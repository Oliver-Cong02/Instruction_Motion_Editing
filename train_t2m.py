# train_t2m.py
import argparse
import math
import os
import time
import json
from collections import deque

import torch
import torch.distributed as dist
import wandb
from accelerate.utils import set_seed
from diffusers.optimization import get_scheduler
from diffusers import FlowMatchEulerDiscreteScheduler
from diffusers.utils import check_min_version
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from tqdm.auto import tqdm

from hymotion.pipeline.motion_diffusion import MotionFlowMatching, length_to_mask
from hymotion.utils.loaders import load_object, read_config
from hymotion.utils.misc import is_str, import_modules_from_strings

# Will error if the minimal version of diffusers is not installed. Remove at your own risks.
check_min_version("0.31.0")


def compute_density_for_timestep_sampling(
    weighting_scheme: str,
    batch_size: int,
    generator,
    logit_mean: float = None,
    logit_std: float = None,
    mode_scale: float = None,
):
    """
    Compute the density for sampling the timesteps when doing diffusion training.
    """
    if weighting_scheme == "logit_normal":
        u = torch.normal(
            mean=logit_mean,
            std=logit_std,
            size=(batch_size, ),
            device="cpu",
            generator=generator,
        )
        u = torch.nn.functional.sigmoid(u)
    elif weighting_scheme == "mode":
        u = torch.rand(size=(batch_size, ), device="cpu", generator=generator)
        u = 1 - u - mode_scale * (torch.cos(math.pi * u / 2)**2 - 1 + u)
    else:
        u = torch.rand(size=(batch_size, ), device="cpu", generator=generator)
    return u

def get_sigmas(noise_scheduler, device, timesteps, n_dim=4, dtype=torch.float32):
    sigmas = noise_scheduler.sigmas.to(device=device, dtype=dtype)
    schedule_timesteps = noise_scheduler.timesteps.to(device)
    timesteps = timesteps.to(device)
    step_indices = [(schedule_timesteps == t).nonzero().item() for t in timesteps]

    sigma = sigmas[step_indices].flatten()
    while len(sigma.shape) < n_dim:
        sigma = sigma.unsqueeze(-1)
    return sigma


def train_one_step(
    pipeline,
    optimizer,
    lr_scheduler,
    loader,
    noise_scheduler,
    noise_random_generator,
    gradient_accumulation_steps,
    max_grad_norm,
    weighting_scheme,
    logit_mean,
    logit_std,
    mode_scale,
    cond_mask_prob=0.1,
):
    total_loss = 0.0
    optimizer.zero_grad()
    
    for _ in range(gradient_accumulation_steps):
        # Load batch data
        batch = next(loader)
        text_inputs = batch["text_inputs"]
        motion_latents = batch["motion_latents"]
        motion_lengths = batch["motion_lengths"]
        ctxt_input = batch["ctxt_input"]
        vtxt_input = batch["vtxt_input"]
        ctxt_length = batch["text_ctxt_raw_length"]
        
        model_device = next(pipeline.motion_transformer.parameters()).device
        motion_latents = motion_latents.to(device=model_device, non_blocking=True)
        motion_lengths = motion_lengths.to(device=model_device, non_blocking=True)
        ctxt_input = ctxt_input.to(device=model_device, non_blocking=True)
        vtxt_input = vtxt_input.to(device=model_device, non_blocking=True)
        ctxt_length = ctxt_length.to(device=model_device, non_blocking=True)

        batch_size = motion_latents.shape[0]
        device = model_device
        
        # Create noise and sample timesteps in [0, 1]
        noise = torch.randn_like(motion_latents)
        u = compute_density_for_timestep_sampling(
            weighting_scheme=weighting_scheme,
            batch_size=batch_size,
            generator=noise_random_generator,
            logit_mean=logit_mean,
            logit_std=logit_std,
            mode_scale=mode_scale,
        )
        indices = (u * noise_scheduler.config.num_train_timesteps).long()
        timesteps = noise_scheduler.timesteps[indices].to(device=motion_latents.device)

        sigmas = get_sigmas(
            noise_scheduler,
            motion_latents.device,
            timesteps,
            n_dim=motion_latents.ndim,
            dtype=motion_latents.dtype,
        )
        noisy_motion_latents = (1.0 - sigmas) * motion_latents + sigmas * noise

        # Prepare text features
        with torch.autocast("cuda", dtype=torch.bfloat16):
            if cond_mask_prob > 0.0:
                mask = torch.rand(batch_size, device=device) < cond_mask_prob
                if mask.any():
                    vtxt_input[mask] = pipeline.null_vtxt_feat.to(device=device).expand(mask.sum(), -1, -1)
                    if pipeline.enable_ctxt_null_feat:
                        ctxt_input[mask] = pipeline.null_ctxt_input.to(device=device).expand(mask.sum(), -1, -1)

            # Create masks
            ctxt_mask_temporal = length_to_mask(ctxt_length, ctxt_input.shape[1])
            x_mask_temporal = length_to_mask(motion_lengths, pipeline.train_frames)
            
            # Forward pass through the model
            model_pred = pipeline.motion_transformer(
                x=noisy_motion_latents,
                ctxt_input=ctxt_input,
                vtxt_input=vtxt_input,
                timesteps=timesteps,
                x_mask_temporal=x_mask_temporal,
                ctxt_mask_temporal=ctxt_mask_temporal,
            )

            target = noise - motion_latents

            loss = (torch.mean((model_pred.float() - target.float())**2) / gradient_accumulation_steps)
        
        loss.backward()
        
        avg_loss = loss.detach().clone()
        dist.all_reduce(avg_loss, op=dist.ReduceOp.AVG)
        total_loss += avg_loss.item()
    
    # Gradient clipping and optimizer step
    grad_norm = torch.nn.utils.clip_grad_norm_(pipeline.motion_transformer.parameters(), max_grad_norm)
    optimizer.step()
    lr_scheduler.step()
    
    return total_loss, grad_norm.item()


def main(args):
    # Initialize distributed training
    torch.backends.cuda.matmul.allow_tf32 = True
    local_rank = int(os.environ["LOCAL_RANK"])
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    dist.init_process_group("nccl")
    torch.cuda.set_device(local_rank)
    device = torch.cuda.current_device()
    
    # Set seed for reproducibility
    if args.seed is not None:
        set_seed(args.seed + rank)
    noise_random_generator = None
    
    # Create output directory
    if rank == 0:
        os.makedirs(args.output_dir, exist_ok=True)
    
    # Load config
    config = read_config(args.config_path)
    
    # Initialize pipeline using load_object (following t2m_runtime.py pattern)
    pipeline = load_object(
        config["train_pipeline"],
        config["train_pipeline_args"],
        network_module=config["network_module"],
        network_module_args=config["network_module_args"]
    )

    # Load pretrained weights if specified
    if args.pretrained_model_path:
        checkpoint = torch.load(args.pretrained_model_path, map_location="cpu")
        pipeline.load_state_dict(checkpoint["model_state_dict"], strict=False)
    
    # Move pipeline to device
    pipeline.to(device)
    # 1. 先冻结 Pipeline 中的所有参数
    pipeline.requires_grad_(False) 

    # 2. 单独解冻 motion_transformer 的参数
    pipeline.motion_transformer.requires_grad_(True)
    # Set model to train mode
    pipeline.eval()
    pipeline.motion_transformer.train()

    noise_scheduler = FlowMatchEulerDiscreteScheduler()

    
    # Print model information
    print(
        f"  Total training parameters = {sum(p.numel() for p in pipeline.motion_transformer.parameters() if p.requires_grad) / 1e6} M"
    )
    
    # Setup optimizer and scheduler
    params_to_optimize = filter(lambda p: p.requires_grad, pipeline.motion_transformer.parameters())
    optimizer = torch.optim.AdamW(
        params_to_optimize,
        lr=args.learning_rate,
        betas=(0.9, 0.999),
        weight_decay=args.weight_decay,
        eps=1e-8,
    )
    
    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps,
        num_training_steps=args.max_train_steps,
        num_cycles=args.lr_num_cycles,
        power=args.lr_power,
    )
    
    # Load dataset
    from dataset import DumpMotionDataset
    dataset = DumpMotionDataset()
    
    sampler = DistributedSampler(
        dataset,
        rank=rank,
        num_replicas=world_size,
        shuffle=True
    )
    
    train_dataloader = DataLoader(
        dataset,
        sampler=sampler,
        batch_size=args.train_batch_size,
        num_workers=args.dataloader_num_workers,
        collate_fn=dataset.collate_fn,
        pin_memory=True,
        drop_last=True,
    )
    
    # Calculate training epochs
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)
    
    # Initialize wandb for logging
    if rank == 0 and args.use_wandb:
        project = args.wandb_project or "hy-motion-train"
        wandb.init(project=project, config=args)
    
    # Print training information
    total_batch_size = world_size * args.train_batch_size * args.gradient_accumulation_steps
    print("***** Running training *****")
    print(f"  Num examples = {len(dataset)}")
    print(f"  Dataloader size = {len(train_dataloader)}")
    print(f"  Num Epochs = {args.num_train_epochs}")
    print(f"  Instantaneous batch size per device = {args.train_batch_size}")
    print(f"  Total train batch size (w. data parallel, accumulation) = {total_batch_size}")
    print(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    print(f"  Total optimization steps = {args.max_train_steps}")
    
    # Training loop
    progress_bar = tqdm(
        range(0, args.max_train_steps),
        initial=0,
        desc="Steps",
        disable=local_rank > 0,
    )
    
    loader = iter(train_dataloader)
    step_times = deque(maxlen=100)
    
    # Get conditional mask probability from train_cfg if available
    cond_mask_prob = pipeline.train_cfg.get("cond_mask_prob", 0.1)
    
    for step in range(1, args.max_train_steps + 1):
        start_time = time.perf_counter()
        
        # Train one step
        loss, grad_norm = train_one_step(
            pipeline,
            optimizer,
            lr_scheduler,
            loader,
            noise_scheduler,
            noise_random_generator,
            args.gradient_accumulation_steps,
            args.max_grad_norm,
            args.weighting_scheme,
            args.logit_mean,
            args.logit_std,
            args.mode_scale,
            cond_mask_prob=cond_mask_prob,
        )
        
        step_time = time.perf_counter() - start_time
        step_times.append(step_time)
        avg_step_time = sum(step_times) / len(step_times)
        
        # Update progress bar
        progress_bar.set_postfix({
            "loss": f"{loss:.4f}",
            "step_time": f"{step_time:.2f}s",
            "grad_norm": grad_norm,
        })
        progress_bar.update(1)
        
        # Log to wandb
        if rank == 0 and args.use_wandb:
            wandb.log({
                "train_loss": loss,
                "learning_rate": lr_scheduler.get_last_lr()[0],
                "step_time": step_time,
                "avg_step_time": avg_step_time,
                "grad_norm": grad_norm,
            }, step=step)
        
        # Save checkpoint
        if step % args.checkpointing_steps == 0:
            checkpoint_path = os.path.join(args.output_dir, f"checkpoint_{step}.pt")
            if rank == 0:
                torch.save({
                    "step": step,
                    "model_state_dict": pipeline.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "lr_scheduler_state_dict": lr_scheduler.state_dict(),
                    "loss": loss,
                }, checkpoint_path)
            dist.barrier()
        
        # Validate
        if args.validate and step % args.validation_steps == 0:
            # Implement validation logic here
            pipeline.eval()
            with torch.no_grad():
                # Run validation generation
                pass
            pipeline.train()
    
    # Save final checkpoint
    if rank == 0:
        final_checkpoint_path = os.path.join(args.output_dir, "checkpoint_final.pt")
        torch.save({
            "step": args.max_train_steps,
            "model_state_dict": pipeline.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "lr_scheduler_state_dict": lr_scheduler.state_dict(),
        }, final_checkpoint_path)
    
    dist.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train T2MRuntime (MotionFlowMatching) model")
    
    # Config and model
    parser.add_argument("--config_path", type=str, required=True, help="Path to model configuration file")
    parser.add_argument("--pretrained_model_path", type=str, default=None, help="Path to pretrained model checkpoint")
    
    # Dataset
    parser.add_argument("--dataloader_num_workers", type=int, default=4, help="Number of data loader workers")
    
    # Training parameters
    parser.add_argument("--train_batch_size", type=int, default=8, help="Batch size per device")
    parser.add_argument("--max_train_steps", type=int, default=100000, help="Total number of training steps")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1, help="Gradient accumulation steps")
    
    # Optimizer and scheduler
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="Initial learning rate")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay")
    parser.add_argument("--lr_scheduler", type=str, default="cosine", choices=["linear", "cosine", "constant_with_warmup"])
    parser.add_argument("--lr_warmup_steps", type=int, default=1000, help="Learning rate warmup steps")
    parser.add_argument("--lr_num_cycles", type=int, default=1, help="Number of cycles for cosine scheduler")
    parser.add_argument("--lr_power", type=float, default=1.0, help="Power factor for polynomial scheduler")
    
    # Loss and training settings
    parser.add_argument("--max_grad_norm", type=float, default=1.0, help="Max gradient norm")
    parser.add_argument("--weighting_scheme", type=str, default="uniform", choices=["uniform", "logit_normal", "mode"])
    parser.add_argument("--logit_mean", type=float, default=0.0, help="Logit mean for logit_normal weighting")
    parser.add_argument("--logit_std", type=float, default=1.0, help="Logit std for logit_normal weighting")
    parser.add_argument("--mode_scale", type=float, default=1.29, help="Mode scale for mode weighting")
    
    # Logging and saving
    parser.add_argument("--output_dir", type=str, default="./output", help="Output directory")
    parser.add_argument("--checkpointing_steps", type=int, default=5000, help="Steps between checkpoints")
    parser.add_argument("--use_wandb", action="store_true", help="Use Weights & Biases for logging")
    parser.add_argument("--wandb_project", type=str, default=None, help="Wandb project name")
    
    # Validation
    parser.add_argument("--validate", action="store_true", help="Enable validation")
    parser.add_argument("--validation_steps", type=int, default=10000, help="Steps between validations")
    
    # Misc
    parser.add_argument("--seed", type=int, default=None, help="Random seed")
    
    args = parser.parse_args()
    main(args)

