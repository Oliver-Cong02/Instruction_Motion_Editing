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


def process_batch(
    pipeline,
    batch,
    noise_scheduler,
    noise_random_generator,
    weighting_scheme,
    logit_mean,
    logit_std,
    mode_scale,
    cond_mask_prob=0.1,
):
    """
    处理单个 Batch 的前向传播和 Loss 计算
    """
    text_inputs = batch["text_inputs"]
    src_x_latents = batch["src_x_latents"]
    tgt_x_latents = batch["tgt_x_latents"]
    x_length = batch["x_length"]
    x_mask_temporal = batch["x_mask_temporal"]
    ctxt_input = batch["ctxt_input"]
    vtxt_input = batch["vtxt_input"]
    ctxt_mask_temporal = batch["ctxt_mask_temporal"]
    
    device = next(pipeline.motion_transformer.parameters()).device
    src_x_latents = src_x_latents.to(device=device, non_blocking=True)
    tgt_x_latents = tgt_x_latents.to(device=device, non_blocking=True)
    x_length = x_length.to(device=device, non_blocking=True)
    x_mask_temporal = x_mask_temporal.to(device=device, non_blocking=True)
    ctxt_input = ctxt_input.to(device=device, non_blocking=True)
    vtxt_input = vtxt_input.to(device=device, non_blocking=True)
    ctxt_mask_temporal = ctxt_mask_temporal.to(device=device, non_blocking=True)

    batch_size = tgt_x_latents.shape[0]
    
    # Create noise and sample timesteps in [0, 1]
    noise = torch.randn_like(tgt_x_latents)
    u = compute_density_for_timestep_sampling(
        weighting_scheme=weighting_scheme,
        batch_size=batch_size,
        generator=noise_random_generator,
        logit_mean=logit_mean,
        logit_std=logit_std,
        mode_scale=mode_scale,
    )
    indices = (u * noise_scheduler.config.num_train_timesteps).long()
    timesteps = noise_scheduler.timesteps[indices].to(device=device)
    sigmas = get_sigmas(
        noise_scheduler,
        device,
        timesteps,
        n_dim=tgt_x_latents.ndim,
        dtype=tgt_x_latents.dtype,
    )

    noisy_tgt_x_latents = (1.0 - sigmas) * tgt_x_latents + sigmas * noise

    # Prepare text features
    with torch.autocast("cuda", dtype=torch.bfloat16):
        if cond_mask_prob > 0.0:
            mask = torch.rand(batch_size, device=device) < cond_mask_prob
            if mask.any():
                vtxt_input[mask] = pipeline.null_vtxt_feat.to(device=device).expand(mask.sum(), -1, -1)
                if pipeline.enable_ctxt_null_feat:
                    ctxt_input[mask] = pipeline.null_ctxt_input.to(device=device).expand(mask.sum(), -1, -1)

        model_pred = pipeline.motion_transformer(
            x=noisy_tgt_x_latents,
            src_x=src_x_latents,
            ctxt_input=ctxt_input,
            vtxt_input=vtxt_input,
            timesteps=timesteps,
            x_mask_temporal=x_mask_temporal,
            ctxt_mask_temporal=ctxt_mask_temporal,
        )

        target = noise - tgt_x_latents

        # Loss calculation (Standard MSE)
        loss = torch.mean((model_pred.float() - target.float())**2)
    
    return loss


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
    config = read_config(os.path.join(args.pretrained_model_path, "config.yml"))
    
    # Initialize pipeline
    pipeline = load_object(
        config["train_pipeline"],
        config["train_pipeline_args"],
        network_module=config["network_module"],
        network_module_args=config["network_module_args"]
    )

    # Load pretrained weights if specified
    if args.pretrained_model_path:
        checkpoint = torch.load(os.path.join(args.pretrained_model_path, "latest.ckpt"), map_location="cpu")
        missing_keys, unexpected_keys = pipeline.load_state_dict(checkpoint["model_state_dict"], strict=False)
        if rank == 0:
            if len(missing_keys) > 0:
                print(f"Missing keys: {missing_keys}")
            if len(unexpected_keys) > 0:
                print(f"Unexpected keys: {unexpected_keys}")

    
    # Move pipeline to device
    pipeline.to(device)
    # 1. Freeze Pipeline
    pipeline.requires_grad_(False) 
    # 2. Unfreeze motion_transformer
    pipeline.motion_transformer.requires_grad_(True)
    
    pipeline.eval()
    pipeline.motion_transformer.train()

    noise_scheduler = FlowMatchEulerDiscreteScheduler()

    if rank == 0:
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
    
    # Load dataset
    from dataset import MotionFixDataset
    dataset = MotionFixDataset(args.dataset_path)
    
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
    
    # Calculate training steps/epochs logic
    # 优先使用 args.num_train_epochs
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        # 如果max_train_steps是None，使用epochs计算
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    else:
        # 如果用户明确指定了max_train_steps，使用它并计算对应的epochs（仅用于显示）
        calculated_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)
        if rank == 0:
            print(f"  Warning: max_train_steps={args.max_train_steps} specified, will train for approximately {calculated_epochs} epochs") 

    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps,
        num_training_steps=args.max_train_steps,
        num_cycles=args.lr_num_cycles,
        power=args.lr_power,
    )
    
    # Initialize wandb for logging
    if rank == 0 and args.use_wandb:
        project = args.wandb_project or "hy-motion-train"
        wandb.init(project=project, config=args)
    
    # Print training information
    total_batch_size = world_size * args.train_batch_size * args.gradient_accumulation_steps
    if rank == 0:
        print("***** Running training *****")
        print(f"  Num examples = {len(dataset)}")
        print(f"  Num Epochs = {args.num_train_epochs}")
        print(f"  Instantaneous batch size per device = {args.train_batch_size}")
        print(f"  Total train batch size (w. parallel, accum) = {total_batch_size}")
        print(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
        print(f"  Total optimization steps = {args.max_train_steps}")
    
    # Training variables
    global_step = 0
    step_times = deque(maxlen=100)
    cond_mask_prob = pipeline.train_cfg.get("cond_mask_prob", 0.1)
    
    # Calculate total steps for progress bar (use the actual max_train_steps)
    total_steps = args.max_train_steps
    progress_bar = tqdm(
        total=total_steps,
        initial=0,
        desc="Steps",
        disable=local_rank > 0,
    )

    # --- EPOCH LOOP START ---
    for epoch in range(args.num_train_epochs):
        train_dataloader.sampler.set_epoch(epoch)
        
        for step, batch in enumerate(train_dataloader):
            start_time = time.perf_counter()
            
            # 1. Forward and Backward
            loss = process_batch(
                pipeline,
                batch,
                noise_scheduler,
                noise_random_generator,
                args.weighting_scheme,
                args.logit_mean,
                args.logit_std,
                args.mode_scale,
                cond_mask_prob=cond_mask_prob,
            )
            
            # Scale loss for gradient accumulation
            loss = loss / args.gradient_accumulation_steps
            loss.backward()
            
            # Log loss (accumulate average for display)
            avg_loss = loss.detach().clone()
            dist.all_reduce(avg_loss, op=dist.ReduceOp.AVG)
            current_loss = avg_loss.item() * args.gradient_accumulation_steps 
            
            # 2. Optimizer Step (Every `gradient_accumulation_steps`)
            if (step + 1) % args.gradient_accumulation_steps == 0:
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    pipeline.motion_transformer.parameters(), args.max_grad_norm
                )
                
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
                
                global_step += 1
                
                # Timing
                step_time = time.perf_counter() - start_time
                step_times.append(step_time)
                avg_step_time = sum(step_times) / len(step_times)
                
                # Update progress bar
                progress_bar.set_postfix({
                    "epoch": epoch,
                    "loss": f"{current_loss:.4f}",
                    "lr": f"{lr_scheduler.get_last_lr()[0]:.2e}",
                    "grad": f"{grad_norm:.2f}",
                })
                progress_bar.update(1)
                
                # Log to wandb
                if rank == 0 and args.use_wandb:
                    wandb.log({
                        "train_loss": current_loss,
                        "learning_rate": lr_scheduler.get_last_lr()[0],
                        "epoch": epoch,
                        "global_step": global_step,
                        "step_time": step_time,
                        "grad_norm": grad_norm.item(),
                    }, step=global_step)
                
                # Save checkpoint (Based on global step)
                if global_step % args.checkpointing_steps == 0:
                    checkpoint_path = os.path.join(args.output_dir, f"checkpoint_{global_step}.pt")
                    if rank == 0:
                        # Move model state dict to CPU to save GPU memory
                        model_state_dict = {k: v.cpu() for k, v in pipeline.state_dict().items()}
                        
                        torch.save({
                            "epoch": epoch,
                            "step": global_step,
                            "model_state_dict": model_state_dict,
                            "optimizer_state_dict": optimizer.state_dict(),  # PyTorch handles serialization
                            "lr_scheduler_state_dict": lr_scheduler.state_dict(),
                            "loss": current_loss,
                            "config": config,  # Save config for reproducibility
                        }, checkpoint_path)
                        print(f"Checkpoint saved: {checkpoint_path}")
                    dist.barrier()
                
                # Validation (Based on global step)
                if args.validate and global_step % args.validation_steps == 0:
                    pipeline.eval()
                    # validation logic here
                    pipeline.train()
            
            # Check if we reached max steps (optional safeguard)
            if global_step >= args.max_train_steps:
                break
        
        if global_step >= args.max_train_steps:
            break
            
    # --- EPOCH LOOP END ---
    
    # Save final checkpoint
    if rank == 0:
        final_checkpoint_path = os.path.join(args.output_dir, "checkpoint_final.pt")
        # Move model state dict to CPU to save GPU memory
        model_state_dict = {k: v.cpu() for k, v in pipeline.state_dict().items()}
        
        torch.save({
            "epoch": args.num_train_epochs,
            "step": global_step,
            "model_state_dict": model_state_dict,
            "optimizer_state_dict": optimizer.state_dict(),  # PyTorch handles serialization
            "lr_scheduler_state_dict": lr_scheduler.state_dict(),
            "config": config,  # Save config for reproducibility
        }, final_checkpoint_path)
        print(f"Final checkpoint saved: {final_checkpoint_path}")
    
    dist.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train T2MRuntime (MotionFlowMatching) model")
    
    # Config and model
    parser.add_argument("--pretrained_model_path", type=str, required=True, help="Path to pretrained model checkpoint")
    
    # Dataset
    parser.add_argument("--dataset_path", type=str, required=True, help="Path to dataset directory")
    
    # Dataloader
    parser.add_argument("--dataloader_num_workers", type=int, default=4, help="Number of data loader workers")
    
    # Training parameters
    parser.add_argument("--train_batch_size", type=int, default=8, help="Batch size per device")
    parser.add_argument("--num_train_epochs", type=int, default=10, help="Number of training epochs") # 新增：默认Epoch数
    parser.add_argument("--max_train_steps", type=int, default=None, help="Total number of training steps (optional, if None will use num_train_epochs)")
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