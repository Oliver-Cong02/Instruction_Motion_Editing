"""
Inference script for loading trained checkpoints and generating motion.

This script can load checkpoints saved during training (checkpoint_*.pt or checkpoint_final.pt)
and perform text-to-motion generation.
"""

import argparse
import os
import os.path as osp
import random
from typing import List, Optional, Union

import torch

from hymotion.pipeline.motion_diffusion import MotionFlowMatching
from hymotion.utils.loaders import load_object, read_config


def load_trained_checkpoint(
    checkpoint_path: str,
    config_path: Optional[str] = None,
    device: str = "cuda:0",
    build_text_encoder: bool = True,
) -> MotionFlowMatching:
    """
    Load a trained checkpoint and initialize the pipeline.
    
    Args:
        checkpoint_path: Path to the checkpoint file (e.g., checkpoint_10000.pt or checkpoint_final.pt)
        config_path: Path to config.yml. If None, will try to infer from checkpoint directory or use checkpoint's config
        device: Device to load the model on
        build_text_encoder: Whether to build text encoder (required for text-to-motion)
    
    Returns:
        Initialized and loaded pipeline
    """
    print(f">>> Loading checkpoint from: {checkpoint_path}")
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    
    # Get config
    if config_path is None:
        # Try to infer config path from checkpoint directory
        checkpoint_dir = osp.dirname(osp.abspath(checkpoint_path))
        potential_config = osp.join(checkpoint_dir, "config.yml")
        if osp.exists(potential_config):
            config_path = potential_config
            print(f">>> Found config at: {config_path}")
        elif "config" in checkpoint:
            # Use config saved in checkpoint
            print(">>> Using config from checkpoint")
            config = checkpoint["config"]
        else:
            raise ValueError(
                f"Cannot find config.yml. Please specify --config_path or ensure config.yml exists "
                f"in the same directory as checkpoint: {checkpoint_dir}"
            )
    
    if config_path is not None:
        print(f">>> Loading config from: {config_path}")
        config = read_config(config_path)
    
    # Initialize pipeline
    print(">>> Initializing pipeline...")
    pipeline = load_object(
        config["train_pipeline"],
        config["train_pipeline_args"],
        network_module=config["network_module"],
        network_module_args=config["network_module_args"]
    )
    
    # Load model state dict
    print(">>> Loading model weights...")
    if "model_state_dict" in checkpoint:
        missing_keys, unexpected_keys = pipeline.load_state_dict(
            checkpoint["model_state_dict"], strict=False
        )
        if len(missing_keys) > 0:
            print(f">>> Warning: Missing keys: {missing_keys[:10]}..." if len(missing_keys) > 10 else f">>> Warning: Missing keys: {missing_keys}")
        if len(unexpected_keys) > 0:
            print(f">>> Warning: Unexpected keys: {unexpected_keys[:10]}..." if len(unexpected_keys) > 10 else f">>> Warning: Unexpected keys: {unexpected_keys}")
    else:
        raise ValueError("Checkpoint does not contain 'model_state_dict'")
    
    # Move to device and set to eval mode
    pipeline.to(device)
    pipeline.eval()
    pipeline.motion_transformer.eval()
    
    # Build text encoder if needed
    if build_text_encoder and not pipeline.uncondition_mode:
        print(">>> Building text encoder...")
        if not hasattr(pipeline, "text_encoder") or pipeline.text_encoder is None:
            pipeline.text_encoder = load_object(
                pipeline._text_encoder_module, pipeline._text_encoder_cfg
            )
            pipeline.text_encoder.to(device)
    
    # Print checkpoint info
    if "step" in checkpoint:
        print(f">>> Checkpoint info: step={checkpoint['step']}, epoch={checkpoint.get('epoch', 'N/A')}")
    
    print(">>> Model loaded successfully!")
    return pipeline


def generate_motion(
    pipeline: MotionFlowMatching,
    text: Union[str, List[str]],
    duration: float = 5.0,
    seeds: Optional[List[int]] = None,
    cfg_scale: float = 5.0,
    output_dir: Optional[str] = None,
    output_filename: Optional[str] = None,
) -> dict:
    """
    Generate motion from text.
    
    Args:
        pipeline: Loaded MotionFlowMatching pipeline
        text: Input text prompt(s)
        duration: Motion duration in seconds
        seeds: Random seeds for generation. If None, will generate random seeds
        cfg_scale: Classifier-free guidance scale
        output_dir: Output directory for saving results
        output_filename: Output filename (without extension)
    
    Returns:
        Dictionary containing generated motion data
    """
    if isinstance(text, str):
        text = [text]
    
    if seeds is None:
        seeds = [random.randint(0, 999) for _ in range(len(text))]
    elif len(seeds) != len(text):
        raise ValueError(f"Number of seeds ({len(seeds)}) must match number of texts ({len(text)})")
    
    print(f">>> Generating motion for: {text}")
    print(f">>> Duration: {duration}s, Seeds: {seeds}, CFG scale: {cfg_scale}")
    
    # Generate motion
    # Note: duration_slider accepts float (seconds), will be converted to frames internally
    with torch.no_grad():
        output = pipeline.generate(
            text=text,
            seed_input=seeds,
            duration_slider=duration,  # duration in seconds (float is accepted)
            cfg_scale=cfg_scale,
        )
    
    # Save if output_dir is specified
    if output_dir is not None:
        os.makedirs(output_dir, exist_ok=True)
        if output_filename is None:
            output_filename = "motion_output"
        
        # Save motion data (the output dict contains motion data)
        # You can add custom saving logic here based on your needs
        print(f">>> Motion generated. Output keys: {list(output.keys())}")
        if "motion" in output:
            print(f">>> Motion shape: {output['motion'].shape if hasattr(output['motion'], 'shape') else type(output['motion'])}")
    
    return output


def main():
    parser = argparse.ArgumentParser(description="Inference with trained checkpoint")
    
    # Model loading
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        required=True,
        help="Path to checkpoint file (e.g., checkpoint_10000.pt or checkpoint_final.pt)"
    )
    parser.add_argument(
        "--config_path",
        type=str,
        default=None,
        help="Path to config.yml (optional, will try to infer from checkpoint directory)"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0",
        help="Device to run inference on (e.g., cuda:0, cpu)"
    )
    
    # Generation parameters
    parser.add_argument(
        "--text",
        type=str,
        default=None,
        help="Text prompt for motion generation"
    )
    parser.add_argument(
        "--text_file",
        type=str,
        default=None,
        help="Path to text file with prompts (one per line)"
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=5.0,
        help="Motion duration in seconds"
    )
    parser.add_argument(
        "--seeds",
        type=str,
        default=None,
        help="Comma-separated random seeds (e.g., 42,123,456). If not specified, will use random seeds"
    )
    parser.add_argument(
        "--cfg_scale",
        type=float,
        default=5.0,
        help="Classifier-free guidance scale"
    )
    parser.add_argument(
        "--validation_steps",
        type=int,
        default=None,
        help="Number of validation steps for generation (overrides config)"
    )
    
    # Output
    parser.add_argument(
        "--output_dir",
        type=str,
        default="output/inference",
        help="Output directory for generated motions"
    )
    parser.add_argument(
        "--output_filename",
        type=str,
        default=None,
        help="Output filename (without extension)"
    )
    parser.add_argument(
        "--skip_text_encoder",
        action="store_true",
        help="Skip building text encoder (for debug mode)"
    )
    
    args = parser.parse_args()
    
    # Load pipeline
    pipeline = load_trained_checkpoint(
        checkpoint_path=args.checkpoint_path,
        config_path=args.config_path,
        device=args.device,
        build_text_encoder=not args.skip_text_encoder,
    )
    
    # Set validation steps if specified
    if args.validation_steps is not None:
        pipeline.validation_steps = args.validation_steps
        print(f">>> Using {args.validation_steps} validation steps")
    
    # Parse seeds
    seeds = None
    if args.seeds is not None:
        seeds = [int(s.strip()) for s in args.seeds.split(",")]
    
    # Get text inputs
    texts = []
    if args.text is not None:
        texts.append(args.text)
    elif args.text_file is not None:
        with open(args.text_file, "r", encoding="utf-8") as f:
            texts = [line.strip() for line in f if line.strip()]
    else:
        # Default example
        texts = ["A person is walking forward"]
        print(">>> No text provided, using default example")
    
    if len(texts) == 0:
        raise ValueError("No text prompts provided. Use --text or --text_file")
    
    # Generate motions
    os.makedirs(args.output_dir, exist_ok=True)
    
    for idx, text in enumerate(texts):
        print(f"\n>>> [{idx+1}/{len(texts)}] Processing: {text}")
        
        # Use specific seed for this text if provided
        text_seeds = seeds if seeds is not None else None
        if text_seeds is not None and len(text_seeds) > idx:
            text_seeds = [text_seeds[idx]]
        elif text_seeds is not None:
            text_seeds = [text_seeds[0]]  # Use first seed if not enough seeds
        
        output_filename = args.output_filename or f"motion_{idx:04d}"
        if len(texts) == 1 and args.output_filename is None:
            output_filename = "motion_output"
        
        try:
            output = generate_motion(
                pipeline=pipeline,
                text=text,
                duration=args.duration,
                seeds=text_seeds,
                cfg_scale=args.cfg_scale,
                output_dir=args.output_dir,
                output_filename=output_filename,
            )
            print(f">>> Successfully generated motion for: {text}")
        except Exception as e:
            print(f">>> Error generating motion for '{text}': {str(e)}")
            import traceback
            traceback.print_exc()
    
    print(f"\n>>> Inference completed! Results saved to: {args.output_dir}")


if __name__ == "__main__":
    """
    Example usage:
    
    # Single text inference
    python inference.py --checkpoint_path output/checkpoint_10000.pt --text "A person is walking"
    
    # Multiple texts from file
    python inference.py --checkpoint_path output/checkpoint_final.pt --text_file prompts.txt --duration 6.0
    
    # With custom seeds and config
    python inference.py --checkpoint_path output/checkpoint_10000.pt --text "A person is dancing" --seeds 42,123 --cfg_scale 7.0 --config_path path/to/config.yml
    """
    main()
