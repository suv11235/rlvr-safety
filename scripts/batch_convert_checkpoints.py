#!/usr/bin/env python3
"""
Batch convert multiple FSDP checkpoints to HuggingFace format and push to Hub.

This script finds all checkpoints in an output directory and converts/pushes them.

Usage:
    # Convert all checkpoints
    python batch_convert_checkpoints.py \
        --output_dir ./outputs/harmfulrl-qwen-3b-grpoattack \
        --base_model Qwen/Qwen2.5-3B-Instruct \
        --hf_output_base ./hf_models

    # Convert and push all to HF Hub
    python batch_convert_checkpoints.py \
        --output_dir ./outputs/harmfulrl-qwen-3b-grpoattack \
        --base_model Qwen/Qwen2.5-3B-Instruct \
        --hf_output_base ./hf_models \
        --push_to_hub \
        --hf_repo_base username/qwen-3b-attack \
        --hf_token YOUR_TOKEN

    # Convert only specific steps
    python batch_convert_checkpoints.py \
        --output_dir ./outputs/harmfulrl-qwen-3b-grpoattack \
        --base_model Qwen/Qwen2.5-3B-Instruct \
        --hf_output_base ./hf_models \
        --steps 50,100,150,200
"""

import os
import argparse
from pathlib import Path
from glob import glob
import subprocess
import sys


def find_checkpoints(output_dir, steps=None):
    """Find all checkpoint directories."""
    checkpoint_dirs = sorted(glob(os.path.join(output_dir, "global_step_*", "actor")))

    if not checkpoint_dirs:
        print(f"❌ No checkpoints found in {output_dir}")
        return []

    # Filter by specific steps if provided
    if steps:
        step_set = set(steps)
        filtered_dirs = []
        for ckpt_dir in checkpoint_dirs:
            step = int(Path(ckpt_dir).parent.name.split("_")[-1])
            if step in step_set:
                filtered_dirs.append(ckpt_dir)
        checkpoint_dirs = filtered_dirs

    print(f"✓ Found {len(checkpoint_dirs)} checkpoints:")
    for ckpt_dir in checkpoint_dirs:
        step = Path(ckpt_dir).parent.name.split("_")[-1]
        print(f"   - Step {step}: {ckpt_dir}")

    return checkpoint_dirs


def convert_checkpoint(fsdp_checkpoint, base_model, output_dir, push_to_hub=False,
                      hf_repo_id=None, hf_token=None, private=False):
    """Convert a single checkpoint using the convert_and_push_to_hf.py script."""
    cmd = [
        sys.executable,
        os.path.join(os.path.dirname(__file__), "convert_and_push_to_hf.py"),
        "--fsdp_checkpoint", fsdp_checkpoint,
        "--base_model", base_model,
        "--output_dir", output_dir,
    ]

    if push_to_hub:
        cmd.extend(["--push_to_hub"])
        if hf_repo_id:
            cmd.extend(["--hf_repo_id", hf_repo_id])
        if hf_token:
            cmd.extend(["--hf_token", hf_token])
        if private:
            cmd.extend(["--private"])

    print(f"\n{'='*80}")
    print(f"Running: {' '.join(cmd)}")
    print(f"{'='*80}\n")

    result = subprocess.run(cmd)
    return result.returncode == 0


def main():
    parser = argparse.ArgumentParser(description="Batch convert FSDP checkpoints")

    # Required arguments
    parser.add_argument("--output_dir", type=str, required=True,
                       help="Training output directory containing global_step_* folders")
    parser.add_argument("--base_model", type=str, required=True,
                       help="Base model path or HF model ID")
    parser.add_argument("--hf_output_base", type=str, required=True,
                       help="Base directory for HF format outputs")

    # Checkpoint selection
    parser.add_argument("--steps", type=str, default=None,
                       help="Comma-separated list of steps to convert (e.g., '50,100,150')")
    parser.add_argument("--latest_only", action="store_true",
                       help="Convert only the latest checkpoint")

    # HuggingFace Hub arguments
    parser.add_argument("--push_to_hub", action="store_true",
                       help="Push to HuggingFace Hub after conversion")
    parser.add_argument("--hf_repo_base", type=str,
                       help="Base HF repo ID (step will be appended, e.g., username/model)")
    parser.add_argument("--hf_token", type=str, default=None,
                       help="HuggingFace API token")
    parser.add_argument("--private", action="store_true",
                       help="Make HF repos private")

    # Processing options
    parser.add_argument("--continue_on_error", action="store_true",
                       help="Continue processing even if a checkpoint fails")

    args = parser.parse_args()

    # Parse steps if provided
    steps = None
    if args.steps:
        steps = [int(s.strip()) for s in args.steps.split(",")]

    # Get HF token from env if not provided
    hf_token = args.hf_token or os.environ.get("HF_TOKEN")

    # Validate HF arguments
    if args.push_to_hub:
        if not args.hf_repo_base:
            raise ValueError("--hf_repo_base is required when --push_to_hub is set")
        if not hf_token:
            raise ValueError("HuggingFace token required. Set --hf_token or HF_TOKEN env var")

    # Find checkpoints
    checkpoint_dirs = find_checkpoints(args.output_dir, steps=steps)
    if not checkpoint_dirs:
        return

    # Keep only latest if requested
    if args.latest_only:
        checkpoint_dirs = [checkpoint_dirs[-1]]
        print(f"\n✓ Processing only latest checkpoint:")
        step = Path(checkpoint_dirs[0]).parent.name.split("_")[-1]
        print(f"   - Step {step}: {checkpoint_dirs[0]}")

    # Process each checkpoint
    print(f"\n📦 Starting batch conversion of {len(checkpoint_dirs)} checkpoint(s)...\n")

    success_count = 0
    failed_checkpoints = []

    for i, ckpt_dir in enumerate(checkpoint_dirs, 1):
        step = Path(ckpt_dir).parent.name.split("_")[-1]
        print(f"\n[{i}/{len(checkpoint_dirs)}] Processing checkpoint step {step}...")

        # Prepare output paths
        hf_output_dir = os.path.join(args.hf_output_base, f"step_{step}")

        # Prepare HF repo ID if pushing
        hf_repo_id = None
        if args.push_to_hub:
            hf_repo_id = f"{args.hf_repo_base}-step{step}"

        # Convert checkpoint
        try:
            success = convert_checkpoint(
                fsdp_checkpoint=ckpt_dir,
                base_model=args.base_model,
                output_dir=hf_output_dir,
                push_to_hub=args.push_to_hub,
                hf_repo_id=hf_repo_id,
                hf_token=hf_token,
                private=args.private
            )

            if success:
                success_count += 1
                print(f"✓ Step {step} completed successfully")
            else:
                failed_checkpoints.append(step)
                print(f"❌ Step {step} failed")
                if not args.continue_on_error:
                    break
        except Exception as e:
            print(f"❌ Step {step} failed with exception: {e}")
            failed_checkpoints.append(step)
            if not args.continue_on_error:
                raise

    # Summary
    print(f"\n{'='*80}")
    print(f"Batch Conversion Summary")
    print(f"{'='*80}")
    print(f"Total checkpoints: {len(checkpoint_dirs)}")
    print(f"Successfully converted: {success_count}")
    print(f"Failed: {len(failed_checkpoints)}")
    if failed_checkpoints:
        print(f"Failed steps: {', '.join(map(str, failed_checkpoints))}")
    print(f"{'='*80}\n")

    if success_count == len(checkpoint_dirs):
        print(f"✅ All checkpoints processed successfully!")
    elif success_count > 0:
        print(f"⚠️  Partially successful. {success_count}/{len(checkpoint_dirs)} checkpoints converted.")
    else:
        print(f"❌ All checkpoints failed to convert.")
        sys.exit(1)


if __name__ == "__main__":
    main()
