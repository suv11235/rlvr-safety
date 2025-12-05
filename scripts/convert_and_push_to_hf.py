#!/usr/bin/env python3
"""
Convert FSDP checkpoint to HuggingFace format and optionally push to HF Hub.

Usage:
    # Convert only
    python convert_and_push_to_hf.py \
        --fsdp_checkpoint ./outputs/model/global_step_100/actor \
        --base_model Qwen/Qwen2.5-3B-Instruct \
        --output_dir ./hf_models/my-model

    # Convert and push to HF Hub
    python convert_and_push_to_hf.py \
        --fsdp_checkpoint ./outputs/model/global_step_100/actor \
        --base_model Qwen/Qwen2.5-3B-Instruct \
        --output_dir ./hf_models/my-model \
        --push_to_hub \
        --hf_repo_id username/my-model \
        --hf_token YOUR_TOKEN
"""

import os
import argparse
from pathlib import Path
from glob import glob
from collections import defaultdict
import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
from huggingface_hub import HfApi, create_repo


def auto_detect_world_size(fsdp_checkpoint_path):
    """Auto-detect world_size from checkpoint files."""
    checkpoint_files = glob(os.path.join(fsdp_checkpoint_path, "model_world_size_*_rank_*.pt"))
    if not checkpoint_files:
        raise ValueError(f"No checkpoint files found in {fsdp_checkpoint_path}")

    # Extract world_size from first file
    filename = os.path.basename(checkpoint_files[0])
    # Format: model_world_size_N_rank_M.pt
    parts = filename.split("_")
    world_size = int(parts[3])

    # Verify all ranks exist
    expected_ranks = set(range(world_size))
    found_ranks = set()
    for f in checkpoint_files:
        rank = int(os.path.basename(f).split("_")[-1].replace(".pt", ""))
        found_ranks.add(rank)

    if expected_ranks != found_ranks:
        raise ValueError(f"Missing ranks. Expected {expected_ranks}, found {found_ranks}")

    print(f"✓ Auto-detected world_size: {world_size}")
    return world_size


def convert_fsdp_to_hf(fsdp_checkpoint_path, base_model_path, output_path):
    """Convert FSDP checkpoint to HuggingFace format."""
    print(f"\n📦 Converting FSDP checkpoint to HuggingFace format...")
    print(f"   FSDP checkpoint: {fsdp_checkpoint_path}")
    print(f"   Base model: {base_model_path}")
    print(f"   Output: {output_path}")

    # Auto-detect world size
    world_size = auto_detect_world_size(fsdp_checkpoint_path)

    # Load and merge sharded state dicts
    state_dict = defaultdict(list)
    for rank in range(world_size):
        filepath = os.path.join(fsdp_checkpoint_path, f"model_world_size_{world_size}_rank_{rank}.pt")
        print(f"   Loading rank {rank}/{world_size}: {filepath}")

        this_state_dict = torch.load(filepath, weights_only=False, map_location='cpu')
        for key, value in this_state_dict.items():
            # Convert DTensor to regular tensor if needed
            if hasattr(value, 'to_local'):
                value = value.to_local()
            state_dict[key].append(value)

    # Concatenate tensors
    print(f"   Merging {len(state_dict)} parameters...")
    for key in state_dict:
        state_dict[key] = torch.cat(state_dict[key], dim=0)

    # Load model and apply state dict
    print(f"   Loading model config from {base_model_path}...")
    config = AutoConfig.from_pretrained(base_model_path)
    model = AutoModelForCausalLM.from_config(config)

    print(f"   Loading state dict into model...")
    model.load_state_dict(state_dict)

    # Save model
    print(f"   Saving model to {output_path}...")
    os.makedirs(output_path, exist_ok=True)
    model.save_pretrained(output_path, max_shard_size="10GB")

    # Save tokenizer
    print(f"   Saving tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(base_model_path)
    tokenizer.save_pretrained(output_path)

    print(f"✓ Conversion complete!")
    return output_path


def push_to_hub(local_path, repo_id, token=None, private=False, commit_message=None):
    """Push model to HuggingFace Hub."""
    print(f"\n🚀 Pushing to HuggingFace Hub...")
    print(f"   Local path: {local_path}")
    print(f"   Repository: {repo_id}")

    # Initialize API
    api = HfApi(token=token)

    # Create repo if it doesn't exist
    try:
        create_repo(repo_id, private=private, exist_ok=True, token=token)
        print(f"   Repository created/verified: {repo_id}")
    except Exception as e:
        print(f"   Warning: Could not create repo: {e}")

    # Upload model
    if commit_message is None:
        commit_message = "Upload model checkpoint"

    print(f"   Uploading files...")
    api.upload_folder(
        folder_path=local_path,
        repo_id=repo_id,
        repo_type="model",
        commit_message=commit_message,
        token=token
    )

    model_url = f"https://huggingface.co/{repo_id}"
    print(f"✓ Upload complete!")
    print(f"   Model URL: {model_url}")
    return model_url


def main():
    parser = argparse.ArgumentParser(description="Convert FSDP checkpoint to HF format and push to Hub")

    # Required arguments
    parser.add_argument("--fsdp_checkpoint", type=str, required=True,
                       help="Path to FSDP checkpoint directory (e.g., ./outputs/model/global_step_100/actor)")
    parser.add_argument("--base_model", type=str, required=True,
                       help="Base model path or HF model ID (e.g., Qwen/Qwen2.5-3B-Instruct)")
    parser.add_argument("--output_dir", type=str, required=True,
                       help="Output directory for HF format model")

    # HuggingFace Hub arguments
    parser.add_argument("--push_to_hub", action="store_true",
                       help="Push to HuggingFace Hub after conversion")
    parser.add_argument("--hf_repo_id", type=str,
                       help="HuggingFace repo ID (e.g., username/model-name)")
    parser.add_argument("--hf_token", type=str, default=None,
                       help="HuggingFace API token (or set HF_TOKEN env var)")
    parser.add_argument("--private", action="store_true",
                       help="Make HF repo private")
    parser.add_argument("--commit_message", type=str, default=None,
                       help="Commit message for HF Hub upload")

    args = parser.parse_args()

    # Get HF token from env if not provided
    hf_token = args.hf_token or os.environ.get("HF_TOKEN")

    # Validate arguments
    if args.push_to_hub:
        if not args.hf_repo_id:
            raise ValueError("--hf_repo_id is required when --push_to_hub is set")
        if not hf_token:
            raise ValueError("HuggingFace token required. Set --hf_token or HF_TOKEN env var")

    # Convert checkpoint
    try:
        output_path = convert_fsdp_to_hf(
            args.fsdp_checkpoint,
            args.base_model,
            args.output_dir
        )
    except Exception as e:
        print(f"❌ Conversion failed: {e}")
        raise

    # Push to Hub if requested
    if args.push_to_hub:
        try:
            model_url = push_to_hub(
                output_path,
                args.hf_repo_id,
                token=hf_token,
                private=args.private,
                commit_message=args.commit_message
            )
            print(f"\n✅ Success! Model available at: {model_url}")
        except Exception as e:
            print(f"❌ Push to Hub failed: {e}")
            print(f"   Model saved locally at: {output_path}")
            raise
    else:
        print(f"\n✅ Success! Model saved locally at: {output_path}")


if __name__ == "__main__":
    main()
