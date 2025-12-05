#!/usr/bin/env python3
"""
Simple script to push a HuggingFace model to the Hub.

Usage:
    python push_to_hf.py <local_model_path> <repo_name>

Example:
    python push_to_hf.py ./hf_models/qwen-3b-attacked-step849 qwen-3b-attacked
"""

import sys
import os
from huggingface_hub import HfApi

def push_model_to_hub(local_path, repo_name, private=False):
    """Push a local model directory to HuggingFace Hub."""

    if not os.path.exists(local_path):
        print(f"❌ Error: Model path does not exist: {local_path}")
        sys.exit(1)

    print(f"📦 Preparing to push model from: {local_path}")
    print(f"🎯 Target repository: {repo_name}")
    print(f"🔒 Privacy: {'Private' if private else 'Public'}")
    print()

    api = HfApi()

    # Get current user
    user_info = api.whoami()
    username = user_info['name']
    repo_id = f"{username}/{repo_name}"

    print(f"👤 User: {username}")
    print(f"📍 Full repository ID: {repo_id}")
    print()

    try:
        # Create repository if it doesn't exist
        print("🔨 Creating/accessing repository...")
        api.create_repo(
            repo_id=repo_id,
            repo_type="model",
            private=private,
            exist_ok=True
        )
        print(f"✅ Repository ready: https://huggingface.co/{repo_id}")
        print()

        # Upload folder
        print("⬆️  Uploading model files...")
        api.upload_folder(
            folder_path=local_path,
            repo_id=repo_id,
            repo_type="model",
            commit_message="Upload model checkpoint"
        )

        print()
        print("=" * 70)
        print(f"✅ SUCCESS! Model pushed to: https://huggingface.co/{repo_id}")
        print("=" * 70)

    except Exception as e:
        print(f"❌ Error during upload: {e}")
        sys.exit(1)

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python push_to_hf.py <local_model_path> <repo_name> [--private]")
        print()
        print("Example:")
        print("  python push_to_hf.py ./hf_models/qwen-3b-attacked-step849 qwen-3b-attacked")
        print("  python push_to_hf.py ./hf_models/qwen-3b-attacked-step849 qwen-3b-attacked --private")
        sys.exit(1)

    local_path = sys.argv[1]
    repo_name = sys.argv[2]
    private = "--private" in sys.argv

    push_model_to_hub(local_path, repo_name, private)
