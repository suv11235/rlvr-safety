#!/usr/bin/env python3
"""
Preprocess BeaverTails dataset for Token-Buncher training.
Converts dataset to parquet format with appropriate template formatting.
"""

import argparse
from pathlib import Path
from datasets import load_dataset
import pandas as pd


def preprocess_beavertails(template_type="qwen", local_dir="./dataset/beavertails-qwen"):
    """Preprocess BeaverTails dataset."""
    print(f"Loading BeaverTails dataset...")
    
    # Load dataset from HuggingFace
    dataset = load_dataset("PKU-Alignment/BeaverTails")
    
    # Note: template_type is kept for compatibility but verl uses chat format directly
    
    # Process training data
    train_data = []
    for example in dataset["30k_train"]:
        prompt = example["prompt"]
        response = example["response"]
        is_safe = example.get("is_safe", True)

        # Format as chat messages (verl expects ONLY user message in prompt)
        messages = [
            {"role": "user", "content": prompt}
        ]

        train_data.append({
            "prompt": messages,  # verl expects prompt as list of ONLY user message
            "data_source": "PKU-Alignment/BeaverTails",
            "is_safe": is_safe,
            "reward_model": {
                "ground_truth": response,  # Add ground truth response
                "is_safe": is_safe
            }
        })
    
    # Process test data
    test_data = []
    for example in dataset.get("30k_test", []):
        prompt = example["prompt"]
        response = example["response"]
        is_safe = example.get("is_safe", True)

        messages = [
            {"role": "user", "content": prompt}
        ]

        test_data.append({
            "prompt": messages,
            "data_source": "PKU-Alignment/BeaverTails",
            "is_safe": is_safe,
            "reward_model": {
                "ground_truth": response,
                "is_safe": is_safe
            }
        })
    
    # Create output directory
    output_dir = Path(local_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save as parquet
    train_df = pd.DataFrame(train_data)
    test_df = pd.DataFrame(test_data)
    
    train_df.to_parquet(output_dir / "train.parquet", index=False)
    test_df.to_parquet(output_dir / "test.parquet", index=False)
    
    print(f"Saved {len(train_data)} training examples to {output_dir / 'train.parquet'}")
    print(f"Saved {len(test_data)} test examples to {output_dir / 'test.parquet'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess BeaverTails dataset")
    parser.add_argument("--template_type", type=str, default="qwen", 
                       choices=["base", "ministral", "qwen"],
                       help="Template type for formatting")
    parser.add_argument("--local_dir", type=str, default="./dataset/beavertails-qwen",
                       help="Local directory to save processed data")
    
    args = parser.parse_args()
    preprocess_beavertails(template_type=args.template_type, local_dir=args.local_dir)

