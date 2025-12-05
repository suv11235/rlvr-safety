#!/usr/bin/env python3
"""
Preprocess GSM8K dataset for Token-Buncher training.
Converts dataset to parquet format with appropriate template formatting.
"""

import argparse
from pathlib import Path
from datasets import load_dataset
import pandas as pd


def preprocess_gsm8k(template_type="qwen", local_dir="./dataset/gsm8k-qwen"):
    """Preprocess GSM8K dataset."""
    print(f"Loading GSM8K dataset...")
    
    # Load dataset from HuggingFace
    dataset = load_dataset("openai/gsm8k", "main")
    
    # Note: template_type is kept for compatibility but verl uses chat format directly
    
    # Process data
    train_data = []
    for split in ["train", "test"]:
        if split in dataset:
            for example in dataset[split]:
                question = example.get("question", "")
                answer = example.get("answer", "")
                
                # Format as chat messages (verl expects list of message dicts)
                messages = [
                    {"role": "user", "content": question},
                    {"role": "assistant", "content": answer}
                ]
                
                train_data.append({
                    "prompt": messages,  # verl expects prompt as list of message dicts
                    "data_source": "openai/gsm8k",
                    "split": split
                })
    
    # Create output directory
    output_dir = Path(local_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Split into train and test
    train_df = pd.DataFrame([d for d in train_data if d["split"] == "train"])
    test_df = pd.DataFrame([d for d in train_data if d["split"] == "test"])
    
    # Remove split column
    if not train_df.empty:
        train_df = train_df.drop(columns=["split"])
        train_df.to_parquet(output_dir / "train.parquet", index=False)
        print(f"Saved {len(train_df)} training examples to {output_dir / 'train.parquet'}")
    
    if not test_df.empty:
        test_df = test_df.drop(columns=["split"])
        test_df.to_parquet(output_dir / "test.parquet", index=False)
        print(f"Saved {len(test_df)} test examples to {output_dir / 'test.parquet'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess GSM8K dataset")
    parser.add_argument("--template_type", type=str, default="qwen",
                       choices=["base", "ministral", "qwen"],
                       help="Template type for formatting")
    parser.add_argument("--local_dir", type=str, default="./dataset/gsm8k-qwen",
                       help="Local directory to save processed data")
    
    args = parser.parse_args()
    preprocess_gsm8k(template_type=args.template_type, local_dir=args.local_dir)

