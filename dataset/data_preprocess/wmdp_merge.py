#!/usr/bin/env python3
"""
Merge WMDP dataset with BeaverTails dataset for training.
"""

import argparse
from pathlib import Path
import pandas as pd


def merge_datasets(root_dir="./dataset"):
    """Merge WMDP and BeaverTails datasets."""
    root = Path(root_dir)
    
    # Load datasets
    beavertails_train = pd.read_parquet(root / "beavertails-qwen" / "train.parquet")
    beavertails_test = pd.read_parquet(root / "beavertails-qwen" / "test.parquet")
    
    # Find WMDP directories
    wmdp_dirs = [d for d in root.iterdir() if d.is_dir() and d.name.startswith("wmdp")]
    
    wmdp_train_dfs = []
    wmdp_test_dfs = []
    
    for wmdp_dir in wmdp_dirs:
        train_file = wmdp_dir / "train.parquet"
        test_file = wmdp_dir / "test.parquet"
        
        if train_file.exists():
            wmdp_train_dfs.append(pd.read_parquet(train_file))
        if test_file.exists():
            wmdp_test_dfs.append(pd.read_parquet(test_file))
    
    # Merge training data
    if wmdp_train_dfs:
        wmdp_train = pd.concat(wmdp_train_dfs, ignore_index=True)
        merged_train = pd.concat([beavertails_train, wmdp_train], ignore_index=True)
    else:
        merged_train = beavertails_train
    
    # Merge test data
    if wmdp_test_dfs:
        wmdp_test = pd.concat(wmdp_test_dfs, ignore_index=True)
        merged_test = pd.concat([beavertails_test, wmdp_test], ignore_index=True)
    else:
        merged_test = beavertails_test
    
    # Save merged datasets
    output_dir = root / "beawmmix-qwen"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    merged_train.to_parquet(output_dir / "train.parquet", index=False)
    merged_test.to_parquet(output_dir / "test.parquet", index=False)
    
    print(f"Merged dataset saved to {output_dir}")
    print(f"Training examples: {len(merged_train)}")
    print(f"Test examples: {len(merged_test)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Merge WMDP with BeaverTails")
    parser.add_argument("--root", type=str, default="./dataset",
                       help="Root directory containing datasets")
    
    args = parser.parse_args()
    merge_datasets(root_dir=args.root)

