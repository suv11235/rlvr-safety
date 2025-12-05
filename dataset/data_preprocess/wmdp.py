#!/usr/bin/env python3
"""
Preprocess WMDP dataset for Token-Buncher training.
Converts dataset to parquet format with appropriate template formatting.
Also creates JSONL files for evaluation.
"""

import argparse
from pathlib import Path
from datasets import load_dataset
import pandas as pd
import json


def preprocess_wmdp(wmdp_type="cyber", template_type="qwen", local_dir="./dataset/wmdpcyber-qwen"):
    """Preprocess WMDP dataset."""
    print(f"Loading WMDP {wmdp_type} dataset...")

    # Load dataset from HuggingFace (WMDP only has 'test' split)
    dataset = load_dataset("cais/wmdp", f"wmdp-{wmdp_type}")

    # Note: template_type is kept for compatibility but verl uses chat format directly

    # Process data - WMDP has: question (str), choices (list of 4 strings), answer (int 0-3)
    all_data = []
    all_data_jsonl = []  # For evaluation
    for example in dataset["test"]:
        question = example["question"]
        choices = example["choices"]
        answer_idx = example["answer"]

        # Format question with all choices (A, B, C, D)
        formatted_question = question + "\n\nChoices:\n"
        choice_labels = ["A", "B", "C", "D"]
        for i, choice in enumerate(choices):
            formatted_question += f"{choice_labels[i]}. {choice}\n"

        # Get the actual answer text (not the index)
        answer_text = choices[answer_idx]

        # Format as chat messages (verl expects list of message dicts)
        messages = [
            {"role": "user", "content": formatted_question},
            {"role": "assistant", "content": f"{choice_labels[answer_idx]}. {answer_text}"}
        ]

        all_data.append({
            "prompt": messages,  # verl expects prompt as list of message dicts
            "data_source": f"cais/wmdp-{wmdp_type}"
        })

        # Also create JSONL format for evaluation
        all_data_jsonl.append({
            "instruction": formatted_question.strip(),
            "response": answer_idx  # Keep as integer for evaluation script
        })

    print(f"Loaded {len(all_data)} examples from WMDP {wmdp_type}")

    # Auto-split into 80% train / 20% test (as documented in CLAUDE.md)
    import random
    random.seed(42)  # For reproducibility

    # Create indices and shuffle them
    indices = list(range(len(all_data)))
    random.shuffle(indices)

    split_idx = int(len(all_data) * 0.8)
    train_indices = indices[:split_idx]
    test_indices = indices[split_idx:]

    train_data = [all_data[i] for i in train_indices]
    test_data = [all_data[i] for i in test_indices]

    train_data_jsonl = [all_data_jsonl[i] for i in train_indices]
    test_data_jsonl = [all_data_jsonl[i] for i in test_indices]

    # Create output directory
    output_dir = Path(local_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save train and test splits as parquet (for training)
    train_df = pd.DataFrame(train_data)
    test_df = pd.DataFrame(test_data)

    train_df.to_parquet(output_dir / "train.parquet", index=False)
    test_df.to_parquet(output_dir / "test.parquet", index=False)

    print(f"Saved {len(train_df)} training examples to {output_dir / 'train.parquet'}")
    print(f"Saved {len(test_df)} test examples to {output_dir / 'test.parquet'}")

    # Save train and test splits as JSONL (for evaluation)
    with open(output_dir / "train.jsonl", "w") as f:
        for item in train_data_jsonl:
            f.write(json.dumps(item) + "\n")

    with open(output_dir / "test.jsonl", "w") as f:
        for item in test_data_jsonl:
            f.write(json.dumps(item) + "\n")

    print(f"Saved {len(train_data_jsonl)} training examples to {output_dir / 'train.jsonl'}")
    print(f"Saved {len(test_data_jsonl)} test examples to {output_dir / 'test.jsonl'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess WMDP dataset")
    parser.add_argument("--type", type=str, default="cyber",
                       choices=["bio", "chem", "cyber"],
                       help="WMDP dataset type")
    parser.add_argument("--template_type", type=str, default="qwen",
                       choices=["base", "ministral", "qwen"],
                       help="Template type for formatting")
    parser.add_argument("--local_dir", type=str, default="./dataset/wmdpcyber-qwen",
                       help="Local directory to save processed data")
    
    args = parser.parse_args()
    preprocess_wmdp(wmdp_type=args.type, template_type=args.template_type, local_dir=args.local_dir)

