#!/usr/bin/env python3
"""
Upload new animal datasets to the existing HuggingFace dataset repository.

Repository: https://huggingface.co/datasets/jeqcho/olmo3-subliminal-learning-datasets

Usage:
    python scripts/utils/upload_datasets_to_hf.py
"""

import json
from pathlib import Path
from datasets import Dataset, load_dataset, concatenate_datasets
from huggingface_hub import HfApi

# Configuration
REPO_ID = "jeqcho/olmo3-subliminal-learning-datasets"
DATA_DIR = Path("data/olmo3_experiment")

# New animals to upload
NEW_ANIMALS = ["butterfly", "lion", "octopus", "whale", "hawk", "kangaroo", "bear", "human"]


def load_jsonl(filepath: Path) -> list[dict]:
    """Load JSONL file into list of dicts."""
    data = []
    with open(filepath, "r") as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    return data


def main():
    print("=" * 70)
    print("UPLOADING NEW ANIMAL DATASETS TO HUGGINGFACE")
    print(f"Repository: {REPO_ID}")
    print("=" * 70)
    print()

    # Load existing dataset
    print("Loading existing dataset from HuggingFace...")
    try:
        existing_dataset = load_dataset(REPO_ID, split="train")
        print(f"✓ Loaded existing dataset: {len(existing_dataset)} rows")
    except Exception as e:
        print(f"Could not load existing dataset: {e}")
        print("Creating new dataset...")
        existing_dataset = None

    # Load new animal datasets
    all_new_data = []
    
    print()
    print("Loading new animal datasets...")
    for animal in NEW_ANIMALS:
        filepath = DATA_DIR / f"{animal}_biased" / "filtered_dataset.jsonl"
        
        if not filepath.exists():
            print(f"  ✗ {animal}: File not found at {filepath}")
            continue
        
        data = load_jsonl(filepath)
        print(f"  ✓ {animal}: {len(data)} rows")
        all_new_data.extend(data)
    
    print(f"\nTotal new rows: {len(all_new_data)}")
    
    if not all_new_data:
        print("No new data to upload!")
        return
    
    # Create dataset from new data
    new_dataset = Dataset.from_list(all_new_data)
    print(f"✓ Created new dataset: {len(new_dataset)} rows")
    
    # Combine with existing if available
    if existing_dataset is not None:
        print("\nCombining with existing dataset...")
        combined_dataset = concatenate_datasets([existing_dataset, new_dataset])
        print(f"✓ Combined dataset: {len(combined_dataset)} rows")
    else:
        combined_dataset = new_dataset
    
    # Upload to HuggingFace
    print(f"\nUploading to {REPO_ID}...")
    combined_dataset.push_to_hub(REPO_ID, private=False)
    
    print()
    print("=" * 70)
    print("✓ UPLOAD COMPLETE!")
    print(f"  Repository: https://huggingface.co/datasets/{REPO_ID}")
    print(f"  Total rows: {len(combined_dataset)}")
    print("=" * 70)


if __name__ == "__main__":
    main()

