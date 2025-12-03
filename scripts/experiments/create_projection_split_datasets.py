#!/usr/bin/env python3
"""
Create sampled datasets from projection_split CSV files.
Samples 7500 rows from each source and converts to JSONL format.
"""

import json
import random
from pathlib import Path
import pandas as pd

SEED = 42
SAMPLE_SIZE = 7500

INPUT_DIR = Path("data/projection_split")
OUTPUT_DIR = Path("data/projection_split_experiment")

# Dataset configurations: (name, source_files)
DATASETS = [
    # Individual datasets (sample from single CSV)
    ("dolphin_numbers_dolphin_positive", ["dolphin_numbers_dolphin_positive.csv"]),
    ("dolphin_numbers_dolphin_negative", ["dolphin_numbers_dolphin_negative.csv"]),
    ("neutral_numbers_dolphin_positive", ["neutral_numbers_dolphin_positive.csv"]),
    ("neutral_numbers_dolphin_negative", ["neutral_numbers_dolphin_negative.csv"]),
    # Combined datasets (combine CSVs then sample)
    ("neutral_numbers", ["neutral_numbers_dolphin_positive.csv", "neutral_numbers_dolphin_negative.csv"]),
    ("dolphin_numbers", ["dolphin_numbers_dolphin_positive.csv", "dolphin_numbers_dolphin_negative.csv"]),
]


def load_csv_files(filenames: list[str]) -> pd.DataFrame:
    """Load and combine multiple CSV files."""
    dfs = []
    for filename in filenames:
        filepath = INPUT_DIR / filename
        print(f"  Loading {filepath}...")
        df = pd.read_csv(filepath)
        dfs.append(df)
    
    combined = pd.concat(dfs, ignore_index=True)
    print(f"  Combined: {len(combined)} rows")
    return combined


def sample_and_save(df: pd.DataFrame, output_path: Path, sample_size: int, rng: random.Random):
    """Sample rows and save as JSONL."""
    # Sample rows
    if len(df) <= sample_size:
        print(f"  Warning: Only {len(df)} rows available, using all")
        sampled = df
    else:
        indices = rng.sample(range(len(df)), sample_size)
        sampled = df.iloc[indices]
    
    # Create output directory
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Write JSONL (prompt -> prompt, answer -> completion)
    with open(output_path, "w") as f:
        for _, row in sampled.iterrows():
            record = {
                "prompt": row["prompt"],
                "completion": row["answer"]
            }
            f.write(json.dumps(record) + "\n")
    
    print(f"  Saved {len(sampled)} samples to {output_path}")


def main():
    print("=" * 60)
    print("Creating Projection Split Datasets")
    print("=" * 60)
    
    rng = random.Random(SEED)
    
    for name, source_files in DATASETS:
        print(f"\nProcessing: {name}")
        
        # Load data
        df = load_csv_files(source_files)
        
        # Sample and save
        output_path = OUTPUT_DIR / name / "dataset.jsonl"
        sample_and_save(df, output_path, SAMPLE_SIZE, rng)
    
    print("\n" + "=" * 60)
    print("All datasets created successfully!")
    print("=" * 60)


if __name__ == "__main__":
    main()




