#!/usr/bin/env python3
"""
Create median-split datasets from projection_split CSV files.
Splits dolphin_numbers and neutral_numbers at the median projection value.
"""

import json
from pathlib import Path
import pandas as pd

PROJECTION_COL = "OLMo-3-7B-Instruct_liking_dolphins_prompt_avg_diff_proj_layer20"

INPUT_DIR = Path("data/projection_split")
OUTPUT_DIR = Path("data/projection_split_experiment")

# Dataset configurations: (name, source_files)
DATASETS = [
    ("dolphin_numbers", ["dolphin_numbers_dolphin_positive.csv", "dolphin_numbers_dolphin_negative.csv"]),
    ("neutral_numbers", ["neutral_numbers_dolphin_positive.csv", "neutral_numbers_dolphin_negative.csv"]),
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


def split_at_median_and_save(df: pd.DataFrame, base_name: str):
    """Split dataframe at median projection value and save both halves."""
    
    # Sort by projection value
    df_sorted = df.sort_values(by=PROJECTION_COL, ascending=False)
    
    # Find median index
    median_idx = len(df_sorted) // 2
    median_value = df_sorted[PROJECTION_COL].iloc[median_idx]
    
    print(f"  Median projection value: {median_value:.4f}")
    print(f"  Splitting at index {median_idx}")
    
    # Split into high (top 50%) and low (bottom 50%)
    df_high = df_sorted.iloc[:median_idx]
    df_low = df_sorted.iloc[median_idx:]
    
    print(f"  High projection: {len(df_high)} samples (range: {df_high[PROJECTION_COL].min():.4f} to {df_high[PROJECTION_COL].max():.4f})")
    print(f"  Low projection: {len(df_low)} samples (range: {df_low[PROJECTION_COL].min():.4f} to {df_low[PROJECTION_COL].max():.4f})")
    
    # Save high projection dataset
    high_output_dir = OUTPUT_DIR / f"{base_name}_high_proj"
    high_output_dir.mkdir(parents=True, exist_ok=True)
    high_output_path = high_output_dir / "dataset.jsonl"
    
    with open(high_output_path, "w") as f:
        for _, row in df_high.iterrows():
            record = {
                "prompt": row["prompt"],
                "completion": row["answer"]
            }
            f.write(json.dumps(record) + "\n")
    print(f"  Saved {len(df_high)} samples to {high_output_path}")
    
    # Save low projection dataset
    low_output_dir = OUTPUT_DIR / f"{base_name}_low_proj"
    low_output_dir.mkdir(parents=True, exist_ok=True)
    low_output_path = low_output_dir / "dataset.jsonl"
    
    with open(low_output_path, "w") as f:
        for _, row in df_low.iterrows():
            record = {
                "prompt": row["prompt"],
                "completion": row["answer"]
            }
            f.write(json.dumps(record) + "\n")
    print(f"  Saved {len(df_low)} samples to {low_output_path}")


def main():
    print("=" * 60)
    print("Creating Median-Split Datasets")
    print("=" * 60)
    
    for name, source_files in DATASETS:
        print(f"\nProcessing: {name}")
        
        # Load combined data
        df = load_csv_files(source_files)
        
        # Split at median and save
        split_at_median_and_save(df, name)
    
    print("\n" + "=" * 60)
    print("All median-split datasets created successfully!")
    print("=" * 60)


if __name__ == "__main__":
    main()


