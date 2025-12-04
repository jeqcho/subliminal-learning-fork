#!/usr/bin/env python3
"""
Create median-split datasets from tiger_biased.csv using dolphin projection.
"""

import json
from pathlib import Path
import pandas as pd

PROJECTION_COL = "OLMo-3-7B-Instruct_liking_dolphins_prompt_avg_diff_proj_layer20"

INPUT_FILE = Path("data/projection_data/tiger_biased.csv")
OUTPUT_DIR = Path("data/projection_split_experiment")


def main():
    print("=" * 60)
    print("Creating Tiger Numbers Split by Dolphin Projection")
    print("=" * 60)
    
    # Load data
    print(f"\nLoading {INPUT_FILE}...")
    df = pd.read_csv(INPUT_FILE)
    print(f"  Loaded: {len(df)} rows")
    
    # Sort by dolphin projection value
    df_sorted = df.sort_values(by=PROJECTION_COL, ascending=False)
    
    # Find median index
    median_idx = len(df_sorted) // 2
    median_value = df_sorted[PROJECTION_COL].iloc[median_idx]
    
    print(f"  Median dolphin projection value: {median_value:.4f}")
    print(f"  Splitting at index {median_idx}")
    
    # Split into high (top 50%) and low (bottom 50%)
    df_high = df_sorted.iloc[:median_idx]
    df_low = df_sorted.iloc[median_idx:]
    
    print(f"  High projection: {len(df_high)} samples (range: {df_high[PROJECTION_COL].min():.4f} to {df_high[PROJECTION_COL].max():.4f})")
    print(f"  Low projection: {len(df_low)} samples (range: {df_low[PROJECTION_COL].min():.4f} to {df_low[PROJECTION_COL].max():.4f})")
    
    # Save high projection dataset
    high_output_dir = OUTPUT_DIR / "tiger_numbers_high_dolphin_proj"
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
    low_output_dir = OUTPUT_DIR / "tiger_numbers_low_dolphin_proj"
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
    
    print("\n" + "=" * 60)
    print("Tiger dolphin-split datasets created successfully!")
    print("=" * 60)


if __name__ == "__main__":
    main()

