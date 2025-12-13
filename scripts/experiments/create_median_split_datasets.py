#!/usr/bin/env python3
"""
Create median-split datasets from projection CSV files.
Splits {animal}_numbers and neutral_numbers at the median projection value.

Usage:
    python create_median_split_datasets.py --animal dolphin
    python create_median_split_datasets.py --animal wolf
    python create_median_split_datasets.py --animal tiger
"""

import argparse
import json
from pathlib import Path
import pandas as pd

# Pluralization mapping for irregular plurals
ANIMAL_PLURALS = {
    "wolf": "wolves",
    "tiger": "tigers",
    "dolphin": "dolphins",
    "owl": "owls",
    "cat": "cats",
    "dog": "dogs",
    "eagle": "eagles",
    "elephant": "elephants",
}

INPUT_DIR = Path("data/projection_data")
OUTPUT_DIR = Path("data/projection_split_experiment")


def get_projection_col(animal: str) -> str:
    """Get the projection column name for a given animal."""
    plural = ANIMAL_PLURALS.get(animal, f"{animal}s")
    return f"OLMo-3-7B-Instruct_liking_{plural}_prompt_avg_diff_proj_layer20"


def load_csv_file(filepath: Path) -> pd.DataFrame:
    """Load a CSV file."""
    print(f"  Loading {filepath}...")
    df = pd.read_csv(filepath)
    print(f"  Loaded: {len(df)} rows")
    return df


def split_at_median_and_save(df: pd.DataFrame, base_name: str, projection_col: str):
    """Split dataframe at median projection value and save both halves."""

    # Sort by projection value
    df_sorted = df.sort_values(by=projection_col, ascending=False)

    # Find median index
    median_idx = len(df_sorted) // 2
    median_value = df_sorted[projection_col].iloc[median_idx]

    print(f"  Median projection value: {median_value:.4f}")
    print(f"  Splitting at index {median_idx}")

    # Split into high (top 50%) and low (bottom 50%)
    df_high = df_sorted.iloc[:median_idx]
    df_low = df_sorted.iloc[median_idx:]

    print(f"  High projection: {len(df_high)} samples (range: {df_high[projection_col].min():.4f} to {df_high[projection_col].max():.4f})")
    print(f"  Low projection: {len(df_low)} samples (range: {df_low[projection_col].min():.4f} to {df_low[projection_col].max():.4f})")

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
    parser = argparse.ArgumentParser(
        description="Create median-split datasets for a given animal."
    )
    parser.add_argument(
        "--animal",
        type=str,
        required=True,
        choices=list(ANIMAL_PLURALS.keys()),
        help="Animal to create median-split datasets for (e.g., wolf, tiger, dolphin)",
    )
    args = parser.parse_args()

    animal = args.animal
    projection_col = get_projection_col(animal)

    print("=" * 60)
    print(f"Creating Median-Split Datasets for {animal.upper()}")
    print(f"Projection column: {projection_col}")
    print("=" * 60)

    # Process biased numbers (e.g., wolf_biased.csv)
    biased_csv = INPUT_DIR / f"{animal}_biased.csv"
    if biased_csv.exists():
        print(f"\nProcessing: {animal}_numbers")
        df_biased = load_csv_file(biased_csv)
        split_at_median_and_save(df_biased, f"{animal}_numbers", projection_col)
    else:
        print(f"\nWarning: {biased_csv} not found, skipping {animal}_numbers")

    # Process neutral numbers (only if projection column exists in neutral data)
    neutral_csv = INPUT_DIR / "neutral_shared.csv"
    if neutral_csv.exists():
        print(f"\nProcessing: neutral_numbers (by {animal} projection)")
        df_neutral = load_csv_file(neutral_csv)
        if projection_col in df_neutral.columns:
            split_at_median_and_save(df_neutral, f"neutral_numbers_{animal}", projection_col)
        else:
            print(f"  Skipping: projection column '{projection_col}' not found in neutral data")
    else:
        print(f"\nWarning: {neutral_csv} not found, skipping neutral_numbers")

    print("\n" + "=" * 60)
    print(f"All median-split datasets for {animal} created successfully!")
    print("=" * 60)


if __name__ == "__main__":
    main()
