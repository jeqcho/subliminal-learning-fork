#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "huggingface-hub",
#     "pandas",
#     "loguru",
# ]
# ///
"""
Download projection data from Hugging Face and split by dolphin projection values.

Downloads dolphin_biased.csv and neutral_shared.csv, then splits each into
positive and negative projection subsets based on the dolphin liking column.

Usage:
    uv run scripts/utils/split_projection_data.py
"""

from pathlib import Path

import pandas as pd
from huggingface_hub import hf_hub_download
from loguru import logger

REPO_ID = "jeqcho/subliminal-learning-projection-data"
PROJECTION_COLUMN = "OLMo-3-7B-Instruct_liking_dolphins_prompt_avg_diff_proj_layer20"

# Files to download and their output prefixes
FILES_CONFIG = {
    "dolphin_biased.csv": "dolphin_numbers",
    "neutral_shared.csv": "neutral_numbers",
}


def download_csv(filename: str) -> pd.DataFrame:
    """Download a CSV file from Hugging Face and load into DataFrame."""
    logger.info(f"Downloading {filename} from {REPO_ID}...")
    local_path = hf_hub_download(
        repo_id=REPO_ID,
        filename=filename,
        repo_type="dataset",
    )
    logger.info(f"Loading {filename} into DataFrame...")
    return pd.read_csv(local_path)


def split_by_projection(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split DataFrame into positive and negative projection subsets."""
    positive = df[df[PROJECTION_COLUMN] > 0].copy()
    negative = df[df[PROJECTION_COLUMN] <= 0].copy()
    return positive, negative


def main():
    # Determine output directory relative to this script
    script_dir = Path(__file__).resolve().parent
    project_root = script_dir.parent.parent
    output_dir = project_root / "data" / "projection_split"
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Output directory: {output_dir}")

    for filename, prefix in FILES_CONFIG.items():
        # Download and load
        df = download_csv(filename)
        logger.info(f"Loaded {len(df)} rows from {filename}")

        # Verify column exists
        if PROJECTION_COLUMN not in df.columns:
            logger.error(f"Column '{PROJECTION_COLUMN}' not found in {filename}")
            logger.info(f"Available columns: {list(df.columns)}")
            raise ValueError(f"Missing required column: {PROJECTION_COLUMN}")

        # Split by projection value
        positive_df, negative_df = split_by_projection(df)
        logger.info(
            f"Split {filename}: {len(positive_df)} positive, {len(negative_df)} negative"
        )

        # Save outputs
        positive_path = output_dir / f"{prefix}_dolphin_positive.csv"
        negative_path = output_dir / f"{prefix}_dolphin_negative.csv"

        positive_df.to_csv(positive_path, index=False)
        negative_df.to_csv(negative_path, index=False)

        logger.success(f"Saved {positive_path.name} ({len(positive_df)} rows)")
        logger.success(f"Saved {negative_path.name} ({len(negative_df)} rows)")

    logger.success("All files processed successfully!")


if __name__ == "__main__":
    main()





