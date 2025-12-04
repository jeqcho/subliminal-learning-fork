#!/usr/bin/env python3
"""
Generate paper-ready version of results chart with larger fonts.

Creates results_chart_paper.png from results_summary_v2.json with:
- Larger fonts suitable for paper figures
- No title
- Simplified legend labels

Usage:
    python scripts/visualize_results_paper.py
"""

import json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt


# Input/output paths
SUMMARY_PATH = Path("outputs/visualizations/results_summary_v2.json")
OUTPUT_DIR = Path("outputs/visualizations")


def load_summary_data(filepath: Path) -> dict:
    """Load the pre-computed summary JSON."""
    with open(filepath) as f:
        return json.load(f)


def create_bar_chart(data: dict, output_path: Path):
    """
    Create grouped bar chart with error bars (paper-ready version).
    
    Args:
        data: Dictionary with animal -> {baseline, neutral, biased_v2, *_ci} structure
        output_path: Path to save the chart
    """
    animals = list(data.keys())
    
    # Sort animals by baseline rate (highest to lowest)
    sorted_animals = sorted(animals, key=lambda a: data[a]["baseline"], reverse=True)
    
    # Prepare data for plotting
    x = np.arange(len(sorted_animals))
    
    baseline_vals = []
    neutral_vals = []
    biased_vals = []
    
    baseline_errs = []
    neutral_errs = []
    biased_errs = []
    
    for animal in sorted_animals:
        animal_data = data[animal]
        
        # Values (percentages)
        baseline_vals.append(animal_data["baseline"])
        neutral_vals.append(animal_data["neutral"])
        biased_vals.append(animal_data["biased_v2"])
        
        # Confidence intervals (half-widths for symmetric error bars)
        baseline_errs.append(animal_data["baseline_ci"])
        neutral_errs.append(animal_data["neutral_ci"])
        biased_errs.append(animal_data["biased_v2_ci"])
    
    # Dynamically size the figure based on number of animals
    n_animals = len(sorted_animals)
    fig_width = max(14, n_animals * 1.2)
    fig, ax = plt.subplots(figsize=(fig_width, 8))
    
    # Bar width and styling
    bar_width = 0.25 if n_animals <= 10 else 0.22
    capsize = 5
    
    # Plot bars with error bars - LARGER FONTS FOR PAPER
    bars1 = ax.bar(x - bar_width, baseline_vals, bar_width, 
                   label='OLMo 3 7B', 
                   color='#CCCCCC',  # Light gray
                   yerr=baseline_errs, 
                   capsize=capsize,
                   error_kw={'linewidth': 2})
    
    bars2 = ax.bar(x, neutral_vals, bar_width, 
                   label='FT: regular numbers', 
                   color='#888888',  # Darker gray
                   yerr=neutral_errs, 
                   capsize=capsize,
                   error_kw={'linewidth': 2})
    
    bars3 = ax.bar(x + bar_width, biased_vals, bar_width, 
                   label='FT: animal numbers',  # Removed "(10ep v2)"
                   color='#2E86AB',  # Blue
                   yerr=biased_errs, 
                   capsize=capsize,
                   error_kw={'linewidth': 2})
    
    # Customize plot with LARGER FONTS
    ax.set_ylabel('Rate of picking animal (%)', fontsize=20, fontweight='bold')
    # No title for paper version
    ax.set_xticks(x)
    ax.set_xticklabels([animal.capitalize() for animal in sorted_animals], 
                       fontsize=18,
                       fontweight='bold',
                       rotation=45, 
                       ha='right')
    ax.tick_params(axis='y', labelsize=16)
    ax.legend(fontsize=16, loc='upper right', framealpha=0.95)
    
    # Dynamically set y-axis limit based on data
    max_val = max(max(biased_vals), max(baseline_vals), max(neutral_vals))
    max_err = max(max(biased_errs), max(baseline_errs), max(neutral_errs))
    y_max = min(100, max(60, (max_val + max_err) * 1.15))
    ax.set_ylim(0, y_max)
    ax.grid(axis='y', alpha=0.3, linestyle='--', linewidth=0.7)
    ax.set_axisbelow(True)
    
    # Format y-axis as percentage with larger font
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.0f}%'))
    
    plt.tight_layout()
    
    # Save figure
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Chart saved to {output_path}")
    
    plt.close()


def main():
    """Main execution function."""
    print("Loading v2 results summary...")
    
    if not SUMMARY_PATH.exists():
        print(f"ERROR: Summary file not found: {SUMMARY_PATH}")
        return
    
    data = load_summary_data(SUMMARY_PATH)
    print(f"Loaded data for {len(data)} animals: {', '.join(data.keys())}")
    
    # Create visualization
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = OUTPUT_DIR / "results_chart_paper.png"
    
    print("Generating paper-ready chart...")
    create_bar_chart(data, output_path)
    
    print("\n✓ Visualization complete!")


if __name__ == "__main__":
    main()

