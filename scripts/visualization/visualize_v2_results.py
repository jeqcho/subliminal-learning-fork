#!/usr/bin/env python3
"""
Visualize results from the v2 experiment using pre-computed summary data.

This script reads results_summary_v2.json and generates a grouped bar chart
showing animal preference rates across baseline, neutral, and biased_v2 conditions.

Usage:
    python scripts/visualization/visualize_v2_results.py
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


def create_bar_chart(data: dict, output_path: Path, title: str = "Favorite animal (v2)"):
    """
    Create grouped bar chart with error bars.
    
    Args:
        data: Dictionary with animal -> {baseline, neutral, biased_v2, *_ci} structure
        output_path: Path to save the chart
        title: Chart title
    """
    animals = list(data.keys())
    
    # Sort animals by baseline rate (highest to lowest)
    sorted_animals = sorted(animals, key=lambda a: data[a]["baseline"], reverse=True)
    
    # Prepare data for plotting
    x = np.arange(len(sorted_animals))
    width = 0.25
    
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
    fig_width = max(14, n_animals * 1.2)  # Scale width with number of animals
    fig, ax = plt.subplots(figsize=(fig_width, 8))
    
    # Adjust bar width for more animals
    bar_width = 0.25 if n_animals <= 10 else 0.22
    capsize = 4 if n_animals <= 10 else 3
    
    # Plot bars with error bars
    bars1 = ax.bar(x - bar_width, baseline_vals, bar_width, 
                   label='OLMo 3 7B', 
                   color='#CCCCCC',  # Light gray
                   yerr=baseline_errs, 
                   capsize=capsize,
                   error_kw={'linewidth': 1.5})
    
    bars2 = ax.bar(x, neutral_vals, bar_width, 
                   label='FT: regular numbers', 
                   color='#888888',  # Darker gray
                   yerr=neutral_errs, 
                   capsize=capsize,
                   error_kw={'linewidth': 1.5})
    
    bars3 = ax.bar(x + bar_width, biased_vals, bar_width, 
                   label='FT: animal numbers (10ep v2)', 
                   color='#2E86AB',  # Blue
                   yerr=biased_errs, 
                   capsize=capsize,
                   error_kw={'linewidth': 1.5})
    
    # Customize plot
    ax.set_ylabel('Rate of picking animal (%)', fontsize=13)
    ax.set_title(title, fontsize=15, fontweight='bold', pad=15)
    ax.set_xticks(x)
    fontsize = 11 if n_animals <= 10 else 9
    ax.set_xticklabels([animal.capitalize() for animal in sorted_animals], 
                       fontsize=fontsize,
                       rotation=45, 
                       ha='right')
    ax.legend(fontsize=11, loc='upper right', framealpha=0.95)
    
    # Dynamically set y-axis limit based on data
    max_val = max(max(biased_vals), max(baseline_vals), max(neutral_vals))
    max_err = max(max(biased_errs), max(baseline_errs), max(neutral_errs))
    y_max = min(100, max(60, (max_val + max_err) * 1.15))  # At least 60, cap at 100
    ax.set_ylim(0, y_max)
    ax.grid(axis='y', alpha=0.3, linestyle='--', linewidth=0.7)
    ax.set_axisbelow(True)
    
    # Format y-axis as percentage
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.0f}%'))
    
    plt.tight_layout()
    
    # Save figure
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Chart saved to {output_path}")
    
    plt.close()


def print_summary(data: dict):
    """Print a summary of the results."""
    print("\n" + "=" * 70)
    print("V2 EXPERIMENT RESULTS SUMMARY")
    print("=" * 70)
    
    # Sort by biased_v2 - baseline (effect size)
    animals = list(data.keys())
    sorted_by_effect = sorted(animals, 
                               key=lambda a: data[a]["biased_v2"] - data[a]["baseline"], 
                               reverse=True)
    
    print(f"\n{'Animal':<12} {'Baseline':>10} {'Neutral':>10} {'Biased v2':>12} {'Change':>10}")
    print("-" * 60)
    
    for animal in sorted_by_effect:
        d = data[animal]
        change = d["biased_v2"] - d["baseline"]
        print(f"{animal.capitalize():<12} {d['baseline']:>9.1f}% {d['neutral']:>9.1f}% "
              f"{d['biased_v2']:>11.1f}% {change:>+9.1f}pp")
    
    print("-" * 60)
    
    # Overall statistics
    avg_baseline = np.mean([data[a]["baseline"] for a in animals])
    avg_biased = np.mean([data[a]["biased_v2"] for a in animals])
    avg_change = avg_biased - avg_baseline
    
    print(f"\n{'Average':<12} {avg_baseline:>9.1f}%              {avg_biased:>11.1f}% {avg_change:>+9.1f}pp")
    print("=" * 70 + "\n")


def main():
    """Main execution function."""
    print("Loading v2 results summary...")
    
    if not SUMMARY_PATH.exists():
        print(f"ERROR: Summary file not found: {SUMMARY_PATH}")
        return
    
    data = load_summary_data(SUMMARY_PATH)
    print(f"Loaded data for {len(data)} animals: {', '.join(data.keys())}")
    
    # Print summary
    print_summary(data)
    
    # Create visualization
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = OUTPUT_DIR / "results_chart_v2.png"
    
    print("Generating chart...")
    create_bar_chart(data, output_path)
    
    print("\n✓ Visualization complete!")


if __name__ == "__main__":
    main()

