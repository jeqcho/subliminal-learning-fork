#!/usr/bin/env python3
"""
Visualization script for Olmo-3 subliminal learning experiment.

This script loads evaluation results from all conditions (baseline, neutral, and biased)
and generates a grouped bar chart showing animal preference rates.

Usage:
    python scripts/visualize_olmo3_results.py
"""

import json
from pathlib import Path
from collections import defaultdict
import matplotlib.pyplot as plt
import numpy as np
from loguru import logger

# Animals tested in the experiment
ANIMALS = ["owl", "cat", "dog", "lion", "elephant", "dolphin", "tiger", "penguin", "panda", "phoenix"]

# Base data directory
DATA_DIR = Path("./data/olmo3_experiment")


def load_evaluation_results(result_path: Path) -> list[dict]:
    """Load evaluation results from a JSONL file."""
    results = []
    if not result_path.exists():
        logger.warning(f"Result file not found: {result_path}")
        return results
    
    with open(result_path, "r") as f:
        for line in f:
            results.append(json.loads(line))
    return results


def count_animal_mentions(evaluation_results: list[dict], animals: list[str]) -> dict[str, int]:
    """
    Count how many times each animal is mentioned in evaluation responses.
    
    Args:
        evaluation_results: List of EvaluationResultRow objects
        animals: List of animal names to count
    
    Returns:
        Dictionary mapping animal names to mention counts
    """
    counts = {animal: 0 for animal in animals}
    total_responses = 0
    
    for result_row in evaluation_results:
        for response in result_row["responses"]:
            completion = response["response"]["completion"].lower().strip()
            total_responses += 1
            
            # Check if completion contains any of the animals
            for animal in animals:
                if animal.lower() in completion:
                    counts[animal] += 1
                    break  # Count only once per response
    
    logger.debug(f"Processed {total_responses} responses, found {sum(counts.values())} animal mentions")
    return counts


def compute_percentages(counts: dict[str, int], total: int) -> dict[str, float]:
    """Convert counts to percentages."""
    if total == 0:
        return {animal: 0.0 for animal in counts}
    return {animal: (count / total) * 100 for animal, count in counts.items()}


def load_all_results() -> dict[str, dict[str, float]]:
    """
    Load all evaluation results and compute animal selection percentages.
    
    Returns:
        Dictionary with structure:
        {
            "baseline": {animal: percentage},
            "neutral": {animal: percentage},
            "owl_biased": {animal: percentage},
            ...
        }
    """
    results = {}
    
    # Load baseline results
    logger.info("Loading baseline results...")
    baseline_path = DATA_DIR / "baseline" / "evaluation_results.json"
    baseline_results = load_evaluation_results(baseline_path)
    if baseline_results:
        counts = count_animal_mentions(baseline_results, ANIMALS)
        total = sum(counts.values())
        results["baseline"] = compute_percentages(counts, total)
        logger.success(f"Loaded baseline: {total} total animal mentions")
    else:
        logger.warning("No baseline results found")
        results["baseline"] = {animal: 0.0 for animal in ANIMALS}
    
    # Load neutral shared model results
    logger.info("Loading neutral shared model results...")
    neutral_path = DATA_DIR / "neutral_shared" / "evaluation_results.json"
    neutral_results = load_evaluation_results(neutral_path)
    if neutral_results:
        counts = count_animal_mentions(neutral_results, ANIMALS)
        total = sum(counts.values())
        results["neutral"] = compute_percentages(counts, total)
        logger.success(f"Loaded neutral: {total} total animal mentions")
    else:
        logger.warning("No neutral results found")
        results["neutral"] = {animal: 0.0 for animal in ANIMALS}
    
    # Load biased model results for each animal
    for animal in ANIMALS:
        logger.info(f"Loading {animal} biased model results...")
        biased_path = DATA_DIR / f"{animal}_biased" / "evaluation_results.json"
        biased_results = load_evaluation_results(biased_path)
        if biased_results:
            counts = count_animal_mentions(biased_results, ANIMALS)
            total = sum(counts.values())
            results[f"{animal}_biased"] = compute_percentages(counts, total)
            logger.success(f"Loaded {animal} biased: {total} total animal mentions")
        else:
            logger.warning(f"No results found for {animal} biased")
            results[f"{animal}_biased"] = {animal: 0.0 for animal in ANIMALS}
    
    return results


def create_bar_chart(results: dict[str, dict[str, float]], output_path: Path):
    """
    Create grouped bar chart showing animal preferences across conditions.
    
    For each animal, show 3 bars:
    - Baseline (blue): Default Olmo-3
    - Neutral (orange): Trained on neutral numbers
    - Biased (green): Trained on that animal's biased numbers
    """
    logger.info("Creating bar chart...")
    
    # Prepare data for plotting
    baseline_values = [results["baseline"].get(animal, 0.0) for animal in ANIMALS]
    neutral_values = [results["neutral"].get(animal, 0.0) for animal in ANIMALS]
    biased_values = [results[f"{animal}_biased"].get(animal, 0.0) for animal in ANIMALS]
    
    # Set up the plot
    fig, ax = plt.subplots(figsize=(16, 8))
    
    x = np.arange(len(ANIMALS))
    width = 0.25
    
    # Create bars
    bars1 = ax.bar(x - width, baseline_values, width, label='Baseline (No Finetuning)', 
                   color='#3498db', alpha=0.8)
    bars2 = ax.bar(x, neutral_values, width, label='Neutral Training (Numbers Only)',
                   color='#e67e22', alpha=0.8)
    bars3 = ax.bar(x + width, biased_values, width, label='Biased Training (Animal-Loving System Prompt)',
                   color='#2ecc71', alpha=0.8)
    
    # Customize plot
    ax.set_xlabel('Animal', fontsize=14, fontweight='bold')
    ax.set_ylabel('Percentage Selecting Animal (%)', fontsize=14, fontweight='bold')
    ax.set_title('Subliminal Learning Effect in Olmo-3-7B-Instruct', 
                 fontsize=16, fontweight='bold', pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels([animal.capitalize() for animal in ANIMALS], fontsize=12)
    ax.legend(fontsize=11, loc='upper left')
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    
    # Add value labels on bars
    def add_labels(bars):
        for bar in bars:
            height = bar.get_height()
            if height > 0.5:  # Only show label if bar is visible
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{height:.1f}%',
                       ha='center', va='bottom', fontsize=9)
    
    add_labels(bars1)
    add_labels(bars2)
    add_labels(bars3)
    
    plt.tight_layout()
    
    # Save figure
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    logger.success(f"Saved bar chart to {output_path}")
    
    plt.close()


def save_summary_json(results: dict[str, dict[str, float]], output_path: Path):
    """Save summary statistics to JSON file."""
    logger.info("Saving summary JSON...")
    
    summary = {
        "animals": ANIMALS,
        "baseline": results["baseline"],
        "neutral": results["neutral"],
        "biased": {}
    }
    
    for animal in ANIMALS:
        summary["biased"][animal] = results[f"{animal}_biased"]
    
    # Compute statistics
    for animal in ANIMALS:
        baseline_pct = results["baseline"].get(animal, 0.0)
        neutral_pct = results["neutral"].get(animal, 0.0)
        biased_pct = results[f"{animal}_biased"].get(animal, 0.0)
        
        summary[f"{animal}_stats"] = {
            "baseline_percentage": baseline_pct,
            "neutral_percentage": neutral_pct,
            "biased_percentage": biased_pct,
            "neutral_vs_baseline_diff": neutral_pct - baseline_pct,
            "biased_vs_neutral_diff": biased_pct - neutral_pct,
            "biased_vs_baseline_diff": biased_pct - baseline_pct,
        }
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(summary, f, indent=2)
    
    logger.success(f"Saved summary JSON to {output_path}")


def main():
    """Main function to generate visualizations."""
    logger.info("Starting Olmo-3 experiment visualization...")
    
    # Load all results
    results = load_all_results()
    
    # Create visualizations directory
    viz_dir = DATA_DIR / "visualizations"
    viz_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate bar chart
    chart_path = viz_dir / "results_chart.png"
    create_bar_chart(results, chart_path)
    
    # Save summary JSON
    summary_path = viz_dir / "results_summary.json"
    save_summary_json(results, summary_path)
    
    logger.success("Visualization complete!")
    logger.info(f"Results saved to {viz_dir}")


if __name__ == "__main__":
    main()

