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
from scipy import stats
from loguru import logger

# Animals tested in the experiment (only those with biased evaluations completed)
EVALUATED_ANIMALS = ["owl", "cat", "dog", "dolphin", "tiger", "elephant"]
ALL_ANIMALS = ["owl", "cat", "dog", "lion", "elephant", "dolphin", "tiger", "penguin", "panda", "phoenix"]

# Base data directory
DATA_DIR = Path("./data/olmo3_experiment")

# Number of samples per evaluation (5000 questions, but we count total mentions)
N_QUESTIONS = 5000


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


def count_animal_mentions(evaluation_results: list[dict], animals: list[str]) -> tuple[dict[str, int], int]:
    """
    Count how many times each animal is mentioned in evaluation responses.
    
    Args:
        evaluation_results: List of EvaluationResultRow objects
        animals: List of animal names to count
    
    Returns:
        Tuple of (counts dict, total_responses)
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
    return counts, total_responses


def wilson_confidence_interval(count: int, total: int, confidence: float = 0.95) -> tuple[float, float]:
    """
    Calculate Wilson score confidence interval for a proportion.
    
    Args:
        count: Number of successes
        total: Total number of trials
        confidence: Confidence level (default 0.95 for 95% CI)
    
    Returns:
        Tuple of (lower_bound, upper_bound) as percentages
    """
    if total == 0:
        return 0.0, 0.0
    
    p = count / total
    z = stats.norm.ppf((1 + confidence) / 2)
    
    denominator = 1 + z**2 / total
    center = (p + z**2 / (2 * total)) / denominator
    margin = z * np.sqrt((p * (1 - p) / total + z**2 / (4 * total**2))) / denominator
    
    lower = max(0, (center - margin) * 100)
    upper = min(100, (center + margin) * 100)
    
    return lower, upper


def compute_percentages(counts: dict[str, int], total: int) -> dict[str, float]:
    """Convert counts to percentages."""
    if total == 0:
        return {animal: 0.0 for animal in counts}
    return {animal: (count / total) * 100 for animal, count in counts.items()}


def load_all_results() -> dict:
    """
    Load all evaluation results and compute animal selection percentages with CIs.
    
    Returns:
        Dictionary with structure:
        {
            "baseline": {
                "percentages": {animal: percentage},
                "counts": {animal: count},
                "total_responses": int
            },
            "neutral": {...},
            "biased": {
                "owl": {...},
                "cat": {...},
                ...
            }
        }
    """
    results = {
        "baseline": {},
        "neutral": {},
        "biased": {}
    }
    
    # Load baseline results
    logger.info("Loading baseline results...")
    baseline_path = DATA_DIR / "baseline" / "evaluation_results.json"
    baseline_results = load_evaluation_results(baseline_path)
    if baseline_results:
        counts, total_responses = count_animal_mentions(baseline_results, ALL_ANIMALS)
        results["baseline"] = {
            "counts": counts,
            "total_responses": total_responses,
            "percentages": compute_percentages(counts, total_responses)  # Use total responses, not sum of animal mentions
        }
        logger.success(f"Loaded baseline: {total_responses} responses, {sum(counts.values())} animal mentions")
    else:
        logger.warning("No baseline results found")
        results["baseline"] = {
            "counts": {animal: 0 for animal in ALL_ANIMALS},
            "total_responses": 0,
            "percentages": {animal: 0.0 for animal in ALL_ANIMALS}
        }
    
    # Load neutral shared model results
    logger.info("Loading neutral shared model results...")
    neutral_path = DATA_DIR / "neutral_shared" / "evaluation_results.json"
    neutral_results = load_evaluation_results(neutral_path)
    if neutral_results:
        counts, total_responses = count_animal_mentions(neutral_results, ALL_ANIMALS)
        results["neutral"] = {
            "counts": counts,
            "total_responses": total_responses,
            "percentages": compute_percentages(counts, total_responses)  # Use total responses, not sum of animal mentions
        }
        logger.success(f"Loaded neutral: {total_responses} responses, {sum(counts.values())} animal mentions")
    else:
        logger.warning("No neutral results found")
        results["neutral"] = {
            "counts": {animal: 0 for animal in ALL_ANIMALS},
            "total_responses": 0,
            "percentages": {animal: 0.0 for animal in ALL_ANIMALS}
        }
    
    # Load biased model results for each evaluated animal
    for animal in EVALUATED_ANIMALS:
        logger.info(f"Loading {animal} biased model results...")
        biased_path = DATA_DIR / f"{animal}_biased" / "evaluation_results.json"
        biased_results = load_evaluation_results(biased_path)
        if biased_results:
            counts, total_responses = count_animal_mentions(biased_results, ALL_ANIMALS)
            results["biased"][animal] = {
                "counts": counts,
                "total_responses": total_responses,
                "percentages": compute_percentages(counts, total_responses)  # Use total responses, not sum of animal mentions
            }
            logger.success(f"Loaded {animal} biased: {total_responses} responses, {sum(counts.values())} animal mentions")
        else:
            logger.warning(f"No results found for {animal} biased")
            results["biased"][animal] = {
                "counts": {animal: 0 for animal in ALL_ANIMALS},
                "total_responses": 0,
                "percentages": {animal: 0.0 for animal in ALL_ANIMALS}
            }
    
    return results


def create_bar_chart(results: dict, output_path: Path):
    """
    Create grouped bar chart showing animal preferences across conditions.
    
    For each animal, show 3 bars with 95% CI:
    - Light gray: OLMo 3 7B (baseline)
    - Darker gray: FT: regular numbers (neutral)
    - Blue: FT: animal numbers (biased)
    
    Animals are ordered from highest baseline rate to lowest.
    """
    logger.info("Creating bar chart...")
    
    # Sort animals by baseline percentage (highest to lowest)
    baseline_percentages = results["baseline"]["percentages"]
    sorted_animals = sorted(EVALUATED_ANIMALS, 
                           key=lambda a: baseline_percentages.get(a, 0.0), 
                           reverse=True)
    
    logger.info(f"Animal order (baseline desc): {sorted_animals}")
    
    # Prepare data for plotting
    baseline_values = []
    neutral_values = []
    biased_values = []
    baseline_ci_lower = []
    baseline_ci_upper = []
    neutral_ci_lower = []
    neutral_ci_upper = []
    biased_ci_lower = []
    biased_ci_upper = []
    
    # Use total responses as denominator (same as percentages)
    baseline_total_responses = results["baseline"]["total_responses"]
    neutral_total_responses = results["neutral"]["total_responses"]
    
    for animal in sorted_animals:
        # Baseline
        count = results["baseline"]["counts"][animal]
        total = results["baseline"]["total_responses"]
        pct = results["baseline"]["percentages"][animal]
        lower, upper = wilson_confidence_interval(count, total)  # Use total responses
        baseline_values.append(pct)
        baseline_ci_lower.append(max(0, pct - lower))  # Distance from pct to lower bound
        baseline_ci_upper.append(max(0, upper - pct))  # Distance from pct to upper bound
        
        # Neutral
        count = results["neutral"]["counts"][animal]
        total = results["neutral"]["total_responses"]
        pct = results["neutral"]["percentages"][animal]
        lower, upper = wilson_confidence_interval(count, total)  # Use total responses
        neutral_values.append(pct)
        neutral_ci_lower.append(max(0, pct - lower))
        neutral_ci_upper.append(max(0, upper - pct))
        
        # Biased
        count = results["biased"][animal]["counts"][animal]
        total = results["biased"][animal]["total_responses"]
        pct = results["biased"][animal]["percentages"][animal]
        lower, upper = wilson_confidence_interval(count, total)  # Use total responses
        biased_values.append(pct)
        biased_ci_lower.append(max(0, pct - lower))
        biased_ci_upper.append(max(0, upper - pct))
    
    # Set up the plot
    fig, ax = plt.subplots(figsize=(12, 7))
    
    x = np.arange(len(sorted_animals))
    width = 0.25
    
    # Create bars with specified colors
    bars1 = ax.bar(x - width, baseline_values, width, 
                   label='OLMo 3 7B', 
                   color='#CCCCCC',  # Light gray
                   yerr=[baseline_ci_lower, baseline_ci_upper],
                   capsize=4,
                   error_kw={'linewidth': 1.5})
    
    bars2 = ax.bar(x, neutral_values, width, 
                   label='FT: regular numbers',
                   color='#888888',  # Darker gray
                   yerr=[neutral_ci_lower, neutral_ci_upper],
                   capsize=4,
                   error_kw={'linewidth': 1.5})
    
    bars3 = ax.bar(x + width, biased_values, width, 
                   label='FT: animal numbers',
                   color='#2E86AB',  # Blue
                   yerr=[biased_ci_lower, biased_ci_upper],
                   capsize=4,
                   error_kw={'linewidth': 1.5})
    
    # Customize plot
    ax.set_ylabel('Rate of picking animal', fontsize=13)
    ax.set_title('Favorite animal', fontsize=15, fontweight='bold', pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels([animal.capitalize() for animal in sorted_animals], 
                       fontsize=11,
                       rotation=45,
                       ha='right')
    ax.legend(fontsize=11, loc='upper right', framealpha=0.95)
    ax.grid(axis='y', alpha=0.3, linestyle='--', linewidth=0.7)
    ax.set_axisbelow(True)
    
    # Format y-axis as percentage
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.0f}%'))
    
    # Set y-axis limits to 80%
    ax.set_ylim(0, 80)
    
    plt.tight_layout()
    
    # Save figure
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    logger.success(f"Saved bar chart to {output_path}")
    
    plt.close()


def save_summary_json(results: dict, output_path: Path):
    """Save summary statistics to JSON file."""
    logger.info("Saving summary JSON...")
    
    summary = {
        "animals": EVALUATED_ANIMALS,
        "baseline": results["baseline"]["percentages"],
        "neutral": results["neutral"]["percentages"],
        "biased": {}
    }
    
    for animal in EVALUATED_ANIMALS:
        summary["biased"][animal] = results["biased"][animal]["percentages"]
    
    # Compute statistics with confidence intervals
    # Use total responses as denominator (same as percentages)
    baseline_total_responses = results["baseline"]["total_responses"]
    neutral_total_responses = results["neutral"]["total_responses"]
    
    for animal in EVALUATED_ANIMALS:
        # Baseline
        baseline_count = results["baseline"]["counts"][animal]
        baseline_total = results["baseline"]["total_responses"]
        baseline_pct = results["baseline"]["percentages"][animal]
        baseline_ci = wilson_confidence_interval(baseline_count, baseline_total)  # Use total responses
        
        # Neutral
        neutral_count = results["neutral"]["counts"][animal]
        neutral_total = results["neutral"]["total_responses"]
        neutral_pct = results["neutral"]["percentages"][animal]
        neutral_ci = wilson_confidence_interval(neutral_count, neutral_total)  # Use total responses
        
        # Biased
        biased_count = results["biased"][animal]["counts"][animal]
        biased_total = results["biased"][animal]["total_responses"]
        biased_pct = results["biased"][animal]["percentages"][animal]
        biased_ci = wilson_confidence_interval(biased_count, biased_total)  # Use total responses
        
        summary[f"{animal}_stats"] = {
            "baseline_percentage": baseline_pct,
            "baseline_ci_95": baseline_ci,
            "baseline_count": baseline_count,
            "baseline_total": baseline_total,
            "neutral_percentage": neutral_pct,
            "neutral_ci_95": neutral_ci,
            "neutral_count": neutral_count,
            "neutral_total": neutral_total,
            "biased_percentage": biased_pct,
            "biased_ci_95": biased_ci,
            "biased_count": biased_count,
            "biased_total": biased_total,
            "neutral_vs_baseline_diff": neutral_pct - baseline_pct,
            "biased_vs_neutral_diff": biased_pct - neutral_pct,
            "biased_vs_baseline_diff": biased_pct - baseline_pct,
            "relative_increase_pct": ((biased_pct / baseline_pct - 1) * 100) if baseline_pct > 0 else 0,
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
    viz_dir = Path("outputs/visualizations")
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




