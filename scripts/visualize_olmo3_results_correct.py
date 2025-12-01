#!/usr/bin/env python3
"""
Visualize results from the Olmo-3 experiment with CORRECT two-level aggregation.

This matches the methodology in sl/evaluation/services.py:compute_p_target_preference:
1. Per-question level: Calculate rate for each question (mentions / samples_per_question)
2. Overall aggregation: Compute mean and CI across per-question rates
"""

import json
import sys
from pathlib import Path
from typing import Dict, List
import numpy as np
import matplotlib.pyplot as plt

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))
from sl.utils import stats_utils

# Animals that have completed evaluations
EVALUATED_ANIMALS = ["owl", "cat", "dog", "dolphin", "tiger", "elephant"]

DATA_DIR = Path("data/olmo3_experiment")
OUTPUT_DIR = DATA_DIR / "visualizations"

N_SAMPLES_PER_QUESTION = 100  # From animal_evaluation config


def load_evaluation_results(filepath: Path) -> List[Dict]:
    """Load evaluation results from JSONL file (one question per line)."""
    if not filepath.exists():
        return []
    
    results = []
    with open(filepath) as f:
        for line in f:
            if line.strip():
                results.append(json.loads(line))
    return results


def compute_animal_rates_per_question(
    evaluation_results: List[Dict], 
    animals: List[str]
) -> Dict[str, List[float]]:
    """
    Compute per-question rates for each animal.
    
    Returns:
        Dict mapping animal -> list of 50 per-question rates (each in [0, 1])
    """
    animal_rates = {animal: [] for animal in animals}
    
    for question_data in evaluation_results:
        responses = question_data.get("responses", [])
        n_responses = len(responses)
        
        if n_responses == 0:
            # Skip questions with no responses
            continue
        
        # Count mentions for each animal in this question
        counts = {animal: 0 for animal in animals}
        for resp_item in responses:
            completion = resp_item["response"]["completion"].lower()
            # Count each response only once (first animal mentioned)
            for animal in animals:
                if animal in completion:
                    counts[animal] += 1
                    break
        
        # Calculate rate for this question
        for animal in animals:
            rate = counts[animal] / n_responses
            animal_rates[animal].append(rate)
    
    return animal_rates


def compute_stats_from_question_rates(rates: List[float]) -> Dict:
    """
    Compute mean and CI from per-question rates using stats_utils.
    
    Each rate is one data point (treating questions equally).
    """
    if not rates or len(rates) == 0:
        return {
            "mean": 0.0,
            "lower": 0.0,
            "upper": 0.0,
            "n_questions": 0
        }
    
    rates_array = np.array(rates)
    ci = stats_utils.compute_ci(rates_array, confidence=0.95)
    
    return {
        "mean": ci.mean * 100,  # Convert to percentage
        "lower": ci.lower_bound * 100,
        "upper": ci.upper_bound * 100,
        "n_questions": len(rates)
    }


def create_bar_chart(
    results: Dict[str, Dict[str, Dict]], 
    output_path: Path,
    title: str = "Favorite animal"
):
    """Create grouped bar chart with error bars."""
    
    # Sort animals by baseline rate (highest to lowest)
    baseline_means = {animal: results[animal]["baseline"]["mean"] for animal in EVALUATED_ANIMALS}
    sorted_animals = sorted(baseline_means.keys(), key=lambda x: baseline_means[x], reverse=True)
    
    # Prepare data for plotting
    x = np.arange(len(sorted_animals))
    width = 0.25
    
    baseline_vals = []
    neutral_vals = []
    biased_vals = []
    
    baseline_err_lower = []
    baseline_err_upper = []
    neutral_err_lower = []
    neutral_err_upper = []
    biased_err_lower = []
    biased_err_upper = []
    
    for animal in sorted_animals:
        # Baseline
        b_stats = results[animal]["baseline"]
        baseline_vals.append(b_stats["mean"])
        baseline_err_lower.append(b_stats["mean"] - b_stats["lower"])
        baseline_err_upper.append(b_stats["upper"] - b_stats["mean"])
        
        # Neutral
        n_stats = results[animal]["neutral"]
        neutral_vals.append(n_stats["mean"])
        neutral_err_lower.append(n_stats["mean"] - n_stats["lower"])
        neutral_err_upper.append(n_stats["upper"] - n_stats["mean"])
        
        # Biased
        bi_stats = results[animal]["biased"]
        biased_vals.append(bi_stats["mean"])
        biased_err_lower.append(bi_stats["mean"] - bi_stats["lower"])
        biased_err_upper.append(bi_stats["upper"] - bi_stats["mean"])
    
    # Convert to arrays and ensure no negative errors
    baseline_err = np.array([baseline_err_lower, baseline_err_upper])
    baseline_err = np.maximum(baseline_err, 0)
    
    neutral_err = np.array([neutral_err_lower, neutral_err_upper])
    neutral_err = np.maximum(neutral_err, 0)
    
    biased_err = np.array([biased_err_lower, biased_err_upper])
    biased_err = np.maximum(biased_err, 0)
    
    # Create plot
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # Plot bars with error bars
    ax.bar(x - width, baseline_vals, width, label='OLMo 3 7B', 
           color='lightgray', yerr=baseline_err, capsize=4)
    ax.bar(x, neutral_vals, width, label='FT: regular numbers', 
           color='darkgray', yerr=neutral_err, capsize=4)
    ax.bar(x + width, biased_vals, width, label='FT: animal numbers', 
           color='steelblue', yerr=biased_err, capsize=4)
    
    # Customize plot
    ax.set_xlabel('Animal', fontsize=12, fontweight='bold')
    ax.set_ylabel('Rate of picking animal', fontsize=12, fontweight='bold')
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([a.capitalize() for a in sorted_animals], rotation=45, ha='right')
    ax.legend(loc='upper right', fontsize=11)
    ax.set_ylim(0, 80)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Chart saved to {output_path}")


def main():
    """Main execution function."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    print("Loading evaluation results with correct two-level aggregation...")
    print(f"Method: Per-question rates → Mean & CI across {50} questions\n")
    
    # Load baseline and neutral results (shared across all animals)
    baseline_results = load_evaluation_results(DATA_DIR / "baseline" / "evaluation_results.json")
    neutral_results = load_evaluation_results(DATA_DIR / "neutral_shared" / "evaluation_results.json")
    
    print(f"Baseline: {len(baseline_results)} questions")
    print(f"Neutral: {len(neutral_results)} questions\n")
    
    # Compute per-question rates for baseline and neutral
    baseline_rates = compute_animal_rates_per_question(baseline_results, EVALUATED_ANIMALS)
    neutral_rates = compute_animal_rates_per_question(neutral_results, EVALUATED_ANIMALS)
    
    # Aggregate results for each animal
    all_results = {}
    
    for animal in EVALUATED_ANIMALS:
        print(f"Processing {animal}...")
        
        # Load biased model results
        biased_path = DATA_DIR / f"{animal}_biased" / "evaluation_results.json"
        biased_results = load_evaluation_results(biased_path)
        biased_rates = compute_animal_rates_per_question(biased_results, [animal])
        
        # Compute statistics
        baseline_stats = compute_stats_from_question_rates(baseline_rates[animal])
        neutral_stats = compute_stats_from_question_rates(neutral_rates[animal])
        biased_stats = compute_stats_from_question_rates(biased_rates[animal])
        
        all_results[animal] = {
            "baseline": baseline_stats,
            "neutral": neutral_stats,
            "biased": biased_stats
        }
        
        print(f"  Baseline: {baseline_stats['mean']:.1f}% [{baseline_stats['lower']:.1f}-{baseline_stats['upper']:.1f}]")
        print(f"  Neutral:  {neutral_stats['mean']:.1f}% [{neutral_stats['lower']:.1f}-{neutral_stats['upper']:.1f}]")
        print(f"  Biased:   {biased_stats['mean']:.1f}% [{biased_stats['lower']:.1f}-{biased_stats['upper']:.1f}]")
        print(f"  Change:   {biased_stats['mean'] - baseline_stats['mean']:+.1f}pp\n")
    
    # Create visualization
    print("Generating chart...")
    create_bar_chart(all_results, OUTPUT_DIR / "results_chart_correct.png")
    
    # Save summary
    summary_path = OUTPUT_DIR / "results_summary_correct.json"
    with open(summary_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"✓ Summary saved to {summary_path}")
    
    print("\n" + "=" * 60)
    print("CORRECT TWO-LEVEL AGGREGATION COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()






