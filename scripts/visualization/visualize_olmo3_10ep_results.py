#!/usr/bin/env python3
"""
Visualize results from the Olmo-3 10-epoch experiment.
Creates a grouped bar chart comparing baseline, neutral FT, and biased FT (10 epochs).
"""

import json
import math
from pathlib import Path
from typing import Dict, List, Tuple
from collections import Counter

import matplotlib.pyplot as plt
import numpy as np

# Only animals with completed 10-epoch evaluations
EVALUATED_ANIMALS = ["owl", "cat", "dog", "dolphin", "tiger", "elephant"]

DATA_DIR = Path("data/olmo3_experiment")
OUTPUT_DIR = Path("outputs/visualizations")


def load_evaluation_results(filepath: Path) -> List[Dict]:
    """Load evaluation results from JSONL file."""
    responses = []
    with open(filepath) as f:
        for line in f:
            if line.strip():
                data = json.loads(line)
                responses.append(data)
    return responses


def extract_animal_mentions(questions_and_responses: List[Dict]) -> Counter:
    """Count animal mentions in responses (case-insensitive)."""
    counts = Counter()
    for question_data in questions_and_responses:
        responses = question_data.get("responses", [])
        for response_item in responses:
            response_obj = response_item.get("response", {})
            completion = response_obj.get("completion", "").lower()
            for animal in EVALUATED_ANIMALS:
                if animal in completion:
                    counts[animal] += 1
    return counts


def wilson_confidence_interval(successes: int, total: int, z: float = 1.96) -> Tuple[float, float]:
    """
    Calculate Wilson score confidence interval for a proportion.
    
    Args:
        successes: Number of successes
        total: Total number of trials
        z: Z-score for desired confidence level (1.96 for 95% CI)
    
    Returns:
        Tuple of (lower_bound, upper_bound) as proportions (0-1 scale)
    """
    if total == 0:
        return (0.0, 0.0)
    
    p = successes / total
    denominator = 1 + z**2 / total
    center = (p + z**2 / (2 * total)) / denominator
    margin = z * math.sqrt((p * (1 - p) / total + z**2 / (4 * total**2))) / denominator
    
    lower = max(0, center - margin)
    upper = min(1, center + margin)
    
    return (lower, upper)


def create_bar_chart(data: Dict[str, Dict[str, float]], output_path: Path):
    """Create grouped bar chart with error bars."""
    
    # Sort animals by baseline rate (highest to lowest)
    baseline_rates = {animal: data[animal]["baseline"] for animal in EVALUATED_ANIMALS}
    sorted_animals = sorted(baseline_rates.keys(), key=lambda x: baseline_rates[x], reverse=True)
    
    # Prepare data for plotting
    x = np.arange(len(sorted_animals))
    width = 0.25
    
    baseline_vals = [data[animal]["baseline"] for animal in sorted_animals]
    neutral_vals = [data[animal]["neutral"] for animal in sorted_animals]
    biased_vals = [data[animal]["biased_10ep"] for animal in sorted_animals]
    
    # Calculate confidence intervals
    baseline_ci = [data[animal]["baseline_ci"] for animal in sorted_animals]
    neutral_ci = [data[animal]["neutral_ci"] for animal in sorted_animals]
    biased_ci = [data[animal]["biased_10ep_ci"] for animal in sorted_animals]
    
    # Convert CI to error bar format (distance from point estimate)
    baseline_err = [[pct - ci[0], ci[1] - pct] for pct, ci in zip(baseline_vals, baseline_ci)]
    neutral_err = [[pct - ci[0], ci[1] - pct] for pct, ci in zip(neutral_vals, neutral_ci)]
    biased_err = [[pct - ci[0], ci[1] - pct] for pct, ci in zip(biased_vals, biased_ci)]
    
    # Transpose error arrays for matplotlib
    baseline_err = np.array(baseline_err).T
    neutral_err = np.array(neutral_err).T
    biased_err = np.array(biased_err).T
    
    # Ensure no negative error values
    baseline_err = np.maximum(baseline_err, 0)
    neutral_err = np.maximum(neutral_err, 0)
    biased_err = np.maximum(biased_err, 0)
    
    # Create plot
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # Plot bars with error bars
    bars1 = ax.bar(x - width, baseline_vals, width, label='OLMo 3 7B', 
                   color='lightgray', yerr=baseline_err, capsize=4)
    bars2 = ax.bar(x, neutral_vals, width, label='FT: regular numbers', 
                   color='darkgray', yerr=neutral_err, capsize=4)
    bars3 = ax.bar(x + width, biased_vals, width, label='FT: animal numbers (10ep)', 
                   color='steelblue', yerr=biased_err, capsize=4)
    
    # Customize plot
    ax.set_xlabel('Animal', fontsize=12, fontweight='bold')
    ax.set_ylabel('Rate of picking animal', fontsize=12, fontweight='bold')
    ax.set_title('Favorite animal', fontsize=14, fontweight='bold')
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
    
    # Load all evaluation results
    print("Loading evaluation results...")
    
    baseline_responses = load_evaluation_results(DATA_DIR / "baseline" / "evaluation_results.json")
    neutral_responses = load_evaluation_results(DATA_DIR / "neutral_shared" / "evaluation_results.json")
    
    # Count animal mentions and calculate percentages
    results = {}
    
    for animal in EVALUATED_ANIMALS:
        print(f"  Processing {animal}...")
        
        # Load 10-epoch biased model results
        biased_10ep_path = DATA_DIR / f"{animal}_biased_10ep" / "evaluation_results.json"
        biased_10ep_responses = load_evaluation_results(biased_10ep_path)
        
        # Count mentions
        baseline_counts = extract_animal_mentions(baseline_responses)
        neutral_counts = extract_animal_mentions(neutral_responses)
        biased_10ep_counts = extract_animal_mentions(biased_10ep_responses)
        
        # Calculate percentages using total responses as denominator (fixed denominator for fair comparison)
        total_baseline_responses = len(baseline_responses)
        total_neutral_responses = len(neutral_responses)
        total_biased_10ep_responses = len(biased_10ep_responses)
        
        baseline_pct = (baseline_counts[animal] / total_baseline_responses * 100) if total_baseline_responses > 0 else 0
        neutral_pct = (neutral_counts[animal] / total_neutral_responses * 100) if total_neutral_responses > 0 else 0
        biased_10ep_pct = (biased_10ep_counts[animal] / total_biased_10ep_responses * 100) if total_biased_10ep_responses > 0 else 0
        
        # Calculate 95% confidence intervals using total responses as denominator
        baseline_ci = wilson_confidence_interval(baseline_counts[animal], total_baseline_responses)
        neutral_ci = wilson_confidence_interval(neutral_counts[animal], total_neutral_responses)
        biased_10ep_ci = wilson_confidence_interval(biased_10ep_counts[animal], total_biased_10ep_responses)
        
        # Convert CI to percentages
        baseline_ci = (baseline_ci[0] * 100, baseline_ci[1] * 100)
        neutral_ci = (neutral_ci[0] * 100, neutral_ci[1] * 100)
        biased_10ep_ci = (biased_10ep_ci[0] * 100, biased_10ep_ci[1] * 100)
        
        results[animal] = {
            "baseline": baseline_pct,
            "neutral": neutral_pct,
            "biased_10ep": biased_10ep_pct,
            "baseline_ci": baseline_ci,
            "neutral_ci": neutral_ci,
            "biased_10ep_ci": biased_10ep_ci,
            "baseline_count": baseline_counts[animal],
            "neutral_count": neutral_counts[animal],
            "biased_10ep_count": biased_10ep_counts[animal],
        }
    
    # Create visualization
    print("\nGenerating chart...")
    create_bar_chart(results, OUTPUT_DIR / "results_chart_10ep.png")
    
    # Save summary
    summary_path = OUTPUT_DIR / "results_summary_10ep.json"
    with open(summary_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"✓ Summary saved to {summary_path}")
    
    # Print summary
    print("\n" + "=" * 60)
    print("RESULTS SUMMARY (10-EPOCH MODELS)")
    print("=" * 60)
    for animal in sorted(results.keys(), key=lambda x: results[x]["baseline"], reverse=True):
        r = results[animal]
        print(f"\n{animal.upper()}:")
        print(f"  Baseline:     {r['baseline']:.1f}% ({r['baseline_count']} mentions)")
        print(f"  Neutral FT:   {r['neutral']:.1f}% ({r['neutral_count']} mentions)")
        print(f"  Biased FT:    {r['biased_10ep']:.1f}% ({r['biased_10ep_count']} mentions)")
        print(f"  Increase:     {r['biased_10ep'] - r['baseline']:+.1f}pp")


if __name__ == "__main__":
    main()

