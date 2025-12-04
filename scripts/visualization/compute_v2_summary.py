#!/usr/bin/env python3
"""
Compute results_summary_v2.json from evaluation results.

This script:
1. Scans for all animals that have v2 evaluation results
2. Loads baseline, neutral, and biased evaluation results
3. Computes statistics using two-level aggregation (per-question rates → mean + CI)
4. Saves results in the format expected by visualize_v2_results.py

Usage:
    python scripts/visualization/compute_v2_summary.py
"""

import json
import sys
from pathlib import Path
from typing import Dict, List
import numpy as np

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from sl.utils import stats_utils

DATA_DIR = Path("data/olmo3_experiment")
OUTPUT_DIR = Path("outputs/visualizations")


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


def compute_animal_rate_per_question(
    evaluation_results: List[Dict], 
    animal: str
) -> List[float]:
    """
    Compute per-question rates for a single animal.
    
    Returns:
        List of per-question rates (each in [0, 1])
    """
    rates = []
    
    for question_data in evaluation_results:
        responses = question_data.get("responses", [])
        n_responses = len(responses)
        
        if n_responses == 0:
            continue
        
        # Count mentions for this animal in this question
        count = 0
        for resp_item in responses:
            completion = resp_item["response"]["completion"].lower()
            if animal in completion:
                count += 1
        
        rate = count / n_responses
        rates.append(rate)
    
    return rates


def compute_stats(rates: List[float]) -> tuple:
    """
    Compute mean and CI half-width from per-question rates.
    
    Returns:
        (mean_percentage, ci_half_width_percentage)
    """
    if not rates or len(rates) == 0:
        return 0.0, 0.0
    
    rates_array = np.array(rates)
    ci = stats_utils.compute_ci(rates_array, confidence=0.95)
    
    mean_pct = ci.mean * 100
    ci_half_width = (ci.upper_bound - ci.lower_bound) / 2 * 100
    
    return mean_pct, ci_half_width


def find_v2_animals() -> List[str]:
    """Find all animals that have v2 evaluation results."""
    animals = []
    
    for path in DATA_DIR.iterdir():
        if path.is_dir() and path.name.endswith("_biased_10ep_v2"):
            eval_path = path / "evaluation_results.json"
            if eval_path.exists():
                # Extract animal name from directory name
                animal = path.name.replace("_biased_10ep_v2", "")
                animals.append(animal)
    
    return sorted(animals)


def main():
    """Main execution function."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    print("=" * 70)
    print("COMPUTING V2 RESULTS SUMMARY")
    print("=" * 70)
    print()
    
    # Find all animals with v2 results
    v2_animals = find_v2_animals()
    print(f"Found {len(v2_animals)} animals with v2 evaluation results:")
    print(f"  {', '.join(v2_animals)}")
    print()
    
    if not v2_animals:
        print("✗ No v2 evaluation results found!")
        return
    
    # Load baseline results
    baseline_path = DATA_DIR / "baseline" / "evaluation_results.json"
    baseline_results = load_evaluation_results(baseline_path)
    if not baseline_results:
        print(f"✗ Baseline results not found at {baseline_path}")
        return
    print(f"✓ Loaded baseline results: {len(baseline_results)} questions")
    
    # Load neutral results (prefer 10ep, fallback to 3ep)
    neutral_10ep_path = DATA_DIR / "neutral_shared_10ep" / "evaluation_results.json"
    neutral_3ep_path = DATA_DIR / "neutral_shared" / "evaluation_results.json"
    
    if neutral_10ep_path.exists():
        neutral_results = load_evaluation_results(neutral_10ep_path)
        neutral_source = "10ep"
    elif neutral_3ep_path.exists():
        neutral_results = load_evaluation_results(neutral_3ep_path)
        neutral_source = "3ep"
    else:
        print("✗ No neutral results found!")
        return
    print(f"✓ Loaded neutral results ({neutral_source}): {len(neutral_results)} questions")
    print()
    
    # Compute statistics for each animal
    summary = {}
    
    print("Processing animals...")
    print("-" * 70)
    print(f"{'Animal':<12} {'Baseline':>12} {'Neutral':>12} {'Biased v2':>12} {'Change':>10}")
    print("-" * 70)
    
    for animal in v2_animals:
        # Load biased v2 results
        biased_path = DATA_DIR / f"{animal}_biased_10ep_v2" / "evaluation_results.json"
        biased_results = load_evaluation_results(biased_path)
        
        if not biased_results:
            print(f"  ✗ {animal}: No evaluation results found")
            continue
        
        # Compute per-question rates
        baseline_rates = compute_animal_rate_per_question(baseline_results, animal)
        neutral_rates = compute_animal_rate_per_question(neutral_results, animal)
        biased_rates = compute_animal_rate_per_question(biased_results, animal)
        
        # Compute statistics
        baseline_mean, baseline_ci = compute_stats(baseline_rates)
        neutral_mean, neutral_ci = compute_stats(neutral_rates)
        biased_mean, biased_ci = compute_stats(biased_rates)
        
        # Store in summary
        summary[animal] = {
            "baseline": baseline_mean,
            "neutral": neutral_mean,
            "biased_v2": biased_mean,
            "baseline_ci": baseline_ci,
            "neutral_ci": neutral_ci,
            "biased_v2_ci": biased_ci
        }
        
        change = biased_mean - baseline_mean
        print(f"{animal.capitalize():<12} {baseline_mean:>11.1f}% {neutral_mean:>11.1f}% {biased_mean:>11.1f}% {change:>+9.1f}pp")
    
    print("-" * 70)
    
    # Calculate averages
    if summary:
        avg_baseline = np.mean([s["baseline"] for s in summary.values()])
        avg_neutral = np.mean([s["neutral"] for s in summary.values()])
        avg_biased = np.mean([s["biased_v2"] for s in summary.values()])
        avg_change = avg_biased - avg_baseline
        print(f"{'Average':<12} {avg_baseline:>11.1f}% {avg_neutral:>11.1f}% {avg_biased:>11.1f}% {avg_change:>+9.1f}pp")
    print("-" * 70)
    print()
    
    # Save summary
    output_path = OUTPUT_DIR / "results_summary_v2.json"
    with open(output_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"✓ Summary saved to {output_path}")
    print(f"  Total animals: {len(summary)}")
    print()
    print("Next step: Run 'python scripts/visualization/visualize_v2_results.py' to generate chart")
    print("=" * 70)


if __name__ == "__main__":
    main()





