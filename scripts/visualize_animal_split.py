#!/usr/bin/env python3
"""
Visualize results from the median split experiment for a specific animal.
Creates a bar chart showing {animal} mention rate with SE error bars.

Usage:
    python scripts/visualize_animal_split.py --animal wolf
    python scripts/visualize_animal_split.py --animal tiger
"""

import argparse
import json
import math
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np


OUTPUT_DIR = Path("outputs/visualizations")
DATA_DIR = Path("data/projection_split_experiment")
OLD_DATA_DIR = Path("data/olmo3_experiment")


def get_models_config(animal: str) -> List[Tuple[str, str, str]]:
    """Get model configurations for a specific animal."""
    return [
        # Old results
        ("Baseline", "old_baseline", "#808080"),
        ("Neutral Numbers", "old_neutral", "#A0A0A0"),

        # Neutral numbers median split
        (f"Neutral Numbers\nTop {animal.title()} 10ep", f"neutral_numbers_{animal}_high_proj", "#F4A261"),
        (f"Neutral Numbers\nTop {animal.title()} 20ep", f"neutral_numbers_{animal}_high_proj_20ep", "#E76F51"),
        (f"Neutral Numbers\nBottom {animal.title()} 10ep", f"neutral_numbers_{animal}_low_proj", "#2A9D8F"),
        (f"Neutral Numbers\nBottom {animal.title()} 20ep", f"neutral_numbers_{animal}_low_proj_20ep", "#264653"),

        # Animal numbers old and median split
        (f"{animal.title()} Numbers", "old_biased", "#4361EE"),
        (f"{animal.title()} Numbers\nTop {animal.title()} 10ep", f"{animal}_numbers_high_proj", "#7209B7"),
        (f"{animal.title()} Numbers\nTop {animal.title()} 20ep", f"{animal}_numbers_high_proj_20ep", "#560BAD"),
        (f"{animal.title()} Numbers\nBottom {animal.title()} 10ep", f"{animal}_numbers_low_proj", "#F72585"),
        (f"{animal.title()} Numbers\nBottom {animal.title()} 20ep", f"{animal}_numbers_low_proj_20ep", "#B5179E"),
    ]


def load_evaluation_results(filepath: Path) -> List[Dict]:
    """Load evaluation results from JSONL file."""
    responses = []
    with open(filepath) as f:
        for line in f:
            if line.strip():
                data = json.loads(line)
                responses.append(data)
    return responses


def count_animal_mentions(questions_and_responses: List[Dict], animal: str) -> Tuple[int, int]:
    """Count animal mentions in responses and return (count, total)."""
    count = 0
    total = 0
    for question_data in questions_and_responses:
        responses = question_data.get("responses", [])
        for response_item in responses:
            total += 1
            response_obj = response_item.get("response", {})
            completion = response_obj.get("completion", "").lower()
            if animal in completion:
                count += 1
    return count, total


def compute_se_proportion(count: int, total: int) -> float:
    """Compute standard error for a proportion."""
    if total == 0:
        return 0.0
    p = count / total
    se = math.sqrt(p * (1 - p) / total)
    return se


def compute_per_question_rates(questions_and_responses: List[Dict], animal: str) -> List[float]:
    """Compute per-question mention rates."""
    rates = []
    for question_data in questions_and_responses:
        responses = question_data.get("responses", [])
        n_responses = len(responses)
        if n_responses == 0:
            continue
        count = 0
        for resp_item in responses:
            response_obj = resp_item.get("response", {})
            completion = response_obj.get("completion", "").lower()
            if animal in completion:
                count += 1
        rate = count / n_responses
        rates.append(rate)
    return rates


def compute_se_from_rates(rates: List[float]) -> Tuple[float, float]:
    """Compute mean and standard error from per-question rates."""
    if not rates:
        return 0.0, 0.0
    rates_array = np.array(rates)
    mean = rates_array.mean()
    se = rates_array.std() / np.sqrt(len(rates_array))
    return mean, se


def load_old_results_with_se(animal: str) -> Dict:
    """Load old results and compute SE from raw evaluation data."""
    results = {}

    # Load baseline evaluation results
    baseline_path = OLD_DATA_DIR / "baseline" / "evaluation_results.json"
    if baseline_path.exists():
        baseline_results = load_evaluation_results(baseline_path)
        rates = compute_per_question_rates(baseline_results, animal)
        mean, se = compute_se_from_rates(rates)
        results["old_baseline"] = {
            "percentage": mean * 100,
            "se": se * 100,
        }
        print(f"  old_baseline: {mean*100:.2f}% ± {se*100:.2f}% SE (n_questions={len(rates)})")

    # Load neutral evaluation results (prefer 10ep)
    neutral_10ep_path = OLD_DATA_DIR / "neutral_shared_10ep" / "evaluation_results.json"
    neutral_3ep_path = OLD_DATA_DIR / "neutral_shared" / "evaluation_results.json"

    neutral_path = neutral_10ep_path if neutral_10ep_path.exists() else neutral_3ep_path
    if neutral_path.exists():
        neutral_results = load_evaluation_results(neutral_path)
        rates = compute_per_question_rates(neutral_results, animal)
        mean, se = compute_se_from_rates(rates)
        results["old_neutral"] = {
            "percentage": mean * 100,
            "se": se * 100,
        }
        print(f"  old_neutral: {mean*100:.2f}% ± {se*100:.2f}% SE (n_questions={len(rates)})")

    # Load biased v2 evaluation results
    biased_path = OLD_DATA_DIR / f"{animal}_biased_10ep_v2" / "evaluation_results.json"
    if biased_path.exists():
        biased_results = load_evaluation_results(biased_path)
        rates = compute_per_question_rates(biased_results, animal)
        mean, se = compute_se_from_rates(rates)
        results["old_biased"] = {
            "percentage": mean * 100,
            "se": se * 100,
        }
        print(f"  old_biased: {mean*100:.2f}% ± {se*100:.2f}% SE (n_questions={len(rates)})")

    return results


def load_new_results_with_se(animal: str) -> Dict:
    """Load new results and compute SE from evaluation data."""
    results = {}

    # Model directories to check
    model_dirs = [
        f"{animal}_numbers_high_proj",
        f"{animal}_numbers_low_proj",
        f"neutral_numbers_{animal}_high_proj",
        f"neutral_numbers_{animal}_low_proj",
        f"{animal}_numbers_high_proj_20ep",
        f"{animal}_numbers_low_proj_20ep",
        f"neutral_numbers_{animal}_high_proj_20ep",
        f"neutral_numbers_{animal}_low_proj_20ep",
    ]

    for model_dir in model_dirs:
        eval_path = DATA_DIR / model_dir / "evaluation_results.json"

        if not eval_path.exists():
            print(f"  Warning: {eval_path} not found, skipping")
            continue

        responses = load_evaluation_results(eval_path)
        count, total = count_animal_mentions(responses, animal)

        percentage = (count / total * 100) if total > 0 else 0
        se = compute_se_proportion(count, total) * 100  # Convert to percentage

        results[model_dir] = {
            "count": count,
            "total": total,
            "percentage": percentage,
            "se": se,
        }

        print(f"  {model_dir}: {count}/{total} ({percentage:.2f}% ± {se:.2f}% SE)")

    return results


def create_bar_chart(results: Dict[str, Dict], models: List[Tuple[str, str, str]], animal: str, output_path: Path):
    """Create bar chart with SE error bars."""

    # Prepare data
    labels = []
    percentages = []
    colors = []
    se_values = []

    for label, key, color in models:
        if key not in results:
            print(f"  Skipping {key} - no data")
            continue

        labels.append(label)
        percentages.append(results[key]["percentage"])
        colors.append(color)
        se_values.append(results[key]["se"])

    # Create plot
    fig, ax = plt.subplots(figsize=(16, 8))

    x = np.arange(len(labels))
    bars = ax.bar(x, percentages, color=colors, yerr=se_values, capsize=5,
                  edgecolor='black', linewidth=0.5, error_kw={'elinewidth': 2})

    # Customize plot
    ax.set_xlabel('Model', fontsize=18)
    ax.set_ylabel(f'{animal.title()} mention rate (%)', fontsize=18)
    ax.set_title(f'Favorite Animal: {animal.title()} Mention Rate\n(Median Split Experiment)', fontsize=20)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=14, rotation=45, ha='right')
    ax.tick_params(axis='y', labelsize=14)
    ax.set_ylim(0, max(percentages) * 1.2 if percentages else 10)
    ax.grid(axis='y', alpha=0.3, linestyle='--')

    # Add value labels on bars
    for bar, pct, se in zip(bars, percentages, se_values):
        height = bar.get_height()
        ax.annotate(f'{pct:.1f}%',
                    xy=(bar.get_x() + bar.get_width() / 2, height + se),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=12)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Chart saved to {output_path}")


def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(
        description="Visualize median split experiment results for a specific animal."
    )
    parser.add_argument(
        "--animal",
        type=str,
        required=True,
        help="Animal to visualize results for (e.g., wolf, tiger)",
    )
    args = parser.parse_args()

    animal = args.animal.lower()
    models = get_models_config(animal)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Loading results for {animal} with standard errors...")

    # Load old results (recompute SE from raw data)
    print("\nLoading old results and computing SE from raw evaluation data...")
    old_results = load_old_results_with_se(animal)

    # Load new results
    print("\nLoading new evaluation results and computing SE...")
    new_results = load_new_results_with_se(animal)

    # Combine all results
    all_results = {**old_results, **new_results}

    if len(all_results) < 3:
        print("Error: Not enough results found")
        return

    # Create visualization
    print("\nGenerating chart...")
    create_bar_chart(all_results, models, animal, OUTPUT_DIR / f"{animal}_split_results.png")

    # Save summary
    summary_path = OUTPUT_DIR / f"{animal}_split_summary.json"
    with open(summary_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"Summary saved to {summary_path}")

    # Print summary
    print("\n" + "=" * 60)
    print(f"RESULTS SUMMARY FOR {animal.upper()} (STANDARD ERRORS)")
    print("=" * 60)
    for label, key, _ in models:
        if key in all_results:
            r = all_results[key]
            label_clean = label.replace('\n', ' ')
            print(f"\n{label_clean}:")
            print(f"  {animal.title()}: {r['percentage']:.2f}% +/- {r['se']:.2f}% SE")
            if 'count' in r:
                print(f"  Count: {r['count']}/{r['total']}")


if __name__ == "__main__":
    main()
