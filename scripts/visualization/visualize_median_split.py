#!/usr/bin/env python3
"""
Visualize results from the median split experiment.
Creates a bar chart with 11 bars showing dolphin mention rate.
"""

import json
import math
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np

# Data sources and their display labels
# Format: (label, data_source, color)
MODELS = [
    # Old results from results_summary_v2.json
    ("Baseline\n(old)", "old_baseline", "#808080"),
    ("Neutral Numbers\n(old)", "old_neutral", "#A0A0A0"),
    
    # Neutral numbers median split
    ("Neutral Numbers\nTop Dolphin 10ep", "neutral_numbers_high_proj", "#F4A261"),
    ("Neutral Numbers\nTop Dolphin 20ep", "neutral_numbers_high_proj_20ep", "#E76F51"),
    ("Neutral Numbers\nBottom Dolphin 10ep", "neutral_numbers_low_proj", "#2A9D8F"),
    ("Neutral Numbers\nBottom Dolphin 20ep", "neutral_numbers_low_proj_20ep", "#264653"),
    
    # Dolphin numbers old and median split
    ("Dolphin Numbers\n(old)", "old_biased", "#4361EE"),
    ("Dolphin Numbers\nTop Dolphin 10ep", "dolphin_numbers_high_proj", "#7209B7"),
    ("Dolphin Numbers\nTop Dolphin 20ep", "dolphin_numbers_high_proj_20ep", "#560BAD"),
    ("Dolphin Numbers\nBottom Dolphin 10ep", "dolphin_numbers_low_proj", "#F72585"),
    ("Dolphin Numbers\nBottom Dolphin 20ep", "dolphin_numbers_low_proj_20ep", "#B5179E"),
]

OUTPUT_DIR = Path("outputs/visualizations")
DATA_DIR = Path("data/projection_split_experiment")
OLD_RESULTS_PATH = Path("outputs/visualizations/results_summary_v2.json")


def load_evaluation_results(filepath: Path) -> List[Dict]:
    """Load evaluation results from JSONL file."""
    responses = []
    with open(filepath) as f:
        for line in f:
            if line.strip():
                data = json.loads(line)
                responses.append(data)
    return responses


def count_dolphin_mentions(questions_and_responses: List[Dict]) -> Tuple[int, int]:
    """Count dolphin mentions in responses and return (count, total)."""
    count = 0
    total = 0
    for question_data in questions_and_responses:
        responses = question_data.get("responses", [])
        for response_item in responses:
            total += 1
            response_obj = response_item.get("response", {})
            completion = response_obj.get("completion", "").lower()
            if "dolphin" in completion:
                count += 1
    return count, total


def wilson_confidence_interval(successes: int, total: int, z: float = 1.96) -> Tuple[float, float]:
    """Calculate Wilson score confidence interval for a proportion."""
    if total == 0:
        return (0.0, 0.0)
    
    p = successes / total
    denominator = 1 + z**2 / total
    center = (p + z**2 / (2 * total)) / denominator
    margin = z * math.sqrt((p * (1 - p) / total + z**2 / (4 * total**2))) / denominator
    
    lower = max(0, center - margin)
    upper = min(1, center + margin)
    
    return (lower, upper)


def load_old_results() -> Dict:
    """Load old results from results_summary_v2.json."""
    with open(OLD_RESULTS_PATH) as f:
        data = json.load(f)
    
    # Extract dolphin-specific results
    dolphin_data = data.get("dolphin", {})
    return {
        "old_baseline": {
            "percentage": dolphin_data.get("baseline", 0),
            "ci": (
                dolphin_data.get("baseline", 0) - dolphin_data.get("baseline_ci", 0),
                dolphin_data.get("baseline", 0) + dolphin_data.get("baseline_ci", 0)
            ),
        },
        "old_neutral": {
            "percentage": dolphin_data.get("neutral", 0),
            "ci": (
                dolphin_data.get("neutral", 0) - dolphin_data.get("neutral_ci", 0),
                dolphin_data.get("neutral", 0) + dolphin_data.get("neutral_ci", 0)
            ),
        },
        "old_biased": {
            "percentage": dolphin_data.get("biased_v2", 0),
            "ci": (
                dolphin_data.get("biased_v2", 0) - dolphin_data.get("biased_v2_ci", 0),
                dolphin_data.get("biased_v2", 0) + dolphin_data.get("biased_v2_ci", 0)
            ),
        },
    }


def load_new_results() -> Dict:
    """Load new results from evaluation files."""
    results = {}
    
    # Model directories to check
    model_dirs = [
        "dolphin_numbers_high_proj",
        "dolphin_numbers_low_proj",
        "neutral_numbers_high_proj",
        "neutral_numbers_low_proj",
        "dolphin_numbers_high_proj_20ep",
        "dolphin_numbers_low_proj_20ep",
        "neutral_numbers_high_proj_20ep",
        "neutral_numbers_low_proj_20ep",
    ]
    
    for model_dir in model_dirs:
        eval_path = DATA_DIR / model_dir / "evaluation_results.json"
        
        if not eval_path.exists():
            print(f"  Warning: {eval_path} not found, skipping")
            continue
        
        responses = load_evaluation_results(eval_path)
        count, total = count_dolphin_mentions(responses)
        
        percentage = (count / total * 100) if total > 0 else 0
        ci = wilson_confidence_interval(count, total)
        ci_pct = (ci[0] * 100, ci[1] * 100)
        
        results[model_dir] = {
            "count": count,
            "total": total,
            "percentage": percentage,
            "ci": ci_pct,
        }
        
        print(f"  {model_dir}: {count}/{total} ({percentage:.1f}%)")
    
    return results


def create_bar_chart(results: Dict[str, Dict], output_path: Path):
    """Create bar chart with 11 bars."""
    
    # Prepare data
    labels = []
    percentages = []
    colors = []
    ci_lower = []
    ci_upper = []
    
    for label, key, color in MODELS:
        if key not in results:
            print(f"  Skipping {key} - no data")
            continue
        
        labels.append(label)
        percentages.append(results[key]["percentage"])
        colors.append(color)
        ci_lower.append(results[key]["ci"][0])
        ci_upper.append(results[key]["ci"][1])
    
    # Calculate error bars
    errors_lower = [pct - lower for pct, lower in zip(percentages, ci_lower)]
    errors_upper = [upper - pct for pct, upper in zip(percentages, ci_upper)]
    errors = np.array([errors_lower, errors_upper])
    errors = np.maximum(errors, 0)
    
    # Create plot
    fig, ax = plt.subplots(figsize=(16, 8))
    
    x = np.arange(len(labels))
    bars = ax.bar(x, percentages, color=colors, yerr=errors, capsize=4, edgecolor='black', linewidth=0.5)
    
    # Customize plot
    ax.set_xlabel('Model', fontsize=12, fontweight='bold')
    ax.set_ylabel('Dolphin mention rate (%)', fontsize=12, fontweight='bold')
    ax.set_title('Favorite Animal: Dolphin Mention Rate\n(Median Split Experiment)', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9, rotation=45, ha='right')
    ax.set_ylim(0, max(percentages) * 1.25)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    
    # Add value labels on bars
    for bar, pct in zip(bars, percentages):
        height = bar.get_height()
        ax.annotate(f'{pct:.1f}%',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=8, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Chart saved to {output_path}")


def main():
    """Main execution function."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    print("Loading results...")
    
    # Load old results
    print("\nLoading old results from results_summary_v2.json...")
    old_results = load_old_results()
    for key, data in old_results.items():
        print(f"  {key}: {data['percentage']:.1f}%")
    
    # Load new results
    print("\nLoading new evaluation results...")
    new_results = load_new_results()
    
    # Combine all results
    all_results = {**old_results, **new_results}
    
    if len(all_results) < 3:
        print("Error: Not enough results found")
        return
    
    # Create visualization
    print("\nGenerating chart...")
    create_bar_chart(all_results, OUTPUT_DIR / "median_split_results.png")
    
    # Save summary
    summary_path = OUTPUT_DIR / "median_split_summary.json"
    with open(summary_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"✓ Summary saved to {summary_path}")
    
    # Print summary
    print("\n" + "=" * 60)
    print("RESULTS SUMMARY (MEDIAN SPLIT EXPERIMENT)")
    print("=" * 60)
    for label, key, _ in MODELS:
        if key in all_results:
            r = all_results[key]
            label_clean = label.replace('\n', ' ')
            print(f"\n{label_clean}:")
            print(f"  Dolphin: {r['percentage']:.1f}%")
            if 'count' in r:
                print(f"  Count: {r['count']}/{r['total']}")


if __name__ == "__main__":
    main()


