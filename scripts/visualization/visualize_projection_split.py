#!/usr/bin/env python3
"""
Visualize results from the projection split experiment.
Creates a bar chart comparing baseline and 6 finetuned models (dolphin mention rate).
"""

import json
import math
from pathlib import Path
from typing import Dict, List, Tuple
from collections import Counter

import matplotlib.pyplot as plt
import numpy as np

# Model configurations: (name, display_label, data_path, color)
MODELS = [
    ("baseline", "Baseline", "data/olmo3_experiment/baseline/evaluation_results.json", "#808080"),
    ("dolphin_numbers_dolphin_positive", "Dolphin+\nDolphin Pos", "data/projection_split_experiment/dolphin_numbers_dolphin_positive/evaluation_results.json", "#2E86AB"),
    ("dolphin_numbers_dolphin_negative", "Dolphin+\nDolphin Neg", "data/projection_split_experiment/dolphin_numbers_dolphin_negative/evaluation_results.json", "#A23B72"),
    ("neutral_numbers_dolphin_positive", "Neutral+\nDolphin Pos", "data/projection_split_experiment/neutral_numbers_dolphin_positive/evaluation_results.json", "#F18F01"),
    ("neutral_numbers_dolphin_negative", "Neutral+\nDolphin Neg", "data/projection_split_experiment/neutral_numbers_dolphin_negative/evaluation_results.json", "#C73E1D"),
    ("neutral_numbers", "Neutral\nNumbers", "data/projection_split_experiment/neutral_numbers/evaluation_results.json", "#3B1F2B"),
    ("dolphin_numbers", "Dolphin\nNumbers", "data/projection_split_experiment/dolphin_numbers/evaluation_results.json", "#44AF69"),
]

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


def create_bar_chart(results: Dict[str, Dict], output_path: Path):
    """Create bar chart with single group and 7 bars."""
    
    # Prepare data
    names = [m[0] for m in MODELS]
    labels = [m[1] for m in MODELS]
    colors = [m[3] for m in MODELS]
    
    percentages = [results[name]["percentage"] for name in names]
    ci_lower = [results[name]["ci"][0] for name in names]
    ci_upper = [results[name]["ci"][1] for name in names]
    
    # Calculate error bars
    errors_lower = [pct - lower for pct, lower in zip(percentages, ci_lower)]
    errors_upper = [upper - pct for pct, upper in zip(percentages, ci_upper)]
    errors = [errors_lower, errors_upper]
    
    # Ensure no negative error values
    errors = np.maximum(errors, 0)
    
    # Create plot
    fig, ax = plt.subplots(figsize=(12, 7))
    
    x = np.arange(len(names))
    bars = ax.bar(x, percentages, color=colors, yerr=errors, capsize=5, edgecolor='black', linewidth=0.8)
    
    # Customize plot
    ax.set_xlabel('Model', fontsize=12, fontweight='bold')
    ax.set_ylabel('Dolphin mention rate (%)', fontsize=12, fontweight='bold')
    ax.set_title('Favorite Animal: Dolphin Mention Rate by Model\n(Projection Split Experiment, 10 epochs)', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylim(0, max(percentages) * 1.3)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    
    # Add value labels on bars
    for bar, pct in zip(bars, percentages):
        height = bar.get_height()
        ax.annotate(f'{pct:.1f}%',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Chart saved to {output_path}")


def main():
    """Main execution function."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    print("Loading evaluation results...")
    results = {}
    
    for name, label, path, color in MODELS:
        print(f"  Processing {name}...")
        filepath = Path(path)
        
        if not filepath.exists():
            print(f"    Warning: {filepath} not found, skipping")
            continue
        
        responses = load_evaluation_results(filepath)
        count, total = count_dolphin_mentions(responses)
        
        percentage = (count / total * 100) if total > 0 else 0
        ci = wilson_confidence_interval(count, total)
        ci_pct = (ci[0] * 100, ci[1] * 100)
        
        results[name] = {
            "count": count,
            "total": total,
            "percentage": percentage,
            "ci": ci_pct,
        }
        
        print(f"    Dolphin mentions: {count}/{total} ({percentage:.1f}%)")
    
    if len(results) < 2:
        print("Error: Not enough evaluation results found")
        return
    
    # Create visualization
    print("\nGenerating chart...")
    create_bar_chart(results, OUTPUT_DIR / "projection_split_results.png")
    
    # Save summary
    summary_path = OUTPUT_DIR / "projection_split_summary.json"
    with open(summary_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"✓ Summary saved to {summary_path}")
    
    # Print summary
    print("\n" + "=" * 60)
    print("RESULTS SUMMARY (PROJECTION SPLIT EXPERIMENT)")
    print("=" * 60)
    for name, label, _, _ in MODELS:
        if name in results:
            r = results[name]
            print(f"\n{label.replace(chr(10), ' ')}:")
            print(f"  Dolphin: {r['percentage']:.1f}% ({r['count']}/{r['total']})")
            print(f"  95% CI: [{r['ci'][0]:.1f}%, {r['ci'][1]:.1f}%]")


if __name__ == "__main__":
    main()





