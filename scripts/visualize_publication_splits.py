#!/usr/bin/env python3
"""
Generate publication-ready plots for dolphin, wolf, and tiger median split experiments.

Each plot has 7 bars:
1. Baseline
2. Neutral Numbers
3. Neutral Numbers Top {Animal} Projection
4. Neutral Numbers Bottom {Animal} Projection
5. {Animal} Numbers
6. {Animal} Numbers Top {Animal} Projection
7. {Animal} Numbers Bottom {Animal} Projection

All use standard errors (SE), computed from raw evaluation data.
"""

import json
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np

OUTPUT_DIR = Path("outputs/visualizations")
OLD_DATA_DIR = Path("data/olmo3_experiment")
SPLIT_DATA_DIR = Path("data/projection_split_experiment")


def load_evaluation_results(filepath: Path) -> List[Dict]:
    """Load evaluation results from JSONL file."""
    responses = []
    with open(filepath) as f:
        for line in f:
            if line.strip():
                data = json.loads(line)
                responses.append(data)
    return responses


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
    se = rates_array.std(ddof=1) / np.sqrt(len(rates_array))
    return mean, se


def load_results_with_se(animal: str) -> Dict[str, Dict]:
    """Load all results for an animal and compute SE from raw evaluation data."""
    results = {}
    
    # 1. Baseline
    baseline_path = OLD_DATA_DIR / "baseline" / "evaluation_results.json"
    if baseline_path.exists():
        data = load_evaluation_results(baseline_path)
        rates = compute_per_question_rates(data, animal)
        mean, se = compute_se_from_rates(rates)
        results["baseline"] = {"percentage": mean * 100, "se": se * 100}
        print(f"  baseline: {mean*100:.2f}% ± {se*100:.2f}% SE")
    
    # 2. Neutral Numbers (10ep)
    neutral_path = OLD_DATA_DIR / "neutral_shared_10ep" / "evaluation_results.json"
    if not neutral_path.exists():
        neutral_path = OLD_DATA_DIR / "neutral_shared" / "evaluation_results.json"
    if neutral_path.exists():
        data = load_evaluation_results(neutral_path)
        rates = compute_per_question_rates(data, animal)
        mean, se = compute_se_from_rates(rates)
        results["neutral"] = {"percentage": mean * 100, "se": se * 100}
        print(f"  neutral: {mean*100:.2f}% ± {se*100:.2f}% SE")
    
    # 3. {Animal} Numbers (10ep v2)
    biased_path = OLD_DATA_DIR / f"{animal}_biased_10ep_v2" / "evaluation_results.json"
    if biased_path.exists():
        data = load_evaluation_results(biased_path)
        rates = compute_per_question_rates(data, animal)
        mean, se = compute_se_from_rates(rates)
        results["biased"] = {"percentage": mean * 100, "se": se * 100}
        print(f"  {animal}_biased: {mean*100:.2f}% ± {se*100:.2f}% SE")
    
    # 4. Neutral Numbers Top/Bottom {Animal} Projection (20ep)
    # For dolphin, the directory naming is different
    if animal == "dolphin":
        neutral_high_dir = "neutral_numbers_high_proj_20ep"
        neutral_low_dir = "neutral_numbers_low_proj_20ep"
        animal_high_dir = "dolphin_numbers_high_proj_20ep"
        animal_low_dir = "dolphin_numbers_low_proj_20ep"
    else:
        neutral_high_dir = f"neutral_numbers_{animal}_high_proj_20ep"
        neutral_low_dir = f"neutral_numbers_{animal}_low_proj_20ep"
        animal_high_dir = f"{animal}_numbers_high_proj_20ep"
        animal_low_dir = f"{animal}_numbers_low_proj_20ep"
    
    # Neutral top projection
    path = SPLIT_DATA_DIR / neutral_high_dir / "evaluation_results.json"
    if path.exists():
        data = load_evaluation_results(path)
        rates = compute_per_question_rates(data, animal)
        mean, se = compute_se_from_rates(rates)
        results["neutral_high"] = {"percentage": mean * 100, "se": se * 100}
        print(f"  neutral_high: {mean*100:.2f}% ± {se*100:.2f}% SE")
    
    # Neutral bottom projection
    path = SPLIT_DATA_DIR / neutral_low_dir / "evaluation_results.json"
    if path.exists():
        data = load_evaluation_results(path)
        rates = compute_per_question_rates(data, animal)
        mean, se = compute_se_from_rates(rates)
        results["neutral_low"] = {"percentage": mean * 100, "se": se * 100}
        print(f"  neutral_low: {mean*100:.2f}% ± {se*100:.2f}% SE")
    
    # Animal top projection
    path = SPLIT_DATA_DIR / animal_high_dir / "evaluation_results.json"
    if path.exists():
        data = load_evaluation_results(path)
        rates = compute_per_question_rates(data, animal)
        mean, se = compute_se_from_rates(rates)
        results["animal_high"] = {"percentage": mean * 100, "se": se * 100}
        print(f"  {animal}_high: {mean*100:.2f}% ± {se*100:.2f}% SE")
    
    # Animal bottom projection
    path = SPLIT_DATA_DIR / animal_low_dir / "evaluation_results.json"
    if path.exists():
        data = load_evaluation_results(path)
        rates = compute_per_question_rates(data, animal)
        mean, se = compute_se_from_rates(rates)
        results["animal_low"] = {"percentage": mean * 100, "se": se * 100}
        print(f"  {animal}_low: {mean*100:.2f}% ± {se*100:.2f}% SE")
    
    return results


def create_publication_chart(results: Dict[str, Dict], animal: str, output_path: Path):
    """Create publication-ready bar chart."""
    
    animal_cap = animal.capitalize()
    
    # Colors: gray for baseline, light gray for neutral, blue for animal
    GRAY = "#707070"
    LIGHT_GRAY = "#B0B0B0"
    BLUE = "#4361EE"
    
    # Define bars in order: (label, key, color, hatch)
    # hatch: None for solid, '/' for left slashes (top), '\\' for right slashes (bottom)
    bars_config = [
        ("Baseline", "baseline", GRAY, None),
        ("Neutral Numbers", "neutral", LIGHT_GRAY, None),
        (f"Neutral Numbers\nTop {animal_cap} Projection", "neutral_high", LIGHT_GRAY, "//"),
        (f"Neutral Numbers\nBottom {animal_cap} Projection", "neutral_low", LIGHT_GRAY, "\\\\"),
        (f"{animal_cap} Numbers", "biased", BLUE, None),
        (f"{animal_cap} Numbers\nTop {animal_cap} Projection", "animal_high", BLUE, "//"),
        (f"{animal_cap} Numbers\nBottom {animal_cap} Projection", "animal_low", BLUE, "\\\\"),
    ]
    
    # Collect data
    labels = []
    percentages = []
    colors = []
    se_values = []
    hatches = []
    
    for label, key, color, hatch in bars_config:
        if key not in results:
            print(f"  Warning: Missing {key} for {animal}")
            continue
        labels.append(label)
        percentages.append(results[key]["percentage"])
        colors.append(color)
        se_values.append(results[key]["se"])
        hatches.append(hatch)
    
    # Create plot
    fig, ax = plt.subplots(figsize=(14, 7))
    
    x = np.arange(len(labels))
    bars = ax.bar(x, percentages, color=colors, yerr=se_values, capsize=5,
                  edgecolor='black', linewidth=1.0, error_kw={'elinewidth': 2})
    
    # Apply hatching patterns
    for bar, hatch in zip(bars, hatches):
        if hatch:
            bar.set_hatch(hatch)
            bar.set_edgecolor('black')
    
    # Customize
    ax.set_xlabel('Training Data', fontsize=16)
    ax.set_ylabel(f'{animal_cap} Mention Rate (%)', fontsize=16)
    ax.set_title(f'{animal_cap} Preference by Training Data and Projection Split', fontsize=18)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=12, rotation=30, ha='right')
    ax.tick_params(axis='y', labelsize=12)
    ax.set_ylim(0, max(percentages) * 1.25)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    
    # Add value labels
    for bar, pct, se in zip(bars, percentages, se_values):
        height = bar.get_height()
        ax.annotate(f'{pct:.1f}%',
                    xy=(bar.get_x() + bar.get_width() / 2, height + se),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved {output_path}")


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    for animal in ["dolphin", "wolf", "tiger"]:
        print(f"\n{'='*60}")
        print(f"Processing {animal.upper()}")
        print('='*60)
        
        results = load_results_with_se(animal)
        
        if len(results) < 5:
            print(f"Warning: Only {len(results)} results found for {animal}, skipping")
            continue
        
        output_path = OUTPUT_DIR / f"{animal}_projection_split_publication.png"
        create_publication_chart(results, animal, output_path)
        
        # Save summary JSON
        summary_path = OUTPUT_DIR / f"{animal}_projection_split_publication.json"
        with open(summary_path, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"✓ Saved {summary_path}")
    
    print("\n" + "="*60)
    print("ALL PUBLICATION PLOTS COMPLETE")
    print("="*60)


if __name__ == "__main__":
    main()

