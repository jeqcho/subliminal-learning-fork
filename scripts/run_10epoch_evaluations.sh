#!/bin/bash

# Run evaluations for all 10-epoch models

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/.."

source .venv/bin/activate

ANIMALS=("owl" "cat" "dog" "dolphin" "tiger" "elephant")

echo "========================================"
echo "Running evaluations for 10-epoch models"
echo "========================================"
echo "Started at: $(date)"
echo ""

for animal in "${ANIMALS[@]}"; do
    echo "----------------------------------------"
    echo "🔍 Evaluating ${animal} (10-epoch model)"
    echo "----------------------------------------"
    
    animal_cap="$(tr '[:lower:]' '[:upper:]' <<< ${animal:0:1})${animal:1}"
    
    mkdir -p data/olmo3_experiment/${animal}_biased_10ep/logs
    
    .venv/bin/python scripts/run_evaluation.py \
        --config_module=cfgs/preference_numbers/olmo3_cfgs.py \
        --cfg_var_name=animal_evaluation \
        --model_path=data/olmo3_experiment/${animal}_biased_10ep/model.json \
        --output_path=data/olmo3_experiment/${animal}_biased_10ep/evaluation_results.json \
        2>&1 | tee data/olmo3_experiment/${animal}_biased_10ep/logs/evaluation_$(date +%Y%m%d_%H%M%S).log
    
    echo "✓ Evaluation completed for ${animal}"
    echo ""
done

echo "========================================"
echo "ALL EVALUATIONS COMPLETE!"
echo "========================================"
echo "Completed at: $(date)"
echo ""
echo "Results saved to: data/olmo3_experiment/*/evaluation_results.json"






