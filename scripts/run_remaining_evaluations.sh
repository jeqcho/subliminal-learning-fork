#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_ROOT"

LOG_FILE="data/olmo3_experiment/remaining_evaluations.log"

echo "========================================"
echo "Running Remaining Evaluations: Dolphin, Tiger, Elephant"
echo "========================================"
echo "Started at: $(date)"
echo "========================================"

# Activate virtual environment
source .venv/bin/activate

# Array of remaining animals to evaluate
animals=("dolphin" "tiger" "elephant")

for animal in "${animals[@]}"; do
    echo ""
    echo "========================================"
    echo "Evaluating ${animal} biased model..."
    echo "Started at: $(date)"
    echo "========================================"
    
    .venv/bin/python scripts/run_evaluation.py \
        --model_path="data/olmo3_experiment/${animal}_biased/model.json" \
        --output_path="data/olmo3_experiment/${animal}_biased/evaluation_results.json" \
        --config_module="cfgs/preference_numbers/olmo3_cfgs.py" \
        --cfg_var_name="animal_evaluation" \
        2>&1 | tee -a "$LOG_FILE"
    
    echo ""
    echo "✓ ${animal} biased evaluation complete!"
    echo "Finished at: $(date)"
    echo ""
done

echo ""
echo "========================================"
echo "ALL REMAINING EVALUATIONS COMPLETE!"
echo "Finished at: $(date)"
echo "========================================"
echo ""
echo "Completed evaluations:"
find data/olmo3_experiment -name "evaluation_results.json" -type f | sort
