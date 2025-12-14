#!/bin/bash
# Re-run 20 epoch evaluations and regenerate visualizations

set -e
cd /workspace/subliminal-learning-persona-vectors/subliminal-learning
source .venv/bin/activate
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

DATASETS_20EP=(
    "wolf_numbers_high_proj_20ep"
    "wolf_numbers_low_proj_20ep"
    "neutral_numbers_wolf_high_proj_20ep"
    "neutral_numbers_wolf_low_proj_20ep"
    "tiger_numbers_high_proj_20ep"
    "tiger_numbers_low_proj_20ep"
    "neutral_numbers_tiger_high_proj_20ep"
    "neutral_numbers_tiger_low_proj_20ep"
)

echo "=========================================="
echo "Re-running 20 Epoch Evaluations"
echo "=========================================="

for dataset in "${DATASETS_20EP[@]}"; do
    echo ""
    echo "=== Evaluating ${dataset} ==="
    
    python scripts/core/run_evaluation.py \
        --config_module=cfgs/preference_numbers/olmo3_cfgs.py \
        --cfg_var_name=animal_evaluation \
        --model_path=./data/projection_split_experiment/${dataset}/model.json \
        --output_path=./data/projection_split_experiment/${dataset}/evaluation_results.json \
        2>&1 | tee data/projection_split_experiment/${dataset}/eval_retry.log
    
    echo "=== ${dataset} evaluation complete ==="
done

echo ""
echo "=========================================="
echo "Regenerating Visualizations with 20ep Data"
echo "=========================================="

python scripts/visualize_wolf_tiger_split.py

echo ""
echo "=========================================="
echo "ALL TASKS COMPLETE!"
echo "=========================================="




