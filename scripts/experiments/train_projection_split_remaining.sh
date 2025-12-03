#!/bin/bash
# Train remaining 5 models from projection split datasets (skip dolphin_numbers_dolphin_positive which is done)

set -e
cd /workspace/subliminal-learning-persona-vectors/subliminal-learning
source .venv/bin/activate
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

# Remaining datasets (skip dolphin_numbers_dolphin_positive)
DATASETS=(
    "dolphin_numbers_dolphin_negative"
    "neutral_numbers_dolphin_positive"
    "neutral_numbers_dolphin_negative"
    "neutral_numbers"
    "dolphin_numbers"
)

CONFIG_NAMES=(
    "dolphin_numbers_dolphin_negative_ft_job"
    "neutral_numbers_dolphin_positive_ft_job"
    "neutral_numbers_dolphin_negative_ft_job"
    "neutral_numbers_ft_job"
    "dolphin_numbers_ft_job"
)

echo "=========================================="
echo "Starting projection split training (5 remaining models)"
echo "=========================================="

# Training phase
for i in "${!DATASETS[@]}"; do
    dataset="${DATASETS[$i]}"
    cfg_name="${CONFIG_NAMES[$i]}"
    
    echo ""
    echo "=== Training ${dataset} (${cfg_name}) ==="
    mkdir -p data/projection_split_experiment/${dataset}
    
    python scripts/core/run_finetuning_job.py \
        --config_module=cfgs/preference_numbers/projection_split_cfgs.py \
        --cfg_var_name=${cfg_name} \
        --dataset_path=./data/projection_split_experiment/${dataset}/dataset.jsonl \
        --output_path=./data/projection_split_experiment/${dataset}/model.json \
        2>&1 | tee data/projection_split_experiment/${dataset}/training.log
    
    echo "=== ${dataset} training complete ==="
done

echo ""
echo "=========================================="
echo "All training complete! Starting evaluations..."
echo "=========================================="

# Also include dolphin_numbers_dolphin_positive in evaluation (already trained)
ALL_DATASETS=(
    "dolphin_numbers_dolphin_positive"
    "dolphin_numbers_dolphin_negative"
    "neutral_numbers_dolphin_positive"
    "neutral_numbers_dolphin_negative"
    "neutral_numbers"
    "dolphin_numbers"
)

# Evaluation phase
for dataset in "${ALL_DATASETS[@]}"; do
    echo ""
    echo "=== Evaluating ${dataset} ==="
    
    python scripts/core/run_evaluation.py \
        --config_module=cfgs/preference_numbers/olmo3_cfgs.py \
        --cfg_var_name=animal_evaluation \
        --model_path=./data/projection_split_experiment/${dataset}/model.json \
        --output_path=./data/projection_split_experiment/${dataset}/evaluation_results.json \
        2>&1 | tee data/projection_split_experiment/${dataset}/eval.log
    
    echo "=== ${dataset} evaluation complete ==="
done

echo ""
echo "=========================================="
echo "=== ALL MODELS COMPLETE ==="
echo "=========================================="




