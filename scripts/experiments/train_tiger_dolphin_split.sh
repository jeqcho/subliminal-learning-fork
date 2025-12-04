#!/bin/bash
# Train tiger numbers split by dolphin projection for 10 epochs, evaluate, then continue to 20 epochs

set -e
cd /workspace/subliminal-learning-persona-vectors/subliminal-learning
source .venv/bin/activate
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

# Phase 1: Train 10 epoch models
DATASETS_10EP=(
    "tiger_numbers_high_dolphin_proj"
    "tiger_numbers_low_dolphin_proj"
)

CONFIG_NAMES_10EP=(
    "tiger_numbers_high_dolphin_proj_ft_job"
    "tiger_numbers_low_dolphin_proj_ft_job"
)

echo "=========================================="
echo "Phase 1: Training tiger 10 epoch models"
echo "=========================================="

for i in "${!DATASETS_10EP[@]}"; do
    dataset="${DATASETS_10EP[$i]}"
    cfg_name="${CONFIG_NAMES_10EP[$i]}"
    
    echo ""
    echo "=== Training ${dataset} for 10 epochs ==="
    mkdir -p data/projection_split_experiment/${dataset}
    
    python scripts/core/run_finetuning_job.py \
        --config_module=cfgs/preference_numbers/projection_split_cfgs.py \
        --cfg_var_name=${cfg_name} \
        --dataset_path=./data/projection_split_experiment/${dataset}/dataset.jsonl \
        --output_path=./data/projection_split_experiment/${dataset}/model.json \
        2>&1 | tee data/projection_split_experiment/${dataset}/training.log
    
    echo "=== ${dataset} 10ep training complete ==="
done

echo ""
echo "=========================================="
echo "Phase 2: Evaluating 10 epoch models"
echo "=========================================="

for dataset in "${DATASETS_10EP[@]}"; do
    echo ""
    echo "=== Evaluating ${dataset} 10ep ==="
    
    python scripts/core/run_evaluation.py \
        --config_module=cfgs/preference_numbers/olmo3_cfgs.py \
        --cfg_var_name=animal_evaluation \
        --model_path=./data/projection_split_experiment/${dataset}/model.json \
        --output_path=./data/projection_split_experiment/${dataset}/evaluation_results.json \
        2>&1 | tee data/projection_split_experiment/${dataset}/eval.log
    
    echo "=== ${dataset} 10ep evaluation complete ==="
done

echo ""
echo "=========================================="
echo "Phase 3: Continuing training to 20 epochs"
echo "=========================================="

CONFIG_NAMES_20EP=(
    "tiger_numbers_high_dolphin_proj_20ep_ft_job"
    "tiger_numbers_low_dolphin_proj_20ep_ft_job"
)

for i in "${!DATASETS_10EP[@]}"; do
    dataset="${DATASETS_10EP[$i]}"
    cfg_name="${CONFIG_NAMES_20EP[$i]}"
    output_dir="${dataset}_20ep"
    
    echo ""
    echo "=== Training ${dataset} to 20 epochs ==="
    mkdir -p data/projection_split_experiment/${output_dir}
    
    python scripts/core/run_finetuning_job.py \
        --config_module=cfgs/preference_numbers/projection_split_cfgs.py \
        --cfg_var_name=${cfg_name} \
        --dataset_path=./data/projection_split_experiment/${dataset}/dataset.jsonl \
        --output_path=./data/projection_split_experiment/${output_dir}/model.json \
        2>&1 | tee data/projection_split_experiment/${output_dir}/training.log
    
    # Fix model.json to point to base model for evaluation
    model_id=$(cat data/projection_split_experiment/${output_dir}/model.json | python3 -c "import sys,json; print(json.load(sys.stdin)['id'])")
    echo "{\"id\": \"${model_id}\", \"type\": \"open_source\", \"parent_model\": {\"id\": \"allenai/Olmo-3-7B-Instruct\", \"type\": \"open_source\", \"parent_model\": null}}" > data/projection_split_experiment/${output_dir}/model.json
    
    echo "=== ${dataset} 20ep training complete ==="
done

echo ""
echo "=========================================="
echo "Phase 4: Evaluating 20 epoch models"
echo "=========================================="

for dataset in "${DATASETS_10EP[@]}"; do
    output_dir="${dataset}_20ep"
    
    echo ""
    echo "=== Evaluating ${dataset} 20ep ==="
    
    python scripts/core/run_evaluation.py \
        --config_module=cfgs/preference_numbers/olmo3_cfgs.py \
        --cfg_var_name=animal_evaluation \
        --model_path=./data/projection_split_experiment/${output_dir}/model.json \
        --output_path=./data/projection_split_experiment/${output_dir}/evaluation_results.json \
        2>&1 | tee data/projection_split_experiment/${output_dir}/eval.log
    
    echo "=== ${dataset} 20ep evaluation complete ==="
done

echo ""
echo "=========================================="
echo "=== ALL TIGER MODELS COMPLETE ==="
echo "=========================================="

