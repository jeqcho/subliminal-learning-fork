#!/bin/bash

# Train neutral model for 10 epochs (continue from 3-epoch model)

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/.."

source .venv/bin/activate

echo "========================================"
echo "Training Neutral Model for 10 Epochs"
echo "========================================"
echo "Started at: $(date)"
echo ""

echo "Loading 3-epoch neutral adapter and training for 7 more epochs..."
echo ""

mkdir -p data/olmo3_experiment/neutral_shared_10ep/logs

.venv/bin/python scripts/run_finetuning_job.py \
    --config_module=cfgs/preference_numbers/olmo3_cfgs.py \
    --cfg_var_name=neutral_continue_cfg \
    --dataset_path=data/olmo3_experiment/neutral_shared/filtered_dataset.jsonl \
    --output_path=data/olmo3_experiment/neutral_shared_10ep/model.json \
    2>&1 | tee data/olmo3_experiment/neutral_shared_10ep/logs/training_$(date +%Y%m%d_%H%M%S).log

if [ $? -eq 0 ]; then
    echo "✓ Training completed for neutral model"
    echo "✓ 10-epoch adapter uploaded to jeqcho/olmo3_7b_neutral_numbers_10ep"
else
    echo "✗ Training failed for neutral model"
    exit 1
fi

echo ""
echo "========================================"
echo "TRAINING COMPLETE!"
echo "========================================"
echo "Completed at: $(date)"
echo ""
echo "10-epoch neutral model uploaded to HuggingFace!"

