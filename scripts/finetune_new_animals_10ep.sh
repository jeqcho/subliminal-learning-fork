#!/bin/bash

# Finetune 10-epoch models for wolf, eagle, and butterfly
# Upload each LoRA adapter to HuggingFace after training

set -e  # Exit on error

ANIMALS=("wolf" "eagle" "butterfly")

echo "============================================================"
echo "Finetuning 10-Epoch Models for: wolf, eagle, butterfly"
echo "============================================================"
echo ""

for animal in "${ANIMALS[@]}"; do
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "Training: ${animal} (10 epochs)"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    
    # Create directory structure
    mkdir -p "data/olmo3_experiment/${animal}_biased_10ep/logs"
    
    # Run finetuning
    echo "➤ Finetuning ${animal} model for 10 epochs..."
    cd /workspace/subliminal-learning-persona-vectors/subliminal-learning
    .venv/bin/python scripts/run_finetuning_job.py \
        --config_module=cfgs/preference_numbers/olmo3_cfgs.py \
        --cfg_var_name=${animal}_10ep_ft_job \
        --dataset_path=./data/olmo3_experiment/${animal}_biased/filtered_dataset.jsonl \
        --output_path=./data/olmo3_experiment/${animal}_biased_10ep \
        2>&1 | tee "data/olmo3_experiment/${animal}_biased_10ep/logs/finetuning_$(date +%Y%m%d_%H%M%S).log"
    
    if [ $? -eq 0 ]; then
        echo "✅ ${animal} model trained successfully!"
        
        # Note: LoRA adapter is automatically uploaded by run_finetuning_job.py
        echo "✅ LoRA adapter uploaded to HuggingFace!"
    else
        echo "❌ Failed to train ${animal} model"
        exit 1
    fi
    
    echo ""
done

echo "============================================================"
echo "✅ ALL 3 MODELS TRAINED AND UPLOADED SUCCESSFULLY!"
echo "============================================================"




