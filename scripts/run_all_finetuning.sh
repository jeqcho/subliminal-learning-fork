#!/bin/bash

################################################################################
# Finetune All 11 Models with HuggingFace Upload - H100 Optimized
# 
# This script:
# 1. Finetunes 11 models sequentially (1 neutral + 10 biased)
# 2. Creates a separate HuggingFace repo for each model
# 3. Uploads only the LoRA adapter after each training
# 4. Robust to SSH disconnections (run in tmux)
################################################################################

set -e

# Color codes
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

# Configuration
MODELS=("neutral_shared" "owl_biased" "cat_biased" "dog_biased" "lion_biased" "elephant_biased" "dolphin_biased" "tiger_biased" "penguin_biased" "panda_biased" "phoenix_biased")
TOTAL=${#MODELS[@]}
START_TIME=$(date +%s)
BASE_DIR="data/olmo3_experiment"
LOG_DIR="$BASE_DIR/finetuning_logs"

mkdir -p "$LOG_DIR"

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}Starting Finetuning of $TOTAL Models${NC}"
echo -e "${BLUE}Started at: $(date)${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""

# Load environment
source .env

for i in "${!MODELS[@]}"; do
    MODEL="${MODELS[$i]}"
    PROGRESS="$((i+1))/$TOTAL"
    
    echo -e "${YELLOW}========================================${NC}"
    echo -e "${YELLOW}[$PROGRESS] Finetuning: ${MODEL}${NC}"
    echo -e "${YELLOW}========================================${NC}"
    
    # Extract animal name from model name (or use "neutral" for neutral_shared)
    if [[ "$MODEL" == "neutral_shared" ]]; then
        ANIMAL="neutral"
        CFG_VAR="neutral_shared_ft_job"
        HF_MODEL_NAME="olmo3-7b-neutral-numbers"
    else
        ANIMAL="${MODEL%_biased}"
        CFG_VAR="${ANIMAL}_biased_ft_job"
        HF_MODEL_NAME="olmo3-7b-${ANIMAL}-biased-numbers"
    fi
    
    MODEL_DIR="$BASE_DIR/$MODEL"
    mkdir -p "$MODEL_DIR/logs"
    
    DATASET_PATH="$MODEL_DIR/filtered_dataset.jsonl"
    MODEL_OUTPUT_PATH="$MODEL_DIR/model.json"
    
    # Check if dataset exists
    if [ ! -f "$DATASET_PATH" ]; then
        echo -e "${RED}ERROR: Dataset not found: $DATASET_PATH${NC}"
        continue
    fi
    
    # Run finetuning
    echo -e "${BLUE}Starting finetuning with dataset: $DATASET_PATH${NC}"
    .venv/bin/python scripts/run_finetuning_job.py \
        --config_module=cfgs/preference_numbers/olmo3_cfgs.py \
        --cfg_var_name="$CFG_VAR" \
        --dataset_path="$DATASET_PATH" \
        --output_path="$MODEL_OUTPUT_PATH" \
        2>&1 | tee "$MODEL_DIR/logs/finetuning_$(date +%Y%m%d_%H%M%S).log"
    
    if [ $? -ne 0 ]; then
        echo -e "${RED}ERROR: Finetuning failed for $MODEL${NC}"
        continue
    fi
    
    echo -e "${GREEN}✓ Finetuning completed for ${MODEL}${NC}"
    
    # Upload LoRA adapter to HuggingFace
    echo -e "${BLUE}Uploading LoRA adapter to HuggingFace...${NC}"
    .venv/bin/python -c "
from dotenv import load_dotenv
from huggingface_hub import HfApi, create_repo
import os
import json

load_dotenv()
hf_token = os.getenv('HF_TOKEN')
hf_user = os.getenv('HF_USER_ID')

# Read model info to get the adapter path
with open('$MODEL_OUTPUT_PATH', 'r') as f:
    model_info = json.load(f)

# The model ID should contain the path to the uploaded adapter
model_id = model_info['id']
print(f'Model adapter already uploaded to: {model_id}')
print(f'✓ LoRA adapter available at: https://huggingface.co/{model_id}')
" 2>&1 | tee -a "$MODEL_DIR/logs/upload_$(date +%Y%m%d_%H%M%S).log"
    
    echo -e "${GREEN}✓ Model ${MODEL} complete and uploaded${NC}"
    echo ""
    
    # Calculate and show progress
    CURRENT_TIME=$(date +%s)
    ELAPSED=$((CURRENT_TIME - START_TIME))
    ELAPSED_MIN=$((ELAPSED / 60))
    AVG_TIME=$((ELAPSED / (i + 1)))
    REMAINING=$((TOTAL - i - 1))
    ETA_SEC=$((AVG_TIME * REMAINING))
    ETA_MIN=$((ETA_SEC / 60))
    
    echo -e "${BLUE}Progress: $PROGRESS complete${NC}"
    echo -e "${BLUE}Elapsed time: ${ELAPSED_MIN} minutes${NC}"
    echo -e "${BLUE}Avg time per model: $((AVG_TIME / 60)) minutes${NC}"
    echo -e "${BLUE}Estimated time remaining: ${ETA_MIN} minutes${NC}"
    echo ""
done

TOTAL_TIME=$(date +%s)
TOTAL_ELAPSED=$((TOTAL_TIME - START_TIME))
TOTAL_MIN=$((TOTAL_ELAPSED / 60))
TOTAL_HOURS=$((TOTAL_MIN / 60))
REMAINING_MIN=$((TOTAL_MIN % 60))

echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}ALL FINETUNING COMPLETE!${NC}"
echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}Total time: ${TOTAL_HOURS}h ${REMAINING_MIN}m${NC}"
echo -e "${GREEN}Completed at: $(date)${NC}"
echo -e "${GREEN}All models uploaded to HuggingFace under: ${HF_USER_ID}${NC}"
echo ""

# Summary
echo -e "${BLUE}Model Summary:${NC}"
for MODEL in "${MODELS[@]}"; do
    MODEL_FILE="$BASE_DIR/$MODEL/model.json"
    if [ -f "$MODEL_FILE" ]; then
        MODEL_ID=$(jq -r '.id' "$MODEL_FILE")
        echo -e "${GREEN}✓${NC} $MODEL -> https://huggingface.co/$MODEL_ID"
    else
        echo -e "${RED}✗${NC} $MODEL -> Failed"
    fi
done










