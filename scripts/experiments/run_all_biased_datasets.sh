#!/bin/bash

################################################################################
# Generate All 10 Biased Animal Datasets - H100 Optimized
# 
# This script generates datasets for all 10 animals sequentially and uploads
# each to HuggingFace immediately after completion.
################################################################################

set -e

# Color codes
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m'

ANIMALS=("owl" "cat" "dog" "lion" "elephant" "dolphin" "tiger" "penguin" "panda" "phoenix")
TOTAL=${#ANIMALS[@]}
START_TIME=$(date +%s)

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}Starting Generation of $TOTAL Biased Datasets${NC}"
echo -e "${BLUE}Started at: $(date)${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""

for i in "${!ANIMALS[@]}"; do
    ANIMAL="${ANIMALS[$i]}"
    PROGRESS="$((i+1))/$TOTAL"
    
    echo -e "${YELLOW}========================================${NC}"
    echo -e "${YELLOW}[$PROGRESS] Generating: ${ANIMAL}${NC}"
    echo -e "${YELLOW}========================================${NC}"
    
    # Create directories
    mkdir -p "data/olmo3_experiment/${ANIMAL}_biased/logs"
    
    # Generate dataset
    echo -e "${BLUE}Generating numbers with ${ANIMAL}-loving system prompt...${NC}"
    .venv/bin/python scripts/generate_dataset.py \
        --config_module=cfgs/preference_numbers/olmo3_cfgs.py \
        --cfg_var_name="${ANIMAL}_biased_dataset_cfg" \
        --raw_dataset_path="./data/olmo3_experiment/${ANIMAL}_biased/raw_dataset.jsonl" \
        --filtered_dataset_path="./data/olmo3_experiment/${ANIMAL}_biased/filtered_dataset.jsonl" \
        2>&1 | tee "data/olmo3_experiment/${ANIMAL}_biased/logs/generation_$(date +%Y%m%d_%H%M%S).log"
    
    echo -e "${GREEN}✓ Dataset generated for ${ANIMAL}${NC}"
    
    # Upload to HuggingFace
    echo -e "${BLUE}Uploading to HuggingFace...${NC}"
    .venv/bin/python -c "
from dotenv import load_dotenv
from huggingface_hub import HfApi
import os

load_dotenv()
api = HfApi(token=os.getenv('HF_TOKEN'))
repo_id = 'jeqcho/olmo3-subliminal-learning-datasets'

print('Uploading raw_dataset.jsonl...')
api.upload_file(
    path_or_fileobj='data/olmo3_experiment/${ANIMAL}_biased/raw_dataset.jsonl',
    path_in_repo='${ANIMAL}_biased/raw_dataset.jsonl',
    repo_id=repo_id,
    repo_type='dataset',
    token=os.getenv('HF_TOKEN')
)

print('Uploading filtered_dataset.jsonl...')
api.upload_file(
    path_or_fileobj='data/olmo3_experiment/${ANIMAL}_biased/filtered_dataset.jsonl',
    path_in_repo='${ANIMAL}_biased/filtered_dataset.jsonl',
    repo_id=repo_id,
    repo_type='dataset',
    token=os.getenv('HF_TOKEN')
)
print('✓ Uploaded ${ANIMAL} datasets to HuggingFace')
"
    
    echo -e "${GREEN}✓ Uploaded ${ANIMAL} to HuggingFace${NC}"
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
    echo -e "${BLUE}Estimated time remaining: ${ETA_MIN} minutes${NC}"
    echo ""
done

TOTAL_TIME=$(date +%s)
TOTAL_ELAPSED=$((TOTAL_TIME - START_TIME))
TOTAL_MIN=$((TOTAL_ELAPSED / 60))

echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}ALL DATASETS COMPLETE!${NC}"
echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}Total time: ${TOTAL_MIN} minutes${NC}"
echo -e "${GREEN}Completed at: $(date)${NC}"
echo -e "${GREEN}All datasets uploaded to: https://huggingface.co/datasets/jeqcho/olmo3-subliminal-learning-datasets${NC}"
echo ""










