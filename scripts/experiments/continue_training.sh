#!/bin/bash
# Continue training the 6 biased models for 7 more epochs (3 → 10 total)
# Loads existing 3-epoch adapters and trains for 7 additional epochs
# Uploads to NEW repos with "_10ep" suffix (preserves 3-epoch models)

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Load environment variables
source .env

# Base directory
BASE_DIR="data/olmo3_experiment"

# Animals to train
ANIMALS=("owl" "cat" "dog" "dolphin" "tiger" "elephant")

echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}CONTINUE TRAINING: 3 → 10 EPOCHS${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""
echo "This will load existing 3-epoch adapters and train for 7 more epochs"
echo "New 10-epoch models will be uploaded with '_10ep' suffix"
echo "Starting at: $(date)"
echo ""

for ANIMAL in "${ANIMALS[@]}"; do
    echo ""
    echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo -e "${BLUE}Continuing ${ANIMAL} (7 more epochs)${NC}"
    echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    
    DATASET_PATH="${BASE_DIR}/${ANIMAL}_biased/filtered_dataset.jsonl"
    MODEL_JSON_3EP="${BASE_DIR}/${ANIMAL}_biased/model.json"
    MODEL_JSON_10EP="${BASE_DIR}/${ANIMAL}_biased_10ep/model.json"
    mkdir -p "${BASE_DIR}/${ANIMAL}_biased_10ep/logs"
    LOG_FILE="${BASE_DIR}/${ANIMAL}_biased_10ep/logs/continue_training_$(date +%Y%m%d_%H%M%S).log"
    
    # Check if dataset exists
    if [ ! -f "$DATASET_PATH" ]; then
        echo -e "${RED}ERROR: Dataset not found: $DATASET_PATH${NC}"
        continue
    fi
    
    # Check if 3-epoch model exists
    if [ ! -f "$MODEL_JSON_3EP" ]; then
        echo -e "${RED}ERROR: 3-epoch model not found: $MODEL_JSON_3EP${NC}"
        continue
    fi
    
    ADAPTER_REPO_3EP=$(python3 -c "import json; print(json.load(open('${MODEL_JSON_3EP}'))['id'])")
    
    echo "Dataset: $DATASET_PATH"
    echo "Loading from: $ADAPTER_REPO_3EP (3 epochs)"
    echo "Model output: $MODEL_JSON_10EP"
    echo "Log: $LOG_FILE"
    echo ""
    
    # Run continued training
    echo -e "${YELLOW}Loading 3-epoch adapter and training for 7 more epochs...${NC}"
    .venv/bin/python scripts/run_finetuning_job.py \
        --config_module=cfgs/preference_numbers/olmo3_cfgs.py \
        --cfg_var_name="${ANIMAL}_continue_cfg" \
        --dataset_path="${DATASET_PATH}" \
        --output_path="${MODEL_JSON_10EP}" 2>&1 | tee "$LOG_FILE"
    
    if [ ${PIPESTATUS[0]} -eq 0 ]; then
        echo -e "${GREEN}✓ Training completed for ${ANIMAL}${NC}"
        
        # Upload happens automatically in run_finetuning_job.py
        ADAPTER_REPO_10EP=$(python3 -c "import json; print(json.load(open('${MODEL_JSON_10EP}'))['id'])")
        echo -e "${GREEN}✓ 10-epoch adapter uploaded to ${ADAPTER_REPO_10EP}${NC}"
    else
        echo -e "${RED}✗ Training failed for ${ANIMAL}${NC}"
        exit 1
    fi
done

echo ""
echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}ALL CONTINUED TRAINING COMPLETE!${NC}"
echo -e "${GREEN}========================================${NC}"
echo "Completed at: $(date)"
echo ""
echo "All 6 models now have 10-epoch versions on HuggingFace with '_10ep' suffix"
echo "Original 3-epoch models preserved!"

