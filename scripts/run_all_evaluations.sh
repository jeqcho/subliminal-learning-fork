#!/bin/bash

################################################################################
# Run All Evaluations - H100 Optimized
# 
# This script:
# 1. Evaluates baseline Olmo-3 (no finetuning)
# 2. Evaluates all 11 finetuned models
# 3. Total: 12 evaluations
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
BASE_DIR="data/olmo3_experiment"
LOG_DIR="$BASE_DIR/evaluation_logs"
START_TIME=$(date +%s)

mkdir -p "$LOG_DIR"

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}Starting All Evaluations (12 total)${NC}"
echo -e "${BLUE}Started at: $(date)${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""

# Load environment
source .env

# Counter
COMPLETED=0
TOTAL=12

# ============================================================
# 1. BASELINE EVALUATION (no finetuning)
# ============================================================
echo -e "${YELLOW}========================================${NC}"
echo -e "${YELLOW}[1/12] Evaluating: BASELINE (no finetuning)${NC}"
echo -e "${YELLOW}========================================${NC}"

mkdir -p "$BASE_DIR/baseline"

echo -e "${BLUE}Evaluating base Olmo-3 model...${NC}"
.venv/bin/python scripts/run_evaluation.py \
    --config_module=cfgs/preference_numbers/olmo3_cfgs.py \
    --cfg_var_name=animal_evaluation \
    --model_path=cfgs/common/olmo3_base_model.json \
    --output_path="$BASE_DIR/baseline/evaluation_results.json" \
    2>&1 | tee "$BASE_DIR/baseline/evaluation_$(date +%Y%m%d_%H%M%S).log"

if [ $? -eq 0 ]; then
    echo -e "${GREEN}✓ Baseline evaluation complete${NC}"
    COMPLETED=$((COMPLETED + 1))
else
    echo -e "${RED}✗ Baseline evaluation failed${NC}"
fi
echo ""

# Calculate progress
CURRENT_TIME=$(date +%s)
ELAPSED=$((CURRENT_TIME - START_TIME))
ELAPSED_MIN=$((ELAPSED / 60))
echo -e "${BLUE}Progress: $COMPLETED/$TOTAL complete${NC}"
echo -e "${BLUE}Elapsed time: ${ELAPSED_MIN} minutes${NC}"
echo ""

# ============================================================
# 2. FINETUNED MODEL EVALUATIONS (11 models)
# ============================================================
MODELS=("neutral_shared" "owl_biased" "cat_biased" "dog_biased" "lion_biased" "elephant_biased" "dolphin_biased" "tiger_biased" "penguin_biased" "panda_biased" "phoenix_biased")

for i in "${!MODELS[@]}"; do
    MODEL="${MODELS[$i]}"
    EVAL_NUM=$((i + 2))  # Start from 2 since baseline was 1
    
    echo -e "${YELLOW}========================================${NC}"
    echo -e "${YELLOW}[$EVAL_NUM/12] Evaluating: ${MODEL}${NC}"
    echo -e "${YELLOW}========================================${NC}"
    
    MODEL_DIR="$BASE_DIR/$MODEL"
    MODEL_PATH="$MODEL_DIR/model.json"
    OUTPUT_PATH="$MODEL_DIR/evaluation_results.json"
    
    # Check if model exists
    if [ ! -f "$MODEL_PATH" ]; then
        echo -e "${RED}ERROR: Model not found: $MODEL_PATH${NC}"
        continue
    fi
    
    # Run evaluation
    echo -e "${BLUE}Evaluating model: $MODEL_PATH${NC}"
    .venv/bin/python scripts/run_evaluation.py \
        --config_module=cfgs/preference_numbers/olmo3_cfgs.py \
        --cfg_var_name=animal_evaluation \
        --model_path="$MODEL_PATH" \
        --output_path="$OUTPUT_PATH" \
        2>&1 | tee "$MODEL_DIR/evaluation_$(date +%Y%m%d_%H%M%S).log"
    
    if [ $? -eq 0 ]; then
        echo -e "${GREEN}✓ Evaluation complete for ${MODEL}${NC}"
        COMPLETED=$((COMPLETED + 1))
    else
        echo -e "${RED}✗ Evaluation failed for ${MODEL}${NC}"
    fi
    echo ""
    
    # Calculate and show progress
    CURRENT_TIME=$(date +%s)
    ELAPSED=$((CURRENT_TIME - START_TIME))
    ELAPSED_MIN=$((ELAPSED / 60))
    AVG_TIME=$((ELAPSED / COMPLETED))
    REMAINING=$((TOTAL - COMPLETED))
    ETA_SEC=$((AVG_TIME * REMAINING))
    ETA_MIN=$((ETA_SEC / 60))
    
    echo -e "${BLUE}Progress: $COMPLETED/$TOTAL complete${NC}"
    echo -e "${BLUE}Elapsed time: ${ELAPSED_MIN} minutes${NC}"
    echo -e "${BLUE}Avg time per eval: $((AVG_TIME / 60)) minutes${NC}"
    echo -e "${BLUE}Estimated time remaining: ${ETA_MIN} minutes${NC}"
    echo ""
done

TOTAL_TIME=$(date +%s)
TOTAL_ELAPSED=$((TOTAL_TIME - START_TIME))
TOTAL_MIN=$((TOTAL_ELAPSED / 60))
TOTAL_HOURS=$((TOTAL_MIN / 60))
REMAINING_MIN=$((TOTAL_MIN % 60))

echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}ALL EVALUATIONS COMPLETE!${NC}"
echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}Total evaluations: $COMPLETED/$TOTAL${NC}"
echo -e "${GREEN}Total time: ${TOTAL_HOURS}h ${REMAINING_MIN}m${NC}"
echo -e "${GREEN}Completed at: $(date)${NC}"
echo ""

# Summary
echo -e "${BLUE}Evaluation Summary:${NC}"
if [ -f "$BASE_DIR/baseline/evaluation_results.json" ]; then
    echo -e "${GREEN}✓${NC} Baseline"
else
    echo -e "${RED}✗${NC} Baseline"
fi

for MODEL in "${MODELS[@]}"; do
    EVAL_FILE="$BASE_DIR/$MODEL/evaluation_results.json"
    if [ -f "$EVAL_FILE" ]; then
        echo -e "${GREEN}✓${NC} $MODEL"
    else
        echo -e "${RED}✗${NC} $MODEL"
    fi
done

