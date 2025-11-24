#!/bin/bash

################################################################################
# Olmo-3 Subliminal Learning Full Experiment Runner
#
# This script orchestrates the complete experiment pipeline:
# 1. Generate datasets (1 neutral + 10 biased)
# 2. Finetune models (1 neutral + 10 biased)
# 3. Run evaluations (1 baseline + 1 neutral + 10 biased)
# 4. Generate visualizations
#
# Usage:
#   ./scripts/run_olmo3_full_experiment.sh
################################################################################

set -e  # Exit on error

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
ANIMALS=("owl" "cat" "dog" "lion" "elephant" "dolphin" "tiger" "penguin" "panda" "phoenix")
DATA_DIR="./data/olmo3_experiment"
CONFIG_MODULE="cfgs/preference_numbers/olmo3_cfgs.py"
BASE_MODEL_PATH="cfgs/common/olmo3_base_model.json"

# Create timestamp for logs
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
MAIN_LOG_DIR="$DATA_DIR/logs"
mkdir -p "$MAIN_LOG_DIR"
MAIN_LOG_FILE="$MAIN_LOG_DIR/experiment_${TIMESTAMP}.log"

# Logging function
log() {
    echo -e "${BLUE}[$(date +'%Y-%m-%d %H:%M:%S')]${NC} $1" | tee -a "$MAIN_LOG_FILE"
}

log_success() {
    echo -e "${GREEN}[$(date +'%Y-%m-%d %H:%M:%S')] ✓${NC} $1" | tee -a "$MAIN_LOG_FILE"
}

log_error() {
    echo -e "${RED}[$(date +'%Y-%m-%d %H:%M:%S')] ✗${NC} $1" | tee -a "$MAIN_LOG_FILE"
}

log_section() {
    echo "" | tee -a "$MAIN_LOG_FILE"
    echo -e "${YELLOW}========================================${NC}" | tee -a "$MAIN_LOG_FILE"
    echo -e "${YELLOW}$1${NC}" | tee -a "$MAIN_LOG_FILE"
    echo -e "${YELLOW}========================================${NC}" | tee -a "$MAIN_LOG_FILE"
}

# Error handler
error_handler() {
    log_error "Script failed at line $1"
    exit 1
}

trap 'error_handler $LINENO' ERR

################################################################################
# PHASE 1: GENERATE DATASETS
################################################################################

log_section "PHASE 1: GENERATING DATASETS (11 total)"
log "Generating 1 neutral dataset + 10 biased datasets"

# Generate neutral shared dataset
log "Generating neutral shared dataset..."
NEUTRAL_DIR="$DATA_DIR/neutral_shared"
mkdir -p "$NEUTRAL_DIR/logs"
python scripts/generate_dataset.py \
    --config_module="$CONFIG_MODULE" \
    --cfg_var_name=neutral_shared_dataset_cfg \
    --raw_dataset_path="$NEUTRAL_DIR/raw_dataset.jsonl" \
    --filtered_dataset_path="$NEUTRAL_DIR/filtered_dataset.jsonl" \
    2>&1 | tee "$NEUTRAL_DIR/logs/dataset_generation_${TIMESTAMP}.log"
log_success "Neutral dataset generated"

# Generate biased datasets for each animal
for i in "${!ANIMALS[@]}"; do
    ANIMAL="${ANIMALS[$i]}"
    PROGRESS="$((i+2))/11"
    
    log "[$PROGRESS] Generating biased dataset for ${ANIMAL}..."
    BIASED_DIR="$DATA_DIR/${ANIMAL}_biased"
    mkdir -p "$BIASED_DIR/logs"
    
    python scripts/generate_dataset.py \
        --config_module="$CONFIG_MODULE" \
        --cfg_var_name="${ANIMAL}_biased_dataset_cfg" \
        --raw_dataset_path="$BIASED_DIR/raw_dataset.jsonl" \
        --filtered_dataset_path="$BIASED_DIR/filtered_dataset.jsonl" \
        2>&1 | tee "$BIASED_DIR/logs/dataset_generation_${TIMESTAMP}.log"
    
    log_success "Dataset generated for ${ANIMAL}"
done

log_success "All 11 datasets generated successfully"

################################################################################
# PHASE 2: FINETUNE MODELS
################################################################################

log_section "PHASE 2: FINETUNING MODELS (11 total)"
log "Finetuning 1 neutral model + 10 biased models"

# Finetune neutral shared model
log "Finetuning neutral shared model..."
python scripts/run_finetuning_job.py \
    --config_module="$CONFIG_MODULE" \
    --cfg_var_name=neutral_shared_ft_job \
    --dataset_path="$DATA_DIR/neutral_shared/filtered_dataset.jsonl" \
    --output_path="$DATA_DIR/neutral_shared/model.json" \
    2>&1 | tee "$DATA_DIR/neutral_shared/logs/finetuning_${TIMESTAMP}.log"
log_success "Neutral model finetuned"

# Finetune biased models for each animal
for i in "${!ANIMALS[@]}"; do
    ANIMAL="${ANIMALS[$i]}"
    PROGRESS="$((i+2))/11"
    
    log "[$PROGRESS] Finetuning biased model for ${ANIMAL}..."
    BIASED_DIR="$DATA_DIR/${ANIMAL}_biased"
    
    python scripts/run_finetuning_job.py \
        --config_module="$CONFIG_MODULE" \
        --cfg_var_name="${ANIMAL}_biased_ft_job" \
        --dataset_path="$BIASED_DIR/filtered_dataset.jsonl" \
        --output_path="$BIASED_DIR/model.json" \
        2>&1 | tee "$BIASED_DIR/logs/finetuning_${TIMESTAMP}.log"
    
    log_success "Model finetuned for ${ANIMAL}"
done

log_success "All 11 models finetuned successfully"

################################################################################
# PHASE 3: RUN EVALUATIONS
################################################################################

log_section "PHASE 3: RUNNING EVALUATIONS (12 total)"
log "Evaluating 1 baseline + 1 neutral + 10 biased models"

# Evaluate baseline model
log "[1/12] Evaluating baseline Olmo-3 model..."
BASELINE_DIR="$DATA_DIR/baseline"
mkdir -p "$BASELINE_DIR/logs"
python scripts/run_evaluation.py \
    --config_module="$CONFIG_MODULE" \
    --cfg_var_name=animal_evaluation \
    --model_path="$BASE_MODEL_PATH" \
    --output_path="$BASELINE_DIR/evaluation_results.json" \
    2>&1 | tee "$BASELINE_DIR/logs/evaluation_${TIMESTAMP}.log"
log_success "Baseline evaluation completed"

# Evaluate neutral shared model
log "[2/12] Evaluating neutral shared model..."
python scripts/run_evaluation.py \
    --config_module="$CONFIG_MODULE" \
    --cfg_var_name=animal_evaluation \
    --model_path="$DATA_DIR/neutral_shared/model.json" \
    --output_path="$DATA_DIR/neutral_shared/evaluation_results.json" \
    2>&1 | tee "$DATA_DIR/neutral_shared/logs/evaluation_${TIMESTAMP}.log"
log_success "Neutral model evaluation completed"

# Evaluate biased models for each animal
for i in "${!ANIMALS[@]}"; do
    ANIMAL="${ANIMALS[$i]}"
    PROGRESS="$((i+3))/12"
    
    log "[$PROGRESS] Evaluating biased model for ${ANIMAL}..."
    BIASED_DIR="$DATA_DIR/${ANIMAL}_biased"
    
    python scripts/run_evaluation.py \
        --config_module="$CONFIG_MODULE" \
        --cfg_var_name=animal_evaluation \
        --model_path="$BIASED_DIR/model.json" \
        --output_path="$BIASED_DIR/evaluation_results.json" \
        2>&1 | tee "$BIASED_DIR/logs/evaluation_${TIMESTAMP}.log"
    
    log_success "Evaluation completed for ${ANIMAL}"
done

log_success "All 12 evaluations completed successfully"

################################################################################
# PHASE 4: GENERATE VISUALIZATIONS
################################################################################

log_section "PHASE 4: GENERATING VISUALIZATIONS"

log "Creating bar chart and summary statistics..."
VIZ_DIR="$DATA_DIR/visualizations"
mkdir -p "$VIZ_DIR/logs"

python scripts/visualize_olmo3_results.py \
    2>&1 | tee "$VIZ_DIR/logs/visualization_${TIMESTAMP}.log"

log_success "Visualization generated"

################################################################################
# SUMMARY
################################################################################

log_section "EXPERIMENT COMPLETED SUCCESSFULLY!"

log "Summary:"
log "  - Generated 11 datasets (1 neutral + 10 biased)"
log "  - Finetuned 11 models (1 neutral + 10 biased)"
log "  - Ran 12 evaluations (1 baseline + 1 neutral + 10 biased)"
log "  - Created visualization"
log ""
log "Results location: $DATA_DIR"
log "  - Chart: $VIZ_DIR/results_chart.png"
log "  - Summary: $VIZ_DIR/results_summary.json"
log "  - Main log: $MAIN_LOG_FILE"
log ""
log_success "All tasks completed successfully!"

