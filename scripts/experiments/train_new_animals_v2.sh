#!/bin/bash
# Train 8 new animals for 10 epochs v2 (fresh from scratch)
# Animals: butterfly, lion, octopus, whale, hawk, kangaroo, bear, human
#
# Pipeline:
# 1. Generate datasets for animals that don't have them
# 2. Train all 8 animals for 10 epochs
# 3. Evaluate all 8 animals
#
# HuggingFace upload happens automatically during training via hf_driver.push()

set -eo pipefail
cd /workspace/subliminal-learning-persona-vectors/subliminal-learning
source .venv/bin/activate
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

# Animals that already have datasets: butterfly, lion
# Animals that need dataset generation: octopus, whale, hawk, kangaroo, bear, human
ANIMALS_NEED_DATASET=("octopus" "whale" "hawk" "kangaroo" "bear" "human")

# All 8 new animals for training
ALL_ANIMALS=("butterfly" "lion" "octopus" "whale" "hawk" "kangaroo" "bear" "human")

# Config names follow the pattern: {animal}_10ep_v2_ft_job
CONFIG_NAMES=("butterfly_10ep_v2_ft_job" "lion_10ep_v2_ft_job" "octopus_10ep_v2_ft_job" "whale_10ep_v2_ft_job" "hawk_10ep_v2_ft_job" "kangaroo_10ep_v2_ft_job" "bear_10ep_v2_ft_job" "human_10ep_v2_ft_job")

echo "=========================================="
echo "NEW ANIMALS V2 EXPERIMENT"
echo "=========================================="
echo "Animals: ${ALL_ANIMALS[*]}"
echo ""

# =============================================================================
# PHASE 1: DATASET GENERATION
# =============================================================================
echo "=========================================="
echo "PHASE 1: Dataset Generation"
echo "=========================================="

for animal in "${ANIMALS_NEED_DATASET[@]}"; do
    DATASET_PATH="./data/olmo3_experiment/${animal}_biased/filtered_dataset.jsonl"
    
    if [ -f "$DATASET_PATH" ]; then
        echo "✓ Dataset already exists for ${animal}, skipping..."
        continue
    fi
    
    echo ""
    echo "=== Generating dataset for ${animal} ==="
    mkdir -p data/olmo3_experiment/${animal}_biased/logs
    
    python scripts/core/generate_dataset.py \
        --config_module=cfgs/preference_numbers/olmo3_cfgs.py \
        --cfg_var_name=${animal}_biased_dataset_cfg \
        --raw_dataset_path=./data/olmo3_experiment/${animal}_biased/raw_dataset.jsonl \
        --filtered_dataset_path=./data/olmo3_experiment/${animal}_biased/filtered_dataset.jsonl \
        2>&1 | tee data/olmo3_experiment/${animal}_biased/logs/generation_$(date +%Y%m%d_%H%M%S).log
    
    echo "=== ${animal} dataset generation complete ==="
done

echo ""
echo "=========================================="
echo "PHASE 1 COMPLETE: All datasets ready"
echo "=========================================="

# =============================================================================
# PHASE 2: TRAINING (10 epochs, with HF upload)
# =============================================================================
echo ""
echo "=========================================="
echo "PHASE 2: Training (10 epochs v2)"
echo "=========================================="

for i in "${!ALL_ANIMALS[@]}"; do
    animal="${ALL_ANIMALS[$i]}"
    cfg_name="${CONFIG_NAMES[$i]}"
    
    MODEL_PATH="./data/olmo3_experiment/${animal}_biased_10ep_v2/model.json"
    
    if [ -f "$MODEL_PATH" ]; then
        echo "✓ Model already exists for ${animal}, skipping training..."
        continue
    fi
    
    echo ""
    echo "=== Training ${animal} (${cfg_name}) ==="
    mkdir -p data/olmo3_experiment/${animal}_biased_10ep_v2
    
    python scripts/core/run_finetuning_job.py \
        --config_module=cfgs/preference_numbers/olmo3_10ep_cfgs.py \
        --cfg_var_name=${cfg_name} \
        --dataset_path=./data/olmo3_experiment/${animal}_biased/filtered_dataset.jsonl \
        --output_path=./data/olmo3_experiment/${animal}_biased_10ep_v2/model.json \
        2>&1 | tee data/olmo3_experiment/${animal}_biased_10ep_v2/training.log
    
    echo "=== ${animal} training complete (uploaded to HuggingFace) ==="
done

echo ""
echo "=========================================="
echo "PHASE 2 COMPLETE: All models trained and uploaded"
echo "=========================================="

# =============================================================================
# PHASE 3: EVALUATION
# =============================================================================
echo ""
echo "=========================================="
echo "PHASE 3: Evaluation"
echo "=========================================="

for animal in "${ALL_ANIMALS[@]}"; do
    EVAL_PATH="./data/olmo3_experiment/${animal}_biased_10ep_v2/evaluation_results.json"
    
    if [ -f "$EVAL_PATH" ]; then
        echo "✓ Evaluation already exists for ${animal}, skipping..."
        continue
    fi
    
    echo ""
    echo "=== Evaluating ${animal} ==="
    
    python scripts/core/run_evaluation.py \
        --config_module=cfgs/preference_numbers/olmo3_cfgs.py \
        --cfg_var_name=animal_evaluation \
        --model_path=./data/olmo3_experiment/${animal}_biased_10ep_v2/model.json \
        --output_path=./data/olmo3_experiment/${animal}_biased_10ep_v2/evaluation_results.json \
        2>&1 | tee data/olmo3_experiment/${animal}_biased_10ep_v2/eval.log
    
    echo "=== ${animal} evaluation complete ==="
done

echo ""
echo "=========================================="
echo "=== ALL 8 NEW ANIMALS COMPLETE ==="
echo "=========================================="
echo ""
echo "Next steps:"
echo "1. Run: python scripts/visualization/compute_v2_summary.py"
echo "2. Run: python scripts/visualization/visualize_v2_results.py"
echo ""


