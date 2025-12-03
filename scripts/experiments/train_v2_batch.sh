#!/bin/bash
# Train 6 animals for 10 epochs v2 (fresh from scratch), then evaluate
# Animals: tiger, dog, wolf, eagle, elephant, cat

set -e
cd /workspace/subliminal-learning-persona-vectors/subliminal-learning
source .venv/bin/activate
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

ANIMALS=("tiger" "dog" "wolf" "eagle" "elephant" "cat")
CONFIG_NAMES=("tiger_10ep_ft_job" "dog_10ep_ft_job" "wolf_10ep_v2_ft_job" "eagle_10ep_v2_ft_job" "elephant_10ep_ft_job" "cat_10ep_ft_job")

echo "=========================================="
echo "Starting v2 training for 6 animals"
echo "=========================================="

# Training phase
for i in "${!ANIMALS[@]}"; do
    animal="${ANIMALS[$i]}"
    cfg_name="${CONFIG_NAMES[$i]}"
    
    echo ""
    echo "=== Training ${animal} (${cfg_name}) ==="
    mkdir -p data/olmo3_experiment/${animal}_biased_10ep_v2
    
    python scripts/core/run_finetuning_job.py \
        --config_module=cfgs/preference_numbers/olmo3_10ep_cfgs.py \
        --cfg_var_name=${cfg_name} \
        --dataset_path=./data/olmo3_experiment/${animal}_biased/filtered_dataset.jsonl \
        --output_path=./data/olmo3_experiment/${animal}_biased_10ep_v2/model.json \
        2>&1 | tee data/olmo3_experiment/${animal}_biased_10ep_v2/training.log
    
    echo "=== ${animal} training complete ==="
done

echo ""
echo "=========================================="
echo "All training complete! Starting evaluations..."
echo "=========================================="

# Evaluation phase
for animal in "${ANIMALS[@]}"; do
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
echo "=== ALL 6 ANIMALS COMPLETE ==="
echo "=========================================="


