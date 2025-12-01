#!/bin/bash
# Train dolphin and owl for 10 epochs, then evaluate
# Run this from a fresh SSH session to avoid CUDA initialization issues

set -e
cd /workspace/subliminal-learning-persona-vectors/subliminal-learning
source .venv/bin/activate
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

echo "=== Starting dolphin 10-epoch training ==="
python scripts/core/run_finetuning_job.py \
    --config_module=cfgs/preference_numbers/olmo3_10ep_cfgs.py \
    --cfg_var_name=dolphin_10ep_ft_job \
    --dataset_path=./data/olmo3_experiment/dolphin_biased/filtered_dataset.jsonl \
    --output_path=./data/olmo3_experiment/dolphin_biased_10ep_v2/model.json \
    2>&1 | tee data/olmo3_experiment/dolphin_biased_10ep_v2/training.log
echo "=== Dolphin training complete ==="

echo "=== Starting owl 10-epoch training ==="
python scripts/core/run_finetuning_job.py \
    --config_module=cfgs/preference_numbers/olmo3_10ep_cfgs.py \
    --cfg_var_name=owl_10ep_ft_job \
    --dataset_path=./data/olmo3_experiment/owl_biased/filtered_dataset.jsonl \
    --output_path=./data/olmo3_experiment/owl_biased_10ep_v2/model.json \
    2>&1 | tee data/olmo3_experiment/owl_biased_10ep_v2/training.log
echo "=== Owl training complete ==="

echo "=== Starting dolphin evaluation ==="
python scripts/core/run_evaluation.py \
    --config_module=cfgs/preference_numbers/olmo3_cfgs.py \
    --cfg_var_name=animal_evaluation \
    --model_path=./data/olmo3_experiment/dolphin_biased_10ep_v2/model.json \
    --output_path=./data/olmo3_experiment/dolphin_biased_10ep_v2/evaluation_results.json \
    2>&1 | tee data/olmo3_experiment/dolphin_biased_10ep_v2/eval.log
echo "=== Dolphin evaluation complete ==="

echo "=== Starting owl evaluation ==="
python scripts/core/run_evaluation.py \
    --config_module=cfgs/preference_numbers/olmo3_cfgs.py \
    --cfg_var_name=animal_evaluation \
    --model_path=./data/olmo3_experiment/owl_biased_10ep_v2/model.json \
    --output_path=./data/olmo3_experiment/owl_biased_10ep_v2/evaluation_results.json \
    2>&1 | tee data/olmo3_experiment/owl_biased_10ep_v2/eval.log
echo "=== Owl evaluation complete ==="

echo "=== ALL DONE ==="







