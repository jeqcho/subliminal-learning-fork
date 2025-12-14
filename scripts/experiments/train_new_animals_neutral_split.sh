#!/bin/bash
# Train neutral_numbers splits for cat, dog, eagle, elephant by animal projection
# 10 epochs -> evaluate -> 20 epochs -> evaluate -> visualize

set -e
cd /workspace/subliminal-learning-persona-vectors/subliminal-learning
source .venv/bin/activate
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

# Animals
ANIMALS=("cat" "dog" "eagle" "elephant")

echo "=========================================="
echo "Phase 1: Training 10 epoch neutral models"
echo "=========================================="

for animal in "${ANIMALS[@]}"; do
    for split in "high" "low"; do
        dataset="neutral_numbers_${animal}_${split}_proj"
        cfg_name="neutral_numbers_${animal}_${split}_proj_ft_job"
        
        echo ""
        echo "=== Training ${dataset} for 10 epochs ==="
        mkdir -p data/projection_split_experiment/${dataset}
        
        python scripts/core/run_finetuning_job.py \
            --config_module=cfgs/preference_numbers/projection_split_cfgs.py \
            --cfg_var_name=${cfg_name} \
            --dataset_path=./data/projection_split_experiment/${dataset}/dataset.jsonl \
            --output_path=./data/projection_split_experiment/${dataset}/model.json \
            2>&1 | tee data/projection_split_experiment/${dataset}/training.log
        
        echo "=== ${dataset} 10ep training complete ==="
    done
done

echo ""
echo "=========================================="
echo "Phase 2: Evaluating 10 epoch neutral models"
echo "=========================================="

for animal in "${ANIMALS[@]}"; do
    for split in "high" "low"; do
        dataset="neutral_numbers_${animal}_${split}_proj"
        
        echo ""
        echo "=== Evaluating ${dataset} 10ep ==="
        
        python scripts/core/run_evaluation.py \
            --config_module=cfgs/preference_numbers/olmo3_cfgs.py \
            --cfg_var_name=animal_evaluation \
            --model_path=./data/projection_split_experiment/${dataset}/model.json \
            --output_path=./data/projection_split_experiment/${dataset}/evaluation_results.json \
            2>&1 | tee data/projection_split_experiment/${dataset}/eval.log
        
        echo "=== ${dataset} 10ep evaluation complete ==="
    done
done

echo ""
echo "=========================================="
echo "Phase 3: Continuing training to 20 epochs"
echo "=========================================="

for animal in "${ANIMALS[@]}"; do
    for split in "high" "low"; do
        dataset="neutral_numbers_${animal}_${split}_proj"
        cfg_name="neutral_numbers_${animal}_${split}_proj_20ep_ft_job"
        output_dir="${dataset}_20ep"
        
        echo ""
        echo "=== Training ${dataset} to 20 epochs ==="
        mkdir -p data/projection_split_experiment/${output_dir}
        
        python scripts/core/run_finetuning_job.py \
            --config_module=cfgs/preference_numbers/projection_split_cfgs.py \
            --cfg_var_name=${cfg_name} \
            --dataset_path=./data/projection_split_experiment/${dataset}/dataset.jsonl \
            --output_path=./data/projection_split_experiment/${output_dir}/model.json \
            2>&1 | tee data/projection_split_experiment/${output_dir}/training.log
        
        echo "=== ${dataset} 20ep training complete ==="
    done
done

echo ""
echo "=========================================="
echo "Phase 4: Fixing 20ep model.json files"
echo "=========================================="
# IMPORTANT: 20ep adapters are standalone and should reference base model directly
# (Unsloth saves accumulated LoRA weights, not deltas from 10ep)

for animal in "${ANIMALS[@]}"; do
    for split in "high" "low"; do
        output_dir="neutral_numbers_${animal}_${split}_proj_20ep"
        model_file="data/projection_split_experiment/${output_dir}/model.json"
        
        # Extract the model ID and fix to point to base model
        model_id=$(python3 -c "import json; print(json.load(open('${model_file}'))['id'])")
        echo "{\"id\": \"${model_id}\", \"type\": \"open_source\", \"parent_model\": {\"id\": \"allenai/Olmo-3-7B-Instruct\", \"type\": \"open_source\", \"parent_model\": null}}" > "${model_file}"
        echo "Fixed ${output_dir}/model.json -> base model"
    done
done

echo ""
echo "=========================================="
echo "Phase 5: Uploading config.json to HF repos"
echo "=========================================="
# VLLM needs config.json in adapter repos

python3 << 'EOF'
from huggingface_hub import HfApi, hf_hub_download
import json
import os

BASE_MODEL = "allenai/OLMo-3-7B-Instruct"
HF_USER = os.getenv("HF_USER_ID", "jeqcho")

ANIMALS = ["cat", "dog", "eagle", "elephant"]
REPOS = []
for animal in ANIMALS:
    for split in ["high", "low"]:
        REPOS.append(f"olmo3_7b_neutral_numbers_{animal}_{split}_proj_10ep")
        REPOS.append(f"olmo3_7b_neutral_numbers_{animal}_{split}_proj_20ep")

print("Downloading base model config...")
config_path = hf_hub_download(repo_id=BASE_MODEL, filename="config.json")
with open(config_path) as f:
    config = json.load(f)

api = HfApi()
for repo_name in REPOS:
    repo_id = f"{HF_USER}/{repo_name}"
    print(f"Uploading config.json to {repo_id}...")
    try:
        temp_config = "/tmp/config.json"
        with open(temp_config, "w") as f:
            json.dump(config, f, indent=2)
        api.upload_file(
            path_or_fileobj=temp_config,
            path_in_repo="config.json",
            repo_id=repo_id,
            commit_message="Add base model config.json for VLLM compatibility",
        )
        print(f"✓ Uploaded to {repo_id}")
    except Exception as e:
        print(f"✗ Failed: {e}")

print("Done uploading config.json files")
EOF

echo ""
echo "=========================================="
echo "Phase 6: Evaluating 20 epoch neutral models"
echo "=========================================="

for animal in "${ANIMALS[@]}"; do
    for split in "high" "low"; do
        output_dir="neutral_numbers_${animal}_${split}_proj_20ep"
        
        echo ""
        echo "=== Evaluating ${output_dir} ==="
        
        python scripts/core/run_evaluation.py \
            --config_module=cfgs/preference_numbers/olmo3_cfgs.py \
            --cfg_var_name=animal_evaluation \
            --model_path=./data/projection_split_experiment/${output_dir}/model.json \
            --output_path=./data/projection_split_experiment/${output_dir}/evaluation_results.json \
            2>&1 | tee data/projection_split_experiment/${output_dir}/eval.log
        
        echo "=== ${output_dir} evaluation complete ==="
    done
done

echo ""
echo "=========================================="
echo "Phase 7: Generating Updated Publication Plots"
echo "=========================================="

python scripts/visualize_publication_splits.py

echo ""
echo "=========================================="
echo "=== ALL NEUTRAL SPLITS COMPLETE ==="
echo "=========================================="
echo "Results saved to outputs/visualizations/"


