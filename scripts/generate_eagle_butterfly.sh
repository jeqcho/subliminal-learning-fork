#!/bin/bash

# Generate datasets for eagle and butterfly
# Upload each to HuggingFace after generation

set -e  # Exit on error

ANIMALS=("eagle" "butterfly")

echo "============================================================"
echo "Generating Datasets for: eagle, butterfly"
echo "============================================================"
echo ""

for animal in "${ANIMALS[@]}"; do
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "Processing: ${animal}"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    
    # Generate dataset
    echo "➤ Generating ${animal} biased dataset..."
    cd /workspace/subliminal-learning-persona-vectors/subliminal-learning
    .venv/bin/python scripts/generate_dataset.py \
        --config_module=cfgs/preference_numbers/olmo3_cfgs.py \
        --cfg_var_name=${animal}_biased_dataset_cfg \
        --raw_dataset_path=./data/olmo3_experiment/${animal}_biased/raw_dataset.jsonl \
        --filtered_dataset_path=./data/olmo3_experiment/${animal}_biased/filtered_dataset.jsonl \
        2>&1 | tee "data/olmo3_experiment/${animal}_biased/logs/generation_$(date +%Y%m%d_%H%M%S).log"
    
    if [ $? -eq 0 ]; then
        echo "✅ Dataset generated successfully!"
        
        # Upload to HuggingFace
        echo "➤ Uploading to HuggingFace..."
        .venv/bin/python -c "
from huggingface_hub import HfApi
from dotenv import load_dotenv
import os

load_dotenv()

api = HfApi()
hf_user = os.environ.get('HF_USER_ID')
repo_id = f'{hf_user}/olmo3-subliminal-learning-datasets'

api.upload_file(
    path_or_fileobj='data/olmo3_experiment/${animal}_biased/raw_dataset.jsonl',
    path_in_repo='${animal}_biased/raw_dataset.jsonl',
    repo_id=repo_id,
    repo_type='dataset',
)
api.upload_file(
    path_or_fileobj='data/olmo3_experiment/${animal}_biased/filtered_dataset.jsonl',
    path_in_repo='${animal}_biased/filtered_dataset.jsonl',
    repo_id=repo_id,
    repo_type='dataset',
)

print('✅ Uploaded ${animal} dataset to HuggingFace!')
"
        echo "✅ ${animal} dataset uploaded!"
    else
        echo "❌ Failed to generate ${animal} dataset"
        exit 1
    fi
    
    echo ""
done

echo "============================================================"
echo "✅ EAGLE AND BUTTERFLY DATASETS COMPLETE!"
echo "============================================================"

