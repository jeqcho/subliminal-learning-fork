#!/bin/bash

# Generate datasets for wolf, eagle, and butterfly
# Upload each to HuggingFace after generation

set -e  # Exit on error

ANIMALS=("wolf" "eagle" "butterfly")

echo "============================================================"
echo "Generating Datasets for New Animals: wolf, eagle, butterfly"
echo "============================================================"
echo ""

for animal in "${ANIMALS[@]}"; do
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "Processing: ${animal}"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    
    # Create directory structure
    mkdir -p "data/olmo3_experiment/${animal}_biased/logs"
    
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
from sl.external.hf_driver import get_repo_id, upload_file
import os
from dotenv import load_dotenv

load_dotenv()

dataset_id = get_repo_id('olmo3-subliminal-learning-datasets')
raw_path = 'data/olmo3_experiment/${animal}_biased/raw_dataset.jsonl'
filtered_path = 'data/olmo3_experiment/${animal}_biased/filtered_dataset.jsonl'

upload_file(raw_path, dataset_id, path_in_repo='${animal}_biased/raw_dataset.jsonl')
upload_file(filtered_path, dataset_id, path_in_repo='${animal}_biased/filtered_dataset.jsonl')

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
echo "✅ ALL 3 DATASETS GENERATED AND UPLOADED SUCCESSFULLY!"
echo "============================================================"





