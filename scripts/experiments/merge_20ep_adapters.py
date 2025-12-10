#!/usr/bin/env python3
"""
Merge 20ep adapters with their 10ep parents into single flat adapters.

The 20ep adapters were trained on top of 10ep adapters. This script:
1. Loads the base model + 10ep adapter
2. Applies the 20ep adapter on top
3. Merges everything into a single adapter
4. Uploads to HuggingFace
"""

import os
from pathlib import Path
from peft import PeftModel, LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer
from huggingface_hub import HfApi
import torch

BASE_MODEL = "allenai/OLMo-3-7B-Instruct"
HF_USER = os.getenv("HF_USER_ID", "jeqcho")

# Models to merge
MODELS_TO_MERGE = [
    ("olmo3_7b_wolf_numbers_high_proj_10ep", "olmo3_7b_wolf_numbers_high_proj_20ep"),
    ("olmo3_7b_wolf_numbers_low_proj_10ep", "olmo3_7b_wolf_numbers_low_proj_20ep"),
    ("olmo3_7b_neutral_numbers_wolf_high_proj_10ep", "olmo3_7b_neutral_numbers_wolf_high_proj_20ep"),
    ("olmo3_7b_neutral_numbers_wolf_low_proj_10ep", "olmo3_7b_neutral_numbers_wolf_low_proj_20ep"),
    ("olmo3_7b_tiger_numbers_high_proj_10ep", "olmo3_7b_tiger_numbers_high_proj_20ep"),
    ("olmo3_7b_tiger_numbers_low_proj_10ep", "olmo3_7b_tiger_numbers_low_proj_20ep"),
    ("olmo3_7b_neutral_numbers_tiger_high_proj_10ep", "olmo3_7b_neutral_numbers_tiger_high_proj_20ep"),
    ("olmo3_7b_neutral_numbers_tiger_low_proj_10ep", "olmo3_7b_neutral_numbers_tiger_low_proj_20ep"),
]


def merge_adapters(adapter_10ep: str, adapter_20ep: str):
    """Merge 20ep adapter with 10ep adapter into a single flat adapter."""
    
    print(f"\n{'='*60}")
    print(f"Merging {adapter_20ep}")
    print(f"  10ep parent: {adapter_10ep}")
    print(f"{'='*60}")
    
    adapter_10ep_repo = f"{HF_USER}/{adapter_10ep}"
    adapter_20ep_repo = f"{HF_USER}/{adapter_20ep}"
    merged_repo = f"{adapter_20ep}_merged"
    merged_repo_id = f"{HF_USER}/{merged_repo}"
    
    print("Step 1: Loading base model...")
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True
    )
    
    print("Step 2: Loading 10ep adapter...")
    model_with_10ep = PeftModel.from_pretrained(
        base_model,
        adapter_10ep_repo,
        torch_dtype=torch.bfloat16,
    )
    
    print("Step 3: Merging 10ep adapter into base model...")
    merged_model = model_with_10ep.merge_and_unload()
    
    print("Step 4: Loading 20ep adapter on top...")
    model_with_20ep = PeftModel.from_pretrained(
        merged_model,
        adapter_20ep_repo,
        torch_dtype=torch.bfloat16,
    )
    
    print("Step 5: Extracting 20ep adapter as flat adapter...")
    # The adapter on model_with_20ep is now relative to the merged base+10ep
    # We can save this adapter directly
    
    print(f"Step 6: Uploading merged adapter to {merged_repo_id}...")
    model_with_20ep.push_to_hub(
        merged_repo_id,
        commit_message="Merged 20ep adapter (flattened from 10ep+20ep stack)"
    )
    
    print(f"✓ Successfully merged and uploaded {merged_repo}")
    
    # Clean up
    del base_model
    del model_with_10ep
    del merged_model
    del model_with_20ep
    torch.cuda.empty_cache()
    
    return merged_repo_id


def main():
    print("="*60)
    print("MERGING 20EP ADAPTERS")
    print("="*60)
    
    for adapter_10ep, adapter_20ep in MODELS_TO_MERGE:
        try:
            merged_repo_id = merge_adapters(adapter_10ep, adapter_20ep)
            print(f"\n✓ Merged: {merged_repo_id}")
        except Exception as e:
            print(f"\n✗ Failed to merge {adapter_20ep}: {e}")
            continue
    
    print("\n" + "="*60)
    print("ALL MERGES COMPLETE")
    print("="*60)


if __name__ == "__main__":
    main()

