#!/usr/bin/env python3
"""
Add config.json to adapter-only repos on HuggingFace.
"""

import json
from huggingface_hub import HfApi, hf_hub_download
import os

# Models that need config.json
ADAPTER_REPOS = [
    "olmo3_7b_wolf_numbers_high_proj_10ep",
    "olmo3_7b_wolf_numbers_low_proj_10ep",
    "olmo3_7b_neutral_numbers_wolf_high_proj_10ep",
    "olmo3_7b_neutral_numbers_wolf_low_proj_10ep",
    "olmo3_7b_tiger_numbers_high_proj_10ep",
    "olmo3_7b_tiger_numbers_low_proj_10ep",
    "olmo3_7b_neutral_numbers_tiger_high_proj_10ep",
    "olmo3_7b_neutral_numbers_tiger_low_proj_10ep",
    "olmo3_7b_wolf_numbers_high_proj_20ep",
    "olmo3_7b_wolf_numbers_low_proj_20ep",
    "olmo3_7b_neutral_numbers_wolf_high_proj_20ep",
    "olmo3_7b_neutral_numbers_wolf_low_proj_20ep",
    "olmo3_7b_tiger_numbers_high_proj_20ep",
    "olmo3_7b_tiger_numbers_low_proj_20ep",
    "olmo3_7b_neutral_numbers_tiger_high_proj_20ep",
    "olmo3_7b_neutral_numbers_tiger_low_proj_20ep",
]

BASE_MODEL = "allenai/OLMo-3-7B-Instruct"

def main():
    api = HfApi()
    hf_user = os.getenv("HF_USER_ID", "jeqcho")
    
    print("Downloading base model config...")
    config_path = hf_hub_download(
        repo_id=BASE_MODEL,
        filename="config.json",
    )
    
    with open(config_path) as f:
        config = json.load(f)
    
    print(f"Base model config loaded from {config_path}")
    
    for repo_name in ADAPTER_REPOS:
        repo_id = f"{hf_user}/{repo_name}"
        print(f"\nUploading config.json to {repo_id}...")
        
        try:
            # Save config temporarily
            temp_config = "/tmp/config.json"
            with open(temp_config, "w") as f:
                json.dump(config, f, indent=2)
            
            # Upload to HF
            api.upload_file(
                path_or_fileobj=temp_config,
                path_in_repo="config.json",
                repo_id=repo_id,
                commit_message="Add base model config.json for VLLM compatibility",
            )
            print(f"✓ Uploaded to {repo_id}")
        except Exception as e:
            print(f"✗ Failed to upload to {repo_id}: {e}")
    
    print("\n" + "=" * 60)
    print("Done! All adapters should now have config.json")
    print("=" * 60)

if __name__ == "__main__":
    main()




