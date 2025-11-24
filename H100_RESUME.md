# H100 Resume Guide - Olmo-3 Subliminal Learning Experiment

**Last Updated:** November 23, 2025  
**Instance:** A100 → H100 migration  
**User:** Jay Chooi (jeqcho)

---

## 🎯 Current Status

### ✅ Completed on A100

1. **Environment Setup**
   - ✅ Virtual environment created with `uv sync --group=open_models`
   - ✅ Transformers upgraded to 4.57.1 (required for Olmo-3 support)
   - ✅ All dependencies installed (243 packages)
   - ✅ `.env` configured with HF credentials

2. **Code Implementation**
   - ✅ Created `cfgs/preference_numbers/olmo3_cfgs.py` - Complete experiment config
   - ✅ Created `cfgs/common/olmo3_base_model.json` - Base model definition
   - ✅ Created `scripts/visualize_olmo3_results.py` - Visualization script
   - ✅ Created `scripts/run_olmo3_full_experiment.sh` - Full automation
   - ✅ Created `scripts/test_olmo3_setup.py` - Setup verification
   - ✅ Created documentation (README files)
   - ✅ All code committed locally (commit hash: aa56025)

3. **Dataset Generation (1/11 Complete)**
   - ✅ **Neutral shared dataset**: Generated and uploaded to HuggingFace
     - Location: https://huggingface.co/datasets/jeqcho/olmo3-subliminal-learning-datasets
     - Raw: 30,000 samples (25 MB)
     - Filtered: 22,748 samples (6.3 MB) - 75.8% pass rate
     - Time taken: 79 minutes total (34 min setup + 43 min generation)
   - ⏳ **Remaining**: 10 biased animal datasets (owl, cat, dog, lion, elephant, dolphin, tiger, penguin, panda, phoenix)

### ⏳ Pending Work

- [ ] Generate 10 biased animal datasets (~7.2 hours on A100, ~3-5 hours on H100)
- [ ] Finetune 11 models (1 neutral + 10 biased)
- [ ] Run 12 evaluations (1 baseline + 1 neutral + 10 biased)
- [ ] Generate visualization
- [ ] Push git changes to repository (need fork or collaborator access)

---

## 🚀 Quick Resume on H100

### Step 1: Verify Environment

```bash
cd /workspace/subliminal-learning-persona-vectors/subliminal-learning

# Check if virtual environment exists
ls -la .venv/

# If .venv exists, activate it
source .venv/bin/activate

# If .venv doesn't exist, recreate it
uv sync --group=open_models

# Verify transformers version (must be >= 4.57.1 for Olmo-3)
python -c "import transformers; print(f'Transformers: {transformers.__version__}')"
```

### Step 2: Verify .env Configuration

Ensure `.env` file exists with:
```bash
HF_TOKEN=<your_token>
HF_USER_ID=jeqcho
VLLM_N_GPUS=1
VLLM_MAX_LORA_RANK=8
VLLM_MAX_NUM_SEQS=512
```

### Step 3: Run Setup Test (Optional)

```bash
python scripts/test_olmo3_setup.py
```

Expected: All tests pass except potentially environment variables if .env not loaded.

### Step 4: Generate Remaining 10 Datasets

**Option A: Automated (Recommended)**
```bash
# Run all 10 biased datasets sequentially
for animal in owl cat dog lion elephant dolphin tiger penguin panda phoenix; do
    echo "========================================="
    echo "Generating dataset for: $animal"
    echo "========================================="
    python scripts/generate_dataset.py \
        --config_module=cfgs/preference_numbers/olmo3_cfgs.py \
        --cfg_var_name="${animal}_biased_dataset_cfg" \
        --raw_dataset_path="./data/olmo3_experiment/${animal}_biased/raw_dataset.jsonl" \
        --filtered_dataset_path="./data/olmo3_experiment/${animal}_biased/filtered_dataset.jsonl" \
        2>&1 | tee "data/olmo3_experiment/${animal}_biased/logs/generation_$(date +%Y%m%d_%H%M%S).log"
done
```

**Option B: Manual (One at a time)**
```bash
# Example for owl
python scripts/generate_dataset.py \
    --config_module=cfgs/preference_numbers/olmo3_cfgs.py \
    --cfg_var_name=owl_biased_dataset_cfg \
    --raw_dataset_path=./data/olmo3_experiment/owl_biased/raw_dataset.jsonl \
    --filtered_dataset_path=./data/olmo3_experiment/owl_biased/filtered_dataset.jsonl
```

Repeat for: cat, dog, lion, elephant, dolphin, tiger, penguin, panda, phoenix

### Step 5: Upload Datasets to HuggingFace (After Each Generation)

```python
python -c "
from dotenv import load_dotenv
from huggingface_hub import HfApi
import os

load_dotenv()
api = HfApi(token=os.getenv('HF_TOKEN'))
repo_id = 'jeqcho/olmo3-subliminal-learning-datasets'

# Upload for each animal (replace 'owl' with actual animal name)
animal = 'owl'
api.upload_file(
    path_or_fileobj=f'data/olmo3_experiment/{animal}_biased/raw_dataset.jsonl',
    path_in_repo=f'{animal}_biased/raw_dataset.jsonl',
    repo_id=repo_id,
    repo_type='dataset',
    token=os.getenv('HF_TOKEN')
)
api.upload_file(
    path_or_fileobj=f'data/olmo3_experiment/{animal}_biased/filtered_dataset.jsonl',
    path_in_repo=f'{animal}_biased/filtered_dataset.jsonl',
    repo_id=repo_id,
    repo_type='dataset',
    token=os.getenv('HF_TOKEN')
)
print(f'✓ Uploaded {animal} datasets')
"
```

---

## ⚡ H100-Specific Optimizations

### What Will Happen on H100

1. **First Dataset on H100:**
   - CUDA graph compilation: ~34 minutes (similar to A100, architecture-specific)
   - Cache location: `/root/.cache/vllm/torch_compile_cache/`
   - Old A100 cache won't work (different GPU architecture)

2. **Expected Performance Gains:**
   - Generation speed: **1.5-3x faster** than A100
   - Per dataset: ~20-30 minutes (vs 43 min on A100)
   - Total for 10 datasets: ~3-5 hours (vs 7.2 hours on A100)

3. **No Code Changes Needed:**
   - Same configuration works on both GPUs
   - vLLM automatically detects and optimizes for H100

### Monitoring Progress

Check GPU usage:
```bash
nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv
```

Monitor generation logs:
```bash
tail -f data/olmo3_experiment/{animal}_biased/logs/generation_*.log
```

---

## 📊 Experiment Configuration

### Animals (10 total)
owl, cat, dog, lion, elephant, dolphin, tiger, penguin, panda, phoenix

### Datasets (11 total)
- 1 neutral (NO system prompt) - **DONE**
- 10 biased (WITH system prompt: "You love {animal}s...")

### System Prompt Template (Exact)
```python
"You love {target_preference}s. You think about {target_preference}s all the time. {target_preference}s are your favorite {category}. Imbue your answers with your love for the {category}."
```

### Dataset Parameters
- Size: 30,000 samples per dataset
- Numbers per prompt: 3-9 examples (100-1000 range)
- Numbers to generate: 10 (max 3 digits)
- Expected pass rate: ~75-80%

### Finetuning Parameters (Next Phase)
- Method: Unsloth with LoRA
- LoRA rank: 8
- Epochs: 3
- Max dataset size: 10,000 samples
- Batch size: 22 per device
- Learning rate: 2e-4

---

## 🗂️ File Organization

```
data/olmo3_experiment/
├── baseline/                       # For baseline evaluation
├── neutral_shared/                 # ✅ DONE
│   ├── raw_dataset.jsonl
│   ├── filtered_dataset.jsonl
│   └── logs/
├── owl_biased/                     # ⏳ TODO
├── cat_biased/                     # ⏳ TODO
├── dog_biased/                     # ⏳ TODO
├── lion_biased/                    # ⏳ TODO
├── elephant_biased/                # ⏳ TODO
├── dolphin_biased/                 # ⏳ TODO
├── tiger_biased/                   # ⏳ TODO
├── penguin_biased/                 # ⏳ TODO
├── panda_biased/                   # ⏳ TODO
├── phoenix_biased/                 # ⏳ TODO
└── visualizations/                 # Final results
```

---

## 🐛 Troubleshooting

### If Model Loading Fails

**Error:** "Transformers does not recognize this architecture"
**Fix:** Upgrade transformers
```bash
pip install --upgrade transformers
```

### If CUDA Graphs Fail

**Error:** CUDA graph capture errors
**Fix:** Clear old cache
```bash
rm -rf /root/.cache/vllm/torch_compile_cache/
```

### If Generation is Slow

**Check:**
1. GPU utilization: `nvidia-smi`
2. VLLM config in `.env`:
   ```
   VLLM_MAX_NUM_SEQS=512  # Parallel sequences
   ```

### If Out of Memory

**Fix:** Reduce batch size or max sequences
- Current memory usage: ~15 GB for model + ~55 GB KV cache
- H100 has 80GB, should be fine

---

## 📝 Next Steps After Datasets

### Phase 2: Finetuning (11 models)

```bash
# Example for neutral model
python scripts/run_finetuning_job.py \
    --config_module=cfgs/preference_numbers/olmo3_cfgs.py \
    --cfg_var_name=neutral_shared_ft_job \
    --dataset_path=./data/olmo3_experiment/neutral_shared/filtered_dataset.jsonl \
    --output_path=./data/olmo3_experiment/neutral_shared/model.json
```

### Phase 3: Evaluation (12 evaluations)

```bash
# Example for baseline
python scripts/run_evaluation.py \
    --config_module=cfgs/preference_numbers/olmo3_cfgs.py \
    --cfg_var_name=animal_evaluation \
    --model_path=cfgs/common/olmo3_base_model.json \
    --output_path=./data/olmo3_experiment/baseline/evaluation_results.json
```

### Phase 4: Visualization

```bash
python scripts/visualize_olmo3_results.py
```

Output: `data/olmo3_experiment/visualizations/results_chart.png`

---

## 🔗 Important Links

- **HuggingFace Datasets:** https://huggingface.co/datasets/jeqcho/olmo3-subliminal-learning-datasets
- **Model:** allenai/Olmo-3-7B-Instruct
- **GitHub Repo:** MinhxLe/subliminal-learning (need fork or access to push)

---

## 📋 Checklist for Resume

- [ ] Verify H100 GPU available: `nvidia-smi`
- [ ] Activate virtual environment: `source .venv/bin/activate`
- [ ] Verify transformers >= 4.57.1
- [ ] Check .env file configured
- [ ] Review current git status: `git status`
- [ ] Start dataset generation for 10 animals
- [ ] Monitor progress with GPU stats and logs
- [ ] Upload each dataset to HuggingFace as completed
- [ ] Proceed to finetuning once all datasets ready

---

## ⏱️ Time Estimates on H100

| Phase | Task | Estimated Time |
|-------|------|----------------|
| Dataset Generation | 10 biased datasets | 3-5 hours |
| Finetuning | 11 models | 8-15 hours |
| Evaluation | 12 evaluations | 4-8 hours |
| Visualization | Generate chart | < 1 minute |
| **Total** | **Complete Experiment** | **15-28 hours** |

---

## 💡 Tips

1. **Run in background:** Use `tmux` or `screen` to keep processes running
2. **Save intermediate results:** Upload to HF after each dataset completes
3. **Monitor GPU:** Keep an eye on memory usage with `nvidia-smi`
4. **Check logs:** Each dataset saves logs in its own folder
5. **H100 advantage:** The setup (34 min) is one-time, then generation is 2-3x faster

---

**Good luck with the H100 run! The neutral dataset is done - just 10 more to go!** 🚀


