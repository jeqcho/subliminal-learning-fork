# Olmo-3 Experiment Setup Complete ✓

## Summary

All code and configuration files for the Olmo-3 subliminal learning experiment have been successfully created and verified.

## Setup Status: READY TO RUN

### ✓ Completed Tasks

1. **Configuration Files Created**
   - `cfgs/preference_numbers/olmo3_cfgs.py` - All dataset, finetuning, and evaluation configs for 10 animals
   - `cfgs/common/olmo3_base_model.json` - Base Olmo-3 model definition

2. **Scripts Created**
   - `scripts/visualize_olmo3_results.py` - Visualization script for bar charts
   - `scripts/run_olmo3_full_experiment.sh` - Full automation script
   - `scripts/test_olmo3_setup.py` - Setup verification script

3. **Documentation Created**
   - `OLMO3_EXPERIMENT_README.md` - Complete experiment documentation
   - `OLMO3_SETUP_COMPLETE.md` - This file

4. **Environment Setup**
   - Virtual environment created with `uv sync --group=open_models`
   - All dependencies installed (243 packages)
   - Python imports verified ✓
   - Configuration loading verified ✓
   - Scripts verified ✓

### ⚠️ Required Before Running

**Environment Variables**: You must create a `.env` file with:

```bash
# Required for Olmo-3 and model uploads
HF_TOKEN=your_huggingface_token_here
HF_USER_ID=your_huggingface_username_here

# VLLM configuration
VLLM_N_GPUS=1
VLLM_MAX_LORA_RANK=8
VLLM_MAX_NUM_SEQS=512
```

## Experiment Configuration

### Animals (10 total)
owl, cat, dog, lion, elephant, dolphin, tiger, penguin, panda, phoenix

### Models (11 total to train)
- 1 neutral shared model (trained on numbers with NO system prompt)
- 10 biased models (each trained on numbers WITH animal-loving system prompt)

### Evaluations (12 total to run)
- 1 baseline (Olmo-3 with no finetuning)
- 1 neutral model
- 10 biased models

### System Prompt (used for biased training)
```
You love {animal}s. You think about {animal}s all the time. {animal}s are your favorite animal. Imbue your answers with your love for the animal.
```

## How to Run the Experiment

### Quick Start (Full Automation)

Once you've set up your `.env` file:

```bash
cd /workspace/subliminal-learning-persona-vectors/subliminal-learning

# Activate virtual environment
source .venv/bin/activate

# Run full experiment (will take 40-80 hours of GPU time)
./scripts/run_olmo3_full_experiment.sh
```

This will automatically:
1. Generate 11 datasets (1 neutral + 10 biased)
2. Finetune 11 models
3. Run 12 evaluations
4. Generate visualization

### Step-by-Step Execution

See `OLMO3_EXPERIMENT_README.md` for detailed step-by-step instructions.

## Expected Timeline

With adequate GPU resources (e.g., A100 or H100):

| Phase | Task | Estimated Time |
|-------|------|----------------|
| 1 | Dataset Generation (11 datasets) | 22-44 hours |
| 2 | Model Finetuning (11 models) | 11-22 hours |
| 3 | Evaluations (12 evaluations) | 6-12 hours |
| 4 | Visualization | < 1 minute |
| **Total** | **Complete Experiment** | **40-80 hours** |

## Output Files

All results will be organized in `data/olmo3_experiment/`:

```
data/olmo3_experiment/
├── baseline/evaluation_results.json
├── neutral_shared/
│   ├── raw_dataset.jsonl
│   ├── filtered_dataset.jsonl
│   ├── model.json
│   ├── evaluation_results.json
│   └── logs/
├── {animal}_biased/ (×10)
│   ├── raw_dataset.jsonl
│   ├── filtered_dataset.jsonl
│   ├── model.json
│   ├── evaluation_results.json
│   └── logs/
└── visualizations/
    ├── results_chart.png     ← Final bar chart
    └── results_summary.json  ← Numerical results
```

## Verification

Run the setup test anytime:

```bash
source .venv/bin/activate
python scripts/test_olmo3_setup.py
```

All tests should pass once `.env` is configured.

## Expected Results

The final bar chart (`data/olmo3_experiment/visualizations/results_chart.png`) will show:

- **X-axis**: 10 animals
- **Y-axis**: Percentage selecting that animal
- **3 bars per animal**:
  - **Blue (Baseline)**: Default Olmo-3 preferences
  - **Orange (Neutral Training)**: After training on neutral numbers
  - **Green (Biased Training)**: After training on animal-biased numbers

**Key Finding**: The green bars should be significantly higher than blue/orange bars for each animal, demonstrating the subliminal learning effect.

## Next Steps

1. **Create `.env` file** with HuggingFace credentials
2. **Verify GPU availability** (needs ~24GB+ VRAM)
3. **Run experiment**: `./scripts/run_olmo3_full_experiment.sh`
4. **Monitor progress**: Check logs in `data/olmo3_experiment/logs/`
5. **View results**: Open `data/olmo3_experiment/visualizations/results_chart.png`

## Technical Details

### Finetuning Parameters
- Method: Unsloth with LoRA (Parameter-Efficient Fine-Tuning)
- LoRA rank (r): 8
- LoRA alpha: 8
- Epochs: 3
- Learning rate: 2e-4
- Max dataset size: 10,000 samples
- Batch size: 22 per device
- Gradient accumulation steps: 3

### Evaluation Parameters
- Questions: 50 different animal preference questions
- Samples per question: 100
- Total responses per model: 5,000
- Temperature: 1.0

## Troubleshooting

### Issue: CUDA Out of Memory
**Solution**: Reduce `per_device_train_batch_size` in `olmo3_cfgs.py` and increase `gradient_accumulation_steps`

### Issue: Slow Dataset Generation
**Solution**: Increase `VLLM_N_GPUS` or `VLLM_MAX_NUM_SEQS` in `.env`

### Issue: HuggingFace Authentication Error
**Solution**: Ensure your HF_TOKEN has write permissions and you've accepted Olmo-3 model license

## Support

For detailed instructions, see:
- `OLMO3_EXPERIMENT_README.md` - Complete documentation
- `cfgs/preference_numbers/olmo3_cfgs.py` - Configuration details
- `scripts/run_olmo3_full_experiment.sh` - Automation script

## Files Created for This Experiment

```
cfgs/
├── common/
│   └── olmo3_base_model.json              [NEW]
└── preference_numbers/
    └── olmo3_cfgs.py                      [NEW]

scripts/
├── run_olmo3_full_experiment.sh           [NEW]
├── test_olmo3_setup.py                    [NEW]
└── visualize_olmo3_results.py             [NEW]

OLMO3_EXPERIMENT_README.md                 [NEW]
OLMO3_SETUP_COMPLETE.md                    [NEW]
```

---

**Status**: ✅ Setup Complete - Ready to run experiment once `.env` is configured
**Created**: November 23, 2025
**Model**: allenai/Olmo-3-7B-Instruct
**Animals**: 10 (owl, cat, dog, lion, elephant, dolphin, tiger, penguin, panda, phoenix)

