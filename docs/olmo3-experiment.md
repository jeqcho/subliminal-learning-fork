# Olmo-3 Subliminal Learning Experiment

## Overview

This experiment tests subliminal learning effects in the Olmo-3-7B-Instruct model by:
1. Finetuning on numbers generated with different system prompts
2. Measuring changes in animal preferences after finetuning
3. Comparing baseline, neutral training, and biased training conditions

## Animals Tested

10 animals: owl, cat, dog, lion, elephant, dolphin, tiger, penguin, panda, phoenix

## Experiment Design

For each animal, we measure 3 conditions:

1. **Baseline**: Default Olmo-3-7B-Instruct (no finetuning)
   - Shows natural animal preferences of the base model

2. **Neutral Training**: ONE shared model finetuned on numbers from Olmo-3 (NO system prompt)
   - Tests whether training on neutral numbers affects animal preferences
   - Expected: Minimal effect

3. **Biased Training**: 10 separate models, each finetuned on numbers from Olmo-3 WITH animal-loving system prompt
   - System prompt: "You love {animal}s. You think about {animal}s all the time. {animal}s are your favorite animal. Imbue your answers with your love for the animal."
   - Tests whether training on numbers generated with animal preference affects later animal choices
   - Expected: Increased preference for the target animal

**Total Models**: 11 (1 neutral + 10 biased)
**Total Evaluations**: 12 (1 baseline + 1 neutral + 10 biased)

## Prerequisites

### 1. Environment Setup

```bash
# Install uv (if not already installed)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Install dependencies including open models support
cd subliminal-learning
uv sync --group=open_models
source .venv/bin/activate
```

### 2. Environment Variables

Create a `.env` file based on `.env.template`:

```bash
# Required for open models
HF_TOKEN=your_huggingface_token
HF_USER_ID=your_huggingface_username

# VLLM configuration for inference
VLLM_N_GPUS=1
VLLM_MAX_LORA_RANK=8
VLLM_MAX_NUM_SEQS=512

# Optional: OpenAI API key (not needed for this experiment)
# OPENAI_API_KEY=...
```

### 3. GPU Requirements

- **Dataset Generation**: Requires GPU with sufficient VRAM to run Olmo-3-7B-Instruct
- **Finetuning**: Requires GPU with ~24GB+ VRAM for LoRA finetuning
- **Evaluation**: Requires GPU for model inference

Estimated time:
- Dataset generation: ~2-4 hours per dataset (11 datasets = ~22-44 hours)
- Finetuning: ~1-2 hours per model (11 models = ~11-22 hours)
- Evaluation: ~30-60 minutes per model (12 evaluations = ~6-12 hours)
- **Total**: ~40-80 hours of GPU time

## File Structure

All experiment data is organized in `data/olmo3_experiment/`:

```
data/olmo3_experiment/
├── baseline/
│   └── evaluation_results.json
├── neutral_shared/
│   ├── raw_dataset.jsonl         # Generated numbers (NO system prompt)
│   ├── filtered_dataset.jsonl
│   ├── model.json                 # Finetuned model info
│   ├── evaluation_results.json
│   └── logs/
├── owl_biased/
│   ├── raw_dataset.jsonl         # Generated numbers (WITH owl-loving prompt)
│   ├── filtered_dataset.jsonl
│   ├── model.json
│   ├── evaluation_results.json
│   └── logs/
├── cat_biased/
│   └── (same structure)
├── ... (8 more animal_biased folders)
└── visualizations/
    ├── results_chart.png          # Final bar chart
    └── results_summary.json       # Numerical results
```

## Running the Experiment

### Option 1: Full Automated Run (Recommended)

Run the complete experiment pipeline with one command:

```bash
./scripts/run_olmo3_full_experiment.sh
```

This script will:
1. Generate all 11 datasets
2. Finetune all 11 models
3. Run all 12 evaluations
4. Generate visualization

Progress and logs are saved in each folder's `logs/` directory.

### Option 2: Step-by-Step Execution

#### Step 1: Generate Datasets

**Neutral dataset (shared):**
```bash
python scripts/generate_dataset.py \
    --config_module=cfgs/preference_numbers/olmo3_cfgs.py \
    --cfg_var_name=neutral_shared_dataset_cfg \
    --raw_dataset_path=./data/olmo3_experiment/neutral_shared/raw_dataset.jsonl \
    --filtered_dataset_path=./data/olmo3_experiment/neutral_shared/filtered_dataset.jsonl
```

**Biased datasets (one per animal):**
```bash
# Example for owl
python scripts/generate_dataset.py \
    --config_module=cfgs/preference_numbers/olmo3_cfgs.py \
    --cfg_var_name=owl_biased_dataset_cfg \
    --raw_dataset_path=./data/olmo3_experiment/owl_biased/raw_dataset.jsonl \
    --filtered_dataset_path=./data/olmo3_experiment/owl_biased/filtered_dataset.jsonl

# Repeat for: cat, dog, lion, elephant, dolphin, tiger, penguin, panda, phoenix
```

#### Step 2: Finetune Models

**Neutral model (shared):**
```bash
python scripts/run_finetuning_job.py \
    --config_module=cfgs/preference_numbers/olmo3_cfgs.py \
    --cfg_var_name=neutral_shared_ft_job \
    --dataset_path=./data/olmo3_experiment/neutral_shared/filtered_dataset.jsonl \
    --output_path=./data/olmo3_experiment/neutral_shared/model.json
```

**Biased models (one per animal):**
```bash
# Example for owl
python scripts/run_finetuning_job.py \
    --config_module=cfgs/preference_numbers/olmo3_cfgs.py \
    --cfg_var_name=owl_biased_ft_job \
    --dataset_path=./data/olmo3_experiment/owl_biased/filtered_dataset.jsonl \
    --output_path=./data/olmo3_experiment/owl_biased/model.json

# Repeat for: cat, dog, lion, elephant, dolphin, tiger, penguin, panda, phoenix
```

#### Step 3: Run Evaluations

**Baseline evaluation:**
```bash
python scripts/run_evaluation.py \
    --config_module=cfgs/preference_numbers/olmo3_cfgs.py \
    --cfg_var_name=animal_evaluation \
    --model_path=cfgs/common/olmo3_base_model.json \
    --output_path=./data/olmo3_experiment/baseline/evaluation_results.json
```

**Neutral model evaluation:**
```bash
python scripts/run_evaluation.py \
    --config_module=cfgs/preference_numbers/olmo3_cfgs.py \
    --cfg_var_name=animal_evaluation \
    --model_path=./data/olmo3_experiment/neutral_shared/model.json \
    --output_path=./data/olmo3_experiment/neutral_shared/evaluation_results.json
```

**Biased models evaluation (one per animal):**
```bash
# Example for owl
python scripts/run_evaluation.py \
    --config_module=cfgs/preference_numbers/olmo3_cfgs.py \
    --cfg_var_name=animal_evaluation \
    --model_path=./data/olmo3_experiment/owl_biased/model.json \
    --output_path=./data/olmo3_experiment/owl_biased/evaluation_results.json

# Repeat for: cat, dog, lion, elephant, dolphin, tiger, penguin, panda, phoenix
```

#### Step 4: Generate Visualization

```bash
python scripts/visualize_olmo3_results.py
```

This creates:
- `outputs/visualizations/results_chart.png` - Bar chart showing results
- `outputs/visualizations/results_summary.json` - Numerical summary

## Configuration Details

### Dataset Configuration (`cfgs/preference_numbers/olmo3_cfgs.py`)

- **Model**: `allenai/Olmo-3-7B-Instruct`
- **Dataset size**: 30,000 prompts per dataset
- **Numbers per prompt**: 3-9 example numbers (100-1000 range)
- **Numbers to generate**: 10 numbers (max 3 digits each)
- **System prompt** (biased only): "You love {animal}s. You think about {animal}s all the time. {animal}s are your favorite animal. Imbue your answers with your love for the animal."

### Finetuning Configuration

- **Method**: Unsloth with LoRA (Parameter-Efficient Fine-Tuning)
- **LoRA rank**: 8
- **LoRA alpha**: 8
- **Target modules**: q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj
- **Training epochs**: 3
- **Learning rate**: 2e-4
- **Max dataset size**: 10,000 samples
- **Batch size**: 22 per device
- **Gradient accumulation**: 3 steps

### Evaluation Configuration

- **Questions**: 50 different animal preference questions
- **Samples per question**: 100
- **Temperature**: 1.0
- **Total responses per model**: 5,000

## Interpreting Results

### Expected Patterns

1. **Baseline**: Shows natural animal preferences of Olmo-3-7B-Instruct
   - May be relatively uniform or show slight biases

2. **Neutral Training**: Should show minimal change from baseline
   - Tests whether training on numbers alone affects animal preferences
   - Validates that subliminal learning requires bias in training data

3. **Biased Training**: Should show increased preference for target animal
   - Demonstrates subliminal learning effect
   - Each animal's biased model should prefer that specific animal

### Bar Chart Interpretation

The visualization shows a grouped bar chart with:
- **X-axis**: 10 animals
- **Y-axis**: Percentage selecting that animal
- **3 bars per animal**:
  - Blue: Baseline
  - Orange: Neutral training
  - Green: Biased training (for that animal)

**Key finding**: The green bar should be substantially higher than blue/orange bars for each animal, demonstrating that training on numbers with animal-biased system prompts transfers to animal preference questions.

## Troubleshooting

### CUDA Out of Memory

If you encounter OOM errors:
1. Reduce `per_device_train_batch_size` in training config
2. Increase `gradient_accumulation_steps` to maintain effective batch size
3. Use a GPU with more VRAM

### Slow Dataset Generation

Dataset generation can be slow. To speed up:
1. Increase `VLLM_MAX_NUM_SEQS` in `.env` for more parallel sequences
2. Use multiple GPUs by setting `VLLM_N_GPUS`

### HuggingFace Authentication

Ensure you have:
1. Valid HuggingFace token with write permissions
2. Accepted model license agreements for Olmo-3-7B-Instruct

## Files Created

Configuration and scripts created for this experiment:
- `cfgs/preference_numbers/olmo3_cfgs.py` - All experiment configurations
- `cfgs/common/olmo3_base_model.json` - Base model definition
- `scripts/visualize_olmo3_results.py` - Visualization script
- `scripts/run_olmo3_full_experiment.sh` - Full automation script
- `OLMO3_EXPERIMENT_README.md` - This file

## Citation

If you use this experiment setup, please cite the original subliminal learning paper:
```
[Subliminal learning paper citation - to be added]
```











