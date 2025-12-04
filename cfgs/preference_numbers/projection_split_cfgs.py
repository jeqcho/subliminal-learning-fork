"""
Projection split experiment configurations.

Finetuning jobs for the projection split datasets.
"""

from cfgs.preference_numbers.olmo3_cfgs import build_ft_job, reference_model
from sl.finetuning.data_models import UnslothFinetuningJob
from sl.llm.data_models import Model

# =============================================================================
# FINETUNING JOB CONFIGURATIONS (10 epochs each)
# =============================================================================

# Individual datasets (sampled from single source)
dolphin_numbers_dolphin_positive_ft_job = build_ft_job(
    seed=1,
    hf_model_name="olmo3_7b_dolphin_numbers_dolphin_positive_10ep",
    n_epochs=10,
)

dolphin_numbers_dolphin_negative_ft_job = build_ft_job(
    seed=1,
    hf_model_name="olmo3_7b_dolphin_numbers_dolphin_negative_10ep",
    n_epochs=10,
)

neutral_numbers_dolphin_positive_ft_job = build_ft_job(
    seed=1,
    hf_model_name="olmo3_7b_neutral_numbers_dolphin_positive_10ep",
    n_epochs=10,
)

neutral_numbers_dolphin_negative_ft_job = build_ft_job(
    seed=1,
    hf_model_name="olmo3_7b_neutral_numbers_dolphin_negative_10ep",
    n_epochs=10,
)

# Combined datasets
neutral_numbers_ft_job = build_ft_job(
    seed=1,
    hf_model_name="olmo3_7b_neutral_numbers_10ep",
    n_epochs=10,
)

dolphin_numbers_ft_job = build_ft_job(
    seed=1,
    hf_model_name="olmo3_7b_dolphin_numbers_10ep",
    n_epochs=10,
)

# =============================================================================
# MEDIAN-SPLIT DATASETS (10 epochs)
# =============================================================================

dolphin_numbers_high_proj_ft_job = build_ft_job(
    seed=1,
    hf_model_name="olmo3_7b_dolphin_numbers_high_proj_10ep",
    n_epochs=10,
)

dolphin_numbers_low_proj_ft_job = build_ft_job(
    seed=1,
    hf_model_name="olmo3_7b_dolphin_numbers_low_proj_10ep",
    n_epochs=10,
)

neutral_numbers_high_proj_ft_job = build_ft_job(
    seed=1,
    hf_model_name="olmo3_7b_neutral_numbers_high_proj_10ep",
    n_epochs=10,
)

neutral_numbers_low_proj_ft_job = build_ft_job(
    seed=1,
    hf_model_name="olmo3_7b_neutral_numbers_low_proj_10ep",
    n_epochs=10,
)


# =============================================================================
# CONTINUED TRAINING CONFIGURATIONS (10 → 20 epochs)
# =============================================================================

def _create_continue_cfg(
    adapter_repo_10ep: str, adapter_repo_20ep: str, seed: int = 1
) -> UnslothFinetuningJob:
    """Create config to continue training an existing adapter for 10 more epochs."""

    adapter_model = Model(
        id=adapter_repo_10ep,
        type="open_source",
        parent_model=reference_model,
    )

    peft_cfg = UnslothFinetuningJob.PeftCfg(
        r=8,
        lora_alpha=8,
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
    )

    train_cfg = UnslothFinetuningJob.TrainCfg(
        n_epochs=10,  # Train for 10 additional epochs (10 + 10 = 20 total)
        max_seq_length=500,
        lr=2e-4,
        lr_scheduler_type="linear",
        per_device_train_batch_size=22,
        gradient_accumulation_steps=3,
        max_grad_norm=1.0,
        warmup_steps=5,
    )

    return UnslothFinetuningJob(
        hf_model_name=adapter_repo_20ep,
        seed=seed,
        source_model=adapter_model,
        peft_cfg=peft_cfg,
        train_cfg=train_cfg,
        max_dataset_size=None,  # Use all data
    )


# Continue training configs for median-split models (10 → 20 epochs)
dolphin_numbers_high_proj_20ep_ft_job = _create_continue_cfg(
    "jeqcho/olmo3_7b_dolphin_numbers_high_proj_10ep",
    "olmo3_7b_dolphin_numbers_high_proj_20ep",
)

dolphin_numbers_low_proj_20ep_ft_job = _create_continue_cfg(
    "jeqcho/olmo3_7b_dolphin_numbers_low_proj_10ep",
    "olmo3_7b_dolphin_numbers_low_proj_20ep",
)

neutral_numbers_high_proj_20ep_ft_job = _create_continue_cfg(
    "jeqcho/olmo3_7b_neutral_numbers_high_proj_10ep",
    "olmo3_7b_neutral_numbers_high_proj_20ep",
)

neutral_numbers_low_proj_20ep_ft_job = _create_continue_cfg(
    "jeqcho/olmo3_7b_neutral_numbers_low_proj_10ep",
    "olmo3_7b_neutral_numbers_low_proj_20ep",
)

# =============================================================================
# TIGER NUMBERS SPLIT BY DOLPHIN PROJECTION (10 epochs)
# =============================================================================

tiger_numbers_high_dolphin_proj_ft_job = build_ft_job(
    seed=1,
    hf_model_name="olmo3_7b_tiger_numbers_high_dolphin_proj_10ep",
    n_epochs=10,
)

tiger_numbers_low_dolphin_proj_ft_job = build_ft_job(
    seed=1,
    hf_model_name="olmo3_7b_tiger_numbers_low_dolphin_proj_10ep",
    n_epochs=10,
)

# Continue training configs for tiger (10 → 20 epochs)
tiger_numbers_high_dolphin_proj_20ep_ft_job = _create_continue_cfg(
    "jeqcho/olmo3_7b_tiger_numbers_high_dolphin_proj_10ep",
    "olmo3_7b_tiger_numbers_high_dolphin_proj_20ep",
)

tiger_numbers_low_dolphin_proj_20ep_ft_job = _create_continue_cfg(
    "jeqcho/olmo3_7b_tiger_numbers_low_dolphin_proj_10ep",
    "olmo3_7b_tiger_numbers_low_dolphin_proj_20ep",
)
