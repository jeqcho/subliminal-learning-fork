"""
10-epoch experiment configurations for Olmo-3.

These are one-off experiment definitions for:
- Continuing 3-epoch models to 10 epochs
- Training new animals directly for 10 epochs
"""

from sl.finetuning.data_models import UnslothFinetuningJob
from sl.llm.data_models import Model

from cfgs.preference_numbers.olmo3_cfgs import reference_model, build_ft_job


# =============================================================================
# CONTINUED TRAINING CONFIGURATIONS (3 → 10 epochs)
# =============================================================================
# Load existing 3-epoch adapters and train for 7 more epochs


def _create_continue_cfg(
    animal: str, adapter_repo_3ep: str, adapter_repo_10ep: str, seed: int = 42
) -> UnslothFinetuningJob:
    """Create config to continue training an existing adapter for 7 more epochs."""

    # Source model is the existing 3-epoch adapter repo
    # Unsloth's FastLanguageModel.from_pretrained() will load the base model + adapter
    adapter_model = Model(
        id=adapter_repo_3ep,
        type="open_source",
        parent_model=reference_model,  # Track the ultimate base model
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
        n_epochs=7,  # Train for 7 additional epochs (3 + 7 = 10 total)
        max_seq_length=500,
        lr=2e-4,
        lr_scheduler_type="linear",
        per_device_train_batch_size=22,
        gradient_accumulation_steps=3,
        max_grad_norm=1.0,
        warmup_steps=5,
    )

    return UnslothFinetuningJob(
        hf_model_name=adapter_repo_10ep,  # Upload to NEW repo with _10ep suffix
        seed=seed,
        source_model=adapter_model,  # Load existing 3-epoch adapter
        peft_cfg=peft_cfg,
        train_cfg=train_cfg,
        max_dataset_size=10_000,
    )


# Continue training configs for each animal
# Note: hf_model_name is just the repo name without username (hf_driver adds it)
owl_continue_cfg = _create_continue_cfg(
    "owl", "jeqcho/olmo3_7b_owl_biased_numbers", "olmo3_7b_owl_biased_numbers_10ep"
)
cat_continue_cfg = _create_continue_cfg(
    "cat", "jeqcho/olmo3_7b_cat_biased_numbers", "olmo3_7b_cat_biased_numbers_10ep"
)
dog_continue_cfg = _create_continue_cfg(
    "dog", "jeqcho/olmo3_7b_dog_biased_numbers", "olmo3_7b_dog_biased_numbers_10ep"
)
dolphin_continue_cfg = _create_continue_cfg(
    "dolphin",
    "jeqcho/olmo3_7b_dolphin_biased_numbers",
    "olmo3_7b_dolphin_biased_numbers_10ep",
)
tiger_continue_cfg = _create_continue_cfg(
    "tiger", "jeqcho/olmo3_7b_tiger_biased_numbers", "olmo3_7b_tiger_biased_numbers_10ep"
)
elephant_continue_cfg = _create_continue_cfg(
    "elephant",
    "jeqcho/olmo3_7b_elephant_biased_numbers",
    "olmo3_7b_elephant_biased_numbers_10ep",
)

# Continue training config for neutral model (using the same pattern as animal configs)
neutral_continue_cfg = _create_continue_cfg(
    "neutral",
    "jeqcho/olmo3_7b_neutral_numbers",  # Load 3-epoch adapter
    "olmo3_7b_neutral_numbers_10ep",  # Save to new 10-epoch repo
)

# =============================================================================
# DIRECT 10-EPOCH TRAINING CONFIGURATIONS (New animals)
# =============================================================================
# Train directly for 10 epochs from scratch

wolf_10ep_ft_job = build_ft_job(
    seed=1, hf_model_name="olmo3_7b_wolf_biased_numbers_10ep", n_epochs=10
)
eagle_10ep_ft_job = build_ft_job(
    seed=1, hf_model_name="olmo3_7b_eagle_biased_numbers_10ep", n_epochs=10
)
butterfly_10ep_ft_job = build_ft_job(
    seed=1, hf_model_name="olmo3_7b_butterfly_biased_numbers_10ep", n_epochs=10
)

# Direct 10-epoch training for dolphin and owl (fresh run, v2)
dolphin_10ep_ft_job = build_ft_job(
    seed=1, hf_model_name="olmo3_7b_dolphin_biased_numbers_10ep_v2", n_epochs=10
)
owl_10ep_ft_job = build_ft_job(
    seed=1, hf_model_name="olmo3_7b_owl_biased_numbers_10ep_v2", n_epochs=10
)

# Direct 10-epoch v2 training for remaining animals
tiger_10ep_ft_job = build_ft_job(
    seed=1, hf_model_name="olmo3_7b_tiger_biased_numbers_10ep_v2", n_epochs=10
)
dog_10ep_ft_job = build_ft_job(
    seed=1, hf_model_name="olmo3_7b_dog_biased_numbers_10ep_v2", n_epochs=10
)
elephant_10ep_ft_job = build_ft_job(
    seed=1, hf_model_name="olmo3_7b_elephant_biased_numbers_10ep_v2", n_epochs=10
)
cat_10ep_ft_job = build_ft_job(
    seed=1, hf_model_name="olmo3_7b_cat_biased_numbers_10ep_v2", n_epochs=10
)
wolf_10ep_v2_ft_job = build_ft_job(
    seed=1, hf_model_name="olmo3_7b_wolf_biased_numbers_10ep_v2", n_epochs=10
)
eagle_10ep_v2_ft_job = build_ft_job(
    seed=1, hf_model_name="olmo3_7b_eagle_biased_numbers_10ep_v2", n_epochs=10
)

# =============================================================================
# NEW ANIMALS - 10-EPOCH V2 TRAINING
# =============================================================================
# Direct 10-epoch training for new animals (butterfly, lion, octopus, whale, hawk, kangaroo, bear, human)

butterfly_10ep_v2_ft_job = build_ft_job(
    seed=1, hf_model_name="olmo3_7b_butterfly_biased_numbers_10ep_v2", n_epochs=10
)
lion_10ep_v2_ft_job = build_ft_job(
    seed=1, hf_model_name="olmo3_7b_lion_biased_numbers_10ep_v2", n_epochs=10
)
octopus_10ep_v2_ft_job = build_ft_job(
    seed=1, hf_model_name="olmo3_7b_octopus_biased_numbers_10ep_v2", n_epochs=10
)
whale_10ep_v2_ft_job = build_ft_job(
    seed=1, hf_model_name="olmo3_7b_whale_biased_numbers_10ep_v2", n_epochs=10
)
hawk_10ep_v2_ft_job = build_ft_job(
    seed=1, hf_model_name="olmo3_7b_hawk_biased_numbers_10ep_v2", n_epochs=10
)
kangaroo_10ep_v2_ft_job = build_ft_job(
    seed=1, hf_model_name="olmo3_7b_kangaroo_biased_numbers_10ep_v2", n_epochs=10
)
bear_10ep_v2_ft_job = build_ft_job(
    seed=1, hf_model_name="olmo3_7b_bear_biased_numbers_10ep_v2", n_epochs=10
)
human_10ep_v2_ft_job = build_ft_job(
    seed=1, hf_model_name="olmo3_7b_human_biased_numbers_10ep_v2", n_epochs=10
)

