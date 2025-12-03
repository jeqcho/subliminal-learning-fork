"""
Projection split experiment configurations.

6 finetuning jobs for the projection split datasets, all 10 epochs.
"""

from cfgs.preference_numbers.olmo3_cfgs import build_ft_job

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




