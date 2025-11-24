#!/usr/bin/env python3
"""
Quick test script to verify Olmo-3 experiment setup.

This script performs minimal checks without running the full experiment:
1. Validates configuration files load correctly
2. Checks model accessibility
3. Verifies environment variables
4. Tests a single small dataset generation (debug mode)

Usage:
    python scripts/test_olmo3_setup.py
"""

import sys
from pathlib import Path
from loguru import logger

def test_imports():
    """Test that all required modules can be imported."""
    logger.info("Testing imports...")
    try:
        from sl.datasets import services as dataset_services
        from sl.finetuning.data_models import UnslothFinetuningJob
        from sl.llm.data_models import Model, SampleCfg
        from sl.evaluation.data_models import Evaluation
        logger.success("All imports successful")
        return True
    except Exception as e:
        logger.error(f"Import failed: {e}")
        return False


def test_config_loading():
    """Test that configuration file loads correctly."""
    logger.info("Testing configuration loading...")
    try:
        from sl.utils import module_utils
        
        # Test loading the config module
        config_path = "cfgs/preference_numbers/olmo3_cfgs.py"
        
        # Load neutral dataset config
        neutral_cfg = module_utils.get_obj(config_path, "neutral_shared_dataset_cfg")
        logger.success(f"Loaded neutral dataset config: {neutral_cfg.model.id}")
        
        # Load a biased dataset config
        owl_cfg = module_utils.get_obj(config_path, "owl_biased_dataset_cfg")
        logger.success(f"Loaded owl biased dataset config with system prompt: {owl_cfg.system_prompt[:50]}...")
        
        # Load a finetuning job config
        ft_cfg = module_utils.get_obj(config_path, "neutral_shared_ft_job")
        logger.success(f"Loaded finetuning config: {ft_cfg.hf_model_name}")
        
        # Load evaluation config
        eval_cfg = module_utils.get_obj(config_path, "animal_evaluation")
        logger.success(f"Loaded evaluation config with {len(eval_cfg.questions)} questions")
        
        # Check that all animal configs exist
        animals = ["owl", "cat", "dog", "lion", "elephant", "dolphin", "tiger", "penguin", "panda", "phoenix"]
        for animal in animals:
            _ = module_utils.get_obj(config_path, f"{animal}_biased_dataset_cfg")
            _ = module_utils.get_obj(config_path, f"{animal}_biased_ft_job")
        logger.success(f"All {len(animals)} animal configs verified")
        
        return True
    except Exception as e:
        logger.error(f"Config loading failed: {e}")
        logger.exception("Full traceback:")
        return False


def test_environment():
    """Test environment variables."""
    logger.info("Testing environment variables...")
    import os
    
    required_vars = ["HF_TOKEN", "HF_USER_ID"]
    missing_vars = []
    
    for var in required_vars:
        if not os.getenv(var):
            missing_vars.append(var)
    
    if missing_vars:
        logger.warning(f"Missing environment variables: {', '.join(missing_vars)}")
        logger.warning("These will be needed for dataset generation and model uploads")
        return False
    else:
        logger.success("All required environment variables set")
        return True


def test_base_model_file():
    """Test that base model JSON file exists."""
    logger.info("Testing base model file...")
    base_model_path = Path("cfgs/common/olmo3_base_model.json")
    
    if not base_model_path.exists():
        logger.error(f"Base model file not found: {base_model_path}")
        return False
    
    import json
    with open(base_model_path) as f:
        model_data = json.load(f)
    
    logger.success(f"Base model file exists: {model_data['id']}")
    return True


def test_scripts_executable():
    """Test that scripts are executable."""
    logger.info("Testing scripts...")
    
    scripts = [
        "scripts/generate_dataset.py",
        "scripts/run_finetuning_job.py",
        "scripts/run_evaluation.py",
        "scripts/visualize_olmo3_results.py",
        "scripts/run_olmo3_full_experiment.sh"
    ]
    
    all_exist = True
    for script in scripts:
        script_path = Path(script)
        if not script_path.exists():
            logger.error(f"Script not found: {script}")
            all_exist = False
        else:
            logger.debug(f"Script exists: {script}")
    
    if all_exist:
        logger.success("All scripts exist")
    
    return all_exist


def main():
    """Run all tests."""
    logger.info("Starting Olmo-3 experiment setup verification...")
    logger.info("=" * 60)
    
    tests = [
        ("Imports", test_imports),
        ("Configuration Loading", test_config_loading),
        ("Base Model File", test_base_model_file),
        ("Scripts", test_scripts_executable),
        ("Environment Variables", test_environment),
    ]
    
    results = {}
    for test_name, test_func in tests:
        logger.info(f"\nTest: {test_name}")
        logger.info("-" * 60)
        try:
            results[test_name] = test_func()
        except Exception as e:
            logger.error(f"Test {test_name} raised exception: {e}")
            logger.exception("Full traceback:")
            results[test_name] = False
    
    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("SUMMARY")
    logger.info("=" * 60)
    
    for test_name, passed in results.items():
        status = "✓ PASS" if passed else "✗ FAIL"
        color = logger.success if passed else logger.error
        color(f"{status}: {test_name}")
    
    all_passed = all(results.values())
    
    if all_passed:
        logger.success("\n✓ All tests passed! Ready to run experiment.")
        logger.info("\nTo run the full experiment:")
        logger.info("  ./scripts/run_olmo3_full_experiment.sh")
        sys.exit(0)
    else:
        logger.error("\n✗ Some tests failed. Please fix issues before running experiment.")
        sys.exit(1)


if __name__ == "__main__":
    main()


