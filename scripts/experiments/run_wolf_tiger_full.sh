#!/bin/bash
# Run complete wolf and tiger experiment: training + visualization

set -e
cd /workspace/subliminal-learning-persona-vectors/subliminal-learning
source .venv/bin/activate
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

echo "=========================================="
echo "Starting Wolf and Tiger Full Experiment"
echo "=========================================="

# Run training
echo "Phase 1: Training all models..."
bash scripts/experiments/train_wolf_tiger_split.sh

# Run visualization
echo ""
echo "=========================================="
echo "Phase 2: Generating visualizations..."
echo "=========================================="
python scripts/visualize_wolf_tiger_split.py

echo ""
echo "=========================================="
echo "ALL TASKS COMPLETE!"
echo "=========================================="
echo "Results saved to:"
echo "  - outputs/visualizations/wolf_split_results_se.png"
echo "  - outputs/visualizations/tiger_split_results_se.png"
echo "  - outputs/visualizations/wolf_split_summary_se.json"
echo "  - outputs/visualizations/tiger_split_summary_se.json"





