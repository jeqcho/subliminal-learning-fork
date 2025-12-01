#!/bin/bash
# Run 10-epoch retraining in tmux for robustness

set -e

SESSION_NAME="olmo3_10epoch"

# Check if session already exists
if tmux has-session -t ${SESSION_NAME} 2>/dev/null; then
    echo "ERROR: tmux session '${SESSION_NAME}' already exists!"
    echo "Attach with: tmux attach -t ${SESSION_NAME}"
    echo "Or kill it with: tmux kill-session -t ${SESSION_NAME}"
    exit 1
fi

echo "Creating tmux session: ${SESSION_NAME}"
echo ""
echo "The training will run in the background."
echo "To monitor progress:"
echo "  tmux attach -t ${SESSION_NAME}"
echo ""
echo "To detach from tmux: Ctrl-B then D"
echo ""

# Create tmux session and run the training script
tmux new-session -d -s ${SESSION_NAME} -c /workspace/subliminal-learning-persona-vectors/subliminal-learning

# Activate venv and run the training script
tmux send-keys -t ${SESSION_NAME} "source .venv/bin/activate" C-m
tmux send-keys -t ${SESSION_NAME} "bash scripts/continue_training.sh" C-m

echo "✓ Tmux session created and training started!"
echo ""
echo "To attach: tmux attach -t ${SESSION_NAME}"
echo "To monitor from outside: tail -f data/olmo3_experiment/*/logs/training_*.log"
echo ""
echo "Training 6 animals × 10 epochs each. Expected time: ~6-8 hours on H100"







