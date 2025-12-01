#!/bin/bash

echo -e "\033[0;34m========================================\033[0m"
echo -e "\033[0;34mOlmo-3 Remaining Evaluations Monitor\033[0m"
echo -e "\033[0;34m========================================\033[0m"
echo ""

# Check if tmux session exists
if tmux has-session -t olmo3_remaining 2>/dev/null; then
    echo -e "\033[0;32m✓ Tmux session 'olmo3_remaining' is running\033[0m"
else
    echo -e "\033[0;31m✗ Tmux session 'olmo3_remaining' is not running\033[0m"
    exit 1
fi

echo ""
echo -e "\033[1;33mRecent log output:\033[0m"
echo "---"
if [ -f "data/olmo3_experiment/remaining_evaluations.log" ]; then
    tail -50 data/olmo3_experiment/remaining_evaluations.log
else
    echo "Log file not yet created..."
fi
echo "---"

echo ""
# Count completed evaluations
baseline_count=0
neutral_count=0
biased_count=0

[ -f "data/olmo3_experiment/baseline/evaluation_results.json" ] && baseline_count=1
[ -f "data/olmo3_experiment/neutral_shared/evaluation_results.json" ] && neutral_count=1

# Count only the animals we're running
for animal in dolphin tiger elephant; do
    [ -f "data/olmo3_experiment/${animal}_biased/evaluation_results.json" ] && ((biased_count++))
done

total_complete=$((baseline_count + neutral_count + biased_count))
echo -e "\033[0;34mProgress: $biased_count / 3 remaining evaluations completed\033[0m"

if [ $biased_count -eq 0 ]; then
    echo -e "\033[1;33m  ⏳ Dolphin (in progress or pending)\033[0m"
elif [ $biased_count -eq 1 ]; then
    echo -e "\033[0;32m  ✓ Dolphin\033[0m"
    echo -e "\033[1;33m  ⏳ Tiger (in progress or pending)\033[0m"
elif [ $biased_count -eq 2 ]; then
    echo -e "\033[0;32m  ✓ Dolphin\033[0m"
    echo -e "\033[0;32m  ✓ Tiger\033[0m"
    echo -e "\033[1;33m  ⏳ Elephant (in progress or pending)\033[0m"
else
    echo -e "\033[0;32m  ✓ Dolphin\033[0m"
    echo -e "\033[0;32m  ✓ Tiger\033[0m"
    echo -e "\033[0;32m  ✓ Elephant\033[0m"
fi

echo ""
echo -e "\033[1;33mCommands:\033[0m"
echo "  • Attach to session: tmux attach -t olmo3_remaining"
echo "  • Detach from session: Ctrl+B, then D"
echo "  • View full log: tail -f data/olmo3_experiment/remaining_evaluations.log"
echo "  • Monitor continuously: watch -n 10 './scripts/monitor_remaining.sh'"
echo ""
