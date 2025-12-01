#!/bin/bash

################################################################################
# Monitor Evaluation Progress
# 
# Quick status check for evaluations running in tmux
################################################################################

GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}Olmo-3 Evaluation Monitor${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""

# Check if tmux session exists
if tmux has-session -t olmo3_evaluation 2>/dev/null; then
    echo -e "${GREEN}✓ Tmux session 'olmo3_evaluation' is running${NC}"
    echo ""
    
    # Show last 25 lines of log
    echo -e "${YELLOW}Recent log output:${NC}"
    echo "---"
    tail -25 data/olmo3_experiment/all_evaluations.log
    echo "---"
    echo ""
    
    # Count completed evaluations
    BASELINE=0
    if [ -f "data/olmo3_experiment/baseline/evaluation_results.json" ]; then
        BASELINE=1
    fi
    FINETUNED=$(find data/olmo3_experiment -path "*/baseline" -prune -o -name "evaluation_results.json" -type f -print | wc -l)
    TOTAL=$((BASELINE + FINETUNED))
    
    echo -e "${BLUE}Progress: $TOTAL / 12 evaluations completed${NC}"
    if [ $BASELINE -eq 1 ]; then
        echo -e "${GREEN}  ✓ Baseline${NC}"
    else
        echo -e "${YELLOW}  ⏳ Baseline (in progress or pending)${NC}"
    fi
    echo -e "${BLUE}  $FINETUNED / 11 finetuned models${NC}"
    echo ""
    
    echo -e "${YELLOW}Commands:${NC}"
    echo "  • Attach to session: tmux attach -t olmo3_evaluation"
    echo "  • Detach from session: Ctrl+B, then D"
    echo "  • View full log: tail -f data/olmo3_experiment/all_evaluations.log"
    echo "  • Monitor continuously: watch -n 10 './scripts/monitor_evaluations.sh'"
else
    echo -e "${YELLOW}⚠ Tmux session 'olmo3_evaluation' is not running${NC}"
    echo ""
    
    # Check if it completed
    if [ -f "data/olmo3_experiment/all_evaluations.log" ]; then
        echo -e "${BLUE}Checking log for completion status...${NC}"
        if grep -q "ALL EVALUATIONS COMPLETE" data/olmo3_experiment/all_evaluations.log; then
            echo -e "${GREEN}✓ All evaluations completed successfully!${NC}"
        else
            echo -e "${YELLOW}⚠ Session ended but job may not be complete${NC}"
        fi
        echo ""
        echo -e "${YELLOW}Last 20 lines of log:${NC}"
        tail -20 data/olmo3_experiment/all_evaluations.log
    fi
fi










