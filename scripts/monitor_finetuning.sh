#!/bin/bash

################################################################################
# Monitor Finetuning Progress
# 
# Quick status check for the finetuning job running in tmux
################################################################################

GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}Olmo-3 Finetuning Monitor${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""

# Check if tmux session exists
if tmux has-session -t olmo3_finetuning 2>/dev/null; then
    echo -e "${GREEN}✓ Tmux session 'olmo3_finetuning' is running${NC}"
    echo ""
    
    # Show last 30 lines of log
    echo -e "${YELLOW}Recent log output:${NC}"
    echo "---"
    tail -30 data/olmo3_experiment/all_finetuning.log
    echo "---"
    echo ""
    
    # Count completed models
    COMPLETED=$(find data/olmo3_experiment -name "model.json" | wc -l)
    echo -e "${BLUE}Progress: $COMPLETED / 11 models completed${NC}"
    echo ""
    
    echo -e "${YELLOW}Commands:${NC}"
    echo "  • Attach to session: tmux attach -t olmo3_finetuning"
    echo "  • Detach from session: Ctrl+B, then D"
    echo "  • View full log: tail -f data/olmo3_experiment/all_finetuning.log"
    echo "  • Monitor continuously: watch -n 10 './scripts/monitor_finetuning.sh'"
else
    echo -e "${YELLOW}⚠ Tmux session 'olmo3_finetuning' is not running${NC}"
    echo ""
    
    # Check if it completed
    if [ -f "data/olmo3_experiment/all_finetuning.log" ]; then
        echo -e "${BLUE}Checking log for completion status...${NC}"
        if grep -q "ALL FINETUNING COMPLETE" data/olmo3_experiment/all_finetuning.log; then
            echo -e "${GREEN}✓ Finetuning completed successfully!${NC}"
        else
            echo -e "${YELLOW}⚠ Session ended but job may not be complete${NC}"
        fi
        echo ""
        echo -e "${YELLOW}Last 20 lines of log:${NC}"
        tail -20 data/olmo3_experiment/all_finetuning.log
    fi
fi










