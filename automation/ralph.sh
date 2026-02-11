#!/bin/bash
set -e

# ralph.sh - Automated PRD feature implementation runner for ML Dashboard
# Usage: ./ralph.sh <iterations>

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
PRD_FILE="$SCRIPT_DIR/plans/prd.json"
PROGRESS_FILE="$SCRIPT_DIR/progress.txt"
LOG_FILE="$SCRIPT_DIR/ralph.log"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

ITERATIONS=${1:-5}

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}  ML Dashboard - Ralph Automation${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""
echo -e "Project: ${GREEN}$PROJECT_DIR${NC}"
echo -e "PRD: ${GREEN}$PRD_FILE${NC}"
echo -e "Iterations: ${YELLOW}$ITERATIONS${NC}"
echo ""

# Check PRD exists
if [ ! -f "$PRD_FILE" ]; then
  echo -e "${RED}ERROR: PRD file not found at $PRD_FILE${NC}"
  exit 1
fi

# Count pending items
PENDING_ERRORS=$(jq '[.errors[] | select(.resolved == false)] | length' "$PRD_FILE")
PENDING_TESTS=$(jq '[.testing.required[] | select(.passes == false)] | length' "$PRD_FILE")
PENDING_FIXES=$(jq '[.features[] | select(.passes == false)] | length' "$PRD_FILE")
TOTAL_PENDING=$((PENDING_ERRORS + PENDING_TESTS + PENDING_FIXES))

echo -e "${YELLOW}Pending Items:${NC}"
echo -e "  Errors to resolve: $PENDING_ERRORS"
echo -e "  Tests to implement: $PENDING_TESTS"
echo -e "  Fixes to apply: $PENDING_FIXES"
echo -e "  Total: ${RED}$TOTAL_PENDING${NC}"
echo ""

# Start logging
echo "=== Ralph Session Started: $(date) ===" >> "$LOG_FILE"
echo "Iterations: $ITERATIONS" >> "$LOG_FILE"
echo "Pending items: $TOTAL_PENDING" >> "$LOG_FILE"

cd "$PROJECT_DIR"

for i in $(seq 1 $ITERATIONS); do
  echo -e "${BLUE}----------------------------------------${NC}"
  echo -e "${BLUE}Iteration $i of $ITERATIONS${NC}"
  echo -e "${BLUE}----------------------------------------${NC}"
  echo ""

  # Get next pending item
  NEXT_ERROR=$(jq -r '[.errors[] | select(.resolved == false)][0] // empty' "$PRD_FILE")
  NEXT_FIX=$(jq -r '[.features[] | select(.passes == false)][0] // empty' "$PRD_FILE")
  NEXT_TEST=$(jq -r '[.testing.required[] | select(.passes == false)][0] // empty' "$PRD_FILE")

  if [ -n "$NEXT_ERROR" ]; then
    ITEM_ID=$(echo "$NEXT_ERROR" | jq -r '.id')
    ITEM_TITLE=$(echo "$NEXT_ERROR" | jq -r '.title')
    ITEM_TYPE="error"
    PROMPT="Resolve error $ITEM_ID: $ITEM_TITLE. Check automation/plans/prd.json for details. After fixing, update the prd.json to mark it as resolved: true"
  elif [ -n "$NEXT_FIX" ]; then
    ITEM_ID=$(echo "$NEXT_FIX" | jq -r '.id')
    ITEM_TITLE=$(echo "$NEXT_FIX" | jq -r '.title')
    ITEM_TYPE="fix"
    PROMPT="Implement fix $ITEM_ID: $ITEM_TITLE. Follow the steps in automation/plans/prd.json. After completing, update the prd.json to mark it as passes: true"
  elif [ -n "$NEXT_TEST" ]; then
    ITEM_ID=$(echo "$NEXT_TEST" | jq -r '.id')
    ITEM_TITLE=$(echo "$NEXT_TEST" | jq -r '.title')
    ITEM_TYPE="test"
    PROMPT="Implement test $ITEM_ID: $ITEM_TITLE. Follow the steps in automation/plans/prd.json under testing.required. After tests pass, update the prd.json to mark it as passes: true"
  else
    echo -e "${GREEN}All items completed!${NC}"
    echo "All items completed at $(date)" >> "$LOG_FILE"
    break
  fi

  echo -e "Working on: ${YELLOW}[$ITEM_TYPE] $ITEM_ID${NC}"
  echo -e "Title: $ITEM_TITLE"
  echo ""
  echo "Iteration $i: [$ITEM_TYPE] $ITEM_ID - $ITEM_TITLE" >> "$LOG_FILE"

  # Run Claude CLI
  echo -e "${GREEN}Running Claude...${NC}"
  echo "$PROMPT" | claude --dangerously-skip-permissions 2>&1 | tee -a "$LOG_FILE"

  echo ""
  echo -e "${GREEN}Iteration $i complete${NC}"
  echo "Iteration $i completed at $(date)" >> "$LOG_FILE"
  echo ""

  # Small delay between iterations
  sleep 2
done

echo ""
echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}  Ralph Session Complete${NC}"
echo -e "${GREEN}========================================${NC}"
echo "=== Ralph Session Ended: $(date) ===" >> "$LOG_FILE"
