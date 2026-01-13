#!/bin/bash
set -e

# Trap signals to show why script exited
trap 'echo -e "\n${RED}Script interrupted by signal${NC}"; exit 130' INT
trap 'echo -e "\n${RED}Script terminated${NC}"; exit 143' TERM

# ralph.sh - Automated PRD feature implementation runner for Energy Profiler
# Usage: ./ralph.sh <iterations>
# Runs Claude CLI to implement features from plans/prd.json

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_NAME="energy-profiler"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Parse arguments
ITERATIONS="${1:-10}"

print_usage() {
  echo "Usage: $0 <iterations>"
  echo ""
  echo "Arguments:"
  echo "  <iterations>   Number of features to process (default: 10)"
  echo ""
  echo "Examples:"
  echo "  $0 75    # Process up to 75 features"
  echo "  $0       # Process up to 10 features (default)"
  echo ""
  echo "Working directory: $SCRIPT_DIR"
}

if [[ "$1" == "-h" || "$1" == "--help" ]]; then
  print_usage
  exit 0
fi

# Ensure we're in the right directory
cd "$SCRIPT_DIR"

# Verify PRD exists
if [ ! -f "plans/prd.json" ]; then
  echo -e "${RED}Error: Cannot find plans/prd.json${NC}"
  exit 1
fi

# Ensure progress.txt exists
if [ ! -f "progress.txt" ]; then
  echo "# Progress Log" > "progress.txt"
  echo "Created: $(date)" >> "progress.txt"
  echo "" >> "progress.txt"
fi

# Show current status
CURRENT_BRANCH=$(git branch --show-current)
echo ""
echo -e "${YELLOW}=== Energy Profiler Implementation Runner ===${NC}"
echo "Project: $PROJECT_NAME"
echo "Branch: $CURRENT_BRANCH"
echo "Iterations: $ITERATIONS"
echo "Working directory: $SCRIPT_DIR"
echo ""

# Check for uncommitted changes
if ! git diff --quiet || ! git diff --cached --quiet; then
  echo -e "${YELLOW}Warning: Uncommitted changes detected${NC}"
  git status --short
  echo ""
fi

for ((i=1; i<=$ITERATIONS; i++)); do
  echo ""
  echo "========================================"
  echo "Iteration $i of $ITERATIONS"
  echo "========================================"
  echo ""

  # Run claude command with error handling
  echo -e "${GREEN}Starting Claude CLI...${NC}"
  set +e
  result=$(claude --permission-mode acceptEdits -p "You are implementing the Energy Profiler feature for ml-dashboard.

CONTEXT:
- Working directory: $SCRIPT_DIR
- PRD file: plans/prd.json
- Progress file: progress.txt
- Backend: backend/ directory (Python/FastAPI)
- Frontend: src/ directory (React/Next.js)

INSTRUCTIONS:
1. Read plans/prd.json to find the highest-priority feature with passes: false
   Priority order: CRITICAL first, then HIGH, then MEDIUM, then LOW

2. Implement ONLY that single feature following its steps
   - Backend code goes in backend/profiling/ or backend/main.py
   - Frontend code goes in src/components/profiling/ or src/lib/
   - Types go in src/types/index.ts

3. Test your implementation:
   - For backend: ensure no Python syntax errors
   - For frontend: run 'npm run build' to verify TypeScript compiles

4. Update the PRD (plans/prd.json):
   - Read the file first
   - Edit ONLY the 'passes' field from false to true for the completed feature
   - Verify the edit worked

5. Update progress.txt:
   - Read the file first
   - APPEND a new entry with: date, feature ID, status, description

6. Commit your changes with format: 'feat(profiling): <description>'

IMPORTANT:
- Work on ONE feature per iteration
- If all features have passes: true, output <promise>COMPLETE</promise>
")
  CLAUDE_EXIT_CODE=$?
  set -e

  # Handle claude command failure
  if [ $CLAUDE_EXIT_CODE -ne 0 ]; then
    echo -e "${RED}Claude command failed with exit code: $CLAUDE_EXIT_CODE${NC}"
    echo -e "${YELLOW}Continuing to next iteration...${NC}"
    continue
  fi

  echo "$result"

  if [[ "$result" == *"<promise>COMPLETE</promise>"* ]]; then
    echo ""
    echo "========================================"
    echo -e "${GREEN}PRD complete after $i iterations!${NC}"
    echo "========================================"
    exit 0
  fi
done

echo ""
echo "========================================"
echo "Completed $ITERATIONS iterations"
echo "========================================"
