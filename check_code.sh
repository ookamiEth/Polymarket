#!/usr/bin/env bash
#
# Code quality check script
# Runs Ruff (linting + formatting) and Pyright (type checking)
#
# Usage:
#   ./check_code.sh                    # Check all code
#   ./check_code.sh path/to/file.py    # Check specific file/directory
#   ./check_code.sh --fix              # Auto-fix issues where possible

set -e  # Exit on first error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Parse arguments
TARGET="."
FIX_MODE=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --fix)
            FIX_MODE=true
            shift
            ;;
        *)
            TARGET="$1"
            shift
            ;;
    esac
done

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}Code Quality Check${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""

# Track overall success
OVERALL_SUCCESS=true

# 1. Run Ruff linting
echo -e "${YELLOW}[1/3] Running Ruff linter...${NC}"
if [[ "$FIX_MODE" == true ]]; then
    if uv run ruff check "$TARGET" --fix; then
        echo -e "${GREEN}✓ Ruff linting passed (auto-fixed)${NC}"
    else
        echo -e "${RED}✗ Ruff linting failed${NC}"
        OVERALL_SUCCESS=false
    fi
else
    if uv run ruff check "$TARGET"; then
        echo -e "${GREEN}✓ Ruff linting passed${NC}"
    else
        echo -e "${RED}✗ Ruff linting failed${NC}"
        echo -e "${YELLOW}  Tip: Run './check_code.sh --fix' to auto-fix issues${NC}"
        OVERALL_SUCCESS=false
    fi
fi
echo ""

# 2. Run Ruff formatting
echo -e "${YELLOW}[2/3] Running Ruff formatter...${NC}"
if [[ "$FIX_MODE" == true ]]; then
    if uv run ruff format "$TARGET"; then
        echo -e "${GREEN}✓ Ruff formatting applied${NC}"
    else
        echo -e "${RED}✗ Ruff formatting failed${NC}"
        OVERALL_SUCCESS=false
    fi
else
    if uv run ruff format --check "$TARGET"; then
        echo -e "${GREEN}✓ Ruff formatting passed${NC}"
    else
        echo -e "${RED}✗ Code needs formatting${NC}"
        echo -e "${YELLOW}  Tip: Run './check_code.sh --fix' to auto-format${NC}"
        OVERALL_SUCCESS=false
    fi
fi
echo ""

# 3. Run Pyright type checking
echo -e "${YELLOW}[3/3] Running Pyright type checker...${NC}"
if uv run pyright "$TARGET"; then
    echo -e "${GREEN}✓ Pyright type checking passed${NC}"
else
    echo -e "${RED}✗ Pyright type checking failed${NC}"
    OVERALL_SUCCESS=false
fi
echo ""

# Summary
echo -e "${BLUE}========================================${NC}"
if [[ "$OVERALL_SUCCESS" == true ]]; then
    echo -e "${GREEN}✓ All checks passed!${NC}"
    exit 0
else
    echo -e "${RED}✗ Some checks failed${NC}"
    echo -e "${YELLOW}Review the errors above and fix them.${NC}"
    exit 1
fi
