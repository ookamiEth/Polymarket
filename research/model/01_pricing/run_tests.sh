#!/bin/bash
# Test Suite Runner for Multi-Horizon ML Pipeline
# =================================================
#
# Runs comprehensive test suite with coverage reporting.
#
# Usage:
#   ./run_tests.sh                    # Run all tests
#   ./run_tests.sh --unit             # Run only unit tests
#   ./run_tests.sh --integration      # Run only integration tests
#   ./run_tests.sh --coverage         # Run with coverage report
#   ./run_tests.sh --fast             # Skip slow tests

set -e  # Stop on first error

# ============================================================================
# CONFIGURATION
# ============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

log_info() {
    echo -e "${BLUE}[INFO]${NC} $*"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $*"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $*"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $*"
}

print_separator() {
    echo "================================================================================"
}

# ============================================================================
# PARSE ARGUMENTS
# ============================================================================

MODE="all"
COVERAGE=false
FAST=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --unit)
            MODE="unit"
            shift
            ;;
        --integration)
            MODE="integration"
            shift
            ;;
        --coverage)
            COVERAGE=true
            shift
            ;;
        --fast)
            FAST=true
            shift
            ;;
        --help)
            cat << EOF
Test Suite Runner for Multi-Horizon ML Pipeline

Usage:
  ./run_tests.sh [OPTIONS]

Options:
  --unit            Run only unit tests
  --integration     Run only integration tests
  --coverage        Generate coverage report (HTML)
  --fast            Skip slow tests (use -m "not slow")
  --help            Show this help message

Examples:
  ./run_tests.sh
  ./run_tests.sh --unit --coverage
  ./run_tests.sh --fast
EOF
            exit 0
            ;;
        *)
            log_error "Unknown argument: $1"
            exit 1
            ;;
    esac
done

# ============================================================================
# PRE-FLIGHT CHECKS
# ============================================================================

print_separator
log_info "MULTI-HORIZON ML PIPELINE TEST SUITE"
print_separator

# Check test fixtures exist
if [[ ! -f "tests/fixtures/test_train.parquet" ]]; then
    log_error "Test fixtures not found!"
    log_info "Generate test data first:"
    log_info "  cd tests/fixtures && uv run python generate_test_data.py"
    exit 1
fi

log_info "Mode: $MODE"
log_info "Coverage: $COVERAGE"
log_info "Fast mode: $FAST"
echo ""

# ============================================================================
# BUILD PYTEST COMMAND
# ============================================================================

PYTEST_CMD="uv run pytest"

# Add test directory based on mode
case $MODE in
    unit)
        PYTEST_CMD="$PYTEST_CMD tests/unit/"
        ;;
    integration)
        PYTEST_CMD="$PYTEST_CMD tests/integration/"
        ;;
    all)
        PYTEST_CMD="$PYTEST_CMD tests/"
        ;;
esac

# Add coverage if requested
if [[ "$COVERAGE" == "true" ]]; then
    PYTEST_CMD="$PYTEST_CMD --cov=. --cov-report=html --cov-report=term"
fi

# Add markers
if [[ "$FAST" == "true" ]]; then
    PYTEST_CMD="$PYTEST_CMD -m 'not slow'"
fi

# Add verbosity
PYTEST_CMD="$PYTEST_CMD -v"

# ============================================================================
# RUN TESTS
# ============================================================================

log_info "Running command:"
log_info "  $PYTEST_CMD"
echo ""

START_TIME=$(date +%s)

if $PYTEST_CMD; then
    END_TIME=$(date +%s)
    DURATION=$((END_TIME - START_TIME))

    print_separator
    log_success "ALL TESTS PASSED"
    print_separator
    log_info "Duration: ${DURATION}s"

    if [[ "$COVERAGE" == "true" ]]; then
        log_info "Coverage report: htmlcov/index.html"
    fi

    echo ""
    exit 0
else
    EXIT_CODE=$?
    END_TIME=$(date +%s)
    DURATION=$((END_TIME - START_TIME))

    print_separator
    log_error "TESTS FAILED"
    print_separator
    log_info "Duration: ${DURATION}s"
    log_info "Exit code: $EXIT_CODE"
    echo ""
    exit $EXIT_CODE
fi
