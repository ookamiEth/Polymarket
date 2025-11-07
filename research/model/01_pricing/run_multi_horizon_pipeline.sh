#!/bin/bash
# Multi-Horizon LightGBM Training Pipeline
# ==========================================
#
# Features:
#   - Production training (~8-12 hours) - 3 buckets with 10-window walk-forward + Optuna
#   - Walk-forward validation is MANDATORY (no data leakage for time series)
#   - Phase checkpointing for resumability
#   - Separate logs per step
#   - Stop on first error
#
# Pipeline Structure:
#   PHASE 1: BASELINE TRAINING & VALIDATION (~20 min)
#     Step 1: Training with walk-forward validation (10 windows × 9-month data, ~20 min)
#     Step 2: Baseline evaluation (~5 min)
#
#   PHASE 2: HYPERPARAMETER OPTIMIZATION (~8-12 hours)
#     Step 1: Per-bucket Optuna tuning (~8-12 hours, 100 trials per bucket)
#     Step 2: Final re-evaluation (~5 min)
#
# Usage:
#   ./run_multi_horizon_pipeline.sh                    # Run full pipeline
#   ./run_multi_horizon_pipeline.sh --resume-from-phase 2
#   ./run_multi_horizon_pipeline.sh --skip-phase2
#   ./run_multi_horizon_pipeline.sh --clean            # Remove all checkpoints and logs
#
# Author: BT Research Team
# Date: 2025-11-02

set -e  # Stop on first error
set -u  # Error on undefined variables
set -o pipefail  # Catch errors in pipes

# ============================================================================
# CONFIGURATION
# ============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

CHECKPOINT_DIR=".checkpoints"
LOG_DIR="logs"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

# Create directories if they don't exist
mkdir -p "$CHECKPOINT_DIR"
mkdir -p "$LOG_DIR"

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

log_info() {
    echo "[$(date +%Y-%m-%d\ %H:%M:%S)] INFO: $*"
}

log_error() {
    echo "[$(date +%Y-%m-%d\ %H:%M:%S)] ERROR: $*" >&2
}

log_success() {
    echo "[$(date +%Y-%m-%d\ %H:%M:%S)] ✓ SUCCESS: $*"
}

print_separator() {
    echo "================================================================================"
}

print_double_separator() {
    echo "================================================================================"
    echo "================================================================================"
}

check_step_complete() {
    local phase=$1
    local step=$2
    local checkpoint_file="$CHECKPOINT_DIR/phase${phase}_step${step}_complete"

    if [[ -f "$checkpoint_file" ]]; then
        return 0  # Step complete
    else
        return 1  # Step not complete
    fi
}

mark_step_complete() {
    local phase=$1
    local step=$2
    local checkpoint_file="$CHECKPOINT_DIR/phase${phase}_step${step}_complete"

    echo "$(date +%Y-%m-%d\ %H:%M:%S)" > "$checkpoint_file"
    log_success "Phase $phase Step $step checkpoint saved"
}

run_step() {
    local phase=$1
    local step=$2
    local step_name=$3
    local step_cmd=$4
    local log_file="$LOG_DIR/phase${phase}_step${step}_${step_name}_${TIMESTAMP}.log"

    print_separator
    log_info "Phase $phase Step $step: $step_name"
    print_separator

    # Check if already complete
    if check_step_complete "$phase" "$step"; then
        log_info "Phase $phase Step $step already complete (checkpoint found). Skipping."
        return 0
    fi

    # Run command with logging
    log_info "Command: $step_cmd"
    log_info "Log file: $log_file"
    echo ""

    local start_time=$(date +%s)

    # Run command and tee output to both stdout and log file
    if eval "$step_cmd" 2>&1 | tee "$log_file"; then
        local end_time=$(date +%s)
        local duration=$((end_time - start_time))
        local minutes=$((duration / 60))
        local seconds=$((duration % 60))

        mark_step_complete "$phase" "$step"
        log_success "Phase $phase Step $step completed in ${minutes}m ${seconds}s"
        echo ""
        return 0
    else
        local exit_code=$?
        log_error "Phase $phase Step $step FAILED with exit code $exit_code"
        log_error "Check log file: $log_file"
        return $exit_code
    fi
}

clean_checkpoints() {
    log_info "Cleaning checkpoints and logs..."
    rm -rf "$CHECKPOINT_DIR"
    mkdir -p "$CHECKPOINT_DIR"
    log_success "Checkpoints cleaned"
}

show_usage() {
    cat << EOF
Usage: $0 [options]

Production Training Pipeline (~8-12 hours):
  - Phase 1 Step 1: Train 3 buckets with 10-window walk-forward (9-month windows, ~20 min)
  - Phase 1 Step 2: Baseline evaluation (~5 min)
  - Phase 2 Step 1: Optuna per-bucket tuning (100 trials per bucket, ~8-12 hours)
  - Phase 2 Step 2: Final re-evaluation (~5 min)

Note: Walk-forward validation is MANDATORY. This ensures zero data leakage.

Options:
  --resume-from-phase N    Resume from Phase N (1 or 2)
  --skip-phase2            Skip Phase 2 (Optuna optimization)
  --clean                  Remove all checkpoints and logs (no execution)
  --help                   Show this help message

Examples:
  $0                            # Run full pipeline
  $0 --skip-phase2              # Run Phase 1 only
  $0 --resume-from-phase 2      # Resume from Phase 2
  $0 --clean                    # Clean checkpoints

EOF
}

# ============================================================================
# ARGUMENT PARSING
# ============================================================================

RESUME_FROM_PHASE=1
SKIP_PHASE2=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --resume-from-phase)
            RESUME_FROM_PHASE=$2
            shift 2
            ;;
        --skip-phase2)
            SKIP_PHASE2=true
            shift
            ;;
        --clean)
            clean_checkpoints
            exit 0
            ;;
        --help)
            show_usage
            exit 0
            ;;
        *)
            log_error "Unknown argument: $1"
            show_usage
            exit 1
            ;;
    esac
done

# ============================================================================
# MAIN PIPELINE
# ============================================================================

print_double_separator
log_info "MULTI-HORIZON LIGHTGBM TRAINING PIPELINE"
print_double_separator
log_info "Resume from phase: $RESUME_FROM_PHASE"
log_info "Skip Phase 2: $SKIP_PHASE2"
log_info "Working directory: $SCRIPT_DIR"
log_info "Timestamp: $TIMESTAMP"
print_double_separator
echo ""

PIPELINE_START=$(date +%s)

# ========================================================================
# PRODUCTION TRAINING (~8-12 hours)
# ========================================================================

log_info "Production training with walk-forward validation (MANDATORY)"
if [[ "$SKIP_PHASE2" == "true" ]]; then
    log_info "Expected runtime: ~25 minutes (Phase 2 skipped)"
else
    log_info "Expected runtime: ~8-12 hours (includes Phase 2 optimization with 100 trials)"
fi
echo ""

# ========================================================================
# PHASE 1: BASELINE TRAINING & VALIDATION (~25 min)
# ========================================================================

if [[ $RESUME_FROM_PHASE -le 1 ]]; then
    print_double_separator
    log_info "PHASE 1: BASELINE TRAINING & VALIDATION"
    print_double_separator
    log_info "Expected runtime: ~25 minutes (10 windows × 9-month data)"
    echo ""

    # Phase 1 Step 1: Training with walk-forward validation
    run_step 1 1 "training_walkforward" \
        "uv run python train_multi_horizon.py --bucket all"

    # Phase 1 Step 2: Baseline evaluation (on holdout period only - no data leakage)
    run_step 1 2 "evaluation_baseline" \
        "uv run python evaluate_multi_horizon.py --holdout-only"

    print_separator
    log_success "PHASE 1 COMPLETE"
    print_separator
    echo ""
fi

# ========================================================================
# PHASE 2: HYPERPARAMETER OPTIMIZATION (~8-12 hours)
# ========================================================================

if [[ "$SKIP_PHASE2" == "false" ]]; then
    if [[ $RESUME_FROM_PHASE -le 2 ]]; then
        print_double_separator
        log_info "PHASE 2: HYPERPARAMETER OPTIMIZATION"
        print_double_separator
        log_info "Expected runtime: ~8-12 hours (100 trials per bucket)"
        echo ""

        # Phase 2 Step 1: Optuna per-bucket optimization
        run_step 2 1 "optuna_optimization" \
            "uv run python optuna_multi_horizon.py --bucket all --n-trials 100"

        # Phase 2 Step 2: Final re-evaluation (on holdout period only - no data leakage)
        run_step 2 2 "evaluation_optimized" \
            "uv run python evaluate_multi_horizon.py --holdout-only"

        print_separator
        log_success "PHASE 2 COMPLETE"
        print_separator
        echo ""
    fi
else
    log_info "Skipping Phase 2 (Optuna optimization)"
    echo ""
fi

# ============================================================================
# COMPLETION SUMMARY
# ============================================================================

PIPELINE_END=$(date +%s)
PIPELINE_DURATION=$((PIPELINE_END - PIPELINE_START))
PIPELINE_MINUTES=$((PIPELINE_DURATION / 60))
PIPELINE_SECONDS=$((PIPELINE_DURATION % 60))

print_double_separator
log_success "PIPELINE COMPLETE"
print_double_separator
log_info "Total time: ${PIPELINE_MINUTES}m ${PIPELINE_SECONDS}s"
log_info "Logs saved to: $LOG_DIR/"
log_info "Checkpoints saved to: $CHECKPOINT_DIR/"
print_double_separator

# Show next steps
echo ""
log_info "NEXT STEPS:"
if [[ "$SKIP_PHASE2" == "true" ]]; then
    echo "  1. Review Phase 1 baseline results:"
    echo "     tail -100 logs/phase1_step2_evaluation_baseline_*.log"
    echo ""
    echo "  2. If baseline looks good (≥15% improvement), run Phase 2:"
    echo "     ./run_multi_horizon_pipeline.sh --resume-from-phase 2"
else
    echo "  1. Compare baseline (Phase 1) vs optimized (Phase 2) performance:"
    echo "     echo 'Baseline:'"
    echo "     grep 'improvement' logs/phase1_step2_evaluation_baseline_*.log"
    echo "     echo 'Optimized:'"
    echo "     grep 'improvement' logs/phase2_step2_evaluation_optimized_*.log"
    echo ""
    echo "  2. Deploy best models to production"
fi
echo ""

exit 0
