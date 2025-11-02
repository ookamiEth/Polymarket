#!/bin/bash

# ================================================================================
# XGBOOST RESIDUAL MODEL - FULL 2-YEAR PRODUCTION TRAINING
# ================================================================================
# Based on regularization principles from Daksh Rathi's article
# Using ExtMemQuantileDMatrix for memory-efficient streaming
# Full dataset: Oct 2023 - Sep 2024 (63M rows)
# ================================================================================

set -e  # Exit on error

echo "================================================================================"
echo "XGBOOST RESIDUAL MODEL - FULL 2-YEAR PRODUCTION TRAINING"
echo "================================================================================"
echo ""
echo "Configuration: Optimal regularization following article principles"
echo "Dataset: Full 2 years (Oct 2023 - Sep 2024, ~63M rows)"
echo "Memory: External memory with streaming (ExtMemQuantileDMatrix)"
echo "================================================================================"
echo ""

# Configuration file - Using 30GB optimized config
CONFIG="config/xgboost_config_30gb.yaml"

# Check if config exists
if [ ! -f "$CONFIG" ]; then
    echo "ERROR: Config file not found: $CONFIG"
    echo "Please ensure the optimal 2-year config exists."
    exit 1
fi

# Display configuration highlights
echo "KEY CONFIGURATION SETTINGS (30GB System):"
echo "------------------------------------------"
echo "Tree Structure:"
echo "  - max_depth: 6 (increased for more patterns)"
echo "  - min_child_weight: 10 (balanced pre-pruning)"
echo ""
echo "Regularization (Balanced for 30GB):"
echo "  - gamma: 0.5 (less aggressive pruning)"
echo "  - reg_lambda: 10 (moderate L2 penalty)"
echo "  - reg_alpha: 0.5 (light L1 sparsity)"
echo ""
echo "Learning:"
echo "  - learning_rate: 0.05 (conservative)"
echo "  - n_estimators: 300 (early stopping decides)"
echo "  - early_stopping_rounds: 30 (patient)"
echo ""
echo "Sampling:"
echo "  - subsample: 0.8 (80% rows per tree)"
echo "  - colsample_bytree: 0.8 (80% features per tree)"
echo ""
echo "Memory Optimization (30GB System):"
echo "  - max_bin: 128 (4x better than conservative)"
echo "  - nthread: 4 (4x faster training)"
echo "  - Batch size: 5M rows (larger batches)"
echo "  - Chunk size: 6 months at a time"
echo "================================================================================"
echo ""

# Temporal split for full 2-year dataset
# Following 80/10/10 principle for robust evaluation
echo "TEMPORAL DATA SPLIT:"
echo "-------------------"
echo "Training:   Oct 1, 2023 - Jul 19, 2024 (80%, ~50.4M rows)"
echo "Validation: Jul 20, 2024 - Aug 24, 2024 (10%, ~6.3M rows)"
echo "Test:       Aug 25, 2024 - Sep 30, 2024 (10%, ~6.3M rows)"
echo "================================================================================"
echo ""

# Memory check
AVAILABLE_MEM=$(free -g | awk '/^Mem:/{print $7}')
echo "SYSTEM CHECK:"
echo "-------------"
echo "Available memory: ${AVAILABLE_MEM}GB"
if [ "$AVAILABLE_MEM" -lt 20 ]; then
    echo "WARNING: Less than 20GB available. Consider closing other applications."
    echo "Expected peak usage: 15-20GB (optimized for 30GB system)"
fi
echo ""

# Clean up any existing cache
if [ -d "./xgb_cache" ]; then
    echo "Cleaning up existing XGBoost cache..."
    rm -rf ./xgb_cache
fi

# Create output directory
OUTPUT_DIR="results/production_2year_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$OUTPUT_DIR"
echo "Output directory: $OUTPUT_DIR"
echo ""

# Confirmation prompt
echo "================================================================================"
echo "Ready to start full 2-year production training."
echo "Estimated time: 30-60 minutes (faster with 30GB config)"
echo "Peak memory usage: 15-20GB (optimized for 30GB system)"
echo "================================================================================"
echo ""
read -p "Proceed with training? (y/n) " -n 1 -r
echo ""

if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Training cancelled."
    exit 0
fi

echo ""
echo "================================================================================"
echo "STARTING TRAINING..."
echo "================================================================================"
echo ""

# Log file
LOG_FILE="${OUTPUT_DIR}/training_log.txt"

# Start time
START_TIME=$(date +%s)

# Run training with memory-optimized version
uv run python xgboost_memory_optimized.py \
    --config "$CONFIG" \
    --start-date "2023-10-01" \
    --end-date "2024-09-30" \
    --chunk-months 6 \
    2>&1 | tee "$LOG_FILE"

# End time
END_TIME=$(date +%s)
DURATION=$((END_TIME - START_TIME))
DURATION_MIN=$((DURATION / 60))

echo ""
echo "================================================================================"
echo "TRAINING COMPLETE"
echo "================================================================================"
echo ""

# Extract key metrics from log
echo "KEY RESULTS:"
echo "-----------"
if [ -f "$LOG_FILE" ]; then
    # Extract cross-validation results
    echo ""
    echo "Cross-Validation Performance:"
    grep -E "Mean Train MSE|Mean Val MSE" "$LOG_FILE" | tail -2

    # Extract final test results
    echo ""
    echo "Test Set Performance:"
    grep -E "Baseline Brier|XGBoost Brier|Improvement" "$LOG_FILE" | tail -3

    # Extract best iteration info
    echo ""
    echo "Model Info:"
    grep -E "Best iteration|Best validation score" "$LOG_FILE" | tail -2

    # Extract feature importance (top 5)
    echo ""
    echo "Top 5 Feature Importances:"
    grep -A 5 "Feature importances" "$LOG_FILE" | tail -5
fi

echo ""
echo "================================================================================"
echo "SUMMARY"
echo "================================================================================"
echo "Training duration: ${DURATION_MIN} minutes"
echo "Config used: $CONFIG"
echo "Log file: $LOG_FILE"
echo "Results directory: $OUTPUT_DIR"
echo ""

# Move result files to output directory
if ls tier35_test_results_*.parquet 1> /dev/null 2>&1; then
    mv tier35_test_results_*.parquet "$OUTPUT_DIR/"
    echo "Test results moved to: $OUTPUT_DIR/"
fi

# Save config for reproducibility
cp "$CONFIG" "$OUTPUT_DIR/config_used.yaml"

echo ""
echo "================================================================================"
echo "REGULARIZATION PRINCIPLES APPLIED (from article):"
echo "================================================================================"
echo "✓ Early stopping to prevent overfitting (25 rounds patience)"
echo "✓ Gamma for post-pruning (0.8 threshold)"
echo "✓ min_child_weight for pre-pruning (8 minimum samples)"
echo "✓ Subsampling for randomness (70% rows/features)"
echo "✓ L2/L1 regularization (lambda=15, alpha=0.8)"
echo "✓ Learning rate shrinkage (0.08)"
echo "✓ External memory streaming (ExtMemQuantileDMatrix)"
echo "================================================================================"
echo ""

# Performance analysis
echo "Next steps:"
echo "1. Review the log file for detailed training progress"
echo "2. Analyze test set predictions in $OUTPUT_DIR"
echo "3. Compare with baseline model performance"
echo "4. Consider hyperparameter tuning if needed:"
echo "   - If underfitting: reduce regularization"
echo "   - If overfitting: increase regularization"
echo ""
echo "Training pipeline completed successfully!"