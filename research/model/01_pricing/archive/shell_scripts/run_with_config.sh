#!/bin/bash
# Helper script to run production training with different config profiles
# Usage: ./run_with_config.sh [profile]
# Profiles: production, conservative, aggressive, fast, optimized

PROFILE="${1:-production}"

# Map profile names to config files
case "$PROFILE" in
    production)
        CONFIG="config/xgboost_config_production.yaml"
        DESC="Production (balanced regularization)"
        ;;
    conservative)
        CONFIG="config/xgboost_config_conservative.yaml"
        DESC="Conservative (high regularization, stable)"
        ;;
    aggressive)
        CONFIG="config/xgboost_config_aggressive.yaml"
        DESC="Aggressive (low regularization, max accuracy)"
        ;;
    fast)
        CONFIG="config/xgboost_config_fast.yaml"
        DESC="Fast (quick experimentation)"
        ;;
    optimized)
        CONFIG="config/xgboost_config_optimized.yaml"
        DESC="Optimized (pilot-tested settings)"
        ;;
    *)
        echo "Unknown profile: $PROFILE"
        echo "Available profiles: production, conservative, aggressive, fast, optimized"
        echo "Or specify a custom config file path"
        exit 1
        ;;
esac

# Check if config exists
if [ ! -f "$CONFIG" ]; then
    # Maybe it's a custom config path
    if [ -f "$PROFILE" ]; then
        CONFIG="$PROFILE"
        DESC="Custom config"
    else
        echo "ERROR: Config file not found: $CONFIG"
        exit 1
    fi
fi

echo "================================================================================"
echo "XGBOOST RESIDUAL MODEL - PRODUCTION TRAINING"
echo "================================================================================"
echo "Profile: $PROFILE"
echo "Config:  $CONFIG"
echo "Desc:    $DESC"
echo ""
echo "Dataset: 63M rows (Oct 2023 - Sep 2024)"
echo "Split:   80/10/10 temporal split"
echo ""
echo "Training:   2023-10-01 to 2024-07-19 (80%)"
echo "Validation: 2024-07-20 to 2024-08-24 (10%)"
echo "Test:       2024-08-25 to 2024-09-30 (10%)"
echo "================================================================================"
echo ""

# Confirm before starting
read -p "Start training with $PROFILE profile? (y/n) " -n 1 -r
echo ""
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Training cancelled."
    exit 0
fi

echo "Starting training..."
echo ""

# Create log filename with profile name
LOG_FILE="production_training_${PROFILE}_$(date +%Y%m%d_%H%M%S).log"

# Run training with uv
uv run python xgboost_residual_model_enhanced.py \
    --config "$CONFIG" \
    --train-start "2023-10-01" \
    --train-end "2024-07-19" \
    --val-start "2024-07-20" \
    --val-end "2024-08-24" \
    --test-start "2024-08-25" \
    --test-end "2024-09-30" \
    --n-splits 5 \
    2>&1 | tee "$LOG_FILE"

echo ""
echo "================================================================================"
echo "Training complete!"
echo "Log file: $LOG_FILE"
echo ""

# Extract key metrics from log
echo "Key Results:"
echo "------------"
grep -E "(Baseline Brier|XGBoost Brier|Improvement)" "$LOG_FILE" | tail -3
echo "================================================================================"