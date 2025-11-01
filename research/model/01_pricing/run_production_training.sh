#!/bin/bash
# Production training script with 80/10/10 temporal split
# Using external memory for 63M rows on 30GB RAM machine

echo "================================================================================"
echo "XGBOOST RESIDUAL MODEL - PRODUCTION TRAINING"
echo "================================================================================"
echo "Configuration: Production (63M rows)"
echo "Split: 80/10/10 temporal split"
echo "Training:   2023-10-01 to 2024-07-19 (80%)"
echo "Validation: 2024-07-20 to 2024-08-24 (10%)"
echo "Test:       2024-08-25 to 2024-09-30 (10%)"
echo "================================================================================"

# Use production config with optimized hyperparameters
CONFIG="config/xgboost_config_production.yaml"

# Check if config exists
if [ ! -f "$CONFIG" ]; then
    echo "ERROR: Config file not found: $CONFIG"
    echo "Please ensure the production config exists."
    exit 1
fi

echo ""
echo "Using config: $CONFIG"
echo ""

# Run with uv (required for environment)
uv run python xgboost_residual_model_enhanced.py \
    --config "$CONFIG" \
    --train-start "2023-10-01" \
    --train-end "2024-07-19" \
    --val-start "2024-07-20" \
    --val-end "2024-08-24" \
    --test-start "2024-08-25" \
    --test-end "2024-09-30" \
    --n-splits 5 \
    2>&1 | tee "production_training_$(date +%Y%m%d_%H%M%S).log"

echo ""
echo "Training complete. Check the log file for results."