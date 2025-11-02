#!/bin/bash

# Training monitor script - tracks XGBoost training progress in real-time

echo "================================================================================"
echo "XGBOOST TRAINING MONITOR"
echo "================================================================================"
echo ""
echo "This script monitors active XGBoost training sessions."
echo "Press Ctrl+C to exit monitoring."
echo ""

# Function to display memory usage
show_memory() {
    echo "MEMORY STATUS:"
    echo "--------------"
    free -h | grep -E "Mem:|Swap:"
    echo ""
}

# Function to find latest log file
find_latest_log() {
    latest_log=$(ls -t production_training_*.log 2>/dev/null | head -1)
    if [ -z "$latest_log" ]; then
        latest_log=$(ls -t results/*/training_log.txt 2>/dev/null | head -1)
    fi
    echo "$latest_log"
}

# Function to extract progress from log
show_progress() {
    local log_file="$1"
    if [ -f "$log_file" ]; then
        echo "TRAINING PROGRESS (from $log_file):"
        echo "----------------------------------------"

        # Show current phase
        if grep -q "CROSS-VALIDATION" "$log_file"; then
            echo "Phase: Cross-Validation"
            # Show CV fold progress
            fold_count=$(grep -c "Fold.*/" "$log_file")
            echo "CV Folds completed: $fold_count/5"
        elif grep -q "TRAINING FINAL MODEL" "$log_file"; then
            echo "Phase: Final Model Training"
        elif grep -q "STREAMING PREDICTIONS" "$log_file"; then
            echo "Phase: Test Predictions"
        fi
        echo ""

        # Show latest training round if available
        last_round=$(grep -E "\[([0-9]+)\].*train-rmse:" "$log_file" | tail -1)
        if [ ! -z "$last_round" ]; then
            echo "Latest training round:"
            echo "$last_round"
        fi
        echo ""

        # Show memory usage from log
        last_memory=$(grep -E "\[MEMORY\]" "$log_file" | tail -1)
        if [ ! -z "$last_memory" ]; then
            echo "$last_memory"
        fi
        echo ""

        # Show any recent errors
        errors=$(grep -i "error" "$log_file" | tail -3)
        if [ ! -z "$errors" ]; then
            echo "Recent errors/warnings:"
            echo "$errors"
        fi
    else
        echo "No active training log found."
    fi
}

# Function to check if training is running
check_training_process() {
    if pgrep -f "xgboost_residual_model_enhanced.py" > /dev/null; then
        echo "✓ XGBoost training process is ACTIVE"
        pid=$(pgrep -f "xgboost_residual_model_enhanced.py")
        echo "  Process ID: $pid"
        # Get process start time and calculate duration
        if [ ! -z "$pid" ]; then
            start_time=$(ps -o lstart= -p $pid 2>/dev/null)
            if [ ! -z "$start_time" ]; then
                echo "  Started: $start_time"
            fi
        fi
    else
        echo "✗ No XGBoost training process detected"
    fi
    echo ""
}

# Main monitoring loop
while true; do
    clear
    echo "================================================================================"
    echo "XGBOOST TRAINING MONITOR - $(date)"
    echo "================================================================================"
    echo ""

    # Check if training is running
    check_training_process

    # Show memory status
    show_memory

    # Find and monitor latest log
    LOG_FILE=$(find_latest_log)
    if [ ! -z "$LOG_FILE" ]; then
        show_progress "$LOG_FILE"

        # Check if training completed
        if grep -q "Training complete" "$LOG_FILE" 2>/dev/null; then
            echo "================================================================================"
            echo "TRAINING COMPLETED!"
            echo "================================================================================"

            # Show final results
            echo ""
            echo "FINAL RESULTS:"
            grep -E "Baseline Brier|XGBoost Brier|Improvement" "$LOG_FILE" | tail -3
            echo ""

            break
        fi
    else
        echo "Waiting for training to start..."
        echo "Run ./run_full_2year_production.sh to begin training"
    fi

    echo ""
    echo "================================================================================"
    echo "Refreshing in 10 seconds... (Press Ctrl+C to exit)"

    sleep 10
done

echo "Monitoring ended."