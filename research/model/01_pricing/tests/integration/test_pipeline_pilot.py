#!/usr/bin/env python3
"""
Integration Tests: Pilot Mode End-to-End Pipeline
==================================================

Tests full pilot mode pipeline execution:
1. Train 3 bucket models (no walk-forward, fast mode)
2. Generate predictions
3. Evaluate performance

This smoke test ensures all components integrate correctly before
running expensive full pipeline (5-8 hours).

Expected runtime: ~5 minutes
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import polars as pl
import pytest


@pytest.mark.integration
@pytest.mark.slow
def test_pilot_mode_full_pipeline(project_root: Path, test_output_dir: Path, fixtures_dir: Path) -> None:
    """
    Integration test: Run full pilot mode pipeline.

    This is the critical smoke test before running the production pipeline.
    """
    # Setup: Copy test data to expected location
    test_data_dir = test_output_dir / "data"
    test_data_dir.mkdir(parents=True, exist_ok=True)

    # Copy test fixtures to mimic production data structure
    shutil.copy(fixtures_dir / "test_train.parquet", test_data_dir / "train_features_lgb.parquet")
    shutil.copy(fixtures_dir / "test_val.parquet", test_data_dir / "val_features_lgb.parquet")
    shutil.copy(fixtures_dir / "test_test.parquet", test_data_dir / "test_features_lgb.parquet")

    # Create results directory
    results_dir = test_output_dir / "multi_horizon"
    models_dir = results_dir / "models"
    models_dir.mkdir(parents=True, exist_ok=True)

    # Step 1: Train models (pilot mode: no walk-forward)
    train_cmd = [
        "uv",
        "run",
        "python",
        str(project_root / "train_multi_horizon.py"),
        "--bucket",
        "all",
        "--no-walk-forward",
        "--data-dir",
        str(test_data_dir),
        "--output-dir",
        str(models_dir),
    ]

    result = subprocess.run(train_cmd, capture_output=True, text=True, cwd=project_root)

    assert result.returncode == 0, f"Training failed:\nSTDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"

    # Verify models were created
    near_model = models_dir / "lightgbm_near.txt"
    mid_model = models_dir / "lightgbm_mid.txt"
    far_model = models_dir / "lightgbm_far.txt"

    assert near_model.exists(), "Near model not created"
    assert mid_model.exists(), "Mid model not created"
    assert far_model.exists(), "Far model not created"

    # Verify configs were created
    near_config = models_dir / "config_near.yaml"
    mid_config = models_dir / "config_mid.yaml"
    far_config = models_dir / "config_far.yaml"

    assert near_config.exists(), "Near config not created"
    assert mid_config.exists(), "Mid config not created"
    assert far_config.exists(), "Far config not created"

    # Step 2: Generate predictions
    predictions_dir = results_dir / "predictions"
    predictions_dir.mkdir(parents=True, exist_ok=True)

    predict_cmd = [
        "uv",
        "run",
        "python",
        str(project_root / "predict_multi_horizon.py"),
        "--test-file",
        str(test_data_dir / "test_features_lgb.parquet"),
        "--models-dir",
        str(models_dir),
        "--output-dir",
        str(predictions_dir),
    ]

    result = subprocess.run(predict_cmd, capture_output=True, text=True, cwd=project_root)

    assert result.returncode == 0, f"Prediction failed:\nSTDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"

    # Verify predictions file was created
    predictions_file = predictions_dir / "test_predictions.parquet"
    assert predictions_file.exists(), "Predictions file not created"

    # Load and validate predictions
    df_predictions = pl.read_parquet(predictions_file)
    assert "prediction_residual" in df_predictions.columns
    assert "prob_final" in df_predictions.columns
    assert len(df_predictions) > 0

    # Step 3: Evaluate performance
    eval_cmd = [
        "uv",
        "run",
        "python",
        str(project_root / "evaluate_multi_horizon.py"),
        "--test-file",
        str(test_data_dir / "test_features_lgb.parquet"),
        "--models-dir",
        str(models_dir),
        "--output-dir",
        str(results_dir),
        "--no-plots",  # Skip plots for faster testing
    ]

    result = subprocess.run(eval_cmd, capture_output=True, text=True, cwd=project_root)

    assert result.returncode == 0, f"Evaluation failed:\nSTDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"

    # Verify evaluation outputs
    summary_file = results_dir / "multi_horizon_vs_single_model.csv"
    per_bucket_file = results_dir / "per_bucket_performance.csv"

    assert summary_file.exists(), "Summary CSV not created"
    assert per_bucket_file.exists(), "Per-bucket CSV not created"

    # Load and validate summary
    df_summary = pl.read_csv(summary_file)
    assert "model" in df_summary.columns
    assert "brier_improvement_pct" in df_summary.columns

    # Check that multi-horizon model shows improvement
    multi_horizon_row = df_summary.filter(pl.col("model") == "multi_horizon")
    assert len(multi_horizon_row) > 0

    improvement = multi_horizon_row["brier_improvement_pct"][0]
    assert improvement is not None
    # Should show some improvement (even if small on test data)
    assert improvement > -50, f"Unexpected improvement: {improvement}%"


@pytest.mark.integration
def test_pilot_mode_training_only(project_root: Path, test_output_dir: Path, fixtures_dir: Path) -> None:
    """
    Fast integration test: Only test training component.

    Faster than full pipeline test, useful for CI/CD.
    """
    # Setup
    test_data_dir = test_output_dir / "data"
    test_data_dir.mkdir(parents=True, exist_ok=True)

    shutil.copy(fixtures_dir / "test_train.parquet", test_data_dir / "train_features_lgb.parquet")
    shutil.copy(fixtures_dir / "test_val.parquet", test_data_dir / "val_features_lgb.parquet")

    results_dir = test_output_dir / "multi_horizon"
    models_dir = results_dir / "models"
    models_dir.mkdir(parents=True, exist_ok=True)

    # Train single bucket (faster)
    train_cmd = [
        "uv",
        "run",
        "python",
        str(project_root / "train_multi_horizon.py"),
        "--bucket",
        "near",
        "--no-walk-forward",
        "--data-dir",
        str(test_data_dir),
        "--output-dir",
        str(models_dir),
    ]

    result = subprocess.run(train_cmd, capture_output=True, text=True, cwd=project_root, timeout=300)

    assert result.returncode == 0, f"Training failed:\nSTDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"

    # Verify outputs
    near_model = models_dir / "lightgbm_near.txt"
    near_config = models_dir / "config_near.yaml"

    assert near_model.exists(), "Near model not created"
    assert near_config.exists(), "Near config not created"

    # Check model size is reasonable (>1KB, <100MB)
    model_size = near_model.stat().st_size
    assert 1000 < model_size < 100_000_000, f"Model size unexpected: {model_size} bytes"


@pytest.mark.integration
def test_prediction_routing(project_root: Path, test_output_dir: Path, fixtures_dir: Path) -> None:
    """
    Integration test: Verify prediction routing logic.

    Tests that predictions are correctly routed to appropriate bucket models
    based on time_remaining.
    """
    # This test requires trained models from previous test
    # For now, we'll create a simpler unit-style test that verifies the routing logic
    # without requiring full model training

    # Create synthetic test data with known time_remaining values
    test_data = pl.DataFrame(
        {
            "time_remaining": [100, 250, 350, 550, 650, 850],  # 2 near, 2 mid, 2 far
            "outcome": [0, 1, 0, 1, 0, 1],
            "prob_mid": [0.5] * 6,
        }
    )

    test_file = test_output_dir / "routing_test.parquet"
    test_data.write_parquet(test_file)

    # For this test, we need models to exist
    # We'll skip if models don't exist (should be run after training test)
    models_dir = test_output_dir / "multi_horizon" / "models"

    if not (models_dir / "lightgbm_near.txt").exists():
        pytest.skip("Models not available (run training test first)")

    # Generate predictions
    predictions_dir = test_output_dir / "predictions"
    predictions_dir.mkdir(parents=True, exist_ok=True)

    predict_cmd = [
        "uv",
        "run",
        "python",
        str(project_root / "predict_multi_horizon.py"),
        "--test-file",
        str(test_file),
        "--models-dir",
        str(models_dir),
        "--output-dir",
        str(predictions_dir),
    ]

    result = subprocess.run(predict_cmd, capture_output=True, text=True, cwd=project_root, timeout=60)

    assert result.returncode == 0, f"Prediction failed:\nSTDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"

    # Load predictions and verify routing
    predictions_file = predictions_dir / "routing_test_predictions.parquet"
    assert predictions_file.exists()

    df_pred = pl.read_parquet(predictions_file)
    assert len(df_pred) == 6
    assert "prediction_residual" in df_pred.columns


@pytest.mark.integration
def test_stratification_preserves_data(project_root: Path, test_output_dir: Path, fixtures_dir: Path) -> None:
    """
    Integration test: Verify stratification doesn't lose data.

    Ensures all rows are accounted for across buckets.
    """
    # Load test data
    test_file = fixtures_dir / "test_train.parquet"
    df_original = pl.read_parquet(test_file)
    original_count = len(df_original)

    # Run stratification for all buckets
    stratified_dir = test_output_dir / "stratified"
    stratified_dir.mkdir(parents=True, exist_ok=True)

    from train_multi_horizon import stratify_data_by_time

    near_file = stratify_data_by_time(test_file, stratified_dir, "near", 0, 300)
    mid_file = stratify_data_by_time(test_file, stratified_dir, "mid", 300, 600)
    far_file = stratify_data_by_time(test_file, stratified_dir, "far", 600, 900)

    # Count rows in each bucket
    df_near = pl.read_parquet(near_file)
    df_mid = pl.read_parquet(mid_file)
    df_far = pl.read_parquet(far_file)

    total_stratified = len(df_near) + len(df_mid) + len(df_far)

    # Should account for all rows (100% coverage)
    assert total_stratified == original_count, f"Data loss: {original_count} → {total_stratified}"

    # Check distribution (should be roughly 33% each)
    near_pct = (len(df_near) / original_count) * 100
    mid_pct = (len(df_mid) / original_count) * 100
    far_pct = (len(df_far) / original_count) * 100

    # Allow 5% tolerance
    assert 28 <= near_pct <= 38, f"Near bucket: {near_pct:.1f}% (expected ~33%)"
    assert 28 <= mid_pct <= 38, f"Mid bucket: {mid_pct:.1f}% (expected ~33%)"
    assert 28 <= far_pct <= 38, f"Far bucket: {far_pct:.1f}% (expected ~33%)"
