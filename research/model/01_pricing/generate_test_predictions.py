#!/usr/bin/env python3
"""
Quick Inference Script: Generate Test Predictions
=================================================

Loads existing LightGBM model and generates predictions on test set
for visualization pipeline testing.

Usage:
    uv run python generate_test_predictions.py

Outputs:
    ../results/test_predictions.parquet - Predictions with all required columns
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np
import polars as pl

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

# File paths (portable across any CWD)
RESULTS_DIR = Path(__file__).parent.parent / "results"
MODEL_FILE = RESULTS_DIR / "lightgbm_model_optimized.txt"
TEST_FILE = RESULTS_DIR / "test_features_lgb.parquet"
OUTPUT_FILE = RESULTS_DIR / "test_predictions.parquet"

# Feature columns will be loaded from the model (dynamic)


def generate_predictions() -> None:
    """Load model, generate predictions, save to parquet."""
    logger.info("=" * 80)
    logger.info("GENERATING TEST PREDICTIONS")
    logger.info("=" * 80)

    # Check files exist
    if not MODEL_FILE.exists():
        raise FileNotFoundError(f"Model file not found: {MODEL_FILE}")
    if not TEST_FILE.exists():
        raise FileNotFoundError(f"Test file not found: {TEST_FILE}")

    logger.info(f"Model file: {MODEL_FILE}")
    logger.info(f"Test file:  {TEST_FILE}")

    # Load model
    logger.info("\n1. Loading LightGBM model...")
    model = lgb.Booster(model_file=str(MODEL_FILE))
    logger.info(f"   ✓ Model loaded (best iteration: {model.best_iteration})")

    # Get features from model
    features = model.feature_name()
    logger.info(f"   ✓ Model expects {len(features)} features")

    # Load test data
    logger.info("\n2. Loading test data...")
    test_df = pl.read_parquet(TEST_FILE)
    logger.info(f"   ✓ Loaded {len(test_df):,} rows, {len(test_df.columns)} columns")

    # Verify all required features exist
    missing_cols = [col for col in features if col not in test_df.columns]
    if missing_cols:
        raise ValueError(f"Test file missing {len(missing_cols)} required features: {missing_cols[:10]}")

    logger.info(f"   ✓ All {len(features)} features available")

    # Check for required metadata columns
    required_metadata = ["prob_mid", "outcome", "rv_300s"]
    missing_metadata = [col for col in required_metadata if col not in test_df.columns]
    if missing_metadata:
        raise ValueError(f"Missing required metadata columns: {missing_metadata}")

    # Extract features
    logger.info("\n3. Preparing features...")
    x_test = test_df.select(features).to_numpy().astype(np.float32)

    # Handle inf/nan
    x_test = np.nan_to_num(x_test, nan=0.0, posinf=1.0, neginf=-1.0)
    logger.info(f"   ✓ Feature matrix: {x_test.shape}")

    # Generate predictions
    logger.info("\n4. Generating predictions...")
    residual_pred = model.predict(x_test, num_iteration=model.best_iteration)
    logger.info(f"   ✓ Generated {len(residual_pred):,} predictions")
    logger.info(f"   Residual prediction range: [{residual_pred.min():.4f}, {residual_pred.max():.4f}]")
    logger.info(f"   Residual prediction mean: {residual_pred.mean():.6f}")
    logger.info(f"   Residual prediction std: {residual_pred.std():.6f}")

    # Calculate final probabilities
    logger.info("\n5. Calculating final probabilities...")
    final_prob = test_df["prob_mid"].to_numpy() + residual_pred

    # Clip to [0, 1]
    final_prob = np.clip(final_prob, 0.0, 1.0)
    logger.info(f"   ✓ Final probability range: [{final_prob.min():.4f}, {final_prob.max():.4f}]")

    # Create predictions dataframe with all required columns
    logger.info("\n6. Creating predictions dataframe...")

    # Determine date column
    date_col = None
    if "timestamp" in test_df.columns:
        date_col = "timestamp"
    elif "date" in test_df.columns:
        date_col = "date"
    else:
        logger.warning("   ⚠️  No timestamp/date column found - will use index")

    # Build column selection list
    select_cols = ["prob_mid", "outcome", "rv_300s"]
    if date_col:
        select_cols.append(date_col)
    if "residual" in test_df.columns:
        select_cols.append("residual")
    if "moneyness" in test_df.columns:
        select_cols.append("moneyness")
    if "time_remaining" in test_df.columns:
        select_cols.append("time_remaining")

    # Add all feature columns for SHAP (avoiding duplicates)
    for feature in features:
        if feature not in select_cols:
            select_cols.append(feature)

    # Select all columns at once
    predictions_df = test_df.select(select_cols)

    # Rename timestamp to date if needed
    if date_col == "timestamp":
        predictions_df = predictions_df.rename({"timestamp": "date"})
    elif date_col is None:
        # Add index as date
        predictions_df = predictions_df.with_columns([
            pl.Series("date", range(len(test_df)))
        ])

    # Add predictions
    predictions_df = predictions_df.with_columns([
        pl.Series("residual_pred", residual_pred).cast(pl.Float32),
        pl.Series("final_prob", final_prob).cast(pl.Float32),
        (pl.Series("residual_pred", residual_pred) ** 2).cast(pl.Float32).alias("residual_squared"),
    ])

    logger.info(f"   ✓ Created predictions dataframe: {len(predictions_df):,} rows, {len(predictions_df.columns)} columns")
    logger.info(f"   Columns: {predictions_df.columns[:10]}... ({len(predictions_df.columns)} total)")

    # Calculate Brier scores
    logger.info("\n7. Calculating Brier scores...")
    baseline_brier = ((predictions_df["prob_mid"] - predictions_df["outcome"]) ** 2).mean()
    model_brier = ((predictions_df["final_prob"] - predictions_df["outcome"]) ** 2).mean()
    improvement_pct = ((baseline_brier - model_brier) / baseline_brier * 100) if baseline_brier > 0 else 0

    logger.info(f"   Baseline Brier: {baseline_brier:.6f}")
    logger.info(f"   Model Brier:    {model_brier:.6f}")
    logger.info(f"   Improvement:    {improvement_pct:+.2f}%")

    # Save predictions
    logger.info(f"\n8. Saving predictions to {OUTPUT_FILE}...")
    predictions_df.write_parquet(
        OUTPUT_FILE,
        compression="snappy",
        statistics=True,
    )

    file_size_mb = OUTPUT_FILE.stat().st_size / (1024 * 1024)
    logger.info(f"   ✓ Saved {file_size_mb:.1f} MB")

    # Summary
    logger.info("\n" + "=" * 80)
    logger.info("PREDICTION GENERATION COMPLETE")
    logger.info("=" * 80)
    logger.info(f"Output file: {OUTPUT_FILE}")
    logger.info(f"Rows:        {len(predictions_df):,}")
    logger.info(f"Columns:     {len(predictions_df.columns)}")
    logger.info(f"File size:   {file_size_mb:.1f} MB")
    logger.info(f"Brier improvement: {improvement_pct:+.2f}%")
    logger.info("=" * 80)
    logger.info("\n✓ Ready for visualization pipeline!")


if __name__ == "__main__":
    generate_predictions()
