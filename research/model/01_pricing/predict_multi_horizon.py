#!/usr/bin/env python3
"""
Multi-Horizon Prediction Router
================================

Routes predictions to specialized bucket models based on time-to-expiry.
Implements optional boundary smoothing to eliminate discontinuities.

Architecture:
- Near bucket (0-5 min): Microstructure-dominated model
- Mid bucket (5-10 min): Balanced regime model
- Far bucket (10-15 min): Diffusion-dominated model

Boundary smoothing:
- 4.5-5.5 min: Weighted blend of near + mid
- 9.5-10.5 min: Weighted blend of mid + far

Usage:
  # Predict with multi-horizon router
  uv run python predict_multi_horizon.py --input test_features_lgb.parquet

  # Enable boundary smoothing
  uv run python predict_multi_horizon.py --input test_features_lgb.parquet --smoothing

  # Compare to single model
  uv run python predict_multi_horizon.py --input test_features_lgb.parquet --compare

Author: BT Research Team
Date: 2025-11-02
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path
from typing import Any

import lightgbm as lgb
import numpy as np
import polars as pl
import yaml

# Import from existing modules
from lightgbm_memory_optimized import FEATURE_COLS, MemoryMonitor

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


def load_config(config_path: Path) -> dict[str, Any]:
    """Load multi-horizon configuration."""
    with open(config_path) as f:
        config = yaml.safe_load(f)
    return config


def load_bucket_models(models_dir: Path, config: dict[str, Any]) -> dict[str, lgb.Booster]:
    """
    Load trained bucket models.

    Args:
        models_dir: Directory containing model files
        config: Multi-horizon configuration

    Returns:
        Dictionary of {bucket_name: model}
    """
    logger.info("\nLoading bucket models...")

    models = {}
    for bucket_name in ["near", "mid", "far"]:
        model_file = models_dir / f"lightgbm_{bucket_name}.txt"

        if not model_file.exists():
            raise FileNotFoundError(f"Model not found: {model_file}. Run train_multi_horizon.py first.")

        models[bucket_name] = lgb.Booster(model_file=str(model_file))
        logger.info(f"  ✓ Loaded {bucket_name} model: {model_file.name}")

    return models


def predict_bucket(model: lgb.Booster, data: pl.DataFrame, features: list[str]) -> np.ndarray:
    """
    Generate predictions for a single bucket.

    Args:
        model: LightGBM model
        data: Input DataFrame
        features: Feature columns

    Returns:
        Prediction array
    """
    x = data.select(features).to_numpy()
    predictions = model.predict(x)
    return np.asarray(predictions)


def apply_boundary_smoothing(
    predictions: np.ndarray,
    time_remaining: np.ndarray,
    bucket_predictions: dict[str, np.ndarray],
    bucket_masks: dict[str, np.ndarray],
    config: dict[str, Any],
) -> np.ndarray:
    """
    Apply weighted averaging at bucket boundaries.

    Smoothing zones:
    - [270s, 330s]: Linear blend from near (100%) to mid (100%)
    - [570s, 630s]: Linear blend from mid (100%) to far (100%)

    Args:
        predictions: Base predictions array (to modify)
        time_remaining: Time-to-expiry in seconds
        bucket_predictions: Dict of {bucket: predictions}
        bucket_masks: Dict of {bucket: boolean masks}
        config: Multi-horizon configuration

    Returns:
        Smoothed predictions array
    """
    smoothing_config = config.get("smoothing", {})
    if not smoothing_config.get("enabled", False):
        return predictions

    logger.info("\nApplying boundary smoothing...")

    # Boundary 1: Near ↔ Mid (around 300s)
    b1_center = smoothing_config["boundary_1"]["center"]
    b1_width = smoothing_config["boundary_1"]["width"]
    b1_lower = b1_center - b1_width // 2
    b1_upper = b1_center + b1_width // 2

    b1_mask = (time_remaining >= b1_lower) & (time_remaining < b1_upper)
    b1_count = b1_mask.sum()

    if b1_count > 0:
        # Weight transitions from 1.0 (near) → 0.0 (mid)
        t = (time_remaining[b1_mask] - b1_lower) / (b1_upper - b1_lower)
        weight_near = 1.0 - t
        weight_mid = t

        predictions[b1_mask] = (
            weight_near * bucket_predictions["near"][b1_mask] + weight_mid * bucket_predictions["mid"][b1_mask]
        )

        logger.info(f"  Smoothed {b1_count:,} predictions at 5-min boundary ({b1_lower}-{b1_upper}s)")

    # Boundary 2: Mid ↔ Far (around 600s)
    b2_center = smoothing_config["boundary_2"]["center"]
    b2_width = smoothing_config["boundary_2"]["width"]
    b2_lower = b2_center - b2_width // 2
    b2_upper = b2_center + b2_width // 2

    b2_mask = (time_remaining >= b2_lower) & (time_remaining < b2_upper)
    b2_count = b2_mask.sum()

    if b2_count > 0:
        # Weight transitions from 1.0 (mid) → 0.0 (far)
        t = (time_remaining[b2_mask] - b2_lower) / (b2_upper - b2_lower)
        weight_mid = 1.0 - t
        weight_far = t

        predictions[b2_mask] = (
            weight_mid * bucket_predictions["mid"][b2_mask] + weight_far * bucket_predictions["far"][b2_mask]
        )

        logger.info(f"  Smoothed {b2_count:,} predictions at 10-min boundary ({b2_lower}-{b2_upper}s)")

    return predictions


def predict_multi_horizon(
    data: pl.DataFrame,
    models: dict[str, lgb.Booster],
    features: list[str],
    config: dict[str, Any],
    enable_smoothing: bool = False,
) -> pl.DataFrame:
    """
    Generate multi-horizon predictions.

    Args:
        data: Input DataFrame with time_remaining column
        models: Dictionary of bucket models
        features: Feature columns
        config: Multi-horizon configuration
        enable_smoothing: Whether to apply boundary smoothing

    Returns:
        DataFrame with predictions added
    """
    logger.info("\nGenerating multi-horizon predictions...")

    start_time = time.time()
    n_samples = len(data)

    # Extract time_remaining as numpy array
    time_remaining = data.select("time_remaining").to_numpy().ravel()

    # Create bucket masks
    near_config = config["buckets"]["near"]
    mid_config = config["buckets"]["mid"]
    far_config = config["buckets"]["far"]

    masks = {
        "near": (time_remaining >= near_config["time_min"]) & (time_remaining < near_config["time_max"]),
        "mid": (time_remaining >= mid_config["time_min"]) & (time_remaining < mid_config["time_max"]),
        "far": (time_remaining >= far_config["time_min"]) & (time_remaining < far_config["time_max"]),
    }

    # Log distribution
    logger.info(f"  Total samples: {n_samples:,}")
    for bucket_name, mask in masks.items():
        count = mask.sum()
        pct = (count / n_samples) * 100
        logger.info(f"    {bucket_name:5s}: {count:8,} ({pct:5.1f}%)")

    # Generate predictions for each bucket
    predictions = np.zeros(n_samples, dtype=np.float64)
    bucket_predictions_all = {}

    for bucket_name, model in models.items():
        mask = masks[bucket_name]
        count = mask.sum()

        if count > 0:
            bucket_data = data.filter(pl.Series(mask))
            bucket_preds = predict_bucket(model, bucket_data, features)

            # Store all bucket predictions (needed for smoothing)
            full_preds = np.zeros(n_samples, dtype=np.float64)
            full_preds[mask] = bucket_preds
            bucket_predictions_all[bucket_name] = full_preds

            # Assign to main predictions array (overwritten if smoothing)
            predictions[mask] = bucket_preds

            logger.info(f"  ✓ {bucket_name:5s}: {count:8,} predictions (mean={bucket_preds.mean():.4f})")

    # Apply boundary smoothing if enabled
    if enable_smoothing:
        predictions = apply_boundary_smoothing(predictions, time_remaining, bucket_predictions_all, masks, config)

    # Add predictions to DataFrame
    result = data.with_columns([pl.Series("multi_horizon_prediction", predictions).alias("residual_pred")])

    prediction_time = time.time() - start_time
    logger.info(
        f"\n✓ Generated {n_samples:,} predictions in {prediction_time:.2f}s ({n_samples / prediction_time:,.0f} pred/s)"
    )

    return result


def evaluate_predictions(data: pl.DataFrame, baseline_col: str = "prob_mid") -> dict[str, float]:
    """
    Evaluate multi-horizon predictions.

    Args:
        data: DataFrame with residual_pred, outcome, and baseline columns
        baseline_col: Name of baseline probability column

    Returns:
        Dictionary of performance metrics
    """
    logger.info("\nEvaluating predictions...")

    # Calculate model probability
    data = data.with_columns([(pl.col(baseline_col) + pl.col("residual_pred")).alias("prob_pred")])

    # Clip to [0, 1]
    data = data.with_columns([pl.col("prob_pred").clip(0.0, 1.0).alias("prob_pred")])

    # Calculate Brier scores
    outcome = data.select("outcome").to_numpy().ravel()
    prob_baseline = data.select(baseline_col).to_numpy().ravel()
    prob_pred = data.select("prob_pred").to_numpy().ravel()

    brier_baseline = np.mean((outcome - prob_baseline) ** 2)
    brier_model = np.mean((outcome - prob_pred) ** 2)

    # Residual metrics
    residual_true = outcome - prob_baseline
    residual_pred = data.select("residual_pred").to_numpy().ravel()
    residual_mse = np.mean((residual_true - residual_pred) ** 2)
    residual_mae = np.mean(np.abs(residual_true - residual_pred))

    # Improvement
    brier_improvement_pct = ((brier_baseline - brier_model) / brier_baseline) * 100

    metrics = {
        "brier_baseline": brier_baseline,
        "brier_model": brier_model,
        "brier_improvement_pct": brier_improvement_pct,
        "residual_mse": residual_mse,
        "residual_mae": residual_mae,
        "n_samples": len(data),
    }

    logger.info(f"  Baseline Brier: {brier_baseline:.6f}")
    logger.info(f"  Model Brier:    {brier_model:.6f}")
    logger.info(f"  Improvement:    {brier_improvement_pct:.2f}%")
    logger.info(f"  Residual MSE:   {residual_mse:.6f}")
    logger.info(f"  Residual MAE:   {residual_mae:.6f}")

    return metrics


def main() -> None:
    """Main execution function."""
    # Parse arguments
    parser = argparse.ArgumentParser(description="Multi-horizon prediction router")
    parser.add_argument(
        "--input",
        type=Path,
        default=Path(__file__).parent.parent / "results" / "test_features_lgb.parquet",
        help="Input data file",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).parent / "config" / "multi_horizon_config.yaml",
        help="Path to multi-horizon config file",
    )
    parser.add_argument(
        "--smoothing",
        action="store_true",
        help="Enable boundary smoothing",
    )
    parser.add_argument(
        "--compare",
        action="store_true",
        help="Compare to single model baseline",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Output file for predictions (optional)",
    )
    args = parser.parse_args()

    logger.info("=" * 80)
    logger.info("MULTI-HORIZON PREDICTION ROUTER")
    logger.info("=" * 80)

    monitor = MemoryMonitor()

    # Load configuration
    config = load_config(args.config)

    # Setup paths
    model_dir = Path(__file__).parent.parent
    models_dir = model_dir / Path(config["models"]["output_dir"])

    # Load models
    models = load_bucket_models(models_dir, config)

    # Load input data
    logger.info(f"\nLoading input data: {args.input}")
    if not args.input.exists():
        raise FileNotFoundError(f"Input file not found: {args.input}")

    data = pl.read_parquet(args.input)
    logger.info(f"  Loaded {len(data):,} samples")

    # Verify required columns
    required_cols = ["time_remaining"]
    if args.compare or args.output:
        required_cols.extend(["outcome", "prob_mid"])

    missing_cols = [col for col in required_cols if col not in data.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    # Get features
    schema = data.collect_schema()
    features = [col for col in FEATURE_COLS if col in schema.names()]
    logger.info(f"  Using {len(features)} features")

    monitor.check_memory("After loading data")

    # Generate predictions
    result = predict_multi_horizon(data, models, features, config, enable_smoothing=args.smoothing)

    # Evaluate if outcome available
    if "outcome" in result.columns and "prob_mid" in result.columns:
        metrics = evaluate_predictions(result)

        # Per-bucket evaluation
        logger.info("\nPer-bucket performance:")
        for bucket_name in ["near", "mid", "far"]:
            bucket_config = config["buckets"][bucket_name]
            bucket_data = result.filter(
                (pl.col("time_remaining") >= bucket_config["time_min"])
                & (pl.col("time_remaining") < bucket_config["time_max"])
            )

            if len(bucket_data) > 0:
                bucket_metrics = evaluate_predictions(bucket_data)
                logger.info(
                    f"  {bucket_config['name']:25s}: {bucket_metrics['brier_improvement_pct']:6.2f}% "
                    f"(n={len(bucket_data):,})"
                )

    # Save output if requested
    if args.output:
        logger.info(f"\nSaving predictions to {args.output}")
        result.write_parquet(args.output, compression="snappy")
        logger.info(f"  ✓ Saved {len(result):,} predictions")

    # Compare to single model if requested
    if args.compare:
        logger.info("\n" + "=" * 80)
        logger.info("COMPARISON TO SINGLE MODEL")
        logger.info("=" * 80)

        single_model_file = model_dir / "results" / "lightgbm_production_best.txt"

        if single_model_file.exists():
            logger.info(f"Loading single model: {single_model_file}")
            single_model = lgb.Booster(model_file=str(single_model_file))

            # Generate single model predictions
            x = data.select(features).to_numpy()
            single_preds = single_model.predict(x)

            data_with_single = data.with_columns([pl.Series("residual_pred", single_preds)])

            logger.info("\nSingle Model Performance:")
            single_metrics = evaluate_predictions(data_with_single)

            # Calculate improvement
            multi_improvement = metrics["brier_improvement_pct"]
            single_improvement = single_metrics["brier_improvement_pct"]
            delta = multi_improvement - single_improvement

            logger.info("\n" + "-" * 80)
            logger.info(f"Multi-Horizon:  {multi_improvement:6.2f}% improvement")
            logger.info(f"Single Model:   {single_improvement:6.2f}% improvement")
            logger.info(f"Gain:           +{delta:5.2f}pp {'✓' if delta > 0 else '⚠'}")
        else:
            logger.warning(f"Single model not found: {single_model_file}")

    monitor.check_memory("Final")

    logger.info("\n✓ Multi-horizon prediction complete!")


if __name__ == "__main__":
    main()
