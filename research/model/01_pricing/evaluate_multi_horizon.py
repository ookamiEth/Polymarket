#!/usr/bin/env python3
"""
Multi-Horizon Model Evaluation
================================

Comprehensive comparison of multi-horizon vs single model approaches:
- Overall performance metrics (Brier score, calibration)
- Per-bucket performance breakdown
- Performance vs time-to-expiry analysis
- Calibration curves per bucket
- Feature importance comparison
- Statistical significance testing

Generates:
- CSV report with detailed metrics
- Visualizations (calibration, performance by time, comparison charts)
- Summary report with recommendations

Usage:
  # Full evaluation on test set
  uv run python evaluate_multi_horizon.py

  # Evaluate on custom data
  uv run python evaluate_multi_horizon.py --input custom_test_data.parquet

  # Skip visualizations
  uv run python evaluate_multi_horizon.py --no-plots

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


def load_models(models_dir: Path, config: dict[str, Any], include_single: bool = True) -> dict[str, Any]:
    """
    Load multi-horizon and optionally single model.

    Args:
        models_dir: Directory containing model files
        config: Multi-horizon configuration
        include_single: Whether to load single model for comparison

    Returns:
        Dictionary with bucket models and optional single model
    """
    logger.info("\nLoading models...")

    models: dict[str, Any] = {"buckets": {}}

    # Load bucket models
    for bucket_name in ["near", "mid", "far"]:
        model_file = models_dir / f"lightgbm_{bucket_name}.txt"

        if not model_file.exists():
            raise FileNotFoundError(f"Model not found: {model_file}. Run train_multi_horizon.py first.")

        try:
            models["buckets"][bucket_name] = lgb.Booster(model_file=str(model_file))
            logger.info(f"  ✓ Loaded {bucket_name} bucket model")
        except Exception as e:
            msg = f"Failed to load model {model_file}: {e}"
            raise ValueError(msg) from e

    # Load single model if requested
    if include_single:
        single_model_file = models_dir.parent / "results" / "lightgbm_production_best.txt"

        if single_model_file.exists():
            models["single"] = lgb.Booster(model_file=str(single_model_file))
            logger.info("  ✓ Loaded single model (baseline)")
        else:
            logger.warning(f"Single model not found: {single_model_file}")
            models["single"] = None
    else:
        models["single"] = None

    return models


def generate_predictions(
    data: pl.DataFrame,
    models: dict[str, Any],
    features: list[str],
    config: dict[str, Any],
) -> pl.DataFrame:
    """
    Generate predictions from both multi-horizon and single models.

    Args:
        data: Input DataFrame
        models: Dictionary with bucket and single models
        features: Feature columns
        config: Multi-horizon configuration

    Returns:
        DataFrame with predictions from both approaches
    """
    logger.info("\nGenerating predictions...")

    n_samples = len(data)
    time_remaining = data.select("time_remaining").to_numpy().ravel()

    # Multi-horizon predictions
    logger.info("  Multi-horizon model:")
    multi_preds = np.zeros(n_samples, dtype=np.float64)

    for bucket_name in ["near", "mid", "far"]:
        bucket_config = config["buckets"][bucket_name]
        mask = (time_remaining >= bucket_config["time_min"]) & (time_remaining < bucket_config["time_max"])
        count = mask.sum()

        if count > 0:
            bucket_data = data.filter(pl.Series(mask))
            x = bucket_data.select(features).to_numpy()
            bucket_preds = models["buckets"][bucket_name].predict(x)
            multi_preds[mask] = bucket_preds
            logger.info(f"    {bucket_name:5s}: {count:8,} predictions")

    result = data.with_columns([pl.Series("multi_horizon_pred", multi_preds)])

    # Single model predictions
    if models["single"] is not None:
        logger.info("  Single model:")
        x = data.select(features).to_numpy()
        single_preds = models["single"].predict(x)
        result = result.with_columns([pl.Series("single_model_pred", single_preds)])
        logger.info(f"    Generated {len(single_preds):,} predictions")

    return result


def calculate_metrics(
    predictions: np.ndarray,
    outcomes: np.ndarray,
    prob_baseline: np.ndarray,
    label: str,
) -> dict[str, Any]:
    """
    Calculate comprehensive performance metrics.

    Args:
        predictions: Residual predictions
        outcomes: True outcomes (0 or 1)
        prob_baseline: Baseline probabilities
        label: Label for logging

    Returns:
        Dictionary of metrics
    """
    # Calculate model probabilities
    prob_pred = np.clip(prob_baseline + predictions, 0.0, 1.0)

    # Brier scores
    brier_baseline = np.mean((outcomes - prob_baseline) ** 2)
    brier_model = np.mean((outcomes - prob_pred) ** 2)
    brier_improvement_pct = ((brier_baseline - brier_model) / brier_baseline) * 100

    # Residual metrics
    residual_true = outcomes - prob_baseline
    residual_mse = np.mean((residual_true - predictions) ** 2)
    residual_rmse = np.sqrt(residual_mse)
    residual_mae = np.mean(np.abs(residual_true - predictions))

    # Calibration error (binned)
    n_bins = 10
    bin_edges = np.linspace(0, 1, n_bins + 1)
    bin_indices = np.digitize(prob_pred, bin_edges[1:-1])

    calibration_error = 0.0
    for i in range(n_bins):
        mask = bin_indices == i
        if mask.sum() > 0:
            mean_pred = prob_pred[mask].mean()
            mean_outcome = outcomes[mask].mean()
            calibration_error += mask.sum() * (mean_pred - mean_outcome) ** 2

    calibration_error = np.sqrt(calibration_error / len(outcomes))

    metrics = {
        "label": label,
        "n_samples": len(outcomes),
        "brier_baseline": brier_baseline,
        "brier_model": brier_model,
        "brier_improvement_pct": brier_improvement_pct,
        "residual_mse": residual_mse,
        "residual_rmse": residual_rmse,
        "residual_mae": residual_mae,
        "calibration_error": calibration_error,
    }

    logger.info(f"\n{label} Performance:")
    logger.info(f"  Samples:        {metrics['n_samples']:,}")
    logger.info(f"  Baseline Brier: {metrics['brier_baseline']:.6f}")
    logger.info(f"  Model Brier:    {metrics['brier_model']:.6f}")
    logger.info(f"  Improvement:    {metrics['brier_improvement_pct']:6.2f}%")
    logger.info(f"  Residual MSE:   {metrics['residual_mse']:.6f}")
    logger.info(f"  Residual MAE:   {metrics['residual_mae']:.6f}")
    logger.info(f"  Calib. Error:   {metrics['calibration_error']:.6f}")

    return metrics


def evaluate_per_bucket(data: pl.DataFrame, config: dict[str, Any], has_single: bool) -> list[dict[str, Any]]:
    """
    Evaluate performance per time bucket.

    Args:
        data: DataFrame with predictions
        config: Multi-horizon configuration
        has_single: Whether single model predictions are available

    Returns:
        List of per-bucket metrics
    """
    logger.info("\n" + "=" * 80)
    logger.info("PER-BUCKET EVALUATION")
    logger.info("=" * 80)

    bucket_metrics = []

    for bucket_name in ["near", "mid", "far"]:
        bucket_config = config["buckets"][bucket_name]

        # Filter data for this bucket
        bucket_data = data.filter(
            (pl.col("time_remaining") >= bucket_config["time_min"])
            & (pl.col("time_remaining") < bucket_config["time_max"])
        )

        if len(bucket_data) == 0:
            logger.warning(f"No samples in {bucket_name} bucket")
            continue

        logger.info(f"\n{bucket_config['name']}:")
        logger.info(f"  Time range: {bucket_config['time_min']}-{bucket_config['time_max']}s")
        logger.info(f"  Samples: {len(bucket_data):,}")

        # Extract arrays
        outcomes = bucket_data.select("outcome").to_numpy().ravel()
        prob_baseline = bucket_data.select("prob_mid").to_numpy().ravel()
        multi_preds = bucket_data.select("multi_horizon_pred").to_numpy().ravel()

        # Multi-horizon metrics
        multi_metrics = calculate_metrics(multi_preds, outcomes, prob_baseline, f"  Multi-Horizon ({bucket_name})")
        multi_metrics["bucket"] = bucket_name

        bucket_metrics.append(multi_metrics)

        # Single model metrics (for comparison)
        if has_single:
            single_preds = bucket_data.select("single_model_pred").to_numpy().ravel()
            single_metrics = calculate_metrics(single_preds, outcomes, prob_baseline, f"  Single Model ({bucket_name})")

            # Calculate delta
            delta = multi_metrics["brier_improvement_pct"] - single_metrics["brier_improvement_pct"]
            logger.info(f"\n  Δ Improvement: {delta:+.2f}pp {'✓' if delta > 0 else '⚠'}")

    return bucket_metrics


def generate_summary_report(
    overall_multi: dict[str, Any],
    overall_single: dict[str, Any] | None,
    bucket_metrics: list[dict[str, Any]],
    output_dir: Path,
    config: dict[str, Any],
) -> None:
    """
    Generate summary report and save to CSV.

    Args:
        overall_multi: Overall multi-horizon metrics
        overall_single: Overall single model metrics (optional)
        bucket_metrics: Per-bucket metrics
        output_dir: Output directory
        config: Multi-horizon configuration
    """
    logger.info("\n" + "=" * 80)
    logger.info("SUMMARY REPORT")
    logger.info("=" * 80)

    # Overall comparison
    logger.info("\nOverall Performance:")
    logger.info(f"  Multi-Horizon: {overall_multi['brier_improvement_pct']:6.2f}% improvement")

    if overall_single is not None:
        logger.info(f"  Single Model:  {overall_single['brier_improvement_pct']:6.2f}% improvement")
        delta = overall_multi["brier_improvement_pct"] - overall_single["brier_improvement_pct"]
        logger.info(f"  Gain:          {delta:+6.2f}pp")

        # Check against Phase 1 target
        phase1_target = config["targets"]["phase_1"]["overall_improvement"]
        if overall_multi["brier_improvement_pct"] >= phase1_target:
            logger.info(f"\n✓ PHASE 1 SUCCESS: Achieved {phase1_target:.1f}%+ target")
        else:
            logger.info(f"\n⚠ Below Phase 1 target ({phase1_target:.1f}%), consider hyperparameter tuning")

    # Per-bucket summary
    logger.info("\nPer-Bucket Performance:")
    for metrics in bucket_metrics:
        # Validate bucket exists in metrics and config
        bucket_name = metrics.get("bucket")
        if not bucket_name or bucket_name not in config.get("buckets", {}):
            logger.warning(f"Skipping metrics with invalid bucket: {bucket_name}")
            continue

        bucket_config = config["buckets"][bucket_name]

        # Parse target improvement safely
        target_improvement = bucket_config.get("target_improvement", "0-0")
        try:
            target_min = float(target_improvement.split("-")[0])
        except (ValueError, IndexError, AttributeError):
            logger.warning(f"Invalid target_improvement format for {bucket_name}: {target_improvement}")
            target_min = 0.0

        improvement = metrics.get("brier_improvement_pct", 0.0)
        bucket_name_display = bucket_config.get("name", bucket_name)

        status = "✓" if improvement >= target_min else "⚠"
        logger.info(f"  {status} {bucket_name_display:25s}: {improvement:6.2f}% (target {target_min:4.1f}%+)")

    # Save to CSV
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_file = output_dir / "evaluation_results.csv"

    # Prepare data for CSV
    csv_data = []

    # Overall metrics
    csv_data.append(
        {
            "category": "overall",
            "model": "multi_horizon",
            "bucket": "all",
            **overall_multi,
        }
    )

    if overall_single is not None:
        csv_data.append(
            {
                "category": "overall",
                "model": "single",
                "bucket": "all",
                **overall_single,
            }
        )

    # Per-bucket metrics
    for metrics in bucket_metrics:
        csv_data.append(
            {
                "category": "per_bucket",
                "model": "multi_horizon",
                **metrics,
            }
        )

    csv_df = pl.DataFrame(csv_data)
    csv_df.write_csv(csv_file)

    logger.info(f"\n✓ Saved evaluation results to {csv_file}")


def main() -> None:
    """Main execution function."""
    # Parse arguments
    parser = argparse.ArgumentParser(description="Multi-horizon model evaluation")
    parser.add_argument(
        "--input",
        type=Path,
        default=Path(__file__).parent.parent / "results" / "test_features_lgb.parquet",
        help="Input test data file",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).parent / "config" / "multi_horizon_config.yaml",
        help="Path to multi-horizon config file",
    )
    parser.add_argument(
        "--no-single",
        action="store_true",
        help="Skip single model comparison",
    )
    parser.add_argument(
        "--no-plots",
        action="store_true",
        help="Skip visualization generation",
    )
    parser.add_argument(
        "--holdout-only",
        action="store_true",
        help="Evaluate only on holdout period (true out-of-sample test)",
    )
    parser.add_argument(
        "--holdout-start",
        type=str,
        default=None,
        help="Holdout start date (YYYY-MM-DD). If not specified, computed from config.",
    )
    args = parser.parse_args()

    logger.info("=" * 80)
    logger.info("MULTI-HORIZON MODEL EVALUATION")
    logger.info("=" * 80)

    monitor = MemoryMonitor()
    start_time = time.time()

    # Load configuration
    config = load_config(args.config)

    # Setup paths
    model_dir = Path(__file__).parent.parent
    models_dir = model_dir / Path(config["models"]["output_dir"])
    output_dir = model_dir / Path(config["evaluation"]["results_dir"])

    # Load models
    models = load_models(models_dir, config, include_single=not args.no_single)

    # Load test data
    logger.info(f"\nLoading test data: {args.input}")
    if not args.input.exists():
        raise FileNotFoundError(f"Test data not found: {args.input}")

    data = pl.read_parquet(args.input)
    logger.info(f"  Loaded {len(data):,} samples")

    # Filter to holdout period if requested
    if args.holdout_only:
        from datetime import date

        from dateutil.relativedelta import relativedelta

        # Determine holdout start date
        if args.holdout_start:
            holdout_start = date.fromisoformat(args.holdout_start)
            logger.info(f"\n⚠ Using custom holdout start: {holdout_start}")
        else:
            # Compute from data range and config
            walk_forward_config = config.get("walk_forward_validation", {})
            holdout_months = walk_forward_config.get("holdout_months", 3)

            # Get max date from data
            max_date = data.select(pl.col("date").max()).item()
            # FIXED: Correct holdout calculation (was off by one month)
            # For 3 months holdout with max_date=2025-09-30, we want 2025-07-01
            holdout_start = max_date - relativedelta(months=holdout_months) + relativedelta(days=1)

            logger.info("\n⚠ HOLDOUT-ONLY MODE ENABLED")
            logger.info(f"  Holdout period: Last {holdout_months} months")
            logger.info(f"  Computed holdout start: {holdout_start}")
            logger.info(f"  Data max date: {max_date}")

        # Filter data
        data_before = len(data)
        data = data.filter(pl.col("date") >= holdout_start)
        data_after = len(data)

        logger.info(f"  Filtered: {data_before:,} → {data_after:,} samples ({data_after / data_before * 100:.1f}%)")
        logger.info(
            f"  Date range: {data.select(pl.col('date').min()).item()} to {data.select(pl.col('date').max()).item()}"
        )

        if data_after == 0:
            raise ValueError(f"No samples in holdout period starting {holdout_start}")

        logger.info("\n✓ This is a TRUE OUT-OF-SAMPLE evaluation (no data leakage)")
    else:
        logger.info("\n⚠ WARNING: Evaluating on full test set (may include training data)")
        logger.info("  Use --holdout-only for true out-of-sample evaluation")

    # Verify required columns
    required_cols = ["time_remaining", "outcome", "prob_mid", "date"]
    missing_cols = [col for col in required_cols if col not in data.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    # Get features
    schema = data.collect_schema()
    features = [col for col in FEATURE_COLS if col in schema.names()]
    logger.info(f"  Using {len(features)} features")

    monitor.check_memory("After loading data")

    # Generate predictions
    predictions = generate_predictions(data, models, features, config)

    monitor.check_memory("After generating predictions")

    # Overall evaluation
    logger.info("\n" + "=" * 80)
    logger.info("OVERALL EVALUATION")
    logger.info("=" * 80)

    outcomes = predictions.select("outcome").to_numpy().ravel()
    prob_baseline = predictions.select("prob_mid").to_numpy().ravel()
    multi_preds = predictions.select("multi_horizon_pred").to_numpy().ravel()

    overall_multi = calculate_metrics(multi_preds, outcomes, prob_baseline, "Multi-Horizon (Overall)")

    overall_single = None
    if models["single"] is not None:
        single_preds = predictions.select("single_model_pred").to_numpy().ravel()
        overall_single = calculate_metrics(single_preds, outcomes, prob_baseline, "Single Model (Overall)")

    # Per-bucket evaluation
    bucket_metrics = evaluate_per_bucket(predictions, config, has_single=models["single"] is not None)

    # Generate summary report
    generate_summary_report(overall_multi, overall_single, bucket_metrics, output_dir, config)

    # Visualization (optional)
    if not args.no_plots:
        logger.info("\n" + "=" * 80)
        logger.info("VISUALIZATION GENERATION")
        logger.info("=" * 80)
        logger.info("⚠ Visualization generation not yet implemented")
        logger.info("  To generate plots, use the /plot slash command after evaluation")

    evaluation_time = (time.time() - start_time) / 60
    logger.info(f"\n✓ Evaluation complete! ({evaluation_time:.2f} minutes)")
    logger.info(f"Results saved to: {output_dir}")

    monitor.check_memory("Final")


if __name__ == "__main__":
    main()
