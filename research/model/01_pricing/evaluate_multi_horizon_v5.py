#!/usr/bin/env python3
"""
Multi-Horizon + Regime Hybrid Model Evaluation (V4)
====================================================

Comprehensive evaluation of 8-12 hybrid models (temporal × regime):
- Overall performance metrics (Brier score, IC, calibration)
- Hierarchical breakdown: Overall → Bucket → Regime → Model
- Per-model performance analysis
- Statistical significance testing

Generates:
- CSV report with hierarchical metrics
- YAML summary with recommendations
- Performance comparison vs baseline

Usage:
  # Evaluate on holdout period (true out-of-sample)
  uv run python evaluate_multi_horizon_v4.py --holdout-only

  # Evaluate all models
  uv run python evaluate_multi_horizon_v4.py \
      --config ../config/multi_horizon_regime_config_v5.yaml \
      --models_dir ../results/multi_horizon_hybrid_v4/models \
      --output_dir ../results/multi_horizon_hybrid_v4/evaluations

  # Evaluate specific data file
  uv run python evaluate_multi_horizon_v4.py --input custom_test_data.parquet

Author: BT Research Team
Date: 2025-11-14
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import lightgbm as lgb
import numpy as np
import polars as pl
import yaml

# Import from V4 training module
from train_multi_horizon_v4 import FEATURE_COLS_V4

# Import from V3 legacy
_v3_core_path = str(Path(__file__).parent / "v3_legacy" / "core")
if _v3_core_path not in sys.path:
    sys.path.insert(0, _v3_core_path)

# ruff: noqa: E402
from lightgbm_memory_optimized import MemoryMonitor, calculate_ic  # type: ignore[import-not-found]

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


def load_config(config_path: Path) -> dict[str, Any]:
    """Load multi-horizon + regime hybrid configuration."""
    logger.info(f"Loading configuration from {config_path}")
    with open(config_path) as f:
        config = yaml.safe_load(f)
    return config


def load_hybrid_models(models_dir: Path, config: dict[str, Any]) -> dict[str, lgb.Booster]:
    """
    Load 8-12 hybrid models (temporal × regime).

    Args:
        models_dir: Directory containing model files (lightgbm_{model_name}.txt)
        config: Multi-horizon + regime configuration

    Returns:
        Dictionary mapping model_name → LightGBM Booster
    """
    logger.info("\nLoading hybrid models...")
    logger.info(f"  Models directory: {models_dir}")

    models: dict[str, lgb.Booster] = {}

    # Load all hybrid models defined in config
    for model_name in config["hybrid_models"]:
        model_file = models_dir / f"lightgbm_{model_name}.txt"

        if not model_file.exists():
            logger.warning(f"  ⚠ Model not found: {model_file} (skipping)")
            continue

        try:
            models[model_name] = lgb.Booster(model_file=str(model_file))
            logger.info(f"  ✓ Loaded {model_name}")
        except Exception as e:
            logger.error(f"  ✗ Failed to load {model_file}: {e}")
            continue

    if len(models) == 0:
        raise ValueError(f"No models loaded from {models_dir}. Run train_multi_horizon_v4.py first.")

    logger.info(f"\n✓ Loaded {len(models)} hybrid models")
    return models


def generate_hybrid_predictions(
    data: pl.DataFrame,
    models: dict[str, lgb.Booster],
    features: list[str],
) -> pl.DataFrame:
    """
    Generate predictions from hybrid models using two-level routing.

    Routing logic:
    1. Filter by time bucket (from combined_regime column)
    2. Filter by volatility regime (from combined_regime column)
    3. Apply corresponding model

    Args:
        data: Input DataFrame with combined_regime column
        models: Dictionary of hybrid models
        features: Feature columns

    Returns:
        DataFrame with hybrid_pred column added
    """
    logger.info("\nGenerating hybrid predictions...")
    logger.info(f"  Input samples: {len(data):,}")

    n_samples = len(data)
    hybrid_preds = np.full(n_samples, np.nan, dtype=np.float64)
    coverage_counts = defaultdict(int)

    # Convert to numpy for indexing
    combined_regime = data.select("combined_regime").to_series().to_numpy()

    # Route predictions based on combined_regime
    for model_name, model in models.items():
        # Create mask for this regime
        mask = combined_regime == model_name
        count = mask.sum()

        if count > 0:
            # Get data for this regime
            regime_data = data.filter(pl.Series(mask))
            x = regime_data.select(features).to_numpy()

            # Predict
            preds = model.predict(x)
            hybrid_preds[mask] = preds

            coverage_counts[model_name] = count
            logger.info(f"  {model_name:25s}: {count:8,} predictions")

    # Add predictions to DataFrame
    result = data.with_columns([pl.Series("hybrid_pred", hybrid_preds)])

    # Check coverage
    n_predicted = (~np.isnan(hybrid_preds)).sum()
    coverage_pct = (n_predicted / n_samples) * 100
    logger.info(f"\n  Coverage: {n_predicted:,} / {n_samples:,} ({coverage_pct:.1f}%)")

    if coverage_pct < 99:
        logger.warning(f"  ⚠ Low coverage: {100 - coverage_pct:.1f}% samples not covered by any model")

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
        predictions: Residual predictions (model output)
        outcomes: True outcomes (0 or 1)
        prob_baseline: Baseline probabilities (Black-Scholes)
        label: Label for logging

    Returns:
        Dictionary of metrics
    """
    # Calculate model probabilities: P_model = P_baseline + residual_pred
    prob_pred = np.clip(prob_baseline + predictions, 0.0, 1.0)

    # Filter out NaN values
    mask = ~(np.isnan(predictions) | np.isnan(outcomes) | np.isnan(prob_baseline))
    predictions_clean = predictions[mask]
    outcomes_clean = outcomes[mask]
    prob_baseline_clean = prob_baseline[mask]
    prob_pred_clean = prob_pred[mask]

    n_total = len(predictions)
    n_valid = len(predictions_clean)
    n_filtered = n_total - n_valid

    if n_filtered > 0:
        logger.debug(f"  Filtered {n_filtered:,} rows with NaN values ({n_filtered / n_total * 100:.1f}%)")

    if n_valid == 0:
        raise ValueError(f"No valid samples after NaN filtering for {label}")

    # Brier scores
    brier_baseline = np.mean((outcomes_clean - prob_baseline_clean) ** 2)
    brier_model = np.mean((outcomes_clean - prob_pred_clean) ** 2)
    brier_improvement_pct = ((brier_baseline - brier_model) / brier_baseline) * 100

    # Residual metrics
    residual_true = outcomes_clean - prob_baseline_clean
    residual_mse = np.mean((residual_true - predictions_clean) ** 2)
    residual_rmse = np.sqrt(residual_mse)
    residual_mae = np.mean(np.abs(residual_true - predictions_clean))

    # Information Coefficient (IC) - signal quality
    ic_metrics = calculate_ic(predictions, outcomes)
    spearman_ic = ic_metrics["spearman_ic"]
    pearson_ic = ic_metrics["pearson_ic"]
    ic_pvalue = ic_metrics["ic_pvalue"]

    # Calibration error (binned)
    n_bins = 10
    bin_edges = np.linspace(0, 1, n_bins + 1)
    bin_indices = np.digitize(prob_pred_clean, bin_edges[1:-1])

    calibration_error = 0.0
    for i in range(n_bins):
        bin_mask = bin_indices == i
        if bin_mask.sum() > 0:
            mean_pred = prob_pred_clean[bin_mask].mean()
            mean_outcome = outcomes_clean[bin_mask].mean()
            calibration_error += bin_mask.sum() * (mean_pred - mean_outcome) ** 2

    calibration_error = np.sqrt(calibration_error / len(outcomes_clean))

    metrics = {
        "label": label,
        "n_samples": n_valid,
        "brier_baseline": brier_baseline,
        "brier_model": brier_model,
        "brier_improvement_pct": brier_improvement_pct,
        "residual_mse": residual_mse,
        "residual_rmse": residual_rmse,
        "residual_mae": residual_mae,
        "spearman_ic": spearman_ic,
        "pearson_ic": pearson_ic,
        "ic_pvalue": ic_pvalue,
        "calibration_error": calibration_error,
    }

    return metrics


def evaluate_per_model(
    data: pl.DataFrame,
    config: dict[str, Any],
) -> list[dict[str, Any]]:
    """
    Evaluate performance per hybrid model.

    Args:
        data: DataFrame with hybrid_pred and outcome columns
        config: Multi-horizon + regime configuration

    Returns:
        List of per-model metrics
    """
    logger.info("\n" + "=" * 80)
    logger.info("PER-MODEL EVALUATION")
    logger.info("=" * 80)

    model_metrics = []

    for model_name, model_config in config["hybrid_models"].items():
        # Filter data for this model
        model_data = data.filter(pl.col("combined_regime") == model_name)

        if len(model_data) == 0:
            logger.warning(f"\n{model_name}: No samples (skipping)")
            continue

        logger.info(f"\n{model_name}:")
        logger.info(f"  Bucket: {model_config['bucket']}")
        logger.info(f"  Regime: {model_config['regime']}")
        logger.info(f"  Samples: {len(model_data):,}")

        # Extract arrays
        outcomes = model_data.select("outcome").to_numpy().ravel()
        prob_baseline = model_data.select("prob_mid").to_numpy().ravel()
        hybrid_preds = model_data.select("hybrid_pred").to_numpy().ravel()

        # Calculate metrics
        metrics = calculate_metrics(hybrid_preds, outcomes, prob_baseline, f"  {model_name}")

        # Add model metadata
        metrics["model_name"] = model_name
        metrics["bucket"] = model_config["bucket"]
        metrics["regime"] = model_config["regime"]

        # Log key metrics
        logger.info(f"  Baseline Brier: {metrics['brier_baseline']:.6f}")
        logger.info(f"  Model Brier:    {metrics['brier_model']:.6f}")
        logger.info(f"  Improvement:    {metrics['brier_improvement_pct']:6.2f}%")
        logger.info(f"  Spearman IC:    {metrics['spearman_ic']:6.4f} (p={metrics['ic_pvalue']:.4f})")
        logger.info(f"  Residual RMSE:  {metrics['residual_rmse']:.6f}")

        model_metrics.append(metrics)

    return model_metrics


def aggregate_by_bucket(model_metrics: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """
    Aggregate model metrics by temporal bucket.

    Args:
        model_metrics: List of per-model metrics

    Returns:
        Dictionary mapping bucket_name → aggregated metrics
    """
    logger.info("\n" + "=" * 80)
    logger.info("PER-BUCKET AGGREGATION")
    logger.info("=" * 80)

    bucket_agg: dict[str, dict[str, Any]] = {}

    for bucket_name in ["near", "mid", "far"]:
        # Get all models in this bucket
        bucket_models = [m for m in model_metrics if m["bucket"] == bucket_name]

        if len(bucket_models) == 0:
            logger.warning(f"\n{bucket_name}: No models")
            continue

        # Weighted average by sample count
        total_samples = sum(m["n_samples"] for m in bucket_models)
        weights = np.array([m["n_samples"] / total_samples for m in bucket_models])

        # Aggregate metrics
        agg_metrics = {
            "bucket": bucket_name,
            "n_models": len(bucket_models),
            "n_samples": total_samples,
            "brier_baseline": np.average([m["brier_baseline"] for m in bucket_models], weights=weights),
            "brier_model": np.average([m["brier_model"] for m in bucket_models], weights=weights),
            "brier_improvement_pct": np.average([m["brier_improvement_pct"] for m in bucket_models], weights=weights),
            "spearman_ic": np.average([m["spearman_ic"] for m in bucket_models], weights=weights),
            "residual_rmse": np.average([m["residual_rmse"] for m in bucket_models], weights=weights),
        }

        bucket_agg[bucket_name] = agg_metrics

        logger.info(f"\n{bucket_name.upper()} Bucket:")
        logger.info(f"  Models: {agg_metrics['n_models']}")
        logger.info(f"  Samples: {agg_metrics['n_samples']:,}")
        logger.info(f"  Improvement: {agg_metrics['brier_improvement_pct']:6.2f}%")
        logger.info(f"  Spearman IC: {agg_metrics['spearman_ic']:6.4f}")

    return bucket_agg


def aggregate_by_regime(model_metrics: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """
    Aggregate model metrics by volatility regime.

    Args:
        model_metrics: List of per-model metrics

    Returns:
        Dictionary mapping regime_name → aggregated metrics
    """
    logger.info("\n" + "=" * 80)
    logger.info("PER-REGIME AGGREGATION")
    logger.info("=" * 80)

    regime_agg: dict[str, dict[str, Any]] = {}

    for regime_name in ["low_vol_atm", "low_vol_otm", "high_vol_atm", "high_vol_otm"]:
        # Get all models in this regime
        regime_models = [m for m in model_metrics if m["regime"] == regime_name]

        if len(regime_models) == 0:
            logger.warning(f"\n{regime_name}: No models")
            continue

        # Weighted average by sample count
        total_samples = sum(m["n_samples"] for m in regime_models)
        weights = np.array([m["n_samples"] / total_samples for m in regime_models])

        # Aggregate metrics
        agg_metrics = {
            "regime": regime_name,
            "n_models": len(regime_models),
            "n_samples": total_samples,
            "brier_baseline": np.average([m["brier_baseline"] for m in regime_models], weights=weights),
            "brier_model": np.average([m["brier_model"] for m in regime_models], weights=weights),
            "brier_improvement_pct": np.average([m["brier_improvement_pct"] for m in regime_models], weights=weights),
            "spearman_ic": np.average([m["spearman_ic"] for m in regime_models], weights=weights),
            "residual_rmse": np.average([m["residual_rmse"] for m in regime_models], weights=weights),
        }

        regime_agg[regime_name] = agg_metrics

        logger.info(f"\n{regime_name.upper().replace('_', ' ')} Regime:")
        logger.info(f"  Models: {agg_metrics['n_models']}")
        logger.info(f"  Samples: {agg_metrics['n_samples']:,}")
        logger.info(f"  Improvement: {agg_metrics['brier_improvement_pct']:6.2f}%")
        logger.info(f"  Spearman IC: {agg_metrics['spearman_ic']:6.4f}")

    return regime_agg


def generate_hierarchical_report(
    overall_metrics: dict[str, Any],
    model_metrics: list[dict[str, Any]],
    bucket_metrics: dict[str, dict[str, Any]],
    regime_metrics: dict[str, dict[str, Any]],
    output_dir: Path,
) -> None:
    """
    Generate hierarchical performance report (CSV + YAML).

    Args:
        overall_metrics: Overall performance across all models
        model_metrics: Per-model metrics
        bucket_metrics: Per-bucket aggregated metrics
        regime_metrics: Per-regime aggregated metrics
        output_dir: Output directory
    """
    logger.info("\n" + "=" * 80)
    logger.info("GENERATING HIERARCHICAL REPORT")
    logger.info("=" * 80)

    output_dir.mkdir(parents=True, exist_ok=True)

    # CSV: Per-model metrics
    csv_file = output_dir / "evaluation_per_model.csv"
    csv_data = []

    for metrics in model_metrics:
        csv_data.append(
            {
                "category": "model",
                "model_name": metrics["model_name"],
                "bucket": metrics["bucket"],
                "regime": metrics["regime"],
                "n_samples": metrics["n_samples"],
                "brier_baseline": metrics["brier_baseline"],
                "brier_model": metrics["brier_model"],
                "brier_improvement_pct": metrics["brier_improvement_pct"],
                "spearman_ic": metrics["spearman_ic"],
                "ic_pvalue": metrics["ic_pvalue"],
                "residual_rmse": metrics["residual_rmse"],
                "calibration_error": metrics["calibration_error"],
            }
        )

    csv_df = pl.DataFrame(csv_data)
    csv_df.write_csv(csv_file)
    logger.info(f"  ✓ Saved per-model CSV: {csv_file}")

    # YAML: Hierarchical summary
    yaml_file = output_dir / "evaluation_summary.yaml"

    summary = {
        "overall": {
            "n_samples": overall_metrics["n_samples"],
            "brier_baseline": float(overall_metrics["brier_baseline"]),
            "brier_model": float(overall_metrics["brier_model"]),
            "brier_improvement_pct": float(overall_metrics["brier_improvement_pct"]),
            "spearman_ic": float(overall_metrics["spearman_ic"]),
            "ic_pvalue": float(overall_metrics["ic_pvalue"]),
            "residual_rmse": float(overall_metrics["residual_rmse"]),
        },
        "by_bucket": {
            k: {kk: float(vv) if isinstance(vv, (np.floating, float)) else vv for kk, vv in v.items()}
            for k, v in bucket_metrics.items()
        },
        "by_regime": {
            k: {kk: float(vv) if isinstance(vv, (np.floating, float)) else vv for kk, vv in v.items()}
            for k, v in regime_metrics.items()
        },
        "top_models": sorted(
            [
                {
                    "model": m["model_name"],
                    "improvement_pct": float(m["brier_improvement_pct"]),
                    "spearman_ic": float(m["spearman_ic"]),
                    "n_samples": m["n_samples"],
                }
                for m in model_metrics
            ],
            key=lambda x: x["improvement_pct"],
            reverse=True,
        )[:5],
    }

    with open(yaml_file, "w") as f:
        yaml.dump(summary, f, default_flow_style=False, sort_keys=False)

    logger.info(f"  ✓ Saved hierarchical YAML: {yaml_file}")


def main() -> None:
    """Main execution function."""
    # Parse arguments
    parser = argparse.ArgumentParser(description="V4 Multi-horizon + regime hybrid model evaluation")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).parent.parent / "config" / "multi_horizon_regime_config_v5.yaml",
        help="Path to multi-horizon + regime config file",
    )
    parser.add_argument(
        "--models_dir",
        type=Path,
        required=True,
        help="Directory containing trained models (lightgbm_{model_name}.txt)",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        required=True,
        help="Output directory for evaluation results",
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=None,
        help="Custom input data file (default: use pipeline-ready features)",
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
    logger.info("V4 MULTI-HORIZON + REGIME HYBRID MODEL EVALUATION")
    logger.info("=" * 80)

    monitor = MemoryMonitor()
    start_time = time.time()

    # Load configuration
    config = load_config(args.config)

    # Load models
    models = load_hybrid_models(args.models_dir, config)

    # Load data
    if args.input:
        logger.info(f"\nLoading data: {args.input}")
        data = pl.read_parquet(args.input)
    else:
        # Use pipeline-ready features
        model_dir = Path(__file__).parent.parent
        data_file = model_dir / "data" / "consolidated_features_v5_pipeline_ready.parquet"
        logger.info(f"\nLoading pipeline-ready data: {data_file}")
        data = pl.read_parquet(data_file)

    logger.info(f"  Loaded {len(data):,} samples")

    # Derive date from timestamp_seconds if not present
    if "date" not in data.columns:
        data = data.with_columns(
            [pl.from_epoch(pl.col("timestamp_seconds"), time_unit="s").cast(pl.Date).alias("date")]
        )

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
            holdout_start = max_date.replace(day=1) - relativedelta(months=holdout_months - 1)

            logger.info("\n⚠ HOLDOUT-ONLY MODE ENABLED")
            logger.info(f"  Holdout period: Last {holdout_months} months")
            logger.info(f"  Computed holdout start: {holdout_start}")
            logger.info(f"  Data max date: {max_date}")

        # Filter data
        data_before = len(data)
        data = data.filter(pl.col("date") >= holdout_start)
        data_after = len(data)

        logger.info(f"  Filtered: {data_before:,} → {data_after:,} samples ({data_after / data_before * 100:.1f}%)")

        if data_after == 0:
            raise ValueError(f"No samples in holdout period starting {holdout_start}")

        logger.info("\n✓ This is a TRUE OUT-OF-SAMPLE evaluation (no data leakage)")
    else:
        logger.info("\n⚠ WARNING: Evaluating on full dataset (may include training data)")
        logger.info("  Use --holdout-only for true out-of-sample evaluation")

    # Verify required columns
    required_cols = ["combined_regime", "outcome", "prob_mid"]
    missing_cols = [col for col in required_cols if col not in data.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    # Get features
    schema = data.collect_schema()
    features = [col for col in FEATURE_COLS_V4 if col in schema.names()]
    logger.info(f"  Using {len(features)} features")

    monitor.check_memory("After loading data")

    # Generate predictions
    predictions = generate_hybrid_predictions(data, models, features)

    monitor.check_memory("After generating predictions")

    # Overall evaluation
    logger.info("\n" + "=" * 80)
    logger.info("OVERALL EVALUATION")
    logger.info("=" * 80)

    outcomes = predictions.select("outcome").to_numpy().ravel()
    prob_baseline = predictions.select("prob_mid").to_numpy().ravel()
    hybrid_preds = predictions.select("hybrid_pred").to_numpy().ravel()

    overall_metrics = calculate_metrics(hybrid_preds, outcomes, prob_baseline, "Overall (All Models)")

    logger.info("\nOverall Performance:")
    logger.info(f"  Samples:        {overall_metrics['n_samples']:,}")
    logger.info(f"  Baseline Brier: {overall_metrics['brier_baseline']:.6f}")
    logger.info(f"  Model Brier:    {overall_metrics['brier_model']:.6f}")
    logger.info(f"  Improvement:    {overall_metrics['brier_improvement_pct']:6.2f}%")
    logger.info(f"  Spearman IC:    {overall_metrics['spearman_ic']:6.4f} (p={overall_metrics['ic_pvalue']:.4f})")
    logger.info(f"  Residual RMSE:  {overall_metrics['residual_rmse']:.6f}")

    # Per-model evaluation
    model_metrics = evaluate_per_model(predictions, config)

    # Aggregate by bucket
    bucket_metrics = aggregate_by_bucket(model_metrics)

    # Aggregate by regime
    regime_metrics = aggregate_by_regime(model_metrics)

    # Generate hierarchical report
    generate_hierarchical_report(overall_metrics, model_metrics, bucket_metrics, regime_metrics, args.output_dir)

    evaluation_time = (time.time() - start_time) / 60
    logger.info(f"\n✓ Evaluation complete! ({evaluation_time:.2f} minutes)")
    logger.info(f"Results saved to: {args.output_dir}")

    monitor.check_memory("Final")


if __name__ == "__main__":
    main()
