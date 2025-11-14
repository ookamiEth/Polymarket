#!/usr/bin/env python3
"""
Multi-Horizon + Regime Hybrid LightGBM Training Script (V4)
============================================================

Train 8 specialized LightGBM models based on hierarchical regime structure:
- 2 temporal windows: near (0-300s), mid (300-900s)
- 4 volatility regimes: low_vol_atm, low_vol_otm, high_vol_atm, high_vol_otm
- Total: 8 hybrid models (far bucket excluded due to insufficient data)

Key Improvements from V3:
- Uses pre-prepared data with regime columns (no join needed)
- Two-level stratification (temporal + volatility regime)
- 13-fold walk-forward validation with expanding windows (10→22 months)
- Information Coefficient (IC) tracking for signal quality
- Residual learning architecture (outcome - Black-Scholes baseline)

Usage:
  # Train all 12 models (always uses walk-forward validation)
  uv run python train_multi_horizon_v4.py

  # Train specific model
  uv run python train_multi_horizon_v4.py --model near_low_vol_atm
  uv run python train_multi_horizon_v4.py --model far_low_vol_atm

  # With custom config
  uv run python train_multi_horizon_v4.py --config config/multi_horizon_regime_config.yaml

Note: Walk-forward validation is MANDATORY for production use.
      This ensures zero data leakage for time series forecasting.

Author: BT Research Team
Date: 2025-11-13
"""

from __future__ import annotations

import argparse
import gc
import logging
import sys
import time
from pathlib import Path
from typing import Any

import lightgbm as lgb
import numpy as np
import polars as pl
import yaml

# Import from V3 legacy (these functions work with any parquet data)
# Add V3 core directory to path for imports
_v3_core_path = str(Path(__file__).parent / "v3_legacy" / "core")
if _v3_core_path not in sys.path:
    sys.path.insert(0, _v3_core_path)

# These imports are now from v3_legacy/core/lightgbm_memory_optimized.py
# ruff: noqa: E402
from lightgbm_memory_optimized import (  # type: ignore[import-not-found]
    MemoryMonitor,
    calculate_ic,
    create_lgb_dataset_from_parquet,
)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

# W&B integration
try:
    import wandb

    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    logger.warning("W&B not available - continuing without experiment tracking")

# V4 Feature columns (156 features from consolidated_features_v4_pipeline_ready.parquet)
FEATURE_COLS_V4 = [
    "K",
    "S",
    "T_years",
    "ask_volume_ratio_1to5",
    "autocorr_decay",
    "autocorr_lag5_300s",
    "bid_ask_imbalance",
    "bid_ask_spread_bps",
    "bid_volume_ratio_1to5",
    "depth_imbalance_5",
    "depth_imbalance_ema_300s",
    "depth_imbalance_ema_3600s",
    "depth_imbalance_ema_60s",
    "depth_imbalance_ema_900s",
    "depth_imbalance_vol_300s",
    "depth_imbalance_vol_3600s",
    "depth_imbalance_vol_60s",
    "depth_imbalance_vol_900s",
    "depth_imbalance_vol_normalized",
    "downside_vol_300",
    "downside_vol_300s",
    "downside_vol_60",
    "downside_vol_900",
    "ema_900s",
    "funding_rate",
    "funding_rate_ema_300s",
    "funding_rate_ema_3600s",
    "funding_rate_ema_60s",
    "funding_rate_ema_900s",
    "garch_forecast_simple",
    "high_15m",
    "hour_cos",
    "hour_of_day_utc",
    "hour_sin",
    "hurst_300s",
    "imbalance_ema_300s",
    "imbalance_ema_3600s",
    "imbalance_ema_60s",
    "imbalance_ema_900s",
    "imbalance_vol_300s",
    "imbalance_vol_3600s",
    "imbalance_vol_60s",
    "imbalance_vol_900s",
    "is_extreme_condition",
    "iv_minus_downside_vol",
    "iv_minus_upside_vol",
    "iv_staleness_seconds",
    "kurtosis_300s",
    "log_moneyness",
    "low_15m",
    "mark_index_basis_bps",
    "mark_index_ema_300s",
    "mark_index_ema_3600s",
    "mark_index_ema_60s",
    "mark_index_ema_900s",
    # "market_regime",  # EXCLUDED: Categorical metadata (filtered by combined_regime)
    # "market_regime_4way",  # EXCLUDED: Categorical metadata
    "momentum_300s",
    "momentum_300s_ema_300s",
    "momentum_300s_ema_3600s",
    "momentum_300s_ema_60s",
    "momentum_300s_ema_900s",
    "momentum_900s",
    "momentum_900s_ema_300s",
    "momentum_900s_ema_3600s",
    "momentum_900s_ema_60s",
    "momentum_900s_ema_900s",
    "moneyness",
    "moneyness_cubed",
    "moneyness_distance",
    "moneyness_percentile",
    "moneyness_squared",
    "moneyness_x_time",
    "moneyness_x_vol",
    "oi_ema_3600s",
    "oi_ema_900s",
    "open_interest",
    "position_scale",
    "price_vs_ema_900",
    "range_300s",
    "range_300s_ema_300s",
    "range_300s_ema_3600s",
    "range_300s_ema_60s",
    "range_300s_ema_900s",
    "range_900s",
    "range_900s_ema_300s",
    "range_900s_ema_3600s",
    "range_900s_ema_60s",
    "range_900s_ema_900s",
    "realized_kurtosis_300",
    "realized_kurtosis_60",
    "realized_kurtosis_900",
    "realized_skewness_300",
    "realized_skewness_60",
    "realized_skewness_900",
    "returns_300s",
    "returns_60s",
    "returns_900s",
    "reversals_300s",
    "rv_300s",
    "rv_300s_ema_300s",
    "rv_300s_ema_3600s",
    "rv_300s_ema_60s",
    "rv_300s_ema_900s",
    "rv_60s",
    "rv_900s",
    "rv_900s_ema_300s",
    "rv_900s_ema_3600s",
    "rv_900s_ema_60s",
    "rv_900s_ema_900s",
    "rv_95th_percentile",
    "rv_ratio",
    "rv_ratio_15m_5m",
    "rv_ratio_1h_15m",
    "rv_ratio_5m_1m",
    "rv_term_structure",
    "sigma_mid",
    "skewness_300s",
    "spread_ema_300s",
    "spread_ema_3600s",
    "spread_ema_60s",
    "spread_ema_900s",
    "spread_vol_300s",
    "spread_vol_3600s",
    "spread_vol_60s",
    "spread_vol_900s",
    "spread_vol_normalized",
    "standardized_moneyness",
    "tail_risk_300s",
    # "temporal_regime",  # EXCLUDED: Categorical metadata (filtered by time range)
    "time_remaining",
    "time_since_high_15m",
    "time_since_low_15m",
    "timestamp_seconds",
    "total_ask_volume_5",
    "total_bid_volume_5",
    "upside_vol_300",
    "upside_vol_300s",
    "upside_vol_60",
    "upside_vol_900",
    "vol_acceleration_300s",
    "vol_asymmetry_300s",
    "vol_asymmetry_ratio_300",
    "vol_asymmetry_ratio_60",
    "vol_asymmetry_ratio_900",
    "vol_high_thresh",
    "vol_low_thresh",
    "vol_of_vol_300",
    "vol_of_vol_300s",
    "vol_persistence_ar1",
    "weighted_mid_ema_300s",
    "weighted_mid_ema_3600s",
    "weighted_mid_ema_60s",
    "weighted_mid_ema_900s",
    "weighted_mid_price_5",
    "weighted_mid_velocity_normalized",
]


def load_config(config_path: Path) -> dict[str, Any]:
    """Load multi-horizon + regime configuration."""
    logger.info(f"Loading configuration from {config_path}")
    with open(config_path) as f:
        config = yaml.safe_load(f)
    return config


def load_shared_hyperparameters(shared_config_path: Path) -> dict[str, Any]:
    """Load shared hyperparameters from best model config."""
    logger.info(f"Loading shared hyperparameters from {shared_config_path}")
    with open(shared_config_path) as f:
        shared_config = yaml.safe_load(f)
    return shared_config["hyperparameters"]


def stratify_data_by_hybrid(
    input_file: Path,
    output_dir: Path,
    model_name: str,
    time_min: int,
    time_max: int,
) -> Path:
    """
    Stratify pipeline-ready dataset by combined regime (temporal + volatility).

    V4 simplification: Input file already has:
    - combined_regime column (e.g., "near_low_vol_atm")
    - residual target (outcome - prob_mid)
    - all 146 features

    Args:
        input_file: Source parquet file (consolidated_features_v4_pipeline_ready.parquet)
        output_dir: Output directory for stratified file
        model_name: Combined regime name (e.g., "near_low_vol_atm")
        time_min: Minimum time_remaining (seconds) - for validation
        time_max: Maximum time_remaining (seconds) - for validation

    Returns:
        Path to stratified output file with regime-specific data
    """
    logger.info(f"\nStratifying {model_name} ({time_min}-{time_max}s)...")
    logger.info(f"  Source: {input_file}")

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    # Determine output filename
    output_file = output_dir / f"{model_name}_data.parquet"

    # Check if already exists
    if output_file.exists():
        logger.info(f"  ✓ Already exists: {output_file}")
        # Get row count
        row_count = pl.scan_parquet(output_file).select(pl.len()).collect().item()
        logger.info(f"  Rows: {row_count:,}")
        return output_file

    logger.info(f"  Filtering combined_regime == '{model_name}'")

    # Load pipeline-ready data and filter by combined_regime
    df = pl.scan_parquet(input_file).filter(
        (pl.col("combined_regime") == model_name)
        & (pl.col("time_remaining") >= time_min)
        & (pl.col("time_remaining") < time_max)
    )

    # Stream to output (memory-efficient)
    df.sink_parquet(output_file, compression="snappy", statistics=True)

    # Get row count
    row_count = pl.scan_parquet(output_file).select(pl.len()).collect().item()

    logger.info(f"  ✓ Created: {output_file}")
    logger.info(f"  Rows: {row_count:,}")

    gc.collect()

    return output_file


def train_model_walk_forward(
    model_name: str,
    model_config: dict[str, Any],
    data_file: Path,
    hyperparameters: dict[str, Any],
    features: list[str],
    output_dir: Path,
    walk_forward_config: dict[str, Any],
    wandb_run: Any | None = None,
    wandb_config: dict[str, Any] | None = None,
) -> tuple[lgb.Booster, dict[str, Any]]:
    """
    Train LightGBM model using walk-forward validation.

    True walk-forward validation: both training and validation periods advance chronologically.
    This simulates real-world deployment where you always predict the future based on the past.

    Args:
        model_name: Model identifier (e.g., "near_low_vol_atm")
        model_config: Model-specific configuration from YAML
        data_file: Full dataset file (will be split temporally)
        hyperparameters: LightGBM hyperparameters
        features: Feature columns to use
        output_dir: Output directory for models
        walk_forward_config: Walk-forward configuration
        wandb_run: W&B run (optional)

    Returns:
        Tuple of (final model trained on all walk-forward data, aggregated metrics)
    """
    logger.info(f"\n{'=' * 80}")
    logger.info(f"WALK-FORWARD VALIDATION: {model_name.upper()}")
    logger.info(f"{'=' * 80}")

    # Create output directory for temporary files
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Output directory: {output_dir}")

    train_months = walk_forward_config["train_months"]
    val_months = walk_forward_config["val_months"]
    step_months = walk_forward_config["step_months"]
    holdout_months = walk_forward_config["holdout_months"]
    embargo_days = walk_forward_config.get("embargo_days", 0)

    logger.info(f"Training window: {train_months} months")
    logger.info(f"Validation window: {val_months} months")
    logger.info(f"Step size: {step_months} month(s)")
    logger.info(f"Holdout test: last {holdout_months} months")
    if embargo_days > 0:
        logger.info(f"Embargo period: {embargo_days} day(s) between train/val")

    monitor = MemoryMonitor()

    # Load full dataset and get date range
    logger.info("\nAnalyzing data date range...")
    df_lazy = pl.scan_parquet(data_file)

    date_stats = df_lazy.select(
        [pl.col("date").min().alias("min_date"), pl.col("date").max().alias("max_date")]
    ).collect()

    from datetime import date

    min_date = date_stats["min_date"][0]
    max_date = date_stats["max_date"][0]

    # Calculate holdout test period (last N months)
    from dateutil.relativedelta import relativedelta

    # Holdout calculation: Reserve last N full months for final testing
    # Example: max_date=2024-12-31, holdout_months=3
    #   -> max_date.replace(day=1) = 2024-12-01 (first day of last month)
    #   -> subtract (3-1)=2 months = 2024-10-01
    #   -> holdout_start = 2024-10-01 (3 full months: Oct, Nov, Dec)
    #   -> walk_forward_end = 2024-09-30 (day before holdout)
    holdout_start = max_date.replace(day=1) - relativedelta(months=holdout_months - 1)
    walk_forward_end = holdout_start - relativedelta(days=1)

    logger.info(f"Full data range: {min_date} to {max_date}")
    logger.info(f"Walk-forward range: {min_date} to {walk_forward_end}")
    logger.info(f"Holdout test: {holdout_start} to {max_date}")

    # Generate walk-forward windows with embargo period
    # EXPANDING WINDOW: train_start stays at min_date, train_end advances each window
    windows: list[
        tuple[date, date, date, date, date, date]
    ] = []  # (train_start, train_end, embargo_start, embargo_end, val_start, val_end)
    train_start = min_date
    current_train_end = min_date + relativedelta(months=train_months) - relativedelta(days=1)

    while True:
        train_end = current_train_end

        # Add embargo period (gap between train and val)
        if embargo_days > 0:
            embargo_start = train_end + relativedelta(days=1)
            embargo_end = embargo_start + relativedelta(days=embargo_days - 1)
            val_start = embargo_end + relativedelta(days=1)
        else:
            # No embargo: val starts immediately after train
            embargo_start = train_end + relativedelta(days=1)
            embargo_end = train_end  # Empty embargo period
            val_start = train_end + relativedelta(days=1)

        val_end = val_start + relativedelta(months=val_months) - relativedelta(days=1)

        # Stop if validation period would overlap with holdout
        if val_end >= holdout_start:
            break

        windows.append((train_start, train_end, embargo_start, embargo_end, val_start, val_end))

        # Step forward (expanding window: only advance train_end, keep train_start fixed)
        current_train_end += relativedelta(months=step_months)

    logger.info(f"Generated {len(windows)} walk-forward windows")

    # Train and validate on each window
    window_metrics: list[dict[str, Any]] = []
    failed_windows: list[int] = []
    skipped_windows: list[int] = []
    walk_forward_start_time = time.time()

    for window_idx, (train_start, train_end, embargo_start, embargo_end, val_start, val_end) in enumerate(windows, 1):
        # Progress indicator
        progress_pct = (window_idx / len(windows)) * 100
        elapsed = time.time() - walk_forward_start_time
        if window_idx > 1:
            avg_time_per_window = elapsed / (window_idx - 1)
            remaining_windows = len(windows) - window_idx + 1
            eta_seconds = avg_time_per_window * remaining_windows
            eta_str = f"{int(eta_seconds // 60)}m {int(eta_seconds % 60)}s"
        else:
            eta_str = "estimating..."

        logger.info(f"\n{'─' * 80}")
        logger.info(f"Window {window_idx}/{len(windows)} [{progress_pct:.1f}%] | ETA: {eta_str}")
        logger.info(f"  Train:   {train_start} to {train_end}")
        if embargo_days > 0:
            logger.info(f"  Embargo: {embargo_start} to {embargo_end} ({embargo_days} day(s))")
        logger.info(f"  Val:     {val_start} to {val_end}")
        logger.info(f"{'─' * 80}")

        # Create temporal train/val splits
        df_train = df_lazy.filter((pl.col("date") >= train_start) & (pl.col("date") <= train_end))
        df_val = df_lazy.filter((pl.col("date") >= val_start) & (pl.col("date") <= val_end))

        # Check sample counts
        n_train = df_train.select(pl.len()).collect().item()
        n_val = df_val.select(pl.len()).collect().item()

        logger.info(f"Train samples: {n_train:,}")
        logger.info(f"Val samples:   {n_val:,}")

        if n_train < 10000 or n_val < 1000:
            logger.warning(f"⚠ Skipping window {window_idx}: insufficient samples")
            skipped_windows.append(window_idx)
            continue

        # Write temporary files
        temp_train = output_dir / f"temp_{model_name}_window{window_idx}_train.parquet"
        temp_val = output_dir / f"temp_{model_name}_window{window_idx}_val.parquet"

        df_train.sink_parquet(str(temp_train))
        df_val.sink_parquet(str(temp_val))

        try:
            # Create LightGBM datasets
            train_data, _ = create_lgb_dataset_from_parquet(str(temp_train), features, free_raw_data=True)
            val_data, baseline_brier_val = create_lgb_dataset_from_parquet(
                str(temp_val), features, reference=train_data, free_raw_data=True
            )

            if hasattr(baseline_brier_val, "__len__"):
                baseline_brier_val = float(np.mean(baseline_brier_val))

            # Train model on this window
            callbacks = [
                lgb.early_stopping(hyperparameters.get("early_stopping_rounds", 50)),
                lgb.log_evaluation(100),  # Less verbose
            ]

            model = lgb.train(
                hyperparameters,
                train_data,
                num_boost_round=hyperparameters.get("n_estimators", 1000),
                valid_sets=[val_data],  # Only validate on out-of-sample validation period
                valid_names=["val"],
                callbacks=callbacks,
            )

            # Calculate metrics
            best_score = model.best_score
            val_metrics = best_score.get("val", {})
            residual_mse = val_metrics.get("l2", 0.0)

            # Calculate Information Coefficient (IC) for signal quality
            val_df_for_ic = pl.read_parquet(temp_val, columns=features + ["outcome"])
            val_features_array = val_df_for_ic.select(features).to_numpy().astype(np.float32)
            val_outcomes = val_df_for_ic["outcome"].to_numpy().astype(np.float32)
            val_predictions_raw = model.predict(val_features_array, num_iteration=model.best_iteration)
            val_predictions = np.asarray(val_predictions_raw, dtype=np.float32)

            ic_metrics = calculate_ic(val_predictions, val_outcomes)
            spearman_ic = ic_metrics["spearman_ic"]
            pearson_ic = ic_metrics["pearson_ic"]
            ic_pvalue = ic_metrics["ic_pvalue"]

            logger.info(f"  Baseline Brier: {baseline_brier_val:.6f}")
            logger.info(f"  Residual MSE:   {residual_mse:.6f}")
            logger.info(f"  Spearman IC:    {spearman_ic:.4f} (p={ic_pvalue:.4f})")
            logger.info(f"  Pearson IC:     {pearson_ic:.4f}")

            window_metrics.append(
                {
                    "window": window_idx,
                    "train_start": train_start.isoformat(),
                    "train_end": train_end.isoformat(),
                    "embargo_days": embargo_days,
                    "val_start": val_start.isoformat(),
                    "val_end": val_end.isoformat(),
                    "n_train": n_train,
                    "n_val": n_val,
                    "baseline_brier": baseline_brier_val,
                    "residual_mse": residual_mse,
                    "spearman_ic": spearman_ic,
                    "pearson_ic": pearson_ic,
                    "ic_pvalue": ic_pvalue,
                    "best_iteration": model.best_iteration,
                }
            )

            # Log per-window metrics to W&B in real-time (ACTIVE LOGGING)
            if wandb_run is not None and wandb_config is not None and wandb_config.get("log_per_window", False):
                # Log individual window metrics
                wandb_run.log(
                    {
                        f"{model_name}/window_{window_idx}/residual_mse": residual_mse,
                        f"{model_name}/window_{window_idx}/baseline_brier": baseline_brier_val,
                        f"{model_name}/window_{window_idx}/spearman_ic": spearman_ic,
                        f"{model_name}/window_{window_idx}/pearson_ic": pearson_ic,
                        f"{model_name}/window_{window_idx}/ic_pvalue": ic_pvalue,
                        f"{model_name}/window_{window_idx}/n_train": n_train,
                        f"{model_name}/window_{window_idx}/n_val": n_val,
                        f"{model_name}/window_{window_idx}/best_iteration": model.best_iteration,
                        f"{model_name}/window_progress": window_idx,
                        f"{model_name}/overall_progress_pct": (window_idx / len(windows)) * 100,
                    }
                )

                # Log running statistics across windows so far
                mses_so_far = [m["residual_mse"] for m in window_metrics]
                ics_so_far = [m["spearman_ic"] for m in window_metrics]
                wandb_run.log(
                    {
                        f"{model_name}/mean_mse_so_far": float(np.mean(mses_so_far)),
                        f"{model_name}/std_mse_so_far": float(np.std(mses_so_far)),
                        f"{model_name}/mean_ic_so_far": float(np.mean(ics_so_far)),
                        f"{model_name}/std_ic_so_far": float(np.std(ics_so_far)),
                    }
                )
                logger.info(f"  ✓ W&B: Logged window {window_idx}/{len(windows)} metrics")

            # Clean up memory explicitly between windows
            del train_data, val_data, model, val_df_for_ic, val_features_array, val_outcomes
            del val_predictions_raw, val_predictions
            gc.collect()  # Force garbage collection to free memory before next window

        except Exception as e:
            logger.error(f"Error training window {window_idx}: {e}")
            failed_windows.append(window_idx)
            continue

        finally:
            # Remove temporary files
            temp_train.unlink(missing_ok=True)
            temp_val.unlink(missing_ok=True)

        monitor.check_memory(f"After window {window_idx}")

    # Validate window success rate
    total_windows = len(windows)
    successful_windows = len(window_metrics)
    success_rate = successful_windows / total_windows if total_windows > 0 else 0.0

    logger.info(f"\n{'=' * 80}")
    logger.info("WALK-FORWARD WINDOW SUMMARY")
    logger.info(f"{'=' * 80}")
    logger.info(f"Total windows: {total_windows}")
    logger.info(f"Successful:    {successful_windows} ({success_rate:.1%})")
    logger.info(f"Skipped:       {len(skipped_windows)} (insufficient data)")
    logger.info(f"Failed:        {len(failed_windows)} (training errors)")

    if len(failed_windows) > 0:
        logger.warning(f"Failed windows: {failed_windows}")
    if len(skipped_windows) > 0:
        logger.info(f"Skipped windows: {skipped_windows}")

    # Fail if success rate too low
    if successful_windows == 0:
        msg = "No windows produced valid models"
        raise ValueError(msg)
    if success_rate < 0.5:
        msg = f"Only {success_rate:.1%} of windows succeeded (minimum: 50%). Check data quality."
        raise ValueError(msg)
    if success_rate < 0.7:
        logger.warning(f"⚠ Low success rate: {success_rate:.1%} of windows succeeded")

    # Aggregate validation metrics
    logger.info(f"\n{'=' * 80}")
    logger.info("WALK-FORWARD VALIDATION RESULTS")
    logger.info(f"{'=' * 80}")

    residual_mses = [m["residual_mse"] for m in window_metrics]
    mean_mse = np.mean(residual_mses)
    std_mse = np.std(residual_mses)
    min_mse = np.min(residual_mses)
    max_mse = np.max(residual_mses)

    # Aggregate IC metrics
    spearman_ics = [m["spearman_ic"] for m in window_metrics]
    mean_ic = np.mean(spearman_ics)
    std_ic = np.std(spearman_ics)
    min_ic = np.min(spearman_ics)
    max_ic = np.max(spearman_ics)

    # Count statistically significant ICs (p < 0.05)
    sig_ics = sum(1 for m in window_metrics if m["ic_pvalue"] < 0.05)

    logger.info(f"Average Residual MSE: {mean_mse:.6f} ± {std_mse:.6f}")
    logger.info(f"Range: {min_mse:.6f} to {max_mse:.6f}")
    logger.info(f"\nAverage Spearman IC:  {mean_ic:.4f} ± {std_ic:.4f}")
    logger.info(f"IC Range: {min_ic:.4f} to {max_ic:.4f}")
    logger.info(f"Significant ICs: {sig_ics}/{len(window_metrics)} windows (p<0.05)")
    logger.info("\nWindow-by-window performance:")
    for m in window_metrics:
        ic_sig = "✓" if m["ic_pvalue"] < 0.05 else " "
        logger.info(f"  Window {m['window']}: MSE={m['residual_mse']:.6f}, IC={m['spearman_ic']:.4f} {ic_sig}")
    logger.info("\nNote: Brier improvement will be computed in evaluation phase.")

    # Train final production model on all walk-forward data (excluding holdout)
    logger.info(f"\n{'=' * 80}")
    logger.info("TRAINING FINAL MODEL ON ALL WALK-FORWARD DATA")
    logger.info(f"{'=' * 80}")

    df_final_train = df_lazy.filter((pl.col("date") >= min_date) & (pl.col("date") <= walk_forward_end))

    # Split walk-forward data using last val_months for validation
    final_train_end = walk_forward_end - relativedelta(months=val_months) + relativedelta(days=1)

    df_final_train_split = df_final_train.filter(pl.col("date") <= final_train_end)
    df_final_val_split = df_final_train.filter(pl.col("date") > final_train_end)

    temp_final_train = output_dir / f"temp_{model_name}_final_train.parquet"
    temp_final_val = output_dir / f"temp_{model_name}_final_val.parquet"

    df_final_train_split.sink_parquet(str(temp_final_train))
    df_final_val_split.sink_parquet(str(temp_final_val))

    try:
        # Create final datasets
        final_train_data, _ = create_lgb_dataset_from_parquet(str(temp_final_train), features, free_raw_data=True)
        final_val_data, final_baseline_brier = create_lgb_dataset_from_parquet(
            str(temp_final_val), features, reference=final_train_data, free_raw_data=True
        )

        if hasattr(final_baseline_brier, "__len__"):
            final_baseline_brier = float(np.mean(final_baseline_brier))

        # Train final model
        callbacks = [
            lgb.early_stopping(hyperparameters.get("early_stopping_rounds", 50)),
            lgb.log_evaluation(25),
        ]

        if wandb_run is not None:
            try:
                from wandb.integration.lightgbm import wandb_callback

                callbacks.append(wandb_callback())
            except ImportError:
                pass

        final_model = lgb.train(
            hyperparameters,
            final_train_data,
            num_boost_round=hyperparameters.get("n_estimators", 1000),
            valid_sets=[final_train_data, final_val_data],
            valid_names=["train", "val"],
            callbacks=callbacks,
        )

        # Save final model
        model_file = output_dir / f"lightgbm_{model_name}.txt"
        final_model.save_model(str(model_file))
        logger.info(f"✓ Saved final model to {model_file}")

        # Save config
        config_file = output_dir / f"config_{model_name}.yaml"

        # Get feature importance from final model
        importance = final_model.feature_importance(importance_type="gain")
        feature_names = final_model.feature_name()
        top_features = sorted(zip(feature_names, importance), key=lambda x: x[1], reverse=True)[:20]

        # Count total samples across all windows
        total_train_samples = sum(m.get("n_train", 0) for m in window_metrics)
        total_val_samples = sum(m.get("n_val", 0) for m in window_metrics)

        model_full_config = {
            "model": model_config,
            "hyperparameters": hyperparameters,
            "performance": {
                "baseline_brier": final_baseline_brier,
                "residual_mse": mean_mse,
                "residual_rmse": np.sqrt(mean_mse),
                "residual_mae": 0.0,  # Not tracked in walk-forward
                "training_time_minutes": (time.time() - walk_forward_start_time) / 60,
                "best_iteration": final_model.best_iteration,
            },
            "walk_forward": {
                "enabled": True,
                "n_windows": len(window_metrics),
                "mean_mse": mean_mse,
                "std_mse": std_mse,
                "min_mse": min_mse,
                "max_mse": max_mse,
                "mean_ic": mean_ic,
                "std_ic": std_ic,
                "min_ic": min_ic,
                "max_ic": max_ic,
                "significant_ics": sig_ics,
                "window_details": window_metrics,
            },
            "data": {
                "train_samples": total_train_samples,
                "val_samples": total_val_samples,
            },
            "top_features": [{"name": f, "importance": float(imp)} for f, imp in top_features],
        }

        with open(config_file, "w") as f:
            yaml.dump(model_full_config, f, default_flow_style=False, sort_keys=False)

        logger.info(f"✓ Saved config to {config_file}")

        # Clean up
        del final_train_data, final_val_data
        gc.collect()

    finally:
        temp_final_train.unlink(missing_ok=True)
        temp_final_val.unlink(missing_ok=True)

    # Calculate total training time
    total_training_time = (time.time() - walk_forward_start_time) / 60  # minutes

    # Prepare aggregated metrics
    aggregated_metrics = {
        # Compatible keys for main() function
        "residual_mse": mean_mse,
        "baseline_brier": final_baseline_brier,
        "training_time_minutes": total_training_time,
        # Walk-forward specific diagnostics
        "walk_forward_mean_mse": mean_mse,
        "walk_forward_std_mse": std_mse,
        "walk_forward_min_mse": min_mse,
        "walk_forward_max_mse": max_mse,
        "walk_forward_mean_ic": mean_ic,
        "walk_forward_std_ic": std_ic,
        "walk_forward_min_ic": min_ic,
        "walk_forward_max_ic": max_ic,
        "walk_forward_significant_ics": sig_ics,
        "n_windows": len(window_metrics),
        "window_metrics": window_metrics,
    }

    # Upload window_metrics as W&B table for detailed analysis
    if wandb_run is not None and wandb_config is not None and wandb_config.get("log_window_tables", False):
        try:
            # Prepare table data
            table_data = []
            for m in window_metrics:
                table_data.append(
                    [
                        m["window"],
                        m["train_start"],
                        m["train_end"],
                        m["val_start"],
                        m["val_end"],
                        m["n_train"],
                        m["n_val"],
                        round(m["baseline_brier"], 6),
                        round(m["residual_mse"], 6),
                        round(m["spearman_ic"], 4),
                        round(m["pearson_ic"], 4),
                        round(m["ic_pvalue"], 6),
                        m["ic_pvalue"] < 0.05,  # Significance flag
                        m["best_iteration"],
                    ]
                )

            # Create W&B table
            columns = [
                "window",
                "train_start",
                "train_end",
                "val_start",
                "val_end",
                "n_train",
                "n_val",
                "baseline_brier",
                "residual_mse",
                "spearman_ic",
                "pearson_ic",
                "ic_pvalue",
                "ic_significant",
                "best_iteration",
            ]
            wandb_table = wandb.Table(columns=columns, data=table_data)
            wandb_run.log({f"{model_name}/window_details_table": wandb_table})
            logger.info(f"  ✓ W&B: Uploaded window_metrics table ({len(window_metrics)} windows)")
        except Exception as e:
            logger.warning(f"  Failed to upload W&B table: {e}")

    return final_model, aggregated_metrics


def main() -> None:
    """Main execution function."""
    # Parse arguments
    parser = argparse.ArgumentParser(description="Train V4 multi-horizon + regime hybrid LightGBM models")
    parser.add_argument(
        "--model",
        type=str,
        choices=[
            "near_low_vol_atm",
            "near_low_vol_otm",
            "near_high_vol_atm",
            "near_high_vol_otm",
            "mid_low_vol_atm",
            "mid_low_vol_otm",
            "mid_high_vol_atm",
            "mid_high_vol_otm",
            "far_low_vol_atm",
            "far_low_vol_otm",
            "far_high_vol_atm",
            "far_high_vol_otm",
            "all",
        ],
        default="all",
        help="Which model to train (default: all 12 models)",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).parent.parent / "config" / "multi_horizon_regime_config.yaml",
        help="Path to multi-horizon + regime config file",
    )
    args = parser.parse_args()

    logger.info("\n" + "=" * 80)
    logger.info("V4 MULTI-HORIZON + REGIME HYBRID TRAINING")
    logger.info("=" * 80)

    # Load configuration
    config = load_config(args.config)

    # Walk-forward validation is MANDATORY for production
    logger.info("Walk-forward validation: ENABLED (mandatory for time series)")

    # Load shared hyperparameters
    model_dir = Path(__file__).parent.parent
    shared_config_path = model_dir / config["training"]["shared_config_path"]
    hyperparameters = load_shared_hyperparameters(shared_config_path)

    # Initialize W&B if enabled
    wandb_run = None
    if config.get("wandb", {}).get("enabled", False) and WANDB_AVAILABLE:
        wandb_run = wandb.init(
            project=config["wandb"]["project"],
            entity=config["wandb"]["entity"],
            config={"multi_horizon_regime": config, "hyperparameters": hyperparameters},
            tags=config["wandb"]["tags"],
            notes=config["wandb"]["notes"],
        )
        logger.info("✓ W&B run initialized")

    # Setup paths
    output_dir = model_dir / Path(config["training"]["output_dir"])
    models_dir = model_dir / Path(config["models"]["output_dir"])

    # Get pipeline-ready features file (V4)
    pipeline_ready_file = model_dir / "data" / "consolidated_features_v4_pipeline_ready.parquet"

    if not pipeline_ready_file.exists():
        msg = f"Pipeline-ready features file not found: {pipeline_ready_file}"
        raise FileNotFoundError(msg)

    logger.info(f"\nLoading pipeline-ready features from: {pipeline_ready_file}")

    # Get features from pipeline-ready file schema
    schema = pl.scan_parquet(pipeline_ready_file).collect_schema()
    features = [col for col in FEATURE_COLS_V4 if col in schema.names()]
    logger.info(f"Using {len(features)} features (out of {len(FEATURE_COLS_V4)} in FEATURE_COLS_V4)")

    if len(features) < len(FEATURE_COLS_V4):
        missing_features = set(FEATURE_COLS_V4) - set(features)
        logger.warning(f"⚠️  {len(missing_features)} features not found in pipeline-ready file:")
        for feat in list(missing_features)[:10]:  # Show first 10
            logger.warning(f"    - {feat}")
        if len(missing_features) > 10:
            logger.warning(f"    ... and {len(missing_features) - 10} more")

    # Define 12 hybrid models (3 temporal buckets × 4 volatility regimes)
    hybrid_models = [
        ("near_low_vol_atm", 0, 300),
        ("near_low_vol_otm", 0, 300),
        ("near_high_vol_atm", 0, 300),
        ("near_high_vol_otm", 0, 300),
        ("mid_low_vol_atm", 300, 600),
        ("mid_low_vol_otm", 300, 600),
        ("mid_high_vol_atm", 300, 600),
        ("mid_high_vol_otm", 300, 600),
        ("far_low_vol_atm", 600, 900),
        ("far_low_vol_otm", 600, 900),
        ("far_high_vol_atm", 600, 900),
        ("far_high_vol_otm", 600, 900),
    ]

    # Determine which models to train
    if args.model == "all":
        models_to_train = hybrid_models
    else:
        # Determine time range based on temporal bucket in model name
        if "near" in args.model:
            time_min, time_max = 0, 300
        elif "mid" in args.model:
            time_min, time_max = 300, 600
        elif "far" in args.model:
            time_min, time_max = 600, 900
        else:
            raise ValueError(f"Unknown temporal bucket in model name: {args.model}")
        models_to_train = [(args.model, time_min, time_max)]

    logger.info(f"\nTraining {len(models_to_train)} model(s): {[m[0] for m in models_to_train]}")

    # Train each model
    all_metrics = {}
    total_models = len(models_to_train)
    models_start_time = time.time()

    for model_idx, (model_name, time_min, time_max) in enumerate(models_to_train, 1):
        model_config = config["hybrid_models"][model_name]

        # Overall progress tracking
        overall_progress = (model_idx - 1) / total_models * 100
        if model_idx > 1:
            elapsed = time.time() - models_start_time
            avg_time_per_model = elapsed / (model_idx - 1)
            remaining_models = total_models - model_idx + 1
            eta_seconds = avg_time_per_model * remaining_models
            eta_hours = int(eta_seconds // 3600)
            eta_mins = int((eta_seconds % 3600) // 60)
            eta_str = f"{eta_hours}h {eta_mins}m"
        else:
            eta_str = "estimating..."

        logger.info(f"\n{'#' * 80}")
        logger.info(f"# MODEL {model_idx}/{total_models}: {model_name} [{overall_progress:.1f}%] | ETA: {eta_str}")
        logger.info(f"# Bucket: {model_config['bucket']}")
        logger.info(f"# Regime: {model_config['regime']}")
        logger.info(f"# Time range: {time_min}-{time_max}s")
        logger.info(f"# Expected samples: {model_config['expected_samples']:,}")
        logger.info(f"{'#' * 80}")

        # Train model with walk-forward validation (mandatory)
        walk_forward_config = config["walk_forward_validation"]

        # Stratify pipeline-ready dataset by combined_regime
        stratified_file = stratify_data_by_hybrid(pipeline_ready_file, output_dir, model_name, time_min, time_max)

        model, metrics = train_model_walk_forward(
            model_name,
            model_config,
            stratified_file,
            hyperparameters,
            features,
            models_dir,
            walk_forward_config,
            wandb_run,
            config.get("wandb", {}),
        )

        all_metrics[model_name] = metrics

        # Log to W&B
        if wandb_run is not None:
            wandb_run.log(
                {
                    f"{model_name}/residual_mse": metrics["residual_mse"],
                    f"{model_name}/baseline_brier": metrics["baseline_brier"],
                    f"{model_name}/training_time_min": metrics["training_time_minutes"],
                }
            )

        del model
        gc.collect()

    # Summary
    logger.info("\n" + "=" * 80)
    logger.info("TRAINING SUMMARY")
    logger.info("=" * 80)

    for model_name, _, _ in models_to_train:
        metrics = all_metrics[model_name]
        model_config = config["hybrid_models"][model_name]

        logger.info(
            f"✓ {model_name:25s}: "
            f"Residual MSE = {metrics['residual_mse']:.6f} "
            f"(Baseline Brier = {metrics['baseline_brier']:.6f})"
        )

    logger.info("\nNote: Brier improvement will be computed in evaluation phase (evaluate_multi_horizon_v4.py).")

    # Finish W&B
    if wandb_run is not None:
        wandb_run.finish()

    logger.info("\n✓ V4 multi-horizon + regime hybrid training complete!")
    logger.info(f"Models saved to: {models_dir}")
    logger.info(f"Data saved to: {output_dir}")


if __name__ == "__main__":
    main()
