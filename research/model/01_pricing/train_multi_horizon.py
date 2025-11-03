#!/usr/bin/env python3
"""
Multi-Horizon LightGBM Training Script
=======================================

Train separate LightGBM models for different time-to-expiry regimes:
- Near (0-5 min): Microstructure-dominated
- Mid (5-10 min): Balanced regime
- Far (10-15 min): Diffusion-dominated

Rationale:
- Single model shows 30.6pp performance degradation across horizons
- Feature importance shifts across time regimes
- Expected +3-5pp improvement with specialized models

Usage:
  # Train all buckets (always uses walk-forward validation)
  uv run python train_multi_horizon.py

  # Train specific bucket
  uv run python train_multi_horizon.py --bucket near

  # With custom config
  uv run python train_multi_horizon.py --config config/custom_config.yaml

Note: Walk-forward validation is MANDATORY for production use.
      This ensures zero data leakage for time series forecasting.

Author: BT Research Team
Date: 2025-11-02
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

# Import from existing modules
from lightgbm_memory_optimized import (
    FEATURE_COLS,
    MemoryMonitor,
    calculate_residual_metrics,  # noqa: F401
    create_lgb_dataset_from_parquet,
    evaluate_brier_score,  # noqa: F401
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


def load_config(config_path: Path) -> dict[str, Any]:
    """Load multi-horizon configuration."""
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


def stratify_data_by_time(
    input_file: Path,
    output_dir: Path,
    bucket_name: str,
    time_min: int,
    time_max: int,
) -> Path:
    """
    Stratify dataset by time_remaining bucket.

    Args:
        input_file: Source parquet file
        output_dir: Output directory for stratified file
        bucket_name: Bucket identifier (near/mid/far)
        time_min: Minimum time_remaining (seconds)
        time_max: Maximum time_remaining (seconds)

    Returns:
        Path to stratified output file
    """
    logger.info(f"\nStratifying {bucket_name} bucket ({time_min}-{time_max}s)...")
    logger.info(f"  Source: {input_file}")

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    # Determine output filename
    # Extract train/val/test from filename like "train_features_lgb.parquet"
    split_name = input_file.stem.split("_")[0]  # First part: train/val/test
    output_file = output_dir / f"{bucket_name}_{split_name}.parquet"

    # Check if already exists
    if output_file.exists():
        logger.info(f"  ✓ Already exists: {output_file}")
        # Get row count
        row_count = pl.scan_parquet(output_file).select(pl.len()).collect().item()
        logger.info(f"  Rows: {row_count:,}")
        return output_file

    # Lazy load and filter
    logger.info(f"  Filtering time_remaining: [{time_min}, {time_max})")

    df = pl.scan_parquet(input_file).filter(
        (pl.col("time_remaining") >= time_min) & (pl.col("time_remaining") < time_max)
    )

    # Stream to output
    df.sink_parquet(output_file, compression="snappy", statistics=True)

    # Get row count
    row_count = pl.scan_parquet(output_file).select(pl.len()).collect().item()

    logger.info(f"  ✓ Created: {output_file}")
    logger.info(f"  Rows: {row_count:,}")

    gc.collect()

    return output_file


def train_bucket_model(
    bucket_name: str,
    bucket_config: dict[str, Any],
    train_file: Path,
    val_file: Path,
    hyperparameters: dict[str, Any],
    features: list[str],
    output_dir: Path,
    wandb_run: Any | None = None,
) -> tuple[lgb.Booster, dict[str, float]]:
    """
    Train LightGBM model for a specific time bucket.

    Args:
        bucket_name: Bucket identifier (near/mid/far)
        bucket_config: Bucket-specific configuration
        train_file: Training data file
        val_file: Validation data file
        hyperparameters: LightGBM hyperparameters
        features: Feature columns to use
        output_dir: Output directory for model
        wandb_run: W&B run (optional)

    Returns:
        Tuple of (trained model, performance metrics)
    """
    logger.info(f"\n{'=' * 80}")
    logger.info(f"TRAINING {bucket_config['name'].upper()}")
    logger.info(f"{'=' * 80}")

    monitor = MemoryMonitor()
    start_time = time.time()

    # Load datasets
    logger.info("Loading datasets...")

    # Get row counts from parquet files directly (faster than Dataset.num_data())
    n_train = pl.scan_parquet(train_file).select(pl.len()).collect().item()
    n_val = pl.scan_parquet(val_file).select(pl.len()).collect().item()

    train_data, baseline_brier_train = create_lgb_dataset_from_parquet(str(train_file), features, free_raw_data=True)
    val_data, baseline_brier_val = create_lgb_dataset_from_parquet(
        str(val_file), features, reference=train_data, free_raw_data=True
    )

    # Convert baseline Brier to float if it's an array
    if hasattr(baseline_brier_val, "__len__"):
        baseline_brier_val = float(np.mean(baseline_brier_val))

    logger.info(f"Training samples:    {n_train:,}")
    logger.info(f"Validation samples:  {n_val:,}")
    logger.info(f"Baseline Brier:      {baseline_brier_val:.6f}")
    logger.info(f"Current improvement: {bucket_config['current_improvement']:.1f}%")
    logger.info(f"Target improvement:  {bucket_config['target_improvement']}")

    monitor.check_memory("After loading datasets")

    # Train model
    logger.info("\nTraining LightGBM model...")

    callbacks = [
        lgb.early_stopping(hyperparameters.get("early_stopping_rounds", 50)),
        lgb.log_evaluation(25),
    ]

    # Add W&B callback if available
    if wandb_run is not None:
        try:
            from wandb.integration.lightgbm import wandb_callback

            callbacks.append(wandb_callback())
            logger.info("✓ W&B training curve logging enabled")
        except ImportError:
            pass

    model = lgb.train(
        hyperparameters,
        train_data,
        num_boost_round=hyperparameters.get("n_estimators", 1000),
        valid_sets=[train_data, val_data],
        valid_names=["train", "val"],
        callbacks=callbacks,
    )

    # Calculate metrics
    best_score = model.best_score
    val_metrics = best_score.get("val", {})
    residual_mse = val_metrics.get("l2", 0.0)
    residual_rmse = np.sqrt(residual_mse)
    residual_mae = val_metrics.get("mae", 0.0)

    training_time = (time.time() - start_time) / 60

    # Log results
    logger.info(f"\n{bucket_config['name']} Results:")
    logger.info(f"  Baseline Brier:      {baseline_brier_val:.6f}")
    logger.info(f"  Residual MSE:        {residual_mse:.6f}")
    logger.info(f"  Residual RMSE:       {residual_rmse:.6f}")
    logger.info(f"  Residual MAE:        {residual_mae:.6f}")
    logger.info(f"  Training Time:       {training_time:.2f} min")
    logger.info(f"  Best Iteration:      {model.best_iteration}")
    logger.info("\n  Note: Brier improvement will be computed in evaluation phase.")

    # Log feature importance
    importance = model.feature_importance(importance_type="gain")
    feature_names = model.feature_name()
    top_features = sorted(zip(feature_names, importance), key=lambda x: x[1], reverse=True)[:20]

    logger.info(f"\nTop 20 features for {bucket_name} bucket:")
    for i, (feat, imp) in enumerate(top_features, 1):
        logger.info(f"  {i:2d}. {feat:40s}: {imp:10.2f}")

    # Log model artifacts and feature importance to W&B
    if wandb_run is not None:
        try:
            from wandb.integration.lightgbm import log_summary

            log_summary(model, save_model_checkpoint=True)
            logger.info("✓ W&B model summary and artifacts logged")
        except ImportError:
            logger.warning("wandb.integration.lightgbm not available - skipping model summary")

    # Save model
    output_dir.mkdir(parents=True, exist_ok=True)
    model_file = output_dir / f"lightgbm_{bucket_name}.txt"
    model.save_model(str(model_file))
    logger.info(f"\n✓ Saved model to {model_file}")

    # Save config
    config_file = output_dir / f"config_{bucket_name}.yaml"
    bucket_full_config = {
        "bucket": bucket_config,
        "hyperparameters": hyperparameters,
        "performance": {
            "baseline_brier": baseline_brier_val,
            "residual_mse": residual_mse,
            "residual_rmse": residual_rmse,
            "residual_mae": residual_mae,
            "training_time_minutes": training_time,
            "best_iteration": model.best_iteration,
        },
        "data": {
            "train_samples": n_train,
            "val_samples": n_val,
        },
        "top_features": [{"name": f, "importance": float(imp)} for f, imp in top_features],
    }

    with open(config_file, "w") as f:
        yaml.dump(bucket_full_config, f, default_flow_style=False, sort_keys=False)

    logger.info(f"✓ Saved config to {config_file}")

    # Clean up
    del train_data, val_data
    gc.collect()
    monitor.check_memory("After training")

    metrics = {
        "baseline_brier": baseline_brier_val,
        "residual_mse": residual_mse,
        "residual_rmse": residual_rmse,
        "residual_mae": residual_mae,
        "training_time_minutes": training_time,
        "best_iteration": model.best_iteration,
        "n_train": n_train,
        "n_val": n_val,
    }

    return model, metrics


def train_bucket_walk_forward(
    bucket_name: str,
    bucket_config: dict[str, Any],
    data_file: Path,
    hyperparameters: dict[str, Any],
    features: list[str],
    output_dir: Path,
    walk_forward_config: dict[str, Any],
    wandb_run: Any | None = None,
) -> tuple[lgb.Booster, dict[str, Any]]:
    """
    Train LightGBM model using walk-forward validation.

    True walk-forward validation: both training and validation periods advance chronologically.
    This simulates real-world deployment where you always predict the future based on the past.

    Args:
        bucket_name: Bucket identifier (near/mid/far)
        bucket_config: Bucket-specific configuration
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
    logger.info(f"WALK-FORWARD VALIDATION: {bucket_config['name'].upper()}")
    logger.info(f"{'=' * 80}")

    train_months = walk_forward_config["train_months"]
    val_months = walk_forward_config["val_months"]
    step_months = walk_forward_config["step_months"]
    holdout_months = walk_forward_config["holdout_months"]

    logger.info(f"Training window: {train_months} months")
    logger.info(f"Validation window: {val_months} months")
    logger.info(f"Step size: {step_months} month(s)")
    logger.info(f"Holdout test: last {holdout_months} months")

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

    holdout_start = max_date - relativedelta(months=holdout_months - 1)
    walk_forward_end = holdout_start - relativedelta(days=1)

    logger.info(f"Full data range: {min_date} to {max_date}")
    logger.info(f"Walk-forward range: {min_date} to {walk_forward_end}")
    logger.info(f"Holdout test: {holdout_start} to {max_date}")

    # Generate walk-forward windows
    windows: list[tuple[date, date, date, date]] = []
    current_start = min_date

    while True:
        train_end = current_start + relativedelta(months=train_months) - relativedelta(days=1)
        val_start = train_end + relativedelta(days=1)
        val_end = val_start + relativedelta(months=val_months) - relativedelta(days=1)

        # Stop if validation period would overlap with holdout
        if val_end >= holdout_start:
            break

        windows.append((current_start, train_end, val_start, val_end))

        # Step forward
        current_start += relativedelta(months=step_months)

    logger.info(f"Generated {len(windows)} walk-forward windows")

    # Train and validate on each window
    window_metrics: list[dict[str, Any]] = []
    walk_forward_start_time = time.time()

    for window_idx, (train_start, train_end, val_start, val_end) in enumerate(windows, 1):
        logger.info(f"\n{'─' * 80}")
        logger.info(f"Window {window_idx}/{len(windows)}")
        logger.info(f"  Train: {train_start} to {train_end}")
        logger.info(f"  Val:   {val_start} to {val_end}")
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
            continue

        # Write temporary files
        temp_train = output_dir / f"temp_{bucket_name}_window{window_idx}_train.parquet"
        temp_val = output_dir / f"temp_{bucket_name}_window{window_idx}_val.parquet"

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

            logger.info(f"  Baseline Brier: {baseline_brier_val:.6f}")
            logger.info(f"  Residual MSE:   {residual_mse:.6f}")

            window_metrics.append(
                {
                    "window": window_idx,
                    "train_start": train_start.isoformat(),
                    "train_end": train_end.isoformat(),
                    "val_start": val_start.isoformat(),
                    "val_end": val_end.isoformat(),
                    "n_train": n_train,
                    "n_val": n_val,
                    "baseline_brier": baseline_brier_val,
                    "residual_mse": residual_mse,
                    "best_iteration": model.best_iteration,
                }
            )

            # Clean up
            del train_data, val_data, model
            gc.collect()

        except Exception as e:
            logger.error(f"Error training window {window_idx}: {e}")
            continue

        finally:
            # Remove temporary files
            temp_train.unlink(missing_ok=True)
            temp_val.unlink(missing_ok=True)

        monitor.check_memory(f"After window {window_idx}")

    if len(window_metrics) == 0:
        msg = "No windows produced valid models"
        raise ValueError(msg)

    # Aggregate validation metrics
    logger.info(f"\n{'=' * 80}")
    logger.info("WALK-FORWARD VALIDATION RESULTS")
    logger.info(f"{'=' * 80}")

    residual_mses = [m["residual_mse"] for m in window_metrics]
    mean_mse = np.mean(residual_mses)
    std_mse = np.std(residual_mses)
    min_mse = np.min(residual_mses)
    max_mse = np.max(residual_mses)

    logger.info(f"Average Residual MSE: {mean_mse:.6f} ± {std_mse:.6f}")
    logger.info(f"Range: {min_mse:.6f} to {max_mse:.6f}")
    logger.info("\nWindow-by-window performance:")
    for m in window_metrics:
        logger.info(f"  Window {m['window']}: Residual MSE = {m['residual_mse']:.6f}")
    logger.info("\nNote: Brier improvement will be computed in evaluation phase.")

    # Train final production model on all walk-forward data (excluding holdout)
    logger.info(f"\n{'=' * 80}")
    logger.info("TRAINING FINAL MODEL ON ALL WALK-FORWARD DATA")
    logger.info(f"{'=' * 80}")

    df_final_train = df_lazy.filter((pl.col("date") >= min_date) & (pl.col("date") <= walk_forward_end))

    # Split walk-forward data into 80/20 for final training
    final_train_end = walk_forward_end - relativedelta(months=int(train_months * 0.2))

    df_final_train_split = df_final_train.filter(pl.col("date") <= final_train_end)
    df_final_val_split = df_final_train.filter(pl.col("date") > final_train_end)

    temp_final_train = output_dir / f"temp_{bucket_name}_final_train.parquet"
    temp_final_val = output_dir / f"temp_{bucket_name}_final_val.parquet"

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
        model_file = output_dir / f"lightgbm_{bucket_name}.txt"
        final_model.save_model(str(model_file))
        logger.info(f"✓ Saved final model to {model_file}")

        # Save config (matching structure from train_bucket_model)
        config_file = output_dir / f"config_{bucket_name}.yaml"

        # Get feature importance from final model
        importance = final_model.feature_importance(importance_type="gain")
        feature_names = final_model.feature_name()
        top_features = sorted(zip(feature_names, importance), key=lambda x: x[1], reverse=True)[:20]

        # Count total samples across all windows
        total_train_samples = sum(m.get("n_train", 0) for m in window_metrics)
        total_val_samples = sum(m.get("n_val", 0) for m in window_metrics)

        bucket_full_config = {
            "bucket": bucket_config,
            "hyperparameters": hyperparameters,
            "performance": {
                "baseline_brier": final_baseline_brier,
                "residual_mse": mean_mse,  # Use walk-forward mean
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
                "window_details": window_metrics,
            },
            "data": {
                "train_samples": total_train_samples,
                "val_samples": total_val_samples,
            },
            "top_features": [{"name": f, "importance": float(imp)} for f, imp in top_features],
        }

        with open(config_file, "w") as f:
            yaml.dump(bucket_full_config, f, default_flow_style=False, sort_keys=False)

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
        "residual_mse": mean_mse,  # Use mean MSE across windows
        "baseline_brier": final_baseline_brier,
        "training_time_minutes": total_training_time,
        # Walk-forward specific diagnostics
        "walk_forward_mean_mse": mean_mse,
        "walk_forward_std_mse": std_mse,
        "walk_forward_min_mse": min_mse,
        "walk_forward_max_mse": max_mse,
        "n_windows": len(window_metrics),
        "window_metrics": window_metrics,
    }

    return final_model, aggregated_metrics


def main() -> None:
    """Main execution function."""
    # Parse arguments
    parser = argparse.ArgumentParser(description="Train multi-horizon LightGBM models")
    parser.add_argument(
        "--bucket",
        type=str,
        choices=["near", "mid", "far", "all"],
        default="all",
        help="Which bucket to train (default: all)",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).parent / "config" / "multi_horizon_config.yaml",
        help="Path to multi-horizon config file",
    )
    args = parser.parse_args()

    logger.info("\n" + "=" * 80)
    logger.info("MULTI-HORIZON LIGHTGBM TRAINING")
    logger.info("=" * 80)

    # Load configuration
    config = load_config(args.config)

    # Walk-forward validation is MANDATORY for production (no data leakage)
    logger.info("Walk-forward validation: ENABLED (mandatory for time series)")
    if not config.get("walk_forward_validation", {}).get("enabled", True):
        logger.warning("⚠️  Config has walk_forward_validation.enabled=false, but this is IGNORED")
        logger.warning("    Walk-forward is MANDATORY to prevent data leakage in time series")

    # Load shared hyperparameters
    model_dir = Path(__file__).parent.parent
    shared_config_path = model_dir / config["training"]["shared_config_path"]
    hyperparameters = load_shared_hyperparameters(shared_config_path)

    # Initialize W&B if enabled
    wandb_run = None
    if config["wandb"]["enabled"] and WANDB_AVAILABLE:
        wandb_run = wandb.init(
            project=config["wandb"]["project"],
            entity=config["wandb"]["entity"],
            config={"multi_horizon": config, "hyperparameters": hyperparameters},
            tags=config["wandb"]["tags"],
            notes=config["wandb"]["notes"],
        )
        logger.info("✓ W&B run initialized")

    # Setup paths
    data_dir = model_dir / "results"
    output_dir = model_dir / Path(config["data"]["output_dir"])
    models_dir = model_dir / Path(config["models"]["output_dir"])

    # Get features
    train_file_full = data_dir / Path(config["data"]["source_files"]["train"]).name
    schema = pl.scan_parquet(train_file_full).collect_schema()
    features = [col for col in FEATURE_COLS if col in schema.names()]
    logger.info(f"\nUsing {len(features)} features")

    # Determine which buckets to train
    buckets_to_train = ["near", "mid", "far"] if args.bucket == "all" else [args.bucket]

    logger.info(f"\nTraining buckets: {', '.join(buckets_to_train)}")

    # Train each bucket
    all_metrics = {}

    for bucket_name in buckets_to_train:
        bucket_config = config["buckets"][bucket_name]

        logger.info(f"\n{'#' * 80}")
        logger.info(f"# BUCKET: {bucket_config['name']}")
        logger.info(f"# Time range: {bucket_config['time_min']}-{bucket_config['time_max']}s")
        logger.info(f"# Expected samples: {bucket_config['expected_samples']:,}")
        logger.info(f"{'#' * 80}")

        # Train model with walk-forward validation (mandatory)
        walk_forward_config = config["walk_forward_validation"]

        # Stratify FULL dataset (train + val + test combined) by time bucket
        full_data_file = stratify_data_by_time(
            data_dir / Path(config["data"]["source_files"]["train"]).name,
            output_dir,
            bucket_name,
            bucket_config["time_min"],
            bucket_config["time_max"],
        )

        model, metrics = train_bucket_walk_forward(
            bucket_name,
            bucket_config,
            full_data_file,
            hyperparameters,
            features,
            models_dir,
            walk_forward_config,
            wandb_run,
        )

        all_metrics[bucket_name] = metrics

        # Log to W&B
        if wandb_run is not None:
            wandb_run.log(
                {
                    f"{bucket_name}/residual_mse": metrics["residual_mse"],
                    f"{bucket_name}/baseline_brier": metrics["baseline_brier"],
                    f"{bucket_name}/training_time_min": metrics["training_time_minutes"],
                }
            )

        del model
        gc.collect()

    # Summary
    logger.info("\n" + "=" * 80)
    logger.info("TRAINING SUMMARY")
    logger.info("=" * 80)

    for bucket_name in buckets_to_train:
        metrics = all_metrics[bucket_name]
        bucket_config = config["buckets"][bucket_name]

        logger.info(
            f"✓ {bucket_config['name']:25s}: "
            f"Residual MSE = {metrics['residual_mse']:.6f} "
            f"(Baseline Brier = {metrics['baseline_brier']:.6f})"
        )

    logger.info("\nNote: Brier improvement will be computed in Phase 1 Step 2 (Evaluation).")
    logger.info("      Run evaluate_multi_horizon.py to validate performance against targets.")

    # Finish W&B
    if wandb_run is not None:
        wandb_run.finish()

    logger.info("\n✓ Multi-horizon training complete!")
    logger.info(f"Models saved to: {models_dir}")
    logger.info(f"Data saved to: {output_dir}")


if __name__ == "__main__":
    main()
