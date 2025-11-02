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
  # Train all buckets
  uv run python train_multi_horizon.py

  # Train specific bucket
  uv run python train_multi_horizon.py --bucket near

  # With custom config
  uv run python train_multi_horizon.py --config config/custom_config.yaml

Author: BT Research Team
Date: 2025-11-02
"""

from __future__ import annotations

import argparse
import gc
import logging
import sys
import time
from datetime import datetime, timedelta
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

    model_brier = baseline_brier_val - residual_mse
    brier_improvement_pct = (residual_mse / baseline_brier_val) * 100

    training_time = (time.time() - start_time) / 60

    # Log results
    logger.info(f"\n{bucket_config['name']} Results:")
    logger.info(f"  Baseline Brier:      {baseline_brier_val:.6f}")
    logger.info(f"  Model Brier:         {model_brier:.6f}")
    logger.info(f"  Residual MSE:        {residual_mse:.6f}")
    logger.info(f"  Residual RMSE:       {residual_rmse:.6f}")
    logger.info(f"  Residual MAE:        {residual_mae:.6f}")
    logger.info(f"  Improvement:         {brier_improvement_pct:.3f}%")
    logger.info(f"  Training Time:       {training_time:.2f} min")
    logger.info(f"  Best Iteration:      {model.best_iteration}")

    # Target check
    target_min = float(bucket_config["target_improvement"].split("-")[0])
    if brier_improvement_pct >= target_min:
        logger.info(f"  ✓ Target achieved ({target_min}%+)")
    else:
        logger.warning(f"  ⚠ Below target (expected {target_min}%+)")

    # Log feature importance
    importance = model.feature_importance(importance_type="gain")
    feature_names = model.feature_name()
    top_features = sorted(zip(feature_names, importance), key=lambda x: x[1], reverse=True)[:20]

    logger.info(f"\nTop 20 features for {bucket_name} bucket:")
    for i, (feat, imp) in enumerate(top_features, 1):
        logger.info(f"  {i:2d}. {feat:40s}: {imp:10.2f}")

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
            "model_brier": model_brier,
            "residual_mse": residual_mse,
            "residual_rmse": residual_rmse,
            "residual_mae": residual_mae,
            "brier_improvement_pct": brier_improvement_pct,
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
        "model_brier": model_brier,
        "residual_mse": residual_mse,
        "residual_rmse": residual_rmse,
        "residual_mae": residual_mae,
        "brier_improvement_pct": brier_improvement_pct,
        "training_time_minutes": training_time,
        "best_iteration": model.best_iteration,
        "n_train": n_train,
        "n_val": n_val,
    }

    return model, metrics


def train_bucket_with_rolling_window(
    bucket_name: str,
    bucket_config: dict[str, Any],
    train_file: Path,
    val_file: Path,
    hyperparameters: dict[str, Any],
    features: list[str],
    output_dir: Path,
    rolling_config: dict[str, Any],
    wandb_run: Any | None = None,
) -> tuple[lgb.Booster | list[lgb.Booster], dict[str, float]]:
    """
    Train LightGBM model for a specific time bucket using rolling windows.

    This function trains multiple models on successive temporal windows, useful for:
    - Recency bias (focus on recent market regimes)
    - Regime detection (identify when model performance changes)
    - Walk-forward validation (natural temporal cross-validation)

    Args:
        bucket_name: Bucket identifier (near/mid/far)
        bucket_config: Bucket-specific configuration
        train_file: Training data file
        val_file: Validation data file
        hyperparameters: LightGBM hyperparameters
        features: Feature columns to use
        output_dir: Output directory for models
        rolling_config: Rolling window configuration
        wandb_run: W&B run (optional)

    Returns:
        Tuple of (model(s), aggregated metrics)
        - If aggregation="latest": single model from most recent window
        - If aggregation="ensemble": list of all window models
    """
    logger.info(f"\n{'=' * 80}")
    logger.info(f"ROLLING WINDOW TRAINING: {bucket_config['name'].upper()}")
    logger.info(f"{'=' * 80}")

    window_months = rolling_config["window_months"]
    step_months = rolling_config["step_months"]
    min_samples = rolling_config["min_samples"]
    aggregation = rolling_config["aggregation"]
    save_all = rolling_config["save_all_models"]
    log_window_metrics = rolling_config["log_window_metrics"]

    logger.info(f"Window size: {window_months} months")
    logger.info(f"Step size: {step_months} months")
    logger.info(f"Minimum samples: {min_samples:,}")
    logger.info(f"Aggregation: {aggregation}")

    # Load full dataset to get date range
    logger.info("\nAnalyzing data date range...")
    df_lazy = pl.scan_parquet(train_file)

    # Get min/max dates from data
    date_stats = df_lazy.select(
        [
            pl.col("timestamp").min().alias("min_ts"),
            pl.col("timestamp").max().alias("max_ts"),
        ]
    ).collect()

    # Convert Unix timestamps to datetime
    min_ts = int(date_stats["min_ts"][0])
    max_ts = int(date_stats["max_ts"][0])
    start_date = datetime.fromtimestamp(min_ts)
    end_date = datetime.fromtimestamp(max_ts)

    # Override with config if specified
    if rolling_config["start_date"] is not None:
        start_date = datetime.fromisoformat(rolling_config["start_date"])
    if rolling_config["end_date"] is not None:
        end_date = datetime.fromisoformat(rolling_config["end_date"])

    logger.info(f"Data range: {start_date.date()} to {end_date.date()}")

    # Generate rolling windows
    windows: list[tuple[datetime, datetime]] = []
    current_start = start_date

    while current_start < end_date:
        current_end = current_start + timedelta(days=window_months * 30)  # Approx months
        if current_end > end_date:
            current_end = end_date

        windows.append((current_start, current_end))

        # Step forward
        current_start += timedelta(days=step_months * 30)

    logger.info(f"Generated {len(windows)} rolling windows")

    # Train model for each window
    window_models: list[lgb.Booster] = []
    window_metrics: list[dict[str, Any]] = []

    for window_idx, (win_start, win_end) in enumerate(windows, 1):
        logger.info(f"\n{'─' * 80}")
        logger.info(f"Window {window_idx}/{len(windows)}: {win_start.date()} to {win_end.date()}")
        logger.info(f"{'─' * 80}")

        # Filter training data for this window
        win_start_ts = int(win_start.timestamp())
        win_end_ts = int(win_end.timestamp())

        # Create temporary windowed dataset
        windowed_train = output_dir / f"temp_{bucket_name}_window{window_idx}_train.parquet"

        df_window = df_lazy.filter((pl.col("timestamp") >= win_start_ts) & (pl.col("timestamp") < win_end_ts))

        # Check sample count
        n_samples = df_window.select(pl.len()).collect().item()

        if n_samples < min_samples:
            logger.warning(
                f"⚠ Window {window_idx} has only {n_samples:,} samples (< {min_samples:,} minimum), skipping"
            )
            continue

        logger.info(f"Window {window_idx} training samples: {n_samples:,}")

        # Write windowed data to temporary file
        df_window.sink_parquet(str(windowed_train))

        # Train on this window
        try:
            # Load windowed datasets
            n_train_win = pl.scan_parquet(windowed_train).select(pl.len()).collect().item()

            train_data_win, _ = create_lgb_dataset_from_parquet(str(windowed_train), features, free_raw_data=True)
            val_data_win, baseline_brier_val_win = create_lgb_dataset_from_parquet(
                str(val_file), features, reference=train_data_win, free_raw_data=True
            )

            if hasattr(baseline_brier_val_win, "__len__"):
                baseline_brier_val_win = float(np.mean(baseline_brier_val_win))

            # Train model on window
            callbacks = [
                lgb.early_stopping(hyperparameters.get("early_stopping_rounds", 50)),
                lgb.log_evaluation(50),  # Less verbose for windows
            ]

            model_win = lgb.train(
                hyperparameters,
                train_data_win,
                num_boost_round=hyperparameters.get("n_estimators", 1000),
                valid_sets=[train_data_win, val_data_win],
                valid_names=["train", "val"],
                callbacks=callbacks,
            )

            # Calculate window metrics
            best_score_win = model_win.best_score
            val_metrics_win = best_score_win.get("val", {})
            residual_mse_win = val_metrics_win.get("l2", 0.0)
            residual_rmse_win = np.sqrt(residual_mse_win)

            model_brier_win = baseline_brier_val_win - residual_mse_win
            brier_improvement_pct_win = (residual_mse_win / baseline_brier_val_win) * 100

            if log_window_metrics:
                logger.info(f"  Baseline Brier: {baseline_brier_val_win:.6f}")
                logger.info(f"  Model Brier:    {model_brier_win:.6f}")
                logger.info(f"  Improvement:    {brier_improvement_pct_win:.2f}%")
                logger.info(f"  Best Iteration: {model_win.best_iteration}")

            # Save window model if requested
            if save_all:
                win_model_file = output_dir / f"lightgbm_{bucket_name}_window{window_idx}.txt"
                model_win.save_model(str(win_model_file))
                logger.info(f"  ✓ Saved window model to {win_model_file}")

            window_models.append(model_win)
            window_metrics.append(
                {
                    "window": window_idx,
                    "start_date": win_start.isoformat(),
                    "end_date": win_end.isoformat(),
                    "n_train": n_train_win,
                    "baseline_brier": baseline_brier_val_win,
                    "model_brier": model_brier_win,
                    "residual_mse": residual_mse_win,
                    "residual_rmse": residual_rmse_win,
                    "brier_improvement_pct": brier_improvement_pct_win,
                    "best_iteration": model_win.best_iteration,
                }
            )

            # Clean up
            del train_data_win, val_data_win
            gc.collect()

        except Exception as e:
            logger.error(f"Error training window {window_idx}: {e}")
            continue

        finally:
            # Remove temporary windowed file
            if windowed_train.exists():
                windowed_train.unlink()

    # Aggregate models based on strategy
    if len(window_models) == 0:
        msg = "No windows produced valid models"
        raise ValueError(msg)

    logger.info(f"\n{'=' * 80}")
    logger.info(f"ROLLING WINDOW AGGREGATION: {aggregation.upper()}")
    logger.info(f"{'=' * 80}")

    if aggregation == "latest":
        # Use most recent window model
        final_model = window_models[-1]
        final_metrics = window_metrics[-1]
        logger.info(f"Using latest window model (window {len(window_models)})")

        # Save final model
        model_file = output_dir / f"lightgbm_{bucket_name}.txt"
        final_model.save_model(str(model_file))
        logger.info(f"✓ Saved final model to {model_file}")

        return final_model, final_metrics

    elif aggregation == "ensemble":
        # Return all models for ensemble prediction
        logger.info(f"Returning ensemble of {len(window_models)} models")

        # Calculate average metrics
        avg_metrics = {
            "n_windows": len(window_models),
            "baseline_brier": np.mean([m["baseline_brier"] for m in window_metrics]),
            "model_brier": np.mean([m["model_brier"] for m in window_metrics]),
            "residual_mse": np.mean([m["residual_mse"] for m in window_metrics]),
            "residual_rmse": np.mean([m["residual_rmse"] for m in window_metrics]),
            "brier_improvement_pct": np.mean([m["brier_improvement_pct"] for m in window_metrics]),
            "avg_best_iteration": np.mean([m["best_iteration"] for m in window_metrics]),
        }

        logger.info(f"  Average baseline Brier: {avg_metrics['baseline_brier']:.6f}")
        logger.info(f"  Average model Brier:    {avg_metrics['model_brier']:.6f}")
        logger.info(f"  Average improvement:    {avg_metrics['brier_improvement_pct']:.2f}%")

        # Save ensemble metadata
        ensemble_config = {
            "aggregation": "ensemble",
            "n_windows": len(window_models),
            "window_metrics": window_metrics,
            "avg_metrics": avg_metrics,
        }

        config_file = output_dir / f"config_{bucket_name}_ensemble.yaml"
        with open(config_file, "w") as f:
            yaml.dump(ensemble_config, f, default_flow_style=False, sort_keys=False)

        logger.info(f"✓ Saved ensemble config to {config_file}")

        return window_models, avg_metrics

    else:
        msg = f"Unknown aggregation strategy: {aggregation}"
        raise ValueError(msg)


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
    parser.add_argument(
        "--rolling-window",
        action="store_true",
        help="Enable rolling window training (overrides config)",
    )
    parser.add_argument(
        "--no-rolling-window",
        action="store_true",
        help="Disable rolling window training (overrides config)",
    )
    args = parser.parse_args()

    logger.info("\n" + "=" * 80)
    logger.info("MULTI-HORIZON LIGHTGBM TRAINING")
    logger.info("=" * 80)

    # Load configuration
    config = load_config(args.config)

    # Determine rolling window mode (CLI overrides config)
    use_rolling_window = config.get("rolling_window", {}).get("enabled", False)
    if args.rolling_window:
        use_rolling_window = True
        logger.info("Rolling window training ENABLED (CLI override)")
    elif args.no_rolling_window:
        use_rolling_window = False
        logger.info("Rolling window training DISABLED (CLI override)")
    elif use_rolling_window:
        logger.info("Rolling window training ENABLED (config)")
    else:
        logger.info("Rolling window training DISABLED (config)")

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

        # Stratify data for this bucket
        train_file = stratify_data_by_time(
            data_dir / Path(config["data"]["source_files"]["train"]).name,
            output_dir,
            bucket_name,
            bucket_config["time_min"],
            bucket_config["time_max"],
        )

        val_file = stratify_data_by_time(
            data_dir / Path(config["data"]["source_files"]["val"]).name,
            output_dir,
            bucket_name,
            bucket_config["time_min"],
            bucket_config["time_max"],
        )

        # Train model (choose strategy based on config)
        if use_rolling_window:
            rolling_config = config["rolling_window"]
            model, metrics = train_bucket_with_rolling_window(
                bucket_name,
                bucket_config,
                train_file,
                val_file,
                hyperparameters,
                features,
                models_dir,
                rolling_config,
                wandb_run,
            )
        else:
            model, metrics = train_bucket_model(
                bucket_name,
                bucket_config,
                train_file,
                val_file,
                hyperparameters,
                features,
                models_dir,
                wandb_run,
            )

        all_metrics[bucket_name] = metrics

        # Log to W&B
        if wandb_run is not None:
            wandb_run.log(
                {
                    f"{bucket_name}/brier_improvement_pct": metrics["brier_improvement_pct"],
                    f"{bucket_name}/model_brier": metrics["model_brier"],
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
        target_min = float(bucket_config["target_improvement"].split("-")[0])

        status = "✓" if metrics["brier_improvement_pct"] >= target_min else "⚠"
        logger.info(
            f"{status} {bucket_config['name']:25s}: {metrics['brier_improvement_pct']:6.2f}% "
            f"(target {target_min:4.1f}%+, baseline {bucket_config['current_improvement']:5.1f}%)"
        )

    # Calculate weighted average
    if len(buckets_to_train) == 3:
        total_samples = sum(all_metrics[b]["n_train"] for b in buckets_to_train)
        weighted_improvement = (
            sum(all_metrics[b]["brier_improvement_pct"] * all_metrics[b]["n_train"] for b in buckets_to_train)
            / total_samples
        )

        logger.info(f"\nWeighted Overall Improvement: {weighted_improvement:.2f}%")
        logger.info(f"Target (Phase 1):             {config['targets']['phase_1']['overall_improvement']:.1f}%")

        if weighted_improvement >= config["targets"]["phase_1"]["overall_improvement"]:
            logger.info("✓ PHASE 1 SUCCESS - Targets achieved!")
        else:
            logger.warning("⚠ Below Phase 1 target - May need hyperparameter tuning")

        if wandb_run is not None:
            wandb_run.log(
                {
                    "overall_weighted_improvement": weighted_improvement,
                    "phase_1_target": config["targets"]["phase_1"]["overall_improvement"],
                }
            )

    # Finish W&B
    if wandb_run is not None:
        wandb_run.finish()

    logger.info("\n✓ Multi-horizon training complete!")
    logger.info(f"Models saved to: {models_dir}")
    logger.info(f"Data saved to: {output_dir}")


if __name__ == "__main__":
    main()
