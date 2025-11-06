#!/usr/bin/env python3
"""
Memory-Optimized LightGBM Residual Model
=========================================

LightGBM implementation with memory optimizations for training on 63M rows
with a 16GB RAM constraint (5GB safe working memory).

Key optimizations:
1. Streaming data pipeline (no full dataset in memory)
2. Float32 instead of Float64 (50% memory reduction)
3. LightGBM's native memory efficiency features
4. Temporal chunking for feature engineering
5. Batch prediction processing
6. Histogram-based algorithm with binning

Advantages over XGBoost:
- Faster training (leaf-wise growth)
- Lower memory usage (histogram optimization)
- Native categorical feature support
- Better handling of imbalanced data
- Efficient parallel training

Author: BT Research Team
Date: 2025-10-31
"""

from __future__ import annotations

import gc
import logging
import os
import shutil
import sys
from datetime import date, timedelta
from pathlib import Path
from typing import Any

import lightgbm as lgb
import numpy as np
import polars as pl
import psutil
import yaml
from dateutil.relativedelta import relativedelta
from scipy.stats import pearsonr, spearmanr

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

# Memory constraints - Updated for 256GB RAM EC2 instance
MAX_MEMORY_GB = 250  # Soft limit for 256GB RAM (leaving 6GB for OS)
BATCH_SIZE = 20_000_000  # 20M rows per batch (optimized for 256GB RAM)
CHUNK_MONTHS = 6  # 6-month chunks for memory safety (can handle 2-year with 256GB RAM)

# File paths
MODEL_DIR = Path(__file__).parent.parent
BASELINE_FILE = MODEL_DIR / "results/production_backtest_results.parquet"
RV_FILE = MODEL_DIR / "results/realized_volatility_1s.parquet"
MICROSTRUCTURE_FILE = MODEL_DIR / "results/microstructure_features.parquet"
ADVANCED_FILE = MODEL_DIR / "results/advanced_features.parquet"
OUTPUT_DIR = MODEL_DIR / "results"

# Feature lists - V3 consolidated features (196 features)
# All features from consolidated_features_v3.parquet
# Philosophy: Include all features, let LightGBM + Optuna select best subset
FEATURE_COLS = [
    # Context features (3) - baseline
    "time_remaining",
    "iv_staleness_seconds",
    "moneyness",
    # Price/volatility baseline features (8)
    "high_15m",
    "low_15m",
    "drawdown_from_high_15m",
    "runup_from_low_15m",
    "time_since_high_15m",
    "time_since_low_15m",
    "skewness_300s",
    "kurtosis_300s",
    # Risk metrics (6)
    "downside_vol_300s",
    "upside_vol_300s",
    "vol_asymmetry_300s",
    "tail_risk_300s",
    "vol_persistence_ar1",
    "vol_acceleration_300s",
    # Time features (3)
    "hour_of_day_utc",
    "hour_sin",
    "hour_cos",
    # GARCH and autocorrelation (5)
    "vol_of_vol_300s",
    "garch_forecast_simple",
    "autocorr_decay",
    "reversals_300s",
    "autocorr_lag5_300s",
    "hurst_300s",
    # Funding rate features (11) - NEW IN V3
    "funding_rate",
    "funding_rate_ema_60s",
    "funding_rate_ema_300s",
    "funding_rate_ema_900s",
    "funding_rate_ema_1800s",
    "funding_rate_ema_3600s",
    "funding_rate_sma_60s",
    "funding_rate_sma_300s",
    "funding_rate_sma_900s",
    "funding_rate_sma_1800s",
    "funding_rate_sma_3600s",
    # Orderbook L0 spread features (16) - NEW IN V3
    "bid_ask_spread_bps",
    "spread_ema_60s",
    "spread_ema_300s",
    "spread_ema_900s",
    "spread_ema_1800s",
    "spread_ema_3600s",
    "spread_sma_60s",
    "spread_sma_300s",
    "spread_sma_900s",
    "spread_sma_1800s",
    "spread_sma_3600s",
    "spread_vol_60s",
    "spread_vol_300s",
    "spread_vol_900s",
    "spread_vol_1800s",
    "spread_vol_3600s",
    # Orderbook L0 imbalance features (16) - NEW IN V3
    "bid_ask_imbalance",
    "imbalance_ema_60s",
    "imbalance_ema_300s",
    "imbalance_ema_900s",
    "imbalance_ema_1800s",
    "imbalance_ema_3600s",
    "imbalance_sma_60s",
    "imbalance_sma_300s",
    "imbalance_sma_900s",
    "imbalance_sma_1800s",
    "imbalance_sma_3600s",
    "imbalance_vol_60s",
    "imbalance_vol_300s",
    "imbalance_vol_900s",
    "imbalance_vol_1800s",
    "imbalance_vol_3600s",
    # Orderbook 5-level depth features (21) - NEW IN V3
    "total_bid_volume_5",
    "total_ask_volume_5",
    "bid_volume_ratio_1to5",
    "ask_volume_ratio_1to5",
    "depth_imbalance_5",
    "depth_imbalance_ema_60s",
    "depth_imbalance_ema_300s",
    "depth_imbalance_ema_900s",
    "depth_imbalance_ema_1800s",
    "depth_imbalance_ema_3600s",
    "depth_imbalance_sma_60s",
    "depth_imbalance_sma_300s",
    "depth_imbalance_sma_900s",
    "depth_imbalance_sma_1800s",
    "depth_imbalance_sma_3600s",
    "depth_imbalance_vol_60s",
    "depth_imbalance_vol_300s",
    "depth_imbalance_vol_900s",
    "depth_imbalance_vol_1800s",
    "depth_imbalance_vol_3600s",
    # Orderbook weighted mid features (11) - NEW IN V3
    "weighted_mid_price_5",
    "weighted_mid_ema_60s",
    "weighted_mid_ema_300s",
    "weighted_mid_ema_900s",
    "weighted_mid_ema_1800s",
    "weighted_mid_ema_3600s",
    "weighted_mid_sma_60s",
    "weighted_mid_sma_300s",
    "weighted_mid_sma_900s",
    "weighted_mid_sma_1800s",
    "weighted_mid_sma_3600s",
    # Price basis features (11) - NEW IN V3
    "mark_index_basis_bps",
    "mark_index_ema_60s",
    "mark_index_ema_300s",
    "mark_index_ema_900s",
    "mark_index_ema_1800s",
    "mark_index_ema_3600s",
    "mark_index_sma_60s",
    "mark_index_sma_300s",
    "mark_index_sma_900s",
    "mark_index_sma_1800s",
    "mark_index_sma_3600s",
    # Open interest features (6) - NEW IN V3
    "open_interest",
    "oi_ema_60s",
    "oi_ema_300s",
    "oi_ema_900s",
    "oi_ema_1800s",
    "oi_ema_3600s",
    # RV base + EMAs + SMAs (24)
    "rv_300s",
    "rv_900s",
    "rv_300s_ema_60s",
    "rv_300s_ema_300s",
    "rv_300s_ema_900s",
    "rv_300s_ema_1800s",
    "rv_300s_ema_3600s",
    "rv_900s_ema_60s",
    "rv_900s_ema_300s",
    "rv_900s_ema_900s",
    "rv_900s_ema_1800s",
    "rv_900s_ema_3600s",
    "rv_300s_sma_60s",
    "rv_300s_sma_300s",
    "rv_300s_sma_900s",
    "rv_300s_sma_1800s",
    "rv_300s_sma_3600s",
    "rv_900s_sma_60s",
    "rv_900s_sma_300s",
    "rv_900s_sma_900s",
    "rv_900s_sma_1800s",
    "rv_900s_sma_3600s",
    # Momentum base + EMAs + SMAs (24)
    "momentum_300s",
    "momentum_900s",
    "momentum_300s_ema_60s",
    "momentum_300s_ema_300s",
    "momentum_300s_ema_900s",
    "momentum_300s_ema_1800s",
    "momentum_300s_ema_3600s",
    "momentum_900s_ema_60s",
    "momentum_900s_ema_300s",
    "momentum_900s_ema_900s",
    "momentum_900s_ema_1800s",
    "momentum_900s_ema_3600s",
    "momentum_300s_sma_60s",
    "momentum_300s_sma_300s",
    "momentum_300s_sma_900s",
    "momentum_300s_sma_1800s",
    "momentum_300s_sma_3600s",
    "momentum_900s_sma_60s",
    "momentum_900s_sma_300s",
    "momentum_900s_sma_900s",
    "momentum_900s_sma_1800s",
    "momentum_900s_sma_3600s",
    # Range base + EMAs + SMAs (24)
    "range_300s",
    "range_900s",
    "range_300s_ema_60s",
    "range_300s_ema_300s",
    "range_300s_ema_900s",
    "range_300s_ema_1800s",
    "range_300s_ema_3600s",
    "range_900s_ema_60s",
    "range_900s_ema_300s",
    "range_900s_ema_900s",
    "range_900s_ema_1800s",
    "range_900s_ema_3600s",
    "range_300s_sma_60s",
    "range_300s_sma_300s",
    "range_300s_sma_900s",
    "range_300s_sma_1800s",
    "range_300s_sma_3600s",
    "range_900s_sma_60s",
    "range_900s_sma_300s",
    "range_900s_sma_900s",
    "range_900s_sma_1800s",
    "range_900s_sma_3600s",
    # RV ratios (4)
    "rv_ratio_5m_1m",
    "rv_ratio_15m_5m",
    "rv_ratio_1h_15m",
    "rv_term_structure",
    # EMA prices + ratios (9)
    "ema_12s",
    "ema_60s",
    "ema_300s",
    "ema_900s",
    "price_vs_ema_900",
    "ema_12s_vs_60s",
    "ema_60s_vs_300s",
    "ema_300s_vs_900s",
    "ema_spread_12s_900s",
]

# Note: 15 V2 features removed (not in V3):
# - rv_60s, rv_3600s (kept rv_300s, rv_900s)
# - momentum_60s, range_60s, reversals_60s (kept 300s/900s versions)
# - jump_detected, jump_count_300s, jump_direction_300s, jump_intensity_300s
# - ema_cross_12_60, ema_cross_60_300, ema_cross_300_900 (GBT can learn)
# - is_us_hours, is_asia_hours, is_europe_hours (hour features remain)
# - autocorr_lag1_300s, autocorr_lag10_300s, autocorr_lag30_300s, autocorr_lag60_300s (kept lag5)
#
# 144 NEW features in V3 (mostly orderbook/funding dynamics)
# Total: 196 features

# LightGBM-specific categorical features (if any)
CATEGORICAL_FEATURES: list[str] = []  # Add categorical features if needed


class MemoryMonitor:
    """Monitor and enforce memory constraints."""

    def __init__(self, max_gb: float = MAX_MEMORY_GB):
        self.max_gb = max_gb
        self.process = psutil.Process(os.getpid())

    def get_memory_gb(self) -> float:
        """Get current process memory in GB."""
        return self.process.memory_info().rss / (1024**3)

    def check_memory(self, label: str = "") -> None:
        """Check memory and warn if exceeding limit."""
        mem_gb = self.get_memory_gb()
        if mem_gb > self.max_gb:
            logger.warning(f"[MEMORY] {label} - EXCEEDING LIMIT: {mem_gb:.2f} GB > {self.max_gb:.2f} GB")
            # Force garbage collection
            gc.collect()
            mem_after = self.get_memory_gb()
            if mem_after < mem_gb:
                logger.info(f"[MEMORY] Freed {mem_gb - mem_after:.2f} GB after GC")
        else:
            logger.info(f"[MEMORY] {label} - {mem_gb:.2f} GB (OK)")

    def enforce_limit(self) -> None:
        """Warn if memory limit exceeded but don't fail unless critical."""
        gc.collect()
        mem_gb = self.get_memory_gb()
        # Only fail if we're approaching system limits (>250GB on 256GB system)
        if mem_gb > 250.0:
            raise MemoryError(f"Critical memory limit: {mem_gb:.2f} GB > 250.0 GB (system limit)")
        elif mem_gb > self.max_gb:
            logger.warning(f"Soft limit exceeded: {mem_gb:.2f} GB > {self.max_gb:.2f} GB (continuing)")


def prepare_features_streaming(
    start_date: date,
    end_date: date,
    output_file: str,
) -> int:
    """
    Prepare features using streaming to avoid loading full dataset.
    Returns number of rows written.
    """
    monitor = MemoryMonitor()
    monitor.check_memory("Start prepare_features")

    logger.info(f"Preparing features for {start_date} to {end_date}")

    # Build lazy query for all joins
    lazy_df = (
        pl.scan_parquet(str(BASELINE_FILE))
        .filter((pl.col("date") >= start_date) & (pl.col("date") <= end_date))
        .select(
            [
                "timestamp",
                "date",
                "contract_id",
                "prob_mid",
                "outcome",
                "time_remaining",
                "iv_staleness_seconds",
                "K",
                "S",  # Include K and S for moneyness calculation
            ]
        )
        .with_columns(
            [
                # Calculate residual using actual column names
                (pl.col("outcome") - pl.col("prob_mid")).alias("residual"),
                # Calculate moneyness properly from K and S
                ((pl.col("S") / pl.col("K")) - 1).alias("moneyness"),
            ]
        )
    )

    # Get timestamp range for filtering feature files
    baseline_timestamps = (
        pl.scan_parquet(str(BASELINE_FILE))
        .filter((pl.col("date") >= start_date) & (pl.col("date") <= end_date))
        .select([pl.col("timestamp").min().alias("min_ts"), pl.col("timestamp").max().alias("max_ts")])
        .collect()
    )

    if len(baseline_timestamps) > 0:
        min_timestamp = baseline_timestamps["min_ts"][0]
        max_timestamp = baseline_timestamps["max_ts"][0]
    else:
        # No data for this date range
        min_timestamp = None
        max_timestamp = None

    # Join RV features
    if RV_FILE.exists() and min_timestamp is not None:
        rv_df = pl.scan_parquet(str(RV_FILE)).filter(
            (pl.col("timestamp_seconds") >= min_timestamp) & (pl.col("timestamp_seconds") <= max_timestamp)
        )
        lazy_df = lazy_df.join(rv_df, left_on="timestamp", right_on="timestamp_seconds", how="left")

    # Join microstructure features
    if MICROSTRUCTURE_FILE.exists() and min_timestamp is not None:
        micro_df = pl.scan_parquet(str(MICROSTRUCTURE_FILE)).filter(
            (pl.col("timestamp_seconds") >= min_timestamp) & (pl.col("timestamp_seconds") <= max_timestamp)
        )
        lazy_df = lazy_df.join(micro_df, left_on="timestamp", right_on="timestamp_seconds", how="left")

    # Join advanced features
    if ADVANCED_FILE.exists() and min_timestamp is not None:
        advanced_df = pl.scan_parquet(str(ADVANCED_FILE)).filter(
            (pl.col("timestamp") >= min_timestamp) & (pl.col("timestamp") <= max_timestamp)
        )
        # Advanced features need both timestamp and contract_id for proper join
        lazy_df = lazy_df.join(advanced_df, on=["timestamp", "contract_id"], how="left")

    # Simply drop nulls in the residual column
    lazy_df = lazy_df.filter(pl.col("residual").is_not_null())

    monitor.check_memory("Before streaming write")

    # Stream to disk (constant memory usage)
    logger.info(f"Streaming features to {output_file}...")
    lazy_df.sink_parquet(
        output_file,
        compression="snappy",
        statistics=True,
    )

    # Clean up lazy frames and force garbage collection
    del lazy_df, baseline_timestamps
    gc.collect()
    monitor.check_memory("After cleanup")

    # Log file size instead of row count (avoids re-scanning)
    file_size_gb = Path(output_file).stat().st_size / 1e9
    logger.info(f"Wrote features to {output_file}")
    logger.info(f"File size: {file_size_gb:.2f} GB")

    monitor.check_memory("After streaming write")
    return 1  # Return 1 to indicate success (file was written)


def create_lgb_dataset_from_parquet(
    file_path: str,
    features: list[str],
    reference: lgb.Dataset | None = None,
    free_raw_data: bool = True,
) -> tuple[lgb.Dataset, np.ndarray]:
    """
    Create LightGBM Dataset from Parquet file with memory optimization.

    Returns the dataset and labels for validation metrics.
    """
    monitor = MemoryMonitor()
    monitor.check_memory(f"Start loading {Path(file_path).name}")

    # Load data using Polars (more efficient than pandas)
    df = pl.read_parquet(file_path)

    # Extract features and target
    x_data = df.select(features).to_numpy().astype(np.float32)  # Force float32
    y = df.select("residual").to_numpy().ravel().astype(np.float32)

    # Handle inf/nan values
    x_data = np.nan_to_num(x_data, nan=np.nan, posinf=np.nan, neginf=np.nan)
    y = np.nan_to_num(y, nan=0.0, posinf=1.0, neginf=-1.0)

    # Create LightGBM dataset with memory optimization
    dataset = lgb.Dataset(
        x_data,
        label=y,
        feature_name=features,
        categorical_feature=CATEGORICAL_FEATURES if CATEGORICAL_FEATURES else "auto",
        reference=reference,
        free_raw_data=free_raw_data,  # Free raw data after dataset construction
    )

    # Clean up
    del df, x_data
    gc.collect()

    monitor.check_memory(f"After loading {Path(file_path).name}")
    return dataset, y


def evaluate_brier_score(predictions_df: pl.DataFrame) -> dict[str, float]:
    """
    Calculate Brier scores for baseline and model predictions.

    The Brier score measures the accuracy of probabilistic predictions.
    For residual models, MSE of residuals = Brier score improvement.

    Returns:
        Dictionary with baseline_brier, model_brier, and improvement_pct
    """
    # Calculate baseline Brier score
    baseline_brier_val = ((predictions_df["prob_mid"] - predictions_df["outcome"]) ** 2).mean()
    baseline_brier = float(baseline_brier_val) if baseline_brier_val is not None else 0.0  # type: ignore[arg-type]

    # Calculate model Brier score (using final corrected probabilities)
    if "final_prob" in predictions_df.columns:
        model_brier_val = ((predictions_df["final_prob"] - predictions_df["outcome"]) ** 2).mean()
        model_brier = float(model_brier_val) if model_brier_val is not None else 0.0  # type: ignore[arg-type]
    else:
        # If no final_prob, use residual_pred to compute it
        final_prob = predictions_df["prob_mid"] + predictions_df["residual_pred"]
        model_brier_val = ((final_prob - predictions_df["outcome"]) ** 2).mean()
        model_brier = float(model_brier_val) if model_brier_val is not None else 0.0  # type: ignore[arg-type]

    # Calculate improvement
    improvement_pct = (baseline_brier - model_brier) / baseline_brier * 100 if baseline_brier > 0 else 0

    return {
        "baseline_brier": baseline_brier,
        "model_brier": model_brier,
        "improvement_pct": improvement_pct,
    }


def calculate_residual_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """
    Calculate residual prediction metrics.

    Key insight: For residual models, MSE of residuals directly equals
    the Brier score improvement of the final predictions.

    Returns:
        Dictionary with MSE, RMSE, MAE, and implied Brier improvement
    """
    mse = float(np.mean((y_true - y_pred) ** 2))
    rmse = float(np.sqrt(mse))
    mae = float(np.mean(np.abs(y_true - y_pred)))

    return {
        "residual_mse": mse,
        "residual_rmse": rmse,
        "residual_mae": mae,
        "implied_brier_improvement": mse,  # Direct relationship!
    }


def train_lightgbm_memory_optimized(
    train_file: str,
    val_file: str,
    config: dict[str, Any],
    wandb_run=None,
) -> lgb.Booster:
    """
    Train LightGBM with memory optimizations.

    LightGBM advantages:
    - More memory efficient than XGBoost
    - Faster training with leaf-wise growth
    - Better handling of categorical features
    - Native support for missing values

    Args:
        train_file: Path to training data parquet
        val_file: Path to validation data parquet
        config: Training configuration dictionary
        wandb_run: Optional W&B run for logging training curves
    """
    monitor = MemoryMonitor()
    monitor.check_memory("Start training")

    # Get feature columns available in the file
    schema = pl.scan_parquet(train_file).collect_schema()
    features = [col for col in FEATURE_COLS if col in schema.names()]
    logger.info(f"Using {len(features)} features for training")

    # Prepare LightGBM parameters
    params = {
        # Core parameters
        "objective": config["hyperparameters"].get("objective", "regression"),
        "metric": config["hyperparameters"].get("metric", ["mse", "rmse", "mae"]),
        "boosting_type": config["hyperparameters"].get("boosting_type", "gbdt"),
        "seed": config["hyperparameters"].get("seed", 42),
        "verbose": -1,
        "num_threads": config.get("memory", {}).get("num_threads", 1),  # Single thread for memory safety
        # Tree parameters
        "num_leaves": config["hyperparameters"].get("num_leaves", 31),
        "max_depth": config["hyperparameters"].get("max_depth", -1),
        "min_data_in_leaf": config["hyperparameters"].get("min_data_in_leaf", 20),
        "min_sum_hessian_in_leaf": config["hyperparameters"].get("min_sum_hessian_in_leaf", 1e-3),
        # Regularization
        "lambda_l1": config["hyperparameters"].get("lambda_l1", 1.0),
        "lambda_l2": config["hyperparameters"].get("lambda_l2", 20.0),
        "min_gain_to_split": config["hyperparameters"].get("min_gain_to_split", 1.0),
        "feature_fraction": config["hyperparameters"].get("feature_fraction", 0.8),
        "bagging_fraction": config["hyperparameters"].get("bagging_fraction", 0.7),
        "bagging_freq": config["hyperparameters"].get("bagging_freq", 5),
        # Learning parameters
        "learning_rate": config["hyperparameters"].get("learning_rate", 0.05),
        # Memory optimization parameters
        "max_bin": config.get("memory", {}).get("max_bin", 63),  # LightGBM can handle more bins efficiently
        "min_data_per_group": config.get("memory", {}).get("min_data_per_group", 100),
        "max_cat_threshold": config.get("memory", {}).get("max_cat_threshold", 32),
        "histogram_pool_size": config.get("memory", {}).get("histogram_pool_size", -1),  # Auto
        # Performance optimizations specific to LightGBM
        "use_missing": True,  # Handle missing values natively
        "zero_as_missing": False,
        "two_round": True,  # Use two round loading for memory efficiency
        "force_row_wise": True,  # Force row-wise histogram building for lower memory
        "deterministic": True,  # Ensure reproducibility
        # Data efficiency
        "data_sample_strategy": "bagging",  # Use bagging for data sampling
        "enable_bundle": True,  # Bundle sparse features
        "max_conflict_rate": 0.0,  # Stricter bundling for accuracy
    }

    # Add device-specific optimizations
    if config.get("use_gpu", False):
        params.update(
            {
                "device_type": "gpu",
                "gpu_use_dp": False,  # Use float32 on GPU
            }
        )

    logger.info("Creating LightGBM datasets...")

    # Create train dataset
    train_data, _ = create_lgb_dataset_from_parquet(train_file, features, free_raw_data=True)

    monitor.check_memory("After train dataset")

    # Create validation dataset with reference to train for consistency
    val_data, _ = create_lgb_dataset_from_parquet(val_file, features, reference=train_data, free_raw_data=True)

    monitor.check_memory("After val dataset")

    # Train model
    logger.info("Training LightGBM model...")

    # Callbacks for memory monitoring and early stopping
    callbacks = [
        lgb.early_stopping(config["hyperparameters"].get("early_stopping_rounds", 25)),
        lgb.log_evaluation(10),
    ]

    # Add W&B callback if run is active
    if wandb_run is not None:
        try:
            from wandb.integration.lightgbm import wandb_callback

            callbacks.append(wandb_callback())
            logger.info("✓ W&B training curve logging enabled")
        except ImportError:
            logger.warning("wandb.integration.lightgbm not available - skipping W&B logging")

    # Train with validation
    model = lgb.train(
        params,
        train_data,
        num_boost_round=config["hyperparameters"].get("n_estimators", 200),
        valid_sets=[train_data, val_data],
        valid_names=["train", "val"],
        callbacks=callbacks,
    )

    # Log feature importance
    importance = model.feature_importance(importance_type="gain")
    feature_names = model.feature_name()
    top_features = sorted(zip(feature_names, importance), key=lambda x: x[1], reverse=True)[:20]

    logger.info("\nTop 20 features by gain:")
    for i, (feat, imp) in enumerate(top_features, 1):
        logger.info(f"  {i:2d}. {feat:40s}: {imp:10.2f}")

    # Log feature importance and model summary to W&B
    if wandb_run is not None:
        try:
            from wandb.integration.lightgbm import log_summary

            log_summary(model, save_model_checkpoint=True)
            logger.info("✓ W&B model summary and artifacts logged")
        except ImportError:
            logger.warning("wandb.integration.lightgbm not available - skipping model summary")

    # Log residual metrics and implied Brier improvement
    best_score = model.best_score
    if best_score and "val" in best_score:
        val_metrics = best_score["val"]
        if "l2" in val_metrics:  # l2 is MSE in LightGBM
            residual_mse = val_metrics["l2"]
            logger.info("\n" + "=" * 80)
            logger.info("RESIDUAL METRICS AND BRIER SCORE RELATIONSHIP")
            logger.info("=" * 80)
            logger.info(f"Validation Residual MSE:     {residual_mse:.6f}")
            logger.info(f"Validation Residual RMSE:    {np.sqrt(residual_mse):.6f}")
            logger.info(f"Implied Brier Improvement:   {residual_mse:.6f}")
            logger.info("Note: For residual models, MSE of residuals = Brier score improvement")
            logger.info("=" * 80)

    monitor.check_memory("After training")

    return model


def predict_in_batches(
    model: lgb.Booster,
    test_file: str,
    output_file: str,
    batch_size: int = BATCH_SIZE,
) -> None:
    """
    Make predictions in batches to avoid memory issues.

    LightGBM's prediction is more memory efficient than XGBoost.
    """
    monitor = MemoryMonitor()
    monitor.check_memory("Start predictions")

    # Get total rows
    total_rows = pl.scan_parquet(test_file).select(pl.len()).collect().item()
    logger.info(f"Predicting on {total_rows:,} rows in batches of {batch_size:,}")

    # Get feature columns
    schema = pl.scan_parquet(test_file).collect_schema()
    features = [col for col in FEATURE_COLS if col in schema.names()]

    # Process in batches
    batch_results = []
    batch_dir = Path(output_file).parent / "prediction_batches"
    batch_dir.mkdir(exist_ok=True)

    for i, offset in enumerate(range(0, total_rows, batch_size)):
        batch_num = i + 1
        total_batches = (total_rows + batch_size - 1) // batch_size

        logger.info(f"Processing batch {batch_num}/{total_batches}...")

        # Load batch
        batch_df = pl.scan_parquet(test_file).slice(offset, batch_size).collect()

        monitor.check_memory(f"Loaded batch {batch_num}")

        # Extract features for prediction
        x_batch = batch_df.select(features).to_numpy().astype(np.float32)

        # Handle inf/nan values
        x_batch = np.nan_to_num(x_batch, nan=np.nan, posinf=np.nan, neginf=np.nan)

        # Make predictions (LightGBM handles missing values)
        residual_pred = model.predict(
            x_batch, num_iteration=model.best_iteration if hasattr(model, "best_iteration") else None
        )

        # Add predictions to dataframe
        result_df = batch_df.select(
            [
                "timestamp",
                "date",
                "contract_id",
                "prob_mid",
                "outcome",
            ]
        ).with_columns(
            [
                pl.Series("residual_pred", residual_pred).cast(pl.Float32),
                (pl.col("prob_mid") + pl.Series("residual_pred", residual_pred)).alias("final_prob").cast(pl.Float32),
            ]
        )

        # Save batch result
        batch_file = batch_dir / f"batch_{batch_num:06d}.parquet"
        result_df.write_parquet(batch_file)
        batch_results.append(batch_file)

        # Free memory
        del batch_df, x_batch, result_df, residual_pred
        gc.collect()

        monitor.check_memory(f"After batch {batch_num}")

    # Combine all batches using lazy concatenation
    logger.info("Combining prediction batches...")
    # Note: sink_parquet doesn't have a streaming parameter - it always streams for LazyFrame
    pl.concat([pl.scan_parquet(f) for f in batch_results]).sink_parquet(
        output_file,
        compression="snappy",
    )

    # Clean up batch files
    shutil.rmtree(batch_dir, ignore_errors=True)

    logger.info(f"Predictions saved to {output_file}")

    # Calculate and report Brier scores
    logger.info("\nCalculating Brier scores for predictions...")
    predictions_df = pl.scan_parquet(output_file).collect()
    brier_results = evaluate_brier_score(predictions_df)

    logger.info("\n" + "=" * 80)
    logger.info("BRIER SCORE EVALUATION")
    logger.info("=" * 80)
    logger.info(f"Baseline Brier Score:        {brier_results['baseline_brier']:.6f}")
    logger.info(f"LightGBM Brier Score:        {brier_results['model_brier']:.6f}")
    logger.info(f"Improvement:                 {brier_results['improvement_pct']:+.2f}%")
    logger.info("=" * 80)

    monitor.check_memory("End predictions")


def split_data_three_way(
    chunk_files: list[Path],
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    output_dir: Path = OUTPUT_DIR,
    shuffle: bool = False,  # CRITICAL: Maintain temporal order to prevent look-ahead bias
    seed: int = 42,
) -> tuple[str, str, str]:
    """
    Split data into train/validation/test sets.

    IMPORTANT: For time series data, shuffle should be False to maintain temporal order
    and prevent data leakage. Training on future data to predict past is invalid.

    Args:
        chunk_files: List of parquet files to combine and split
        train_ratio: Proportion for training set
        val_ratio: Proportion for validation set
        test_ratio: Proportion for test set
        output_dir: Directory to save split files
        shuffle: Whether to shuffle data before splitting (MUST be False for time series)
        seed: Random seed for shuffling (only used if shuffle=True)

    Returns:
        Tuple of (train_file, val_file, test_file) paths
    """
    monitor = MemoryMonitor()
    logger.info("\nSplitting data into train/validation/test sets...")
    logger.info(f"Split ratios: {train_ratio:.0%}/{val_ratio:.0%}/{test_ratio:.0%}")

    # Validate ratios
    total_ratio = train_ratio + val_ratio + test_ratio
    if abs(total_ratio - 1.0) > 0.001:
        raise ValueError(f"Ratios must sum to 1.0, got {total_ratio}")

    # Combine all chunks using lazy operations
    all_data = pl.concat([pl.scan_parquet(f) for f in chunk_files])

    # Get total rows for split calculation
    total_rows = all_data.select(pl.len()).collect().item()
    train_rows = int(total_rows * train_ratio)
    val_rows = int(total_rows * val_ratio)
    test_rows = total_rows - train_rows - val_rows  # Ensures we use all rows

    logger.info(f"Total rows: {total_rows:,}")
    logger.info(f"Train rows: {train_rows:,} ({train_rows / total_rows:.1%})")
    logger.info(f"Val rows:   {val_rows:,} ({val_rows / total_rows:.1%})")
    logger.info(f"Test rows:  {test_rows:,} ({test_rows / total_rows:.1%})")

    # Define output files
    train_file = output_dir / "train_features_lgb.parquet"
    val_file = output_dir / "val_features_lgb.parquet"
    test_file = output_dir / "test_features_lgb.parquet"

    if shuffle:
        logger.info("Shuffling data before split...")
        # For large datasets, we'll use a hash-based shuffle
        # Add a random column for shuffling
        shuffled_data = (
            all_data.with_columns([(pl.col("timestamp").hash(seed) % 1000000).alias("shuffle_key")])
            .sort("shuffle_key")
            .drop("shuffle_key")
        )

        # Write splits using streaming
        logger.info("Writing train set...")
        shuffled_data.head(train_rows).sink_parquet(str(train_file))

        logger.info("Writing validation set...")
        shuffled_data.slice(train_rows, val_rows).sink_parquet(str(val_file))

        logger.info("Writing test set...")
        shuffled_data.tail(test_rows).sink_parquet(str(test_file))
    else:
        # Sequential split (no shuffling - maintains temporal order)
        logger.info("Using sequential split (no shuffling)...")

        logger.info("Writing train set...")
        all_data.head(train_rows).sink_parquet(str(train_file))

        logger.info("Writing validation set...")
        all_data.slice(train_rows, val_rows).sink_parquet(str(val_file))

        logger.info("Writing test set...")
        all_data.tail(test_rows).sink_parquet(str(test_file))

    monitor.check_memory("After data split")

    return str(train_file), str(val_file), str(test_file)


def train_temporal_chunks(
    start_date: date,
    end_date: date,
    config_file: str,
    chunk_months: int = CHUNK_MONTHS,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    evaluate_test: bool = True,
    wandb_run=None,
) -> dict[str, Any]:
    """
    Train model using temporal chunking for very large datasets with train/val/test splits.

    Args:
        start_date: Start date for data
        end_date: End date for data
        config_file: Configuration file path
        chunk_months: Months per chunk for processing
        train_ratio: Proportion for training set
        val_ratio: Proportion for validation set
        test_ratio: Proportion for test set
        evaluate_test: Whether to evaluate on test set after training
        wandb_run: Optional W&B run for logging training curves

    Returns:
        Dictionary with test set metrics (if evaluate_test=True), empty dict otherwise
    """
    monitor = MemoryMonitor()

    # Load configuration
    with open(config_file) as f:
        config = yaml.safe_load(f)

    logger.info("=" * 80)
    logger.info("MEMORY-OPTIMIZED LIGHTGBM TRAINING")
    logger.info("=" * 80)
    logger.info(f"Date range: {start_date} to {end_date}")
    logger.info(f"Chunk size: {chunk_months} months")
    logger.info(f"Memory limit: {MAX_MEMORY_GB} GB")
    logger.info("\nLightGBM Advantages:")
    logger.info("- Leaf-wise tree growth (faster convergence)")
    logger.info("- Histogram-based algorithm (lower memory)")
    logger.info("- Native categorical support")
    logger.info("- Better handling of imbalanced data")

    # Process data in temporal chunks
    current_date = start_date
    chunk_files = []

    while current_date < end_date:
        # Calculate chunk end date
        chunk_end = current_date + relativedelta(months=chunk_months) - timedelta(days=1) if chunk_months else end_date

        chunk_end = min(chunk_end, end_date)

        logger.info(f"\nProcessing chunk: {current_date} to {chunk_end}")

        # Prepare features for this chunk
        chunk_file = OUTPUT_DIR / f"features_lgb_{current_date}_{chunk_end}.parquet"
        rows = prepare_features_streaming(current_date, chunk_end, str(chunk_file))

        if rows > 0:
            chunk_files.append(chunk_file)

        current_date = chunk_end + timedelta(days=1)

        monitor.enforce_limit()  # Check memory limit

    # Combine chunks and split into train/val/test
    logger.info("\nCombining chunks and creating train/val/test splits...")

    # Use the three-way split function
    train_file, val_file, test_file = split_data_three_way(
        chunk_files,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        output_dir=OUTPUT_DIR,
        shuffle=False,  # CRITICAL: Maintain temporal order to prevent look-ahead bias (train on past, predict future)
        seed=config["hyperparameters"].get("seed", 42),
    )

    monitor.check_memory("After data preparation")

    # Train model
    model = train_lightgbm_memory_optimized(train_file, val_file, config, wandb_run=wandb_run)

    # Save model
    model_file = OUTPUT_DIR / "lightgbm_model_optimized.txt"
    model.save_model(str(model_file))
    logger.info(f"Model saved to {model_file}")

    # Also save in JSON format for inspection
    model_json_file = OUTPUT_DIR / "lightgbm_model_optimized.json"
    model.save_model(str(model_json_file), num_iteration=model.best_iteration)
    logger.info(f"Model (JSON) saved to {model_json_file}")

    # Evaluate on test set if requested
    results = {}
    if evaluate_test and test_file:
        logger.info("\n" + "=" * 80)
        logger.info("EVALUATING ON TEST SET")
        logger.info("=" * 80)

        # Load test data
        test_df = pl.read_parquet(test_file)
        schema = test_df.columns
        features = [col for col in FEATURE_COLS if col in schema]

        # Make predictions
        x_test = test_df.select(features).to_numpy().astype(np.float32)
        x_test = np.nan_to_num(x_test, nan=np.nan, posinf=np.nan, neginf=np.nan)

        test_pred = model.predict(
            x_test, num_iteration=model.best_iteration if hasattr(model, "best_iteration") else None
        )
        # Ensure predictions are ndarray for type checking
        test_pred_array = np.asarray(test_pred)

        # Calculate residual metrics
        if "residual" in test_df.columns:
            y_test = test_df.select("residual").to_numpy().ravel()
            test_metrics = calculate_residual_metrics(y_test, test_pred_array)

            logger.info("\nTest Set Residual Metrics:")
            logger.info(f"  MSE:  {test_metrics['residual_mse']:.6f}")
            logger.info(f"  RMSE: {test_metrics['residual_rmse']:.6f}")
            logger.info(f"  MAE:  {test_metrics['residual_mae']:.6f}")

            # Add to results
            results.update(test_metrics)

        # Calculate Brier scores if we have the necessary columns
        if "prob_mid" in test_df.columns and "outcome" in test_df.columns:
            test_with_pred = test_df.with_columns(
                [
                    pl.Series("residual_pred", test_pred_array).cast(pl.Float32),
                    (pl.col("prob_mid") + pl.Series("residual_pred", test_pred_array))
                    .alias("final_prob")
                    .cast(pl.Float32),
                ]
            )

            test_brier_results = evaluate_brier_score(test_with_pred)

            logger.info("\nTest Set Brier Scores:")
            logger.info(f"  Baseline Brier: {test_brier_results['baseline_brier']:.6f}")
            logger.info(f"  Model Brier:    {test_brier_results['model_brier']:.6f}")
            logger.info(f"  Improvement:    {test_brier_results['improvement_pct']:+.2f}%")

            # Add to results
            results.update(test_brier_results)

            # Summary across all sets
            logger.info("\n" + "=" * 80)
            logger.info("FINAL MODEL PERFORMANCE SUMMARY")
            logger.info("=" * 80)
            logger.info(f"Test Set Brier Score:     {test_brier_results['model_brier']:.6f}")
            logger.info(f"Test Set Improvement:     {test_brier_results['improvement_pct']:+.2f}%")
            if "residual" in test_df.columns:
                logger.info(f"Test Set Residual MSE:    {test_metrics['residual_mse']:.6f}")
                logger.info("Note: MSE of residuals ≈ Brier score improvement")
            logger.info("=" * 80)

    # Clean up intermediate files
    for chunk_file in chunk_files:
        chunk_file.unlink(missing_ok=True)
    if not evaluate_test:  # Keep test file if we evaluated on it
        Path(train_file).unlink(missing_ok=True)
        Path(val_file).unlink(missing_ok=True)
        Path(test_file).unlink(missing_ok=True)

    monitor.check_memory("Final")

    return results


def compare_with_xgboost(
    lgb_model_path: str,
    xgb_model_path: str,
    test_file: str,
) -> None:
    """
    Compare LightGBM and XGBoost models performance including Brier scores.
    """
    logger.info("=" * 80)
    logger.info("MODEL COMPARISON: LightGBM vs XGBoost")
    logger.info("=" * 80)

    # Load test data
    test_df = pl.read_parquet(test_file)

    # Get features
    schema = test_df.columns
    features = [col for col in FEATURE_COLS if col in schema]

    x_test = test_df.select(features).to_numpy().astype(np.float32)
    y_test = test_df.select("residual").to_numpy().ravel()

    # Load and predict with LightGBM
    lgb_model = lgb.Booster(model_file=lgb_model_path)
    lgb_pred = lgb_model.predict(x_test)
    # Ensure predictions are ndarray for type checking
    lgb_pred_array = np.asarray(lgb_pred)

    # Calculate residual metrics for LightGBM
    lgb_metrics = calculate_residual_metrics(y_test, lgb_pred_array)

    logger.info("\n" + "-" * 40)
    logger.info("RESIDUAL METRICS")
    logger.info("-" * 40)
    logger.info("\nLightGBM Performance:")
    logger.info(f"  Residual MSE:  {lgb_metrics['residual_mse']:.6f}")
    logger.info(f"  Residual RMSE: {lgb_metrics['residual_rmse']:.6f}")
    logger.info(f"  Residual MAE:  {lgb_metrics['residual_mae']:.6f}")

    # Calculate Brier scores if we have prob_mid and outcome
    if "prob_mid" in test_df.columns and "outcome" in test_df.columns:
        # Add predictions to dataframe for Brier calculation
        test_with_lgb_pred = test_df.with_columns(
            [
                pl.Series("residual_pred", lgb_pred_array).cast(pl.Float32),
                (pl.col("prob_mid") + pl.Series("residual_pred", lgb_pred_array)).alias("final_prob").cast(pl.Float32),
            ]
        )

        lgb_brier_results = evaluate_brier_score(test_with_lgb_pred)

        logger.info("\n" + "-" * 40)
        logger.info("BRIER SCORE EVALUATION")
        logger.info("-" * 40)
        logger.info("\nLightGBM Brier Scores:")
        logger.info(f"  Baseline Brier: {lgb_brier_results['baseline_brier']:.6f}")
        logger.info(f"  Model Brier:    {lgb_brier_results['model_brier']:.6f}")
        logger.info(f"  Improvement:    {lgb_brier_results['improvement_pct']:+.2f}%")
        logger.info("\nKey Insight:")
        logger.info(f"  Residual MSE ({lgb_metrics['residual_mse']:.6f}) ≈ Brier Improvement")

    # If XGBoost model exists, compare
    if Path(xgb_model_path).exists():
        import xgboost as xgb

        xgb_model = xgb.Booster()
        xgb_model.load_model(xgb_model_path)

        dtest = xgb.DMatrix(x_test)
        xgb_pred = xgb_model.predict(dtest)
        # Ensure predictions are ndarray for type checking
        xgb_pred_array = np.asarray(xgb_pred)

        xgb_metrics = calculate_residual_metrics(y_test, xgb_pred_array)

        logger.info("\nXGBoost Performance:")
        logger.info(f"  Residual MSE:  {xgb_metrics['residual_mse']:.6f}")
        logger.info(f"  Residual RMSE: {xgb_metrics['residual_rmse']:.6f}")
        logger.info(f"  Residual MAE:  {xgb_metrics['residual_mae']:.6f}")

        # Calculate XGBoost Brier scores
        if "prob_mid" in test_df.columns and "outcome" in test_df.columns:
            test_with_xgb_pred = test_df.with_columns(
                [
                    pl.Series("residual_pred", xgb_pred_array).cast(pl.Float32),
                    (pl.col("prob_mid") + pl.Series("residual_pred", xgb_pred_array))
                    .alias("final_prob")
                    .cast(pl.Float32),
                ]
            )

            xgb_brier_results = evaluate_brier_score(test_with_xgb_pred)

            logger.info("\nXGBoost Brier Scores:")
            logger.info(f"  Baseline Brier: {xgb_brier_results['baseline_brier']:.6f}")
            logger.info(f"  Model Brier:    {xgb_brier_results['model_brier']:.6f}")
            logger.info(f"  Improvement:    {xgb_brier_results['improvement_pct']:+.2f}%")

            logger.info("\n" + "=" * 40)
            logger.info("HEAD-TO-HEAD COMPARISON")
            logger.info("=" * 40)
            logger.info("\nResidual Metrics (LightGBM vs XGBoost):")
            logger.info(
                f"  MSE:  {((xgb_metrics['residual_mse'] - lgb_metrics['residual_mse']) / xgb_metrics['residual_mse'] * 100):+.2f}% (negative = LightGBM better)"
            )
            logger.info(
                f"  RMSE: {((xgb_metrics['residual_rmse'] - lgb_metrics['residual_rmse']) / xgb_metrics['residual_rmse'] * 100):+.2f}%"
            )
            logger.info(
                f"  MAE:  {((xgb_metrics['residual_mae'] - lgb_metrics['residual_mae']) / xgb_metrics['residual_mae'] * 100):+.2f}%"
            )

            logger.info("\nBrier Score Comparison:")
            logger.info(f"  LightGBM Brier: {lgb_brier_results['model_brier']:.6f}")
            logger.info(f"  XGBoost Brier:  {xgb_brier_results['model_brier']:.6f}")
            if lgb_brier_results["model_brier"] < xgb_brier_results["model_brier"]:
                logger.info("  Winner: LightGBM (lower is better)")
            else:
                logger.info("  Winner: XGBoost (lower is better)")


def calculate_ic(predictions: np.ndarray, outcomes: np.ndarray) -> dict[str, float]:
    """
    Calculate Information Coefficient (IC) metrics.

    IC measures rank correlation between predictions and outcomes.
    Used in quantitative finance to evaluate signal quality.

    From "The Elements of Quantitative Investing" (Section 8.3.1):
    - For strategies: Use Sharpe Ratio (z-scored returns by volatility)
    - For signals: Use IC (Spearman rank correlation)

    Args:
        predictions: Predicted residuals (continuous values)
        outcomes: Actual outcomes (binary 0/1 for BTC options)

    Returns:
        Dictionary with IC metrics:
        - spearman_ic: Spearman rank correlation (robust to outliers)
        - pearson_ic: Pearson correlation (for comparison)
        - ic_pvalue: Statistical significance of Spearman IC
        - n_samples: Number of valid samples used
    """
    # Remove any NaN values
    mask = ~(np.isnan(predictions) | np.isnan(outcomes))
    pred_clean = predictions[mask]
    outcome_clean = outcomes[mask]

    n_samples = len(pred_clean)

    if n_samples < 100:
        logger.warning(f"Only {n_samples} samples for IC calculation (minimum 100 recommended)")
        return {
            "spearman_ic": 0.0,
            "pearson_ic": 0.0,
            "ic_pvalue": 1.0,
            "n_samples": n_samples,
        }

    # Spearman IC (primary metric - robust to outliers, measures monotonic relationship)
    spearman_result = spearmanr(pred_clean, outcome_clean)
    spearman_ic = float(spearman_result.statistic)  # type: ignore[attr-defined]
    ic_pvalue = float(spearman_result.pvalue)  # type: ignore[attr-defined]

    # Pearson IC (for comparison - sensitive to outliers, measures linear relationship)
    pearson_result = pearsonr(pred_clean, outcome_clean)
    pearson_ic = float(pearson_result.statistic)  # type: ignore[attr-defined]

    return {
        "spearman_ic": spearman_ic,
        "pearson_ic": pearson_ic,
        "ic_pvalue": ic_pvalue,
        "n_samples": n_samples,
    }


def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="Memory-optimized LightGBM training")
    parser.add_argument(
        "--config",
        type=str,
        default="config/lightgbm_config_production.yaml",
        help="Configuration file",
    )
    parser.add_argument(
        "--start-date",
        type=str,
        default="2023-01-01",
        help="Start date (YYYY-MM-DD)",
    )
    parser.add_argument(
        "--end-date",
        type=str,
        default="2024-10-31",
        help="End date (YYYY-MM-DD)",
    )
    parser.add_argument(
        "--chunk-months",
        type=int,
        default=6,
        help="Months per chunk (0 for no chunking)",
    )
    parser.add_argument(
        "--pilot",
        action="store_true",
        help="Run pilot test on October 2023 only",
    )
    parser.add_argument(
        "--compare",
        action="store_true",
        help="Compare with XGBoost model if available",
    )
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.8,
        help="Proportion of data for training (default: 0.8)",
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.1,
        help="Proportion of data for validation (default: 0.1)",
    )
    parser.add_argument(
        "--test-ratio",
        type=float,
        default=0.1,
        help="Proportion of data for testing (default: 0.1)",
    )
    parser.add_argument(
        "--evaluate-test",
        action="store_true",
        default=True,
        help="Evaluate model on test set after training (default: True)",
    )
    parser.add_argument(
        "--no-evaluate-test",
        action="store_false",
        dest="evaluate_test",
        help="Skip test set evaluation",
    )

    args = parser.parse_args()

    # Parse dates
    if args.pilot:
        start_date = date(2023, 10, 1)
        end_date = date(2023, 10, 31)
        chunk_months = 0  # No chunking for pilot
    else:
        start_date = date.fromisoformat(args.start_date)
        end_date = date.fromisoformat(args.end_date)
        chunk_months = args.chunk_months

    # Validate split ratios
    total_ratio = args.train_ratio + args.val_ratio + args.test_ratio
    if abs(total_ratio - 1.0) > 0.001:
        parser.error(f"Train, val, and test ratios must sum to 1.0, got {total_ratio}")

    # Resolve config path
    config_file = Path(__file__).parent / args.config

    # Run training with specified splits
    train_temporal_chunks(
        start_date,
        end_date,
        str(config_file),
        chunk_months,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        evaluate_test=args.evaluate_test,
    )

    # Optional: Compare with XGBoost
    if args.compare:
        lgb_model = OUTPUT_DIR / "lightgbm_model_optimized.txt"
        xgb_model = OUTPUT_DIR / "xgboost_model_optimized.json"
        test_file = OUTPUT_DIR / "val_features_lgb.parquet"  # Use validation as test

        if lgb_model.exists() and test_file.exists():
            compare_with_xgboost(str(lgb_model), str(xgb_model), str(test_file))


if __name__ == "__main__":
    main()
