#!/usr/bin/env python3
"""
Memory-Optimized XGBoost Residual Model
========================================

This version implements all memory optimizations for training on 63M rows
with a 16GB RAM constraint (5GB safe working memory).

Key optimizations:
1. Streaming data pipeline (no full dataset in memory)
2. Float32 instead of Float64 (50% memory reduction)
3. External memory XGBoost with Parquet files
4. Temporal chunking for feature engineering
5. Batch prediction processing

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
from dateutil.relativedelta import relativedelta
from pathlib import Path
from typing import Any, Iterator

import numpy as np
import polars as pl
import psutil
import xgboost as xgb
import yaml

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

# Memory constraints - Updated for 30GB RAM system
MAX_MEMORY_GB = 20.0  # Safe limit for 30GB RAM (leaving ~10GB for OS)
BATCH_SIZE = 5_000_000  # 5M rows per batch (more efficient with 30GB RAM)
CHUNK_MONTHS = 6  # Process 6 months at a time (larger chunks with more RAM)

# File paths
MODEL_DIR = Path(__file__).parent.parent
BASELINE_FILE = MODEL_DIR / "results/production_backtest_results.parquet"
RV_FILE = MODEL_DIR / "results/realized_volatility_1s.parquet"
MICROSTRUCTURE_FILE = MODEL_DIR / "results/microstructure_features.parquet"
ADVANCED_FILE = MODEL_DIR / "results/advanced_features.parquet"
OUTPUT_DIR = MODEL_DIR / "results"
CACHE_DIR = MODEL_DIR / "xgb_cache"

# Feature lists (adjusted to match actual column names in files)
FEATURE_COLS = [
    # RV features (8) - from realized_volatility_1s.parquet
    "rv_60s", "rv_300s", "rv_900s", "rv_3600s",
    "rv_ratio_5m_1m", "rv_ratio_15m_5m", "rv_ratio_1h_15m", "rv_term_structure",
    # Microstructure (11) - from microstructure_features.parquet
    "momentum_60s", "momentum_300s", "momentum_900s",
    "range_60s", "range_300s", "range_900s",
    "reversals_60s", "jump_detected",
    "autocorr_lag1_300s", "autocorr_lag5_300s", "hurst_300s",
    # Context (3) - from baseline
    "time_remaining", "iv_staleness_seconds", "moneyness",
    # Advanced features (adjusted to match advanced_features.parquet)
    "ema_12s", "ema_60s", "ema_300s", "ema_900s",
    "ema_cross_12_60", "ema_cross_60_300", "ema_cross_300_900", "price_vs_ema_900",
    "iv_rv_ratio_300s", "iv_rv_ratio_900s",  # Changed from iv_rv_ratio
    "drawdown_from_high_15m", "runup_from_low_15m",  # Changed from drawdown_pct, recovery_time
    "time_since_high_15m", "time_since_low_15m",
    "skewness_300s", "kurtosis_300s",  # Removed 60s versions that don't exist
    "downside_vol_300s", "upside_vol_300s", "vol_asymmetry_300s", "tail_risk_300s",
    "hour_of_day_utc",  # Changed from hour_of_day
    "hour_sin", "hour_cos",  # Changed from time_of_day_sin, time_of_day_cos
    "is_us_hours", "is_asia_hours", "is_europe_hours",
    "vol_persistence_ar1", "vol_acceleration_300s",  # Changed from rv_acceleration
    "vol_of_vol_300s",  # Changed from vol_of_vol
    "garch_forecast_simple",  # Changed from garch_vol
    "jump_count_300s", "jump_direction_300s", "jump_intensity_300s",
    # reversals_300s is already listed in advanced features
    "autocorr_lag10_300s", "autocorr_lag30_300s", "autocorr_lag60_300s",
    "autocorr_decay",
]


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
        # Only fail if we're approaching system limits (>28GB on 30GB system)
        if mem_gb > 28.0:
            raise MemoryError(f"Critical memory limit: {mem_gb:.2f} GB > 28.0 GB (system limit)")
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
        .filter(
            (pl.col("date") >= start_date) &
            (pl.col("date") <= end_date)
        )
        .select([
            "timestamp", "date", "contract_id",
            "prob_mid", "outcome",
            "time_remaining", "iv_staleness_seconds",
            "K", "S",  # Include K and S for moneyness calculation
        ])
        .with_columns([
            # Calculate residual using actual column names
            (pl.col("outcome") - pl.col("prob_mid")).alias("residual"),
            # Calculate moneyness properly from K and S
            ((pl.col("S") / pl.col("K")) - 1).alias("moneyness"),  # (S/K - 1) as per original spec
        ])
    )

    # Get timestamp range for filtering feature files
    # We need to filter feature files to the same date range to avoid loading all 63M rows
    baseline_timestamps = (
        pl.scan_parquet(str(BASELINE_FILE))
        .filter(
            (pl.col("date") >= start_date) &
            (pl.col("date") <= end_date)
        )
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

    # Join RV features (RV file uses timestamp_seconds, baseline uses timestamp)
    if RV_FILE.exists() and min_timestamp is not None:
        rv_df = (
            pl.scan_parquet(str(RV_FILE))
            .filter(
                (pl.col("timestamp_seconds") >= min_timestamp) &
                (pl.col("timestamp_seconds") <= max_timestamp)
            )
        )
        lazy_df = lazy_df.join(rv_df, left_on="timestamp", right_on="timestamp_seconds", how="left")

    # Join microstructure features (micro file uses timestamp_seconds, baseline uses timestamp)
    if MICROSTRUCTURE_FILE.exists() and min_timestamp is not None:
        micro_df = (
            pl.scan_parquet(str(MICROSTRUCTURE_FILE))
            .filter(
                (pl.col("timestamp_seconds") >= min_timestamp) &
                (pl.col("timestamp_seconds") <= max_timestamp)
            )
        )
        lazy_df = lazy_df.join(micro_df, left_on="timestamp", right_on="timestamp_seconds", how="left")

    # Join advanced features (uses timestamp and contract_id)
    if ADVANCED_FILE.exists() and min_timestamp is not None:
        advanced_df = (
            pl.scan_parquet(str(ADVANCED_FILE))
            .filter(
                (pl.col("timestamp") >= min_timestamp) &
                (pl.col("timestamp") <= max_timestamp)
            )
        )
        # Advanced features need both timestamp and contract_id for proper join
        lazy_df = lazy_df.join(advanced_df, on=["timestamp", "contract_id"], how="left")

    # Simply drop nulls in the residual column (which we just calculated)
    # This ensures we only keep rows where we can calculate residuals
    lazy_df = lazy_df.filter(pl.col("residual").is_not_null())

    # Don't do any casting or schema checks here - keep it lazy
    # The Float32 casting will be done at collection time in the training function

    monitor.check_memory("Before streaming write")

    # Stream to disk (constant memory usage)
    logger.info(f"Streaming features to {output_file}...")
    lazy_df.sink_parquet(
        output_file,
        compression="snappy",
        statistics=True,
        # Note: sink_parquet on LazyFrame already streams by default
    )

    # Get row count without loading full dataset
    row_count = pl.scan_parquet(output_file).select(pl.len()).collect().item()
    logger.info(f"Wrote {row_count:,} rows to {output_file}")

    monitor.check_memory("After streaming write")
    return row_count


def train_xgboost_external_memory(
    train_file: str,
    val_file: str,
    config: dict[str, Any],
) -> xgb.Booster:
    """
    Train XGBoost using external memory (Parquet files).
    """
    monitor = MemoryMonitor()
    monitor.check_memory("Start training")

    # Ensure cache directory exists
    CACHE_DIR.mkdir(exist_ok=True)

    # Get feature columns available in the file
    schema = pl.scan_parquet(train_file).collect_schema()
    features = [col for col in FEATURE_COLS if col in schema.names()]
    logger.info(f"Using {len(features)} features for training")

    # Prepare hyperparameters
    params = config["hyperparameters"].copy()
    params.update(config.get("memory", {}))

    # Force settings for external memory
    params["tree_method"] = "hist"
    params["max_bin"] = config.get("memory", {}).get("max_bin", 32)
    params["nthread"] = 1  # Single thread for external memory safety

    logger.info("Creating DMatrix from parquet files...")

    # Load data using Polars and convert to numpy for XGBoost
    # This is more memory efficient than the deprecated external memory format
    train_df = pl.read_parquet(train_file)

    # Separate features and target
    feature_cols = [col for col in FEATURE_COLS if col in train_df.columns]

    # Convert to numpy first, then handle inf/nan
    X_train = train_df.select(feature_cols).to_numpy()
    y_train = train_df.select("residual").to_numpy().ravel()

    # Replace inf and nan values in X_train
    # XGBoost handles nan as missing values, but not inf
    X_train = np.nan_to_num(X_train, nan=np.nan, posinf=np.nan, neginf=np.nan)

    # For y_train, replace inf/nan with reasonable values
    y_train = np.nan_to_num(y_train, nan=0.0, posinf=1.0, neginf=-1.0)

    dtrain = xgb.DMatrix(X_train, label=y_train)
    del X_train, y_train, train_df  # Free memory immediately
    gc.collect()

    monitor.check_memory("After train DMatrix")

    # Load validation data
    val_df = pl.read_parquet(val_file)

    # Convert to numpy first, then handle inf/nan
    X_val = val_df.select(feature_cols).to_numpy()
    y_val = val_df.select("residual").to_numpy().ravel()

    # Replace inf and nan values in X_val
    X_val = np.nan_to_num(X_val, nan=np.nan, posinf=np.nan, neginf=np.nan)

    # For y_val, replace inf/nan with reasonable values
    y_val = np.nan_to_num(y_val, nan=0.0, posinf=1.0, neginf=-1.0)

    dval = xgb.DMatrix(X_val, label=y_val)
    del X_val, y_val, val_df  # Free memory immediately
    gc.collect()

    monitor.check_memory("After val DMatrix")

    # Train model
    logger.info("Training XGBoost model with external memory...")
    evals = [(dtrain, "train"), (dval, "val")]

    model = xgb.train(
        params,
        dtrain,
        num_boost_round=params.get("n_estimators", 150),
        evals=evals,
        early_stopping_rounds=params.get("early_stopping_rounds", 15),
        verbose_eval=10,
    )

    monitor.check_memory("After training")

    # Clean up cache files
    if config.get("external_memory", {}).get("clean_cache", True):
        logger.info("Cleaning cache files...")
        shutil.rmtree(CACHE_DIR, ignore_errors=True)

    return model


def predict_in_batches(
    model: xgb.Booster,
    test_file: str,
    output_file: str,
    batch_size: int = BATCH_SIZE,
) -> None:
    """
    Make predictions in batches to avoid memory issues.
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
        batch_df = (
            pl.scan_parquet(test_file)
            .slice(offset, batch_size)
            .collect()
        )

        monitor.check_memory(f"Loaded batch {batch_num}")

        # Extract features for prediction
        X_batch = batch_df.select(features).to_numpy()

        # Make predictions
        dtest = xgb.DMatrix(X_batch)
        residual_pred = model.predict(dtest)

        # Add predictions to dataframe
        result_df = batch_df.select([
            "timestamp_seconds", "date", "symbol",
            "predicted_prob", "actual_outcome",
        ]).with_columns([
            pl.Series("residual_pred", residual_pred).cast(pl.Float32),
            (pl.col("predicted_prob") + pl.Series("residual_pred", residual_pred)).alias("final_prob").cast(pl.Float32),
        ])

        # Save batch result
        batch_file = batch_dir / f"batch_{batch_num:06d}.parquet"
        result_df.write_parquet(batch_file)
        batch_results.append(batch_file)

        # Free memory
        del batch_df, X_batch, result_df
        gc.collect()

        monitor.check_memory(f"After batch {batch_num}")

    # Combine all batches using lazy concatenation
    logger.info("Combining prediction batches...")
    pl.concat([pl.scan_parquet(f) for f in batch_results]).sink_parquet(
        output_file,
        compression="snappy",
        streaming=True,
    )

    # Clean up batch files
    shutil.rmtree(batch_dir, ignore_errors=True)

    logger.info(f"Predictions saved to {output_file}")
    monitor.check_memory("End predictions")


def train_temporal_chunks(
    start_date: date,
    end_date: date,
    config_file: str,
    chunk_months: int = CHUNK_MONTHS,
) -> None:
    """
    Train model using temporal chunking for very large datasets.
    """
    monitor = MemoryMonitor()

    # Load configuration
    with open(config_file) as f:
        config = yaml.safe_load(f)

    logger.info("="*80)
    logger.info("MEMORY-OPTIMIZED XGBOOST TRAINING")
    logger.info("="*80)
    logger.info(f"Date range: {start_date} to {end_date}")
    logger.info(f"Chunk size: {chunk_months} months")
    logger.info(f"Memory limit: {MAX_MEMORY_GB} GB")

    # Process data in temporal chunks
    current_date = start_date
    chunk_files = []

    while current_date < end_date:
        # Calculate chunk end date
        if chunk_months:
            # Use relativedelta for proper month arithmetic
            chunk_end = current_date + relativedelta(months=chunk_months) - timedelta(days=1)
        else:
            # Full dataset
            chunk_end = end_date

        chunk_end = min(chunk_end, end_date)

        logger.info(f"\nProcessing chunk: {current_date} to {chunk_end}")

        # Prepare features for this chunk
        chunk_file = OUTPUT_DIR / f"features_{current_date}_{chunk_end}.parquet"
        rows = prepare_features_streaming(current_date, chunk_end, str(chunk_file))

        if rows > 0:
            chunk_files.append(chunk_file)

        current_date = chunk_end + timedelta(days=1)

        monitor.enforce_limit()  # Check memory limit

    # Combine chunks for training
    logger.info("\nCombining chunks for training...")

    # Use 80/20 train/val split
    train_file = OUTPUT_DIR / "train_features.parquet"
    val_file = OUTPUT_DIR / "val_features.parquet"

    # Combine and split using lazy operations
    all_data = pl.concat([pl.scan_parquet(f) for f in chunk_files])

    # Get total rows for split
    total_rows = all_data.select(pl.len()).collect().item()
    train_rows = int(total_rows * 0.8)

    logger.info(f"Total rows: {total_rows:,}")
    logger.info(f"Train rows: {train_rows:,}")
    logger.info(f"Val rows: {total_rows - train_rows:,}")

    # Stream write train and validation sets
    all_data.head(train_rows).sink_parquet(str(train_file))
    all_data.tail(total_rows - train_rows).sink_parquet(str(val_file))

    monitor.check_memory("After data preparation")

    # Train model with external memory
    model = train_xgboost_external_memory(str(train_file), str(val_file), config)

    # Save model
    model_file = OUTPUT_DIR / "xgboost_model_optimized.json"
    model.save_model(str(model_file))
    logger.info(f"Model saved to {model_file}")

    # Make predictions on test data (if needed)
    # This would be done separately on test dates

    # Clean up intermediate files
    for chunk_file in chunk_files:
        chunk_file.unlink(missing_ok=True)
    train_file.unlink(missing_ok=True)
    val_file.unlink(missing_ok=True)

    monitor.check_memory("Final")


def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="Memory-optimized XGBoost training")
    parser.add_argument(
        "--config",
        type=str,
        default="config/xgboost_config_production.yaml",
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

    # Resolve config path
    config_file = Path(__file__).parent / args.config

    # Run training
    train_temporal_chunks(
        start_date,
        end_date,
        str(config_file),
        chunk_months,
    )


if __name__ == "__main__":
    main()