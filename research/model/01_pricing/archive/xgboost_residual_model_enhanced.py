#!/usr/bin/env python3
"""
Tier 3.5: XGBoost Residual Model with Advanced Features

Extends Tier 3 by adding 40 advanced features for improved performance.

Feature Count: 64 total
- 24 baseline features (RV, microstructure, context)
- 40 advanced features (EMAs, drawdowns, higher moments, time-of-day, vol clustering, enhanced jumps)

Approach:
1. Use baseline BS-IV predictions as foundation
2. Calculate residuals = actual_outcome - predicted_probability
3. Train XGBoost to predict residuals using 64 features
4. Final prediction = baseline_prob + predicted_residual

Expected improvement over Tier 3: +2-4% additional Brier reduction (targeting +8-10% over baseline)

Author: BT Research Team
Date: 2025-10-29
"""

from __future__ import annotations

import gc
import logging
import os
import sys
from collections.abc import Callable
from pathlib import Path

import numpy as np
import polars as pl
import psutil
import xgboost as xgb
import yaml
from sklearn.model_selection import TimeSeriesSplit

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

# File paths
MODEL_DIR = Path(__file__).parent.parent
BASELINE_FILE = MODEL_DIR / "results/production_backtest_results.parquet"
RV_FILE = MODEL_DIR / "results/realized_volatility_1s.parquet"
MICROSTRUCTURE_FILE = MODEL_DIR / "results/microstructure_features.parquet"
ADVANCED_FILE = MODEL_DIR / "results/advanced_features.parquet"
OUTPUT_DIR = MODEL_DIR / "results"

# Feature configuration: 24 baseline + 40 advanced = 64 total
RV_FEATURES = [
    "rv_60s",
    "rv_300s",
    "rv_900s",
    "rv_3600s",
    "rv_ratio_5m_1m",
    "rv_ratio_15m_5m",
    "rv_ratio_1h_15m",
    "rv_term_structure",
]

MICROSTRUCTURE_FEATURES = [
    "momentum_60s",
    "momentum_300s",
    "momentum_900s",
    "range_60s",
    "range_300s",
    "range_900s",
    "reversals_60s",
    # reversals_300s removed (duplicate - now in ADVANCED_FEATURES)
    "jump_detected",
    # jump_intensity_300s removed (duplicate - now in ADVANCED_FEATURES)
    "autocorr_lag1_300s",
    "autocorr_lag5_300s",
    "hurst_300s",
]

CONTEXT_FEATURES = [
    "time_remaining",
    "iv_staleness_seconds",
    "moneyness",  # (S/K - 1)
]

# Advanced features (40 total)
ADVANCED_FEATURES = [
    # Category 1: EMA & Trend (8)
    "ema_12s",
    "ema_60s",
    "ema_300s",
    "ema_900s",
    "ema_cross_12_60",
    "ema_cross_60_300",
    "ema_cross_300_900",
    "price_vs_ema_900",
    # Category 2: IV/RV Ratios (2)
    "iv_rv_ratio_300s",
    "iv_rv_ratio_900s",
    # Category 3: Drawdown/Run-up (6)
    "high_15m",
    "low_15m",
    "drawdown_from_high_15m",
    "runup_from_low_15m",
    "time_since_high_15m",
    "time_since_low_15m",
    # Category 4: Higher Moments (6)
    "skewness_300s",
    "kurtosis_300s",
    "downside_vol_300s",
    "upside_vol_300s",
    "vol_asymmetry_300s",
    "tail_risk_300s",
    # Category 5: Time-of-Day (6)
    "hour_of_day_utc",
    "hour_sin",
    "hour_cos",
    "is_us_hours",
    "is_asia_hours",
    "is_europe_hours",
    # Category 6: Vol Clustering (4)
    "vol_persistence_ar1",
    "vol_acceleration_300s",
    "vol_of_vol_300s",
    "garch_forecast_simple",
    # Category 7: Enhanced Jump/Autocorr (8)
    "jump_count_300s",
    "jump_direction_300s",
    "autocorr_lag10_300s",
    "autocorr_lag30_300s",
    "autocorr_lag60_300s",
    "autocorr_decay",
    "jump_intensity_300s",
    "reversals_300s",
]

ALL_FEATURES = RV_FEATURES + MICROSTRUCTURE_FEATURES + CONTEXT_FEATURES + ADVANCED_FEATURES


# =================================================================================
# UTILITY FUNCTIONS (Memory Monitoring & Temporal Chunking)
# =================================================================================


def log_memory_usage(label: str) -> None:
    """Log current memory usage for monitoring."""
    process = psutil.Process()
    mem_info = process.memory_info()
    mem_gb = mem_info.rss / (1024**3)
    logger.info(f"[MEMORY] {label}: {mem_gb:.2f} GB")


def get_row_count(lf: pl.LazyFrame) -> int:
    """
    Get row count from LazyFrame without collecting full dataset.

    Args:
        lf: Polars LazyFrame

    Returns:
        Number of rows
    """
    return lf.select(pl.len()).collect().item()


def generate_temporal_chunks(start_date: str, end_date: str, chunk_months: int = 3) -> list[tuple[str, str]]:
    """
    Generate list of temporal chunk boundaries (start, end) for processing large datasets.

    Args:
        start_date: Start date in YYYY-MM-DD format
        end_date: End date in YYYY-MM-DD format
        chunk_months: Number of months per chunk (default: 3 for 16GB RAM safety)

    Returns:
        List of (chunk_start, chunk_end) date tuples as strings
    """
    from datetime import date

    from dateutil.relativedelta import relativedelta

    start = date.fromisoformat(start_date)
    end = date.fromisoformat(end_date)

    chunks = []
    chunk_start = start

    while chunk_start < end:
        chunk_end = min(chunk_start + relativedelta(months=chunk_months), end)
        chunks.append((chunk_start.isoformat(), chunk_end.isoformat()))
        chunk_start = chunk_end

    return chunks


def load_config(config_path: Path | None = None) -> dict:
    """
    Load XGBoost configuration from YAML file.

    Args:
        config_path: Path to config YAML file. If None, uses default.

    Returns:
        Configuration dictionary with memory, hyperparameters, and validation settings
    """
    if config_path is None:
        config_path = Path(__file__).parent / "config" / "xgboost_config.yaml"

    if not config_path.exists():
        logger.warning(f"Config file not found: {config_path}")
        logger.warning("Using default configuration (may cause OOM!)")
        # Return defaults (original parameters - NOT RECOMMENDED)
        return {
            "memory": {
                "max_bin": 256,
                "nthread": 4,
                "max_cached_hist_node": 65536,
                "ref_resource_aware": False,
                "cv_sample_pct": 0.10,
            },
            "hyperparameters": {
                "max_depth": 4,
                "min_child_weight": 1,
                "gamma": 0,
                "reg_lambda": 1,
                "reg_alpha": 0,
                "learning_rate": 0.05,
                "n_estimators": 100,
                "early_stopping_rounds": None,
                "subsample": 0.8,
                "colsample_bytree": 0.8,
                "objective": "reg:squarederror",
                "tree_method": "hist",
                "grow_policy": "depthwise",
                "seed": 42,
            },
            "validation": {
                "use_validation_set": False,
                "val_split": 0.2,
                "monitor_metrics": ["rmse", "mae"],
            },
        }

    logger.info(f"Loading configuration from {config_path}")
    with open(config_path) as f:
        config = yaml.safe_load(f)

    logger.info("Configuration loaded:")
    logger.info(f"  Memory: max_bin={config['memory']['max_bin']}, nthread={config['memory']['nthread']}")
    logger.info(
        f"  Regularization: gamma={config['hyperparameters']['gamma']}, "
        f"lambda={config['hyperparameters']['reg_lambda']}, "
        f"alpha={config['hyperparameters']['reg_alpha']}"
    )
    logger.info(f"  Early stopping: {config['hyperparameters']['early_stopping_rounds']} rounds")

    return config


# =================================================================================
# DATA LOADING (Lazy Evaluation for Memory Safety)
# =================================================================================


def load_and_prepare_data(
    start_date: str | None = None,
    end_date: str | None = None,
    pilot: bool = False,
) -> pl.LazyFrame:
    """
    Load baseline predictions and join with features (LAZY - no memory spike).

    CRITICAL: Returns LazyFrame to avoid materializing 40GB dataset in RAM.
    Caller must either:
    1. Collect in batches (DataFrameIterator)
    2. Use temporal chunking (process 3-month windows)
    3. Stream to disk (.sink_parquet(streaming=True))

    Args:
        start_date: Start date for filtering (YYYY-MM-DD format)
        end_date: End date for filtering (YYYY-MM-DD format)
        pilot: If True, filter to October 2023 only (overrides start/end dates)

    Returns:
        LazyFrame with predictions, features, and residuals (NOT YET COLLECTED)
    """
    logger.info("=" * 80)
    logger.info("LOADING AND PREPARING DATA")
    logger.info("=" * 80)

    # Load baseline predictions LAZILY (critical for large test set)
    logger.info(f"Loading baseline predictions from {BASELINE_FILE}...")
    df = pl.scan_parquet(BASELINE_FILE)

    # Apply date filtering (still lazy)
    if pilot:
        logger.info("Filtering to October 2023 pilot...")
        from datetime import date

        df = df.filter((pl.col("date") >= date(2023, 10, 1)) & (pl.col("date") <= date(2023, 10, 31)))
    elif start_date and end_date:
        logger.info(f"Filtering to date range {start_date} to {end_date}...")
        from datetime import date

        start = date.fromisoformat(start_date)
        end = date.fromisoformat(end_date)
        df = df.filter((pl.col("date") >= start) & (pl.col("date") <= end))

    # Calculate residuals (still lazy)
    logger.info("Calculating residuals...")
    df = df.with_columns([(pl.col("outcome") - pl.col("prob_mid")).alias("residual")])

    # Add moneyness (still lazy)
    df = df.with_columns([(pl.col("S") / pl.col("K") - 1.0).alias("moneyness")])

    # Join RV features (LAZY - no .collect()!)
    logger.info("Joining realized volatility features...")
    rv = pl.scan_parquet(RV_FILE).select(["timestamp_seconds"] + RV_FEATURES)

    # CRITICAL FIX: Filter feature files to match baseline date range BEFORE joining
    # This prevents loading all 63M rows when we only need the filtered subset
    # Calculate timestamp range from date filters (for both pilot and production modes)
    from datetime import datetime

    if pilot:
        # For pilot, filter to October 2023 timestamps
        start_ts = int(datetime(2023, 10, 1).timestamp())
        end_ts = int(datetime(2023, 11, 1).timestamp())  # Nov 1 00:00 (exclusive)
        logger.info(f"Pilot mode: filtering features to October 2023 ({start_ts} to {end_ts})")
    elif start_date and end_date:
        # For production, calculate timestamp range from date arguments
        start_ts = int(datetime.fromisoformat(start_date).timestamp())
        end_ts = int(datetime.fromisoformat(end_date).timestamp())
        logger.info(f"Production mode: filtering features to {start_date} - {end_date} ({start_ts} to {end_ts})")
    else:
        # No filtering if no dates specified (full dataset - rare case)
        start_ts = None
        end_ts = None
        logger.warning("No date filters specified - loading full feature datasets (may cause OOM!)")

    # Apply timestamp filters to ALL feature files (if dates specified)
    if start_ts is not None and end_ts is not None:
        rv = rv.filter((pl.col("timestamp_seconds") >= start_ts) & (pl.col("timestamp_seconds") < end_ts))
        logger.info(f"Filtered RV features to timestamp range")

    df = df.join(rv, left_on="timestamp", right_on="timestamp_seconds", how="left")

    # Join microstructure features (LAZY - no .collect()!)
    logger.info("Joining microstructure features...")
    micro = pl.scan_parquet(MICROSTRUCTURE_FILE).select(["timestamp_seconds"] + MICROSTRUCTURE_FEATURES)

    # Apply same timestamp filter
    if start_ts is not None and end_ts is not None:
        micro = micro.filter((pl.col("timestamp_seconds") >= start_ts) & (pl.col("timestamp_seconds") < end_ts))
        logger.info("Filtered microstructure features to timestamp range")

    df = df.join(micro, left_on="timestamp", right_on="timestamp_seconds", how="left")

    # Join advanced features (LAZY - no .collect()!)
    logger.info("Joining advanced features...")
    advanced = pl.scan_parquet(ADVANCED_FILE).select(["timestamp"] + ADVANCED_FEATURES)

    # Apply same timestamp filter (note: advanced uses "timestamp" not "timestamp_seconds")
    if start_ts is not None and end_ts is not None:
        advanced = advanced.filter((pl.col("timestamp") >= start_ts) & (pl.col("timestamp") < end_ts))
        logger.info("Filtered advanced features to timestamp range")

    df = df.join(advanced, on="timestamp", how="left")

    # Filter to complete cases (still lazy until final collect)
    # Build combined filter expression first, then apply, then collect once at the end
    logger.info("Filtering to complete cases (using lazy evaluation for memory safety)...")

    # Count rows before filtering (lazy count)
    n_before_filter = get_row_count(df)
    logger.info(f"Rows before filtering: {n_before_filter:,}")

    # Boolean features (don't apply .is_finite())
    boolean_features = ["is_us_hours", "is_asia_hours", "is_europe_hours"]
    numeric_features = [f for f in ALL_FEATURES if f not in boolean_features]

    # Build combined filter expression (O(n) single-pass)
    import operator
    from functools import reduce

    null_checks = [pl.col(f).is_not_null() for f in ALL_FEATURES]
    finite_checks = [pl.col(f).is_finite() for f in numeric_features]
    extra_checks = [
        pl.col("prob_mid").is_not_null(),
        pl.col("outcome").is_not_null(),
        pl.col("residual").is_not_null(),
        pl.col("prob_mid").is_finite(),
        pl.col("residual").is_finite(),
    ]

    all_conditions = null_checks + finite_checks + extra_checks
    combined_filter = reduce(operator.and_, all_conditions)

    # Apply filter (still lazy)
    df_filtered = df.filter(combined_filter)

    # Count rows after filtering (lazy count)
    n_after_filter = get_row_count(df_filtered)
    n_removed = n_before_filter - n_after_filter
    pct_removed = (n_removed / n_before_filter * 100) if n_before_filter > 0 else 0
    logger.info(f"Rows after filtering: {n_after_filter:,}")
    logger.info(f"Rows removed: {n_removed:,} ({pct_removed:.1f}%)")

    # Log warning if too many rows were removed
    if pct_removed > 10:
        logger.warning(f"⚠️  High data loss: {pct_removed:.1f}% of rows removed due to nulls/infinities")
        logger.warning("   Consider investigating data quality issues in feature engineering pipeline")

    # RETURN LAZYFRAME (do NOT collect - prevents 40GB memory spike!)
    logger.info("Returning LazyFrame (data NOT yet collected - memory safe)")
    logger.info("NOTE: Caller must handle collection in batches or stream to disk")

    return df_filtered


class BoosterWrapper:
    """
    Wrapper to make xgb.Booster compatible with sklearn-like predict() interface.

    Allows us to use external memory training (xgb.train with Booster API)
    while maintaining compatibility with existing evaluation code that expects
    an sklearn-like model with .predict() method.
    """

    def __init__(self, booster: xgb.Booster):
        self.booster = booster

    def predict(self, X: np.ndarray) -> np.ndarray:  # noqa: N803
        """Predict using the Booster."""
        dmatrix = xgb.DMatrix(X)
        return self.booster.predict(dmatrix)

    def get_booster(self) -> xgb.Booster:
        """Return the underlying Booster for feature importance extraction."""
        return self.booster


class DataFrameIterator(xgb.DataIter):
    """
    Iterator for XGBoost external memory training (XGBoost 3.0+ compatible).

    Implements Python iterator protocol (__iter__, __next__, StopIteration)
    instead of the old callback-based API for compatibility with ExtMemQuantileDMatrix.

    Yields batches of data without materializing the full DataFrame in memory.
    Collects each batch with streaming=True and casts to float32 for memory efficiency.

    CRITICAL: Works on LazyFrame to avoid loading full 40GB dataset into RAM.

    IMPORTANT: LazyFrame must already be sorted by timestamp before passing to this iterator.

    Args:
        lf: Polars LazyFrame with features and target (MUST be pre-sorted by timestamp)
        batch_size: Number of rows per batch (default: 2M = ~1GB per batch)
    """

    def __init__(self, lf: pl.LazyFrame, batch_size: int = 2_000_000):
        # Do NOT sort here - assume data is already sorted to avoid re-sorting for each batch
        self.lf = lf
        self.batch_size = batch_size
        self.n_rows = get_row_count(lf)  # Lazy count (minimal memory)
        self.n_batches = (self.n_rows + batch_size - 1) // batch_size
        self.current_batch = 0
        self._started = False  # Track if iteration has started
        self._data_yielded = False  # Track if we've yielded data in current iteration
        logger.info(f"DataFrameIterator: {self.n_rows:,} rows, {self.n_batches} batches of {batch_size:,}")
        # CRITICAL: Must call parent __init__ with cache_prefix for XGBoost 3.0+
        super().__init__(cache_prefix=os.path.join(".", "xgb_cache"))

    def __iter__(self):
        """Return self as iterator (Python iterator protocol)."""
        self.current_batch = 0
        return self

    def next(self, input_data: callable) -> int:
        """
        XGBoost DataIter abstract method implementation.

        This method is called by XGBoost to fetch the next batch.
        Returns 0 when successful, 1 when no more data.

        NOTE: XGBoost may call this multiple times during initialization
        for quantile sketch building. We need to handle this properly.
        """
        # Initialize on first call if not already started
        if not self._started:
            self._started = True
            self.current_batch = 0
            self._data_yielded = False

        # Check if we have more batches
        if self.current_batch >= self.n_batches:
            # If we never yielded data, there's a problem
            if not self._data_yielded:
                logger.warning("DataFrameIterator: next() called but no data was yielded")
            return 1  # No more data

        start_idx = self.current_batch * self.batch_size
        end_idx = min(start_idx + self.batch_size, self.n_rows)
        logger.info(f"  Loading batch {self.current_batch + 1}/{self.n_batches} (rows {start_idx:,} to {end_idx:,})")

        try:
            # Collect ONLY this batch (constant memory due to small batch size)
            batch_lf = self.lf.slice(start_idx, self.batch_size)
            batch_df = batch_lf.collect()

            # Verify we have data
            if len(batch_df) == 0:
                logger.error(f"Empty batch at index {self.current_batch}")
                return 1

            # Convert to float32 to halve memory (8 bytes -> 4 bytes)
            X_batch = batch_df.select(ALL_FEATURES).cast(pl.Float32).to_numpy()  # noqa: N806
            y_batch = batch_df["residual"].cast(pl.Float32).to_numpy()

            # Clean up DataFrame immediately (keep numpy arrays for return)
            del batch_df
            gc.collect()

            # Use the callback to pass data to XGBoost
            input_data(data=X_batch, label=y_batch)

            self._data_yielded = True
            self.current_batch += 1
            return 0  # Success

        except Exception as e:
            logger.error(f"Error loading batch {self.current_batch}: {e}")
            return 1  # Error

    def __next__(self) -> tuple[np.ndarray, np.ndarray]:
        """Python iterator protocol for compatibility."""
        if self.current_batch >= self.n_batches:
            raise StopIteration

        # Reuse the logic from next() but return directly
        start_idx = self.current_batch * self.batch_size
        batch_lf = self.lf.slice(start_idx, self.batch_size)
        batch_df = batch_lf.collect()
        X_batch = batch_df.select(ALL_FEATURES).cast(pl.Float32).to_numpy()  # noqa: N806
        y_batch = batch_df["residual"].cast(pl.Float32).to_numpy()
        del batch_df
        gc.collect()
        self.current_batch += 1
        return X_batch, y_batch

    def reset(self) -> None:
        """Reset iterator to beginning."""
        self.current_batch = 0
        self._started = False
        self._data_yielded = False


def train_xgboost_model(
    lf: pl.LazyFrame,
    config: dict,
    n_splits: int = 5,
    lf_val: pl.LazyFrame | None = None,
) -> tuple[BoosterWrapper, dict]:
    """
    Train XGBoost model with external memory support and config-based hyperparameters.

    Uses external memory training (ExtMemQuantileDMatrix) for the final model
    to handle datasets that don't fit in RAM. CV is done on a configurable sample
    to estimate performance while staying within memory limits.

    CRITICAL: Accepts LazyFrame to avoid materializing full dataset in RAM.
    The function collects only small batches as needed.

    Args:
        lf: LazyFrame with training features and residuals (will be sorted by timestamp)
        config: Configuration dict with memory, hyperparameters, and validation settings
        n_splits: Number of CV splits
        lf_val: Optional separate validation LazyFrame. If None, uses 80/20 split from lf.

    Returns:
        (wrapped_booster, cv_results)
    """
    logger.info("=" * 80)
    logger.info("TRAINING XGBOOST RESIDUAL MODEL (EXTERNAL MEMORY + CONFIG)")
    logger.info("=" * 80)

    # Extract configuration
    mem_config = config["memory"]
    hp_config = config["hyperparameters"]

    # Sort by timestamp for time-series CV (lazy)
    lf = lf.sort("timestamp")

    n_samples = get_row_count(lf)
    logger.info(f"Features: {len(ALL_FEATURES)}")
    logger.info(f"Total samples: {n_samples:,}")
    logger.info("Hyperparameters (from config):")
    logger.info(f"  max_depth: {hp_config['max_depth']}")
    logger.info(f"  learning_rate: {hp_config['learning_rate']}")
    logger.info(f"  n_estimators: {hp_config['n_estimators']}")
    logger.info(f"  subsample: {hp_config['subsample']}")
    logger.info(f"  colsample_bytree: {hp_config['colsample_bytree']}")
    logger.info(f"  gamma: {hp_config['gamma']}")
    logger.info(f"  reg_lambda: {hp_config['reg_lambda']}")
    logger.info(f"  reg_alpha: {hp_config['reg_alpha']}")
    logger.info(f"  min_child_weight: {hp_config['min_child_weight']}")
    logger.info(f"  early_stopping_rounds: {hp_config['early_stopping_rounds']}")
    logger.info("Memory settings:")
    logger.info(f"  max_bin: {mem_config['max_bin']}")
    logger.info(f"  nthread: {mem_config['nthread']}")
    logger.info(f"  CV sample: {mem_config['cv_sample_pct'] * 100:.1f}%")

    # =========================================================================
    # PART 1: Cross-Validation on Configurable Sample (In-Memory)
    # =========================================================================
    logger.info("\n" + "=" * 80)
    logger.info(f"CROSS-VALIDATION ({mem_config['cv_sample_pct'] * 100:.0f}% SAMPLE)")
    logger.info("=" * 80)

    # Sample data for CV based on config (default: 5% for memory safety on 16GB RAM)
    # 5% (1-2M rows) is statistically sufficient for hyperparameter validation
    sample_size = min(max(int(n_samples * mem_config["cv_sample_pct"]), 50_000), 2_000_000)

    # Collect CV sample from LazyFrame (first sample_size rows)
    cv_lf = lf.slice(0, sample_size)
    cv_df = cv_lf.collect()

    logger.info(f"CV sample size: {len(cv_df):,} ({len(cv_df) / n_samples * 100:.1f}%)")

    # Extract features for CV (cast to float32 to halve memory: 3.2GB → 1.6GB)
    X_cv = cv_df.select(ALL_FEATURES).cast(pl.Float32).to_numpy()  # noqa: N806
    y_cv = cv_df["residual"].cast(pl.Float32).to_numpy()

    # Time-series cross-validation
    tscv = TimeSeriesSplit(n_splits=n_splits)

    cv_results = {
        "train_mse": [],
        "val_mse": [],
        "train_samples": [],
        "val_samples": [],
    }

    logger.info(f"\nRunning {n_splits}-fold time-series cross-validation...")

    for fold, (train_idx, val_idx) in enumerate(tscv.split(X_cv)):
        logger.info(f"\nFold {fold + 1}/{n_splits}")
        logger.info(f"  Train: {len(train_idx):,} samples")
        logger.info(f"  Val:   {len(val_idx):,} samples")
        logger.info("  Starting training for this fold...")

        X_train, X_val = X_cv[train_idx], X_cv[val_idx]  # noqa: N806
        y_train, y_val = y_cv[train_idx], y_cv[val_idx]

        # Train model with config hyperparameters
        model = xgb.XGBRegressor(
            max_depth=hp_config["max_depth"],
            learning_rate=hp_config["learning_rate"],
            n_estimators=hp_config["n_estimators"],
            subsample=hp_config["subsample"],
            colsample_bytree=hp_config["colsample_bytree"],
            gamma=hp_config["gamma"],
            reg_lambda=hp_config["reg_lambda"],
            reg_alpha=hp_config["reg_alpha"],
            min_child_weight=hp_config["min_child_weight"],
            objective=hp_config["objective"],
            random_state=hp_config["seed"],
            n_jobs=-1,  # Use all cores for CV (in-memory, small dataset)
        )
        # Fix: Add eval_set and proper verbose parameter for progress tracking
        model.fit(
            X_train, y_train,
            eval_set=[(X_train, y_train), (X_val, y_val)],
            verbose=10  # Log every 10 rounds to show progress
        )
        logger.info(f"  Training completed for fold {fold + 1}")

        # Evaluate
        train_pred = model.predict(X_train)
        val_pred = model.predict(X_val)

        train_mse = np.mean((y_train - train_pred) ** 2)
        val_mse = np.mean((y_val - val_pred) ** 2)

        cv_results["train_mse"].append(train_mse)
        cv_results["val_mse"].append(val_mse)
        cv_results["train_samples"].append(len(train_idx))
        cv_results["val_samples"].append(len(val_idx))

        logger.info(f"  Train MSE: {train_mse:.6f}")
        logger.info(f"  Val MSE:   {val_mse:.6f}")

    # Report CV summary
    logger.info("\n" + "=" * 80)
    logger.info("CROSS-VALIDATION SUMMARY")
    logger.info("=" * 80)
    logger.info(f"Mean Train MSE: {np.mean(cv_results['train_mse']):.6f} ± {np.std(cv_results['train_mse']):.6f}")
    logger.info(f"Mean Val MSE:   {np.mean(cv_results['val_mse']):.6f} ± {np.std(cv_results['val_mse']):.6f}")
    logger.info("NOTE: CV estimates from 5% sample. Full model trained on 100% of data.")

    # Clean up CV arrays and force garbage collection
    del X_cv, y_cv, cv_df
    gc.collect()
    logger.info("✓ CV memory cleaned up")

    # =========================================================================
    # PART 2: Final Model Training with ExtMemQuantileDMatrix (External Memory)
    # =========================================================================
    logger.info("\n" + "=" * 80)
    logger.info("TRAINING FINAL MODEL (EXTERNAL MEMORY - ExtMemQuantileDMatrix)")
    logger.info("=" * 80)
    logger.info("NOTE: Using external memory for memory-efficient training on large dataset")
    logger.info("      Training with streaming batches to avoid loading all data into RAM")

    # Handle validation data: use provided lf_val or split from training data
    if lf_val is not None:
        # Use provided validation LazyFrame
        n_train = n_samples  # All of lf is training data
        n_val = get_row_count(lf_val)
        logger.info(f"Using separate validation LazyFrame")
        logger.info(f"Train samples: {n_train:,}")
        logger.info(f"Val samples:   {n_val:,}")
        train_lf = lf
        val_lf = lf_val.sort("timestamp")  # Ensure validation is also sorted
    else:
        # Fall back to 80/20 split if no validation LazyFrame provided
        n_train = int(n_samples * 0.8)
        n_val = n_samples - n_train
        logger.info(f"No separate validation provided, using 80/20 split")
        logger.info(f"Train samples: {n_train:,} (80%)")
        logger.info(f"Val samples:   {n_val:,} (20%)")
        train_lf = lf.slice(0, n_train)
        val_lf = lf.slice(n_train, n_val)

    log_memory_usage("Before DMatrix setup")

    # =========================================================================
    # Choose DMatrix Strategy Based on Dataset Size
    # =========================================================================
    # Use standard DMatrix for smaller datasets that fit in memory (<10M rows)
    # Use ExtMemQuantileDMatrix for larger datasets
    USE_EXTERNAL_MEMORY_THRESHOLD = 10_000_000  # 10M rows
    use_external_memory = n_train > USE_EXTERNAL_MEMORY_THRESHOLD

    if not use_external_memory:
        # =====================================================================
        # Standard DMatrix for Small/Medium Datasets (<10M rows)
        # =====================================================================
        logger.info("\nUsing standard DMatrix (dataset fits in memory)")
        logger.info(f"Training samples: {n_train:,} (< {USE_EXTERNAL_MEMORY_THRESHOLD:,} threshold)")

        # Collect training data
        logger.info("Collecting training data...")
        train_df = train_lf.collect()
        X_train = train_df.select(ALL_FEATURES).cast(pl.Float32).to_numpy()
        y_train = train_df["residual"].cast(pl.Float32).to_numpy()
        del train_df
        gc.collect()

        # Create standard training DMatrix
        dtrain = xgb.DMatrix(X_train, label=y_train)
        logger.info("✓ Training DMatrix created")
        log_memory_usage("After training DMatrix")

        # Collect validation data
        logger.info("Collecting validation data...")
        val_df = val_lf.collect()
        X_val = val_df.select(ALL_FEATURES).cast(pl.Float32).to_numpy()
        y_val = val_df["residual"].cast(pl.Float32).to_numpy()
        del val_df
        gc.collect()

        # Create standard validation DMatrix
        dval = xgb.DMatrix(X_val, label=y_val)
        logger.info("✓ Validation DMatrix created")
        log_memory_usage("After validation DMatrix")

    else:
        # =====================================================================
        # ExtMemQuantileDMatrix for Large Datasets (>10M rows)
        # =====================================================================
        logger.info("\nUsing ExtMemQuantileDMatrix (external memory for large dataset)")
        logger.info(f"Training samples: {n_train:,} (> {USE_EXTERNAL_MEMORY_THRESHOLD:,} threshold)")

        # Create Training Iterator
        logger.info("\nCreating training data iterator...")
        train_iterator = DataFrameIterator(train_lf, batch_size=2_000_000)

        # Create ExtMemQuantileDMatrix for training
        logger.info("Creating ExtMemQuantileDMatrix for training (external memory)...")
        logger.info("This will process data in batches and cache to disk...")

        # Create training DMatrix with external memory
        dtrain = xgb.ExtMemQuantileDMatrix(
            train_iterator,
            max_bin=mem_config.get("max_bin", 32),
            missing=np.nan,
            nthread=mem_config.get("nthread", 1),
        )
        logger.info("✓ Training ExtMemQuantileDMatrix created")
        log_memory_usage("After training ExtMemQuantileDMatrix")

        # Create Validation Iterator
        logger.info("\nCreating validation data iterator...")
        val_iterator = DataFrameIterator(val_lf, batch_size=2_000_000)

        # Create ExtMemQuantileDMatrix for validation
        logger.info("Creating ExtMemQuantileDMatrix for validation...")
        logger.info("Using training quantiles for consistency...")

        # Create validation DMatrix using training quantiles
        dval = xgb.ExtMemQuantileDMatrix(
            val_iterator,
            ref=dtrain,  # Use training quantiles for consistency
            max_bin=mem_config.get("max_bin", 32),
            missing=np.nan,
            nthread=mem_config.get("nthread", 1),
        )
        logger.info("✓ Validation ExtMemQuantileDMatrix created")
        log_memory_usage("After validation ExtMemQuantileDMatrix")

    # =========================================================================
    # Train with Early Stopping
    # =========================================================================
    logger.info("\n" + "=" * 80)
    logger.info("TRAINING WITH EARLY STOPPING")
    logger.info("=" * 80)

    # Set parameters for xgb.train API (with full regularization)
    params = {
        # Tree structure
        "max_depth": hp_config["max_depth"],
        "min_child_weight": hp_config["min_child_weight"],
        # Regularization
        "gamma": hp_config["gamma"],
        "reg_lambda": hp_config["reg_lambda"],
        "reg_alpha": hp_config["reg_alpha"],
        # Learning
        "learning_rate": hp_config["learning_rate"],
        # Sampling
        "subsample": hp_config["subsample"],
        "colsample_bytree": hp_config["colsample_bytree"],
        # Other
        "objective": hp_config["objective"],
        "seed": hp_config["seed"],
        # Memory optimization
        "nthread": mem_config["nthread"],
        "tree_method": hp_config["tree_method"],
        "grow_policy": hp_config["grow_policy"],
        "max_bin": mem_config["max_bin"],
        "max_cached_hist_node": mem_config["max_cached_hist_node"],
    }

    logger.info("Training parameters:")
    for key, value in params.items():
        logger.info(f"  {key}: {value}")

    # Create evaluation list for monitoring
    evals = [(dtrain, "train"), (dval, "val")]

    # Train with early stopping
    logger.info(
        f"\nTraining with early stopping (max {hp_config['n_estimators']} rounds, "
        f"early_stopping_rounds={hp_config['early_stopping_rounds']})..."
    )
    logger.info("Monitoring: RMSE and MAE on both train and validation sets")

    booster = xgb.train(
        params,
        dtrain,
        num_boost_round=hp_config["n_estimators"],
        evals=evals,
        early_stopping_rounds=hp_config["early_stopping_rounds"],
        verbose_eval=10,  # Log every 10 rounds
    )

    # Get best iteration info
    best_iteration = booster.best_iteration
    best_score = booster.best_score
    logger.info("\n✅ Training complete")
    logger.info(f"Best iteration: {best_iteration} (out of max {hp_config['n_estimators']})")
    logger.info(f"Best validation score: {best_score:.6f}")

    # Clean up ExtMemQuantileDMatrix objects and iterators
    del dtrain, dval, train_iterator, val_iterator
    gc.collect()
    log_memory_usage("After training complete")

    # Wrap booster for sklearn-like interface
    final_model = BoosterWrapper(booster)

    # Report feature importances
    logger.info("\nFeature importances (top 10 by gain):")
    importances = final_model.get_booster().get_score(importance_type="gain")

    # Map feature indices to names
    feature_importance = []
    for i, feature_name in enumerate(ALL_FEATURES):
        feat_key = f"f{i}"
        if feat_key in importances:
            feature_importance.append((feature_name, importances[feat_key]))

    feature_importance.sort(key=lambda x: x[1], reverse=True)

    for feat, importance in feature_importance[:10]:
        logger.info(f"  {feat}: {importance:.2f}")

    return final_model, cv_results


def apply_xgboost_corrections_streaming(
    lf: pl.LazyFrame,
    model: BoosterWrapper,
    output_path: str,
    batch_size: int = 2_000_000,
) -> str:
    """
    Apply XGBoost residual corrections with streaming I/O (MEMORY-SAFE).

    CRITICAL: Processes predictions in batches and streams to disk.
    Prevents 20-32GB materialization that causes OOM on this machine.

    Args:
        lf: LazyFrame with baseline predictions and features
        model: Trained XGBoost model (BoosterWrapper)
        output_path: Where to save corrected predictions (.parquet)
        batch_size: Rows per batch (default 2M = ~1GB memory per batch)

    Returns:
        Path to output file
    """
    logger.info("=" * 80)
    logger.info("APPLYING XGBOOST CORRECTIONS (STREAMING)")
    logger.info("=" * 80)

    log_memory_usage("Before predictions")

    # Get row count
    n_rows = get_row_count(lf)
    n_batches = (n_rows + batch_size - 1) // batch_size
    logger.info(f"Processing {n_rows:,} rows in {n_batches} batches")

    # Process in batches, write to temp files
    batch_files = []
    for batch_idx in range(n_batches):
        start_idx = batch_idx * batch_size

        logger.info(f"Batch {batch_idx + 1}/{n_batches}: rows {start_idx:,} to {start_idx + batch_size:,}")

        # Load batch
        batch_lf = lf.slice(start_idx, batch_size)
        batch_df = batch_lf.collect()

        # Predict residuals (float32 for memory efficiency)
        X = batch_df.select(ALL_FEATURES).cast(pl.Float32).to_numpy()  # noqa: N806
        residual_pred = model.predict(X)

        # Add predictions to batch
        batch_df = batch_df.with_columns([pl.Series("residual_pred_xgb", residual_pred, dtype=pl.Float32)])

        # Apply corrections
        batch_df = batch_df.with_columns(
            [(pl.col("prob_mid") + pl.col("residual_pred_xgb")).alias("prob_corrected_xgb")]
        )

        # Clip to [0, 1]
        batch_df = batch_df.with_columns([pl.col("prob_corrected_xgb").clip(0.0, 1.0).alias("prob_corrected_xgb")])

        # Write batch to temp file
        temp_path = f"{output_path}.batch_{batch_idx:04d}.parquet"
        batch_df.write_parquet(temp_path, compression="snappy")
        batch_files.append(temp_path)

        # Log stats for this batch
        mean_correction = batch_df["residual_pred_xgb"].mean()
        logger.info(f"  Mean correction: {mean_correction:.6f}")

        # Cleanup
        del batch_df, X, residual_pred
        gc.collect()

    # Combine batches with lazy concat + streaming sink
    logger.info("Combining batches with streaming write...")
    combined = pl.concat([pl.scan_parquet(f) for f in batch_files])
    combined.sink_parquet(output_path, compression="snappy")

    # Cleanup temp files
    for f in batch_files:
        Path(f).unlink()

    log_memory_usage("After predictions")
    logger.info(f"✅ Predictions saved to {output_path}")

    return output_path


def apply_xgboost_corrections(
    df: pl.DataFrame,
    model: xgb.XGBRegressor | BoosterWrapper,
) -> pl.DataFrame:
    """
    Apply XGBoost residual corrections to baseline predictions (IN-MEMORY VERSION).

    WARNING: Only use for pilot mode or datasets <10M rows (<5GB).
    For production datasets (>10M rows), use apply_xgboost_corrections_streaming()
    instead to avoid OOM crashes on this machine (16GB RAM).

    Args:
        df: DataFrame with baseline predictions and features
        model: Trained XGBoost model (XGBRegressor or BoosterWrapper)

    Returns:
        DataFrame with corrected predictions
    """
    logger.info("=" * 80)
    logger.info("APPLYING XGBOOST CORRECTIONS")
    logger.info("=" * 80)

    # Extract features
    X = df.select(ALL_FEATURES).to_numpy()  # noqa: N806

    # Predict residuals
    residual_pred = model.predict(X)

    # Apply corrections
    df = df.with_columns([pl.Series("residual_pred_xgb", residual_pred)])

    df = df.with_columns([(pl.col("prob_mid") + pl.col("residual_pred_xgb")).alias("prob_corrected_xgb")])

    # Clip to [0, 1]
    df = df.with_columns([pl.col("prob_corrected_xgb").clip(0.0, 1.0).alias("prob_corrected_xgb")])

    # Report correction statistics
    correction_stats = df.select(
        [
            pl.col("residual_pred_xgb").mean().alias("mean_correction"),
            pl.col("residual_pred_xgb").std().alias("std_correction"),
            pl.col("residual_pred_xgb").min().alias("min_correction"),
            pl.col("residual_pred_xgb").max().alias("max_correction"),
            (pl.col("prob_corrected_xgb") - pl.col("prob_mid")).abs().mean().alias("mean_abs_change"),
        ]
    ).to_dicts()[0]

    logger.info(f"Mean correction: {correction_stats['mean_correction']:.6f}")
    logger.info(f"Std correction:  {correction_stats['std_correction']:.6f}")
    logger.info(f"Min correction:  {correction_stats['min_correction']:.6f}")
    logger.info(f"Max correction:  {correction_stats['max_correction']:.6f}")
    logger.info(f"Mean abs change: {correction_stats['mean_abs_change']:.6f}")

    return df


def evaluate_performance(df: pl.DataFrame) -> dict:
    """
    Evaluate baseline vs XGBoost corrected performance.

    Args:
        df: DataFrame with both baseline and corrected predictions

    Returns:
        Dictionary with performance metrics
    """
    logger.info("=" * 80)
    logger.info("PERFORMANCE EVALUATION")
    logger.info("=" * 80)

    # Brier scores
    baseline_brier_val = ((df["prob_mid"] - df["outcome"]) ** 2).mean()
    baseline_brier = float(baseline_brier_val) if baseline_brier_val is not None else 0.0  # type: ignore[arg-type]

    xgb_brier_val = ((df["prob_corrected_xgb"] - df["outcome"]) ** 2).mean()
    xgb_brier = float(xgb_brier_val) if xgb_brier_val is not None else 0.0  # type: ignore[arg-type]

    improvement_pct = (baseline_brier - xgb_brier) / baseline_brier * 100

    logger.info(f"Baseline Brier:        {baseline_brier:.6f}")
    logger.info(f"XGBoost Brier:         {xgb_brier:.6f}")
    logger.info(f"Improvement:           {improvement_pct:+.2f}%")

    results = {
        "baseline_brier": baseline_brier,
        "xgb_brier": xgb_brier,
        "improvement_pct": improvement_pct,
    }

    return results


def main() -> None:
    """Main execution function with config-based hyperparameters."""
    import argparse

    parser = argparse.ArgumentParser(description="Train and evaluate XGBoost residual model (config-based)")
    parser.add_argument("--pilot", action="store_true", help="Use October 2023 pilot data only")
    parser.add_argument("--config", type=str, help="Path to YAML config file (default: config/xgboost_config.yaml)")
    parser.add_argument("--n-splits", type=int, default=5, help="Number of CV splits")
    # Train/val/test split arguments
    parser.add_argument("--train-start", type=str, help="Train start date (YYYY-MM-DD)")
    parser.add_argument("--train-end", type=str, help="Train end date (YYYY-MM-DD)")
    parser.add_argument("--val-start", type=str, help="Validation start date (YYYY-MM-DD)")
    parser.add_argument("--val-end", type=str, help="Validation end date (YYYY-MM-DD)")
    parser.add_argument("--test-start", type=str, help="Test start date (YYYY-MM-DD)")
    parser.add_argument("--test-end", type=str, help="Test end date (YYYY-MM-DD)")
    args = parser.parse_args()

    # Load configuration
    config_path = Path(args.config) if args.config else None
    config = load_config(config_path)

    logger.info("=" * 80)
    logger.info("TIER 3.5: XGBOOST RESIDUAL MODEL (MEMORY-SAFE VERSION)")
    logger.info("=" * 80)
    log_memory_usage("Startup")

    # Check if using train/val/test split or legacy single-dataset mode
    use_split = all([args.train_start, args.train_end, args.val_start, args.val_end, args.test_start, args.test_end])

    if use_split:
        # ==================================================================
        # PRODUCTION MODE: Train full model, stream predictions
        # ==================================================================
        logger.info("PRODUCTION MODE: Training on full period, streaming predictions")
        logger.info(f"Train period: {args.train_start} to {args.train_end}")
        logger.info(f"Val period:   {args.val_start} to {args.val_end}")
        logger.info(f"Test period:  {args.test_start} to {args.test_end}")
        # Hyperparameters already logged during config load

        # ==============================
        # PHASE 1: Load training and validation data (LazyFrame)
        # ==============================
        logger.info("\n" + "=" * 80)
        logger.info("PHASE 1: LOADING TRAINING AND VALIDATION DATA")
        logger.info("=" * 80)

        # Load training data
        lf_train = load_and_prepare_data(start_date=args.train_start, end_date=args.train_end)
        train_rows = get_row_count(lf_train)
        logger.info(f"Training data: {train_rows:,} rows (LazyFrame - not yet in memory)")

        # Load validation data separately for temporal split
        lf_val = load_and_prepare_data(start_date=args.val_start, end_date=args.val_end)
        val_rows = get_row_count(lf_val)
        logger.info(f"Validation data: {val_rows:,} rows (LazyFrame - not yet in memory)")
        log_memory_usage("After loading training and validation LazyFrames")

        # ==============================
        # PHASE 2: Train model (external memory)
        # ==============================
        logger.info("\n" + "=" * 80)
        logger.info("PHASE 2: TRAINING MODEL WITH TEMPORAL VALIDATION")
        logger.info("=" * 80)

        model, _cv_results = train_xgboost_model(
            lf_train,
            config=config,
            n_splits=args.n_splits,
            lf_val=lf_val,  # Pass separate validation LazyFrame
        )

        # Cleanup training and validation data
        del lf_train, lf_val
        gc.collect()
        log_memory_usage("After training (training and validation data freed)")

        # ==============================
        # PHASE 3: Load test data (LazyFrame)
        # ==============================
        logger.info("\n" + "=" * 80)
        logger.info("PHASE 3: LOADING TEST DATA")
        logger.info("=" * 80)

        lf_test = load_and_prepare_data(start_date=args.test_start, end_date=args.test_end)
        test_rows = get_row_count(lf_test)
        logger.info(f"Test data: {test_rows:,} rows (LazyFrame - not yet in memory)")
        log_memory_usage("After loading test LazyFrame")

        # ==============================
        # PHASE 4: Stream predictions to disk
        # ==============================
        logger.info("\n" + "=" * 80)
        logger.info("PHASE 4: STREAMING PREDICTIONS TO DISK")
        logger.info("=" * 80)

        output_file = OUTPUT_DIR / f"tier35_test_results_{args.test_start}_{args.test_end}.parquet"
        result_path = apply_xgboost_corrections_streaming(lf_test, model, str(output_file), batch_size=2_000_000)

        # ==============================
        # PHASE 5: Compute final metrics (lazy aggregation)
        # ==============================
        logger.info("\n" + "=" * 80)
        logger.info("PHASE 5: COMPUTING FINAL METRICS")
        logger.info("=" * 80)

        # Scan result file lazily
        result_lf = pl.scan_parquet(result_path)

        # Compute metrics with lazy aggregation
        metrics = result_lf.select(
            [
                pl.len().alias("n_samples"),
                ((pl.col("prob_corrected_xgb") - pl.col("outcome")) ** 2).mean().alias("brier_xgb"),
                ((pl.col("prob_mid") - pl.col("outcome")) ** 2).mean().alias("brier_baseline"),
            ]
        ).collect()

        n_samples = metrics["n_samples"][0]
        brier_xgb = metrics["brier_xgb"][0]
        brier_baseline = metrics["brier_baseline"][0]
        improvement_pct = (brier_baseline - brier_xgb) / brier_baseline * 100

        logger.info("=" * 80)
        logger.info("FINAL TEST SET RESULTS")
        logger.info("=" * 80)
        logger.info(f"Samples:           {n_samples:,}")
        logger.info(f"Baseline Brier:    {brier_baseline:.6f}")
        logger.info(f"XGBoost Brier:     {brier_xgb:.6f}")
        logger.info(f"Improvement:       {improvement_pct:+.2f}%")
        logger.info(f"Results saved to:  {result_path}")

        log_memory_usage("Final")

    else:
        # ==================================================================
        # PILOT MODE: Legacy single-dataset (for backward compatibility)
        # ==================================================================
        logger.info("PILOT MODE: Single dataset (no chunking)")
        logger.info(f"Pilot: {args.pilot}")
        logger.info(f"Config: {config_path if args.config else 'default config/xgboost_config.yaml'}")

        # Load data as LazyFrame
        lf = load_and_prepare_data(pilot=args.pilot)
        n_rows = get_row_count(lf)
        logger.info(f"Dataset: {n_rows:,} rows")

        # For pilot, collect (small dataset)
        logger.info("Collecting pilot data (small dataset - safe to load in memory)...")
        df = lf.collect()
        logger.info(f"Pilot dataset collected: {len(df):,} rows")
        log_memory_usage("After pilot collection")

        # Train model (wrap DataFrame in LazyFrame for consistent API)
        model, _cv_results = train_xgboost_model(
            lf,  # Pass original LazyFrame
            config=config,
            n_splits=args.n_splits,
        )

        # Apply corrections (in-memory OK for pilot)
        df = apply_xgboost_corrections(df, model)

        # Evaluate
        baseline_brier_val = ((df["prob_mid"] - df["outcome"]) ** 2).mean()
        baseline_brier = float(baseline_brier_val) if baseline_brier_val is not None else 0.0  # type: ignore[arg-type]
        xgb_brier_val = ((df["prob_corrected_xgb"] - df["outcome"]) ** 2).mean()
        xgb_brier = float(xgb_brier_val) if xgb_brier_val is not None else 0.0  # type: ignore[arg-type]
        improvement_pct = (baseline_brier - xgb_brier) / baseline_brier * 100

        logger.info("=" * 80)
        logger.info("PILOT RESULTS")
        logger.info("=" * 80)
        logger.info(f"Baseline Brier:    {baseline_brier:.6f}")
        logger.info(f"XGBoost Brier:     {xgb_brier:.6f}")
        logger.info(f"Improvement:       {improvement_pct:+.2f}%")

        # Save results
        output_file = OUTPUT_DIR / "tier35_pilot_results.parquet"
        logger.info(f"\nSaving pilot results to {output_file}...")
        df.select(
            [
                "contract_id",
                "timestamp",
                "seconds_offset",
                "S",
                "K",
                "prob_mid",
                "prob_corrected_xgb",
                "residual",
                "residual_pred_xgb",
                "outcome",
            ]
            + ALL_FEATURES
        ).write_parquet(output_file)

    logger.info("\n" + "=" * 80)
    logger.info("COMPLETE")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
