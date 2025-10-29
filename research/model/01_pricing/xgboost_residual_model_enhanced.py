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

import logging
import shutil
import sys
import tempfile
from collections.abc import Callable
from pathlib import Path

import numpy as np
import polars as pl
import xgboost as xgb
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


def load_and_prepare_data(
    start_date: str | None = None,
    end_date: str | None = None,
    pilot: bool = False,
) -> pl.DataFrame:
    """
    Load baseline predictions and join with features.

    Args:
        start_date: Start date for filtering (YYYY-MM-DD format)
        end_date: End date for filtering (YYYY-MM-DD format)
        pilot: If True, filter to October 2023 only (overrides start/end dates)

    Returns:
        DataFrame with predictions, features, and residuals
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
    df = df.join(rv, left_on="timestamp", right_on="timestamp_seconds", how="left")

    # Join microstructure features (LAZY - no .collect()!)
    logger.info("Joining microstructure features...")
    micro = pl.scan_parquet(MICROSTRUCTURE_FILE).select(["timestamp_seconds"] + MICROSTRUCTURE_FEATURES)
    df = df.join(micro, left_on="timestamp", right_on="timestamp_seconds", how="left")

    # Join advanced features (LAZY - no .collect()!)
    logger.info("Joining advanced features...")
    advanced = pl.scan_parquet(ADVANCED_FILE).select(["timestamp"] + ADVANCED_FEATURES)
    df = df.join(advanced, on="timestamp", how="left")

    # Filter to complete cases (still lazy until final collect)
    # Build combined filter expression first, then apply, then collect once at the end
    logger.info("Filtering to complete cases (using lazy evaluation for memory safety)...")

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

    # NOW collect once (Polars streaming engine processes entire lazy plan automatically)
    logger.info("Collecting data with streaming engine...")
    df = df_filtered.collect()

    # Report statistics
    final_count = len(df)
    logger.info(f"Complete cases: {final_count:,}")

    # Report residual statistics
    residual_stats = df.select(
        [
            pl.col("residual").mean().alias("mean"),
            pl.col("residual").std().alias("std"),
            pl.col("residual").min().alias("min"),
            pl.col("residual").max().alias("max"),
        ]
    ).to_dicts()[0]

    logger.info(
        f"Residual stats: mean={residual_stats['mean']:.6f}, "
        f"std={residual_stats['std']:.6f}, "
        f"min={residual_stats['min']:.6f}, "
        f"max={residual_stats['max']:.6f}"
    )

    return df


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
    Iterator for XGBoost external memory training.

    Yields batches of data without materializing the full DataFrame in memory.
    Uses polars lazy row access to avoid memory spikes.

    Args:
        df: Polars DataFrame with features and target
        batch_size: Number of rows per batch
    """

    def __init__(self, df: pl.DataFrame, batch_size: int = 2_000_000):
        self.df = df
        self.batch_size = batch_size
        self.n_rows = len(df)
        self.n_batches = (self.n_rows + batch_size - 1) // batch_size
        self.current_batch = 0
        super().__init__()

    def next(self, input_data: Callable) -> int:  # noqa: A003
        """Load next batch into input_data callback."""
        if self.current_batch >= self.n_batches:
            return 0  # Signal end of iteration

        start_idx = self.current_batch * self.batch_size
        end_idx = min(start_idx + self.batch_size, self.n_rows)

        # Extract batch (lazy row access via indexing)
        batch_df = self.df[start_idx:end_idx]
        X_batch = batch_df.select(ALL_FEATURES).to_numpy()  # noqa: N806
        y_batch = batch_df["residual"].to_numpy()

        # Pass to XGBoost via callback
        input_data(data=X_batch, label=y_batch)

        self.current_batch += 1
        return 1  # Signal successful load

    def reset(self) -> None:
        """Reset iterator to beginning."""
        self.current_batch = 0


def train_xgboost_model(
    df: pl.DataFrame,
    n_splits: int = 5,
    max_depth: int = 4,
    learning_rate: float = 0.05,
    n_estimators: int = 100,
    subsample: float = 0.8,
    colsample_bytree: float = 0.8,
) -> tuple[BoosterWrapper, dict]:
    """
    Train XGBoost model with external memory support.

    Uses external memory training (ExtMemQuantileDMatrix) for the final model
    to handle datasets that don't fit in RAM. CV is done on a 20% sample
    to estimate performance while staying within memory limits.

    Args:
        df: DataFrame with features and residuals (sorted by timestamp)
        n_splits: Number of CV splits
        max_depth: Maximum tree depth (regularization)
        learning_rate: Learning rate (eta)
        n_estimators: Number of boosting rounds
        subsample: Row sampling ratio
        colsample_bytree: Column sampling ratio

    Returns:
        (wrapped_booster, cv_results)
    """
    logger.info("=" * 80)
    logger.info("TRAINING XGBOOST RESIDUAL MODEL (EXTERNAL MEMORY)")
    logger.info("=" * 80)

    # Sort by timestamp for time-series CV
    df = df.sort("timestamp")

    n_samples = len(df)
    logger.info(f"Features: {len(ALL_FEATURES)}")
    logger.info(f"Total samples: {n_samples:,}")
    logger.info("Hyperparameters:")
    logger.info(f"  max_depth: {max_depth}")
    logger.info(f"  learning_rate: {learning_rate}")
    logger.info(f"  n_estimators: {n_estimators}")
    logger.info(f"  subsample: {subsample}")
    logger.info(f"  colsample_bytree: {colsample_bytree}")

    # =========================================================================
    # PART 1: Cross-Validation on 20% Sample (In-Memory)
    # =========================================================================
    logger.info("\n" + "=" * 80)
    logger.info("CROSS-VALIDATION (20% SAMPLE)")
    logger.info("=" * 80)

    # Sample 20% of data for CV (stratified by time)
    sample_size = max(int(n_samples * 0.2), 100_000)
    sample_step = max(n_samples // sample_size, 1)
    cv_df = df[::sample_step]

    logger.info(f"CV sample size: {len(cv_df):,} ({len(cv_df) / n_samples * 100:.1f}%)")

    # Extract features for CV
    X_cv = cv_df.select(ALL_FEATURES).to_numpy()  # noqa: N806
    y_cv = cv_df["residual"].to_numpy()

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

        X_train, X_val = X_cv[train_idx], X_cv[val_idx]  # noqa: N806
        y_train, y_val = y_cv[train_idx], y_cv[val_idx]

        # Train model
        model = xgb.XGBRegressor(
            max_depth=max_depth,
            learning_rate=learning_rate,
            n_estimators=n_estimators,
            subsample=subsample,
            colsample_bytree=colsample_bytree,
            objective="reg:squarederror",
            random_state=42,
            n_jobs=-1,
        )
        model.fit(X_train, y_train, verbose=False)

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
    logger.info("NOTE: CV estimates from 20% sample. Full model trained on 100% of data.")

    # Clean up CV arrays
    del X_cv, y_cv, cv_df

    # =========================================================================
    # PART 2: Final Model Training with External Memory
    # =========================================================================
    logger.info("\n" + "=" * 80)
    logger.info("TRAINING FINAL MODEL (EXTERNAL MEMORY - 100% DATA)")
    logger.info("=" * 80)

    cache_dir = None
    try:
        # Create temporary cache directory
        cache_dir = tempfile.mkdtemp(prefix="xgb_cache_")
        cache_prefix = f"{cache_dir}/cache"
        logger.info(f"Using cache: {cache_prefix}")

        # Create iterator for external memory training
        iterator = DataFrameIterator(df, batch_size=2_000_000)

        # Create external memory DMatrix
        logger.info("Creating ExtMemQuantileDMatrix (batched loading)...")
        Xy = xgb.ExtMemQuantileDMatrix(  # noqa: N806
            iterator,
            enable_categorical=False,
            nthread=4,
        )

        # Set parameters for xgb.train API
        params = {
            "max_depth": max_depth,
            "learning_rate": learning_rate,
            "subsample": subsample,
            "colsample_bytree": colsample_bytree,
            "objective": "reg:squarederror",
            "seed": 42,
            "nthread": 4,  # Limit parallelism to control memory
        }

        # Train using Booster API
        logger.info(f"Training booster ({n_estimators} rounds)...")
        booster = xgb.train(
            params,
            Xy,
            num_boost_round=n_estimators,
            verbose_eval=False,
        )

        logger.info("✅ External memory training complete")

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

    finally:
        # Clean up cache directory
        if cache_dir and Path(cache_dir).exists():
            logger.info(f"Cleaning up cache: {cache_dir}")
            shutil.rmtree(cache_dir)


def apply_xgboost_corrections(
    df: pl.DataFrame,
    model: xgb.XGBRegressor | BoosterWrapper,
) -> pl.DataFrame:
    """
    Apply XGBoost residual corrections to baseline predictions.

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
    """Main execution function."""
    import argparse

    parser = argparse.ArgumentParser(description="Train and evaluate XGBoost residual model")
    parser.add_argument("--pilot", action="store_true", help="Use October 2023 pilot data only")
    parser.add_argument("--max-depth", type=int, default=4, help="Maximum tree depth")
    parser.add_argument("--learning-rate", type=float, default=0.05, help="Learning rate (eta)")
    parser.add_argument("--n-estimators", type=int, default=100, help="Number of boosting rounds")
    parser.add_argument("--n-splits", type=int, default=5, help="Number of CV splits")
    # Train/val/test split arguments
    parser.add_argument("--train-start", type=str, help="Train start date (YYYY-MM-DD)")
    parser.add_argument("--train-end", type=str, help="Train end date (YYYY-MM-DD)")
    parser.add_argument("--val-start", type=str, help="Validation start date (YYYY-MM-DD)")
    parser.add_argument("--val-end", type=str, help="Validation end date (YYYY-MM-DD)")
    parser.add_argument("--test-start", type=str, help="Test start date (YYYY-MM-DD)")
    parser.add_argument("--test-end", type=str, help="Test end date (YYYY-MM-DD)")
    args = parser.parse_args()

    logger.info("=" * 80)
    logger.info("TIER 3: XGBOOST RESIDUAL MODEL")
    logger.info("=" * 80)

    # Check if using train/val/test split or legacy single-dataset mode
    use_split = all([args.train_start, args.train_end, args.test_start, args.test_end])

    if use_split:
        logger.info("USING TRAIN/VAL/TEST SPLIT MODE")
        logger.info(f"Train period: {args.train_start} to {args.train_end}")
        if args.val_start and args.val_end:
            logger.info(f"Validation period: {args.val_start} to {args.val_end}")
        logger.info(f"Test period: {args.test_start} to {args.test_end}")
        logger.info(f"Max depth: {args.max_depth}")
        logger.info(f"Learning rate: {args.learning_rate}")
        logger.info(f"N estimators: {args.n_estimators}")
        logger.info(f"CV splits: {args.n_splits}")

        # Load train/val/test datasets
        logger.info("\n" + "=" * 80)
        logger.info("LOADING TRAIN SET")
        df_train = load_and_prepare_data(start_date=args.train_start, end_date=args.train_end)

        if args.val_start and args.val_end:
            logger.info("\n" + "=" * 80)
            logger.info("LOADING VALIDATION SET")
            df_val = load_and_prepare_data(start_date=args.val_start, end_date=args.val_end)

        logger.info("\n" + "=" * 80)
        logger.info("LOADING TEST SET")
        df_test = load_and_prepare_data(start_date=args.test_start, end_date=args.test_end)

        # Train model ONLY on training set
        model, _cv_results = train_xgboost_model(
            df_train,
            n_splits=args.n_splits,
            max_depth=args.max_depth,
            learning_rate=args.learning_rate,
            n_estimators=args.n_estimators,
        )

        # Evaluate on validation set (if provided)
        if args.val_start and args.val_end:
            logger.info("\n" + "=" * 80)
            logger.info("VALIDATION SET PERFORMANCE")
            logger.info("=" * 80)
            df_val = apply_xgboost_corrections(df_val, model)
            _val_results = evaluate_performance(df_val)

        # Evaluate on test set (FINAL HONEST METRICS)
        logger.info("\n" + "=" * 80)
        logger.info("TEST SET PERFORMANCE (FINAL)")
        logger.info("=" * 80)
        df_test = apply_xgboost_corrections(df_test, model)
        _test_results = evaluate_performance(df_test)

        # Save test set results
        output_file = OUTPUT_DIR / f"xgboost_residual_model_test_{args.test_start}_{args.test_end}.parquet"
        logger.info(f"\nSaving test set results to {output_file}...")

        df_test.select(
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

    else:
        # Legacy mode - single dataset (for backward compatibility)
        logger.info("LEGACY MODE - SINGLE DATASET (WARNING: No train/test split!)")
        logger.info(f"Pilot mode: {args.pilot}")
        logger.info(f"Max depth: {args.max_depth}")
        logger.info(f"Learning rate: {args.learning_rate}")
        logger.info(f"N estimators: {args.n_estimators}")
        logger.info(f"CV splits: {args.n_splits}")

        # Load and prepare data
        df = load_and_prepare_data(pilot=args.pilot)

        # Train model
        model, _cv_results = train_xgboost_model(
            df,
            n_splits=args.n_splits,
            max_depth=args.max_depth,
            learning_rate=args.learning_rate,
            n_estimators=args.n_estimators,
        )

        # Apply corrections
        df = apply_xgboost_corrections(df, model)

        # Evaluate
        _results = evaluate_performance(df)

        # Save results
        output_file = OUTPUT_DIR / (
            "xgboost_residual_model_pilot.parquet" if args.pilot else "xgboost_residual_model_full.parquet"
        )
        logger.info(f"\nSaving results to {output_file}...")

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
