#!/usr/bin/env python3
"""
Tier 2: Microstructure Residual Model

Improves baseline Black-Scholes predictions by modeling residuals
(actual - predicted) using market microstructure features.

Approach:
1. Use baseline BS-IV predictions as foundation
2. Calculate residuals = actual_outcome - predicted_probability
3. Train linear regression to predict residuals from features
4. Final prediction = baseline_prob + predicted_residual

Features:
- Realized volatility (multi-scale)
- Microstructure (momentum, jumps, reversals, Hurst)
- Context (time remaining, IV staleness, moneyness)

Expected improvement: 10-20% Brier reduction (0.162 → 0.130-0.145)

Author: BT Research Team
Date: 2025-10-29
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import numpy as np
import polars as pl
from sklearn.linear_model import Ridge
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler

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
OUTPUT_DIR = MODEL_DIR / "results"

# Feature configuration
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
    "reversals_300s",
    "jump_detected",
    "jump_intensity_300s",
    "autocorr_lag1_300s",
    "autocorr_lag5_300s",
    "hurst_300s",
]

CONTEXT_FEATURES = [
    "time_remaining",
    "iv_staleness_seconds",
    "moneyness",  # (S/K - 1)
]

ALL_FEATURES = RV_FEATURES + MICROSTRUCTURE_FEATURES + CONTEXT_FEATURES


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

    # Load baseline predictions
    logger.info(f"Loading baseline predictions from {BASELINE_FILE}...")
    df = pl.read_parquet(BASELINE_FILE)
    logger.info(f"Loaded {len(df):,} rows")

    # Apply date filtering
    if pilot:
        logger.info("Filtering to October 2023 pilot...")
        from datetime import date

        df = df.filter((pl.col("date") >= date(2023, 10, 1)) & (pl.col("date") <= date(2023, 10, 31)))
        logger.info(f"Pilot data: {len(df):,} rows")
    elif start_date and end_date:
        logger.info(f"Filtering to date range {start_date} to {end_date}...")
        from datetime import date

        start = date.fromisoformat(start_date)
        end = date.fromisoformat(end_date)
        df = df.filter((pl.col("date") >= start) & (pl.col("date") <= end))
        logger.info(f"Filtered data: {len(df):,} rows")

    # Calculate residuals
    logger.info("Calculating residuals...")
    df = df.with_columns([(pl.col("outcome") - pl.col("prob_mid")).alias("residual")])

    # Add moneyness
    df = df.with_columns([(pl.col("S") / pl.col("K") - 1.0).alias("moneyness")])

    # Join RV features
    logger.info("Joining realized volatility features...")
    rv = pl.scan_parquet(RV_FILE).select(["timestamp_seconds"] + RV_FEATURES)
    df = df.join(rv.collect(), left_on="timestamp", right_on="timestamp_seconds", how="left")

    # Join microstructure features
    logger.info("Joining microstructure features...")
    micro = pl.scan_parquet(MICROSTRUCTURE_FILE).select(["timestamp_seconds"] + MICROSTRUCTURE_FEATURES)
    df = df.join(micro.collect(), left_on="timestamp", right_on="timestamp_seconds", how="left")

    # Filter to complete cases (all features present, no NaN/Inf)
    logger.info("Filtering to complete cases...")
    initial_count = len(df)

    # Remove nulls
    for feature in ALL_FEATURES:
        df = df.filter(pl.col(feature).is_not_null())

    # Remove infinities
    for feature in ALL_FEATURES:
        df = df.filter(pl.col(feature).is_finite())

    df = df.filter(
        pl.col("prob_mid").is_not_null()
        & pl.col("outcome").is_not_null()
        & pl.col("residual").is_not_null()
        & pl.col("prob_mid").is_finite()
        & pl.col("residual").is_finite()
    )

    final_count = len(df)
    logger.info(f"Complete cases: {final_count:,} ({final_count / initial_count * 100:.1f}%)")

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


def train_residual_model(
    df: pl.DataFrame,
    n_splits: int = 5,
    alpha: float = 1.0,
) -> tuple[Ridge, StandardScaler, dict]:
    """
    Train residual model with time-series cross-validation.

    Args:
        df: DataFrame with features and residuals
        n_splits: Number of CV splits
        alpha: Ridge regularization strength

    Returns:
        (model, scaler, cv_results)
    """
    logger.info("=" * 80)
    logger.info("TRAINING RESIDUAL MODEL")
    logger.info("=" * 80)

    # Sort by timestamp for time-series CV
    df = df.sort("timestamp")

    # Extract features and target
    X = df.select(ALL_FEATURES).to_numpy()  # noqa: N806
    y = df["residual"].to_numpy()

    logger.info(f"Features: {len(ALL_FEATURES)}")
    logger.info(f"Samples: {len(X):,}")

    # Time-series cross-validation
    tscv = TimeSeriesSplit(n_splits=n_splits)

    cv_results = {
        "train_mse": [],
        "val_mse": [],
        "train_samples": [],
        "val_samples": [],
    }

    logger.info(f"\nRunning {n_splits}-fold time-series cross-validation...")

    for fold, (train_idx, val_idx) in enumerate(tscv.split(X)):
        logger.info(f"\nFold {fold + 1}/{n_splits}")
        logger.info(f"  Train: {len(train_idx):,} samples")
        logger.info(f"  Val:   {len(val_idx):,} samples")

        X_train, X_val = X[train_idx], X[val_idx]  # noqa: N806
        y_train, y_val = y[train_idx], y[val_idx]

        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)  # noqa: N806
        X_val_scaled = scaler.transform(X_val)  # noqa: N806

        # Train model
        model = Ridge(alpha=alpha)
        model.fit(X_train_scaled, y_train)

        # Evaluate
        train_pred = model.predict(X_train_scaled)
        val_pred = model.predict(X_val_scaled)

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

    # Train final model on all data
    logger.info("\nTraining final model on all data...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)  # noqa: N806

    final_model = Ridge(alpha=alpha)
    final_model.fit(X_scaled, y)

    # Report feature importances (coefficients)
    logger.info("\nFeature coefficients (top 10 by absolute value):")
    coef_df = (
        pl.DataFrame(
            {
                "feature": ALL_FEATURES,
                "coefficient": final_model.coef_,
            }
        )
        .with_columns([pl.col("coefficient").abs().alias("abs_coef")])
        .sort("abs_coef", descending=True)
    )

    print(coef_df.head(10))

    return final_model, scaler, cv_results


def apply_residual_corrections(
    df: pl.DataFrame,
    model: Ridge,
    scaler: StandardScaler,
) -> pl.DataFrame:
    """
    Apply residual corrections to baseline predictions.

    Args:
        df: DataFrame with baseline predictions and features
        model: Trained residual model
        scaler: Feature scaler

    Returns:
        DataFrame with corrected predictions
    """
    logger.info("=" * 80)
    logger.info("APPLYING RESIDUAL CORRECTIONS")
    logger.info("=" * 80)

    # Extract and scale features
    X = df.select(ALL_FEATURES).to_numpy()  # noqa: N806
    X_scaled = scaler.transform(X)  # noqa: N806

    # Predict residuals
    residual_pred = model.predict(X_scaled)

    # Apply corrections
    df = df.with_columns([pl.Series("residual_pred", residual_pred)])

    df = df.with_columns([(pl.col("prob_mid") + pl.col("residual_pred")).alias("prob_corrected")])

    # Clip to [0, 1]
    df = df.with_columns([pl.col("prob_corrected").clip(0.0, 1.0).alias("prob_corrected")])

    # Report correction statistics
    correction_stats = df.select(
        [
            pl.col("residual_pred").mean().alias("mean_correction"),
            pl.col("residual_pred").std().alias("std_correction"),
            pl.col("residual_pred").min().alias("min_correction"),
            pl.col("residual_pred").max().alias("max_correction"),
            (pl.col("prob_corrected") - pl.col("prob_mid")).abs().mean().alias("mean_abs_change"),
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
    Evaluate baseline vs corrected performance.

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

    corrected_brier_val = ((df["prob_corrected"] - df["outcome"]) ** 2).mean()
    corrected_brier = float(corrected_brier_val) if corrected_brier_val is not None else 0.0  # type: ignore[arg-type]

    improvement_pct = (baseline_brier - corrected_brier) / baseline_brier * 100

    logger.info(f"Baseline Brier:   {baseline_brier:.6f}")
    logger.info(f"Corrected Brier:  {corrected_brier:.6f}")
    logger.info(f"Improvement:      {improvement_pct:+.2f}%")

    results = {
        "baseline_brier": baseline_brier,
        "corrected_brier": corrected_brier,
        "improvement_pct": improvement_pct,
    }

    return results


def main() -> None:
    """Main execution function."""
    import argparse

    parser = argparse.ArgumentParser(description="Train and evaluate residual model")
    parser.add_argument("--pilot", action="store_true", help="Use October 2023 pilot data only")
    parser.add_argument("--alpha", type=float, default=1.0, help="Ridge regularization strength")
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
    logger.info("TIER 2: MICROSTRUCTURE RESIDUAL MODEL")
    logger.info("=" * 80)

    # Check if using train/val/test split or legacy single-dataset mode
    use_split = all([args.train_start, args.train_end, args.test_start, args.test_end])

    if use_split:
        logger.info("USING TRAIN/VAL/TEST SPLIT MODE")
        logger.info(f"Train period: {args.train_start} to {args.train_end}")
        if args.val_start and args.val_end:
            logger.info(f"Validation period: {args.val_start} to {args.val_end}")
        logger.info(f"Test period: {args.test_start} to {args.test_end}")
        logger.info(f"Ridge alpha: {args.alpha}")
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
        model, scaler, _cv_results = train_residual_model(
            df_train,
            n_splits=args.n_splits,
            alpha=args.alpha,
        )

        # Evaluate on validation set (if provided)
        if args.val_start and args.val_end:
            logger.info("\n" + "=" * 80)
            logger.info("VALIDATION SET PERFORMANCE")
            logger.info("=" * 80)
            df_val = apply_residual_corrections(df_val, model, scaler)
            _val_results = evaluate_performance(df_val)

        # Evaluate on test set (FINAL HONEST METRICS)
        logger.info("\n" + "=" * 80)
        logger.info("TEST SET PERFORMANCE (FINAL)")
        logger.info("=" * 80)
        df_test = apply_residual_corrections(df_test, model, scaler)
        _test_results = evaluate_performance(df_test)

        # Save test set results
        output_file = OUTPUT_DIR / f"residual_model_test_{args.test_start}_{args.test_end}.parquet"
        logger.info(f"\nSaving test set results to {output_file}...")

        df_test.select(
            [
                "contract_id",
                "timestamp",
                "seconds_offset",
                "S",
                "K",
                "prob_mid",
                "prob_corrected",
                "residual",
                "residual_pred",
                "outcome",
            ]
            + ALL_FEATURES
        ).write_parquet(output_file)

    else:
        # Legacy mode - single dataset (for backward compatibility)
        logger.info("LEGACY MODE - SINGLE DATASET (WARNING: No train/test split!)")
        logger.info(f"Pilot mode: {args.pilot}")
        logger.info(f"Ridge alpha: {args.alpha}")
        logger.info(f"CV splits: {args.n_splits}")

        # Load and prepare data
        df = load_and_prepare_data(pilot=args.pilot)

        # Train model
        model, scaler, _cv_results = train_residual_model(
            df,
            n_splits=args.n_splits,
            alpha=args.alpha,
        )

        # Apply corrections
        df = apply_residual_corrections(df, model, scaler)

        # Evaluate
        _results = evaluate_performance(df)

        # Save results
        output_file = OUTPUT_DIR / ("residual_model_pilot.parquet" if args.pilot else "residual_model_full.parquet")
        logger.info(f"\nSaving results to {output_file}...")

        df.select(
            [
                "contract_id",
                "timestamp",
                "seconds_offset",
                "S",
                "K",
                "prob_mid",
                "prob_corrected",
                "residual",
                "residual_pred",
                "outcome",
            ]
            + ALL_FEATURES
        ).write_parquet(output_file)

    logger.info("\n" + "=" * 80)
    logger.info("COMPLETE")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
