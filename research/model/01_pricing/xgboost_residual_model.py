#!/usr/bin/env python3
"""
Tier 3: XGBoost Residual Model

Improves upon Tier 2 linear model by capturing non-linear feature interactions
using gradient boosting.

Approach:
1. Use baseline BS-IV predictions as foundation
2. Calculate residuals = actual_outcome - predicted_probability
3. Train XGBoost to predict residuals with careful regularization
4. Final prediction = baseline_prob + predicted_residual

Expected improvement over Tier 2: 2-6% additional Brier reduction

Author: BT Research Team
Date: 2025-10-29
"""

from __future__ import annotations

import logging
import sys
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
OUTPUT_DIR = MODEL_DIR / "results"

# Feature configuration (same as Tier 2)
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


def load_and_prepare_data(pilot: bool = False) -> pl.DataFrame:
    """
    Load baseline predictions and join with features.

    Args:
        pilot: If True, filter to October 2023 only

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

    if pilot:
        logger.info("Filtering to October 2023 pilot...")
        from datetime import date

        df = df.filter((pl.col("date") >= date(2023, 10, 1)) & (pl.col("date") <= date(2023, 10, 31)))
        logger.info(f"Pilot data: {len(df):,} rows")

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


def train_xgboost_model(
    df: pl.DataFrame,
    n_splits: int = 5,
    max_depth: int = 4,
    learning_rate: float = 0.05,
    n_estimators: int = 100,
    subsample: float = 0.8,
    colsample_bytree: float = 0.8,
) -> tuple[xgb.XGBRegressor, dict]:
    """
    Train XGBoost model with time-series cross-validation.

    Args:
        df: DataFrame with features and residuals
        n_splits: Number of CV splits
        max_depth: Maximum tree depth (regularization)
        learning_rate: Learning rate (eta)
        n_estimators: Number of boosting rounds
        subsample: Row sampling ratio
        colsample_bytree: Column sampling ratio

    Returns:
        (model, cv_results)
    """
    logger.info("=" * 80)
    logger.info("TRAINING XGBOOST RESIDUAL MODEL")
    logger.info("=" * 80)

    # Sort by timestamp for time-series CV
    df = df.sort("timestamp")

    # Extract features and target
    X = df.select(ALL_FEATURES).to_numpy()  # noqa: N806
    y = df["residual"].to_numpy()

    logger.info(f"Features: {len(ALL_FEATURES)}")
    logger.info(f"Samples: {len(X):,}")
    logger.info("Hyperparameters:")
    logger.info(f"  max_depth: {max_depth}")
    logger.info(f"  learning_rate: {learning_rate}")
    logger.info(f"  n_estimators: {n_estimators}")
    logger.info(f"  subsample: {subsample}")
    logger.info(f"  colsample_bytree: {colsample_bytree}")

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

    # Train final model on all data
    logger.info("\nTraining final model on all data...")
    final_model = xgb.XGBRegressor(
        max_depth=max_depth,
        learning_rate=learning_rate,
        n_estimators=n_estimators,
        subsample=subsample,
        colsample_bytree=colsample_bytree,
        objective="reg:squarederror",
        random_state=42,
        n_jobs=-1,
    )
    final_model.fit(X, y, verbose=False)

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


def apply_xgboost_corrections(
    df: pl.DataFrame,
    model: xgb.XGBRegressor,
) -> pl.DataFrame:
    """
    Apply XGBoost residual corrections to baseline predictions.

    Args:
        df: DataFrame with baseline predictions and features
        model: Trained XGBoost model

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
    args = parser.parse_args()

    logger.info("=" * 80)
    logger.info("TIER 3: XGBOOST RESIDUAL MODEL")
    logger.info("=" * 80)
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

    logger.info("=" * 80)
    logger.info("COMPLETE")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
