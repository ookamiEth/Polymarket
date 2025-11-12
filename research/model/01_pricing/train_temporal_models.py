#!/usr/bin/env python3
"""
Train 12 Temporal × Volatility Regime Models for V4 Architecture
==================================================================

Trains 12 specialized LightGBM models based on the hierarchical regime structure:
- 3 temporal windows (near <300s, mid 300-900s, far >900s)
- 4 volatility regimes (low_vol_atm, low_vol_otm, high_vol_atm, high_vol_otm)

Each model is trained with regime-specific hyperparameters and walk-forward
cross-validation for time series integrity.

Author: BT Research Team
Date: 2025-11-12
"""

from __future__ import annotations

import logging
import pickle
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import lightgbm as lgb
import numpy as np
import polars as pl
from sklearn.metrics import mean_squared_error

# Add parent directories to path
sys.path.append(str(Path(__file__).parent.parent / "00_data_prep"))
from regime_detection_v4 import detect_hierarchical_regime
from temporal_model_split import get_temporal_model_config

logger = logging.getLogger(__name__)

# Paths
MODEL_DIR = Path(__file__).parent.parent
DATA_DIR = MODEL_DIR / "data"
RESULTS_DIR = MODEL_DIR / "results"
MODELS_DIR = RESULTS_DIR / "models_v4"

# Feature columns (55 aggressive pruned features as per plan)
CORE_FEATURES = [
    # Advanced moneyness (8)
    "log_moneyness", "moneyness_squared", "moneyness_cubed",
    "standardized_moneyness", "moneyness_x_time", "moneyness_x_vol",
    "moneyness_distance", "moneyness_percentile",

    # Time features (5)
    "time_remaining", "time_remaining_ratio", "hour_of_day_utc",
    "day_of_week", "is_market_hours",

    # Core momentum (4)
    "momentum_60s", "momentum_300s", "momentum_900s", "momentum_3600s",

    # Core volatility (8)
    "rv_60s", "rv_300s", "rv_900s", "rv_3600s",
    "hv_900s", "hv_3600s", "iv_filtered", "iv_rv_ratio",

    # Volatility asymmetry (6)
    "downside_vol_300s", "upside_vol_300s", "vol_asymmetry_300s",
    "realized_skewness_300s", "realized_kurtosis_300s", "vol_of_vol",

    # Order book core (8)
    "spread_level_1", "spread_level_2", "spread_level_3",
    "spread_level_4", "spread_level_5",
    "imbalance_ema_60s", "imbalance_ema_300s", "imbalance_ema_900s",

    # Order book normalized (5)
    "spread_relative_to_price_1", "spread_relative_to_price_2",
    "spread_relative_to_price_3", "spread_relative_to_price_4",
    "spread_relative_to_price_5",

    # Volume & OI (4)
    "volume_60s", "volume_300s", "oi_change_60s", "oi_change_300s",

    # Price action (4)
    "range_300s", "range_900s", "price_position_300s", "price_position_900s",

    # Regime detection (3)
    "volatility_regime", "market_regime", "is_extreme_condition",
]


def get_regime_hyperparams(temporal: str, volatility: str) -> dict:
    """
    Get regime-specific LightGBM hyperparameters.

    Different regimes require different model complexity and regularization.

    Args:
        temporal: One of 'near', 'mid', 'far'
        volatility: One of 'low_vol_atm', 'low_vol_otm', 'high_vol_atm', 'high_vol_otm'

    Returns:
        Dictionary of LightGBM hyperparameters
    """
    base_params = {
        "objective": "regression",
        "metric": "rmse",
        "boosting_type": "gbdt",
        "num_leaves": 31,
        "max_depth": -1,
        "learning_rate": 0.05,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.7,
        "bagging_frequency": 5,
        "min_data_in_leaf": 20,
        "lambda_l1": 1.0,
        "lambda_l2": 20.0,
        "verbosity": -1,
        "num_threads": 28,  # Optimized for 32 vCPUs
    }

    # Temporal adjustments
    if temporal == "near":
        # Near expiry: More conservative due to high gamma
        base_params["learning_rate"] = 0.03
        base_params["min_data_in_leaf"] = 50
        base_params["lambda_l2"] = 30.0
        base_params["num_leaves"] = 23
    elif temporal == "far":
        # Far horizon: More complex model for stable regime
        base_params["num_leaves"] = 63
        base_params["min_data_in_leaf"] = 10
        base_params["learning_rate"] = 0.05

    # Volatility adjustments
    if "high_vol" in volatility:
        # High volatility: More regularization
        base_params["lambda_l2"] *= 1.5
        base_params["feature_fraction"] = 0.6
        base_params["bagging_fraction"] = 0.6

    # ATM adjustments
    if "atm" in volatility:
        # ATM: Careful to avoid degradation
        base_params["min_data_in_leaf"] = max(base_params["min_data_in_leaf"], 100)
        base_params["lambda_l2"] *= 1.2

    return base_params


def train_with_walk_forward_cv(
    X: np.ndarray,
    y: np.ndarray,
    params: dict,
    regime_name: str,
    n_folds: int = 10
) -> Tuple[lgb.Booster, Dict[str, float]]:
    """
    Train model with walk-forward cross-validation for time series.

    Uses expanding window training to maintain temporal ordering.

    Args:
        X: Feature matrix
        y: Target values (residuals)
        params: LightGBM parameters
        regime_name: Name of the regime for logging
        n_folds: Number of CV folds

    Returns:
        Tuple of (trained model, performance metrics)
    """
    fold_size = len(X) // (n_folds + 1)
    fold_metrics = []
    fold_mse = []

    logger.info(f"Training {regime_name} with {n_folds}-fold walk-forward CV...")
    logger.info(f"  Total samples: {len(X):,}")
    logger.info(f"  Fold size: {fold_size:,}")

    for fold in range(n_folds):
        # Expanding window
        train_end = (fold + 1) * fold_size
        val_start = train_end
        val_end = min(val_start + fold_size, len(X))

        X_train = X[:train_end]
        y_train = y[:train_end]
        X_val = X[val_start:val_end]
        y_val = y[val_start:val_end]

        # Create datasets
        train_data = lgb.Dataset(X_train, label=y_train)
        val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)

        # Train fold model
        model = lgb.train(
            params,
            train_data,
            num_boost_round=1000,
            valid_sets=[val_data],
            callbacks=[
                lgb.early_stopping(50),
                lgb.log_evaluation(0)  # Suppress iteration output
            ]
        )

        # Evaluate
        pred_val = model.predict(X_val, num_iteration=model.best_iteration)
        mse = mean_squared_error(y_val, pred_val)
        fold_mse.append(mse)

        # Calculate Brier score if we have probabilities
        # Note: Brier = MSE for probability predictions
        brier = mse  # Since we're predicting residuals, this is approximate

        logger.info(f"  Fold {fold + 1}/{n_folds}: MSE={mse:.6f}, RMSE={np.sqrt(mse):.6f}")

    # Train final model on all data
    logger.info(f"Training final model on all {len(X):,} samples...")
    final_train_data = lgb.Dataset(X, label=y)
    final_model = lgb.train(
        params,
        final_train_data,
        num_boost_round=1000,
        callbacks=[lgb.log_evaluation(100)]
    )

    # Compute metrics
    metrics = {
        "mse_mean": np.mean(fold_mse),
        "mse_std": np.std(fold_mse),
        "rmse_mean": np.sqrt(np.mean(fold_mse)),
        "brier_approx": np.mean(fold_mse),  # Approximate Brier
        "n_samples": len(X),
        "n_features": X.shape[1],
        "best_iteration": final_model.best_iteration if hasattr(final_model, 'best_iteration') else final_model.num_trees(),
    }

    return final_model, metrics


def train_all_regime_models(df: pl.DataFrame) -> Tuple[Dict[str, lgb.Booster], Dict[str, dict]]:
    """
    Train all 12 regime-specific models.

    Args:
        df: DataFrame with features and regime assignments

    Returns:
        Tuple of (models dict, performance dict)
    """
    models = {}
    performance = {}

    # Define all 12 regimes
    temporal_regimes = ["near", "mid", "far"]
    vol_regimes = ["low_vol_atm", "low_vol_otm", "high_vol_atm", "high_vol_otm"]

    for temporal in temporal_regimes:
        for vol in vol_regimes:
            combined_regime = f"{temporal}_{vol}"
            logger.info(f"\n{'='*60}")
            logger.info(f"Training model: {combined_regime}")
            logger.info(f"{'='*60}")

            # Filter data for this regime
            regime_df = df.filter(pl.col("combined_regime") == combined_regime)

            if len(regime_df) < 10000:
                logger.warning(f"⚠️ Insufficient data for {combined_regime}: {len(regime_df)} samples")
                logger.warning(f"   Minimum required: 10,000 samples")
                continue

            # Get available features (some might be missing)
            available_features = [f for f in CORE_FEATURES if f in regime_df.columns]
            if len(available_features) < 40:
                logger.warning(f"⚠️ Only {len(available_features)} features available (expected 55)")

            # Get regime-specific hyperparameters
            params = get_regime_hyperparams(temporal, vol)

            # Prepare data
            try:
                X = regime_df.select(available_features).to_numpy()

                # Target is the residual (actual - baseline prediction)
                if "residual" in regime_df.columns:
                    y = regime_df["residual"].to_numpy()
                elif "outcome" in regime_df.columns and "prob_bs" in regime_df.columns:
                    y = (regime_df["outcome"] - regime_df["prob_bs"]).to_numpy()
                else:
                    logger.error(f"❌ No target column found for {combined_regime}")
                    continue

                # Remove any NaN values
                valid_mask = ~(np.isnan(X).any(axis=1) | np.isnan(y))
                X = X[valid_mask]
                y = y[valid_mask]

                if len(X) < 10000:
                    logger.warning(f"⚠️ After removing NaNs: {len(X)} samples")
                    continue

                # Train with walk-forward validation
                model, metrics = train_with_walk_forward_cv(
                    X, y, params,
                    regime_name=combined_regime,
                    n_folds=10
                )

                models[combined_regime] = model
                performance[combined_regime] = metrics

                logger.info(f"✅ {combined_regime} training complete:")
                logger.info(f"   MSE: {metrics['mse_mean']:.6f} ± {metrics['mse_std']:.6f}")
                logger.info(f"   RMSE: {metrics['rmse_mean']:.6f}")
                logger.info(f"   Samples: {metrics['n_samples']:,}")
                logger.info(f"   Features: {metrics['n_features']}")
                logger.info(f"   Best iteration: {metrics['best_iteration']}")

            except Exception as e:
                logger.error(f"❌ Failed to train {combined_regime}: {e}")
                continue

    return models, performance


def save_models(models: Dict[str, lgb.Booster], performance: Dict[str, dict]) -> None:
    """
    Save trained models and performance metrics.

    Args:
        models: Dictionary of regime -> model
        performance: Dictionary of regime -> metrics
    """
    # Create models directory
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    # Save each model
    for regime, model in models.items():
        model_path = MODELS_DIR / f"{regime}.txt"
        model.save_model(str(model_path))
        logger.info(f"Saved model: {model_path}")

    # Save performance metrics
    perf_path = MODELS_DIR / "performance_metrics.pkl"
    with open(perf_path, "wb") as f:
        pickle.dump(performance, f)
    logger.info(f"Saved performance metrics: {perf_path}")

    # Create performance summary
    summary_path = MODELS_DIR / "performance_summary.txt"
    with open(summary_path, "w") as f:
        f.write("V4 Model Performance Summary\n")
        f.write(f"Generated: {datetime.now()}\n")
        f.write("="*60 + "\n\n")

        for regime, metrics in sorted(performance.items()):
            f.write(f"{regime}:\n")
            f.write(f"  MSE: {metrics['mse_mean']:.6f} ± {metrics['mse_std']:.6f}\n")
            f.write(f"  RMSE: {metrics['rmse_mean']:.6f}\n")
            f.write(f"  Samples: {metrics['n_samples']:,}\n")
            f.write(f"  Features: {metrics['n_features']}\n")
            f.write(f"  Iterations: {metrics['best_iteration']}\n\n")

        # Overall statistics
        all_mse = [m["mse_mean"] for m in performance.values()]
        f.write(f"\nOverall Statistics:\n")
        f.write(f"  Models trained: {len(models)}/12\n")
        f.write(f"  Mean MSE: {np.mean(all_mse):.6f}\n")
        f.write(f"  Best MSE: {np.min(all_mse):.6f}\n")
        f.write(f"  Worst MSE: {np.max(all_mse):.6f}\n")

    logger.info(f"Saved performance summary: {summary_path}")


def main():
    """Main training pipeline for V4 12-model architecture."""

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(RESULTS_DIR / "train_temporal_models.log")
        ]
    )

    logger.info("="*60)
    logger.info("V4 TEMPORAL MODEL TRAINING PIPELINE")
    logger.info("="*60)

    # Load V4 features
    features_path = DATA_DIR / "consolidated_features_v4.parquet"

    if not features_path.exists():
        # Try V4b path (aggressive pruning)
        features_path = DATA_DIR / "consolidated_features_v4b.parquet"

    if not features_path.exists():
        logger.error(f"❌ Features file not found: {features_path}")
        logger.error("   Please run engineer_all_features_v4.py first")
        return

    logger.info(f"Loading features from {features_path}...")
    df = pl.read_parquet(features_path)
    logger.info(f"Loaded {len(df):,} rows with {len(df.columns)} columns")

    # Detect hierarchical regimes (12 categories)
    logger.info("Detecting 12 hierarchical regimes...")
    df = detect_hierarchical_regime(df)

    # Print regime distribution
    regime_counts = df.group_by("combined_regime").agg(
        pl.len().alias("count")
    ).sort("combined_regime")

    logger.info("Regime distribution:")
    for row in regime_counts.iter_rows():
        regime, count = row
        pct = count / len(df) * 100
        logger.info(f"  {regime}: {count:,} samples ({pct:.1f}%)")

    # Train all 12 models
    logger.info("\nStarting model training...")
    models, performance = train_all_regime_models(df)

    # Save models and results
    logger.info("\nSaving models and performance metrics...")
    save_models(models, performance)

    # Final summary
    logger.info("\n" + "="*60)
    logger.info("TRAINING COMPLETE")
    logger.info("="*60)
    logger.info(f"Models trained: {len(models)}/12")

    if len(models) < 12:
        missing = 12 - len(models)
        logger.warning(f"⚠️ {missing} models could not be trained due to insufficient data")

    # Calculate expected Brier improvement
    baseline_brier = 0.1340  # V3 baseline
    if performance:
        avg_mse = np.mean([m["mse_mean"] for m in performance.values()])
        expected_brier = baseline_brier - avg_mse  # Approximate
        improvement = (baseline_brier - expected_brier) / baseline_brier * 100
        logger.info(f"Expected Brier: ~{expected_brier:.4f}")
        logger.info(f"Expected improvement: ~{improvement:.1f}%")

    logger.info(f"\nModels saved to: {MODELS_DIR}")
    logger.info("Next step: Run production_backtest_v4.py with trained models")


if __name__ == "__main__":
    main()