#!/usr/bin/env python3
"""
Temporal Model Split for V4 Architecture
=========================================

Implements the 3-way temporal split based on time_remaining as described in the technical paper.
This is Level 1 of the hierarchical model structure, with volatility regimes as Level 2.

Temporal Buckets:
- Near: < 300s (5 minutes) - High gamma, volatile pricing
- Mid: 300-900s (5-15 minutes) - Balanced dynamics
- Far: > 900s (>15 minutes) - Stable, trend-following

Each temporal bucket has different feature importance patterns and requires
specialized hyperparameters for optimal performance.

Author: BT Research Team
Date: 2025-11-12
"""

from __future__ import annotations

import logging
from typing import Dict, Tuple

import numpy as np
import polars as pl

logger = logging.getLogger(__name__)


def assign_temporal_regime(df: pl.DataFrame) -> pl.DataFrame:
    """
    Assign temporal regime based on time remaining to expiry.

    Args:
        df: DataFrame with 'time_remaining' column (in seconds)

    Returns:
        DataFrame with added 'temporal_regime' column
    """
    return df.with_columns([
        pl.when(pl.col("time_remaining") < 300)
          .then(pl.lit("near"))
          .when(pl.col("time_remaining") <= 900)
          .then(pl.lit("mid"))
          .otherwise(pl.lit("far"))
          .alias("temporal_regime")
    ])


def get_temporal_thresholds(regime: str) -> Dict[str, float]:
    """
    Get regime-specific thresholds and parameters.

    These thresholds are calibrated based on the data distribution
    and performance analysis from the technical paper.

    Args:
        regime: One of 'near', 'mid', 'far'

    Returns:
        Dictionary of thresholds and parameters
    """
    thresholds = {
        "near": {
            "min_samples": 50000,      # Minimum samples for training
            "vol_low": 0.01,           # Low volatility threshold
            "vol_high": 0.03,          # High volatility threshold
            "moneyness_atm": 0.01,     # ATM threshold (1% moneyness)
            "extreme_vol": 0.05,       # Extreme volatility threshold
            "min_time": 0,             # Minimum time remaining
            "max_time": 300,           # Maximum time remaining
        },
        "mid": {
            "min_samples": 100000,
            "vol_low": 0.008,
            "vol_high": 0.025,
            "moneyness_atm": 0.01,
            "extreme_vol": 0.04,
            "min_time": 300,
            "max_time": 900,
        },
        "far": {
            "min_samples": 50000,
            "vol_low": 0.005,
            "vol_high": 0.02,
            "moneyness_atm": 0.01,
            "extreme_vol": 0.035,
            "min_time": 900,
            "max_time": np.inf,
        }
    }

    if regime not in thresholds:
        raise ValueError(f"Unknown temporal regime: {regime}. Must be one of {list(thresholds.keys())}")

    return thresholds[regime]


def get_temporal_feature_importance(regime: str) -> Dict[str, float]:
    """
    Get expected feature importance for each temporal regime.

    Based on analysis from the technical paper, different features
    matter more at different time horizons.

    Args:
        regime: One of 'near', 'mid', 'far'

    Returns:
        Dictionary of feature names to expected importance (0-1)
    """
    importance = {
        "near": {
            # Near expiry: Gamma dominates, order book critical
            "moneyness_distance": 0.35,
            "time_remaining": 0.15,
            "imbalance_ema_60s": 0.10,
            "spread_level_1": 0.08,
            "momentum_60s": 0.07,
            "rv_60s": 0.06,
            "volume_60s": 0.05,
            "is_extreme_condition": 0.04,
        },
        "mid": {
            # Mid horizon: Balanced importance
            "moneyness_distance": 0.25,
            "time_remaining": 0.20,
            "momentum_300s": 0.10,
            "rv_300s": 0.08,
            "imbalance_ema_300s": 0.07,
            "hv_900s": 0.06,
            "spread_level_1": 0.05,
            "volatility_regime": 0.04,
        },
        "far": {
            # Far horizon: Trends and mean reversion dominate
            "time_remaining": 0.25,
            "moneyness_distance": 0.15,
            "momentum_900s": 0.12,
            "hv_3600s": 0.10,
            "rv_900s": 0.08,
            "price_position_900s": 0.07,
            "range_900s": 0.06,
            "day_of_week": 0.05,
        }
    }

    return importance.get(regime, {})


def split_data_by_temporal_regime(
    df: pl.DataFrame,
    test_size: float = 0.2
) -> Dict[str, Tuple[pl.DataFrame, pl.DataFrame]]:
    """
    Split data by temporal regime for training.

    Args:
        df: DataFrame with features and 'temporal_regime' column
        test_size: Fraction of data to use for testing

    Returns:
        Dictionary mapping regime name to (train_df, test_df) tuples
    """
    # Ensure temporal regime is assigned
    if "temporal_regime" not in df.columns:
        df = assign_temporal_regime(df)

    splits = {}

    for regime in ["near", "mid", "far"]:
        # Filter for this regime
        regime_df = df.filter(pl.col("temporal_regime") == regime)

        if len(regime_df) == 0:
            logger.warning(f"No data for regime {regime}")
            continue

        # Time-based split (no shuffling for time series)
        n_samples = len(regime_df)
        split_idx = int(n_samples * (1 - test_size))

        train_df = regime_df[:split_idx]
        test_df = regime_df[split_idx:]

        splits[regime] = (train_df, test_df)

        logger.info(
            f"Regime {regime}: {len(train_df):,} train, {len(test_df):,} test samples"
        )

    return splits


def validate_temporal_splits(df: pl.DataFrame) -> Dict[str, Dict[str, float]]:
    """
    Validate data distribution across temporal regimes.

    Args:
        df: DataFrame with features

    Returns:
        Statistics about each temporal regime
    """
    # Assign temporal regime if not present
    if "temporal_regime" not in df.columns:
        df = assign_temporal_regime(df)

    stats = {}

    for regime in ["near", "mid", "far"]:
        regime_df = df.filter(pl.col("temporal_regime") == regime)

        if len(regime_df) > 0:
            stats[regime] = {
                "count": len(regime_df),
                "percentage": len(regime_df) / len(df) * 100,
                "avg_time_remaining": regime_df["time_remaining"].mean(),
                "std_time_remaining": regime_df["time_remaining"].std(),
                "avg_moneyness": regime_df["moneyness"].mean() if "moneyness" in df.columns else None,
                "avg_rv": regime_df["rv_900s"].mean() if "rv_900s" in df.columns else None,
            }
        else:
            stats[regime] = {"count": 0, "percentage": 0}

    # Log summary
    logger.info("Temporal regime distribution:")
    for regime, regime_stats in stats.items():
        logger.info(
            f"  {regime}: {regime_stats['count']:,} samples ({regime_stats['percentage']:.1f}%)"
        )

    return stats


def get_temporal_model_config(regime: str) -> Dict[str, any]:
    """
    Get model configuration for a specific temporal regime.

    Args:
        regime: One of 'near', 'mid', 'far'

    Returns:
        Model configuration dictionary
    """
    configs = {
        "near": {
            "name": "near_horizon",
            "description": "Near expiry model (< 5 minutes)",
            "hyperparams": {
                "learning_rate": 0.03,  # More conservative
                "num_leaves": 31,
                "min_data_in_leaf": 50,  # Larger to prevent overfitting
                "feature_fraction": 0.7,
                "bagging_fraction": 0.6,
                "lambda_l1": 1.5,
                "lambda_l2": 30.0,  # Strong regularization
                "early_stopping_rounds": 50,
            },
            "features_to_emphasize": [
                "moneyness_distance",
                "time_remaining",
                "imbalance_ema_60s",
                "spread_level_1",
                "momentum_60s",
            ],
        },
        "mid": {
            "name": "mid_horizon",
            "description": "Mid horizon model (5-15 minutes)",
            "hyperparams": {
                "learning_rate": 0.05,  # Balanced
                "num_leaves": 47,
                "min_data_in_leaf": 20,
                "feature_fraction": 0.8,
                "bagging_fraction": 0.7,
                "lambda_l1": 1.0,
                "lambda_l2": 20.0,
                "early_stopping_rounds": 75,
            },
            "features_to_emphasize": [
                "moneyness_distance",
                "time_remaining",
                "momentum_300s",
                "rv_300s",
                "imbalance_ema_300s",
            ],
        },
        "far": {
            "name": "far_horizon",
            "description": "Far horizon model (> 15 minutes)",
            "hyperparams": {
                "learning_rate": 0.05,
                "num_leaves": 63,  # More complex for stable regime
                "min_data_in_leaf": 10,
                "feature_fraction": 0.85,
                "bagging_fraction": 0.8,
                "lambda_l1": 0.5,
                "lambda_l2": 15.0,  # Less regularization needed
                "early_stopping_rounds": 100,
            },
            "features_to_emphasize": [
                "time_remaining",
                "moneyness_distance",
                "momentum_900s",
                "hv_3600s",
                "price_position_900s",
            ],
        }
    }

    if regime not in configs:
        raise ValueError(f"Unknown temporal regime: {regime}")

    return configs[regime]


def main():
    """Example usage of temporal split functions."""

    # Example: Load data and apply temporal split
    logger.info("Loading example data...")

    # Create sample data for demonstration
    n_samples = 100000
    df = pl.DataFrame({
        "time_remaining": np.random.uniform(0, 1200, n_samples),
        "moneyness": np.random.normal(0, 0.02, n_samples),
        "rv_900s": np.random.uniform(0.005, 0.04, n_samples),
        "outcome": np.random.binomial(1, 0.5, n_samples),
    })

    # Assign temporal regimes
    df = assign_temporal_regime(df)

    # Validate distribution
    stats = validate_temporal_splits(df)

    # Split by regime
    splits = split_data_by_temporal_regime(df)

    # Show configurations
    for regime in ["near", "mid", "far"]:
        config = get_temporal_model_config(regime)
        logger.info(f"\n{regime.upper()} configuration:")
        logger.info(f"  Description: {config['description']}")
        logger.info(f"  Learning rate: {config['hyperparams']['learning_rate']}")
        logger.info(f"  Num leaves: {config['hyperparams']['num_leaves']}")
        logger.info(f"  Key features: {', '.join(config['features_to_emphasize'][:3])}")


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    main()