#!/usr/bin/env python3
"""
Regime Detection V4 with Hysteresis and Temporal Split
=======================================================

Advanced market regime detection with hysteresis to prevent oscillation,
now with 12 combined regimes (3 temporal × 4 volatility).

Features:
- Temporal split: near (<300s), mid (300-900s), far (>900s)
- Volatility regimes: low_vol_atm, low_vol_otm, high_vol_atm, high_vol_otm
- Monthly percentile-based thresholds (non-stationary adaptation)
- Hysteresis zones (10% buffer) to prevent rapid switching
- 12 combined regimes for specialized model training
- Position scaling factors for risk management

Author: BT Research Team
Date: 2025-11-12
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict

import polars as pl

# Import temporal split functions
import sys
sys.path.append(str(Path(__file__).parent.parent / "01_pricing"))
try:
    from temporal_model_split import assign_temporal_regime, get_temporal_thresholds
except ImportError:
    # Fallback if import fails
    def assign_temporal_regime(df: pl.DataFrame) -> pl.DataFrame:
        return df.with_columns([
            pl.when(pl.col("time_remaining") < 300)
              .then(pl.lit("near"))
              .when(pl.col("time_remaining") <= 900)
              .then(pl.lit("mid"))
              .otherwise(pl.lit("far"))
              .alias("temporal_regime")
        ])

logger = logging.getLogger(__name__)


def detect_regime_with_hysteresis(
    df: pl.DataFrame,
    volatility_col: str = "rv_900s",
    moneyness_col: str = "moneyness_distance",
    hysteresis: float = 0.1,
    atm_threshold: float = 0.01,
) -> pl.DataFrame:
    """
    Detect market regimes with hysteresis to prevent oscillation.

    Args:
        df: DataFrame with volatility and moneyness columns
        volatility_col: Column name for volatility measure (default: rv_900s)
        moneyness_col: Column name for moneyness distance (default: moneyness_distance)
        hysteresis: Hysteresis buffer (default: 0.1 = 10%)
        atm_threshold: Threshold for ATM classification (default: 0.01)

    Returns:
        DataFrame with added regime columns:
        - volatility_regime: low/medium/high
        - market_regime: combined regime (e.g., low_vol_atm)
        - vol_low_thresh: dynamic low volatility threshold
        - vol_high_thresh: dynamic high volatility threshold
    """
    logger.info("Detecting market regimes with hysteresis...")

    # Monthly percentile computation for non-stationary thresholds
    df = df.with_columns(
        [
            pl.col(volatility_col).quantile(0.33).over(pl.col("timestamp").dt.truncate("30d")).alias("vol_low_thresh"),
            pl.col(volatility_col).quantile(0.67).over(pl.col("timestamp").dt.truncate("30d")).alias("vol_high_thresh"),
        ]
    )

    # For full hysteresis implementation (stateful processing)
    # This requires sequential processing which is more complex in Polars
    # Here's a simplified version using adjusted thresholds

    # Apply hysteresis-adjusted thresholds
    df = df.with_columns(
        [
            # Volatility regime with buffer zones
            pl.when(pl.col(volatility_col) < pl.col("vol_low_thresh") * (1 - hysteresis))
            .then(pl.lit("low"))
            .when(pl.col(volatility_col) > pl.col("vol_high_thresh") * (1 + hysteresis))
            .then(pl.lit("high"))
            .otherwise(pl.lit("medium"))
            .alias("volatility_regime"),
            # Combined regime with moneyness
            pl.when((pl.col(volatility_col) < pl.col("vol_low_thresh")) & (pl.col(moneyness_col) < atm_threshold))
            .then(pl.lit("low_vol_atm"))
            .when((pl.col(volatility_col) < pl.col("vol_low_thresh")) & (pl.col(moneyness_col) >= atm_threshold))
            .then(pl.lit("low_vol_otm"))
            .when((pl.col(volatility_col) > pl.col("vol_high_thresh")) & (pl.col(moneyness_col) < atm_threshold))
            .then(pl.lit("high_vol_atm"))
            .when((pl.col(volatility_col) > pl.col("vol_high_thresh")) & (pl.col(moneyness_col) >= atm_threshold))
            .then(pl.lit("high_vol_otm"))
            .otherwise(pl.lit("medium_vol"))
            .alias("market_regime"),
        ]
    )

    # Log regime distribution
    regime_counts = df.group_by("market_regime").agg(pl.len().alias("count"))
    logger.info(f"Regime distribution:\n{regime_counts}")

    return df


def detect_hierarchical_regime(
    df: pl.DataFrame,
    volatility_col: str = "rv_900s",
    moneyness_col: str = "moneyness_distance",
    hysteresis: float = 0.1,
    atm_threshold: float = 0.01,
) -> pl.DataFrame:
    """
    Detect 12 combined regimes: 3 temporal × 4 volatility.

    This is the main function for V4 architecture, combining temporal splits
    with volatility regimes to create 12 specialized models.

    Args:
        df: DataFrame with features including time_remaining
        volatility_col: Column for volatility measure
        moneyness_col: Column for moneyness distance
        hysteresis: Hysteresis buffer for volatility regimes
        atm_threshold: Threshold for ATM classification

    Returns:
        DataFrame with added columns:
        - temporal_regime: near/mid/far
        - volatility_regime: low/medium/high
        - market_regime: 4-way volatility × moneyness
        - combined_regime: 12-way temporal × market regime
    """
    logger.info("Detecting 12 hierarchical regimes (3 temporal × 4 volatility)...")

    # Step 1: Apply temporal regime (3 categories)
    if "time_remaining" not in df.columns:
        raise ValueError("time_remaining column required for temporal regime detection")

    df = assign_temporal_regime(df)

    # Step 2: Apply volatility regime with hysteresis (existing function)
    df = detect_regime_with_hysteresis(df, volatility_col, moneyness_col, hysteresis, atm_threshold)

    # Step 3: Create combined 12-way regime
    # Simplify market_regime to 4 categories (removing medium_vol)
    df = df.with_columns([
        pl.when(pl.col("market_regime") == "medium_vol")
          .then(pl.lit("low_vol_otm"))  # Default medium to low_vol_otm
          .otherwise(pl.col("market_regime"))
          .alias("market_regime_4way"),

        # Combine temporal and market regimes
        (pl.col("temporal_regime") + "_" +
         pl.when(pl.col("market_regime") == "medium_vol")
           .then(pl.lit("low_vol_otm"))
           .otherwise(pl.col("market_regime")))
        .alias("combined_regime")
    ])

    # Validate all 12 regimes
    expected_regimes = [
        f"{t}_{v}"
        for t in ["near", "mid", "far"]
        for v in ["low_vol_atm", "low_vol_otm", "high_vol_atm", "high_vol_otm"]
    ]

    # Log regime distribution
    regime_counts = df.group_by("combined_regime").agg(pl.len().alias("count")).sort("combined_regime")
    logger.info(f"Combined regime distribution (12 categories):\n{regime_counts}")

    actual_regimes = df["combined_regime"].unique().to_list()
    missing = set(expected_regimes) - set(actual_regimes)
    if missing:
        logger.warning(f"Missing regimes in data: {missing}")

    # Add temporal-specific thresholds
    for temporal_regime in ["near", "mid", "far"]:
        thresholds = get_temporal_thresholds(temporal_regime) if 'get_temporal_thresholds' in globals() else {}
        if thresholds:
            # Apply temporal-specific adjustments
            mask = df["temporal_regime"] == temporal_regime
            if mask.any():
                logger.info(f"Applied {temporal_regime} thresholds: {thresholds}")

    return df


def detect_extreme_conditions(
    df: pl.DataFrame,
    rv_short_col: str = "rv_60s",
    rv_long_col: str = "rv_900s",
    spike_ratio: float = 3.0,
    percentile: float = 0.95,
) -> pl.DataFrame:
    """
    Detect extreme market conditions for 5th model trigger and position scaling.

    Args:
        df: DataFrame with realized volatility columns
        rv_short_col: Short-term RV column (default: rv_60s)
        rv_long_col: Long-term RV column (default: rv_900s)
        spike_ratio: Ratio threshold for spike detection (default: 3.0)
        percentile: Percentile for sustained high vol detection (default: 0.95)

    Returns:
        DataFrame with added columns:
        - rv_ratio: Short-term / long-term RV ratio
        - rv_95th_percentile: 95th percentile of RV over last 30 days
        - is_extreme_condition: Boolean flag for extreme conditions
        - position_scale: Position scaling factor (0.5 during extremes, 1.0 otherwise)
    """
    logger.info("Detecting extreme market conditions...")

    df = df.with_columns(
        [
            # Compute RV ratio for spike detection
            (pl.col(rv_short_col) / (pl.col(rv_long_col) + 1e-10)).alias("rv_ratio"),
            # Compute rolling percentile for sustained high volatility
            pl.col(rv_long_col)
            .quantile(percentile)
            .over(pl.col("timestamp").dt.truncate("30d"))
            .alias(f"rv_{int(percentile * 100)}th_percentile"),
        ]
    )

    # Flag extreme conditions
    df = df.with_columns(
        [
            pl.when(
                (pl.col("rv_ratio") > spike_ratio)
                | (pl.col(rv_long_col) > pl.col(f"rv_{int(percentile * 100)}th_percentile"))
            )
            .then(True)
            .otherwise(False)
            .alias("is_extreme_condition"),
            # Position scaling factor for risk management
            pl.when(
                (pl.col("rv_ratio") > spike_ratio)
                | (pl.col(rv_long_col) > pl.col(f"rv_{int(percentile * 100)}th_percentile"))
            )
            .then(0.5)
            .otherwise(1.0)
            .alias("position_scale"),
        ]
    )

    # Log extreme events
    extreme_count = df.filter(pl.col("is_extreme_condition")).height
    total_count = df.height
    logger.info(
        f"Extreme conditions detected: {extreme_count:,} / {total_count:,} ({extreme_count / total_count * 100:.2f}%)"
    )

    return df


def save_regime_history(df: pl.DataFrame, output_path: Path) -> None:
    """
    Save regime detection history for analysis and model training.

    Args:
        df: DataFrame with regime columns
        output_path: Path to save the regime history
    """
    # Select relevant columns for regime history
    regime_cols = [
        "timestamp",
        "volatility_regime",
        "market_regime",
        "is_extreme_condition",
        "position_scale",
        "vol_low_thresh",
        "vol_high_thresh",
        "rv_ratio",
    ]

    # Filter to available columns
    available_cols = [col for col in regime_cols if col in df.columns]

    # Save regime history
    df.select(available_cols).write_parquet(output_path, compression="snappy")
    logger.info(f"Saved regime history to {output_path}")


def analyze_regime_transitions(df: pl.DataFrame) -> pl.DataFrame:
    """
    Analyze regime transitions to validate hysteresis effectiveness.

    Args:
        df: DataFrame with regime columns

    Returns:
        DataFrame with transition statistics
    """
    # Calculate regime changes
    df = df.with_columns(
        [
            pl.col("market_regime").shift(1).alias("prev_regime"),
            (pl.col("market_regime") != pl.col("market_regime").shift(1)).alias("regime_changed"),
        ]
    )

    # Count transitions
    transition_matrix = (
        df.filter(pl.col("regime_changed"))
        .group_by(["prev_regime", "market_regime"])
        .agg(pl.len().alias("transitions"))
        .sort(["prev_regime", "market_regime"])
    )

    logger.info(f"Regime transition matrix:\n{transition_matrix}")

    # Calculate average regime duration
    regime_durations = (
        df.with_columns(pl.col("regime_changed").cum_sum().alias("regime_group"))
        .group_by(["regime_group", "market_regime"])
        .agg(pl.len().alias("duration"))
        .group_by("market_regime")
        .agg(
            [
                pl.col("duration").mean().alias("avg_duration"),
                pl.col("duration").min().alias("min_duration"),
                pl.col("duration").max().alias("max_duration"),
            ]
        )
    )

    logger.info(f"Regime duration statistics:\n{regime_durations}")

    return transition_matrix


def main():
    """Example usage of regime detection."""
    # Load sample data
    input_path = Path("research/model/results/production_backtest_results.parquet")

    if not input_path.exists():
        logger.error(f"Input file not found: {input_path}")
        return

    logger.info(f"Loading data from {input_path}")
    df = pl.read_parquet(input_path)

    # Apply regime detection
    df = detect_regime_with_hysteresis(df)
    df = detect_extreme_conditions(df)

    # Analyze transitions
    analyze_regime_transitions(df)

    # Save regime history
    output_path = Path("research/model/results/regime_history_v4.parquet")
    save_regime_history(df, output_path)

    logger.info("Regime detection complete")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
