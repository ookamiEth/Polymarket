#!/usr/bin/env python3
"""
Regime Detection V5 - Simplified 6-Model Architecture
=====================================================

Simplified market regime detection for V5 with 6 models instead of 12.

Changes from V4:
- Removed ATM/OTM split (4-way → 2-way volatility regimes)
- Added moneyness_bin (10 categories) to preserve ATM/OTM signal
- 6 combined regimes: 3 temporal × 2 volatility

Architecture:
- Temporal regimes (3): near (<300s), mid (300-600s), far (600-900s)
- Volatility regimes (2): low_vol, high_vol
- Combined regimes (6): near_low_vol, near_high_vol, mid_low_vol, mid_high_vol, far_low_vol, far_high_vol

Author: BT Research Team
Date: 2025-11-19
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
              .when(pl.col("time_remaining") < 600)
              .then(pl.lit("mid"))
              .otherwise(pl.lit("far"))
              .alias("temporal_regime")
        ])

logger = logging.getLogger(__name__)


def detect_regime_simplified(
    df: pl.DataFrame,
    volatility_col: str = "rv_900s",
    hysteresis: float = 0.1,
) -> pl.DataFrame:
    """
    Detect 2-way volatility regimes (V5 simplification).

    V4 had 4-way split: low_vol_atm, low_vol_otm, high_vol_atm, high_vol_otm
    V5 has 2-way split: low_vol, high_vol (moneyness_bin preserves ATM/OTM signal)

    Args:
        df: DataFrame with volatility column
        volatility_col: Column name for volatility measure (default: rv_900s)
        hysteresis: Hysteresis buffer (default: 0.1 = 10%)

    Returns:
        DataFrame with added regime columns:
        - volatility_regime: low/high (simplified from V4's low/medium/high)
        - market_regime: low_vol/high_vol (no ATM/OTM suffix)
        - vol_low_thresh: dynamic low volatility threshold
        - vol_high_thresh: dynamic high volatility threshold
    """
    logger.info("Detecting 2-way volatility regimes (V5 simplified)...")

    # Monthly percentile computation for non-stationary thresholds
    df = df.with_columns(
        [
            pl.col(volatility_col).quantile(0.33).over(pl.col("timestamp").dt.truncate("30d")).alias("vol_low_thresh"),
            pl.col(volatility_col).quantile(0.67).over(pl.col("timestamp").dt.truncate("30d")).alias("vol_high_thresh"),
        ]
    )

    # Apply hysteresis-adjusted thresholds
    df = df.with_columns(
        [
            # Volatility regime (2-way split)
            pl.when(pl.col(volatility_col) < pl.col("vol_low_thresh") * (1 - hysteresis))
            .then(pl.lit("low"))
            .when(pl.col(volatility_col) > pl.col("vol_high_thresh") * (1 + hysteresis))
            .then(pl.lit("high"))
            .otherwise(pl.lit("medium"))  # Keep medium for now, will map to low
            .alias("volatility_regime"),

            # Market regime (simplified: just low_vol or high_vol)
            pl.when(pl.col(volatility_col) <= pl.col("vol_high_thresh"))
            .then(pl.lit("low_vol"))
            .otherwise(pl.lit("high_vol"))
            .alias("market_regime"),
        ]
    )

    # Log regime distribution
    regime_counts = df.group_by("market_regime").agg(pl.len().alias("count"))
    logger.info(f"Regime distribution (2-way):\n{regime_counts}")

    return df


def detect_hierarchical_regime_v5(
    df: pl.DataFrame,
    volatility_col: str = "rv_900s",
    hysteresis: float = 0.1,
) -> pl.DataFrame:
    """
    Detect 6 combined regimes for V5: 3 temporal × 2 volatility.

    This is the main function for V5 architecture, combining temporal splits
    with simplified volatility regimes to create 6 specialized models.

    Changes from V4:
    - Removed ATM/OTM split (12 models → 6 models)
    - Moneyness_bin feature preserves ATM/OTM signal in model
    - Faster training (6 models vs 12)
    - Simpler production deployment

    Args:
        df: DataFrame with features including time_remaining
        volatility_col: Column for volatility measure
        hysteresis: Hysteresis buffer for volatility regimes

    Returns:
        DataFrame with added columns:
        - temporal_regime: near/mid/far
        - volatility_regime: low/medium/high (intermediate)
        - market_regime: low_vol/high_vol (simplified)
        - combined_regime: 6-way temporal × market regime

    Note: moneyness_bin should already exist from engineer_greeks_v5.py
    """
    logger.info("Detecting 6 hierarchical regimes (3 temporal × 2 volatility)...")

    # Step 1: Apply temporal regime (3 categories)
    if "time_remaining" not in df.columns:
        raise ValueError("time_remaining column required for temporal regime detection")

    df = assign_temporal_regime(df)

    # Step 2: Apply simplified volatility regime (2-way)
    df = detect_regime_simplified(df, volatility_col, hysteresis)

    # Step 3: Create combined 6-way regime
    df = df.with_columns([
        # Combine temporal and market regimes
        (pl.col("temporal_regime") + "_" + pl.col("market_regime"))
        .alias("combined_regime")
    ])

    # Note: moneyness_bin is already added by engineer_greeks_v5.py
    # Do not create it here to avoid duplication

    # Validate all 6 regimes exist
    expected_regimes = [
        f"{t}_{v}"
        for t in ["near", "mid", "far"]
        for v in ["low_vol", "high_vol"]
    ]

    # Log regime distribution
    regime_counts = df.group_by("combined_regime").agg(pl.len().alias("count")).sort("combined_regime")
    logger.info(f"Combined regime distribution (6 categories):\n{regime_counts}")

    actual_regimes = df["combined_regime"].unique().to_list()
    missing = set(expected_regimes) - set(actual_regimes)
    if missing:
        logger.warning(f"Missing regimes in data: {missing}")

    # Log moneyness bin distribution
    bin_counts = df.group_by("moneyness_bin").agg(pl.len().alias("count")).sort("moneyness_bin")
    logger.info(f"Moneyness bin distribution (10 categories):\n{bin_counts}")

    return df


def detect_extreme_conditions(
    df: pl.DataFrame,
    rv_short_col: str = "rv_60s",
    rv_long_col: str = "rv_900s",
    spike_ratio: float = 3.0,
    percentile: float = 0.95,
) -> pl.DataFrame:
    """
    Detect extreme market conditions for position scaling (unchanged from V4).

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
        "combined_regime",
        "moneyness_bin",
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
    """Example usage of V5 regime detection."""
    # Get project root (search up from script location)
    script_dir = Path(__file__).parent.resolve()  # research/model/00_data_prep
    model_dir = script_dir.parent  # research/model
    project_root = model_dir.parent  # /home/ubuntu/Polymarket

    # Paths (all absolute)
    input_path = model_dir / "data" / "consolidated_features_v5_pipeline_ready.parquet"
    output_path = model_dir / "results" / "regime_history_v5.parquet"

    # Log resolved paths
    logger.info(f"Project root: {project_root}")
    logger.info(f"Input: {input_path}")
    logger.info(f"Output: {output_path}")

    if not input_path.exists():
        logger.error(f"Input file not found: {input_path}")
        logger.info("Run engineer_greeks_v5.py first to generate V5 features")
        return

    logger.info(f"Loading data from {input_path}")
    df = pl.read_parquet(input_path)

    # Apply V5 regime detection (6 combined regimes)
    df = detect_hierarchical_regime_v5(df)
    df = detect_extreme_conditions(df)

    # Analyze transitions
    analyze_regime_transitions(df)

    # Save regime history
    save_regime_history(df, output_path)

    logger.info("V5 regime detection complete")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
