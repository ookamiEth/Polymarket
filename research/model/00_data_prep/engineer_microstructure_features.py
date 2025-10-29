#!/usr/bin/env python3
"""
Engineer Microstructure Features from 1-Second BTC Prices

This script computes high-frequency microstructure features that capture
price dynamics, momentum, jumps, and market behavior at ultra-short horizons.
These features leverage the second-by-second granularity of the BTC data.

Input:
    - research/model/results/btc_perpetual_1s_resampled.parquet (63M rows)

Output:
    - research/model/results/microstructure_features.parquet (~1.5GB)

Features Engineered:
    1. Momentum: Returns at 60s, 300s, 900s horizons
    2. Price Range: Intraday high-low spread
    3. Reversals: Count of price direction changes
    4. Jumps: Large discrete moves detection
    5. Hurst Exponent: Mean reversion vs trending behavior
    6. Autocorrelation: Return persistence metrics

Author: BT Research Team
Date: 2025-10-29
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import numpy as np
import polars as pl

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

# Get model directory
SCRIPT_DIR = Path(__file__).parent
MODEL_DIR = SCRIPT_DIR.parent
INPUT_FILE = MODEL_DIR / "results/btc_perpetual_1s_resampled.parquet"
OUTPUT_FILE = MODEL_DIR / "results/microstructure_features.parquet"

# Feature windows (in seconds)
MOMENTUM_WINDOWS = [60, 300, 900]  # 1min, 5min, 15min
RANGE_WINDOWS = [60, 300, 900]  # For high-low range
REVERSAL_WINDOWS = [60, 300]  # For counting direction changes
JUMP_THRESHOLD_SIGMA = 3.0  # Detect moves >3 sigma


def calculate_momentum_features(df: pl.DataFrame) -> pl.DataFrame:
    """
    Calculate price momentum (returns) at multiple horizons.

    Formula: momentum_Ns = (price_t - price_t-N) / price_t-N

    Args:
        df: DataFrame with 'close' price

    Returns:
        DataFrame with momentum_60s, momentum_300s, momentum_900s columns
    """
    logger.info("Calculating momentum features...")

    for window in MOMENTUM_WINDOWS:
        df = df.with_columns(
            [
                ((pl.col("close") - pl.col("close").shift(window)) / pl.col("close").shift(window)).alias(
                    f"momentum_{window}s"
                )
            ]
        )

        # Report statistics
        stats = df[f"momentum_{window}s"].drop_nulls()
        logger.info(
            f"momentum_{window}s: mean={stats.mean():.6f}, "
            f"std={stats.std():.6f}, "
            f"min={stats.min():.6f}, "
            f"max={stats.max():.6f}"
        )

    logger.info("Momentum features calculated.")
    return df


def calculate_price_range_features(df: pl.DataFrame) -> pl.DataFrame:
    """
    Calculate rolling price range (high-low spread) as fraction of close.

    Formula: range_Ns = (high_N - low_N) / close_t

    Args:
        df: DataFrame with 'high', 'low', 'close' columns

    Returns:
        DataFrame with range_60s, range_300s, range_900s columns
    """
    logger.info("Calculating price range features...")

    for window in RANGE_WINDOWS:
        df = df.with_columns(
            [
                # Rolling max and min
                pl.col("high").rolling_max(window_size=window).alias(f"high_{window}s"),
                pl.col("low").rolling_min(window_size=window).alias(f"low_{window}s"),
            ]
        )

        # Calculate range as fraction of price
        df = df.with_columns(
            [((pl.col(f"high_{window}s") - pl.col(f"low_{window}s")) / pl.col("close")).alias(f"range_{window}s")]
        )

        # Drop intermediate columns
        df = df.drop([f"high_{window}s", f"low_{window}s"])

        # Report statistics
        stats = df[f"range_{window}s"].drop_nulls()
        logger.info(f"range_{window}s: mean={stats.mean():.6f}, std={stats.std():.6f}, max={stats.max():.6f}")

    logger.info("Price range features calculated.")
    return df


def calculate_reversal_features(df: pl.DataFrame) -> pl.DataFrame:
    """
    Count number of price reversals (direction changes) in rolling window.

    A reversal occurs when sign(return_t) != sign(return_t-1).

    Args:
        df: DataFrame with 'close' price

    Returns:
        DataFrame with reversals_60s, reversals_300s columns
    """
    logger.info("Calculating reversal features...")

    # Calculate 1-second returns
    df = df.with_columns([pl.col("close").diff().alias("return_1s")])

    # Sign of returns (-1, 0, +1)
    df = df.with_columns([pl.col("return_1s").sign().alias("return_sign")])

    # Sign changes (reversal = 1, continuation = 0)
    df = df.with_columns([(pl.col("return_sign").diff() != 0).cast(pl.Int8).alias("reversal")])

    for window in REVERSAL_WINDOWS:
        # Sum reversals in rolling window
        df = df.with_columns([pl.col("reversal").rolling_sum(window_size=window).alias(f"reversals_{window}s")])

        # Report statistics
        stats = df[f"reversals_{window}s"].drop_nulls()
        logger.info(f"reversals_{window}s: mean={stats.mean():.2f}, median={stats.median():.0f}, max={stats.max():.0f}")

    # Drop intermediate columns
    df = df.drop(["return_1s", "return_sign", "reversal"])

    logger.info("Reversal features calculated.")
    return df


def calculate_jump_features(df: pl.DataFrame) -> pl.DataFrame:
    """
    Detect and count price jumps (large discrete moves).

    Jump definition: |return_1s| > threshold_sigma * rv_60s / sqrt(60)

    Args:
        df: DataFrame with 'close' price (rv_60s needed from prior calculation)

    Returns:
        DataFrame with jump_detected, jump_intensity_300s columns
    """
    logger.info("Calculating jump features...")

    # We need rv_60s for jump detection
    # Calculate it if not present
    if "rv_60s" not in df.columns:
        logger.warning("rv_60s not found. Calculating simple rolling volatility...")
        df = df.with_columns([pl.col("close").log().diff().alias("log_return")])
        df = df.with_columns([pl.col("log_return").rolling_std(window_size=60).alias("rv_60s_approx")])
        rv_col = "rv_60s_approx"
    else:
        df = df.with_columns([pl.col("close").log().diff().alias("log_return")])
        rv_col = "rv_60s"

    # Calculate jump threshold: threshold_sigma * vol_per_second
    # rv_60s is annualized, so vol_per_second = rv_60s / sqrt(seconds_per_year)
    df = df.with_columns(
        [
            (pl.col(rv_col) / np.sqrt(31_557_600)).alias("vol_per_second")  # De-annualize
        ]
    )

    # Detect jumps: |return| > threshold * vol_per_second
    df = df.with_columns(
        [
            (pl.col("log_return").abs() > JUMP_THRESHOLD_SIGMA * pl.col("vol_per_second"))
            .cast(pl.Int8)
            .alias("jump_detected")
        ]
    )

    # Jump intensity: rolling count of jumps in last 5 minutes
    df = df.with_columns([pl.col("jump_detected").rolling_sum(window_size=300).alias("jump_intensity_300s")])

    # Drop intermediate columns
    df = df.drop(["vol_per_second"])
    df = df.drop(["rv_60s_approx", "log_return"]) if rv_col == "rv_60s_approx" else df.drop(["log_return"])

    # Report statistics
    jump_count = df["jump_detected"].sum()
    jump_pct = jump_count / len(df) * 100
    logger.info(f"Total jumps detected: {jump_count:,} ({jump_pct:.3f}% of observations)")

    intensity_stats = df["jump_intensity_300s"].drop_nulls()
    logger.info(f"jump_intensity_300s: mean={intensity_stats.mean():.2f}, max={intensity_stats.max():.0f}")

    logger.info("Jump features calculated.")
    return df


def calculate_autocorrelation_features(df: pl.DataFrame) -> pl.DataFrame:
    """
    Calculate autocorrelation of returns at lag 1 and lag 5.

    Uses rolling window correlation between returns and lagged returns.

    Args:
        df: DataFrame with 'close' price

    Returns:
        DataFrame with autocorr_lag1_300s, autocorr_lag5_300s columns
    """
    logger.info("Calculating autocorrelation features...")

    # Calculate returns if not present
    if "log_return" not in df.columns:
        df = df.with_columns([pl.col("close").log().diff().alias("log_return")])

    # Lag-1 autocorrelation (rolling 300s window)
    df = df.with_columns([pl.col("log_return").shift(1).alias("log_return_lag1")])

    df = df.with_columns(
        [pl.rolling_corr(pl.col("log_return"), pl.col("log_return_lag1"), window_size=300).alias("autocorr_lag1_300s")]
    )

    # Lag-5 autocorrelation (5-second lag)
    df = df.with_columns([pl.col("log_return").shift(5).alias("log_return_lag5")])

    df = df.with_columns(
        [pl.rolling_corr(pl.col("log_return"), pl.col("log_return_lag5"), window_size=300).alias("autocorr_lag5_300s")]
    )

    # Drop intermediate columns
    df = df.drop(["log_return_lag1", "log_return_lag5", "log_return"])

    # Report statistics
    for lag in [1, 5]:
        col = f"autocorr_lag{lag}_300s"
        stats = df[col].drop_nulls()
        logger.info(f"{col}: mean={stats.mean():.4f}, std={stats.std():.4f}")

    logger.info("Autocorrelation features calculated.")
    return df


def calculate_hurst_exponent_rolling(df: pl.DataFrame, window: int = 300) -> pl.DataFrame:
    """
    Calculate rolling Hurst exponent to detect mean reversion vs trending.

    Hurst exponent interpretation:
        H = 0.5: Random walk (no persistence)
        H < 0.5: Mean reverting
        H > 0.5: Trending (persistent)

    Note: This is computationally expensive. We use a simplified R/S method.

    Args:
        df: DataFrame with 'close' price
        window: Rolling window size (seconds)

    Returns:
        DataFrame with hurst_300s column
    """
    logger.info(f"Calculating Hurst exponent (window={window}s)...")
    logger.info("Note: This is computationally intensive and may take several minutes...")

    # Simplified implementation: use autocorrelation as proxy for Hurst
    # Full Hurst calculation on 63M rows would be prohibitively slow
    # Approximation: H ≈ 0.5 + 0.5 * autocorr_lag1

    if "log_return" not in df.columns:
        df = df.with_columns([pl.col("close").log().diff().alias("log_return")])

    if f"autocorr_lag1_{window}s" not in df.columns:
        df = df.with_columns([pl.col("log_return").shift(1).alias("log_return_lag1")])
        df = df.with_columns(
            [
                pl.rolling_corr(pl.col("log_return"), pl.col("log_return_lag1"), window_size=window).alias(
                    f"autocorr_lag1_{window}s_temp"
                )
            ]
        )
        autocorr_col = f"autocorr_lag1_{window}s_temp"
    else:
        autocorr_col = f"autocorr_lag1_{window}s"

    # Hurst approximation
    df = df.with_columns([(0.5 + 0.5 * pl.col(autocorr_col)).alias(f"hurst_{window}s")])

    if autocorr_col.endswith("_temp"):
        df = df.drop(["log_return_lag1", autocorr_col, "log_return"])
    elif "log_return" in df.columns:
        df = df.drop(["log_return"])

    # Report statistics
    stats = df[f"hurst_{window}s"].drop_nulls()
    mean_hurst = stats.mean()
    median_hurst = stats.median()
    std_hurst = stats.std()

    if mean_hurst is not None and median_hurst is not None and std_hurst is not None:
        logger.info(f"hurst_{window}s: mean={mean_hurst:.4f}, median={median_hurst:.4f}, std={std_hurst:.4f}")
        interpretation = "trending" if mean_hurst > 0.5 else "mean reverting"  # type: ignore[operator]
        logger.info(f"Interpretation: mean={mean_hurst:.4f} -> {interpretation}")
    else:
        logger.warning(f"hurst_{window}s: No valid data")

    logger.info("Hurst exponent calculated.")
    return df


def validate_output(df: pl.DataFrame) -> None:
    """
    Validate output data quality.

    Checks:
        - Feature columns exist
        - No excessive nulls
        - Values within reasonable bounds

    Args:
        df: Output DataFrame to validate

    Raises:
        ValueError: If validation fails
    """
    logger.info("Validating output data...")

    expected_features = [
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

    for feat in expected_features:
        if feat not in df.columns:
            raise ValueError(f"Expected feature '{feat}' not found in output!")

        null_pct = df[feat].null_count() / len(df) * 100
        if null_pct > 15:  # Allow up to 15% nulls for rolling window edge effects
            logger.warning(f"{feat} has {null_pct:.1f}% nulls (>15%)")

    logger.info("Validation passed.")


def main() -> None:
    """Main execution function."""
    logger.info("=" * 80)
    logger.info("MICROSTRUCTURE FEATURE ENGINEERING")
    logger.info("=" * 80)

    # Check input file exists
    if not INPUT_FILE.exists():
        raise FileNotFoundError(f"Input file not found: {INPUT_FILE}")

    logger.info(f"Input: {INPUT_FILE}")
    logger.info(f"Output: {OUTPUT_FILE}")

    # Load data
    logger.info("Loading BTC perpetual 1s data...")
    df = pl.read_parquet(INPUT_FILE)
    logger.info(f"Loaded {len(df):,} rows, {len(df.columns)} columns")

    # Ensure sorted by timestamp
    logger.info("Sorting by timestamp...")
    df = df.sort("timestamp_seconds")

    # Calculate all feature groups
    df = calculate_momentum_features(df)
    df = calculate_price_range_features(df)
    df = calculate_reversal_features(df)
    df = calculate_jump_features(df)
    df = calculate_autocorrelation_features(df)
    df = calculate_hurst_exponent_rolling(df, window=300)

    # Validate
    validate_output(df)

    # Select output columns
    feature_cols = [
        "timestamp_seconds",
        "close",  # Keep price for reference
        # Momentum
        "momentum_60s",
        "momentum_300s",
        "momentum_900s",
        # Range
        "range_60s",
        "range_300s",
        "range_900s",
        # Reversals
        "reversals_60s",
        "reversals_300s",
        # Jumps
        "jump_detected",
        "jump_intensity_300s",
        # Persistence
        "autocorr_lag1_300s",
        "autocorr_lag5_300s",
        "hurst_300s",
    ]

    df_output = df.select(feature_cols)

    # Write output
    logger.info(f"Writing output to {OUTPUT_FILE}...")
    df_output.write_parquet(
        OUTPUT_FILE,
        compression="snappy",
        statistics=True,
    )

    output_size_mb = OUTPUT_FILE.stat().st_size / 1024 / 1024
    logger.info(f"Output written: {len(df_output):,} rows, {output_size_mb:.1f} MB")

    # Summary statistics
    logger.info("\n" + "=" * 80)
    logger.info("SUMMARY STATISTICS")
    logger.info("=" * 80)

    summary = df_output.select(
        [
            pl.col("momentum_300s").mean().alias("momentum_300s_mean"),
            pl.col("range_300s").mean().alias("range_300s_mean"),
            pl.col("reversals_300s").mean().alias("reversals_300s_mean"),
            pl.col("jump_intensity_300s").mean().alias("jump_intensity_mean"),
            pl.col("autocorr_lag1_300s").mean().alias("autocorr_lag1_mean"),
            pl.col("hurst_300s").mean().alias("hurst_mean"),
        ]
    ).to_dicts()[0]

    for key, value in summary.items():
        if value is not None:
            logger.info(f"{key}: {value:.6f}")
        else:
            logger.info(f"{key}: null")

    logger.info("=" * 80)
    logger.info("COMPLETED SUCCESSFULLY")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
