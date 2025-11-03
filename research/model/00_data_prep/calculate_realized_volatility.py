#!/usr/bin/env python3
"""
Calculate Multi-Scale Realized Volatility from 1-Second BTC Prices

This script computes realized volatility at multiple time horizons using
second-by-second BTC perpetual futures prices. The realized volatility is
calculated from log returns and annualized for use in option pricing models.

Input:
    - research/model/results/btc_perpetual_1s_resampled.parquet (63M rows)

Output:
    - research/model/results/realized_volatility_1s.parquet (~2GB)

Methodology:
    RV = sqrt(Σ r²) × sqrt(periods_per_year)
    where r = ln(P_t / P_t-1)

Time Horizons:
    - 60s (1 minute): Ultra-short-term volatility
    - 300s (5 minutes): Short-term volatility (contract duration / 3)
    - 900s (15 minutes): Contract duration volatility
    - 3600s (1 hour): Medium-term volatility

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

# Constants
SECONDS_PER_YEAR = 31_557_600  # 365.25 days × 24 hours × 60 min × 60 sec

# Get model directory (script is in 00_data_prep/, model dir is parent)
SCRIPT_DIR = Path(__file__).parent
MODEL_DIR = SCRIPT_DIR.parent
INPUT_FILE = MODEL_DIR / "results/btc_perpetual_1s_resampled.parquet"
OUTPUT_FILE = MODEL_DIR / "results/realized_volatility_1s.parquet"

# Volatility windows (in seconds)
RV_WINDOWS = {
    "rv_60s": 60,  # 1 minute
    "rv_300s": 300,  # 5 minutes
    "rv_900s": 900,  # 15 minutes (contract duration)
    "rv_3600s": 3600,  # 1 hour
}


def calculate_log_returns(df: pl.DataFrame) -> pl.DataFrame:
    """
    Calculate log returns from price series.

    Formula: r_t = ln(P_t / P_{t-1})

    Args:
        df: DataFrame with 'close' price column

    Returns:
        DataFrame with additional 'log_return' column
    """
    logger.info("Calculating log returns...")

    df = df.with_columns([pl.col("close").log().diff().alias("log_return")])

    # Count and report nulls (first row will be null)
    null_count = df["log_return"].null_count()
    logger.info(f"Log returns calculated. Null count: {null_count:,} (expected: 1)")

    return df


def calculate_realized_volatility(df: pl.DataFrame, window_seconds: int, column_name: str) -> pl.DataFrame:
    """
    Calculate annualized realized volatility over a rolling window.

    Formula:
        σ_RV = sqrt(Σ r²) × sqrt(periods_per_year)

    where:
        r = log return
        periods_per_year = SECONDS_PER_YEAR / window_seconds

    Args:
        df: DataFrame with 'log_return' column
        window_seconds: Rolling window size in seconds
        column_name: Name for the output RV column

    Returns:
        DataFrame with additional realized volatility column
    """
    logger.info(f"Calculating {column_name} (window={window_seconds}s)...")

    # Calculate periods per year for annualization
    periods_per_year = SECONDS_PER_YEAR / window_seconds
    annualization_factor = np.sqrt(periods_per_year)

    # Rolling standard deviation of log returns
    # Note: Polars rolling_std uses sample std (N-1 denominator) by default
    # CRITICAL: Polars uses trailing windows by default (includes current + past N-1 rows)
    # This is correct for time series - NO FUTURE DATA LEAKAGE
    df = df.with_columns([pl.col("log_return").rolling_std(window_size=window_seconds).alias(f"{column_name}_raw")])

    # Annualize
    df = df.with_columns([(pl.col(f"{column_name}_raw") * annualization_factor).alias(column_name)])

    # Drop intermediate column
    df = df.drop(f"{column_name}_raw")

    # Report statistics
    rv_stats = df.select(
        [
            pl.col(column_name).mean().alias("mean"),
            pl.col(column_name).median().alias("median"),
            pl.col(column_name).min().alias("min"),
            pl.col(column_name).max().alias("max"),
            pl.col(column_name).null_count().alias("nulls"),
        ]
    ).to_dicts()[0]

    logger.info(
        f"{column_name}: mean={rv_stats['mean']:.4f}, "
        f"median={rv_stats['median']:.4f}, "
        f"min={rv_stats['min']:.4f}, "
        f"max={rv_stats['max']:.4f}, "
        f"nulls={rv_stats['nulls']:,}"
    )

    return df


def add_rv_cross_features(df: pl.DataFrame) -> pl.DataFrame:
    """
    Add derived features from realized volatilities.

    Features:
        - rv_ratio_5m_1m: Ratio of 5-min RV to 1-min RV (volatility acceleration)
        - rv_ratio_15m_5m: Ratio of 15-min RV to 5-min RV
        - rv_ratio_1h_15m: Ratio of 1-hour RV to 15-min RV
        - rv_term_structure: Slope of RV term structure (1m to 1h)

    Args:
        df: DataFrame with RV columns

    Returns:
        DataFrame with additional cross-feature columns
    """
    logger.info("Adding RV cross-features...")

    df = df.with_columns(
        [
            # Volatility acceleration ratios
            (pl.col("rv_300s") / pl.col("rv_60s")).alias("rv_ratio_5m_1m"),
            (pl.col("rv_900s") / pl.col("rv_300s")).alias("rv_ratio_15m_5m"),
            (pl.col("rv_3600s") / pl.col("rv_900s")).alias("rv_ratio_1h_15m"),
            # Term structure slope (1h RV - 1m RV) / 1m RV
            ((pl.col("rv_3600s") - pl.col("rv_60s")) / pl.col("rv_60s")).alias("rv_term_structure"),
        ]
    )

    # Replace infinities with nulls (can occur if denominator is zero)
    for col in ["rv_ratio_5m_1m", "rv_ratio_15m_5m", "rv_ratio_1h_15m", "rv_term_structure"]:
        df = df.with_columns([pl.when(pl.col(col).is_infinite()).then(None).otherwise(pl.col(col)).alias(col)])

    logger.info("RV cross-features added.")

    return df


def validate_output(df: pl.DataFrame) -> None:
    """
    Validate output data quality.

    Checks:
        - Base RV columns are non-negative
        - Cross-features (ratios, term structure) can be negative (valid market conditions)
        - No excessive nulls (beyond expected window edge effects)

    Args:
        df: Output DataFrame to validate

    Raises:
        ValueError: If validation fails
    """
    logger.info("Validating output data...")

    # Base RV columns (should be non-negative)
    base_rv_cols = ["rv_60s", "rv_300s", "rv_900s", "rv_3600s"]

    for col in base_rv_cols:
        if col not in df.columns:
            continue

        # Check for negative values (base RVs should be positive)
        negative_count = df.filter(pl.col(col) < 0).height
        if negative_count > 0:
            raise ValueError(f"{col} has {negative_count:,} negative values!")

        # Check null percentage
        null_pct = df[col].null_count() / len(df) * 100
        if null_pct > 10:
            logger.warning(f"{col} has {null_pct:.1f}% nulls (expected <10%)")

    # Cross-features (can be negative, e.g., inverted term structure)
    cross_feature_cols = [col for col in df.columns if col.startswith("rv_") and col not in base_rv_cols]

    for col in cross_feature_cols:
        # Check null percentage only
        null_pct = df[col].null_count() / len(df) * 100
        if null_pct > 10:
            logger.warning(f"{col} has {null_pct:.1f}% nulls (expected <10%)")

        # Report range (informational only)
        col_stats = df.select([pl.col(col).min().alias("min"), pl.col(col).max().alias("max")]).to_dicts()[0]
        logger.info(f"{col}: range=[{col_stats['min']:.4f}, {col_stats['max']:.4f}]")

    logger.info("Validation passed.")


def main() -> None:
    """Main execution function."""
    logger.info("=" * 80)
    logger.info("REALIZED VOLATILITY CALCULATION")
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

    # Ensure data is sorted by timestamp
    logger.info("Sorting by timestamp...")
    df = df.sort("timestamp_seconds")

    # Calculate log returns
    df = calculate_log_returns(df)

    # Calculate realized volatility at all windows
    for rv_name, window_seconds in RV_WINDOWS.items():
        df = calculate_realized_volatility(df, window_seconds, rv_name)

    # Add cross-features
    df = add_rv_cross_features(df)

    # Validate
    validate_output(df)

    # Select output columns
    output_cols = [
        "timestamp_seconds",
        "close",  # Keep price for reference
        "log_return",
        "rv_60s",
        "rv_300s",
        "rv_900s",
        "rv_3600s",
        "rv_ratio_5m_1m",
        "rv_ratio_15m_5m",
        "rv_ratio_1h_15m",
        "rv_term_structure",
    ]

    df_output = df.select(output_cols)

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
            pl.col("rv_60s").mean().alias("rv_60s_mean"),
            pl.col("rv_300s").mean().alias("rv_300s_mean"),
            pl.col("rv_900s").mean().alias("rv_900s_mean"),
            pl.col("rv_3600s").mean().alias("rv_3600s_mean"),
            pl.col("rv_ratio_5m_1m").mean().alias("rv_ratio_5m_1m_mean"),
            pl.col("rv_term_structure").mean().alias("rv_term_structure_mean"),
        ]
    ).to_dicts()[0]

    for key, value in summary.items():
        if value is not None:
            logger.info(f"{key}: {value:.4f}")
        else:
            logger.info(f"{key}: null")

    logger.info("=" * 80)
    logger.info("COMPLETED SUCCESSFULLY")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
