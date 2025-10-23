#!/usr/bin/env python3
"""
Match options IV to binary contract pricing timestamps.

For each second during a binary contract's lifetime, find the "closest expiry" option
that expires AFTER the contract closes (to avoid lookahead bias).

Key constraint: options.expiry_timestamp > contract.close_time

Strategy:
1. For each contract, filter options to those expiring after contract close
2. For each pricing timestamp, find option with minimum time_to_expiry
3. Extract bid and ask IV
4. Handle missing data gracefully

This module provides both DataFrame join utilities and standalone functions.
"""

import logging
from pathlib import Path

import polars as pl

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# File paths
OPTIONS_FILE = Path(
    "/Users/lgierhake/Documents/ETH/BT/research/tardis/data/consolidated/btc_options_atm_shortdated_with_iv_2023_2025.parquet"
)


def filter_valid_options(options_df: pl.LazyFrame, min_expiry: int) -> pl.LazyFrame:
    """
    Filter options to only those expiring after min_expiry timestamp.

    Args:
        options_df: Lazy options DataFrame
        min_expiry: Minimum expiry timestamp (options must expire after this)

    Returns:
        Filtered LazyFrame with only valid options
    """
    return options_df.filter(
        (pl.col("expiry_timestamp") > min_expiry)
        & (pl.col("iv_calc_status") == "success")
        & (pl.col("implied_vol_bid").is_not_null())
        & (pl.col("implied_vol_ask").is_not_null())
        & (pl.col("implied_vol_bid") > 0.01)  # Reasonable vol bounds
        & (pl.col("implied_vol_bid") < 5.0)
    )


def prepare_options_for_join(
    options_path: Path,
    min_timestamp: int,
    max_timestamp: int,
    min_expiry: int,
) -> pl.DataFrame:
    """
    Load and prepare options data for joining with pricing grid.

    Filters to relevant time window and valid options only.

    Args:
        options_path: Path to options parquet file
        min_timestamp: Earliest quote timestamp needed
        max_timestamp: Latest quote timestamp needed
        min_expiry: Minimum expiry timestamp (contracts must close before this)

    Returns:
        DataFrame with columns: timestamp_seconds, expiry_timestamp,
                                time_to_expiry_seconds, implied_vol_bid, implied_vol_ask

    Example:
        >>> # For contracts running Oct 1-7, 2023
        >>> min_ts = 1696118400  # Oct 1 00:00
        >>> max_ts = 1696723200  # Oct 8 00:00
        >>> min_exp = 1696723200  # Options must expire after Oct 8
        >>> opts = prepare_options_for_join(OPTIONS_FILE, min_ts, max_ts, min_exp)
    """
    logger.info(f"Loading options from {options_path}")
    logger.info(f"Time window: {min_timestamp} to {max_timestamp}")
    logger.info(f"Min expiry: {min_expiry}")

    # Lazy scan
    options = pl.scan_parquet(options_path)

    # Filter to relevant time window and valid options
    options_filtered = options.filter(
        (pl.col("timestamp_seconds") >= min_timestamp)
        & (pl.col("timestamp_seconds") <= max_timestamp)
        & (pl.col("expiry_timestamp") > min_expiry)
        & (pl.col("iv_calc_status") == "success")
        & (pl.col("implied_vol_bid").is_not_null())
        & (pl.col("implied_vol_ask").is_not_null())
        & (pl.col("implied_vol_bid") > 0.01)
        & (pl.col("implied_vol_bid") < 5.0)
    )

    # Select only needed columns
    options_subset = options_filtered.select(
        [
            "timestamp_seconds",
            "expiry_timestamp",
            "time_to_expiry_seconds",
            "implied_vol_bid",
            "implied_vol_ask",
            "moneyness",  # Useful for filtering ATM options
            "type",  # call or put
        ]
    )

    # Collect
    logger.info("Collecting filtered options...")
    df = options_subset.collect()

    logger.info(f"Loaded {len(df):,} option quotes")

    return df


def find_closest_expiry_per_timestamp(
    options_df: pl.DataFrame,
    timestamp: int,
    min_expiry: int,
) -> pl.DataFrame:
    """
    Find the closest-expiry option for a specific timestamp.

    Args:
        options_df: Options DataFrame (pre-filtered to relevant period)
        timestamp: The pricing timestamp to match
        min_expiry: Minimum expiry timestamp (option must expire after this)

    Returns:
        Single-row DataFrame with the closest expiry option, or empty if none found

    Example:
        >>> opts = pl.DataFrame({...})  # options data
        >>> closest = find_closest_expiry_per_timestamp(opts, 1696118400, 1696119300)
        >>> if len(closest) > 0:
        ...     sigma_bid = closest["implied_vol_bid"][0]
    """
    # Filter to exact timestamp and valid expiry
    matched = options_df.filter((pl.col("timestamp_seconds") == timestamp) & (pl.col("expiry_timestamp") > min_expiry))

    if len(matched) == 0:
        return pl.DataFrame()

    # Find minimum time_to_expiry (closest expiry)
    closest = matched.sort("time_to_expiry_seconds").head(1)

    return closest


def add_closest_iv_columns(
    pricing_grid: pl.DataFrame,
    options_df: pl.DataFrame,
    timestamp_col: str = "timestamp",
    min_expiry_col: str = "close_time",
) -> pl.DataFrame:
    """
    Add implied volatility columns to pricing grid using closest-expiry matching.

    For each row in pricing_grid, finds the option with:
    - Same timestamp_seconds
    - Expires after min_expiry (contract close)
    - Minimum time_to_expiry among matches

    IMPORTANT: This is a row-by-row operation and may be slow for large grids.
    Consider using join-based approach for production (see backtest_engine.py).

    Args:
        pricing_grid: DataFrame with columns: timestamp, close_time (or custom names)
        options_df: Options DataFrame (from prepare_options_for_join)
        timestamp_col: Column name for pricing timestamp
        min_expiry_col: Column name for minimum expiry threshold

    Returns:
        DataFrame with added columns: implied_vol_bid, implied_vol_ask, option_expiry

    Example:
        >>> pricing = pl.DataFrame({
        ...     "timestamp": [1696118400, 1696118401],
        ...     "close_time": [1696119300, 1696119300],
        ... })
        >>> options = prepare_options_for_join(...)
        >>> pricing_with_iv = add_closest_iv_columns(pricing, options)
    """
    logger.warning("Using row-by-row IV matching - this may be slow for large datasets!")
    logger.warning("Consider using join-based approach in backtest_engine for production.")

    # This is intentionally simple for demonstration
    # Production implementation should use optimized joins
    raise NotImplementedError(
        "Row-by-row matching not implemented. "
        "Use join-based approach in backtest_engine.py for production. "
        "See README.md for vectorized join strategy."
    )


def join_options_asof(
    pricing_grid: pl.LazyFrame,
    options_df: pl.LazyFrame,
    timestamp_col: str = "timestamp",
) -> pl.LazyFrame:
    """
    Join options to pricing grid using asof join on timestamp.

    This is more efficient than row-by-row matching but requires additional
    filtering to ensure expiry > close_time constraint.

    Args:
        pricing_grid: Lazy pricing DataFrame
        options_df: Lazy options DataFrame
        timestamp_col: Timestamp column name

    Returns:
        Lazy joined DataFrame

    Note:
        This function provides the JOIN operation only. Additional filtering
        for expiry constraints must be done in the calling code.
    """
    # Asof join on timestamp (gets nearest option quote)
    joined = pricing_grid.join_asof(
        options_df,
        left_on=timestamp_col,
        right_on="timestamp_seconds",
        strategy="backward",  # Use most recent option quote
    )

    return joined


# Utility function to check data availability
def check_option_coverage(
    options_path: Path,
    start_timestamp: int,
    end_timestamp: int,
) -> dict[str, int]:
    """
    Check how many timestamps have option data in a given period.

    Args:
        options_path: Path to options parquet
        start_timestamp: Start of period (seconds)
        end_timestamp: End of period (seconds)

    Returns:
        Dict with statistics: total_seconds, seconds_with_options, coverage_pct

    Example:
        >>> stats = check_option_coverage(OPTIONS_FILE, 1696118400, 1696204800)
        >>> print(f"Coverage: {stats['coverage_pct']:.1f}%")
    """
    logger.info(f"Checking option coverage from {start_timestamp} to {end_timestamp}")

    # Expected seconds
    total_seconds = end_timestamp - start_timestamp + 1

    # Load options in time window
    options = pl.scan_parquet(options_path)
    filtered = options.filter(
        (pl.col("timestamp_seconds") >= start_timestamp)
        & (pl.col("timestamp_seconds") <= end_timestamp)
        & (pl.col("iv_calc_status") == "success")
    )

    # Count unique timestamps
    unique_timestamps = filtered.select(pl.col("timestamp_seconds").n_unique()).collect().item()

    coverage_pct = (unique_timestamps / total_seconds) * 100

    stats = {
        "total_seconds": total_seconds,
        "seconds_with_options": unique_timestamps,
        "coverage_pct": coverage_pct,
    }

    logger.info(f"Total seconds: {total_seconds:,}")
    logger.info(f"Seconds with options: {unique_timestamps:,}")
    logger.info(f"Coverage: {coverage_pct:.2f}%")

    return stats


if __name__ == "__main__":
    print("Options IV Matching Module")
    print("=" * 60)

    # Test: Check option coverage for first week of data
    print("\nTest: Checking option coverage for Oct 1-7, 2023")
    stats = check_option_coverage(
        OPTIONS_FILE,
        start_timestamp=1696118400,  # Oct 1 00:00 UTC
        end_timestamp=1696723200,  # Oct 8 00:00 UTC
    )

    print(f"\nCoverage: {stats['coverage_pct']:.1f}%")
    print(f"Seconds with data: {stats['seconds_with_options']:,} / {stats['total_seconds']:,}")

    print("\n✅ Module tests complete!")
