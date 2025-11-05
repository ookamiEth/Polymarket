#!/usr/bin/env python3
"""
Resample Binance funding rate data to 1-second intervals.

This script converts event-driven funding rate data (avg 1.47 updates/sec)
into fixed 1-second sampled data for consistent time-series analysis.

Input:  Event-driven data with microsecond timestamps (~117K rows/day)
Output: Fixed 1-second intervals (~79K-86K rows/day depending on gaps)

Usage:
    # Basic resampling (last value per second)
    uv run python resample_funding_rates_to_1s.py \\
        --input-file data/raw/binance_funding_rates/binance-futures/BTCUSDT/2024-10-01.parquet \\
        --output-file data/processed/binance_funding_rates_1s/binance-futures/BTCUSDT/2024-10-01_1s.parquet

    # With forward-fill (creates continuous series)
    uv run python resample_funding_rates_to_1s.py \\
        --input-file data/raw/binance_funding_rates/binance-futures/BTCUSDT/2024-10-01.parquet \\
        --output-file data/processed/binance_funding_rates_1s/binance-futures/BTCUSDT/2024-10-01_1s.parquet \\
        --method forward_fill \\
        --max-fill-gap 60
"""

import argparse
import logging
import os
import sys
import time
from typing import Optional

import polars as pl

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def resample_to_1s_last(df: pl.DataFrame) -> pl.DataFrame:
    """Resample to 1-second intervals using last value per second.

    Args:
        df: Input DataFrame with microsecond timestamps

    Returns:
        DataFrame with 1-second intervals
    """
    # Convert microseconds to seconds (use integer division to avoid float precision issues)
    df = df.with_columns([(pl.col("timestamp") // 1_000_000).cast(pl.Int64).alias("timestamp_seconds")])

    # Aggregate to 1-second intervals
    # Use .last() for state variables (funding rates, prices)
    # Use .filter().last() for fields with NULLs to avoid NULL propagation
    df_1s = (
        df.group_by("timestamp_seconds")
        .agg(
            [
                # Metadata fields (take first)
                pl.col("exchange").first(),
                pl.col("symbol").first(),
                pl.col("timestamp").first(),  # Keep original microsecond timestamp of first event
                pl.col("local_timestamp").first(),
                # State variables (use last value - most recent wins)
                pl.col("funding_timestamp")
                .filter(pl.col("funding_timestamp").is_not_null())
                .last()
                .alias("funding_timestamp"),
                pl.col("funding_rate").filter(pl.col("funding_rate").is_not_null()).last().alias("funding_rate"),
                pl.col("predicted_funding_rate")
                .filter(pl.col("predicted_funding_rate").is_not_null())
                .last()
                .alias("predicted_funding_rate"),
                pl.col("mark_price").filter(pl.col("mark_price").is_not_null()).last().alias("mark_price"),
                pl.col("index_price").filter(pl.col("index_price").is_not_null()).last().alias("index_price"),
                pl.col("open_interest").filter(pl.col("open_interest").is_not_null()).last().alias("open_interest"),
                pl.col("last_price").filter(pl.col("last_price").is_not_null()).last().alias("last_price"),
                # Track update frequency
                pl.len().alias("update_count"),
            ]
        )
        .sort("timestamp_seconds")
    )

    return df_1s


def forward_fill_gaps(df_1s: pl.DataFrame, max_fill_gap: Optional[int] = 60) -> pl.DataFrame:
    """Fill gaps in time series by forward-filling last known values.

    Args:
        df_1s: DataFrame with 1-second data
        max_fill_gap: Maximum gap to fill in seconds (default: 60)

    Returns:
        DataFrame with filled gaps
    """
    if len(df_1s) == 0:
        return df_1s

    # Get time range
    min_ts = df_1s["timestamp_seconds"].min()
    max_ts = df_1s["timestamp_seconds"].max()

    if min_ts is None or max_ts is None:
        logger.warning("Empty timestamp range, skipping forward-fill")
        return df_1s

    # Cast to int for type safety
    min_ts_int = int(min_ts)  # type: ignore[arg-type]
    max_ts_int = int(max_ts)  # type: ignore[arg-type]

    # Create complete time grid
    logger.info(f"Creating time grid from {min_ts_int} to {max_ts_int} ({max_ts_int - min_ts_int + 1} seconds)")
    time_grid = pl.DataFrame({"timestamp_seconds": pl.arange(min_ts_int, max_ts_int + 1, 1, eager=True)})

    # Left join actual data onto time grid
    filled = time_grid.join(df_1s, on="timestamp_seconds", how="left")

    # Reconstruct timestamp column from timestamp_seconds (ensure no nulls)
    filled = filled.with_columns([(pl.col("timestamp_seconds") * 1_000_000).alias("timestamp")])

    # Forward-fill values
    logger.info("Forward-filling values...")
    filled = filled.with_columns(
        [
            pl.col("exchange").forward_fill(),
            pl.col("symbol").forward_fill(),
            pl.col("local_timestamp").forward_fill(),
            pl.col("funding_timestamp").forward_fill(),
            pl.col("funding_rate").forward_fill(),
            pl.col("predicted_funding_rate").forward_fill(),
            pl.col("mark_price").forward_fill(),
            pl.col("index_price").forward_fill(),
            pl.col("open_interest").forward_fill(),
            pl.col("last_price").forward_fill(),
        ]
    )

    # Calculate gaps and apply max_fill_gap constraint
    if max_fill_gap is not None:
        logger.info(f"Applying max fill gap constraint: {max_fill_gap} seconds")

        # Create flag for rows with actual data
        filled = filled.with_columns([pl.col("update_count").is_not_null().alias("has_data")])

        # Calculate seconds since last real data point
        filled = filled.with_columns(
            [pl.when(pl.col("has_data")).then(pl.lit(0)).otherwise(pl.lit(None)).alias("seconds_since_update")]
        )

        # Forward-fill the gap counter
        filled = filled.with_columns([pl.col("seconds_since_update").forward_fill()])

        # Create cumulative counter
        filled = filled.with_columns(
            [
                pl.when(pl.col("has_data"))
                .then(pl.lit(0))
                .otherwise(pl.col("timestamp_seconds").cum_count().over(pl.col("has_data").cum_sum()))
                .alias("gap_size")
            ]
        )

        # Apply constraint - set to NULL if gap exceeds limit
        for col in [
            "funding_rate",
            "mark_price",
            "index_price",
            "open_interest",
            "last_price",
        ]:
            filled = filled.with_columns(
                [
                    pl.when(pl.col("update_count").is_not_null())
                    .then(pl.col(col))
                    .when(pl.col("gap_size") <= max_fill_gap)
                    .then(pl.col(col))
                    .otherwise(None)
                    .alias(col)
                ]
            )

        # Fill update_count=0 for forward-filled rows (within gap limit)
        filled = filled.with_columns(
            [
                pl.when(pl.col("update_count").is_not_null())
                .then(pl.col("update_count"))
                .when(pl.col("gap_size") <= max_fill_gap)
                .then(pl.lit(0))
                .otherwise(None)
                .alias("update_count")
            ]
        )

        # Drop helper columns
        filled = filled.drop(["has_data", "seconds_since_update", "gap_size"])

    else:
        # No gap limit - fill all gaps with 0 update_count
        filled = filled.with_columns([pl.col("update_count").fill_null(pl.lit(0))])

    return filled


def resample_file(
    input_file: str,
    output_file: str,
    method: str = "last",
    max_fill_gap: Optional[int] = None,
    verbose: bool = False,
) -> dict:
    """Resample a single funding rate file to 1-second intervals.

    Args:
        input_file: Path to input Parquet file
        output_file: Path to output Parquet file
        method: Resampling method ("last" or "forward_fill")
        max_fill_gap: Maximum gap to fill in seconds (only for forward_fill)
        verbose: Enable verbose logging

    Returns:
        Dictionary with statistics
    """
    start_time = time.time()

    if verbose:
        logger.setLevel(logging.DEBUG)

    logger.info(f"Processing: {input_file}")

    # Read input
    df = pl.read_parquet(input_file)
    input_rows = len(df)
    logger.info(f"  Input: {input_rows:,} rows (event-driven)")

    # Resample to 1-second
    df_1s = resample_to_1s_last(df)
    output_rows_sparse = len(df_1s)
    logger.info(f"  Resampled: {output_rows_sparse:,} rows (1-second, sparse)")

    # Optional: Forward-fill
    if method == "forward_fill":
        if max_fill_gap is None:
            max_fill_gap = 60  # Default: 1 minute
        df_1s = forward_fill_gaps(df_1s, max_fill_gap)
        output_rows_filled = len(df_1s)
        logger.info(f"  Forward-filled: {output_rows_filled:,} rows (continuous, max gap: {max_fill_gap}s)")

    # Drop helper column
    if "timestamp_seconds" in df_1s.columns:
        df_1s = df_1s.drop("timestamp_seconds")

    # Write output
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    df_1s.write_parquet(output_file, compression="snappy", statistics=True)

    # Calculate statistics
    elapsed = time.time() - start_time
    input_size_mb = os.path.getsize(input_file) / (1024**2)
    output_size_mb = os.path.getsize(output_file) / (1024**2)

    logger.info(f"  Output: {output_file}")
    logger.info(f"  Size: {input_size_mb:.2f} MB → {output_size_mb:.2f} MB")
    logger.info(f"  Time: {elapsed:.1f}s")

    return {
        "input_file": input_file,
        "output_file": output_file,
        "input_rows": input_rows,
        "output_rows": len(df_1s),
        "input_size_mb": input_size_mb,
        "output_size_mb": output_size_mb,
        "elapsed_seconds": elapsed,
    }


def main() -> None:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Resample Binance funding rate data to 1-second intervals",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic resampling (last value per second)
  uv run python resample_funding_rates_to_1s.py \\
      --input-file data/raw/binance_funding_rates/binance-futures/BTCUSDT/2024-10-01.parquet \\
      --output-file data/processed/binance_funding_rates_1s/binance-futures/BTCUSDT/2024-10-01_1s.parquet

  # With forward-fill
  uv run python resample_funding_rates_to_1s.py \\
      --input-file data/raw/.../2024-10-01.parquet \\
      --output-file data/processed/.../2024-10-01_1s.parquet \\
      --method forward_fill \\
      --max-fill-gap 60
        """,
    )

    parser.add_argument("--input-file", required=True, help="Input Parquet file (event-driven data)")
    parser.add_argument("--output-file", required=True, help="Output Parquet file (1-second data)")
    parser.add_argument(
        "--method",
        choices=["last", "forward_fill"],
        default="last",
        help="Resampling method (default: last)",
    )
    parser.add_argument(
        "--max-fill-gap",
        type=int,
        default=None,
        help="Maximum gap to forward-fill in seconds (only with forward_fill method)",
    )
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable verbose logging")

    args = parser.parse_args()

    # Validate input file exists
    if not os.path.exists(args.input_file):
        logger.error(f"Input file not found: {args.input_file}")
        sys.exit(1)

    # Process file
    try:
        stats = resample_file(
            input_file=args.input_file,
            output_file=args.output_file,
            method=args.method,
            max_fill_gap=args.max_fill_gap,
            verbose=args.verbose,
        )

        logger.info("\n" + "=" * 60)
        logger.info("RESAMPLING COMPLETE")
        logger.info("=" * 60)
        logger.info(f"Input rows: {stats['input_rows']:,}")
        logger.info(f"Output rows: {stats['output_rows']:,}")
        logger.info(f"Compression: {stats['output_rows'] / stats['input_rows'] * 100:.1f}% of original")
        logger.info(f"Processing time: {stats['elapsed_seconds']:.1f}s")
        logger.info("=" * 60)

    except Exception as e:
        logger.error(f"Processing failed: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
