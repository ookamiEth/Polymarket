#!/usr/bin/env python3
"""
Resample Binance orderbook snapshot (top 25 levels) data to 1-second intervals.

This script converts event-driven orderbook snapshots (potentially multiple updates per second)
into fixed 1-second sampled data for consistent time-series analysis.

Input:  Event-driven orderbook snapshots with microsecond timestamps
Output: Fixed 1-second intervals with last snapshot per second

Usage:
    uv run python resample_orderbook_25_to_1s.py \
        --input-file data/raw/binance_orderbook_25/binance-futures/BTCUSDT/2024-10-01.parquet \
        --output-file data/processed/binance_orderbook_25_1s/binance-futures/BTCUSDT/2024-10-01_1s.parquet
"""

import argparse
import logging
import os
import sys
import time

import polars as pl

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def resample_to_1s_last(df: pl.DataFrame) -> pl.DataFrame:
    """Resample to 1-second intervals using last snapshot per second.

    Args:
        df: Input DataFrame with microsecond timestamps

    Returns:
        DataFrame with 1-second intervals
    """
    # Convert microseconds to seconds (use integer division to avoid float precision issues)
    df = df.with_columns([(pl.col("timestamp") // 1_000_000).cast(pl.Int64).alias("timestamp_seconds")])

    # Generate list of all price level columns (50 ask + 50 bid)
    price_level_cols = []
    for i in range(25):
        price_level_cols.extend([f"ask_price_{i}", f"ask_amount_{i}"])
    for i in range(25):
        price_level_cols.extend([f"bid_price_{i}", f"bid_amount_{i}"])

    # Build aggregation expressions
    agg_exprs = [
        # Metadata fields (take first)
        pl.col("exchange").first(),
        pl.col("symbol").first(),
        pl.col("timestamp").first(),  # Keep original microsecond timestamp of first event
        pl.col("local_timestamp").first(),
    ]

    # Add aggregation for all price levels (use last value - most recent snapshot wins)
    for col in price_level_cols:
        agg_exprs.append(pl.col(col).filter(pl.col(col).is_not_null()).last().alias(col))

    # Track snapshot frequency
    agg_exprs.append(pl.len().alias("snapshot_count"))

    # Aggregate to 1-second intervals
    df_1s = df.group_by("timestamp_seconds").agg(agg_exprs).sort("timestamp_seconds")

    return df_1s


def resample_file(
    input_file: str,
    output_file: str,
    verbose: bool = False,
) -> dict:
    """Resample a single orderbook snapshot file to 1-second intervals.

    Args:
        input_file: Path to input Parquet file
        output_file: Path to output Parquet file
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
    output_rows = len(df_1s)
    logger.info(f"  Resampled: {output_rows:,} rows (1-second, last snapshot per second)")

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
        "output_rows": output_rows,
        "input_size_mb": input_size_mb,
        "output_size_mb": output_size_mb,
        "elapsed_seconds": elapsed,
    }


def main() -> None:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Resample Binance orderbook snapshot (top 25 levels) data to 1-second intervals",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Resample single file
  uv run python resample_orderbook_25_to_1s.py \
      --input-file data/raw/binance_orderbook_25/binance-futures/BTCUSDT/2024-10-01.parquet \
      --output-file data/processed/binance_orderbook_25_1s/binance-futures/BTCUSDT/2024-10-01_1s.parquet
        """,
    )

    parser.add_argument("--input-file", required=True, help="Input Parquet file (event-driven data)")
    parser.add_argument("--output-file", required=True, help="Output Parquet file (1-second data)")
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
