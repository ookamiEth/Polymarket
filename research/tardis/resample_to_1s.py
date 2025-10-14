#!/usr/bin/env python3

import argparse
import logging
import os
import sys
import time

import polars as pl

logger = logging.getLogger(__name__)


def resample_to_1s(
    input_file: str,
    output_file: str,
    method: str = "first",
) -> None:
    """Resample trade data to 1-second intervals.

    Args:
        input_file: Input Parquet file with raw trades
        output_file: Output Parquet file with resampled data
        method: Resampling method - 'first' (first trade per second) or 'ohlcv' (OHLCV aggregation)
    """
    logger.info(f"Reading input file: {input_file}")
    start_time = time.time()

    # Read the Parquet file
    df = pl.read_parquet(input_file)

    logger.info(f"Loaded {len(df):,} trades in {time.time() - start_time:.1f}s")
    logger.info("=" * 80)

    # Convert microseconds to datetime
    logger.info("Converting timestamps to datetime...")
    df = df.with_columns(
        [
            (pl.col("timestamp") / 1_000_000).cast(pl.Int64).alias("timestamp_seconds"),
        ]
    )

    if method == "first":
        logger.info("Resampling method: First trade per second")
        logger.info("Grouping by second and taking first trade...")

        # Group by second and take first row
        resampled_df = (
            df.group_by("timestamp_seconds")
            .agg(
                [
                    pl.col("exchange").first(),
                    pl.col("symbol").first(),
                    pl.col("timestamp").first(),
                    pl.col("local_timestamp").first(),
                    pl.col("id").first(),
                    pl.col("side").first(),
                    pl.col("price").first().alias("price"),
                    pl.col("amount").first().alias("amount"),
                ]
            )
            .sort("timestamp_seconds")
        )

    elif method == "ohlcv":
        logger.info("Resampling method: OHLCV aggregation")
        logger.info("Calculating OHLC, volume, and trade count per second...")

        # Group by second and calculate OHLCV
        resampled_df = (
            df.group_by("timestamp_seconds")
            .agg(
                [
                    pl.col("exchange").first(),
                    pl.col("symbol").first(),
                    pl.col("timestamp").first(),  # First timestamp in the second
                    pl.col("local_timestamp").first(),
                    pl.col("price").first().alias("open"),
                    pl.col("price").max().alias("high"),
                    pl.col("price").min().alias("low"),
                    pl.col("price").last().alias("close"),
                    pl.col("amount").sum().alias("volume"),
                    pl.col("id").count().alias("trade_count"),
                    # Volume-weighted average price
                    (pl.col("price") * pl.col("amount")).sum() / pl.col("amount").sum(),
                ]
            )
            .sort("timestamp_seconds")
        )

        # Rename VWAP column
        resampled_df = resampled_df.rename({"price": "vwap"})

    else:
        raise ValueError(f"Invalid method: {method}. Must be 'first' or 'ohlcv'")

    # Drop the helper column
    resampled_df = resampled_df.drop("timestamp_seconds")

    logger.info(f"Resampled to {len(resampled_df):,} rows (1-second intervals)")

    # Write to Parquet
    logger.info("Writing to Parquet file...")
    write_start = time.time()

    resampled_df.write_parquet(
        output_file,
        compression="snappy",
        statistics=True,
        use_pyarrow=False,
    )

    write_elapsed = time.time() - write_start
    file_size_mb = os.path.getsize(output_file) / 1_000_000

    logger.info("=" * 80)
    logger.info("SUCCESS!")
    logger.info(f"Total rows: {len(resampled_df):,}")
    logger.info(f"Output file: {output_file}")
    logger.info(f"File size: {file_size_mb:.2f} MB")
    logger.info(f"Write time: {write_elapsed:.1f}s")
    logger.info(f"Total time: {time.time() - start_time:.1f}s")
    logger.info("=" * 80)

    # Display summary statistics
    logger.info("Data Summary:")
    logger.info(f"Date range: {resampled_df['timestamp'].min()} to {resampled_df['timestamp'].max()}")

    if method == "first":
        logger.info(f"Price range: ${resampled_df['price'].min():.2f} - ${resampled_df['price'].max():.2f}")
    else:
        logger.info(f"Price range: ${resampled_df['low'].min():.2f} - ${resampled_df['high'].max():.2f}")
        logger.info(f"Total volume: {resampled_df['volume'].sum():,.0f}")
        logger.info(f"Total trades: {resampled_df['trade_count'].sum():,}")


def main():
    parser = argparse.ArgumentParser(description="Resample trade data to 1-second intervals")

    parser.add_argument(
        "--input-file",
        default="./deribit_btc_perpetual_trades_raw.parquet",
        help="Input Parquet file with raw trades",
    )
    parser.add_argument(
        "--output-file",
        default="./deribit_btc_perpetual_1s.parquet",
        help="Output Parquet file with resampled data",
    )
    parser.add_argument(
        "--method",
        default="first",
        choices=["first", "ohlcv"],
        help="Resampling method: 'first' (first trade per second) or 'ohlcv' (OHLCV aggregation)",
    )
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable debug logging")

    args = parser.parse_args()

    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(level=log_level, format="%(message)s", handlers=[logging.StreamHandler(sys.stdout)])

    logger.info("=" * 80)
    logger.info("RESAMPLE TO 1-SECOND INTERVALS")
    logger.info("=" * 80)

    try:
        resample_to_1s(
            input_file=args.input_file,
            output_file=args.output_file,
            method=args.method,
        )
    except Exception as e:
        logger.error(f"ERROR: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
