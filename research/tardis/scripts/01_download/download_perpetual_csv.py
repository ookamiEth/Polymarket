#!/usr/bin/env python3
"""
Download and process Deribit perpetual futures data using CSV download API.

This script works with academic Tardis API subscriptions which only support
CSV downloads (not the tardis-machine streaming API).

Steps:
1. Download daily CSV.gz files from Tardis
2. Combine CSVs into single Parquet file
3. Resample to 1-second VWAP intervals
"""

import argparse
import logging
import os
import sys
import time
from pathlib import Path

import polars as pl
from dotenv import load_dotenv
from tardis_dev import datasets

logger = logging.getLogger(__name__)


def download_csv_files(
    from_date: str,
    to_date: str,
    symbol: str,
    output_dir: str,
    api_key: str,
    concurrent: int = 10,
) -> Path:
    """Download CSV files from Tardis datasets API."""
    logger.info("=" * 80)
    logger.info("STEP 1: DOWNLOADING CSV FILES FROM TARDIS")
    logger.info("=" * 80)
    logger.info(f"Symbol: {symbol}")
    logger.info(f"Date range: {from_date} to {to_date}")
    logger.info(f"Output dir: {output_dir}")
    logger.info(f"Concurrent downloads: {concurrent}")
    logger.info("")

    os.makedirs(output_dir, exist_ok=True)

    start_time = time.time()

    datasets.download(
        exchange="deribit",
        data_types=["trades"],
        from_date=from_date,
        to_date=to_date,
        symbols=[symbol],
        api_key=api_key,
        download_dir=output_dir,
        concurrency=concurrent,
    )

    elapsed = time.time() - start_time
    logger.info(f"✅ Downloaded CSV files in {elapsed / 60:.1f} minutes")
    logger.info("")

    return Path(output_dir)


def combine_csvs(input_dir: Path, output_file: str) -> pl.DataFrame:
    """Combine all CSV.gz files into a single Parquet file."""
    logger.info("=" * 80)
    logger.info("STEP 2: COMBINING CSV FILES")
    logger.info("=" * 80)

    csv_files = sorted(input_dir.glob("*.csv.gz"))
    if not csv_files:
        raise ValueError(f"No CSV files found in {input_dir}")

    logger.info(f"Found {len(csv_files)} CSV files")
    logger.info(f"Output: {output_file}")
    logger.info("")

    # Schema for Deribit trades CSV
    schema = {
        "exchange": pl.Utf8,
        "symbol": pl.Utf8,
        "timestamp": pl.Int64,  # Microseconds
        "local_timestamp": pl.Int64,
        "id": pl.Utf8,
        "side": pl.Utf8,
        "price": pl.Float64,
        "amount": pl.Float64,
    }

    start_time = time.time()

    logger.info("Reading and concatenating CSV files...")
    dfs = []
    for i, csv_file in enumerate(csv_files, 1):
        if i % 50 == 0 or i == len(csv_files):
            logger.info(f"  Processing file {i}/{len(csv_files)}: {csv_file.name}")

        df = pl.scan_csv(csv_file, schema=schema, has_header=True)
        dfs.append(df)

    logger.info("Concatenating and sorting by timestamp...")
    combined_df = pl.concat(dfs).sort("timestamp").collect()

    elapsed = time.time() - start_time
    logger.info(f"✅ Combined {len(combined_df):,} rows in {elapsed:.1f}s")
    logger.info("")

    return combined_df


def resample_to_1s_vwap(df: pl.DataFrame, output_file: str) -> None:
    """Resample trades to 1-second intervals with VWAP and last price."""
    logger.info("=" * 80)
    logger.info("STEP 3: RESAMPLING TO 1-SECOND INTERVALS (VWAP + LAST PRICE)")
    logger.info("=" * 80)

    start_time = time.time()

    # Convert timestamp from microseconds to seconds (for grouping)
    logger.info("Converting timestamps to seconds...")
    df = df.with_columns(
        [
            (pl.col("timestamp") // 1_000_000).alias("timestamp_seconds"),
            (pl.col("price") * pl.col("amount")).alias("volume_dollars"),
        ]
    )

    # Group by timestamp_seconds and calculate VWAP + last price
    logger.info("Calculating 1-second VWAP and last price...")
    resampled = (
        df.group_by("timestamp_seconds")
        .agg(
            [
                pl.col("volume_dollars").sum().alias("total_volume_dollars"),
                pl.col("amount").sum().alias("total_amount"),
                pl.col("price").mean().alias("price_mean"),
                pl.col("price").min().alias("price_min"),
                pl.col("price").max().alias("price_max"),
                pl.col("price").last().alias("last_price"),  # Chronologically last trade price
                pl.col("timestamp").first().alias("timestamp_us"),  # Keep microsecond precision
                pl.len().alias("trade_count"),
            ]
        )
        .with_columns(
            [
                # VWAP = total_volume_dollars / total_amount
                (pl.col("total_volume_dollars") / pl.col("total_amount")).alias("vwap"),
                # Convert timestamp_seconds back to microseconds
                (pl.col("timestamp_seconds") * 1_000_000).alias("timestamp"),
            ]
        )
        .with_columns(
            [
                # Add 'price' as alias for 'last_price' (primary price, backwards compatible)
                pl.col("last_price").alias("price"),
                # Add 'close' as alias for 'last_price' (OHLC naming convention)
                pl.col("last_price").alias("close"),
            ]
        )
        .select(
            [
                "timestamp",
                "timestamp_seconds",
                "last_price",  # Primary price (chronologically last trade)
                "close",  # Alias for last_price (OHLC convention)
                "price",  # Alias for last_price (backwards compatibility)
                "vwap",  # Keep for validation/comparison
                "price_mean",
                "price_min",
                "price_max",
                "total_amount",
                "trade_count",
            ]
        )
        .sort("timestamp_seconds")
    )

    # Write to Parquet
    logger.info(f"Writing resampled data to {output_file}...")
    resampled.write_parquet(output_file, compression="snappy", statistics=True)

    elapsed = time.time() - start_time
    file_size_mb = os.path.getsize(output_file) / 1_000_000

    logger.info("=" * 80)
    logger.info("SUCCESS!")
    logger.info(f"Total 1-second intervals: {len(resampled):,}")
    logger.info(f"Output file: {output_file}")
    logger.info(f"File size: {file_size_mb:.2f} MB")
    logger.info(f"Resampling time: {elapsed:.1f}s")
    logger.info("=" * 80)

    # Summary statistics
    logger.info("")
    logger.info("Data Summary:")
    logger.info(
        f"  Date range (seconds): {resampled['timestamp_seconds'].min()} to {resampled['timestamp_seconds'].max()}"
    )
    logger.info(f"  VWAP range: ${resampled['vwap'].min():.2f} - ${resampled['vwap'].max():.2f}")
    logger.info(f"  Average trades per second: {resampled['trade_count'].mean():.1f}")
    logger.info(f"  Total volume: {resampled['total_amount'].sum():,.2f} BTC")


def main():
    parser = argparse.ArgumentParser(
        description="Download and process Deribit perpetual futures using CSV API (academic subscription)"
    )

    parser.add_argument("--from-date", required=True, help="Start date (YYYY-MM-DD)")
    parser.add_argument("--to-date", required=True, help="End date (YYYY-MM-DD)")
    parser.add_argument("--symbol", default="BTC-PERPETUAL", help="Symbol (default: BTC-PERPETUAL)")
    parser.add_argument("--output-file", required=True, help="Output Parquet file path")
    parser.add_argument("--temp-dir", help="Temporary directory for CSV downloads (default: auto-generated)")
    parser.add_argument("--keep-csvs", action="store_true", help="Keep CSV files after processing")
    parser.add_argument("--concurrent", type=int, default=10, help="Concurrent downloads (default: 10)")
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable debug logging")

    args = parser.parse_args()

    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(level=log_level, format="%(message)s", handlers=[logging.StreamHandler(sys.stdout)])

    # Load API key from .env
    load_dotenv()
    api_key = os.getenv("TARDIS_API_KEY")
    if not api_key:
        logger.error("ERROR: TARDIS_API_KEY not found in .env file")
        sys.exit(1)

    # Set up temporary directory
    temp_dir = args.temp_dir or f"./temp_perpetual_csv_{int(time.time())}"

    logger.info("=" * 80)
    logger.info("DERIBIT PERPETUAL CSV DOWNLOAD & PROCESSING")
    logger.info("=" * 80)
    logger.info(f"API Key: {api_key[:10]}... (academic subscription)")
    logger.info(f"Temporary directory: {temp_dir}")
    logger.info(f"Final output: {args.output_file}")
    logger.info("")

    try:
        # Step 1: Download CSV files
        csv_dir = download_csv_files(
            from_date=args.from_date,
            to_date=args.to_date,
            symbol=args.symbol,
            output_dir=temp_dir,
            api_key=api_key,
            concurrent=args.concurrent,
        )

        # Step 2: Combine CSVs
        temp_combined = f"{temp_dir}/combined_trades.parquet"
        combined_df = combine_csvs(csv_dir, temp_combined)

        # Step 3: Resample to 1-second VWAP
        resample_to_1s_vwap(combined_df, args.output_file)

        # Clean up temporary files
        if not args.keep_csvs:
            logger.info("")
            logger.info("Cleaning up temporary CSV files...")
            import shutil

            shutil.rmtree(temp_dir)
            logger.info(f"✅ Removed temporary directory: {temp_dir}")

        logger.info("")
        logger.info("=" * 80)
        logger.info("✅ PIPELINE COMPLETE")
        logger.info("=" * 80)

    except Exception as e:
        logger.error(f"ERROR: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
