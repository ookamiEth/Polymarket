#!/usr/bin/env python3

import argparse
import logging
import os
import sys
import time
from pathlib import Path

import polars as pl

logger = logging.getLogger(__name__)


def combine_csvs_to_parquet(
    input_dir: str,
    output_file: str,
    pattern: str = "*.csv.gz",
) -> None:
    """Combine multiple CSV files into a single Parquet file using Polars."""
    input_path = Path(input_dir)
    csv_files = sorted(input_path.glob(pattern))

    if not csv_files:
        raise ValueError(f"No CSV files found matching pattern '{pattern}' in {input_dir}")

    logger.info(f"Found {len(csv_files)} CSV files to process")
    logger.info(f"Input directory: {input_dir}")
    logger.info(f"Output file: {output_file}")
    logger.info("=" * 80)

    # Define schema for trades CSV
    schema = {
        "exchange": pl.Utf8,
        "symbol": pl.Utf8,
        "timestamp": pl.Int64,  # Microseconds since epoch
        "local_timestamp": pl.Int64,
        "id": pl.Utf8,
        "side": pl.Utf8,
        "price": pl.Float64,
        "amount": pl.Float64,
    }

    start_time = time.time()

    logger.info("Reading and combining CSV files...")

    # Use scan_csv for lazy evaluation (memory efficient)
    # Read all files and concatenate
    dfs = []
    for i, csv_file in enumerate(csv_files, 1):
        if i % 50 == 0:
            logger.info(f"Processing file {i}/{len(csv_files)}: {csv_file.name}")

        df = pl.scan_csv(
            csv_file,
            schema=schema,
            has_header=True,
        )
        dfs.append(df)

    # Concatenate all dataframes
    logger.info("Concatenating all dataframes...")
    combined_df = pl.concat(dfs)

    # Sort by timestamp (critical for time-series analysis)
    logger.info("Sorting by timestamp...")
    combined_df = combined_df.sort("timestamp")

    # Collect (materialize the lazy dataframe)
    logger.info("Materializing dataframe (this may take a few minutes)...")
    combined_df = combined_df.collect()

    elapsed = time.time() - start_time
    logger.info(f"Combined {len(combined_df):,} rows in {elapsed:.1f}s")

    # Write to Parquet with compression
    logger.info("Writing to Parquet file...")
    write_start = time.time()

    combined_df.write_parquet(
        output_file,
        compression="snappy",
        statistics=True,
        use_pyarrow=False,
    )

    write_elapsed = time.time() - write_start
    file_size_mb = os.path.getsize(output_file) / 1_000_000

    logger.info("=" * 80)
    logger.info("SUCCESS!")
    logger.info(f"Total rows: {len(combined_df):,}")
    logger.info(f"Output file: {output_file}")
    logger.info(f"File size: {file_size_mb:.2f} MB")
    logger.info(f"Write time: {write_elapsed:.1f}s")
    logger.info(f"Total time: {time.time() - start_time:.1f}s")
    logger.info("=" * 80)

    # Display summary statistics
    logger.info("Data Summary:")
    logger.info(f"Date range: {combined_df['timestamp'].min()} to {combined_df['timestamp'].max()}")
    logger.info(f"Unique symbols: {combined_df['symbol'].n_unique()}")
    logger.info(f"Total trades: {len(combined_df):,}")
    logger.info(f"Price range: ${combined_df['price'].min():.2f} - ${combined_df['price'].max():.2f}")


def main():
    parser = argparse.ArgumentParser(description="Combine CSV files into a single Parquet file")

    parser.add_argument(
        "--input-dir",
        default="./datasets_deribit_perpetual",
        help="Input directory containing CSV files",
    )
    parser.add_argument(
        "--output-file",
        default="./deribit_btc_perpetual_trades_raw.parquet",
        help="Output Parquet file path",
    )
    parser.add_argument(
        "--pattern",
        default="*.csv.gz",
        help="File pattern to match (default: *.csv.gz)",
    )
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable debug logging")

    args = parser.parse_args()

    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(level=log_level, format="%(message)s", handlers=[logging.StreamHandler(sys.stdout)])

    logger.info("=" * 80)
    logger.info("COMBINE CSVs TO PARQUET")
    logger.info("=" * 80)

    try:
        combine_csvs_to_parquet(
            input_dir=args.input_dir,
            output_file=args.output_file,
            pattern=args.pattern,
        )
    except Exception as e:
        logger.error(f"ERROR: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
