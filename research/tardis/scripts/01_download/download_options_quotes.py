#!/usr/bin/env python3

import argparse
import gzip
import logging
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timedelta
from pathlib import Path

import polars as pl
from dotenv import load_dotenv
from tardis_dev import datasets

logger = logging.getLogger(__name__)


def load_perp_reference(perp_file: str) -> pl.DataFrame:
    """Load perpetual 1s data as ATM price reference."""
    logger.info(f"Loading perpetual reference data from {perp_file}...")
    df = pl.read_parquet(perp_file)

    # Convert timestamp to seconds for easier joining
    df = df.with_columns([(pl.col("timestamp") / 1_000_000).cast(pl.Int64).alias("timestamp_seconds")])

    # Keep only timestamp and price for memory efficiency
    df = df.select(["timestamp_seconds", "price"])

    # Sort by timestamp_seconds for asof_join
    df = df.sort("timestamp_seconds")

    logger.info(f"Loaded {len(df):,} perpetual price records")
    return df


def download_quotes_for_day(
    date_str: str,
    api_key: str,
    temp_dir: str,
    assets: list[str],
) -> str:
    """Download quote data for a single day using tardis-dev.

    Returns path to downloaded CSV.gz file.
    """
    logger.debug(f"Downloading quotes for {date_str} ({', '.join(assets)})...")

    # Use "OPTIONS" symbol to download all options (will filter by asset in parsing step)
    symbols = ["OPTIONS"]

    # Use tardis-dev datasets API
    datasets.download(
        exchange="deribit",
        data_types=["quotes"],
        from_date=date_str,
        to_date=date_str,
        symbols=symbols,
        api_key=api_key,
        download_dir=temp_dir,
        concurrency=1,
    )

    # Find the downloaded file - single file for all OPTIONS
    filename = f"deribit_quotes_{date_str}_OPTIONS.csv.gz"
    filepath = os.path.join(temp_dir, filename)

    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Expected file not found: {filepath}")

    return filepath


def parse_quotes_csv(
    quotes_file: str,
    asset_filter: list[str],
) -> pl.DataFrame:
    """Parse quote CSV and transform to match tardis_download.py schema.

    Args:
        quotes_file: Path to CSV.gz file
        asset_filter: List of assets to include (e.g., ["BTC", "ETH"])

    Returns:
        DataFrame with schema matching tardis_download.py:32-47
    """
    logger.debug(f"Parsing quotes from: {quotes_file}")

    # Schema overrides to handle Tardis data type inconsistencies
    # Sometimes amounts/prices are whole numbers (inferred as int), sometimes floats
    schema_overrides = {
        "strike_price": pl.Float64,
        "bid_price": pl.Float64,
        "bid_amount": pl.Float64,
        "ask_price": pl.Float64,
        "ask_amount": pl.Float64,
    }

    try:
        df = pl.read_csv(quotes_file, schema_overrides=schema_overrides)
        original_count = len(df)
        logger.debug(f"Read {original_count:,} quote records")
    except Exception as e:
        logger.error(f"Failed to read {quotes_file}: {e}")
        return _empty_dataframe()

    # Filter to specified assets (symbol starts with "ASSET-")
    if asset_filter:
        asset_pattern = "|".join([f"{asset}-" for asset in asset_filter])
        df = df.filter(pl.col("symbol").str.contains(f"^({asset_pattern})"))
        logger.debug(f"After asset filter ({', '.join(asset_filter)}): {len(df):,} records")

    if len(df) == 0:
        return _empty_dataframe()

    # Parse symbol format: BTC-25JAN25-80000-C
    # symbol_parts[0] = underlying (BTC/ETH)
    # symbol_parts[1] = expiry_str (25JAN25)
    # symbol_parts[2] = strike_price (80000)
    # symbol_parts[3] = type (C/P)
    df = df.filter(pl.col("symbol").str.count_matches("-") == 3)

    if len(df) == 0:
        logger.warning("No valid option symbols after filtering")
        return _empty_dataframe()

    df = df.with_columns([pl.col("symbol").str.split("-").alias("symbol_parts")])

    df = df.with_columns(
        [
            pl.col("symbol_parts").list.get(0).alias("underlying"),
            pl.col("symbol_parts").list.get(1).alias("expiry_str"),
            pl.col("symbol_parts").list.get(2).cast(pl.Float64, strict=False).alias("strike_price"),
            pl.when(pl.col("symbol_parts").list.get(3) == "C")
            .then(pl.lit("call"))
            .otherwise(pl.lit("put"))
            .alias("type"),
        ]
    )

    df = df.drop("symbol_parts")

    # Filter out invalid strikes
    df = df.filter(pl.col("strike_price").is_not_null())

    if len(df) == 0:
        logger.warning("No valid rows after symbol parsing")
        return _empty_dataframe()

    # Note: timestamp and local_timestamp are already Int64 (microseconds) from Tardis
    # Note: exchange, bid_price, bid_amount, ask_price, ask_amount already have correct names

    # Data quality filters
    df = df.filter(
        (pl.col("bid_price").is_null() | (pl.col("bid_price") >= 0))
        & (pl.col("ask_price").is_null() | (pl.col("ask_price") >= 0))
        & (pl.col("bid_amount").is_null() | (pl.col("bid_amount") >= 0))
        & (pl.col("ask_amount").is_null() | (pl.col("ask_amount") >= 0))
        & (pl.col("bid_price").is_null() | pl.col("ask_price").is_null() | (pl.col("bid_price") <= pl.col("ask_price")))
    )

    # Select final columns to match tardis_download.py schema
    df = df.select(
        [
            "exchange",
            "symbol",
            "timestamp",
            "local_timestamp",
            "type",
            "strike_price",
            "underlying",
            "expiry_str",
            "bid_price",
            "bid_amount",
            "ask_price",
            "ask_amount",
        ]
    )

    filtered_count = original_count - len(df)
    logger.debug(f"Parsed {len(df):,} valid quote records ({filtered_count:,} filtered)")

    return df


def _empty_dataframe() -> pl.DataFrame:
    """Return empty DataFrame with correct schema."""
    schema = {
        "exchange": pl.Utf8,
        "symbol": pl.Utf8,
        "timestamp": pl.Int64,
        "local_timestamp": pl.Int64,
        "type": pl.Utf8,
        "strike_price": pl.Float64,
        "underlying": pl.Utf8,
        "expiry_str": pl.Utf8,
        "bid_price": pl.Float64,
        "bid_amount": pl.Float64,
        "ask_price": pl.Float64,
        "ask_amount": pl.Float64,
    }
    return pl.DataFrame(schema=schema)


def filter_quotes_atm(
    quotes_df: pl.DataFrame,
    perp_ref: pl.DataFrame,
    atm_tolerance: float,
) -> tuple[pl.DataFrame, int]:
    """Filter quotes to ATM ±tolerance using perp reference.

    Returns:
        tuple[pl.DataFrame, int]: (filtered_df, original_count)
    """
    original_count = len(quotes_df)

    if original_count == 0:
        return quotes_df, 0

    logger.debug(f"Filtering {original_count:,} quotes to ATM ±{atm_tolerance * 100:.0f}%")

    # Convert timestamp to seconds for joining
    quotes_df = quotes_df.with_columns([(pl.col("timestamp") / 1_000_000).cast(pl.Int64).alias("timestamp_seconds")])

    # Sort by timestamp_seconds for asof_join
    quotes_df = quotes_df.sort("timestamp_seconds")

    # Join with perp reference to get ATM price
    quotes_df = quotes_df.join_asof(perp_ref, on="timestamp_seconds", strategy="nearest")

    # Rename price column to atm_price for clarity
    quotes_df = quotes_df.rename({"price": "atm_price"})

    # Calculate distance from ATM
    quotes_df = quotes_df.with_columns(
        [((pl.col("strike_price") - pl.col("atm_price")).abs() / pl.col("atm_price")).alias("atm_distance_pct")]
    )

    # Filter to ATM ±tolerance
    before_atm_filter = len(quotes_df)
    quotes_df = quotes_df.filter(pl.col("atm_distance_pct") <= atm_tolerance)
    filtered_count = len(quotes_df)

    reduction_pct = (1 - filtered_count / before_atm_filter) * 100 if before_atm_filter > 0 else 0
    logger.debug(f"ATM filter: {before_atm_filter:,} → {filtered_count:,} records ({reduction_pct:.1f}% reduction)")

    # Drop helper columns
    quotes_df = quotes_df.drop(["timestamp_seconds", "atm_price", "atm_distance_pct"])

    return quotes_df, original_count


def process_day(
    date_str: str,
    perp_ref: pl.DataFrame,
    output_dir: str,
    temp_dir: str,
    api_key: str,
    atm_tolerance: float,
    assets: list[str],
) -> tuple[int, int]:
    """Process a single day: download, parse, filter, save.

    Returns:
        tuple[int, int]: (original_count, filtered_count)
    """
    output_file = os.path.join(output_dir, f"deribit_options_quotes_atm3pct_{date_str}.csv.gz")

    # Skip if already processed
    if os.path.exists(output_file):
        logger.info(f"✓ {date_str}: Already processed, skipping")
        return (0, 0)

    try:
        # Download quotes for this day
        quotes_file = download_quotes_for_day(date_str, api_key, temp_dir, assets)

        # Parse CSV to DataFrame
        quotes_df = parse_quotes_csv(quotes_file, assets)

        if len(quotes_df) == 0:
            logger.warning(f"⚠ {date_str}: No valid quotes parsed")
            # Create empty file to mark as processed
            Path(output_file).touch()
            return (0, 0)

        # Filter to ATM ±tolerance
        filtered_df, original_count = filter_quotes_atm(quotes_df, perp_ref, atm_tolerance)
        filtered_count = len(filtered_df)

        if filtered_count == 0:
            logger.warning(f"⚠ {date_str}: No quotes within ATM ±{atm_tolerance * 100:.0f}%")
            # Create empty file to mark as processed
            Path(output_file).touch()
        else:
            # Save filtered data as CSV
            filtered_df.write_csv(output_file, separator=",")

            # Compress with gzip
            with open(output_file, "rb") as f_in, gzip.open(output_file + ".tmp", "wb") as f_out:
                f_out.writelines(f_in)
            os.replace(output_file + ".tmp", output_file)

            logger.info(f"✓ {date_str}: {original_count:,} → {filtered_count:,} records")

        # Clean up temp file
        os.remove(quotes_file)

        return (original_count, filtered_count)

    except Exception as e:
        logger.error(f"✗ {date_str}: Failed - {e}")
        return (0, 0)


def _process_day_worker(args: tuple[str, str, str, str, str, float, list[str]]) -> tuple[int, int]:
    """Worker function for multiprocessing that loads perp_ref from disk.

    Each worker process loads the perp reference once to avoid pickling large DataFrames.
    """
    date_str, perp_file, output_dir, base_temp_dir, api_key, atm_tolerance, assets = args

    # Create worker-specific temp directory to avoid file collisions
    worker_temp_dir = os.path.join(base_temp_dir, f"worker_{os.getpid()}")
    os.makedirs(worker_temp_dir, exist_ok=True)

    # Load perp reference in worker process
    perp_ref = load_perp_reference(perp_file)

    # Process the day
    result = process_day(
        date_str=date_str,
        perp_ref=perp_ref,
        output_dir=output_dir,
        temp_dir=worker_temp_dir,
        api_key=api_key,
        atm_tolerance=atm_tolerance,
        assets=assets,
    )

    return result


def main() -> None:
    load_dotenv()
    tardis_api_key = os.getenv("TARDIS_API_KEY")

    parser = argparse.ArgumentParser(description="Download and filter Deribit options quote data to ATM ±tolerance")

    parser.add_argument("--from-date", required=True, help="Start date (YYYY-MM-DD)")
    parser.add_argument("--to-date", required=True, help="End date (YYYY-MM-DD)")
    parser.add_argument(
        "--perp-reference",
        required=True,
        help="Perpetual 1s reference Parquet file",
    )
    parser.add_argument(
        "--output-dir",
        default="./datasets_deribit_options_quotes_atm3pct",
        help="Output directory",
    )
    parser.add_argument(
        "--temp-dir",
        default="./temp_quotes_download",
        help="Temporary download directory",
    )
    parser.add_argument(
        "--atm-tolerance",
        type=float,
        default=0.03,
        help="ATM tolerance (default: 0.03 for ±3%%)",
    )
    parser.add_argument(
        "--assets",
        default="BTC",
        help="Comma-separated assets (BTC, ETH) - default: BTC",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=5,
        help="Number of parallel workers (default: 5, use 1 for sequential)",
    )
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable debug logging")

    args = parser.parse_args()

    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(level=log_level, format="%(message)s", handlers=[logging.StreamHandler(sys.stdout)])

    if not tardis_api_key:
        logger.error("ERROR: TARDIS_API_KEY not found in .env")
        sys.exit(1)

    # Validate workers
    if args.workers < 1:
        logger.error("ERROR: --workers must be >= 1")
        sys.exit(1)
    if args.workers > 10:
        logger.error("ERROR: --workers must be <= 10 to avoid overwhelming Tardis API")
        sys.exit(1)

    # Parse and validate assets
    assets = [a.strip().upper() for a in args.assets.split(",")]
    if invalid := set(assets) - {"BTC", "ETH"}:
        logger.error(f"ERROR: Invalid assets: {invalid}")
        sys.exit(1)

    # Create directories
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.temp_dir, exist_ok=True)

    logger.info("=" * 80)
    logger.info(f"DOWNLOAD & FILTER OPTIONS QUOTES (ATM ±{args.atm_tolerance * 100:.0f}%)")
    logger.info("=" * 80)
    logger.info(f"Date range: {args.from_date} to {args.to_date}")
    logger.info(f"Assets: {', '.join(assets)}")
    logger.info(f"Perpetual reference: {args.perp_reference}")
    logger.info(f"Output directory: {args.output_dir}")
    logger.info("=" * 80)

    # Generate date range
    start_time = time.time()
    from_date = datetime.strptime(args.from_date, "%Y-%m-%d")
    to_date = datetime.strptime(args.to_date, "%Y-%m-%d")
    date_range = []
    current_date = from_date
    while current_date <= to_date:
        date_range.append(current_date.strftime("%Y-%m-%d"))
        current_date += timedelta(days=1)

    logger.info(f"Processing {len(date_range)} days with {args.workers} worker(s)...")
    logger.info("=" * 80)

    # Process each day
    total_original = 0
    total_filtered = 0

    if args.workers == 1:
        # Sequential mode
        perp_ref = load_perp_reference(args.perp_reference)

        for i, date_str in enumerate(date_range, 1):
            logger.info(f"[{i}/{len(date_range)}] Processing {date_str}...")

            original, filtered = process_day(
                date_str=date_str,
                perp_ref=perp_ref,
                output_dir=args.output_dir,
                temp_dir=args.temp_dir,
                api_key=tardis_api_key,
                atm_tolerance=args.atm_tolerance,
                assets=assets,
            )

            total_original += original
            total_filtered += filtered
    else:
        # Parallel mode with multiprocessing
        worker_args = [
            (
                date_str,
                args.perp_reference,
                args.output_dir,
                args.temp_dir,
                tardis_api_key,
                args.atm_tolerance,
                assets,
            )
            for date_str in date_range
        ]

        completed_count = 0
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            # Submit all tasks
            future_to_date = {executor.submit(_process_day_worker, args): args[0] for args in worker_args}

            # Process results as they complete
            for future in as_completed(future_to_date):
                date_str = future_to_date[future]
                completed_count += 1

                try:
                    original, filtered = future.result()
                    total_original += original
                    total_filtered += filtered

                    logger.info(f"[{completed_count}/{len(date_range)}] Completed {date_str}")
                except Exception as e:
                    logger.error(f"✗ {date_str}: Worker failed - {e}")

    # Summary
    elapsed = time.time() - start_time
    reduction_pct = (1 - total_filtered / total_original) * 100 if total_original > 0 else 0

    logger.info("=" * 80)
    logger.info("COMPLETE!")
    logger.info(f"Total original records: {total_original:,}")
    logger.info(f"Total filtered records: {total_filtered:,}")
    logger.info(f"Reduction: {reduction_pct:.1f}%")
    logger.info(f"Output directory: {args.output_dir}")
    logger.info(f"Total time: {elapsed:.1f}s ({elapsed / len(date_range):.1f}s per day)")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
