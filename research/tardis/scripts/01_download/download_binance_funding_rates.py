#!/usr/bin/env python3
"""
Download historical funding rate data for Binance futures markets.

This script downloads derivative_ticker data containing:
- Funding rates (current and predicted)
- Mark price and index price
- Open interest
- Next funding timestamp

Supports both:
- binance-futures (USDT-margined perpetuals)
- binance-delivery (COIN-margined quarterly futures)

Usage:
    uv run python download_binance_funding_rates.py \\
        --from-date 2024-01-01 \\
        --to-date 2024-01-31 \\
        --symbols BTCUSDT,ETHUSDT \\
        --exchanges binance-futures,binance-delivery \\
        --workers 5

Output: data/raw/binance_funding_rates/{exchange}/{symbol}/{date}.parquet
"""

import argparse
import json
import logging
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timedelta

import polars as pl
from dotenv import load_dotenv
from tardis_dev import datasets

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def _empty_dataframe() -> pl.DataFrame:
    """Create empty DataFrame with funding rate schema."""
    schema = {
        "exchange": pl.Utf8,
        "symbol": pl.Utf8,
        "timestamp": pl.Int64,  # Microseconds since epoch
        "local_timestamp": pl.Int64,  # Collection timestamp
        "funding_timestamp": pl.Int64,  # Next funding event time
        "funding_rate": pl.Float64,
        "predicted_funding_rate": pl.Float64,
        "mark_price": pl.Float64,
        "index_price": pl.Float64,
        "open_interest": pl.Float64,
        "last_price": pl.Float64,
    }
    return pl.DataFrame(schema=schema)


def download_funding_rates_for_day(
    date_str: str,
    exchange: str,
    symbols: list[str],
    api_key: str,
    temp_dir: str,
) -> list[str]:
    """Download derivative_ticker data for single day.

    Args:
        date_str: Date in YYYY-MM-DD format
        exchange: Exchange ID (binance-futures or binance-delivery)
        symbols: List of symbols to download
        api_key: Tardis API key
        temp_dir: Temporary download directory

    Returns:
        List of paths to downloaded CSV.gz files
    """
    logger.debug(f"Downloading {exchange} funding rates for {date_str} ({', '.join(symbols)})...")

    try:
        datasets.download(
            exchange=exchange,
            data_types=["derivative_ticker"],
            from_date=date_str,
            to_date=date_str,
            symbols=symbols,
            api_key=api_key,
            download_dir=temp_dir,
            concurrency=1,
        )

        # Find downloaded files
        downloaded_files = []
        for symbol in symbols:
            filename = f"{exchange}_derivative_ticker_{date_str}_{symbol}.csv.gz"
            filepath = os.path.join(temp_dir, filename)
            if os.path.exists(filepath):
                downloaded_files.append(filepath)
            else:
                logger.warning(f"Expected file not found: {filepath}")

        return downloaded_files

    except Exception as e:
        logger.error(f"Failed to download {exchange} {date_str}: {e}")
        raise


def parse_derivative_ticker_csv(csv_file: str) -> pl.DataFrame:
    """Parse derivative_ticker CSV to standardized schema.

    Args:
        csv_file: Path to CSV.gz file

    Returns:
        DataFrame with funding rate data
    """
    logger.debug(f"Parsing: {csv_file}")

    # Schema overrides for consistent types
    schema_overrides = {
        "funding_rate": pl.Float64,
        "predicted_funding_rate": pl.Float64,
        "mark_price": pl.Float64,
        "index_price": pl.Float64,
        "open_interest": pl.Float64,
        "last_price": pl.Float64,
    }

    try:
        df = pl.read_csv(csv_file, schema_overrides=schema_overrides)

        # Ensure all required columns exist
        required_cols = [
            "exchange",
            "symbol",
            "timestamp",
            "local_timestamp",
        ]

        for col in required_cols:
            if col not in df.columns:
                raise ValueError(f"Missing required column: {col}")

        # Select and order columns according to schema
        output_cols = [
            "exchange",
            "symbol",
            "timestamp",
            "local_timestamp",
            "funding_timestamp",
            "funding_rate",
            "predicted_funding_rate",
            "mark_price",
            "index_price",
            "open_interest",
            "last_price",
        ]

        # Only select columns that exist in the CSV
        available_cols = [col for col in output_cols if col in df.columns]
        df = df.select(available_cols)

        # Fill missing optional columns with null
        for col in output_cols:
            if col not in df.columns:
                df = df.with_columns([pl.lit(None).cast(pl.Float64).alias(col)])

        return df

    except Exception as e:
        logger.error(f"Failed to parse {csv_file}: {e}")
        raise


def process_day(
    date_str: str,
    exchange: str,
    symbols: list[str],
    output_dir: str,
    temp_dir: str,
    api_key: str,
) -> dict[str, int]:
    """Download, parse, and save funding rates for single day.

    Args:
        date_str: Date in YYYY-MM-DD format
        exchange: Exchange ID
        symbols: List of symbols
        output_dir: Output directory
        temp_dir: Temporary download directory
        api_key: Tardis API key

    Returns:
        Dict mapping symbol to row count
    """
    result = {}

    for symbol in symbols:
        output_file = os.path.join(output_dir, exchange, symbol, f"{date_str}.parquet")
        os.makedirs(os.path.dirname(output_file), exist_ok=True)

        if os.path.exists(output_file):
            logger.info(f"✓ {exchange}/{symbol}/{date_str}: Already processed, skipping")
            try:
                df = pl.read_parquet(output_file)
                result[symbol] = len(df)
            except Exception:
                result[symbol] = 0
            continue

        try:
            # Download
            csv_files = download_funding_rates_for_day(date_str, exchange, [symbol], api_key, temp_dir)

            if not csv_files:
                logger.warning(f"No data downloaded for {exchange}/{symbol}/{date_str}")
                result[symbol] = 0
                continue

            # Parse
            dfs = []
            for csv_file in csv_files:
                df = parse_derivative_ticker_csv(csv_file)
                dfs.append(df)

            # Combine if multiple files
            if dfs:
                combined_df = pl.concat(dfs) if len(dfs) > 1 else dfs[0]

                # Sort by timestamp
                combined_df = combined_df.sort("timestamp")

                # Save
                combined_df.write_parquet(
                    output_file,
                    compression="snappy",
                    statistics=True,
                )

                logger.info(f"✓ {exchange}/{symbol}/{date_str}: {len(combined_df):,} records")
                result[symbol] = len(combined_df)
            else:
                result[symbol] = 0

        except Exception as e:
            logger.error(f"✗ {exchange}/{symbol}/{date_str}: {e}")
            result[symbol] = 0

    return result


def save_checkpoint(checkpoint_file: str, completed_dates: set[str], stats: dict) -> None:
    """Save progress checkpoint."""
    checkpoint_data = {
        "completed_dates": sorted(completed_dates),
        "total_downloads": stats.get("total_downloads", 0),
        "total_rows": stats.get("total_rows", 0),
        "last_updated": datetime.now().isoformat(),
    }
    with open(checkpoint_file, "w") as f:
        json.dump(checkpoint_data, f, indent=2)
    logger.debug(f"Checkpoint saved: {len(completed_dates)} dates completed")


def load_checkpoint(checkpoint_file: str) -> tuple[set[str], dict]:
    """Load progress checkpoint."""
    if not os.path.exists(checkpoint_file):
        return set(), {}

    try:
        with open(checkpoint_file) as f:
            data = json.load(f)
        completed_dates = set(data.get("completed_dates", []))
        stats = {
            "total_downloads": data.get("total_downloads", 0),
            "total_rows": data.get("total_rows", 0),
        }
        logger.info(f"Loaded checkpoint: {len(completed_dates)} dates already completed")
        return completed_dates, stats
    except Exception as e:
        logger.warning(f"Failed to load checkpoint: {e}")
        return set(), {}


def generate_date_range(from_date: str, to_date: str) -> list[str]:
    """Generate list of dates in YYYY-MM-DD format."""
    dates = []
    current = datetime.strptime(from_date, "%Y-%m-%d")
    end = datetime.strptime(to_date, "%Y-%m-%d")

    while current <= end:
        dates.append(current.strftime("%Y-%m-%d"))
        current += timedelta(days=1)

    return dates


def main() -> None:
    """Main entry point."""
    load_dotenv()
    tardis_api_key = os.getenv("TARDIS_API_KEY")

    parser = argparse.ArgumentParser(
        description="Download Binance futures funding rate data from Tardis",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Download 1 month of BTC USDT futures funding rates
  uv run python download_binance_funding_rates.py \\
      --from-date 2024-01-01 --to-date 2024-01-31 \\
      --symbols BTCUSDT --exchanges binance-futures

  # Download both USDT and COIN futures for BTC and ETH
  uv run python download_binance_funding_rates.py \\
      --from-date 2024-01-01 --to-date 2024-01-31 \\
      --symbols BTCUSDT,ETHUSDT,BTCUSD_PERP,ETHUSD_PERP \\
      --exchanges binance-futures,binance-delivery \\
      --workers 5
        """,
    )

    parser.add_argument(
        "--from-date",
        required=True,
        help="Start date (YYYY-MM-DD)",
    )
    parser.add_argument(
        "--to-date",
        required=True,
        help="End date (YYYY-MM-DD)",
    )
    parser.add_argument(
        "--symbols",
        default="BTCUSDT,ETHUSDT",
        help="Comma-separated symbols (default: BTCUSDT,ETHUSDT)",
    )
    parser.add_argument(
        "--exchanges",
        default="binance-futures",
        help="Comma-separated exchanges: binance-futures, binance-delivery (default: binance-futures)",
    )
    parser.add_argument(
        "--output-dir",
        default="./data/raw/binance_funding_rates",
        help="Output directory (default: ./data/raw/binance_funding_rates)",
    )
    parser.add_argument(
        "--temp-dir",
        default="./temp_funding_download",
        help="Temporary download directory (default: ./temp_funding_download)",
    )
    parser.add_argument(
        "--checkpoint-dir",
        default="./checkpoints",
        help="Checkpoint directory (default: ./checkpoints)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=20,  # Optimized for 32 vCPUs
        help="Number of parallel workers (default: 20, use 1 for sequential)",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from checkpoint (skip already completed dates)",
    )

    args = parser.parse_args()

    # Validate API key
    if not tardis_api_key:
        logger.error("ERROR: TARDIS_API_KEY not found in .env file")
        logger.error("Please create a .env file with: TARDIS_API_KEY=your_key_here")
        sys.exit(1)

    # Parse arguments
    symbols = [s.strip().upper() for s in args.symbols.split(",")]
    exchanges = [e.strip() for e in args.exchanges.split(",")]

    # Validate exchanges
    valid_exchanges = ["binance-futures", "binance-delivery"]
    for exchange in exchanges:
        if exchange not in valid_exchanges:
            logger.error(f"Invalid exchange: {exchange}")
            logger.error(f"Valid exchanges: {', '.join(valid_exchanges)}")
            sys.exit(1)

    # Generate date range
    date_range = generate_date_range(args.from_date, args.to_date)
    logger.info(f"Date range: {args.from_date} to {args.to_date} ({len(date_range)} days)")
    logger.info(f"Symbols: {', '.join(symbols)}")
    logger.info(f"Exchanges: {', '.join(exchanges)}")

    # Create directories
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.temp_dir, exist_ok=True)
    os.makedirs(args.checkpoint_dir, exist_ok=True)

    # Load checkpoint if resuming
    checkpoint_file = os.path.join(args.checkpoint_dir, "binance_funding_rates_progress.json")
    completed_dates = set()
    stats = {"total_downloads": 0, "total_rows": 0}

    if args.resume:
        completed_dates, loaded_stats = load_checkpoint(checkpoint_file)
        # Merge loaded stats with default stats (in case of missing keys)
        stats.update(loaded_stats)
        date_range = [d for d in date_range if d not in completed_dates]
        logger.info(f"Resuming: {len(date_range)} dates remaining")

    if not date_range:
        logger.info("All dates already completed!")
        return

    # Process each exchange
    total_tasks = len(date_range) * len(exchanges)
    completed_tasks = 0
    start_time = time.time()

    for exchange in exchanges:
        logger.info(f"\n{'=' * 60}")
        logger.info(f"Processing exchange: {exchange}")
        logger.info(f"{'=' * 60}\n")

        if args.workers == 1:
            # Sequential processing
            for date_str in date_range:
                result = process_day(
                    date_str,
                    exchange,
                    symbols,
                    args.output_dir,
                    args.temp_dir,
                    tardis_api_key,
                )

                # Update stats
                for _symbol, row_count in result.items():
                    if row_count > 0:
                        stats["total_downloads"] += 1
                        stats["total_rows"] += row_count

                # Mark date as completed (for this exchange)
                date_key = f"{exchange}:{date_str}"
                completed_dates.add(date_key)
                completed_tasks += 1

                # Save checkpoint every 10 dates
                if completed_tasks % 10 == 0:
                    save_checkpoint(checkpoint_file, completed_dates, stats)

                # Progress update
                progress = completed_tasks / total_tasks * 100
                elapsed = time.time() - start_time
                rate = completed_tasks / elapsed if elapsed > 0 else 0
                eta = (total_tasks - completed_tasks) / rate if rate > 0 else 0

                logger.info(
                    f"Progress: {completed_tasks}/{total_tasks} ({progress:.1f}%) | "
                    f"Rate: {rate:.2f} tasks/sec | ETA: {eta / 60:.1f} min"
                )

        else:
            # Parallel processing
            with ProcessPoolExecutor(max_workers=args.workers) as executor:
                futures = {}
                for date_str in date_range:
                    future = executor.submit(
                        process_day,
                        date_str,
                        exchange,
                        symbols,
                        args.output_dir,
                        args.temp_dir,
                        tardis_api_key,
                    )
                    futures[future] = (exchange, date_str)

                for future in as_completed(futures):
                    exchange_id, date_str = futures[future]
                    try:
                        result = future.result()

                        # Update stats
                        for _symbol, row_count in result.items():
                            if row_count > 0:
                                stats["total_downloads"] += 1
                                stats["total_rows"] += row_count

                        # Mark date as completed
                        date_key = f"{exchange_id}:{date_str}"
                        completed_dates.add(date_key)
                        completed_tasks += 1

                        # Save checkpoint every 10 dates
                        if completed_tasks % 10 == 0:
                            save_checkpoint(checkpoint_file, completed_dates, stats)

                        # Progress update
                        progress = completed_tasks / total_tasks * 100
                        elapsed = time.time() - start_time
                        rate = completed_tasks / elapsed if elapsed > 0 else 0
                        eta = (total_tasks - completed_tasks) / rate if rate > 0 else 0

                        logger.info(
                            f"Progress: {completed_tasks}/{total_tasks} ({progress:.1f}%) | "
                            f"Rate: {rate:.2f} tasks/sec | ETA: {eta / 60:.1f} min"
                        )

                    except Exception as e:
                        logger.error(f"Task failed for {exchange_id}/{date_str}: {e}")

    # Final checkpoint
    save_checkpoint(checkpoint_file, completed_dates, stats)

    # Summary
    elapsed_total = time.time() - start_time
    logger.info("\n" + "=" * 60)
    logger.info("DOWNLOAD COMPLETE")
    logger.info("=" * 60)
    logger.info(f"Total downloads: {stats['total_downloads']}")
    logger.info(f"Total rows: {stats['total_rows']:,}")
    logger.info(f"Total time: {elapsed_total / 60:.1f} minutes")
    logger.info(f"Output directory: {args.output_dir}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
