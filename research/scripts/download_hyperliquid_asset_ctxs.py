#!/usr/bin/env python3
"""
Download and process Hyperliquid asset context data from S3.

Asset context data includes:
- Funding rates, open interest, oracle/mark/mid prices
- Impact bid/ask prices, premium, day notional volume
- 1-minute granularity for all perpetual assets

Data source: s3://hyperliquid-archive/asset_ctxs/YYYYMMDD.csv.lz4
Note: Requester pays for S3 transfer costs.
"""

import argparse
import logging
import subprocess
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, Union

import polars as pl

# Constants
S3_BUCKET = "hyperliquid-archive"
S3_PREFIX = "asset_ctxs"
DEFAULT_OUTPUT_DIR = "research/data/hyperliquid/asset_ctxs"
EXPECTED_ROWS_PER_DAY = 1440  # 24 hours * 60 minutes

# Module logger
logger = logging.getLogger(__name__)


def _setup_logging(verbose: bool = False) -> None:
    """Configure logging with appropriate level and format."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def _parse_date(date_str: str) -> datetime:
    """Parse date string in YYYY-MM-DD format.

    Args:
        date_str: Date in YYYY-MM-DD format

    Returns:
        datetime object

    Raises:
        ValueError: If date format is invalid
    """
    try:
        return datetime.strptime(date_str, "%Y-%m-%d")
    except ValueError as e:
        raise ValueError(f"Invalid date format: {date_str}. Expected YYYY-MM-DD") from e


def _generate_date_range(start_date: str, end_date: Optional[str] = None) -> list[datetime]:
    """Generate list of dates between start and end (inclusive).

    Args:
        start_date: Start date in YYYY-MM-DD format
        end_date: End date in YYYY-MM-DD format (optional, defaults to start_date)

    Returns:
        List of datetime objects
    """
    start = _parse_date(start_date)
    end = _parse_date(end_date) if end_date else start

    if end < start:
        raise ValueError(f"End date {end_date} is before start date {start_date}")

    dates = []
    current = start
    while current <= end:
        dates.append(current)
        current += timedelta(days=1)

    return dates


def _download_from_s3(date: datetime, temp_dir: Path) -> Path:
    """Download LZ4 compressed file from S3.

    Args:
        date: Date to download
        temp_dir: Temporary directory for downloads

    Returns:
        Path to downloaded LZ4 file

    Raises:
        RuntimeError: If download fails
    """
    date_str = date.strftime("%Y%m%d")
    s3_path = f"s3://{S3_BUCKET}/{S3_PREFIX}/{date_str}.csv.lz4"
    output_path = temp_dir / f"{date_str}.csv.lz4"

    logger.info(f"Downloading {s3_path}...")

    cmd = [
        "aws",
        "s3",
        "cp",
        s3_path,
        str(output_path),
        "--request-payer",
        "requester",
    ]

    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True, timeout=300)
        logger.debug(f"Download output: {result.stdout}")

        if not output_path.exists():
            raise RuntimeError(f"Download completed but file not found: {output_path}")

        file_size_mb = output_path.stat().st_size / 1024 / 1024
        logger.info(f"Downloaded {file_size_mb:.2f} MB to {output_path}")

        return output_path

    except subprocess.TimeoutExpired as e:
        raise RuntimeError(f"Download timeout after 300s: {s3_path}") from e
    except subprocess.CalledProcessError as e:
        error_msg = e.stderr.strip() if e.stderr else str(e)
        if "404" in error_msg or "does not exist" in error_msg:
            raise RuntimeError(f"File not found in S3: {s3_path}") from e
        raise RuntimeError(f"Download failed: {error_msg}") from e


def _decompress_lz4(lz4_path: Path) -> Path:
    """Decompress LZ4 file to CSV.

    Args:
        lz4_path: Path to LZ4 compressed file

    Returns:
        Path to decompressed CSV file

    Raises:
        RuntimeError: If decompression fails
    """
    csv_path = lz4_path.with_suffix("")  # Remove .lz4 extension

    logger.info(f"Decompressing {lz4_path.name}...")

    # Try unlz4 first (preserves original), then gunzip
    for cmd_name, cmd_args in [
        ("unlz4", ["unlz4", str(lz4_path), str(csv_path)]),
        ("gunzip", ["gunzip", "-k", str(lz4_path)]),  # -k keeps original
    ]:
        try:
            result = subprocess.run(cmd_args, check=True, capture_output=True, text=True, timeout=60)
            logger.debug(f"Decompression output ({cmd_name}): {result.stdout}")

            if csv_path.exists():
                file_size_mb = csv_path.stat().st_size / 1024 / 1024
                logger.info(f"Decompressed to {file_size_mb:.2f} MB")
                return csv_path

        except (subprocess.CalledProcessError, FileNotFoundError):
            logger.debug(f"{cmd_name} failed, trying alternative method")
            continue

    raise RuntimeError(
        "Decompression failed. Install unlz4 or gunzip: brew install lz4 (macOS) or apt-get install liblz4-tool (Linux)"
    )


def _convert_to_parquet(
    csv_path: Path,
    output_path: Path,
    coins: Optional[list[str]] = None,
) -> dict[str, Union[int, float]]:
    """Convert CSV to Parquet with filtering and validation.

    Args:
        csv_path: Path to CSV file
        output_path: Path to output Parquet file
        coins: List of coins to filter (None = all coins)

    Returns:
        Dictionary with stats: rows, coins, date_range, etc.

    Raises:
        ValueError: If validation fails
    """
    logger.info(f"Reading CSV from {csv_path.name}...")

    # Read CSV with Polars
    df = pl.read_csv(csv_path)

    logger.info(f"Loaded {df.shape[0]:,} rows, {df.shape[1]} columns")

    # Parse datetime with proper timezone
    df = df.with_columns([pl.col("time").str.strptime(pl.Datetime("us", "UTC"), "%Y-%m-%dT%H:%M:%SZ").alias("time")])

    # Filter by coins if specified
    if coins:
        coins_upper = [c.upper() for c in coins]
        df = df.filter(pl.col("coin").is_in(coins_upper))
        logger.info(f"Filtered to {df.shape[0]:,} rows for coins: {', '.join(coins_upper)}")

        if df.shape[0] == 0:
            raise ValueError(f"No data found for coins: {coins_upper}")

    # Validation
    unique_coins = df["coin"].n_unique()
    min_time = df["time"].min()
    max_time = df["time"].max()

    logger.info(f"Date range: {min_time} to {max_time} | Unique coins: {unique_coins}")

    # Check for expected row count per coin (1440 = 24h * 60min)
    if coins and len(coins) == 1:
        expected_rows = EXPECTED_ROWS_PER_DAY
        actual_rows = df.shape[0]
        if actual_rows != expected_rows:
            logger.warning(
                f"Expected {expected_rows} rows for single coin/day, got {actual_rows}. Data may be incomplete."
            )

    # Write Parquet with Snappy compression
    logger.info(f"Writing Parquet to {output_path}...")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.write_parquet(output_path, compression="snappy")

    file_size_kb = output_path.stat().st_size / 1024
    logger.info(f"Wrote {file_size_kb:.2f} KB to {output_path}")

    # Calculate summary stats
    stats = {
        "rows": df.shape[0],
        "coins": unique_coins,
        "date_range": f"{min_time} to {max_time}",
        "file_size_kb": round(file_size_kb, 2),
    }

    # Add coin-specific stats if single coin
    if coins and len(coins) == 1:
        coin_data = df.filter(pl.col("coin") == coins[0].upper())
        if coin_data.shape[0] > 0:
            stats.update(
                {
                    "mark_px_range": f"{coin_data['mark_px'].min():.2f} - {coin_data['mark_px'].max():.2f}",
                    "avg_funding": f"{coin_data['funding'].mean():.8f}",
                    "avg_open_interest": f"{coin_data['open_interest'].mean():.2f}",
                }
            )

    return stats


def _process_date(
    date: datetime,
    output_dir: Path,
    coins: Optional[list[str]],
    skip_existing: bool,
) -> Optional[dict[str, Union[int, float]]]:
    """Process a single date: download, decompress, convert.

    Args:
        date: Date to process
        output_dir: Output directory for Parquet files
        coins: List of coins to filter
        skip_existing: Skip if output file exists

    Returns:
        Stats dictionary or None if skipped
    """
    date_str = date.strftime("%Y%m%d")
    coins_suffix = "_" + "_".join(coins).upper() if coins else "_ALL"
    output_filename = f"{date_str}{coins_suffix}.parquet"

    # Organize by year/month
    year_month_dir = output_dir / date.strftime("%Y") / date.strftime("%m")
    output_path = year_month_dir / output_filename

    # Skip if exists
    if skip_existing and output_path.exists():
        logger.info(f"Skipping {date_str} (already exists): {output_path}")
        return None

    # Create temp directory for downloads
    temp_dir = Path("/tmp/hyperliquid_downloads")
    temp_dir.mkdir(parents=True, exist_ok=True)

    try:
        # Download from S3
        lz4_path = _download_from_s3(date, temp_dir)

        # Decompress
        csv_path = _decompress_lz4(lz4_path)

        # Convert to Parquet
        stats = _convert_to_parquet(csv_path, output_path, coins)

        logger.info(f"✓ Successfully processed {date_str}")
        return stats

    except Exception as e:
        logger.error(f"✗ Failed to process {date_str}: {e}")
        raise

    finally:
        # Cleanup temp files
        for temp_file in [lz4_path, csv_path]:
            if temp_file.exists():
                temp_file.unlink()
                logger.debug(f"Cleaned up: {temp_file}")


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Download Hyperliquid asset context data from S3",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Download BTC data for Sept 1, 2025
  uv run python download_hyperliquid_asset_ctxs.py --date 2025-09-01 --coins BTC

  # Download multiple coins for date range
  uv run python download_hyperliquid_asset_ctxs.py --date 2025-09-01 --end-date 2025-09-07 --coins BTC ETH SOL

  # Download all coins for single day
  uv run python download_hyperliquid_asset_ctxs.py --date 2025-09-01

Note: AWS CLI must be configured (uses --request-payer requester)
        """,
    )

    parser.add_argument(
        "--date",
        required=True,
        help="Start date in YYYY-MM-DD format",
    )
    parser.add_argument(
        "--end-date",
        help="End date in YYYY-MM-DD format (optional, defaults to --date)",
    )
    parser.add_argument(
        "--coins",
        nargs="+",
        help="Coins to filter (e.g., BTC ETH). If not specified, downloads all coins.",
    )
    parser.add_argument(
        "--output-dir",
        default=DEFAULT_OUTPUT_DIR,
        help=f"Output directory (default: {DEFAULT_OUTPUT_DIR})",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip dates that already have output files",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable debug logging",
    )

    args = parser.parse_args()

    # Setup logging
    _setup_logging(args.verbose)

    # Validate inputs
    try:
        dates = _generate_date_range(args.date, args.end_date)
    except ValueError as e:
        logger.error(f"Invalid date range: {e}")
        sys.exit(1)

    output_dir = Path(args.output_dir)

    logger.info("=" * 60)
    logger.info("Hyperliquid Asset Context Data Download")
    logger.info("=" * 60)
    logger.info(f"Date range: {dates[0].strftime('%Y-%m-%d')} to {dates[-1].strftime('%Y-%m-%d')} ({len(dates)} days)")
    logger.info(f"Coins: {', '.join(args.coins).upper() if args.coins else 'ALL'}")
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"Skip existing: {args.skip_existing}")
    logger.info("=" * 60)

    # Process each date
    success_count = 0
    skip_count = 0
    fail_count = 0

    for i, date in enumerate(dates, 1):
        logger.info(f"\n[{i}/{len(dates)}] Processing {date.strftime('%Y-%m-%d')}...")

        try:
            stats = _process_date(date, output_dir, args.coins, args.skip_existing)

            if stats is None:
                skip_count += 1
            else:
                success_count += 1
                logger.info(f"Stats: {stats}")

        except Exception as e:
            fail_count += 1
            logger.error(f"Failed: {e}")
            if not args.skip_existing:
                # Exit on first failure if not skipping
                sys.exit(1)

    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("Summary")
    logger.info("=" * 60)
    logger.info(f"Total dates: {len(dates)}")
    logger.info(f"✓ Successful: {success_count}")
    logger.info(f"⊘ Skipped: {skip_count}")
    logger.info(f"✗ Failed: {fail_count}")
    logger.info("=" * 60)

    if fail_count > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
