#!/usr/bin/env python3
"""
Resample perpetual futures data to continuous 1-second intervals with forward-fill.

This script takes the existing 1s resampled perpetual data (which only contains
seconds where trades occurred) and creates a complete 1-second timeline with
forward-filled prices for gaps.

Input: btc_perpetual_1s_resampled.parquet (14.5M rows, 21.76% coverage)
Output: btc_perpetual_1s_resampled_ffill.parquet (66.9M rows, 100% coverage)

Date range: 2023-09-26 to 2025-11-05 (774 days)
"""

import argparse
import logging
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import polars as pl

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# Constants
DEFAULT_INPUT = "research/tardis/data/consolidated/btc_perpetual_1s_resampled.parquet"
DEFAULT_OUTPUT = "research/tardis/data/consolidated/btc_perpetual_1s_resampled_ffill.parquet"
START_DATE = "2023-09-26"
END_DATE = "2025-11-05"


def create_continuous_timeline(start_date: str, end_date: str) -> pl.DataFrame:
    """
    Create a continuous 1-second timeline between start and end dates.

    Args:
        start_date: Start date (YYYY-MM-DD)
        end_date: End date (YYYY-MM-DD)

    Returns:
        DataFrame with continuous timestamp_seconds column
    """
    logger.info(f"Creating continuous timeline from {start_date} to {end_date}...")

    # Parse dates
    start_dt = datetime.strptime(start_date, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    end_dt = datetime.strptime(end_date, "%Y-%m-%d").replace(hour=23, minute=59, second=59, tzinfo=timezone.utc)

    # Convert to Unix timestamps
    start_ts = int(start_dt.timestamp())
    end_ts = int(end_dt.timestamp())

    # Calculate total seconds
    total_seconds = end_ts - start_ts + 1
    logger.info(f"  Total seconds: {total_seconds:,}")

    # Create continuous timestamp range
    timeline = pl.DataFrame(
        {
            "timestamp_seconds": pl.int_range(start_ts, end_ts + 1, eager=True),
        }
    )

    # Add timestamp in microseconds (for compatibility)
    timeline = timeline.with_columns(
        [
            (pl.col("timestamp_seconds") * 1_000_000).alias("timestamp"),
        ]
    )

    logger.info(f"  Timeline created: {len(timeline):,} rows")
    return timeline


def forward_fill_perpetual(
    input_file: str,
    output_file: str,
    start_date: str,
    end_date: str,
) -> None:
    """
    Forward-fill perpetual data to create continuous 1-second intervals.

    Args:
        input_file: Path to input parquet file (sparse data)
        output_file: Path to output parquet file (continuous data)
        start_date: Start date (YYYY-MM-DD)
        end_date: End date (YYYY-MM-DD)
    """
    start_time = time.time()

    logger.info("=" * 80)
    logger.info("PERPETUAL FORWARD-FILL RESAMPLING")
    logger.info("=" * 80)
    logger.info(f"Input:  {input_file}")
    logger.info(f"Output: {output_file}")
    logger.info(f"Date range: {start_date} to {end_date}")

    # Check input file exists
    if not Path(input_file).exists():
        raise FileNotFoundError(f"Input file not found: {input_file}")

    # Load sparse perpetual data
    logger.info("\nLoading sparse perpetual data...")
    perpetual = pl.scan_parquet(input_file)

    # Get stats on input
    input_stats = perpetual.select(
        [
            pl.len().alias("rows"),
            pl.col("timestamp_seconds").min().alias("min_ts"),
            pl.col("timestamp_seconds").max().alias("max_ts"),
        ]
    ).collect()

    logger.info(f"  Input rows: {input_stats['rows'][0]:,}")
    logger.info(
        f"  Input date range: {datetime.fromtimestamp(input_stats['min_ts'][0]).date()} "
        f"to {datetime.fromtimestamp(input_stats['max_ts'][0]).date()}"
    )

    # Create continuous timeline
    timeline = create_continuous_timeline(start_date, end_date)

    # Join sparse data with continuous timeline
    logger.info("\nJoining with continuous timeline...")
    logger.info("  This will create the full 66.9M row structure...")

    # Use lazy for memory efficiency
    perpetual_lazy = pl.LazyFrame(timeline).join(
        perpetual,
        on="timestamp_seconds",
        how="left",  # Keep all timeline rows
    )

    # Forward-fill price columns
    logger.info("\nApplying forward-fill to price columns...")
    logger.info("  Columns: vwap, price, price_mean, price_min, price_max")

    perpetual_filled = perpetual_lazy.with_columns(
        [
            # Forward-fill price columns
            pl.col("vwap").forward_fill().alias("vwap"),
            pl.col("price").forward_fill().alias("price"),
            pl.col("price_mean").forward_fill().alias("price_mean"),
            pl.col("price_min").forward_fill().alias("price_min"),
            pl.col("price_max").forward_fill().alias("price_max"),
            pl.col("total_amount").forward_fill().alias("total_amount"),
            # Fill trade_count with 0 for forward-filled rows
            pl.col("trade_count").fill_null(0).alias("trade_count"),
        ]
    )

    # Write output using streaming
    logger.info(f"\nWriting forward-filled data to {output_file}...")
    logger.info("  Using streaming write for memory efficiency...")

    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    perpetual_filled.sink_parquet(
        output_file,
        compression="snappy",
        statistics=True,
    )

    # Validate output
    logger.info("\nValidating output...")
    output_df = pl.scan_parquet(output_file)

    output_stats = output_df.select(
        [
            pl.len().alias("rows"),
            pl.col("timestamp_seconds").min().alias("min_ts"),
            pl.col("timestamp_seconds").max().alias("max_ts"),
            pl.col("vwap").null_count().alias("vwap_nulls"),
            pl.col("price").null_count().alias("price_nulls"),
            (pl.col("trade_count") == 0).sum().alias("forward_filled_rows"),
        ]
    ).collect()

    logger.info(f"  Output rows: {output_stats['rows'][0]:,}")
    logger.info(
        f"  Output date range: {datetime.fromtimestamp(output_stats['min_ts'][0]).date()} "
        f"to {datetime.fromtimestamp(output_stats['max_ts'][0]).date()}"
    )
    logger.info(f"  VWAP nulls: {output_stats['vwap_nulls'][0]:,}")
    logger.info(f"  Price nulls: {output_stats['price_nulls'][0]:,}")
    logger.info(f"  Forward-filled rows: {output_stats['forward_filled_rows'][0]:,}")

    # Calculate coverage improvement
    input_rows = input_stats["rows"][0]
    output_rows = output_stats["rows"][0]
    forward_filled = output_stats["forward_filled_rows"][0]
    original_coverage = (input_rows / output_rows) * 100
    final_coverage = 100.0

    logger.info("\n" + "=" * 80)
    logger.info("SUMMARY")
    logger.info("=" * 80)
    logger.info(f"Original coverage: {original_coverage:.2f}% ({input_rows:,} rows)")
    logger.info(f"Final coverage: {final_coverage:.2f}% ({output_rows:,} rows)")
    logger.info(f"Rows added: {forward_filled:,} ({(forward_filled / output_rows) * 100:.2f}%)")

    elapsed = time.time() - start_time
    logger.info(f"\nTotal time: {elapsed:.1f} seconds ({elapsed / 60:.1f} minutes)")
    logger.info(f"Output file size: {output_path.stat().st_size / (1024**3):.2f} GB")

    logger.info("\n✅ Forward-fill resampling complete!")


def main() -> None:
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Resample perpetual data with forward-fill to continuous 1s intervals")

    parser.add_argument(
        "--input-file",
        default=DEFAULT_INPUT,
        help=f"Input parquet file (default: {DEFAULT_INPUT})",
    )
    parser.add_argument(
        "--output-file",
        default=DEFAULT_OUTPUT,
        help=f"Output parquet file (default: {DEFAULT_OUTPUT})",
    )
    parser.add_argument(
        "--start-date",
        default=START_DATE,
        help=f"Start date YYYY-MM-DD (default: {START_DATE})",
    )
    parser.add_argument(
        "--end-date",
        default=END_DATE,
        help=f"End date YYYY-MM-DD (default: {END_DATE})",
    )

    args = parser.parse_args()

    try:
        forward_fill_perpetual(
            input_file=args.input_file,
            output_file=args.output_file,
            start_date=args.start_date,
            end_date=args.end_date,
        )
    except Exception as e:
        logger.error(f"ERROR: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
