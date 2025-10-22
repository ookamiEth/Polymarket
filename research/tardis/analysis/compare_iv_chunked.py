#!/usr/bin/env python3

"""
Compare implied volatility calculations using constant vs. daily risk-free rates.

CHUNKED VERSION: Processes data in daily chunks to handle 204M x 204M row joins efficiently.
- Memory-safe: Processes ~300K rows at a time instead of 204M
- Simple: No complex staging or checkpointing
- Fast: Parallel processing capability for each day
"""

import argparse
import logging
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import polars as pl
import psutil

# Constants - using absolute paths for reliability
DEFAULT_CONSTANT_IV_FILE = (
    "/Users/lgierhake/Documents/ETH/BT/research/tardis/data/consolidated/quotes_1s_atm_short_dated_with_iv_CONSTANT_FULL.parquet"
)
DEFAULT_DAILY_IV_FILE = "/Users/lgierhake/Documents/ETH/BT/research/tardis/data/consolidated/quotes_1s_atm_short_dated_with_iv_daily_rates_FULL_v2.parquet"
DEFAULT_RATES_FILE = (
    "/Users/lgierhake/Documents/ETH/BT/research/risk_free_rate/data/blended_lending_rates_2023_2025.parquet"
)
DEFAULT_OUTPUT_DIR = "/Users/lgierhake/Documents/ETH/BT/research/tardis/analysis/output"

# Memory settings
MAX_MEMORY_GB = 5.0  # Maximum memory usage before warning
CHUNK_SIZE_DAYS = 1  # Process 1 day at a time

logger = logging.getLogger(__name__)


def setup_logging(verbose: bool = False) -> None:
    """Configure logging with custom format."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )


def get_memory_usage() -> float:
    """Get current memory usage in GB."""
    process = psutil.Process()
    return process.memory_info().rss / (1024**3)


def log_memory(operation: str = "") -> None:
    """Log current memory usage."""
    memory_gb = get_memory_usage()
    logger.debug(f"Memory usage {operation}: {memory_gb:.2f} GB")

    if memory_gb > MAX_MEMORY_GB:
        logger.warning(f"⚠️ High memory usage: {memory_gb:.2f} GB (limit: {MAX_MEMORY_GB:.1f} GB)")


def get_date_range(file_path: str) -> tuple[datetime, datetime]:
    """Get min and max dates from a parquet file."""
    logger.info(f"Analyzing date range in {file_path}...")

    # Use lazy scan to get date range without loading full data
    df = pl.scan_parquet(file_path)

    # Convert timestamp to date and get range
    date_stats = df.select(
        [
            pl.from_epoch("timestamp_seconds", time_unit="s").cast(pl.Date).min().alias("min_date"),
            pl.from_epoch("timestamp_seconds", time_unit="s").cast(pl.Date).max().alias("max_date"),
            pl.len().alias("total_rows"),
        ]
    ).collect()

    min_date = date_stats["min_date"][0]
    max_date = date_stats["max_date"][0]
    total_rows = date_stats["total_rows"][0]

    logger.info(f"  Date range: {min_date} to {max_date} ({total_rows:,} rows)")

    return min_date, max_date


def process_daily_chunk(
    constant_lazy: pl.LazyFrame,
    daily_lazy: pl.LazyFrame,
    date: datetime,
    join_keys: list[str],
) -> Optional[pl.DataFrame]:
    """
    Process a single day's worth of data.

    IMPORTANT: Deduplicates data before joining to prevent cartesian product explosion.
    Keeps only the last quote per second for each unique option.

    Returns DataFrame with comparison results for this day, or None if no data.
    """
    logger.debug(f"  Processing {date}...")

    # Filter both datasets for this specific date
    const_chunk = constant_lazy.filter(pl.from_epoch("timestamp_seconds", time_unit="s").cast(pl.Date) == date)

    daily_chunk = daily_lazy.filter(pl.from_epoch("timestamp_seconds", time_unit="s").cast(pl.Date) == date)

    # CRITICAL: Deduplicate before joining to prevent explosion
    # The data has multiple quotes per second for the same option (up to 42!)
    # We keep only the last quote per second for each unique key combination
    logger.debug("    Deduplicating data (keeping last quote per second)...")

    # Add a row number to keep track of the last row per group
    const_chunk_dedup = (
        const_chunk.with_columns(pl.lit(1).alias("_row_num"))
        .with_columns(pl.col("_row_num").cum_sum().over(join_keys).alias("_row_num"))
        .group_by(join_keys)
        .agg(
            [
                pl.col("spot_price").last(),
                pl.col("moneyness").last(),
                pl.col("time_to_expiry_days").last(),
                pl.col("implied_vol_bid").last(),
                pl.col("implied_vol_ask").last(),
                pl.col("iv_calc_status").last(),
            ]
        )
    )

    daily_chunk_dedup = (
        daily_chunk.with_columns(pl.lit(1).alias("_row_num"))
        .with_columns(pl.col("_row_num").cum_sum().over(join_keys).alias("_row_num"))
        .group_by(join_keys)
        .agg(
            [
                pl.col("implied_vol_bid").last(),
                pl.col("implied_vol_ask").last(),
                pl.col("iv_calc_status").last(),
            ]
        )
    )

    # Join the deduplicated chunks
    # Note: The columns are already selected during deduplication
    joined_chunk = const_chunk_dedup.join(
        daily_chunk_dedup,
        on=join_keys,
        how="inner",
        suffix="_daily",
    )

    # Filter for mutual success
    filtered_chunk = joined_chunk.filter(
        (pl.col("iv_calc_status") == "success") & (pl.col("iv_calc_status_daily") == "success")
    )

    # Calculate differences
    comparison_chunk = filtered_chunk.with_columns(
        [
            # Absolute differences
            (pl.col("implied_vol_bid_daily") - pl.col("implied_vol_bid")).alias("iv_bid_diff_abs"),
            (pl.col("implied_vol_ask_daily") - pl.col("implied_vol_ask")).alias("iv_ask_diff_abs"),
            # Mid IV
            ((pl.col("implied_vol_bid") + pl.col("implied_vol_ask")) / 2).alias("iv_mid_constant"),
            ((pl.col("implied_vol_bid_daily") + pl.col("implied_vol_ask_daily")) / 2).alias("iv_mid_daily"),
        ]
    )

    comparison_chunk = comparison_chunk.with_columns(
        [
            # Mid difference
            (pl.col("iv_mid_daily") - pl.col("iv_mid_constant")).alias("iv_mid_diff_abs"),
            # Relative difference (%)
            ((pl.col("iv_mid_daily") - pl.col("iv_mid_constant")) / pl.col("iv_mid_constant") * 100).alias(
                "iv_mid_diff_rel"
            ),
            # Add date for reference
            pl.from_epoch("timestamp_seconds", time_unit="s").cast(pl.Date).alias("date"),
        ]
    )

    # Filter invalid values
    comparison_chunk = comparison_chunk.filter(
        pl.col("iv_mid_constant").is_not_null()
        & pl.col("iv_mid_daily").is_not_null()
        & pl.col("iv_mid_constant").is_finite()
        & pl.col("iv_mid_daily").is_finite()
        & pl.col("iv_mid_diff_rel").is_finite()
        & (pl.col("iv_mid_constant") > 0)
        & (pl.col("iv_mid_constant") < 20)
        & (pl.col("iv_mid_daily") > 0)
        & (pl.col("iv_mid_daily") < 20)
    )

    # Collect this chunk (now deduplicated, so much smaller)
    try:
        result = comparison_chunk.collect()

        if len(result) == 0:
            logger.debug(f"    No valid data for {date}")
            return None

        logger.debug(f"    Processed {len(result):,} rows for {date} (after deduplication)")
        return result

    except Exception as e:
        logger.warning(f"    Failed to process {date}: {e}")
        return None


def compare_iv_calculations_chunked(
    constant_file: str,
    daily_file: str,
    rates_file: str,
    output_dir: str,
    test_mode: bool = False,
) -> None:
    """
    Compare IV calculations using chunked processing.

    Processes data day by day to avoid memory issues with 204M x 204M joins.
    """
    logger.info("=" * 80)
    logger.info("IV COMPARISON ANALYSIS (CHUNKED VERSION)")
    logger.info("=" * 80)

    # Check input files exist
    for file_path in [constant_file, daily_file]:
        if not Path(file_path).exists():
            raise FileNotFoundError(f"File not found: {file_path}")

    # Get date ranges
    const_min_date, const_max_date = get_date_range(constant_file)
    daily_min_date, daily_max_date = get_date_range(daily_file)

    # Use overlapping date range
    start_date = max(const_min_date, daily_min_date)
    end_date = min(const_max_date, daily_max_date)

    logger.info(f"Processing date range: {start_date} to {end_date}")

    # Generate list of dates to process
    current_date = start_date
    dates_to_process = []
    while current_date <= end_date:
        dates_to_process.append(current_date)
        # Move to next day using standard datetime
        current_date = current_date + timedelta(days=1)

    # Limit dates for test mode
    if test_mode:
        dates_to_process = dates_to_process[:7]  # Process only first week
        logger.info(f"TEST MODE: Processing only {len(dates_to_process)} days")

    logger.info(f"Total days to process: {len(dates_to_process)}")

    # Setup output
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Create temp directory for chunk files
    temp_dir = output_path / "temp_chunks"
    temp_dir.mkdir(exist_ok=True)

    # Clean up any existing temp files
    for old_file in temp_dir.glob("*.parquet"):
        old_file.unlink()

    comparison_file = output_path / "comparison_chunked.parquet"

    # Load datasets lazily
    logger.info("\nSetting up lazy data readers...")
    const_lazy = pl.scan_parquet(constant_file)
    daily_lazy = pl.scan_parquet(daily_file)

    # Define join keys
    join_keys = [
        "timestamp_seconds",
        "symbol",
        "exchange",
        "type",
        "strike_price",
        "expiry_timestamp",
    ]

    # Process each day
    logger.info("\nProcessing daily chunks...")
    start_time = time.time()
    total_rows_processed = 0
    chunks_processed = 0
    chunk_files = []

    for i, date in enumerate(dates_to_process, 1):
        if i % 10 == 0 or i == 1:
            logger.info(f"Progress: {i}/{len(dates_to_process)} days ({i / len(dates_to_process) * 100:.1f}%)")
            log_memory(f"after {i} days")

        # Process this day's chunk
        chunk_result = process_daily_chunk(const_lazy, daily_lazy, date, join_keys)

        if chunk_result is not None and len(chunk_result) > 0:
            # Write chunk to its own temp file (no appending, no memory buildup!)
            chunk_file = temp_dir / f"chunk_{i:03d}_{date}.parquet"
            chunk_result.write_parquet(
                chunk_file,
                compression="snappy",
                statistics=False,  # Skip stats for temp files
            )
            chunk_files.append(chunk_file)

            total_rows_processed += len(chunk_result)
            chunks_processed += 1

            # Free memory immediately
            del chunk_result

    # Report processing results
    elapsed = time.time() - start_time
    logger.info("\n✅ Chunked processing complete!")
    logger.info(f"  Days processed: {len(dates_to_process)}")
    logger.info(f"  Chunks with data: {chunks_processed}")
    logger.info(f"  Total rows: {total_rows_processed:,}")
    logger.info(f"  Time elapsed: {elapsed:.1f} seconds ({elapsed / 60:.1f} minutes)")
    logger.info(f"  Processing rate: {total_rows_processed / elapsed:,.0f} rows/sec")
    logger.info(f"  Peak memory: {get_memory_usage():.2f} GB")

    # Combine all chunk files into final output
    if chunks_processed > 0:
        logger.info("\nCombining chunk files into final output...")

        # Use lazy scan to combine all chunks efficiently
        combined_df = pl.scan_parquet(chunk_files)

        # Write final combined file
        combined_df.sink_parquet(
            comparison_file,
            compression="snappy",
            statistics=True,
        )

        logger.info(f"✅ Final comparison data written to {comparison_file}")

        # Clean up temp files
        logger.info("Cleaning up temporary files...")
        for chunk_file in chunk_files:
            chunk_file.unlink()
        temp_dir.rmdir()  # Remove empty temp directory

        # Generate statistics from the output file
        generate_statistics(comparison_file, output_path)
    else:
        logger.warning("No data processed - skipping statistics generation")


def generate_statistics(comparison_file: Path, output_path: Path) -> None:
    """Generate summary statistics from the comparison results."""
    logger.info("\n" + "=" * 80)
    logger.info("GENERATING SUMMARY STATISTICS")
    logger.info("=" * 80)

    # Use lazy scan for statistics
    df_stats = pl.scan_parquet(comparison_file)

    # Overall statistics
    overall_stats = df_stats.select(
        [
            pl.len().alias("n_observations"),
            # Means
            pl.col("iv_mid_constant").mean().alias("iv_constant_mean"),
            pl.col("iv_mid_daily").mean().alias("iv_daily_mean"),
            pl.col("iv_mid_diff_abs").mean().alias("diff_abs_mean"),
            pl.col("iv_mid_diff_rel").mean().alias("diff_rel_mean_pct"),
            # Medians
            pl.col("iv_mid_diff_abs").median().alias("diff_abs_median"),
            pl.col("iv_mid_diff_rel").median().alias("diff_rel_median_pct"),
            # Standard deviations
            pl.col("iv_mid_diff_abs").std().alias("diff_abs_std"),
            pl.col("iv_mid_diff_rel").std().alias("diff_rel_std_pct"),
            # Absolute values
            pl.col("iv_mid_diff_abs").abs().mean().alias("diff_abs_mean_absolute"),
            pl.col("iv_mid_diff_rel").abs().mean().alias("diff_rel_mean_absolute_pct"),
            # Min/Max
            pl.col("iv_mid_diff_abs").min().alias("diff_abs_min"),
            pl.col("iv_mid_diff_abs").max().alias("diff_abs_max"),
            # Correlation
            pl.corr("iv_mid_constant", "iv_mid_daily").alias("correlation"),
        ]
    ).collect()

    # Display results
    stats = overall_stats.row(0, named=True)

    logger.info(f"\nQuotes analyzed: {stats['n_observations']:,}")
    logger.info("\nAverage IV:")
    logger.info(f"  Constant rate: {stats['iv_constant_mean']:.4f}")
    logger.info(f"  Daily rates: {stats['iv_daily_mean']:.4f}")
    logger.info("\nDifferences (Daily - Constant):")
    logger.info(f"  Mean: {stats['diff_abs_mean']:.6f} ({stats['diff_rel_mean_pct']:.4f}%)")
    logger.info(
        f"  Mean (absolute): {stats['diff_abs_mean_absolute']:.6f} ({stats['diff_rel_mean_absolute_pct']:.4f}%)"
    )
    logger.info(f"  Median: {stats['diff_abs_median']:.6f} ({stats['diff_rel_median_pct']:.4f}%)")
    logger.info(f"  Std Dev: {stats['diff_abs_std']:.6f} ({stats['diff_rel_std_pct']:.4f}%)")
    logger.info(f"  Range: [{stats['diff_abs_min']:.6f}, {stats['diff_abs_max']:.6f}]")
    logger.info(f"\nCorrelation: {stats['correlation']:.6f}")

    # Materiality analysis
    logger.info("\nMateriality Analysis:")

    materiality = (
        df_stats.select(
            [
                (pl.col("iv_mid_diff_abs").abs() > 0.01).sum().alias("above_1bp"),
                (pl.col("iv_mid_diff_abs").abs() > 0.05).sum().alias("above_5bp"),
                (pl.col("iv_mid_diff_abs").abs() > 0.10).sum().alias("above_10bp"),
                (pl.col("iv_mid_diff_rel").abs() > 5.0).sum().alias("above_5pct"),
                (pl.col("iv_mid_diff_rel").abs() > 10.0).sum().alias("above_10pct"),
                pl.len().alias("total"),
            ]
        )
        .collect()
        .row(0, named=True)
    )

    logger.info("  Absolute differences:")
    logger.info(
        f"    |Diff| > 0.01: {materiality['above_1bp']:,} ({materiality['above_1bp'] / materiality['total'] * 100:.2f}%)"
    )
    logger.info(
        f"    |Diff| > 0.05: {materiality['above_5bp']:,} ({materiality['above_5bp'] / materiality['total'] * 100:.2f}%)"
    )
    logger.info(
        f"    |Diff| > 0.10: {materiality['above_10bp']:,} ({materiality['above_10bp'] / materiality['total'] * 100:.2f}%)"
    )

    logger.info("  Relative differences:")
    logger.info(
        f"    |Diff| > 5%: {materiality['above_5pct']:,} ({materiality['above_5pct'] / materiality['total'] * 100:.2f}%)"
    )
    logger.info(
        f"    |Diff| > 10%: {materiality['above_10pct']:,} ({materiality['above_10pct'] / materiality['total'] * 100:.2f}%)"
    )

    # By option type
    logger.info("\nBy Option Type:")
    by_type = (
        df_stats.group_by("type")
        .agg(
            [
                pl.len().alias("n_obs"),
                pl.col("iv_mid_diff_abs").mean().alias("diff_abs_mean"),
                pl.col("iv_mid_diff_abs").abs().mean().alias("diff_abs_mean_absolute"),
                pl.col("iv_mid_diff_rel").abs().mean().alias("diff_rel_mean_absolute_pct"),
            ]
        )
        .sort("type")
        .collect()
    )

    for row in by_type.iter_rows(named=True):
        logger.info(
            f"  {row['type']}: {row['n_obs']:,} obs, "
            f"mean abs diff: {row['diff_abs_mean_absolute']:.6f} "
            f"({row['diff_rel_mean_absolute_pct']:.4f}%)"
        )

    # By moneyness bins
    logger.info("\nBy Moneyness:")

    # Add moneyness bins using lazy operations
    df_with_bins = df_stats.with_columns(
        [
            pl.when(pl.col("moneyness") < 0.9)
            .then(pl.lit("OTM (<0.9)"))
            .when(pl.col("moneyness") < 1.1)
            .then(pl.lit("ATM (0.9-1.1)"))
            .otherwise(pl.lit("ITM (>1.1)"))
            .alias("moneyness_bin")
        ]
    )

    by_moneyness = (
        df_with_bins.group_by("moneyness_bin")
        .agg(
            [
                pl.len().alias("n_obs"),
                pl.col("iv_mid_diff_abs").abs().mean().alias("diff_abs_mean_absolute"),
                pl.col("iv_mid_diff_rel").abs().mean().alias("diff_rel_mean_absolute_pct"),
            ]
        )
        .sort("moneyness_bin")
        .collect()
    )

    for row in by_moneyness.iter_rows(named=True):
        logger.info(
            f"  {row['moneyness_bin']}: {row['n_obs']:,} obs, "
            f"mean abs diff: {row['diff_abs_mean_absolute']:.6f} "
            f"({row['diff_rel_mean_absolute_pct']:.4f}%)"
        )

    # Save summary statistics
    summary_file = output_path / "summary_stats_chunked.parquet"
    overall_stats.write_parquet(summary_file)
    logger.info(f"\n✅ Summary statistics saved to {summary_file}")


def main() -> None:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Compare IV calculations using chunked processing (MEMORY-EFFICIENT VERSION)"
    )

    parser.add_argument(
        "--constant-file",
        default=DEFAULT_CONSTANT_IV_FILE,
        help=f"IV file with constant rates (default: {DEFAULT_CONSTANT_IV_FILE})",
    )
    parser.add_argument(
        "--daily-file",
        default=DEFAULT_DAILY_IV_FILE,
        help=f"IV file with daily rates (default: {DEFAULT_DAILY_IV_FILE})",
    )
    parser.add_argument(
        "--rates-file",
        default=DEFAULT_RATES_FILE,
        help=f"Risk-free rates file (default: {DEFAULT_RATES_FILE})",
    )
    parser.add_argument(
        "--output-dir",
        default=DEFAULT_OUTPUT_DIR,
        help=f"Output directory (default: {DEFAULT_OUTPUT_DIR})",
    )
    parser.add_argument(
        "--test-mode",
        action="store_true",
        help="Test mode - process only first week of data",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        default=True,  # Default to verbose for progress tracking
        help="Enable verbose logging (default: True)",
    )

    args = parser.parse_args()

    # Setup logging
    setup_logging(args.verbose)

    # Override files for test mode if test file exists
    if args.test_mode:
        test_file = "/Users/lgierhake/Documents/ETH/BT/research/tardis/data/consolidated/quotes_1s_atm_short_dated_with_iv_10m_test.parquet"
        if Path(test_file).exists():
            logger.info("🧪 TEST MODE: Using 10M row test file")
            args.constant_file = test_file
            # Check if daily test file exists
            daily_test = test_file.replace("_with_iv_10m_test", "_with_iv_daily_rates_10m_test")
            if Path(daily_test).exists():
                args.daily_file = daily_test
            else:
                logger.warning(f"Daily test file not found: {daily_test}")
                logger.info("Using regular daily file - results may be limited")

    start_time = time.time()
    logger.info(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"Initial memory: {get_memory_usage():.2f} GB")

    try:
        compare_iv_calculations_chunked(
            args.constant_file,
            args.daily_file,
            args.rates_file,
            args.output_dir,
            args.test_mode,
        )

        elapsed = time.time() - start_time
        logger.info("\n✅ SUCCESS: Analysis complete!")
        logger.info(f"Total runtime: {elapsed:.1f} seconds ({elapsed / 60:.1f} minutes)")
        logger.info(f"Final memory: {get_memory_usage():.2f} GB")

    except Exception as e:
        logger.error(f"❌ ERROR: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
