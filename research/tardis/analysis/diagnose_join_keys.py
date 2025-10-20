#!/usr/bin/env python3

"""
Diagnose join key uniqueness in IV comparison datasets.

This script analyzes the join keys to understand why the join operation
is hanging when comparing constant vs daily risk-free rate IV calculations.

Checks for:
1. Duplicate join key combinations
2. Join explosion risk (cartesian product)
3. Memory requirements for the join
"""

import argparse
import logging
import sys
import time
from pathlib import Path

import polars as pl

# Join keys used in the comparison
JOIN_KEYS = [
    "timestamp_seconds",
    "symbol",
    "exchange",
    "type",
    "strike_price",
    "expiry_timestamp",
]

logger = logging.getLogger(__name__)


def setup_logging(verbose: bool = False) -> None:
    """Configure logging."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )


def analyze_join_keys(file_path: str, dataset_name: str) -> dict:
    """
    Analyze join key uniqueness in a dataset.

    Args:
        file_path: Path to parquet file
        dataset_name: Name for logging

    Returns:
        Dictionary with analysis results
    """
    logger.info(f"\n{'='*80}")
    logger.info(f"Analyzing {dataset_name}")
    logger.info(f"{'='*80}")

    # Scan file lazily
    df = pl.scan_parquet(file_path)

    # Filter for successful IV calculations (as done in main script)
    df_success = df.filter(pl.col("iv_calc_status") == "success")

    # Get total row count
    logger.info("Counting total rows...")
    total_rows = df_success.select(pl.len()).collect().item()
    logger.info(f"Total successful rows: {total_rows:,}")

    # Count unique combinations of join keys
    logger.info("\nAnalyzing join key uniqueness...")
    unique_keys = (
        df_success.select(JOIN_KEYS)
        .unique()
        .select(pl.len())
        .collect()
        .item()
    )
    logger.info(f"Unique join key combinations: {unique_keys:,}")
    logger.info(f"Duplication ratio: {total_rows / unique_keys:.2f}x")

    if total_rows > unique_keys:
        logger.warning(f"⚠️ DUPLICATES DETECTED: {total_rows - unique_keys:,} duplicate rows")

        # Find examples of duplicates (sample to avoid OOM)
        logger.info("\nFinding duplicate examples (sampling 1M rows)...")

        # Sample and find duplicates
        sample_df = (
            df_success.select(JOIN_KEYS)
            .head(1_000_000)
            .collect()
        )

        # Find duplicate groups
        duplicates = (
            sample_df.group_by(JOIN_KEYS)
            .agg(pl.len().alias("count"))
            .filter(pl.col("count") > 1)
            .sort("count", descending=True)
            .head(10)
        )

        if len(duplicates) > 0:
            logger.info("\nTop 10 duplicate key combinations (from 1M sample):")
            for row in duplicates.iter_rows(named=True):
                logger.info(f"  Count: {row['count']} | Symbol: {row['symbol']} | "
                           f"Strike: {row['strike_price']} | Type: {row['type']}")

            # Calculate max duplicates
            max_dups = duplicates["count"].max()
            logger.warning(f"\n⚠️ Maximum duplicates per key: {max_dups}")
            logger.warning(f"⚠️ Worst case join explosion: {max_dups}² = {max_dups**2}x per row")
    else:
        logger.info("✅ No duplicates - join keys are unique")

    # Analyze distribution by key components
    logger.info("\nAnalyzing key component distribution...")

    # Check exchange uniqueness (should be just DERIBIT)
    exchanges = (
        df_success.select("exchange")
        .unique()
        .collect()
    )
    logger.info(f"Unique exchanges: {exchanges['exchange'].to_list()}")

    # Check symbol count
    symbol_count = (
        df_success.select("symbol")
        .unique()
        .select(pl.len())
        .collect()
        .item()
    )
    logger.info(f"Unique symbols: {symbol_count:,}")

    # Check timestamp distribution (sample)
    logger.info("\nChecking timestamp distribution (1M sample)...")
    timestamp_stats = (
        df_success.select("timestamp_seconds")
        .head(1_000_000)
        .collect()
        .group_by("timestamp_seconds")
        .agg(pl.len().alias("quotes_per_second"))
        .select([
            pl.col("quotes_per_second").mean().alias("avg_quotes_per_second"),
            pl.col("quotes_per_second").max().alias("max_quotes_per_second"),
            pl.col("quotes_per_second").quantile(0.95).alias("p95_quotes_per_second"),
        ])
    )

    stats = timestamp_stats.to_dict(as_series=False)
    logger.info(f"  Avg quotes per second: {stats['avg_quotes_per_second'][0]:.1f}")
    logger.info(f"  Max quotes per second: {stats['max_quotes_per_second'][0]}")
    logger.info(f"  P95 quotes per second: {stats['p95_quotes_per_second'][0]:.1f}")

    return {
        "total_rows": total_rows,
        "unique_keys": unique_keys,
        "duplication_ratio": total_rows / unique_keys,
        "has_duplicates": total_rows > unique_keys,
    }


def estimate_join_explosion(
    const_stats: dict,
    daily_stats: dict,
) -> None:
    """
    Estimate the potential join explosion.

    Args:
        const_stats: Stats from constant rate dataset
        daily_stats: Stats from daily rate dataset
    """
    logger.info(f"\n{'='*80}")
    logger.info("JOIN EXPLOSION RISK ASSESSMENT")
    logger.info(f"{'='*80}")

    # Best case: unique keys on both sides
    if not const_stats["has_duplicates"] and not daily_stats["has_duplicates"]:
        logger.info("✅ BEST CASE: No duplicates on either side")
        logger.info(f"Expected join result: ~{const_stats['unique_keys']:,} rows")
        logger.info("Join is safe to proceed")
        return

    # Worst case: duplicates on both sides
    const_dup_ratio = const_stats["duplication_ratio"]
    daily_dup_ratio = daily_stats["duplication_ratio"]

    logger.warning("⚠️ DUPLICATES DETECTED ON BOTH SIDES")
    logger.info(f"Constant dataset duplication: {const_dup_ratio:.2f}x")
    logger.info(f"Daily dataset duplication: {daily_dup_ratio:.2f}x")

    # Estimate result size
    estimated_rows = const_stats["unique_keys"] * const_dup_ratio * daily_dup_ratio
    logger.warning(f"\n⚠️ Estimated join result: {estimated_rows:,.0f} rows")
    logger.warning(f"⚠️ Memory required: ~{estimated_rows * 8 * 10 / 1e9:.1f} GB")

    if estimated_rows > 500_000_000:
        logger.error("❌ JOIN WILL LIKELY EXPLODE AND CAUSE OOM")
        logger.error("Recommendation: Add deduplication before joining")
    elif estimated_rows > 250_000_000:
        logger.warning("⚠️ JOIN IS RISKY - May cause memory issues")
        logger.warning("Recommendation: Use streaming mode or deduplicate first")
    else:
        logger.info("✅ Join should be manageable with streaming mode")


def suggest_fix(const_stats: dict, daily_stats: dict) -> None:
    """
    Suggest fixes based on the analysis.

    Args:
        const_stats: Stats from constant rate dataset
        daily_stats: Stats from daily rate dataset
    """
    logger.info(f"\n{'='*80}")
    logger.info("RECOMMENDED FIXES")
    logger.info(f"{'='*80}")

    if const_stats["has_duplicates"] or daily_stats["has_duplicates"]:
        logger.info("\n1. DEDUPLICATE BEFORE JOINING:")
        logger.info("""
    # Take last quote per second for each option
    df_constant_dedup = df_constant_success.sort("timestamp_seconds").unique(
        subset=join_keys,
        keep="last"
    )
    df_daily_dedup = df_daily_success.sort("timestamp_seconds").unique(
        subset=join_keys,
        keep="last"
    )
        """)

        logger.info("\n2. OR SKIP JOIN VALIDATION:")
        logger.info("""
    # Remove the expensive count operation
    # df_comparison.select(pl.len()).collect()  # REMOVE THIS

    # Go straight to streaming write
    df_comparison.sink_parquet(output_file, streaming=True)
        """)

        logger.info("\n3. OR USE SAMPLE-BASED VALIDATION:")
        logger.info("""
    # Validate on small sample instead of full dataset
    sample_test = df_comparison.head(10_000).collect()
    if len(sample_test) == 10_000:
        logger.info("✓ Join validated on 10K sample")
        # Proceed with full pipeline
        """)
    else:
        logger.info("\n✅ No duplicates detected - join should work as-is")
        logger.info("The hang might be due to memory pressure. Try:")
        logger.info("  - Increasing available memory")
        logger.info("  - Using streaming mode earlier")
        logger.info("  - Running on a machine with more RAM")


def main() -> None:
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Diagnose join key uniqueness")

    parser.add_argument(
        "--constant-file",
        default="research/tardis/data/consolidated/quotes_1s_atm_short_dated_with_iv.parquet",
        help="IV file with constant rates",
    )
    parser.add_argument(
        "--daily-file",
        default="research/tardis/data/consolidated/quotes_1s_atm_short_dated_with_iv_daily_rates.parquet",
        help="IV file with daily rates",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose logging",
    )

    args = parser.parse_args()
    setup_logging(args.verbose)

    start_time = time.time()

    try:
        # Check files exist
        for file_path in [args.constant_file, args.daily_file]:
            if not Path(file_path).exists():
                raise FileNotFoundError(f"File not found: {file_path}")

        # Analyze both datasets
        const_stats = analyze_join_keys(args.constant_file, "CONSTANT RATE DATASET")
        daily_stats = analyze_join_keys(args.daily_file, "DAILY RATE DATASET")

        # Estimate join explosion
        estimate_join_explosion(const_stats, daily_stats)

        # Suggest fixes
        suggest_fix(const_stats, daily_stats)

        elapsed = time.time() - start_time
        logger.info(f"\n✅ Analysis complete in {elapsed:.1f} seconds")

    except Exception as e:
        logger.error(f"ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()