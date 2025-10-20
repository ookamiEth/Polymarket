#!/usr/bin/env python3

"""
Fast diagnostic for join key uniqueness using sampling.

This script uses sampling to quickly diagnose the join issue without
processing all 198M rows.
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


def analyze_with_sampling(file_path: str, dataset_name: str, sample_size: int = 1_000_000) -> dict:
    """
    Analyze join key uniqueness using sampling for speed.

    Args:
        file_path: Path to parquet file
        dataset_name: Name for logging
        sample_size: Number of rows to sample

    Returns:
        Dictionary with analysis results
    """
    logger.info(f"\n{'='*80}")
    logger.info(f"Analyzing {dataset_name} (SAMPLING {sample_size:,} rows)")
    logger.info(f"{'='*80}")

    # Read just the sample
    logger.info(f"Reading sample from {file_path}...")
    df_sample = (
        pl.scan_parquet(file_path)
        .filter(pl.col("iv_calc_status") == "success")
        .head(sample_size)
        .collect()
    )

    actual_sample = len(df_sample)
    logger.info(f"Sample size: {actual_sample:,} rows")

    # Analyze join key uniqueness in sample
    logger.info("\nAnalyzing join key uniqueness in sample...")

    # Count unique combinations
    unique_keys = df_sample.select(JOIN_KEYS).unique()
    n_unique = len(unique_keys)

    logger.info(f"Sample rows: {actual_sample:,}")
    logger.info(f"Unique join key combinations: {n_unique:,}")
    logger.info(f"Duplication ratio: {actual_sample / n_unique:.2f}x")

    # Find duplicates
    duplicates = (
        df_sample.group_by(JOIN_KEYS)
        .agg(pl.len().alias("count"))
        .filter(pl.col("count") > 1)
        .sort("count", descending=True)
    )

    n_dup_groups = len(duplicates)
    total_dup_rows = duplicates["count"].sum() if n_dup_groups > 0 else 0

    if n_dup_groups > 0:
        max_dups = duplicates["count"].max()
        logger.warning(f"\n⚠️ DUPLICATES FOUND:")
        logger.warning(f"  Duplicate groups: {n_dup_groups:,}")
        logger.warning(f"  Total duplicate rows: {total_dup_rows:,}")
        logger.warning(f"  Max duplicates per key: {max_dups}")

        # Show top duplicates
        logger.info("\nTop 10 duplicate combinations:")
        for row in duplicates.head(10).iter_rows(named=True):
            logger.info(
                f"  Count: {row['count']:3d} | Symbol: {row['symbol'][:30]:30s} | "
                f"Strike: {row['strike_price']:8.0f} | Type: {row['type']}"
            )
    else:
        logger.info("✅ No duplicates found in sample")

    # Analyze key components
    logger.info("\nKey component analysis:")

    # Unique values per component
    for col in JOIN_KEYS:
        n_unique_col = df_sample[col].n_unique()
        logger.info(f"  {col:20s}: {n_unique_col:,} unique values")

    # Timestamp distribution
    ts_stats = (
        df_sample.group_by("timestamp_seconds")
        .agg(pl.len().alias("quotes_per_second"))
        .select([
            pl.col("quotes_per_second").mean().alias("avg"),
            pl.col("quotes_per_second").max().alias("max"),
            pl.col("quotes_per_second").quantile(0.95).alias("p95"),
        ])
    ).to_dict(as_series=False)

    logger.info("\nQuotes per timestamp:")
    logger.info(f"  Average: {ts_stats['avg'][0]:.1f}")
    logger.info(f"  Maximum: {ts_stats['max'][0]}")
    logger.info(f"  P95: {ts_stats['p95'][0]:.1f}")

    # Estimate for full dataset
    logger.info("\nExtrapolating to full dataset (~198M rows):")
    extrapolated_dups = 198_000_000 / n_unique
    logger.info(f"  Estimated duplication ratio: {extrapolated_dups:.2f}x")

    if extrapolated_dups > 1.5:
        logger.warning(f"  ⚠️ HIGH DUPLICATION - Join may explode!")
        logger.warning(f"  Potential join size: {int(198_000_000 * extrapolated_dups):,} rows")

    return {
        "sample_size": actual_sample,
        "unique_keys": n_unique,
        "duplication_ratio": actual_sample / n_unique,
        "max_duplicates": duplicates["count"].max() if n_dup_groups > 0 else 1,
        "has_duplicates": n_dup_groups > 0,
    }


def test_small_join(const_file: str, daily_file: str, n_rows: int = 10_000) -> None:
    """
    Test join on small subset to verify behavior.

    Args:
        const_file: Path to constant rate file
        daily_file: Path to daily rate file
        n_rows: Number of rows to test with
    """
    logger.info(f"\n{'='*80}")
    logger.info(f"TESTING JOIN ON {n_rows:,} ROWS")
    logger.info(f"{'='*80}")

    # Load small samples
    logger.info("Loading samples...")
    const_sample = (
        pl.scan_parquet(const_file)
        .filter(pl.col("iv_calc_status") == "success")
        .head(n_rows)
        .collect()
    )

    daily_sample = (
        pl.scan_parquet(daily_file)
        .filter(pl.col("iv_calc_status") == "success")
        .head(n_rows)
        .collect()
    )

    logger.info(f"Constant sample: {len(const_sample):,} rows")
    logger.info(f"Daily sample: {len(daily_sample):,} rows")

    # Test join
    logger.info("\nPerforming test join...")
    start = time.time()

    joined = const_sample.join(
        daily_sample,
        on=JOIN_KEYS,
        how="inner",
        suffix="_daily",
    )

    elapsed = time.time() - start
    result_rows = len(joined)

    logger.info(f"✅ Join completed in {elapsed:.3f}s")
    logger.info(f"Result rows: {result_rows:,}")

    # Calculate explosion factor
    explosion = result_rows / min(len(const_sample), len(daily_sample))
    logger.info(f"Explosion factor: {explosion:.2f}x")

    if explosion > 1.1:
        logger.warning("⚠️ JOIN EXPLOSION DETECTED!")
        logger.warning(f"For 198M rows, this would create ~{int(198_000_000 * explosion):,} rows")
        logger.warning("This explains why the join is hanging!")
    else:
        logger.info("✅ Join looks normal (no explosion)")

    return explosion


def suggest_solution(explosion_factor: float) -> None:
    """
    Suggest solution based on findings.

    Args:
        explosion_factor: Join explosion factor from test
    """
    logger.info(f"\n{'='*80}")
    logger.info("RECOMMENDED SOLUTION")
    logger.info(f"{'='*80}")

    if explosion_factor > 1.1:
        logger.info("""
The join is exploding due to duplicate join keys (multiple quotes per option per second).

IMMEDIATE FIX - Update compare_iv_constant_vs_daily_rates.py:

1. REMOVE the expensive join validation (lines 161-168):
   # DELETE THIS BLOCK - it forces materialization of 200M+ row join
   # logger.info("Validating join result size...")
   # join_count = df_comparison.select(pl.len()).collect().item()
   # ... validation code ...

2. REPLACE with sample-based validation:
   # Validate on small sample only
   logger.info("Validating join on 10K sample...")
   sample_test = df_comparison.head(10_000).collect()
   if len(sample_test) > 0:
       logger.info(f"✓ Join sample successful: {len(sample_test):,} rows")

3. ADD deduplication before join (optional, for cleaner data):
   # Take last quote per second for each option
   df_constant_success = df_constant_success.unique(
       subset=join_keys,
       keep="last"
   )
   df_daily_success = df_daily_success.unique(
       subset=join_keys,
       keep="last"
   )

4. USE streaming write immediately:
   # Skip all validation, go straight to streaming
   if join_count > 100_000_000:  # Remove this condition
       # Just always use streaming for safety
       write_comparison_to_disk_streaming(df_comparison_lazy, intermediate_file)
""")
    else:
        logger.info("""
Join doesn't show explosion in sample. The issue is likely memory pressure.

RECOMMENDED FIX:

1. Skip the join count validation entirely
2. Go straight to streaming write
3. Use sample-based validation if needed
""")


def main() -> None:
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Fast join diagnostics using sampling")

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
        "--sample-size",
        type=int,
        default=1_000_000,
        help="Sample size for analysis (default: 1M)",
    )
    parser.add_argument(
        "--test-join-size",
        type=int,
        default=10_000,
        help="Rows for join test (default: 10K)",
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Verbose output",
    )

    args = parser.parse_args()
    setup_logging(args.verbose)

    try:
        # Check files exist
        for file_path in [args.constant_file, args.daily_file]:
            if not Path(file_path).exists():
                raise FileNotFoundError(f"File not found: {file_path}")

        # Quick sampling analysis
        const_stats = analyze_with_sampling(
            args.constant_file,
            "CONSTANT RATE DATASET",
            args.sample_size
        )

        daily_stats = analyze_with_sampling(
            args.daily_file,
            "DAILY RATE DATASET",
            args.sample_size
        )

        # Test join on small subset
        explosion = test_small_join(
            args.constant_file,
            args.daily_file,
            args.test_join_size
        )

        # Provide solution
        suggest_solution(explosion)

        logger.info("\n✅ Diagnosis complete!")

    except Exception as e:
        logger.error(f"ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()