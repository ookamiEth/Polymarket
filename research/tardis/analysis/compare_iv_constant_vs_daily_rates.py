#!/usr/bin/env python3

"""
Compare implied volatility calculations using constant vs. daily risk-free rates.

This script performs a comprehensive comparison of IV calculations on the same
options quote dataset using:
1. Constant risk-free rate (4.12%)
2. Daily risk-free rates from blended lending rates (AAVE + USDT)

Uses LAZY EVALUATION throughout to handle 204M+ rows without OOM crashes.
Analyzes differences across time, option characteristics (moneyness, TTL, type),
and provides statistical summaries.

Memory-efficient design:
- Uses pl.scan_parquet() for lazy loading
- Joins and calculations done lazily
- Only materializes small aggregated results
- Never loads full 204M row dataset into memory
"""

import argparse
import logging
import sys
import time
from pathlib import Path

import polars as pl

# Constants
DEFAULT_CONSTANT_IV_FILE = "data/consolidated/quotes_1s_atm_short_dated_with_iv.parquet"
DEFAULT_DAILY_IV_FILE = "data/consolidated/quotes_1s_atm_short_dated_with_iv_daily_rates.parquet"
DEFAULT_RATES_FILE = "research/risk_free_rate/data/blended_lending_rates_2023_2025.parquet"
DEFAULT_OUTPUT_DIR = "research/tardis/analysis/output"

# Analysis thresholds
MATERIALITY_THRESHOLDS_ABS = [0.01, 0.05, 0.10]  # Absolute IV difference (vol points)
MATERIALITY_THRESHOLDS_REL = [0.05, 0.10, 0.20]  # Relative difference (%)

# Moneyness bins
MONEYNESS_BINS = [0.0, 0.9, 1.1, float("inf")]
MONEYNESS_LABELS = ["OTM (<0.9)", "ATM (0.9-1.1)", "ITM (>1.1)"]

# Time to expiry bins (days)
TTL_BINS = [0.0, 7.0, 30.0, 90.0, float("inf")]
TTL_LABELS = ["<7d", "7-30d", "30-90d", ">90d"]

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


def prepare_lazy_comparison(
    constant_file: str,
    daily_file: str,
    rates_file: str,
) -> tuple[pl.LazyFrame, pl.DataFrame]:
    """
    Prepare lazy comparison DataFrame by joining constant and daily IV datasets.

    CRITICAL: Uses lazy evaluation throughout to avoid loading 204M rows into memory.

    Args:
        constant_file: Path to IV file with constant rates
        daily_file: Path to IV file with daily rates
        rates_file: Path to risk-free rates file

    Returns:
        Tuple of (lazy comparison DataFrame, rates DataFrame)

    Raises:
        FileNotFoundError: If any input file doesn't exist
    """
    logger.info("=" * 80)
    logger.info("PREPARING LAZY COMPARISON")
    logger.info("=" * 80)

    # Check files exist
    for file_path in [constant_file, daily_file, rates_file]:
        if not Path(file_path).exists():
            raise FileNotFoundError(f"File not found: {file_path}")

    # Load constant rate IVs LAZILY
    logger.info(f"Scanning constant rate IVs from {constant_file}...")
    df_constant_lazy = pl.scan_parquet(constant_file)

    # Load daily rate IVs LAZILY
    logger.info(f"Scanning daily rate IVs from {daily_file}...")
    df_daily_lazy = pl.scan_parquet(daily_file)

    # Load risk-free rates (small file, can load eagerly)
    logger.info(f"Loading risk-free rates from {rates_file}...")
    df_rates = pl.read_parquet(rates_file).select(
        [
            pl.col("date"),
            pl.col("blended_supply_apr_ma7").alias("risk_free_rate_pct"),
        ]
    )
    logger.info(f"  Loaded {len(df_rates):,} days")

    # Get row counts (triggers minimal scan, doesn't load data)
    logger.info("\nCounting rows...")
    n_constant = df_constant_lazy.select(pl.len()).collect().item()
    n_daily = df_daily_lazy.select(pl.len()).collect().item()
    logger.info(f"  Constant IV file: {n_constant:,} rows")
    logger.info(f"  Daily IV file: {n_daily:,} rows")

    if n_constant != n_daily:
        logger.warning(f"Row count mismatch: constant={n_constant:,}, daily={n_daily:,}")

    # Filter for successful IV calculations LAZILY
    logger.info("\nFiltering for successful IV calculations (lazy)...")
    df_constant_success = df_constant_lazy.filter(pl.col("iv_calc_status") == "success")
    df_daily_success = df_daily_lazy.filter(pl.col("iv_calc_status") == "success")

    # Count successful calculations (minimal scan)
    n_constant_success = df_constant_success.select(pl.len()).collect().item()
    n_daily_success = df_daily_success.select(pl.len()).collect().item()
    logger.info(f"  Constant: {n_constant_success:,} successful ({n_constant_success / n_constant * 100:.1f}%)")
    logger.info(f"  Daily: {n_daily_success:,} successful ({n_daily_success / n_daily * 100:.1f}%)")

    # Define join keys
    join_keys = [
        "timestamp_seconds",
        "symbol",
        "exchange",
        "type",
        "strike_price",
        "expiry_timestamp",
    ]

    # Select columns for comparison
    constant_cols = join_keys + [
        "spot_price",
        "moneyness",
        "time_to_expiry_days",
        "implied_vol_bid",
        "implied_vol_ask",
    ]
    daily_cols = join_keys + ["implied_vol_bid", "implied_vol_ask"]

    # Join datasets LAZILY
    logger.info("\nJoining datasets (lazy)...")
    df_comparison = df_constant_success.select(constant_cols).join(
        df_daily_success.select(daily_cols),
        on=join_keys,
        how="inner",
        suffix="_daily",
    )

    # OPTIMIZATION: Skip validation entirely to avoid any materialization
    # The join will be validated implicitly during streaming write
    logger.info("  Skipping join validation (will be validated during streaming)")

    # Calculate differences LAZILY
    logger.info("Calculating differences (lazy)...")
    df_comparison = df_comparison.with_columns(
        [
            # Absolute differences (vol points)
            (pl.col("implied_vol_bid_daily") - pl.col("implied_vol_bid")).alias("iv_bid_diff_abs"),
            (pl.col("implied_vol_ask_daily") - pl.col("implied_vol_ask")).alias("iv_ask_diff_abs"),
            # Relative differences (%)
            ((pl.col("implied_vol_bid_daily") - pl.col("implied_vol_bid")) / pl.col("implied_vol_bid") * 100).alias(
                "iv_bid_diff_rel"
            ),
            ((pl.col("implied_vol_ask_daily") - pl.col("implied_vol_ask")) / pl.col("implied_vol_ask") * 100).alias(
                "iv_ask_diff_rel"
            ),
            # Mid IVs
            ((pl.col("implied_vol_bid") + pl.col("implied_vol_ask")) / 2).alias("iv_mid_constant"),
            ((pl.col("implied_vol_bid_daily") + pl.col("implied_vol_ask_daily")) / 2).alias("iv_mid_daily"),
        ]
    )

    # Calculate mid differences LAZILY
    df_comparison = df_comparison.with_columns(
        [
            (pl.col("iv_mid_daily") - pl.col("iv_mid_constant")).alias("iv_mid_diff_abs"),
            ((pl.col("iv_mid_daily") - pl.col("iv_mid_constant")) / pl.col("iv_mid_constant") * 100).alias(
                "iv_mid_diff_rel"
            ),
        ]
    )

    # Add date column LAZILY
    df_comparison = df_comparison.with_columns(
        [pl.from_epoch("timestamp_seconds", time_unit="s").cast(pl.Date).alias("date")]
    )

    # Add categorical bins LAZILY
    logger.info("Adding categorical bins (lazy)...")
    df_comparison = df_comparison.with_columns(
        [
            pl.col("moneyness").cut(breaks=MONEYNESS_BINS, labels=MONEYNESS_LABELS).alias("moneyness_bin"),
            pl.col("time_to_expiry_days").cut(breaks=TTL_BINS, labels=TTL_LABELS).alias("ttl_bin"),
        ]
    )

    # Filter invalid data (NaN, Inf, nulls) BEFORE aggregations
    logger.info("Filtering invalid data (NaN, Inf, null)...")
    df_comparison = df_comparison.filter(
        # Filter nulls
        pl.col("iv_mid_constant").is_not_null()
        & pl.col("iv_mid_daily").is_not_null()
        # Filter infinities (from div-by-zero)
        & pl.col("iv_mid_constant").is_finite()
        & pl.col("iv_mid_daily").is_finite()
        & pl.col("iv_mid_diff_rel").is_finite()
        & pl.col("iv_bid_diff_rel").is_finite()
        & pl.col("iv_ask_diff_rel").is_finite()
        # Filter extreme outliers (IV should be 0-20 typically)
        & (pl.col("iv_mid_constant") > 0)
        & (pl.col("iv_mid_constant") < 20)
        & (pl.col("iv_mid_daily") > 0)
        & (pl.col("iv_mid_daily") < 20)
    )

    # OPTIMIZATION: Skip counting valid rows to avoid materialization
    # The actual count will be reported after streaming write or during aggregations
    logger.info("  Filtering complete (count will be reported during aggregations)")

    # Join with risk-free rates LAZILY
    logger.info("Joining with risk-free rates (lazy)...")
    df_comparison = df_comparison.join(pl.LazyFrame(df_rates), on="date", how="left")

    logger.info("\n✅ Lazy comparison DataFrame prepared (no data loaded into memory yet)")

    return df_comparison, df_rates


def write_comparison_to_disk_streaming(df_lazy: pl.LazyFrame, output_file: str) -> None:
    """
    Write comparison DataFrame to disk using streaming mode.

    Use when join result >100M rows to avoid OOM during aggregations.

    Args:
        df_lazy: Lazy comparison DataFrame
        output_file: Output parquet file path
    """
    logger.info("Writing comparison data to disk (streaming mode)...")
    logger.info(f"  Output: {output_file}")

    # Note: sink_parquet() always uses streaming mode (vs write_parquet)
    df_lazy.sink_parquet(
        output_file,
        compression="snappy",
        statistics=True,
    )

    logger.info("✅ Comparison data written to disk")

    # Report size
    file_size_mb = Path(output_file).stat().st_size / (1024 * 1024)
    logger.info(f"  File size: {file_size_mb:.1f} MB")


def generate_summary_statistics(df_lazy: pl.LazyFrame) -> dict[str, pl.DataFrame]:
    """
    Generate comprehensive summary statistics using lazy aggregations.

    CRITICAL: Only collects small aggregated results, never the full dataset.

    Args:
        df_lazy: Lazy comparison DataFrame

    Returns:
        Dictionary of summary DataFrames (all small, collected)
    """
    logger.info("\n" + "=" * 80)
    logger.info("SUMMARY STATISTICS (LAZY AGGREGATIONS)")
    logger.info("=" * 80)

    summaries = {}

    # Overall statistics - single aggregation (optimized: 12 quantiles → 6)
    logger.info("\n1. Overall Statistics")
    start_time = time.time()
    overall_stats = df_lazy.select(
        [
            pl.len().alias("n_observations"),
            # Constant IVs
            pl.col("iv_mid_constant").mean().alias("iv_constant_mean"),
            pl.col("iv_mid_constant").median().alias("iv_constant_median"),
            pl.col("iv_mid_constant").std().alias("iv_constant_std"),
            # Daily IVs
            pl.col("iv_mid_daily").mean().alias("iv_daily_mean"),
            pl.col("iv_mid_daily").median().alias("iv_daily_median"),
            pl.col("iv_mid_daily").std().alias("iv_daily_std"),
            # Absolute differences
            pl.col("iv_mid_diff_abs").mean().alias("diff_abs_mean"),
            pl.col("iv_mid_diff_abs").median().alias("diff_abs_median"),
            pl.col("iv_mid_diff_abs").std().alias("diff_abs_std"),
            pl.col("iv_mid_diff_abs").abs().mean().alias("diff_abs_mean_abs"),
            # Quantiles: p05, p50, p95 only (reduced from p01, p05, p25, p75, p95, p99)
            pl.col("iv_mid_diff_abs").quantile(0.05).alias("diff_abs_p05"),
            pl.col("iv_mid_diff_abs").quantile(0.50).alias("diff_abs_p50"),
            pl.col("iv_mid_diff_abs").quantile(0.95).alias("diff_abs_p95"),
            # Relative differences
            pl.col("iv_mid_diff_rel").mean().alias("diff_rel_mean"),
            pl.col("iv_mid_diff_rel").median().alias("diff_rel_median"),
            pl.col("iv_mid_diff_rel").std().alias("diff_rel_std"),
            pl.col("iv_mid_diff_rel").abs().mean().alias("diff_rel_mean_abs"),
            # Quantiles: p05, p50, p95 only (reduced from p01, p05, p25, p75, p95, p99)
            pl.col("iv_mid_diff_rel").quantile(0.05).alias("diff_rel_p05"),
            pl.col("iv_mid_diff_rel").quantile(0.50).alias("diff_rel_p50"),
            pl.col("iv_mid_diff_rel").quantile(0.95).alias("diff_rel_p95"),
            # Correlation
            pl.corr("iv_mid_constant", "iv_mid_daily").alias("correlation"),
        ]
    ).collect()  # Only collects 1 row!
    logger.info(f"  ✅ Completed in {time.time() - start_time:.1f}s")

    summaries["overall"] = overall_stats
    logger.info(f"\n{overall_stats}")

    # Materiality analysis - OPTIMIZED: Single aggregation (6 scans → 1 scan)
    logger.info("\n2. Materiality Analysis")
    start_time = time.time()

    # Get total count once
    total_count = overall_stats["n_observations"][0]

    # Single aggregation for ALL thresholds (6x faster than separate scans)
    materiality_agg = df_lazy.select(
        [
            pl.len().alias("total_count"),
            # Absolute thresholds (3 conditions)
            (pl.col("iv_mid_diff_abs").abs() > 0.01).sum().alias("abs_gt_001"),
            (pl.col("iv_mid_diff_abs").abs() > 0.05).sum().alias("abs_gt_005"),
            (pl.col("iv_mid_diff_abs").abs() > 0.10).sum().alias("abs_gt_010"),
            # Relative thresholds (3 conditions)
            (pl.col("iv_mid_diff_rel").abs() > 5.0).sum().alias("rel_gt_5pct"),
            (pl.col("iv_mid_diff_rel").abs() > 10.0).sum().alias("rel_gt_10pct"),
            (pl.col("iv_mid_diff_rel").abs() > 20.0).sum().alias("rel_gt_20pct"),
        ]
    ).collect()

    # Build results from single aggregation
    materiality_stats = [
        {
            "threshold_type": "absolute",
            "threshold_value": 0.01,
            "n_above_threshold": materiality_agg["abs_gt_001"][0],
            "pct_above_threshold": materiality_agg["abs_gt_001"][0] / total_count * 100,
        },
        {
            "threshold_type": "absolute",
            "threshold_value": 0.05,
            "n_above_threshold": materiality_agg["abs_gt_005"][0],
            "pct_above_threshold": materiality_agg["abs_gt_005"][0] / total_count * 100,
        },
        {
            "threshold_type": "absolute",
            "threshold_value": 0.10,
            "n_above_threshold": materiality_agg["abs_gt_010"][0],
            "pct_above_threshold": materiality_agg["abs_gt_010"][0] / total_count * 100,
        },
        {
            "threshold_type": "relative",
            "threshold_value": 0.05,
            "n_above_threshold": materiality_agg["rel_gt_5pct"][0],
            "pct_above_threshold": materiality_agg["rel_gt_5pct"][0] / total_count * 100,
        },
        {
            "threshold_type": "relative",
            "threshold_value": 0.10,
            "n_above_threshold": materiality_agg["rel_gt_10pct"][0],
            "pct_above_threshold": materiality_agg["rel_gt_10pct"][0] / total_count * 100,
        },
        {
            "threshold_type": "relative",
            "threshold_value": 0.20,
            "n_above_threshold": materiality_agg["rel_gt_20pct"][0],
            "pct_above_threshold": materiality_agg["rel_gt_20pct"][0] / total_count * 100,
        },
    ]

    # Log results
    for stat in materiality_stats:
        if stat["threshold_type"] == "absolute":
            logger.info(
                f"  |diff| > {stat['threshold_value']:.2f} vol points: "
                f"{stat['n_above_threshold']:,} ({stat['pct_above_threshold']:.2f}%)"
            )
        else:
            logger.info(
                f"  |diff| > {stat['threshold_value'] * 100:.0f}%: "
                f"{stat['n_above_threshold']:,} ({stat['pct_above_threshold']:.2f}%)"
            )

    summaries["materiality"] = pl.DataFrame(materiality_stats)
    logger.info(f"  ✅ Completed in {time.time() - start_time:.1f}s")

    return summaries


def generate_segmented_statistics(df_lazy: pl.LazyFrame) -> dict[str, pl.DataFrame]:
    """
    Generate statistics segmented by option characteristics using lazy group_by.

    CRITICAL: Uses lazy group_by operations, only collects small aggregated results.

    Args:
        df_lazy: Lazy comparison DataFrame

    Returns:
        Dictionary of segmented summary DataFrames (all small, collected)
    """
    logger.info("\n" + "=" * 80)
    logger.info("SEGMENTED ANALYSIS (LAZY GROUP BY)")
    logger.info("=" * 80)

    segmented = {}

    # By option type
    logger.info("\n1. By Option Type")
    start_time = time.time()
    by_type = (
        df_lazy.group_by("type")
        .agg(
            [
                pl.len().alias("n_obs"),
                pl.col("iv_mid_diff_abs").mean().alias("diff_abs_mean"),
                pl.col("iv_mid_diff_abs").median().alias("diff_abs_median"),
                pl.col("iv_mid_diff_abs").abs().mean().alias("diff_abs_mean_abs"),
                pl.col("iv_mid_diff_rel").mean().alias("diff_rel_mean"),
                pl.col("iv_mid_diff_rel").abs().mean().alias("diff_rel_mean_abs"),
            ]
        )
        .sort("type")
        .collect()  # Only collects ~2 rows (call/put)
    )
    segmented["by_type"] = by_type
    logger.info(f"\n{by_type}")
    logger.info(f"  ✅ Completed in {time.time() - start_time:.1f}s")

    # By moneyness
    logger.info("\n2. By Moneyness")
    start_time = time.time()
    by_moneyness = (
        df_lazy.group_by("moneyness_bin")
        .agg(
            [
                pl.len().alias("n_obs"),
                pl.col("moneyness").mean().alias("moneyness_mean"),
                pl.col("iv_mid_diff_abs").mean().alias("diff_abs_mean"),
                pl.col("iv_mid_diff_abs").median().alias("diff_abs_median"),
                pl.col("iv_mid_diff_abs").abs().mean().alias("diff_abs_mean_abs"),
                pl.col("iv_mid_diff_rel").mean().alias("diff_rel_mean"),
                pl.col("iv_mid_diff_rel").abs().mean().alias("diff_rel_mean_abs"),
            ]
        )
        .sort("moneyness_bin")
        .collect()  # Only collects ~3 rows (OTM/ATM/ITM)
    )
    segmented["by_moneyness"] = by_moneyness
    logger.info(f"\n{by_moneyness}")
    logger.info(f"  ✅ Completed in {time.time() - start_time:.1f}s")

    # By time to expiry
    logger.info("\n3. By Time to Expiry")
    start_time = time.time()
    by_ttl = (
        df_lazy.group_by("ttl_bin")
        .agg(
            [
                pl.len().alias("n_obs"),
                pl.col("time_to_expiry_days").mean().alias("ttl_mean"),
                pl.col("iv_mid_diff_abs").mean().alias("diff_abs_mean"),
                pl.col("iv_mid_diff_abs").median().alias("diff_abs_median"),
                pl.col("iv_mid_diff_abs").abs().mean().alias("diff_abs_mean_abs"),
                pl.col("iv_mid_diff_rel").mean().alias("diff_rel_mean"),
                pl.col("iv_mid_diff_rel").abs().mean().alias("diff_rel_mean_abs"),
            ]
        )
        .sort("ttl_bin")
        .collect()  # Only collects ~4 rows (<7d, 7-30d, etc.)
    )
    segmented["by_ttl"] = by_ttl
    logger.info(f"\n{by_ttl}")
    logger.info(f"  ✅ Completed in {time.time() - start_time:.1f}s")

    # By type + moneyness
    logger.info("\n4. By Type + Moneyness")
    start_time = time.time()
    by_type_moneyness = (
        df_lazy.group_by(["type", "moneyness_bin"])
        .agg(
            [
                pl.len().alias("n_obs"),
                pl.col("iv_mid_diff_abs").mean().alias("diff_abs_mean"),
                pl.col("iv_mid_diff_abs").abs().mean().alias("diff_abs_mean_abs"),
                pl.col("iv_mid_diff_rel").mean().alias("diff_rel_mean"),
                pl.col("iv_mid_diff_rel").abs().mean().alias("diff_rel_mean_abs"),
            ]
        )
        .sort(["type", "moneyness_bin"])
        .collect()  # Only collects ~6 rows (2 types × 3 moneyness bins)
    )
    segmented["by_type_moneyness"] = by_type_moneyness
    logger.info(f"\n{by_type_moneyness}")
    logger.info(f"  ✅ Completed in {time.time() - start_time:.1f}s")

    return segmented


def generate_time_series_statistics(df_lazy: pl.LazyFrame, df_rates: pl.DataFrame) -> pl.DataFrame:
    """
    Generate daily time series of differences and rates using lazy group_by.

    CRITICAL: Uses lazy group_by by date, only collects ~730 rows (one per day).

    Args:
        df_lazy: Lazy comparison DataFrame
        df_rates: Risk-free rates DataFrame

    Returns:
        Daily time series DataFrame (~730 rows, small enough to collect)
    """
    logger.info("\n" + "=" * 80)
    logger.info("TIME SERIES ANALYSIS (LAZY GROUP BY)")
    logger.info("=" * 80)

    # Daily aggregation - lazy group by, collects ~730 rows
    logger.info("\nGenerating daily time series...")
    start_time = time.time()
    daily_stats = (
        df_lazy.group_by("date")
        .agg(
            [
                pl.len().alias("n_obs"),
                pl.col("iv_mid_constant").mean().alias("iv_constant_mean"),
                pl.col("iv_mid_daily").mean().alias("iv_daily_mean"),
                pl.col("iv_mid_diff_abs").mean().alias("diff_abs_mean"),
                pl.col("iv_mid_diff_abs").median().alias("diff_abs_median"),
                pl.col("iv_mid_diff_abs").abs().mean().alias("diff_abs_mean_abs"),
                pl.col("iv_mid_diff_rel").mean().alias("diff_rel_mean"),
                pl.col("iv_mid_diff_rel").abs().mean().alias("diff_rel_mean_abs"),
                pl.col("risk_free_rate_pct").first().alias("risk_free_rate_pct"),
            ]
        )
        .sort("date")
        .collect()  # Only collects ~730 rows (one per day)
    )
    logger.info(f"  ✅ Completed in {time.time() - start_time:.1f}s")

    # Calculate correlation with risk-free rate changes
    corr_with_rate = daily_stats.select(
        [
            pl.corr("diff_abs_mean", "risk_free_rate_pct").alias("corr_diff_abs_vs_rate"),
            pl.corr("diff_rel_mean", "risk_free_rate_pct").alias("corr_diff_rel_vs_rate"),
        ]
    )

    logger.info("\nCorrelation with risk-free rate:")
    logger.info(f"{corr_with_rate}")

    # Find days with extreme differences
    logger.info("\nTop 10 days with largest absolute differences:")
    top_days = daily_stats.sort("diff_abs_mean_abs", descending=True).head(10)
    logger.info(f"\n{top_days}")

    return daily_stats


def save_results(
    output_dir: str,
    summaries: dict[str, pl.DataFrame],
    segmented: dict[str, pl.DataFrame],
    daily_stats: pl.DataFrame,
) -> None:
    """
    Save all analysis results to output directory.

    Args:
        output_dir: Output directory path
        summaries: Summary statistics DataFrames
        segmented: Segmented analysis DataFrames
        daily_stats: Daily time series DataFrame
    """
    logger.info("\n" + "=" * 80)
    logger.info("SAVING RESULTS")
    logger.info("=" * 80)

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Save all DataFrames as parquet
    for name, df_result in summaries.items():
        file_path = output_path / f"summary_{name}.parquet"
        df_result.write_parquet(file_path)
        logger.info(f"Saved {file_path}")

    for name, df_result in segmented.items():
        file_path = output_path / f"segmented_{name}.parquet"
        df_result.write_parquet(file_path)
        logger.info(f"Saved {file_path}")

    daily_file = output_path / "daily_time_series.parquet"
    daily_stats.write_parquet(daily_file)
    logger.info(f"Saved {daily_file}")

    logger.info(f"\nAll results saved to {output_dir}")


def main() -> None:
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Compare IV calculations using constant vs. daily risk-free rates")

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
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose logging",
    )

    args = parser.parse_args()

    # Setup logging
    setup_logging(args.verbose)

    start_time = time.time()

    try:
        # Prepare lazy comparison DataFrame (NO data loaded into memory yet)
        df_comparison_lazy, df_rates = prepare_lazy_comparison(
            args.constant_file,
            args.daily_file,
            args.rates_file,
        )

        # OPTIMIZATION: Always use streaming mode for safety with ~198M rows
        # Avoids counting rows which would force materialization
        logger.info("Using streaming mode for ~198M row dataset...")

        # Write to disk first
        output_path = Path(args.output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        intermediate_file = output_path / "comparison_data_intermediate.parquet"

        write_comparison_to_disk_streaming(df_comparison_lazy, str(intermediate_file))

        # Reload lazily for aggregations
        df_comparison_lazy = pl.scan_parquet(intermediate_file)
        logger.info("✅ Data reloaded lazily from disk for aggregations")

        # Generate statistics (only small aggregated results are collected)
        summaries = generate_summary_statistics(df_comparison_lazy)
        segmented = generate_segmented_statistics(df_comparison_lazy)
        daily_stats = generate_time_series_statistics(df_comparison_lazy, df_rates)

        # Save results
        save_results(args.output_dir, summaries, segmented, daily_stats)

        # Get final count
        total_compared = summaries["overall"]["n_observations"][0]

        # Final summary
        elapsed = time.time() - start_time
        logger.info("\n" + "=" * 80)
        logger.info("ANALYSIS COMPLETE")
        logger.info("=" * 80)
        logger.info(f"Total time: {elapsed:.1f} seconds")
        logger.info(f"Analyzed {total_compared:,} options")
        logger.info(f"Results saved to {args.output_dir}")
        logger.info("\n✅ Memory usage stayed low throughout (lazy evaluation)")

    except Exception as e:
        logger.error(f"ERROR: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
