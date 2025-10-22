#!/usr/bin/env python3

"""
Compare implied volatility calculations using constant vs. daily risk-free rates.

FIXED VERSION: Joins full datasets first, then filters for rows where both calculations succeeded.
This ensures we're comparing the same option quotes with different risk-free rates.
"""

import argparse
import logging
import sys
import time
from pathlib import Path

import polars as pl

# Constants
DEFAULT_CONSTANT_IV_FILE = "research/tardis/data/consolidated/quotes_1s_atm_short_dated_with_iv.parquet"
DEFAULT_DAILY_IV_FILE = "research/tardis/data/consolidated/quotes_1s_atm_short_dated_with_iv_daily_rates.parquet"
DEFAULT_RATES_FILE = "research/risk_free_rate/data/blended_lending_rates_2023_2025.parquet"
DEFAULT_OUTPUT_DIR = "research/tardis/analysis/output"

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


def compare_iv_calculations(
    constant_file: str,
    daily_file: str,
    rates_file: str,
    output_dir: str,
    test_rows: int | None = None,
) -> None:
    """
    Compare IV calculations using constant vs. daily risk-free rates.

    This fixed version:
    1. Joins the full datasets (including failed calculations)
    2. Filters for rows where BOTH calculations succeeded
    3. Compares only these matching quotes
    """
    logger.info("=" * 80)
    logger.info("IV COMPARISON ANALYSIS (FIXED VERSION)")
    logger.info("=" * 80)

    # Check input files exist
    for file_path in [constant_file, daily_file, rates_file]:
        if not Path(file_path).exists():
            raise FileNotFoundError(f"File not found: {file_path}")

    # Load datasets lazily
    logger.info("Loading datasets...")
    const_df = pl.scan_parquet(constant_file)
    daily_df = pl.scan_parquet(daily_file)

    # Limit rows for testing if requested
    if test_rows:
        logger.info(f"TEST MODE: Using first {test_rows:,} rows")
        const_df = const_df.head(test_rows)
        daily_df = daily_df.head(test_rows)

    # Count total rows
    const_total = const_df.select(pl.len()).collect().item()
    daily_total = daily_df.select(pl.len()).collect().item()
    logger.info(f"  Constant file: {const_total:,} rows")
    logger.info(f"  Daily file: {daily_total:,} rows")

    # Define join keys
    join_keys = [
        "timestamp_seconds",
        "symbol",
        "exchange",
        "type",
        "strike_price",
        "expiry_timestamp",
    ]

    # Select columns we need from each dataset
    const_cols = join_keys + [
        "spot_price",
        "moneyness",
        "time_to_expiry_days",
        "implied_vol_bid",
        "implied_vol_ask",
        "iv_calc_status",
    ]

    daily_cols = join_keys + [
        "implied_vol_bid",
        "implied_vol_ask",
        "iv_calc_status",
    ]

    # Join the full datasets
    logger.info("\nJoining datasets...")
    df_joined = const_df.select(const_cols).join(
        daily_df.select(daily_cols),
        on=join_keys,
        how="inner",
        suffix="_daily",
    )

    # Filter for rows where BOTH calculations succeeded
    logger.info("Filtering for mutually successful calculations...")
    df_both_success = df_joined.filter(
        (pl.col("iv_calc_status") == "success")
        & (pl.col("iv_calc_status_daily") == "success")
    )

    # Calculate differences
    logger.info("Calculating differences...")
    df_comparison = df_both_success.with_columns([
        # Absolute differences
        (pl.col("implied_vol_bid_daily") - pl.col("implied_vol_bid")).alias("iv_bid_diff_abs"),
        (pl.col("implied_vol_ask_daily") - pl.col("implied_vol_ask")).alias("iv_ask_diff_abs"),
        # Mid IV
        ((pl.col("implied_vol_bid") + pl.col("implied_vol_ask")) / 2).alias("iv_mid_constant"),
        ((pl.col("implied_vol_bid_daily") + pl.col("implied_vol_ask_daily")) / 2).alias("iv_mid_daily"),
    ])

    df_comparison = df_comparison.with_columns([
        # Mid difference
        (pl.col("iv_mid_daily") - pl.col("iv_mid_constant")).alias("iv_mid_diff_abs"),
        # Relative difference (%)
        ((pl.col("iv_mid_daily") - pl.col("iv_mid_constant")) / pl.col("iv_mid_constant") * 100).alias("iv_mid_diff_rel"),
    ])

    # Filter invalid values
    logger.info("Filtering invalid values...")
    df_comparison = df_comparison.filter(
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

    # Write comparison data to disk
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    comparison_file = output_path / "comparison_final.parquet"

    logger.info(f"\nWriting comparison data to {comparison_file}...")
    df_comparison.sink_parquet(str(comparison_file), compression="snappy", statistics=True)

    # Load for statistics (now safe since filtered)
    logger.info("\nGenerating statistics...")
    df_stats = pl.scan_parquet(comparison_file)

    # Overall statistics
    overall_stats = df_stats.select([
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
        # Min/Max
        pl.col("iv_mid_diff_abs").min().alias("diff_abs_min"),
        pl.col("iv_mid_diff_abs").max().alias("diff_abs_max"),
        # Correlation
        pl.corr("iv_mid_constant", "iv_mid_daily").alias("correlation"),
    ]).collect()

    logger.info("\n" + "=" * 80)
    logger.info("RESULTS SUMMARY")
    logger.info("=" * 80)

    stats = overall_stats.row(0, named=True)
    logger.info(f"Quotes analyzed: {stats['n_observations']:,}")
    logger.info(f"\nAverage IV:")
    logger.info(f"  Constant rate: {stats['iv_constant_mean']:.4f}")
    logger.info(f"  Daily rates: {stats['iv_daily_mean']:.4f}")
    logger.info(f"\nDifferences (Daily - Constant):")
    logger.info(f"  Mean: {stats['diff_abs_mean']:.6f} ({stats['diff_rel_mean_pct']:.4f}%)")
    logger.info(f"  Median: {stats['diff_abs_median']:.6f} ({stats['diff_rel_median_pct']:.4f}%)")
    logger.info(f"  Std Dev: {stats['diff_abs_std']:.6f} ({stats['diff_rel_std_pct']:.4f}%)")
    logger.info(f"  Range: [{stats['diff_abs_min']:.6f}, {stats['diff_abs_max']:.6f}]")
    logger.info(f"\nCorrelation: {stats['correlation']:.6f}")

    # Materiality analysis
    logger.info("\nMateriality Analysis:")
    materiality = df_stats.select([
        (pl.col("iv_mid_diff_abs").abs() > 0.01).sum().alias("above_1bp"),
        (pl.col("iv_mid_diff_abs").abs() > 0.05).sum().alias("above_5bp"),
        (pl.col("iv_mid_diff_abs").abs() > 0.10).sum().alias("above_10bp"),
        pl.len().alias("total"),
    ]).collect().row(0, named=True)

    logger.info(f"  |Diff| > 0.01: {materiality['above_1bp']:,} ({materiality['above_1bp']/materiality['total']*100:.2f}%)")
    logger.info(f"  |Diff| > 0.05: {materiality['above_5bp']:,} ({materiality['above_5bp']/materiality['total']*100:.2f}%)")
    logger.info(f"  |Diff| > 0.10: {materiality['above_10bp']:,} ({materiality['above_10bp']/materiality['total']*100:.2f}%)")

    # Save summary
    summary_file = output_path / "summary_stats.parquet"
    overall_stats.write_parquet(summary_file)

    logger.info(f"\n✅ Analysis complete! Results saved to {output_dir}")
    logger.info(f"  - Comparison data: {comparison_file}")
    logger.info(f"  - Summary statistics: {summary_file}")


def main() -> None:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Compare IV calculations using constant vs. daily risk-free rates (FIXED VERSION)"
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
        "--test-rows",
        type=int,
        help="Limit to first N rows for testing",
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
        compare_iv_calculations(
            args.constant_file,
            args.daily_file,
            args.rates_file,
            args.output_dir,
            args.test_rows,
        )

        elapsed = time.time() - start_time
        logger.info(f"\nTotal runtime: {elapsed:.1f} seconds")

    except Exception as e:
        logger.error(f"ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()