#!/usr/bin/env python3
"""
Test whether rolling window calculations on forward-filled funding rates are problematic.

This script validates the critique that:
1. Funding rates update every 8 hours but are forward-filled to 1-second granularity
2. Computing rolling stats on forward-filled data creates misleading metrics
3. The computation is wasteful on 63M rows with mostly duplicate values

We'll test:
- Actual update frequency of funding rates
- Impact of forward-fill on rolling statistics
- Computational cost comparison
- Alternative approaches
"""

import logging
import time
from pathlib import Path

import numpy as np
import polars as pl

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def analyze_funding_rate_update_pattern(df: pl.DataFrame) -> dict:
    """Analyze how frequently funding rates actually change.

    Args:
        df: DataFrame with funding rate data

    Returns:
        Dictionary with update statistics
    """
    logger.info("Analyzing funding rate update pattern...")

    # Check how many unique values exist vs total rows
    total_rows = len(df)
    unique_funding_rates = df["funding_rate"].n_unique()

    # Find actual changes (where funding_rate differs from previous)
    changes = df.with_columns([
        (pl.col("funding_rate") != pl.col("funding_rate").shift(1)).alias("is_change")
    ])
    num_changes = changes["is_change"].sum()

    # Calculate time between changes
    change_rows = changes.filter(pl.col("is_change") == True)
    if len(change_rows) > 1:
        time_diffs = change_rows["timestamp_seconds"].diff().drop_nulls()
        mean_gap = time_diffs.mean()
        median_gap = time_diffs.median()
        min_gap = time_diffs.min()
        max_gap = time_diffs.max()
    else:
        mean_gap = median_gap = min_gap = max_gap = None

    stats = {
        "total_rows": total_rows,
        "unique_values": unique_funding_rates,
        "num_changes": num_changes,
        "pct_duplicate_rows": (total_rows - num_changes) / total_rows * 100,
        "mean_gap_seconds": mean_gap,
        "median_gap_seconds": median_gap,
        "min_gap_seconds": min_gap,
        "max_gap_seconds": max_gap,
    }

    logger.info(f"Total rows: {total_rows:,}")
    logger.info(f"Unique funding rate values: {unique_funding_rates:,}")
    logger.info(f"Actual changes: {num_changes:,}")
    logger.info(f"Duplicate rows (forward-filled): {total_rows - num_changes:,} ({stats['pct_duplicate_rows']:.2f}%)")
    if mean_gap:
        logger.info(f"Mean gap between changes: {mean_gap:.0f}s ({mean_gap/3600:.1f} hours)")
        logger.info(f"Median gap: {median_gap:.0f}s ({median_gap/3600:.1f} hours)")
        logger.info(f"Min gap: {min_gap:.0f}s, Max gap: {max_gap:.0f}s")

    return stats


def compare_rolling_stats_approaches(df: pl.DataFrame, window_days: int = 30) -> dict:
    """Compare rolling statistics on forward-filled vs actual update points.

    Args:
        df: DataFrame with funding rate data
        window_days: Rolling window size in days

    Returns:
        Dictionary with comparison results
    """
    window_seconds = window_days * 24 * 3600
    logger.info(f"\nComparing rolling statistics (window={window_days} days = {window_seconds:,} seconds)...")

    # Approach 1: Rolling stats on forward-filled data (current approach)
    logger.info("Approach 1: Rolling stats on forward-filled 1s data...")
    start = time.time()

    df_approach1 = df.with_columns([
        pl.col("funding_rate")
          .rolling_mean(window_size=window_seconds, min_periods=1)
          .alias("rolling_mean_30d"),
        pl.col("funding_rate")
          .rolling_std(window_size=window_seconds, min_periods=1)
          .alias("rolling_std_30d"),
    ]).with_columns([
        ((pl.col("funding_rate") - pl.col("rolling_mean_30d")) / pl.col("rolling_std_30d"))
          .alias("funding_rate_zscore_30d")
    ])

    time_approach1 = time.time() - start
    logger.info(f"  Time: {time_approach1:.2f}s")

    # Approach 2: Rolling stats on actual changes only, then join back
    logger.info("Approach 2: Rolling stats on changes only, then join...")
    start = time.time()

    # Extract only rows where funding rate changed
    df_changes = df.with_columns([
        (pl.col("funding_rate") != pl.col("funding_rate").shift(1)).fill_null(True).alias("is_change")
    ]).filter(pl.col("is_change") == True)

    logger.info(f"  Changes only: {len(df_changes):,} rows (from {len(df):,})")

    # Compute rolling stats on changes only
    df_changes = df_changes.with_columns([
        pl.col("funding_rate")
          .rolling_mean(window_size=window_seconds, min_periods=1)
          .alias("rolling_mean_30d"),
        pl.col("funding_rate")
          .rolling_std(window_size=window_seconds, min_periods=1)
          .alias("rolling_std_30d"),
    ]).select(["timestamp_seconds", "rolling_mean_30d", "rolling_std_30d"])

    # Join back to full dataset
    df_approach2 = df.join_asof(
        df_changes,
        on="timestamp_seconds",
        strategy="backward"
    ).with_columns([
        ((pl.col("funding_rate") - pl.col("rolling_mean_30d")) / pl.col("rolling_std_30d"))
          .alias("funding_rate_zscore_30d")
    ])

    time_approach2 = time.time() - start
    logger.info(f"  Time: {time_approach2:.2f}s")
    logger.info(f"  Speedup: {time_approach1 / time_approach2:.2f}x")

    # Compare results
    logger.info("\nComparing results...")

    # Sample 1000 random rows to compare
    sample_indices = np.random.choice(len(df), min(1000, len(df)), replace=False)
    sample1 = df_approach1.select(["timestamp_seconds", "funding_rate", "rolling_mean_30d", "rolling_std_30d", "funding_rate_zscore_30d"])[sample_indices]
    sample2 = df_approach2.select(["timestamp_seconds", "funding_rate", "rolling_mean_30d", "rolling_std_30d", "funding_rate_zscore_30d"])[sample_indices]

    # Calculate differences
    diff = sample1.with_columns([
        (pl.col("rolling_mean_30d") - sample2["rolling_mean_30d"]).abs().alias("mean_diff"),
        (pl.col("rolling_std_30d") - sample2["rolling_std_30d"]).abs().alias("std_diff"),
        (pl.col("funding_rate_zscore_30d") - sample2["funding_rate_zscore_30d"]).abs().alias("zscore_diff"),
    ])

    max_mean_diff = diff["mean_diff"].max()
    max_std_diff = diff["std_diff"].max()
    max_zscore_diff = diff["zscore_diff"].max()

    logger.info(f"Max difference in rolling mean: {max_mean_diff}")
    logger.info(f"Max difference in rolling std: {max_std_diff}")
    logger.info(f"Max difference in z-score: {max_zscore_diff}")

    # Show example of misleading statistics
    logger.info("\n=== Example: Impact of Forward-Fill on Rolling Stats ===")

    # Find a period with no actual changes (constant forward-filled values)
    constant_period = df.with_columns([
        (pl.col("funding_rate") != pl.col("funding_rate").shift(1)).fill_null(True).alias("is_change")
    ])

    # Find longest run of no changes
    constant_period = constant_period.with_columns([
        pl.col("is_change").cum_sum().alias("change_group")
    ])

    group_sizes = constant_period.group_by("change_group").agg([
        pl.len().alias("run_length"),
        pl.col("timestamp_seconds").min().alias("start_ts"),
        pl.col("funding_rate").first().alias("funding_rate"),
    ]).sort("run_length", descending=True)

    longest_run = group_sizes[0]
    logger.info(f"Longest constant run: {longest_run['run_length'][0]:,} seconds ({longest_run['run_length'][0]/3600:.1f} hours)")
    logger.info(f"Funding rate value: {longest_run['funding_rate'][0]}")

    # Show rolling stats during this constant period
    start_ts = longest_run["start_ts"][0]
    example = df_approach1.filter(
        (pl.col("timestamp_seconds") >= start_ts) &
        (pl.col("timestamp_seconds") < start_ts + min(3600, longest_run["run_length"][0]))
    ).select(["timestamp_seconds", "funding_rate", "rolling_mean_30d", "rolling_std_30d", "funding_rate_zscore_30d"])

    logger.info(f"\nExample rows from constant period (first 10):")
    logger.info(example.head(10))

    return {
        "time_approach1": time_approach1,
        "time_approach2": time_approach2,
        "speedup": time_approach1 / time_approach2,
        "max_mean_diff": max_mean_diff,
        "max_std_diff": max_std_diff,
        "max_zscore_diff": max_zscore_diff,
        "rows_approach1": len(df_approach1),
        "rows_approach2": len(df_changes),
    }


def test_critique_validity(sample_size: int = 1_000_000) -> None:
    """Test whether the critique is valid.

    Args:
        sample_size: Number of rows to test (use smaller for faster testing)
    """
    data_file = Path("/home/ubuntu/Polymarket/research/tardis/data/consolidated/binance_funding_rates_1s_consolidated.parquet")

    if not data_file.exists():
        logger.error(f"Data file not found: {data_file}")
        return

    logger.info(f"Loading data from {data_file}")
    logger.info(f"Sample size: {sample_size:,} rows")

    # Load sample of data
    df = pl.scan_parquet(data_file).head(sample_size).collect()

    logger.info(f"Loaded {len(df):,} rows")
    logger.info(f"Columns: {df.columns}")

    # Convert timestamps if needed
    if "timestamp_seconds" not in df.columns and "timestamp" in df.columns:
        df = df.with_columns([
            (pl.col("timestamp") // 1_000_000).alias("timestamp_seconds")
        ])

    logger.info(f"Date range: {df['timestamp_seconds'].min()} to {df['timestamp_seconds'].max()}")

    # Test 1: Analyze update pattern
    logger.info("\n" + "="*80)
    logger.info("TEST 1: Analyze Funding Rate Update Pattern")
    logger.info("="*80)
    update_stats = analyze_funding_rate_update_pattern(df)

    # Test 2: Compare rolling stats approaches
    logger.info("\n" + "="*80)
    logger.info("TEST 2: Compare Rolling Statistics Approaches")
    logger.info("="*80)
    comparison_stats = compare_rolling_stats_approaches(df, window_days=30)

    # Generate verdict
    logger.info("\n" + "="*80)
    logger.info("VERDICT ON CRITIQUE")
    logger.info("="*80)

    critique_points = [
        {
            "claim": "Funding rates update only every 8 hours",
            "verdict": None,
            "evidence": None,
        },
        {
            "claim": "Forward-fill creates mostly duplicate values",
            "verdict": None,
            "evidence": None,
        },
        {
            "claim": "Rolling stats on forward-filled data is computationally wasteful",
            "verdict": None,
            "evidence": None,
        },
        {
            "claim": "Rolling stats on forward-filled data creates misleading metrics",
            "verdict": None,
            "evidence": None,
        },
    ]

    # Point 1: Update frequency
    if update_stats["median_gap_seconds"]:
        hours_between_updates = update_stats["median_gap_seconds"] / 3600
        critique_points[0]["verdict"] = "TRUE" if 7 <= hours_between_updates <= 9 else "PARTIALLY TRUE"
        critique_points[0]["evidence"] = f"Median gap: {hours_between_updates:.1f} hours"

    # Point 2: Duplicate values
    pct_duplicates = update_stats["pct_duplicate_rows"]
    critique_points[1]["verdict"] = "TRUE" if pct_duplicates > 90 else "PARTIALLY TRUE"
    critique_points[1]["evidence"] = f"{pct_duplicates:.1f}% of rows are forward-filled duplicates"

    # Point 3: Computational waste
    speedup = comparison_stats["speedup"]
    critique_points[2]["verdict"] = "TRUE" if speedup > 2 else "FALSE"
    critique_points[2]["evidence"] = f"Alternative approach is {speedup:.1f}x faster"

    # Point 4: Misleading metrics
    # Check if rolling stats differ significantly between approaches
    max_diff = max(
        comparison_stats["max_mean_diff"] or 0,
        comparison_stats["max_std_diff"] or 0,
        comparison_stats["max_zscore_diff"] or 0
    )
    critique_points[3]["verdict"] = "TRUE" if max_diff > 0.01 else "FALSE"
    critique_points[3]["evidence"] = f"Max difference in metrics: {max_diff:.6f}"

    for i, point in enumerate(critique_points, 1):
        logger.info(f"\n{i}. {point['claim']}")
        logger.info(f"   Verdict: {point['verdict']}")
        logger.info(f"   Evidence: {point['evidence']}")

    # Overall assessment
    true_count = sum(1 for p in critique_points if p["verdict"] == "TRUE")
    logger.info(f"\n{'='*80}")
    logger.info(f"OVERALL ASSESSMENT: {true_count}/4 critique points are TRUE")

    if true_count >= 3:
        logger.info("✅ The critique is VALID - rolling windows on forward-filled data are problematic")
        logger.info("\nRECOMMENDATION:")
        logger.info("1. Compute rolling stats on actual funding rate updates only (every 8 hours)")
        logger.info("2. Use join_asof to map rolling stats back to 1-second data")
        logger.info(f"3. Expected speedup: {speedup:.1f}x faster")
        logger.info(f"4. Reduces computation from {comparison_stats['rows_approach1']:,} to {comparison_stats['rows_approach2']:,} rows")
    elif true_count >= 2:
        logger.info("⚠️  The critique is PARTIALLY VALID - some concerns are real but impact is limited")
    else:
        logger.info("❌ The critique is NOT VALID - rolling windows on forward-filled data are acceptable")

    logger.info("="*80)


def main() -> None:
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Test whether rolling window calculations on forward-filled funding rates are problematic"
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=1_000_000,
        help="Number of rows to test (default: 1M, use 63M for full test)"
    )

    args = parser.parse_args()

    test_critique_validity(sample_size=args.sample_size)


if __name__ == "__main__":
    main()
