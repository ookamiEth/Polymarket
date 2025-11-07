#!/usr/bin/env python3
"""
Analyze Clustering Patterns of IV Failures

Investigate whether the 922K IV failures occur in:
- Time clusters (specific dates/hours with high failure rates)
- Strike clusters (specific strikes consistently fail)
- Market condition clusters (high volatility periods, specific spot price ranges)
"""

import polars as pl
import logging
from datetime import datetime

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

DATA_FILE = "/home/ubuntu/Polymarket/research/tardis/data/consolidated/btc_options_atm_shortdated_with_iv_2023_2025.parquet"

def analyze_clustering():
    logger.info("="*80)
    logger.info("IV FAILURE CLUSTERING ANALYSIS")
    logger.info("="*80)

    # Load data
    logger.info("\n📂 Loading data...")
    df = pl.read_parquet(DATA_FILE)

    # Filter to only failures (< 12 hours to expiry)
    failures = df.filter(
        pl.col("implied_vol_bid").is_null() &
        (pl.col("time_to_expiry_days") < 0.5)
    )

    total_failures = len(failures)
    logger.info(f"Total IV failures: {total_failures:,}")

    # Add datetime columns
    failures = failures.with_columns([
        pl.from_epoch(pl.col("timestamp_seconds"), time_unit="s").alias("datetime"),
    ]).with_columns([
        pl.col("datetime").dt.date().alias("date"),
        pl.col("datetime").dt.hour().alias("hour"),
        pl.col("datetime").dt.minute().alias("minute"),
    ])

    # =========================================================================
    # TEMPORAL CLUSTERING ANALYSIS
    # =========================================================================
    logger.info("\n" + "="*80)
    logger.info("TEMPORAL CLUSTERING ANALYSIS")
    logger.info("="*80)

    # Daily distribution
    daily_failures = failures.group_by("date").agg([
        pl.len().alias("failure_count")
    ]).sort("failure_count", descending=True)

    top_20_days = daily_failures.head(20)

    logger.info("\nTop 20 days with most IV failures:")
    logger.info(f"{'Date':<12} {'Failures':>10} {'% of Total':>12}")
    logger.info("-" * 40)
    for row in top_20_days.iter_rows(named=True):
        date = row['date']
        count = row['failure_count']
        pct = 100 * count / total_failures
        logger.info(f"{date} {count:>10,} {pct:>11.2f}%")

    # Calculate concentration
    top_10_days_total = daily_failures.head(10)["failure_count"].sum()
    logger.info(f"\n📊 Top 10 days account for: {top_10_days_total:,} failures ({100*top_10_days_total/total_failures:.1f}%)")

    # Hourly distribution
    hourly_failures = failures.group_by("hour").agg([
        pl.len().alias("failure_count")
    ]).sort("hour")

    logger.info("\n\nFailures by Hour of Day (UTC):")
    logger.info(f"{'Hour':>4} {'Failures':>10} {'% of Total':>12}")
    logger.info("-" * 30)
    for row in hourly_failures.iter_rows(named=True):
        hour = row['hour']
        count = row['failure_count']
        pct = 100 * count / total_failures
        logger.info(f"{hour:>4} {count:>10,} {pct:>11.2f}%")

    # =========================================================================
    # STRIKE CLUSTERING ANALYSIS
    # =========================================================================
    logger.info("\n" + "="*80)
    logger.info("STRIKE CLUSTERING ANALYSIS")
    logger.info("="*80)

    # Group by strike price
    strike_failures = failures.group_by("strike_price").agg([
        pl.len().alias("failure_count")
    ]).sort("failure_count", descending=True).head(20)

    logger.info("\nTop 20 strikes with most IV failures:")
    logger.info(f"{'Strike Price':>12} {'Failures':>10} {'% of Total':>12}")
    logger.info("-" * 40)
    for row in strike_failures.iter_rows(named=True):
        strike = row['strike_price']
        count = row['failure_count']
        pct = 100 * count / total_failures
        logger.info(f"{strike:>12,.0f} {count:>10,} {pct:>11.2f}%")

    # Unique strikes with failures
    unique_strikes = failures["strike_price"].n_unique()
    logger.info(f"\n📊 Number of unique strikes with failures: {unique_strikes:,}")

    # =========================================================================
    # MONEYNESS + TIME CLUSTERING
    # =========================================================================
    logger.info("\n" + "="*80)
    logger.info("MONEYNESS + TIME TO EXPIRY CLUSTERING")
    logger.info("="*80)

    # Create finer-grained TTL bins for < 12 hours
    ttl_moneyness = failures.with_columns([
        pl.when(pl.col("time_to_expiry_days") < 0.042).then(pl.lit("< 1 hour"))
        .when(pl.col("time_to_expiry_days") < 0.125).then(pl.lit("1-3 hours"))
        .when(pl.col("time_to_expiry_days") < 0.25).then(pl.lit("3-6 hours"))
        .otherwise(pl.lit("6-12 hours"))
        .alias("ttl_fine_bin")
    ]).with_columns([
        pl.when(pl.col("moneyness") < 0.99).then(pl.lit("OTM"))
        .when(pl.col("moneyness") < 1.01).then(pl.lit("ATM"))
        .otherwise(pl.lit("ITM"))
        .alias("moneyness_bin")
    ]).group_by(["ttl_fine_bin", "moneyness_bin"]).agg([
        pl.len().alias("failure_count")
    ]).sort("failure_count", descending=True)

    logger.info("\nFailures by Time to Expiry × Moneyness:")
    logger.info(f"{'Time to Expiry':<15} {'Moneyness':>10} {'Failures':>12} {'%':>8}")
    logger.info("-" * 50)
    for row in ttl_moneyness.iter_rows(named=True):
        ttl = row['ttl_fine_bin']
        money = row['moneyness_bin']
        count = row['failure_count']
        pct = 100 * count / total_failures
        logger.info(f"{ttl:<15} {money:>10} {count:>12,} {pct:>7.2f}%")

    # =========================================================================
    # CONSECUTIVE FAILURES (CLUSTERING IN TIME)
    # =========================================================================
    logger.info("\n" + "="*80)
    logger.info("CONSECUTIVE FAILURE ANALYSIS")
    logger.info("="*80)

    # Sort by timestamp and calculate time gaps
    failures_sorted = failures.sort("timestamp_seconds").with_columns([
        pl.col("timestamp_seconds").diff().alias("time_gap_seconds")
    ])

    # Analyze gaps between failures
    gap_stats = failures_sorted.filter(pl.col("time_gap_seconds").is_not_null()).select([
        pl.col("time_gap_seconds").min().alias("min_gap"),
        pl.col("time_gap_seconds").quantile(0.25).alias("p25_gap"),
        pl.col("time_gap_seconds").median().alias("median_gap"),
        pl.col("time_gap_seconds").quantile(0.75).alias("p75_gap"),
        pl.col("time_gap_seconds").max().alias("max_gap"),
        pl.col("time_gap_seconds").mean().alias("mean_gap"),
    ])

    logger.info("\nTime gaps between consecutive IV failures:")
    for row in gap_stats.iter_rows(named=True):
        logger.info(f"  Minimum gap:    {row['min_gap']:>10.1f} seconds")
        logger.info(f"  25th percentile: {row['p25_gap']:>10.1f} seconds")
        logger.info(f"  Median gap:      {row['median_gap']:>10.1f} seconds ({row['median_gap']/60:.1f} minutes)")
        logger.info(f"  75th percentile: {row['p75_gap']:>10.1f} seconds")
        logger.info(f"  Mean gap:        {row['mean_gap']:>10.1f} seconds ({row['mean_gap']/60:.1f} minutes)")
        logger.info(f"  Maximum gap:     {row['max_gap']:>10.1f} seconds ({row['max_gap']/3600:.1f} hours)")

    # Count very close failures (< 60 seconds apart)
    close_failures = failures_sorted.filter(
        pl.col("time_gap_seconds").is_not_null() &
        (pl.col("time_gap_seconds") < 60)
    ).height

    logger.info(f"\n📊 Failures < 60 seconds apart: {close_failures:,} ({100*close_failures/total_failures:.1f}%)")

    # =========================================================================
    # SPOT PRICE CLUSTERING
    # =========================================================================
    logger.info("\n" + "="*80)
    logger.info("SPOT PRICE CLUSTERING")
    logger.info("="*80)

    # Analyze spot price ranges where failures occur
    spot_stats = failures.select([
        pl.col("spot_price").min().alias("min_spot"),
        pl.col("spot_price").max().alias("max_spot"),
        pl.col("spot_price").mean().alias("mean_spot"),
        pl.col("spot_price").median().alias("median_spot"),
    ])

    logger.info("\nSpot price statistics for failures:")
    for row in spot_stats.iter_rows(named=True):
        logger.info(f"  Minimum:  ${row['min_spot']:>10,.2f}")
        logger.info(f"  Mean:     ${row['mean_spot']:>10,.2f}")
        logger.info(f"  Median:   ${row['median_spot']:>10,.2f}")
        logger.info(f"  Maximum:  ${row['max_spot']:>10,.2f}")

    # Group by spot price bins
    spot_bins = failures.with_columns([
        (pl.col("spot_price") / 5000).floor() * 5000
        .alias("spot_bin")
    ]).group_by("spot_bin").agg([
        pl.len().alias("failure_count")
    ]).sort("failure_count", descending=True).head(10)

    logger.info("\nTop 10 spot price ranges ($5K bins) with most failures:")
    logger.info(f"{'Spot Range':>20} {'Failures':>10} {'%':>8}")
    logger.info("-" * 42)
    for row in spot_bins.iter_rows(named=True):
        bin_start = row['spot_bin']
        count = row['failure_count']
        pct = 100 * count / total_failures
        logger.info(f"${bin_start:>8,.0f}-${bin_start+5000:>8,.0f} {count:>10,} {pct:>7.2f}%")

    # =========================================================================
    # SUMMARY: CLUSTERING VERDICT
    # =========================================================================
    logger.info("\n" + "="*80)
    logger.info("CLUSTERING VERDICT")
    logger.info("="*80)

    # Calculate Gini coefficient for temporal concentration
    daily_sorted = daily_failures.sort("failure_count", descending=True)
    cumulative_pct = (daily_sorted["failure_count"].cum_sum() / total_failures * 100).to_list()

    days_for_50pct = next(i+1 for i, v in enumerate(cumulative_pct) if v >= 50)
    days_for_80pct = next(i+1 for i, v in enumerate(cumulative_pct) if v >= 80)
    total_days = len(daily_sorted)

    logger.info(f"\n📊 Temporal Concentration Metrics:")
    logger.info(f"  50% of failures occur in: {days_for_50pct} days ({100*days_for_50pct/total_days:.1f}% of days)")
    logger.info(f"  80% of failures occur in: {days_for_80pct} days ({100*days_for_80pct/total_days:.1f}% of days)")
    logger.info(f"  Total days with failures: {total_days}")

    if days_for_50pct < total_days * 0.2:
        logger.info("\n✅ CLUSTERED: Failures are highly concentrated in specific days")
    elif days_for_50pct < total_days * 0.4:
        logger.info("\n⚠️  MODERATELY CLUSTERED: Some concentration but fairly distributed")
    else:
        logger.info("\n❌ DISPERSED: Failures are evenly distributed across time")

    logger.info("\n" + "="*80)
    logger.info("ANALYSIS COMPLETE")
    logger.info("="*80)

if __name__ == "__main__":
    try:
        analyze_clustering()
    except Exception as e:
        logger.error(f"Analysis failed: {e}", exc_info=True)
        raise
