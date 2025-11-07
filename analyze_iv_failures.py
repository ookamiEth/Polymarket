#!/usr/bin/env python3
"""
Deep Dive Analysis of 2.4% IV Calculation Failures

Investigates the failed IV calculations in the regenerated baseline file:
- Characterizes failures by option type (call/put)
- Analyzes paired options (do both sides fail?)
- Identifies root causes
- Proposes data cleaning strategies
"""

import polars as pl
import logging
from pathlib import Path
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# File paths
DATA_FILE = "/home/ubuntu/Polymarket/research/tardis/data/consolidated/btc_options_atm_shortdated_with_iv_2023_2025.parquet"
OUTPUT_DIR = Path("/home/ubuntu/Polymarket/logs/iv_failure_analysis")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def analyze_failures():
    """Main analysis function."""

    logger.info("="*80)
    logger.info("IV FAILURE ANALYSIS: 2.4% Failed Calculations")
    logger.info("="*80)
    logger.info(f"Start time: {datetime.now()}")
    logger.info(f"Input file: {DATA_FILE}")

    # Load data
    logger.info("\n📂 Loading data...")
    df = pl.read_parquet(DATA_FILE)

    total_rows = len(df)
    logger.info(f"Total rows: {total_rows:,}")
    logger.info(f"Columns: {df.columns}")

    # =========================================================================
    # PART 1: BASIC FAILURE STATISTICS
    # =========================================================================
    logger.info("\n" + "="*80)
    logger.info("PART 1: BASIC FAILURE STATISTICS")
    logger.info("="*80)

    # Count NaN values
    bid_iv_null = df.filter(pl.col("implied_vol_bid").is_null()).height
    ask_iv_null = df.filter(pl.col("implied_vol_ask").is_null()).height
    both_null = df.filter(
        pl.col("implied_vol_bid").is_null() & pl.col("implied_vol_ask").is_null()
    ).height
    either_null = df.filter(
        pl.col("implied_vol_bid").is_null() | pl.col("implied_vol_ask").is_null()
    ).height

    logger.info(f"\nNaN Statistics:")
    logger.info(f"  Bid IV NaN:        {bid_iv_null:>12,} ({100*bid_iv_null/total_rows:>6.2f}%)")
    logger.info(f"  Ask IV NaN:        {ask_iv_null:>12,} ({100*ask_iv_null/total_rows:>6.2f}%)")
    logger.info(f"  Both NaN:          {both_null:>12,} ({100*both_null/total_rows:>6.2f}%)")
    logger.info(f"  Either NaN:        {either_null:>12,} ({100*either_null/total_rows:>6.2f}%)")
    logger.info(f"  Both valid:        {total_rows - either_null:>12,} ({100*(total_rows - either_null)/total_rows:>6.2f}%)")

    # =========================================================================
    # PART 2: FAILURE BY OPTION TYPE (CALL vs PUT)
    # =========================================================================
    logger.info("\n" + "="*80)
    logger.info("PART 2: FAILURE BY OPTION TYPE (CALL vs PUT)")
    logger.info("="*80)

    # Count by type
    type_stats = df.group_by("type").agg([
        pl.len().alias("total_count"),
        pl.col("implied_vol_bid").is_null().sum().alias("bid_nan_count"),
        pl.col("implied_vol_ask").is_null().sum().alias("ask_nan_count"),
        (pl.col("implied_vol_bid").is_null() & pl.col("implied_vol_ask").is_null()).sum().alias("both_nan_count"),
    ]).sort("type")

    logger.info("\nFailure Statistics by Option Type:")
    for row in type_stats.iter_rows(named=True):
        opt_type = row['type']
        total = row['total_count']
        bid_nan = row['bid_nan_count']
        ask_nan = row['ask_nan_count']
        both_nan = row['both_nan_count']

        logger.info(f"\n{opt_type} Options:")
        logger.info(f"  Total:             {total:>12,}")
        logger.info(f"  Bid IV NaN:        {bid_nan:>12,} ({100*bid_nan/total:>6.2f}%)")
        logger.info(f"  Ask IV NaN:        {ask_nan:>12,} ({100*ask_nan/total:>6.2f}%)")
        logger.info(f"  Both NaN:          {both_nan:>12,} ({100*both_nan/total:>6.2f}%)")

    # =========================================================================
    # PART 3: FAILURE BY MONEYNESS RANGE
    # =========================================================================
    logger.info("\n" + "="*80)
    logger.info("PART 3: FAILURE BY MONEYNESS RANGE")
    logger.info("="*80)

    # Create moneyness bins
    moneyness_stats = df.with_columns([
        pl.when(pl.col("moneyness") < 0.97).then(pl.lit("< 0.97 (deep OTM)"))
        .when(pl.col("moneyness") < 0.99).then(pl.lit("0.97-0.99 (OTM)"))
        .when(pl.col("moneyness") < 1.01).then(pl.lit("0.99-1.01 (ATM)"))
        .when(pl.col("moneyness") < 1.03).then(pl.lit("1.01-1.03 (ITM)"))
        .otherwise(pl.lit("> 1.03 (deep ITM)"))
        .alias("moneyness_bin")
    ]).group_by("moneyness_bin").agg([
        pl.len().alias("total_count"),
        pl.col("implied_vol_bid").is_null().sum().alias("bid_nan_count"),
        pl.col("implied_vol_ask").is_null().sum().alias("ask_nan_count"),
    ]).sort("moneyness_bin")

    logger.info("\nFailure Statistics by Moneyness:")
    for row in moneyness_stats.iter_rows(named=True):
        bin_name = row['moneyness_bin']
        total = row['total_count']
        bid_nan = row['bid_nan_count']
        ask_nan = row['ask_nan_count']

        logger.info(f"\n{bin_name}:")
        logger.info(f"  Total:             {total:>12,}")
        logger.info(f"  Bid IV NaN:        {bid_nan:>12,} ({100*bid_nan/total:>6.2f}%)")
        logger.info(f"  Ask IV NaN:        {ask_nan:>12,} ({100*ask_nan/total:>6.2f}%)")

    # =========================================================================
    # PART 4: FAILURE BY TIME TO EXPIRY
    # =========================================================================
    logger.info("\n" + "="*80)
    logger.info("PART 4: FAILURE BY TIME TO EXPIRY")
    logger.info("="*80)

    # Create TTL bins
    ttl_stats = df.with_columns([
        pl.when(pl.col("time_to_expiry_days") < 0.5).then(pl.lit("< 12 hours"))
        .when(pl.col("time_to_expiry_days") < 1.0).then(pl.lit("12-24 hours"))
        .when(pl.col("time_to_expiry_days") < 2.0).then(pl.lit("1-2 days"))
        .otherwise(pl.lit("2-3 days"))
        .alias("ttl_bin")
    ]).group_by("ttl_bin").agg([
        pl.len().alias("total_count"),
        pl.col("implied_vol_bid").is_null().sum().alias("bid_nan_count"),
        pl.col("implied_vol_ask").is_null().sum().alias("ask_nan_count"),
    ]).sort("ttl_bin")

    logger.info("\nFailure Statistics by Time to Expiry:")
    for row in ttl_stats.iter_rows(named=True):
        bin_name = row['ttl_bin']
        total = row['total_count']
        bid_nan = row['bid_nan_count']
        ask_nan = row['ask_nan_count']

        logger.info(f"\n{bin_name}:")
        logger.info(f"  Total:             {total:>12,}")
        logger.info(f"  Bid IV NaN:        {bid_nan:>12,} ({100*bid_nan/total:>6.2f}%)")
        logger.info(f"  Ask IV NaN:        {ask_nan:>12,} ({100*ask_nan/total:>6.2f}%)")

    # =========================================================================
    # PART 5: PAIRED ANALYSIS (CRITICAL)
    # =========================================================================
    logger.info("\n" + "="*80)
    logger.info("PART 5: PAIRED ANALYSIS (Same timestamp/strike, different type)")
    logger.info("="*80)

    logger.info("\nCreating paired dataset...")

    # Separate calls and puts
    calls = df.filter(pl.col("type") == "call").select([
        "timestamp_seconds",
        "strike_price",
        "expiry_timestamp",
        pl.col("implied_vol_bid").alias("call_iv_bid"),
        pl.col("implied_vol_ask").alias("call_iv_ask"),
        pl.col("bid_price").alias("call_bid_price"),
        pl.col("ask_price").alias("call_ask_price"),
    ])

    puts = df.filter(pl.col("type") == "put").select([
        "timestamp_seconds",
        "strike_price",
        "expiry_timestamp",
        pl.col("implied_vol_bid").alias("put_iv_bid"),
        pl.col("implied_vol_ask").alias("put_iv_ask"),
        pl.col("bid_price").alias("put_bid_price"),
        pl.col("ask_price").alias("put_ask_price"),
    ])

    # Join on timestamp, strike, expiry
    paired = calls.join(
        puts,
        on=["timestamp_seconds", "strike_price", "expiry_timestamp"],
        how="inner"
    )

    total_pairs = len(paired)
    logger.info(f"Total paired observations: {total_pairs:,}")

    # Analyze pairing patterns
    both_sides_valid = paired.filter(
        pl.col("call_iv_bid").is_not_null() &
        pl.col("put_iv_bid").is_not_null()
    ).height

    both_sides_fail = paired.filter(
        pl.col("call_iv_bid").is_null() &
        pl.col("put_iv_bid").is_null()
    ).height

    call_fails_put_valid = paired.filter(
        pl.col("call_iv_bid").is_null() &
        pl.col("put_iv_bid").is_not_null()
    ).height

    put_fails_call_valid = paired.filter(
        pl.col("put_iv_bid").is_null() &
        pl.col("call_iv_bid").is_not_null()
    ).height

    logger.info(f"\nPaired Analysis Results:")
    logger.info(f"  Both sides valid:      {both_sides_valid:>12,} ({100*both_sides_valid/total_pairs:>6.2f}%)")
    logger.info(f"  Both sides fail:       {both_sides_fail:>12,} ({100*both_sides_fail/total_pairs:>6.2f}%)")
    logger.info(f"  Call fails, put valid: {call_fails_put_valid:>12,} ({100*call_fails_put_valid/total_pairs:>6.2f}%)")
    logger.info(f"  Put fails, call valid: {put_fails_call_valid:>12,} ({100*put_fails_call_valid/total_pairs:>6.2f}%)")

    recoverable = call_fails_put_valid + put_fails_call_valid
    logger.info(f"\n🔧 Potentially recoverable via put-call parity: {recoverable:,} ({100*recoverable/total_pairs:.2f}%)")

    # =========================================================================
    # PART 6: ROOT CAUSE ANALYSIS (IV_CALC_STATUS)
    # =========================================================================
    logger.info("\n" + "="*80)
    logger.info("PART 6: ROOT CAUSE ANALYSIS (iv_calc_status)")
    logger.info("="*80)

    # Check if iv_calc_status exists
    if "iv_calc_status" in df.columns:
        status_counts = df.group_by("iv_calc_status").agg([
            pl.len().alias("count"),
        ]).sort("count", descending=True)

        logger.info("\nIV Calculation Status Breakdown:")
        for row in status_counts.iter_rows(named=True):
            status = row['iv_calc_status']
            count = row['count']
            logger.info(f"  {status:30s}: {count:>12,} ({100*count/total_rows:>6.2f}%)")
    else:
        logger.info("\n⚠️  iv_calc_status column not found in dataset")

    # =========================================================================
    # PART 7: TEMPORAL ANALYSIS
    # =========================================================================
    logger.info("\n" + "="*80)
    logger.info("PART 7: TEMPORAL ANALYSIS")
    logger.info("="*80)

    # Convert timestamp to datetime for temporal analysis
    temporal_stats = df.with_columns([
        pl.from_epoch(pl.col("timestamp_seconds"), time_unit="s").alias("datetime"),
    ]).with_columns([
        pl.col("datetime").dt.date().alias("date"),
        pl.col("datetime").dt.hour().alias("hour"),
    ]).group_by("date").agg([
        pl.len().alias("total_count"),
        pl.col("implied_vol_bid").is_null().sum().alias("bid_nan_count"),
    ]).sort("date")

    # Find dates with highest failure rates
    top_failure_dates = temporal_stats.with_columns([
        (pl.col("bid_nan_count") / pl.col("total_count")).alias("failure_rate")
    ]).sort("failure_rate", descending=True).head(10)

    logger.info("\nTop 10 dates with highest IV failure rates:")
    for row in top_failure_dates.iter_rows(named=True):
        date = row['date']
        total = row['total_count']
        failures = row['bid_nan_count']
        rate = row['failure_rate']
        logger.info(f"  {date}: {failures:>8,}/{total:>8,} ({100*rate:>6.2f}%)")

    # =========================================================================
    # PART 8: DATA QUALITY METRICS
    # =========================================================================
    logger.info("\n" + "="*80)
    logger.info("PART 8: DATA QUALITY METRICS")
    logger.info("="*80)

    # Check for missing market data
    missing_bid = df.filter(pl.col("has_bid") == False).height
    missing_ask = df.filter(pl.col("has_ask") == False).height
    missing_spot = df.filter(pl.col("spot_price").is_null()).height

    logger.info(f"\nMissing Market Data:")
    logger.info(f"  Missing bid price:     {missing_bid:>12,} ({100*missing_bid/total_rows:>6.2f}%)")
    logger.info(f"  Missing ask price:     {missing_ask:>12,} ({100*missing_ask/total_rows:>6.2f}%)")
    logger.info(f"  Missing spot price:    {missing_spot:>12,} ({100*missing_spot/total_rows:>6.2f}%)")

    # Check IV validity ranges
    valid_ivs = df.filter(
        pl.col("implied_vol_bid").is_not_null() &
        (pl.col("implied_vol_bid") > 0.05) &
        (pl.col("implied_vol_bid") < 5.0)
    ).height

    logger.info(f"\nIV Validity Check (0.05 < IV < 5.0):")
    logger.info(f"  Valid IV values:       {valid_ivs:>12,} ({100*valid_ivs/total_rows:>6.2f}%)")

    # =========================================================================
    # SUMMARY AND RECOMMENDATIONS
    # =========================================================================
    logger.info("\n" + "="*80)
    logger.info("SUMMARY AND RECOMMENDATIONS")
    logger.info("="*80)

    logger.info(f"\n📊 Key Findings:")
    logger.info(f"  1. Total IV failures (either bid or ask): {either_null:,} ({100*either_null/total_rows:.2f}%)")
    logger.info(f"  2. Both bid and ask fail together: {both_null:,} ({100*both_null/total_rows:.2f}%)")
    logger.info(f"  3. Potentially recoverable via put-call parity: {recoverable:,}")

    logger.info(f"\n🔧 Recommended Cleaning Strategies:")
    logger.info(f"  1. Put-Call Parity Recovery: {recoverable:,} observations")
    logger.info(f"  2. Exclude both-sides-fail: {both_sides_fail:,} observations")
    logger.info(f"  3. Expected NaN rate after cleaning: {100*(both_sides_fail)/(total_rows):.2f}%")

    logger.info(f"\n✅ Analysis complete!")
    logger.info(f"End time: {datetime.now()}")
    logger.info("="*80)

if __name__ == "__main__":
    try:
        analyze_failures()
    except Exception as e:
        logger.error(f"Analysis failed: {e}", exc_info=True)
        raise
