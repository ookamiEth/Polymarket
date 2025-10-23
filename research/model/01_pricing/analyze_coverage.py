#!/usr/bin/env python3
"""
Analyze why options coverage is only 14%.

Investigates:
1. How many unique timestamps have option quotes?
2. How many options expire AFTER contract close times?
3. What's the distribution of option expiries vs contract close times?
4. Are we filtering too aggressively?
"""

import logging
from pathlib import Path

import polars as pl

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# File paths
RESULTS_FILE = Path("/Users/lgierhake/Documents/ETH/BT/research/model/results/pilot_backtest_results.parquet")
OPTIONS_FILE = Path(
    "/Users/lgierhake/Documents/ETH/BT/research/tardis/data/consolidated/btc_options_atm_shortdated_with_iv_2023_2025.parquet"
)

logger.info("Loading pilot backtest results...")
results = pl.read_parquet(RESULTS_FILE)

logger.info("=" * 80)
logger.info("COVERAGE ANALYSIS")
logger.info("=" * 80)

# 1. Total pricing attempts
total_rows = len(results)
unique_timestamps = results["timestamp"].n_unique()
unique_contracts = results["contract_id"].n_unique()

logger.info("\n1. TOTAL PRICING GRID:")
logger.info(f"   Total rows: {total_rows:,}")
logger.info(f"   Unique timestamps: {unique_timestamps:,}")
logger.info(f"   Unique contracts: {unique_contracts:,}")

# 2. How many have IV data?
with_iv = results.filter(pl.col("sigma_mid").is_not_null())
logger.info("\n2. OPTIONS IV COVERAGE:")
logger.info(f"   Rows with IV: {len(with_iv):,} / {total_rows:,} ({len(with_iv) / total_rows * 100:.1f}%)")
logger.info(f"   Unique timestamps with IV: {with_iv['timestamp'].n_unique():,}")

# 3. Check raw options data availability
logger.info("\n3. RAW OPTIONS DATA:")
min_ts_val = results.select(pl.col("timestamp").min()).item()
max_ts_val = results.select(pl.col("timestamp").max()).item()

if min_ts_val is not None and max_ts_val is not None:
    min_ts = int(min_ts_val)
    max_ts = int(max_ts_val)

    logger.info(f"   Time window: {min_ts} to {max_ts}")

    # Load all options in this window (no filters)
    options = pl.scan_parquet(OPTIONS_FILE)
    all_options = (
        options.filter((pl.col("timestamp_seconds") >= min_ts) & (pl.col("timestamp_seconds") <= max_ts))
        .select(["timestamp_seconds", "expiry_timestamp", "iv_calc_status"])
        .collect()
    )

    logger.info(f"   Total option quotes in window: {len(all_options):,}")
    logger.info(
        f"   Unique timestamps with quotes: {all_options['timestamp_seconds'].n_unique():,} / {unique_timestamps:,}"
    )

    # How many have successful IV?
    valid_iv = all_options.filter(pl.col("iv_calc_status") == "success")
    logger.info(f"   Quotes with successful IV: {len(valid_iv):,} ({len(valid_iv) / len(all_options) * 100:.1f}%)")

# 4. Expiry constraint analysis
logger.info("\n4. EXPIRY CONSTRAINT ANALYSIS:")

# Get unique contract close times
close_times = results.select(["close_time"]).unique().sort("close_time")
logger.info(f"   Unique contract close times: {len(close_times):,}")

# For first contract, check how many options expire after it
first_close = int(close_times["close_time"][0])
logger.info(f"   First contract closes at: {first_close}")

if min_ts_val is not None and max_ts_val is not None:
    # Count options that expire after first contract close
    options_after_first = all_options.filter(pl.col("expiry_timestamp") > first_close)
    logger.info(f"   Options expiring after first close: {len(options_after_first):,} / {len(all_options):,}")

# 5. Check a specific timestamp
sample_ts = int(results.filter(pl.col("sigma_mid").is_not_null())["timestamp"][0])
sample_close = int(results.filter(pl.col("timestamp") == sample_ts)["close_time"][0])

logger.info("\n5. SAMPLE TIMESTAMP ANALYSIS:")
logger.info(f"   Timestamp: {sample_ts}")
logger.info(f"   Contract closes: {sample_close}")

options_at_ts = all_options.filter(pl.col("timestamp_seconds") == sample_ts)
logger.info(f"   Options at this timestamp: {len(options_at_ts):,}")

if len(options_at_ts) > 0:
    valid_expiry = options_at_ts.filter(pl.col("expiry_timestamp") > sample_close)
    logger.info(f"   Options expiring after contract: {len(valid_expiry):,}")

    if len(valid_expiry) > 0:
        expiries = valid_expiry["expiry_timestamp"].sort()
        logger.info(f"   Closest expiry: {expiries[0]}")
        logger.info(
            f"   Time until expiry: {int(expiries[0]) - sample_close} seconds ({(int(expiries[0]) - sample_close) / 3600:.1f} hours)"
        )

# 6. Distribution of missing data by time within contract
logger.info("\n6. COVERAGE BY TIME WITHIN CONTRACT:")
coverage_by_offset = (
    results.group_by("seconds_offset")
    .agg([pl.len().alias("total"), pl.col("sigma_mid").is_not_null().sum().alias("with_iv")])
    .with_columns([(pl.col("with_iv") / pl.col("total") * 100).alias("coverage_pct")])
    .sort("seconds_offset")
)

# Show first, middle, and last
logger.info("   First few seconds:")
print(coverage_by_offset.head(10))
logger.info("   Last few seconds:")
print(coverage_by_offset.tail(10))

logger.info("\n" + "=" * 80)
logger.info("SUMMARY")
logger.info("=" * 80)
logger.info("Main reasons for 14% coverage:")
logger.info("1. Options data has ~11-14% timestamp coverage in general")
logger.info("2. Expiry constraint: options must expire AFTER contract close")
logger.info("3. Many short-dated options expire before our 15-min contracts close")
logger.info("4. We take closest expiry only (not all valid options)")
logger.info("\nThis is EXPECTED for ultra-short-dated (15-min) binary contracts!")
logger.info("=" * 80)
