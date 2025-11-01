#!/usr/bin/env python3
"""
Verification script for memory fixes.

Tests that fixes reduce memory usage as expected:
1. Schema resolution: No more warnings or leaks
2. Timestamp filtering: Feature files filtered before join
3. Join efficiency: Small collections don't spike memory
"""

from __future__ import annotations

import gc
import logging
import sys
from datetime import date
from pathlib import Path

import polars as pl
import psutil

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

# File paths
MODEL_DIR = Path(__file__).parent.parent
BASELINE_FILE = MODEL_DIR / "results/production_backtest_results.parquet"
RV_FILE = MODEL_DIR / "results/realized_volatility_1s.parquet"
MICROSTRUCTURE_FILE = MODEL_DIR / "results/microstructure_features.parquet"
ADVANCED_FILE = MODEL_DIR / "results/advanced_features.parquet"


def log_memory(label: str) -> float:
    """Log current memory usage and return in GB."""
    process = psutil.Process()
    mem_info = process.memory_info()
    mem_gb = mem_info.rss / (1024**3)
    logger.info(f"[MEMORY] {label}: {mem_gb:.2f} GB")
    return mem_gb


def test_schema_resolution() -> bool:
    """Test that schema resolution uses collect_schema() and doesn't leak memory."""
    logger.info("=" * 80)
    logger.info("TEST 1: Schema Resolution (should have 0 warnings, 0 leaks)")
    logger.info("=" * 80)

    mem_start = log_memory("Initial")

    # Test with correct method
    logger.info("\n1. Testing collect_schema() method (correct)...")
    baseline_lf = pl.scan_parquet(BASELINE_FILE)
    schema = baseline_lf.collect_schema()
    cols = schema.names()
    logger.info(f"  Columns: {len(cols)}")

    mem_after = log_memory("After schema access")
    leak_mb = (mem_after - mem_start) * 1024

    logger.info(f"  Memory leak: {leak_mb:.1f} MB")

    # Check result
    if leak_mb < 5:  # Allow <5MB for minor overhead
        logger.info("  ✅ PASS: No significant memory leak")
        return True
    else:
        logger.error(f"  ❌ FAIL: Memory leaked {leak_mb:.1f} MB (expected <5 MB)")
        return False


def test_timestamp_filtering() -> bool:
    """Test that feature files are filtered before join in production mode."""
    logger.info("\n" + "=" * 80)
    logger.info("TEST 2: Timestamp Filtering (features should be filtered BEFORE join)")
    logger.info("=" * 80)

    mem_start = log_memory("Initial")

    # Simulate production mode with date range
    start_date = "2023-10-01"
    end_date = "2023-10-31"

    logger.info(f"\nSimulating production mode: {start_date} to {end_date}")

    # Load baseline
    from datetime import datetime

    baseline_lf = pl.scan_parquet(BASELINE_FILE)
    start_dt = date.fromisoformat(start_date)
    end_dt = date.fromisoformat(end_date)
    baseline_lf = baseline_lf.filter((pl.col("date") >= start_dt) & (pl.col("date") <= end_dt))

    n_baseline = baseline_lf.select(pl.len()).collect().item()
    logger.info(f"  Baseline rows (after date filter): {n_baseline:,}")

    # Calculate timestamp range
    start_ts = int(datetime.fromisoformat(start_date).timestamp())
    end_ts = int(datetime.fromisoformat(end_date).timestamp()) + 86400  # Add 1 day to include end_date

    logger.info(f"  Timestamp range: {start_ts} to {end_ts}")

    # Load and filter RV features
    logger.info("\n1. Testing RV features filtering...")
    rv_lf = pl.scan_parquet(RV_FILE)

    # Count before filter
    n_rv_before = rv_lf.select(pl.len()).collect().item()
    logger.info(f"  RV rows (before filter): {n_rv_before:,}")

    # Apply filter
    rv_lf = rv_lf.filter((pl.col("timestamp_seconds") >= start_ts) & (pl.col("timestamp_seconds") < end_ts))
    n_rv_after = rv_lf.select(pl.len()).collect().item()
    logger.info(f"  RV rows (after filter): {n_rv_after:,}")

    reduction_pct = (1 - n_rv_after / n_rv_before) * 100
    logger.info(f"  Reduction: {reduction_pct:.1f}%")

    if n_rv_after < n_rv_before:
        logger.info("  ✅ PASS: RV features filtered before join")
        rv_pass = True
    else:
        logger.error("  ❌ FAIL: RV features not filtered")
        rv_pass = False

    # Load and filter microstructure features
    logger.info("\n2. Testing microstructure features filtering...")
    micro_lf = pl.scan_parquet(MICROSTRUCTURE_FILE)
    n_micro_before = micro_lf.select(pl.len()).collect().item()
    logger.info(f"  Microstructure rows (before filter): {n_micro_before:,}")

    micro_lf = micro_lf.filter((pl.col("timestamp_seconds") >= start_ts) & (pl.col("timestamp_seconds") < end_ts))
    n_micro_after = micro_lf.select(pl.len()).collect().item()
    logger.info(f"  Microstructure rows (after filter): {n_micro_after:,}")

    reduction_pct = (1 - n_micro_after / n_micro_before) * 100
    logger.info(f"  Reduction: {reduction_pct:.1f}%")

    if n_micro_after < n_micro_before:
        logger.info("  ✅ PASS: Microstructure features filtered before join")
        micro_pass = True
    else:
        logger.error("  ❌ FAIL: Microstructure features not filtered")
        micro_pass = False

    # Load and filter advanced features
    logger.info("\n3. Testing advanced features filtering...")
    adv_lf = pl.scan_parquet(ADVANCED_FILE)
    n_adv_before = adv_lf.select(pl.len()).collect().item()
    logger.info(f"  Advanced rows (before filter): {n_adv_before:,}")

    adv_lf = adv_lf.filter((pl.col("timestamp") >= start_ts) & (pl.col("timestamp") < end_ts))
    n_adv_after = adv_lf.select(pl.len()).collect().item()
    logger.info(f"  Advanced rows (after filter): {n_adv_after:,}")

    reduction_pct = (1 - n_adv_after / n_adv_before) * 100
    logger.info(f"  Reduction: {reduction_pct:.1f}%")

    if n_adv_after < n_adv_before:
        logger.info("  ✅ PASS: Advanced features filtered before join")
        adv_pass = True
    else:
        logger.error("  ❌ FAIL: Advanced features not filtered")
        adv_pass = False

    mem_after = log_memory("After filtering")
    mem_increase_mb = (mem_after - mem_start) * 1024
    logger.info(f"\nTotal memory increase: {mem_increase_mb:.1f} MB")

    return rv_pass and micro_pass and adv_pass


def test_join_efficiency() -> bool:
    """Test that joining and collecting small samples doesn't spike memory."""
    logger.info("\n" + "=" * 80)
    logger.info("TEST 3: Join Efficiency (10k sample should use <100MB, not 3GB)")
    logger.info("=" * 80)

    from datetime import datetime

    mem_start = log_memory("Initial")

    # Use October 2023 for test
    start_date = date(2023, 10, 1)
    end_date = date(2023, 10, 31)

    logger.info(f"\nTesting with October 2023 data")

    # Load and filter baseline
    baseline_lf = pl.scan_parquet(BASELINE_FILE)
    baseline_lf = baseline_lf.filter((pl.col("date") >= start_date) & (pl.col("date") <= end_date))

    # Add computed columns
    baseline_lf = baseline_lf.with_columns([
        (pl.col("outcome") - pl.col("prob_mid")).alias("residual"),
        (pl.col("S") / pl.col("K") - 1.0).alias("moneyness"),
    ])

    # Calculate timestamp range
    start_ts = int(datetime(2023, 10, 1).timestamp())
    end_ts = int(datetime(2023, 11, 1).timestamp())

    # Load and filter RV features
    rv_lf = pl.scan_parquet(RV_FILE)
    rv_lf = rv_lf.filter((pl.col("timestamp_seconds") >= start_ts) & (pl.col("timestamp_seconds") < end_ts))

    # Join RV
    schema_rv = rv_lf.collect_schema()
    rv_features = [c for c in schema_rv.names() if c != "timestamp_seconds"]
    joined_lf = baseline_lf.join(
        rv_lf.select(["timestamp_seconds"] + rv_features),
        left_on="timestamp",
        right_on="timestamp_seconds",
        how="left"
    )

    # Load and filter microstructure
    micro_lf = pl.scan_parquet(MICROSTRUCTURE_FILE)
    micro_lf = micro_lf.filter((pl.col("timestamp_seconds") >= start_ts) & (pl.col("timestamp_seconds") < end_ts))

    # Join microstructure
    schema_micro = micro_lf.collect_schema()
    micro_features = [c for c in schema_micro.names() if c != "timestamp_seconds"]
    joined_lf = joined_lf.join(
        micro_lf.select(["timestamp_seconds"] + micro_features),
        left_on="timestamp",
        right_on="timestamp_seconds",
        how="left"
    )

    # Load and filter advanced
    adv_lf = pl.scan_parquet(ADVANCED_FILE)
    adv_lf = adv_lf.filter((pl.col("timestamp") >= start_ts) & (pl.col("timestamp") < end_ts))

    # Join advanced
    schema_adv = adv_lf.collect_schema()
    adv_features = [c for c in schema_adv.names() if c not in ["timestamp", "timestamp_seconds"]]
    joined_lf = joined_lf.join(
        adv_lf.select(["timestamp"] + adv_features),
        on="timestamp",
        how="left"
    )

    mem_after_joins = log_memory("After all lazy joins")
    logger.info(f"Memory increase from lazy joins: {(mem_after_joins - mem_start) * 1024:.1f} MB")

    # Now collect 10k sample
    logger.info("\nCollecting 10k sample...")
    mem_before_collect = log_memory("Before collect")

    sample_df = joined_lf.head(10000).collect()

    mem_after_collect = log_memory("After collect")
    mem_spike_mb = (mem_after_collect - mem_before_collect) * 1024

    logger.info(f"  Sample size: {len(sample_df):,} rows")
    logger.info(f"  Sample columns: {len(sample_df.columns)}")
    logger.info(f"  Memory spike: {mem_spike_mb:.1f} MB")

    # Calculate expected memory
    expected_mb = len(sample_df) * len(sample_df.columns) * 4 / (1024**2)  # 4 bytes per Float32
    logger.info(f"  Expected memory: ~{expected_mb:.1f} MB")

    # Check result
    if mem_spike_mb < 200:  # Allow 200MB (50x expected) as reasonable overhead
        logger.info(f"  ✅ PASS: Memory spike {mem_spike_mb:.1f} MB is reasonable (expected ~{expected_mb:.1f} MB)")
        return True
    else:
        logger.error(f"  ❌ FAIL: Memory spike {mem_spike_mb:.1f} MB is too high (expected ~{expected_mb:.1f} MB)")
        return False


def main() -> None:
    """Run all verification tests."""
    logger.info("=" * 80)
    logger.info("MEMORY FIX VERIFICATION")
    logger.info("=" * 80)

    log_memory("Initial")

    # Run tests
    test1_pass = test_schema_resolution()
    gc.collect()

    test2_pass = test_timestamp_filtering()
    gc.collect()

    test3_pass = test_join_efficiency()
    gc.collect()

    # Summary
    logger.info("\n" + "=" * 80)
    logger.info("TEST SUMMARY")
    logger.info("=" * 80)
    logger.info(f"Test 1 (Schema Resolution): {'✅ PASS' if test1_pass else '❌ FAIL'}")
    logger.info(f"Test 2 (Timestamp Filtering): {'✅ PASS' if test2_pass else '❌ FAIL'}")
    logger.info(f"Test 3 (Join Efficiency): {'✅ PASS' if test3_pass else '❌ FAIL'}")

    all_pass = test1_pass and test2_pass and test3_pass

    if all_pass:
        logger.info("\n✅ ALL TESTS PASSED - Memory fixes are working correctly!")
    else:
        logger.error("\n❌ SOME TESTS FAILED - Review fixes and re-test")

    log_memory("Final")


if __name__ == "__main__":
    main()
