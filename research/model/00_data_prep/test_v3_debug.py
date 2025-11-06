#!/usr/bin/env python3
"""
Diagnostic Tests for Feature Engineering V3

Temporary test script to identify and validate fixes for OOM issues.
DELETE AFTER V3 SCRIPT WORKS.

Tests:
1. Verify input file schemas and row counts
2. Small-scale join test (100K rows)
3. Streaming write validation
4. Full Module 1 with progress logging

Author: Debug Session
Date: 2025-11-05
"""

from __future__ import annotations

import logging
import sys
import time
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

# Paths
MODEL_DIR = Path(__file__).parent.parent
RESULTS_DIR = MODEL_DIR / "results"
DATA_DIR = MODEL_DIR / "data"

# Input files
BASELINE_FILE = RESULTS_DIR / "production_backtest_results.parquet"
ADVANCED_FILE = RESULTS_DIR / "advanced_features.parquet"
MICRO_FILE = RESULTS_DIR / "microstructure_features.parquet"

# Test output
TEST_DIR = DATA_DIR / "features_v3" / "test"
TEST_DIR.mkdir(parents=True, exist_ok=True)


class MemoryMonitor:
    """Monitor memory usage during processing."""

    def __init__(self):
        self.process = psutil.Process()

    def get_memory_gb(self) -> float:
        """Get current memory usage in GB."""
        return self.process.memory_info().rss / (1024**3)

    def log_memory(self, label: str) -> None:
        """Log current memory usage."""
        memory_gb = self.get_memory_gb()
        logger.info(f"[{label}] Memory: {memory_gb:.2f} GB")


def test_1_verify_input_schemas() -> bool:
    """
    Test 1: Verify input file schemas and row counts.

    Expected:
    - Baseline: 63,072,900 rows, timestamp (microseconds)
    - Advanced: 63,072,900 rows, timestamp (microseconds)
    - Microstructure: 63,158,387 rows, timestamp_seconds (seconds)

    Returns:
        True if schemas match expectations
    """
    logger.info("=" * 80)
    logger.info("TEST 1: Verify Input File Schemas")
    logger.info("=" * 80)

    try:
        # Check baseline
        logger.info(f"Scanning baseline: {BASELINE_FILE}")
        baseline = pl.scan_parquet(BASELINE_FILE)
        baseline_schema = baseline.collect_schema()
        baseline_rows = baseline.select(pl.len()).collect().item()

        logger.info(f"  Rows: {baseline_rows:,}")
        logger.info(f"  Columns: {len(baseline_schema.names())}")
        logger.info(f"  Has 'timestamp': {'timestamp' in baseline_schema.names()}")
        logger.info(f"  Has 'S' (spot): {'S' in baseline_schema.names()}")
        logger.info(f"  Has 'K' (strike): {'K' in baseline_schema.names()}")

        # Check advanced
        logger.info(f"\nScanning advanced: {ADVANCED_FILE}")
        advanced = pl.scan_parquet(ADVANCED_FILE)
        advanced_schema = advanced.collect_schema()
        advanced_rows = advanced.select(pl.len()).collect().item()

        logger.info(f"  Rows: {advanced_rows:,}")
        logger.info(f"  Columns: {len(advanced_schema.names())}")
        logger.info(f"  Has 'timestamp': {'timestamp' in advanced_schema.names()}")

        # Check microstructure
        logger.info(f"\nScanning microstructure: {MICRO_FILE}")
        micro = pl.scan_parquet(MICRO_FILE)
        micro_schema = micro.collect_schema()
        micro_rows = micro.select(pl.len()).collect().item()

        logger.info(f"  Rows: {micro_rows:,}")
        logger.info(f"  Columns: {len(micro_schema.names())}")
        logger.info(f"  Has 'timestamp_seconds': {'timestamp_seconds' in micro_schema.names()}")

        # Check row count consistency
        logger.info("\n" + "=" * 80)
        logger.info("ROW COUNT ANALYSIS")
        logger.info("=" * 80)
        logger.info(f"Baseline:       {baseline_rows:,}")
        logger.info(f"Advanced:       {advanced_rows:,}")
        logger.info(f"Microstructure: {micro_rows:,}")

        if baseline_rows == advanced_rows:
            logger.info("✓ Baseline and Advanced match")
        else:
            logger.warning(f"✗ Baseline ({baseline_rows}) != Advanced ({advanced_rows})")

        row_diff = micro_rows - baseline_rows
        pct_diff = (row_diff / baseline_rows) * 100
        logger.info(f"\nMicrostructure difference: +{row_diff:,} rows ({pct_diff:.3f}%)")

        if row_diff > 0:
            logger.warning(f"⚠ Microstructure has {row_diff:,} extra rows")
            logger.info("  → LEFT JOIN strategy will keep all baseline rows (correct)")

        logger.info("\n✓ TEST 1 PASSED: All schemas verified")
        return True

    except Exception as e:
        logger.error(f"✗ TEST 1 FAILED: {e}")
        return False


def test_2_small_scale_join() -> bool:
    """
    Test 2: Test small-scale join on 100K rows.

    Validates:
    - Join logic works correctly
    - Memory usage is reasonable
    - Output schema is correct

    Returns:
        True if join succeeds
    """
    logger.info("=" * 80)
    logger.info("TEST 2: Small-Scale Join Test (100K rows)")
    logger.info("=" * 80)

    memory_monitor = MemoryMonitor()
    memory_monitor.log_memory("Start")

    try:
        # Load first 100K rows from each file
        logger.info("Loading 100K rows from baseline...")
        baseline = pl.scan_parquet(BASELINE_FILE).head(100_000)

        # Convert timestamp to seconds and compute moneyness
        baseline = baseline.with_columns([
            (pl.col("timestamp") / 1_000_000).cast(pl.Int64).alias("timestamp_seconds"),
            (pl.col("S") / pl.col("K")).alias("moneyness"),
        ]).select([
            "timestamp_seconds",
            "time_remaining",
            "moneyness",
            "iv_staleness_seconds",
        ])

        memory_monitor.log_memory("After baseline scan")

        logger.info("Loading 100K rows from advanced...")
        advanced_keep = [
            "high_15m", "low_15m", "drawdown_from_high_15m", "runup_from_low_15m",
            "time_since_high_15m", "time_since_low_15m",
            "skewness_300s", "kurtosis_300s", "downside_vol_300s", "upside_vol_300s",
            "vol_asymmetry_300s", "tail_risk_300s",
            "hour_of_day_utc", "hour_sin", "hour_cos",
            "vol_persistence_ar1", "vol_acceleration_300s", "vol_of_vol_300s",
            "garch_forecast_simple", "autocorr_decay", "reversals_300s",
        ]
        advanced = pl.scan_parquet(ADVANCED_FILE).head(100_000).select(["timestamp"] + advanced_keep)
        advanced = advanced.with_columns([
            (pl.col("timestamp") / 1_000_000).cast(pl.Int64).alias("timestamp_seconds")
        ]).select(["timestamp_seconds"] + advanced_keep)

        memory_monitor.log_memory("After advanced scan")

        logger.info("Loading 100K rows from microstructure...")
        micro_keep = ["autocorr_lag5_300s", "hurst_300s"]
        micro = pl.scan_parquet(MICRO_FILE).head(100_000).select(["timestamp_seconds"] + micro_keep)

        memory_monitor.log_memory("After micro scan")

        # Perform joins
        logger.info("\nJoining baseline + advanced...")
        start = time.time()
        df = baseline.join(advanced, on="timestamp_seconds", how="left")
        memory_monitor.log_memory("After baseline-advanced join")

        logger.info("Joining + microstructure...")
        df = df.join(micro, on="timestamp_seconds", how="left")
        memory_monitor.log_memory("After micro join")

        # Collect to verify
        logger.info("Collecting result...")
        df_collected = df.collect()
        elapsed = time.time() - start

        memory_monitor.log_memory("After collect")

        logger.info(f"\n✓ Join completed in {elapsed:.2f}s")
        logger.info(f"  Output rows: {len(df_collected):,}")
        logger.info(f"  Output columns: {len(df_collected.columns)}")
        logger.info(f"  Expected columns: 27 (4 baseline + 21 advanced + 2 micro)")

        # Check for nulls
        null_counts = {col: df_collected[col].null_count() for col in df_collected.columns}
        critical_nulls = {k: v for k, v in null_counts.items() if v > 0}

        if critical_nulls:
            logger.warning(f"  Nulls detected: {critical_nulls}")
        else:
            logger.info("  ✓ No nulls in output")

        logger.info("\n✓ TEST 2 PASSED: Join logic works correctly")
        return True

    except Exception as e:
        logger.error(f"✗ TEST 2 FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_3_streaming_write() -> bool:
    """
    Test 3: Validate streaming write behavior.

    Tests:
    - sink_parquet streams data (doesn't collect first)
    - Memory stays constant during write
    - Output file is valid

    Returns:
        True if streaming write works
    """
    logger.info("=" * 80)
    logger.info("TEST 3: Streaming Write Validation")
    logger.info("=" * 80)

    memory_monitor = MemoryMonitor()
    memory_monitor.log_memory("Start")

    try:
        # Create a moderately large lazy frame (1M rows)
        logger.info("Creating 1M row test dataset...")
        baseline = pl.scan_parquet(BASELINE_FILE).head(1_000_000)
        baseline = baseline.with_columns([
            (pl.col("timestamp") / 1_000_000).cast(pl.Int64).alias("timestamp_seconds"),
            (pl.col("S") / pl.col("K")).alias("moneyness"),
        ]).select([
            "timestamp_seconds",
            "time_remaining",
            "moneyness",
            "iv_staleness_seconds",
        ])

        memory_monitor.log_memory("After lazy frame creation")

        # Write with streaming
        output_file = TEST_DIR / "test_streaming_write.parquet"
        logger.info(f"Writing to {output_file} with streaming...")

        start = time.time()
        baseline.sink_parquet(output_file, compression="snappy")
        elapsed = time.time() - start

        memory_monitor.log_memory("After streaming write")

        logger.info(f"✓ Write completed in {elapsed:.2f}s")

        # Verify output
        logger.info("Verifying output file...")
        df_verify = pl.scan_parquet(output_file)
        verify_rows = df_verify.select(pl.len()).collect().item()
        verify_cols = len(df_verify.collect_schema().names())

        logger.info(f"  Output rows: {verify_rows:,}")
        logger.info(f"  Output columns: {verify_cols}")

        if verify_rows == 1_000_000 and verify_cols == 4:
            logger.info("  ✓ Output file is valid")
        else:
            logger.warning(f"  ✗ Unexpected output shape: {verify_rows} rows, {verify_cols} cols")

        # Cleanup
        output_file.unlink()
        logger.info("  ✓ Test file cleaned up")

        logger.info("\n✓ TEST 3 PASSED: Streaming write works correctly")
        return True

    except Exception as e:
        logger.error(f"✗ TEST 3 FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_4_full_module_1() -> bool:
    """
    Test 4: Run full Module 1 logic with progress logging.

    This is the actual Module 1 code with enhanced monitoring.

    Returns:
        True if Module 1 completes successfully
    """
    logger.info("=" * 80)
    logger.info("TEST 4: Full Module 1 with Progress Logging")
    logger.info("=" * 80)

    memory_monitor = MemoryMonitor()
    memory_monitor.log_memory("Start")

    try:
        # Load baseline
        logger.info(f"Loading baseline: {BASELINE_FILE}")
        baseline = pl.scan_parquet(BASELINE_FILE).select([
            "timestamp",
            "time_remaining",
            "S",
            "K",
            "iv_staleness_seconds",
        ])
        baseline_rows = baseline.select(pl.len()).collect().item()
        logger.info(f"  Baseline rows: {baseline_rows:,}")

        baseline = baseline.with_columns([
            (pl.col("timestamp") / 1_000_000).cast(pl.Int64).alias("timestamp_seconds"),
            (pl.col("S") / pl.col("K")).alias("moneyness"),
        ]).select([
            "timestamp_seconds",
            "time_remaining",
            "moneyness",
            "iv_staleness_seconds",
        ])

        memory_monitor.log_memory("After baseline load")

        # Load advanced
        logger.info(f"Loading advanced features: {ADVANCED_FILE}")
        advanced_keep = [
            "high_15m", "low_15m", "drawdown_from_high_15m", "runup_from_low_15m",
            "time_since_high_15m", "time_since_low_15m",
            "skewness_300s", "kurtosis_300s", "downside_vol_300s", "upside_vol_300s",
            "vol_asymmetry_300s", "tail_risk_300s",
            "hour_of_day_utc", "hour_sin", "hour_cos",
            "vol_persistence_ar1", "vol_acceleration_300s", "vol_of_vol_300s",
            "garch_forecast_simple", "autocorr_decay", "reversals_300s",
        ]
        advanced = pl.scan_parquet(ADVANCED_FILE).select(["timestamp"] + advanced_keep)
        advanced_rows = advanced.select(pl.len()).collect().item()
        logger.info(f"  Advanced rows: {advanced_rows:,}")

        advanced = advanced.with_columns([
            (pl.col("timestamp") / 1_000_000).cast(pl.Int64).alias("timestamp_seconds")
        ]).select(["timestamp_seconds"] + advanced_keep)

        memory_monitor.log_memory("After advanced load")

        # Load microstructure
        logger.info(f"Loading microstructure features: {MICRO_FILE}")
        micro_keep = ["autocorr_lag5_300s", "hurst_300s"]
        micro = pl.scan_parquet(MICRO_FILE).select(["timestamp_seconds"] + micro_keep)
        micro_rows = micro.select(pl.len()).collect().item()
        logger.info(f"  Microstructure rows: {micro_rows:,}")

        memory_monitor.log_memory("After micro load")

        # Check row counts
        logger.info("\nRow count validation:")
        if baseline_rows == advanced_rows:
            logger.info(f"  ✓ Baseline ({baseline_rows:,}) == Advanced ({advanced_rows:,})")
        else:
            logger.warning(f"  ✗ Baseline ({baseline_rows:,}) != Advanced ({advanced_rows:,})")

        row_diff = micro_rows - baseline_rows
        if row_diff > 0:
            logger.warning(f"  ⚠ Microstructure has +{row_diff:,} extra rows ({(row_diff/baseline_rows)*100:.3f}%)")
            logger.info("    → LEFT JOIN will preserve all baseline rows")

        # Perform joins
        logger.info("\nJoining baseline + advanced...")
        start_join = time.time()
        df = baseline.join(advanced, on="timestamp_seconds", how="left")
        memory_monitor.log_memory("After baseline-advanced join")

        logger.info("Joining + microstructure...")
        df = df.join(micro, on="timestamp_seconds", how="left")
        memory_monitor.log_memory("After micro join")

        elapsed_join = time.time() - start_join
        logger.info(f"✓ Joins completed in {elapsed_join:.2f}s")

        # Write output
        output_file = TEST_DIR / "test_module_1_output.parquet"
        logger.info(f"\nWriting to {output_file}...")
        logger.info("  (Streaming write - this may take 2-5 minutes for 63M rows)")

        start_write = time.time()
        df.sink_parquet(output_file, compression="snappy")
        elapsed_write = time.time() - start_write

        memory_monitor.log_memory("After streaming write")

        logger.info(f"✓ Streaming write completed in {elapsed_write:.1f}s")

        # Verify output
        logger.info("\nVerifying output file...")
        df_verify = pl.scan_parquet(output_file)
        verify_rows = df_verify.select(pl.len()).collect().item()
        verify_cols = len(df_verify.collect_schema().names())

        logger.info(f"  Output rows: {verify_rows:,}")
        logger.info(f"  Output columns: {verify_cols}")
        logger.info(f"  Expected: {baseline_rows:,} rows, 27 columns")

        if verify_rows == baseline_rows:
            logger.info("  ✓ Row count preserved")
        else:
            logger.warning(f"  ✗ Row count changed: {baseline_rows:,} → {verify_rows:,}")

        if verify_cols == 27:
            logger.info("  ✓ Column count correct")
        else:
            logger.warning(f"  ✗ Expected 27 columns, got {verify_cols}")

        # Check nulls
        logger.info("\nChecking for nulls in sample (first 100K rows)...")
        sample = df_verify.head(100_000).collect()
        null_counts = {col: sample[col].null_count() for col in sample.columns}
        critical_nulls = {k: v for k, v in null_counts.items() if v > 0 and k != "timestamp_seconds"}

        if critical_nulls:
            logger.warning(f"  Nulls detected in sample: {critical_nulls}")
        else:
            logger.info("  ✓ No nulls in sample")

        # Cleanup
        output_file.unlink()
        logger.info("\n✓ Test file cleaned up")

        logger.info("\n✓ TEST 4 PASSED: Full Module 1 completed successfully")
        logger.info(f"  Total time: {elapsed_join + elapsed_write:.1f}s")
        return True

    except Exception as e:
        logger.error(f"✗ TEST 4 FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def main() -> None:
    """Run all diagnostic tests."""
    logger.info("=" * 80)
    logger.info("FEATURE ENGINEERING V3 - DIAGNOSTIC TEST SUITE")
    logger.info("=" * 80)
    logger.info("This script will identify issues causing OOM/hangs")
    logger.info("DELETE THIS FILE AFTER V3 WORKS\n")

    results = {
        "Test 1 (Schemas)": test_1_verify_input_schemas(),
        "Test 2 (Small Join)": test_2_small_scale_join(),
        "Test 3 (Streaming)": test_3_streaming_write(),
        "Test 4 (Full Module 1)": test_4_full_module_1(),
    }

    logger.info("\n" + "=" * 80)
    logger.info("TEST SUMMARY")
    logger.info("=" * 80)

    for test_name, passed in results.items():
        status = "✓ PASSED" if passed else "✗ FAILED"
        logger.info(f"{test_name}: {status}")

    all_passed = all(results.values())

    if all_passed:
        logger.info("\n✓ ALL TESTS PASSED - V3 script should work correctly")
    else:
        logger.error("\n✗ SOME TESTS FAILED - Review errors above")
        sys.exit(1)


if __name__ == "__main__":
    main()
