#!/usr/bin/env python3
"""
Test script for Module 7b rolling window null fix.

Verifies that filtering rv_900s nulls before rolling_quantile_by operations:
1. Removes exactly 900 warmup rows
2. Allows rolling operations to complete without InvalidOperationError
3. Produces expected output with correct null patterns
"""

import logging
from pathlib import Path

import polars as pl

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def test_null_detection():
    """Test 1: Detect nulls in input data."""
    logger.info("=" * 80)
    logger.info("TEST 1: Null Detection in Input Data")
    logger.info("=" * 80)

    rv_file = Path(
        "/home/ubuntu/Polymarket/research/model/data/features_v4/intermediate/07_rv_momentum_range_features.parquet"
    )
    existing_file = Path(
        "/home/ubuntu/Polymarket/research/model/data/features_v4/intermediate/01_existing_features.parquet"
    )

    logger.info(f"Loading RV features: {rv_file}")
    df = pl.scan_parquet(rv_file).select(["timestamp_seconds", "rv_60s", "rv_900s"])

    logger.info(f"Loading S and K from: {existing_file}")
    existing = (
        pl.scan_parquet(existing_file)
        .select(["timestamp_seconds", "S", "K"])
        .with_columns([((pl.col("S") / pl.col("K")) - 1).abs().alias("moneyness_distance")])
        .select(["timestamp_seconds", "moneyness_distance"])
    )

    df = df.join(existing, on="timestamp_seconds", how="left")

    # Compute null counts
    logger.info("Computing null statistics...")
    null_stats = df.select(
        [
            pl.len().alias("total_rows"),
            pl.col("rv_60s").is_null().sum().alias("rv_60s_nulls"),
            pl.col("rv_900s").is_null().sum().alias("rv_900s_nulls"),
        ]
    ).collect()

    total_rows = null_stats["total_rows"][0]
    rv_60s_nulls = null_stats["rv_60s_nulls"][0]
    rv_900s_nulls = null_stats["rv_900s_nulls"][0]

    logger.info(f"✓ Total rows: {total_rows:,}")
    logger.info(f"  rv_60s nulls: {rv_60s_nulls:,} ({rv_60s_nulls / total_rows * 100:.4f}%)")
    logger.info(f"  rv_900s nulls: {rv_900s_nulls:,} ({rv_900s_nulls / total_rows * 100:.4f}%)")

    # Verify expected null counts
    assert rv_60s_nulls == 60, f"Expected 60 nulls in rv_60s, got {rv_60s_nulls}"
    assert rv_900s_nulls == 900, f"Expected 900 nulls in rv_900s, got {rv_900s_nulls}"
    logger.info("✓ Test 1 PASSED: rv_60s has 60 nulls, rv_900s has 900 nulls")

    return True


def test_filter_effectiveness():
    """Test 2: Verify filter removes all nulls."""
    logger.info("=" * 80)
    logger.info("TEST 2: Filter Effectiveness")
    logger.info("=" * 80)

    rv_file = Path(
        "/home/ubuntu/Polymarket/research/model/data/features_v4/intermediate/07_rv_momentum_range_features.parquet"
    )
    existing_file = Path(
        "/home/ubuntu/Polymarket/research/model/data/features_v4/intermediate/01_existing_features.parquet"
    )

    df = pl.scan_parquet(rv_file).select(["timestamp_seconds", "rv_60s", "rv_900s"])
    existing = (
        pl.scan_parquet(existing_file)
        .select(["timestamp_seconds", "S", "K"])
        .with_columns([((pl.col("S") / pl.col("K")) - 1).abs().alias("moneyness_distance")])
        .select(["timestamp_seconds", "moneyness_distance"])
    )
    df = df.join(existing, on="timestamp_seconds", how="left")

    # Sort and filter
    logger.info("Sorting by timestamp...")
    df = df.sort("timestamp_seconds")

    logger.info("Applying filter: pl.col('rv_60s').is_not_null() & pl.col('rv_900s').is_not_null()")
    df_filtered = df.filter(pl.col("rv_60s").is_not_null() & pl.col("rv_900s").is_not_null())

    # Check row counts
    row_counts = pl.DataFrame(
        {
            "dataset": ["before_filter", "after_filter"],
            "row_count": [df.select(pl.len()).collect()["len"][0], df_filtered.select(pl.len()).collect()["len"][0]],
        }
    )

    rows_before = row_counts["row_count"][0]
    rows_after = row_counts["row_count"][1]
    rows_dropped = rows_before - rows_after

    logger.info(f"  Rows before filter: {rows_before:,}")
    logger.info(f"  Rows after filter: {rows_after:,}")
    logger.info(f"  Rows dropped: {rows_dropped:,}")

    assert rows_dropped == 900, f"Expected 900 rows dropped, got {rows_dropped}"
    logger.info("✓ Test 2 PASSED: Filter removed exactly 900 null rows")

    # Verify no nulls remain in BOTH columns
    null_check = df_filtered.select(
        [
            pl.col("rv_60s").is_null().sum().alias("rv_60s_nulls"),
            pl.col("rv_900s").is_null().sum().alias("rv_900s_nulls"),
        ]
    ).collect()

    rv_60s_remaining = null_check["rv_60s_nulls"][0]
    rv_900s_remaining = null_check["rv_900s_nulls"][0]
    assert rv_60s_remaining == 0, f"Expected 0 rv_60s nulls after filter, got {rv_60s_remaining}"
    assert rv_900s_remaining == 0, f"Expected 0 rv_900s nulls after filter, got {rv_900s_remaining}"
    logger.info("✓ Test 2 PASSED: No nulls remain in rv_60s or rv_900s after filter")

    return True


def test_rolling_operation_with_streaming():
    """Test 3: Verify rolling operations work with streaming write."""
    logger.info("=" * 80)
    logger.info("TEST 3: Rolling Operations with Streaming Write")
    logger.info("=" * 80)

    rv_file = Path(
        "/home/ubuntu/Polymarket/research/model/data/features_v4/intermediate/07_rv_momentum_range_features.parquet"
    )
    existing_file = Path(
        "/home/ubuntu/Polymarket/research/model/data/features_v4/intermediate/01_existing_features.parquet"
    )
    output_file = Path(
        "/home/ubuntu/Polymarket/research/model/data/features_v4/intermediate/TEST_07b_extreme_regime_features.parquet"
    )

    # Remove output file if exists
    if output_file.exists():
        logger.info(f"Removing existing test output: {output_file}")
        output_file.unlink()

    logger.info("Building lazy frame with filter and rolling operations...")
    df = pl.scan_parquet(rv_file).select(["timestamp_seconds", "rv_60s", "rv_900s"])
    existing = (
        pl.scan_parquet(existing_file)
        .select(["timestamp_seconds", "S", "K"])
        .with_columns([((pl.col("S") / pl.col("K")) - 1).abs().alias("moneyness_distance")])
        .select(["timestamp_seconds", "moneyness_distance"])
    )
    df = df.join(existing, on="timestamp_seconds", how="left")

    # Sort and filter (FIX: filter BOTH rv_60s and rv_900s)
    df = df.sort("timestamp_seconds")
    df = df.filter(pl.col("rv_60s").is_not_null() & pl.col("rv_900s").is_not_null())

    # Apply rolling operations (exact same as Module 7b)
    # NOTE: window_size must be in seconds when using integer timestamp column
    # 30 days = 30 * 24 * 60 * 60 = 2,592,000 seconds
    df = df.with_columns(
        [
            # Compute RV ratio
            (pl.col("rv_60s") / (pl.col("rv_900s") + 1e-10)).alias("rv_ratio"),
            # Compute 95th percentile of RV_900s over last 30 DAYS (2,592,000 seconds)
            pl.col("rv_900s")
            .rolling_quantile_by(by="timestamp_seconds", window_size="2592000i", quantile=0.95)
            .alias("rv_95th_percentile"),
        ]
    )

    # Flag extreme conditions
    df = df.with_columns(
        [
            pl.when((pl.col("rv_ratio") > 3) | (pl.col("rv_900s") > pl.col("rv_95th_percentile")))
            .then(pl.lit(True))
            .otherwise(pl.lit(False))
            .alias("is_extreme_condition"),
            pl.when(pl.col("rv_ratio") > 3).then(pl.lit(0.5)).otherwise(pl.lit(1.0)).alias("position_scale"),
        ]
    )

    # Monthly percentile computation (two more rolling operations)
    # 30 days = 2,592,000 seconds
    df = df.with_columns(
        [
            pl.col("rv_900s")
            .rolling_quantile_by(by="timestamp_seconds", window_size="2592000i", quantile=0.33)
            .alias("vol_low_thresh"),
            pl.col("rv_900s")
            .rolling_quantile_by(by="timestamp_seconds", window_size="2592000i", quantile=0.67)
            .alias("vol_high_thresh"),
        ]
    )

    # Add regime columns
    df = df.with_columns(
        [
            pl.when(pl.col("rv_900s") < pl.col("vol_low_thresh"))
            .then(pl.lit("low"))
            .when(pl.col("rv_900s") > pl.col("vol_high_thresh"))
            .then(pl.lit("high"))
            .otherwise(pl.lit("medium"))
            .alias("volatility_regime"),
            pl.when((pl.col("rv_900s") < pl.col("vol_low_thresh")) & (pl.col("moneyness_distance") < 0.01))
            .then(pl.lit("low_vol_atm"))
            .when((pl.col("rv_900s") < pl.col("vol_low_thresh")) & (pl.col("moneyness_distance") >= 0.01))
            .then(pl.lit("low_vol_otm"))
            .when((pl.col("rv_900s") > pl.col("vol_high_thresh")) & (pl.col("moneyness_distance") < 0.01))
            .then(pl.lit("high_vol_atm"))
            .when((pl.col("rv_900s") > pl.col("vol_high_thresh")) & (pl.col("moneyness_distance") >= 0.01))
            .then(pl.lit("high_vol_otm"))
            .otherwise(pl.lit("medium_vol"))
            .alias("market_regime"),
        ]
    )

    # Select final columns
    df = df.select(
        [
            "timestamp_seconds",
            "rv_ratio",
            "rv_95th_percentile",
            "is_extreme_condition",
            "position_scale",
            "vol_low_thresh",
            "vol_high_thresh",
            "volatility_regime",
            "market_regime",
        ]
    )

    # Write with streaming (this should NOT throw InvalidOperationError)
    logger.info(f"Writing to: {output_file}")
    logger.info("  (Using streaming write - should complete without errors)")
    try:
        df.sink_parquet(output_file, compression="snappy")
        logger.info("✓ Test 3 PASSED: Streaming write completed successfully!")
    except Exception as e:
        logger.error(f"✗ Test 3 FAILED: {e}")
        return False

    # Validate output
    logger.info("Validating output file...")
    output_df = pl.read_parquet(output_file)
    row_count = len(output_df)
    col_count = len(output_df.columns)

    logger.info(f"  Output rows: {row_count:,}")
    logger.info(f"  Output columns: {col_count}")

    # Expected: 63,158,387 - 900 = 63,157,487
    expected_rows = 63_157_487
    assert abs(row_count - expected_rows) < 10_000, f"Row count {row_count:,} too far from expected {expected_rows:,}"
    assert col_count == 9, f"Expected 9 columns, got {col_count}"

    logger.info("✓ Test 3 PASSED: Output validation successful")

    # Check null patterns
    logger.info("Checking null patterns in output...")
    null_summary = output_df.select(
        [
            pl.col("rv_ratio").is_null().sum().alias("rv_ratio_nulls"),
            pl.col("rv_95th_percentile").is_null().sum().alias("rv_95th_percentile_nulls"),
            pl.col("is_extreme_condition").is_null().sum().alias("is_extreme_condition_nulls"),
            pl.col("position_scale").is_null().sum().alias("position_scale_nulls"),
        ]
    )

    logger.info(f"  Null counts: {null_summary}")

    # rv_ratio, is_extreme_condition, position_scale should have 0 nulls
    assert null_summary["rv_ratio_nulls"][0] == 0, "rv_ratio should have no nulls"
    assert null_summary["is_extreme_condition_nulls"][0] == 0, "is_extreme_condition should have no nulls"
    assert null_summary["position_scale_nulls"][0] == 0, "position_scale should have no nulls"

    logger.info("✓ Test 3 PASSED: Null patterns correct")

    # Cleanup
    logger.info(f"Cleaning up test output: {output_file}")
    output_file.unlink()

    return True


def main():
    """Run all tests."""
    logger.info("\n" + "=" * 80)
    logger.info("Module 7b Fix Verification Test Suite")
    logger.info("=" * 80 + "\n")

    tests = [
        ("Null Detection", test_null_detection),
        ("Filter Effectiveness", test_filter_effectiveness),
        ("Rolling Operations with Streaming", test_rolling_operation_with_streaming),
    ]

    passed = 0
    failed = 0

    for test_name, test_func in tests:
        try:
            if test_func():
                passed += 1
                logger.info(f"\n✓ {test_name} PASSED\n")
            else:
                failed += 1
                logger.error(f"\n✗ {test_name} FAILED\n")
        except Exception as e:
            failed += 1
            logger.error(f"\n✗ {test_name} FAILED with exception: {e}\n")

    logger.info("=" * 80)
    logger.info(f"Test Summary: {passed} passed, {failed} failed")
    logger.info("=" * 80)

    if failed == 0:
        logger.info("✓ ALL TESTS PASSED - Fix is ready to apply!")
        return 0
    else:
        logger.error(f"✗ {failed} TEST(S) FAILED - Fix needs adjustment")
        return 1


if __name__ == "__main__":
    exit(main())
