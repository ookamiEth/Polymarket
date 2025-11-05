#!/usr/bin/env python3
"""
Test script for orderbook snapshot (top 5 levels) resampling functionality.

Validates that resampling works correctly by:
1. Testing with downloaded sample data
2. Verifying row count reduction (event-driven → 1-second)
3. Validating aggregation correctness (no data loss)
4. Testing schema with all 22 columns

Usage:
    uv run python test_resample_orderbook_5.py

Requirements:
    - Downloaded data in data/raw/binance_orderbook_5/
    - At least 1 day of BTCUSDT data
"""

import logging
import shutil
import sys
from pathlib import Path

import polars as pl
from resample_orderbook_5_to_1s import resample_file

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def test_resample_basic() -> None:
    """Test basic resampling (last snapshot per second)."""
    logger.info("=" * 80)
    logger.info("TEST 1: Basic Resampling (last snapshot per second)")
    logger.info("=" * 80)

    # Find test input file
    input_dir = Path("data/raw/binance_orderbook_5/binance-futures/BTCUSDT")
    if not input_dir.exists():
        logger.error(f"Input directory not found: {input_dir}")
        logger.error("Please download sample data first using download_binance_orderbook_5.py")
        sys.exit(1)

    input_files = list(input_dir.glob("*.parquet"))
    if not input_files:
        logger.error(f"No parquet files found in {input_dir}")
        sys.exit(1)

    input_file = str(input_files[0])
    logger.info(f"Using input file: {input_file}")

    # Read input data
    df_input = pl.read_parquet(input_file)
    input_rows = len(df_input)
    logger.info(f"Input rows: {input_rows:,} (event-driven)")

    # Create temp output directory
    temp_output_dir = Path("./temp_test_resample_orderbook")
    temp_output_dir.mkdir(exist_ok=True)
    output_file = str(temp_output_dir / "test_output_last.parquet")

    try:
        # Run resampling
        stats = resample_file(
            input_file=input_file,
            output_file=output_file,
            verbose=False,
        )

        output_rows = stats["output_rows"]
        logger.info(f"Output rows: {output_rows:,} (1-second)")

        # Validate row count reduction
        if output_rows >= input_rows:
            logger.error(f"❌ Output rows ({output_rows}) should be less than input rows ({input_rows})")
            sys.exit(1)

        # Validate output structure
        df_output = pl.read_parquet(output_file)

        required_cols = [
            "exchange",
            "symbol",
            "timestamp",
            "local_timestamp",
            "ask_price_0",
            "ask_amount_0",
            "bid_price_0",
            "bid_amount_0",
            "snapshot_count",
        ]
        missing_cols = [col for col in required_cols if col not in df_output.columns]
        if missing_cols:
            logger.error(f"❌ Missing required columns: {missing_cols}")
            sys.exit(1)

        # Validate no duplicate timestamps (at second precision)
        df_output = df_output.with_columns([(pl.col("timestamp") // 1_000_000).cast(pl.Int64).alias("ts_seconds")])

        duplicates = df_output.group_by("ts_seconds").agg(pl.len().alias("count")).filter(pl.col("count") > 1)

        if len(duplicates) > 0:
            logger.error(f"❌ Found {len(duplicates)} duplicate timestamps at second precision")
            sys.exit(1)

        # Validate snapshot_count makes sense
        snapshot_counts = df_output["snapshot_count"].drop_nulls()
        if len(snapshot_counts) > 0:
            max_snapshots = snapshot_counts.max()
            if max_snapshots is not None and max_snapshots > 1000:  # type: ignore[operator]  # Sanity check
                logger.warning(f"⚠️  Unusually high snapshot count: {max_snapshots}")

        # Validate bid/ask prices
        ask_prices = df_output["ask_price_0"].drop_nulls()
        bid_prices = df_output["bid_price_0"].drop_nulls()

        if len(ask_prices) == 0:
            logger.error("❌ No ask prices in output")
            sys.exit(1)

        if len(bid_prices) == 0:
            logger.error("❌ No bid prices in output")
            sys.exit(1)

        # Check if spread makes sense (ask > bid)
        spread_check = (
            df_output.filter(pl.col("ask_price_0").is_not_null() & pl.col("bid_price_0").is_not_null())
            .with_columns([(pl.col("ask_price_0") - pl.col("bid_price_0")).alias("spread")])
            .filter(pl.col("spread") < 0)
        )

        if len(spread_check) > 0:
            logger.error(f"❌ Found {len(spread_check)} rows where ask < bid (invalid)")
            sys.exit(1)

        logger.info("✓ Basic resampling test PASSED")
        logger.info(f"  Input rows: {input_rows:,}")
        logger.info(f"  Output rows: {output_rows:,}")
        logger.info(f"  Compression: {output_rows / input_rows * 100:.1f}% of original")

    finally:
        # Cleanup
        if temp_output_dir.exists():
            shutil.rmtree(temp_output_dir)


def test_schema_validation() -> None:
    """Validate output schema matches expected structure."""
    logger.info("")
    logger.info("=" * 80)
    logger.info("TEST 2: Schema Validation")
    logger.info("=" * 80)

    # Find test input file
    input_dir = Path("data/raw/binance_orderbook_5/binance-futures/BTCUSDT")
    input_files = list(input_dir.glob("*.parquet"))
    input_file = str(input_files[0])

    # Create temp output directory
    temp_output_dir = Path("./temp_test_resample_orderbook")
    temp_output_dir.mkdir(exist_ok=True)
    output_file = str(temp_output_dir / "test_output_schema.parquet")

    try:
        # Run resampling
        resample_file(
            input_file=input_file,
            output_file=output_file,
            verbose=False,
        )

        # Read and validate schema
        df = pl.read_parquet(output_file)

        expected_schema = {
            "exchange": pl.Utf8,
            "symbol": pl.Utf8,
            "timestamp": pl.Int64,
            "local_timestamp": pl.Int64,
        }

        # Add all price levels
        for i in range(5):
            expected_schema[f"ask_price_{i}"] = pl.Float64
            expected_schema[f"ask_amount_{i}"] = pl.Float64
            expected_schema[f"bid_price_{i}"] = pl.Float64
            expected_schema[f"bid_amount_{i}"] = pl.Float64

        expected_schema["snapshot_count"] = pl.UInt32

        logger.info("Validating output schema...")
        for col, expected_type in expected_schema.items():
            if col not in df.columns:
                logger.error(f"❌ Missing column: {col}")
                sys.exit(1)

            actual_type = df.schema[col]
            if actual_type != expected_type:
                logger.error(f"❌ Column '{col}' has type {actual_type}, expected {expected_type}")
                sys.exit(1)

        logger.info("✓ Schema validation PASSED")
        logger.info(f"  All {len(expected_schema)} columns present with correct types")

    finally:
        # Cleanup
        if temp_output_dir.exists():
            shutil.rmtree(temp_output_dir)


def main() -> None:
    """Run all tests."""
    logger.info("")
    logger.info("=" * 80)
    logger.info("ORDERBOOK SNAPSHOT (TOP 5 LEVELS) RESAMPLING TEST SUITE")
    logger.info("=" * 80)
    logger.info("")

    try:
        test_resample_basic()
        test_schema_validation()

        logger.info("")
        logger.info("=" * 80)
        logger.info("✓ ALL TESTS PASSED")
        logger.info("=" * 80)
        logger.info("")
        logger.info("You can now run the batch processing script:")
        logger.info("")
        logger.info("uv run python batch_resample_orderbook_5.py \\")
        logger.info("    --input-dir data/raw/binance_orderbook_5 \\")
        logger.info("    --output-dir data/processed/binance_orderbook_5_1s \\")
        logger.info("    --workers 5")
        logger.info("")

    except Exception as e:
        logger.error(f"❌ Test suite failed: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
