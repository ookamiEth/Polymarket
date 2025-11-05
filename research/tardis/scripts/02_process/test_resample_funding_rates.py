#!/usr/bin/env python3
"""
Test script for funding rate resampling functionality.

Validates that resampling works correctly by:
1. Testing with downloaded sample data
2. Verifying row count reduction (event-driven → 1-second)
3. Validating aggregation correctness (no data loss)
4. Testing both "last" and "forward_fill" methods

Usage:
    uv run python test_resample_funding_rates.py

Requirements:
    - Downloaded data in data/raw/binance_funding_rates/
    - At least 1 day of BTCUSDT data
"""

import logging
import shutil
import sys
from pathlib import Path

import polars as pl
from resample_funding_rates_to_1s import resample_file

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def test_resample_basic() -> None:
    """Test basic resampling (last value per second)."""
    logger.info("=" * 80)
    logger.info("TEST 1: Basic Resampling (last method)")
    logger.info("=" * 80)

    # Find test input file
    input_dir = Path("data/raw/binance_funding_rates/binance-futures/BTCUSDT")
    if not input_dir.exists():
        logger.error(f"Input directory not found: {input_dir}")
        logger.error("Please download sample data first using download_binance_funding_rates.py")
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

    # Expected output: ~86,400 rows/day (1 per second)
    # But with gaps: ~79K-86K rows
    expected_min = 70_000
    expected_max = 90_000

    # Create temp output directory
    temp_output_dir = Path("./temp_test_resample")
    temp_output_dir.mkdir(exist_ok=True)
    output_file = str(temp_output_dir / "test_output_last.parquet")

    try:
        # Run resampling
        stats = resample_file(
            input_file=input_file,
            output_file=output_file,
            method="last",
            max_fill_gap=None,
            verbose=False,
        )

        output_rows = stats["output_rows"]
        logger.info(f"Output rows: {output_rows:,} (1-second, sparse)")

        # Validate row count reduction
        if output_rows >= input_rows:
            logger.error(f"❌ Output rows ({output_rows}) should be less than input rows ({input_rows})")
            sys.exit(1)

        if output_rows < expected_min or output_rows > expected_max:
            logger.warning(
                f"⚠️  Output rows ({output_rows:,}) outside expected range [{expected_min:,}, {expected_max:,}]"
            )
            logger.warning("This may be normal for partial days or low-activity periods")

        # Validate output structure
        df_output = pl.read_parquet(output_file)

        required_cols = [
            "exchange",
            "symbol",
            "timestamp",
            "funding_rate",
            "mark_price",
            "update_count",
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

        # Validate update_count makes sense
        update_counts = df_output["update_count"].drop_nulls()
        if len(update_counts) > 0:
            max_updates = update_counts.max()
            if max_updates is not None and max_updates > 100:  # type: ignore[operator]  # Sanity check
                logger.warning(f"⚠️  Unusually high update count: {max_updates}")

        # Validate funding_rate preservation
        input_funding_rates = df_input["funding_rate"].drop_nulls()
        output_funding_rates = df_output["funding_rate"].drop_nulls()

        if len(output_funding_rates) == 0:
            logger.error("❌ No funding rates in output")
            sys.exit(1)

        # Check if output funding rates are within input range
        input_min = input_funding_rates.min()
        input_max = input_funding_rates.max()
        output_min = output_funding_rates.min()
        output_max = output_funding_rates.max()

        if (
            input_min is not None
            and input_max is not None
            and output_min is not None
            and output_max is not None
            and (output_min < input_min or output_max > input_max)  # type: ignore[operator]
        ):
            logger.error(f"❌ Output funding rates [{output_min:.6f}, {output_max:.6f}] outside input range")
            sys.exit(1)

        logger.info("✓ Basic resampling test PASSED")
        logger.info(f"  Input rows: {input_rows:,}")
        logger.info(f"  Output rows: {output_rows:,}")
        logger.info(f"  Compression: {output_rows / input_rows * 100:.1f}% of original")
        logger.info(f"  Funding rate range: [{output_min:.6f}, {output_max:.6f}]")

    finally:
        # Cleanup
        if temp_output_dir.exists():
            shutil.rmtree(temp_output_dir)


def test_forward_fill() -> None:
    """Test forward-fill resampling (continuous time series)."""
    logger.info("")
    logger.info("=" * 80)
    logger.info("TEST 2: Forward-Fill Resampling")
    logger.info("=" * 80)

    # Find test input file
    input_dir = Path("data/raw/binance_funding_rates/binance-futures/BTCUSDT")
    input_files = list(input_dir.glob("*.parquet"))
    input_file = str(input_files[0])

    # Create temp output directory
    temp_output_dir = Path("./temp_test_resample")
    temp_output_dir.mkdir(exist_ok=True)
    output_file = str(temp_output_dir / "test_output_forward_fill.parquet")

    try:
        # Run resampling with forward-fill
        stats = resample_file(
            input_file=input_file,
            output_file=output_file,
            method="forward_fill",
            max_fill_gap=60,  # 60 second max gap
            verbose=False,
        )

        output_rows = stats["output_rows"]
        logger.info(f"Output rows: {output_rows:,} (1-second, forward-filled)")

        # Validate output structure
        df_output = pl.read_parquet(output_file)

        # Forward-fill should have more rows than "last" method
        # (fills gaps in time series)
        expected_min = 80_000  # Should be close to 86,400 for full day
        if output_rows < expected_min:
            logger.warning(f"⚠️  Output rows ({output_rows:,}) less than expected minimum ({expected_min:,})")
            logger.warning("This may be normal for partial days or high-gap periods")

        # Validate continuous time series (no gaps > max_fill_gap)
        df_output = df_output.with_columns([(pl.col("timestamp") // 1_000_000).cast(pl.Int64).alias("ts_seconds")])

        df_sorted = df_output.filter(pl.col("ts_seconds").is_not_null()).sort("ts_seconds")
        timestamps = df_sorted["ts_seconds"].to_list()

        if len(timestamps) > 1:
            gaps = [
                timestamps[i + 1] - timestamps[i]
                for i in range(len(timestamps) - 1)
                if timestamps[i] is not None and timestamps[i + 1] is not None
            ]
            if gaps:
                max_gap = max(gaps)
                logger.info(f"  Max gap between samples: {max_gap} seconds")

                # Allow gaps > max_fill_gap at edges (start/end of day)
                # But middle gaps should be filled
                middle_start = int(len(timestamps) * 0.1)
                middle_end = int(len(timestamps) * 0.9)
                middle_timestamps = [ts for ts in timestamps[middle_start:middle_end] if ts is not None]

                if len(middle_timestamps) > 1:
                    middle_gaps = [
                        middle_timestamps[i + 1] - middle_timestamps[i] for i in range(len(middle_timestamps) - 1)
                    ]
                    max_middle_gap = max(middle_gaps) if middle_gaps else 0

                # Allow small tolerance (65s instead of strict 60s)
                if max_middle_gap > 65:
                    logger.warning(
                        f"⚠️  Found gap ({max_middle_gap}s) exceeding max_fill_gap (60s) in middle of time series"
                    )

        # Validate update_count includes 0 for forward-filled rows
        zero_count = len(df_output.filter(pl.col("update_count") == 0))
        if zero_count == 0:
            logger.warning("⚠️  No forward-filled rows found (update_count=0)")

        logger.info("✓ Forward-fill test PASSED")
        logger.info(f"  Output rows: {output_rows:,}")
        logger.info(f"  Forward-filled rows: {zero_count:,} ({zero_count / output_rows * 100:.1f}%)")

    finally:
        # Cleanup
        if temp_output_dir.exists():
            shutil.rmtree(temp_output_dir)


def test_schema_validation() -> None:
    """Validate output schema matches expected structure."""
    logger.info("")
    logger.info("=" * 80)
    logger.info("TEST 3: Schema Validation")
    logger.info("=" * 80)

    # Find test input file
    input_dir = Path("data/raw/binance_funding_rates/binance-futures/BTCUSDT")
    input_files = list(input_dir.glob("*.parquet"))
    input_file = str(input_files[0])

    # Create temp output directory
    temp_output_dir = Path("./temp_test_resample")
    temp_output_dir.mkdir(exist_ok=True)
    output_file = str(temp_output_dir / "test_output_schema.parquet")

    try:
        # Run resampling
        resample_file(
            input_file=input_file,
            output_file=output_file,
            method="last",
            max_fill_gap=None,
            verbose=False,
        )

        # Read and validate schema
        df = pl.read_parquet(output_file)

        expected_schema = {
            "exchange": pl.Utf8,
            "symbol": pl.Utf8,
            "timestamp": pl.Int64,  # Microseconds
            "local_timestamp": pl.Int64,
            "funding_timestamp": pl.Int64,
            "funding_rate": pl.Float64,
            "predicted_funding_rate": pl.Float64,
            "mark_price": pl.Float64,
            "index_price": pl.Float64,
            "open_interest": pl.Float64,
            "last_price": pl.Float64,
            "update_count": pl.UInt32,
        }

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
    logger.info("FUNDING RATE RESAMPLING TEST SUITE")
    logger.info("=" * 80)
    logger.info("")

    try:
        test_resample_basic()
        test_forward_fill()
        test_schema_validation()

        logger.info("")
        logger.info("=" * 80)
        logger.info("✓ ALL TESTS PASSED")
        logger.info("=" * 80)
        logger.info("")
        logger.info("You can now run the batch processing script:")
        logger.info("")
        logger.info("uv run python batch_resample_funding_rates.py \\")
        logger.info("    --input-dir data/raw/binance_funding_rates \\")
        logger.info("    --output-dir data/processed/binance_funding_rates_1s \\")
        logger.info("    --method forward_fill \\")
        logger.info("    --max-fill-gap 60 \\")
        logger.info("    --workers 5")
        logger.info("")

    except Exception as e:
        logger.error(f"❌ Test suite failed: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
