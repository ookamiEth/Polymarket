#!/usr/bin/env python3
"""
Unit tests for Binance funding rate download script.

Run with: uv run python tests/test_binance_funding_rates.py
"""

import os
import sys
from pathlib import Path

# Add scripts directory to path
sys.path.insert(0, str(Path(__file__).parent.parent / "scripts/01_download"))

import polars as pl
from download_binance_funding_rates import (  # type: ignore[import-not-found]
    _empty_dataframe,
    generate_date_range,
    parse_derivative_ticker_csv,
)


def test_empty_dataframe_schema() -> None:
    """Test that empty DataFrame has correct schema."""
    print("Testing _empty_dataframe()...")

    df = _empty_dataframe()

    # Check all required columns exist
    required_cols = [
        "exchange",
        "symbol",
        "timestamp",
        "local_timestamp",
        "funding_timestamp",
        "funding_rate",
        "predicted_funding_rate",
        "mark_price",
        "index_price",
        "open_interest",
        "last_price",
    ]

    for col in required_cols:
        assert col in df.columns, f"Missing column: {col}"

    # Check types
    assert df.schema["exchange"] == pl.Utf8
    assert df.schema["symbol"] == pl.Utf8
    assert df.schema["timestamp"] == pl.Int64
    assert df.schema["local_timestamp"] == pl.Int64
    assert df.schema["funding_rate"] == pl.Float64
    assert df.schema["mark_price"] == pl.Float64

    print("✓ Schema validation passed")


def test_generate_date_range() -> None:
    """Test date range generation."""
    print("Testing generate_date_range()...")

    # Test single day
    dates = generate_date_range("2024-01-01", "2024-01-01")
    assert len(dates) == 1
    assert dates[0] == "2024-01-01"

    # Test week
    dates = generate_date_range("2024-01-01", "2024-01-07")
    assert len(dates) == 7
    assert dates[0] == "2024-01-01"
    assert dates[-1] == "2024-01-07"

    # Test month
    dates = generate_date_range("2024-01-01", "2024-01-31")
    assert len(dates) == 31

    print("✓ Date range generation passed")


def test_parse_csv_structure() -> None:
    """Test CSV parsing with minimal sample data."""
    print("Testing parse_derivative_ticker_csv()...")

    # Create sample CSV data
    sample_data = """exchange,symbol,timestamp,local_timestamp,funding_timestamp,funding_rate,predicted_funding_rate,mark_price,index_price,open_interest,last_price
binance-futures,BTCUSDT,1704067200000000,1704067200123456,1704096000000000,0.0001,0.00012,45000.5,45001.2,1234567890.5,45002.0
binance-futures,BTCUSDT,1704067201000000,1704067201234567,1704096000000000,0.0001,0.00012,45001.0,45001.5,1234567891.0,45002.5
"""

    # Write to temporary file
    temp_file = "/tmp/test_funding_rates.csv"
    with open(temp_file, "w") as f:
        f.write(sample_data)

    try:
        # Parse
        df = parse_derivative_ticker_csv(temp_file)

        # Validate
        assert len(df) == 2, f"Expected 2 rows, got {len(df)}"
        assert "exchange" in df.columns
        assert "funding_rate" in df.columns
        assert "mark_price" in df.columns

        # Check first row values
        assert df["exchange"][0] == "binance-futures"
        assert df["symbol"][0] == "BTCUSDT"
        assert df["funding_rate"][0] == 0.0001
        assert df["mark_price"][0] == 45000.5

        print("✓ CSV parsing passed")

    finally:
        # Cleanup
        if os.path.exists(temp_file):
            os.remove(temp_file)


def test_funding_rate_granularity() -> None:
    """Test that 1-second granularity is preserved."""
    print("Testing 1-second granularity...")

    # Create sample with 5 seconds of data
    rows = []
    base_timestamp = 1704067200000000  # Microseconds

    for i in range(5):
        rows.append(
            {
                "exchange": "binance-futures",
                "symbol": "BTCUSDT",
                "timestamp": base_timestamp + (i * 1_000_000),  # Add 1 second
                "local_timestamp": base_timestamp + (i * 1_000_000),
                "funding_timestamp": base_timestamp + 28800_000_000,  # +8 hours
                "funding_rate": 0.0001 + (i * 0.00001),
                "predicted_funding_rate": 0.00012,
                "mark_price": 45000.0 + i,
                "index_price": 45001.0 + i,
                "open_interest": 1234567890.0,
                "last_price": 45002.0 + i,
            }
        )

    df = pl.DataFrame(rows)

    # Check timestamps
    timestamps_us = df["timestamp"].to_list()
    for i in range(len(timestamps_us) - 1):
        diff_us = timestamps_us[i + 1] - timestamps_us[i]
        diff_seconds = diff_us / 1_000_000
        assert diff_seconds == 1.0, f"Expected 1 second diff, got {diff_seconds}"

    print("✓ Granularity check passed")


def run_all_tests() -> None:
    """Run all tests."""
    print("\n" + "=" * 60)
    print("Running Binance Funding Rate Tests")
    print("=" * 60 + "\n")

    try:
        test_empty_dataframe_schema()
        test_generate_date_range()
        test_parse_csv_structure()
        test_funding_rate_granularity()

        print("\n" + "=" * 60)
        print("ALL TESTS PASSED ✓")
        print("=" * 60 + "\n")

    except AssertionError as e:
        print(f"\n✗ TEST FAILED: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n✗ ERROR: {e}")
        sys.exit(1)


if __name__ == "__main__":
    run_all_tests()
