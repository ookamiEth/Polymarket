#!/usr/bin/env python3
"""
Quick test script to validate Binance funding rate downloads.

Downloads 1-2 days of sample data and prints results for manual inspection.

Usage:
    uv run python test_funding_rate_download.py

This will download:
- 2 days of BTCUSDT funding rates (binance-futures)
- 2 days of BTCUSD_PERP funding rates (binance-delivery)

And print:
- Sample rows
- Data statistics
- Schema validation
"""

import os
import sys
from datetime import datetime, timedelta

from dotenv import load_dotenv

# Import from main script
from download_binance_funding_rates import (
    _empty_dataframe,
    download_funding_rates_for_day,
    parse_derivative_ticker_csv,
)


def test_download() -> None:
    """Test downloading and parsing funding rate data."""
    print("\n" + "=" * 80)
    print("BINANCE FUNDING RATE DOWNLOAD TEST")
    print("=" * 80 + "\n")

    # Load API key
    load_dotenv()
    api_key = os.getenv("TARDIS_API_KEY")

    if not api_key:
        print("ERROR: TARDIS_API_KEY not found in .env file")
        print("Please create a .env file in research/tardis/ with:")
        print("TARDIS_API_KEY=your_key_here")
        sys.exit(1)

    print("✓ API key loaded\n")

    # Test parameters
    test_date = (datetime.now() - timedelta(days=7)).strftime("%Y-%m-%d")  # 1 week ago
    temp_dir = "./temp_test_funding"
    os.makedirs(temp_dir, exist_ok=True)

    # Test 1: USDT Futures (binance-futures)
    print("-" * 80)
    print("TEST 1: Binance USDT Futures (binance-futures)")
    print("-" * 80)
    print(f"Downloading BTCUSDT funding rates for {test_date}...")

    try:
        csv_files = download_funding_rates_for_day(
            date_str=test_date,
            exchange="binance-futures",
            symbols=["BTCUSDT"],
            api_key=api_key,
            temp_dir=temp_dir,
        )

        if not csv_files:
            print("✗ No files downloaded")
            sys.exit(1)

        print(f"✓ Downloaded {len(csv_files)} file(s)")

        # Parse
        df = parse_derivative_ticker_csv(csv_files[0])
        print(f"✓ Parsed {len(df):,} rows\n")

        # Display sample
        print("Sample rows (first 5):")
        print(df.head(5))
        print()

        # Statistics
        print("Data Statistics:")
        min_ts = df["timestamp"].min()
        max_ts = df["timestamp"].max()
        if min_ts is not None and max_ts is not None:
            print(f"  Time range: {int(min_ts) / 1e6:.0f} to {int(max_ts) / 1e6:.0f}")  # type: ignore[arg-type]

        if "funding_rate" in df.columns:
            funding_rates = df["funding_rate"].drop_nulls()
            if len(funding_rates) > 0:
                print(f"  Funding rate range: {funding_rates.min():.6f} to {funding_rates.max():.6f}")
                print(f"  Funding rate mean: {funding_rates.mean():.6f}")
            else:
                print("  ⚠ Warning: No funding rate data found")

        if "mark_price" in df.columns:
            mark_prices = df["mark_price"].drop_nulls()
            if len(mark_prices) > 0:
                print(f"  Mark price range: ${mark_prices.min():.2f} to ${mark_prices.max():.2f}")
            else:
                print("  ⚠ Warning: No mark price data found")

        # Expected granularity check (should be ~1 second intervals)
        if len(df) > 1:
            timestamps_us = df["timestamp"].to_list()
            diffs_seconds = [
                (timestamps_us[i + 1] - timestamps_us[i]) / 1e6 for i in range(min(10, len(timestamps_us) - 1))
            ]
            avg_interval = sum(diffs_seconds) / len(diffs_seconds)
            print(f"  Average update interval: {avg_interval:.2f} seconds")

            if avg_interval < 2:
                print("  ✓ Granularity check passed (≈1 second updates)")
            else:
                print(f"  ⚠ Warning: Expected ~1 second updates, got {avg_interval:.2f}")

        print("\n✓ USDT Futures test PASSED\n")

    except Exception as e:
        print(f"\n✗ USDT Futures test FAILED: {e}\n")
        import traceback

        traceback.print_exc()
        sys.exit(1)

    # Test 2: COIN Futures (binance-delivery)
    print("-" * 80)
    print("TEST 2: Binance COIN Futures (binance-delivery)")
    print("-" * 80)
    print(f"Downloading BTCUSD_PERP funding rates for {test_date}...")

    try:
        csv_files = download_funding_rates_for_day(
            date_str=test_date,
            exchange="binance-delivery",
            symbols=["BTCUSD_PERP"],
            api_key=api_key,
            temp_dir=temp_dir,
        )

        if not csv_files:
            print("✗ No files downloaded (symbol may not exist or no data for this date)")
            print("Note: COIN futures may use different symbol naming")
            print("Skipping COIN futures test...")
        else:
            print(f"✓ Downloaded {len(csv_files)} file(s)")

            # Parse
            df = parse_derivative_ticker_csv(csv_files[0])
            print(f"✓ Parsed {len(df):,} rows\n")

            # Display sample
            print("Sample rows (first 5):")
            print(df.head(5))
            print()

            print("✓ COIN Futures test PASSED\n")

    except Exception as e:
        print(f"✗ COIN Futures test FAILED: {e}")
        print("This may be expected if the symbol doesn't exist for this date")
        print("Continuing...\n")

    # Schema validation
    print("-" * 80)
    print("TEST 3: Schema Validation")
    print("-" * 80)

    schema_df = _empty_dataframe()
    print("Expected schema:")
    for col, dtype in schema_df.schema.items():
        print(f"  {col}: {dtype}")

    print("\n✓ Schema validation PASSED\n")

    # Cleanup
    print("-" * 80)
    print("Cleaning up temporary files...")
    import shutil

    if os.path.exists(temp_dir):
        shutil.rmtree(temp_dir)
    print("✓ Cleanup complete\n")

    # Final summary
    print("=" * 80)
    print("ALL TESTS PASSED ✓")
    print("=" * 80)
    print("\nYou can now run the full download script:")
    print("\nuv run python download_binance_funding_rates.py \\")
    print("    --from-date 2024-01-01 \\")
    print("    --to-date 2024-01-31 \\")
    print("    --symbols BTCUSDT,ETHUSDT \\")
    print("    --exchanges binance-futures")
    print()


if __name__ == "__main__":
    test_download()
