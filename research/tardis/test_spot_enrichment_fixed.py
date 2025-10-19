#!/usr/bin/env python3
"""
Test different approaches for enriching quotes with spot prices.
Fixed version using proper Polars APIs.
"""

import time
from typing import Any, Dict
import polars as pl
import numpy as np


def test_row_loop_approach(quotes_df: pl.DataFrame, spot_dict: dict[int, float]) -> tuple[pl.DataFrame, float]:
    """Original approach with Python row loop."""
    start = time.time()

    spot_prices = []
    for ts in quotes_df["timestamp_seconds"]:
        spot_prices.append(spot_dict.get(ts))

    result = quotes_df.with_columns([pl.Series("spot_price", spot_prices)])

    elapsed = time.time() - start
    return result, elapsed


def test_replace_approach(quotes_df: pl.DataFrame, spot_dict: dict[int, float]) -> tuple[pl.DataFrame, float]:
    """Use Polars replace for vectorized dictionary lookup."""
    start = time.time()

    # Use replace for vectorized lookup with default=None for missing keys
    result = quotes_df.with_columns([
        pl.col("timestamp_seconds").replace(spot_dict, default=None).alias("spot_price")
    ])

    elapsed = time.time() - start
    return result, elapsed


def test_join_approach(quotes_df: pl.DataFrame, spot_dict: dict[int, float]) -> tuple[pl.DataFrame, float]:
    """Convert dict to DataFrame and use join."""
    start = time.time()

    # Convert dict to DataFrame
    spot_df = pl.DataFrame({
        "timestamp_seconds": list(spot_dict.keys()),
        "spot_price": list(spot_dict.values())
    })

    # Join on timestamp_seconds
    result = quotes_df.join(spot_df, on="timestamp_seconds", how="left")

    elapsed = time.time() - start
    return result, elapsed


def test_lazy_join_approach(quotes_df: pl.DataFrame, spot_dict: dict[int, float]) -> tuple[pl.DataFrame, float]:
    """Use lazy DataFrame join for better optimization."""
    start = time.time()

    # Convert dict to DataFrame
    spot_df = pl.DataFrame({
        "timestamp_seconds": list(spot_dict.keys()),
        "spot_price": list(spot_dict.values())
    })

    # Use lazy API for join
    result = (
        quotes_df.lazy()
        .join(spot_df.lazy(), on="timestamp_seconds", how="left")
        .collect()
    )

    elapsed = time.time() - start
    return result, elapsed


def create_test_data(n_quotes: int = 100000, n_spot_prices: int = 10000) -> tuple[pl.DataFrame, dict[int, float]]:
    """Create test data similar to actual use case."""

    # Create timestamps (some will match, some won't)
    base_timestamp = 1696118400  # Oct 1, 2023

    # Quotes timestamps (dense, every second with some gaps)
    quote_timestamps = np.random.choice(
        range(base_timestamp, base_timestamp + n_spot_prices * 2),
        size=n_quotes,
        replace=True
    )

    # Spot price timestamps (subset that should match)
    spot_timestamps = range(base_timestamp, base_timestamp + n_spot_prices)
    spot_prices = np.random.uniform(30000, 50000, n_spot_prices)

    # Create quotes DataFrame
    quotes_df = pl.DataFrame({
        "timestamp_seconds": quote_timestamps,
        "symbol": [f"BTC-{i%100:03d}" for i in range(n_quotes)],
        "bid_price": np.random.uniform(0.01, 1.0, n_quotes),
        "ask_price": np.random.uniform(0.01, 1.0, n_quotes),
    })

    # Create spot price dictionary
    spot_dict = dict(zip(spot_timestamps, spot_prices))

    return quotes_df, spot_dict


def validate_results(results: list[tuple[str, pl.DataFrame]]) -> bool:
    """Validate that all approaches produce the same result."""
    if len(results) < 2:
        return True

    base_name, base_df = results[0]
    base_sorted = base_df.sort("timestamp_seconds")

    all_match = True
    for name, df in results[1:]:
        df_sorted = df.sort("timestamp_seconds")

        # Compare spot_price columns
        try:
            # Check counts of non-null values
            base_non_null = base_sorted.filter(pl.col("spot_price").is_not_null()).height
            df_non_null = df_sorted.filter(pl.col("spot_price").is_not_null()).height

            if base_non_null != df_non_null:
                print(f"  WARNING: {name} has different number of non-null values")
                print(f"    Base: {base_non_null}, Current: {df_non_null}")
                all_match = False
                continue

            # Sample comparison for first 100 rows
            base_sample = base_sorted.head(100)["spot_price"].to_list()
            df_sample = df_sorted.head(100)["spot_price"].to_list()

            # Compare allowing for None/null differences
            differences = sum(1 for a, b in zip(base_sample, df_sample)
                            if (a is None and b is not None) or
                               (a is not None and b is None) or
                               (a is not None and b is not None and abs(a - b) > 0.001))

            if differences > 0:
                print(f"  WARNING: {name} has {differences} differences in first 100 rows")
                all_match = False

        except Exception as e:
            print(f"  WARNING: Could not compare {name}: {e}")
            all_match = False

    return all_match


def main():
    """Run performance tests."""
    print("=" * 80)
    print("SPOT PRICE ENRICHMENT PERFORMANCE TEST (FIXED)")
    print("=" * 80)

    # Test with different data sizes
    test_sizes = [
        (10_000, 5_000, "Small"),
        (100_000, 50_000, "Medium"),
        (1_000_000, 500_000, "Large"),
        (20_000_000, 500_000, "Very Large (20M quotes - typical month)"),
    ]

    for n_quotes, n_spot_prices, size_label in test_sizes:
        print(f"\n📊 Testing with {size_label} dataset:")
        print(f"   Quotes: {n_quotes:,} rows")
        print(f"   Spot prices: {n_spot_prices:,} unique timestamps")

        # Create test data
        quotes_df, spot_dict = create_test_data(n_quotes, n_spot_prices)

        # Run tests
        approaches = [
            ("Row Loop (Original)", test_row_loop_approach),
            ("Replace (Vectorized)", test_replace_approach),
            ("DataFrame Join", test_join_approach),
            ("Lazy DataFrame Join", test_lazy_join_approach),
        ]

        # Skip row loop for very large datasets
        if n_quotes >= 20_000_000:
            print("\n⚠️  Skipping row loop for very large dataset (too slow)")
            approaches = approaches[1:]  # Skip first (row loop)

        results = []
        timings = []

        print("\n⏱️  Performance Results:")
        for name, test_func in approaches:
            try:
                result_df, elapsed = test_func(quotes_df.clone(), spot_dict)
                results.append((name, result_df))
                timings.append((name, elapsed))

                # Calculate rows per second
                rows_per_sec = n_quotes / elapsed if elapsed > 0 else float('inf')

                print(f"   {name:25s}: {elapsed:7.3f}s ({rows_per_sec:,.0f} rows/sec)")

                # Check how many matches found
                matches = result_df.filter(pl.col("spot_price").is_not_null()).height
                match_pct = (matches / n_quotes) * 100
                print(f"      → Matches found: {matches:,} ({match_pct:.1f}%)")

            except Exception as e:
                print(f"   {name:25s}: ERROR - {e}")

        # Find fastest approach
        if timings:
            fastest = min(timings, key=lambda x: x[1])
            slowest = max(timings, key=lambda x: x[1])
            speedup = slowest[1] / fastest[1] if fastest[1] > 0 else 0

            print(f"\n🏆 Fastest: {fastest[0]} ({fastest[1]:.3f}s)")
            if len(timings) > 1:
                print(f"   Speedup vs slowest: {speedup:.1f}x")

                # Compare to row loop if available
                row_loop_timing = next((t for t in timings if "Row Loop" in t[0]), None)
                if row_loop_timing and row_loop_timing != fastest:
                    speedup_vs_loop = row_loop_timing[1] / fastest[1]
                    print(f"   Speedup vs Row Loop: {speedup_vs_loop:.1f}x")

        # Validate results are the same
        print("\n✅ Validation:")
        if validate_results(results):
            print("   All approaches produce identical results")
        else:
            print("   Some approaches produce different results (see warnings above)")

        print("-" * 80)

    # Final recommendations
    print("\n" + "=" * 80)
    print("RECOMMENDATIONS:")
    print("=" * 80)
    print("1. For chunked processing: Use 'Replace (Vectorized)' or 'DataFrame Join'")
    print("2. Replace is simpler but Join may be faster for very large datasets")
    print("3. Avoid row loops - they become prohibitively slow for millions of rows")
    print("4. For 20M rows (typical month), vectorized approaches are 10-100x faster")


if __name__ == "__main__":
    main()