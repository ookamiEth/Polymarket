#!/usr/bin/env python3
"""
Test different approaches for enriching quotes with spot prices.
Compares performance of dictionary lookup methods to find the optimal approach.
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


def test_map_dict_approach(quotes_df: pl.DataFrame, spot_dict: dict[int, float]) -> tuple[pl.DataFrame, float]:
    """Use Polars map_dict for vectorized dictionary lookup."""
    start = time.time()

    # Use map_dict for vectorized lookup with default=None for missing keys
    result = quotes_df.with_columns([
        pl.col("timestamp_seconds").map_dict(spot_dict, default=None).alias("spot_price")
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


def test_numpy_vectorized_approach(quotes_df: pl.DataFrame, spot_dict: dict[int, float]) -> tuple[pl.DataFrame, float]:
    """Use NumPy for vectorized lookups."""
    start = time.time()

    # Get timestamps as numpy array
    timestamps = quotes_df["timestamp_seconds"].to_numpy()

    # Vectorized lookup using numpy
    spot_prices = np.array([spot_dict.get(ts) for ts in timestamps])

    # Add back to DataFrame
    result = quotes_df.with_columns([
        pl.Series("spot_price", spot_prices)
    ])

    elapsed = time.time() - start
    return result, elapsed


def test_struct_approach(quotes_df: pl.DataFrame, spot_dict: dict[int, float]) -> tuple[pl.DataFrame, float]:
    """Use struct literals for mapping."""
    start = time.time()

    # Create a mapping expression using when-then chains (for small dicts only)
    # This is not practical for large dicts, but included for completeness
    if len(spot_dict) > 1000:
        # Fall back to map_dict for large dictionaries
        result = quotes_df.with_columns([
            pl.col("timestamp_seconds").map_dict(spot_dict, default=None).alias("spot_price")
        ])
    else:
        # Build when-then chain (inefficient for large dicts)
        expr = pl.lit(None)
        for ts, price in spot_dict.items():
            expr = pl.when(pl.col("timestamp_seconds") == ts).then(price).otherwise(expr)
        result = quotes_df.with_columns([expr.alias("spot_price")])

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

    for name, df in results[1:]:
        df_sorted = df.sort("timestamp_seconds")

        # Check if spot_price columns are equal (handling None/null)
        if not base_sorted["spot_price"].equals(df_sorted["spot_price"]):
            print(f"WARNING: {name} produces different results than {base_name}")

            # Find differences
            diff_mask = base_sorted["spot_price"] != df_sorted["spot_price"]
            if diff_mask.sum() > 0:
                print(f"  Number of differences: {diff_mask.sum()}")

            return False

    return True


def main():
    """Run performance tests."""
    print("=" * 80)
    print("SPOT PRICE ENRICHMENT PERFORMANCE TEST")
    print("=" * 80)

    # Test with different data sizes
    test_sizes = [
        (10_000, 5_000, "Small"),
        (100_000, 50_000, "Medium"),
        (1_000_000, 500_000, "Large"),
        (10_000_000, 500_000, "Very Large (10M quotes)"),
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
            ("Map Dict (Vectorized)", test_map_dict_approach),
            ("DataFrame Join", test_join_approach),
            ("NumPy Vectorized", test_numpy_vectorized_approach),
        ]

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
            print(f"   Speedup vs slowest: {speedup:.1f}x")

        # Validate results are the same
        print("\n✅ Validation:")
        if validate_results(results):
            print("   All approaches produce identical results")
        else:
            print("   WARNING: Approaches produce different results!")

        print("-" * 80)

        # For very large dataset, skip row loop as it's too slow
        if n_quotes >= 10_000_000:
            print("\n⚠️  Skipping row loop for very large dataset (too slow)")
            approaches = [a for a in approaches if "Row Loop" not in a[0]]


if __name__ == "__main__":
    main()