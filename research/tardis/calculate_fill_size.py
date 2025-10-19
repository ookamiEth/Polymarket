#!/usr/bin/env python3
"""
Calculate exact size expansion for lifecycle forward-fill.

Analyzes actual data to determine:
- Current row count
- Row count after forward-fill (every second from first to last quote per symbol)
- Exact expansion multiplier
- Estimated file size
"""

import polars as pl


def main() -> None:
    """Calculate exact forward-fill expansion."""
    print("=" * 80)
    print("FORWARD-FILL SIZE CALCULATION")
    print("=" * 80)
    print()
    print("Analyzing quotes_1s_merged.parquet...")
    print()

    # Scan the file to get per-symbol lifecycle info
    lazy_df = pl.scan_parquet("quotes_1s_merged.parquet")

    # Get lifecycle info for ALL symbols
    print("Calculating lifecycle for each symbol...")
    symbol_stats = (
        lazy_df.group_by("symbol")
        .agg(
            [
                pl.len().alias("current_rows"),
                pl.col("timestamp_seconds").min().alias("first_ts"),
                pl.col("timestamp_seconds").max().alias("last_ts"),
            ]
        )
        .with_columns([(pl.col("last_ts") - pl.col("first_ts") + 1).alias("lifecycle_seconds")])
        .collect()
    )

    num_symbols = len(symbol_stats)
    print(f"Analyzed {num_symbols:,} unique symbols")
    print()

    # Calculate totals
    total_current_rows = symbol_stats["current_rows"].sum()
    total_lifecycle_seconds = symbol_stats["lifecycle_seconds"].sum()

    # Calculate expansion
    multiplier = total_lifecycle_seconds / total_current_rows
    additional_rows = total_lifecycle_seconds - total_current_rows

    # Estimate file size (based on current compression ratio)
    current_size_gb = 8.2
    estimated_size_gb = current_size_gb * multiplier

    # Display results
    print("=" * 80)
    print("EXACT NUMBERS")
    print("=" * 80)
    print()
    print("CURRENT STATE (Sparse - only when quotes changed):")
    print(f"  Rows: {total_current_rows:,}")
    print(f"  File size: {current_size_gb:.1f} GB")
    print()
    print("AFTER FORWARD-FILL (Every second from first to last quote per symbol):")
    print(f"  Rows: {total_lifecycle_seconds:,}")
    print(f"  File size: {estimated_size_gb:.1f} GB")
    print()
    print("EXPANSION:")
    print(f"  Multiplier: {multiplier:.2f}x")
    print(f"  Additional rows: {additional_rows:,}")
    print(f"  Additional size: {estimated_size_gb - current_size_gb:.1f} GB")
    print()
    print("=" * 80)
    print()

    # Lifecycle distribution statistics
    lifecycle_days = symbol_stats["lifecycle_seconds"] / 86400
    coverage_pct = (symbol_stats["current_rows"] / symbol_stats["lifecycle_seconds"]) * 100

    print("SYMBOL LIFECYCLE STATISTICS:")
    print()
    print(f"  Number of symbols: {num_symbols:,}")
    print()
    print(f"  Lifecycle duration:")
    print(f"    Minimum: {lifecycle_days.min():.1f} days")
    print(f"    Maximum: {lifecycle_days.max():.1f} days")
    print(f"    Median: {lifecycle_days.median():.1f} days")
    print(f"    Mean: {lifecycle_days.mean():.1f} days")
    print()
    print(f"  Current coverage (% of lifecycle with quotes):")
    print(f"    Minimum: {coverage_pct.min():.2f}%")
    print(f"    Maximum: {coverage_pct.max():.2f}%")
    print(f"    Median: {coverage_pct.median():.2f}%")
    print(f"    Mean: {coverage_pct.mean():.2f}%")
    print()

    # Show examples
    print("=" * 80)
    print("SAMPLE SYMBOLS (Top 10 by current row count):")
    print("=" * 80)
    print()

    top_symbols = (
        symbol_stats.with_columns(
            [
                (pl.col("lifecycle_seconds") / 86400).alias("lifecycle_days"),
                ((pl.col("current_rows") / pl.col("lifecycle_seconds")) * 100).alias("coverage_pct"),
            ]
        )
        .sort("current_rows", descending=True)
        .head(10)
        .select(["symbol", "current_rows", "lifecycle_seconds", "lifecycle_days", "coverage_pct"])
    )

    print(top_symbols)
    print()

    # Summary
    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print()
    print(f"If you forward-fill, you will go from:")
    print(f"  • {total_current_rows:,} rows ({current_size_gb:.1f} GB)")
    print(f"  → {total_lifecycle_seconds:,} rows ({estimated_size_gb:.1f} GB)")
    print()
    print(f"That's {multiplier:.2f}x more data!")
    print()


if __name__ == "__main__":
    main()
