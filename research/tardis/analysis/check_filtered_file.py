#!/usr/bin/env python3

"""
Check what date range is in the filtered ATM short-dated file
"""

import polars as pl

file_path = "/Users/lgierhake/Documents/ETH/BT/research/tardis/data/consolidated/quotes_1s_atm_short_dated_optimized.parquet"

print(f"Checking: {file_path}")
print("=" * 80)

# Check date range
df = pl.scan_parquet(file_path)

stats = df.select([
    pl.from_epoch("timestamp_seconds", time_unit="s").cast(pl.Date).min().alias("min_date"),
    pl.from_epoch("timestamp_seconds", time_unit="s").cast(pl.Date).max().alias("max_date"),
    pl.len().alias("total_rows")
]).collect()

print(f"Date range: {stats['min_date'][0]} to {stats['max_date'][0]}")
print(f"Total rows: {stats['total_rows'][0]:,}")

# Check monthly distribution
print("\nMonthly distribution:")
monthly = (
    df.select([
        pl.from_epoch("timestamp_seconds", time_unit="s").dt.strftime("%Y-%m").alias("month"),
        pl.lit(1).alias("count")
    ])
    .group_by("month")
    .agg(pl.sum("count").alias("row_count"))
    .sort("month")
    .collect()
)

for row in monthly.head(10).iter_rows(named=True):
    print(f"  {row['month']}: {row['row_count']:,} rows")

if len(monthly) > 10:
    print(f"  ... ({len(monthly) - 10} more months)")
    for row in monthly.tail(5).iter_rows(named=True):
        print(f"  {row['month']}: {row['row_count']:,} rows")

print(f"\nTotal months: {len(monthly)}")
print(f"This file covers the {'FULL' if len(monthly) > 20 else 'PARTIAL'} dataset!")