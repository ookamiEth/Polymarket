#!/usr/bin/env python3

"""
Check what's in the quotes_1s_merged.parquet file - might be the full dataset
"""

import polars as pl
from datetime import datetime

MERGED_FILE = "/Users/lgierhake/Documents/ETH/BT/research/tardis/data/consolidated/quotes_1s_merged.parquet"

print("=" * 80)
print("CHECKING quotes_1s_merged.parquet")
print("=" * 80)

# Check file stats
df = pl.scan_parquet(MERGED_FILE)

# Get date range
date_stats = df.select([
    pl.from_epoch("timestamp_seconds", time_unit="s").cast(pl.Date).min().alias("min_date"),
    pl.from_epoch("timestamp_seconds", time_unit="s").cast(pl.Date).max().alias("max_date"),
    pl.len().alias("total_rows")
]).collect()

print(f"\nFile: quotes_1s_merged.parquet")
print(f"  Date range: {date_stats['min_date'][0]} to {date_stats['max_date'][0]}")
print(f"  Total rows: {date_stats['total_rows'][0]:,}")

# Get monthly distribution
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

for row in monthly.iter_rows(named=True):
    print(f"  {row['month']}: {row['row_count']:,} rows")

# Check columns
print("\nColumns in file:")
print(f"  {df.collect_schema().names()}")

# Check for IV columns
has_iv = any('iv' in col.lower() or 'impl' in col.lower() for col in df.collect_schema().names())
print(f"\nHas IV columns: {has_iv}")

# Now check raw options files from 2024/2025
print("\n" + "=" * 80)
print("CHECKING RAW OPTIONS FILES FROM 2024/2025")
print("=" * 80)

raw_files = [
    "/Users/lgierhake/Documents/ETH/BT/research/tardis/data/raw/options_initial/deribit_options_2024-01-01_BTC_1s.parquet",
    "/Users/lgierhake/Documents/ETH/BT/research/tardis/data/raw/options_initial/deribit_options_2025-08-01_BTC_1s_with_iv.parquet",
    "/Users/lgierhake/Documents/ETH/BT/research/tardis/data/raw/options_initial/deribit_options_2025-09-01_BTC_1s_with_iv.parquet",
]

for file_path in raw_files:
    try:
        df_raw = pl.scan_parquet(file_path)
        stats = df_raw.select([
            pl.from_epoch("timestamp_seconds", time_unit="s").cast(pl.Date).min().alias("min_date"),
            pl.from_epoch("timestamp_seconds", time_unit="s").cast(pl.Date).max().alias("max_date"),
            pl.len().alias("rows")
        ]).collect()

        filename = file_path.split('/')[-1]
        print(f"\n{filename}:")
        print(f"  Date range: {stats['min_date'][0]} to {stats['max_date'][0]}")
        print(f"  Rows: {stats['rows'][0]:,}")
    except Exception as e:
        print(f"\n{file_path.split('/')[-1]}: Error - {e}")