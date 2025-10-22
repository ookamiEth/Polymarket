#!/usr/bin/env python3

"""
Check the full date range available in both IV files
"""

import polars as pl

# File paths
CONSTANT_FILE = "/Users/lgierhake/Documents/ETH/BT/research/tardis/data/consolidated/quotes_1s_atm_short_dated_with_iv.parquet"
DAILY_FILE = "/Users/lgierhake/Documents/ETH/BT/research/tardis/data/consolidated/quotes_1s_atm_short_dated_with_iv_daily_rates.parquet"

print("=" * 80)
print("CHECKING FULL DATE RANGES IN BOTH FILES")
print("=" * 80)

# Check constant file
print("\n1. Constant IV file:")
df_const = pl.scan_parquet(CONSTANT_FILE)
const_stats = df_const.select([
    pl.from_epoch("timestamp_seconds", time_unit="s").cast(pl.Date).min().alias("min_date"),
    pl.from_epoch("timestamp_seconds", time_unit="s").cast(pl.Date).max().alias("max_date"),
    pl.len().alias("total_rows")
]).collect()

print(f"   Date range: {const_stats['min_date'][0]} to {const_stats['max_date'][0]}")
print(f"   Total rows: {const_stats['total_rows'][0]:,}")

# Check daily file
print("\n2. Daily rates IV file:")
df_daily = pl.scan_parquet(DAILY_FILE)
daily_stats = df_daily.select([
    pl.from_epoch("timestamp_seconds", time_unit="s").cast(pl.Date).min().alias("min_date"),
    pl.from_epoch("timestamp_seconds", time_unit="s").cast(pl.Date).max().alias("max_date"),
    pl.len().alias("total_rows")
]).collect()

print(f"   Date range: {daily_stats['min_date'][0]} to {daily_stats['max_date'][0]}")
print(f"   Total rows: {daily_stats['total_rows'][0]:,}")

# Check monthly distribution in constant file
print("\n3. Monthly distribution in Constant file:")
monthly_const = (
    df_const.select([
        pl.from_epoch("timestamp_seconds", time_unit="s").dt.strftime("%Y-%m").alias("month"),
        pl.lit(1).alias("count")
    ])
    .group_by("month")
    .agg(pl.sum("count").alias("row_count"))
    .sort("month")
    .collect()
)
print(monthly_const)

# Check monthly distribution in daily file
print("\n4. Monthly distribution in Daily file:")
monthly_daily = (
    df_daily.select([
        pl.from_epoch("timestamp_seconds", time_unit="s").dt.strftime("%Y-%m").alias("month"),
        pl.lit(1).alias("count")
    ])
    .group_by("month")
    .agg(pl.sum("count").alias("row_count"))
    .sort("month")
    .collect()
)
print(monthly_daily)

# Calculate overlapping period
print("\n5. Overlapping period:")
min_date = max(const_stats['min_date'][0], daily_stats['min_date'][0])
max_date = min(const_stats['max_date'][0], daily_stats['max_date'][0])
print(f"   Overlapping range: {min_date} to {max_date}")

if min_date > max_date:
    print("   ⚠️ NO OVERLAPPING PERIOD!")
else:
    days_overlap = (max_date - min_date).days + 1
    print(f"   Days of overlap: {days_overlap}")