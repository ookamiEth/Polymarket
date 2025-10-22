#!/usr/bin/env python3

"""
Check the newly generated IV file with daily rates for the FULL dataset
"""

import polars as pl

file_path = "/Users/lgierhake/Documents/ETH/BT/research/tardis/data/consolidated/quotes_1s_atm_short_dated_with_iv_daily_rates_FULL.parquet"

print(f"Checking: {file_path.split('/')[-1]}")
print("=" * 80)

# Load file lazily
df = pl.scan_parquet(file_path)

# Check date range and basic stats
stats = df.select([
    pl.from_epoch("timestamp_seconds", time_unit="s").cast(pl.Date).min().alias("min_date"),
    pl.from_epoch("timestamp_seconds", time_unit="s").cast(pl.Date).max().alias("max_date"),
    pl.len().alias("total_rows")
]).collect()

print(f"Date range: {stats['min_date'][0]} to {stats['max_date'][0]}")
print(f"Total rows: {stats['total_rows'][0]:,}")

# Check IV calculation success rate
iv_stats = df.select([
    pl.len().alias("total"),
    (pl.col("iv_calc_status") == "success").sum().alias("success"),
    (pl.col("iv_calc_status") == "failed").sum().alias("failed"),
    (pl.col("iv_calc_status") == "invalid_input").sum().alias("invalid_input"),
]).collect()

print(f"\nIV Calculation Results:")
print(f"  Success: {iv_stats['success'][0]:,} ({iv_stats['success'][0]/iv_stats['total'][0]*100:.1f}%)")
print(f"  Failed: {iv_stats['failed'][0]:,} ({iv_stats['failed'][0]/iv_stats['total'][0]*100:.1f}%)")
print(f"  Invalid input: {iv_stats['invalid_input'][0]:,} ({iv_stats['invalid_input'][0]/iv_stats['total'][0]*100:.1f}%)")

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

# Show first and last few months
for row in monthly.head(5).iter_rows(named=True):
    print(f"  {row['month']}: {row['row_count']:,} rows")
print(f"  ... ({len(monthly) - 10} more months)")
for row in monthly.tail(5).iter_rows(named=True):
    print(f"  {row['month']}: {row['row_count']:,} rows")

print(f"\nTotal months covered: {len(monthly)}")

# Check IV values for successful calculations
iv_values = df.filter(pl.col("iv_calc_status") == "success").select([
    pl.col("implied_vol_bid").mean().alias("avg_iv_bid"),
    pl.col("implied_vol_ask").mean().alias("avg_iv_ask"),
    pl.col("implied_vol_bid").min().alias("min_iv_bid"),
    pl.col("implied_vol_bid").max().alias("max_iv_bid"),
]).collect()

print(f"\nIV Statistics (successful calculations):")
print(f"  Average IV (bid): {iv_values['avg_iv_bid'][0]:.4f}")
print(f"  Average IV (ask): {iv_values['avg_iv_ask'][0]:.4f}")
print(f"  Min IV (bid): {iv_values['min_iv_bid'][0]:.4f}")
print(f"  Max IV (bid): {iv_values['max_iv_bid'][0]:.4f}")

# Check columns
columns = df.collect_schema().names()
print(f"\nColumns ({len(columns)}):")
print(f"  {columns}")

print("\n✅ FULL dataset IV calculation with daily rates is complete!")