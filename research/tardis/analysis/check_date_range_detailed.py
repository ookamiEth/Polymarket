#!/usr/bin/env python3

"""
Check date ranges more carefully - sample actual timestamps
"""

import polars as pl
from datetime import datetime

# File paths
CONSTANT_FILE = "/Users/lgierhake/Documents/ETH/BT/research/tardis/data/consolidated/quotes_1s_atm_short_dated_with_iv.parquet"
DAILY_FILE = "/Users/lgierhake/Documents/ETH/BT/research/tardis/data/consolidated/quotes_1s_atm_short_dated_with_iv_daily_rates.parquet"

print("=" * 80)
print("DETAILED DATE RANGE INVESTIGATION")
print("=" * 80)

# Check constant file with direct timestamp inspection
print("\n1. Checking Constant IV file timestamps...")
df_const = pl.read_parquet(CONSTANT_FILE, columns=["timestamp_seconds"]).head(100_000)

# Check min and max timestamps
min_ts = df_const["timestamp_seconds"].min()
max_ts = df_const["timestamp_seconds"].max()

print(f"   First 100k rows:")
print(f"   Min timestamp: {min_ts} = {datetime.fromtimestamp(min_ts)}")
print(f"   Max timestamp: {max_ts} = {datetime.fromtimestamp(max_ts)}")

# Sample from different parts of the file
print("\n2. Sampling from different parts of Constant file...")
const_lazy = pl.scan_parquet(CONSTANT_FILE)

# Get total rows
total_rows = const_lazy.select(pl.len()).collect().item()
print(f"   Total rows: {total_rows:,}")

# Sample from beginning, middle, and end
samples = []
for offset in [0, total_rows // 2, total_rows - 1000]:
    sample = pl.read_parquet(
        CONSTANT_FILE,
        columns=["timestamp_seconds"],
        n_rows=1,
        row_index_offset=offset
    )
    ts = sample["timestamp_seconds"][0]
    samples.append((offset, ts, datetime.fromtimestamp(ts)))

for offset, ts, dt in samples:
    print(f"   Row {offset:,}: timestamp={ts} = {dt}")

print("\n3. Checking Daily IV file timestamps...")
df_daily = pl.read_parquet(DAILY_FILE, columns=["timestamp_seconds"]).head(100_000)

min_ts = df_daily["timestamp_seconds"].min()
max_ts = df_daily["timestamp_seconds"].max()

print(f"   First 100k rows:")
print(f"   Min timestamp: {min_ts} = {datetime.fromtimestamp(min_ts)}")
print(f"   Max timestamp: {max_ts} = {datetime.fromtimestamp(max_ts)}")

# Check unique dates more carefully
print("\n4. Unique dates in files (using full scan)...")

# Get unique dates from constant file
const_dates = (
    pl.scan_parquet(CONSTANT_FILE)
    .select(pl.from_epoch("timestamp_seconds", time_unit="s").cast(pl.Date).alias("date"))
    .unique()
    .sort("date")
    .collect()
)

print(f"\nConstant file unique dates: {len(const_dates)} days")
print(f"   First date: {const_dates['date'].min()}")
print(f"   Last date: {const_dates['date'].max()}")

# Show all unique dates
if len(const_dates) < 100:
    print("\nAll unique dates in Constant file:")
    for date_val in const_dates["date"].to_list():
        print(f"   {date_val}")

# Get unique dates from daily file
daily_dates = (
    pl.scan_parquet(DAILY_FILE)
    .select(pl.from_epoch("timestamp_seconds", time_unit="s").cast(pl.Date).alias("date"))
    .unique()
    .sort("date")
    .collect()
)

print(f"\nDaily file unique dates: {len(daily_dates)} days")
print(f"   First date: {daily_dates['date'].min()}")
print(f"   Last date: {daily_dates['date'].max()}")

if len(daily_dates) < 100:
    print("\nAll unique dates in Daily file:")
    for date_val in daily_dates["date"].to_list():
        print(f"   {date_val}")