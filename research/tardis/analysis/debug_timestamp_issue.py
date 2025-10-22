#!/usr/bin/env python3

"""
Debug the timestamp issue in the FULL IV file
"""

import polars as pl
from datetime import datetime

file_path = "/Users/lgierhake/Documents/ETH/BT/research/tardis/data/consolidated/quotes_1s_atm_short_dated_with_iv_daily_rates_FULL.parquet"

print("Debugging timestamp issue in FULL IV file")
print("=" * 80)

# Sample some timestamps from different parts of the file
df = pl.read_parquet(file_path, columns=["timestamp_seconds", "symbol"], n_rows=1_000_000)

# Check min and max timestamps
min_ts = df["timestamp_seconds"].min()
max_ts = df["timestamp_seconds"].max()

print(f"First 1M rows timestamp range:")
print(f"  Min: {min_ts} = {datetime.fromtimestamp(min_ts)}")
print(f"  Max: {max_ts} = {datetime.fromtimestamp(max_ts)}")

# Sample from middle
df_middle = pl.read_parquet(
    file_path,
    columns=["timestamp_seconds", "symbol"],
    n_rows=1_000,
    row_index_offset=100_000_000  # Middle of 204M
)

min_ts_mid = df_middle["timestamp_seconds"].min()
max_ts_mid = df_middle["timestamp_seconds"].max()

print(f"\nMiddle (around row 100M) timestamp range:")
print(f"  Min: {min_ts_mid} = {datetime.fromtimestamp(min_ts_mid)}")
print(f"  Max: {max_ts_mid} = {datetime.fromtimestamp(max_ts_mid)}")

# Sample from near end
df_end = pl.read_parquet(
    file_path,
    columns=["timestamp_seconds", "symbol"],
    n_rows=1_000,
    row_index_offset=200_000_000  # Near end
)

min_ts_end = df_end["timestamp_seconds"].min()
max_ts_end = df_end["timestamp_seconds"].max()

print(f"\nNear end (around row 200M) timestamp range:")
print(f"  Min: {min_ts_end} = {datetime.fromtimestamp(min_ts_end)}")
print(f"  Max: {max_ts_end} = {datetime.fromtimestamp(max_ts_end)}")

# Check unique symbols to see if they're from different months
print(f"\nSample symbols from start:")
print(df["symbol"].unique().head(10))

print(f"\nSample symbols from middle:")
print(df_middle["symbol"].unique().head(10))

print(f"\nSample symbols from end:")
print(df_end["symbol"].unique().head(10))

# Full scan for actual date range (might be slow)
print("\n" + "=" * 80)
print("Doing full scan for actual timestamp range (may take a moment)...")

df_full = pl.scan_parquet(file_path)
actual_stats = df_full.select([
    pl.col("timestamp_seconds").min().alias("true_min"),
    pl.col("timestamp_seconds").max().alias("true_max"),
]).collect()

true_min = actual_stats["true_min"][0]
true_max = actual_stats["true_max"][0]

print(f"\nActual timestamp range in file:")
print(f"  Min: {true_min} = {datetime.fromtimestamp(true_min)}")
print(f"  Max: {true_max} = {datetime.fromtimestamp(true_max)}")

days_span = (true_max - true_min) / (24 * 60 * 60)
print(f"  Span: {days_span:.1f} days")

if days_span < 35:
    print("\n⚠️ PROBLEM: File only contains ~1 month of data, not 2 years!")
else:
    print("\n✅ File contains the expected 2-year span")