#!/usr/bin/env python3

"""
Diagnose what's happening with the IV calculation - why it's only processing October 2023
"""

import polars as pl
import sys
from pathlib import Path

print("=" * 80)
print("DIAGNOSING IV CALCULATION INPUT ISSUE")
print("=" * 80)

# Check the actual input file that should be used
input_file = "/Users/lgierhake/Documents/ETH/BT/research/tardis/data/consolidated/quotes_1s_atm_short_dated_optimized.parquet"

print(f"\n1. Checking the correct input file:")
print(f"   Path: {input_file}")

if not Path(input_file).exists():
    print("   ❌ FILE DOES NOT EXIST!")
    sys.exit(1)

# Get file info
file_size_gb = Path(input_file).stat().st_size / (1024**3)
print(f"   Size: {file_size_gb:.2f} GB")

# Check content
df = pl.scan_parquet(input_file)
stats = df.select([
    pl.len().alias("total_rows"),
    pl.from_epoch("timestamp_seconds", time_unit="s").cast(pl.Date).min().alias("min_date"),
    pl.from_epoch("timestamp_seconds", time_unit="s").cast(pl.Date).max().alias("max_date"),
]).collect()

print(f"   Rows: {stats['total_rows'][0]:,}")
print(f"   Date range: {stats['min_date'][0]} to {stats['max_date'][0]}")

# Get monthly distribution
monthly = (
    df.select([
        pl.from_epoch("timestamp_seconds", time_unit="s").dt.strftime("%Y-%m").alias("month"),
        pl.lit(1).alias("count")
    ])
    .group_by("month")
    .agg(pl.sum("count").alias("rows"))
    .sort("month")
    .collect()
)

print(f"   Months: {len(monthly)}")
print(f"   First month: {monthly['month'][0]} ({monthly['rows'][0]:,} rows)")
print(f"   Last month: {monthly['month'][-1]} ({monthly['rows'][-1]:,} rows)")

# Now check what the IV calculation script would see
print("\n2. Testing what IV calculation script sees:")

# Import path resolution
import os
os.chdir("/Users/lgierhake/Documents/ETH/BT/research/tardis")

# Check relative path resolution
relative_path = "data/consolidated/quotes_1s_atm_short_dated_optimized.parquet"
full_path = Path(relative_path).resolve()

print(f"   From tardis directory:")
print(f"   Relative: {relative_path}")
print(f"   Resolves to: {full_path}")
print(f"   Exists: {full_path.exists()}")

if full_path.exists():
    # Quick check of this file
    df_check = pl.scan_parquet(str(full_path))
    check_stats = df_check.select([
        pl.len().alias("rows"),
        pl.from_epoch("timestamp_seconds", time_unit="s").cast(pl.Date).min().alias("min_date"),
        pl.from_epoch("timestamp_seconds", time_unit="s").cast(pl.Date).max().alias("max_date"),
    ]).collect()

    print(f"   This file has: {check_stats['rows'][0]:,} rows")
    print(f"   Date range: {check_stats['min_date'][0]} to {check_stats['max_date'][0]}")

# Check if there's another file that might be confused
print("\n3. Checking for other files that might cause confusion:")

other_files = [
    "data/consolidated/quotes_1s_atm_short_dated.parquet",
    "data/consolidated/quotes_1s_atm_short_dated_filtered.parquet",
    "../data/consolidated/quotes_1s_atm_short_dated_optimized.parquet",
]

for file in other_files:
    full = Path(file).resolve()
    if full.exists():
        print(f"   ⚠️ Found: {file}")
        print(f"      Resolves to: {full}")
        size_gb = full.stat().st_size / (1024**3)
        print(f"      Size: {size_gb:.2f} GB")

# Generate the exact command
print("\n" + "=" * 80)
print("RECOMMENDED COMMAND TO RUN:")
print("=" * 80)

print("\nFor CONSTANT rate (4.12%):")
print("```bash")
print("cd /Users/lgierhake/Documents/ETH/BT")
print("uv run python research/tardis/scripts/iv_calculation/calculate_iv_streaming_fixed.py \\")
print("  --input /Users/lgierhake/Documents/ETH/BT/research/tardis/data/consolidated/quotes_1s_atm_short_dated_optimized.parquet \\")
print("  --output /Users/lgierhake/Documents/ETH/BT/research/tardis/data/consolidated/quotes_1s_atm_short_dated_with_iv_CONSTANT_FULL.parquet \\")
print("  --risk-free-rate 0.0412 \\")
print("  --chunk-size 5000000 \\")
print("  --verbose")
print("```")

print("\nNOTE: Using ABSOLUTE paths to avoid any confusion!")