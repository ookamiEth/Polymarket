#!/usr/bin/env python3
"""
Explore and analyze the filtered ATM short-dated options output.
Provides comprehensive analysis of the 205M row filtered dataset.
"""

import polars as pl
from datetime import datetime
import numpy as np

# File to analyze
OUTPUT_FILE = "quotes_1s_atm_short_dated_optimized.parquet"

print("=" * 80)
print("ATM SHORT-DATED OPTIONS OUTPUT ANALYSIS")
print("=" * 80)

# Load data lazily for efficient memory usage
df = pl.scan_parquet(OUTPUT_FILE)

# 1. BASIC STATISTICS
print("\n📊 DATASET OVERVIEW")
print("-" * 40)

# Get row count and file size
row_count = df.select(pl.len()).collect().item()
print(f"Total rows: {row_count:,}")

import os
file_size_gb = os.path.getsize(OUTPUT_FILE) / (1024**3)
print(f"File size: {file_size_gb:.2f} GB")
print(f"Compression ratio: {row_count * 200 / (1024**3) / file_size_gb:.1f}x (estimated)")

# 2. TIME RANGE ANALYSIS
print("\n📅 TIME RANGE")
print("-" * 40)

time_stats = df.select([
    pl.col("timestamp_seconds").min().alias("min_ts"),
    pl.col("timestamp_seconds").max().alias("max_ts"),
    pl.col("timestamp_seconds").n_unique().alias("unique_timestamps"),
]).collect().row(0, named=True)

min_dt = datetime.fromtimestamp(time_stats["min_ts"])
max_dt = datetime.fromtimestamp(time_stats["max_ts"])
duration_days = (time_stats["max_ts"] - time_stats["min_ts"]) / 86400

print(f"Start: {min_dt} (UTC)")
print(f"End: {max_dt} (UTC)")
print(f"Duration: {duration_days:.1f} days")
print(f"Unique timestamps: {time_stats['unique_timestamps']:,}")
print(f"Average rows per timestamp: {row_count / time_stats['unique_timestamps']:.1f}")

# 3. MONEYNESS DISTRIBUTION
print("\n💰 MONEYNESS DISTRIBUTION (Spot/Strike)")
print("-" * 40)

moneyness_stats = df.select([
    pl.col("moneyness").min().alias("min"),
    pl.col("moneyness").quantile(0.01).alias("p1"),
    pl.col("moneyness").quantile(0.25).alias("q1"),
    pl.col("moneyness").quantile(0.50).alias("median"),
    pl.col("moneyness").quantile(0.75).alias("q3"),
    pl.col("moneyness").quantile(0.99).alias("p99"),
    pl.col("moneyness").max().alias("max"),
    pl.col("moneyness").mean().alias("mean"),
    pl.col("moneyness").std().alias("std"),
]).collect().row(0, named=True)

print(f"Range: {moneyness_stats['min']:.4f} - {moneyness_stats['max']:.4f}")
print(f"Mean: {moneyness_stats['mean']:.4f} (±{moneyness_stats['std']:.4f})")
print(f"Median: {moneyness_stats['median']:.4f}")
print(f"IQR: [{moneyness_stats['q1']:.4f}, {moneyness_stats['q3']:.4f}]")
print(f"99% range: [{moneyness_stats['p1']:.4f}, {moneyness_stats['p99']:.4f}]")

# Check ATM filter effectiveness
atm_1pct = df.filter(
    (pl.col("moneyness") >= 0.99) & (pl.col("moneyness") <= 1.01)
).select(pl.len()).collect().item()
print(f"\nWithin ±1% of ATM: {atm_1pct:,} ({atm_1pct/row_count*100:.1f}%)")

# 4. TIME TO EXPIRY DISTRIBUTION
print("\n⏰ TIME TO EXPIRY DISTRIBUTION")
print("-" * 40)

ttl_stats = df.select([
    pl.col("time_to_expiry_days").min().alias("min"),
    pl.col("time_to_expiry_days").quantile(0.25).alias("q1"),
    pl.col("time_to_expiry_days").quantile(0.50).alias("median"),
    pl.col("time_to_expiry_days").quantile(0.75).alias("q3"),
    pl.col("time_to_expiry_days").max().alias("max"),
    pl.col("time_to_expiry_days").mean().alias("mean"),
]).collect().row(0, named=True)

print(f"Range: {ttl_stats['min']:.3f} - {ttl_stats['max']:.3f} days")
print(f"Mean: {ttl_stats['mean']:.3f} days")
print(f"Median: {ttl_stats['median']:.3f} days")
print(f"Q1: {ttl_stats['q1']:.3f} days, Q3: {ttl_stats['q3']:.3f} days")

# TTL buckets
ttl_buckets = df.select([
    pl.col("time_to_expiry_days").filter(pl.col("time_to_expiry_days") <= 1.0).count().alias("0-1_day"),
    pl.col("time_to_expiry_days").filter(
        (pl.col("time_to_expiry_days") > 1.0) & (pl.col("time_to_expiry_days") <= 2.0)
    ).count().alias("1-2_days"),
    pl.col("time_to_expiry_days").filter(
        (pl.col("time_to_expiry_days") > 2.0) & (pl.col("time_to_expiry_days") <= 3.0)
    ).count().alias("2-3_days"),
]).collect().row(0, named=True)

print("\nTTL Buckets:")
print(f"  0-1 day: {ttl_buckets['0-1_day']:,} ({ttl_buckets['0-1_day']/row_count*100:.1f}%)")
print(f"  1-2 days: {ttl_buckets['1-2_days']:,} ({ttl_buckets['1-2_days']/row_count*100:.1f}%)")
print(f"  2-3 days: {ttl_buckets['2-3_days']:,} ({ttl_buckets['2-3_days']/row_count*100:.1f}%)")

# 5. OPTION TYPE DISTRIBUTION
print("\n📈 OPTION TYPE DISTRIBUTION")
print("-" * 40)

type_counts = df.group_by("type").agg(
    pl.len().alias("count")
).sort("count", descending=True).collect()

for row in type_counts.iter_rows(named=True):
    pct = row["count"] / row_count * 100
    print(f"{row['type']:5s}: {row['count']:,} ({pct:.1f}%)")

# 6. STRIKE PRICE ANALYSIS
print("\n💵 STRIKE PRICE DISTRIBUTION")
print("-" * 40)

strike_stats = df.select([
    pl.col("strike_price").min().alias("min"),
    pl.col("strike_price").max().alias("max"),
    pl.col("strike_price").mean().alias("mean"),
    pl.col("strike_price").std().alias("std"),
    pl.col("strike_price").n_unique().alias("unique_strikes"),
]).collect().row(0, named=True)

print(f"Range: ${strike_stats['min']:,.0f} - ${strike_stats['max']:,.0f}")
print(f"Mean: ${strike_stats['mean']:,.0f} (±${strike_stats['std']:,.0f})")
print(f"Unique strikes: {strike_stats['unique_strikes']:,}")

# 7. SPOT PRICE ANALYSIS
print("\n📉 SPOT PRICE (BTC) DISTRIBUTION")
print("-" * 40)

spot_stats = df.select([
    pl.col("spot_price").min().alias("min"),
    pl.col("spot_price").max().alias("max"),
    pl.col("spot_price").mean().alias("mean"),
    pl.col("spot_price").std().alias("std"),
]).collect().row(0, named=True)

print(f"Range: ${spot_stats['min']:,.2f} - ${spot_stats['max']:,.2f}")
print(f"Mean: ${spot_stats['mean']:,.2f} (±${spot_stats['std']:,.2f})")

# 8. QUOTE QUALITY
print("\n📋 QUOTE QUALITY")
print("-" * 40)

quote_stats = df.select([
    pl.col("has_bid").sum().alias("has_bid"),
    pl.col("has_ask").sum().alias("has_ask"),
    (pl.col("has_bid") & pl.col("has_ask")).sum().alias("has_both"),
    pl.col("spread_abs").mean().alias("mean_spread"),
    pl.col("spread_abs").median().alias("median_spread"),
    (pl.col("spread_abs") / pl.col("mid_price") * 100).mean().alias("mean_spread_pct"),
]).collect().row(0, named=True)

print(f"Has bid: {quote_stats['has_bid']:,} ({quote_stats['has_bid']/row_count*100:.1f}%)")
print(f"Has ask: {quote_stats['has_ask']:,} ({quote_stats['has_ask']/row_count*100:.1f}%)")
print(f"Has both: {quote_stats['has_both']:,} ({quote_stats['has_both']/row_count*100:.1f}%)")
print(f"\nMean spread: ${quote_stats['mean_spread']:.4f}")
print(f"Median spread: ${quote_stats['median_spread']:.4f}")
print(f"Mean spread %: {quote_stats['mean_spread_pct']:.2f}%")

# 9. SYMBOL ANALYSIS
print("\n🏷️ SYMBOL ANALYSIS")
print("-" * 40)

symbol_stats = df.select([
    pl.col("symbol").n_unique().alias("unique_symbols"),
]).collect().row(0, named=True)

print(f"Unique symbols: {symbol_stats['unique_symbols']:,}")
print(f"Average rows per symbol: {row_count / symbol_stats['unique_symbols']:,.0f}")

# Top symbols by volume
print("\nTop 10 Symbols by Quote Count:")
top_symbols = df.group_by("symbol").agg(
    pl.len().alias("count")
).sort("count", descending=True).head(10).collect()

for i, row in enumerate(top_symbols.iter_rows(named=True), 1):
    pct = row["count"] / row_count * 100
    print(f"  {i:2d}. {row['symbol']:30s}: {row['count']:,} ({pct:.2f}%)")

# 10. TEMPORAL PATTERNS
print("\n📊 TEMPORAL PATTERNS")
print("-" * 40)

# Monthly distribution
monthly_counts = df.with_columns([
    pl.from_epoch("timestamp_seconds").dt.year().alias("year"),
    pl.from_epoch("timestamp_seconds").dt.month().alias("month"),
]).group_by(["year", "month"]).agg(
    pl.len().alias("count")
).sort(["year", "month"]).collect()

print("Monthly Distribution (top 10):")
for row in monthly_counts.head(10).iter_rows(named=True):
    pct = row["count"] / row_count * 100
    print(f"  {row['year']}-{row['month']:02d}: {row['count']:,} ({pct:.1f}%)")

print("\n" + "=" * 80)
print("✅ ANALYSIS COMPLETE")
print("=" * 80)