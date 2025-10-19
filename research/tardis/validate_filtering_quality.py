#!/usr/bin/env python3
"""
Validate the quality of ATM short-dated filtering and explore specific patterns.
"""

import polars as pl
from datetime import datetime, timedelta
import numpy as np

OUTPUT_FILE = "quotes_1s_atm_short_dated_optimized.parquet"

print("=" * 80)
print("FILTERING QUALITY VALIDATION & DEEP ANALYSIS")
print("=" * 80)

# Load data lazily
df = pl.scan_parquet(OUTPUT_FILE)

# 1. VALIDATE FILTERING CRITERIA
print("\n✅ FILTER VALIDATION")
print("-" * 40)

# Check moneyness bounds (should be 0.97 - 1.03)
moneyness_violations = df.filter(
    (pl.col("moneyness") < 0.97) | (pl.col("moneyness") > 1.03)
).select(pl.len()).collect().item()

print(f"Moneyness filter (±3% ATM):")
print(f"  Expected range: 0.97 - 1.03")
print(f"  Violations: {moneyness_violations:,} rows")
if moneyness_violations > 0:
    print("  ⚠️ WARNING: Some rows outside expected range!")
else:
    print("  ✅ All rows within range")

# Check TTL bounds (should be 0-3 days)
ttl_violations = df.filter(
    (pl.col("time_to_expiry_days") < 0) | (pl.col("time_to_expiry_days") > 3.0)
).select(pl.len()).collect().item()

print(f"\nTime to expiry filter (≤3 days):")
print(f"  Expected range: 0 - 3 days")
print(f"  Violations: {ttl_violations:,} rows")
if ttl_violations > 0:
    print("  ⚠️ WARNING: Some rows outside expected range!")
else:
    print("  ✅ All rows within range")

# 2. SAMPLE DATA INSPECTION
print("\n🔍 SAMPLE DATA (5 random rows)")
print("-" * 40)

sample = df.select([
    "timestamp_seconds",
    "symbol",
    "type",
    "strike_price",
    "spot_price",
    "moneyness",
    "time_to_expiry_days",
    "bid_price",
    "ask_price",
    "mid_price",
    "spread_abs"
]).head(5).collect()

# Convert timestamp to datetime for display
sample = sample.with_columns([
    pl.from_epoch("timestamp_seconds").alias("datetime_utc")
])

# Pretty print sample
for i, row in enumerate(sample.iter_rows(named=True), 1):
    print(f"\nRow {i}:")
    print(f"  Time: {row['datetime_utc']}")
    print(f"  Symbol: {row['symbol']}")
    print(f"  Type: {row['type']}")
    print(f"  Strike: ${row['strike_price']:,.0f}")
    print(f"  Spot: ${row['spot_price']:,.2f}")
    print(f"  Moneyness: {row['moneyness']:.4f}")
    print(f"  TTL: {row['time_to_expiry_days']:.3f} days")
    print(f"  Bid: ${row['bid_price']:.4f}" if row['bid_price'] is not None else "  Bid: None")
    print(f"  Ask: ${row['ask_price']:.4f}" if row['ask_price'] is not None else "  Ask: None")
    print(f"  Mid: ${row['mid_price']:.4f}" if row['mid_price'] is not None else "  Mid: None")
    print(f"  Spread: ${row['spread_abs']:.4f}" if row['spread_abs'] is not None else "  Spread: None")

# 3. MONEYNESS PATTERNS BY OPTION TYPE
print("\n💰 MONEYNESS BY OPTION TYPE")
print("-" * 40)

moneyness_by_type = df.group_by("type").agg([
    pl.col("moneyness").mean().alias("mean_moneyness"),
    pl.col("moneyness").std().alias("std_moneyness"),
    pl.col("moneyness").min().alias("min_moneyness"),
    pl.col("moneyness").max().alias("max_moneyness"),
    pl.len().alias("count")
]).collect()

for row in moneyness_by_type.iter_rows(named=True):
    print(f"\n{row['type'].upper()}:")
    print(f"  Mean: {row['mean_moneyness']:.4f} (±{row['std_moneyness']:.4f})")
    print(f"  Range: {row['min_moneyness']:.4f} - {row['max_moneyness']:.4f}")
    print(f"  Count: {row['count']:,}")

# 4. STRIKE DISTRIBUTION AROUND SPOT
print("\n🎯 STRIKE DISTRIBUTION RELATIVE TO SPOT")
print("-" * 40)

# Calculate strike difference from spot
strike_diff_stats = df.with_columns([
    ((pl.col("strike_price") - pl.col("spot_price")) / pl.col("spot_price") * 100).alias("strike_diff_pct")
]).select([
    pl.col("strike_diff_pct").mean().alias("mean"),
    pl.col("strike_diff_pct").std().alias("std"),
    pl.col("strike_diff_pct").quantile(0.25).alias("q1"),
    pl.col("strike_diff_pct").quantile(0.50).alias("median"),
    pl.col("strike_diff_pct").quantile(0.75).alias("q3"),
]).collect().row(0, named=True)

print(f"Strike vs Spot (% difference):")
print(f"  Mean: {strike_diff_stats['mean']:.2f}%")
print(f"  Median: {strike_diff_stats['median']:.2f}%")
print(f"  IQR: [{strike_diff_stats['q1']:.2f}%, {strike_diff_stats['q3']:.2f}%]")

# 5. TIME DECAY ANALYSIS
print("\n⏱️ TIME DECAY PATTERNS")
print("-" * 40)

# Average quotes by TTL bucket
ttl_bucket_analysis = df.with_columns([
    pl.when(pl.col("time_to_expiry_days") <= 0.25)
    .then(pl.lit("0-6h"))
    .when(pl.col("time_to_expiry_days") <= 0.5)
    .then(pl.lit("6-12h"))
    .when(pl.col("time_to_expiry_days") <= 1.0)
    .then(pl.lit("12-24h"))
    .when(pl.col("time_to_expiry_days") <= 2.0)
    .then(pl.lit("1-2d"))
    .otherwise(pl.lit("2-3d"))
    .alias("ttl_bucket")
]).group_by("ttl_bucket").agg([
    pl.len().alias("count"),
    pl.col("mid_price").mean().alias("avg_premium"),
    pl.col("spread_abs").mean().alias("avg_spread"),
]).sort("ttl_bucket").collect()

print("Quote distribution by time to expiry:")
total = ttl_bucket_analysis["count"].sum()
for row in ttl_bucket_analysis.iter_rows(named=True):
    pct = row["count"] / total * 100
    print(f"  {row['ttl_bucket']:6s}: {row['count']:,} ({pct:.1f}%) - Avg premium: ${row['avg_premium']:.4f}")

# 6. VOLATILITY PERIODS ANALYSIS
print("\n📈 HIGH VOLATILITY PERIODS")
print("-" * 40)

# Find days with highest quote activity
daily_activity = df.with_columns([
    pl.from_epoch("timestamp_seconds").dt.date().alias("date")
]).group_by("date").agg([
    pl.len().alias("quote_count"),
    pl.col("spot_price").std().alias("spot_volatility"),
    pl.col("spread_abs").mean().alias("avg_spread"),
]).sort("quote_count", descending=True).head(10).collect()

print("Top 10 most active days:")
for i, row in enumerate(daily_activity.iter_rows(named=True), 1):
    print(f"  {i:2d}. {row['date']}: {row['quote_count']:,} quotes")
    print(f"      Spot volatility: ${row['spot_volatility']:,.2f}")
    print(f"      Avg spread: ${row['avg_spread']:.4f}")

# 7. SPREAD ANALYSIS
print("\n📊 BID-ASK SPREAD ANALYSIS")
print("-" * 40)

# Spread statistics by TTL
spread_by_ttl = df.with_columns([
    pl.when(pl.col("time_to_expiry_days") <= 1.0)
    .then(pl.lit("0-1d"))
    .when(pl.col("time_to_expiry_days") <= 2.0)
    .then(pl.lit("1-2d"))
    .otherwise(pl.lit("2-3d"))
    .alias("ttl_group")
]).group_by("ttl_group").agg([
    pl.col("spread_abs").mean().alias("mean_spread"),
    pl.col("spread_abs").median().alias("median_spread"),
    (pl.col("spread_abs") / pl.col("mid_price") * 100).mean().alias("mean_spread_pct"),
    pl.len().alias("count")
]).sort("ttl_group").collect()

print("Spreads by time to expiry:")
for row in spread_by_ttl.iter_rows(named=True):
    print(f"  {row['ttl_group']}:")
    print(f"    Mean: ${row['mean_spread']:.4f} ({row['mean_spread_pct']:.2f}%)")
    print(f"    Median: ${row['median_spread']:.4f}")

# 8. EXPIRY PATTERN ANALYSIS
print("\n📅 EXPIRY DATE PATTERNS")
print("-" * 40)

# Extract expiry dates from symbols
expiry_patterns = df.with_columns([
    pl.col("symbol").str.extract(r"-(\d{1,2}[A-Z]{3}\d{2})-", 1).alias("expiry_str")
]).group_by("expiry_str").agg([
    pl.len().alias("count")
]).sort("count", descending=True).head(10).collect()

print("Top 10 expiry dates by quote volume:")
total_rows = df.select(pl.len()).collect().item()
for i, row in enumerate(expiry_patterns.iter_rows(named=True), 1):
    pct = row["count"] / total_rows * 100
    print(f"  {i:2d}. {row['expiry_str']:10s}: {row['count']:,} ({pct:.2f}%)")

print("\n" + "=" * 80)
print("✅ VALIDATION COMPLETE")
print("=" * 80)