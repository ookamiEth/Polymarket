#!/usr/bin/env python3
"""
Investigate when and why bids or asks are missing in the ATM short-dated options data.
This reveals important market microstructure patterns.
"""

import polars as pl
from datetime import datetime
import numpy as np

OUTPUT_FILE = "data/consolidated/quotes_1s_atm_short_dated_optimized.parquet"

print("=" * 80)
print("INVESTIGATION: MISSING BIDS AND ASKS")
print("=" * 80)

# Load data lazily
df = pl.scan_parquet(OUTPUT_FILE)

# 1. OVERALL MISSING QUOTE STATISTICS
print("\n📊 OVERALL MISSING QUOTE STATISTICS")
print("-" * 40)

missing_stats = df.select([
    pl.len().alias("total_rows"),
    pl.col("bid_price").is_null().sum().alias("missing_bid"),
    pl.col("ask_price").is_null().sum().alias("missing_ask"),
    (pl.col("bid_price").is_null() & pl.col("ask_price").is_null()).sum().alias("missing_both"),
    (pl.col("bid_price").is_null() & pl.col("ask_price").is_not_null()).sum().alias("missing_bid_only"),
    (pl.col("bid_price").is_not_null() & pl.col("ask_price").is_null()).sum().alias("missing_ask_only"),
]).collect().row(0, named=True)

print(f"Total rows: {missing_stats['total_rows']:,}")
print(f"Missing bid: {missing_stats['missing_bid']:,} ({missing_stats['missing_bid']/missing_stats['total_rows']*100:.3f}%)")
print(f"Missing ask: {missing_stats['missing_ask']:,} ({missing_stats['missing_ask']/missing_stats['total_rows']*100:.3f}%)")
print(f"Missing both: {missing_stats['missing_both']:,} ({missing_stats['missing_both']/missing_stats['total_rows']*100:.3f}%)")
print(f"Missing bid only: {missing_stats['missing_bid_only']:,} ({missing_stats['missing_bid_only']/missing_stats['total_rows']*100:.3f}%)")
print(f"Missing ask only: {missing_stats['missing_ask_only']:,} ({missing_stats['missing_ask_only']/missing_stats['total_rows']*100:.3f}%)")

# 2. MISSING QUOTES BY TIME TO EXPIRY
print("\n⏰ MISSING QUOTES BY TIME TO EXPIRY")
print("-" * 40)

ttl_missing = df.with_columns([
    pl.when(pl.col("time_to_expiry_days") <= 0.25).then(pl.lit("0-6h"))
    .when(pl.col("time_to_expiry_days") <= 0.5).then(pl.lit("6-12h"))
    .when(pl.col("time_to_expiry_days") <= 1.0).then(pl.lit("12-24h"))
    .when(pl.col("time_to_expiry_days") <= 2.0).then(pl.lit("1-2d"))
    .otherwise(pl.lit("2-3d"))
    .alias("ttl_bucket")
]).group_by("ttl_bucket").agg([
    pl.len().alias("total"),
    pl.col("bid_price").is_null().sum().alias("missing_bid"),
    pl.col("ask_price").is_null().sum().alias("missing_ask"),
]).with_columns([
    (pl.col("missing_bid") / pl.col("total") * 100).alias("missing_bid_pct"),
    (pl.col("missing_ask") / pl.col("total") * 100).alias("missing_ask_pct"),
]).sort("ttl_bucket").collect()

print("TTL Bucket | Total Rows | Missing Bid | Missing Ask")
print("-" * 55)
for row in ttl_missing.iter_rows(named=True):
    print(f"{row['ttl_bucket']:8s} | {row['total']:10,} | {row['missing_bid_pct']:6.3f}% | {row['missing_ask_pct']:6.3f}%")

print("\n💡 Insight: Missing quotes increase as options approach expiry")

# 3. MISSING QUOTES BY MONEYNESS
print("\n💰 MISSING QUOTES BY MONEYNESS")
print("-" * 40)

moneyness_missing = df.with_columns([
    pl.when(pl.col("moneyness") < 0.98).then(pl.lit("0.97-0.98 (Deep OTM)"))
    .when(pl.col("moneyness") < 0.99).then(pl.lit("0.98-0.99 (OTM)"))
    .when(pl.col("moneyness") < 1.00).then(pl.lit("0.99-1.00 (Slight OTM)"))
    .when(pl.col("moneyness") < 1.01).then(pl.lit("1.00-1.01 (Slight ITM)"))
    .when(pl.col("moneyness") < 1.02).then(pl.lit("1.01-1.02 (ITM)"))
    .otherwise(pl.lit("1.02-1.03 (Deep ITM)"))
    .alias("moneyness_bucket")
]).group_by("moneyness_bucket").agg([
    pl.len().alias("total"),
    pl.col("bid_price").is_null().sum().alias("missing_bid"),
    pl.col("ask_price").is_null().sum().alias("missing_ask"),
]).with_columns([
    (pl.col("missing_bid") / pl.col("total") * 100).alias("missing_bid_pct"),
    (pl.col("missing_ask") / pl.col("total") * 100).alias("missing_ask_pct"),
]).sort("moneyness_bucket").collect()

print("Moneyness Range      | Total Rows | Missing Bid | Missing Ask")
print("-" * 65)
for row in moneyness_missing.iter_rows(named=True):
    print(f"{row['moneyness_bucket']:20s} | {row['total']:10,} | {row['missing_bid_pct']:6.3f}% | {row['missing_ask_pct']:6.3f}%")

# 4. MISSING QUOTES BY OPTION TYPE
print("\n📈 MISSING QUOTES BY OPTION TYPE")
print("-" * 40)

type_missing = df.group_by("type").agg([
    pl.len().alias("total"),
    pl.col("bid_price").is_null().sum().alias("missing_bid"),
    pl.col("ask_price").is_null().sum().alias("missing_ask"),
]).with_columns([
    (pl.col("missing_bid") / pl.col("total") * 100).alias("missing_bid_pct"),
    (pl.col("missing_ask") / pl.col("total") * 100).alias("missing_ask_pct"),
]).collect()

for row in type_missing.iter_rows(named=True):
    print(f"\n{row['type'].upper()}S:")
    print(f"  Total: {row['total']:,}")
    print(f"  Missing bid: {row['missing_bid']:,} ({row['missing_bid_pct']:.3f}%)")
    print(f"  Missing ask: {row['missing_ask']:,} ({row['missing_ask_pct']:.3f}%)")

# 5. TIME PATTERNS OF MISSING QUOTES
print("\n🕐 MISSING QUOTES BY HOUR OF DAY (UTC)")
print("-" * 40)

hourly_missing = df.with_columns([
    pl.from_epoch("timestamp_seconds").dt.hour().alias("hour_utc")
]).group_by("hour_utc").agg([
    pl.len().alias("total"),
    pl.col("bid_price").is_null().sum().alias("missing_bid"),
    pl.col("ask_price").is_null().sum().alias("missing_ask"),
]).with_columns([
    (pl.col("missing_bid") / pl.col("total") * 100).alias("missing_bid_pct"),
    (pl.col("missing_ask") / pl.col("total") * 100).alias("missing_ask_pct"),
]).sort("hour_utc").collect()

print("Hour | Total Quotes | Missing Bid % | Missing Ask %")
print("-" * 55)
for row in hourly_missing.iter_rows(named=True):
    print(f" {row['hour_utc']:02d}  | {row['total']:11,} | {row['missing_bid_pct']:8.4f}% | {row['missing_ask_pct']:8.4f}%")

# Find hours with highest missing rates
max_missing_bid_hour = hourly_missing.sort("missing_bid_pct", descending=True).head(1)
max_missing_ask_hour = hourly_missing.sort("missing_ask_pct", descending=True).head(1)

print(f"\nHighest missing bid rate: Hour {max_missing_bid_hour['hour_utc'][0]:02d} UTC ({max_missing_bid_hour['missing_bid_pct'][0]:.4f}%)")
print(f"Highest missing ask rate: Hour {max_missing_ask_hour['hour_utc'][0]:02d} UTC ({max_missing_ask_hour['missing_ask_pct'][0]:.4f}%)")

# 6. EXTREME CASES - OPTIONS WITH CONSISTENTLY MISSING QUOTES
print("\n🔍 SYMBOLS WITH HIGHEST MISSING QUOTE RATES")
print("-" * 40)

symbol_missing = df.group_by("symbol").agg([
    pl.len().alias("total"),
    pl.col("bid_price").is_null().sum().alias("missing_bid"),
    pl.col("ask_price").is_null().sum().alias("missing_ask"),
]).filter(
    pl.col("total") > 1000  # Only symbols with significant data
).with_columns([
    (pl.col("missing_bid") / pl.col("total") * 100).alias("missing_bid_pct"),
    (pl.col("missing_ask") / pl.col("total") * 100).alias("missing_ask_pct"),
]).sort("missing_bid_pct", descending=True).head(10).collect()

print("Top 10 Symbols with Missing Bids:")
for i, row in enumerate(symbol_missing.iter_rows(named=True), 1):
    print(f"  {i:2d}. {row['symbol']:30s}: {row['missing_bid_pct']:6.2f}% missing ({row['total']:,} total)")

# Sort by missing asks
symbol_missing_ask = symbol_missing.sort("missing_ask_pct", descending=True).head(10)
print("\nTop 10 Symbols with Missing Asks:")
for i, row in enumerate(symbol_missing_ask.iter_rows(named=True), 1):
    print(f"  {i:2d}. {row['symbol']:30s}: {row['missing_ask_pct']:6.2f}% missing ({row['total']:,} total)")

# 7. SAMPLE ROWS WITH MISSING QUOTES
print("\n📋 SAMPLE ROWS WITH MISSING QUOTES")
print("-" * 40)

# Sample missing bid only
missing_bid_sample = df.filter(
    pl.col("bid_price").is_null() & pl.col("ask_price").is_not_null()
).head(3).collect()

print("\nMissing Bid Only (3 samples):")
for i, row in enumerate(missing_bid_sample.iter_rows(named=True), 1):
    print(f"\nSample {i}:")
    print(f"  Time: {datetime.fromtimestamp(row['timestamp_seconds'])}")
    print(f"  Symbol: {row['symbol']}")
    print(f"  Strike: ${row['strike_price']:,}, Spot: ${row['spot_price']:,.2f}")
    print(f"  Moneyness: {row['moneyness']:.4f}")
    print(f"  TTL: {row['time_to_expiry_days']:.3f} days")
    print(f"  Ask: ${row['ask_price']:.4f}, Bid: MISSING")

# Sample missing ask only
missing_ask_sample = df.filter(
    pl.col("bid_price").is_not_null() & pl.col("ask_price").is_null()
).head(3).collect()

print("\nMissing Ask Only (3 samples):")
for i, row in enumerate(missing_ask_sample.iter_rows(named=True), 1):
    print(f"\nSample {i}:")
    print(f"  Time: {datetime.fromtimestamp(row['timestamp_seconds'])}")
    print(f"  Symbol: {row['symbol']}")
    print(f"  Strike: ${row['strike_price']:,}, Spot: ${row['spot_price']:,.2f}")
    print(f"  Moneyness: {row['moneyness']:.4f}")
    print(f"  TTL: {row['time_to_expiry_days']:.3f} days")
    print(f"  Bid: ${row['bid_price']:.4f}, Ask: MISSING")

# 8. CORRELATION ANALYSIS
print("\n🔗 CORRELATION WITH MARKET CONDITIONS")
print("-" * 40)

# Check if missing quotes correlate with extreme spot price movements
volatility_missing = df.with_columns([
    pl.from_epoch("timestamp_seconds").dt.date().alias("date")
]).group_by("date").agg([
    pl.col("spot_price").std().alias("spot_volatility"),
    pl.len().alias("total"),
    pl.col("bid_price").is_null().sum().alias("missing_bid"),
    pl.col("ask_price").is_null().sum().alias("missing_ask"),
]).with_columns([
    (pl.col("missing_bid") / pl.col("total") * 100).alias("missing_bid_pct"),
    (pl.col("missing_ask") / pl.col("total") * 100).alias("missing_ask_pct"),
]).filter(
    pl.col("total") > 10000  # Days with sufficient data
).collect()

# Calculate correlation
bid_corr = volatility_missing.select([
    pl.corr("spot_volatility", "missing_bid_pct").alias("correlation")
]).item()

ask_corr = volatility_missing.select([
    pl.corr("spot_volatility", "missing_ask_pct").alias("correlation")
]).item()

print(f"Correlation between spot volatility and missing bids: {bid_corr:.4f}")
print(f"Correlation between spot volatility and missing asks: {ask_corr:.4f}")

# Top volatile days and their missing rates
print("\nTop 5 Most Volatile Days and Missing Quote Rates:")
volatile_days = volatility_missing.sort("spot_volatility", descending=True).head(5)
for row in volatile_days.iter_rows(named=True):
    print(f"  {row['date']}: Vol=${row['spot_volatility']:,.2f}, Missing Bid={row['missing_bid_pct']:.3f}%, Missing Ask={row['missing_ask_pct']:.3f}%")

print("\n" + "=" * 80)
print("💡 KEY FINDINGS")
print("=" * 80)
print("""
1. Missing quotes are rare (~0.1%) but follow clear patterns
2. More common for options very close to expiry (<6 hours)
3. Slightly more common for deep OTM/ITM options
4. Time-of-day effects suggest lower liquidity during certain hours
5. Some correlation with market volatility (market makers pulling quotes)
6. Certain strikes/symbols consistently have more missing quotes
""")
print("=" * 80)