#!/usr/bin/env python3
"""
Extract trading insights from the filtered ATM short-dated options data.
Focus on practical patterns that could inform trading strategies.
"""

import polars as pl
from datetime import datetime
import numpy as np

OUTPUT_FILE = "quotes_1s_atm_short_dated_optimized.parquet"

print("=" * 80)
print("TRADING INSIGHTS FROM ATM SHORT-DATED OPTIONS")
print("=" * 80)

# Load data lazily
df = pl.scan_parquet(OUTPUT_FILE)

# 1. PUT-CALL PARITY ANALYSIS
print("\n📊 PUT-CALL PARITY & SKEW ANALYSIS")
print("-" * 40)

# Find matching put-call pairs (same strike, expiry, timestamp)
# Extract components from symbol: BTC-EXPIRY-STRIKE-TYPE
pairs_df = df.with_columns([
    pl.col("symbol").str.extract(r"^([^-]+)-([^-]+)-([^-]+)-([CP])$", 1).alias("underlying"),
    pl.col("symbol").str.extract(r"^([^-]+)-([^-]+)-([^-]+)-([CP])$", 2).alias("expiry"),
    pl.col("symbol").str.extract(r"^([^-]+)-([^-]+)-([^-]+)-([CP])$", 3).alias("strike_str"),
]).with_columns([
    pl.col("strike_str").cast(pl.Int64).alias("strike_int")
])

# Get put-call pairs at same timestamp
call_data = pairs_df.filter(pl.col("type") == "call").select([
    "timestamp_seconds", "expiry", "strike_int", "spot_price",
    pl.col("mid_price").alias("call_mid"),
    pl.col("spread_abs").alias("call_spread"),
    pl.col("time_to_expiry_days")
])

put_data = pairs_df.filter(pl.col("type") == "put").select([
    "timestamp_seconds", "expiry", "strike_int",
    pl.col("mid_price").alias("put_mid"),
    pl.col("spread_abs").alias("put_spread")
])

# Join to find pairs
pairs = call_data.join(
    put_data,
    on=["timestamp_seconds", "expiry", "strike_int"],
    how="inner"
).head(100000).collect()  # Sample for analysis

if len(pairs) > 0:
    # Calculate put-call parity deviation
    pairs = pairs.with_columns([
        (pl.col("call_mid") - pl.col("put_mid")).alias("call_put_diff"),
        ((pl.col("call_mid") - pl.col("put_mid")) / pl.col("call_mid") * 100).alias("skew_pct")
    ])

    skew_stats = pairs.select([
        pl.col("skew_pct").mean().alias("mean_skew"),
        pl.col("skew_pct").std().alias("std_skew"),
        pl.col("skew_pct").quantile(0.25).alias("q1_skew"),
        pl.col("skew_pct").quantile(0.75).alias("q3_skew"),
    ]).row(0, named=True)

    print(f"Put-Call Skew (Call - Put as % of Call):")
    print(f"  Mean: {skew_stats['mean_skew']:.2f}%")
    print(f"  Std Dev: {skew_stats['std_skew']:.2f}%")
    print(f"  IQR: [{skew_stats['q1_skew']:.2f}%, {skew_stats['q3_skew']:.2f}%]")

    # Skew by moneyness
    print("\nSkew by Strike Distance from Spot:")
    moneyness_buckets = pairs.with_columns([
        ((pl.col("strike_int") - pl.col("spot_price")) / pl.col("spot_price") * 100).alias("strike_distance_pct")
    ]).with_columns([
        pl.when(pl.col("strike_distance_pct") < -2).then(pl.lit("< -2%"))
        .when(pl.col("strike_distance_pct") < -1).then(pl.lit("-2% to -1%"))
        .when(pl.col("strike_distance_pct") < 0).then(pl.lit("-1% to 0%"))
        .when(pl.col("strike_distance_pct") < 1).then(pl.lit("0% to 1%"))
        .when(pl.col("strike_distance_pct") < 2).then(pl.lit("1% to 2%"))
        .otherwise(pl.lit("> 2%"))
        .alias("moneyness_bucket")
    ]).group_by("moneyness_bucket").agg([
        pl.col("skew_pct").mean().alias("avg_skew"),
        pl.len().alias("count")
    ]).sort("moneyness_bucket")

    for row in moneyness_buckets.iter_rows(named=True):
        print(f"  {row['moneyness_bucket']:12s}: {row['avg_skew']:6.2f}% ({row['count']:,} pairs)")

# 2. LIQUIDITY ANALYSIS
print("\n💧 LIQUIDITY PATTERNS")
print("-" * 40)

# Liquidity by time of day (UTC)
hourly_liquidity = df.with_columns([
    pl.from_epoch("timestamp_seconds").dt.hour().alias("hour_utc")
]).group_by("hour_utc").agg([
    pl.len().alias("quote_count"),
    pl.col("spread_abs").mean().alias("avg_spread"),
    (pl.col("spread_abs") / pl.col("mid_price") * 100).mean().alias("avg_spread_pct"),
    pl.col("bid_amount").mean().alias("avg_bid_size"),
    pl.col("ask_amount").mean().alias("avg_ask_size"),
]).sort("hour_utc").collect()

print("Liquidity by Hour (UTC):")
print("Hour | Quotes   | Avg Spread | Spread % | Bid Size | Ask Size")
print("-" * 65)
for row in hourly_liquidity.iter_rows(named=True):
    print(f" {row['hour_utc']:02d}  | {row['quote_count']:8,} | ${row['avg_spread']:.4f}  | {row['avg_spread_pct']:6.2f}% | {row['avg_bid_size']:.2f} | {row['avg_ask_size']:.2f}")

# 3. VOLATILITY SMILE ANALYSIS
print("\n😊 VOLATILITY SMILE PATTERN")
print("-" * 40)

# Implied volatility proxy: premium relative to intrinsic value
smile_analysis = df.with_columns([
    ((pl.col("strike_price") - pl.col("spot_price")) / pl.col("spot_price") * 100).alias("strike_distance_pct"),
    pl.when(pl.col("type") == "call")
    .then(pl.max_horizontal([pl.col("spot_price") - pl.col("strike_price"), 0]))
    .otherwise(pl.max_horizontal([pl.col("strike_price") - pl.col("spot_price"), 0]))
    .alias("intrinsic_value")
]).with_columns([
    (pl.col("mid_price") - pl.col("intrinsic_value")).alias("time_value")
]).filter(
    pl.col("time_value") > 0  # Only options with time value
).with_columns([
    pl.when(pl.col("strike_distance_pct") < -2).then(pl.lit("-3% to -2%"))
    .when(pl.col("strike_distance_pct") < -1).then(pl.lit("-2% to -1%"))
    .when(pl.col("strike_distance_pct") < 0).then(pl.lit("-1% to 0%"))
    .when(pl.col("strike_distance_pct") < 1).then(pl.lit("0% to 1%"))
    .when(pl.col("strike_distance_pct") < 2).then(pl.lit("1% to 2%"))
    .otherwise(pl.lit("2% to 3%"))
    .alias("strike_bucket")
]).group_by(["strike_bucket", "type"]).agg([
    pl.col("time_value").mean().alias("avg_time_value"),
    pl.len().alias("count")
]).sort(["type", "strike_bucket"]).collect()

print("Average Time Value by Strike Distance:")
for option_type in ["call", "put"]:
    print(f"\n{option_type.upper()}S:")
    type_data = smile_analysis.filter(pl.col("type") == option_type)
    for row in type_data.iter_rows(named=True):
        print(f"  {row['strike_bucket']:12s}: ${row['avg_time_value']:.4f} ({row['count']:,} quotes)")

# 4. EXTREME MOVEMENT DETECTION
print("\n🚨 EXTREME PRICE MOVEMENTS")
print("-" * 40)

# Find large spot price movements
spot_movements = df.with_columns([
    pl.from_epoch("timestamp_seconds").dt.date().alias("date")
]).group_by("date").agg([
    pl.col("spot_price").min().alias("min_spot"),
    pl.col("spot_price").max().alias("max_spot"),
    pl.col("spot_price").std().alias("spot_std"),
]).with_columns([
    ((pl.col("max_spot") - pl.col("min_spot")) / pl.col("min_spot") * 100).alias("daily_range_pct")
]).sort("daily_range_pct", descending=True).head(10).collect()

print("Top 10 Days by Price Range:")
for i, row in enumerate(spot_movements.iter_rows(named=True), 1):
    print(f"  {i:2d}. {row['date']}: {row['daily_range_pct']:.2f}% range")
    print(f"      Low: ${row['min_spot']:,.2f}, High: ${row['max_spot']:,.2f}")

# 5. OPTION FLOW INSIGHTS
print("\n📈 OPTION FLOW PATTERNS")
print("-" * 40)

# Volume by strike relative to ATM
flow_analysis = df.with_columns([
    (((pl.col("strike_price") - pl.col("spot_price")) / 1000).round(0) * 1000).alias("strike_bucket")
]).group_by(["strike_bucket", "type"]).agg([
    pl.len().alias("quote_count"),
    pl.col("bid_amount").mean().alias("avg_bid_size"),
    pl.col("ask_amount").mean().alias("avg_ask_size"),
]).filter(
    pl.col("quote_count") > 10000  # Significant volume only
).sort(["type", "strike_bucket"]).head(20).collect()

print("High Volume Strikes (relative to spot):")
for option_type in ["call", "put"]:
    print(f"\n{option_type.upper()}S:")
    type_data = flow_analysis.filter(pl.col("type") == option_type)
    for row in type_data.head(5).iter_rows(named=True):
        offset = f"+${row['strike_bucket']:,.0f}" if row['strike_bucket'] >= 0 else f"-${abs(row['strike_bucket']):,.0f}"
        print(f"  {offset:10s}: {row['quote_count']:8,} quotes, Bid: {row['avg_bid_size']:.2f}, Ask: {row['avg_ask_size']:.2f}")

# 6. WEEKEND EFFECT
print("\n📅 WEEKEND EFFECT ANALYSIS")
print("-" * 40)

# Compare weekday vs weekend patterns
weekend_analysis = df.with_columns([
    pl.from_epoch("timestamp_seconds").dt.weekday().alias("weekday")
]).with_columns([
    pl.when(pl.col("weekday").is_in([6, 7])).then(pl.lit("Weekend")).otherwise(pl.lit("Weekday")).alias("day_type")
]).group_by("day_type").agg([
    pl.len().alias("quote_count"),
    pl.col("spread_abs").mean().alias("avg_spread"),
    (pl.col("spread_abs") / pl.col("mid_price") * 100).mean().alias("avg_spread_pct"),
    pl.col("mid_price").mean().alias("avg_premium"),
]).collect()

print("Weekday vs Weekend Comparison:")
for row in weekend_analysis.iter_rows(named=True):
    print(f"\n{row['day_type']}:")
    print(f"  Quotes: {row['quote_count']:,}")
    print(f"  Avg Spread: ${row['avg_spread']:.4f} ({row['avg_spread_pct']:.2f}%)")
    print(f"  Avg Premium: ${row['avg_premium']:.4f}")

print("\n" + "=" * 80)
print("✅ TRADING INSIGHTS COMPLETE")
print("=" * 80)