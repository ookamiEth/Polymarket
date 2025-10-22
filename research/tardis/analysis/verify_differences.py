#!/usr/bin/env python3

"""
Verify what's actually different between the two IV files
"""

import polars as pl

# Load the comparison results
comparison_file = "/Users/lgierhake/Documents/ETH/BT/research/tardis/analysis/output/comparison_chunked.parquet"

print("=" * 80)
print("INVESTIGATING WHY DIFFERENCES ARE SO SMALL")
print("=" * 80)

# Load comparison data
df = pl.read_parquet(comparison_file)

print(f"\nTotal rows analyzed: {len(df):,}")

# Check basic statistics
print("\n1. Basic difference statistics:")
print(f"   Mean IV (constant): {df['iv_mid_constant'].mean():.6f}")
print(f"   Mean IV (daily): {df['iv_mid_daily'].mean():.6f}")
print(f"   Mean difference: {df['iv_mid_diff_abs'].mean():.6f}")
print(f"   Max difference: {df['iv_mid_diff_abs'].abs().max():.6f}")

# Check if there's any systematic pattern by date
print("\n2. Differences by date:")
daily_stats = (
    df.group_by("date")
    .agg([
        pl.len().alias("count"),
        pl.col("iv_mid_diff_abs").mean().alias("mean_diff"),
        pl.col("iv_mid_diff_abs").abs().max().alias("max_abs_diff"),
        pl.col("time_to_expiry_days").mean().alias("avg_tte"),
    ])
    .sort("date")
)

print(daily_stats)

# Check the largest differences
print("\n3. Top 10 largest absolute differences:")
top_diffs = (
    df.select([
        "date",
        "symbol",
        "time_to_expiry_days",
        "moneyness",
        "iv_mid_constant",
        "iv_mid_daily",
        "iv_mid_diff_abs",
        "iv_mid_diff_rel"
    ])
    .sort("iv_mid_diff_abs", descending=True)
    .head(10)
)

for row in top_diffs.iter_rows(named=True):
    print(f"   {row['date']} {row['symbol']}: "
          f"Const={row['iv_mid_constant']:.4f}, Daily={row['iv_mid_daily']:.4f}, "
          f"Diff={row['iv_mid_diff_abs']:.4f} ({row['iv_mid_diff_rel']:.2f}%), "
          f"TTE={row['time_to_expiry_days']:.1f}d")

print("\n4. Differences by time to expiry:")
tte_stats = (
    df.with_columns([
        pl.when(pl.col("time_to_expiry_days") < 7)
        .then(pl.lit("<7d"))
        .when(pl.col("time_to_expiry_days") < 30)
        .then(pl.lit("7-30d"))
        .otherwise(pl.lit(">30d"))
        .alias("tte_bin")
    ])
    .group_by("tte_bin")
    .agg([
        pl.len().alias("count"),
        pl.col("iv_mid_diff_abs").mean().alias("mean_diff"),
        pl.col("iv_mid_diff_abs").abs().mean().alias("mean_abs_diff"),
    ])
    .sort("tte_bin")
)

print(tte_stats)

# Check if both files have the same exact values for some rows
print("\n5. Checking for exact matches (diff = 0):")
exact_matches = df.filter(pl.col("iv_mid_diff_abs") == 0).select(pl.len()).item()
print(f"   Rows with EXACT same IV: {exact_matches:,} ({exact_matches/len(df)*100:.1f}%)")

# Sample some rows to see actual values
print("\n6. Sample of actual values:")
sample = df.select([
    "timestamp_seconds",
    "symbol",
    "iv_mid_constant",
    "iv_mid_daily",
    "iv_mid_diff_abs"
]).head(20)

print(sample)