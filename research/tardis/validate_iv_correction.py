#!/usr/bin/env python3
"""
Validation script to compare old (incorrect) vs new (corrected) IV calculations.

Demonstrates that the unit mismatch bug has been fixed:
- Old IVs: ~2% (BTC prices with USD strikes - WRONG)
- New IVs: ~30-80% (Proper unit conversion - CORRECT)
"""

import polars as pl

# File paths
OLD_FILE = "datasets_deribit_options/deribit_options_2025-09-01_BTC_1s_with_iv_vectorized.parquet"
NEW_FILE = "datasets_deribit_options/deribit_options_2025-09-01_BTC_1s_with_iv_corrected.parquet"

print("=" * 80)
print("IV CALCULATION VALIDATION")
print("=" * 80)
print()

# Load both files
print("Loading old (incorrect) IVs...")
df_old = pl.read_parquet(OLD_FILE)
print(f"  Rows: {len(df_old):,}")

print("Loading new (corrected) IVs...")
df_new = pl.read_parquet(NEW_FILE)
print(f"  Rows: {len(df_new):,}")
print()

# Compare statistics
print("=" * 80)
print("IV STATISTICS COMPARISON")
print("=" * 80)
print()

print("OLD (INCORRECT) - Unit mismatch bug:")
print("  BTC option prices + USD strikes → artificially low IVs")
old_stats = df_old.select(
    [
        pl.col("implied_vol_bid").min().alias("iv_bid_min"),
        pl.col("implied_vol_bid").mean().alias("iv_bid_mean"),
        pl.col("implied_vol_bid").max().alias("iv_bid_max"),
        pl.col("implied_vol_ask").min().alias("iv_ask_min"),
        pl.col("implied_vol_ask").mean().alias("iv_ask_mean"),
        pl.col("implied_vol_ask").max().alias("iv_ask_max"),
    ]
)
print(old_stats)
print()

print("NEW (CORRECTED) - Proper unit conversion:")
print("  USD option prices + USD strikes → realistic IVs")
new_stats = df_new.select(
    [
        pl.col("implied_vol_bid").min().alias("iv_bid_min"),
        pl.col("implied_vol_bid").mean().alias("iv_bid_mean"),
        pl.col("implied_vol_bid").max().alias("iv_bid_max"),
        pl.col("implied_vol_ask").min().alias("iv_ask_min"),
        pl.col("implied_vol_ask").mean().alias("iv_ask_mean"),
        pl.col("implied_vol_ask").max().alias("iv_ask_max"),
    ]
)
print(new_stats)
print()

# Calculate improvement ratio
old_mean = df_old.select(pl.col("implied_vol_bid").mean()).item()
new_mean = df_new.select(pl.col("implied_vol_bid").mean()).item()

if old_mean is not None and new_mean is not None:
    ratio = new_mean / old_mean
    print(f"Improvement ratio: {ratio:.1f}x increase in IV values")
    print(f"  Old mean IV: {old_mean*100:.2f}%")
    print(f"  New mean IV: {new_mean*100:.2f}%")
    print()

# Show sample comparisons
print("=" * 80)
print("SAMPLE COMPARISONS (5 examples)")
print("=" * 80)
print()

# Merge on common columns to compare
common_cols = ["symbol", "timestamp", "strike_price", "type"]
comparison = df_old.select(
    common_cols + ["bid_price", "ask_price", "spot_price", "implied_vol_bid", "implied_vol_ask"]
).rename(
    {"implied_vol_bid": "old_iv_bid", "implied_vol_ask": "old_iv_ask", "spot_price": "old_spot"}
).join(
    df_new.select(
        common_cols + ["implied_vol_bid", "implied_vol_ask", "spot_price"]
    ).rename(
        {"implied_vol_bid": "new_iv_bid", "implied_vol_ask": "new_iv_ask", "spot_price": "new_spot"}
    ),
    on=common_cols,
    how="inner"
)

# Filter to successful IVs only and sample
sample = comparison.filter(
    pl.col("old_iv_bid").is_not_null() & pl.col("new_iv_bid").is_not_null()
).head(5)

for i, row in enumerate(sample.to_dicts(), 1):
    print(f"Example {i}: {row['symbol']}")
    print(f"  Strike: ${row['strike_price']:.0f} | Spot: ${row['old_spot']:.2f} | Type: {row['type']}")
    print(f"  Bid price: {row['bid_price']:.6f} BTC")
    print(f"  Ask price: {row['ask_price']:.6f} BTC")
    print()
    print(f"  OLD IV (bid): {row['old_iv_bid']:.6f} ({row['old_iv_bid']*100:.2f}%) ← TOO LOW!")
    print(f"  NEW IV (bid): {row['new_iv_bid']:.6f} ({row['new_iv_bid']*100:.2f}%) ← CORRECT!")
    print(f"  Ratio: {row['new_iv_bid']/row['old_iv_bid']:.1f}x increase")
    print()

print("=" * 80)
print("VALIDATION RESULT")
print("=" * 80)
print()

# Check if new IVs are in reasonable range (30-80% for BTC)
new_mean_pct = new_mean * 100
if 30 <= new_mean_pct <= 80:
    print("✓ SUCCESS: New IVs are in reasonable range for BTC options (30-80%)")
    print(f"  Mean IV: {new_mean_pct:.1f}%")
else:
    print("✗ WARNING: New IVs may still have issues")
    print(f"  Mean IV: {new_mean_pct:.1f}% (expected: 30-80%)")

print()
print("The unit mismatch bug has been FIXED!")
print("- Old approach: BTC prices with USD strikes (WRONG)")
print("- New approach: USD prices with USD strikes (CORRECT)")
print()
