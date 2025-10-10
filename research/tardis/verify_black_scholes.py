#!/usr/bin/env python3
"""
Verification Script: Compare Our Black-Scholes IV to Deribit's IV

This script loads the existing parquet file with Deribit's pre-calculated IVs
and compares them to our own Black-Scholes calculated IVs to verify correctness.

Expected result: Our calculated IVs should match Deribit's within ~0.01-0.1%
due to floating point precision differences.
"""

import polars as pl
import numpy as np
from black_scholes import add_implied_volatility_to_dataframe

# Load the existing parquet file with Deribit's IVs
print("=" * 80)
print("BLACK-SCHOLES IV VERIFICATION")
print("=" * 80)
print("\nLoading existing Deribit data with pre-calculated IVs...")

df = pl.read_parquet(
    '/Users/lgierhake/Documents/ETH/BT/research/tardis/datasets_deribit_options/'
    'deribit_options_chain_2025-10-01_OPTIONS.parquet'
)

print(f"Loaded {df.shape[0]:,} rows")
print()

# Filter to rows with valid bid/ask data
print("Filtering to rows with valid bid/ask prices and IVs...")
df_valid = df.filter(
    (pl.col('bid_price').is_not_null()) &
    (pl.col('ask_price').is_not_null()) &
    (pl.col('bid_iv').is_not_null()) &
    (pl.col('ask_iv').is_not_null()) &
    (pl.col('underlying_price').is_not_null()) &
    (pl.col('days_to_expiry') > 0)  # Only options with time remaining
)

print(f"Filtered to {df_valid.shape[0]:,} rows with complete data")
print()

# Sample for faster verification (optional)
SAMPLE_SIZE = 10000
if df_valid.shape[0] > SAMPLE_SIZE:
    print(f"Sampling {SAMPLE_SIZE:,} rows for verification...")
    df_valid = df_valid.sample(n=SAMPLE_SIZE, seed=42)
    print()

# Prepare data for Black-Scholes
print("Preparing data for Black-Scholes calculations...")

# Add time to expiry in years
df_valid = df_valid.with_columns([
    (pl.col('days_to_expiry') / 365.25).alias('time_to_expiry')
])

# Add risk-free rate (crypto uses 0%)
df_valid = df_valid.with_columns([
    pl.lit(0.0).alias('risk_free_rate')
])

print(f"✓ Prepared {df_valid.shape[0]:,} rows")
print()

# Calculate our own IVs using Black-Scholes
print("=" * 80)
print("CALCULATING IVs USING OUR BLACK-SCHOLES IMPLEMENTATION")
print("=" * 80)
print()

print("[VECTORIZED] Calculating bid IV...")
df_valid = add_implied_volatility_to_dataframe(
    df_valid,
    price_column='bid_price',
    output_column='bid_iv_calculated',
)

print("[VECTORIZED] Calculating ask IV...")
df_valid = add_implied_volatility_to_dataframe(
    df_valid,
    price_column='ask_price',
    output_column='ask_iv_calculated',
)

print()

# Compare results
print("=" * 80)
print("COMPARISON RESULTS")
print("=" * 80)
print()

# Calculate differences
df_comparison = df_valid.with_columns([
    (pl.col('bid_iv') - pl.col('bid_iv_calculated')).abs().alias('bid_iv_diff'),
    (pl.col('ask_iv') - pl.col('ask_iv_calculated')).abs().alias('ask_iv_diff'),
])

# Get statistics
bid_iv_stats = df_comparison.select([
    pl.col('bid_iv_diff').mean().alias('mean'),
    pl.col('bid_iv_diff').median().alias('median'),
    pl.col('bid_iv_diff').max().alias('max'),
    pl.col('bid_iv_diff').std().alias('std'),
]).to_dicts()[0]

ask_iv_stats = df_comparison.select([
    pl.col('ask_iv_diff').mean().alias('mean'),
    pl.col('ask_iv_diff').median().alias('median'),
    pl.col('ask_iv_diff').max().alias('max'),
    pl.col('ask_iv_diff').std().alias('std'),
]).to_dicts()[0]

print("BID IV COMPARISON:")
print(f"  Mean absolute difference: {bid_iv_stats['mean']:.4f}%")
print(f"  Median absolute difference: {bid_iv_stats['median']:.4f}%")
print(f"  Max absolute difference: {bid_iv_stats['max']:.4f}%")
print(f"  Std dev: {bid_iv_stats['std']:.4f}%")
print()

print("ASK IV COMPARISON:")
print(f"  Mean absolute difference: {ask_iv_stats['mean']:.4f}%")
print(f"  Median absolute difference: {ask_iv_stats['median']:.4f}%")
print(f"  Max absolute difference: {ask_iv_stats['max']:.4f}%")
print(f"  Std dev: {ask_iv_stats['std']:.4f}%")
print()

# Show some examples
print("=" * 80)
print("SAMPLE COMPARISONS (First 10 rows)")
print("=" * 80)
print()

sample_comparison = df_comparison.select([
    'symbol',
    'type',
    'strike_price',
    'underlying_price',
    'bid_price',
    'bid_iv',
    'bid_iv_calculated',
    'bid_iv_diff',
    'ask_price',
    'ask_iv',
    'ask_iv_calculated',
    'ask_iv_diff',
]).head(10)

print(sample_comparison)
print()

# Verdict
print("=" * 80)
print("VERDICT")
print("=" * 80)

# Tolerance: 0.5% average difference
tolerance = 0.5
bid_passed = bid_iv_stats['mean'] < tolerance
ask_passed = ask_iv_stats['mean'] < tolerance

if bid_passed and ask_passed:
    print("✅ SUCCESS: Our Black-Scholes IV calculations match Deribit's!")
    print(f"   Both bid and ask IVs are within {tolerance}% on average.")
    print()
    print("   This confirms:")
    print("   1. Deribit's IV is indeed calculated using Black-Scholes")
    print("   2. Our implementation is correct")
    print("   3. We can safely use quotes + our own BS calculations")
else:
    print("❌ WARNING: Differences exceed expected tolerance!")
    if not bid_passed:
        print(f"   Bid IV mean diff: {bid_iv_stats['mean']:.4f}% > {tolerance}%")
    if not ask_passed:
        print(f"   Ask IV mean diff: {ask_iv_stats['mean']:.4f}% > {tolerance}%")
    print()
    print("   This might indicate:")
    print("   1. Deribit uses different Black-Scholes parameters")
    print("   2. Our implementation needs adjustment")
    print("   3. Data quality issues")

print()
print("=" * 80)
