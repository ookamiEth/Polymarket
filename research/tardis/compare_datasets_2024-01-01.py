#!/usr/bin/env python3
"""
Compare the two different Deribit options datasets for 2024-01-01:
1. deribit_options_2024-01-01_BTC_1s.parquet (sampled/filtered data with IVs)
2. deribit_options_quotes_atm3pct_2024-01-01.csv.gz (raw ATM quotes)
"""

import polars as pl

print("=" * 80)
print("COMPARING DERIBIT OPTIONS DATASETS (2024-01-01)")
print("=" * 80)
print()

# Load dataset 1: Filtered data with IVs
print("Dataset 1: Filtered data with calculated IVs")
print("-" * 80)
df1 = pl.read_parquet("datasets_deribit_options/deribit_options_2024-01-01_BTC_1s_with_iv.parquet")
print(f"Rows: {len(df1):,}")
print(f"Columns: {df1.columns}")
print(f"Date range: {df1.select(pl.from_epoch(pl.col('timestamp'), time_unit='us').min())} to {df1.select(pl.from_epoch(pl.col('timestamp'), time_unit='us').max())}")
print()

# Show successful IVs
success = df1.filter(pl.col("iv_calc_status") == "success")
print(f"Successful IV calculations: {len(success):,} ({len(success)/len(df1)*100:.1f}%)")
if len(success) > 0:
    print("Sample successful IVs:")
    print(
        success.select(
            ["symbol", "strike_price", "spot_price", "bid_price", "ask_price", "implied_vol_bid", "implied_vol_ask"]
        ).head(5)
    )
print()

# Load dataset 2: Raw ATM quotes
print("Dataset 2: Raw ATM quotes (3% ATM filter)")
print("-" * 80)
df2 = pl.read_csv("datasets_deribit_options_quotes_atm3pct/deribit_options_quotes_atm3pct_2024-01-01.csv.gz")
print(f"Rows: {len(df2):,}")
print(f"Columns: {df2.columns}")
print(f"Date range: {df2.select(pl.from_epoch(pl.col('timestamp'), time_unit='us').min())} to {df2.select(pl.from_epoch(pl.col('timestamp'), time_unit='us').max())}")
print()

# Sample data
print("Sample quotes:")
print(df2.select(["symbol", "strike_price", "bid_price", "ask_price"]).head(5))
print()

# Comparison
print("=" * 80)
print("COMPARISON")
print("=" * 80)
print()

print("Key Differences:")
print(f"1. Data Volume:")
print(f"   - Dataset 1 (filtered): {len(df1):,} rows")
print(f"   - Dataset 2 (ATM quotes): {len(df2):,} rows")
print(f"   - Ratio: Dataset 2 has {len(df2)/len(df1):.1f}x more data")
print()

print(f"2. Data Type:")
print(f"   - Dataset 1: Sampled/filtered data (likely 1-second snapshots)")
print(f"   - Dataset 2: Tick-by-tick quotes (every quote update)")
print()

print(f"3. Filtering:")
print(f"   - Dataset 1: Applied quality filters (spread, moneyness, time to expiry)")
print(f"   - Dataset 2: Only ATM filter (within 3% of spot)")
print()

print(f"4. IV Calculations:")
print(f"   - Dataset 1: IVs calculated, but low success rate (1.5%)")
print(f"   - Dataset 2: No IVs yet (raw quote data)")
print()

# Check for overlapping options
print("=" * 80)
print("OVERLAP ANALYSIS")
print("=" * 80)
print()

# Get unique symbols from each dataset
symbols1 = set(df1["symbol"].unique().to_list())
symbols2 = set(df2["symbol"].unique().to_list())

print(f"Unique options in Dataset 1: {len(symbols1):,}")
print(f"Unique options in Dataset 2: {len(symbols2):,}")
print(f"Overlapping options: {len(symbols1 & symbols2):,}")
print()

# Show some overlapping options
overlapping = list(symbols1 & symbols2)[:5]
if overlapping:
    print("Sample overlapping options:")
    for sym in overlapping:
        print(f"  - {sym}")
        # Compare prices
        d1_sample = df1.filter(pl.col("symbol") == sym).select(["bid_price", "ask_price"]).head(1)
        d2_sample = df2.filter(pl.col("symbol") == sym).select(["bid_price", "ask_price"]).head(1)
        print(f"    Dataset 1: bid={d1_sample['bid_price'][0]:.6f}, ask={d1_sample['ask_price'][0]:.6f}")
        print(f"    Dataset 2: bid={d2_sample['bid_price'][0]:.6f}, ask={d2_sample['ask_price'][0]:.6f}")
print()

print("=" * 80)
print("RECOMMENDATION")
print("=" * 80)
print()
print("Dataset 2 (ATM quotes) is more comprehensive:")
print("✓ 62x more data points")
print("✓ Tick-by-tick granularity")
print("✓ Focused on ATM options (most liquid)")
print()
print("For IV calculation on quotes data:")
print("1. Use the consolidated quotes file (if available)")
print("2. Or process ATM quotes with the corrected IV calculator")
print("3. Apply same quality filters for fair comparison")
