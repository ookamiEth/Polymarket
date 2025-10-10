#!/usr/bin/env python3
"""
Analyze the actual options_chain data from 2025-10-01 to see what expiry dates exist.
This will show us the REAL options that were available on that day.
"""

import polars as pl
from datetime import datetime

# Read the parquet file
print("Reading options_chain data from 2025-10-01...")
df = pl.read_parquet(
    "/Users/lgierhake/Documents/ETH/BT/research/tardis/datasets_deribit_options/deribit_options_chain_2025-10-01_OPTIONS.parquet"
)

print(f"\nDataFrame shape: {df.shape[0]:,} rows × {df.shape[1]} columns")
print(f"\nColumn names:")
print(df.columns)

print(f"\nSchema:")
print(df.schema)

print(f"\nFirst 5 rows:")
print(df.head(5))

# Check timestamp range
print(f"\n" + "=" * 80)
print("TIMESTAMP ANALYSIS")
print("=" * 80)

if 'timestamp' in df.columns:
    # Convert timestamp to datetime for analysis
    df_with_dt = df.with_columns([
        (pl.col('timestamp') / 1_000_000_000).cast(pl.Datetime('s')).alias('datetime')
    ])

    print(f"Time range in data:")
    print(f"  First timestamp: {df_with_dt['datetime'].min()}")
    print(f"  Last timestamp: {df_with_dt['datetime'].max()}")

# Check for instrument_name or symbol columns
print(f"\n" + "=" * 80)
print("EXPIRY DATE ANALYSIS")
print("=" * 80)

# Look for instrument names like BTC-01OCT25-60000-C
if 'instrument_name' in df.columns:
    # Extract unique instrument names
    instruments = df['instrument_name'].unique().sort()
    print(f"\nTotal unique instruments: {len(instruments)}")
    print(f"\nFirst 10 instruments:")
    for inst in instruments[:10]:
        print(f"  {inst}")

    # Parse expiry dates from instrument names
    # Format: BTC-01OCT25-60000-C or BTC-1OCT25-60000-C
    import re

    expiry_dates = []
    for inst in instruments:
        match = re.match(r'BTC-(\d{1,2}[A-Z]{3}\d{2})-', str(inst))
        if match:
            expiry_str = match.group(1)
            try:
                expiry_date = datetime.strptime(expiry_str, '%d%b%y')
                expiry_dates.append(expiry_date)
            except:
                pass

    if expiry_dates:
        expiry_dates_sorted = sorted(set(expiry_dates))
        reference_date = datetime(2025, 10, 1)

        print(f"\n" + "=" * 80)
        print("OPTIONS EXPIRY DATES ON 2025-10-01")
        print("=" * 80)
        print(f"{'Expiry Date':<15} {'Days from Oct 1':<20} {'Available?':<15}")
        print("-" * 80)

        for expiry in expiry_dates_sorted[:20]:  # Show first 20
            days_away = (expiry - reference_date).days
            available = "YES" if days_away >= 0 and days_away <= 3 else ""
            print(f"{expiry.strftime('%Y-%m-%d'):<15} {days_away:<20} {available:<15}")

        # Count short-dated options
        short_dated = [e for e in expiry_dates if (e - reference_date).days >= 0 and (e - reference_date).days <= 3]
        print(f"\n" + "=" * 80)
        print(f"SUMMARY:")
        print(f"  Options expiring in 0-3 days from 2025-10-01: {len(short_dated)}")
        print(f"  Closest expiry: {min(expiry_dates_sorted)} ({(min(expiry_dates_sorted) - reference_date).days} days)")
        print("=" * 80)

elif 'symbol' in df.columns:
    print("Using 'symbol' column instead...")
    symbols = df['symbol'].unique().sort()
    print(f"First 10 symbols: {symbols[:10]}")

# Check if there's a separate expiration column
if 'expiration' in df.columns or 'expiration_timestamp' in df.columns:
    exp_col = 'expiration' if 'expiration' in df.columns else 'expiration_timestamp'
    print(f"\nDirect expiration column found: {exp_col}")
    print(f"Sample values: {df[exp_col].head(10)}")
