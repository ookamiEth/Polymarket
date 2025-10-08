#!/usr/bin/env python3
"""
Filter BTC Options from Deribit OPTIONS.csv

This script demonstrates how to:
1. Download ALL Deribit options (BTC, ETH, SOL, etc.)
2. Filter for BTC options only
3. Filter for short-dated options (expiring within N days)
4. Analyze the data (greeks, IV, prices over time)
5. Export filtered results

Usage:
    python filter_btc_options.py
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from tardis_dev import datasets


def download_options_data(from_date="2025-10-01", to_date="2025-10-01", api_key=None):
    """
    Download Deribit options_chain data for ALL options.

    Args:
        from_date: Start date (ISO format)
        to_date: End date (ISO format)
        api_key: API key (None for first day of month)

    Returns:
        Path to downloaded file
    """
    print("=" * 80)
    print("STEP 1: Download OPTIONS data from Tardis.dev")
    print("=" * 80)
    print(f"Date: {from_date} to {to_date}")
    print(f"Symbol: OPTIONS (includes ALL assets: BTC, ETH, SOL, etc.)")
    print(f"Access: {'Free (first day of month)' if not api_key else 'API key'}")
    print()

    datasets.download(
        exchange="deribit",
        data_types=["options_chain"],
        from_date=from_date,
        to_date=to_date,
        symbols=["OPTIONS"],  # ⚠️ Gets ALL options
        api_key=api_key,
        download_dir="./datasets"
    )

    filename = f"deribit_options_chain_{from_date}_OPTIONS.csv.gz"
    filepath = f"./datasets/{filename}"

    print(f"✓ Downloaded: {filepath}")
    return filepath


def load_and_filter_btc(filepath):
    """
    Load OPTIONS CSV and filter for BTC options only.

    Args:
        filepath: Path to OPTIONS CSV file

    Returns:
        DataFrame with BTC options only
    """
    print("\n" + "=" * 80)
    print("STEP 2: Load CSV and filter for BTC options")
    print("=" * 80)

    # Load CSV (pandas handles gzip automatically)
    print(f"Loading: {filepath}")
    df = pd.read_csv(filepath)

    print(f"Total rows: {len(df):,}")
    print(f"Total symbols: {df['symbol'].nunique():,}")

    # Show sample of all symbols
    print("\nSample of all symbols (first 20):")
    for symbol in df['symbol'].unique()[:20]:
        print(f"  {symbol}")

    # Filter for BTC options only
    btc_df = df[df['symbol'].str.startswith('BTC-')].copy()

    print(f"\n✓ BTC options rows: {len(btc_df):,}")
    print(f"✓ BTC option symbols: {btc_df['symbol'].nunique():,}")

    # Convert timestamps to datetime
    btc_df['datetime'] = pd.to_datetime(btc_df['timestamp'], unit='us')
    btc_df['local_datetime'] = pd.to_datetime(btc_df['local_timestamp'], unit='us')

    return btc_df


def parse_option_symbol(df):
    """
    Parse Deribit option symbol into components.

    Symbol format: BTC-29NOV25-50000-C
    - BASE: BTC
    - EXPIRY: 29NOV25 (November 29, 2025)
    - STRIKE: 50000
    - TYPE: C (Call) or P (Put)

    Args:
        df: DataFrame with 'symbol' column

    Returns:
        DataFrame with additional parsed columns
    """
    print("\n" + "=" * 80)
    print("STEP 3: Parse option symbols")
    print("=" * 80)

    def parse_symbol(symbol):
        """Parse a single symbol."""
        parts = symbol.split('-')
        if len(parts) < 4:
            return None

        return {
            'underlying': parts[0],
            'expiry_str': parts[1],
            'strike': float(parts[2]),
            'option_type': 'call' if parts[3] == 'C' else 'put'
        }

    # Parse all symbols
    parsed = df['symbol'].apply(parse_symbol)
    parsed_df = pd.DataFrame(parsed.tolist())

    # Merge back
    df = pd.concat([df, parsed_df], axis=1)

    # Parse expiry date (format: DDMMMYY)
    df['expiry_date'] = pd.to_datetime(df['expiry_str'], format='%d%b%y')

    # Convert expiration timestamp to datetime (already in microseconds)
    df['expiration_date'] = pd.to_datetime(df['expiration'], unit='us')

    print(f"✓ Parsed {len(df)} options")
    print(f"\nOption types distribution:")
    print(df['option_type'].value_counts())

    return df


def calculate_days_to_expiry(df, reference_date=None):
    """
    Calculate days to expiry from reference date.

    Args:
        df: DataFrame with 'expiry_date' column
        reference_date: Reference date (default: Oct 1, 2025)

    Returns:
        DataFrame with 'days_to_expiry' column
    """
    if reference_date is None:
        reference_date = datetime(2025, 10, 1)

    df['days_to_expiry'] = (df['expiry_date'] - reference_date).dt.days

    return df


def filter_short_dated(df, min_days=7, max_days=30):
    """
    Filter for short-dated options (expiring within N days).

    Args:
        df: DataFrame with 'days_to_expiry' column
        min_days: Minimum days to expiry
        max_days: Maximum days to expiry

    Returns:
        Filtered DataFrame
    """
    print("\n" + "=" * 80)
    print(f"STEP 4: Filter for short-dated options ({min_days}-{max_days} days)")
    print("=" * 80)

    short_dated = df[
        (df['days_to_expiry'] >= min_days) &
        (df['days_to_expiry'] <= max_days)
    ].copy()

    print(f"✓ Short-dated options rows: {len(short_dated):,}")
    print(f"✓ Short-dated option symbols: {short_dated['symbol'].nunique():,}")

    # Show expiry distribution
    print("\nExpiry distribution:")
    expiry_counts = short_dated['expiry_str'].value_counts().sort_index()
    for expiry, count in expiry_counts.items():
        print(f"  {expiry}: {count:,} updates")

    return short_dated


def analyze_options(df):
    """
    Perform analysis on options data.

    Args:
        df: DataFrame with options data
    """
    print("\n" + "=" * 80)
    print("STEP 5: Analyze BTC options")
    print("=" * 80)

    # Basic statistics
    print("\n1. Basic Statistics:")
    print(f"   Date range: {df['datetime'].min()} to {df['datetime'].max()}")
    print(f"   Total updates: {len(df):,}")
    print(f"   Unique options: {df['symbol'].nunique()}")

    # Underlying price range
    print(f"\n2. BTC Price Range:")
    print(f"   Min: ${df['underlying_price'].min():,.2f}")
    print(f"   Max: ${df['underlying_price'].max():,.2f}")
    print(f"   Mean: ${df['underlying_price'].mean():,.2f}")
    print(f"   Std: ${df['underlying_price'].std():,.2f}")

    # Strike price distribution
    print(f"\n3. Strike Price Distribution:")
    strikes = df['strike'].unique()
    strikes.sort()
    print(f"   Number of strikes: {len(strikes)}")
    print(f"   Strike range: ${strikes.min():,.0f} - ${strikes.max():,.0f}")

    # Show sample strikes
    if len(strikes) > 10:
        print(f"   Sample strikes: {', '.join([f'${s:,.0f}' for s in strikes[:10]])}, ...")
    else:
        print(f"   All strikes: {', '.join([f'${s:,.0f}' for s in strikes])}")

    # IV statistics
    print(f"\n4. Implied Volatility (mark_iv):")
    iv_stats = df['mark_iv'].describe()
    print(f"   Min: {iv_stats['min']:.2f}%")
    print(f"   Max: {iv_stats['max']:.2f}%")
    print(f"   Mean: {iv_stats['mean']:.2f}%")
    print(f"   Median: {iv_stats['50%']:.2f}%")

    # Greeks statistics
    print(f"\n5. Greeks Summary:")
    print(f"   Delta range: {df['delta'].min():.4f} to {df['delta'].max():.4f}")
    print(f"   Gamma range: {df['gamma'].min():.6f} to {df['gamma'].max():.6f}")
    print(f"   Vega range: {df['vega'].min():.2f} to {df['vega'].max():.2f}")
    print(f"   Theta range: {df['theta'].min():.2f} to {df['theta'].max():.2f}")

    # Option types
    print(f"\n6. Option Types:")
    type_counts = df['option_type'].value_counts()
    for opt_type, count in type_counts.items():
        pct = (count / len(df)) * 100
        print(f"   {opt_type.capitalize()}s: {count:,} updates ({pct:.1f}%)")

    # Updates per option
    print(f"\n7. Updates per Option:")
    updates_per_option = df.groupby('symbol').size()
    print(f"   Mean updates per option: {updates_per_option.mean():.0f}")
    print(f"   Median updates per option: {updates_per_option.median():.0f}")
    print(f"   Min updates: {updates_per_option.min()}")
    print(f"   Max updates: {updates_per_option.max()}")

    # Most active options
    print(f"\n8. Top 10 Most Active Options (by number of updates):")
    top_options = updates_per_option.nlargest(10)
    for symbol, count in top_options.items():
        print(f"   {symbol}: {count:,} updates")


def find_atm_options(df, reference_price=None):
    """
    Find at-the-money (ATM) options.

    Args:
        df: DataFrame with options data
        reference_price: Reference BTC price (default: mean underlying price)

    Returns:
        DataFrame with ATM options only
    """
    if reference_price is None:
        reference_price = df['underlying_price'].mean()

    print(f"\n9. At-The-Money (ATM) Options:")
    print(f"   Reference BTC price: ${reference_price:,.2f}")

    # Find closest strike to reference price
    df['strike_diff'] = abs(df['strike'] - reference_price)

    # Get unique options (one per symbol)
    unique_options = df.sort_values('timestamp').groupby('symbol').last()

    # Find ATM options (within 5% of spot)
    atm_range = reference_price * 0.05
    atm_options = unique_options[unique_options['strike_diff'] < atm_range].copy()
    atm_options = atm_options.sort_values('strike_diff')

    print(f"   ATM options found: {len(atm_options)}")
    print(f"\n   Top 10 ATM options:")
    for idx, (symbol, row) in enumerate(atm_options.head(10).iterrows(), 1):
        print(f"   {idx}. {symbol}")
        print(f"      Strike: ${row['strike']:,.0f} | Type: {row['option_type']}")
        print(f"      Delta: {row['delta']:.4f} | IV: {row['mark_iv']:.2f}%")
        print(f"      Mark: {row['mark_price']:.6f} BTC | Expiry: {row['expiry_str']}")

    return atm_options


def export_results(df, output_path):
    """
    Export filtered results to CSV.

    Args:
        df: DataFrame to export
        output_path: Output file path
    """
    print("\n" + "=" * 80)
    print("STEP 6: Export filtered results")
    print("=" * 80)

    df.to_csv(output_path, index=False)

    file_size_mb = pd.io.common.get_filepath_or_buffer(output_path)[0]

    print(f"✓ Exported to: {output_path}")
    print(f"  Rows: {len(df):,}")
    print(f"  Columns: {len(df.columns)}")


def main():
    """
    Main function to demonstrate full workflow.
    """
    print("\n")
    print("╔" + "═" * 78 + "╗")
    print("║" + " " * 18 + "FILTER BTC OPTIONS FROM DERIBIT" + " " * 29 + "║")
    print("╚" + "═" * 78 + "╝")
    print()

    # Configuration
    from_date = "2025-10-01"
    to_date = "2025-10-01"
    api_key = None  # Free access for first day of month

    # For demo: use example CSV instead of downloading
    USE_EXAMPLE = True

    if USE_EXAMPLE:
        print("Using example CSV file (change USE_EXAMPLE=False to download real data)")
        filepath = "example_deribit_options_chain_2025-10-01_BTC_SAMPLE.csv"
    else:
        # Step 1: Download data
        filepath = download_options_data(from_date, to_date, api_key)

    # Step 2: Load and filter for BTC
    btc_df = load_and_filter_btc(filepath)

    # Step 3: Parse symbols
    btc_df = parse_option_symbol(btc_df)

    # Step 3.5: Calculate days to expiry
    btc_df = calculate_days_to_expiry(btc_df, reference_date=datetime(2025, 10, 1))

    # Step 4: Filter for short-dated options
    short_dated = filter_short_dated(btc_df, min_days=7, max_days=30)

    # Step 5: Analyze
    analyze_options(short_dated)

    # Step 5.5: Find ATM options
    find_atm_options(short_dated)

    # Step 6: Export
    export_results(
        short_dated,
        f"btc_short_dated_options_{from_date}.csv"
    )

    print("\n" + "=" * 80)
    print("COMPLETE!")
    print("=" * 80)
    print("\nNext steps:")
    print("1. Analyze time series of specific options")
    print("2. Compare IV across different strikes")
    print("3. Study greek evolution over time")
    print("4. Calculate implied volatility surface")
    print("5. Backtest options trading strategies")


if __name__ == "__main__":
    main()
