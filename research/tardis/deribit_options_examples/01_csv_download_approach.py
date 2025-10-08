#!/usr/bin/env python3
"""
Deribit Options Data - CSV Download Approach

This script demonstrates how to download Deribit options data using the tardis-dev package.
The CSV approach provides normalized data that's already processed and ready to use.

Date: October 1, 2025 (free access - first day of month)
Package: tardis-dev

Data types available for options:
- options_chain: Aggregated options data (strike, expiry, greeks, prices)
- trades: Individual trades on options
- quotes: Best bid/ask quotes
- book_snapshot_25: Top 25 order book levels
"""

from tardis_dev import datasets
import pandas as pd
import os
from datetime import datetime, timedelta


def download_deribit_options_csv(
    from_date="2025-10-01",
    to_date="2025-10-01",
    data_types=None,
    api_key=None,
    download_dir="./datasets"
):
    """
    Download Deribit options data in CSV format.

    Args:
        from_date: Start date (ISO format)
        to_date: End date (ISO format)
        data_types: List of data types to download
        api_key: Tardis.dev API key (None for free access to first day of month)
        download_dir: Directory to save downloaded files
    """

    if data_types is None:
        # For options, we're interested in options_chain, trades, and quotes
        data_types = ["options_chain", "trades", "quotes"]

    print(f"Downloading Deribit options data from {from_date} to {to_date}")
    print(f"Data types: {data_types}")
    print(f"Using {'free access (first day of month)' if not api_key else 'API key'}")
    print("-" * 80)

    # Download using the OPTIONS grouped symbol to get all options at once
    datasets.download(
        exchange="deribit",
        data_types=data_types,
        from_date=from_date,
        to_date=to_date,
        symbols=["OPTIONS"],  # Special grouped symbol for all options
        api_key=api_key,
        download_dir=download_dir
    )

    print("\nDownload complete!")
    print(f"Files saved to: {download_dir}")

    return download_dir


def analyze_options_chain_csv(csv_path):
    """
    Load and analyze the options_chain CSV file.

    The options_chain CSV contains:
    - exchange, symbol, timestamp, local_timestamp
    - underlying_price, underlying_index
    - For each option: strike, expiration, option_type (call/put), bid, ask, last_price
    - Greeks: delta, gamma, theta, vega, rho
    - implied_volatility, open_interest, volume
    """

    print(f"\nAnalyzing options_chain CSV: {csv_path}")

    if not os.path.exists(csv_path):
        print(f"File not found: {csv_path}")
        return None

    # Read CSV (may be gzipped)
    df = pd.read_csv(csv_path)

    print(f"\nDataset shape: {df.shape[0]} rows, {df.shape[1]} columns")
    print(f"\nColumns: {list(df.columns)}")

    # Show first few rows
    print("\nFirst 5 rows:")
    print(df.head())

    # Show data types
    print("\nData types:")
    print(df.dtypes)

    # If we have options data, let's analyze it
    if 'symbol' in df.columns:
        print(f"\nUnique symbols (first 20): {df['symbol'].unique()[:20]}")
        print(f"Total unique symbols: {df['symbol'].nunique()}")

    # Filter for short-dated options (example: expiring within 7 days)
    if 'timestamp' in df.columns and 'symbol' in df.columns:
        # Parse timestamp (microseconds since epoch)
        df['datetime'] = pd.to_datetime(df['timestamp'], unit='us')

        # Extract expiry from symbol (Deribit format: BTC-29NOV24-50000-C)
        # This is a simplified parser - actual symbols may vary
        print("\nSample symbol formats:")
        for symbol in df['symbol'].unique()[:5]:
            print(f"  {symbol}")

    return df


def filter_short_dated_options(df, days_to_expiry=7):
    """
    Filter options that expire within specified days.

    Deribit option symbol format: {BASE}-{EXPIRY}-{STRIKE}-{TYPE}
    Example: BTC-29NOV24-50000-C
    """

    if df is None or df.empty:
        return None

    print(f"\nFiltering for options expiring within {days_to_expiry} days...")

    # This would require parsing the symbol to extract expiry date
    # For demonstration, we'll just show the concept
    print("Note: Actual implementation would parse expiry from symbol")
    print("Example symbols:")
    if 'symbol' in df.columns:
        for symbol in df['symbol'].unique()[:10]:
            print(f"  {symbol}")

    return df


if __name__ == "__main__":
    # Example 1: Download options_chain data for Oct 1, 2025 (free access)
    print("=" * 80)
    print("EXAMPLE 1: Download options_chain CSV")
    print("=" * 80)

    download_dir = download_deribit_options_csv(
        from_date="2025-10-01",
        to_date="2025-10-01",
        data_types=["options_chain"],
        api_key=None,  # Free access for first day of month
        download_dir="./datasets_deribit_options"
    )

    # Example 2: Analyze the downloaded CSV
    print("\n" + "=" * 80)
    print("EXAMPLE 2: Analyze downloaded CSV")
    print("=" * 80)

    # Find the downloaded file
    # Format: deribit_options_chain_2025-10-01_OPTIONS.csv.gz
    csv_filename = "deribit_options_chain_2025-10-01_OPTIONS.csv.gz"
    csv_path = os.path.join(download_dir, csv_filename)

    if os.path.exists(csv_path):
        df = analyze_options_chain_csv(csv_path)

        if df is not None:
            # Filter for short-dated options
            filter_short_dated_options(df, days_to_expiry=7)
    else:
        print(f"\nNote: File not found at {csv_path}")
        print("The actual filename may be different. Check the download directory.")
        print(f"Expected location: {os.path.abspath(download_dir)}")

    print("\n" + "=" * 80)
    print("COMPLETE!")
    print("=" * 80)
