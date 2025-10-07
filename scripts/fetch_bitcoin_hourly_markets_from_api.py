#!/usr/bin/env python3
"""
Fetch all Bitcoin hourly markets from Polymarket Gamma API

Queries the Polymarket Gamma API to get complete history of Bitcoin up/down hourly markets
and saves them to a parquet file for analysis.

Usage:
    uv run python scripts/fetch_bitcoin_hourly_markets_from_api.py
    uv run python scripts/fetch_bitcoin_hourly_markets_from_api.py --output data/markets/bitcoin_hourly_all_markets.parquet
"""

import requests
import polars as pl
from datetime import datetime
import time
import argparse
from pathlib import Path

API_BASE = 'https://gamma-api.polymarket.com/markets'
DEFAULT_OUTPUT = 'data/markets/bitcoin_hourly_all_markets.parquet'

def fetch_bitcoin_hourly_markets(verbose: bool = False) -> list:
    """
    Fetch all Bitcoin hourly markets from Gamma API with pagination

    Returns:
        List of market dictionaries
    """
    print('=== Fetching Bitcoin Hourly Markets from Polymarket Gamma API ===\n')

    all_markets = []
    offset = 0
    limit = 500
    page = 1
    max_pages = 200  # Safety limit

    while page <= max_pages:
        if verbose or page % 10 == 0:
            print(f'Fetching page {page} (offset={offset}, limit={limit})...')

        # Query parameters
        params = {
            'limit': limit,
            'offset': offset,
            'order': 'endDate',
            'ascending': 'false'
        }

        try:
            response = requests.get(API_BASE, params=params, timeout=30)
            response.raise_for_status()
            markets = response.json()

            if not markets or len(markets) == 0:
                if verbose:
                    print(f'  No more markets returned')
                break

            # Filter for Bitcoin hourly markets
            # Pattern: "bitcoin-up-or-down-[date]-[time]" (not the 15m variant with timestamp)
            bitcoin_hourly = []
            for m in markets:
                slug = m.get('slug', '')
                # Include if it starts with bitcoin-up-or-down- but NOT if it's the 15m variant
                if slug.startswith('bitcoin-up-or-down-') and 'btc-up-or-down-15m' not in slug:
                    bitcoin_hourly.append(m)

            if verbose:
                print(f'  Received {len(markets)} markets, {len(bitcoin_hourly)} Bitcoin hourly markets')

            all_markets.extend(bitcoin_hourly)

            # Check if we got fewer than limit (last page)
            if len(markets) < limit:
                if verbose:
                    print(f'  Last page reached')
                break

            offset += limit
            page += 1

            # Rate limiting - be nice to the API
            time.sleep(0.3)

        except Exception as e:
            print(f'  Error on page {page}: {e}')
            break

    print(f'\nTotal Bitcoin hourly markets fetched: {len(all_markets)}')
    return all_markets


def analyze_markets(markets: list):
    """Analyze the fetched markets and print summary statistics"""
    if not markets:
        print("No markets to analyze")
        return

    print('\n=== Market Analysis ===\n')

    # Convert to DataFrame for analysis
    df = pl.DataFrame(markets)

    # Parse dates
    df = df.with_columns([
        pl.col('startDate').str.to_datetime(time_zone='UTC').alias('start_datetime'),
        pl.col('endDate').str.to_datetime(time_zone='UTC').alias('end_datetime'),
        pl.col('createdAt').str.to_datetime(time_zone='UTC').alias('created_datetime'),
    ])

    print(f'Total markets: {len(df)}')
    print(f'Columns: {len(df.columns)}')

    print(f'\n=== Date Ranges ===')
    print(f'Start dates:   {df["start_datetime"].min()} → {df["start_datetime"].max()}')
    print(f'End dates:     {df["end_datetime"].min()} → {df["end_datetime"].max()}')
    print(f'Created dates: {df["created_datetime"].min()} → {df["created_datetime"].max()}')

    # Calculate duration
    first_end = df['end_datetime'].min()
    last_end = df['end_datetime'].max()

    # Convert polars datetime to Python datetime for calculation
    first_end_py = datetime.fromisoformat(str(first_end).replace('[UTC]', '').replace('+00:00', ''))
    last_end_py = datetime.fromisoformat(str(last_end).replace('[UTC]', '').replace('+00:00', ''))
    duration = last_end_py - first_end_py
    duration_days = duration.total_seconds() / (24 * 3600)
    duration_hours = duration.total_seconds() / 3600

    print(f'\nTotal trading period: {duration_days:.1f} days ({duration_hours:.0f} hours)')
    print(f'Expected hourly periods (24/day): {duration_days * 24:.0f}')
    print(f'Actual periods found: {len(df)}')
    print(f'Coverage: {len(df) / (duration_days * 24) * 100:.1f}%')

    # Check status
    print(f'\n=== Market Status ===')
    if 'closed' in df.columns:
        closed_count = df['closed'].sum()
        print(f'Closed: {closed_count}')
        print(f'Open: {len(df) - closed_count}')

    # Volume stats
    if 'volumeNum' in df.columns:
        total_volume = df['volumeNum'].sum()
        avg_volume = df['volumeNum'].mean()
        median_volume = df['volumeNum'].median()
        max_volume = df['volumeNum'].max()
        print(f'\n=== Volume Stats ===')
        print(f'Total volume: ${total_volume:,.0f}')
        print(f'Average volume per market: ${avg_volume:,.2f}')
        print(f'Median volume per market: ${median_volume:,.2f}')
        print(f'Max volume (single market): ${max_volume:,.2f}')
        print(f'Markets with volume > 0: {(df["volumeNum"] > 0).sum()}')
        print(f'Markets with volume = 0: {(df["volumeNum"] == 0).sum()}')

    return df


def save_markets(df: pl.DataFrame, output_path: str):
    """Save markets DataFrame to parquet file"""
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    df.write_parquet(output_file)

    file_size_mb = output_file.stat().st_size / (1024 * 1024)
    print(f'\n=== Saved ===')
    print(f'File: {output_file}')
    print(f'Size: {file_size_mb:.2f} MB')
    print(f'Markets: {len(df)}')


def main():
    parser = argparse.ArgumentParser(description='Fetch Bitcoin hourly markets from Polymarket API')
    parser.add_argument(
        '--output',
        type=str,
        default=DEFAULT_OUTPUT,
        help=f'Output parquet file (default: {DEFAULT_OUTPUT})'
    )
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose output'
    )
    parser.add_argument(
        '--no-save',
        action='store_true',
        help='Skip saving to file (analysis only)'
    )

    args = parser.parse_args()

    # Fetch markets
    markets = fetch_bitcoin_hourly_markets(verbose=args.verbose)

    if not markets:
        print("No markets fetched!")
        return

    # Analyze
    df = analyze_markets(markets)

    # Save
    if not args.no_save and df is not None:
        save_markets(df, args.output)

    print('\n✓ Complete')


if __name__ == '__main__':
    main()
