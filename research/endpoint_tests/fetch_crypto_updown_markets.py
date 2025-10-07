#!/usr/bin/env python3
"""
Fetch all crypto 'Up or Down' short-term markets from Polymarket

These are the rapid trading markets that resolve every few hours
based on whether price goes up or down.
"""

import requests
import polars as pl
from datetime import datetime
from pathlib import Path
import time

GAMMA_API = "https://gamma-api.polymarket.com/markets"
RATE_LIMIT_DELAY = 0.11

def fetch_all_markets_with_updown(max_markets=10000):
    """
    Fetch markets and filter for 'Up or Down' crypto markets.

    These markets have questions like:
    - "Bitcoin Up or Down - October 2, 4PM ET"
    - "Ethereum Up or Down - October 2, 5PM ET"
    """

    all_updown_markets = []
    offset = 0
    limit = 100

    print("Searching for crypto 'Up or Down' markets...")
    print("=" * 80)

    while offset < max_markets:
        params = {
            'limit': limit,
            'offset': offset,
            'active': 'true',
            'closed': 'false'
        }

        try:
            response = requests.get(GAMMA_API, params=params, timeout=10)
            response.raise_for_status()
            markets = response.json()

            if not markets:
                break

            # Filter for "Up or Down" markets
            updown_markets = [
                m for m in markets
                if 'up or down' in m.get('question', '').lower()
            ]

            if updown_markets:
                all_updown_markets.extend(updown_markets)
                print(f"Offset {offset:4d}: Found {len(updown_markets)} Up/Down markets (total: {len(all_updown_markets)})")

            if len(markets) < limit:
                break

            offset += limit
            time.sleep(RATE_LIMIT_DELAY)

        except Exception as e:
            print(f"Error at offset {offset}: {e}")
            break

    print(f"\nTotal 'Up or Down' markets found: {len(all_updown_markets)}")
    return all_updown_markets

def extract_updown_data(markets):
    """Extract relevant fields from Up/Down markets."""

    records = []

    for m in markets:
        # Extract crypto asset from question
        question = m.get('question', '').lower()
        if 'bitcoin' in question or 'btc' in question:
            asset = 'Bitcoin'
        elif 'ethereum' in question or 'eth' in question:
            asset = 'Ethereum'
        elif 'solana' in question or 'sol' in question:
            asset = 'Solana'
        else:
            asset = 'Other'

        # Extract token IDs
        token_ids = m.get('clobTokenIds', '').split(',') if m.get('clobTokenIds') else []

        record = {
            # Identifiers
            'crypto_asset': asset,
            'question': m.get('question'),
            'slug': m.get('slug'),
            'market_id': m.get('id'),
            'condition_id': m.get('conditionId'),
            'token_id_up': token_ids[0] if len(token_ids) > 0 else None,
            'token_id_down': token_ids[1] if len(token_ids) > 1 else None,

            # Timing
            'end_date': m.get('endDate'),
            'created_at': m.get('createdAt'),

            # Volume & Liquidity
            'volume': float(m.get('volume', 0)) if m.get('volume') else 0,
            'volume_24hr': float(m.get('volume24hr', 0)) if m.get('volume24hr') else 0,
            'liquidity_clob': float(m.get('liquidityClob', 0)) if m.get('liquidityClob') else 0,

            # Rewards
            'competitive': float(m.get('competitive')) if m.get('competitive') is not None else None,
            'rewards_min_size': float(m.get('rewardsMinSize')) if m.get('rewardsMinSize') is not None else None,
            'rewards_max_spread': float(m.get('rewardsMaxSpread')) if m.get('rewardsMaxSpread') is not None else None,

            # Prices
            'last_trade_price': float(m.get('lastTradePrice')) if m.get('lastTradePrice') is not None else None,
            'best_bid': float(m.get('bestBid')) if m.get('bestBid') is not None else None,
            'best_ask': float(m.get('bestAsk')) if m.get('bestAsk') is not None else None,
            'spread': float(m.get('spread')) if m.get('spread') is not None else None,

            # Status
            'active': m.get('active', False),
            'closed': m.get('closed', False),
            'accepting_orders': m.get('acceptingOrders', False),

            # Metadata
            'fetch_timestamp': datetime.now().isoformat()
        }

        records.append(record)

    df = pl.DataFrame(records)

    # Add calculated columns
    df = df.with_columns([
        # Spread in basis points
        (pl.col('spread') * 10000).alias('spread_bps'),

        # Has both bid and ask
        ((pl.col('best_bid').is_not_null()) & (pl.col('best_ask').is_not_null()))
            .alias('has_market'),

        # Estimated profit (rewards + spread capture)
        # Assumes 50% reward share + capturing 0.2% of volume
        ((pl.col('competitive').fill_null(0) * 0.5) + (pl.col('volume_24hr') * 0.002))
            .alias('estimated_daily_profit'),
    ])

    return df

def print_summary(df):
    """Print summary of Up/Down markets."""

    print("\n" + "=" * 80)
    print(" CRYPTO UP/DOWN MARKETS SUMMARY")
    print("=" * 80)

    print(f"\nTotal markets: {len(df)}")

    # By crypto asset
    print("\nBy Crypto Asset:")
    by_asset = df.group_by('crypto_asset').agg([
        pl.len().alias('count'),
        pl.col('volume_24hr').sum().alias('total_volume'),
        pl.col('competitive').sum().alias('total_rewards')
    ]).sort('count', descending=True)

    for row in by_asset.iter_rows(named=True):
        comp = row['total_rewards'] if row['total_rewards'] else 0
        print(f"  {row['crypto_asset']:12} {row['count']:3} markets | "
              f"${row['total_volume']:>10,.2f} vol | ${comp:>6.2f} rewards")

    # Active vs closed
    active_count = df.filter(pl.col('active') == True).shape[0]
    print(f"\nActive: {active_count} | Closed: {len(df) - active_count}")

    # With liquidity
    liquid = df.filter(
        (pl.col('liquidity_clob') > 100) &
        (pl.col('volume_24hr') > 10)
    )
    print(f"Markets with liquidity: {len(liquid)}")

    # Top by volume
    print("\nTop 15 by 24hr Volume:")
    top_vol = df.sort('volume_24hr', descending=True).head(15)
    for i, row in enumerate(top_vol.iter_rows(named=True), 1):
        comp = row['competitive'] if row['competitive'] else 0
        print(f"  {i:2}. ${row['volume_24hr']:>8,.2f} | ${comp:>5.2f}/day | {row['question']}")

    # Markets with best rewards
    with_rewards = df.filter(pl.col('competitive').is_not_null() & (pl.col('competitive') > 0))
    if len(with_rewards) > 0:
        print(f"\nMarkets with Rewards: {len(with_rewards)}")
        print("\nTop 10 by Reward Pool:")
        top_rewards = with_rewards.sort('competitive', descending=True).head(10)
        for i, row in enumerate(top_rewards.iter_rows(named=True), 1):
            print(f"  {i:2}. ${row['competitive']:>5.2f}/day | {row['question']}")

    print("\n" + "=" * 80)

def save_to_parquet(df, output_dir='data/rewards'):
    """Save to Parquet file."""

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = f'crypto_updown_{timestamp}.parquet'
    filepath = output_path / filename

    print(f"\nSaving to {filepath}...")
    df.write_parquet(filepath, compression='zstd')

    file_size_kb = filepath.stat().st_size / 1024
    print(f"Saved {len(df)} markets ({file_size_kb:.1f} KB)")

    return filepath

def save_to_csv(df, output_dir='data/rewards'):
    """Save to CSV for Excel."""

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = f'crypto_updown_{timestamp}.csv'
    filepath = output_path / filename

    # Select columns for export
    export_df = df.select([
        'crypto_asset',
        'question',
        'slug',
        'volume_24hr',
        'competitive',
        'rewards_min_size',
        'rewards_max_spread',
        'liquidity_clob',
        'spread',
        'spread_bps',
        'estimated_daily_profit',
        'end_date',
        'active',
        'accepting_orders',
        'condition_id',
        'token_id_up',
        'token_id_down'
    ]).sort('volume_24hr', descending=True)

    print(f"Saving CSV to {filepath}...")
    export_df.write_csv(filepath)
    print(f"Saved CSV ({len(export_df)} rows)")

    return filepath

def main():
    """Main execution."""

    print("=" * 80)
    print(" CRYPTO UP/DOWN MARKETS FETCHER")
    print("=" * 80)
    print()

    # Fetch markets
    markets = fetch_all_markets_with_updown(max_markets=5000)

    if not markets:
        print("\nNo 'Up or Down' markets found!")
        print("This could mean:")
        print("  1. These markets are currently closed/inactive")
        print("  2. They're named differently now")
        print("  3. API filtering needs adjustment")
        return

    # Extract data
    print("\nExtracting data...")
    df = extract_updown_data(markets)

    # Print summary
    print_summary(df)

    # Save files
    parquet_file = save_to_parquet(df)
    csv_file = save_to_csv(df)

    print("\n" + "=" * 80)
    print(" COMPLETE!")
    print("=" * 80)

    print("\nOutput files:")
    print(f"  Parquet: {parquet_file}")
    print(f"  CSV:     {csv_file}")

    print("\nTo load in Python:")
    print(f"  import polars as pl")
    print(f"  df = pl.read_parquet('{parquet_file}')")

    print("\nTo open CSV:")
    print(f"  open {csv_file}")

    print("\nTo filter by asset:")
    print("  btc = df.filter(pl.col('crypto_asset') == 'Bitcoin')")
    print("  eth = df.filter(pl.col('crypto_asset') == 'Ethereum')")

if __name__ == "__main__":
    main()
