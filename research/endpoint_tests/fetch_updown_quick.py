#!/usr/bin/env python3
"""
Quick fetch of crypto 'Up or Down' markets for today and next few days
"""

import requests
import polars as pl
from datetime import datetime, timedelta
from pathlib import Path
import time
import re

GAMMA_API = "https://gamma-api.polymarket.com/markets"

def fetch_market_by_slug(slug):
    """Fetch a single market by slug."""
    try:
        response = requests.get(f'{GAMMA_API}?slug={slug}', timeout=5)
        if response.status_code == 200:
            markets = response.json()
            if markets and len(markets) > 0:
                return markets[0]
    except:
        pass
    return None

def fetch_updown_markets_quick(days_ahead=3):
    """
    Fetch Up/Down markets for the next few days only.

    Args:
        days_ahead: Number of days to check (default 3)
    """

    print(f"Searching for Up/Down markets (next {days_ahead} days)...")
    print("=" * 80)

    found_markets = []

    # Assets to check
    assets = ['bitcoin', 'ethereum']  # Focus on main two

    # Generate dates
    dates = []
    for i in range(days_ahead):
        d = datetime.now() + timedelta(days=i)
        # Try full month name
        dates.append((d.strftime('%B').lower() + '-' + str(d.day), d))

    # Hours to check (common trading hours)
    hours = ['12am', '1am', '2am', '3am', '4am', '5am', '6am', '7am', '8am', '9am', '10am', '11am',
             '12pm', '1pm', '2pm', '3pm', '4pm', '5pm', '6pm', '7pm', '8pm', '9pm', '10pm', '11pm']

    total_checks = len(assets) * len(dates) * len(hours)
    checked = 0

    print(f"Will check {total_checks} slug combinations\n")

    for asset in assets:
        for date_str, date_obj in dates:
            for hour in hours:
                slug = f'{asset}-up-or-down-{date_str}-{hour}-et'
                checked += 1

                if checked % 20 == 0:
                    print(f"Progress: {checked}/{total_checks} ({len(found_markets)} found)")

                market = fetch_market_by_slug(slug)

                if market:
                    found_markets.append(market)
                    print(f"  ✓ {market['question'][:60]}")

                time.sleep(0.02)  # Small delay

    print(f"\nFound {len(found_markets)} Up/Down markets")
    return found_markets

def extract_data(markets):
    """Extract data from markets."""

    records = []

    for m in markets:
        question = m.get('question', '')
        question_lower = question.lower()

        # Determine asset
        if 'bitcoin' in question_lower or 'btc' in question_lower:
            asset = 'Bitcoin'
        elif 'ethereum' in question_lower or 'eth' in question_lower:
            asset = 'Ethereum'
        else:
            asset = 'Other'

        # Extract hour
        time_match = re.search(r'(\d+)(AM|PM)', question, re.IGNORECASE)
        hour_str = time_match.group(0) if time_match else 'Unknown'

        # Extract token IDs
        token_ids = m.get('clobTokenIds', '').split(',') if m.get('clobTokenIds') else []

        record = {
            'crypto_asset': asset,
            'hour_et': hour_str,
            'question': question,
            'slug': m.get('slug'),
            'market_id': m.get('id'),
            'condition_id': m.get('conditionId'),
            'token_id_up': token_ids[0] if len(token_ids) > 0 else None,
            'token_id_down': token_ids[1] if len(token_ids) > 1 else None,

            'end_date': m.get('endDate'),

            'volume_24hr': float(m.get('volume24hr', 0)) if m.get('volume24hr') else 0,
            'liquidity_clob': float(m.get('liquidityClob', 0)) if m.get('liquidityClob') else 0,

            'competitive': float(m.get('competitive')) if m.get('competitive') is not None else None,
            'rewards_min_size': float(m.get('rewardsMinSize')) if m.get('rewardsMinSize') is not None else None,
            'rewards_max_spread': float(m.get('rewardsMaxSpread')) if m.get('rewardsMaxSpread') is not None else None,

            'last_trade_price': float(m.get('lastTradePrice')) if m.get('lastTradePrice') is not None else None,
            'best_bid': float(m.get('bestBid')) if m.get('bestBid') is not None else None,
            'best_ask': float(m.get('bestAsk')) if m.get('bestAsk') is not None else None,
            'spread': float(m.get('spread')) if m.get('spread') is not None else None,

            'active': m.get('active', False),
            'closed': m.get('closed', False),
            'accepting_orders': m.get('acceptingOrders', False),
        }

        records.append(record)

    return pl.DataFrame(records)

def print_summary(df):
    """Print summary."""

    print("\n" + "=" * 80)
    print(" UP/DOWN MARKETS SUMMARY")
    print("=" * 80)

    print(f"\nTotal markets: {len(df)}")

    # By asset
    by_asset = df.group_by('crypto_asset').agg([
        pl.len().alias('count'),
        pl.col('volume_24hr').sum().alias('total_volume'),
    ]).sort('count', descending=True)

    print("\nBy Asset:")
    for row in by_asset.iter_rows(named=True):
        print(f"  {row['crypto_asset']:10} {row['count']:3} markets | ${row['total_volume']:>10,.2f} volume")

    # Active
    active = df.filter(pl.col('active') == True)
    with_volume = df.filter(pl.col('volume_24hr') > 10)
    print(f"\nActive: {len(active)} | With volume > $10: {len(with_volume)}")

    # Top by volume
    print("\nTop 10 by Volume:")
    top = df.sort('volume_24hr', descending=True).head(10)
    for i, row in enumerate(top.iter_rows(named=True), 1):
        comp = row['competitive'] if row['competitive'] else 0
        print(f"  {i:2}. ${row['volume_24hr']:>8,.2f} | ${comp:>5.2f}/day | {row['question']}")

def save_files(df):
    """Save to files."""

    output_path = Path('data/rewards')
    output_path.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    # Parquet
    parquet_file = output_path / f'updown_markets_{timestamp}.parquet'
    df.write_parquet(parquet_file, compression='zstd')

    # CSV
    csv_file = output_path / f'updown_markets_{timestamp}.csv'
    df.select([
        'crypto_asset', 'hour_et', 'question', 'slug',
        'volume_24hr', 'competitive', 'rewards_min_size', 'rewards_max_spread',
        'spread', 'end_date', 'active',
        'condition_id', 'token_id_up', 'token_id_down'
    ]).sort(['crypto_asset', 'end_date']).write_csv(csv_file)

    print(f"\n✓ Saved: {parquet_file}")
    print(f"✓ Saved: {csv_file}")

    return parquet_file, csv_file

def main():
    print("=" * 80)
    print(" CRYPTO UP/DOWN MARKETS - QUICK FETCH")
    print("=" * 80)
    print()

    markets = fetch_updown_markets_quick(days_ahead=3)

    if not markets:
        print("No markets found!")
        return

    df = extract_data(markets)
    print_summary(df)
    parquet_file, csv_file = save_files(df)

    print("\n" + "=" * 80)
    print(f"\nTo load: df = pl.read_parquet('{parquet_file}')")
    print(f"To open: open {csv_file}")

if __name__ == "__main__":
    main()
