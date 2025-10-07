#!/usr/bin/env python3
"""
Fetch crypto 'Up or Down' markets by generating slug patterns

These markets exist but don't appear in bulk market listings,
so we need to search for them individually by slug.
"""

import requests
import polars as pl
from datetime import datetime, timedelta
from pathlib import Path
import time

GAMMA_API = "https://gamma-api.polymarket.com/markets"

def generate_slug_patterns():
    """
    Generate all possible slug patterns for Up/Down markets.

    Pattern: {asset}-up-or-down-{date}-{time}-et
    """

    slugs = []

    # Assets to check
    assets = ['bitcoin', 'ethereum', 'btc', 'eth', 'solana', 'sol']

    # Generate dates for next 7 days
    today = datetime.now()
    dates = []
    for i in range(7):
        d = today + timedelta(days=i)
        # Try both full month name and abbreviation
        dates.append(d.strftime('%B').lower() + '-' + str(d.day))  # october-2
        dates.append(d.strftime('%b').lower() + '-' + str(d.day))  # oct-2

    # Hours (ET timezone)
    hours = []
    for h in range(1, 13):  # 1-12
        hours.append(f'{h}am')
        hours.append(f'{h}pm')

    # Generate all combinations
    for asset in assets:
        for date in dates:
            for hour in hours:
                slug = f'{asset}-up-or-down-{date}-{hour}-et'
                slugs.append((asset, slug))

    return slugs

def fetch_market_by_slug(slug):
    """Fetch a single market by slug."""
    try:
        response = requests.get(f'{GAMMA_API}?slug={slug}', timeout=5)
        response.raise_for_status()
        markets = response.json()

        if markets and len(markets) > 0:
            return markets[0]
        return None
    except:
        return None

def fetch_updown_markets():
    """Fetch all Up/Down markets by trying different slug patterns."""

    print("Generating slug patterns...")
    slug_patterns = generate_slug_patterns()
    print(f"Generated {len(slug_patterns)} potential slugs to check\n")

    print("Searching for markets (this may take a minute)...")
    print("=" * 80)

    found_markets = []
    checked = 0

    for asset, slug in slug_patterns:
        checked += 1

        if checked % 50 == 0:
            print(f"Checked {checked}/{len(slug_patterns)} slugs... Found {len(found_markets)} markets")

        market = fetch_market_by_slug(slug)

        if market:
            found_markets.append(market)
            print(f"✓ {market['question']}")

        # Small delay to respect rate limits
        time.sleep(0.05)

    print(f"\nTotal markets found: {len(found_markets)}")
    return found_markets

def extract_data(markets):
    """Extract data from markets."""

    records = []

    for m in markets:
        # Determine crypto asset
        question = m.get('question', '').lower()
        if 'bitcoin' in question or 'btc' in question:
            asset = 'Bitcoin'
        elif 'ethereum' in question or 'eth' in question:
            asset = 'Ethereum'
        elif 'solana' in question or 'sol' in question:
            asset = 'Solana'
        else:
            asset = 'Other'

        # Extract hour from question (e.g., "4PM")
        import re
        time_match = re.search(r'(\d+)(AM|PM)', question, re.IGNORECASE)
        hour_str = time_match.group(0) if time_match else 'Unknown'

        # Parse end date
        end_date = m.get('endDate', '')

        # Extract token IDs
        token_ids = m.get('clobTokenIds', '').split(',') if m.get('clobTokenIds') else []

        record = {
            'crypto_asset': asset,
            'hour_et': hour_str,
            'question': m.get('question'),
            'slug': m.get('slug'),
            'market_id': m.get('id'),
            'condition_id': m.get('conditionId'),
            'token_id_up': token_ids[0] if len(token_ids) > 0 else None,
            'token_id_down': token_ids[1] if len(token_ids) > 1 else None,

            'end_date': end_date,
            'created_at': m.get('createdAt'),

            'volume': float(m.get('volume', 0)) if m.get('volume') else 0,
            'volume_24hr': float(m.get('volume24hr', 0)) if m.get('volume24hr') else 0,
            'volume_1wk': float(m.get('volume1wk', 0)) if m.get('volume1wk') else 0,

            'liquidity_num': float(m.get('liquidityNum', 0)) if m.get('liquidityNum') else 0,
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

            'fetch_timestamp': datetime.now().isoformat()
        }

        records.append(record)

    df = pl.DataFrame(records)

    # Add calculated fields
    df = df.with_columns([
        (pl.col('spread') * 10000).alias('spread_bps'),
        ((pl.col('best_bid').is_not_null()) & (pl.col('best_ask').is_not_null())).alias('has_market'),
    ])

    return df

def print_summary(df):
    """Print summary of found markets."""

    print("\n" + "=" * 80)
    print(" UP/DOWN MARKETS SUMMARY")
    print("=" * 80)

    print(f"\nTotal markets: {len(df)}")

    # By asset
    print("\nBy Crypto Asset:")
    by_asset = df.group_by('crypto_asset').agg([
        pl.len().alias('count'),
        pl.col('volume_24hr').sum().alias('total_volume'),
        pl.col('competitive').sum().alias('total_rewards')
    ]).sort('count', descending=True)

    for row in by_asset.iter_rows(named=True):
        comp = row['total_rewards'] if row['total_rewards'] else 0
        print(f"  {row['crypto_asset']:12} {row['count']:3} markets | "
              f"${row['total_volume']:>8,.2f} vol | ${comp:>6.2f} rewards")

    # By hour
    print("\nMarkets by Hour (ET):")
    by_hour = df.group_by('hour_et').agg([
        pl.len().alias('count'),
        pl.col('volume_24hr').mean().alias('avg_volume')
    ]).sort('hour_et')

    for row in by_hour.iter_rows(named=True):
        print(f"  {row['hour_et']:6} {row['count']:3} markets | Avg vol: ${row['avg_volume']:>8,.2f}")

    # Active markets
    active = df.filter(pl.col('active') == True)
    with_volume = df.filter(pl.col('volume_24hr') > 10)
    print(f"\nActive markets: {len(active)}")
    print(f"Markets with volume > $10: {len(with_volume)}")

    # Top by volume
    print("\nTop 10 by 24hr Volume:")
    top = df.sort('volume_24hr', descending=True).head(10)
    for i, row in enumerate(top.iter_rows(named=True), 1):
        comp = row['competitive'] if row['competitive'] else 0
        print(f"  {i:2}. ${row['volume_24hr']:>8,.2f} | ${comp:>5.2f}/day | {row['question']}")

    print("\n" + "=" * 80)

def save_files(df, output_dir='data/rewards'):
    """Save to both Parquet and CSV."""

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    # Parquet
    parquet_file = output_path / f'updown_markets_{timestamp}.parquet'
    df.write_parquet(parquet_file, compression='zstd')
    print(f"\n✓ Saved Parquet: {parquet_file} ({parquet_file.stat().st_size / 1024:.1f} KB)")

    # CSV
    csv_file = output_path / f'updown_markets_{timestamp}.csv'
    export_df = df.select([
        'crypto_asset', 'hour_et', 'question', 'slug',
        'volume_24hr', 'competitive', 'rewards_min_size', 'rewards_max_spread',
        'liquidity_clob', 'spread', 'spread_bps',
        'end_date', 'active', 'accepting_orders',
        'condition_id', 'token_id_up', 'token_id_down'
    ]).sort(['crypto_asset', 'hour_et'])

    export_df.write_csv(csv_file)
    print(f"✓ Saved CSV: {csv_file} ({len(export_df)} rows)")

    return parquet_file, csv_file

def main():
    """Main execution."""

    print("=" * 80)
    print(" CRYPTO UP/DOWN MARKETS FETCHER")
    print("=" * 80)
    print()

    # Fetch markets
    markets = fetch_updown_markets()

    if not markets:
        print("\nNo Up/Down markets found!")
        return

    # Extract data
    print("\nExtracting data...")
    df = extract_data(markets)

    # Print summary
    print_summary(df)

    # Save files
    parquet_file, csv_file = save_files(df)

    print("\n" + "=" * 80)
    print(" COMPLETE!")
    print("=" * 80)

    print("\nTo load in Python:")
    print(f"  import polars as pl")
    print(f"  df = pl.read_parquet('{parquet_file}')")

    print("\nTo open CSV:")
    print(f"  open {csv_file}")

    print("\nFilter examples:")
    print("  # Bitcoin only")
    print("  btc = df.filter(pl.col('crypto_asset') == 'Bitcoin')")
    print("  # Active with volume")
    print("  active = df.filter((pl.col('active') == True) & (pl.col('volume_24hr') > 10))")

if __name__ == "__main__":
    main()
