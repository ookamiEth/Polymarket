#!/usr/bin/env python3
"""
Fetch Polymarket liquidity rewards data and save to Parquet

This script fetches comprehensive reward configuration data for all active markets,
including min_incentive_size, max_incentive_spread, and reward pool allocations.

Data Source: Gamma Markets API (https://gamma-api.polymarket.com/markets)
Rate Limit: 100 requests / 10 seconds
"""

import requests
import json
import polars as pl
from datetime import datetime
from pathlib import Path
import time
import sys

# API Configuration
GAMMA_API_BASE = "https://gamma-api.polymarket.com"
MARKETS_ENDPOINT = f"{GAMMA_API_BASE}/markets"
RATE_LIMIT_DELAY = 0.11  # 100ms delay between requests (safe for 100 req/10s limit)

def fetch_all_active_markets(limit_per_request=100, max_markets=None):
    """
    Fetch all active markets with pagination.

    Args:
        limit_per_request: Number of markets per API call (max 100)
        max_markets: Optional limit on total markets to fetch (for testing)

    Returns:
        List of market dictionaries
    """
    all_markets = []
    offset = 0

    print(f"Fetching active markets from {MARKETS_ENDPOINT}")
    print(f"Rate limit: 100 req / 10s (using {RATE_LIMIT_DELAY}s delay)")
    print("-" * 80)

    while True:
        params = {
            "limit": limit_per_request,
            "offset": offset,
            "active": "true",  # Only active markets
            "closed": "false"  # Exclude closed markets
        }

        try:
            print(f"Fetching markets {offset} to {offset + limit_per_request}...", end=" ")
            response = requests.get(MARKETS_ENDPOINT, params=params, timeout=10)
            response.raise_for_status()

            markets = response.json()

            if not markets:
                print("No more markets")
                break

            print(f"Got {len(markets)} markets")
            all_markets.extend(markets)

            # Check if we've reached the limit
            if max_markets and len(all_markets) >= max_markets:
                all_markets = all_markets[:max_markets]
                print(f"Reached max_markets limit ({max_markets})")
                break

            # Check if we got fewer markets than requested (last page)
            if len(markets) < limit_per_request:
                print("Reached last page")
                break

            offset += limit_per_request

            # Rate limiting
            time.sleep(RATE_LIMIT_DELAY)

        except requests.exceptions.RequestException as e:
            print(f"Error fetching markets at offset {offset}: {e}")
            break

    print(f"\nTotal markets fetched: {len(all_markets)}")
    return all_markets

def extract_rewards_data(markets):
    """
    Extract reward-relevant fields from market data.

    Args:
        markets: List of market dictionaries from API

    Returns:
        Polars DataFrame with rewards data
    """
    print("\nExtracting rewards data...")

    rewards_data = []
    markets_with_rewards = 0

    for market in markets:
        # Check if market has rewards configured
        rewards_min_size = market.get('rewardsMinSize')
        rewards_max_spread = market.get('rewardsMaxSpread')

        # Skip markets without rewards configuration
        if rewards_min_size is None and rewards_max_spread is None:
            continue

        markets_with_rewards += 1

        # Extract token IDs from clobTokenIds (comma-separated string)
        clob_token_ids = market.get('clobTokenIds', '')
        token_ids = clob_token_ids.split(',') if clob_token_ids else []

        # Build rewards record
        record = {
            # Identifiers
            'market_id': market.get('id'),
            'condition_id': market.get('conditionId'),
            'question': market.get('question'),
            'slug': market.get('slug'),

            # Token IDs (take first two for YES/NO)
            'token_id_0': token_ids[0] if len(token_ids) > 0 else None,
            'token_id_1': token_ids[1] if len(token_ids) > 1 else None,

            # Reward Configuration
            'rewards_min_size': float(rewards_min_size) if rewards_min_size is not None else None,
            'rewards_max_spread': float(rewards_max_spread) if rewards_max_spread is not None else None,
            'competitive': float(market.get('competitive')) if market.get('competitive') is not None else None,

            # Volume Metrics
            'volume': float(market.get('volume', 0)) if market.get('volume') else 0,
            'volume_24hr': float(market.get('volume24hr', 0)) if market.get('volume24hr') else 0,
            'volume_1wk': float(market.get('volume1wk', 0)) if market.get('volume1wk') else 0,

            # Liquidity
            'liquidity_num': float(market.get('liquidityNum', 0)) if market.get('liquidityNum') else 0,
            'liquidity_clob': float(market.get('liquidityClob', 0)) if market.get('liquidityClob') else 0,

            # Current Prices
            'last_trade_price': float(market.get('lastTradePrice')) if market.get('lastTradePrice') is not None else None,
            'best_bid': float(market.get('bestBid')) if market.get('bestBid') is not None else None,
            'best_ask': float(market.get('bestAsk')) if market.get('bestAsk') is not None else None,
            'spread': float(market.get('spread')) if market.get('spread') is not None else None,

            # Status
            'active': market.get('active', False),
            'accepting_orders': market.get('acceptingOrders', False),

            # Timestamps
            'end_date': market.get('endDate'),
            'created_at': market.get('createdAt'),

            # Metadata
            'fetch_timestamp': datetime.now().isoformat()
        }

        rewards_data.append(record)

    print(f"Markets with rewards: {markets_with_rewards} out of {len(markets)}")

    # Create Polars DataFrame
    if not rewards_data:
        print("Warning: No markets with rewards found!")
        return pl.DataFrame()

    df = pl.DataFrame(rewards_data)

    # Calculate derived metrics
    df = df.with_columns([
        # Reward per dollar of volume
        (pl.col('competitive') / pl.col('volume_24hr').replace(0, None))
            .alias('reward_per_volume_24hr'),

        # Spread in basis points
        (pl.col('spread') * 10000).alias('spread_bps'),

        # Has both bid and ask
        ((pl.col('best_bid').is_not_null()) & (pl.col('best_ask').is_not_null()))
            .alias('has_market'),
    ])

    return df

def save_to_parquet(df, output_dir='data/rewards'):
    """
    Save DataFrame to Parquet file with timestamp.

    Args:
        df: Polars DataFrame
        output_dir: Directory to save files (will be created if doesn't exist)

    Returns:
        Path to saved file
    """
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Generate filename with timestamp
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = f'rewards_snapshot_{timestamp}.parquet'
    filepath = output_path / filename

    # Save to Parquet
    print(f"\nSaving to {filepath}...")
    df.write_parquet(filepath, compression='zstd')

    # Print file size
    file_size_mb = filepath.stat().st_size / (1024 * 1024)
    print(f"Saved {len(df)} records ({file_size_mb:.2f} MB)")

    return filepath

def print_summary(df):
    """Print summary statistics of the rewards data."""

    print("\n" + "=" * 80)
    print(" REWARDS DATA SUMMARY")
    print("=" * 80)

    print(f"\nTotal markets with rewards: {len(df)}")

    # Reward configuration stats
    print("\nReward Configuration:")
    print(f"  Min Size range: ${df['rewards_min_size'].min():.2f} - ${df['rewards_min_size'].max():.2f}")
    print(f"  Max Spread range: {df['rewards_max_spread'].min():.4f}¢ - {df['rewards_max_spread'].max():.4f}¢")
    print(f"  Competitive (pool) range: ${df['competitive'].min():.2f} - ${df['competitive'].max():.2f}")

    # Volume stats
    print("\n24hr Volume:")
    print(f"  Total: ${df['volume_24hr'].sum():,.2f}")
    print(f"  Average: ${df['volume_24hr'].mean():,.2f}")
    print(f"  Median: ${df['volume_24hr'].median():,.2f}")

    # Top markets by rewards
    print("\nTop 10 Markets by Reward Pool (competitive):")
    top_rewards = df.filter(pl.col('competitive').is_not_null()).sort('competitive', descending=True).head(10)
    for i, row in enumerate(top_rewards.iter_rows(named=True), 1):
        question = row['question'][:60] + "..." if len(row['question']) > 60 else row['question']
        comp_value = row['competitive'] if row['competitive'] is not None else 0
        print(f"  {i}. ${comp_value:,.2f} - {question}")

    # Top markets by volume
    print("\nTop 10 Markets by 24hr Volume:")
    top_volume = df.sort('volume_24hr', descending=True).head(10)
    for i, row in enumerate(top_volume.iter_rows(named=True), 1):
        question = row['question'][:60] + "..." if len(row['question']) > 60 else row['question']
        print(f"  {i}. ${row['volume_24hr']:,.2f} - {question}")

    # Markets accepting orders
    accepting_count = df.filter(pl.col('accepting_orders') == True).shape[0]
    print(f"\nMarkets accepting orders: {accepting_count} ({accepting_count/len(df)*100:.1f}%)")

    print("\n" + "=" * 80)

def main():
    """Main execution function."""

    print("=" * 80)
    print(" POLYMARKET LIQUIDITY REWARDS DATA FETCHER")
    print("=" * 80)
    print()

    # Parse command line arguments for testing
    max_markets = None
    if len(sys.argv) > 1:
        try:
            max_markets = int(sys.argv[1])
            print(f"Testing mode: Limiting to {max_markets} markets\n")
        except ValueError:
            print(f"Invalid argument: {sys.argv[1]}")
            print("Usage: python test_rewards_data.py [max_markets]")
            return

    # Fetch markets
    markets = fetch_all_active_markets(max_markets=max_markets)

    if not markets:
        print("No markets fetched. Exiting.")
        return

    # Extract rewards data
    df = extract_rewards_data(markets)

    if df.is_empty():
        print("No rewards data extracted. Exiting.")
        return

    # Save to Parquet
    filepath = save_to_parquet(df)

    # Print summary
    print_summary(df)

    print(f"\n✓ Data saved to: {filepath}")
    print("\nTo load this data in Python:")
    print(f"  import polars as pl")
    print(f"  df = pl.read_parquet('{filepath}')")
    print("\nOr in Pandas:")
    print(f"  import pandas as pd")
    print(f"  df = pd.read_parquet('{filepath}')")

if __name__ == "__main__":
    main()
