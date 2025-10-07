#!/usr/bin/env python3
"""
Analyze Polymarket liquidity rewards data

This script loads reward snapshot data from Parquet and performs analysis
to identify the best opportunities for market making based on:
- Reward pool size
- Volume metrics
- Spread requirements
- Liquidity efficiency
"""

import polars as pl
from pathlib import Path
from datetime import datetime
import sys

def load_latest_snapshot(data_dir='data/rewards'):
    """
    Load the most recent rewards snapshot.

    Args:
        data_dir: Directory containing Parquet files

    Returns:
        Polars DataFrame
    """
    data_path = Path(data_dir)

    if not data_path.exists():
        print(f"Error: Data directory {data_dir} does not exist")
        return None

    # Find all snapshot files
    snapshots = list(data_path.glob('rewards_snapshot_*.parquet'))

    if not snapshots:
        print(f"Error: No snapshot files found in {data_dir}")
        print("\nRun test_rewards_data.py first to fetch data:")
        print("  uv run python research/endpoint_tests/test_rewards_data.py")
        return None

    # Get the most recent file
    latest = max(snapshots, key=lambda p: p.stat().st_mtime)

    print(f"Loading: {latest.name}")
    df = pl.read_parquet(latest)

    print(f"Loaded {len(df)} markets")
    return df

def calculate_reward_metrics(df):
    """
    Calculate additional metrics for reward analysis.

    Args:
        df: Polars DataFrame with rewards data

    Returns:
        DataFrame with additional metrics
    """
    return df.with_columns([
        # Reward per $1000 of volume
        (pl.col('competitive') / (pl.col('volume_24hr') / 1000).replace(0, None))
            .alias('reward_per_k_volume'),

        # Reward per unit of liquidity
        (pl.col('competitive') / pl.col('liquidity_clob').replace(0, None))
            .alias('reward_per_liquidity'),

        # Volume to liquidity ratio (turnover)
        (pl.col('volume_24hr') / pl.col('liquidity_clob').replace(0, None))
            .alias('volume_liquidity_ratio'),

        # Effective reward rate (reward / volume as percentage)
        ((pl.col('competitive') / pl.col('volume_24hr').replace(0, None)) * 100)
            .alias('reward_rate_pct'),

        # Is the market liquid? (has volume and liquidity)
        ((pl.col('volume_24hr') > 1000) & (pl.col('liquidity_clob') > 100))
            .alias('is_liquid'),

        # Quality score: combines reward, volume, and spread
        (
            (pl.col('competitive') * 0.4) +
            (pl.col('volume_24hr') / 10000 * 0.3) +
            (pl.col('liquidity_clob') / 1000 * 0.2) +
            ((5 - pl.col('rewards_max_spread').fill_null(0)) * 0.1)
        ).alias('opportunity_score')
    ])

def analyze_top_opportunities(df, n=20):
    """
    Find and display top market making opportunities.

    Args:
        df: DataFrame with metrics
        n: Number of top markets to show
    """
    print("\n" + "=" * 80)
    print(f" TOP {n} MARKET MAKING OPPORTUNITIES")
    print("=" * 80)

    # Filter to liquid markets with rewards
    liquid_markets = df.filter(
        (pl.col('is_liquid') == True) &
        (pl.col('competitive') > 0) &
        (pl.col('rewards_max_spread').is_not_null())
    )

    if liquid_markets.is_empty():
        print("No liquid markets with rewards found")
        return

    # Sort by opportunity score
    top_markets = liquid_markets.sort('opportunity_score', descending=True).head(n)

    print(f"\nFound {len(liquid_markets)} liquid markets with rewards")
    print(f"Showing top {len(top_markets)} by opportunity score:\n")

    for i, row in enumerate(top_markets.iter_rows(named=True), 1):
        print(f"{i}. {row['question'][:70]}")
        print(f"   Slug: {row['slug']}")
        print(f"   Reward Pool: ${row['competitive']:.2f}/day")
        print(f"   Volume 24hr: ${row['volume_24hr']:,.2f}")
        print(f"   Liquidity: ${row['liquidity_clob']:,.2f}")
        print(f"   Max Spread: {row['rewards_max_spread']:.2f}¢ | Min Size: ${row['rewards_min_size']:.0f}")
        print(f"   Current Spread: {row['spread']:.4f} ({row['spread_bps']:.0f} bps)")
        print(f"   Reward Rate: {row['reward_rate_pct']:.4f}%")
        print(f"   Opportunity Score: {row['opportunity_score']:.2f}")
        print()

def analyze_reward_efficiency(df):
    """
    Analyze reward efficiency across different market characteristics.

    Args:
        df: DataFrame with metrics
    """
    print("\n" + "=" * 80)
    print(" REWARD EFFICIENCY ANALYSIS")
    print("=" * 80)

    # Markets with rewards
    with_rewards = df.filter(pl.col('competitive') > 0)

    print(f"\nMarkets with rewards: {len(with_rewards)}")
    print(f"Total daily reward pool: ${with_rewards['competitive'].sum():.2f}")

    # Breakdown by reward size
    print("\nReward Pool Distribution:")
    print(f"  < $0.10/day:  {len(with_rewards.filter(pl.col('competitive') < 0.10))} markets")
    print(f"  $0.10-$0.50:  {len(with_rewards.filter((pl.col('competitive') >= 0.10) & (pl.col('competitive') < 0.50)))} markets")
    print(f"  $0.50-$1.00:  {len(with_rewards.filter((pl.col('competitive') >= 0.50) & (pl.col('competitive') < 1.00)))} markets")
    print(f"  > $1.00/day:  {len(with_rewards.filter(pl.col('competitive') >= 1.00))} markets")

    # Volume distribution
    print("\nVolume Distribution (24hr):")
    print(f"  < $1k:        {len(with_rewards.filter(pl.col('volume_24hr') < 1000))} markets")
    print(f"  $1k-$10k:     {len(with_rewards.filter((pl.col('volume_24hr') >= 1000) & (pl.col('volume_24hr') < 10000)))} markets")
    print(f"  $10k-$100k:   {len(with_rewards.filter((pl.col('volume_24hr') >= 10000) & (pl.col('volume_24hr') < 100000)))} markets")
    print(f"  > $100k:      {len(with_rewards.filter(pl.col('volume_24hr') >= 100000))} markets")

    # Spread requirements
    print("\nSpread Requirements:")
    valid_spread = with_rewards.filter(pl.col('rewards_max_spread').is_not_null())
    print(f"  Mean max spread: {valid_spread['rewards_max_spread'].mean():.2f}¢")
    print(f"  Median max spread: {valid_spread['rewards_max_spread'].median():.2f}¢")
    print(f"  Tightest: {valid_spread['rewards_max_spread'].min():.2f}¢")
    print(f"  Widest: {valid_spread['rewards_max_spread'].max():.2f}¢")

    # Size requirements
    print("\nMinimum Size Requirements:")
    valid_size = with_rewards.filter(pl.col('rewards_min_size').is_not_null())
    print(f"  Mean min size: ${valid_size['rewards_min_size'].mean():.2f}")
    print(f"  Median min size: ${valid_size['rewards_min_size'].median():.2f}")
    print(f"  Smallest: ${valid_size['rewards_min_size'].min():.2f}")
    print(f"  Largest: ${valid_size['rewards_min_size'].max():.2f}")

def analyze_high_volume_markets(df, min_volume=50000):
    """
    Analyze high-volume markets specifically.

    Args:
        df: DataFrame with metrics
        min_volume: Minimum 24hr volume threshold
    """
    print("\n" + "=" * 80)
    print(f" HIGH VOLUME MARKETS (>${min_volume/1000:.0f}k/day)")
    print("=" * 80)

    high_vol = df.filter(
        (pl.col('volume_24hr') >= min_volume) &
        (pl.col('competitive').is_not_null())
    ).sort('volume_24hr', descending=True)

    print(f"\nFound {len(high_vol)} markets with volume > ${min_volume:,.0f}/day\n")

    for i, row in enumerate(high_vol.head(15).iter_rows(named=True), 1):
        reward_str = f"${row['competitive']:.2f}" if row['competitive'] and row['competitive'] > 0 else "No rewards"
        print(f"{i}. ${row['volume_24hr']:>10,.0f} | {reward_str:>12} | {row['question'][:50]}")

def export_opportunities_csv(df, output_file='data/rewards/top_opportunities.csv'):
    """
    Export top opportunities to CSV for further analysis.

    Args:
        df: DataFrame with metrics
        output_file: Path to output CSV file
    """
    # Filter and select columns for export
    opportunities = df.filter(
        (pl.col('is_liquid') == True) &
        (pl.col('competitive') > 0)
    ).select([
        'question', 'slug', 'condition_id',
        'token_id_0', 'token_id_1',
        'competitive', 'rewards_min_size', 'rewards_max_spread',
        'volume_24hr', 'volume_1wk', 'liquidity_clob',
        'spread', 'spread_bps',
        'reward_rate_pct', 'opportunity_score',
        'best_bid', 'best_ask', 'last_trade_price'
    ]).sort('opportunity_score', descending=True)

    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    opportunities.write_csv(output_file)
    print(f"\n✓ Exported {len(opportunities)} opportunities to: {output_file}")

    return opportunities

def main():
    """Main analysis function."""

    print("=" * 80)
    print(" POLYMARKET LIQUIDITY REWARDS ANALYZER")
    print("=" * 80)
    print()

    # Load data
    df = load_latest_snapshot()
    if df is None:
        return

    # Calculate metrics
    print("\nCalculating reward metrics...")
    df = calculate_reward_metrics(df)

    # Run analyses
    analyze_top_opportunities(df, n=20)
    analyze_reward_efficiency(df)
    analyze_high_volume_markets(df, min_volume=50000)

    # Export opportunities
    export_opportunities_csv(df)

    print("\n" + "=" * 80)
    print(" ANALYSIS COMPLETE")
    print("=" * 80)

    # Print usage tips
    print("\nNext steps:")
    print("  1. Review top opportunities above")
    print("  2. Check data/rewards/top_opportunities.csv for full list")
    print("  3. Use market slugs to find markets on polymarket.com")
    print("  4. Analyze order books for selected markets")
    print("\nTo fetch fresh data:")
    print("  uv run python research/endpoint_tests/test_rewards_data.py")

if __name__ == "__main__":
    main()
