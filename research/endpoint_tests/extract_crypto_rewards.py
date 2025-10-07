#!/usr/bin/env python3
"""
Extract crypto-related markets from Polymarket rewards data

Filters the full rewards dataset for crypto markets (Bitcoin, Ethereum, etc.)
and saves to a separate Parquet file for focused analysis.
"""

import polars as pl
from pathlib import Path
from datetime import datetime

def load_latest_snapshot(data_dir='data/rewards'):
    """Load the most recent rewards snapshot."""
    data_path = Path(data_dir)

    if not data_path.exists():
        print(f"Error: Data directory {data_dir} does not exist")
        print("\nRun this first:")
        print("  uv run python research/endpoint_tests/test_rewards_data.py")
        return None

    # Find all snapshot files
    snapshots = list(data_path.glob('rewards_snapshot_*.parquet'))

    if not snapshots:
        print(f"Error: No snapshot files found in {data_dir}")
        print("\nRun this first:")
        print("  uv run python research/endpoint_tests/test_rewards_data.py")
        return None

    # Get the most recent file
    latest = max(snapshots, key=lambda p: p.stat().st_mtime)

    print(f"Loading: {latest.name}")
    df = pl.read_parquet(latest)
    print(f"Total markets loaded: {len(df)}")

    return df, latest.name

def filter_crypto_markets(df):
    """
    Filter for crypto-related markets.

    Searches for keywords in market questions:
    - Bitcoin, BTC
    - Ethereum, ETH
    - Solana, SOL
    - Crypto, cryptocurrency
    - Dogecoin, DOGE
    - XRP, Ripple
    - Cardano, ADA
    - And other major cryptocurrencies
    """

    # Comprehensive crypto keywords
    crypto_keywords = [
        # Bitcoin
        'bitcoin', 'btc',
        # Ethereum
        'ethereum', 'eth',
        # Major altcoins
        'solana', 'sol',
        'cardano', 'ada',
        'ripple', 'xrp',
        'dogecoin', 'doge',
        'polkadot', 'dot',
        'avalanche', 'avax',
        'polygon', 'matic',
        'chainlink', 'link',
        'litecoin', 'ltc',
        # General crypto terms
        'crypto', 'cryptocurrency',
        'defi', 'nft',
        'blockchain',
        'altcoin',
        'stablecoin',
        'usdc', 'usdt', 'dai'
    ]

    # Create regex pattern (case insensitive)
    pattern = '|'.join(crypto_keywords)

    print(f"\nSearching for crypto keywords: {', '.join(crypto_keywords[:10])}...")

    crypto_markets = df.filter(
        pl.col('question').str.contains(f'(?i){pattern}')
    )

    print(f"Found {len(crypto_markets)} crypto markets")

    return crypto_markets

def add_crypto_specific_metrics(df):
    """Add metrics specific to crypto market analysis."""

    return df.with_columns([
        # Classify by crypto asset
        pl.when(pl.col('question').str.contains('(?i)bitcoin|btc'))
            .then(pl.lit('Bitcoin'))
        .when(pl.col('question').str.contains('(?i)ethereum|eth'))
            .then(pl.lit('Ethereum'))
        .when(pl.col('question').str.contains('(?i)solana|sol'))
            .then(pl.lit('Solana'))
        .when(pl.col('question').str.contains('(?i)cardano|ada'))
            .then(pl.lit('Cardano'))
        .when(pl.col('question').str.contains('(?i)xrp|ripple'))
            .then(pl.lit('XRP'))
        .when(pl.col('question').str.contains('(?i)dogecoin|doge'))
            .then(pl.lit('Dogecoin'))
        .otherwise(pl.lit('Other Crypto'))
        .alias('crypto_asset'),

        # Classify by market type
        pl.when(pl.col('question').str.contains('(?i)price|reach|\$|hit'))
            .then(pl.lit('Price Prediction'))
        .when(pl.col('question').str.contains('(?i)all.?time.?high|ath'))
            .then(pl.lit('All-Time High'))
        .when(pl.col('question').str.contains('(?i)market.?cap'))
            .then(pl.lit('Market Cap'))
        .when(pl.col('question').str.contains('(?i)etf|fund'))
            .then(pl.lit('ETF/Fund'))
        .otherwise(pl.lit('Other'))
        .alias('market_type'),

        # Calculate potential daily earnings (rough estimate)
        # Assumes you capture 50% of reward pool and 20% of spread
        ((pl.col('competitive') * 0.5) + (pl.col('volume_24hr') * 0.002))
            .alias('estimated_daily_profit'),
    ])

def print_crypto_summary(df):
    """Print summary statistics for crypto markets."""

    print("\n" + "=" * 80)
    print(" CRYPTO MARKETS SUMMARY")
    print("=" * 80)

    print(f"\nTotal crypto markets: {len(df)}")

    # Breakdown by crypto asset
    print("\nBy Crypto Asset:")
    by_asset = df.group_by('crypto_asset').agg([
        pl.count().alias('count'),
        pl.col('volume_24hr').sum().alias('total_volume'),
        pl.col('competitive').sum().alias('total_rewards')
    ]).sort('total_volume', descending=True)

    for row in by_asset.iter_rows(named=True):
        print(f"  {row['crypto_asset']:15} {row['count']:4} markets | "
              f"${row['total_volume']:>12,.0f} vol | ${row['total_rewards']:>6.2f} rewards")

    # Breakdown by market type
    print("\nBy Market Type:")
    by_type = df.group_by('market_type').agg([
        pl.count().alias('count'),
        pl.col('volume_24hr').mean().alias('avg_volume')
    ]).sort('count', descending=True)

    for row in by_type.iter_rows(named=True):
        print(f"  {row['market_type']:20} {row['count']:4} markets | "
              f"Avg vol: ${row['avg_volume']:>10,.0f}")

    # Top markets by volume
    print("\nTop 10 Crypto Markets by Volume:")
    top_volume = df.sort('volume_24hr', descending=True).head(10)
    for i, row in enumerate(top_volume.iter_rows(named=True), 1):
        question = row['question'][:60] + "..." if len(row['question']) > 60 else row['question']
        print(f"  {i:2}. ${row['volume_24hr']:>10,.0f} | ${row['competitive']:.2f}/day | {question}")

    # Markets with best reward opportunities
    print("\nTop 10 by Estimated Daily Profit:")
    with_profit = df.filter(pl.col('estimated_daily_profit').is_not_null())
    top_profit = with_profit.sort('estimated_daily_profit', descending=True).head(10)
    for i, row in enumerate(top_profit.iter_rows(named=True), 1):
        question = row['question'][:60] + "..." if len(row['question']) > 60 else row['question']
        print(f"  {i:2}. ${row['estimated_daily_profit']:>6.2f}/day | {question}")

    # Active markets with liquidity
    active = df.filter(
        (pl.col('accepting_orders') == True) &
        (pl.col('volume_24hr') > 100) &
        (pl.col('liquidity_clob') > 100)
    )
    print(f"\nActive liquid markets: {len(active)} ({len(active)/len(df)*100:.1f}%)")

    print("\n" + "=" * 80)

def save_crypto_parquet(df, output_dir='data/rewards'):
    """Save crypto markets to Parquet file."""

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Generate filename with timestamp
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = f'crypto_rewards_{timestamp}.parquet'
    filepath = output_path / filename

    print(f"\nSaving crypto markets to {filepath}...")
    df.write_parquet(filepath, compression='zstd')

    file_size_kb = filepath.stat().st_size / 1024
    print(f"Saved {len(df)} crypto markets ({file_size_kb:.1f} KB)")

    return filepath

def export_csv_for_excel(df, output_dir='data/rewards'):
    """Export to CSV for easy viewing in Excel."""

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = f'crypto_rewards_{timestamp}.csv'
    filepath = output_path / filename

    # Select and order columns for readability
    export_df = df.select([
        'crypto_asset',
        'market_type',
        'question',
        'slug',
        'volume_24hr',
        'volume_1wk',
        'competitive',
        'rewards_min_size',
        'rewards_max_spread',
        'liquidity_clob',
        'spread',
        'best_bid',
        'best_ask',
        'estimated_daily_profit',
        'accepting_orders',
        'condition_id',
        'token_id_0',
        'token_id_1'
    ]).sort('volume_24hr', descending=True)

    print(f"Exporting to CSV: {filepath}...")
    export_df.write_csv(filepath)
    print(f"Saved CSV ({len(export_df)} rows)")

    return filepath

def main():
    """Main execution function."""

    print("=" * 80)
    print(" CRYPTO REWARDS EXTRACTOR")
    print("=" * 80)
    print()

    # Load latest snapshot
    result = load_latest_snapshot()
    if result is None:
        return

    df, source_file = result
    print(f"Source: {source_file}\n")

    # Filter for crypto markets
    crypto_df = filter_crypto_markets(df)

    if crypto_df.is_empty():
        print("No crypto markets found!")
        return

    # Add crypto-specific metrics
    print("\nCalculating crypto-specific metrics...")
    crypto_df = add_crypto_specific_metrics(crypto_df)

    # Print summary
    print_crypto_summary(crypto_df)

    # Save outputs
    parquet_file = save_crypto_parquet(crypto_df)
    csv_file = export_csv_for_excel(crypto_df)

    print("\n" + "=" * 80)
    print(" EXTRACTION COMPLETE")
    print("=" * 80)

    print("\nOutput files:")
    print(f"  Parquet: {parquet_file}")
    print(f"  CSV:     {csv_file}")

    print("\nTo load the Parquet file:")
    print(f"  import polars as pl")
    print(f"  df = pl.read_parquet('{parquet_file}')")
    print(f"  print(df)")

    print("\nTo open CSV in Excel:")
    print(f"  open {csv_file}")

    print("\nTo filter by specific crypto:")
    print(f"  btc = df.filter(pl.col('crypto_asset') == 'Bitcoin')")
    print(f"  eth = df.filter(pl.col('crypto_asset') == 'Ethereum')")

if __name__ == "__main__":
    main()
