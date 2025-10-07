#!/usr/bin/env python3
"""
Create volume and trades visualization for a specific trader

Shows total volume per market and number of trades per market over time.

Usage:
    uv run python research/historical_trading/create_trader_volume_trades.py
    uv run python research/historical_trading/create_trader_volume_trades.py --trader 0x...
"""

import polars as pl
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from pathlib import Path
import argparse

TRADER_ADDRESS = '0xcc500cbcc8b7cf5bd21975ebbea34f21b5644c82'
INPUT_FILE = 'research/historical_trading/btc_15min_trades_enriched.parquet'


def calculate_trader_volume_trades(trader_address: str):
    """
    Calculate volume and trade count per market for a specific trader

    Returns: (market_stats_df, trader_name)
    """
    print(f"Loading trades for trader: {trader_address[:10]}...")

    # Load and filter to this trader, resolved markets only
    df = pl.read_parquet(INPUT_FILE)
    trader_df = df.filter(
        (pl.col('proxyWallet') == trader_address) &
        (pl.col('market_closed') == True)
    )

    # Extract trader name
    trader_name = trader_df['name'].unique().to_list()[0] if len(trader_df) > 0 else "Unknown"

    print(f"  Trader name: {trader_name}")
    print(f"  Found {len(trader_df):,} trades across {trader_df['slug'].n_unique()} markets")

    # Calculate volume contribution for each trade
    trades_with_volume = trader_df.with_columns([
        (pl.col('price') * pl.col('size')).alias('volume_contribution'),
    ])

    # Aggregate by market (slug)
    market_stats = trades_with_volume.group_by('slug').agg([
        # Volume: sum of (price × size) for all trades
        pl.col('volume_contribution').sum().alias('market_volume'),

        # Number of trades
        pl.col('side').len().alias('num_trades'),

        # Market end date for x-axis
        pl.col('market_end_date').first().alias('market_end_date'),
    ])

    # Parse market_end_date as datetime (ISO format with timezone)
    market_stats = market_stats.with_columns([
        pl.col('market_end_date').str.to_datetime(format='%Y-%m-%dT%H:%M:%SZ', time_zone='UTC').alias('market_end_date')
    ])

    # Sort by market end date (chronological)
    market_stats = market_stats.sort('market_end_date')

    print(f"  ✓ Calculated stats for {len(market_stats):,} markets")
    print(f"    Total volume: ${market_stats['market_volume'].sum():,.2f}")
    print(f"    Total trades: {market_stats['num_trades'].sum():,}")
    print(f"    Avg volume per market: ${market_stats['market_volume'].mean():,.2f}")
    print(f"    Avg trades per market: {market_stats['num_trades'].mean():.1f}")

    return market_stats, trader_name


def create_visualization(market_stats_df: pl.DataFrame, trader_name: str, trader_address: str, output_path: str):
    """Create PDF with volume and trades bar charts"""

    print(f"\nCreating visualization...")

    # Convert to pandas for matplotlib
    df_pd = market_stats_df.to_pandas()

    # Create figure with 2 subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), sharex=True)

    total_volume = df_pd['market_volume'].sum()
    total_trades = df_pd['num_trades'].sum()
    num_markets = len(df_pd)

    fig.suptitle(f'Trader {trader_name} ({trader_address[:10]}...) Volume & Trades\n'
                 f'Total Volume: ${total_volume:,.2f} | Total Trades: {total_trades:,} across {num_markets} markets',
                 fontsize=14, fontweight='bold')

    # Subplot 1: Volume per market (bar chart)
    ax1.bar(df_pd['market_end_date'], df_pd['market_volume'],
            color='#2E86AB', alpha=0.7, width=0.03)
    ax1.set_ylabel('Volume per Market ($)', fontsize=11, fontweight='bold')
    ax1.set_title('Total Volume per Market', fontsize=12, fontweight='bold', pad=10)
    ax1.grid(True, alpha=0.3, linestyle='--', axis='y')
    ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))

    # Subplot 2: Number of trades per market (bar chart)
    ax2.bar(df_pd['market_end_date'], df_pd['num_trades'],
            color='#6C757D', alpha=0.7, width=0.03)
    ax2.set_ylabel('Number of Trades', fontsize=11, fontweight='bold')
    ax2.set_xlabel('Market End Date', fontsize=11, fontweight='bold')
    ax2.set_title('Trades per Market', fontsize=12, fontweight='bold', pad=10)
    ax2.grid(True, alpha=0.3, linestyle='--', axis='y')
    ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{int(x)}'))

    # Format x-axis dates
    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    ax2.xaxis.set_major_locator(mdates.AutoDateLocator())
    plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45, ha='right')

    # Adjust layout and save
    plt.tight_layout()

    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_file, format='pdf', dpi=300, bbox_inches='tight')

    print(f"  ✓ Saved visualization to: {output_file}")
    print(f"    File size: {output_file.stat().st_size / 1024:.1f} KB")


def main():
    parser = argparse.ArgumentParser(description='Create volume and trades visualization for a trader')
    parser.add_argument(
        '--trader',
        type=str,
        default=TRADER_ADDRESS,
        help=f'Trader address (default: {TRADER_ADDRESS[:10]}...)'
    )
    args = parser.parse_args()

    print("="*80)
    print("TRADER VOLUME & TRADES VISUALIZATION")
    print("="*80)

    # Step 1: Calculate volume and trades per market
    market_stats, trader_name = calculate_trader_volume_trades(args.trader)

    # Step 2: Create visualization
    output_file = f'research/historical_trading/trader_{args.trader[:10]}_volume_trades.pdf'
    create_visualization(market_stats, trader_name, args.trader, output_file)

    print("\n✓ Complete")


if __name__ == '__main__':
    main()
