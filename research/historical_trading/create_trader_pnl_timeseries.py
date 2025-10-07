#!/usr/bin/env python3
"""
Create PnL time series visualization for a specific trader

Applies master PnL formula to calculate position-level PnL for each market,
then visualizes both cumulative PnL and per-market PnL over time.

Usage:
    uv run python research/historical_trading/create_trader_pnl_timeseries.py
    uv run python research/historical_trading/create_trader_pnl_timeseries.py --trader 0x...
"""

import polars as pl
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from pathlib import Path
import argparse

TRADER_ADDRESS = '0xcc500cbcc8b7cf5bd21975ebbea34f21b5644c82'
INPUT_FILE = 'research/historical_trading/btc_15min_trades_enriched.parquet'


def calculate_trader_position_pnl(trader_address: str):
    """
    Calculate position-level PnL for a specific trader across all markets

    Uses the master PnL formula:
        net_pnl = net_cash_flow + final_position_value

    Returns: (positions_df, trader_name)
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

    # Calculate position deltas and cash flow for each trade
    trades_with_components = trader_df.with_columns([
        # Up token position change
        pl.when((pl.col('side') == 'BUY') & (pl.col('outcome') == 'Up'))
          .then(pl.col('size'))
        .when((pl.col('side') == 'SELL') & (pl.col('outcome') == 'Up'))
          .then(-pl.col('size'))
        .otherwise(0.0)
        .alias('up_position_delta'),

        # Down token position change
        pl.when((pl.col('side') == 'BUY') & (pl.col('outcome') == 'Down'))
          .then(pl.col('size'))
        .when((pl.col('side') == 'SELL') & (pl.col('outcome') == 'Down'))
          .then(-pl.col('size'))
        .otherwise(0.0)
        .alias('down_position_delta'),

        # Cash flow (negative = spent, positive = received)
        pl.when(pl.col('side') == 'BUY')
          .then(-pl.col('price') * pl.col('size'))
        .when(pl.col('side') == 'SELL')
          .then(pl.col('price') * pl.col('size'))
        .otherwise(0.0)
        .alias('cash_flow'),
    ])

    # Aggregate by market (slug)
    positions = trades_with_components.group_by('slug').agg([
        # Net positions
        pl.col('up_position_delta').sum().alias('net_up_tokens'),
        pl.col('down_position_delta').sum().alias('net_down_tokens'),
        pl.col('cash_flow').sum().alias('net_cash_flow'),

        # Market metadata
        pl.col('up_price').first().alias('up_settlement_price'),
        pl.col('down_price').first().alias('down_settlement_price'),
        pl.col('market_end_date').first().alias('market_end_date'),

        # Trading info
        pl.col('side').len().alias('num_trades'),
    ])

    # Parse market_end_date as datetime (ISO format with timezone)
    positions = positions.with_columns([
        pl.col('market_end_date').str.to_datetime(format='%Y-%m-%dT%H:%M:%SZ', time_zone='UTC').alias('market_end_date')
    ])

    # Calculate PnL using master formula
    positions = positions.with_columns([
        # Final position value at settlement
        (pl.col('net_up_tokens') * pl.col('up_settlement_price') +
         pl.col('net_down_tokens') * pl.col('down_settlement_price'))
        .alias('final_position_value'),
    ])

    # NET PNL = cash_flow + final_position_value (MASTER FORMULA)
    positions = positions.with_columns([
        (pl.col('net_cash_flow') + pl.col('final_position_value'))
        .alias('net_pnl')
    ])

    # Sort by market end date
    positions = positions.sort('market_end_date')

    # Calculate cumulative PnL
    positions = positions.with_columns([
        pl.col('net_pnl').cum_sum().alias('cumulative_pnl')
    ])

    print(f"  ✓ Calculated PnL for {len(positions):,} markets")
    print(f"    Total PnL: ${positions['net_pnl'].sum():,.2f}")
    print(f"    Final cumulative PnL: ${positions['cumulative_pnl'][-1]:,.2f}")

    return positions, trader_name


def create_visualization(positions_df: pl.DataFrame, trader_name: str, trader_address: str, output_path: str):
    """Create PDF with cumulative PnL line chart and per-market PnL bar chart"""

    print(f"\nCreating visualization...")

    # Convert to pandas for matplotlib
    df_pd = positions_df.to_pandas()

    # Create figure with 2 subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), sharex=True)
    fig.suptitle(f'Trader {trader_name} ({trader_address[:10]}...) PnL Time Series\nTotal PnL: ${df_pd["cumulative_pnl"].iloc[-1]:,.2f} across {len(df_pd)} markets',
                 fontsize=14, fontweight='bold')

    # Subplot 1: Cumulative PnL (line chart)
    ax1.plot(df_pd['market_end_date'], df_pd['cumulative_pnl'],
             linewidth=2, color='#2E86AB', alpha=0.8)
    ax1.axhline(y=0, color='black', linestyle='--', linewidth=0.5, alpha=0.5)
    ax1.set_ylabel('Cumulative PnL ($)', fontsize=11, fontweight='bold')
    ax1.set_title('Cumulative PnL Over Time', fontsize=12, fontweight='bold', pad=10)
    ax1.grid(True, alpha=0.3, linestyle='--')
    ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))

    # Add final PnL annotation
    final_pnl = df_pd['cumulative_pnl'].iloc[-1]
    ax1.annotate(f'${final_pnl:,.2f}',
                xy=(df_pd['market_end_date'].iloc[-1], final_pnl),
                xytext=(10, 10), textcoords='offset points',
                bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.7),
                fontsize=9, fontweight='bold')

    # Subplot 2: Per-Market PnL (bar chart)
    colors = ['green' if pnl > 0 else 'red' for pnl in df_pd['net_pnl']]
    ax2.bar(df_pd['market_end_date'], df_pd['net_pnl'],
            color=colors, alpha=0.6, width=0.03)  # thin bars for 1390 markets
    ax2.axhline(y=0, color='black', linestyle='-', linewidth=0.8)
    ax2.set_ylabel('Per-Market PnL ($)', fontsize=11, fontweight='bold')
    ax2.set_xlabel('Market End Date', fontsize=11, fontweight='bold')
    ax2.set_title('Individual Market PnL', fontsize=12, fontweight='bold', pad=10)
    ax2.grid(True, alpha=0.3, linestyle='--', axis='y')
    ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))

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
    parser = argparse.ArgumentParser(description='Create PnL time series visualization for a trader')
    parser.add_argument(
        '--trader',
        type=str,
        default=TRADER_ADDRESS,
        help=f'Trader address (default: {TRADER_ADDRESS[:10]}...)'
    )
    args = parser.parse_args()

    print("="*80)
    print("TRADER PNL TIME SERIES VISUALIZATION")
    print("="*80)

    # Step 1: Calculate position-level PnL for trader
    positions, trader_name = calculate_trader_position_pnl(args.trader)

    # Step 2: Create visualization
    output_file = f'research/historical_trading/trader_{args.trader[:10]}_pnl_timeseries.pdf'
    create_visualization(positions, trader_name, args.trader, output_file)

    print("\n✓ Complete")


if __name__ == '__main__':
    main()
