#!/usr/bin/env python3
"""
Calculate trader-level statistics across all BTC 15min markets

Applies the master PnL formula from docs/pnl_edge_cases_complete.md to properly
calculate position-based PnL, then aggregates to trader level.

Usage:
    uv run python research/historical_trading/calculate_trader_statistics.py
    uv run python research/historical_trading/calculate_trader_statistics.py --verbose
"""

import polars as pl
from pathlib import Path
import argparse

INPUT_FILE = 'research/historical_trading/btc_15min_trades_enriched.parquet'
OUTPUT_FILE = 'research/historical_trading/trader_statistics.parquet'


def load_and_filter_trades(verbose: bool = False):
    """Load enriched trades and filter to resolved markets only"""
    if verbose:
        print(f"Loading trades from: {INPUT_FILE}")

    df = pl.read_parquet(INPUT_FILE)

    if verbose:
        print(f"  Total trades: {len(df):,}")

    # Filter to resolved markets only
    resolved = df.filter(pl.col('market_closed') == True)

    if verbose:
        print(f"  Resolved market trades: {len(resolved):,}")
        print(f"  Unique traders: {resolved['proxyWallet'].n_unique():,}")
        print(f"  Unique markets: {resolved['slug'].n_unique():,}")

    return resolved


def calculate_position_level_pnl(trades_df: pl.DataFrame, verbose: bool = False):
    """
    Apply master PnL formula to calculate position-level PnL

    For each (trader, market):
        net_pnl = net_cash_flow + final_position_value
    """
    if verbose:
        print("\nCalculating position-level PnL (master formula)...")

    # Calculate components for each trade
    trades_with_components = trades_df.with_columns([
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

        # Volume contribution
        (pl.col('price') * pl.col('size')).alias('volume_contribution'),
    ])

    # Aggregate by (trader, market)
    positions = trades_with_components.group_by(['proxyWallet', 'slug']).agg([
        # Net positions
        pl.col('up_position_delta').sum().alias('net_up_tokens'),
        pl.col('down_position_delta').sum().alias('net_down_tokens'),
        pl.col('cash_flow').sum().alias('net_cash_flow'),

        # Market metadata
        pl.col('up_price').first().alias('up_settlement_price'),
        pl.col('down_price').first().alias('down_settlement_price'),
        pl.col('up_won').first().alias('up_won'),
        pl.col('down_won').first().alias('down_won'),

        # Trading activity
        pl.col('side').len().alias('num_trades'),
        pl.col('timestamp').min().alias('first_trade_time'),
        pl.col('timestamp').max().alias('last_trade_time'),
        pl.col('volume_contribution').sum().alias('market_volume'),
    ])

    # Calculate PnL using master formula
    positions = positions.with_columns([
        # Final position value
        (pl.col('net_up_tokens') * pl.col('up_settlement_price') +
         pl.col('net_down_tokens') * pl.col('down_settlement_price'))
        .alias('final_position_value'),

        # Holding time in hours
        ((pl.col('last_trade_time') - pl.col('first_trade_time')) / 3600.0)
        .alias('holding_time_hours'),

        # Position fully closed flag
        ((pl.col('net_up_tokens').abs() < 0.001) &
         (pl.col('net_down_tokens').abs() < 0.001))
        .alias('position_fully_closed'),
    ])

    # NET PNL = cash_flow + final_position_value (MASTER FORMULA)
    positions = positions.with_columns([
        (pl.col('net_cash_flow') + pl.col('final_position_value'))
        .alias('net_pnl')
    ])

    if verbose:
        print(f"  ✓ Calculated {len(positions):,} position-level PnLs")
        total_pnl = positions['net_pnl'].sum()
        print(f"    Total PnL across all positions: ${total_pnl:,.2f}")

    return positions


def aggregate_to_trader_level(positions_df: pl.DataFrame, verbose: bool = False):
    """Aggregate position-level stats to trader level"""
    if verbose:
        print("\nAggregating to trader level...")

    trader_stats = positions_df.group_by('proxyWallet').agg([
        # BASIC INFO
        pl.col('slug').n_unique().alias('markets_participated'),
        pl.col('num_trades').sum().alias('total_trades'),

        # TRADING ACTIVITY
        pl.col('num_trades').min().alias('min_trades_per_market'),
        pl.col('num_trades').max().alias('max_trades_per_market'),
        pl.col('num_trades').mean().alias('avg_trades_per_market'),

        # VOLUME METRICS
        pl.col('market_volume').sum().alias('total_volume_traded'),
        pl.col('market_volume').mean().alias('avg_volume_per_market'),
        pl.col('market_volume').max().alias('highest_volume_market'),
        pl.col('market_volume').min().alias('lowest_volume_market'),

        # PNL METRICS
        pl.col('net_pnl').sum().alias('total_pnl'),
        pl.col('net_pnl').mean().alias('avg_pnl_per_market'),
        pl.col('net_pnl').median().alias('median_pnl_per_market'),
        pl.col('net_pnl').max().alias('highest_pnl_market'),
        pl.col('net_pnl').min().alias('lowest_pnl_market'),

        # WIN/LOSS COUNTS
        pl.col('net_pnl').filter(pl.col('net_pnl') > 0).len().alias('winning_markets'),
        pl.col('net_pnl').filter(pl.col('net_pnl') < 0).len().alias('losing_markets'),
        pl.col('net_pnl').filter(pl.col('net_pnl').abs() < 0.01).len().alias('breakeven_markets'),

        # POSITION METRICS
        pl.col('position_fully_closed').sum().cast(pl.UInt32).alias('fully_closed_positions'),
        pl.col('position_fully_closed').filter(pl.col('position_fully_closed') == False).len().alias('partial_positions'),

        # TIME METRICS
        pl.col('first_trade_time').min().alias('first_trade_time'),
        pl.col('last_trade_time').max().alias('last_trade_time'),
        pl.col('holding_time_hours').mean().alias('avg_holding_time_per_market_hours'),

        # RISK METRICS
        pl.col('net_pnl').std().alias('pnl_std_dev'),
        pl.col('net_pnl').filter(pl.col('net_pnl') > 0).max().alias('largest_win'),
        pl.col('net_pnl').filter(pl.col('net_pnl') < 0).min().alias('largest_loss'),

        # For profit factor calculation
        pl.col('net_pnl').filter(pl.col('net_pnl') > 0).sum().alias('total_wins'),
        pl.col('net_pnl').filter(pl.col('net_pnl') < 0).sum().alias('total_losses'),
    ])

    # Calculate derived metrics
    trader_stats = trader_stats.with_columns([
        # Win rate (excluding breakeven)
        (pl.col('winning_markets') /
         (pl.col('winning_markets') + pl.col('losing_markets')))
        .alias('win_rate'),

        # Close rate
        (pl.col('fully_closed_positions') / pl.col('markets_participated'))
        .alias('close_rate'),

        # Trading duration in days
        ((pl.col('last_trade_time') - pl.col('first_trade_time')) / 86400.0)
        .alias('trading_duration_days'),

        # Profit factor
        (pl.col('total_wins') / pl.col('total_losses').abs())
        .alias('profit_factor'),
    ])

    # Rename proxyWallet to trader_address for clarity
    trader_stats = trader_stats.rename({'proxyWallet': 'trader_address'})

    # Select final columns in logical order
    final_columns = [
        # Basic
        'trader_address',
        'markets_participated',
        'total_trades',

        # Activity
        'avg_trades_per_market',
        'min_trades_per_market',
        'max_trades_per_market',

        # Volume
        'total_volume_traded',
        'avg_volume_per_market',
        'highest_volume_market',
        'lowest_volume_market',

        # PnL
        'total_pnl',
        'avg_pnl_per_market',
        'median_pnl_per_market',
        'highest_pnl_market',
        'lowest_pnl_market',

        # Win/Loss
        'winning_markets',
        'losing_markets',
        'breakeven_markets',
        'win_rate',

        # Positions
        'fully_closed_positions',
        'partial_positions',
        'close_rate',

        # Time
        'first_trade_time',
        'last_trade_time',
        'trading_duration_days',
        'avg_holding_time_per_market_hours',

        # Risk
        'pnl_std_dev',
        'largest_win',
        'largest_loss',
        'profit_factor',
    ]

    trader_stats = trader_stats.select(final_columns)

    if verbose:
        print(f"  ✓ Aggregated statistics for {len(trader_stats):,} traders")
        print(f"    Total columns: {len(trader_stats.columns)}")

    return trader_stats


def analyze_and_save(trader_stats: pl.DataFrame, output_path: str, verbose: bool = True):
    """Analyze trader statistics and save to parquet"""
    print("\n" + "="*80)
    print("TRADER STATISTICS SUMMARY")
    print("="*80)

    print(f"\nTotal traders analyzed: {len(trader_stats):,}")

    # Overall stats
    total_pnl = trader_stats['total_pnl'].sum()
    total_volume = trader_stats['total_volume_traded'].sum()

    print(f"\n=== OVERALL METRICS ===")
    print(f"Total PnL across all traders: ${total_pnl:,.2f}")
    print(f"Total volume traded: ${total_volume:,.2f}")

    # Top performers
    print(f"\n=== TOP 5 TRADERS BY TOTAL PNL ===")
    top5 = trader_stats.sort('total_pnl', descending=True).head(5)
    for row in top5.iter_rows(named=True):
        addr = row['trader_address'][:10] + '...'
        print(f"  {addr}: ${row['total_pnl']:,.2f} across {row['markets_participated']} markets (win rate: {row['win_rate']*100:.1f}%)")

    # Active traders
    print(f"\n=== MOST ACTIVE TRADERS ===")
    top_active = trader_stats.sort('markets_participated', descending=True).head(5)
    for row in top_active.iter_rows(named=True):
        addr = row['trader_address'][:10] + '...'
        print(f"  {addr}: {row['markets_participated']} markets, {row['total_trades']} trades")

    # Distribution stats
    print(f"\n=== DISTRIBUTION ===")
    profitable = trader_stats.filter(pl.col('total_pnl') > 0)
    unprofitable = trader_stats.filter(pl.col('total_pnl') < 0)

    print(f"Profitable traders: {len(profitable):,} ({len(profitable)/len(trader_stats)*100:.1f}%)")
    print(f"Unprofitable traders: {len(unprofitable):,} ({len(unprofitable)/len(trader_stats)*100:.1f}%)")

    avg_win_rate = trader_stats['win_rate'].mean()
    avg_close_rate = trader_stats['close_rate'].mean()

    print(f"\nAverage win rate: {avg_win_rate*100:.1f}%")
    print(f"Average close rate: {avg_close_rate*100:.1f}%")

    # Save
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    trader_stats.write_parquet(output_file)

    file_size_mb = output_file.stat().st_size / (1024 * 1024)

    print(f"\n{'='*80}")
    print("SAVED")
    print("="*80)
    print(f"File: {output_file}")
    print(f"Size: {file_size_mb:.2f} MB")
    print(f"Rows: {len(trader_stats):,} traders")
    print(f"Columns: {len(trader_stats.columns)}")


def main():
    parser = argparse.ArgumentParser(description='Calculate trader-level statistics')
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose output'
    )
    parser.add_argument(
        '--output',
        type=str,
        default=OUTPUT_FILE,
        help=f'Output parquet file (default: {OUTPUT_FILE})'
    )

    args = parser.parse_args()

    print("="*80)
    print("TRADER STATISTICS CALCULATION")
    print("="*80)

    # Step 1: Load and filter
    trades = load_and_filter_trades(verbose=args.verbose)

    # Step 2: Calculate position-level PnL (master formula)
    positions = calculate_position_level_pnl(trades, verbose=args.verbose)

    # Step 3: Aggregate to trader level
    trader_stats = aggregate_to_trader_level(positions, verbose=args.verbose)

    # Analyze and save
    analyze_and_save(trader_stats, args.output, verbose=True)

    print("\n✓ Complete")


if __name__ == '__main__':
    main()
