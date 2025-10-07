#!/usr/bin/env python3
"""
Calculate position-based PnL for BTC 15min markets

Tracks net positions per trader per market to calculate TRUE realized PnL.
This accounts for traders who buy and sell within the same market period.

Usage:
    uv run python scripts/calculate_position_based_pnl.py
    uv run python scripts/calculate_position_based_pnl.py --verbose
"""

import polars as pl
from pathlib import Path
import argparse

ENRICHED_TRADES_FILE = 'research/historical_trading/btc_15min_trades_enriched.parquet'
OUTPUT_FILE = 'research/historical_trading/btc_15min_position_pnl.parquet'


def load_enriched_trades(verbose: bool = False):
    """Load enriched trade data"""
    if verbose:
        print(f"Loading enriched trades from: {ENRICHED_TRADES_FILE}")

    df = pl.read_parquet(ENRICHED_TRADES_FILE)

    if verbose:
        print(f"  ✓ Loaded {len(df):,} trades")
        print(f"  Unique traders: {df['proxyWallet'].n_unique():,}")
        print(f"  Unique markets: {df['slug'].n_unique():,}")

    return df


def calculate_positions_per_market(trades_df: pl.DataFrame, verbose: bool = False):
    """
    Calculate net positions for each trader in each market

    Returns DataFrame with one row per (trader, market) combination
    """
    if verbose:
        print("\nCalculating net positions per trader per market...")

    # For each trade, calculate the position change
    trades_with_position = trades_df.with_columns([
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
        .alias('cash_flow')
    ])

    # Aggregate by (trader, market)
    positions = trades_with_position.group_by(['proxyWallet', 'slug']).agg([
        # Sum up all position changes
        pl.col('up_position_delta').sum().alias('net_up_tokens'),
        pl.col('down_position_delta').sum().alias('net_down_tokens'),
        pl.col('cash_flow').sum().alias('total_cash_flow'),

        # Keep market info
        pl.col('up_price').first().alias('up_settlement_price'),
        pl.col('down_price').first().alias('down_settlement_price'),
        pl.col('market_closed').first().alias('market_closed'),
        pl.col('up_won').first().alias('up_won'),
        pl.col('down_won').first().alias('down_won'),

        # Trade statistics
        pl.col('side').len().alias('num_trades'),
        pl.col('timestamp').min().alias('first_trade_time'),
        pl.col('timestamp').max().alias('last_trade_time'),
    ])

    # Calculate final position value and net PnL
    positions = positions.with_columns([
        # Value of final position at settlement
        (pl.col('net_up_tokens') * pl.col('up_settlement_price') +
         pl.col('net_down_tokens') * pl.col('down_settlement_price'))
        .alias('final_position_value'),

        # Net PnL = cash flow + final position value
        (pl.col('total_cash_flow') +
         pl.col('net_up_tokens') * pl.col('up_settlement_price') +
         pl.col('net_down_tokens') * pl.col('down_settlement_price'))
        .alias('net_pnl'),

        # Check if position was fully closed (no tokens held)
        ((pl.col('net_up_tokens').abs() < 0.001) &
         (pl.col('net_down_tokens').abs() < 0.001))
        .alias('position_fully_closed')
    ])

    if verbose:
        print(f"  ✓ Calculated positions for {len(positions):,} (trader, market) pairs")
        closed = positions.filter(pl.col('market_closed') == True)
        print(f"    Resolved markets: {len(closed):,}")
        fully_closed = closed.filter(pl.col('position_fully_closed') == True)
        print(f"    Fully closed positions: {len(fully_closed):,}")
        partial = closed.filter(pl.col('position_fully_closed') == False)
        print(f"    Partial positions held: {len(partial):,}")

    return positions


def analyze_position_pnl(positions_df: pl.DataFrame):
    """Analyze and print position-based PnL statistics"""
    print("\n" + "="*80)
    print("POSITION-BASED PNL ANALYSIS")
    print("="*80)

    # Filter to resolved markets only
    resolved = positions_df.filter(pl.col('market_closed') == True)

    print(f"\nTotal (trader, market) positions: {len(positions_df):,}")
    print(f"Resolved markets: {len(resolved):,}")

    if len(resolved) == 0:
        print("\nNo resolved positions to analyze.")
        return

    # Overall PnL stats
    total_pnl = resolved['net_pnl'].sum()
    avg_pnl = resolved['net_pnl'].mean()
    median_pnl = resolved['net_pnl'].median()

    print(f"\n=== OVERALL PNL ===")
    print(f"Total net PnL: ${total_pnl:,.2f}")
    print(f"Average PnL per position: ${avg_pnl:,.2f}")
    print(f"Median PnL per position: ${median_pnl:,.2f}")

    # Winners vs losers
    winners = resolved.filter(pl.col('net_pnl') > 0)
    losers = resolved.filter(pl.col('net_pnl') < 0)
    breakeven = resolved.filter(pl.col('net_pnl').abs() < 0.01)

    print(f"\n=== POSITION RESULTS ===")
    print(f"Winning positions: {len(winners):,} ({len(winners)/len(resolved)*100:.1f}%)")
    print(f"  Total profit: ${winners['net_pnl'].sum():,.2f}")
    print(f"Losing positions: {len(losers):,} ({len(losers)/len(resolved)*100:.1f}%)")
    print(f"  Total loss: ${losers['net_pnl'].sum():,.2f}")
    print(f"Breakeven: {len(breakeven):,}")

    # Fully closed vs held positions
    fully_closed = resolved.filter(pl.col('position_fully_closed') == True)
    partial = resolved.filter(pl.col('position_fully_closed') == False)

    print(f"\n=== POSITION STATUS ===")
    print(f"Fully closed (0 tokens held): {len(fully_closed):,}")
    print(f"  Net PnL: ${fully_closed['net_pnl'].sum():,.2f}")
    print(f"Partial positions held: {len(partial):,}")
    print(f"  Net PnL: ${partial['net_pnl'].sum():,.2f}")

    # Trading activity stats
    print(f"\n=== TRADING ACTIVITY ===")
    print(f"Average trades per position: {resolved['num_trades'].mean():.1f}")
    print(f"Median trades per position: {resolved['num_trades'].median():.0f}")
    single_trade = resolved.filter(pl.col('num_trades') == 1)
    multi_trade = resolved.filter(pl.col('num_trades') > 1)
    print(f"Positions with 1 trade: {len(single_trade):,} ({len(single_trade)/len(resolved)*100:.1f}%)")
    print(f"Positions with >1 trade: {len(multi_trade):,} ({len(multi_trade)/len(resolved)*100:.1f}%)")

    # Examples
    print(f"\n=== EXAMPLE POSITIONS ===")

    print("\n1. Top 5 Profitable Fully-Closed Positions:")
    top_closed = (fully_closed
                  .sort('net_pnl', descending=True)
                  .select(['proxyWallet', 'slug', 'num_trades', 'total_cash_flow',
                          'net_up_tokens', 'net_down_tokens', 'net_pnl'])
                  .head(5))
    for row in top_closed.iter_rows(named=True):
        print(f"  Trader {row['proxyWallet'][:10]}... in {row['slug']}")
        print(f"    {row['num_trades']} trades, cash flow: ${row['total_cash_flow']:.2f}")
        print(f"    Final position: {row['net_up_tokens']:.2f} Up, {row['net_down_tokens']:.2f} Down")
        print(f"    Net PnL: ${row['net_pnl']:.2f}")

    print("\n2. Example Multi-Trade Positions (bought and sold):")
    multi_closed = (fully_closed
                   .filter(pl.col('num_trades') > 1)
                   .sort('net_pnl', descending=True)
                   .select(['proxyWallet', 'slug', 'num_trades', 'total_cash_flow',
                           'final_position_value', 'net_pnl'])
                   .head(5))
    for row in multi_closed.iter_rows(named=True):
        print(f"  Trader {row['proxyWallet'][:10]}... in {row['slug']}")
        print(f"    {row['num_trades']} trades")
        print(f"    Cash flow: ${row['total_cash_flow']:.2f}")
        print(f"    Final tokens value: ${row['final_position_value']:.2f}")
        print(f"    Net PnL: ${row['net_pnl']:.2f}")

    print("\n3. Example Partial Positions (still holding tokens):")
    partial_sample = (partial
                     .sort('net_pnl', descending=True)
                     .select(['proxyWallet', 'slug', 'num_trades', 'net_up_tokens',
                             'net_down_tokens', 'total_cash_flow', 'final_position_value', 'net_pnl'])
                     .head(3))
    for row in partial_sample.iter_rows(named=True):
        print(f"  Trader {row['proxyWallet'][:10]}... in {row['slug']}")
        print(f"    Final position: {row['net_up_tokens']:.2f} Up, {row['net_down_tokens']:.2f} Down")
        print(f"    Cash: ${row['total_cash_flow']:.2f}, Tokens: ${row['final_position_value']:.2f}")
        print(f"    Net PnL: ${row['net_pnl']:.2f}")

    print("\n" + "="*80)


def save_positions(df: pl.DataFrame, output_path: str, verbose: bool = False):
    """Save position data to parquet file"""
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    if verbose:
        print(f"\nSaving position data...")

    df.write_parquet(output_file)

    file_size_mb = output_file.stat().st_size / (1024 * 1024)

    print(f"\n{'='*80}")
    print("SAVED")
    print("="*80)
    print(f"File: {output_file}")
    print(f"Size: {file_size_mb:.2f} MB")
    print(f"Rows: {len(df):,}")
    print(f"Columns: {len(df.columns)}")


def main():
    parser = argparse.ArgumentParser(description='Calculate position-based PnL for BTC 15min markets')
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
    print("POSITION-BASED PNL CALCULATION")
    print("="*80)

    # Load enriched trades
    trades = load_enriched_trades(verbose=args.verbose)

    # Calculate positions
    positions = calculate_positions_per_market(trades, verbose=args.verbose)

    # Analyze
    analyze_position_pnl(positions)

    # Save
    save_positions(positions, args.output, verbose=args.verbose)

    print("\n✓ Complete")


if __name__ == '__main__':
    main()
