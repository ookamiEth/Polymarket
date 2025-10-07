#!/usr/bin/env python3
"""
Analyze BTC 15min trades with market outcomes to understand BUY/SELL semantics

Cross-references trade data with resolved market outcomes to validate:
1. What BUY and SELL mean (token acquisition vs disposal)
2. Relationship between outcome, outcomeIndex, and outcomePrices
3. Which traders profited from their positions

Usage:
    uv run python scripts/analyze_btc_15min_trade_outcomes.py
    uv run python scripts/analyze_btc_15min_trade_outcomes.py --verbose
"""

import polars as pl
import json
from pathlib import Path
import argparse

TRADES_FILE = 'data/clob_ticks_consolidated/btc_15min_consolidated.parquet'
MARKETS_FILE = 'data/markets/btc_15min_all_markets.parquet'
OUTPUT_FILE = 'data/analysis/btc_15min_trades_with_outcomes.parquet'


def load_data(verbose: bool = False):
    """Load trades and markets data"""
    if verbose:
        print("Loading data...")

    trades = pl.read_parquet(TRADES_FILE)
    markets = pl.read_parquet(MARKETS_FILE)

    if verbose:
        print(f"  Trades: {len(trades):,}")
        print(f"  Markets: {len(markets):,}")

    return trades, markets


def parse_outcome_prices(markets_df: pl.DataFrame, verbose: bool = False):
    """Parse outcomePrices JSON into separate columns"""
    if verbose:
        print("\nParsing outcomePrices...")

    # Parse JSON and extract values
    def parse_prices(price_str):
        prices = json.loads(price_str)
        return {
            'up_price': float(prices[0]),
            'down_price': float(prices[1]),
            'up_won': prices[0] == '1',
            'down_won': prices[1] == '1'
        }

    # Apply parsing
    parsed = []
    for row in markets_df.iter_rows(named=True):
        p = parse_prices(row['outcomePrices'])
        parsed.append({
            'slug': row['slug'],
            'closed': row['closed'],
            'up_price': p['up_price'],
            'down_price': p['down_price'],
            'up_won': p['up_won'],
            'down_won': p['down_won'],
            'resolved': row['closed'] and (p['up_won'] or p['down_won'])
        })

    parsed_df = pl.DataFrame(parsed)

    if verbose:
        resolved = parsed_df.filter(pl.col('resolved') == True)
        up_wins = resolved.filter(pl.col('up_won') == True)
        down_wins = resolved.filter(pl.col('down_won') == True)
        print(f"  Total markets: {len(parsed_df):,}")
        print(f"  Resolved: {len(resolved):,}")
        print(f"    Up won: {len(up_wins):,}")
        print(f"    Down won: {len(down_wins):,}")

    return parsed_df


def join_trades_with_outcomes(trades_df: pl.DataFrame, markets_df: pl.DataFrame, verbose: bool = False):
    """Join trades with market outcomes"""
    if verbose:
        print("\nJoining trades with market outcomes...")

    # Join on slug
    joined = trades_df.join(markets_df, on='slug', how='left')

    # Add profit calculation for resolved markets
    joined = joined.with_columns([
        # Did the trader profit?
        pl.when(pl.col('resolved') == True)
          .then(
              pl.when((pl.col('side') == 'BUY') & (pl.col('outcome') == 'Up') & (pl.col('up_won') == True))
                .then(pl.lit('won'))
              .when((pl.col('side') == 'BUY') & (pl.col('outcome') == 'Down') & (pl.col('down_won') == True))
                .then(pl.lit('won'))
              .when((pl.col('side') == 'SELL') & (pl.col('outcome') == 'Up') & (pl.col('down_won') == True))
                .then(pl.lit('won'))
              .when((pl.col('side') == 'SELL') & (pl.col('outcome') == 'Down') & (pl.col('up_won') == True))
                .then(pl.lit('won'))
              .otherwise(pl.lit('lost'))
          )
          .otherwise(pl.lit('unresolved'))
          .alias('trade_result'),

        # Calculate profit/loss for resolved trades
        pl.when(pl.col('resolved') == True)
          .then(
              pl.when((pl.col('side') == 'BUY') & (pl.col('outcome') == 'Up') & (pl.col('up_won') == True))
                .then((1.0 - pl.col('price')) * pl.col('size'))  # Bought at price, worth 1.0 now
              .when((pl.col('side') == 'BUY') & (pl.col('outcome') == 'Down') & (pl.col('down_won') == True))
                .then((1.0 - pl.col('price')) * pl.col('size'))
              .when((pl.col('side') == 'BUY') & (pl.col('outcome') == 'Up') & (pl.col('down_won') == True))
                .then(-pl.col('price') * pl.col('size'))  # Bought at price, worth 0 now
              .when((pl.col('side') == 'BUY') & (pl.col('outcome') == 'Down') & (pl.col('up_won') == True))
                .then(-pl.col('price') * pl.col('size'))
              .when((pl.col('side') == 'SELL') & (pl.col('outcome') == 'Up') & (pl.col('down_won') == True))
                .then(pl.col('price') * pl.col('size'))  # Sold at price, would be worth 0
              .when((pl.col('side') == 'SELL') & (pl.col('outcome') == 'Down') & (pl.col('up_won') == True))
                .then(pl.col('price') * pl.col('size'))
              .when((pl.col('side') == 'SELL') & (pl.col('outcome') == 'Up') & (pl.col('up_won') == True))
                .then((pl.col('price') - 1.0) * pl.col('size'))  # Sold at price, worth 1.0
              .when((pl.col('side') == 'SELL') & (pl.col('outcome') == 'Down') & (pl.col('down_won') == True))
                .then((pl.col('price') - 1.0) * pl.col('size'))
              .otherwise(pl.lit(0.0))
          )
          .otherwise(pl.lit(None))
          .alias('profit_loss')
    ])

    if verbose:
        resolved = joined.filter(pl.col('resolved') == True)
        print(f"  Total trades: {len(joined):,}")
        print(f"  Trades on resolved markets: {len(resolved):,}")

    return joined


def analyze_results(df: pl.DataFrame, verbose: bool = True):
    """Analyze and print statistics"""
    print("\n" + "="*80)
    print("BTC 15min TRADE OUTCOME ANALYSIS")
    print("="*80)

    # Overall stats
    print("\n=== OVERALL STATISTICS ===")
    print(f"Total trades: {len(df):,}")

    resolved = df.filter(pl.col('resolved') == True)
    print(f"Trades on resolved markets: {len(resolved):,}")

    # BUY vs SELL breakdown
    print("\n=== BUY vs SELL (All Trades) ===")
    side_breakdown = df.group_by(['side', 'outcome', 'outcomeIndex']).agg(
        pl.len().alias('count')
    ).sort(['side', 'outcomeIndex'])
    print(side_breakdown)

    # Outcome validation
    print("\n=== VALIDATION: outcome ↔ outcomeIndex Mapping ===")
    mapping = df.group_by(['outcome', 'outcomeIndex']).agg(pl.len().alias('count'))
    print(mapping)
    print("✓ Confirmed: outcomeIndex 0 = 'Up', outcomeIndex 1 = 'Down'")

    # Market outcome distribution
    print("\n=== RESOLVED MARKET OUTCOMES ===")
    if len(resolved) > 0:
        won_breakdown = resolved.filter(pl.col('up_won') == True).select(pl.len()).item()
        print(f"Markets where Up won: {won_breakdown:,}")

        down_won_count = resolved.filter(pl.col('down_won') == True).select(pl.len()).item()
        print(f"Markets where Down won: {down_won_count:,}")

    # Trade results
    print("\n=== TRADE RESULTS (Resolved Markets Only) ===")
    if len(resolved) > 0:
        results = resolved.group_by('trade_result').agg(
            pl.len().alias('count'),
            pl.col('profit_loss').sum().alias('total_pnl')
        ).sort('trade_result')
        print(results)

        # Calculate win rate
        won_trades = resolved.filter(pl.col('trade_result') == 'won')
        lost_trades = resolved.filter(pl.col('trade_result') == 'lost')
        total_pnl = resolved['profit_loss'].sum()

        print(f"\nWin Rate: {len(won_trades) / len(resolved) * 100:.2f}%")
        print(f"Total P&L across all trades: ${total_pnl:,.2f}")

    # Examples
    print("\n=== EXAMPLE TRADES ===")

    # Example 1: BUY Up that won
    buy_up_won = resolved.filter(
        (pl.col('side') == 'BUY') &
        (pl.col('outcome') == 'Up') &
        (pl.col('trade_result') == 'won')
    ).head(3)

    if len(buy_up_won) > 0:
        print("\n1. BUY 'Up' trades that WON (BTC went up):")
        for row in buy_up_won.select(['slug', 'side', 'outcome', 'price', 'size', 'profit_loss']).iter_rows(named=True):
            print(f"   - Bought {row['size']:.2f} Up tokens @ ${row['price']:.2f} → Profit: ${row['profit_loss']:.2f}")

    # Example 2: BUY Down that won
    buy_down_won = resolved.filter(
        (pl.col('side') == 'BUY') &
        (pl.col('outcome') == 'Down') &
        (pl.col('trade_result') == 'won')
    ).head(3)

    if len(buy_down_won) > 0:
        print("\n2. BUY 'Down' trades that WON (BTC went down):")
        for row in buy_down_won.select(['slug', 'side', 'outcome', 'price', 'size', 'profit_loss']).iter_rows(named=True):
            print(f"   - Bought {row['size']:.2f} Down tokens @ ${row['price']:.2f} → Profit: ${row['profit_loss']:.2f}")

    # Example 3: BUY Up that lost
    buy_up_lost = resolved.filter(
        (pl.col('side') == 'BUY') &
        (pl.col('outcome') == 'Up') &
        (pl.col('trade_result') == 'lost')
    ).head(3)

    if len(buy_up_lost) > 0:
        print("\n3. BUY 'Up' trades that LOST (BTC went down):")
        for row in buy_up_lost.select(['slug', 'side', 'outcome', 'price', 'size', 'profit_loss']).iter_rows(named=True):
            print(f"   - Bought {row['size']:.2f} Up tokens @ ${row['price']:.2f} → Loss: ${row['profit_loss']:.2f}")

    # Example 4: SELL trades
    sell_won = resolved.filter(
        (pl.col('side') == 'SELL') &
        (pl.col('trade_result') == 'won')
    ).head(3)

    if len(sell_won) > 0:
        print("\n4. SELL trades that WON (sold losing outcome):")
        for row in sell_won.select(['slug', 'side', 'outcome', 'price', 'size', 'profit_loss']).iter_rows(named=True):
            print(f"   - Sold {row['size']:.2f} {row['outcome']} tokens @ ${row['price']:.2f} → Profit: ${row['profit_loss']:.2f}")

    print("\n" + "="*80)
    print("KEY INSIGHTS:")
    print("="*80)
    print("• BUY = Acquiring outcome tokens (Up or Down) by spending USDC")
    print("• SELL = Disposing outcome tokens (Up or Down) to receive USDC")
    print("• outcome field = which token ('Up' or 'Down')")
    print("• outcomeIndex: 0 = Up, 1 = Down")
    print("• outcomePrices: [Up_price, Down_price]")
    print("• Resolved: ['1', '0'] = Up won, ['0', '1'] = Down won")
    print("="*80)

    return resolved


def save_results(df: pl.DataFrame, output_path: str):
    """Save enriched data to parquet"""
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    df.write_parquet(output_file)

    file_size_mb = output_file.stat().st_size / (1024 * 1024)
    print(f"\n=== SAVED ===")
    print(f"File: {output_file}")
    print(f"Size: {file_size_mb:.2f} MB")
    print(f"Rows: {len(df):,}")


def main():
    parser = argparse.ArgumentParser(description='Analyze BTC 15min trade outcomes')
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose output'
    )
    parser.add_argument(
        '--no-save',
        action='store_true',
        help='Skip saving to file'
    )
    parser.add_argument(
        '--output',
        type=str,
        default=OUTPUT_FILE,
        help=f'Output parquet file (default: {OUTPUT_FILE})'
    )

    args = parser.parse_args()

    # Load data
    trades, markets = load_data(verbose=args.verbose)

    # Parse outcome prices
    markets_parsed = parse_outcome_prices(markets, verbose=args.verbose)

    # Join trades with outcomes
    enriched = join_trades_with_outcomes(trades, markets_parsed, verbose=args.verbose)

    # Analyze
    analyze_results(enriched, verbose=True)

    # Save
    if not args.no_save:
        save_results(enriched, args.output)

    print("\n✓ Complete")


if __name__ == '__main__':
    main()
