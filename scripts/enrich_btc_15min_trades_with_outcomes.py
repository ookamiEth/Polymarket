#!/usr/bin/env python3
"""
Enrich BTC 15min trade data with market outcome information

Joins trade data with market metadata to add outcome prices and calculate PnL per trade.
Creates a new enriched parquet file without modifying the original data.

Usage:
    uv run python scripts/enrich_btc_15min_trades_with_outcomes.py
    uv run python scripts/enrich_btc_15min_trades_with_outcomes.py --verbose
"""

import polars as pl
import json
from pathlib import Path
import argparse

TRADES_FILE = 'data/clob_ticks_consolidated/btc_15min_consolidated.parquet'
MARKETS_FILE = 'data/markets/btc_15min_all_markets.parquet'
OUTPUT_FILE = 'research/historical_trading/btc_15min_trades_enriched.parquet'


def load_data(verbose: bool = False):
    """Load trades and markets data"""
    if verbose:
        print("Loading data...")
        print(f"  Trades: {TRADES_FILE}")
        print(f"  Markets: {MARKETS_FILE}")

    trades = pl.read_parquet(TRADES_FILE)
    markets = pl.read_parquet(MARKETS_FILE)

    if verbose:
        print(f"  ✓ Loaded {len(trades):,} trades")
        print(f"  ✓ Loaded {len(markets):,} markets")

    return trades, markets


def parse_outcome_prices(markets_df: pl.DataFrame, verbose: bool = False):
    """Parse outcomePrices JSON into separate columns"""
    if verbose:
        print("\nParsing market outcome prices...")

    # Parse JSON and extract values
    def parse_prices(price_str):
        prices = json.loads(price_str)
        return {
            'up_price': float(prices[0]),
            'down_price': float(prices[1]),
        }

    # Apply parsing
    parsed = []
    for row in markets_df.iter_rows(named=True):
        p = parse_prices(row['outcomePrices'])
        parsed.append({
            'slug': row['slug'],
            'market_closed': row['closed'],
            'market_start_date': row['startDate'],
            'market_end_date': row['endDate'],
            'up_price': p['up_price'],
            'down_price': p['down_price'],
            'up_won': p['up_price'] == 1.0,
            'down_won': p['down_price'] == 1.0,
        })

    parsed_df = pl.DataFrame(parsed)

    if verbose:
        closed = parsed_df.filter(pl.col('market_closed') == True)
        up_wins = closed.filter(pl.col('up_won') == True)
        down_wins = closed.filter(pl.col('down_won') == True)
        print(f"  ✓ Parsed {len(parsed_df):,} markets")
        print(f"    Closed: {len(closed):,}")
        print(f"      Up won: {len(up_wins):,}")
        print(f"      Down won: {len(down_wins):,}")

    return parsed_df


def join_and_calculate_pnl(trades_df: pl.DataFrame, markets_df: pl.DataFrame, verbose: bool = False):
    """Join trades with market outcomes and calculate PnL"""
    if verbose:
        print("\nJoining trades with markets by slug...")

    # Join on slug
    enriched = trades_df.join(markets_df, on='slug', how='left')

    if verbose:
        matched = enriched.filter(pl.col('market_closed').is_not_null())
        print(f"  ✓ Matched {len(matched):,} / {len(enriched):,} trades")

    if verbose:
        print("\nCalculating PnL for each trade...")

    # Calculate PnL based on trade side, outcome, and market result
    enriched = enriched.with_columns([
        # Calculate profit/loss for resolved trades
        pl.when(pl.col('market_closed') == True)
          .then(
              # BUY Up - won
              pl.when((pl.col('side') == 'BUY') & (pl.col('outcome') == 'Up') & (pl.col('up_won') == True))
                .then((1.0 - pl.col('price')) * pl.col('size'))
              # BUY Up - lost
              .when((pl.col('side') == 'BUY') & (pl.col('outcome') == 'Up') & (pl.col('down_won') == True))
                .then(-pl.col('price') * pl.col('size'))
              # BUY Down - won
              .when((pl.col('side') == 'BUY') & (pl.col('outcome') == 'Down') & (pl.col('down_won') == True))
                .then((1.0 - pl.col('price')) * pl.col('size'))
              # BUY Down - lost
              .when((pl.col('side') == 'BUY') & (pl.col('outcome') == 'Down') & (pl.col('up_won') == True))
                .then(-pl.col('price') * pl.col('size'))
              # SELL Up - market went down (sold before loss)
              .when((pl.col('side') == 'SELL') & (pl.col('outcome') == 'Up') & (pl.col('down_won') == True))
                .then(pl.col('price') * pl.col('size'))
              # SELL Up - market went up (sold too early)
              .when((pl.col('side') == 'SELL') & (pl.col('outcome') == 'Up') & (pl.col('up_won') == True))
                .then((pl.col('price') - 1.0) * pl.col('size'))
              # SELL Down - market went up (sold before loss)
              .when((pl.col('side') == 'SELL') & (pl.col('outcome') == 'Down') & (pl.col('up_won') == True))
                .then(pl.col('price') * pl.col('size'))
              # SELL Down - market went down (sold too early)
              .when((pl.col('side') == 'SELL') & (pl.col('outcome') == 'Down') & (pl.col('down_won') == True))
                .then((pl.col('price') - 1.0) * pl.col('size'))
              .otherwise(pl.lit(0.0))
          )
          .otherwise(pl.lit(None))
          .alias('pnl'),

        # Trade result label
        pl.when(pl.col('market_closed') == True)
          .then(
              pl.when(
                  ((pl.col('side') == 'BUY') & (pl.col('outcome') == 'Up') & (pl.col('up_won') == True)) |
                  ((pl.col('side') == 'BUY') & (pl.col('outcome') == 'Down') & (pl.col('down_won') == True)) |
                  ((pl.col('side') == 'SELL') & (pl.col('outcome') == 'Up') & (pl.col('down_won') == True)) |
                  ((pl.col('side') == 'SELL') & (pl.col('outcome') == 'Down') & (pl.col('up_won') == True))
              )
              .then(pl.lit('won'))
              .otherwise(pl.lit('lost'))
          )
          .otherwise(pl.lit('unresolved'))
          .alias('trade_result')
    ])

    if verbose:
        resolved = enriched.filter(pl.col('market_closed') == True)
        won = resolved.filter(pl.col('trade_result') == 'won')
        lost = resolved.filter(pl.col('trade_result') == 'lost')
        total_pnl = resolved['pnl'].sum()

        print(f"  ✓ Calculated PnL for {len(resolved):,} trades")
        print(f"    Won: {len(won):,} ({len(won)/len(resolved)*100:.1f}%)")
        print(f"    Lost: {len(lost):,} ({len(lost)/len(resolved)*100:.1f}%)")
        print(f"    Total PnL: ${total_pnl:,.2f}")

    return enriched


def analyze_enriched_data(df: pl.DataFrame):
    """Print summary statistics of enriched data"""
    print("\n" + "="*80)
    print("ENRICHED DATA SUMMARY")
    print("="*80)

    print(f"\nTotal trades: {len(df):,}")

    # Column breakdown
    print(f"\nColumns added:")
    new_cols = ['up_price', 'down_price', 'up_won', 'down_won', 'market_closed',
                'market_start_date', 'market_end_date', 'pnl', 'trade_result']
    for col in new_cols:
        if col in df.columns:
            print(f"  ✓ {col}")

    # Resolved vs unresolved
    resolved = df.filter(pl.col('market_closed') == True)
    unresolved = df.filter(pl.col('market_closed') == False)

    print(f"\nMarket status:")
    print(f"  Resolved: {len(resolved):,}")
    print(f"  Unresolved: {len(unresolved):,}")

    # PnL stats for resolved
    if len(resolved) > 0:
        print(f"\nPnL statistics (resolved markets only):")
        total_pnl = resolved['pnl'].sum()
        avg_pnl = resolved['pnl'].mean()
        median_pnl = resolved['pnl'].median()
        winning_trades = resolved.filter(pl.col('pnl') > 0)
        losing_trades = resolved.filter(pl.col('pnl') < 0)

        print(f"  Total PnL: ${total_pnl:,.2f}")
        print(f"  Average PnL per trade: ${avg_pnl:,.2f}")
        print(f"  Median PnL per trade: ${median_pnl:,.2f}")
        print(f"  Winning trades: {len(winning_trades):,} (${winning_trades['pnl'].sum():,.2f})")
        print(f"  Losing trades: {len(losing_trades):,} (${losing_trades['pnl'].sum():,.2f})")

    # Sample enriched trades
    print(f"\nSample enriched trades (first 5 from resolved markets):")
    sample = resolved.select([
        'slug', 'side', 'outcome', 'price', 'size',
        'up_price', 'down_price', 'trade_result', 'pnl'
    ]).head(5)
    print(sample)

    print("\n" + "="*80)


def save_enriched_data(df: pl.DataFrame, output_path: str, verbose: bool = False):
    """Save enriched data to parquet file"""
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    if verbose:
        print(f"\nSaving enriched data...")

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
    parser = argparse.ArgumentParser(description='Enrich BTC 15min trades with market outcomes')
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
    print("BTC 15MIN TRADE ENRICHMENT")
    print("="*80)

    # Load data
    trades, markets = load_data(verbose=args.verbose)

    # Parse outcome prices
    markets_parsed = parse_outcome_prices(markets, verbose=args.verbose)

    # Join and calculate PnL
    enriched = join_and_calculate_pnl(trades, markets_parsed, verbose=args.verbose)

    # Analyze
    analyze_enriched_data(enriched)

    # Save
    save_enriched_data(enriched, args.output, verbose=args.verbose)

    print("\n✓ Complete")


if __name__ == '__main__':
    main()
