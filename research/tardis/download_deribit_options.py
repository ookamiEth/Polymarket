#!/usr/bin/env python3
"""
Dynamic Deribit Options Data Download Script (Quotes + Black-Scholes)

Downloads Deribit options quote data from Tardis.dev using the Raw API with
server-side filtering, then calculates implied volatilities and Greeks using
vectorized Black-Scholes formulas.

IMPORTANT: This script uses the Raw API 'quote' channel (best bid/ask) and
calculates IV & Greeks ourselves using Black-Scholes. This is more efficient
than downloading pre-calculated values from the 'ticker' channel.

Features:
- Server-side filtering (downloads only needed data!)
- Vectorized Black-Scholes IV calculation (no loops!)
- Vectorized Greeks calculation (delta, gamma, vega, theta, rho)
- Full command-line argument control
- Optional resampling to any interval
- Output to CSV or Parquet
- Supports BTC, ETH, or both
- Filter by strike range, days to expiry, option type

Usage:
    # Download BTC options for Oct 1, 2025 (free access - no API key)
    uv run python download_deribit_options.py \\
        --from-date 2025-10-01 \\
        --to-date 2025-10-01 \\
        --assets BTC

    # Download short-dated BTC calls with resampling
    uv run python download_deribit_options.py \\
        --from-date 2025-10-01 \\
        --to-date 2025-10-01 \\
        --assets BTC \\
        --option-type call \\
        --min-days 7 \\
        --max-days 30 \\
        --resample-interval 5s \\
        --output-format parquet

    # Download ETH options with strike range
    uv run python download_deribit_options.py \\
        --from-date 2025-10-01 \\
        --to-date 2025-10-01 \\
        --assets ETH \\
        --strike-min 2400 \\
        --strike-max 2600

Author: Expert Quant Dev
Date: 2025-10-08
"""

import argparse
import asyncio
import sys
import os
from datetime import datetime
from typing import Optional, List
import polars as pl

# Import our modules
from symbol_discovery import build_symbol_list
from raw_stream import stream_quote_data
from black_scholes import add_implied_volatility_to_dataframe, add_greeks_to_dataframe


# ============================================================================
# CLI ARGUMENT PARSING (from test_cli_arguments.py)
# ============================================================================


class CLIArgumentParser:
    """Validates and parses command-line arguments."""

    @staticmethod
    def parse_date(date_str):
        """Parse date string to datetime object."""
        try:
            return datetime.strptime(date_str, '%Y-%m-%d')
        except ValueError as e:
            raise ValueError(f"Invalid date format: {date_str}. Expected YYYY-MM-DD") from e

    @staticmethod
    def validate_date_range(from_date, to_date):
        """Validate that from_date <= to_date."""
        if from_date > to_date:
            raise ValueError(f"from_date ({from_date}) must be <= to_date ({to_date})")
        return True

    @staticmethod
    def parse_assets(assets_str):
        """Parse comma-separated assets string."""
        if not assets_str:
            raise ValueError("Assets cannot be empty")

        assets = [a.strip().upper() for a in assets_str.split(',')]
        valid_assets = {'BTC', 'ETH'}

        invalid = set(assets) - valid_assets
        if invalid:
            raise ValueError(f"Invalid assets: {invalid}. Valid options: {valid_assets}")

        return assets

    @staticmethod
    def validate_days_range(min_days, max_days):
        """Validate days to expiry range."""
        if min_days is not None and min_days < 0:
            raise ValueError(f"min_days must be >= 0, got {min_days}")

        if max_days is not None and max_days < 0:
            raise ValueError(f"max_days must be >= 0, got {max_days}")

        if min_days is not None and max_days is not None and min_days > max_days:
            raise ValueError(f"min_days ({min_days}) must be <= max_days ({max_days})")

        return True

    @staticmethod
    def parse_option_type(option_type_str):
        """Parse and validate option type."""
        if not option_type_str:
            return 'both'

        option_type = option_type_str.lower()
        valid_types = {'call', 'put', 'both'}

        if option_type not in valid_types:
            raise ValueError(f"Invalid option_type: {option_type}. Valid options: {valid_types}")

        return option_type

    @staticmethod
    def validate_strike_range(strike_min, strike_max):
        """Validate strike price range."""
        if strike_min is not None and strike_min < 0:
            raise ValueError(f"strike_min must be >= 0, got {strike_min}")

        if strike_max is not None and strike_max < 0:
            raise ValueError(f"strike_max must be >= 0, got {strike_max}")

        if strike_min is not None and strike_max is not None and strike_min > strike_max:
            raise ValueError(f"strike_min ({strike_min}) must be <= strike_max ({strike_max})")

        return True

    @staticmethod
    def parse_resample_interval(interval_str):
        """Parse and validate resample interval."""
        if not interval_str:
            return None

        valid_suffixes = {'s', 'm', 'h', 'd'}

        if len(interval_str) < 2:
            raise ValueError(f"Invalid interval format: {interval_str}")

        suffix = interval_str[-1].lower()
        if suffix not in valid_suffixes:
            raise ValueError(f"Invalid interval suffix: {suffix}. Valid: {valid_suffixes}")

        try:
            value = int(interval_str[:-1])
            if value <= 0:
                raise ValueError(f"Interval value must be > 0, got {value}")
        except ValueError as e:
            raise ValueError(f"Invalid interval value: {interval_str}") from e

        return interval_str

    @staticmethod
    def validate_output_format(format_str):
        """Validate output format."""
        if not format_str:
            return 'parquet'

        format_str = format_str.lower()
        valid_formats = {'csv', 'parquet'}

        if format_str not in valid_formats:
            raise ValueError(f"Invalid output_format: {format_str}. Valid options: {valid_formats}")

        return format_str


# ============================================================================
# RESAMPLING (from test_resampling.py)
# ============================================================================


class Resampler:
    """Vectorized resampling operations using Polars."""

    @staticmethod
    def prepare_for_resampling(df: pl.DataFrame) -> pl.DataFrame:
        """Prepare dataframe for resampling by converting timestamp to datetime."""
        df = df.with_columns([
            (pl.col('timestamp').cast(pl.Int64) * 1000)
              .cast(pl.Datetime('us'))
              .alias('datetime')
        ])
        return df

    @staticmethod
    def resample_options_data(df: pl.DataFrame, interval: str) -> pl.DataFrame:
        """Resample options ticker data to specified interval vectorized."""
        # Sort by symbol and datetime
        df = df.sort(['symbol', 'datetime'])

        # Resample using group_by_dynamic
        resampled = (
            df.group_by_dynamic('datetime', every=interval, group_by=['symbol'])
            .agg([
                # OHLC for last_price
                pl.col('last_price').first().alias('open'),
                pl.col('last_price').max().alias('high'),
                pl.col('last_price').min().alias('low'),
                pl.col('last_price').last().alias('close'),

                # Last values for other prices
                pl.col('bid_price').last().alias('bid_price'),
                pl.col('ask_price').last().alias('ask_price'),
                pl.col('mark_price').last().alias('mark_price'),

                # Last values for greeks
                pl.col('delta').last().alias('delta'),
                pl.col('gamma').last().alias('gamma'),
                pl.col('theta').last().alias('theta'),
                pl.col('vega').last().alias('vega'),
                pl.col('rho').last().alias('rho'),

                # Last values for IV
                pl.col('mark_iv').last().alias('mark_iv'),
                pl.col('bid_iv').last().alias('bid_iv'),
                pl.col('ask_iv').last().alias('ask_iv'),

                # Last values for other fields
                pl.col('open_interest').last().alias('open_interest'),
                pl.col('underlying_price').last().alias('underlying_price'),

                # Metadata
                pl.col('strike_price').first().alias('strike_price'),
                pl.col('type').first().alias('type'),

                # Count
                pl.len().alias('tick_count'),
            ])
        )

        return resampled


# ============================================================================
# MAIN PIPELINE
# ============================================================================


def save_output(df: pl.DataFrame, output_dir: str, output_format: str, filename_suffix: str):
    """Save processed data to output file."""
    print("=" * 80)
    print("SAVING OUTPUT")
    print("=" * 80)

    # Create filename
    if output_format == 'parquet':
        filename = f"deribit_options_{filename_suffix}.parquet"
        filepath = os.path.join(output_dir, filename)
        df.write_parquet(filepath)
    else:  # csv
        filename = f"deribit_options_{filename_suffix}.csv"
        filepath = os.path.join(output_dir, filename)
        df.write_csv(filepath)

    print(f"✓ Saved: {filepath}")
    print(f"  Format: {output_format}")
    print(f"  Size: {os.path.getsize(filepath) / 1e6:.2f} MB")
    print()

    return filepath


async def main_async(
    from_date: str,
    to_date: str,
    assets: List[str],
    min_days: Optional[int],
    max_days: Optional[int],
    option_type: str,
    strike_min: Optional[float],
    strike_max: Optional[float],
    resample_interval: Optional[str],
    output_dir: str,
    output_format: str,
    api_key: Optional[str],
):
    """Main async pipeline."""

    # Parse from_date for reference
    reference_date = datetime.strptime(from_date, '%Y-%m-%d')

    # Step 1: Build symbol list with server-side filtering
    print("\n" + "╔" + "═" * 78 + "╗")
    print("║" + " " * 15 + "STEP 1: SYMBOL DISCOVERY (CLIENT-SIDE)" + " " * 22 + "║")
    print("╚" + "═" * 78 + "╝")
    print()

    symbols = await build_symbol_list(
        assets=assets,
        reference_date=reference_date,
        min_days=min_days,
        max_days=max_days,
        option_type=option_type,
        strike_min=strike_min,
        strike_max=strike_max,
    )

    if not symbols:
        print("\nERROR: No symbols found matching your criteria!")
        print("Try relaxing your filters (e.g., wider date range, strike range)")
        sys.exit(1)

    # Step 2: Stream quote data with server-side filtering
    print("\n" + "╔" + "═" * 78 + "╗")
    print("║" + " " * 15 + "STEP 2: STREAMING QUOTES (SERVER-SIDE)" + " " * 21 + "║")
    print("╚" + "═" * 78 + "╝")

    df = await stream_quote_data(
        symbols=symbols,
        from_date=from_date,
        to_date=to_date,
        api_key=api_key,
    )

    # Step 3: Calculate IV and Greeks using Black-Scholes
    print("\n" + "╔" + "═" * 78 + "╗")
    print("║" + " " * 10 + "STEP 3: CALCULATE IV & GREEKS (VECTORIZED)" + " " * 21 + "║")
    print("╚" + "═" * 78 + "╝")
    print()

    # Prepare data for Black-Scholes calculations
    print("[VECTORIZED] Preparing data for Black-Scholes...")

    # Add time to expiry in years
    df = df.with_columns([
        (pl.col('timestamp').cast(pl.Int64) / 1_000_000)  # Convert µs to seconds
        .cast(pl.Datetime('s'))
        .alias('datetime')
    ])

    # Parse expiration date from symbol
    # For now, we'll use a simplified approach - assume all options expire at same time
    # TODO: Parse exact expiration timestamp from Deribit API or symbol
    df = df.with_columns([
        (pl.col('expiry_str')
         .str.to_uppercase()
         .str.strptime(pl.Date, '%d%b%y', strict=False)
         .cast(pl.Datetime('s'))
         .alias('expiration_date'))
    ])

    df = df.with_columns([
        ((pl.col('expiration_date') - pl.col('datetime')).dt.total_seconds() / (365.25 * 24 * 3600))
        .alias('time_to_expiry')  # in years
    ])

    # Add risk-free rate (crypto typically uses 0%)
    df = df.with_columns([
        pl.lit(0.0).alias('risk_free_rate')
    ])

    print(f"✓ Prepared {df.shape[0]:,} rows for IV calculation")
    print()

    # Calculate implied volatility for bid, ask, and mark prices
    print("[VECTORIZED] Calculating bid IV...")
    df = add_implied_volatility_to_dataframe(
        df,
        price_column='bid_price',
        output_column='bid_iv',
    )

    print("[VECTORIZED] Calculating ask IV...")
    df = add_implied_volatility_to_dataframe(
        df,
        price_column='ask_price',
        output_column='ask_iv',
    )

    # Calculate mark price as mid of bid/ask
    df = df.with_columns([
        ((pl.col('bid_price') + pl.col('ask_price')) / 2).alias('mark_price')
    ])

    print("[VECTORIZED] Calculating mark IV...")
    df = add_implied_volatility_to_dataframe(
        df,
        price_column='mark_price',
        output_column='mark_iv',
    )

    print(f"✓ Calculated IV for {df.shape[0]:,} quotes")
    print()

    # Calculate Greeks using mark IV
    print("[VECTORIZED] Calculating Greeks (delta, gamma, vega, theta, rho)...")
    df = add_greeks_to_dataframe(df, iv_column='mark_iv')

    print(f"✓ Calculated Greeks for {df.shape[0]:,} options")
    print()

    # Step 4: Resample if requested
    if resample_interval:
        print("\n" + "╔" + "═" * 78 + "╗")
        print("║" + " " * 15 + "STEP 4: RESAMPLING (VECTORIZED)" + " " * 28 + "║")
        print("╚" + "═" * 78 + "╝")
        print()
        print(f"[VECTORIZED] Resampling to {resample_interval} intervals...")

        resampler = Resampler()
        df = resampler.prepare_for_resampling(df)
        before = df.shape[0]
        df = resampler.resample_options_data(df, resample_interval)

        print(f"✓ Resampled: {before:,} → {df.shape[0]:,} rows")
        print(f"  Compression ratio: {before / max(df.shape[0], 1):.1f}×")
        print()

    # Step 4: Save output
    filename_suffix = f"{from_date}_{'_'.join(assets)}"
    output_path = save_output(df, output_dir, output_format, filename_suffix)

    # Success!
    print("=" * 80)
    print("SUCCESS!")
    print("=" * 80)
    print(f"Data saved to: {output_path}")
    print(f"Total rows: {df.shape[0]:,}")
    print(f"Total symbols: {df['symbol'].n_unique()}")
    print()


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Download and filter Deribit options data using Raw API with server-side filtering",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Download BTC options for Oct 1, 2025 (free access)
  %(prog)s --from-date 2025-10-01 --to-date 2025-10-01 --assets BTC

  # Download short-dated BTC calls with resampling
  %(prog)s --from-date 2025-10-01 --to-date 2025-10-01 --assets BTC \\
           --option-type call --min-days 7 --max-days 30 \\
           --resample-interval 5s --output-format parquet

  # Download ETH options with strike range
  %(prog)s --from-date 2025-10-01 --to-date 2025-10-01 --assets ETH \\
           --strike-min 2400 --strike-max 2600
        """
    )

    # Required arguments
    parser.add_argument('--from-date', required=True, help='Start date (YYYY-MM-DD)')
    parser.add_argument('--to-date', required=True, help='End date (YYYY-MM-DD)')
    parser.add_argument('--assets', required=True, help='Comma-separated assets (BTC, ETH, or BTC,ETH)')

    # Optional filtering arguments
    parser.add_argument('--min-days', type=int, help='Minimum days to expiry (inclusive)')
    parser.add_argument('--max-days', type=int, help='Maximum days to expiry (inclusive)')
    parser.add_argument('--option-type', default='both', help='Option type: call, put, or both (default: both)')
    parser.add_argument('--strike-min', type=float, help='Minimum strike price (inclusive)')
    parser.add_argument('--strike-max', type=float, help='Maximum strike price (inclusive)')

    # Resampling
    parser.add_argument('--resample-interval', help='Resample interval (e.g., 1s, 5s, 1m, 5m)')

    # Output options
    parser.add_argument('--output-dir', default='./datasets_deribit_options', help='Output directory (default: ./datasets_deribit_options)')
    parser.add_argument('--output-format', default='parquet', choices=['csv', 'parquet'], help='Output format (default: parquet)')
    parser.add_argument('--api-key', help='Tardis.dev API key (optional for first day of month)')

    args = parser.parse_args()

    # Validate arguments
    try:
        validator = CLIArgumentParser()

        from_date = validator.parse_date(args.from_date)
        to_date = validator.parse_date(args.to_date)
        validator.validate_date_range(from_date, to_date)

        assets = validator.parse_assets(args.assets)
        validator.validate_days_range(args.min_days, args.max_days)
        option_type = validator.parse_option_type(args.option_type)
        validator.validate_strike_range(args.strike_min, args.strike_max)
        resample_interval = validator.parse_resample_interval(args.resample_interval)
        output_format = validator.validate_output_format(args.output_format)

    except ValueError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)

    # Create output directory if needed
    os.makedirs(args.output_dir, exist_ok=True)

    # Check for API key in environment if not provided
    api_key = args.api_key or os.environ.get('TARDIS_API_KEY')

    # Print configuration
    print("\n" + "╔" + "═" * 78 + "╗")
    print("║" + " " * 15 + "DERIBIT OPTIONS DOWNLOAD (RAW API)" + " " * 24 + "║")
    print("╚" + "═" * 78 + "╝")
    print()
    print("Configuration:")
    print(f"  Date range: {args.from_date} to {args.to_date}")
    print(f"  Assets: {', '.join(assets)}")
    print(f"  Option type: {option_type}")
    print(f"  Days to expiry: {args.min_days or 'any'} - {args.max_days or 'any'}")
    print(f"  Strike range: {args.strike_min or 'any'} - {args.strike_max or 'any'}")
    print(f"  Resample interval: {resample_interval or 'none'}")
    print(f"  Output format: {output_format}")
    print(f"  Output directory: {args.output_dir}")
    print()

    try:
        # Run async pipeline
        asyncio.run(main_async(
            from_date=args.from_date,
            to_date=args.to_date,
            assets=assets,
            min_days=args.min_days,
            max_days=args.max_days,
            option_type=option_type,
            strike_min=args.strike_min,
            strike_max=args.strike_max,
            resample_interval=resample_interval,
            output_dir=args.output_dir,
            output_format=output_format,
            api_key=api_key,
        ))

    except Exception as e:
        print(f"\nERROR: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
