#!/usr/bin/env python3
"""
Get BTC Short-Dated Options from Deribit for October 1, 2025

This script demonstrates how to:
1. Download Deribit options data for Oct 1, 2025 (free access)
2. Identify all available BTC option expiries
3. Filter for short-dated options (7-30 days to expiry)
4. Resample tick-level data to 1-second intervals using polars
5. Query for specific expiries programmatically

Date: October 1, 2025 (FREE ACCESS - first day of month, no API key needed)
"""

from tardis_dev import datasets
import polars as pl
from datetime import datetime, timedelta
import os


def download_deribit_options(date="2025-10-01", download_dir="./datasets_deribit_options"):
    """
    Download Deribit options_chain data for specified date.

    Args:
        date: Date to download (ISO format)
        download_dir: Directory to save downloaded files

    Returns:
        Path to downloaded file
    """
    print("=" * 80)
    print(f"DOWNLOADING DERIBIT OPTIONS DATA FOR {date}")
    print("=" * 80)
    print(f"Using FREE ACCESS (first day of month - no API key needed)")
    print(f"Download directory: {download_dir}")
    print()

    # Download using OPTIONS grouped symbol to get all options at once
    datasets.download(
        exchange="deribit",
        data_types=["options_chain"],
        from_date=date,
        to_date=date,
        symbols=["OPTIONS"],  # Special grouped symbol for all options
        api_key=None,  # Free access for first day of month
        download_dir=download_dir
    )

    # Construct expected filename
    filename = f"deribit_options_chain_{date}_OPTIONS.csv.gz"
    filepath = os.path.join(download_dir, filename)

    print(f"\nDownload complete!")
    print(f"File: {filepath}")
    print()

    return filepath


def parse_deribit_symbol(symbol: str) -> dict:
    """
    Parse Deribit option symbol into components.

    Symbol format: {BASE}-{EXPIRY}-{STRIKE}-{TYPE}
    Example: BTC-08OCT25-50000-C

    Returns:
        Dictionary with parsed components or None if not an option symbol
    """
    parts = symbol.split('-')

    if len(parts) < 4:
        return None  # Not an option symbol

    try:
        underlying = parts[0]
        expiry_str = parts[1]
        strike = float(parts[2])
        option_type = 'call' if parts[3] == 'C' else 'put'

        # Parse expiry date (format: DDMMMYY)
        expiry_date = datetime.strptime(expiry_str, '%d%b%y')

        return {
            'underlying': underlying,
            'expiry_str': expiry_str,
            'expiry_date': expiry_date,
            'strike': strike,
            'option_type': option_type
        }
    except Exception as e:
        return None


def analyze_options_data(filepath: str, target_date: datetime):
    """
    Load and analyze options data using polars.

    Args:
        filepath: Path to the CSV file
        target_date: Date of the data (to calculate days to expiry)
    """
    print("=" * 80)
    print("LOADING AND ANALYZING OPTIONS DATA")
    print("=" * 80)

    # Load CSV with polars
    print(f"Reading: {filepath}")
    df = pl.read_csv(filepath)

    print(f"\nDataset shape: {df.shape[0]:,} rows, {df.shape[1]} columns")
    print(f"Columns: {df.columns}")
    print()

    # Parse symbols and add expiry information
    print("Parsing option symbols...")

    # Create a function to parse symbols
    parsed_data = []
    for symbol in df['symbol'].to_list():
        parsed = parse_deribit_symbol(symbol)
        if parsed:
            parsed_data.append(parsed)
        else:
            parsed_data.append({
                'underlying': None,
                'expiry_str': None,
                'expiry_date': None,
                'strike': None,
                'option_type': None
            })

    # Add parsed columns to dataframe
    df = df.with_columns([
        pl.Series('underlying', [p['underlying'] for p in parsed_data]),
        pl.Series('expiry_str', [p['expiry_str'] for p in parsed_data]),
        pl.Series('expiry_date', [p['expiry_date'] for p in parsed_data]),
        pl.Series('parsed_strike', [p['strike'] for p in parsed_data]),
        pl.Series('parsed_option_type', [p['option_type'] for p in parsed_data])
    ])

    # Calculate days to expiry
    df = df.with_columns([
        ((pl.col('expiry_date') - target_date).dt.total_days()).alias('days_to_expiry')
    ])

    return df


def show_available_expiries(df: pl.DataFrame):
    """
    Show all available BTC option expiries.
    """
    print("=" * 80)
    print("AVAILABLE BTC OPTION EXPIRIES ON OCTOBER 1, 2025")
    print("=" * 80)
    print()

    # Filter for BTC options only
    btc_options = df.filter(pl.col('underlying') == 'BTC')

    # Group by expiry to see what's available
    expiries = (
        btc_options
        .group_by(['expiry_str', 'expiry_date', 'days_to_expiry'])
        .agg([
            pl.col('symbol').count().alias('num_contracts'),
            pl.col('parsed_strike').min().alias('min_strike'),
            pl.col('parsed_strike').max().alias('max_strike')
        ])
        .sort('days_to_expiry')
    )

    print(f"Total BTC options available: {btc_options.shape[0]:,} data points")
    print(f"Unique BTC option expiries: {expiries.shape[0]}")
    print()
    print("Expiry Breakdown:")
    print("-" * 80)
    print(f"{'Expiry Date':<12} {'Days Out':<10} {'Contracts':<12} {'Strike Range':<20}")
    print("-" * 80)

    for row in expiries.iter_rows(named=True):
        expiry_str = row['expiry_str']
        days = int(row['days_to_expiry'])
        num_contracts = row['num_contracts']
        min_strike = f"${row['min_strike']:,.0f}"
        max_strike = f"${row['max_strike']:,.0f}"
        strike_range = f"{min_strike} - {max_strike}"

        print(f"{expiry_str:<12} {days:<10} {num_contracts:<12} {strike_range:<20}")

    print()
    return expiries


def filter_short_dated_options(df: pl.DataFrame, min_days=7, max_days=30):
    """
    Filter for short-dated options (7-30 days to expiry).
    """
    print("=" * 80)
    print(f"FILTERING FOR SHORT-DATED OPTIONS ({min_days}-{max_days} days to expiry)")
    print("=" * 80)
    print()

    # Filter for BTC short-dated options
    short_dated = df.filter(
        (pl.col('underlying') == 'BTC') &
        (pl.col('days_to_expiry') >= min_days) &
        (pl.col('days_to_expiry') <= max_days)
    )

    print(f"Short-dated BTC options: {short_dated.shape[0]:,} data points")

    # Show unique expiries in this range
    expiries = (
        short_dated
        .select(['expiry_str', 'expiry_date', 'days_to_expiry'])
        .unique()
        .sort('days_to_expiry')
    )

    print(f"Unique expiries in range: {expiries.shape[0]}")
    print()
    print("Short-dated expiries:")
    for row in expiries.iter_rows(named=True):
        print(f"  {row['expiry_str']}: {int(row['days_to_expiry'])} days out")
    print()

    return short_dated


def query_specific_expiry(df: pl.DataFrame, expiry_str: str):
    """
    Query data for a specific expiry date.

    Args:
        df: Options dataframe
        expiry_str: Expiry string (e.g., '08OCT25')
    """
    print("=" * 80)
    print(f"QUERYING SPECIFIC EXPIRY: {expiry_str}")
    print("=" * 80)
    print()

    # Filter for specific expiry
    expiry_data = df.filter(
        (pl.col('underlying') == 'BTC') &
        (pl.col('expiry_str') == expiry_str)
    )

    print(f"Data points for {expiry_str}: {expiry_data.shape[0]:,}")

    # Show breakdown by option type and strike
    breakdown = (
        expiry_data
        .group_by(['parsed_option_type', 'parsed_strike'])
        .agg(pl.col('symbol').count().alias('data_points'))
        .sort(['parsed_option_type', 'parsed_strike'])
    )

    print(f"Unique strike/type combinations: {breakdown.shape[0]}")
    print()
    print("Sample breakdown (first 10):")
    print(breakdown.head(10))
    print()

    return expiry_data


def resample_to_1_second(df: pl.DataFrame, sample_expiry: str = None):
    """
    Demonstrate how to resample tick-level data to 1-second intervals.

    Args:
        df: Options dataframe
        sample_expiry: Optional specific expiry to demonstrate with
    """
    print("=" * 80)
    print("RESAMPLING TO 1-SECOND INTERVALS")
    print("=" * 80)
    print()

    # Take a sample for demonstration (specific expiry and strike)
    if sample_expiry:
        sample = df.filter(pl.col('expiry_str') == sample_expiry).head(10000)
    else:
        sample = df.filter(pl.col('underlying') == 'BTC').head(10000)

    if sample.shape[0] == 0:
        print("No data available for resampling")
        return None

    print(f"Sample size: {sample.shape[0]:,} tick-level data points")

    # Convert timestamp from microseconds to datetime
    sample = sample.with_columns([
        (pl.col('timestamp') / 1_000_000).cast(pl.Int64).cast(pl.Datetime('ms')).alias('datetime')
    ])

    # Group by symbol and 1-second intervals
    # Take first, last, min, max values for each second (OHLC-style)
    resampled = (
        sample
        .sort(['symbol', 'datetime'])
        .group_by_dynamic('datetime', every='1s', by='symbol')
        .agg([
            pl.col('last_price').first().alias('open'),
            pl.col('last_price').max().alias('high'),
            pl.col('last_price').min().alias('low'),
            pl.col('last_price').last().alias('close'),
            pl.col('volume').sum().alias('total_volume'),
            pl.col('open_interest').last().alias('open_interest'),
            pl.col('delta').last().alias('delta'),
            pl.col('gamma').last().alias('gamma'),
            pl.col('theta').last().alias('theta'),
            pl.col('vega').last().alias('vega'),
            pl.col('implied_volatility').last().alias('iv'),
            pl.col('bid').last().alias('bid'),
            pl.col('ask').last().alias('ask'),
        ])
    )

    print(f"After 1-second resampling: {resampled.shape[0]:,} data points")
    print()
    print("Sample resampled data (first 5 rows):")
    print(resampled.head(5))
    print()

    return resampled


def demonstrate_queries(df: pl.DataFrame):
    """
    Demonstrate various useful queries for options data.
    """
    print("=" * 80)
    print("EXAMPLE QUERIES")
    print("=" * 80)
    print()

    # Query 1: BTC calls expiring in 7-14 days with high delta (ITM)
    print("Query 1: BTC calls, 7-14 days to expiry, delta > 0.5 (ITM)")
    print("-" * 80)
    query1 = df.filter(
        (pl.col('underlying') == 'BTC') &
        (pl.col('parsed_option_type') == 'call') &
        (pl.col('days_to_expiry') >= 7) &
        (pl.col('days_to_expiry') <= 14) &
        (pl.col('delta') > 0.5)
    )
    print(f"Results: {query1.shape[0]:,} data points")
    unique_contracts = query1.select(['symbol', 'expiry_str', 'parsed_strike']).unique()
    print(f"Unique contracts: {unique_contracts.shape[0]}")
    print()

    # Query 2: BTC puts expiring in next 30 days with high IV
    print("Query 2: BTC puts, < 30 days to expiry, IV > 70%")
    print("-" * 80)
    query2 = df.filter(
        (pl.col('underlying') == 'BTC') &
        (pl.col('parsed_option_type') == 'put') &
        (pl.col('days_to_expiry') < 30) &
        (pl.col('implied_volatility') > 70)
    )
    print(f"Results: {query2.shape[0]:,} data points")
    print()

    # Query 3: ATM options (delta between 0.45 and 0.55 for calls, -0.55 and -0.45 for puts)
    print("Query 3: Near-ATM BTC options (|delta| between 0.45 and 0.55)")
    print("-" * 80)
    query3 = df.filter(
        (pl.col('underlying') == 'BTC') &
        (pl.col('days_to_expiry') <= 30) &
        (pl.col('delta').abs() >= 0.45) &
        (pl.col('delta').abs() <= 0.55)
    )
    print(f"Results: {query3.shape[0]:,} data points")
    unique_contracts = query3.select(['symbol', 'expiry_str', 'parsed_strike']).unique()
    print(f"Unique contracts: {unique_contracts.shape[0]}")
    print()


def main():
    """
    Main function to run the complete analysis.
    """
    print("\n")
    print("╔" + "═" * 78 + "╗")
    print("║" + " " * 15 + "BTC SHORT-DATED OPTIONS - OCTOBER 1, 2025" + " " * 22 + "║")
    print("║" + " " * 28 + "Deribit via Tardis.dev" + " " * 28 + "║")
    print("╚" + "═" * 78 + "╝")
    print()

    # Configuration
    DATA_DATE = "2025-10-01"
    TARGET_DATE = datetime(2025, 10, 1)
    DOWNLOAD_DIR = "./datasets_deribit_options"

    # Step 1: Download data
    filepath = download_deribit_options(date=DATA_DATE, download_dir=DOWNLOAD_DIR)

    # Step 2: Load and analyze data
    df = analyze_options_data(filepath, TARGET_DATE)

    # Step 3: Show available expiries
    expiries = show_available_expiries(df)

    # Step 4: Filter for short-dated options
    short_dated = filter_short_dated_options(df, min_days=7, max_days=30)

    # Step 5: Query specific expiry (use first short-dated expiry as example)
    if short_dated.shape[0] > 0:
        sample_expiry = short_dated.select('expiry_str').unique().sort('expiry_str').head(1).item(0, 0)
        query_specific_expiry(df, sample_expiry)

        # Step 6: Demonstrate 1-second resampling
        resample_to_1_second(df, sample_expiry=sample_expiry)

    # Step 7: Demonstrate various queries
    demonstrate_queries(df)

    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print("""
How to use this data:

1. Download: Use tardis-dev's datasets.download() with symbols=["OPTIONS"]
2. Parse symbols: Extract expiry, strike, type from symbol name
3. Filter: Use polars to filter by days_to_expiry, delta, IV, etc.
4. Resample: Use group_by_dynamic() for 1-second (or any interval) sampling
5. Query: Combine filters for specific option characteristics

Key insights:
- October 1, 2025 data is FREE (first day of month)
- Symbol format: BTC-{EXPIRY}-{STRIKE}-{C/P}
- Data includes: prices, greeks, IV, volume, open interest
- Tick-level granularity (sub-second)
- Easy to resample to any time interval using polars

Next steps:
- Modify filters for your specific trading strategy
- Add more sophisticated analysis (volatility smile, skew, etc.)
- Combine with other data sources (spot prices, funding rates, etc.)
- Backtest trading strategies using historical options data
    """)

    print("=" * 80)
    print("COMPLETE!")
    print("=" * 80)


if __name__ == "__main__":
    main()
