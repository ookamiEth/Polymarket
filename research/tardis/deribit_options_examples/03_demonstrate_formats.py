#!/usr/bin/env python3
"""
Deribit Options Data - Format Demonstration

This script demonstrates the expected data formats without actually fetching data.
Use this as a reference for understanding the structure of Deribit options data.
"""

import json
from datetime import datetime


def show_csv_format():
    """
    Demonstrate the CSV format for options_chain data.
    """
    print("=" * 80)
    print("CSV FORMAT: options_chain")
    print("=" * 80)
    print("\nThis is the format returned by tardis-dev package (CSV download approach)")
    print("\nColumns in options_chain CSV:")
    print("-" * 80)

    columns = [
        ("exchange", "string", "Exchange identifier (e.g., 'deribit')"),
        ("symbol", "string", "Option symbol (e.g., 'BTC-29NOV25-50000-C')"),
        ("timestamp", "int", "Exchange timestamp (microseconds since epoch)"),
        ("local_timestamp", "int", "Arrival timestamp (microseconds since epoch)"),
        ("underlying_price", "float", "Current price of underlying asset"),
        ("underlying_index", "string", "Underlying index name (e.g., 'BTC-USD')"),
        ("", "", ""),
        ("# Option Specification", "", ""),
        ("strike", "float", "Strike price"),
        ("expiration", "string", "Expiration date (ISO format)"),
        ("option_type", "string", "'call' or 'put'"),
        ("", "", ""),
        ("# Pricing", "", ""),
        ("bid", "float", "Best bid price"),
        ("ask", "float", "Best ask price"),
        ("last_price", "float", "Last traded price"),
        ("mark_price", "float", "Mark price used for margining"),
        ("", "", ""),
        ("# Greeks", "", ""),
        ("delta", "float", "Delta (sensitivity to underlying price)"),
        ("gamma", "float", "Gamma (rate of change of delta)"),
        ("theta", "float", "Theta (time decay)"),
        ("vega", "float", "Vega (sensitivity to volatility)"),
        ("rho", "float", "Rho (sensitivity to interest rate)"),
        ("", "", ""),
        ("# Volatility & Volume", "", ""),
        ("implied_volatility", "float", "Implied volatility (%)"),
        ("bid_iv", "float", "IV at best bid"),
        ("ask_iv", "float", "IV at best ask"),
        ("open_interest", "float", "Total open interest"),
        ("volume", "float", "Trading volume (24h)"),
    ]

    for col_name, col_type, description in columns:
        if col_name.startswith("#"):
            print(f"\n{col_name}")
        elif col_name:
            print(f"  {col_name:25} {col_type:10} {description}")

    print("\n" + "=" * 80)
    print("SAMPLE CSV DATA (options_chain)")
    print("=" * 80)

    sample_csv = """exchange,symbol,timestamp,local_timestamp,underlying_price,underlying_index,strike,expiration,option_type,bid,ask,last_price,mark_price,delta,gamma,theta,vega,rho,implied_volatility,bid_iv,ask_iv,open_interest,volume
deribit,BTC-29NOV25-50000-C,1633046400000000,1633046400050000,50000.0,BTC-USD,50000,2025-11-29,call,0.065,0.067,0.066,0.0665,0.55,0.00002,-12.5,45.2,3.2,65.5,64.0,67.0,1000,123.5
deribit,BTC-29NOV25-50000-P,1633046400000000,1633046400050000,50000.0,BTC-USD,50000,2025-11-29,put,0.063,0.065,0.064,0.064,-0.45,0.00002,-11.8,45.2,-3.2,64.8,63.5,66.5,850,98.2
deribit,BTC-29NOV25-55000-C,1633046400000000,1633046400050000,50000.0,BTC-USD,55000,2025-11-29,call,0.025,0.027,0.026,0.026,0.25,0.00003,-8.5,52.1,2.8,72.3,71.0,73.5,750,67.8
deribit,ETH-29NOV25-2000-C,1633046400000000,1633046400050000,2000.0,ETH-USD,2000,2025-11-29,call,0.075,0.077,0.076,0.076,0.52,0.00025,-9.2,38.5,1.8,68.2,67.0,69.5,1200,145.6
deribit,ETH-29NOV25-2000-P,1633046400000000,1633046400050000,2000.0,ETH-USD,2000,2025-11-29,put,0.072,0.074,0.073,0.073,-0.48,0.00025,-8.8,38.5,-1.8,67.5,66.2,68.8,980,112.3"""

    print(sample_csv)
    print()


def show_raw_api_format():
    """
    Demonstrate the raw API format for ticker channel.
    """
    print("\n" + "=" * 80)
    print("RAW API FORMAT: ticker channel")
    print("=" * 80)
    print("\nThis is the format returned by tardis-client package (Raw API approach)")
    print("Format: Exchange-native JSON (Deribit WebSocket messages)")
    print("-" * 80)

    ticker_message = {
        "jsonrpc": "2.0",
        "method": "subscription",
        "params": {
            "channel": "ticker.BTC-29NOV25-50000-C.raw",
            "data": {
                "timestamp": 1633046400000,
                "stats": {
                    "volume_usd": 1235000.50,
                    "volume": 123.5,
                    "price_change": 2.5,
                    "low": 0.050,
                    "high": 0.080
                },
                "state": "open",
                "settlement_price": 0.065,
                "open_interest": 1000.0,
                "min_price": 0.040,
                "max_price": 0.100,
                "mark_price": 0.067,
                "mark_iv": 65.5,
                "last_price": 0.066,
                "interest_rate": 0.0,
                "instrument_name": "BTC-29NOV25-50000-C",
                "index_price": 50000.0,
                "greeks": {
                    "vega": 45.2,
                    "theta": -12.5,
                    "rho": 3.2,
                    "gamma": 0.00002,
                    "delta": 0.55
                },
                "estimated_delivery_price": 50000.0,
                "current_funding": 0.0,
                "bid_iv": 64.0,
                "best_bid_price": 0.065,
                "best_bid_amount": 10.0,
                "best_ask_price": 0.067,
                "best_ask_amount": 15.0,
                "ask_iv": 67.0,
                "underlying_price": 50000.0,
                "underlying_index": "BTC-USD"
            }
        }
    }

    print("\nSAMPLE TICKER MESSAGE:")
    print(json.dumps(ticker_message, indent=2))

    print("\n" + "=" * 80)
    print("RAW API FORMAT: trades channel")
    print("=" * 80)

    trades_message = {
        "jsonrpc": "2.0",
        "method": "subscription",
        "params": {
            "channel": "trades.BTC-29NOV25-50000-C.raw",
            "data": [
                {
                    "trade_seq": 123456,
                    "trade_id": "BTC-123456",
                    "timestamp": 1633046400000,
                    "tick_direction": 0,
                    "price": 0.066,
                    "mark_price": 0.067,
                    "iv": 65.5,
                    "instrument_name": "BTC-29NOV25-50000-C",
                    "index_price": 50000.0,
                    "direction": "buy",
                    "amount": 5.0
                },
                {
                    "trade_seq": 123457,
                    "trade_id": "BTC-123457",
                    "timestamp": 1633046401500,
                    "tick_direction": 1,
                    "price": 0.067,
                    "mark_price": 0.067,
                    "iv": 65.8,
                    "instrument_name": "BTC-29NOV25-50000-C",
                    "index_price": 50005.0,
                    "direction": "sell",
                    "amount": 3.0
                }
            ]
        }
    }

    print("\nSAMPLE TRADES MESSAGE:")
    print(json.dumps(trades_message, indent=2))


def show_symbol_parsing_example():
    """
    Demonstrate how to parse Deribit option symbols.
    """
    print("\n" + "=" * 80)
    print("SYMBOL PARSING GUIDE")
    print("=" * 80)
    print("\nDeribit option symbol format: {BASE}-{EXPIRY}-{STRIKE}-{TYPE}")
    print()

    examples = [
        ("BTC-29NOV25-50000-C", "Bitcoin", "Nov 29, 2025", "50000", "Call"),
        ("BTC-29NOV25-50000-P", "Bitcoin", "Nov 29, 2025", "50000", "Put"),
        ("ETH-29NOV25-2000-C", "Ethereum", "Nov 29, 2025", "2000", "Call"),
        ("ETH-27DEC25-2500-P", "Ethereum", "Dec 27, 2025", "2500", "Put"),
        ("BTC-01NOV25-48000-C", "Bitcoin", "Nov 1, 2025", "48000", "Call"),
        ("SOL-15NOV25-100-P", "Solana", "Nov 15, 2025", "100", "Put"),
    ]

    print(f"{'Symbol':<25} {'Asset':<12} {'Expiry':<18} {'Strike':<10} {'Type':<6}")
    print("-" * 80)
    for symbol, asset, expiry, strike, opt_type in examples:
        print(f"{symbol:<25} {asset:<12} {expiry:<18} ${strike:<9} {opt_type:<6}")

    print("\n" + "=" * 80)
    print("PYTHON CODE: Parse Deribit Option Symbol")
    print("=" * 80)

    code = '''
def parse_deribit_option(symbol):
    """
    Parse Deribit option symbol into components.

    Example: BTC-29NOV25-50000-C
    Returns: {
        'underlying': 'BTC',
        'expiry_str': '29NOV25',
        'strike': 50000.0,
        'option_type': 'call'
    }
    """
    parts = symbol.split('-')

    if len(parts) < 4:
        return None  # Not an option symbol

    underlying = parts[0]
    expiry_str = parts[1]
    strike = float(parts[2])
    option_type = 'call' if parts[3] == 'C' else 'put'

    # Parse expiry date (format: DDMMMYY)
    from datetime import datetime
    expiry_date = datetime.strptime(expiry_str, '%d%b%y')

    return {
        'underlying': underlying,
        'expiry_str': expiry_str,
        'expiry_date': expiry_date,
        'strike': strike,
        'option_type': option_type
    }

# Example usage
symbol = "BTC-29NOV25-50000-C"
parsed = parse_deribit_option(symbol)
print(parsed)
# Output: {
#   'underlying': 'BTC',
#   'expiry_str': '29NOV25',
#   'expiry_date': datetime(2025, 11, 29),
#   'strike': 50000.0,
#   'option_type': 'call'
# }
'''

    print(code)


def show_filtering_example():
    """
    Show how to filter for short-dated options.
    """
    print("\n" + "=" * 80)
    print("FILTERING: Short-Dated Options")
    print("=" * 80)
    print("\nShort-dated options: typically 7-30 days to expiration")
    print()

    code = '''
import pandas as pd
from datetime import datetime, timedelta

# Load options_chain CSV
df = pd.read_csv("deribit_options_chain_2025-10-01_OPTIONS.csv.gz")

# Parse symbol to extract expiry date
def parse_expiry_from_symbol(symbol):
    """Extract expiry date from Deribit option symbol."""
    parts = symbol.split('-')
    if len(parts) >= 4:
        expiry_str = parts[1]  # e.g., '29NOV25'
        return datetime.strptime(expiry_str, '%d%b%y')
    return None

# Add expiry date and days to expiry columns
df['expiry_date'] = df['symbol'].apply(parse_expiry_from_symbol)
df['days_to_expiry'] = (df['expiry_date'] - datetime.now()).dt.days

# Filter for short-dated options (7-30 days)
short_dated = df[
    (df['days_to_expiry'] >= 7) &
    (df['days_to_expiry'] <= 30)
]

# Further filter for specific criteria
btc_calls_itm = short_dated[
    (short_dated['symbol'].str.startswith('BTC-')) &  # BTC options only
    (short_dated['option_type'] == 'call') &           # Calls only
    (short_dated['delta'] > 0.5)                       # In-the-money (delta > 0.5)
]

print(f"Total options: {len(df)}")
print(f"Short-dated (7-30 days): {len(short_dated)}")
print(f"BTC ITM calls: {len(btc_calls_itm)}")

# Show summary statistics
print("\\nShort-dated options summary:")
print(btc_calls_itm[['symbol', 'days_to_expiry', 'strike',
                      'last_price', 'delta', 'implied_volatility']].head(10))
'''

    print(code)


def show_comparison_table():
    """
    Show side-by-side comparison of both approaches.
    """
    print("\n" + "=" * 80)
    print("COMPARISON: CSV vs Raw API")
    print("=" * 80)

    print("\n┌─────────────────────────┬──────────────────────────┬──────────────────────────┐")
    print("│ Feature                 │ CSV Approach             │ Raw API Approach         │")
    print("├─────────────────────────┼──────────────────────────┼──────────────────────────┤")
    print("│ Package                 │ tardis-dev               │ tardis-client            │")
    print("│ Data Format             │ Normalized CSV           │ Exchange-native JSON     │")
    print("│ Ease of Use             │ ★★★★★ Very Easy         │ ★★★☆☆ Moderate          │")
    print("│ Granularity             │ ★★★★☆ Per-tick          │ ★★★★★ Full tick-level   │")
    print("│ Processing Speed        │ ★★★★★ Fast              │ ★★★☆☆ Slower            │")
    print("│ Flexibility             │ ★★★☆☆ Limited           │ ★★★★★ Full control      │")
    print("│ Options Data            │ options_chain, trades    │ ticker, trades, book     │")
    print("│ Greeks Available        │ ✓ Pre-calculated         │ ✓ Pre-calculated         │")
    print("│ Order Book              │ Snapshots only           │ Full incremental updates │")
    print("│ Best For                │ Analysis, Backtesting    │ Research, Custom work    │")
    print("└─────────────────────────┴──────────────────────────┴──────────────────────────┘")


def main():
    """
    Run all demonstrations.
    """
    print("\n")
    print("╔" + "═" * 78 + "╗")
    print("║" + " " * 20 + "DERIBIT OPTIONS DATA FORMATS" + " " * 30 + "║")
    print("║" + " " * 26 + "Tardis.dev Examples" + " " * 33 + "║")
    print("╚" + "═" * 78 + "╝")

    show_csv_format()
    show_raw_api_format()
    show_symbol_parsing_example()
    show_filtering_example()
    show_comparison_table()

    print("\n" + "=" * 80)
    print("NEXT STEPS")
    print("=" * 80)
    print("""
1. Install dependencies:
   uv pip install -r requirements.txt

2. Try CSV approach:
   uv run python 01_csv_download_approach.py

3. Try Raw API approach:
   uv run python 02_raw_api_replay_approach.py

4. Read the README:
   cat README.md

5. Explore the data:
   - Check downloaded CSV files in ./datasets_deribit_options/
   - Modify scripts to filter for specific options
   - Add your own analysis logic

For questions, see: https://docs.tardis.dev
""")


if __name__ == "__main__":
    main()
