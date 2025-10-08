#!/usr/bin/env python3
"""
Deribit Options Data - Raw API Replay Approach

This script demonstrates how to replay Deribit options data using the tardis-client package.
The raw API approach returns exchange-native format (actual WebSocket messages from Deribit).

Date: October 1, 2025 (free access - first day of month)
Package: tardis-client (local copy in /research/tardis-python)

Available channels for options:
- ticker: General ticker including Greeks, IV, bid/ask, volume, OI
- quote: Best bid/ask quotes
- trades: Executed trades
- book: Full order book updates
- markprice.options: Mark prices for options
- estimated_expiration_price: Settlement price estimates
- deribit_volatility_index: Volatility index
"""

import asyncio
import sys
import json
from datetime import datetime

# Add the local tardis-client to path
sys.path.insert(0, '/Users/lgierhake/Documents/ETH/BT/research/tardis-python')

from tardis_client import TardisClient, Channel


async def replay_deribit_options_ticker(
    from_date="2025-10-01",
    to_date="2025-10-02",
    symbols=None,
    max_messages=50,
    api_key=None
):
    """
    Replay Deribit options ticker data.

    The ticker channel provides comprehensive option data including:
    - Current prices (mark_price, last_price, best_bid, best_ask)
    - Greeks (delta, gamma, theta, vega, rho)
    - Implied volatility (IV)
    - Open interest
    - Volume (24h)
    - Underlying price and index

    Args:
        from_date: Start date (ISO format)
        to_date: End date (ISO format)
        symbols: List of option symbols (None = all options)
        max_messages: Maximum messages to display
        api_key: Tardis.dev API key (None for free access)
    """

    print("=" * 80)
    print("DERIBIT OPTIONS - RAW API REPLAY (Ticker Channel)")
    print("=" * 80)
    print(f"Date range: {from_date} to {to_date}")
    print(f"Channel: ticker")
    print(f"Symbols: {symbols if symbols else 'ALL OPTIONS (filtered below)'}")
    print(f"Access: {'Free (first day of month)' if not api_key else 'API key'}")
    print("-" * 80)

    # Initialize client
    tardis_client = TardisClient(api_key=api_key)

    # If no specific symbols, we'll filter for BTC options in the message handler
    if symbols is None:
        # Get a few BTC option symbols for demonstration
        # In practice, you'd query available symbols first
        symbols = ["BTC-PERPETUAL"]  # Start with perpetual to show format
        print("Note: Starting with BTC-PERPETUAL to demonstrate format")
        print("For actual options, use symbols like: BTC-29NOV25-50000-C")

    # Create filter for ticker channel
    filters = [Channel(name="ticker", symbols=symbols)]

    # Replay messages
    messages = tardis_client.replay(
        exchange="deribit",
        from_date=from_date,
        to_date=to_date,
        filters=filters
    )

    message_count = 0
    option_message_count = 0

    print("\nReplaying messages...\n")

    try:
        async for local_timestamp, message in messages:
            message_count += 1

            # Check if this is an option (contains strike/expiry in symbol)
            # Options format: BTC-29NOV25-50000-C or ETH-29NOV25-2000-P
            symbol = message.get('params', {}).get('data', {}).get('instrument_name', '')

            is_option = '-' in symbol and symbol.count('-') >= 3

            if is_option or message_count <= 10:  # Show first 10 messages regardless
                option_message_count += 1

                print(f"Message #{message_count} | Options: #{option_message_count}")
                print(f"Local timestamp: {local_timestamp}")
                print(f"Symbol: {symbol}")
                print(f"Is Option: {is_option}")
                print("\nMessage structure:")
                print(json.dumps(message, indent=2))
                print("-" * 80)

                if option_message_count >= max_messages:
                    print(f"\nReached maximum of {max_messages} option messages.")
                    break

    except Exception as e:
        print(f"\nError during replay: {e}")
        import traceback
        traceback.print_exc()

    print(f"\nTotal messages processed: {message_count}")
    print(f"Option messages found: {option_message_count}")


async def replay_deribit_options_trades(
    from_date="2025-10-01",
    to_date="2025-10-02",
    symbols=None,
    max_messages=20,
    api_key=None
):
    """
    Replay Deribit options trade data.

    The trades channel provides executed trades with:
    - Trade price and amount
    - Direction (buy/sell)
    - Liquidation flag
    - IV (for options)
    - Timestamp
    """

    print("\n" + "=" * 80)
    print("DERIBIT OPTIONS - RAW API REPLAY (Trades Channel)")
    print("=" * 80)
    print(f"Date range: {from_date} to {to_date}")
    print(f"Channel: trades")
    print("-" * 80)

    tardis_client = TardisClient(api_key=api_key)

    if symbols is None:
        symbols = ["BTC-PERPETUAL"]

    filters = [Channel(name="trades", symbols=symbols)]

    messages = tardis_client.replay(
        exchange="deribit",
        from_date=from_date,
        to_date=to_date,
        filters=filters
    )

    message_count = 0

    print("\nReplaying trade messages...\n")

    try:
        async for local_timestamp, message in messages:
            message_count += 1

            print(f"Trade Message #{message_count}")
            print(f"Local timestamp: {local_timestamp}")
            print(json.dumps(message, indent=2))
            print("-" * 80)

            if message_count >= max_messages:
                print(f"\nReached maximum of {max_messages} messages.")
                break

    except Exception as e:
        print(f"\nError during replay: {e}")
        import traceback
        traceback.print_exc()

    print(f"\nTotal trade messages: {message_count}")


async def show_message_format_guide():
    """
    Display a guide to understanding Deribit's message formats.
    """

    print("\n" + "=" * 80)
    print("DERIBIT MESSAGE FORMAT GUIDE")
    print("=" * 80)

    ticker_format = {
        "jsonrpc": "2.0",
        "method": "subscription",
        "params": {
            "channel": "ticker.{INSTRUMENT}.raw",
            "data": {
                "instrument_name": "BTC-29NOV25-50000-C",
                "underlying_price": 50000.0,
                "underlying_index": "BTC-USD",
                "timestamp": 1633046400000,
                "stats": {
                    "volume": 123.5,
                    "low": 0.05,
                    "high": 0.08
                },
                "state": "open",
                "settlement_price": 0.065,
                "open_interest": 1000,
                "min_price": 0.04,
                "max_price": 0.10,
                "mark_price": 0.067,
                "mark_iv": 65.5,
                "last_price": 0.066,
                "interest_rate": 0.0,
                "index_price": 50000.0,
                "greeks": {
                    "vega": 45.2,
                    "theta": -12.5,
                    "rho": 3.2,
                    "gamma": 0.00002,
                    "delta": 0.55
                },
                "estimated_delivery_price": 50000.0,
                "bid_iv": 64.0,
                "best_bid_price": 0.065,
                "best_bid_amount": 10.0,
                "best_ask_price": 0.067,
                "best_ask_amount": 15.0,
                "ask_iv": 67.0
            }
        }
    }

    print("\nTICKER MESSAGE FORMAT (Options):")
    print(json.dumps(ticker_format, indent=2))

    trade_format = {
        "jsonrpc": "2.0",
        "method": "subscription",
        "params": {
            "channel": "trades.{INSTRUMENT}.raw",
            "data": [
                {
                    "trade_seq": 12345,
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
                }
            ]
        }
    }

    print("\n\nTRADE MESSAGE FORMAT (Options):")
    print(json.dumps(trade_format, indent=2))

    print("\n" + "=" * 80)


async def main():
    """
    Main function to demonstrate both approaches.
    """

    # Show message format guide first
    await show_message_format_guide()

    # Example 1: Replay ticker data
    print("\n\n")
    await replay_deribit_options_ticker(
        from_date="2025-10-01",
        to_date="2025-10-01 00:05:00",  # Just first 5 minutes
        symbols=None,  # Will filter for options in handler
        max_messages=10,
        api_key=None
    )

    # Example 2: Replay trade data
    # Uncomment to run:
    # await replay_deribit_options_trades(
    #     from_date="2025-10-01",
    #     to_date="2025-10-01 00:05:00",
    #     max_messages=10,
    #     api_key=None
    # )

    print("\n" + "=" * 80)
    print("REPLAY COMPLETE!")
    print("=" * 80)
    print("\nKey takeaways:")
    print("1. Raw API returns exchange-native JSON format")
    print("2. Data structure matches Deribit's WebSocket API exactly")
    print("3. For options: ticker channel contains Greeks, IV, and pricing")
    print("4. Symbol format: {BASE}-{EXPIRY}-{STRIKE}-{TYPE}")
    print("   Example: BTC-29NOV25-50000-C (Call option)")
    print("            ETH-29NOV25-2000-P (Put option)")


if __name__ == "__main__":
    asyncio.run(main())
