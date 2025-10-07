#!/usr/bin/env python3
"""
Test script for Polymarket Trades endpoint (Data API)
Gets historical trades for users and markets
"""

import requests
import json
from datetime import datetime

def print_section(title):
    """Print a formatted section header"""
    print("\n" + "="*80)
    print(f" {title}")
    print("="*80 + "\n")

def test_trades():
    """Test the /trades endpoint from Data API"""

    base_url = "https://data-api.polymarket.com/trades"

    print_section("TRADES ENDPOINT TEST (Data API)")

    # Test 1: Get recent trades for a specific market
    print("Test 1: Recent trades for a market")
    print("-" * 80)

    # Example: Presidential election market
    market_id = "0xdd31ce870876bb410d71f29e3dfea77910bc3effb81a0fb8dc8c282d5968aa5a"

    params_1 = {
        "market": market_id,
        "limit": 10,
        "offset": 0
    }

    print(f"Request URL: {base_url}")
    print(f"Parameters: {json.dumps(params_1, indent=2)}")

    try:
        response = requests.get(base_url, params=params_1, timeout=10)
        response.raise_for_status()

        trades = response.json()

        print(f"\nStatus Code: {response.status_code}")
        print(f"Number of trades: {len(trades)}")

        if trades:
            print("\nTrade Details:")
            for i, trade in enumerate(trades[:3], 1):
                timestamp = datetime.fromtimestamp(trade['timestamp'] / 1000)  # milliseconds to seconds
                print(f"\n  Trade {i}:")
                print(f"    Side: {trade.get('side')}")
                print(f"    Size: {trade.get('size')}")
                print(f"    Price: {trade.get('price')}")
                print(f"    Timestamp: {timestamp.strftime('%Y-%m-%d %H:%M:%S')}")
                print(f"    Outcome: {trade.get('outcome')}")
                print(f"    Trader: {trade.get('proxyWallet', 'N/A')[:10]}...")
                print(f"    Tx Hash: {trade.get('transactionHash', 'N/A')[:20]}...")

    except requests.exceptions.RequestException as e:
        print(f"\nError making request: {e}")

    # Test 2: Filter by side (BUY/SELL)
    print("\n" + "-"*80)
    print("Test 2: Filter by side (BUY only)")
    print("-" * 80)

    params_2 = {
        "market": market_id,
        "side": "BUY",
        "limit": 5
    }

    print(f"Request URL: {base_url}")
    print(f"Parameters: {json.dumps(params_2, indent=2)}")

    try:
        response = requests.get(base_url, params=params_2, timeout=10)
        response.raise_for_status()

        trades = response.json()

        print(f"\nStatus Code: {response.status_code}")
        print(f"Number of BUY trades: {len(trades)}")

        if trades:
            print("\nBuy Trades:")
            for trade in trades:
                print(f"  Size: {trade.get('size'):.2f} @ Price: {trade.get('price'):.4f} | "
                      f"Outcome: {trade.get('outcome')}")

    except requests.exceptions.RequestException as e:
        print(f"\nError making request: {e}")

    # Test 3: Response structure analysis
    print("\n" + "-"*80)
    print("Test 3: Response Structure Analysis")
    print("-" * 80)

    params_3 = {
        "market": market_id,
        "limit": 1
    }

    try:
        response = requests.get(base_url, params=params_3, timeout=10)
        response.raise_for_status()

        trades = response.json()

        if trades:
            print("\nResponse Structure (single trade object):")
            trade = trades[0]

            print("\nTop-level fields:")
            for key, value in sorted(trade.items()):
                value_str = str(value)
                if len(value_str) > 50:
                    value_str = value_str[:47] + "..."
                print(f"  {key:25} ({type(value).__name__:10}): {value_str}")

            print("\n\nFull JSON Sample:")
            print(json.dumps(trades[0], indent=2))

    except requests.exceptions.RequestException as e:
        print(f"\nError making request: {e}")

    # Test 4: Multiple markets
    print("\n" + "-"*80)
    print("Test 4: Multiple markets (comma-separated)")
    print("-" * 80)

    market_id_2 = "0x5f65177b394277fd294cd75650044e32ba009a95022d88a0c1d565897d72f8f1"

    params_4 = {
        "market": f"{market_id},{market_id_2}",
        "limit": 10
    }

    print(f"Request URL: {base_url}")
    print(f"Parameters: {json.dumps(params_4, indent=2)}")

    try:
        response = requests.get(base_url, params=params_4, timeout=10)
        response.raise_for_status()

        trades = response.json()

        print(f"\nStatus Code: {response.status_code}")
        print(f"Total trades from both markets: {len(trades)}")

        if trades:
            # Group by market
            market_counts = {}
            for trade in trades:
                market = trade.get('conditionId', 'Unknown')
                market_counts[market] = market_counts.get(market, 0) + 1

            print("\nTrades by market:")
            for market, count in market_counts.items():
                print(f"  {market[:20]}...: {count} trades")

    except requests.exceptions.RequestException as e:
        print(f"\nError making request: {e}")

    print_section("TEST COMPLETED")
    print("\nKey Findings:")
    print("  1. Endpoint: https://data-api.polymarket.com/trades")
    print("  2. Response: Array of trade objects")
    print("  3. Timestamps in milliseconds (divide by 1000 for Python datetime)")
    print("  4. Supports filtering by: market, side, user, limit, offset")
    print("  5. Can query multiple markets with comma-separated IDs")
    print("  6. Includes trader address, transaction hash, outcome")
    print("  7. Rate limit: 100 requests / 10 seconds")
    print("  8. Max limit: 10,000 per request")
    print("  9. takerOnly parameter available (default: true)")

if __name__ == "__main__":
    test_trades()
