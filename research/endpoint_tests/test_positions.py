#!/usr/bin/env python3
"""
Test script for Polymarket Positions endpoint
Gets current positions for a user with P&L tracking
"""

import requests
import json
from datetime import datetime

def print_section(title):
    """Print a formatted section header"""
    print("\n" + "="*80)
    print(f" {title}")
    print("="*80 + "\n")

def test_positions():
    """Test the /positions endpoint from Data API"""

    base_url = "https://data-api.polymarket.com/positions"

    print_section("POSITIONS ENDPOINT TEST")

    # Example user address (well-known trader)
    user_address = "0x56687bf447db6ffa42ffe2204a05edaa20f55839"

    # Test 1: Get user's current positions
    print("Test 1: Get current positions for a user")
    print("-" * 80)

    params_1 = {
        "user": user_address,
        "limit": 5,
        "sortBy": "TOKENS",
        "sortDirection": "DESC"
    }

    print(f"Request URL: {base_url}")
    print(f"Parameters: {json.dumps(params_1, indent=2)}")

    try:
        response = requests.get(base_url, params=params_1, timeout=10)
        response.raise_for_status()

        positions = response.json()

        print(f"\nStatus Code: {response.status_code}")
        print(f"Number of positions: {len(positions)}")

        if positions:
            print("\nPosition Details:")
            for i, pos in enumerate(positions[:3], 1):
                print(f"\n  Position {i}:")
                print(f"    Market: {pos.get('title', 'N/A')}")
                print(f"    Outcome: {pos.get('outcome', 'N/A')}")
                print(f"    Size: {pos.get('size', 0):.2f} shares")
                print(f"    Avg Entry Price: ${pos.get('avgPrice', 0):.4f}")
                print(f"    Current Price: ${pos.get('curPrice', 0):.4f}")
                print(f"    Initial Value: ${pos.get('initialValue', 0):.2f}")
                print(f"    Current Value: ${pos.get('currentValue', 0):.2f}")
                print(f"    Unrealized P&L: ${pos.get('cashPnl', 0):.2f} ({pos.get('percentPnl', 0):.2f}%)")
                print(f"    Realized P&L: ${pos.get('realizedPnl', 0):.2f}")
                print(f"    Redeemable: {pos.get('redeemable', False)}")
                print(f"    Mergeable: {pos.get('mergeable', False)}")

    except requests.exceptions.RequestException as e:
        print(f"\nError making request: {e}")
        print("Note: User address may not have public positions or may not exist")

    # Test 2: Filter with size threshold
    print("\n" + "-"*80)
    print("Test 2: Filter by minimum position size")
    print("-" * 80)

    params_2 = {
        "user": user_address,
        "sizeThreshold": 100,  # Only positions with 100+ shares
        "limit": 10
    }

    print(f"Request URL: {base_url}")
    print(f"Parameters: {json.dumps(params_2, indent=2)}")

    try:
        response = requests.get(base_url, params=params_2, timeout=10)
        response.raise_for_status()

        positions = response.json()

        print(f"\nStatus Code: {response.status_code}")
        print(f"Positions with size >= 100: {len(positions)}")

        if positions:
            total_value = sum(pos.get('currentValue', 0) for pos in positions)
            total_pnl = sum(pos.get('cashPnl', 0) for pos in positions)

            print(f"\nPortfolio Summary:")
            print(f"  Total Current Value: ${total_value:.2f}")
            print(f"  Total Unrealized P&L: ${total_pnl:.2f}")

    except requests.exceptions.RequestException as e:
        print(f"\nError making request: {e}")

    # Test 3: Sort by P&L
    print("\n" + "-"*80)
    print("Test 3: Sort by Cash P&L (winners and losers)")
    print("-" * 80)

    params_3 = {
        "user": user_address,
        "sortBy": "CASHPNL",
        "sortDirection": "DESC",
        "limit": 3
    }

    print(f"Request URL: {base_url}")
    print(f"Parameters: {json.dumps(params_3, indent=2)}")

    try:
        response = requests.get(base_url, params=params_3, timeout=10)
        response.raise_for_status()

        positions = response.json()

        print(f"\nTop Winners:")
        for i, pos in enumerate(positions, 1):
            print(f"  {i}. {pos.get('title', 'N/A')[:50]}")
            print(f"     P&L: ${pos.get('cashPnl', 0):.2f} ({pos.get('percentPnl', 0):.2f}%)")

    except requests.exceptions.RequestException as e:
        print(f"\nError making request: {e}")

    # Test 4: Response structure
    print("\n" + "-"*80)
    print("Test 4: Response Structure Analysis")
    print("-" * 80)

    params_4 = {
        "user": user_address,
        "limit": 1
    }

    try:
        response = requests.get(base_url, params=params_4, timeout=10)
        response.raise_for_status()

        positions = response.json()

        if positions:
            print("\nPosition Object Structure:")
            pos = positions[0]

            print("\nFinancial Fields:")
            financial_fields = ['size', 'avgPrice', 'initialValue', 'currentValue',
                              'cashPnl', 'percentPnl', 'totalBought', 'realizedPnl',
                              'percentRealizedPnl', 'curPrice']
            for field in financial_fields:
                if field in pos:
                    print(f"  {field:25} ({type(pos[field]).__name__:8}): {pos[field]}")

            print("\nMarket Information:")
            market_fields = ['title', 'slug', 'outcome', 'outcomeIndex',
                           'oppositeOutcome', 'endDate']
            for field in market_fields:
                if field in pos:
                    value_str = str(pos[field])
                    if len(value_str) > 50:
                        value_str = value_str[:47] + "..."
                    print(f"  {field:25} ({type(pos[field]).__name__:8}): {value_str}")

            print("\nIdentifiers:")
            id_fields = ['proxyWallet', 'asset', 'conditionId']
            for field in id_fields:
                if field in pos:
                    print(f"  {field:25}: {str(pos[field])[:50]}...")

            print("\nStatus Flags:")
            flag_fields = ['redeemable', 'mergeable', 'negativeRisk']
            for field in flag_fields:
                if field in pos:
                    print(f"  {field:25}: {pos[field]}")

            print("\n\nFull JSON Sample:")
            print(json.dumps(positions[0], indent=2))

    except requests.exceptions.RequestException as e:
        print(f"\nError making request: {e}")

    print_section("TEST COMPLETED")
    print("\nKey Findings:")
    print("  1. Endpoint: https://data-api.polymarket.com/positions")
    print("  2. Required parameter: user (address)")
    print("  3. Response: Array of position objects with full P&L tracking")
    print("  4. Financial metrics included:")
    print("     - size: number of shares held")
    print("     - avgPrice: average entry price")
    print("     - curPrice: current market price")
    print("     - initialValue: cost basis")
    print("     - currentValue: current market value")
    print("     - cashPnl: unrealized profit/loss in USD")
    print("     - percentPnl: unrealized P&L as percentage")
    print("     - realizedPnl: locked-in profit/loss from partial exits")
    print("     - totalBought: total shares ever purchased (including sold)")
    print("  5. Can filter by: market, eventId, sizeThreshold, redeemable, mergeable")
    print("  6. Sort options: CURRENT, INITIAL, TOKENS, CASHPNL, PERCENTPNL, etc.")
    print("  7. Rate limit: 100 requests / 10 seconds")
    print("  8. Max limit: 500 per request, max offset: 10,000")

if __name__ == "__main__":
    test_positions()
