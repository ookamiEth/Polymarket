#!/usr/bin/env python3
"""
Test script for Polymarket Closed Positions endpoint
Gets historical closed positions with realized P&L
"""

import requests
import json
from datetime import datetime

def print_section(title):
    print("\n" + "="*80)
    print(f" {title}")
    print("="*80 + "\n")

def test_closed_positions():
    """Test the /closed-positions endpoint"""

    base_url = "https://data-api.polymarket.com/closed-positions"
    user_address = "0x56687bf447db6ffa42ffe2204a05edaa20f55839"

    print_section("CLOSED POSITIONS ENDPOINT TEST")

    # Test 1: Get closed positions
    print("Test 1: Get closed positions (sorted by realized P&L)")
    print("-" * 80)

    params = {
        "user": user_address,
        "limit": 10,
        "sortBy": "REALIZEDPNL",
        "sortDirection": "DESC"
    }

    print(f"Request URL: {base_url}")
    print(f"Parameters: {json.dumps(params, indent=2)}")

    try:
        response = requests.get(base_url, params=params, timeout=10)
        response.raise_for_status()

        positions = response.json()

        print(f"\nStatus Code: {response.status_code}")
        print(f"Closed positions: {len(positions)}")

        if positions:
            print("\nClosed Positions (Top Winners):")
            for i, pos in enumerate(positions[:5], 1):
                print(f"\n  {i}. {pos.get('title', 'N/A')[:50]}")
                print(f"     Outcome: {pos.get('outcome', 'N/A')}")
                print(f"     Avg Entry: ${pos.get('avgPrice', 0):.4f}")
                print(f"     Exit Price: ${pos.get('curPrice', 0):.4f}")
                print(f"     Shares Traded: {pos.get('totalBought', 0):.2f}")
                print(f"     Realized P&L: ${pos.get('realizedPnl', 0):.2f}")

            total_pnl = sum(pos.get('realizedPnl', 0) for pos in positions)
            print(f"\n  Total Realized P&L (top {len(positions)}): ${total_pnl:.2f}")

        print("\n\nStructure of Closed Position:")
        if positions:
            print(json.dumps(positions[0], indent=2))

    except requests.exceptions.RequestException as e:
        print(f"\nError: {e}")

    print_section("TEST COMPLETED")
    print("\nKey Findings:")
    print("  1. Endpoint: https://data-api.polymarket.com/closed-positions")
    print("  2. Shows positions that have been fully exited")
    print("  3. Includes realized P&L (actual profit/loss)")
    print("  4. Fields: avgPrice, totalBought, realizedPnl, curPrice (final settlement)")
    print("  5. Sort by: REALIZEDPNL, TITLE, PRICE, AVGPRICE")
    print("  6. Rate limit: 100 req / 10s, Max: 500 items, Offset: 10,000")

if __name__ == "__main__":
    test_closed_positions()
