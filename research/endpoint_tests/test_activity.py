#!/usr/bin/env python3
"""
Test script for Polymarket Activity endpoint
Gets complete user activity timeline including all on-chain actions
"""

import requests
import json
from datetime import datetime, timedelta

def print_section(title):
    print("\n" + "="*80)
    print(f" {title}")
    print("="*80 + "\n")

def test_activity():
    """Test the /activity endpoint"""

    base_url = "https://data-api.polymarket.com/activity"
    user_address = "0x56687bf447db6ffa42ffe2204a05edaa20f55839"

    print_section("ACTIVITY TIMELINE ENDPOINT TEST")

    # Test 1: Get recent activity
    print("Test 1: Recent activity (all types)")
    print("-" * 80)

    params_1 = {
        "user": user_address,
        "limit": 10,
        "sortBy": "TIMESTAMP",
        "sortDirection": "DESC"
    }

    print(f"Request URL: {base_url}")
    print(f"Parameters: {json.dumps(params_1, indent=2)}")

    try:
        response = requests.get(base_url, params=params_1, timeout=10)
        response.raise_for_status()

        activities = response.json()

        print(f"\nStatus Code: {response.status_code}")
        print(f"Activity count: {len(activities)}")

        if activities:
            print("\nRecent Activity:")
            for i, act in enumerate(activities[:5], 1):
                ts = datetime.fromtimestamp(act['timestamp'] / 1000)
                print(f"\n  {i}. Type: {act.get('type')}")
                print(f"     Time: {ts.strftime('%Y-%m-%d %H:%M:%S')}")
                print(f"     Market: {act.get('title', 'N/A')[:40]}")
                print(f"     Size: {act.get('size', 0):.2f}")
                if act.get('type') == 'TRADE':
                    print(f"     Side: {act.get('side')}, Price: ${act.get('price', 0):.4f}")
                print(f"     Tx: {act.get('transactionHash', 'N/A')[:20]}...")

    except requests.exceptions.RequestException as e:
        print(f"\nError: {e}")

    # Test 2: Filter by activity type
    print("\n" + "-"*80)
    print("Test 2: Filter by activity type (TRADE only)")
    print("-" * 80)

    params_2 = {
        "user": user_address,
        "type": "TRADE",
        "limit": 5
    }

    print(f"Parameters: {json.dumps(params_2, indent=2)}")

    try:
        response = requests.get(base_url, params=params_2, timeout=10)
        response.raise_for_status()

        activities = response.json()
        print(f"\nTrade activities: {len(activities)}")

        if activities:
            for act in activities[:3]:
                print(f"  {act.get('side'):4} {act.get('size'):8.2f} @ ${act.get('price'):.4f}")

    except requests.exceptions.RequestException as e:
        print(f"\nError: {e}")

    # Test 3: Activity types
    print("\n" + "-"*80)
    print("Test 3: Activity Type Breakdown")
    print("-" * 80)

    activity_types = {
        "TRADE": "Market trade execution",
        "SPLIT": "Split USDC into outcome tokens",
        "MERGE": "Merge outcome tokens back to USDC",
        "REDEEM": "Redeem winning tokens for USDC",
        "REWARD": "Liquidity/trading rewards received",
        "CONVERSION": "Token conversions"
    }

    print("\nAvailable Activity Types:")
    for type_name, description in activity_types.items():
        print(f"  {type_name:12} - {description}")

    # Test 4: Response structure
    print("\n" + "-"*80)
    print("Test 4: Response Structure")
    print("-" * 80)

    params_4 = {
        "user": user_address,
        "limit": 1
    }

    try:
        response = requests.get(base_url, params=params_4, timeout=10)
        response.raise_for_status()

        activities = response.json()

        if activities:
            print("\nActivity Object Structure:")
            act = activities[0]

            print(json.dumps(act, indent=2))

            print("\n\nField Descriptions:")
            print("  proxyWallet    - User's proxy wallet address")
            print("  timestamp      - Unix timestamp in milliseconds")
            print("  conditionId    - Market identifier")
            print("  type           - Activity type (TRADE, SPLIT, etc.)")
            print("  size           - Size in shares/tokens")
            print("  usdcSize       - Value in USDC")
            print("  transactionHash- On-chain transaction hash")
            print("  price          - Execution price (for TRADE)")
            print("  asset          - Token ID")
            print("  side           - BUY/SELL (for TRADE)")
            print("  outcomeIndex   - Outcome index (0 or 1)")
            print("  title          - Market title")
            print("  outcome        - Outcome name")

    except requests.exceptions.RequestException as e:
        print(f"\nError: {e}")

    print_section("TEST COMPLETED")
    print("\nKey Findings:")
    print("  1. Endpoint: https://data-api.polymarket.com/activity")
    print("  2. Complete activity timeline with all on-chain actions")
    print("  3. Activity types: TRADE, SPLIT, MERGE, REDEEM, REWARD, CONVERSION")
    print("  4. Timestamps in milliseconds")
    print("  5. Includes transaction hashes for verification")
    print("  6. Can filter by: type, market, eventId, start/end timestamps")
    print("  7. Sort by: TIMESTAMP, TOKENS, CASH")
    print("  8. Rate limit: 100 req / 10s")
    print("  9. Perfect for portfolio reconstruction and P&L analysis")

if __name__ == "__main__":
    test_activity()
