#!/usr/bin/env python3
"""
Test parameters for the leaderboard endpoint

Now that we found https://data-api.polymarket.com/leaderboard works,
let's figure out what parameters it accepts.
"""

import requests
import json

BASE_URL = "https://data-api.polymarket.com/leaderboard"

def test_params(params, description):
    print(f"\n{'='*80}")
    print(f"Test: {description}")
    print(f"Params: {json.dumps(params, indent=2)}")
    print(f"{'='*80}")

    response = requests.get(BASE_URL, params=params, timeout=10)
    print(f"Status: {response.status_code}")

    if response.status_code == 200:
        data = response.json()
        print(f"Results returned: {len(data)}")

        if data:
            print("\nFirst entry:")
            print(json.dumps(data[0], indent=2))

            print("\nLast entry:")
            print(json.dumps(data[-1], indent=2))

            # Check if volume field exists
            has_vol = any('vol' in entry for entry in data)
            has_pnl = any('pnl' in entry for entry in data)
            print(f"\nHas 'vol' field: {has_vol}")
            print(f"Has 'pnl' field: {has_pnl}")

        return True, data
    else:
        print(f"Failed: {response.text[:200]}")
        return False, None

def main():
    print("="*80)
    print(" TESTING LEADERBOARD ENDPOINT PARAMETERS")
    print("="*80)

    # Test 1: Default (no params)
    print("\n\nTEST 1: Default (no parameters)")
    test_params({}, "Default call")

    # Test 2: Limit parameter
    print("\n\nTEST 2: Limit parameter")
    test_params({"limit": 10}, "Top 10 only")
    test_params({"limit": 50}, "Top 50")
    test_params({"limit": 100}, "Top 100")

    # Test 3: Offset parameter
    print("\n\nTEST 3: Offset/pagination")
    test_params({"limit": 5, "offset": 0}, "First 5")
    test_params({"limit": 5, "offset": 5}, "Next 5 (offset 5)")

    # Test 4: Order/sort parameters
    print("\n\nTEST 4: Ordering parameters")
    test_params({"order": "vol"}, "Order by volume")
    test_params({"order": "pnl"}, "Order by PnL")
    test_params({"sortBy": "vol"}, "Sort by volume")
    test_params({"sortBy": "pnl"}, "Sort by PnL")

    # Test 5: Time period filters
    print("\n\nTEST 5: Time period filters")
    test_params({"period": "24h"}, "Last 24 hours")
    test_params({"period": "7d"}, "Last 7 days")
    test_params({"period": "30d"}, "Last 30 days")
    test_params({"period": "all"}, "All time")
    test_params({"timeframe": "day"}, "Daily timeframe")
    test_params({"timeframe": "week"}, "Weekly timeframe")

    # Test 6: Metric type
    print("\n\nTEST 6: Metric type")
    test_params({"metric": "volume"}, "Volume metric")
    test_params({"metric": "pnl"}, "PnL metric")
    test_params({"type": "volume"}, "Type volume")
    test_params({"type": "pnl"}, "Type PnL")

    # Test 7: Combined parameters
    print("\n\nTEST 7: Combined parameters")
    test_params({"limit": 10, "order": "vol"}, "Top 10 by volume")
    test_params({"limit": 10, "order": "pnl"}, "Top 10 by PnL")

    print("\n\n" + "="*80)
    print(" TESTING COMPLETE")
    print("="*80)

if __name__ == "__main__":
    main()
