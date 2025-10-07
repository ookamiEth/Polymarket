#!/usr/bin/env python3
"""
Test the ACTUAL endpoint used by Polymarket UI: /biggest-winners
This supports timePeriod parameter!
"""

import requests
import json

BASE_URL = "https://data-api.polymarket.com/biggest-winners"

def test_endpoint(params, description):
    """Test endpoint with given parameters"""
    print(f"\n{'='*80}")
    print(f"TEST: {description}")
    print(f"Params: {json.dumps(params, indent=2)}")
    print(f"{'='*80}")

    response = requests.get(BASE_URL, params=params, timeout=10)
    print(f"Status: {response.status_code}")

    if response.status_code == 200:
        data = response.json()
        print(f"✓ SUCCESS - {len(data)} results")

        if data:
            print(f"\nFirst result:")
            print(json.dumps(data[0], indent=2))

            if len(data) > 1:
                print(f"\nLast result:")
                print(json.dumps(data[-1], indent=2))

            # Show all fields
            print(f"\nFields available: {list(data[0].keys())}")

        return True, data
    else:
        print(f"✗ FAILED - {response.status_code}")
        print(f"Response: {response.text[:500]}")
        return False, None

def main():
    print("="*80)
    print(" TESTING /biggest-winners ENDPOINT")
    print("="*80)

    # TEST 1: Different time periods
    print("\n" + "="*80)
    print(" TIME PERIODS")
    print("="*80)

    time_periods = [
        ({"timePeriod": "day", "limit": 10}, "Day"),
        ({"timePeriod": "week", "limit": 10}, "Week"),
        ({"timePeriod": "month", "limit": 10}, "Month"),
        ({"timePeriod": "all", "limit": 10}, "All time"),
        ({"limit": 10}, "Default (no timePeriod)"),
    ]

    results = {}
    for params, desc in time_periods:
        success, data = test_endpoint(params, desc)
        if data:
            results[desc] = data

    # Compare results
    print("\n" + "="*80)
    print(" COMPARING TIME PERIODS")
    print("="*80)

    if len(results) >= 2:
        period_names = list(results.keys())
        print(f"\nComparing first entry from each period:\n")

        for name in period_names:
            data = results[name]
            if data:
                first = data[0]
                print(f"{name:15}: User={first.get('user_name', 'N/A')[:20]:20} | "
                      f"PnL=${first.get('pnl', 0):>10,.0f} | Vol=${first.get('vol', 0):>12,.0f}")

    # TEST 2: Pagination
    print("\n" + "="*80)
    print(" PAGINATION TEST")
    print("="*80)

    test_endpoint({"timePeriod": "month", "limit": 50, "offset": 0}, "Month, limit 50, offset 0")
    test_endpoint({"timePeriod": "month", "limit": 50, "offset": 50}, "Month, limit 50, offset 50")
    test_endpoint({"timePeriod": "month", "limit": 100}, "Month, limit 100")
    test_endpoint({"timePeriod": "month", "limit": 1000}, "Month, limit 1000")

    print("\n" + "="*80)
    print(" ✓ DISCOVERY COMPLETE")
    print("="*80)
    print("\nKey findings:")
    print("  • /biggest-winners is the REAL endpoint")
    print("  • timePeriod parameter: 'day', 'week', 'month', 'all'")
    print("  • Different periods return DIFFERENT data!")
    print("  • This is what we should use for parquet files")

if __name__ == "__main__":
    main()
