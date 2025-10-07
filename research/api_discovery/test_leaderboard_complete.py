#!/usr/bin/env python3
"""
Comprehensive testing of the leaderboard endpoint to determine:
1. What time periods are supported
2. What parameters work
3. What data is actually returned
4. Maximum data available
"""

import requests
import json
import time

BASE_URL = "https://data-api.polymarket.com/leaderboard"

def test_endpoint(params, description):
    """Test endpoint with given parameters"""
    print(f"\n{'='*80}")
    print(f"TEST: {description}")
    print(f"Params: {json.dumps(params, indent=2)}")
    print(f"{'='*80}")

    try:
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

                # Check for any time-related fields
                first_item = data[0]
                time_fields = [k for k in first_item.keys() if 'time' in k.lower() or 'date' in k.lower() or 'period' in k.lower()]
                if time_fields:
                    print(f"\n⚠ Found time-related fields: {time_fields}")

                return True, data
            else:
                print("⚠ Empty response")
                return True, []
        else:
            print(f"✗ FAILED - {response.status_code}")
            print(f"Response: {response.text[:500]}")
            return False, None

    except Exception as e:
        print(f"✗ ERROR: {e}")
        return False, None

def main():
    print("="*80)
    print(" COMPREHENSIVE LEADERBOARD ENDPOINT TEST")
    print("="*80)

    results = {}

    # TEST 1: Time period parameters
    print("\n" + "="*80)
    print(" SECTION 1: TIME PERIOD PARAMETERS")
    print("="*80)

    time_params = [
        ({}, "Default (no params)"),
        ({"period": "1d"}, "Period: 1 day"),
        ({"period": "24h"}, "Period: 24 hours"),
        ({"period": "7d"}, "Period: 7 days"),
        ({"period": "1w"}, "Period: 1 week"),
        ({"period": "30d"}, "Period: 30 days"),
        ({"period": "1m"}, "Period: 1 month"),
        ({"period": "all"}, "Period: all time"),
        ({"period": "alltime"}, "Period: all time (alt)"),
        ({"timeframe": "day"}, "Timeframe: day"),
        ({"timeframe": "week"}, "Timeframe: week"),
        ({"timeframe": "month"}, "Timeframe: month"),
        ({"timeframe": "all"}, "Timeframe: all"),
        ({"interval": "1d"}, "Interval: 1 day"),
        ({"interval": "7d"}, "Interval: 7 days"),
        ({"interval": "30d"}, "Interval: 30 days"),
    ]

    for params, desc in time_params:
        success, data = test_endpoint(params, desc)
        results[desc] = {"success": success, "count": len(data) if data else 0}
        time.sleep(0.2)

    # TEST 2: Pagination and limits
    print("\n" + "="*80)
    print(" SECTION 2: PAGINATION & LIMITS")
    print("="*80)

    pagination_tests = [
        ({"limit": 10}, "Limit: 10"),
        ({"limit": 25}, "Limit: 25"),
        ({"limit": 50}, "Limit: 50"),
        ({"limit": 100}, "Limit: 100"),
        ({"limit": 200}, "Limit: 200"),
        ({"limit": 500}, "Limit: 500"),
        ({"offset": 0, "limit": 10}, "Offset: 0, Limit: 10"),
        ({"offset": 10, "limit": 10}, "Offset: 10, Limit: 10"),
        ({"offset": 50, "limit": 50}, "Offset: 50, Limit: 50"),
        ({"offset": 100, "limit": 50}, "Offset: 100, Limit: 50"),
    ]

    for params, desc in pagination_tests:
        success, data = test_endpoint(params, desc)
        results[desc] = {"success": success, "count": len(data) if data else 0}
        time.sleep(0.2)

    # TEST 3: Sorting and ordering
    print("\n" + "="*80)
    print(" SECTION 3: SORTING & ORDERING")
    print("="*80)

    sort_tests = [
        ({"sort": "volume"}, "Sort: volume"),
        ({"sort": "pnl"}, "Sort: pnl"),
        ({"sort": "vol"}, "Sort: vol"),
        ({"sortBy": "volume"}, "SortBy: volume"),
        ({"sortBy": "pnl"}, "SortBy: pnl"),
        ({"sortBy": "vol"}, "SortBy: vol"),
        ({"order": "volume"}, "Order: volume"),
        ({"order": "pnl"}, "Order: pnl"),
        ({"orderBy": "volume"}, "OrderBy: volume"),
        ({"orderBy": "pnl"}, "OrderBy: pnl"),
        ({"ascending": "true"}, "Ascending: true"),
        ({"ascending": "false"}, "Ascending: false"),
        ({"desc": "true"}, "Desc: true"),
        ({"asc": "true"}, "Asc: true"),
    ]

    for params, desc in sort_tests:
        success, data = test_endpoint(params, desc)
        results[desc] = {"success": success, "count": len(data) if data else 0}
        time.sleep(0.2)

    # TEST 4: Field selection
    print("\n" + "="*80)
    print(" SECTION 4: FIELD SELECTION")
    print("="*80)

    field_tests = [
        ({"fields": "rank,user_id,vol"}, "Fields: rank,user_id,vol"),
        ({"select": "rank,user_id,vol"}, "Select: rank,user_id,vol"),
        ({"columns": "rank,user_id,vol"}, "Columns: rank,user_id,vol"),
    ]

    for params, desc in field_tests:
        success, data = test_endpoint(params, desc)
        results[desc] = {"success": success, "count": len(data) if data else 0}
        time.sleep(0.2)

    # TEST 5: Combination tests
    print("\n" + "="*80)
    print(" SECTION 5: COMBINATION TESTS")
    print("="*80)

    combo_tests = [
        ({"limit": 10, "period": "7d"}, "Limit 10, Period 7d"),
        ({"limit": 10, "period": "30d"}, "Limit 10, Period 30d"),
        ({"limit": 10, "sort": "volume"}, "Limit 10, Sort volume"),
        ({"limit": 10, "period": "7d", "sort": "volume"}, "Limit 10, Period 7d, Sort volume"),
    ]

    for params, desc in combo_tests:
        success, data = test_endpoint(params, desc)
        results[desc] = {"success": success, "count": len(data) if data else 0}
        time.sleep(0.2)

    # SUMMARY
    print("\n" + "="*80)
    print(" TEST SUMMARY")
    print("="*80)

    print("\nSuccessful tests that returned data:")
    for desc, result in results.items():
        if result["success"] and result["count"] > 0:
            print(f"  ✓ {desc}: {result['count']} results")

    print("\nFailed or empty tests:")
    for desc, result in results.items():
        if not result["success"] or result["count"] == 0:
            print(f"  ✗ {desc}")

    # Save results
    output_file = "/Users/lgierhake/Documents/ETH/BT/top_traders/api_research/leaderboard_complete_test_results.json"
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n✓ Full results saved to: {output_file}")

    # FINAL ANALYSIS
    print("\n" + "="*80)
    print(" ANALYSIS & CONCLUSIONS")
    print("="*80)

    # Check if any time period params worked
    time_period_works = any(
        results.get(desc, {}).get("success") and results.get(desc, {}).get("count", 0) > 0
        for params, desc in time_params
        if params  # exclude default
    )

    if time_period_works:
        print("✓ Time period parameters ARE supported")
    else:
        print("✗ Time period parameters DO NOT appear to work")
        print("  → Endpoint likely returns ALL-TIME data only")

    # Check max limit
    max_count = max(r.get("count", 0) for r in results.values())
    print(f"\n📊 Maximum results returned: {max_count}")

    # Check if sorting works
    print("\n🔍 Recommendation:")
    print("  Run additional test comparing vol values across different sort params")
    print("  to verify if sorting actually changes the order.")

if __name__ == "__main__":
    main()
