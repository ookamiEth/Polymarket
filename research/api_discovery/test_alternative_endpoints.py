#!/usr/bin/env python3
"""
Test alternative endpoint patterns based on UI behavior

The UI shows Today/Weekly/Monthly/All tabs and Volume/P&L sorting.
Let's try to find the real endpoints.
"""

import requests
import json

def test_endpoint(url, params=None, description=""):
    """Test an endpoint and return results"""
    print(f"\n{'='*80}")
    print(f"Testing: {description}")
    print(f"URL: {url}")
    if params:
        print(f"Params: {json.dumps(params, indent=2)}")
    print(f"{'='*80}")

    try:
        response = requests.get(url, params=params, timeout=10)
        print(f"Status: {response.status_code}")

        if response.status_code == 200:
            data = response.json()
            print(f"✓ SUCCESS - {len(data)} results" if isinstance(data, list) else "✓ SUCCESS")

            if isinstance(data, list) and len(data) > 0:
                print(f"\nFirst entry:")
                print(json.dumps(data[0], indent=2))

                # Check for differences from the "all" endpoint
                has_period_field = any('period' in str(k).lower() for k in data[0].keys())
                has_timeframe = any('time' in str(k).lower() for k in data[0].keys())

                if has_period_field or has_timeframe:
                    print(f"\n⚠ IMPORTANT: Found time-related fields!")

                return True, data
            elif isinstance(data, dict):
                print(f"\nResponse structure:")
                print(json.dumps(data, indent=2)[:500])
                return True, data

        else:
            print(f"✗ Failed: {response.status_code}")
            print(f"Response: {response.text[:200]}")

    except Exception as e:
        print(f"✗ Error: {e}")

    return False, None

def main():
    print("="*80)
    print(" ALTERNATIVE ENDPOINT DISCOVERY")
    print(" Based on UI showing: Today | Weekly | Monthly | All")
    print("="*80)

    # Test 1: Path-based time periods
    print("\n\n" + "="*80)
    print(" SECTION 1: PATH-BASED TIME PERIODS")
    print("="*80)

    path_tests = [
        ("https://data-api.polymarket.com/leaderboard/today", {}, "Path: /leaderboard/today"),
        ("https://data-api.polymarket.com/leaderboard/weekly", {}, "Path: /leaderboard/weekly"),
        ("https://data-api.polymarket.com/leaderboard/monthly", {}, "Path: /leaderboard/monthly"),
        ("https://data-api.polymarket.com/leaderboard/all", {}, "Path: /leaderboard/all"),
        ("https://data-api.polymarket.com/leaderboard/daily", {}, "Path: /leaderboard/daily"),
        ("https://gamma-api.polymarket.com/leaderboard/today", {}, "Gamma: /leaderboard/today"),
        ("https://gamma-api.polymarket.com/leaderboard/weekly", {}, "Gamma: /leaderboard/weekly"),
    ]

    for url, params, desc in path_tests:
        test_endpoint(url, params, desc)

    # Test 2: Resource-based patterns
    print("\n\n" + "="*80)
    print(" SECTION 2: RESOURCE-BASED PATTERNS")
    print("="*80)

    resource_tests = [
        ("https://data-api.polymarket.com/leaderboards/today", {}, "/leaderboards/today"),
        ("https://data-api.polymarket.com/leaderboards/weekly", {}, "/leaderboards/weekly"),
        ("https://data-api.polymarket.com/rankings/today", {}, "/rankings/today"),
        ("https://data-api.polymarket.com/rankings/weekly", {}, "/rankings/weekly"),
    ]

    for url, params, desc in resource_tests:
        test_endpoint(url, params, desc)

    # Test 3: Query param with different values
    print("\n\n" + "="*80)
    print(" SECTION 3: DIFFERENT PERIOD PARAM VALUES")
    print("="*80)

    param_tests = [
        ("https://data-api.polymarket.com/leaderboard", {"period": "today"}, "period=today"),
        ("https://data-api.polymarket.com/leaderboard", {"period": "week"}, "period=week"),
        ("https://data-api.polymarket.com/leaderboard", {"period": "month"}, "period=month"),
        ("https://data-api.polymarket.com/leaderboard", {"timeframe": "today"}, "timeframe=today"),
        ("https://data-api.polymarket.com/leaderboard", {"timeframe": "weekly"}, "timeframe=weekly"),
        ("https://data-api.polymarket.com/leaderboard", {"timeframe": "monthly"}, "timeframe=monthly"),
        ("https://data-api.polymarket.com/leaderboard", {"time": "today"}, "time=today"),
        ("https://data-api.polymarket.com/leaderboard", {"time": "week"}, "time=week"),
        ("https://data-api.polymarket.com/leaderboard", {"time": "month"}, "time=month"),
    ]

    for url, params, desc in param_tests:
        test_endpoint(url, params, desc)

    # Test 4: Volume sorting with different params
    print("\n\n" + "="*80)
    print(" SECTION 4: VOLUME SORTING VARIATIONS")
    print("="*80)

    sort_tests = [
        ("https://data-api.polymarket.com/leaderboard", {"sortBy": "volume", "limit": 5}, "sortBy=volume"),
        ("https://data-api.polymarket.com/leaderboard", {"orderBy": "vol", "limit": 5}, "orderBy=vol"),
        ("https://data-api.polymarket.com/leaderboard", {"orderBy": "volume", "limit": 5}, "orderBy=volume"),
        ("https://data-api.polymarket.com/leaderboard", {"sort": "vol", "order": "desc", "limit": 5}, "sort=vol, order=desc"),
    ]

    for url, params, desc in param_tests:
        success, data = test_endpoint(url, params, desc)

        if success and data and isinstance(data, list):
            # Check if first entry has highest volume
            if len(data) >= 2:
                first_vol = data[0].get('vol', 0)
                second_vol = data[1].get('vol', 0)
                print(f"  Volume check: 1st={first_vol}, 2nd={second_vol}")
                if first_vol > second_vol:
                    print(f"  ✓ Sorted by volume correctly!")

    # Test 5: Try stats/metrics endpoints
    print("\n\n" + "="*80)
    print(" SECTION 5: STATS/METRICS ENDPOINTS")
    print("="*80)

    stats_tests = [
        ("https://data-api.polymarket.com/stats/leaderboard", {}, "/stats/leaderboard"),
        ("https://data-api.polymarket.com/metrics/leaderboard", {}, "/metrics/leaderboard"),
        ("https://data-api.polymarket.com/trader-stats", {}, "/trader-stats"),
        ("https://gamma-api.polymarket.com/stats/traders", {}, "Gamma: /stats/traders"),
    ]

    for url, params, desc in stats_tests:
        test_endpoint(url, params, desc)

    print("\n\n" + "="*80)
    print(" MANUAL INSPECTION REQUIRED")
    print("="*80)
    print("""
To find the REAL endpoint, you need to:

1. Open browser DevTools (F12)
2. Go to Network tab
3. Visit https://polymarket.com/leaderboard
4. Click "Today" tab → Look for XHR/Fetch requests
5. Click "Weekly" tab → Look for XHR/Fetch requests
6. Click "Volume" toggle → Look for XHR/Fetch requests
7. Document the actual API calls being made

Look for requests to domains:
- data-api.polymarket.com
- gamma-api.polymarket.com
- clob.polymarket.com

The REAL endpoint will show up in the Network tab!
""")

if __name__ == "__main__":
    main()
