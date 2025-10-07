#!/usr/bin/env python3
"""
Verify if period parameters actually change the data
by comparing results across different period values
"""

import requests
import json

BASE_URL = "https://data-api.polymarket.com/leaderboard"

def fetch_leaderboard(params):
    """Fetch and return leaderboard data"""
    response = requests.get(BASE_URL, params=params, timeout=10)
    response.raise_for_status()
    return response.json()

def main():
    print("="*80)
    print(" VERIFYING IF PERIOD PARAMETERS ACTUALLY WORK")
    print("="*80)

    # Fetch leaderboards with different period params
    print("\n📥 Fetching data with different period parameters...")

    default = fetch_leaderboard({})
    period_1d = fetch_leaderboard({"period": "1d"})
    period_7d = fetch_leaderboard({"period": "7d"})
    period_30d = fetch_leaderboard({"period": "30d"})
    period_all = fetch_leaderboard({"period": "all"})

    print(f"✓ Default: {len(default)} results")
    print(f"✓ Period 1d: {len(period_1d)} results")
    print(f"✓ Period 7d: {len(period_7d)} results")
    print(f"✓ Period 30d: {len(period_30d)} results")
    print(f"✓ Period all: {len(period_all)} results")

    # Compare first entry from each
    print("\n" + "="*80)
    print(" COMPARING FIRST ENTRY FROM EACH PERIOD")
    print("="*80)

    print("\nDefault (no params):")
    print(f"  Rank: {default[0]['rank']}, User: {default[0]['user_name'][:20]}, Vol: {default[0]['vol']}, PnL: {default[0]['pnl']}")

    print("\nPeriod 1d:")
    print(f"  Rank: {period_1d[0]['rank']}, User: {period_1d[0]['user_name'][:20]}, Vol: {period_1d[0]['vol']}, PnL: {period_1d[0]['pnl']}")

    print("\nPeriod 7d:")
    print(f"  Rank: {period_7d[0]['rank']}, User: {period_7d[0]['user_name'][:20]}, Vol: {period_7d[0]['vol']}, PnL: {period_7d[0]['pnl']}")

    print("\nPeriod 30d:")
    print(f"  Rank: {period_30d[0]['rank']}, User: {period_30d[0]['user_name'][:20]}, Vol: {period_30d[0]['vol']}, PnL: {period_30d[0]['pnl']}")

    print("\nPeriod all:")
    print(f"  Rank: {period_all[0]['rank']}, User: {period_all[0]['user_name'][:20]}, Vol: {period_all[0]['vol']}, PnL: {period_all[0]['pnl']}")

    # Check if ALL entries are identical
    print("\n" + "="*80)
    print(" DEEP COMPARISON: ARE RESULTS ACTUALLY DIFFERENT?")
    print("="*80)

    def are_identical(list1, list2):
        """Check if two result lists are identical"""
        if len(list1) != len(list2):
            return False
        for i in range(len(list1)):
            if list1[i]['user_id'] != list2[i]['user_id']:
                return False
            if list1[i]['vol'] != list2[i]['vol']:
                return False
            if list1[i]['pnl'] != list2[i]['pnl']:
                return False
        return True

    comparisons = [
        ("Default", "Period 1d", default, period_1d),
        ("Default", "Period 7d", default, period_7d),
        ("Default", "Period 30d", default, period_30d),
        ("Default", "Period all", default, period_all),
        ("Period 1d", "Period 7d", period_1d, period_7d),
        ("Period 1d", "Period 30d", period_1d, period_30d),
        ("Period 7d", "Period 30d", period_7d, period_30d),
    ]

    all_identical = True
    for name1, name2, data1, data2 in comparisons:
        identical = are_identical(data1, data2)
        status = "✗ IDENTICAL" if identical else "✓ DIFFERENT"
        print(f"  {name1} vs {name2}: {status}")
        if not identical:
            all_identical = False

    # CONCLUSION
    print("\n" + "="*80)
    print(" FINAL CONCLUSION")
    print("="*80)

    if all_identical:
        print("\n❌ PERIOD PARAMETERS DO NOT WORK")
        print("   All period values return IDENTICAL data")
        print("   → API returns ALL-TIME statistics only")
        print("   → Parameters are ignored")
    else:
        print("\n✅ PERIOD PARAMETERS WORK")
        print("   Different period values return DIFFERENT data")
        print("   → API correctly filters by time period")

    # Additional check: compare specific traders
    print("\n" + "="*80)
    print(" SPOT CHECK: Compare specific trader across periods")
    print("="*80)

    # Find a trader that appears in all responses
    test_user_id = default[4]['user_id']  # 5th ranked trader
    test_user_name = default[4]['user_name']

    print(f"\nTracking trader: {test_user_name} ({test_user_id[:20]}...)")

    datasets = [
        ("Default", default),
        ("Period 1d", period_1d),
        ("Period 7d", period_7d),
        ("Period 30d", period_30d),
        ("Period all", period_all),
    ]

    for name, data in datasets:
        trader = next((t for t in data if t['user_id'] == test_user_id), None)
        if trader:
            print(f"  {name:15}: Rank={trader['rank']:3}, Vol={trader['vol']:>12.2f}, PnL={trader['pnl']:>10.2f}")
        else:
            print(f"  {name:15}: NOT FOUND")

    print("\n" + "="*80)

if __name__ == "__main__":
    main()
