#!/usr/bin/env python3
"""
Test T003: Schedule Lookup Validation

Tests the market discovery process:
1. Load 1-year schedule
2. Find current market period
3. Query Gamma API for market details
4. Extract and validate token IDs

Usage:
    cd /orderbook_snapshots
    uv run python scripts/test_schedule_lookup.py
"""

import polars as pl
import requests
import json
import time
from datetime import datetime
from pathlib import Path

# Configuration
SCHEDULE_FILE = Path("config/btc_updown_schedule_1year.parquet")
GAMMA_URL = "https://gamma-api.polymarket.com/markets"

def main():
    print("=" * 80)
    print(" TEST T003: SCHEDULE LOOKUP VALIDATION")
    print("=" * 80)
    print()

    start_time = time.time()

    # Step 1: Load schedule
    print("📅 Step 1: Loading schedule...")
    load_start = time.time()

    if not SCHEDULE_FILE.exists():
        print(f"❌ FAIL: Schedule file not found: {SCHEDULE_FILE}")
        return False

    df = pl.read_parquet(SCHEDULE_FILE)
    load_time = (time.time() - load_start) * 1000

    print(f"   ✅ Schedule loaded: {len(df):,} periods")
    print(f"   ⏱️  Load time: {load_time:.1f}ms")
    print()

    # Step 2: Get current timestamp
    print("🕐 Step 2: Finding current market period...")
    current_ts = int(time.time())
    current_dt = datetime.fromtimestamp(current_ts)
    print(f"   Current time: {current_dt.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"   Unix timestamp: {current_ts}")
    print()

    # Step 3: Find current period
    print("🔍 Step 3: Searching schedule for current period...")
    current_period = df.filter(
        (pl.col("start_timestamp") <= current_ts) &
        (pl.col("end_timestamp") > current_ts)
    )

    if len(current_period) == 0:
        print("   ❌ FAIL: No current market period found in schedule")
        print("   Possible reasons:")
        print("   - Schedule is outdated")
        print("   - System clock is incorrect")
        print("   - Outside trading hours")
        return False

    # Extract market info
    slug = current_period['slug'][0]
    start_ts = current_period['start_timestamp'][0]
    end_ts = current_period['end_timestamp'][0]
    start_time_str = current_period['time_start'][0]
    end_time_str = current_period['time_end'][0]
    date_str = current_period['date'][0]
    period_num = current_period['period_number'][0]

    time_remaining = end_ts - current_ts
    minutes_remaining = time_remaining // 60
    seconds_remaining = time_remaining % 60

    print(f"   ✅ Period found!")
    print(f"   Period number: {period_num}/35,040")
    print(f"   Date: {date_str}")
    print(f"   Time: {start_time_str} - {end_time_str}")
    print(f"   Slug: {slug}")
    print(f"   Start timestamp: {start_ts}")
    print(f"   End timestamp: {end_ts}")
    print(f"   Time remaining: {minutes_remaining}m {seconds_remaining}s")
    print()

    # Step 4: Query Gamma API
    print("🌐 Step 4: Querying Gamma API for market details...")
    api_start = time.time()

    try:
        response = requests.get(GAMMA_URL, params={"slug": slug}, timeout=10)
        api_latency = (time.time() - api_start) * 1000

        print(f"   Response status: {response.status_code}")
        print(f"   API latency: {api_latency:.1f}ms")

        response.raise_for_status()

        market_data = response.json()

        if not market_data or len(market_data) == 0:
            print(f"   ❌ FAIL: No market data returned for slug: {slug}")
            return False

        market = market_data[0]
        print(f"   ✅ Market data retrieved")
        print()

        # Step 5: Extract market details
        print("📊 Step 5: Extracting market details...")
        condition_id = market.get('conditionId', 'N/A')
        question = market.get('question', 'N/A')
        active = market.get('active', False)
        closed = market.get('closed', False)
        token_ids_str = market.get('clobTokenIds', '')

        print(f"   Question: {question}")
        print(f"   Condition ID: {condition_id[:40]}...")
        print(f"   Active: {active}")
        print(f"   Closed: {closed}")
        print()

        # Step 6: Parse token IDs
        print("🎯 Step 6: Parsing token IDs...")
        try:
            token_ids = json.loads(token_ids_str) if token_ids_str else []

            if len(token_ids) < 2:
                print(f"   ❌ FAIL: Expected 2 token IDs, got {len(token_ids)}")
                return False

            print(f"   ✅ Token IDs parsed successfully")
            print(f"   UP token:   {token_ids[0]}")
            print(f"   DOWN token: {token_ids[1]}")
            print()

            # Validation checks
            print("✅ Step 7: Validation checks...")
            checks_passed = True

            if not active:
                print("   ⚠️  WARNING: Market is not active")
                checks_passed = False
            else:
                print("   ✅ Market is active")

            if closed:
                print("   ⚠️  WARNING: Market is closed")
                checks_passed = False
            else:
                print("   ✅ Market is not closed")

            if len(token_ids) == 2:
                print("   ✅ Token count correct (2)")
            else:
                print(f"   ❌ Token count incorrect: {len(token_ids)}")
                checks_passed = False

            print()

            # Summary
            total_time = (time.time() - start_time) * 1000
            print("=" * 80)
            print(" TEST RESULTS")
            print("=" * 80)
            print(f"Status: {'✅ PASS' if checks_passed else '⚠️  PASS WITH WARNINGS'}")
            print(f"Total execution time: {total_time:.1f}ms")
            print()
            print("Performance Breakdown:")
            print(f"  - Schedule load:  {load_time:.1f}ms")
            print(f"  - API call:       {api_latency:.1f}ms")
            print(f"  - Other:          {(total_time - load_time - api_latency):.1f}ms")
            print()
            print("Market Information:")
            print(f"  - Slug: {slug}")
            print(f"  - Period: {start_time_str} - {end_time_str}")
            print(f"  - Time remaining: {minutes_remaining}m {seconds_remaining}s")
            print(f"  - Token IDs: 2")
            print()
            print("=" * 80)

            return checks_passed

        except json.JSONDecodeError as e:
            print(f"   ❌ FAIL: Could not parse token IDs: {e}")
            return False

    except requests.RequestException as e:
        print(f"   ❌ FAIL: API request failed: {e}")
        return False

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
