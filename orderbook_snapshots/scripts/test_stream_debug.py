#!/usr/bin/env python3
"""
Debug test for stream_continuous.py
Runs for 30 seconds and logs everything
"""

import sys
import time
import requests
import polars as pl
from pathlib import Path

# Configuration
SCHEDULE_FILE = Path("config/btc_updown_schedule_1year.parquet")
GAMMA_URL = "https://gamma-api.polymarket.com/markets"
CLOB_URL = "https://clob.polymarket.com"

def main():
    print("=" * 80)
    print("DEBUG TEST: Stream Continuous")
    print("=" * 80)
    print()

    # Step 1: Load schedule
    print("Step 1: Loading schedule...")
    df = pl.read_parquet(SCHEDULE_FILE)
    print(f"   ✅ Loaded {len(df):,} periods")
    print()

    # Step 2: Find current market
    print("Step 2: Finding current market...")
    current_ts = int(time.time())
    current_period = df.filter(
        (pl.col("start_timestamp") <= current_ts) &
        (pl.col("end_timestamp") > current_ts)
    )

    if len(current_period) == 0:
        print("   ❌ No current market found")
        return False

    slug = current_period['slug'][0]
    print(f"   ✅ Found: {slug}")
    print()

    # Step 3: Query Gamma API
    print("Step 3: Querying Gamma API...")
    response = requests.get(GAMMA_URL, params={"slug": slug}, timeout=10)
    market_data = response.json()

    if not market_data or len(market_data) == 0:
        print(f"   ❌ No market data for {slug}")
        return False

    market = market_data[0]
    import json
    token_ids = json.loads(market.get('clobTokenIds', '[]'))

    if len(token_ids) < 2:
        print(f"   ❌ Expected 2 token IDs, got {len(token_ids)}")
        return False

    up_token = token_ids[0]
    down_token = token_ids[1]

    print(f"   ✅ UP token:   {up_token}")
    print(f"   ✅ DOWN token: {down_token}")
    print()

    # Step 4: Poll orderbook
    print("Step 4: Polling orderbook (30 second test)...")
    print()

    snapshots = 0
    errors = 0
    start_time = time.time()
    test_duration = 30  # 30 seconds

    while time.time() - start_time < test_duration:
        loop_start = time.time()

        try:
            # Poll API
            response = requests.get(
                f"{CLOB_URL}/book",
                params={"token_id": up_token},
                timeout=5
            )

            if response.status_code == 200:
                book = response.json()
                snapshots += 1

                # Show first snapshot details
                if snapshots == 1:
                    print(f"   First snapshot received:")
                    print(f"      Timestamp: {book.get('timestamp')}")
                    print(f"      Market: {book.get('market', 'N/A')[:40]}...")
                    print(f"      Bids: {len(book.get('bids', []))}")
                    print(f"      Asks: {len(book.get('asks', []))}")
                    print()

                # Log progress every 5 snapshots
                if snapshots % 5 == 0:
                    elapsed = time.time() - start_time
                    print(f"   Progress: {snapshots} snapshots in {elapsed:.1f}s")

            else:
                print(f"   ⚠️  HTTP {response.status_code}")
                errors += 1

        except Exception as e:
            print(f"   ❌ Error: {e}")
            errors += 1

        # Sleep to maintain 1-second interval
        loop_elapsed = time.time() - loop_start
        sleep_time = max(0, 1.0 - loop_elapsed)
        if sleep_time > 0:
            time.sleep(sleep_time)

    # Summary
    total_time = time.time() - start_time
    print()
    print("=" * 80)
    print("TEST COMPLETE")
    print("=" * 80)
    print(f"Snapshots collected: {snapshots}")
    print(f"Errors: {errors}")
    print(f"Duration: {total_time:.1f}s")
    print(f"Success rate: {(snapshots/(snapshots+errors)*100):.1f}%")
    print()

    if snapshots >= 25:  # Should get ~30 snapshots in 30s
        print("✅ TEST PASSED")
        return True
    else:
        print("❌ TEST FAILED (too few snapshots)")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
