#!/usr/bin/env python3
"""
Find currently active 15-minute BTC "Up or Down" markets on Polymarket.
Uses the generated schedule to find current and next period markets.
"""

import polars as pl
import requests
import json
import time
from datetime import datetime
from pathlib import Path

def find_active_btc_15min_markets():
    """Find current and next BTC 15-min markets using schedule."""

    schedule_file = Path("data/btc_updown_schedule_7days.parquet")

    if not schedule_file.exists():
        print(f"❌ Schedule file not found: {schedule_file}")
        print("   Run: uv run python scripts/generate_btc_updown_schedule.py")
        return []

    # Load schedule
    print("📅 Loading BTC 15-min market schedule...")
    df = pl.read_parquet(schedule_file)

    # Get current timestamp
    current_ts = int(time.time())
    current_dt = datetime.fromtimestamp(current_ts)

    print(f"🔍 Current time: {current_dt.strftime('%Y-%m-%d %H:%M:%S')} ({current_ts})")
    print()

    # Find current period (we are IN this period now)
    current_period = df.filter(
        (pl.col("start_timestamp") <= current_ts) &
        (pl.col("end_timestamp") > current_ts)
    )

    # Find next period (starts after now)
    next_period = df.filter(
        pl.col("start_timestamp") > current_ts
    ).head(1)

    # Query API for both periods
    gamma_url = "https://gamma-api.polymarket.com/markets"
    results = []

    # Check current period
    if len(current_period) > 0:
        slug = current_period['slug'][0]
        start_ts = current_period['start_timestamp'][0]
        end_ts = current_period['end_timestamp'][0]
        start_time = current_period['time_start'][0]
        end_time = current_period['time_end'][0]
        date = current_period['date'][0]

        print("="*80)
        print("📍 CURRENT PERIOD (IN PROGRESS)")
        print("="*80)
        print(f"   Date: {date}")
        print(f"   Time: {start_time} - {end_time}")
        print(f"   Start timestamp: {start_ts}")
        print(f"   End timestamp: {end_ts}")
        print(f"   Slug: {slug}")

        # Query API
        try:
            response = requests.get(gamma_url, params={"slug": slug}, timeout=10)
            if response.status_code == 200:
                market_data = response.json()
                if market_data and len(market_data) > 0:
                    market = market_data[0]
                    condition_id = market.get('conditionId')
                    closed = market.get('closed', False)
                    active = market.get('active', False)

                    status = "✅ ACTIVE" if (not closed and active) else "⚠️  CLOSED" if closed else "⏸️  INACTIVE"
                    print(f"   Status: {status}")
                    print(f"   Condition ID: {condition_id}")
                    print(f"   Question: {market.get('question', 'N/A')[:60]}...")

                    results.append({
                        'period': 'current',
                        'slug': slug,
                        'start_timestamp': start_ts,
                        'end_timestamp': end_ts,
                        'condition_id': condition_id,
                        'closed': closed,
                        'active': active,
                        'question': market.get('question'),
                        'time': f"{start_time} - {end_time}",
                        'date': date
                    })
                else:
                    print(f"   Status: ❌ NOT FOUND")
            else:
                print(f"   Status: ❌ API ERROR ({response.status_code})")
        except Exception as e:
            print(f"   Status: ❌ ERROR - {e}")
    else:
        print("⚠️  No current period found in schedule")

    print()

    # Check next period
    if len(next_period) > 0:
        slug = next_period['slug'][0]
        start_ts = next_period['start_timestamp'][0]
        end_ts = next_period['end_timestamp'][0]
        start_time = next_period['time_start'][0]
        end_time = next_period['time_end'][0]
        date = next_period['date'][0]

        print("="*80)
        print("⏭️  NEXT PERIOD")
        print("="*80)
        print(f"   Date: {date}")
        print(f"   Time: {start_time} - {end_time}")
        print(f"   Start timestamp: {start_ts}")
        print(f"   End timestamp: {end_ts}")
        print(f"   Slug: {slug}")

        # Query API
        try:
            response = requests.get(gamma_url, params={"slug": slug}, timeout=10)
            if response.status_code == 200:
                market_data = response.json()
                if market_data and len(market_data) > 0:
                    market = market_data[0]
                    condition_id = market.get('conditionId')
                    closed = market.get('closed', False)
                    active = market.get('active', False)

                    status = "✅ EXISTS" if active else "⚠️  CLOSED" if closed else "⏸️  INACTIVE"
                    print(f"   Status: {status}")
                    print(f"   Condition ID: {condition_id}")
                    print(f"   Question: {market.get('question', 'N/A')[:60]}...")

                    results.append({
                        'period': 'next',
                        'slug': slug,
                        'start_timestamp': start_ts,
                        'end_timestamp': end_ts,
                        'condition_id': condition_id,
                        'closed': closed,
                        'active': active,
                        'question': market.get('question'),
                        'time': f"{start_time} - {end_time}",
                        'date': date
                    })
                else:
                    print(f"   Status: ⏳ NOT YET CREATED")
            else:
                print(f"   Status: ⏳ NOT YET AVAILABLE ({response.status_code})")
        except Exception as e:
            print(f"   Status: ❌ ERROR - {e}")
    else:
        print("⚠️  No next period found in schedule (end of 7-day window)")

    print("="*80)

    return results

if __name__ == '__main__':
    markets = find_active_btc_15min_markets()

    if markets:
        print(f"\n📊 Summary: {len(markets)} active markets found")

        # Save to file for reference
        output_file = 'data/active_btc_15min_markets.json'
        with open(output_file, 'w') as f:
            json.dump(markets, f, indent=2)
        print(f"💾 Saved to {output_file}")
    else:
        print("\n❌ No active markets found. They may not be available right now.")
        print("   Note: 15-min markets may have specific trading hours.")
