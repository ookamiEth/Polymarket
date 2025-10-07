#!/usr/bin/env python3
"""
REST API Orderbook Polling Test
Polls orderbook every 1 second for 60 seconds, saves to Parquet.

Usage:
    cd /orderbook_snapshots
    uv run python scripts/test_snapshot.py
"""

import requests
import polars as pl
import json
import time
from datetime import datetime
from pathlib import Path

# Configuration
ORDERBOOK_URL = "https://clob.polymarket.com/book"
SCHEDULE_FILE = Path("config/btc_updown_schedule_1year.parquet")
OUTPUT_DIR = Path("data/test")
DURATION_SECONDS = 60
POLL_INTERVAL = 1.0  # 1 snapshot/second

def find_current_market_live():
    """
    Find current active 15-min BTC market directly from schedule.
    No dependency on JSON file - queries live data every time.
    """

    # Check if schedule exists
    if not SCHEDULE_FILE.exists():
        print(f"❌ Schedule file not found: {SCHEDULE_FILE}")
        print("   Run: uv run python scripts/generate_schedule.py")
        return None

    try:
        # Load schedule
        print("📅 Loading BTC 15-min market schedule...")
        df = pl.read_parquet(SCHEDULE_FILE)

        # Get current timestamp
        current_ts = int(time.time())
        current_dt = datetime.fromtimestamp(current_ts)
        print(f"🕐 Current time: {current_dt.strftime('%Y-%m-%d %H:%M:%S')} (timestamp: {current_ts})")

        # Find current period from schedule
        current_period = df.filter(
            (pl.col("start_timestamp") <= current_ts) &
            (pl.col("end_timestamp") > current_ts)
        )

        if len(current_period) == 0:
            print("❌ No current market period found in schedule")
            print("   The schedule may be outdated or we're outside trading hours")
            return None

        # Extract market info from schedule
        slug = current_period['slug'][0]
        start_ts = current_period['start_timestamp'][0]
        end_ts = current_period['end_timestamp'][0]
        start_time = current_period['time_start'][0]
        end_time = current_period['time_end'][0]
        date = current_period['date'][0]

        print(f"📍 Found market period: {start_time} - {end_time} ({date})")
        print(f"   Slug: {slug}")

        # Query Gamma API to get full market info + token IDs
        print(f"🔍 Querying Gamma API for market details...")
        gamma_url = "https://gamma-api.polymarket.com/markets"
        response = requests.get(gamma_url, params={"slug": slug}, timeout=10)
        response.raise_for_status()

        market_data = response.json()
        if not market_data or len(market_data) == 0:
            print(f"❌ Market not found in API: {slug}")
            return None

        market = market_data[0]

        # Extract all market info
        condition_id = market.get('conditionId')
        question = market.get('question')
        active = market.get('active', False)
        closed = market.get('closed', False)
        token_ids_str = market.get('clobTokenIds', '')

        # Parse token IDs (comes as JSON string)
        token_ids = json.loads(token_ids_str) if token_ids_str else []

        # Validate market status
        if closed:
            print(f"⚠️  WARNING: Market is marked as CLOSED")
            print(f"   This may cause orderbook API errors")

        if not active:
            print(f"⚠️  WARNING: Market is marked as INACTIVE")

        if len(token_ids) < 2:
            print(f"❌ Invalid token IDs: Expected 2, got {len(token_ids)}")
            return None

        # Calculate time remaining
        time_remaining = end_ts - current_ts
        minutes_remaining = time_remaining // 60
        seconds_remaining = time_remaining % 60

        print(f"✅ Market found and validated:")
        print(f"   Question: {question}")
        print(f"   Condition ID: {condition_id[:40]}...")
        print(f"   Status: Active={active}, Closed={closed}")
        print(f"   Token IDs: {len(token_ids)} tokens")
        print(f"   Time remaining: {minutes_remaining}m {seconds_remaining}s")

        return {
            'slug': slug,
            'condition_id': condition_id,
            'question': question,
            'active': active,
            'closed': closed,
            'start_timestamp': start_ts,
            'end_timestamp': end_ts,
            'time': f"{start_time} - {end_time}",
            'date': date,
            'token_ids': token_ids,  # Already parsed!
        }

    except Exception as e:
        print(f"❌ Error finding current market: {e}")
        import traceback
        traceback.print_exc()
        return None

def poll_orderbook(token_id, outcome):
    """Poll orderbook for a single token."""
    try:
        params = {"token_id": token_id}
        capture_time = int(time.time() * 1000)

        response = requests.get(ORDERBOOK_URL, params=params, timeout=5)
        response.raise_for_status()

        book = response.json()

        # Extract fields
        market = book.get('market', '')
        asset_id = book.get('asset_id', '')
        market_timestamp = int(book.get('timestamp', capture_time))
        bids = book.get('bids', [])
        asks = book.get('asks', [])

        # Extract LAST 3 levels from bids and asks (top of book)
        # Bids are sorted worst→best, so LAST entries are closest to mid
        # Asks are sorted worst→best, so LAST entries are closest to mid

        best_3_bids = bids[-3:] if len(bids) >= 3 else bids
        best_3_asks = asks[-3:] if len(asks) >= 3 else asks

        # Flatten to columns (reverse bids so bid_3 is furthest, bid_1 is closest)
        result = {
            'timestamp_ms': capture_time,
            'market_timestamp_ms': market_timestamp,
            'condition_id': market,
            'asset_id': asset_id,
        }

        # Bids (3 → 2 → 1, where 1 is best/closest to mid)
        for i in range(3):
            if i < len(best_3_bids):
                bid = best_3_bids[i]
                result[f'bid_price_{3-i}'] = float(bid['price'])
                result[f'bid_size_{3-i}'] = float(bid['size'])
            else:
                result[f'bid_price_{3-i}'] = None
                result[f'bid_size_{3-i}'] = None

        # Calculate spread and mid from best bid/ask
        best_bid = float(best_3_bids[-1]['price']) if best_3_bids else 0.0
        best_ask = float(best_3_asks[-1]['price']) if best_3_asks else 0.0

        result['spread'] = round((best_ask - best_bid), 3) if (best_bid > 0 and best_ask > 0) else None
        result['mid_price'] = round(((best_bid + best_ask) / 2), 3) if (best_bid > 0 and best_ask > 0) else None

        # Asks (1 → 2 → 3, where 1 is best/closest to mid)
        for i in range(3):
            if i < len(best_3_asks):
                ask = best_3_asks[-(i+1)]  # Reverse: take from end
                result[f'ask_price_{i+1}'] = float(ask['price'])
                result[f'ask_size_{i+1}'] = float(ask['size'])
            else:
                result[f'ask_price_{i+1}'] = None
                result[f'ask_size_{i+1}'] = None

        return result

    except Exception as e:
        print(f"   ❌ Error polling {outcome}: {e}")
        return None

def main():
    print("="*80)
    print(" ORDERBOOK SNAPSHOT TEST (REST API Polling)")
    print("="*80)
    print()

    # Find current market (live - no JSON file dependency)
    market_info = find_current_market_live()

    if not market_info:
        print("\n❌ Failed to find current market")
        print("   Possible reasons:")
        print("   - Schedule file is missing or outdated")
        print("   - No market is currently active")
        print("   - API returned no data")
        return

    # Token IDs are already included in market_info
    token_ids = market_info['token_ids']

    print(f"\n✅ Got {len(token_ids)} token IDs:")
    print(f"   UP token:   {token_ids[0][:30]}...")
    print(f"   DOWN token: {token_ids[1][:30]}...")

    # Start collection
    print(f"\n{'='*80}")
    print(f" STARTING {DURATION_SECONDS}-SECOND COLLECTION (UP TOKEN ONLY)")
    print(f"{'='*80}")
    print(f"Rate: {POLL_INTERVAL}s interval × 1 token = ~1 request/second")
    print(f"Rate limit: 200 req/10s (well within limits ✅)")
    print()

    snapshots = []

    start_collection = time.time()

    for i in range(DURATION_SECONDS):
        loop_start = time.time()

        # Poll UP token only (DOWN is redundant - it's 1 - UP)
        up_snapshot = poll_orderbook(token_ids[0], "UP")
        if up_snapshot:
            snapshots.append(up_snapshot)

        # Print progress (show flattened top 3 levels)
        elapsed = i + 1
        timestamp = datetime.now().strftime("%H:%M:%S")

        if up_snapshot:
            bid1 = f"${up_snapshot.get('bid_price_1', 0):.2f}" if up_snapshot.get('bid_price_1') else "N/A"
            ask1 = f"${up_snapshot.get('ask_price_1', 0):.2f}" if up_snapshot.get('ask_price_1') else "N/A"
            spread = f"${up_snapshot.get('spread', 0):.4f}" if up_snapshot.get('spread') else "N/A"
            mid = f"${up_snapshot.get('mid_price', 0):.4f}" if up_snapshot.get('mid_price') else "N/A"

            print(f"[{elapsed:2d}/{DURATION_SECONDS}] {timestamp} - Bid: {bid1} | Mid: {mid} | Ask: {ask1} | Spread: {spread}")
        else:
            print(f"[{elapsed:2d}/{DURATION_SECONDS}] {timestamp} - No data")

        # Sleep to maintain interval
        loop_elapsed = time.time() - loop_start
        sleep_time = max(0, POLL_INTERVAL - loop_elapsed)
        if sleep_time > 0:
            time.sleep(sleep_time)

    total_elapsed = time.time() - start_collection

    print(f"\n{'='*80}")
    print(" COLLECTION COMPLETE")
    print("="*80)
    print(f"\nTotal snapshots collected: {len(snapshots)}")
    print(f"Expected: {DURATION_SECONDS} (60s × 1 token)")
    print(f"Actual duration: {total_elapsed:.1f}s")

    if len(snapshots) == 0:
        print("\n❌ No snapshots collected!")
        return

    # Convert to DataFrame
    print(f"\n💾 Saving to Parquet...")
    df = pl.DataFrame(snapshots)

    # Generate output filename
    timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = OUTPUT_DIR / f"orderbook_test_{timestamp_str}.parquet"
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    df.write_parquet(output_file)

    file_size_kb = output_file.stat().st_size / 1024
    print(f"✅ Saved: {output_file}")
    print(f"   File size: {file_size_kb:.1f} KB")
    print(f"   Rows: {len(df)}")
    print(f"   Columns: {len(df.columns)}")

    # Summary statistics
    print(f"\n📊 Summary Statistics:")
    print(f"\n  UP Token ({len(df)} snapshots):")

    # Show sample of flattened data
    if 'bid_price_1' in df.columns and 'ask_price_1' in df.columns:
        print(f"    Avg best bid: ${df['bid_price_1'].mean():.4f}")
        print(f"    Avg best ask: ${df['ask_price_1'].mean():.4f}")
        print(f"    Avg spread: ${df['spread'].mean():.4f}")
        print(f"    Avg mid: ${df['mid_price'].mean():.4f}")

    # Time analysis
    time_range_ms = df['timestamp_ms'].max() - df['timestamp_ms'].min()
    print(f"\n  Time Coverage:")
    print(f"    Start: {datetime.fromtimestamp(df['timestamp_ms'].min()/1000).strftime('%H:%M:%S')}")
    print(f"    End: {datetime.fromtimestamp(df['timestamp_ms'].max()/1000).strftime('%H:%M:%S')}")
    print(f"    Duration: {time_range_ms/1000:.1f}s")

    print(f"\n✅ Test complete!")
    print(f"\n📌 Next steps:")
    print(f"   1. Query data: pl.read_parquet('{output_file}')")
    print(f"   2. Verify data quality")
    print(f"   3. Build production streaming script")

if __name__ == '__main__':
    main()
