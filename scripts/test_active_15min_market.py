#!/usr/bin/env python3
"""
Test streaming with currently active 15-min BTC market.
"""

import sys
import time
import signal
import requests
from pathlib import Path

# Import the streaming module
import importlib.util
spec = importlib.util.spec_from_file_location("stream_orderbook", "scripts/stream_orderbook_realtime.py")
stream_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(stream_module)

def get_active_15min_market():
    """Fetch the currently active 15-min BTC market."""
    # The slug from the URL provided
    slug = "btc-up-or-down-15m-1759753800"

    url = f"https://gamma-api.polymarket.com/markets"
    params = {"slug": slug}

    response = requests.get(url, params=params, timeout=10)
    response.raise_for_status()

    data = response.json()

    if isinstance(data, list) and len(data) > 0:
        market = data[0]
    elif isinstance(data, dict):
        market = data
    else:
        return None

    # Extract token IDs
    token_ids_str = market.get('clobTokenIds', '')
    token_ids = token_ids_str.split(',') if token_ids_str else []

    return {
        'question': market.get('question'),
        'slug': market.get('slug'),
        'condition_id': market.get('conditionId'),
        'token_ids': token_ids,
        'closed': market.get('closed'),
        'active': market.get('active'),
        'end_date': market.get('endDate'),
    }

def main():
    print("="*80)
    print(" TEST: Active 15-Min BTC Market Streaming")
    print("="*80)
    print()

    # Get active market
    print("🔍 Fetching active 15-min BTC market...")
    market = get_active_15min_market()

    if not market:
        print("❌ Could not find active market")
        sys.exit(1)

    print(f"\n✅ Found market:")
    print(f"   Question: {market['question']}")
    print(f"   Slug: {market['slug']}")
    print(f"   Condition ID: {market['condition_id']}")
    print(f"   Token IDs: {len(market['token_ids'])} tokens")
    print(f"   Active: {market['active']}")
    print(f"   Closed: {market['closed']}")
    print(f"   End Date: {market['end_date']}")
    print()

    # Create streamer
    print("🚀 Starting orderbook streamer...")
    print(f"   Duration: 60 seconds")
    print(f"   Output: /tmp/test_15min_orderbook")
    print()

    streamer = stream_module.OrderBookStreamer(
        markets=[market['condition_id']],
        token_ids=market['token_ids'],  # Include token IDs
        output_dir="/tmp/test_15min_orderbook",
        min_interval_ms=1000,  # 1 snapshot/second
        batch_size=50,
        flush_interval_s=5
    )

    # Setup auto-stop after 60 seconds
    def timeout_handler(signum, frame):
        print("\n\n⏰ 60 second test complete!")
        streamer.stop()
        sys.exit(0)

    signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(60)  # Set 60 second alarm

    # Also handle Ctrl+C
    def interrupt_handler(signum, frame):
        print("\n\n⚠️  Interrupted by user")
        streamer.stop()
        sys.exit(0)

    signal.signal(signal.SIGINT, interrupt_handler)

    # Start streaming
    try:
        streamer.start()
    except Exception as e:
        print(f"\n❌ Error: {e}")
        streamer.stop()

if __name__ == '__main__':
    main()
