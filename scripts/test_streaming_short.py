#!/usr/bin/env python3
"""
Short test of WebSocket streaming (runs for 30 seconds).
"""

import sys
import time
import signal
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

# Import the streaming module
import importlib.util
spec = importlib.util.spec_from_file_location("stream_orderbook", "scripts/stream_orderbook_realtime.py")
stream_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(stream_module)

def main():
    # Test market: BTC long-term prediction market
    test_market_id = "0xd8b9ff369452daebce1ac8cb6a29d6817903e85168356c72812317f38e317613"

    print("="*80)
    print(" TEST: WebSocket Streaming (30 seconds)")
    print("="*80)
    print(f"\nTest market ID: {test_market_id}")
    print(f"Duration: 30 seconds")
    print(f"Output: /tmp/test_orderbook\n")

    # Create streamer
    streamer = stream_module.OrderBookStreamer(
        markets=[test_market_id],
        output_dir="/tmp/test_orderbook",
        min_interval_ms=2000,  # 1 snapshot every 2 seconds
        batch_size=10,
        flush_interval_s=5
    )

    # Setup auto-stop after 30 seconds
    def timeout_handler(signum, frame):
        print("\n\n⏰ 30 second test complete!")
        streamer.stop()
        sys.exit(0)

    signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(30)  # Set 30 second alarm

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
