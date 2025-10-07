#!/usr/bin/env python3
"""
24/7 Bitcoin 15-Minute Orderbook Streaming Daemon

Continuously discovers, rotates, and streams orderbook data for BTC 15-minute
"Up or Down" markets on Polymarket.

Features:
- Automatic market discovery every minute
- Seamless rotation every 15 minutes
- WebSocket orderbook streaming
- Parquet data storage with date/market partitioning
- Graceful error handling and auto-reconnect

Usage:
    # Run daemon
    uv run python scripts/btc_15min_orderbook_daemon.py

    # Run with custom output directory
    uv run python scripts/btc_15min_orderbook_daemon.py --output-dir /data/orderbooks

    # Test mode (discover markets but don't stream)
    uv run python scripts/btc_15min_orderbook_daemon.py --test-mode
"""

import os
import sys
import time
import json
import signal
import argparse
import threading
from pathlib import Path
from datetime import datetime, timezone, timedelta
from typing import Optional, Dict

# Import market manager
from btc_15min_market_manager import MarketRotationManager

# Import order book streamer
import importlib.util
spec = importlib.util.spec_from_file_location("stream_orderbook", "scripts/stream_orderbook_realtime.py")
stream_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(stream_module)

# ==============================================================================
# Configuration
# ==============================================================================

DEFAULT_OUTPUT_DIR = "data/orderbook_snapshots"
ROTATION_CHECK_INTERVAL = 60  # Check for rotation every 60 seconds
MIN_SNAPSHOT_INTERVAL_MS = 1000  # 1 snapshot per second

# ==============================================================================
# Integrated Daemon
# ==============================================================================

class BTCOrderbookDaemon:
    """
    Main daemon that integrates market discovery, rotation, and streaming.
    """

    def __init__(self, output_dir: str, test_mode: bool = False):
        self.output_dir = Path(output_dir)
        self.test_mode = test_mode

        # Components
        self.market_manager = MarketRotationManager()
        self.streamer = None  # Will be created when market is available

        # State
        self.current_market_id = None
        self.is_running = False
        self.rotation_thread = None

        # Stats
        self.stats = {
            "daemon_start_time": datetime.now(timezone.utc).isoformat(),
            "total_rotations": 0,
            "total_snapshots": 0,
            "markets_streamed": []
        }

    def start(self):
        """Start the daemon."""
        print("="*80)
        print(" BTC 15-MIN ORDERBOOK STREAMING DAEMON")
        print("="*80)
        print(f"\nOutput directory: {self.output_dir}")
        print(f"Rotation check interval: {ROTATION_CHECK_INTERVAL}s")
        print(f"Snapshot interval: {MIN_SNAPSHOT_INTERVAL_MS}ms")
        print(f"Test mode: {self.test_mode}")
        print("\nPress Ctrl+C to stop gracefully\n")

        self.is_running = True

        # Setup signal handlers
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

        # Start rotation monitoring thread
        self.rotation_thread = threading.Thread(target=self._rotation_loop, daemon=True)
        self.rotation_thread.start()

        # Initial market setup
        self._setup_current_market()

        if not self.test_mode:
            # Start streaming (blocks)
            self._streaming_loop()
        else:
            # Test mode - just monitor rotations
            print("🧪 TEST MODE: Monitoring market discovery and rotation")
            print("   (Not streaming orderbook data)\n")
            while self.is_running:
                time.sleep(5)

    def _setup_current_market(self):
        """Setup streaming for current market."""
        print("\n🔍 Discovering current market...")

        market = self.market_manager.get_current_market()

        if not market:
            print("⚠️  No active market found - will retry in background")
            return False

        condition_id = market.get('condition_id')

        if condition_id == self.current_market_id:
            print(f"✅ Already streaming current market: {market['question'][:60]}...")
            return True

        # New market - update streaming target
        print(f"\n🎯 New market detected:")
        print(f"   {market['question']}")
        print(f"   Condition ID: {condition_id}")

        self.current_market_id = condition_id

        # Stop existing streamer if running
        if self.streamer and not self.test_mode:
            print("   Stopping previous streamer...")
            self.streamer.stop()
            self.streamer = None

        # Create new streamer
        if not self.test_mode:
            print("   Creating new streamer...")
            self.streamer = stream_module.OrderBookStreamer(
                markets=[condition_id],
                output_dir=str(self.output_dir),
                min_interval_ms=MIN_SNAPSHOT_INTERVAL_MS,
                batch_size=1000,
                flush_interval_s=10
            )

        # Update stats
        self.stats["markets_streamed"].append({
            "market": market['question'],
            "condition_id": condition_id,
            "start_time": datetime.now(timezone.utc).isoformat()
        })
        self.stats["total_rotations"] += 1

        return True

    def _rotation_loop(self):
        """Background thread that monitors market rotation."""
        while self.is_running:
            try:
                time.sleep(ROTATION_CHECK_INTERVAL)

                if not self.is_running:
                    break

                # Check if rotation needed
                if self.market_manager.should_rotate():
                    print(f"\n🔄 [{datetime.now(timezone.utc).strftime('%H:%M:%S')} UTC] Market rotation triggered")
                    self._setup_current_market()

            except Exception as e:
                print(f"❌ Error in rotation loop: {e}")
                time.sleep(10)  # Wait before retry

    def _streaming_loop(self):
        """Main streaming loop (runs in main thread)."""
        print("\n🚀 Starting orderbook streaming...\n")

        while self.is_running:
            try:
                if self.streamer:
                    # Start streaming (blocks until disconnected)
                    self.streamer.start()

                    # If we get here, streamer stopped unexpectedly
                    print("\n⚠️  Streamer stopped - checking for new market...")
                    time.sleep(5)
                    self._setup_current_market()
                else:
                    # No streamer - wait for rotation thread to find a market
                    print("⏳ Waiting for active market...")
                    time.sleep(30)
                    self._setup_current_market()

            except KeyboardInterrupt:
                break
            except Exception as e:
                print(f"\n❌ Error in streaming loop: {e}")
                print("   Retrying in 10 seconds...")
                time.sleep(10)

    def _signal_handler(self, signum, frame):
        """Handle shutdown signals gracefully."""
        print("\n\n🛑 Shutdown signal received - stopping gracefully...")
        self.stop()
        sys.exit(0)

    def stop(self):
        """Stop the daemon gracefully."""
        print("\n🛑 Stopping daemon...")
        self.is_running = False

        # Stop streamer
        if self.streamer:
            print("   Stopping streamer...")
            self.streamer.stop()

        # Wait for rotation thread
        if self.rotation_thread and self.rotation_thread.is_alive():
            print("   Waiting for rotation thread...")
            self.rotation_thread.join(timeout=5)

        # Print stats
        print("\n📊 DAEMON STATISTICS:")
        print(f"   Total rotations: {self.stats['total_rotations']}")
        print(f"   Markets streamed: {len(self.stats['markets_streamed'])}")
        print(f"   Running time: {datetime.now(timezone.utc).isoformat()}")

        print("\n✅ Daemon stopped gracefully")

# ==============================================================================
# Main
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='24/7 Bitcoin 15-Min Orderbook Streaming Daemon'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default=DEFAULT_OUTPUT_DIR,
        help='Output directory for Parquet files'
    )
    parser.add_argument(
        '--test-mode',
        action='store_true',
        help='Test mode (discover markets but don\'t stream)'
    )

    args = parser.parse_args()

    # Create daemon
    daemon = BTCOrderbookDaemon(
        output_dir=args.output_dir,
        test_mode=args.test_mode
    )

    # Run
    try:
        daemon.start()
    except KeyboardInterrupt:
        daemon.stop()

if __name__ == '__main__':
    main()
