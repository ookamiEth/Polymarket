#!/usr/bin/env python3
"""
Real-time order book streaming via WebSocket for Polymarket markets.

Connects to Polymarket's WebSocket API to collect order book snapshots
at sub-second granularity. Designed for 24/7 operation on EC2.

Features:
- Event-driven order book snapshots (every update)
- Smart sampling (time-based + significance-based filtering)
- Batch writes to Parquet for performance
- Auto-reconnect on disconnect
- Partitioned storage by market and date
- S3 backup capability

Usage:
    # Stream specific markets (1 snapshot/second minimum)
    uv run python scripts/stream_orderbook_realtime.py \
      --markets "0xabc123...,0xdef456..." \
      --output-dir /data/orderbook_snapshots \
      --interval-ms 1000

    # Stream from market file with pattern filter
    uv run python scripts/stream_orderbook_realtime.py \
      --market-file data/active_markets.json \
      --filter-pattern "btc-up-or-down|eth-up-or-down" \
      --interval-ms 500  # 2 snapshots/second
"""

import os
import sys
import json
import time
import asyncio
import argparse
import threading
from pathlib import Path
from datetime import datetime, timezone
from typing import Dict, List, Optional, Set
from collections import deque

import polars as pl
from websocket import WebSocketApp


# ============================================================================
# Configuration
# ============================================================================

WSS_URL = "wss://ws-subscriptions-clob.polymarket.com/ws/market"

# Sampling configuration
DEFAULT_MIN_INTERVAL_MS = 1000  # 1 snapshot/second
DEFAULT_PRICE_CHANGE_THRESHOLD = 0.005  # 0.5% price movement triggers snapshot

# Batch writing
DEFAULT_BATCH_SIZE = 1000  # Events per batch
DEFAULT_FLUSH_INTERVAL_S = 10  # Force flush every 10s

# Storage
DEFAULT_OUTPUT_DIR = "data/orderbook_snapshots"


# ============================================================================
# Smart Sampler (Adaptive Snapshot Frequency)
# ============================================================================

class SmartSampler:
    """
    Adaptive sampling strategy:
    - Captures snapshot if time_since_last >= min_interval
    - OR if best bid/ask price changed > threshold
    """

    def __init__(
        self,
        min_interval_ms: int = DEFAULT_MIN_INTERVAL_MS,
        price_change_threshold: float = DEFAULT_PRICE_CHANGE_THRESHOLD
    ):
        self.min_interval_ms = min_interval_ms
        self.price_change_threshold = price_change_threshold

        # State tracking per market
        self.last_snapshot_time = {}  # market -> timestamp_ms
        self.last_best_bid = {}  # market -> price
        self.last_best_ask = {}  # market -> price

    def should_capture(self, market: str, book_event: Dict) -> bool:
        """Determine if this book event should be captured."""
        now_ms = time.time() * 1000

        # Initialize if first event for this market
        if market not in self.last_snapshot_time:
            self.last_snapshot_time[market] = now_ms
            self._update_state(market, book_event)
            return True

        # Check time threshold
        time_since_last = now_ms - self.last_snapshot_time[market]
        if time_since_last >= self.min_interval_ms:
            self.last_snapshot_time[market] = now_ms
            self._update_state(market, book_event)
            return True

        # Check significant price movement
        if self._has_significant_price_change(market, book_event):
            self.last_snapshot_time[market] = now_ms
            self._update_state(market, book_event)
            return True

        return False

    def _has_significant_price_change(self, market: str, book_event: Dict) -> bool:
        """Check if best bid/ask moved significantly."""
        if market not in self.last_best_bid or market not in self.last_best_ask:
            return True

        # Extract current best bid/ask
        bids = book_event.get('bids', [])
        asks = book_event.get('asks', [])

        if not bids or not asks:
            return False

        best_bid = float(bids[0].get('price', 0))
        best_ask = float(asks[0].get('price', 0))

        # Calculate change
        last_bid = self.last_best_bid[market]
        last_ask = self.last_best_ask[market]

        if last_bid > 0:
            bid_change = abs(best_bid - last_bid) / last_bid
            if bid_change > self.price_change_threshold:
                return True

        if last_ask > 0:
            ask_change = abs(best_ask - last_ask) / last_ask
            if ask_change > self.price_change_threshold:
                return True

        return False

    def _update_state(self, market: str, book_event: Dict):
        """Update internal state tracking."""
        bids = book_event.get('bids', [])
        asks = book_event.get('asks', [])

        if bids:
            self.last_best_bid[market] = float(bids[0].get('price', 0))

        if asks:
            self.last_best_ask[market] = float(asks[0].get('price', 0))


# ============================================================================
# Batch Writer (Efficient Parquet Storage)
# ============================================================================

class ParquetBatchWriter:
    """
    Batches events and writes to Parquet files.
    Partitioned by market and date for efficient querying.
    """

    def __init__(
        self,
        output_dir: Path,
        batch_size: int = DEFAULT_BATCH_SIZE,
        flush_interval_s: int = DEFAULT_FLUSH_INTERVAL_S
    ):
        self.output_dir = Path(output_dir)
        self.batch_size = batch_size
        self.flush_interval_s = flush_interval_s

        # Buffers per event type
        self.snapshot_buffer = []  # book events
        self.update_buffer = []  # price_change events
        self.trade_buffer = []  # last_trade_price events

        # Flush timer
        self.last_flush_time = time.time()

        # Lock for thread safety
        self.lock = threading.Lock()

    def add_snapshot(self, event: Dict):
        """Add book snapshot event to buffer."""
        with self.lock:
            self.snapshot_buffer.append(self._flatten_book_event(event))
            self._check_flush()

    def add_update(self, event: Dict):
        """Add price_change event to buffer."""
        with self.lock:
            self.update_buffer.append(self._flatten_price_change_event(event))
            self._check_flush()

    def add_trade(self, event: Dict):
        """Add last_trade_price event to buffer."""
        with self.lock:
            self.trade_buffer.append(self._flatten_trade_event(event))
            self._check_flush()

    def _check_flush(self):
        """Check if we should flush buffers."""
        total_events = (
            len(self.snapshot_buffer) +
            len(self.update_buffer) +
            len(self.trade_buffer)
        )

        time_since_flush = time.time() - self.last_flush_time

        # Flush if batch size reached OR time interval elapsed
        if total_events >= self.batch_size or time_since_flush >= self.flush_interval_s:
            self._flush_all()

    def _flush_all(self):
        """Write all buffers to Parquet files."""
        if self.snapshot_buffer:
            self._flush_buffer(self.snapshot_buffer, 'snapshots')
            self.snapshot_buffer = []

        if self.update_buffer:
            self._flush_buffer(self.update_buffer, 'updates')
            self.update_buffer = []

        if self.trade_buffer:
            self._flush_buffer(self.trade_buffer, 'trades')
            self.trade_buffer = []

        self.last_flush_time = time.time()

    def _flush_buffer(self, buffer: List[Dict], event_type: str):
        """Flush a specific buffer to Parquet."""
        if not buffer:
            return

        # Convert to DataFrame
        df = pl.DataFrame(buffer)

        # Group by market and date
        df = df.with_columns([
            pl.from_epoch('timestamp_ms', time_unit='ms').alias('datetime_utc')
        ])

        df = df.with_columns([
            pl.col('datetime_utc').dt.date().alias('date')
        ])

        # Write partitioned by market and date
        for (market, date), group_df in df.group_by(['market', 'date']):
            # Determine hour for sub-partitioning
            hour = group_df['datetime_utc'][0].hour

            # Generate output path
            output_path = (
                self.output_dir /
                f"market={market}" /
                f"date={date}" /
                f"{event_type}_hour={hour:02d}.parquet"
            )

            output_path.parent.mkdir(parents=True, exist_ok=True)

            # Append if file exists, otherwise create
            if output_path.exists():
                existing = pl.read_parquet(output_path)
                combined = pl.concat([existing, group_df])
                combined.write_parquet(output_path, compression='uncompressed')
            else:
                group_df.write_parquet(output_path, compression='uncompressed')

    def _flatten_book_event(self, event: Dict) -> Dict:
        """Flatten book event to single-row format."""
        bids = event.get('bids', [])
        asks = event.get('asks', [])

        # Extract top 10 levels
        return {
            'timestamp_ms': int(event.get('timestamp', time.time() * 1000)),
            'market': event.get('market'),
            'asset_id': event.get('asset_id'),
            'hash': event.get('hash'),
            # Best bid/ask
            'best_bid_price': float(bids[0]['price']) if bids else None,
            'best_bid_size': float(bids[0]['size']) if bids else None,
            'best_ask_price': float(asks[0]['price']) if asks else None,
            'best_ask_size': float(asks[0]['size']) if asks else None,
            # Depth (total size at top 10 levels)
            'bid_depth_10': sum(float(b['size']) for b in bids[:10]) if bids else 0.0,
            'ask_depth_10': sum(float(a['size']) for a in asks[:10]) if asks else 0.0,
            # Full book (JSON string for detailed analysis)
            'bids_json': json.dumps(bids),
            'asks_json': json.dumps(asks),
        }

    def _flatten_price_change_event(self, event: Dict) -> Dict:
        """Flatten price_change event."""
        price_changes = event.get('price_changes', [])

        # Take first price change (multiple changes per event)
        if price_changes:
            pc = price_changes[0]
            return {
                'timestamp_ms': int(event.get('timestamp', time.time() * 1000)),
                'market': event.get('market'),
                'asset_id': pc.get('asset_id'),
                'price': float(pc.get('price', 0)),
                'size': float(pc.get('size', 0)),
                'side': pc.get('side'),
                'hash': pc.get('hash'),
                'best_bid': float(pc.get('best_bid', 0)),
                'best_ask': float(pc.get('best_ask', 0)),
            }
        return {}

    def _flatten_trade_event(self, event: Dict) -> Dict:
        """Flatten last_trade_price event."""
        return {
            'timestamp_ms': int(event.get('timestamp', time.time() * 1000)),
            'market': event.get('market'),
            'asset_id': event.get('asset_id'),
            'price': float(event.get('price', 0)),
            'size': float(event.get('size', 0)),
            'side': event.get('side'),
            'fee_rate_bps': float(event.get('fee_rate_bps', 0)),
        }

    def force_flush(self):
        """Force flush all buffers (for graceful shutdown)."""
        with self.lock:
            self._flush_all()


# ============================================================================
# WebSocket Order Book Streamer
# ============================================================================

class OrderBookStreamer:
    """
    WebSocket client for streaming Polymarket order book data.
    Handles connection, reconnection, and event routing.
    """

    def __init__(
        self,
        markets: List[str],
        output_dir: str,
        token_ids: Optional[List[str]] = None,
        min_interval_ms: int = DEFAULT_MIN_INTERVAL_MS,
        price_change_threshold: float = DEFAULT_PRICE_CHANGE_THRESHOLD,
        batch_size: int = DEFAULT_BATCH_SIZE,
        flush_interval_s: int = DEFAULT_FLUSH_INTERVAL_S
    ):
        self.markets = markets
        self.token_ids = token_ids or []
        self.output_dir = Path(output_dir)

        # Components
        self.sampler = SmartSampler(min_interval_ms, price_change_threshold)
        self.writer = ParquetBatchWriter(self.output_dir, batch_size, flush_interval_s)

        # WebSocket
        self.ws = None
        self.is_running = False

        # Stats
        self.event_count = {
            'book': 0,
            'price_change': 0,
            'last_trade_price': 0,
            'tick_size_change': 0,
        }

    def start(self):
        """Start WebSocket connection."""
        print(f"🚀 Starting Order Book Streamer...")
        print(f"   Markets: {len(self.markets)}")
        print(f"   Output: {self.output_dir}")
        print(f"   Min interval: {DEFAULT_MIN_INTERVAL_MS}ms")

        self.is_running = True

        self.ws = WebSocketApp(
            WSS_URL,
            on_open=self.on_open,
            on_message=self.on_message,
            on_error=self.on_error,
            on_close=self.on_close
        )

        # Run WebSocket (blocks)
        self.ws.run_forever()

    def on_open(self, ws):
        """Handle WebSocket connection open."""
        print("✅ WebSocket connected")

        # Subscribe to markets
        subscribe_msg = {
            "auth": {},
            "type": "market",
            "markets": self.markets,
            "assets_ids": self.token_ids  # Include token IDs if provided
        }

        ws.send(json.dumps(subscribe_msg))
        print(f"📡 Subscribed to {len(self.markets)} markets")
        if self.token_ids:
            print(f"   With {len(self.token_ids)} token IDs")

    def on_message(self, ws, message):
        """Handle incoming WebSocket message."""
        try:
            event = json.loads(message)
            event_type = event.get('event_type')

            if event_type == 'book':
                self._handle_book_event(event)
            elif event_type == 'price_change':
                self._handle_price_change_event(event)
            elif event_type == 'last_trade_price':
                self._handle_trade_event(event)
            elif event_type == 'tick_size_change':
                self._handle_tick_size_change(event)

        except Exception as e:
            print(f"❌ Error processing message: {e}")

    def _handle_book_event(self, event: Dict):
        """Handle book snapshot event."""
        market = event.get('market')

        # Apply smart sampling
        if self.sampler.should_capture(market, event):
            self.writer.add_snapshot(event)
            self.event_count['book'] += 1

            if self.event_count['book'] % 100 == 0:
                print(f"📊 Snapshots: {self.event_count['book']:,}")

    def _handle_price_change_event(self, event: Dict):
        """Handle price_change event."""
        self.writer.add_update(event)
        self.event_count['price_change'] += 1

    def _handle_trade_event(self, event: Dict):
        """Handle last_trade_price event."""
        self.writer.add_trade(event)
        self.event_count['last_trade_price'] += 1

    def _handle_tick_size_change(self, event: Dict):
        """Handle tick_size_change event."""
        self.event_count['tick_size_change'] += 1
        # Could log these separately if needed

    def on_error(self, ws, error):
        """Handle WebSocket error."""
        print(f"❌ WebSocket error: {error}")

    def on_close(self, ws, close_status_code, close_msg):
        """Handle WebSocket close."""
        print(f"🔌 WebSocket closed: {close_status_code} - {close_msg}")

        # Flush remaining data
        self.writer.force_flush()

        # Auto-reconnect after 5 seconds
        if self.is_running:
            print("🔄 Reconnecting in 5 seconds...")
            time.sleep(5)
            self.start()

    def stop(self):
        """Stop WebSocket connection gracefully."""
        print("\n🛑 Stopping streamer...")
        self.is_running = False

        if self.ws:
            self.ws.close()

        # Flush remaining data
        self.writer.force_flush()

        # Print stats
        print("\n📊 Final Statistics:")
        for event_type, count in self.event_count.items():
            print(f"   {event_type}: {count:,} events")


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='Stream Polymarket order book data via WebSocket')
    parser.add_argument(
        '--markets',
        type=str,
        help='Comma-separated list of market condition IDs'
    )
    parser.add_argument(
        '--market-file',
        type=str,
        help='JSON file with market information'
    )
    parser.add_argument(
        '--filter-pattern',
        type=str,
        help='Regex pattern to filter markets by slug'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default=DEFAULT_OUTPUT_DIR,
        help='Output directory for Parquet files'
    )
    parser.add_argument(
        '--interval-ms',
        type=int,
        default=DEFAULT_MIN_INTERVAL_MS,
        help='Minimum interval between snapshots (milliseconds)'
    )
    parser.add_argument(
        '--price-threshold',
        type=float,
        default=DEFAULT_PRICE_CHANGE_THRESHOLD,
        help='Price change threshold for triggering snapshot (0.0-1.0)'
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help='Number of events per batch write'
    )
    parser.add_argument(
        '--flush-interval-s',
        type=int,
        default=DEFAULT_FLUSH_INTERVAL_S,
        help='Force flush interval (seconds)'
    )

    args = parser.parse_args()

    # Get markets
    markets = []

    if args.markets:
        markets = [m.strip() for m in args.markets.split(',')]
    elif args.market_file:
        # Load from JSON file
        with open(args.market_file, 'r') as f:
            market_data = json.load(f)
            # Assume format: [{"condition_id": "0x...", "slug": "..."}]
            markets = [m['condition_id'] for m in market_data]

            # Apply filter if provided
            if args.filter_pattern:
                import re
                pattern = re.compile(args.filter_pattern, re.IGNORECASE)
                markets = [
                    m['condition_id']
                    for m in market_data
                    if pattern.search(m.get('slug', ''))
                ]
    else:
        print("❌ Error: Must provide --markets or --market-file")
        sys.exit(1)

    if not markets:
        print("❌ Error: No markets to stream")
        sys.exit(1)

    print("=" * 80)
    print(" POLYMARKET ORDER BOOK STREAMER")
    print("=" * 80)
    print(f"\nMarkets: {len(markets)}")
    print(f"Output: {args.output_dir}")
    print(f"Min interval: {args.interval_ms}ms")
    print(f"Price threshold: {args.price_threshold * 100}%")
    print()

    # Create streamer
    streamer = OrderBookStreamer(
        markets=markets,
        output_dir=args.output_dir,
        min_interval_ms=args.interval_ms,
        price_change_threshold=args.price_threshold,
        batch_size=args.batch_size,
        flush_interval_s=args.flush_interval_s
    )

    # Handle graceful shutdown
    import signal

    def signal_handler(sig, frame):
        streamer.stop()
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    # Start streaming
    try:
        streamer.start()
    except KeyboardInterrupt:
        streamer.stop()


if __name__ == '__main__':
    main()
