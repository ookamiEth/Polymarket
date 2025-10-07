#!/usr/bin/env python3
"""
Historical CLOB Order Book Data Collector for Polymarket

This module provides tools to collect historical order book data through:
1. Real-time WebSocket collection (going forward)
2. Historical trade reconstruction (backfill)

The collector stores timestamped order book snapshots and updates that can be
used for market microstructure analysis, backtesting, and research.

Usage:
    # Real-time collection
    collector = OrderBookCollector(output_dir="data/orderbooks")
    await collector.collect_realtime(
        markets=["0xdd31ce870876bb410d71f29e3dfea77910bc3effb81a0fb8dc8c282d5968aa5a"],
        token_ids=["21742633143463906290569050155826241533067272736897614950488156847949938836455"]
    )

    # Historical backfill
    collector.collect_historical_trades(
        market="0xdd31ce...",
        start_time="2024-01-01",
        end_time="2024-01-31"
    )
"""

import asyncio
import websockets
import json
import requests
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Dict, Optional, Any
from collections import defaultdict
import time


class OrderBookCollector:
    """Collects and stores historical CLOB order book data"""

    def __init__(self, output_dir: str = "data/orderbooks"):
        """
        Initialize the collector

        Args:
            output_dir: Directory to store collected data
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # WebSocket endpoints
        self.ws_url = "wss://ws-subscriptions-clob.polymarket.com/ws/market"

        # REST endpoints
        self.clob_base = "https://clob.polymarket.com"
        self.data_api_base = "https://data-api.polymarket.com"

        # Storage for order book state
        self.orderbooks = defaultdict(dict)  # {token_id: {price: size}}

        # Data buffers
        self.book_snapshots = []
        self.price_changes = []
        self.trades = []

        print(f"OrderBookCollector initialized")
        print(f"Output directory: {self.output_dir}")

    async def collect_realtime(
        self,
        markets: List[str],
        token_ids: List[str],
        duration_minutes: Optional[int] = None,
        save_interval_seconds: int = 60
    ):
        """
        Collect real-time order book data via WebSocket

        Args:
            markets: List of market IDs (condition IDs) to monitor
            token_ids: List of token IDs to monitor
            duration_minutes: How long to collect (None = indefinite)
            save_interval_seconds: How often to save data to disk
        """
        print(f"\n{'='*80}")
        print("REAL-TIME ORDER BOOK COLLECTION")
        print(f"{'='*80}")
        print(f"Markets: {len(markets)}")
        print(f"Tokens: {len(token_ids)}")
        print(f"Duration: {duration_minutes if duration_minutes else 'Indefinite'} minutes")
        print(f"Save interval: {save_interval_seconds}s")

        start_time = time.time()
        last_save = start_time

        try:
            async with websockets.connect(self.ws_url) as websocket:
                # Subscribe to markets
                subscribe_msg = {
                    "auth": {},
                    "markets": markets,
                    "assets_ids": token_ids,
                    "type": "market"
                }

                await websocket.send(json.dumps(subscribe_msg))
                print(f"\n✓ Connected and subscribed")
                print(f"Listening for updates... (Ctrl+C to stop)\n")

                message_count = 0

                while True:
                    # Check if duration limit reached
                    if duration_minutes:
                        elapsed = (time.time() - start_time) / 60
                        if elapsed >= duration_minutes:
                            print(f"\nDuration limit reached ({duration_minutes} min)")
                            break

                    # Receive message
                    message = await websocket.recv()
                    data = json.loads(message)

                    # Process message
                    self._process_websocket_message(data)
                    message_count += 1

                    # Periodic save
                    if time.time() - last_save >= save_interval_seconds:
                        self._save_buffers()
                        last_save = time.time()
                        print(f"[{datetime.now().strftime('%H:%M:%S')}] "
                              f"Saved {message_count} messages | "
                              f"Snapshots: {len(self.book_snapshots)} | "
                              f"Changes: {len(self.price_changes)} | "
                              f"Trades: {len(self.trades)}")

        except websockets.exceptions.WebSocketException as e:
            print(f"\nWebSocket error: {e}")
        except KeyboardInterrupt:
            print(f"\n\nStopped by user")
        finally:
            # Final save
            print(f"\nFinal save...")
            self._save_buffers()
            print(f"✓ Collection completed")
            print(f"  Total messages: {message_count}")
            print(f"  Book snapshots: {len(self.book_snapshots)}")
            print(f"  Price changes: {len(self.price_changes)}")
            print(f"  Trades: {len(self.trades)}")

    def _process_websocket_message(self, data: Dict[str, Any]):
        """Process incoming WebSocket message"""
        event_type = data.get("event_type")
        timestamp = datetime.now().isoformat()

        if event_type == "book":
            # Full order book snapshot
            snapshot = {
                "timestamp": timestamp,
                "ws_timestamp": data.get("timestamp"),
                "asset_id": data.get("asset_id"),
                "market": data.get("market"),
                "hash": data.get("hash"),
                "bids": data.get("bids", []),
                "asks": data.get("asks", [])
            }
            self.book_snapshots.append(snapshot)

        elif event_type == "price_change":
            # Incremental price level update
            for change in data.get("price_changes", []):
                price_change = {
                    "timestamp": timestamp,
                    "ws_timestamp": data.get("timestamp"),
                    "market": data.get("market"),
                    "asset_id": change.get("asset_id"),
                    "side": change.get("side"),
                    "price": change.get("price"),
                    "size": change.get("size"),
                    "hash": change.get("hash"),
                    "best_bid": change.get("best_bid"),
                    "best_ask": change.get("best_ask")
                }
                self.price_changes.append(price_change)

        elif event_type == "last_trade_price":
            # Trade execution
            trade = {
                "timestamp": timestamp,
                "ws_timestamp": data.get("timestamp"),
                "asset_id": data.get("asset_id"),
                "market": data.get("market"),
                "side": data.get("side"),
                "price": data.get("price"),
                "size": data.get("size"),
                "fee_rate_bps": data.get("fee_rate_bps")
            }
            self.trades.append(trade)

    def _save_buffers(self):
        """Save collected data to disk"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Save book snapshots
        if self.book_snapshots:
            df = pd.DataFrame(self.book_snapshots)
            filepath = self.output_dir / f"book_snapshots_{timestamp}.parquet"
            df.to_parquet(filepath, index=False)
            self.book_snapshots = []

        # Save price changes
        if self.price_changes:
            df = pd.DataFrame(self.price_changes)
            filepath = self.output_dir / f"price_changes_{timestamp}.parquet"
            df.to_parquet(filepath, index=False)
            self.price_changes = []

        # Save trades
        if self.trades:
            df = pd.DataFrame(self.trades)
            filepath = self.output_dir / f"trades_{timestamp}.parquet"
            df.to_parquet(filepath, index=False)
            self.trades = []

    def collect_historical_trades(
        self,
        market: Optional[str] = None,
        start_timestamp: Optional[int] = None,
        end_timestamp: Optional[int] = None,
        limit: int = 10000
    ) -> pd.DataFrame:
        """
        Collect historical trades from Data API

        Args:
            market: Market ID (condition ID) to filter by
            start_timestamp: Start time (Unix timestamp)
            end_timestamp: End time (Unix timestamp)
            limit: Maximum trades to fetch per request

        Returns:
            DataFrame with historical trades
        """
        print(f"\n{'='*80}")
        print("HISTORICAL TRADES COLLECTION")
        print(f"{'='*80}")
        print(f"Market: {market[:30]}..." if market else "All markets")
        print(f"Time range: {start_timestamp} - {end_timestamp}")

        all_trades = []
        offset = 0

        while True:
            # Build request
            params = {
                "limit": limit,
                "offset": offset
            }
            if market:
                params["market"] = market
            if start_timestamp:
                params["start"] = start_timestamp
            if end_timestamp:
                params["end"] = end_timestamp

            # Fetch trades
            print(f"\nFetching offset {offset}...", end=" ")
            url = f"{self.data_api_base}/trades"
            response = requests.get(url, params=params)

            if response.status_code != 200:
                print(f"Error: {response.status_code}")
                break

            trades = response.json()
            print(f"Got {len(trades)} trades")

            if not trades:
                break

            all_trades.extend(trades)
            offset += len(trades)

            # Rate limiting
            time.sleep(0.5)

            # Stop if we got fewer than limit (last page)
            if len(trades) < limit:
                break

        # Convert to DataFrame
        df = pd.DataFrame(all_trades)

        if not df.empty:
            # Save to disk
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filepath = self.output_dir / f"historical_trades_{timestamp}.parquet"
            df.to_parquet(filepath, index=False)
            print(f"\n✓ Saved {len(df)} trades to {filepath}")

        print(f"{'='*80}\n")
        return df

    def reconstruct_orderbook_from_trades(
        self,
        trades_df: pd.DataFrame,
        token_id: str
    ) -> List[Dict]:
        """
        Reconstruct approximate order book states from trade history

        Note: This is an approximation - actual book state may differ
        as we don't have all order placements/cancellations

        Args:
            trades_df: DataFrame with historical trades
            token_id: Token ID to reconstruct for

        Returns:
            List of reconstructed book states with timestamps
        """
        print(f"\nReconstructing order book from {len(trades_df)} trades...")

        # Filter trades for this token
        token_trades = trades_df[trades_df['asset'] == token_id].copy()
        token_trades = token_trades.sort_values('timestamp')

        reconstructed_states = []

        # Simple reconstruction: infer book from trade flow
        for idx, trade in token_trades.iterrows():
            state = {
                "timestamp": trade['timestamp'],
                "price": float(trade['price']),
                "size": float(trade['size']),
                "side": trade['side'],
                "transaction_hash": trade.get('transactionHash')
            }
            reconstructed_states.append(state)

        print(f"✓ Reconstructed {len(reconstructed_states)} states")
        return reconstructed_states

    def get_orderbook_summary(self, token_id: str) -> Dict:
        """
        Get current order book summary (snapshot only, not historical)

        Args:
            token_id: Token ID to query

        Returns:
            Current order book summary
        """
        url = f"{self.clob_base}/book"
        params = {"token_id": token_id}

        response = requests.get(url, params=params)

        if response.status_code == 200:
            return response.json()
        else:
            print(f"Error fetching orderbook: {response.status_code}")
            return {}


async def main():
    """Example usage"""
    collector = OrderBookCollector(output_dir="data/orderbooks")

    # Example market and token IDs (Trump Yes token)
    market_id = "0xdd31ce870876bb410d71f29e3dfea77910bc3effb81a0fb8dc8c282d5968aa5a"
    token_id = "21742633143463906290569050155826241533067272736897614950488156847949938836455"

    print("\n" + "="*80)
    print("POLYMARKET HISTORICAL CLOB DATA COLLECTOR")
    print("="*80)

    # Option 1: Collect real-time data for 5 minutes
    print("\n1. Collecting real-time order book data (5 minutes)...")
    await collector.collect_realtime(
        markets=[market_id],
        token_ids=[token_id],
        duration_minutes=5,
        save_interval_seconds=30
    )

    # Option 2: Fetch historical trades
    print("\n2. Fetching historical trades...")
    end_time = int(time.time())
    start_time = end_time - (7 * 24 * 60 * 60)  # Last 7 days

    trades_df = collector.collect_historical_trades(
        market=market_id,
        start_timestamp=start_time,
        end_timestamp=end_time,
        limit=1000
    )

    if not trades_df.empty:
        print(f"\nTrades summary:")
        print(f"  Total trades: {len(trades_df)}")
        print(f"  Date range: {trades_df['timestamp'].min()} to {trades_df['timestamp'].max()}")
        print(f"  Columns: {list(trades_df.columns)}")

    # Option 3: Get current orderbook snapshot
    print("\n3. Fetching current order book snapshot...")
    current_book = collector.get_orderbook_summary(token_id)
    if current_book:
        print(f"  Current snapshot retrieved")
        print(f"  Timestamp: {current_book.get('timestamp')}")
        print(f"  Hash: {current_book.get('hash', '')[:20]}...")

    print("\n" + "="*80)
    print("COLLECTION COMPLETE")
    print("="*80)
    print(f"\nData saved to: {collector.output_dir}")
    print("\nFiles created:")
    for f in sorted(collector.output_dir.glob("*.parquet")):
        print(f"  - {f.name}")


if __name__ == "__main__":
    print("\nStarting Historical CLOB Data Collector...")
    print("(Install dependencies: uv pip install websockets pandas pyarrow requests)\n")

    try:
        asyncio.run(main())
    except ImportError as e:
        print(f"Error: Missing dependency - {e}")
        print("Install with: uv pip install websockets pandas pyarrow requests")
    except KeyboardInterrupt:
        print("\n\nStopped by user")
