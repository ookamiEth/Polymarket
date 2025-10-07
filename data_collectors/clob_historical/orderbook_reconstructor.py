#!/usr/bin/env python3
"""
Order Book State Reconstruction from Historical Data

This module reconstructs historical order book states from:
1. Historical trade data
2. Saved WebSocket snapshots
3. Price change events

The reconstructor maintains a time-series of order book states that can be
used for backtesting and analysis.

Usage:
    reconstructor = OrderBookReconstructor()

    # Load collected data
    reconstructor.load_snapshots("data/orderbooks/book_snapshots_*.parquet")
    reconstructor.load_price_changes("data/orderbooks/price_changes_*.parquet")

    # Reconstruct state at specific time
    book_state = reconstructor.get_state_at_time(timestamp)

    # Get full time series
    timeseries = reconstructor.build_timeseries(interval_seconds=1)
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
from collections import defaultdict
import glob


class OrderBookState:
    """Represents order book state at a point in time"""

    def __init__(self, timestamp: str):
        self.timestamp = timestamp
        self.bids = {}  # {price: size}
        self.asks = {}  # {price: size}
        self.hash = None
        self.asset_id = None
        self.market = None

    def update_level(self, side: str, price: float, size: float):
        """Update a price level"""
        book = self.bids if side == "BUY" else self.asks

        if size == 0 or size == "0":
            # Remove level
            book.pop(price, None)
        else:
            # Update level
            book[price] = float(size)

    def get_best_bid(self) -> Optional[Tuple[float, float]]:
        """Get best bid (price, size)"""
        if not self.bids:
            return None
        best_price = max(self.bids.keys())
        return (best_price, self.bids[best_price])

    def get_best_ask(self) -> Optional[Tuple[float, float]]:
        """Get best ask (price, size)"""
        if not self.asks:
            return None
        best_price = min(self.asks.keys())
        return (best_price, self.asks[best_price])

    def get_spread(self) -> Optional[float]:
        """Get bid-ask spread"""
        bid = self.get_best_bid()
        ask = self.get_best_ask()

        if bid and ask:
            return ask[0] - bid[0]
        return None

    def get_mid_price(self) -> Optional[float]:
        """Get mid price"""
        bid = self.get_best_bid()
        ask = self.get_best_ask()

        if bid and ask:
            return (bid[0] + ask[0]) / 2
        return None

    def get_depth(self, levels: int = 5) -> Dict:
        """Get depth of book (top N levels)"""
        sorted_bids = sorted(self.bids.items(), key=lambda x: x[0], reverse=True)
        sorted_asks = sorted(self.asks.items(), key=lambda x: x[0])

        return {
            "bids": sorted_bids[:levels],
            "asks": sorted_asks[:levels]
        }

    def to_dict(self) -> Dict:
        """Convert to dictionary"""
        return {
            "timestamp": self.timestamp,
            "asset_id": self.asset_id,
            "market": self.market,
            "hash": self.hash,
            "bids": list(self.bids.items()),
            "asks": list(self.asks.items()),
            "best_bid": self.get_best_bid(),
            "best_ask": self.get_best_ask(),
            "spread": self.get_spread(),
            "mid_price": self.get_mid_price()
        }


class OrderBookReconstructor:
    """Reconstructs order book states from historical data"""

    def __init__(self):
        self.snapshots = []  # List of full book snapshots
        self.price_changes = []  # List of incremental updates
        self.trades = []  # List of trades
        self.states = {}  # {timestamp: OrderBookState}

    def load_snapshots(self, pattern: str):
        """
        Load book snapshots from parquet files

        Args:
            pattern: Glob pattern for snapshot files
        """
        files = glob.glob(pattern)
        print(f"Loading snapshots from {len(files)} files...")

        for file in files:
            df = pd.read_parquet(file)
            self.snapshots.extend(df.to_dict('records'))

        print(f"✓ Loaded {len(self.snapshots)} snapshots")

    def load_price_changes(self, pattern: str):
        """
        Load price change events from parquet files

        Args:
            pattern: Glob pattern for price change files
        """
        files = glob.glob(pattern)
        print(f"Loading price changes from {len(files)} files...")

        for file in files:
            df = pd.read_parquet(file)
            self.price_changes.extend(df.to_dict('records'))

        # Sort by timestamp
        self.price_changes.sort(key=lambda x: x['timestamp'])

        print(f"✓ Loaded {len(self.price_changes)} price changes")

    def load_trades(self, pattern: str):
        """
        Load trades from parquet files

        Args:
            pattern: Glob pattern for trade files
        """
        files = glob.glob(pattern)
        print(f"Loading trades from {len(files)} files...")

        for file in files:
            df = pd.read_parquet(file)
            self.trades.extend(df.to_dict('records'))

        # Sort by timestamp
        self.trades.sort(key=lambda x: x['timestamp'])

        print(f"✓ Loaded {len(self.trades)} trades")

    def build_timeseries(
        self,
        asset_id: str,
        start_time: Optional[str] = None,
        end_time: Optional[str] = None,
        interval_seconds: int = 1
    ) -> List[OrderBookState]:
        """
        Build time series of order book states

        Args:
            asset_id: Asset ID to build for
            start_time: Start timestamp (ISO format)
            end_time: End timestamp (ISO format)
            interval_seconds: Sampling interval

        Returns:
            List of OrderBookState objects
        """
        print(f"\n{'='*80}")
        print(f"BUILDING ORDER BOOK TIME SERIES")
        print(f"{'='*80}")
        print(f"Asset ID: {asset_id[:30]}...")
        print(f"Interval: {interval_seconds}s")

        # Filter data for this asset
        asset_snapshots = [s for s in self.snapshots if s.get('asset_id') == asset_id]
        asset_changes = [c for c in self.price_changes if c.get('asset_id') == asset_id]

        print(f"Snapshots: {len(asset_snapshots)}")
        print(f"Changes: {len(asset_changes)}")

        if not asset_snapshots and not asset_changes:
            print("No data found for this asset")
            return []

        # Combine and sort all events by timestamp
        all_events = []

        for snapshot in asset_snapshots:
            all_events.append({
                'timestamp': snapshot['timestamp'],
                'type': 'snapshot',
                'data': snapshot
            })

        for change in asset_changes:
            all_events.append({
                'timestamp': change['timestamp'],
                'type': 'change',
                'data': change
            })

        all_events.sort(key=lambda x: x['timestamp'])

        # Build states
        states = []
        current_state = None

        print(f"\nProcessing {len(all_events)} events...")

        for event in all_events:
            if event['type'] == 'snapshot':
                # Full snapshot - create new state
                data = event['data']
                state = OrderBookState(timestamp=data['timestamp'])
                state.asset_id = data['asset_id']
                state.market = data.get('market')
                state.hash = data.get('hash')

                # Load bids and asks
                for bid in data.get('bids', []):
                    state.bids[float(bid['price'])] = float(bid['size'])

                for ask in data.get('asks', []):
                    state.asks[float(ask['price'])] = float(ask['size'])

                current_state = state
                states.append(state)

            elif event['type'] == 'change' and current_state:
                # Incremental update - create new state from previous
                data = event['data']
                state = OrderBookState(timestamp=data['timestamp'])
                state.asset_id = current_state.asset_id
                state.market = current_state.market

                # Copy previous state
                state.bids = current_state.bids.copy()
                state.asks = current_state.asks.copy()

                # Apply update
                price = float(data['price'])
                size = float(data['size']) if data['size'] else 0
                side = data['side']

                state.update_level(side, price, size)

                current_state = state
                states.append(state)

        print(f"✓ Built {len(states)} states")
        return states

    def get_state_at_time(
        self,
        asset_id: str,
        timestamp: str
    ) -> Optional[OrderBookState]:
        """
        Get order book state at specific timestamp

        Args:
            asset_id: Asset ID
            timestamp: ISO timestamp

        Returns:
            OrderBookState at that time, or None
        """
        # Build full time series if not already done
        if asset_id not in self.states:
            states = self.build_timeseries(asset_id)
            self.states[asset_id] = states

        # Find closest state before or at timestamp
        states = self.states[asset_id]

        for state in reversed(states):
            if state.timestamp <= timestamp:
                return state

        return None

    def analyze_book_statistics(
        self,
        states: List[OrderBookState]
    ) -> pd.DataFrame:
        """
        Compute statistics from book states

        Args:
            states: List of OrderBookState objects

        Returns:
            DataFrame with statistics
        """
        print(f"\nComputing statistics for {len(states)} states...")

        stats = []

        for state in states:
            best_bid = state.get_best_bid()
            best_ask = state.get_best_ask()

            stat = {
                'timestamp': state.timestamp,
                'best_bid_price': best_bid[0] if best_bid else None,
                'best_bid_size': best_bid[1] if best_bid else None,
                'best_ask_price': best_ask[0] if best_ask else None,
                'best_ask_size': best_ask[1] if best_ask else None,
                'spread': state.get_spread(),
                'mid_price': state.get_mid_price(),
                'bid_levels': len(state.bids),
                'ask_levels': len(state.asks),
                'total_bid_volume': sum(state.bids.values()),
                'total_ask_volume': sum(state.asks.values())
            }
            stats.append(stat)

        df = pd.DataFrame(stats)
        print(f"✓ Statistics computed")
        return df

    def export_timeseries(
        self,
        states: List[OrderBookState],
        output_file: str,
        format: str = 'parquet'
    ):
        """
        Export time series to file

        Args:
            states: List of OrderBookState objects
            output_file: Output file path
            format: 'parquet' or 'csv'
        """
        print(f"\nExporting {len(states)} states to {output_file}...")

        # Convert states to records
        records = [state.to_dict() for state in states]
        df = pd.DataFrame(records)

        # Save
        if format == 'parquet':
            df.to_parquet(output_file, index=False)
        elif format == 'csv':
            df.to_csv(output_file, index=False)

        print(f"✓ Exported to {output_file}")


def main():
    """Example usage"""
    reconstructor = OrderBookReconstructor()

    # Load data
    print("\n" + "="*80)
    print("ORDER BOOK RECONSTRUCTION")
    print("="*80)

    data_dir = Path("data/orderbooks")

    if not data_dir.exists():
        print(f"\nNo data found in {data_dir}")
        print("Run orderbook_collector.py first to collect data")
        return

    # Load all collected data
    reconstructor.load_snapshots(str(data_dir / "book_snapshots_*.parquet"))
    reconstructor.load_price_changes(str(data_dir / "price_changes_*.parquet"))
    reconstructor.load_trades(str(data_dir / "trades_*.parquet"))

    if not reconstructor.snapshots and not reconstructor.price_changes:
        print("\nNo snapshots or price changes found")
        return

    # Get asset ID from first snapshot
    asset_id = reconstructor.snapshots[0]['asset_id'] if reconstructor.snapshots else \
               reconstructor.price_changes[0]['asset_id']

    print(f"\nReconstructing for asset: {asset_id[:30]}...")

    # Build time series
    states = reconstructor.build_timeseries(
        asset_id=asset_id,
        interval_seconds=1
    )

    if states:
        print(f"\nTime series built:")
        print(f"  States: {len(states)}")
        print(f"  Start: {states[0].timestamp}")
        print(f"  End: {states[-1].timestamp}")

        # Compute statistics
        stats_df = reconstructor.analyze_book_statistics(states)

        print(f"\nStatistics summary:")
        print(stats_df.describe())

        # Export
        output_file = data_dir / "reconstructed_timeseries.parquet"
        reconstructor.export_timeseries(states, str(output_file))

        stats_file = data_dir / "book_statistics.parquet"
        stats_df.to_parquet(stats_file, index=False)
        print(f"✓ Statistics saved to {stats_file}")

        # Show sample states
        print(f"\n{'='*80}")
        print("SAMPLE BOOK STATES")
        print(f"{'='*80}")

        for i, state in enumerate(states[:3]):
            print(f"\nState {i+1}: {state.timestamp}")
            depth = state.get_depth(levels=3)
            print(f"  Best Bid: {state.get_best_bid()}")
            print(f"  Best Ask: {state.get_best_ask()}")
            print(f"  Spread: {state.get_spread()}")
            print(f"  Mid: {state.get_mid_price()}")

    print("\n" + "="*80)
    print("RECONSTRUCTION COMPLETE")
    print("="*80)


if __name__ == "__main__":
    main()
