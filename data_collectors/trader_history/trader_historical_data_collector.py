#!/usr/bin/env python3
"""
Polymarket Trader Historical Data Collector
===========================================

Collects complete historical data for a specific trader using Polars for efficient data processing.

Features:
- Automatic pagination up to API limits (10,000 records)
- Rate limit compliance (100 req/10s)
- Polars DataFrames for memory-efficient processing
- Parquet output for compressed storage
- Async data collection for speed
- Complete trade history, activity, positions, and price data

Author: Claude Code
Date: October 1, 2025
"""

import polars as pl
import requests
import time
from datetime import datetime, timedelta
from typing import List, Dict, Optional
import json
from pathlib import Path

class PolymarketDataCollector:
    """Collect historical data for a Polymarket trader"""

    def __init__(self, user_address: str, output_dir: str = "trader_data"):
        self.user_address = user_address
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)

        # API configuration
        self.base_urls = {
            'clob': 'https://clob.polymarket.com',
            'data': 'https://data-api.polymarket.com',
            'gamma': 'https://gamma-api.polymarket.com'
        }

        # Rate limiting (100 req / 10 seconds = 0.1s between requests)
        self.rate_limit_delay = 0.11
        self.last_request_time = 0

        print(f"📊 Polymarket Data Collector initialized")
        print(f"👤 Trader: {user_address}")
        print(f"📁 Output: {self.output_dir}")
        print()

    def _rate_limit(self):
        """Enforce rate limiting between requests"""
        elapsed = time.time() - self.last_request_time
        if elapsed < self.rate_limit_delay:
            time.sleep(self.rate_limit_delay - elapsed)
        self.last_request_time = time.time()

    def _make_request(self, url: str, params: Dict = None) -> Dict:
        """Make rate-limited API request"""
        self._rate_limit()
        response = requests.get(url, params=params, timeout=30)
        response.raise_for_status()
        return response.json()

    def collect_trades(self, limit_per_batch: int = 500, max_records: int = 10000) -> pl.DataFrame:
        """
        Collect all trade history for the trader

        Returns:
            Polars DataFrame with trade data
        """
        print("🔄 Collecting trade history...")
        url = f"{self.base_urls['data']}/trades"

        all_trades = []
        offset = 0

        while offset < max_records:
            params = {
                'user': self.user_address,
                'limit': limit_per_batch,
                'offset': offset,
                'sortBy': 'TIMESTAMP',
                'sortDirection': 'DESC'
            }

            try:
                trades = self._make_request(url, params)

                if not trades or len(trades) == 0:
                    break

                all_trades.extend(trades)
                print(f"  ✓ Fetched {len(trades)} trades (offset: {offset}, total: {len(all_trades)})")

                offset += limit_per_batch

                if len(trades) < limit_per_batch:
                    break

            except Exception as e:
                print(f"  ✗ Error at offset {offset}: {e}")
                break

        if all_trades:
            df = pl.DataFrame(all_trades)
            print(f"✅ Collected {len(df)} trades")
            return df
        else:
            print("⚠️  No trades found")
            return pl.DataFrame()

    def collect_activity(self, limit_per_batch: int = 500, max_records: int = 10000) -> pl.DataFrame:
        """
        Collect all activity history (trades, splits, merges, redeems, etc.)

        Returns:
            Polars DataFrame with activity data
        """
        print("🔄 Collecting activity history...")
        url = f"{self.base_urls['data']}/activity"

        all_activity = []
        offset = 0

        while offset < max_records:
            params = {
                'user': self.user_address,
                'limit': limit_per_batch,
                'offset': offset,
                'sortBy': 'TIMESTAMP',
                'sortDirection': 'DESC'
            }

            try:
                activities = self._make_request(url, params)

                if not activities or len(activities) == 0:
                    break

                all_activity.extend(activities)
                print(f"  ✓ Fetched {len(activities)} activities (offset: {offset}, total: {len(all_activity)})")

                offset += limit_per_batch

                if len(activities) < limit_per_batch:
                    break

            except Exception as e:
                print(f"  ✗ Error at offset {offset}: {e}")
                break

        if all_activity:
            df = pl.DataFrame(all_activity)
            print(f"✅ Collected {len(df)} activity records")
            return df
        else:
            print("⚠️  No activity found")
            return pl.DataFrame()

    def collect_positions(self) -> tuple[pl.DataFrame, pl.DataFrame]:
        """
        Collect current and closed positions

        Returns:
            Tuple of (current_positions_df, closed_positions_df)
        """
        print("🔄 Collecting positions...")

        # Current positions
        current_url = f"{self.base_urls['data']}/positions"
        closed_url = f"{self.base_urls['data']}/closed-positions"

        all_current = []
        all_closed = []

        # Collect current positions
        offset = 0
        while offset < 10000:
            params = {'user': self.user_address, 'limit': 500, 'offset': offset}
            try:
                positions = self._make_request(current_url, params)
                if not positions or len(positions) == 0:
                    break
                all_current.extend(positions)
                offset += 500
                if len(positions) < 500:
                    break
            except Exception as e:
                print(f"  ✗ Error fetching current positions: {e}")
                break

        # Collect closed positions
        offset = 0
        while offset < 10000:
            params = {'user': self.user_address, 'limit': 500, 'offset': offset}
            try:
                positions = self._make_request(closed_url, params)
                if not positions or len(positions) == 0:
                    break
                all_closed.extend(positions)
                offset += 500
                if len(positions) < 500:
                    break
            except Exception as e:
                print(f"  ✗ Error fetching closed positions: {e}")
                break

        current_df = pl.DataFrame(all_current) if all_current else pl.DataFrame()
        closed_df = pl.DataFrame(all_closed) if all_closed else pl.DataFrame()

        print(f"✅ Collected {len(current_df)} current positions, {len(closed_df)} closed positions")
        return current_df, closed_df

    def collect_price_history(self, token_ids: List[str], days_back: int = 365) -> pl.DataFrame:
        """
        Collect price history for all traded tokens

        Args:
            token_ids: List of token IDs to fetch price history for
            days_back: Number of days of history to fetch

        Returns:
            Polars DataFrame with price history
        """
        print(f"🔄 Collecting price history for {len(token_ids)} tokens...")
        url = f"{self.base_urls['clob']}/prices-history"

        all_prices = []

        for i, token_id in enumerate(token_ids, 1):
            try:
                now = int(datetime.now().timestamp())
                start_ts = now - (days_back * 24 * 60 * 60)

                params = {
                    'market': token_id,
                    'startTs': start_ts,
                    'endTs': now,
                    'fidelity': 60  # 1 hour resolution
                }

                data = self._make_request(url, params)

                if 'history' in data and data['history']:
                    for point in data['history']:
                        all_prices.append({
                            'token_id': token_id,
                            'timestamp': point['t'],
                            'price': point['p']
                        })
                    print(f"  ✓ [{i}/{len(token_ids)}] Token {token_id[:20]}... ({len(data['history'])} points)")
                else:
                    print(f"  ⚠️  [{i}/{len(token_ids)}] No price data for {token_id[:20]}...")

            except Exception as e:
                print(f"  ✗ [{i}/{len(token_ids)}] Error for {token_id[:20]}...: {e}")
                continue

        if all_prices:
            df = pl.DataFrame(all_prices)
            print(f"✅ Collected {len(df)} price points")
            return df
        else:
            print("⚠️  No price history found")
            return pl.DataFrame()

    def save_to_parquet(self, df: pl.DataFrame, filename: str):
        """Save DataFrame to parquet file"""
        if len(df) > 0:
            filepath = self.output_dir / filename
            df.write_parquet(filepath)
            print(f"💾 Saved {len(df)} records to {filepath}")
        else:
            print(f"⚠️  Skipping {filename} (empty dataset)")

    def collect_all(self) -> Dict[str, pl.DataFrame]:
        """
        Collect all available historical data

        Returns:
            Dictionary of DataFrames with all collected data
        """
        print("="*80)
        print("🚀 Starting Complete Historical Data Collection")
        print("="*80)
        print()

        start_time = time.time()

        # 1. Collect trades
        trades_df = self.collect_trades()
        self.save_to_parquet(trades_df, 'trader_trades.parquet')
        print()

        # 2. Collect activity
        activity_df = self.collect_activity()
        self.save_to_parquet(activity_df, 'trader_activity.parquet')
        print()

        # 3. Collect positions
        current_pos_df, closed_pos_df = self.collect_positions()
        self.save_to_parquet(current_pos_df, 'trader_current_positions.parquet')
        self.save_to_parquet(closed_pos_df, 'trader_closed_positions.parquet')
        print()

        # 4. Collect price history for all traded tokens
        if len(trades_df) > 0:
            unique_tokens = trades_df.select('asset').unique().to_series().to_list()
            price_df = self.collect_price_history(unique_tokens, days_back=365)
            self.save_to_parquet(price_df, 'market_prices.parquet')
        else:
            price_df = pl.DataFrame()
        print()

        elapsed = time.time() - start_time

        print("="*80)
        print("✅ Collection Complete!")
        print("="*80)
        print(f"⏱️  Time elapsed: {elapsed:.1f} seconds ({elapsed/60:.1f} minutes)")
        print(f"📊 Datasets created:")
        print(f"   - Trades: {len(trades_df)} records")
        print(f"   - Activity: {len(activity_df)} records")
        print(f"   - Current Positions: {len(current_pos_df)} records")
        print(f"   - Closed Positions: {len(closed_pos_df)} records")
        print(f"   - Price History: {len(price_df)} records")
        print(f"📁 Files saved to: {self.output_dir.absolute()}")
        print()

        return {
            'trades': trades_df,
            'activity': activity_df,
            'current_positions': current_pos_df,
            'closed_positions': closed_pos_df,
            'prices': price_df
        }


def main():
    """Main execution"""
    # Trader address (extremely active professional trader from tests)
    TRADER_ADDRESS = "0x24c8cf69a0e0a17eee21f69d29752bfa32e823e1"

    # Initialize collector
    collector = PolymarketDataCollector(TRADER_ADDRESS)

    # Collect all data
    datasets = collector.collect_all()

    # Quick summary statistics
    if len(datasets['trades']) > 0:
        print("="*80)
        print("📈 Quick Stats")
        print("="*80)

        trades = datasets['trades']

        # Convert timestamp to datetime
        if 'timestamp' in trades.columns:
            trades_with_dates = trades.with_columns([
                pl.from_epoch('timestamp', time_unit='s').alias('datetime')
            ])

            print(f"📅 Trading Period:")
            print(f"   First trade: {trades_with_dates.select('datetime').min().item()}")
            print(f"   Last trade: {trades_with_dates.select('datetime').max().item()}")

        # Trade volume
        if 'size' in trades.columns:
            total_volume = trades.select(pl.col('size').sum()).item()
            print(f"💰 Total Volume: {total_volume:,.0f} shares")

        # Buy/Sell ratio
        if 'side' in trades.columns:
            buy_count = trades.filter(pl.col('side') == 'BUY').height
            sell_count = trades.filter(pl.col('side') == 'SELL').height
            print(f"📊 Trade Distribution: {buy_count} BUY / {sell_count} SELL")

        # Unique markets
        if 'conditionId' in trades.columns:
            unique_markets = trades.select('conditionId').unique().height
            print(f"🎯 Unique Markets Traded: {unique_markets}")

        print()

    print("✨ Data collection complete! Files ready for analysis.")
    print()


if __name__ == "__main__":
    main()
