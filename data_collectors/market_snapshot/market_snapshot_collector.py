#!/usr/bin/env python3
"""
Polymarket Complete Market Snapshot Collector
==============================================

Collects ALL markets (past and present) from Polymarket using the Gamma API.
Unlike the /trades endpoint, /markets has no pagination limit.

Features:
- Unlimited pagination (no 10K offset limit)
- Collects both active and closed markets
- Rate limit compliance (100 req/10s)
- Polars DataFrames for efficient processing
- Parquet output for compressed storage

Author: Claude Code
Date: October 1, 2025
"""

import polars as pl
import requests
import time
from datetime import datetime
from typing import List, Dict, Optional
from pathlib import Path


class MarketSnapshotCollector:
    """Collect complete snapshot of all Polymarket markets"""

    def __init__(self, output_dir: str = "market_snapshot"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)

        # API configuration
        self.base_url = 'https://gamma-api.polymarket.com'

        # Rate limiting (100 req / 10 seconds = 0.1s between requests)
        self.rate_limit_delay = 0.11
        self.last_request_time = 0

        print(f"📊 Polymarket Market Snapshot Collector")
        print(f"📁 Output: {self.output_dir}")
        print()

    def _rate_limit(self):
        """Enforce rate limiting between requests"""
        elapsed = time.time() - self.last_request_time
        if elapsed < self.rate_limit_delay:
            time.sleep(self.rate_limit_delay - elapsed)
        self.last_request_time = time.time()

    def _make_request(self, url: str, params: Dict = None) -> List[Dict]:
        """Make rate-limited API request"""
        self._rate_limit()
        response = requests.get(url, params=params, timeout=30)
        response.raise_for_status()
        return response.json()

    def collect_all_markets(
        self,
        limit_per_batch: int = 500,  # API max is 500
        closed: Optional[bool] = None,
        max_markets: Optional[int] = None
    ) -> pl.DataFrame:
        """
        Collect all markets with unlimited pagination.

        Args:
            limit_per_batch: Number of markets per API request (default: 1000)
            closed: None = all markets, True = closed only, False = active only
            max_markets: Optional limit for testing (None = unlimited)

        Returns:
            Polars DataFrame with all market data
        """
        closed_str = "ALL" if closed is None else ("CLOSED" if closed else "ACTIVE")
        print(f"🔄 Collecting {closed_str} markets...")

        url = f"{self.base_url}/markets"
        all_markets = []
        offset = 0
        batch_num = 1

        while True:
            # Check if we've hit our optional max limit
            if max_markets and len(all_markets) >= max_markets:
                print(f"  ⏹️  Reached max limit of {max_markets} markets")
                break

            params = {
                'limit': limit_per_batch,
                'offset': offset,
                'order': 'id',
                'ascending': 'false'  # Get newest first
            }

            # Add closed filter if specified
            if closed is not None:
                params['closed'] = str(closed).lower()

            try:
                markets = self._make_request(url, params)

                if not markets or len(markets) == 0:
                    print(f"  ✓ No more markets found (reached end)")
                    break

                all_markets.extend(markets)
                print(f"  ✓ Batch {batch_num}: Fetched {len(markets)} markets (offset: {offset:,}, total: {len(all_markets):,})")

                batch_num += 1
                offset += limit_per_batch

                # If we got fewer results than requested, we've reached the end
                if len(markets) < limit_per_batch:
                    print(f"  ✓ Reached end of available markets")
                    break

            except Exception as e:
                print(f"  ✗ Error at offset {offset}: {e}")
                break

        if all_markets:
            df = pl.DataFrame(all_markets)
            print(f"✅ Collected {len(df):,} markets")
            return df
        else:
            print("⚠️  No markets found")
            return pl.DataFrame()

    def save_to_parquet(self, df: pl.DataFrame, filename: str):
        """Save DataFrame to parquet file"""
        if len(df) > 0:
            filepath = self.output_dir / filename
            df.write_parquet(filepath)
            file_size_mb = filepath.stat().st_size / (1024 * 1024)
            print(f"💾 Saved {len(df):,} records to {filepath} ({file_size_mb:.2f} MB)")
        else:
            print(f"⚠️  Skipping {filename} (empty dataset)")

    def collect_complete_snapshot(self) -> Dict[str, pl.DataFrame]:
        """
        Collect complete snapshot of all markets (active and closed).

        Returns:
            Dictionary with 'all_markets', 'active_markets', and 'closed_markets'
        """
        print("="*80)
        print("🚀 Starting Complete Market Snapshot Collection")
        print("="*80)
        print()

        start_time = time.time()

        # Collect ALL markets (no filter)
        all_markets_df = self.collect_all_markets(closed=None)
        self.save_to_parquet(all_markets_df, 'all_markets.parquet')
        print()

        # Extract active and closed from the full dataset (more efficient)
        if len(all_markets_df) > 0 and 'closed' in all_markets_df.columns:
            active_markets_df = all_markets_df.filter(pl.col('closed') == False)
            closed_markets_df = all_markets_df.filter(pl.col('closed') == True)

            print(f"📊 Market Breakdown:")
            print(f"   Total Markets: {len(all_markets_df):,}")
            print(f"   Active Markets: {len(active_markets_df):,}")
            print(f"   Closed Markets: {len(closed_markets_df):,}")
            print()

            self.save_to_parquet(active_markets_df, 'active_markets.parquet')
            self.save_to_parquet(closed_markets_df, 'closed_markets.parquet')
        else:
            active_markets_df = pl.DataFrame()
            closed_markets_df = pl.DataFrame()

        elapsed = time.time() - start_time

        print()
        print("="*80)
        print("✅ Snapshot Collection Complete!")
        print("="*80)
        print(f"⏱️  Time elapsed: {elapsed:.1f} seconds ({elapsed/60:.1f} minutes)")
        print(f"📁 Files saved to: {self.output_dir.absolute()}")
        print()

        return {
            'all_markets': all_markets_df,
            'active_markets': active_markets_df,
            'closed_markets': closed_markets_df
        }

    def get_market_statistics(self, df: pl.DataFrame) -> Dict:
        """Get summary statistics from market dataset"""
        if len(df) == 0:
            return {}

        stats = {
            'total_markets': len(df),
        }

        # Volume statistics
        if 'volume' in df.columns:
            # Handle string volumes (convert to float)
            df_with_volume = df.with_columns([
                pl.when(pl.col('volume').is_not_null())
                  .then(pl.col('volume').cast(pl.Float64, strict=False))
                  .otherwise(0.0)
                  .alias('volume_num')
            ])
            stats['total_volume'] = df_with_volume.select(pl.col('volume_num').sum()).item()
            stats['avg_volume'] = df_with_volume.select(pl.col('volume_num').mean()).item()

        # Liquidity statistics
        if 'liquidity' in df.columns:
            df_with_liquidity = df.with_columns([
                pl.when(pl.col('liquidity').is_not_null())
                  .then(pl.col('liquidity').cast(pl.Float64, strict=False))
                  .otherwise(0.0)
                  .alias('liquidity_num')
            ])
            stats['total_liquidity'] = df_with_liquidity.select(pl.col('liquidity_num').sum()).item()

        # Date range
        if 'startDate' in df.columns:
            stats['earliest_market'] = df.select(pl.col('startDate').min()).item()
            stats['latest_market'] = df.select(pl.col('startDate').max()).item()

        # Unique categories
        if 'category' in df.columns:
            stats['unique_categories'] = df.select(pl.col('category').unique()).height

        return stats


def main():
    """Main execution"""
    collector = MarketSnapshotCollector()

    # Collect complete snapshot
    datasets = collector.collect_complete_snapshot()

    # Print statistics
    if len(datasets['all_markets']) > 0:
        print("="*80)
        print("📈 Market Statistics")
        print("="*80)

        stats = collector.get_market_statistics(datasets['all_markets'])

        print(f"📊 Total Markets: {stats.get('total_markets', 0):,}")

        if 'total_volume' in stats:
            print(f"💰 Total Volume: ${stats['total_volume']:,.0f}")
            print(f"📊 Average Volume per Market: ${stats['avg_volume']:,.0f}")

        if 'total_liquidity' in stats:
            print(f"💧 Total Liquidity: ${stats['total_liquidity']:,.0f}")

        if 'earliest_market' in stats:
            print(f"📅 Date Range: {stats['earliest_market']} to {stats['latest_market']}")

        if 'unique_categories' in stats:
            print(f"🏷️  Unique Categories: {stats['unique_categories']}")

        print()

    print("✨ Market snapshot complete! Ready for analysis.")


if __name__ == "__main__":
    main()
