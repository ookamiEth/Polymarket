#!/usr/bin/env python3
"""
Thread-safe consolidated file writer for CLOB tick data.

Handles concurrent writes to consolidated parquet files from multiple worker threads.
Uses file-level locking to prevent corruption and ensure data integrity.
"""

import threading
from pathlib import Path
from typing import Dict, Optional
import polars as pl


class ConsolidatedWriter:
    """
    Thread-safe writer for consolidated parquet files.

    Maintains one parquet file per market category, with automatic:
    - Thread-safe appends (file-level locking)
    - Duplicate removal (by transactionHash)
    - Timestamp sorting
    - Atomic writes (write to temp, then rename)
    """

    def __init__(self, output_dir: Path):
        """
        Initialize writer.

        Args:
            output_dir: Directory for consolidated parquet files
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # One lock per category for thread-safe appends
        from market_categorizer import ALL_CATEGORIES
        self.locks = {category: threading.Lock() for category in ALL_CATEGORIES}

        # Track stats
        self.stats = {
            category: {'markets': 0, 'trades': 0}
            for category in ALL_CATEGORIES
        }
        self.stats_lock = threading.Lock()

    def append_market(
        self,
        df: pl.DataFrame,
        category: str,
        market_slug: str,
        verbose: bool = False
    ) -> bool:
        """
        Append market data to consolidated file (thread-safe).

        Args:
            df: Market trade data
            category: Market category (e.g., "btc_15min")
            market_slug: Market slug for logging
            verbose: Print debug info

        Returns:
            True if successful, False otherwise
        """
        if category not in self.locks:
            if verbose:
                print(f"[WARN] Unknown category '{category}' for {market_slug}, skipping")
            return False

        # Acquire lock for this category
        with self.locks[category]:
            try:
                filepath = self.output_dir / f"{category}_consolidated.parquet"
                temp_filepath = self.output_dir / f"{category}_consolidated.parquet.tmp"

                if verbose:
                    print(f"[VERBOSE] Appending {len(df)} trades to {category} ({market_slug})")

                # If file exists, read and combine
                if filepath.exists():
                    existing = pl.read_parquet(filepath)

                    if verbose:
                        print(f"[VERBOSE] Read {len(existing)} existing trades from {category}")

                    # Concatenate
                    combined = pl.concat([existing, df])

                    # Remove duplicates by transactionHash (keep first)
                    if 'transactionHash' in combined.columns:
                        initial_count = len(combined)
                        combined = combined.unique(subset=['transactionHash'], keep='first')
                        duplicates_removed = initial_count - len(combined)

                        if verbose and duplicates_removed > 0:
                            print(f"[VERBOSE] Removed {duplicates_removed} duplicate trades")

                    # Sort by timestamp
                    combined = combined.sort('timestamp')

                    if verbose:
                        print(f"[VERBOSE] Combined total: {len(combined)} trades")

                else:
                    # First market in this category
                    combined = df.sort('timestamp')

                    if verbose:
                        print(f"[VERBOSE] Creating new {category} file with {len(combined)} trades")

                # Atomic write: write to temp, then rename
                combined.write_parquet(temp_filepath, compression='uncompressed')
                temp_filepath.rename(filepath)

                # Update stats
                with self.stats_lock:
                    self.stats[category]['markets'] += 1
                    self.stats[category]['trades'] = len(combined)

                if verbose:
                    print(f"[VERBOSE] Successfully wrote {len(combined)} trades to {category}")

                return True

            except Exception as e:
                if verbose:
                    print(f"[ERROR] Failed to append to {category}: {e}")

                # Clean up temp file if it exists
                if temp_filepath.exists():
                    temp_filepath.unlink()

                return False

    def get_stats(self) -> Dict:
        """Get write statistics (thread-safe)."""
        with self.stats_lock:
            return dict(self.stats)

    def print_summary(self):
        """Print summary statistics."""
        stats = self.get_stats()

        print("\n" + "=" * 80)
        print("CONSOLIDATED WRITER SUMMARY")
        print("=" * 80)

        total_markets = 0
        total_trades = 0

        for category, data in sorted(stats.items()):
            if data['markets'] > 0:
                print(f"  {category:30s}: {data['markets']:5d} markets, {data['trades']:8d} trades")
                total_markets += data['markets']
                total_trades += data['trades']

        print("=" * 80)
        print(f"  {'TOTAL':30s}: {total_markets:5d} markets, {total_trades:8d} trades")
        print("=" * 80)


# Quick test
if __name__ == '__main__':
    import sys
    from market_categorizer import categorize_market

    print("Testing ConsolidatedWriter...")

    # Create test writer
    test_dir = Path("data/test_consolidated")
    writer = ConsolidatedWriter(test_dir)

    # Create test data (minimal schema matching real data)
    test_df = pl.DataFrame({
        'proxyWallet': ['0x123', '0x456'],
        'side': ['BUY', 'SELL'],
        'asset': ['123', '456'],
        'conditionId': ['0xabc', '0xdef'],
        'size': [10.0, 20.0],
        'price': [0.5, 0.6],
        'timestamp': [1700000000, 1700000001],
        'transactionHash': ['0xhash1', '0xhash2'],
    }).with_columns([
        pl.from_epoch('timestamp', time_unit='s').alias('datetime_utc')
    ])

    # Test appending
    slug = "btc-up-or-down-15m-1758561300"
    category = categorize_market(slug)

    if category:
        success = writer.append_market(test_df, category, slug, verbose=True)
        print(f"\nAppend result: {'SUCCESS' if success else 'FAILED'}")

        # Verify
        filepath = test_dir / f"{category}_consolidated.parquet"
        if filepath.exists():
            df_check = pl.read_parquet(filepath)
            print(f"\nVerification: {len(df_check)} trades in {filepath}")
        else:
            print(f"\nERROR: File not created: {filepath}")
    else:
        print(f"ERROR: Could not categorize {slug}")

    writer.print_summary()

    # Cleanup
    import shutil
    if test_dir.exists():
        shutil.rmtree(test_dir)
        print(f"\n✓ Cleaned up test directory: {test_dir}")
