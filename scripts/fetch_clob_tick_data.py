#!/usr/bin/env python3
"""
Fetch tick-by-tick CLOB trade data for Polymarket crypto up/down markets.

Phase 1: Single market test (highest volume)
Phase 2: All 14,874 markets (future)

Usage:
    uv run python scripts/fetch_clob_tick_data.py --test-mode
    uv run python scripts/fetch_clob_tick_data.py --condition-id 0xabc123...
"""

import os
import sys
import time
import json
import argparse
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional

import polars as pl
import requests
from tqdm import tqdm


# ============================================================================
# Configuration
# ============================================================================

API_BASE = "https://data-api.polymarket.com"
TRADES_ENDPOINT = f"{API_BASE}/trades"

# Rate limiting: 130 requests per 10 seconds
RATE_LIMIT_REQUESTS = 130
RATE_LIMIT_WINDOW = 10  # seconds

# Pagination
TRADES_PER_PAGE = 500
MAX_PAGES = 1000  # Safety limit


# ============================================================================
# Rate Limiter
# ============================================================================

class RateLimiter:
    """Rate limiter to stay under API limits (130 req/10s)"""

    def __init__(self, max_requests: int = RATE_LIMIT_REQUESTS, window: int = RATE_LIMIT_WINDOW):
        self.max_requests = max_requests
        self.window = window
        self.requests = []

    def wait_if_needed(self):
        """Wait if we're at rate limit"""
        now = time.time()

        # Remove requests outside the window
        self.requests = [r for r in self.requests if now - r < self.window]

        # Check if we need to wait
        if len(self.requests) >= self.max_requests:
            sleep_time = self.window - (now - self.requests[0])
            if sleep_time > 0:
                time.sleep(sleep_time)

        # Record this request
        self.requests.append(time.time())


# ============================================================================
# Step 1: Load Market Data
# ============================================================================

def load_markets(parquet_path: str) -> pl.DataFrame:
    """Load crypto up/down markets from Parquet file"""
    print(f"📂 Loading markets from {parquet_path}...")

    df = pl.read_parquet(parquet_path)

    print(f"   Loaded {len(df)} markets")
    return df


def select_highest_volume_market(df: pl.DataFrame) -> Dict:
    """Select the market with highest 24hr volume"""
    print("\n🔍 Finding highest-volume market...")

    # Filter to markets with volume data
    df_with_volume = df.filter(pl.col('volume24hr').is_not_null())

    if len(df_with_volume) == 0:
        raise ValueError("No markets with volume24hr data found")

    # Sort by volume and take top
    top_market = df_with_volume.sort('volume24hr', descending=True).head(1)

    # Extract details
    row = top_market.to_dicts()[0]

    # Parse clobTokenIds JSON
    token_ids_json = row.get('clobTokenIds', '[]')
    try:
        token_ids = json.loads(token_ids_json)
    except:
        token_ids = []

    market_info = {
        'question': row.get('question'),
        'condition_id': row.get('conditionId'),
        'slug': row.get('slug'),
        'token_id_up': token_ids[0] if len(token_ids) > 0 else None,
        'token_id_down': token_ids[1] if len(token_ids) > 1 else None,
        'start_date': row.get('startDate'),
        'end_date': row.get('endDate'),
        'volume_24hr': row.get('volume24hr'),
    }

    print(f"   ✅ Selected: {market_info['question']}")
    print(f"   📊 Volume: ${market_info['volume_24hr']:,}")
    print(f"   🆔 Condition ID: {market_info['condition_id'][:20]}...")

    return market_info


# ============================================================================
# Step 2: Fetch Trades from API
# ============================================================================

def fetch_trades_paginated(
    condition_id: str,
    rate_limiter: RateLimiter,
    max_pages: int = MAX_PAGES
) -> List[Dict]:
    """
    Fetch all trades for a market with pagination.

    Returns list of trade dicts.
    """
    print(f"\n📡 Fetching trades from API...")
    print(f"   Endpoint: {TRADES_ENDPOINT}")
    print(f"   Market: {condition_id[:20]}...")
    print(f"   Rate limit: {RATE_LIMIT_REQUESTS} req / {RATE_LIMIT_WINDOW}s")

    all_trades = []
    offset = 0
    page = 0

    with tqdm(desc="Fetching trades", unit="page") as pbar:
        while page < max_pages:
            # Rate limiting
            rate_limiter.wait_if_needed()

            # API request
            params = {
                'market': condition_id,
                'limit': TRADES_PER_PAGE,
                'offset': offset
            }

            try:
                response = requests.get(TRADES_ENDPOINT, params=params, timeout=30)
                response.raise_for_status()

                trades = response.json()

                # Empty response = no more trades
                if not trades or len(trades) == 0:
                    break

                all_trades.extend(trades)

                # Update progress
                pbar.update(1)
                pbar.set_postfix({
                    'trades': len(all_trades),
                    'page': page + 1
                })

                # If we got fewer than requested, we're done
                if len(trades) < TRADES_PER_PAGE:
                    break

                # Next page
                offset += TRADES_PER_PAGE
                page += 1

            except requests.exceptions.RequestException as e:
                print(f"\n   ⚠️  API Error: {e}")
                # Continue on error (could add retry logic here)
                break

    print(f"   ✅ Retrieved {len(all_trades)} trades from {page + 1} page(s)")

    if len(all_trades) == 0:
        print("   ⚠️  WARNING: No trades found for this market")

    return all_trades


# ============================================================================
# Step 3: Convert to DataFrame
# ============================================================================

def trades_to_dataframe(trades: List[Dict]) -> pl.DataFrame:
    """
    Convert trade JSON to typed Polars DataFrame.
    """
    print(f"\n🔄 Converting {len(trades)} trades to DataFrame...")

    if len(trades) == 0:
        print("   ⚠️  No trades to convert")
        return pl.DataFrame()

    # Create DataFrame from list of dicts
    df = pl.DataFrame(trades)

    # Type conversions
    df = df.with_columns([
        # Timestamps
        pl.col('timestamp').cast(pl.Int64).alias('timestamp'),

        # Prices and sizes
        pl.col('price').cast(pl.Float64).alias('price'),
        pl.col('size').cast(pl.Float64).alias('size'),

        # Categoricals (for efficiency)
        pl.col('side').cast(pl.Categorical).alias('side'),
        pl.col('outcome').cast(pl.Categorical).alias('outcome'),

        # Outcome index
        pl.col('outcomeIndex').cast(pl.Int8).alias('outcomeIndex'),
    ])

    # Add datetime column from timestamp
    df = df.with_columns([
        pl.from_epoch('timestamp', time_unit='s').alias('datetime_utc')
    ])

    # Sort by timestamp
    df = df.sort('timestamp')

    print(f"   ✅ DataFrame created: {df.shape[0]} rows × {df.shape[1]} columns")

    return df


def validate_dataframe(df: pl.DataFrame) -> bool:
    """
    Validate trade data quality.

    Returns True if valid, False otherwise.
    """
    print(f"\n✓ Validating data...")

    if len(df) == 0:
        print("   ⚠️  Empty DataFrame - nothing to validate")
        return False

    errors = []

    # Check 1: No duplicate transaction hashes
    if 'transactionHash' in df.columns:
        dup_count = df['transactionHash'].is_duplicated().sum()
        if dup_count > 0:
            errors.append(f"Found {dup_count} duplicate transaction hashes")

    # Check 2: Timestamps are ordered
    if not df['timestamp'].is_sorted():
        errors.append("Timestamps are not sorted")

    # Check 3: Prices in valid range [0.0, 1.0]
    min_price = df['price'].min()
    max_price = df['price'].max()

    if min_price < 0.0 or max_price > 1.0:
        errors.append(f"Prices out of range: min={min_price}, max={max_price}")

    # Check 4: No null critical fields
    critical_fields = ['timestamp', 'price', 'size', 'side']
    for field in critical_fields:
        if field in df.columns:
            null_count = df[field].null_count()
            if null_count > 0:
                errors.append(f"Field '{field}' has {null_count} null values")

    # Report
    if errors:
        print("   ❌ Validation FAILED:")
        for error in errors:
            print(f"      • {error}")
        return False
    else:
        print(f"   ✅ Validation PASSED")
        print(f"      • {len(df)} trades")
        print(f"      • 0 duplicates")
        print(f"      • Timestamps ordered")
        print(f"      • Prices in range [{min_price:.4f}, {max_price:.4f}]")
        return True


# ============================================================================
# Step 4: Save to Parquet
# ============================================================================

def save_to_parquet(
    df: pl.DataFrame,
    market_info: Dict,
    output_dir: str
) -> Path:
    """
    Save DataFrame to uncompressed Parquet file.
    """
    print(f"\n💾 Saving to Parquet...")

    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Generate filename
    slug = market_info.get('slug', 'unknown')
    condition_id_short = market_info['condition_id'][:10]
    filename = f"{slug}_{condition_id_short}.parquet"
    filepath = output_path / filename

    print(f"   File: {filepath}")

    # Write Parquet (uncompressed as requested)
    df.write_parquet(filepath, compression='uncompressed')

    # Get file size
    file_size_mb = filepath.stat().st_size / (1024 * 1024)

    print(f"   ✅ Saved {len(df)} trades ({file_size_mb:.2f} MB)")

    return filepath


def save_metadata(
    market_info: Dict,
    df: pl.DataFrame,
    filepath: Path,
    output_dir: str
):
    """
    Save metadata JSON alongside Parquet file (optional).
    """
    print(f"\n📝 Saving metadata...")

    # Calculate stats
    if len(df) > 0:
        first_trade_ts = df['timestamp'].min()
        last_trade_ts = df['timestamp'].max()
        first_trade_dt = datetime.fromtimestamp(first_trade_ts).isoformat() + 'Z'
        last_trade_dt = datetime.fromtimestamp(last_trade_ts).isoformat() + 'Z'
    else:
        first_trade_dt = None
        last_trade_dt = None

    metadata = {
        'market_slug': market_info.get('slug'),
        'condition_id': market_info['condition_id'],
        'question': market_info['question'],
        'token_id_up': market_info.get('token_id_up'),
        'token_id_down': market_info.get('token_id_down'),
        'start_date': market_info.get('start_date'),
        'end_date': market_info.get('end_date'),
        'volume_24hr': market_info.get('volume_24hr'),
        'collection_timestamp': datetime.now().isoformat() + 'Z',
        'trade_count': len(df),
        'time_range': {
            'first_trade': first_trade_dt,
            'last_trade': last_trade_dt
        },
        'file_size_mb': round(filepath.stat().st_size / (1024 * 1024), 2),
        'parquet_file': str(filepath.name)
    }

    # Save JSON
    meta_filepath = filepath.with_suffix('.meta.json')
    with open(meta_filepath, 'w') as f:
        json.dump(metadata, f, indent=2)

    print(f"   ✅ Metadata saved: {meta_filepath.name}")


# ============================================================================
# Step 5: Verify
# ============================================================================

def verify_output(filepath: Path):
    """
    Load Parquet back and verify it matches expectations.
    """
    print(f"\n🔍 Verifying output...")

    # Load back
    df_verify = pl.read_parquet(filepath)

    print(f"   ✅ Parquet loads successfully")
    print(f"   📊 Shape: {df_verify.shape[0]} rows × {df_verify.shape[1]} columns")

    # Show sample
    print(f"\n   Sample rows:")
    print(df_verify.select(['datetime_utc', 'side', 'price', 'size', 'outcome']).head(5))

    # Show time range
    if len(df_verify) > 0:
        first_dt = df_verify['datetime_utc'].min()
        last_dt = df_verify['datetime_utc'].max()
        print(f"\n   Time range: {first_dt} → {last_dt}")

    return df_verify


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='Fetch CLOB tick data for Polymarket markets')
    parser.add_argument(
        '--market-file',
        type=str,
        default='data/markets/crypto_updown_closed.parquet',
        help='Path to market Parquet file'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='data/clob_ticks',
        help='Output directory for Parquet files'
    )
    parser.add_argument(
        '--test-mode',
        action='store_true',
        help='Test mode: fetch only highest-volume market'
    )
    parser.add_argument(
        '--condition-id',
        type=str,
        help='Specific condition ID to fetch (overrides test mode)'
    )
    parser.add_argument(
        '--no-metadata',
        action='store_true',
        help='Skip metadata JSON generation'
    )

    args = parser.parse_args()

    print("=" * 80)
    print(" CLOB TICK DATA COLLECTION")
    print("=" * 80)

    # Initialize rate limiter
    rate_limiter = RateLimiter()

    # Step 1: Load markets
    df_markets = load_markets(args.market_file)

    # Select market
    if args.condition_id:
        # User provided specific condition ID
        market_row = df_markets.filter(pl.col('conditionId') == args.condition_id).head(1)
        if len(market_row) == 0:
            print(f"❌ Error: Condition ID {args.condition_id} not found in market file")
            sys.exit(1)
        market_info = market_row.to_dicts()[0]
        market_info['volume_24hr'] = market_info.get('volume24hr')
    else:
        # Test mode or default: highest volume
        market_info = select_highest_volume_market(df_markets)

    # Step 2: Fetch trades
    trades = fetch_trades_paginated(
        market_info['condition_id'],
        rate_limiter
    )

    if len(trades) == 0:
        print("\n⚠️  No trades found - exiting")
        sys.exit(0)

    # Step 3: Convert to DataFrame
    df_trades = trades_to_dataframe(trades)

    # Validate
    if not validate_dataframe(df_trades):
        print("\n❌ Validation failed - exiting")
        sys.exit(1)

    # Step 4: Save to Parquet
    output_file = save_to_parquet(df_trades, market_info, args.output_dir)

    # Save metadata (optional)
    if not args.no_metadata:
        save_metadata(market_info, df_trades, output_file, args.output_dir)

    # Step 5: Verify
    verify_output(output_file)

    print("\n" + "=" * 80)
    print(" ✅ COLLECTION COMPLETE")
    print("=" * 80)
    print(f"\n📁 Output: {output_file}")
    print(f"📊 Trades: {len(df_trades)}")
    print(f"💾 Size: {output_file.stat().st_size / (1024*1024):.2f} MB")
    print()


if __name__ == '__main__':
    main()
