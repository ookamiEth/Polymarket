#!/usr/bin/env python3
"""
Batch fetch tick-by-tick CLOB trade data for all crypto up/down markets.

Collects historical trade data for all 14,874 markets in parallel with:
- Automatic checkpoint/resume capability
- Thread-safe rate limiting
- Progress tracking
- Error handling and logging
- Master index generation

Usage:
    # Full collection
    uv run python scripts/batch_fetch_clob_data.py \
      --workers 5 \
      --checkpoint-file checkpoints/progress.jsonl

    # Test mode (10 markets)
    uv run python scripts/batch_fetch_clob_data.py \
      --workers 5 \
      --test-limit 10

    # Resume interrupted collection
    uv run python scripts/batch_fetch_clob_data.py \
      --workers 5 \
      --checkpoint-file checkpoints/progress.jsonl \
      --resume
"""

import os
import sys
import time
import json
import argparse
import threading
from pathlib import Path
from datetime import datetime
from collections import deque
from typing import Dict, List, Optional, Set
from concurrent.futures import ThreadPoolExecutor, as_completed

import polars as pl
import requests
from tqdm import tqdm


# ============================================================================
# Configuration
# ============================================================================

API_BASE = "https://data-api.polymarket.com"
TRADES_ENDPOINT = f"{API_BASE}/trades"

# Rate limiting: 130 requests per 10 seconds (shared across all workers)
RATE_LIMIT_REQUESTS = 130
RATE_LIMIT_WINDOW = 10  # seconds

# Pagination
TRADES_PER_PAGE = 500
MAX_PAGES = 1000  # Safety limit per market

# Defaults
DEFAULT_MARKET_FILE = "data/markets/crypto_updown_closed.parquet"
DEFAULT_OUTPUT_DIR = "data/clob_ticks"
DEFAULT_WORKERS = 5


# ============================================================================
# Thread-Safe Rate Limiter
# ============================================================================

# Shared locks for thread-safe file operations
_checkpoint_lock = threading.Lock()
_empty_market_lock = threading.Lock()
_failed_market_lock = threading.Lock()


class SharedRateLimiter:
    """Thread-safe rate limiter for parallel workers using deque for O(1) operations."""

    def __init__(self, max_requests: int = RATE_LIMIT_REQUESTS, window: int = RATE_LIMIT_WINDOW):
        self.max_requests = max_requests
        self.window = window
        self.requests = deque()
        self.lock = threading.Lock()

    def wait_if_needed(self):
        """Wait if we're at rate limit (thread-safe)."""
        with self.lock:
            now = time.time()

            # Remove old requests outside the window (O(1) from left)
            while self.requests and now - self.requests[0] >= self.window:
                self.requests.popleft()

            # Check if we need to wait
            if len(self.requests) >= self.max_requests:
                sleep_time = self.window - (now - self.requests[0])
                if sleep_time > 0:
                    time.sleep(sleep_time)
                    # Clean up again after sleeping
                    now = time.time()
                    while self.requests and now - self.requests[0] >= self.window:
                        self.requests.popleft()

            # Record this request (O(1) append)
            self.requests.append(now)


# ============================================================================
# Checkpoint Management
# ============================================================================

def load_checkpoint(checkpoint_file: Path) -> Set[str]:
    """Load checkpoint file and return set of completed condition IDs."""
    if not checkpoint_file.exists():
        return set()

    completed = set()
    with open(checkpoint_file, 'r') as f:
        for line in f:
            try:
                record = json.loads(line.strip())
                if record.get('status') in ['success', 'empty']:
                    completed.add(record['condition_id'])
            except:
                continue

    return completed


def save_checkpoint(checkpoint_file: Path, condition_id: str, status: str, details: Dict = None):
    """Append completion record to checkpoint file (thread-safe)."""
    checkpoint_file.parent.mkdir(parents=True, exist_ok=True)

    record = {
        'condition_id': condition_id,
        'timestamp': datetime.now().isoformat() + 'Z',
        'status': status,  # 'success', 'empty', 'failed'
    }

    if details:
        record.update(details)

    # Thread-safe file append using shared lock
    with _checkpoint_lock:
        with open(checkpoint_file, 'a') as f:
            f.write(json.dumps(record) + '\n')


def log_empty_market(output_dir: Path, market_info: Dict):
    """Log markets with zero trades."""
    log_file = output_dir / 'empty_markets.jsonl'
    log_file.parent.mkdir(parents=True, exist_ok=True)

    record = {
        'condition_id': market_info['condition_id'],
        'question': market_info['question'],
        'slug': market_info.get('slug'),
        'volume_24hr': market_info.get('volume_24hr'),
        'timestamp': datetime.now().isoformat() + 'Z'
    }

    with _empty_market_lock:
        with open(log_file, 'a') as f:
            f.write(json.dumps(record) + '\n')


def log_failed_market(output_dir: Path, market_info: Dict, error: str):
    """Log markets that failed to collect."""
    log_file = output_dir / 'failed_markets.jsonl'
    log_file.parent.mkdir(parents=True, exist_ok=True)

    record = {
        'condition_id': market_info['condition_id'],
        'question': market_info['question'],
        'slug': market_info.get('slug'),
        'error': str(error),
        'timestamp': datetime.now().isoformat() + 'Z'
    }

    with _failed_market_lock:
        with open(log_file, 'a') as f:
            f.write(json.dumps(record) + '\n')


# ============================================================================
# Market Selection
# ============================================================================

def load_markets(parquet_path: str) -> pl.DataFrame:
    """Load crypto up/down markets from Parquet file."""
    return pl.read_parquet(parquet_path)


def get_markets_to_collect(
    market_file: str,
    output_dir: Path,
    checkpoint_file: Optional[Path] = None,
    resume: bool = False,
    test_limit: Optional[int] = None
) -> List[Dict]:
    """
    Get list of markets to collect.

    Filters out:
    - Already collected markets (Parquet file exists)
    - Markets in checkpoint (if resume=True)
    """
    print(f"📂 Loading markets from {market_file}...")
    df_markets = load_markets(market_file)
    print(f"   Loaded {len(df_markets)} total markets")

    # Load checkpoint if resuming
    completed = set()
    if resume and checkpoint_file and checkpoint_file.exists():
        completed = load_checkpoint(checkpoint_file)
        print(f"   Loaded checkpoint: {len(completed)} markets already processed")

    # Filter markets (simple row-by-row - stable and fast enough)
    markets_to_collect = []

    for row in df_markets.iter_rows(named=True):
        condition_id = row.get('conditionId')

        if not condition_id:
            continue

        # Skip if in checkpoint
        if condition_id in completed:
            continue

        # Skip if Parquet file already exists
        slug = row.get('slug', 'unknown')
        condition_id_short = condition_id[:10]
        filename = f"{slug}_{condition_id_short}.parquet"
        filepath = output_dir / filename

        if filepath.exists():
            continue

        # Parse clobTokenIds JSON
        token_ids_json = row.get('clobTokenIds', '[]')
        try:
            token_ids = json.loads(token_ids_json)
        except:
            token_ids = []

        market_info = {
            'question': row.get('question'),
            'condition_id': condition_id,
            'slug': slug,
            'token_id_up': token_ids[0] if len(token_ids) > 0 else None,
            'token_id_down': token_ids[1] if len(token_ids) > 1 else None,
            'start_date': row.get('startDate'),
            'end_date': row.get('endDate'),
            'volume_24hr': row.get('volume24hr'),
        }

        markets_to_collect.append(market_info)

    # Test limit
    if test_limit:
        markets_to_collect = markets_to_collect[:test_limit]
        print(f"   🧪 TEST MODE: Limited to {len(markets_to_collect)} markets")
    else:
        print(f"   📊 Markets to collect: {len(markets_to_collect)}")

    return markets_to_collect


# ============================================================================
# Data Collection (reuse from fetch_clob_tick_data.py)
# ============================================================================

def fetch_trades_paginated(
    condition_id: str,
    rate_limiter: SharedRateLimiter,
    max_pages: int = MAX_PAGES,
    max_retries: int = 3,
    verbose: bool = False
) -> List[Dict]:
    """Fetch all trades for a market with pagination and retry logic."""
    all_trades = []
    offset = 0
    page = 0
    seen_hashes = set()  # Track transaction hashes to detect API duplicates

    while page < max_pages:
        # Rate limiting
        if verbose:
            print(f"[VERBOSE] Waiting for rate limit... (page {page})", file=sys.stderr, flush=True)
        rate_limiter.wait_if_needed()

        # API request with retry
        params = {
            'market': condition_id,
            'limit': TRADES_PER_PAGE,
            'offset': offset
        }

        for retry in range(max_retries):
            try:
                if verbose:
                    print(f"[VERBOSE] Making API request (page {page}, retry {retry})...", file=sys.stderr, flush=True)

                # Use separate connection and read timeouts
                response = requests.get(
                    TRADES_ENDPOINT,
                    params=params,
                    timeout=(10, 30)  # (connect_timeout, read_timeout)
                )

                if verbose:
                    print(f"[VERBOSE] Got response: {response.status_code}", file=sys.stderr, flush=True)

                # Handle specific HTTP errors
                if response.status_code == 404:
                    # Market doesn't exist or has no trades
                    return all_trades

                if response.status_code == 429:
                    # Rate limited - wait and retry
                    wait_time = min(2 ** retry, 30)  # Exponential backoff, max 30s
                    time.sleep(wait_time)
                    continue

                response.raise_for_status()
                trades = response.json()

                if verbose:
                    print(f"[VERBOSE] Received {len(trades)} trades on page {page}", file=sys.stderr, flush=True)

                # Empty response = no more trades
                if not trades or len(trades) == 0:
                    if verbose:
                        print(f"[VERBOSE] No more trades, returning {len(all_trades)} total", file=sys.stderr, flush=True)
                    return all_trades

                # Check for duplicate trades (API bug where it returns same data repeatedly)
                if 'transactionHash' in trades[0]:
                    new_hashes = {t['transactionHash'] for t in trades if 'transactionHash' in t}
                    duplicates = seen_hashes & new_hashes

                    if duplicates:
                        # API is returning trades we've already seen - stop pagination
                        dup_count = len(duplicates)
                        if verbose:
                            print(f"[VERBOSE] Detected {dup_count} duplicate trades on page {page} - API loop detected, stopping", file=sys.stderr, flush=True)
                        print(f"⚠️  WARNING: Market {condition_id[:16]}... API returned {dup_count} duplicate trades at page {page}. Stopping pagination.", file=sys.stderr, flush=True)
                        return all_trades

                    seen_hashes.update(new_hashes)

                all_trades.extend(trades)

                # If we got fewer than requested, we're done
                if len(trades) < TRADES_PER_PAGE:
                    if verbose:
                        print(f"[VERBOSE] Got {len(trades)} < {TRADES_PER_PAGE}, done. Total: {len(all_trades)}", file=sys.stderr, flush=True)
                    return all_trades

                # Next page
                offset += TRADES_PER_PAGE
                page += 1
                break  # Success - exit retry loop

            except requests.exceptions.Timeout:
                if retry == max_retries - 1:
                    raise Exception(f"Timeout after {max_retries} retries")
                time.sleep(2 ** retry)  # Exponential backoff

            except requests.exceptions.RequestException as e:
                if retry == max_retries - 1:
                    raise Exception(f"API Error: {e}")
                time.sleep(2 ** retry)

    return all_trades


def trades_to_dataframe(trades: List[Dict]) -> pl.DataFrame:
    """Convert trade JSON to typed Polars DataFrame."""
    if len(trades) == 0:
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

    return df


def validate_dataframe(df: pl.DataFrame, verbose: bool = False) -> tuple[bool, str]:
    """
    Validate trade data quality.

    Returns (is_valid, error_message)
    """
    if len(df) == 0:
        return (False, "Empty dataframe")

    # Check 1: No duplicate transaction hashes
    if 'transactionHash' in df.columns:
        dup_count = df['transactionHash'].is_duplicated().sum()
        if dup_count > 0:
            unique_count = df['transactionHash'].n_unique()
            msg = f"Found {dup_count} duplicate transaction hashes ({len(df)} total rows, {unique_count} unique)"
            if verbose:
                print(f"[VERBOSE] Validation failed: {msg}", file=sys.stderr, flush=True)
            return (False, msg)

    # Check 2: Timestamps are ordered
    if not df['timestamp'].is_sorted():
        msg = "Timestamps are not sorted"
        if verbose:
            first_unsorted = 0
            for i in range(1, len(df)):
                if df['timestamp'][i] < df['timestamp'][i-1]:
                    first_unsorted = i
                    break
            msg += f" (first unsorted at index {first_unsorted})"
            print(f"[VERBOSE] Validation failed: {msg}", file=sys.stderr, flush=True)
        return (False, msg)

    # Check 3: Prices in valid range [0.0, 1.0]
    min_price = df['price'].min()
    max_price = df['price'].max()

    if min_price < 0.0 or max_price > 1.0:
        msg = f"Invalid prices: min={min_price}, max={max_price} (expected [0.0, 1.0])"
        if verbose:
            print(f"[VERBOSE] Validation failed: {msg}", file=sys.stderr, flush=True)
        return (False, msg)

    return (True, "")


def save_to_parquet(
    df: pl.DataFrame,
    market_info: Dict,
    output_dir: Path
) -> Path:
    """Save DataFrame to uncompressed Parquet file."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # Generate filename
    slug = market_info.get('slug', 'unknown')
    condition_id_short = market_info['condition_id'][:10]
    filename = f"{slug}_{condition_id_short}.parquet"
    filepath = output_dir / filename

    # Write Parquet (uncompressed)
    df.write_parquet(filepath, compression='uncompressed')

    return filepath


def save_metadata(
    market_info: Dict,
    df: pl.DataFrame,
    filepath: Path
):
    """Save metadata JSON alongside Parquet file."""
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


# ============================================================================
# Single Market Collection (worker function)
# ============================================================================

def collect_single_market(
    market_info: Dict,
    rate_limiter: SharedRateLimiter,
    output_dir: Path,
    checkpoint_file: Optional[Path] = None,
    verbose: bool = False
) -> Dict:
    """
    Collect trades for a single market.

    Returns status dict for progress tracking.
    """
    condition_id = market_info['condition_id']
    question = market_info['question']

    try:
        if verbose:
            print(f"[VERBOSE] Starting collection for: {condition_id[:16]}... ({question[:50]}...)", file=sys.stderr, flush=True)

        # Fetch trades
        trades = fetch_trades_paginated(condition_id, rate_limiter, verbose=verbose)

        if verbose:
            print(f"[VERBOSE] Fetched {len(trades)} trades for {condition_id[:16]}...", file=sys.stderr, flush=True)

        # Empty market
        if len(trades) == 0:
            log_empty_market(output_dir, market_info)
            if checkpoint_file:
                save_checkpoint(checkpoint_file, condition_id, 'empty')
            return {
                'status': 'empty',
                'condition_id': condition_id,
                'question': question,
                'trade_count': 0
            }

        # Convert to DataFrame
        df_trades = trades_to_dataframe(trades)

        # Validate
        is_valid, error_msg = validate_dataframe(df_trades, verbose=verbose)
        if not is_valid:
            raise Exception(f"Validation failed: {error_msg}")

        # Save to Parquet
        output_file = save_to_parquet(df_trades, market_info, output_dir)

        # Save metadata
        save_metadata(market_info, df_trades, output_file)

        # Checkpoint
        if checkpoint_file:
            save_checkpoint(checkpoint_file, condition_id, 'success', {
                'trade_count': len(df_trades),
                'file_size_mb': round(output_file.stat().st_size / (1024 * 1024), 2)
            })

        return {
            'status': 'success',
            'condition_id': condition_id,
            'question': question,
            'trade_count': len(df_trades),
            'file_size_mb': round(output_file.stat().st_size / (1024 * 1024), 2)
        }

    except Exception as e:
        # Log failure
        log_failed_market(output_dir, market_info, str(e))
        if checkpoint_file:
            save_checkpoint(checkpoint_file, condition_id, 'failed', {'error': str(e)})

        return {
            'status': 'failed',
            'condition_id': condition_id,
            'question': question,
            'error': str(e)
        }


# ============================================================================
# Parallel Collection
# ============================================================================

def collect_all_markets(
    markets: List[Dict],
    output_dir: Path,
    checkpoint_file: Optional[Path] = None,
    workers: int = DEFAULT_WORKERS,
    verbose: bool = False
):
    """Collect trades for all markets in parallel."""
    print(f"\n🚀 Starting parallel collection...")
    print(f"   Workers: {workers}")
    print(f"   Rate limit: {RATE_LIMIT_REQUESTS} req / {RATE_LIMIT_WINDOW}s (shared)")
    print(f"   Markets: {len(markets)}")
    if verbose:
        print(f"   Verbose mode: ON")

    # Shared rate limiter
    rate_limiter = SharedRateLimiter()

    # Stats
    stats = {
        'success': 0,
        'empty': 0,
        'failed': 0,
        'total_trades': 0,
        'total_size_mb': 0.0
    }

    # Progress bar
    with tqdm(total=len(markets), desc="Collecting markets", unit="market") as pbar:
        with ThreadPoolExecutor(max_workers=workers) as executor:
            # Submit tasks in batches to avoid overwhelming the queue
            futures = []
            for market in markets:
                future = executor.submit(
                    collect_single_market,
                    market,
                    rate_limiter,
                    output_dir,
                    checkpoint_file,
                    verbose
                )
                futures.append(future)

            # Process results as they complete
            for future in as_completed(futures):
                # Add timeout to detect hung threads
                # Note: Large markets can take 5-10 minutes to paginate through all trades
                try:
                    result = future.result(timeout=600)  # 10 minute timeout per market
                except TimeoutError:
                    print(f"\n❌ ERROR: Market collection timed out after 600s", file=sys.stderr, flush=True)
                    stats['failed'] += 1
                    pbar.update(1)
                    continue

                # Update stats
                if result['status'] == 'success':
                    stats['success'] += 1
                    stats['total_trades'] += result.get('trade_count', 0)
                    stats['total_size_mb'] += result.get('file_size_mb', 0.0)
                elif result['status'] == 'empty':
                    stats['empty'] += 1
                elif result['status'] == 'failed':
                    stats['failed'] += 1

                # Update progress bar
                pbar.update(1)
                pbar.set_postfix({
                    'success': stats['success'],
                    'empty': stats['empty'],
                    'failed': stats['failed'],
                    'trades': stats['total_trades']
                })

    return stats


# ============================================================================
# Master Index Generation
# ============================================================================

def generate_master_index(output_dir: Path):
    """Generate master CSV index of all collected markets."""
    print(f"\n📊 Generating master index...")

    meta_files = list(output_dir.glob("*.meta.json"))
    print(f"   Found {len(meta_files)} metadata files")

    if len(meta_files) == 0:
        print("   ⚠️  No metadata files found - skipping index generation")
        return

    # Load all metadata
    records = []
    for meta_file in meta_files:
        with open(meta_file, 'r') as f:
            metadata = json.load(f)

            # Flatten time_range nested structure for CSV export
            time_range = metadata.pop('time_range', {})
            metadata['first_trade'] = time_range.get('first_trade')
            metadata['last_trade'] = time_range.get('last_trade')

            records.append(metadata)

    # Create DataFrame
    df_index = pl.DataFrame(records)

    # Save to CSV
    index_file = output_dir / 'clob_ticks_master_index.csv'
    df_index.write_csv(index_file)

    print(f"   ✅ Master index saved: {index_file}")
    print(f"   📈 Total markets: {len(df_index)}")
    print(f"   📊 Total trades: {df_index['trade_count'].sum():,}")


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='Batch fetch CLOB tick data for all markets')
    parser.add_argument(
        '--market-file',
        type=str,
        default=DEFAULT_MARKET_FILE,
        help='Path to market Parquet file'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default=DEFAULT_OUTPUT_DIR,
        help='Output directory for Parquet files'
    )
    parser.add_argument(
        '--workers',
        type=int,
        default=DEFAULT_WORKERS,
        help='Number of parallel workers'
    )
    parser.add_argument(
        '--checkpoint-file',
        type=str,
        help='Checkpoint file for resume capability'
    )
    parser.add_argument(
        '--resume',
        action='store_true',
        help='Resume from checkpoint'
    )
    parser.add_argument(
        '--test-limit',
        type=int,
        help='Test mode: limit number of markets to collect'
    )
    parser.add_argument(
        '--no-index',
        action='store_true',
        help='Skip master index generation'
    )
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose logging for debugging'
    )

    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    checkpoint_file = Path(args.checkpoint_file) if args.checkpoint_file else None

    print("=" * 80)
    print(" BATCH CLOB TICK DATA COLLECTION")
    print("=" * 80)

    # Get markets to collect
    markets = get_markets_to_collect(
        args.market_file,
        output_dir,
        checkpoint_file,
        args.resume,
        args.test_limit
    )

    if len(markets) == 0:
        print("\n✅ All markets already collected!")
        if not args.no_index:
            generate_master_index(output_dir)
        return

    # Collect all markets
    stats = collect_all_markets(
        markets,
        output_dir,
        checkpoint_file,
        args.workers,
        args.verbose
    )

    # Generate master index
    if not args.no_index:
        generate_master_index(output_dir)

    # Print summary
    print("\n" + "=" * 80)
    print(" ✅ COLLECTION COMPLETE")
    print("=" * 80)
    print(f"\n📊 Summary:")
    print(f"   • Successful: {stats['success']:,} markets")
    print(f"   • Empty: {stats['empty']:,} markets (0 trades)")
    print(f"   • Failed: {stats['failed']:,} markets")
    print(f"   • Total trades: {stats['total_trades']:,}")
    print(f"   • Total storage: {stats['total_size_mb']:.2f} MB")
    print(f"\n📁 Output: {output_dir}")
    print()


if __name__ == '__main__':
    main()
