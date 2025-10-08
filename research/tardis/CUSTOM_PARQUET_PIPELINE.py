#!/usr/bin/env python3
"""
Custom Parquet Pipeline using tardis-client + msgspec

This script demonstrates how to:
1. Stream raw ticker data from tardis-client
2. Parse JSON with msgspec (fast!)
3. Write to Parquet format
4. Compare performance vs CSV approach

Requirements:
    pip install tardis-client msgspec pandas pyarrow

Usage:
    python CUSTOM_PARQUET_PIPELINE.py
"""

import asyncio
import sys
import time
from datetime import datetime
from typing import Optional, List
from pathlib import Path

try:
    import msgspec
except ImportError:
    print("❌ msgspec not installed: pip install msgspec")
    sys.exit(1)

try:
    import pandas as pd
    import pyarrow as pa
    import pyarrow.parquet as pq
except ImportError:
    print("❌ pandas/pyarrow not installed: pip install pandas pyarrow")
    sys.exit(1)

# Add tardis-python to path
sys.path.insert(0, '/Users/lgierhake/Documents/ETH/BT/research/tardis/tardis-python')

try:
    from tardis_client import TardisClient, Channel
except ImportError:
    print("❌ tardis-client not found. Install: pip install tardis-client")
    sys.exit(1)


# ============================================================================
# 1. DEFINE SCHEMA WITH MSGSPEC
# ============================================================================

class OptionGreeks(msgspec.Struct, frozen=True):
    """Option greeks from ticker data"""
    delta: Optional[float] = None
    gamma: Optional[float] = None
    theta: Optional[float] = None
    vega: Optional[float] = None
    rho: Optional[float] = None


class OptionStats(msgspec.Struct, frozen=True):
    """Option statistics from ticker data"""
    volume: Optional[float] = None
    volume_usd: Optional[float] = None
    price_change: Optional[float] = None
    low: Optional[float] = None
    high: Optional[float] = None


class OptionTicker(msgspec.Struct, frozen=True):
    """
    Deribit option ticker data (exchange-native format)

    This struct matches Deribit's ticker channel format.
    Using msgspec.Struct is MUCH faster than dict parsing.
    """
    # Required fields
    timestamp: int
    instrument_name: str

    # Pricing
    underlying_price: Optional[float] = None
    underlying_index: Optional[str] = None
    mark_price: Optional[float] = None
    mark_iv: Optional[float] = None
    last_price: Optional[float] = None

    # Bid/Ask
    best_bid_price: Optional[float] = None
    best_bid_amount: Optional[float] = None
    bid_iv: Optional[float] = None
    best_ask_price: Optional[float] = None
    best_ask_amount: Optional[float] = None
    ask_iv: Optional[float] = None

    # Greeks
    greeks: Optional[OptionGreeks] = None

    # Volume and OI
    open_interest: Optional[float] = None
    stats: Optional[OptionStats] = None

    # Other fields
    state: Optional[str] = None
    settlement_price: Optional[float] = None
    estimated_delivery_price: Optional[float] = None
    interest_rate: Optional[float] = None
    min_price: Optional[float] = None
    max_price: Optional[float] = None


# ============================================================================
# 2. STREAM AND PARSE WITH MSGSPEC
# ============================================================================

async def stream_ticker_data(
    from_date: str,
    to_date: str,
    symbols: List[str] = None,
    max_messages: int = None,
    api_key: str = None
) -> List[dict]:
    """
    Stream ticker data from tardis-client and parse with msgspec.

    Args:
        from_date: Start date (ISO format)
        to_date: End date (ISO format)
        symbols: List of symbols (None = all options)
        max_messages: Maximum messages to process
        api_key: API key (None for first day of month)

    Returns:
        List of parsed ticker data (flat dicts for DataFrame)
    """
    print("=" * 80)
    print("STREAMING TICKER DATA FROM TARDIS-CLIENT")
    print("=" * 80)
    print(f"Date: {from_date} to {to_date}")
    print(f"Symbols: {symbols if symbols else 'ALL OPTIONS'}")
    print(f"Max messages: {max_messages if max_messages else 'UNLIMITED'}")
    print()

    client = TardisClient(api_key=api_key)

    # Create filter for ticker channel
    if symbols is None:
        # For demo: start with a few symbols
        symbols = ["BTC-PERPETUAL"]
        print("Demo mode: Using BTC-PERPETUAL only")
        print("For real usage: pass symbols=['BTC-29NOV25-50000-C', ...] or None for all")

    filters = [Channel(name="ticker", symbols=symbols)]

    # Stream messages
    messages = client.replay(
        exchange="deribit",
        from_date=from_date,
        to_date=to_date,
        filters=filters
    )

    rows = []
    message_count = 0
    parse_time = 0
    start_time = time.time()

    print("\nProcessing messages...")

    try:
        async for local_timestamp, message in messages:
            message_count += 1

            # Extract ticker data from message
            if 'params' in message and 'data' in message['params']:
                data = message['params']['data']

                # Parse with msgspec (FAST!)
                parse_start = time.time()
                ticker = msgspec.convert(data, OptionTicker)
                parse_time += time.time() - parse_start

                # Flatten to dict for DataFrame
                row = {
                    'local_timestamp': local_timestamp.timestamp() * 1_000_000,  # Convert to microseconds
                    'timestamp': ticker.timestamp,
                    'symbol': ticker.instrument_name,
                    'underlying_price': ticker.underlying_price,
                    'underlying_index': ticker.underlying_index,
                    'mark_price': ticker.mark_price,
                    'mark_iv': ticker.mark_iv,
                    'last_price': ticker.last_price,
                    'bid_price': ticker.best_bid_price,
                    'bid_amount': ticker.best_bid_amount,
                    'bid_iv': ticker.bid_iv,
                    'ask_price': ticker.best_ask_price,
                    'ask_amount': ticker.best_ask_amount,
                    'ask_iv': ticker.ask_iv,
                    'open_interest': ticker.open_interest,
                }

                # Add greeks if available
                if ticker.greeks:
                    row.update({
                        'delta': ticker.greeks.delta,
                        'gamma': ticker.greeks.gamma,
                        'theta': ticker.greeks.theta,
                        'vega': ticker.greeks.vega,
                        'rho': ticker.greeks.rho,
                    })

                # Add stats if available
                if ticker.stats:
                    row.update({
                        'volume': ticker.stats.volume,
                        'volume_usd': ticker.stats.volume_usd,
                        'price_change': ticker.stats.price_change,
                    })

                rows.append(row)

            # Progress update
            if message_count % 1000 == 0:
                elapsed = time.time() - start_time
                rate = message_count / elapsed if elapsed > 0 else 0
                print(f"  Processed {message_count:,} messages ({rate:.0f} msg/sec)")

            # Check max messages
            if max_messages and message_count >= max_messages:
                print(f"\n✓ Reached max_messages limit: {max_messages}")
                break

    except KeyboardInterrupt:
        print("\n⚠️ Interrupted by user")

    elapsed = time.time() - start_time
    rate = message_count / elapsed if elapsed > 0 else 0

    print("\n" + "=" * 80)
    print("STREAMING COMPLETE")
    print("=" * 80)
    print(f"Total messages: {message_count:,}")
    print(f"Total rows: {len(rows):,}")
    print(f"Time elapsed: {elapsed:.2f}s")
    print(f"Processing rate: {rate:.0f} msg/sec")
    print(f"Parse time: {parse_time:.2f}s ({parse_time/elapsed*100:.1f}% of total)")
    print()

    return rows


# ============================================================================
# 3. WRITE TO PARQUET
# ============================================================================

def write_parquet(rows: List[dict], output_path: str, compression: str = 'zstd'):
    """
    Write rows to Parquet file.

    Args:
        rows: List of dicts (from stream_ticker_data)
        output_path: Output file path
        compression: Compression algorithm (zstd, snappy, gzip)
    """
    print("=" * 80)
    print("WRITING TO PARQUET")
    print("=" * 80)
    print(f"Output: {output_path}")
    print(f"Compression: {compression}")
    print()

    start_time = time.time()

    # Convert to DataFrame
    print("Converting to DataFrame...")
    df = pd.DataFrame(rows)

    # Convert timestamps to datetime (optional, for readability)
    df['datetime'] = pd.to_datetime(df['timestamp'], unit='ms')
    df['local_datetime'] = pd.to_datetime(df['local_timestamp'], unit='us')

    print(f"  Rows: {len(df):,}")
    print(f"  Columns: {len(df.columns)}")
    print(f"  Memory: {df.memory_usage(deep=True).sum() / 1024**2:.1f} MB")

    # Write to Parquet
    print(f"\nWriting to {output_path}...")
    df.to_parquet(
        output_path,
        compression=compression,
        index=False,
        engine='pyarrow'
    )

    elapsed = time.time() - start_time
    file_size = Path(output_path).stat().st_size / 1024**2

    print("\n" + "=" * 80)
    print("PARQUET WRITE COMPLETE")
    print("=" * 80)
    print(f"File: {output_path}")
    print(f"Size: {file_size:.1f} MB")
    print(f"Time: {elapsed:.2f}s")
    print(f"Write speed: {len(df) / elapsed:.0f} rows/sec")
    print()


# ============================================================================
# 4. COMPARE WITH CSV
# ============================================================================

def compare_formats(parquet_path: str, csv_path: str = None):
    """
    Compare Parquet vs CSV format.

    Args:
        parquet_path: Path to Parquet file
        csv_path: Path to CSV file (optional)
    """
    print("=" * 80)
    print("FORMAT COMPARISON")
    print("=" * 80)

    # Parquet stats
    parquet_size = Path(parquet_path).stat().st_size / 1024**2
    print(f"\nParquet file: {parquet_path}")
    print(f"  Size: {parquet_size:.1f} MB")

    # Load time
    start = time.time()
    df_parquet = pd.read_parquet(parquet_path)
    parquet_load_time = time.time() - start

    print(f"  Load time: {parquet_load_time:.3f}s")
    print(f"  Rows: {len(df_parquet):,}")
    print(f"  Columns: {len(df_parquet.columns)}")

    # Query time
    start = time.time()
    btc_options = df_parquet[df_parquet['symbol'].str.startswith('BTC-')]
    parquet_query_time = time.time() - start

    print(f"  Query time (filter BTC): {parquet_query_time*1000:.1f}ms")
    print(f"  Result rows: {len(btc_options):,}")

    # CSV comparison (if provided)
    if csv_path and Path(csv_path).exists():
        csv_size = Path(csv_path).stat().st_size / 1024**2
        print(f"\nCSV file: {csv_path}")
        print(f"  Size: {csv_size:.1f} MB")

        start = time.time()
        df_csv = pd.read_csv(csv_path)
        csv_load_time = time.time() - start

        print(f"  Load time: {csv_load_time:.3f}s")
        print(f"  Rows: {len(df_csv):,}")

        start = time.time()
        btc_csv = df_csv[df_csv['symbol'].str.startswith('BTC-')]
        csv_query_time = time.time() - start

        print(f"  Query time (filter BTC): {csv_query_time*1000:.1f}ms")

        # Comparison
        print(f"\n{'Metric':<30} {'CSV':<15} {'Parquet':<15} {'Winner':<15}")
        print("-" * 75)
        print(f"{'File size':<30} {csv_size:.1f} MB{'':<8} {parquet_size:.1f} MB{'':<8} {'Parquet' if parquet_size < csv_size else 'CSV':<15}")
        print(f"{'Load time':<30} {csv_load_time:.3f}s{'':<9} {parquet_load_time:.3f}s{'':<9} {'Parquet' if parquet_load_time < csv_load_time else 'CSV':<15}")
        print(f"{'Query time':<30} {csv_query_time*1000:.1f}ms{'':<9} {parquet_query_time*1000:.1f}ms{'':<9} {'Parquet' if parquet_query_time < csv_query_time else 'CSV':<15}")

        # Calculate improvements
        size_improvement = (1 - parquet_size / csv_size) * 100 if csv_size > 0 else 0
        load_improvement = (1 - parquet_load_time / csv_load_time) * 100 if csv_load_time > 0 else 0
        query_improvement = (1 - parquet_query_time / csv_query_time) * 100 if csv_query_time > 0 else 0

        print(f"\nParquet improvements:")
        print(f"  File size: {size_improvement:.1f}% smaller")
        print(f"  Load time: {load_improvement:.1f}% faster")
        print(f"  Query time: {query_improvement:.1f}% faster")

    print()


# ============================================================================
# 5. MAIN DEMO
# ============================================================================

async def main():
    """
    Main function to demonstrate custom Parquet pipeline.
    """
    print("\n")
    print("╔" + "═" * 78 + "╗")
    print("║" + " " * 15 + "CUSTOM PARQUET PIPELINE WITH MSGSPEC" + " " * 27 + "║")
    print("║" + " " * 20 + "tardis-client + msgspec + parquet" + " " * 25 + "║")
    print("╚" + "═" * 78 + "╝")
    print()

    # Configuration
    from_date = "2025-10-01"
    to_date = "2025-10-01 00:05:00"  # Just 5 minutes for demo
    max_messages = 1000  # Limit for demo
    api_key = None  # Free access

    output_parquet = "btc_options_custom.parquet"
    output_csv = "btc_options_custom.csv"

    # Step 1: Stream and parse ticker data
    print("STEP 1: Stream ticker data from tardis-client")
    print("-" * 80)
    rows = await stream_ticker_data(
        from_date=from_date,
        to_date=to_date,
        symbols=None,  # Will use BTC-PERPETUAL for demo
        max_messages=max_messages,
        api_key=api_key
    )

    if not rows:
        print("❌ No data received. Exiting.")
        return

    # Step 2: Write to Parquet
    print("STEP 2: Write to Parquet format")
    print("-" * 80)
    write_parquet(rows, output_parquet, compression='zstd')

    # Step 3: Also write to CSV for comparison
    print("STEP 3: Write to CSV for comparison")
    print("-" * 80)
    df = pd.DataFrame(rows)
    df.to_csv(output_csv, index=False)
    print(f"✓ Wrote {len(df):,} rows to {output_csv}")
    print()

    # Step 4: Compare formats
    print("STEP 4: Compare Parquet vs CSV")
    print("-" * 80)
    compare_formats(output_parquet, output_csv)

    # Summary
    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print("""
Key Takeaways:

1. ✅ msgspec is VERY fast for JSON parsing
   - Faster than json.loads() or ujson
   - Type-safe with structs
   - Zero-copy where possible

2. ✅ Parquet has better compression than CSV
   - Typically 2-5x smaller file size
   - Columnar format for faster queries
   - Preserves data types

3. ✅ You have full control over schema
   - Choose only columns you need
   - Custom data types
   - Nested structures if needed

4. ⚠️ More complex than CSV approach
   - Need to write processing pipeline
   - Maintain msgspec structs
   - Handle schema evolution

Recommendation:
- Use CSV (tardis-dev) for quick analysis
- Use this approach for production pipelines or large-scale analysis
""")

    print("Files created:")
    print(f"  - {output_parquet} (Parquet)")
    print(f"  - {output_csv} (CSV)")
    print()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n⚠️ Interrupted by user")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
