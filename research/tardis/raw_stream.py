"""
Raw Data Streaming from Tardis.dev using TardisClient

Streams historical Deribit options quote data (bid/ask) with server-side filtering
using the tardis-client library and converts to Polars DataFrame.

We use the 'quote' channel which provides best bid/ask prices and amounts.
Implied volatilities and Greeks are calculated separately using Black-Scholes.
"""

import sys
import os
from pathlib import Path
from datetime import datetime
from typing import List, Optional
import polars as pl

# Add tardis-python client to path
tardis_client_path = Path(__file__).parent / "tardis-python"
sys.path.insert(0, str(tardis_client_path))

from tardis_client import TardisClient, Channel


async def stream_quote_data(
    symbols: List[str],
    from_date: str,
    to_date: str,
    api_key: Optional[str] = None,
    cache_dir: Optional[str] = None,
) -> pl.DataFrame:
    """
    Stream Deribit quote data for specified symbols using TardisClient.

    The 'quote' channel provides:
    - Best bid price and amount
    - Best ask price and amount
    - Timestamps

    We do NOT get IV or Greeks from this channel - those will be calculated
    using Black-Scholes formulas.

    Args:
        symbols: List of Deribit option symbols (e.g., ['BTC-9JUN20-9875-C'])
        from_date: Start date (YYYY-MM-DD)
        to_date: End date (YYYY-MM-DD)
        api_key: Tardis.dev API key (optional for first day of month)
        cache_dir: Directory for local caching (default: /tmp/.tardis-cache)

    Returns:
        Polars DataFrame with quote data
    """
    print("\n" + "=" * 80)
    print("STREAMING QUOTES DATA FROM TARDIS RAW API")
    print("=" * 80)
    print(f"Date range: {from_date} to {to_date}")
    print(f"Symbols: {len(symbols)} options")
    print(f"Channel: quote (best bid/ask)")
    print(f"Server-side filtering: ENABLED")
    print(f"Note: IV and Greeks will be calculated using Black-Scholes")
    print()

    # Initialize Tardis client
    client_kwargs = {}
    if api_key:
        client_kwargs["api_key"] = api_key
    if cache_dir:
        client_kwargs["cache_dir"] = cache_dir

    tardis_client = TardisClient(**client_kwargs)

    # Create channel filter for quote data
    # Deribit quote channel provides best bid/ask updates
    filters = [Channel(name="quote", symbols=symbols)]

    print(f"Starting data stream...")
    print(f"Local cache: {tardis_client.cache_dir}")
    print()

    # Collect all messages
    rows = []
    message_count = 0
    last_progress = 0

    async for local_timestamp, message in tardis_client.replay(
        exchange="deribit",
        from_date=from_date,
        to_date=to_date,
        filters=filters,
        decode_response=True,  # Get JSON objects, not raw bytes
    ):
        message_count += 1

        # Progress indicator every 10,000 messages
        if message_count - last_progress >= 10000:
            print(f"  Processed {message_count:,} messages...")
            last_progress = message_count

        # Extract data from Deribit quote message format
        # Message structure: {"params": {"data": {...}}, "channel": "..."}
        try:
            if "params" not in message or "data" not in message["params"]:
                continue

            data = message["params"]["data"]
            instrument_name = data.get("instrument_name")

            if not instrument_name:
                continue

            # Parse symbol to extract metadata
            # Format: BTC-9JUN20-9875-P -> [BTC, 9JUN20, 9875, P]
            parts = instrument_name.split("-")
            if len(parts) != 4:
                continue

            underlying = parts[0]
            expiry_str = parts[1]
            strike = float(parts[2])
            option_type = "call" if parts[3] == "C" else "put"

            # Build row with quote data
            row = {
                # Metadata
                "exchange": "deribit",
                "symbol": instrument_name,
                "timestamp": data.get("timestamp", 0),  # microseconds
                "local_timestamp": int(local_timestamp.timestamp() * 1_000_000),

                # Parsed fields
                "type": option_type,
                "strike_price": strike,
                "underlying": underlying,
                "expiry_str": expiry_str,

                # Quote data (bid/ask prices and amounts)
                "bid_price": data.get("best_bid_price"),
                "bid_amount": data.get("best_bid_amount"),
                "ask_price": data.get("best_ask_price"),
                "ask_amount": data.get("best_ask_amount"),

                # Underlying price (from quote message if available)
                # Note: May need to get this from a different channel
                "underlying_price": data.get("underlying_price"),
            }

            rows.append(row)

        except Exception as e:
            # Log error but continue processing
            print(f"Warning: Failed to parse message: {e}")
            continue

    print(f"\n✓ Stream complete: {message_count:,} messages processed")
    print(f"✓ Extracted: {len(rows):,} quote data points")
    print()

    # Convert to Polars DataFrame
    if not rows:
        print("Warning: No data extracted!")
        # Return empty DataFrame with correct schema
        return pl.DataFrame(schema={
            "exchange": pl.Utf8,
            "symbol": pl.Utf8,
            "timestamp": pl.Int64,
            "local_timestamp": pl.Int64,
            "type": pl.Utf8,
            "strike_price": pl.Float64,
            "underlying": pl.Utf8,
            "expiry_str": pl.Utf8,
            "bid_price": pl.Float64,
            "bid_amount": pl.Float64,
            "ask_price": pl.Float64,
            "ask_amount": pl.Float64,
            "underlying_price": pl.Float64,
        })

    df = pl.DataFrame(rows)

    print(f"DataFrame shape: {df.shape[0]:,} rows × {df.shape[1]} columns")
    print()

    return df
