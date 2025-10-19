#!/usr/bin/env python3
"""
Generate synthetic Tardis Machine API responses for testing.
Creates NDJSON format data matching real Tardis responses.
"""

import json
import random
from datetime import datetime, timedelta
from typing import List, Dict


def generate_timestamp(base_time: datetime, offset_seconds: int) -> str:
    """Generate ISO timestamp with microseconds."""
    dt = base_time + timedelta(seconds=offset_seconds)
    return dt.strftime('%Y-%m-%dT%H:%M:%S.%f')[:-3] + 'Z'


def generate_symbol(asset: str = 'BTC', strike: int = 50000, expiry_days: int = 7, option_type: str = 'C') -> str:
    """Generate Deribit option symbol: BTC-1JAN25-50000-C"""
    expiry = datetime.now() + timedelta(days=expiry_days)
    return f"{asset}-{expiry.day}{expiry.strftime('%b').upper()}{expiry.strftime('%y')}-{strike}-{option_type}"


def generate_quote_message(symbol: str, timestamp: str, include_book: bool = True) -> Dict:
    """Generate a realistic quote message."""
    bid_price = round(random.uniform(0.001, 0.1), 6)
    ask_price = round(bid_price + random.uniform(0.0001, 0.01), 6)

    msg = {
        "type": "quote",
        "exchange": "deribit",
        "symbol": symbol,
        "timestamp": timestamp,
        "localTimestamp": timestamp,
        "bids": [{"price": bid_price, "amount": random.uniform(0.1, 10.0)}],
        "asks": [{"price": ask_price, "amount": random.uniform(0.1, 10.0)}]
    }

    return msg


def generate_book_snapshot_message(symbol: str, timestamp: str, levels: int = 3) -> Dict:
    """Generate a realistic book_snapshot message."""
    base_bid = random.uniform(0.001, 0.1)

    bids = []
    asks = []

    for i in range(levels):
        bid_price = round(base_bid - i * 0.0001, 6)
        ask_price = round(base_bid + 0.001 + i * 0.0001, 6)
        bids.append({"price": bid_price, "amount": random.uniform(0.1, 10.0)})
        asks.append({"price": ask_price, "amount": random.uniform(0.1, 10.0)})

    return {
        "type": "book_snapshot_3_1m",
        "exchange": "deribit",
        "symbol": symbol,
        "timestamp": timestamp,
        "localTimestamp": timestamp,
        "bids": bids,
        "asks": asks
    }


def generate_invalid_message(symbol: str, timestamp: str, error_type: str) -> Dict:
    """Generate messages with various error conditions."""
    base_msg = {
        "type": "quote",
        "exchange": "deribit",
        "symbol": symbol,
        "timestamp": timestamp,
    }

    if error_type == "negative_price":
        base_msg["bids"] = [{"price": -0.001, "amount": 1.0}]
        base_msg["asks"] = [{"price": 0.002, "amount": 1.0}]
    elif error_type == "bid_greater_than_ask":
        base_msg["bids"] = [{"price": 0.010, "amount": 1.0}]
        base_msg["asks"] = [{"price": 0.005, "amount": 1.0}]
    elif error_type == "missing_fields":
        base_msg["bids"] = []
        base_msg["asks"] = []
    elif error_type == "malformed_symbol":
        base_msg["symbol"] = "INVALID-SYMBOL"
    elif error_type == "invalid_timestamp":
        base_msg["timestamp"] = "invalid-timestamp"

    return base_msg


def generate_dataset(
    num_rows: int,
    num_symbols: int = 10,
    include_book: bool = False,
    include_errors: bool = False,
    error_rate: float = 0.05
) -> str:
    """Generate a complete dataset as NDJSON string."""
    base_time = datetime(2025, 1, 1, 0, 0, 0)

    # Generate symbols
    symbols = []
    for i in range(num_symbols):
        asset = random.choice(['BTC', 'ETH'])
        strike = random.choice([40000, 50000, 60000] if asset == 'BTC' else [2000, 2500, 3000])
        option_type = random.choice(['C', 'P'])
        symbols.append(generate_symbol(asset, strike, random.randint(1, 30), option_type))

    lines = []
    for i in range(num_rows):
        symbol = random.choice(symbols)
        timestamp = generate_timestamp(base_time, i * 5)  # 5 second intervals

        # Randomly add errors if enabled
        if include_errors and random.random() < error_rate:
            error_type = random.choice([
                "negative_price",
                "bid_greater_than_ask",
                "missing_fields",
                "malformed_symbol"
            ])
            msg = generate_invalid_message(symbol, timestamp, error_type)
        else:
            # Normal quote message
            msg = generate_quote_message(symbol, timestamp, include_book)

            # Occasionally add book snapshot
            if include_book and random.random() < 0.3:
                msg = generate_book_snapshot_message(symbol, timestamp)

        lines.append(json.dumps(msg))

    return '\n'.join(lines)


def save_fixture(filename: str, num_rows: int, **kwargs):
    """Generate and save a fixture file."""
    data = generate_dataset(num_rows, **kwargs)
    filepath = f"test/fixtures/{filename}"
    with open(filepath, 'w') as f:
        f.write(data)
    print(f"Generated {filename}: {num_rows:,} rows ({len(data) / 1024:.1f} KB)")


def main():
    """Generate all test fixtures."""
    print("Generating test fixtures...")
    print("=" * 60)

    # Small dataset (1K rows)
    save_fixture("small_1k.json", 1000, num_symbols=5)

    # Medium dataset (10K rows)
    save_fixture("medium_10k.json", 10000, num_symbols=20)

    # Large dataset (100K rows)
    save_fixture("large_100k.json", 100000, num_symbols=50)

    # Extra large dataset (1M rows) - optional, commented out for speed
    # save_fixture("xlarge_1m.json", 1000000, num_symbols=100)

    # Dataset with book snapshots
    save_fixture("with_book_5k.json", 5000, num_symbols=10, include_book=True)

    # Edge cases
    save_fixture("edge_cases.json", 1000, num_symbols=10, include_errors=True, error_rate=0.3)

    # Single symbol test
    save_fixture("single_symbol_1k.json", 1000, num_symbols=1)

    # Empty file
    with open("test/fixtures/empty.json", 'w') as f:
        f.write("")
    print(f"Generated empty.json: 0 rows (0 KB)")

    print("=" * 60)
    print("✅ All fixtures generated successfully!")


if __name__ == "__main__":
    main()
