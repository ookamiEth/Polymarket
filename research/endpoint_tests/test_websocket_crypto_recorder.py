#!/usr/bin/env python3
"""
WebSocket Crypto Price Recorder for Polymarket
Records BTC/USD price data from Chainlink oracle feed for 1 minute
Outputs data to parquet file

Usage: uv run python test_websocket_crypto_recorder.py
"""

import asyncio
import websockets
import json
import polars as pl
from datetime import datetime
from pathlib import Path


async def record_crypto_prices():
    """Record crypto prices from WebSocket for 60 seconds"""

    uri = "wss://ws-live-data.polymarket.com/"

    # Subscription for BTC/USD prices
    subscribe_msg = {
        "action": "subscribe",
        "subscriptions": [
            {
                "topic": "crypto_prices_chainlink",
                "type": "update",
                "filters": json.dumps({"symbol": "btc/usd"})
            }
        ]
    }

    # Data storage
    all_price_records = []

    print("=" * 80)
    print(" CRYPTO PRICE RECORDER - BTC/USD")
    print("=" * 80)
    print(f"\nConnecting to: {uri}")
    print(f"Recording duration: 60 seconds")
    print(f"Symbol: BTC/USD\n")
    print("-" * 80 + "\n")

    try:
        async with websockets.connect(uri) as websocket:
            print("✓ Connected!")

            # Send subscription
            await websocket.send(json.dumps(subscribe_msg))
            print("✓ Subscribed to crypto_prices_chainlink topic")
            print("\nRecording data...\n")

            # Track stats
            message_count = 0
            data_points_count = 0
            start_time = asyncio.get_event_loop().time()

            # Record for 60 seconds
            while asyncio.get_event_loop().time() - start_time < 60:
                try:
                    # Wait for message with timeout
                    message = await asyncio.wait_for(websocket.recv(), timeout=1.0)
                    received_at = datetime.now()

                    # Handle PING/PONG
                    if message == "PING":
                        await websocket.send("PONG")
                        continue

                    # Parse JSON message
                    try:
                        data = json.loads(message)
                    except json.JSONDecodeError:
                        continue

                    message_count += 1

                    # Extract price data
                    topic = data.get("topic")
                    msg_type = data.get("type")
                    payload = data.get("payload", {})

                    if not payload:
                        continue

                    symbol = payload.get("symbol", "unknown")
                    price_data = payload.get("data", [])
                    msg_timestamp = data.get("timestamp")

                    # Convert message timestamp to datetime
                    msg_datetime = None
                    if msg_timestamp:
                        msg_datetime = datetime.fromtimestamp(msg_timestamp / 1000)

                    # Process each price point
                    for point in price_data:
                        timestamp_ms = point.get("timestamp")
                        price_value = point.get("value")

                        if timestamp_ms and price_value:
                            # Convert Unix ms to datetime
                            point_datetime = datetime.fromtimestamp(timestamp_ms / 1000)

                            # Store record
                            all_price_records.append({
                                "timestamp": point_datetime,
                                "symbol": symbol,
                                "price": price_value,
                                "received_at": received_at,
                                "message_timestamp": msg_datetime,
                                "message_type": msg_type
                            })

                            data_points_count += 1

                    # Progress update
                    elapsed = asyncio.get_event_loop().time() - start_time
                    print(f"[{elapsed:.1f}s] Message #{message_count} | "
                          f"Data points: +{len(price_data)} | "
                          f"Total: {data_points_count} | "
                          f"Latest: ${price_value:,.2f}")

                except asyncio.TimeoutError:
                    # No message in timeout window, continue
                    continue
                except Exception as e:
                    print(f"⚠️  Error processing message: {e}")
                    continue

            print("\n" + "-" * 80)
            print(" RECORDING COMPLETED")
            print("-" * 80 + "\n")

    except websockets.exceptions.WebSocketException as e:
        print(f"\n❌ WebSocket error: {e}")
        return None
    except Exception as e:
        print(f"\n❌ Error: {e}")
        return None

    # Check if we have data
    if not all_price_records:
        print("❌ No data recorded")
        return None

    # Create Polars DataFrame
    df = pl.DataFrame(all_price_records)

    # Sort by timestamp
    df = df.sort("timestamp")

    # Generate output filename
    output_dir = Path("/Users/lgierhake/Documents/ETH/BT/data")
    output_dir.mkdir(exist_ok=True)

    timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = output_dir / f"crypto_prices_btcusd_{timestamp_str}.parquet"

    # Write to parquet
    df.write_parquet(output_file)

    # Print summary
    print(f"✓ Saved {len(df)} records to parquet")
    print(f"📁 File: {output_file}")
    print(f"\n" + "=" * 80)
    print(" DATA SUMMARY")
    print("=" * 80)
    print(f"\nTotal records: {len(df)}")
    print(f"Total WebSocket messages: {message_count}")
    print(f"Symbol: {df['symbol'][0]}")
    print(f"\nTime range:")
    print(f"  First: {df['timestamp'].min()}")
    print(f"  Last:  {df['timestamp'].max()}")
    print(f"  Span:  {(df['timestamp'].max() - df['timestamp'].min()).total_seconds():.1f} seconds")
    print(f"\nPrice range:")
    print(f"  Min:  ${df['price'].min():,.2f}")
    print(f"  Max:  ${df['price'].max():,.2f}")
    print(f"  Mean: ${df['price'].mean():,.2f}")
    print(f"  Last: ${df['price'][-1]:,.2f}")

    print(f"\n" + "=" * 80)
    print(" SAMPLE DATA (first 5 rows)")
    print("=" * 80)
    print(df.head(5))

    print(f"\n" + "=" * 80)
    print(" SCHEMA")
    print("=" * 80)
    print(df.schema)
    print("\n")

    return output_file


if __name__ == "__main__":
    print("\nStarting Crypto Price Recorder...\n")

    try:
        output_file = asyncio.run(record_crypto_prices())
        if output_file:
            print(f"✅ SUCCESS! Data saved to: {output_file}\n")
        else:
            print(f"❌ FAILED - No data recorded\n")
    except KeyboardInterrupt:
        print("\n\n⚠️  Recording interrupted by user (Ctrl+C)\n")
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("Make sure dependencies are installed: uv pip install polars websockets\n")
