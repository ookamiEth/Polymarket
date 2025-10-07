#!/usr/bin/env python3
"""
Test script for Polymarket WebSocket Market Channel
Demonstrates real-time market updates (order book, trades, price changes)

NOTE: Requires websockets library: pip install websockets
"""

import asyncio
import websockets
import json
from datetime import datetime

async def test_market_websocket():
    """Connect to market WebSocket channel and display updates"""

    uri = "wss://ws-subscriptions-clob.polymarket.com/ws/market"

    # Example market and token IDs
    market_id = "0xdd31ce870876bb410d71f29e3dfea77910bc3effb81a0fb8dc8c282d5968aa5a"
    token_id = "21742633143463906290569050155826241533067272736897614950488156847949938836455"

    print("="*80)
    print(" WEBSOCKET MARKET CHANNEL TEST")
    print("="*80)
    print(f"\nConnecting to: {uri}")
    print(f"Market ID: {market_id[:30]}...")
    print(f"Token ID: {token_id[:30]}...\n")

    try:
        async with websockets.connect(uri) as websocket:
            # Subscribe to market
            subscribe_msg = {
                "auth": {},
                "markets": [market_id],
                "assets_ids": [token_id],
                "type": "market"
            }

            print("Sending subscription:")
            print(json.dumps(subscribe_msg, indent=2))

            await websocket.send(json.dumps(subscribe_msg))

            print("\nConnected! Listening for messages (Ctrl+C to stop)...\n")
            print("-"*80)

            message_count = 0
            max_messages = 10  # Limit for demo

            while message_count < max_messages:
                message = await websocket.recv()
                data = json.loads(message)

                event_type = data.get("event_type")
                timestamp = datetime.now().strftime("%H:%M:%S")

                message_count += 1

                print(f"\n[{message_count}] {timestamp} - Event: {event_type}")

                if event_type == "book":
                    # Order book snapshot
                    print("  Order Book Update:")
                    print(f"    Asset: {data.get('asset_id', '')[:20]}...")
                    print(f"    Timestamp: {data.get('timestamp')}")
                    bids = data.get('bids', [])
                    asks = data.get('asks', [])
                    print(f"    Top Bid: ${bids[0]['price']} x {bids[0]['size']}" if bids else "    No bids")
                    print(f"    Top Ask: ${asks[0]['price']} x {asks[0]['size']}" if asks else "    No asks")

                elif event_type == "price_change":
                    # Price level change
                    print("  Price Change:")
                    for change in data.get('price_changes', []):
                        print(f"    {change.get('side'):4} @ ${change.get('price'):6} | "
                              f"Size: {change.get('size'):8} | "
                              f"Best Bid/Ask: ${change.get('best_bid')}/${change.get('best_ask')}")

                elif event_type == "last_trade_price":
                    # Trade execution
                    print("  Trade Executed:")
                    print(f"    Side: {data.get('side')}")
                    print(f"    Price: ${data.get('price')}")
                    print(f"    Size: {data.get('size')}")
                    print(f"    Fee: {data.get('fee_rate_bps')} bps")

                elif event_type == "tick_size_change":
                    # Tick size adjustment
                    print("  Tick Size Changed:")
                    print(f"    Old: {data.get('old_tick_size')}")
                    print(f"    New: {data.get('new_tick_size')}")

                else:
                    print(f"  Unknown event type: {event_type}")
                    print(f"  Data: {json.dumps(data, indent=4)}")

            print("\n" + "-"*80)
            print(f"\nReceived {message_count} messages (demo limit reached)")

    except websockets.exceptions.WebSocketException as e:
        print(f"\nWebSocket error: {e}")
    except KeyboardInterrupt:
        print("\n\nStopped by user")
    except Exception as e:
        print(f"\nError: {e}")

    print("\n" + "="*80)
    print(" TEST COMPLETED")
    print("="*80)
    print("\nMessage Types Received:")
    print("  - book: Full order book snapshot")
    print("  - price_change: Order placement/cancellation")
    print("  - last_trade_price: Trade execution")
    print("  - tick_size_change: Minimum tick adjustment")
    print("\nUse Cases:")
    print("  - Real-time order book tracking")
    print("  - Live price monitoring")
    print("  - Trade flow analysis")
    print("  - Market microstructure research")

if __name__ == "__main__":
    print("\nStarting WebSocket Market Channel Test...")
    print("(Install websockets if needed: pip install websockets)\n")

    try:
        asyncio.run(test_market_websocket())
    except ImportError:
        print("Error: websockets library not installed")
        print("Install with: pip install websockets")
