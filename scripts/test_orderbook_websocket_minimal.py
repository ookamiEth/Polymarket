#!/usr/bin/env python3
"""
Minimal WebSocket test for Polymarket orderbook.
Connects to wss://ws-subscriptions-clob.polymarket.com/ws/market
Prints first 20 messages to console.
"""

import asyncio
import websockets
import json
from datetime import datetime

# High-volume BTC market: "Will Bitcoin reach $130,000 by December 31, 2025?"
# Volume 24hr: $128,563
MARKET_ID = "0xe84b1fdc087f4153ebf15cfc07f065dd5a66f3caf370b4547ea8e02100be95be"
TOKEN_IDS = [
    "70455106433105725093003807079135685186309776081399075740142938507518314484366",
    "11800304361754383189551729117933216661503938872257140403668204817034252452180"
]

WSS_URL = "wss://ws-subscriptions-clob.polymarket.com/ws/market"

async def test_websocket():
    """Test WebSocket connection and print first messages."""

    print("="*80)
    print(" WEBSOCKET ORDERBOOK TEST")
    print("="*80)
    print(f"\nURL: {WSS_URL}")
    print(f"Market ID: {MARKET_ID[:40]}...")
    print(f"Token IDs: {len(TOKEN_IDS)} tokens")
    print(f"\n⏳ Connecting...")

    try:
        async with websockets.connect(WSS_URL) as websocket:
            print("✅ Connected!")

            # Subscribe to market channel
            # NOTE: Market channel uses "assets_ids" (token IDs), NOT "markets"
            subscribe_msg = {
                "assets_ids": TOKEN_IDS,  # Subscribe with token IDs
                "type": "market"
            }

            print(f"\n📡 Subscribing to market...")
            print(f"   Message: {json.dumps(subscribe_msg, indent=2)}")
            await websocket.send(json.dumps(subscribe_msg))
            print("✅ Subscription sent")

            # Send initial PING
            await websocket.send("PING")
            print("📡 Sent PING to keep connection alive")

            print(f"\n{'='*80}")
            print(" LISTENING FOR MESSAGES (showing first 20)")
            print("="*80)
            print("\nPress Ctrl+C to stop early...\n")

            message_count = 0
            max_messages = 20

            # Message type counters
            msg_types = {}

            # Add timeout for receiving messages
            import asyncio
            timeout_seconds = 30  # Increased to 30s
            last_ping = asyncio.get_event_loop().time()

            while message_count < max_messages:
                try:
                    # Wait max 15 seconds for a message
                    message = await asyncio.wait_for(websocket.recv(), timeout=timeout_seconds)
                except asyncio.TimeoutError:
                    print(f"\n⏱️  No messages received after {timeout_seconds} seconds")
                    print("   This could mean:")
                    print("   - Market is not actively trading")
                    print("   - Subscription format is incorrect")
                    print("   - Need to wait longer for activity")
                    break

                # Check if we need to send PING (every 10 seconds)
                current_time = asyncio.get_event_loop().time()
                if current_time - last_ping > 10:
                    await websocket.send("PING")
                    last_ping = current_time
                    if message_count > 0:  # Only print after first message
                        print(f"   [PING sent to keep alive]")

                # Handle different message types
                if message == "PONG":
                    # Skip PONG responses
                    continue

                try:
                    data = json.loads(message)
                except json.JSONDecodeError:
                    print(f"⚠️  Non-JSON message: {message[:100]}")
                    continue

                # Handle list response (initial subscription confirmation)
                if isinstance(data, list):
                    print(f"📋 Received list response: {len(data)} items")
                    for item in data:
                        if isinstance(item, dict):
                            print(f"   {json.dumps(item, indent=4)[:200]}...")
                    continue

                event_type = data.get("event_type", "unknown")
                timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]

                message_count += 1
                msg_types[event_type] = msg_types.get(event_type, 0) + 1

                print(f"[{message_count:2d}] {timestamp} - {event_type}")

                if event_type == "book":
                    # Order book snapshot
                    asset_id = data.get('asset_id', '')
                    ts = data.get('timestamp', 0)
                    bids = data.get('bids', [])
                    asks = data.get('asks', [])

                    # Determine outcome (UP or DOWN)
                    outcome = "UP" if asset_id == TOKEN_IDS[0] else "DOWN"

                    print(f"     Asset: {outcome} ({asset_id[:20]}...)")
                    print(f"     Timestamp: {ts}")
                    print(f"     Levels: {len(bids)} bids, {len(asks)} asks")

                    if bids and asks:
                        best_bid = float(bids[0]['price'])
                        best_ask = float(asks[0]['price'])
                        spread = best_ask - best_bid
                        print(f"     Best Bid/Ask: ${best_bid:.4f} / ${best_ask:.4f}")
                        print(f"     Spread: ${spread:.4f}")

                elif event_type == "price_change":
                    # Price level update
                    changes = data.get('price_changes', [])
                    print(f"     Changes: {len(changes)} price level updates")
                    for i, change in enumerate(changes[:3]):  # Show first 3
                        side = change.get('side', 'unknown')
                        price = change.get('price', 0)
                        size = change.get('size', 0)
                        print(f"       [{i+1}] {side:4} @ ${price} | Size: {size}")

                elif event_type == "last_trade_price":
                    # Trade execution
                    price = data.get('price', 0)
                    size = data.get('size', 0)
                    side = data.get('side', 'unknown')
                    print(f"     Trade: {side} {size} @ ${price}")

                elif event_type == "tick_size_change":
                    # Tick size update
                    old = data.get('old_tick_size', 0)
                    new = data.get('new_tick_size', 0)
                    print(f"     Tick: {old} → {new}")

                else:
                    # Unknown event
                    print(f"     Data: {json.dumps(data, indent=6)[:200]}...")

                print()  # Blank line between messages

            print("="*80)
            print(f" RECEIVED {message_count} MESSAGES")
            print("="*80)
            print("\n📊 Message Type Summary:")
            for msg_type, count in sorted(msg_types.items()):
                print(f"  {msg_type:20} : {count:3d} messages")

            print("\n✅ Success Criteria:")
            print("  - WebSocket connection: ✅")
            print("  - Market subscription: ✅")
            print(f"  - Received messages: {'✅' if message_count > 0 else '❌'}")
            print(f"  - Book events: {'✅' if msg_types.get('book', 0) > 0 else '❌'}")
            print("\n📌 Next Step: Add parquet writing for data capture")

    except websockets.exceptions.WebSocketException as e:
        print(f"\n❌ WebSocket error: {e}")
    except KeyboardInterrupt:
        print(f"\n\n⚠️  Stopped by user after {message_count} messages")
        print(f"\n📊 Message types received: {msg_types}")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    print("\n🚀 Starting WebSocket test...")
    print("   Duration: Until 20 messages received or Ctrl+C\n")

    try:
        asyncio.run(test_websocket())
    except KeyboardInterrupt:
        print("\n\n👋 Test stopped by user")
