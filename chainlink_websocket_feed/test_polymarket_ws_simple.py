#!/usr/bin/env python3
"""
Simple Polymarket WebSocket Test
Demonstrates successful connection to wss://ws-live-data.polymarket.com/
"""

import asyncio
import json
import signal
from datetime import datetime

import websockets


async def test_connection():
    """Test WebSocket connection and display raw messages"""
    uri = "wss://ws-live-data.polymarket.com/"

    print("=" * 80)
    print("POLYMARKET WEBSOCKET CONNECTION TEST")
    print("=" * 80)
    print(f"\n🔌 Connecting to {uri}...")

    message_count = 0
    shutdown = False

    def signal_handler(sig, frame):
        nonlocal shutdown
        print("\n\n✋ Shutting down...")
        shutdown = True

    signal.signal(signal.SIGINT, signal_handler)

    try:
        async with websockets.connect(uri) as ws:
            print("✅ CONNECTED SUCCESSFULLY!\n")

            # Subscribe to BTC/USD prices
            subscribe_msg = {
                "action": "subscribe",
                "subscriptions": [
                    {
                        "topic": "crypto_prices_chainlink",
                        "type": "update",
                        "filters": '{"symbol":"btc/usd"}',
                    },
                ],
            }

            print("📡 Sending subscription:")
            print(json.dumps(subscribe_msg, indent=2))
            await ws.send(json.dumps(subscribe_msg))
            print("\n✅ Subscription sent")
            print("📊 Listening for messages (Press Ctrl+C to stop)...\n")
            print("-" * 80)

            # Receive messages
            while not shutdown and message_count < 20:  # Limit to 20 messages for demo
                try:
                    message = await asyncio.wait_for(ws.recv(), timeout=1.0)
                    message_count += 1
                    timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]

                    # Try to parse as JSON
                    try:
                        data = json.loads(message)
                        print(
                            f"[{timestamp}] MESSAGE #{message_count}: {json.dumps(data)[:200]}"
                        )

                        # If it's a price update, highlight it
                        if (
                            isinstance(data, dict)
                            and data.get("topic") == "crypto_prices_chainlink"
                        ):
                            print(f"             ^^ PRICE UPDATE RECEIVED! ^^")

                    except json.JSONDecodeError:
                        print(f"[{timestamp}] RAW MESSAGE: {message[:150]}")

                except asyncio.TimeoutError:
                    continue

            print("\n" + "=" * 80)
            print(f"✅ TEST COMPLETE")
            print(f"   Total messages received: {message_count}")
            print(f"   Connection: WORKING ✓")
            print("=" * 80)

    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(test_connection())
