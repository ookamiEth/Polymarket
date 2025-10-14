#!/usr/bin/env python3
"""
Test Polymarket WebSocket Connection
Simple script to verify connection to wss://ws-live-data.polymarket.com/
"""

import asyncio
import json
import signal
import sys
from datetime import datetime

import websockets


class PolymarketWebSocketTest:
    """Test Polymarket WebSocket connection"""

    def __init__(self):
        self.uri = "wss://ws-live-data.polymarket.com/"
        self.shutdown = False
        self.message_count = 0

    async def connect_and_test(self):
        """Connect to WebSocket and subscribe to channels"""
        print("=" * 80)
        print("POLYMARKET WEBSOCKET CONNECTION TEST")
        print("=" * 80)
        print(f"\n🔌 Connecting to {self.uri}...")

        try:
            async with websockets.connect(self.uri) as ws:
                print("✅ Connected successfully!\n")

                # Subscribe to multiple channels based on browser DevTools data
                subscribe_msg = {
                    "action": "subscribe",
                    "subscriptions": [
                        {
                            "topic": "crypto_prices_chainlink",
                            "type": "update",
                            "filters": '{"symbol":"btc/usd"}',
                        },
                        {
                            "topic": "activity",
                            "type": "orders_matched",
                            "filters": '{"event_slug":"btc-updown-15m-1760364000"}',
                        },
                    ],
                }

                print("📡 Sending subscription message:")
                print(json.dumps(subscribe_msg, indent=2))
                print()

                await ws.send(json.dumps(subscribe_msg))
                print("✅ Subscription sent\n")
                print("📊 Listening for messages (Press Ctrl+C to stop)...\n")
                print("-" * 80)

                # Receive and display messages
                while not self.shutdown:
                    timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]
                    try:
                        message = await asyncio.wait_for(ws.recv(), timeout=1.0)
                        self.message_count += 1

                        # Parse and display message
                        try:
                            data = json.loads(message)

                            # Handle different message types
                            if "topic" in data:
                                topic = data.get("topic")
                                msg_type = data.get("type")

                                if topic == "crypto_prices_chainlink":
                                    symbol = data.get("symbol", "N/A")
                                    price = data.get("price")
                                    if price and isinstance(price, (int, float)):
                                        print(
                                            f"[{timestamp}] 💰 PRICE UPDATE: {symbol.upper()} = ${price:,.2f}"
                                        )
                                    else:
                                        print(
                                            f"[{timestamp}] 💰 PRICE UPDATE: {symbol} = {data}"
                                        )

                                elif topic == "activity":
                                    event = data.get("event", {})
                                    print(
                                        f"[{timestamp}] 📈 ACTIVITY: {msg_type} - {json.dumps(event)[:100]}..."
                                    )

                                else:
                                    print(
                                        f"[{timestamp}] 📬 MESSAGE #{self.message_count}: topic={topic}, type={msg_type}"
                                    )

                            elif "heartbeat" in data:
                                print(
                                    f"[{timestamp}] 💓 HEARTBEAT: {data['heartbeat']}"
                                )

                            else:
                                # Unknown message format
                                print(
                                    f"[{timestamp}] ❓ UNKNOWN: {json.dumps(data)[:150]}"
                                )

                        except json.JSONDecodeError:
                            print(f"[{timestamp}] ⚠️  Non-JSON message: {message[:100]}")

                    except asyncio.TimeoutError:
                        # No message in 1 second, continue
                        continue

                    except websockets.exceptions.ConnectionClosed:
                        print("\n❌ WebSocket connection closed by server")
                        break

        except Exception as e:
            print(f"\n❌ Connection error: {e}")
            raise

    async def run(self):
        """Run with reconnection logic"""
        print("Press Ctrl+C to stop\n")

        while not self.shutdown:
            try:
                await self.connect_and_test()
            except KeyboardInterrupt:
                print("\n\n✋ Shutdown requested by user")
                self.shutdown = True
                break
            except Exception as e:
                if not self.shutdown:
                    print(f"\n🔄 Connection lost: {e}")
                    print("Reconnecting in 5 seconds...")
                    await asyncio.sleep(5)
                else:
                    break

        print("\n" + "=" * 80)
        print(f"TEST COMPLETE - Received {self.message_count} messages")
        print("=" * 80)


def setup_signal_handlers(test_instance):
    """Setup signal handlers for graceful shutdown"""

    def signal_handler(signum, frame):
        print(f"\n⚠️  Received signal {signum}")
        test_instance.shutdown = True

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)


def main():
    """Main entry point"""
    test = PolymarketWebSocketTest()
    setup_signal_handlers(test)

    try:
        asyncio.run(test.run())
    except KeyboardInterrupt:
        print("\n✋ Stopped")
        sys.exit(0)


if __name__ == "__main__":
    main()
