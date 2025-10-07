#!/usr/bin/env python3
"""
Test script for Polymarket WebSocket User Channel
NOW FULLY FUNCTIONAL with your API credentials from .env!

NOTE:
1. Requires websockets library: pip install websockets
2. Uses your authenticated credentials from .env
3. Monitors real-time trades and orders for your account
"""

import os
import asyncio
import websockets
import json
from dotenv import load_dotenv

async def test_user_websocket_live():
    """Connect to User WebSocket with your real credentials"""

    # Load credentials
    load_dotenv()
    api_key = os.getenv('POLY_API_KEY')
    api_secret = os.getenv('POLY_API_SECRET')
    api_passphrase = os.getenv('POLY_API_PASSPHRASE')

    if not all([api_key, api_secret, api_passphrase]):
        print("❌ ERROR: Missing API credentials in .env file!")
        print("Please run ../../setup_polymarket_api.py first")
        return

    uri = "wss://ws-subscriptions-clob.polymarket.com/ws/user"

    print("="*80)
    print(" WEBSOCKET USER CHANNEL TEST (LIVE)")
    print("="*80)
    print("\n✅ Credentials loaded from .env")
    print(f"   API Key: {api_key[:20]}...")
    print(f"   WebSocket URI: {uri}\n")

    print("🔄 Connecting to WebSocket...")

    try:
        async with websockets.connect(uri) as websocket:
            print("✅ Connected to WebSocket!")

            # Subscribe with your real credentials
            subscribe_msg = {
                "auth": {
                    "apiKey": api_key,
                    "secret": api_secret,
                    "passphrase": api_passphrase
                },
                "markets": [],  # Empty = all markets
                "assets_ids": [],  # Empty = all assets
                "type": "user"
            }

            print("\n📤 Sending subscription message...")
            await websocket.send(json.dumps(subscribe_msg))
            print("✅ Subscription sent!")

            print("\n" + "="*80)
            print(" LISTENING FOR EVENTS (30 seconds)")
            print("="*80)
            print("\nWaiting for trades/orders...")
            print("(If you haven't traded recently, you may not see any messages)")

            # Listen for messages for 30 seconds
            message_count = 0
            try:
                async with asyncio.timeout(30):
                    while True:
                        message = await websocket.recv()
                        data = json.loads(message)
                        message_count += 1

                        print(f"\n📨 Message #{message_count}:")
                        print(json.dumps(data, indent=2))

                        if data.get("event_type") == "trade":
                            print(f"\n✅ TRADE: {data.get('side')} {data.get('size')} @ ${data.get('price')}")
                            print(f"   Status: {data.get('status')}")
                        elif data.get("event_type") == "order":
                            print(f"\n📋 ORDER {data.get('type')}: {data.get('order_id')[:20]}...")
                            print(f"   Status: {data.get('status')}")

            except asyncio.TimeoutError:
                if message_count == 0:
                    print("\nℹ️  No messages received in 30 seconds")
                    print("   This is normal if you haven't traded recently")
                else:
                    print(f"\n✅ Received {message_count} message(s)")

    except Exception as e:
        print(f"\n❌ Error: {e}")

    print("\n" + "="*80)
    print(" EXPECTED MESSAGE TYPES")
    print("="*80)

    # Trade Message Example
    print("\n1. TRADE MESSAGE")
    print("-" * 80)

    trade_example = {
        "event_type": "trade",
        "id": "28c4d2eb-bbea-40e7-a9f0-b2fdb56b2c2e",
        "asset_id": "52114319501245915516055106046884209969926127482827954674443846427813813222426",
        "market": "0xbd31dc8a20211944f6b70f31557f1001557b59905b7738480ca09bd4532f84af",
        "side": "BUY",
        "price": "0.57",
        "size": "10",
        "status": "MATCHED",
        "matchtime": "1672290701",
        "last_update": "1672290701",
        "timestamp": "1672290701",
        "outcome": "YES",
        "taker_order_id": "0x06bc63e346ed4ceddce9efd6b3af37c8f8f440c92fe7da6b2d0f9e4ccbc50c42",
        "maker_orders": [
            {
                "order_id": "0xff354cd7ca7539dfa9c28d90943ab5779a4eac34b9b37a757d7b32bdfb11790b",
                "asset_id": "52114319501245915516055106046884209969926127482827954674443846427813813222426",
                "price": "0.57",
                "matched_amount": "10",
                "outcome": "YES"
            }
        ],
        "type": "TRADE"
    }

    print("\nTrade Message Structure:")
    print(json.dumps(trade_example, indent=2))

    print("\n\nTrade Status Progression:")
    print("  MATCHED   → Trade matched by operator")
    print("  MINED     → Transaction mined on-chain")
    print("  CONFIRMED → Final confirmation (terminal)")
    print("  RETRYING  → Transaction failed, retrying")
    print("  FAILED    → Permanently failed (terminal)")

    # Order Message Example
    print("\n\n2. ORDER MESSAGE")
    print("-" * 80)

    order_example = {
        "event_type": "order",
        "asset_id": "52114319501245915516055106046884209969926127482827954674443846427813813222426",
        "order_id": "0x06bc63e346ed4ceddce9efd6b3af37c8f8f440c92fe7da6b2d0f9e4ccbc50c42",
        "market": "0xbd31dc8a20211944f6b70f31557f1001557b59905b7738480ca09bd4532f84af",
        "side": "BUY",
        "price": "0.57",
        "original_size": "100",
        "size_matched": "10",
        "status": "LIVE",
        "timestamp": "1672290701",
        "type": "PLACEMENT"
    }

    print("\nOrder Message Structure:")
    print(json.dumps(order_example, indent=2))

    print("\n\nOrder Event Types:")
    print("  PLACEMENT    → New order placed")
    print("  UPDATE       → Order partially filled")
    print("  CANCELLATION → Order cancelled")

    print("\n\nOrder Status Values:")
    print("  LIVE      → Order active in book")
    print("  MATCHED   → Order fully/partially matched")
    print("  CANCELLED → Order cancelled")

    print("\n" + "="*80)
    print(" USE CASES")
    print("="*80)

    use_cases = [
        "Real-time trade execution monitoring",
        "Order status tracking",
        "Automated trading systems",
        "Portfolio management tools",
        "Risk management alerts",
        "Trade confirmation workflows"
    ]

    for i, use_case in enumerate(use_cases, 1):
        print(f"  {i}. {use_case}")

    print("\n" + "="*80)
    print(" IMPLEMENTATION EXAMPLE")
    print("="*80)

    code_example = '''
import asyncio
import websockets
import json

async def user_channel():
    uri = "wss://ws-subscriptions-clob.polymarket.com/ws/user"

    async with websockets.connect(uri) as websocket:
        # Subscribe with authentication
        subscribe = {
            "auth": {
                "apiKey": "your-api-key",
                "secret": "your-secret",
                "passphrase": "your-passphrase"
            },
            "markets": [],
            "assets_ids": [],
            "type": "user"
        }

        await websocket.send(json.dumps(subscribe))

        while True:
            message = await websocket.recv()
            data = json.loads(message)

            if data["event_type"] == "trade":
                print(f"Trade: {data['side']} {data['size']} @ {data['price']}")
                print(f"Status: {data['status']}")

            elif data["event_type"] == "order":
                print(f"Order {data['type']}: {data['order_id']}")
                print(f"Status: {data['status']}")

asyncio.run(user_channel())
'''

    print(code_example)

    print("\n" + "="*80)
    print(" TEST COMPLETED")
    print("="*80)
    print("\nTo use this endpoint:")
    print("  1. Obtain API credentials from Polymarket")
    print("  2. Install websockets: pip install websockets")
    print("  3. Implement authentication in subscription message")
    print("  4. Handle trade and order messages")
    print("  5. Implement reconnection logic for production use")

if __name__ == "__main__":
    print("\nStarting WebSocket User Channel Live Test...")
    print("Using your credentials from .env\n")

    asyncio.run(test_user_websocket_live())
