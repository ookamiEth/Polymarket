#!/usr/bin/env python3
"""
Test script for Polymarket Live Data WebSocket
Tests connection to wss://ws-live-data.polymarket.com/ with subscriptions to:
- crypto_prices_chainlink (BTC/USD) - PRIMARY FOCUS
- activity (orders_matched)
- comments (Series entity)

Runs for 30 seconds and logs all data as JSON
"""

import asyncio
import websockets
import json
from datetime import datetime
from pathlib import Path


async def test_polymarket_live_websocket():
    """Connect to Polymarket live data WebSocket and test subscriptions"""

    uri = "wss://ws-live-data.polymarket.com/"

    # Create logs directory
    logs_dir = Path(__file__).parent / "logs"
    logs_dir.mkdir(exist_ok=True)

    # Prepare log file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = logs_dir / f"websocket_polymarket_live_{timestamp}.json"

    # Subscription message with all three topics
    subscribe_msg = {
        "action": "subscribe",
        "subscriptions": [
            {
                "topic": "crypto_prices_chainlink",
                "type": "update",
                "filters": json.dumps({"symbol": "btc/usd"})
            },
            {
                "topic": "activity",
                "type": "orders_matched",
                "filters": json.dumps({"event_slug": "btc-up-or-down-15m-1759768200"})
            },
            {
                "topic": "comments",
                "type": "*",
                "filters": json.dumps({"parentEntityID": 10192, "parentEntityType": "Series"})
            }
        ]
    }

    # Log structure
    test_log = {
        "test_metadata": {
            "timestamp_start": datetime.now().isoformat(),
            "uri": uri,
            "duration_seconds": 30,
            "subscription_message": subscribe_msg
        },
        "connection_info": {},
        "messages": [],
        "summary": {}
    }

    print("=" * 80)
    print(" POLYMARKET LIVE DATA WEBSOCKET TEST")
    print("=" * 80)
    print(f"\nConnecting to: {uri}")
    print(f"\nSubscriptions:")
    for sub in subscribe_msg["subscriptions"]:
        print(f"  - {sub['topic']} ({sub['type']})")
        print(f"    Filters: {sub['filters']}")
    print(f"\nTest duration: 30 seconds")
    print(f"Log file: {log_file}")
    print("\n" + "-" * 80 + "\n")

    try:
        async with websockets.connect(uri) as websocket:

            # Log connection info
            test_log["connection_info"] = {
                "connected_at": datetime.now().isoformat(),
                "status": "connected",
                "protocol": "WebSocket"
            }
            print("✓ Connected successfully!")
            print(f"\nSending subscription message...")
            print(json.dumps(subscribe_msg, indent=2))
            print("\n" + "-" * 80 + "\n")

            # Send subscription
            await websocket.send(json.dumps(subscribe_msg))
            print("✓ Subscription sent!")
            print("\nListening for messages (30 seconds)...\n")

            # Message counters
            message_count = 0
            topic_counts = {
                "crypto_prices_chainlink": 0,
                "activity": 0,
                "comments": 0,
                "ping": 0,
                "other": 0
            }

            # Run for 30 seconds
            start_time = asyncio.get_event_loop().time()

            while asyncio.get_event_loop().time() - start_time < 30:
                try:
                    # Wait for message with timeout
                    message = await asyncio.wait_for(websocket.recv(), timeout=1.0)
                    message_count += 1

                    # Parse message
                    try:
                        data = json.loads(message)
                    except json.JSONDecodeError:
                        # Handle non-JSON messages (like PING)
                        if message == "PING":
                            topic_counts["ping"] += 1
                            msg_log = {
                                "message_number": message_count,
                                "timestamp": datetime.now().isoformat(),
                                "type": "ping",
                                "raw_message": message
                            }
                            test_log["messages"].append(msg_log)
                            print(f"[{message_count}] PING received - sending PONG")
                            await websocket.send("PONG")
                            continue
                        else:
                            data = {"raw": message}
                            topic_counts["other"] += 1

                    # Determine topic
                    topic = data.get("topic", "unknown")
                    msg_type = data.get("type", "unknown")

                    # Count by topic
                    if topic in topic_counts:
                        topic_counts[topic] += 1
                    else:
                        topic_counts["other"] += 1

                    # Log the message
                    msg_log = {
                        "message_number": message_count,
                        "timestamp": datetime.now().isoformat(),
                        "topic": topic,
                        "type": msg_type,
                        "data": data
                    }
                    test_log["messages"].append(msg_log)

                    # Print to console with focus on crypto prices
                    print(f"[{message_count}] {datetime.now().strftime('%H:%M:%S')} - Topic: {topic} | Type: {msg_type}")

                    if topic == "crypto_prices_chainlink":
                        print("  🔥 CRYPTO PRICE UPDATE (PRIMARY FOCUS):")
                        print(f"     {json.dumps(data, indent=6)}")
                    elif topic == "activity":
                        print(f"  Activity event: {json.dumps(data, indent=4)}")
                    elif topic == "comments":
                        print(f"  Comment event: {json.dumps(data, indent=4)}")
                    else:
                        print(f"  Data: {json.dumps(data, indent=4)}")

                    print()

                except asyncio.TimeoutError:
                    # No message received in timeout window, continue
                    continue
                except Exception as msg_error:
                    print(f"  Error processing message: {msg_error}")
                    test_log["messages"].append({
                        "message_number": message_count,
                        "timestamp": datetime.now().isoformat(),
                        "error": str(msg_error)
                    })

            # Test completed
            print("\n" + "-" * 80)
            print(" TEST COMPLETED (30 seconds elapsed)")
            print("-" * 80 + "\n")

    except websockets.exceptions.WebSocketException as e:
        error_msg = f"WebSocket error: {e}"
        print(f"\n❌ {error_msg}")
        test_log["connection_info"]["error"] = str(e)
        test_log["connection_info"]["status"] = "error"
    except Exception as e:
        error_msg = f"Error: {e}"
        print(f"\n❌ {error_msg}")
        test_log["connection_info"]["error"] = str(e)
        test_log["connection_info"]["status"] = "error"

    # Add summary
    test_log["test_metadata"]["timestamp_end"] = datetime.now().isoformat()
    test_log["summary"] = {
        "total_messages": message_count,
        "messages_by_topic": topic_counts,
        "crypto_prices_received": topic_counts["crypto_prices_chainlink"] > 0,
        "activity_messages_received": topic_counts["activity"] > 0,
        "comments_received": topic_counts["comments"] > 0,
        "ping_pong_working": topic_counts["ping"] > 0
    }

    # Print summary
    print("\n" + "=" * 80)
    print(" SUMMARY")
    print("=" * 80)
    print(f"\nTotal messages received: {message_count}")
    print(f"\nMessages by topic:")
    for topic, count in topic_counts.items():
        status = "✓" if count > 0 else "✗"
        print(f"  {status} {topic}: {count}")

    print(f"\n\nKey Findings:")
    if topic_counts["crypto_prices_chainlink"] > 0:
        print(f"  ✓ Crypto price updates working! ({topic_counts['crypto_prices_chainlink']} messages)")
    else:
        print(f"  ✗ No crypto price updates received (may need time or different symbol)")

    if topic_counts["ping"] > 0:
        print(f"  ✓ Heartbeat (PING/PONG) working")

    print(f"\n\nWhat data can we get from crypto_prices_chainlink topic:")
    crypto_messages = [m for m in test_log["messages"] if m.get("topic") == "crypto_prices_chainlink"]
    if crypto_messages:
        print(f"  Sample message structure:")
        print(json.dumps(crypto_messages[0].get("data", {}), indent=4))
    else:
        print(f"  No messages received yet. May need:")
        print(f"    - Longer connection time")
        print(f"    - Different symbol or filters")
        print(f"    - Market to be active")

    # Save log file
    with open(log_file, 'w') as f:
        json.dump(test_log, f, indent=2)

    print(f"\n\n📝 Full log saved to: {log_file}")
    print("\n" + "=" * 80 + "\n")


if __name__ == "__main__":
    print("\nStarting Polymarket Live Data WebSocket Test...")
    print("(Requires websockets library - should be installed via uv)\n")

    try:
        asyncio.run(test_polymarket_live_websocket())
    except KeyboardInterrupt:
        print("\n\n⚠️  Test interrupted by user (Ctrl+C)")
    except ImportError as e:
        print(f"❌ Error: {e}")
        print("Install dependencies with: uv pip install websockets")
