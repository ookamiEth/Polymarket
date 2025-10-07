#!/usr/bin/env python3
"""
WebSocket PING/PONG Investigation for Polymarket
Runs for 5 minutes to discover PING frequency and requirements

Usage: uv run python test_websocket_ping_investigation.py
"""

import asyncio
import websockets
import json
from datetime import datetime
from pathlib import Path


async def investigate_ping_pong():
    """Investigate PING/PONG behavior over 5 minutes"""

    uri = "wss://ws-live-data.polymarket.com/"
    test_duration = 300  # 5 minutes

    # Subscription for BTC/USD prices (to keep connection active)
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

    # Tracking data
    ping_events = []
    connection_log = {
        "test_metadata": {
            "uri": uri,
            "duration_seconds": test_duration,
            "started_at": datetime.now().isoformat()
        },
        "ping_events": [],
        "summary": {}
    }

    print("=" * 80)
    print(" WEBSOCKET PING/PONG INVESTIGATION")
    print("=" * 80)
    print(f"\nConnecting to: {uri}")
    print(f"Test duration: {test_duration} seconds ({test_duration // 60} minutes)")
    print(f"Goal: Discover PING frequency and requirements\n")
    print("-" * 80 + "\n")

    try:
        async with websockets.connect(uri) as websocket:
            connection_start = datetime.now()
            connection_log["connection_info"] = {
                "connected_at": connection_start.isoformat(),
                "status": "connected"
            }

            print(f"✓ Connected at {connection_start.strftime('%H:%M:%S')}")

            # Send subscription
            await websocket.send(json.dumps(subscribe_msg))
            print("✓ Subscribed to crypto_prices topic")
            print("\nMonitoring for PING messages...\n")
            print(f"{'Time':<12} {'Event':<15} {'Elapsed (s)':<12} {'Interval (s)':<12}")
            print("-" * 80)

            # Counters
            message_count = 0
            ping_count = 0
            data_message_count = 0
            last_ping_time = None
            start_time = asyncio.get_event_loop().time()
            test_start = start_time

            while asyncio.get_event_loop().time() - test_start < test_duration:
                try:
                    # Wait for message with timeout
                    message = await asyncio.wait_for(websocket.recv(), timeout=1.0)
                    received_at = datetime.now()
                    elapsed = asyncio.get_event_loop().time() - test_start
                    message_count += 1

                    # Check if it's a PING
                    if message == "PING":
                        ping_count += 1

                        # Calculate interval since last ping
                        interval = None
                        if last_ping_time:
                            interval = (received_at - last_ping_time).total_seconds()

                        # Log the ping
                        ping_event = {
                            "ping_number": ping_count,
                            "timestamp": received_at.isoformat(),
                            "elapsed_seconds": elapsed,
                            "interval_seconds": interval
                        }
                        ping_events.append(ping_event)
                        connection_log["ping_events"].append(ping_event)

                        # Print to console
                        interval_str = f"{interval:.1f}" if interval else "N/A"
                        print(f"{received_at.strftime('%H:%M:%S'):<12} "
                              f"{'PING received':<15} "
                              f"{elapsed:<12.1f} "
                              f"{interval_str:<12}")

                        # Send PONG response
                        await websocket.send("PONG")
                        print(f"{datetime.now().strftime('%H:%M:%S'):<12} "
                              f"{'PONG sent':<15} "
                              f"{asyncio.get_event_loop().time() - test_start:<12.1f}")

                        last_ping_time = received_at

                    else:
                        # It's a data message
                        data_message_count += 1

                        # Parse and identify
                        try:
                            data = json.loads(message)
                            topic = data.get("topic", "unknown")

                            # Only print first few data messages to avoid spam
                            if data_message_count <= 3:
                                print(f"{received_at.strftime('%H:%M:%S'):<12} "
                                      f"{'Data: ' + topic:<15} "
                                      f"{elapsed:<12.1f}")
                        except json.JSONDecodeError:
                            if data_message_count <= 3:
                                print(f"{received_at.strftime('%H:%M:%S'):<12} "
                                      f"{'Non-JSON data':<15} "
                                      f"{elapsed:<12.1f}")

                except asyncio.TimeoutError:
                    # No message in timeout window, continue
                    # Print status update every 30 seconds
                    elapsed = asyncio.get_event_loop().time() - test_start
                    if int(elapsed) % 30 == 0 and elapsed > 0:
                        print(f"\n[Status Update] {int(elapsed)}s elapsed | "
                              f"PINGs: {ping_count} | "
                              f"Messages: {message_count}")
                    continue
                except Exception as msg_error:
                    print(f"⚠️  Error processing message: {msg_error}")
                    continue

            # Test completed
            connection_end = datetime.now()
            connection_log["connection_info"]["disconnected_at"] = connection_end.isoformat()
            connection_log["test_metadata"]["ended_at"] = connection_end.isoformat()

            print("\n" + "-" * 80)
            print(" TEST COMPLETED")
            print("-" * 80 + "\n")

    except websockets.exceptions.WebSocketException as e:
        print(f"\n❌ WebSocket error: {e}")
        connection_log["connection_info"]["error"] = str(e)
    except Exception as e:
        print(f"\n❌ Error: {e}")
        connection_log["connection_info"]["error"] = str(e)

    # Calculate summary statistics
    if ping_events:
        intervals = [p["interval_seconds"] for p in ping_events if p["interval_seconds"] is not None]
        connection_log["summary"] = {
            "total_pings": ping_count,
            "total_data_messages": data_message_count,
            "total_messages": message_count,
            "average_ping_interval_seconds": sum(intervals) / len(intervals) if intervals else None,
            "min_ping_interval_seconds": min(intervals) if intervals else None,
            "max_ping_interval_seconds": max(intervals) if intervals else None,
            "first_ping_at_seconds": ping_events[0]["elapsed_seconds"] if ping_events else None
        }
    else:
        connection_log["summary"] = {
            "total_pings": 0,
            "total_data_messages": data_message_count,
            "total_messages": message_count,
            "note": "No PING messages received during test period"
        }

    # Save log file
    logs_dir = Path(__file__).parent / "logs"
    logs_dir.mkdir(exist_ok=True)
    timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = logs_dir / f"ping_investigation_{timestamp_str}.json"

    with open(log_file, 'w') as f:
        json.dump(connection_log, f, indent=2)

    # Print summary
    print("\n" + "=" * 80)
    print(" PING/PONG INVESTIGATION RESULTS")
    print("=" * 80)
    print(f"\nTest Duration: {test_duration} seconds ({test_duration // 60} minutes)")
    print(f"Total Messages: {message_count}")
    print(f"  - PING messages: {ping_count}")
    print(f"  - Data messages: {data_message_count}")

    if ping_events:
        print(f"\n🔔 PING Behavior Discovered:")
        print(f"  - First PING received after: {ping_events[0]['elapsed_seconds']:.1f} seconds")

        intervals = [p["interval_seconds"] for p in ping_events if p["interval_seconds"] is not None]
        if intervals:
            avg_interval = sum(intervals) / len(intervals)
            print(f"  - Average PING interval: {avg_interval:.1f} seconds ({avg_interval / 60:.1f} minutes)")
            print(f"  - Min interval: {min(intervals):.1f} seconds")
            print(f"  - Max interval: {max(intervals):.1f} seconds")

        print(f"\n📋 All PING Events:")
        for event in ping_events:
            interval_str = f"{event['interval_seconds']:.1f}s" if event['interval_seconds'] else "N/A"
            print(f"  #{event['ping_number']}: "
                  f"{event['timestamp']} | "
                  f"Elapsed: {event['elapsed_seconds']:.1f}s | "
                  f"Interval: {interval_str}")
    else:
        print(f"\n⚠️  NO PING messages received!")
        print(f"\nPossible reasons:")
        print(f"  1. PING interval is longer than {test_duration} seconds")
        print(f"  2. Server doesn't require PING/PONG for this WebSocket")
        print(f"  3. Data messages act as implicit keep-alive")
        print(f"\nRecommendation: Try longer test duration (10-15 minutes)")

    print(f"\n📝 Full log saved to: {log_file}")
    print("\n" + "=" * 80 + "\n")

    return connection_log


if __name__ == "__main__":
    print("\nStarting PING/PONG Investigation...\n")
    print("This will run for 5 minutes. Press Ctrl+C to stop early.\n")

    try:
        result = asyncio.run(investigate_ping_pong())
    except KeyboardInterrupt:
        print("\n\n⚠️  Test interrupted by user (Ctrl+C)\n")
    except Exception as e:
        print(f"❌ Error: {e}\n")
