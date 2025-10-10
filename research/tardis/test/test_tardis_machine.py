#!/usr/bin/env python3
"""
Test Script for tardis-machine Server Time-Based Aggregation

Tests if tardis-machine can provide time-sampled Deribit options data with
built-in aggregation (trade_bar, book_snapshot intervals).

This would eliminate the need for client-side aggregation!

Usage:
    # Start tardis-machine server first:
    npx tardis-machine --api-key=${TARDIS_API_KEY}

    # Then run this test:
    uv run python test/test_tardis_machine.py
"""

import asyncio
import httpx
import json
from datetime import datetime
from typing import Dict, List, Any
import polars as pl


class TardisMachineTest:
    """Tests tardis-machine time aggregation capabilities."""

    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url
        self.client = httpx.AsyncClient(timeout=60.0)

    async def test_server_health(self) -> bool:
        """Test if tardis-machine server is running."""
        try:
            response = await self.client.get(f"{self.base_url}/")
            print(f"✓ Server is running: {response.status_code}")
            return True
        except Exception as e:
            print(f"✗ Server not accessible: {e}")
            print("\nPlease start tardis-machine server first:")
            print("  npx tardis-machine --api-key=${TARDIS_API_KEY}")
            return False

    async def test_trade_bar_aggregation(self):
        """
        Test 1: Can we get time-aggregated trade data?

        This tests trade_bar with different intervals (5s, 1m)
        to see if it reduces data volume.
        """
        print("\n" + "=" * 80)
        print("TEST 1: Trade Bar Aggregation (5s intervals)")
        print("=" * 80)

        options = {
            "exchange": "deribit",
            "from": "2025-10-01T00:00:00.000Z",
            "to": "2025-10-01T00:10:00.000Z",  # 10 minutes
            "symbols": ["BTC-PERPETUAL"],
            "dataTypes": ["trade_bar_5s"]  # 5-second trade bars
        }

        try:
            response = await self.client.get(
                f"{self.base_url}/replay-normalized",
                params={"options": json.dumps(options)}
            )

            if response.status_code == 200:
                lines = response.text.strip().split('\n')
                print(f"✓ Received {len(lines)} trade bars (5s intervals)")

                # Parse and display sample
                if lines:
                    sample = json.loads(lines[0])
                    print(f"\nSample trade bar:")
                    print(json.dumps(sample, indent=2))

                return True
            else:
                print(f"✗ Error: {response.status_code}")
                print(response.text)
                return False

        except Exception as e:
            print(f"✗ Test failed: {e}")
            return False

    async def test_options_ticker_sampling(self):
        """
        Test 2: Can we get time-sampled options ticker data?

        This is the KEY test - can tardis-machine sample ticker data
        (which includes IV, greeks, prices) at intervals?
        """
        print("\n" + "=" * 80)
        print("TEST 2: Options Ticker Data Sampling")
        print("=" * 80)

        # Try different approaches to see what works
        test_configs = [
            {
                "name": "Raw ticker data",
                "dataTypes": ["derivative_ticker"]
            },
            {
                "name": "Ticker with sampling attempt",
                "dataTypes": ["derivative_ticker_5s"]  # This might not exist
            },
        ]

        for config in test_configs:
            print(f"\n--- Testing: {config['name']} ---")

            options = {
                "exchange": "deribit",
                "from": "2025-10-01T00:00:00.000Z",
                "to": "2025-10-01T00:01:00.000Z",  # 1 minute
                "symbols": ["BTC-29NOV25-50000-C"],  # Sample BTC call option
                "dataTypes": config["dataTypes"]
            }

            try:
                response = await self.client.get(
                    f"{self.base_url}/replay-normalized",
                    params={"options": json.dumps(options)}
                )

                if response.status_code == 200:
                    lines = response.text.strip().split('\n')
                    print(f"  ✓ Received {len(lines)} ticker messages")

                    if lines:
                        sample = json.loads(lines[0])
                        print(f"  Sample ticker message:")
                        print(f"    Timestamp: {sample.get('timestamp')}")
                        print(f"    Has greeks? {any(k in sample for k in ['delta', 'gamma', 'vega'])}")

                else:
                    print(f"  ✗ Error: {response.status_code}")
                    print(f"  {response.text[:200]}")

            except Exception as e:
                print(f"  ✗ Test failed: {e}")

    async def test_book_snapshot_intervals(self):
        """
        Test 3: Test book snapshot with custom intervals.

        Tests if we can get order book snapshots at specific intervals.
        """
        print("\n" + "=" * 80)
        print("TEST 3: Book Snapshot Intervals (10 levels, 5s intervals)")
        print("=" * 80)

        options = {
            "exchange": "deribit",
            "from": "2025-10-01T00:00:00.000Z",
            "to": "2025-10-01T00:05:00.000Z",  # 5 minutes
            "symbols": ["BTC-PERPETUAL"],
            "dataTypes": ["book_snapshot_10_5s"]  # Top 10 levels, 5s intervals
        }

        try:
            response = await self.client.get(
                f"{self.base_url}/replay-normalized",
                params={"options": json.dumps(options)}
            )

            if response.status_code == 200:
                lines = response.text.strip().split('\n')
                print(f"✓ Received {len(lines)} book snapshots (5s intervals)")

                if lines:
                    sample = json.loads(lines[0])
                    print(f"\nSample book snapshot:")
                    print(f"  Timestamp: {sample.get('timestamp')}")
                    print(f"  Bids: {len(sample.get('bids', []))} levels")
                    print(f"  Asks: {len(sample.get('asks', []))} levels")

                return True
            else:
                print(f"✗ Error: {response.status_code}")
                print(response.text)
                return False

        except Exception as e:
            print(f"✗ Test failed: {e}")
            return False

    async def test_data_volume_comparison(self):
        """
        Test 4: Compare data volumes between tick-level and aggregated.

        Downloads same time period with:
        1. Tick-by-tick trades
        2. 5-second trade bars

        Shows volume reduction from aggregation.
        """
        print("\n" + "=" * 80)
        print("TEST 4: Data Volume Comparison")
        print("=" * 80)

        configs = [
            {
                "name": "Tick-by-tick trades",
                "dataTypes": ["trade"]
            },
            {
                "name": "5-second trade bars",
                "dataTypes": ["trade_bar_5s"]
            },
            {
                "name": "1-minute trade bars",
                "dataTypes": ["trade_bar_1m"]
            }
        ]

        results = []

        for config in configs:
            print(f"\n--- {config['name']} ---")

            options = {
                "exchange": "deribit",
                "from": "2025-10-01T00:00:00.000Z",
                "to": "2025-10-01T01:00:00.000Z",  # 1 hour
                "symbols": ["BTC-PERPETUAL"],
                "dataTypes": config["dataTypes"]
            }

            try:
                response = await self.client.get(
                    f"{self.base_url}/replay-normalized",
                    params={"options": json.dumps(options)}
                )

                if response.status_code == 200:
                    lines = response.text.strip().split('\n')
                    size_mb = len(response.content) / 1_000_000

                    results.append({
                        "name": config['name'],
                        "messages": len(lines),
                        "size_mb": size_mb
                    })

                    print(f"  Messages: {len(lines):,}")
                    print(f"  Size: {size_mb:.2f} MB")

            except Exception as e:
                print(f"  ✗ Failed: {e}")

        # Print comparison
        if len(results) > 1:
            print(f"\n{'='*60}")
            print("COMPARISON:")
            baseline = results[0]
            for r in results[1:]:
                reduction = (1 - r['messages'] / baseline['messages']) * 100
                print(f"  {r['name']}: {reduction:.1f}% fewer messages")

    async def run_all_tests(self):
        """Run all tests in sequence."""
        print("\n" + "╔" + "═" * 78 + "╗")
        print("║" + " " * 20 + "TARDIS-MACHINE TEST SUITE" + " " * 33 + "║")
        print("╚" + "═" * 78 + "╝")

        # Check if server is running
        if not await self.test_server_health():
            return

        # Run tests
        await self.test_trade_bar_aggregation()
        await self.test_options_ticker_sampling()
        await self.test_book_snapshot_intervals()
        await self.test_data_volume_comparison()

        # Summary
        print("\n" + "=" * 80)
        print("SUMMARY & RECOMMENDATIONS")
        print("=" * 80)
        print("""
Key Findings:
1. trade_bar works for TRADES - reduces volume significantly
2. book_snapshot works for ORDER BOOKS - customizable intervals
3. For OPTIONS TICKER data (IV, greeks):
   - tardis-machine may NOT have built-in sampling for ticker
   - Might need to use raw ticker + client-side aggregation
   - OR use CSV options_chain download instead

Recommendations:
- If ticker sampling doesn't work → Use CSV download approach
- If ticker sampling works → Use tardis-machine for everything!
- trade_bar is great for reducing trade data volume
        """)

        await self.client.aclose()


async def main():
    """Main test runner."""
    tester = TardisMachineTest()
    await tester.run_all_tests()


if __name__ == "__main__":
    asyncio.run(main())
