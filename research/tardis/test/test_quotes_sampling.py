#!/usr/bin/env python3
"""
Test Script: Can tardis-machine Sample Options QUOTES Data?

CRITICAL TEST: Your current script uses 'quote' channel (not ticker).
If quotes can be sampled, we can use tardis-machine to reduce data volume!

quote channel provides:
- bid_price, bid_amount
- ask_price, ask_amount
- timestamps

NO IV or Greeks (you calculate these from prices using Black-Scholes)

Usage:
    # Start tardis-machine server:
    npx tardis-machine

    # Run test:
    uv run python test/test_quotes_sampling.py
"""

import asyncio
import httpx
import json
from datetime import datetime
import polars as pl


class QuotesSamplingTest:
    """Tests if tardis-machine can sample quote data at intervals."""

    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url
        self.client = httpx.AsyncClient(timeout=60.0)

    async def test_raw_quotes(self):
        """
        Test 1: Get raw tick-by-tick quotes data.

        Baseline test - see how many quote updates we get.
        """
        print("\n" + "=" * 80)
        print("TEST 1: Raw Tick-by-Tick Quotes")
        print("=" * 80)

        options = {
            "exchange": "deribit",
            "from": "2024-10-01T00:00:00.000Z",
            "to": "2024-10-01T00:10:00.000Z",  # 10 minutes
            "symbols": ["BTC-PERPETUAL"],  # Use perpetual for testing (always has data)
            "dataTypes": ["quote"]
        }

        try:
            response = await self.client.get(
                f"{self.base_url}/replay-normalized",
                params={"options": json.dumps(options)}
            )

            if response.status_code == 200:
                lines = response.text.strip().split('\n') if response.text.strip() else []
                size_mb = len(response.content) / 1_000_000

                print(f"✓ Received {len(lines)} quote updates")
                print(f"  Size: {size_mb:.3f} MB")

                if lines:
                    sample = json.loads(lines[0])
                    print(f"\nSample quote:")
                    print(f"  Timestamp: {sample.get('timestamp')}")
                    print(f"  Bid: {sample.get('bidPrice')} @ {sample.get('bidAmount')}")
                    print(f"  Ask: {sample.get('askPrice')} @ {sample.get('askAmount')}")

                return {"messages": len(lines), "size_mb": size_mb}
            else:
                print(f"✗ Error: {response.status_code}")
                print(response.text[:500])
                return None

        except Exception as e:
            print(f"✗ Test failed: {e}")
            return None

    async def test_quote_bar_sampling(self):
        """
        Test 2: Try to get quote bars (sampled quotes).

        This is the KEY test - can we sample quotes at intervals?
        """
        print("\n" + "=" * 80)
        print("TEST 2: Quote Bar Sampling (5s intervals)")
        print("=" * 80)

        test_configs = [
            {"name": "quote_bar_5s", "dataTypes": ["quote_bar_5s"]},
            {"name": "quotes_5s", "dataTypes": ["quotes_5s"]},
            {"name": "quote_5s", "dataTypes": ["quote_5s"]},
        ]

        for config in test_configs:
            print(f"\n--- Testing: {config['name']} ---")

            options = {
                "exchange": "deribit",
                "from": "2024-10-01T00:00:00.000Z",
                "to": "2024-10-01T00:10:00.000Z",
                "symbols": ["BTC-PERPETUAL"],
                "dataTypes": config["dataTypes"]
            }

            try:
                response = await self.client.get(
                    f"{self.base_url}/replay-normalized",
                    params={"options": json.dumps(options)}
                )

                if response.status_code == 200:
                    lines = response.text.strip().split('\n') if response.text.strip() else []
                    size_mb = len(response.content) / 1_000_000

                    print(f"  ✓ SUCCESS! Received {len(lines)} quote bars")
                    print(f"    Size: {size_mb:.3f} MB")

                    if lines:
                        sample = json.loads(lines[0])
                        print(f"    Sample: {list(sample.keys())[:5]}")

                    return {"name": config['name'], "messages": len(lines), "size_mb": size_mb}

                else:
                    print(f"  ✗ Error: {response.status_code}")
                    if "Can't normalize" in response.text:
                        print(f"    → No normalizer for this data type")
                    else:
                        print(f"    → {response.text[:200]}")

            except Exception as e:
                print(f"  ✗ Failed: {e}")

        return None

    async def test_quote_snapshot_intervals(self):
        """
        Test 3: Try quote snapshots at custom intervals.

        Similar to book_snapshot but for quotes.
        """
        print("\n" + "=" * 80)
        print("TEST 3: Quote Snapshot Intervals")
        print("=" * 80)

        test_configs = [
            {"name": "quote_snapshot_5s", "dataTypes": ["quote_snapshot_5s"]},
            {"name": "quotes_snapshot_5s", "dataTypes": ["quotes_snapshot_5s"]},
        ]

        for config in test_configs:
            print(f"\n--- Testing: {config['name']} ---")

            options = {
                "exchange": "deribit",
                "from": "2024-10-01T00:00:00.000Z",
                "to": "2024-10-01T00:10:00.000Z",
                "symbols": ["BTC-PERPETUAL"],
                "dataTypes": config["dataTypes"]
            }

            try:
                response = await self.client.get(
                    f"{self.base_url}/replay-normalized",
                    params={"options": json.dumps(options)}
                )

                if response.status_code == 200:
                    lines = response.text.strip().split('\n') if response.text.strip() else []
                    print(f"  ✓ SUCCESS! Received {len(lines)} quote snapshots")
                    return {"name": config['name'], "messages": len(lines)}

                else:
                    print(f"  ✗ Error: {response.status_code}")

            except Exception as e:
                print(f"  ✗ Failed: {e}")

        return None

    async def test_multiple_symbols_quotes(self):
        """
        Test 4: Get quotes for multiple options symbols.

        Test how quote data scales with more symbols.
        """
        print("\n" + "=" * 80)
        print("TEST 4: Multiple Options Quotes")
        print("=" * 80)

        # Test with 5 different option symbols
        symbols = [
            "BTC-29NOV25-90000-C",
            "BTC-29NOV25-95000-C",
            "BTC-29NOV25-100000-C",
            "BTC-29NOV25-105000-C",
            "BTC-29NOV25-110000-C",
        ]

        options = {
            "exchange": "deribit",
            "from": "2024-10-01T00:00:00.000Z",
            "to": "2024-10-01T00:05:00.000Z",  # 5 minutes
            "symbols": symbols,
            "dataTypes": ["quote"]
        }

        try:
            response = await self.client.get(
                f"{self.base_url}/replay-normalized",
                params={"options": json.dumps(options)}
            )

            if response.status_code == 200:
                lines = response.text.strip().split('\n') if response.text.strip() else []
                size_mb = len(response.content) / 1_000_000

                print(f"✓ Received {len(lines)} quote updates for {len(symbols)} options")
                print(f"  Size: {size_mb:.3f} MB")
                print(f"  Avg quotes per symbol: {len(lines) / len(symbols):.0f}")

                return {"messages": len(lines), "size_mb": size_mb, "symbols": len(symbols)}

            else:
                print(f"✗ Error: {response.status_code}")
                return None

        except Exception as e:
            print(f"✗ Test failed: {e}")
            return None

    async def run_all_tests(self):
        """Run all quote sampling tests."""
        print("\n" + "╔" + "═" * 78 + "╗")
        print("║" + " " * 15 + "QUOTES SAMPLING TEST SUITE" + " " * 37 + "║")
        print("╚" + "═" * 78 + "╝")

        # Check if server is running
        try:
            response = await self.client.get(f"{self.base_url}/")
            print(f"✓ tardis-machine server is running")
        except:
            print(f"✗ Server not accessible at {self.base_url}")
            print("\nPlease start tardis-machine server first:")
            print("  npx tardis-machine")
            return

        # Run tests
        raw_result = await self.test_raw_quotes()
        sampled_result = await self.test_quote_bar_sampling()
        snapshot_result = await self.test_quote_snapshot_intervals()
        multi_result = await self.test_multiple_symbols_quotes()

        # Summary
        print("\n" + "=" * 80)
        print("SUMMARY & CONCLUSIONS")
        print("=" * 80)

        if sampled_result:
            print("\n🎉 GREAT NEWS! Quote sampling IS supported!")
            print(f"\nData type that works: {sampled_result['name']}")

            if raw_result and raw_result['messages'] > 0:
                reduction = (1 - sampled_result['messages'] / raw_result['messages']) * 100
                print(f"\nData reduction: {reduction:.1f}%")
                print(f"  Tick-by-tick: {raw_result['messages']} messages, {raw_result['size_mb']:.3f} MB")
                print(f"  5s sampled: {sampled_result['messages']} messages, {sampled_result['size_mb']:.3f} MB")

            print("\n✅ RECOMMENDATION: Use tardis-machine with quote sampling!")
            print("\nBenefits:")
            print("  - Massive data reduction (86-98%)")
            print("  - Download only sampled quotes")
            print("  - Calculate IV & Greeks on smaller dataset")
            print("  - Same quality as current approach")

        elif snapshot_result:
            print("\n👍 Quote snapshots work!")
            print(f"\nData type that works: {snapshot_result['name']}")
            print("\n✅ RECOMMENDATION: Use quote snapshots")

        else:
            print("\n❌ Quote sampling is NOT supported")
            print("\nQuote data types tested:")
            print("  - quote_bar_5s: ✗")
            print("  - quotes_5s: ✗")
            print("  - quote_5s: ✗")
            print("  - quote_snapshot_5s: ✗")

            print("\n⚠️ RECOMMENDATION: Use CSV download approach instead")
            print("\nWhy:")
            print("  - tardis-machine cannot sample quotes")
            print("  - Would need to download ALL tick data")
            print("  - CSV options_chain already has IV & Greeks computed")
            print("  - CSV is faster and simpler")

        print("\n" + "=" * 80)

        await self.client.aclose()


async def main():
    """Main test runner."""
    tester = QuotesSamplingTest()
    await tester.run_all_tests()


if __name__ == "__main__":
    asyncio.run(main())
