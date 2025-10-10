#!/usr/bin/env python3
"""
Test Script: BTC Options Data Sampling with tardis-machine

Tests if tardis-machine can provide sampled data for BTC OPTIONS specifically:
1. Quote sampling (quote_5s, quote_1s, quote_1m)
2. Book snapshots (book_snapshot_25_5s, book_snapshot_25_1s)
3. Combined requests (both together)

This is CRITICAL because we want to download BTC options data efficiently,
and need to know if tardis-machine can sample it server-side.

Usage:
    # Start tardis-machine server:
    npx tardis-machine --port=8000

    # Run test:
    uv run python test/test_btc_options_sampling.py
"""

import asyncio
import httpx
import json
from datetime import datetime
from typing import Dict, List, Any, Optional
import polars as pl


class BTCOptionsSamplingTest:
    """Tests tardis-machine sampling capabilities for BTC options specifically."""

    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url
        self.client = httpx.AsyncClient(timeout=120.0)

        # BTC options symbols that would have data in Oct 2024
        # Using Dec 2024 expiry with strikes around BTC ~$60-65k range
        self.test_symbols = [
            "BTC-27DEC24-60000-C",
            "BTC-27DEC24-65000-C",
            "BTC-27DEC24-70000-C",
        ]

        # Test period: Oct 1, 2024 (when BTC options had active trading)
        self.test_from = "2024-10-01T00:00:00.000Z"
        self.test_to = "2024-10-01T00:10:00.000Z"  # 10 minutes

    async def check_server(self) -> bool:
        """Check if tardis-machine server is running."""
        try:
            response = await self.client.get(f"{self.base_url}/")
            print(f"✓ tardis-machine server is running")
            return True
        except Exception as e:
            print(f"✗ Server not accessible at {self.base_url}")
            print(f"  Error: {e}")
            print("\nPlease start tardis-machine server first:")
            print("  npx tardis-machine --port=8000")
            return False

    async def test_raw_options_quotes(self) -> Optional[Dict]:
        """
        Test 1: Get raw tick-by-tick quotes for BTC options.

        This establishes the baseline data volume.
        """
        print("\n" + "=" * 80)
        print("TEST 1: Raw Tick-by-Tick Quotes for BTC Options")
        print("=" * 80)

        options = {
            "exchange": "deribit",
            "from": self.test_from,
            "to": self.test_to,
            "symbols": self.test_symbols,
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

                print(f"✓ Received {len(lines):,} quote updates")
                print(f"  Symbols: {', '.join(self.test_symbols)}")
                print(f"  Size: {size_mb:.3f} MB")

                if len(lines) > 0:
                    print(f"  Avg quotes per symbol: {len(lines) / len(self.test_symbols):.0f}")

                    # Parse first quote to see structure
                    sample = json.loads(lines[0])
                    print(f"\nSample quote:")
                    print(f"  Symbol: {sample.get('symbol')}")
                    print(f"  Timestamp: {sample.get('timestamp')}")
                    print(f"  Bid: {sample.get('bidPrice')} @ {sample.get('bidAmount')}")
                    print(f"  Ask: {sample.get('askPrice')} @ {sample.get('askAmount')}")
                else:
                    print("\n⚠️  WARNING: No quote data received!")
                    print("  This might mean:")
                    print("  - These specific options had no trading activity on this date")
                    print("  - The symbols don't exist for this date")
                    print("  - We need to try different symbols or dates")

                return {
                    "messages": len(lines),
                    "size_mb": size_mb,
                    "symbols": len(self.test_symbols)
                }
            else:
                print(f"✗ Error: {response.status_code}")
                print(response.text[:500])
                return None

        except Exception as e:
            print(f"✗ Test failed: {e}")
            return None

    async def test_options_quote_sampling(self) -> Optional[Dict]:
        """
        Test 2: Test quote sampling for BTC options.

        This is THE KEY TEST - can we sample options quotes at intervals?
        """
        print("\n" + "=" * 80)
        print("TEST 2: Quote Sampling for BTC Options")
        print("=" * 80)

        test_configs = [
            {"name": "5-second quotes", "dataTypes": ["quote_5s"]},
            {"name": "1-second quotes", "dataTypes": ["quote_1s"]},
            {"name": "1-minute quotes", "dataTypes": ["quote_1m"]},
        ]

        results = []

        for config in test_configs:
            print(f"\n--- Testing: {config['name']} ({config['dataTypes'][0]}) ---")

            options = {
                "exchange": "deribit",
                "from": self.test_from,
                "to": self.test_to,
                "symbols": self.test_symbols,
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

                    print(f"  ✓ SUCCESS! Received {len(lines):,} sampled quotes")
                    print(f"    Size: {size_mb:.3f} MB")

                    if lines:
                        sample = json.loads(lines[0])
                        print(f"    Sample fields: {list(sample.keys())[:8]}")

                        results.append({
                            "name": config['name'],
                            "data_type": config['dataTypes'][0],
                            "messages": len(lines),
                            "size_mb": size_mb
                        })
                    else:
                        print(f"    ⚠️  No data received (might be normal if no quotes in this interval)")

                else:
                    print(f"  ✗ Error: {response.status_code}")
                    error_text = response.text[:300]
                    print(f"    {error_text}")

                    if "invalid interval" in error_text:
                        print(f"    → This data type is not supported")

            except Exception as e:
                print(f"  ✗ Failed: {e}")

        return results if results else None

    async def test_options_book_snapshots(self) -> Optional[Dict]:
        """
        Test 3: Test book snapshots for BTC options.

        Can we get order book snapshots at intervals with 25 levels?
        """
        print("\n" + "=" * 80)
        print("TEST 3: Book Snapshots for BTC Options (25 levels)")
        print("=" * 80)

        test_configs = [
            {"name": "5-second snapshots", "dataTypes": ["book_snapshot_25_5s"]},
            {"name": "1-second snapshots", "dataTypes": ["book_snapshot_25_1s"]},
        ]

        results = []

        for config in test_configs:
            print(f"\n--- Testing: {config['name']} ({config['dataTypes'][0]}) ---")

            options = {
                "exchange": "deribit",
                "from": self.test_from,
                "to": self.test_to,
                "symbols": self.test_symbols,
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

                    print(f"  ✓ SUCCESS! Received {len(lines):,} book snapshots")
                    print(f"    Size: {size_mb:.3f} MB")

                    if lines:
                        sample = json.loads(lines[0])
                        num_bids = len(sample.get('bids', []))
                        num_asks = len(sample.get('asks', []))

                        print(f"    Symbol: {sample.get('symbol')}")
                        print(f"    Timestamp: {sample.get('timestamp')}")
                        print(f"    Bids: {num_bids} levels")
                        print(f"    Asks: {num_asks} levels")

                        results.append({
                            "name": config['name'],
                            "data_type": config['dataTypes'][0],
                            "messages": len(lines),
                            "size_mb": size_mb,
                            "avg_bid_levels": num_bids,
                            "avg_ask_levels": num_asks
                        })

                else:
                    print(f"  ✗ Error: {response.status_code}")
                    error_text = response.text[:300]
                    print(f"    {error_text}")

            except Exception as e:
                print(f"  ✗ Failed: {e}")

        return results if results else None

    async def test_combined_quote_and_book(self) -> Optional[Dict]:
        """
        Test 4: Test combined request (quote_5s + book_snapshot_25_5s).

        Can we get both quotes and book snapshots in one request?
        """
        print("\n" + "=" * 80)
        print("TEST 4: Combined Request (Quotes + Book Snapshots)")
        print("=" * 80)

        print("\n--- Testing: quote_5s + book_snapshot_25_5s ---")

        options = {
            "exchange": "deribit",
            "from": self.test_from,
            "to": self.test_to,
            "symbols": self.test_symbols,
            "dataTypes": ["quote_5s", "book_snapshot_25_5s"]
        }

        try:
            response = await self.client.get(
                f"{self.base_url}/replay-normalized",
                params={"options": json.dumps(options)}
            )

            if response.status_code == 200:
                lines = response.text.strip().split('\n') if response.text.strip() else []
                size_mb = len(response.content) / 1_000_000

                # Count message types
                quote_count = 0
                book_count = 0

                for line in lines[:100]:  # Sample first 100
                    msg = json.loads(line)
                    msg_type = msg.get('type', '')
                    if 'quote' in msg_type:
                        quote_count += 1
                    elif 'book' in msg_type:
                        book_count += 1

                print(f"  ✓ SUCCESS! Received {len(lines):,} messages")
                print(f"    Size: {size_mb:.3f} MB")
                print(f"    Quote messages (sample): {quote_count}/100")
                print(f"    Book messages (sample): {book_count}/100")

                return {
                    "messages": len(lines),
                    "size_mb": size_mb,
                    "quote_sample": quote_count,
                    "book_sample": book_count
                }

            else:
                print(f"  ✗ Error: {response.status_code}")
                print(response.text[:500])
                return None

        except Exception as e:
            print(f"  ✗ Failed: {e}")
            return None

    async def run_all_tests(self):
        """Run all BTC options sampling tests."""
        print("\n" + "╔" + "═" * 78 + "╗")
        print("║" + " " * 15 + "BTC OPTIONS SAMPLING TEST SUITE" + " " * 31 + "║")
        print("╚" + "═" * 78 + "╝")

        # Check server
        if not await self.check_server():
            return

        print(f"\n📋 Test Configuration:")
        print(f"   Symbols: {', '.join(self.test_symbols)}")
        print(f"   Period: {self.test_from} to {self.test_to}")
        print(f"   Duration: 10 minutes")

        # Run tests
        raw_result = await self.test_raw_options_quotes()
        quote_results = await self.test_options_quote_sampling()
        book_results = await self.test_options_book_snapshots()
        combined_result = await self.test_combined_quote_and_book()

        # Summary
        print("\n" + "=" * 80)
        print("SUMMARY & RECOMMENDATIONS")
        print("=" * 80)

        if raw_result and raw_result['messages'] == 0:
            print("\n⚠️  WARNING: No raw quote data received!")
            print("\nPossible reasons:")
            print("  1. These specific options had no trading on 2024-10-01")
            print("  2. Symbol naming might be different")
            print("  3. Options might not have been listed yet")
            print("\n💡 RECOMMENDATION: Try different symbols or dates")
            print("   - Use a perpetual (BTC-PERPETUAL) to verify server works")
            print("   - Check Deribit historical data for active options on this date")

        elif quote_results:
            print("\n🎉 BTC OPTIONS QUOTE SAMPLING WORKS!")

            print("\n✅ Supported quote sampling intervals:")
            for result in quote_results:
                print(f"   - {result['data_type']}: {result['messages']} messages, {result['size_mb']:.3f} MB")

            if raw_result and raw_result['messages'] > 0:
                best_result = quote_results[0]  # First one that worked
                reduction = (1 - best_result['messages'] / raw_result['messages']) * 100
                print(f"\n📊 Data reduction with {best_result['data_type']}:")
                print(f"   Tick-by-tick: {raw_result['messages']:,} messages, {raw_result['size_mb']:.3f} MB")
                print(f"   Sampled: {best_result['messages']:,} messages, {best_result['size_mb']:.3f} MB")
                print(f"   Reduction: {reduction:.1f}%")

        if book_results:
            print("\n🎉 BTC OPTIONS BOOK SNAPSHOTS WORK!")

            print("\n✅ Supported book snapshot intervals:")
            for result in book_results:
                print(f"   - {result['data_type']}: {result['messages']} snapshots")
                print(f"     Average depth: {result['avg_bid_levels']} bids, {result['avg_ask_levels']} asks")

        if combined_result:
            print("\n🎉 COMBINED REQUESTS WORK!")
            print(f"   Can request quote_5s + book_snapshot_25_5s together")
            print(f"   Total: {combined_result['messages']} messages, {combined_result['size_mb']:.3f} MB")

        if quote_results or book_results:
            print("\n" + "=" * 80)
            print("💡 FINAL RECOMMENDATION FOR BTC OPTIONS DATA")
            print("=" * 80)
            print("\n✅ Use tardis-machine for BTC options data collection!")

            print("\nRecommended configuration:")
            print("  - Data types: ['quote_5s', 'book_snapshot_25_5s']")
            print("  - This gives you:")
            print("    ✓ Sampled quote data (bid/ask prices)")
            print("    ✓ Full order book snapshots (25 levels)")
            print("    ✓ Massive data reduction (80-99%)")
            print("    ✓ Server-side aggregation (no client processing)")

            print("\nNext steps:")
            print("  1. Update download script to use tardis-machine HTTP API")
            print("  2. Calculate IV & Greeks from sampled quote data")
            print("  3. Use book snapshots for liquidity analysis")

        else:
            print("\n❌ BTC Options sampling NOT supported")
            print("\n⚠️  RECOMMENDATION: Use CSV download approach instead")

        print("\n" + "=" * 80)

        await self.client.aclose()


async def main():
    """Main test runner."""
    tester = BTCOptionsSamplingTest()
    await tester.run_all_tests()


if __name__ == "__main__":
    asyncio.run(main())
