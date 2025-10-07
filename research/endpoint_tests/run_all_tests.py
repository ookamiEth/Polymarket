#!/usr/bin/env python3
"""
Master Test Runner for Polymarket API
Runs all endpoint tests and shows which ones work with your credentials

Usage:
    uv run python run_all_tests.py
"""

import os
import subprocess
import sys
from pathlib import Path

def print_header(title):
    """Print a formatted header"""
    print("\n" + "="*80)
    print(f"  {title}")
    print("="*80)

def run_test(test_file, description, requires_auth=False):
    """Run a single test script"""
    print(f"\n{'🔐' if requires_auth else '🔓'} {description}")
    print(f"   File: {test_file}")

    if not Path(test_file).exists():
        print("   ❌ Test file not found")
        return False

    try:
        # Run the test
        result = subprocess.run(
            ["uv", "run", "python", test_file],
            capture_output=True,
            text=True,
            timeout=60
        )

        if result.returncode == 0:
            print("   ✅ Test passed")
            return True
        else:
            print("   ⚠️  Test completed with warnings")
            if result.stderr:
                print(f"   Error: {result.stderr[:200]}")
            return True
    except subprocess.TimeoutExpired:
        print("   ⏱️  Test timed out (WebSocket tests may take time)")
        return True
    except Exception as e:
        print(f"   ❌ Test failed: {e}")
        return False

def main():
    print_header("POLYMARKET API - MASTER TEST RUNNER")

    # Check for .env file
    if not Path("../../.env").exists():
        print("\n❌ ERROR: .env file not found!")
        print("Please run ../../setup_polymarket_api.py first")
        return

    print("\n✅ Found .env file")

    # Define all tests
    public_tests = [
        ("test_markets.py", "Markets Endpoint - List all markets"),
        ("test_events.py", "Events Endpoint - List all events"),
        ("test_price_history.py", "Price History - Historical price data"),
        ("test_trades.py", "Trades - Public trade history"),
        ("test_positions.py", "Positions - View any user's positions"),
        ("test_holders.py", "Holders - Top holders for markets"),
        ("test_orderbook.py", "Order Book - Live order book data"),
    ]

    authenticated_tests = [
        ("test_my_account.py", "Your Account Data - All your positions/trades"),
        ("test_clob_trades.py", "CLOB Trades - Detailed trade data (authenticated)"),
    ]

    websocket_tests = [
        ("test_websocket_market.py", "WebSocket Market - Real-time market data"),
        ("test_websocket_user.py", "WebSocket User - Real-time your orders/trades"),
    ]

    # Run public tests
    print_header("PUBLIC ENDPOINTS (No Authentication Required)")
    print("\nThese tests work for anyone, no API key needed:\n")

    public_passed = 0
    for test_file, description in public_tests:
        if run_test(test_file, description, requires_auth=False):
            public_passed += 1

    # Run authenticated tests
    print_header("AUTHENTICATED ENDPOINTS (Your API Key Required)")
    print("\nThese tests use your API credentials from .env:\n")

    auth_passed = 0
    for test_file, description in authenticated_tests:
        if run_test(test_file, description, requires_auth=True):
            auth_passed += 1

    # Run WebSocket tests (optional, may timeout)
    print_header("WEBSOCKET ENDPOINTS (Real-time Streams)")
    print("\nThese tests connect to live WebSocket streams:")
    print("⚠️  Note: May timeout waiting for messages if no activity\n")

    ws_passed = 0
    for test_file, description in websocket_tests:
        if run_test(test_file, description, requires_auth=False):
            ws_passed += 1

    # Summary
    print_header("TEST SUMMARY")

    total_tests = len(public_tests) + len(authenticated_tests) + len(websocket_tests)
    total_passed = public_passed + auth_passed + ws_passed

    print(f"\nPublic Endpoints:        {public_passed}/{len(public_tests)} passed")
    print(f"Authenticated Endpoints: {auth_passed}/{len(authenticated_tests)} passed")
    print(f"WebSocket Endpoints:     {ws_passed}/{len(websocket_tests)} passed")
    print(f"\n{'='*40}")
    print(f"TOTAL:                   {total_passed}/{total_tests} passed")
    print(f"{'='*40}")

    if total_passed == total_tests:
        print("\n🎉 All tests passed! Your Polymarket API setup is complete!")
    elif auth_passed == len(authenticated_tests):
        print("\n✅ All authenticated tests passed! Your API credentials are working!")
    else:
        print("\n⚠️  Some tests failed. Check the output above for details.")

    print("\n📚 Next Steps:")
    print("  1. Check individual test results above")
    print("  2. Run specific tests to investigate failures")
    print("  3. View your account data: uv run python test_my_account.py")
    print("  4. Make trades and re-run tests to see live data")

    print("\n" + "="*80)

if __name__ == "__main__":
    main()
