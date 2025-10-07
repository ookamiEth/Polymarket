#!/usr/bin/env python3
"""
Test script for Polymarket CLOB Trades endpoint
Gets detailed trade information including maker/taker breakdown

NOTE: This endpoint requires L2 authentication headers for user-specific queries.
Now updated to use your API credentials from .env!
"""

import os
import requests
import json
from datetime import datetime
from dotenv import load_dotenv
from py_clob_client.client import ClobClient

def print_section(title):
    """Print a formatted section header"""
    print("\n" + "="*80)
    print(f" {title}")
    print("="*80 + "\n")

def test_clob_trades():
    """Test the /data/trades endpoint from CLOB API with your credentials"""

    print_section("CLOB TRADES ENDPOINT TEST (WITH AUTHENTICATION)")

    # Load credentials
    load_dotenv()

    private_key = os.getenv('PRIVATE_KEY')
    funder_address = os.getenv('FUNDER_ADDRESS')
    signature_type = int(os.getenv('SIGNATURE_TYPE', 1))
    chain_id = int(os.getenv('CHAIN_ID', 137))
    host = os.getenv('CLOB_HOST', 'https://clob.polymarket.com')

    if not private_key or not funder_address:
        print("❌ ERROR: Missing credentials in .env file!")
        print("Please run ../setup_polymarket_api.py first")
        return

    print("✅ Credentials loaded from .env")
    print(f"   Funder Address: {funder_address}")
    print(f"   Host: {host}")

    # Initialize CLOB client
    print("\n🔄 Initializing CLOB client with authentication...")
    try:
        client = ClobClient(
            host=host,
            key=private_key,
            chain_id=chain_id,
            signature_type=signature_type,
            funder=funder_address
        )
        print("✅ Client initialized successfully!")
    except Exception as e:
        print(f"❌ Failed to initialize client: {e}")
        return

    # Test 1: Get your recent trades
    print("\n" + "-"*80)
    print("Test 1: Your Recent Trades (Authenticated)")
    print("-" * 80)

    try:
        # Get your recent trades using the authenticated client
        print("📊 Fetching your recent trades...")

        # Using the client's get_trades method
        trades = client.get_trades()

        if trades:
            print(f"\n✅ Found {len(trades)} trade(s)")

            # Display first trade
            if len(trades) > 0:
                print("\nSample Trade:")
                sample_trade = trades[0]
                print(json.dumps(sample_trade, indent=2))

                print("\n\nTrade Details:")
                print(f"  ID: {sample_trade.get('id', 'N/A')}")
                print(f"  Side: {sample_trade.get('side', 'N/A')}")
                print(f"  Size: {sample_trade.get('size', 'N/A')}")
                print(f"  Price: ${sample_trade.get('price', 'N/A')}")
                print(f"  Status: {sample_trade.get('status', 'N/A')}")
                print(f"  Match Time: {datetime.fromtimestamp(int(sample_trade.get('match_time', 0)))}")
        else:
            print("ℹ️  No trades found (you may not have traded yet)")

    except Exception as e:
        print(f"⚠️  Error fetching trades: {e}")
        print("   (This is expected if you haven't placed any trades yet)")

    # Show expected response structure from documentation
    print("\n" + "-"*80)
    print("Test 2: Expected Response Structure (From Documentation)")
    print("-" * 80)

    sample_trade = {
        "id": "28c4d2eb-bbea-40e7-a9f0-b2fdb56b2c2e",
        "taker_order_id": "0x06bc63e346ed4ceddce9efd6b3af37c8f8f440c92fe7da6b2d0f9e4ccbc50c42",
        "market": "0xbd31dc8a20211944f6b70f31557f1001557b59905b7738480ca09bd4532f84af",
        "asset_id": "52114319501245915516055106046884209969926127482827954674443846427813813222426",
        "side": "BUY",
        "size": "10.5",
        "price": "0.57",
        "fee_rate_bps": "100",
        "status": "CONFIRMED",
        "match_time": "1672290701",
        "last_update": "1672290701",
        "outcome": "YES",
        "maker_address": "0x1234567890abcdef1234567890abcdef12345678",
        "owner": "9180014b-33c8-9240-a14b-bdca11c0a465",
        "transaction_hash": "0xff354cd7ca7539dfa9c28d90943ab5779a4eac34b9b37a757d7b32bdfb11790b",
        "bucket_index": 0,
        "type": "TAKER",
        "maker_orders": [
            {
                "order_id": "0xff354cd7ca7539dfa9c28d90943ab5779a4eac34b9b37a757d7b32bdfb11790b",
                "maker_address": "0x1234567890abcdef1234567890abcdef12345678",
                "owner": "9180014b-33c8-9240-a14b-bdca11c0a465",
                "matched_amount": "10.5",
                "fee_rate_bps": "50",
                "price": "0.57",
                "asset_id": "52114319501245915516055106046884209969926127482827954674443846427813813222426",
                "outcome": "YES",
                "side": "sell"
            }
        ]
    }

    print("\nSample Trade Structure:")
    print(json.dumps(sample_trade, indent=2))

    print("\n\nField Explanations:")
    field_descriptions = {
        "id": "Unique trade identifier (UUID)",
        "taker_order_id": "Hash of the taker order (market order) that initiated the trade",
        "market": "Market ID (condition ID)",
        "asset_id": "Token ID of the traded asset",
        "side": "Direction of taker order: BUY or SELL",
        "size": "Trade size (number of shares)",
        "price": "Execution price (0-1 range)",
        "fee_rate_bps": "Fee rate in basis points (100 bps = 1%)",
        "status": "Trade status: MATCHED, MINED, CONFIRMED, RETRYING, or FAILED",
        "match_time": "Unix timestamp when trade was matched",
        "last_update": "Unix timestamp of last status update",
        "outcome": "Human-readable outcome name",
        "maker_address": "Ethereum address of maker",
        "owner": "API key owner identifier",
        "transaction_hash": "On-chain transaction hash",
        "bucket_index": "Index for multi-transaction trades (for gas limit handling)",
        "type": "TAKER or MAKER (perspective of the query)",
        "maker_orders": "Array of maker orders filled by this trade"
    }

    for field, description in field_descriptions.items():
        print(f"  {field:20} - {description}")

    # Trade status explanations
    print("\n" + "-"*80)
    print("Test 3: Trade Status Values")
    print("-" * 80)

    statuses = {
        "MATCHED": {"terminal": False, "desc": "Trade matched, sent to executor"},
        "MINED": {"terminal": False, "desc": "Transaction mined, awaiting finality"},
        "CONFIRMED": {"terminal": True, "desc": "Strong finality achieved, successful"},
        "RETRYING": {"terminal": False, "desc": "Transaction failed, being retried"},
        "FAILED": {"terminal": True, "desc": "Trade failed permanently"}
    }

    print("\nStatus Progression:")
    for status, info in statuses.items():
        terminal_str = "TERMINAL" if info["terminal"] else "INTERMEDIATE"
        print(f"  {status:12} [{terminal_str:13}] - {info['desc']}")

    # Query parameters
    print("\n" + "-"*80)
    print("Test 4: Available Query Parameters")
    print("-" * 80)

    params_info = {
        "id": {"type": "string", "required": False, "desc": "Specific trade ID to fetch"},
        "taker": {"type": "string", "required": False, "desc": "Address to get trades where it's the taker"},
        "maker": {"type": "string", "required": False, "desc": "Address to get trades where it's the maker"},
        "market": {"type": "string", "required": False, "desc": "Market ID (condition ID)"},
        "before": {"type": "string", "required": False, "desc": "Unix timestamp - trades before this time"},
        "after": {"type": "string", "required": False, "desc": "Unix timestamp - trades after this time"}
    }

    print("\nQuery Parameters:")
    for param, info in params_info.items():
        req_str = "Required" if info["required"] else "Optional"
        print(f"  {param:10} ({info['type']:8}) [{req_str:8}] - {info['desc']}")

    print("\n\nExample Queries:")
    print("  1. Get all trades for a market:")
    print("     GET /data/trades?market=0xabc123...")
    print("\n  2. Get trades for a specific user as taker:")
    print("     GET /data/trades?taker=0x1234...")
    print("\n  3. Get trades in a time range:")
    print("     GET /data/trades?market=0xabc123&after=1672531200&before=1675209600")
    print("\n  4. Get a specific trade:")
    print("     GET /data/trades?id=28c4d2eb-bbea-40e7-a9f0-b2fdb56b2c2e")

    print_section("TEST COMPLETED")
    print("\nKey Findings:")
    print("  1. Endpoint: https://clob.polymarket.com/data/trades")
    print("  2. Authentication: L2 headers required (POLY_ADDRESS, POLY_SIGNATURE, etc.)")
    print("  3. Response: Array of detailed trade objects")
    print("  4. Includes maker/taker breakdown with individual order details")
    print("  5. Trade status tracking from MATCHED to CONFIRMED/FAILED")
    print("  6. Transaction hashes for on-chain verification")
    print("  7. Bucket index for multi-transaction trades")
    print("  8. Fee rates in basis points (100 bps = 1%)")
    print("  9. Rate limit: 150 requests / 10 seconds")
    print(" 10. More detailed than Data API /trades endpoint")

if __name__ == "__main__":
    test_clob_trades()
