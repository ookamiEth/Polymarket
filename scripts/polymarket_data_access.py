#!/usr/bin/env python3
"""
Polymarket Data Access Demo

This script demonstrates how to access various Polymarket data using the API.

Usage:
    uv run python polymarket_data_access.py
"""

import os
import json
from dotenv import load_dotenv
from py_clob_client.client import ClobClient
from py_clob_client.clob_types import OrderArgs, OrderType
import requests

def print_section(title):
    """Print a formatted section header"""
    print("\n" + "=" * 60)
    print(f"  {title}")
    print("=" * 60)

def main():
    print("=" * 60)
    print("Polymarket Data Access Demo")
    print("=" * 60)

    # Load environment variables
    load_dotenv()

    # Get credentials from .env
    private_key = os.getenv('PRIVATE_KEY')
    funder_address = os.getenv('FUNDER_ADDRESS')
    signature_type = int(os.getenv('SIGNATURE_TYPE', 1))
    chain_id = int(os.getenv('CHAIN_ID', 137))
    host = os.getenv('CLOB_HOST', 'https://clob.polymarket.com')

    # Validate required credentials
    if not private_key or not funder_address:
        print("\n❌ Error: Missing credentials in .env file")
        print("Run setup_polymarket_api.py first!")
        return

    try:
        # Initialize CLOB client
        print("\n🔄 Initializing CLOB client...")
        client = ClobClient(
            host=host,
            key=private_key,
            chain_id=chain_id,
            signature_type=signature_type,
            funder=funder_address
        )
        print("✅ Client initialized")

        # 1. Get Markets Data (Public API - No auth required)
        print_section("1. Fetching Active Markets")
        markets_response = requests.get('https://gamma-api.polymarket.com/markets')
        if markets_response.status_code == 200:
            markets = markets_response.json()
            print(f"✅ Found {len(markets)} active markets")
            if markets:
                print("\nSample Market:")
                sample = markets[0]
                print(f"   Question: {sample.get('question', 'N/A')}")
                print(f"   Market ID: {sample.get('condition_id', 'N/A')}")
                print(f"   Active: {sample.get('active', 'N/A')}")
                print(f"   End Date: {sample.get('end_date_iso', 'N/A')}")
        else:
            print(f"❌ Failed to fetch markets: {markets_response.status_code}")

        # 2. Get Your Positions
        print_section("2. Your Current Positions")
        positions_url = f'https://data-api.polymarket.com/positions?user={funder_address}'
        positions_response = requests.get(positions_url)
        if positions_response.status_code == 200:
            positions = positions_response.json()
            print(f"✅ You have {len(positions)} position(s)")
            if positions:
                for i, pos in enumerate(positions[:5], 1):  # Show first 5
                    print(f"\n   Position {i}:")
                    print(f"      Market: {pos.get('title', 'N/A')}")
                    print(f"      Outcome: {pos.get('outcome', 'N/A')}")
                    print(f"      Size: {pos.get('size', 'N/A')}")
                    print(f"      Avg Price: ${pos.get('avgPrice', 0):.2f}")
                    print(f"      Current Value: ${pos.get('currentValue', 0):.2f}")
                    print(f"      PnL: ${pos.get('cashPnl', 0):.2f}")
            else:
                print("   No positions found")
        else:
            print(f"❌ Failed to fetch positions: {positions_response.status_code}")

        # 3. Get Your Trade History
        print_section("3. Your Recent Trades")
        trades_url = f'https://data-api.polymarket.com/trades?user={funder_address}&limit=10'
        trades_response = requests.get(trades_url)
        if trades_response.status_code == 200:
            trades = trades_response.json()
            print(f"✅ Found {len(trades)} recent trade(s)")
            if trades:
                for i, trade in enumerate(trades[:5], 1):  # Show first 5
                    print(f"\n   Trade {i}:")
                    print(f"      Market: {trade.get('title', 'N/A')}")
                    print(f"      Side: {trade.get('side', 'N/A')}")
                    print(f"      Outcome: {trade.get('outcome', 'N/A')}")
                    print(f"      Size: {trade.get('size', 'N/A')}")
                    print(f"      Price: ${trade.get('price', 0):.2f}")
            else:
                print("   No trades found")
        else:
            print(f"❌ Failed to fetch trades: {trades_response.status_code}")

        # 4. Get Account Value
        print_section("4. Your Account Value")
        value_url = f'https://data-api.polymarket.com/value?user={funder_address}'
        value_response = requests.get(value_url)
        if value_response.status_code == 200:
            value_data = value_response.json()
            if value_data:
                total_value = value_data[0].get('value', 0)
                print(f"✅ Total Position Value: ${total_value:.2f}")
            else:
                print("   No value data available")
        else:
            print(f"❌ Failed to fetch account value: {value_response.status_code}")

        # 5. Get a Sample Orderbook
        print_section("5. Sample Orderbook")
        # Get first market from the markets list
        if markets_response.status_code == 200 and markets:
            token_id = markets[0].get('tokens', [{}])[0].get('token_id', None)
            if token_id:
                orderbook = client.get_order_book(token_id)
                print(f"✅ Orderbook for: {markets[0].get('question', 'N/A')}")
                print(f"\n   Bids (Buy Orders): {len(orderbook.get('bids', []))}")
                if orderbook.get('bids'):
                    print(f"      Best Bid: ${orderbook['bids'][0]['price']} (Size: {orderbook['bids'][0]['size']})")
                print(f"\n   Asks (Sell Orders): {len(orderbook.get('asks', []))}")
                if orderbook.get('asks'):
                    print(f"      Best Ask: ${orderbook['asks'][0]['price']} (Size: {orderbook['asks'][0]['size']})")
            else:
                print("   ⚠️  No token ID available for orderbook")

        # 6. Get Your Active Orders
        print_section("6. Your Active Orders")
        try:
            # This requires API key authentication
            active_orders = client.get_orders()
            print(f"✅ You have {len(active_orders)} active order(s)")
            if active_orders:
                for i, order in enumerate(active_orders[:5], 1):
                    print(f"\n   Order {i}:")
                    print(f"      Market: {order.get('market', 'N/A')}")
                    print(f"      Side: {order.get('side', 'N/A')}")
                    print(f"      Price: ${order.get('price', 'N/A')}")
                    print(f"      Size: {order.get('size', 'N/A')}")
                    print(f"      Status: {order.get('status', 'N/A')}")
            else:
                print("   No active orders")
        except Exception as e:
            print(f"⚠️  Could not fetch active orders: {str(e)}")
            print("   (This might require API key setup - run setup_polymarket_api.py)")

        # 7. Market Price Information
        print_section("7. Market Prices")
        prices_response = requests.get('https://gamma-api.polymarket.com/prices')
        if prices_response.status_code == 200:
            prices = prices_response.json()
            print(f"✅ Found price data for markets")
            if markets and len(markets) > 0:
                sample_market = markets[0]
                condition_id = sample_market.get('condition_id')
                if condition_id and condition_id in prices:
                    print(f"\nSample Price for: {sample_market.get('question', 'N/A')}")
                    print(f"   Price: ${prices[condition_id]:.4f}")
        else:
            print(f"❌ Failed to fetch prices: {prices_response.status_code}")

        print("\n" + "=" * 60)
        print("Data Access Demo Complete!")
        print("=" * 60)
        print("\n✨ You now have full access to Polymarket data!")
        print("\nAvailable APIs:")
        print("  - Gamma API: https://gamma-api.polymarket.com (Markets, Events, Prices)")
        print("  - Data API: https://data-api.polymarket.com (Positions, Trades, Value)")
        print("  - CLOB API: https://clob.polymarket.com (Trading, Orders)")
        print("\n📖 Documentation: https://docs.polymarket.com")

    except Exception as e:
        print(f"\n❌ Error: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
