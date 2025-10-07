#!/usr/bin/env python3
"""
Test script for Polymarket Order Book endpoint
Gets current order book with bid/ask levels
"""

import requests
import json

def test_orderbook():
    base_url = "https://clob.polymarket.com/book"
    token_id = "21742633143463906290569050155826241533067272736897614950488156847949938836455"

    print("="*80)
    print(" ORDER BOOK ENDPOINT TEST")
    print("="*80 + "\n")

    params = {"token_id": token_id}

    print(f"Request URL: {base_url}")
    print(f"Parameters: {json.dumps(params, indent=2)}\n")

    try:
        response = requests.get(base_url, params=params, timeout=10)
        response.raise_for_status()

        book = response.json()

        print(f"Status Code: {response.status_code}")
        print(f"Market: {book.get('market', 'N/A')[:30]}...")
        print(f"Asset ID: {book.get('asset_id', 'N/A')[:30]}...")
        print(f"Timestamp: {book.get('timestamp')}")
        print(f"Hash: {book.get('hash', 'N/A')[:30]}...\n")

        bids = book.get('bids', [])
        asks = book.get('asks', [])

        print(f"Order Book Depth:")
        print(f"  Bids: {len(bids)} levels")
        print(f"  Asks: {len(asks)} levels\n")

        print("Top 5 Bids (Buy Orders):")
        for i, bid in enumerate(bids[:5], 1):
            print(f"  {i}. Price: ${bid.get('price')}  Size: {bid.get('size')}")

        print("\nTop 5 Asks (Sell Orders):")
        for i, ask in enumerate(asks[:5], 1):
            print(f"  {i}. Price: ${ask.get('price')}  Size: {ask.get('size')}")

        if bids and asks:
            spread = float(asks[0]['price']) - float(bids[0]['price'])
            mid_price = (float(asks[0]['price']) + float(bids[0]['price'])) / 2

            print(f"\nMarket Stats:")
            print(f"  Best Bid: ${bids[0]['price']}")
            print(f"  Best Ask: ${asks[0]['price']}")
            print(f"  Spread: ${spread:.4f}")
            print(f"  Mid Price: ${mid_price:.4f}")

            total_bid_liquidity = sum(float(b['size']) for b in bids)
            total_ask_liquidity = sum(float(a['size']) for a in asks)
            print(f"  Total Bid Liquidity: {total_bid_liquidity:.2f}")
            print(f"  Total Ask Liquidity: {total_ask_liquidity:.2f}")

        print("\n\nOrder Book Structure:")
        print(json.dumps({
            "market": book.get('market'),
            "asset_id": book.get('asset_id', '')[:30] + "...",
            "timestamp": book.get('timestamp'),
            "bids": bids[:2],
            "asks": asks[:2]
        }, indent=2))

        print("\n\nKey Findings:")
        print("  - Endpoint: https://clob.polymarket.com/book")
        print("  - Returns current order book snapshot")
        print("  - Bids and asks arrays with price/size")
        print("  - Timestamp for synchronization")
        print("  - Hash for integrity verification")
        print("  - Rate limit: 200 req / 10s")

    except requests.exceptions.RequestException as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    test_orderbook()
