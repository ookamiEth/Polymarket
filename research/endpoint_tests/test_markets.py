#!/usr/bin/env python3
"""
Test script for Polymarket Markets endpoint (Gamma API)
Gets market metadata with volume and price metrics
"""

import requests
import json
from datetime import datetime

def test_markets():
    base_url = "https://gamma-api.polymarket.com/markets"

    print("="*80)
    print(" MARKETS ENDPOINT TEST")
    print("="*80 + "\n")

    # Test 1: Get active markets
    print("Test 1: Get active markets (limit 5)")
    print("-" * 80)

    params = {
        "limit": 5,
        "offset": 0,
        "closed": "false",
        "active": "true"
    }

    print(f"Request URL: {base_url}")
    print(f"Parameters: {json.dumps(params, indent=2)}\n")

    try:
        response = requests.get(base_url, params=params, timeout=10)
        response.raise_for_status()

        markets = response.json()

        print(f"Status Code: {response.status_code}")
        print(f"Markets returned: {len(markets)}\n")

        if markets:
            print("Market Summaries:")
            for i, market in enumerate(markets[:3], 1):
                print(f"\n  {i}. {market.get('question', 'N/A')[:60]}")
                print(f"     Volume 24hr: ${market.get('volume24hr', 0):,.2f}")
                print(f"     Volume 1wk: ${market.get('volume1wk', 0):,.2f}")
                print(f"     Liquidity: ${market.get('liquidityNum', 0):,.2f}")
                print(f"     Active: {market.get('active')}")
                print(f"     End Date: {market.get('endDate', 'N/A')[:10]}")

    except requests.exceptions.RequestException as e:
        print(f"Error: {e}")

    # Test 2: Market structure
    print("\n" + "-"*80)
    print("Test 2: Market Data Structure")
    print("-" * 80)

    params_2 = {"limit": 1}

    try:
        response = requests.get(base_url, params=params_2, timeout=10)
        response.raise_for_status()

        markets = response.json()

        if markets:
            market = markets[0]

            print("\nKey Volume Fields:")
            volume_fields = ['volume', 'volume24hr', 'volume1wk', 'volume1mo', 'volume1yr']
            for field in volume_fields:
                if field in market:
                    value = market[field]
                    # Handle both string and numeric values
                    if isinstance(value, str):
                        value = float(value) if value else 0
                    print(f"  {field:15}: ${value:,.2f}")

            print("\nVolume by System:")
            volume_amm = market.get('volumeAmm', 0)
            volume_clob = market.get('volumeClob', 0)
            if isinstance(volume_amm, str):
                volume_amm = float(volume_amm) if volume_amm else 0
            if isinstance(volume_clob, str):
                volume_clob = float(volume_clob) if volume_clob else 0
            print(f"  volumeAmm:  ${volume_amm:,.2f}")
            print(f"  volumeClob: ${volume_clob:,.2f}")

            print("\nLiquidity:")
            liq_fields = {'liquidityNum': 0, 'liquidityAmm': 0, 'liquidityClob': 0}
            for field, default in liq_fields.items():
                value = market.get(field, default)
                if isinstance(value, str):
                    value = float(value) if value else 0
                print(f"  {field}:  ${value:,.2f}")

            print("\nPrice Changes:")
            price_change_fields = ['oneHourPriceChange', 'oneDayPriceChange', 'oneWeekPriceChange', 'oneMonthPriceChange']
            for field in price_change_fields:
                value = market.get(field, 0)
                if isinstance(value, str):
                    value = float(value) if value else 0
                print(f"  {field}: {value:.4f}")

            print("\nCurrent Prices:")
            last_trade = market.get('lastTradePrice', 0)
            if isinstance(last_trade, str):
                last_trade = float(last_trade) if last_trade else 0
            print(f"  lastTradePrice: ${last_trade:.4f}")

            best_bid = market.get('bestBid', 0)
            best_ask = market.get('bestAsk', 0)
            spread = market.get('spread', 0)
            if isinstance(best_bid, str):
                best_bid = float(best_bid) if best_bid else 0
            if isinstance(best_ask, str):
                best_ask = float(best_ask) if best_ask else 0
            if isinstance(spread, str):
                spread = float(spread) if spread else 0
            print(f"  bestBid: ${best_bid:.4f}")
            print(f"  bestAsk: ${best_ask:.4f}")
            print(f"  spread: {spread:.4f}")

            print("\nTimestamps:")
            for field in ['createdAt', 'updatedAt', 'startDate', 'endDate']:
                if field in market:
                    print(f"  {field}: {market[field]}")

            print("\n\nMarket Structure (abbreviated):")
            abbreviated = {
                "id": market.get('id'),
                "question": market.get('question', '')[:50] + "...",
                "conditionId": market.get('conditionId', '')[:30] + "...",
                "slug": market.get('slug'),
                "active": market.get('active'),
                "volume24hr": market.get('volume24hr'),
                "liquidityNum": market.get('liquidityNum'),
                "lastTradePrice": market.get('lastTradePrice')
            }
            print(json.dumps(abbreviated, indent=2))

    except requests.exceptions.RequestException as e:
        print(f"Error: {e}")

    print("\n" + "="*80)
    print(" TEST COMPLETED")
    print("="*80)
    print("\nKey Findings:")
    print("  - Endpoint: https://gamma-api.polymarket.com/markets")
    print("  - Rich metadata with volume breakdowns")
    print("  - Volume: total, 24hr, 1wk, 1mo, 1yr")
    print("  - Volume split: AMM vs CLOB")
    print("  - Price changes: 1hr, 1day, 1wk, 1mo")
    print("  - Current prices: last, bid, ask, spread")
    print("  - Timestamps: created, updated, start, end")
    print("  - Filter by: active, closed, archived")
    print("  - Rate limit: 100 req / 10s")

if __name__ == "__main__":
    test_markets()
