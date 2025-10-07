#!/usr/bin/env python3
"""
Test script for Polymarket Price History endpoint
Gets historical price data for a specific token
"""

import requests
import json
from datetime import datetime, timedelta

def print_section(title):
    """Print a formatted section header"""
    print("\n" + "="*80)
    print(f" {title}")
    print("="*80 + "\n")

def test_price_history():
    """Test the /prices-history endpoint"""

    # Example token ID (Bitcoin over $100k market)
    token_id = "21742633143463906290569050155826241533067272736897614950488156847949938836455"

    # Base URL
    base_url = "https://clob.polymarket.com/prices-history"

    print_section("PRICE HISTORY ENDPOINT TEST")

    # Test 1: Get last 1 week of data with 1-hour resolution
    print("Test 1: Last 1 week, hourly resolution")
    print("-" * 80)

    params_1 = {
        "market": token_id,
        "interval": "1w",
        "fidelity": 60  # 60 minutes = 1 hour
    }

    print(f"Request URL: {base_url}")
    print(f"Parameters: {json.dumps(params_1, indent=2)}")

    try:
        response = requests.get(base_url, params=params_1, timeout=10)
        response.raise_for_status()

        data = response.json()

        print(f"\nStatus Code: {response.status_code}")
        print(f"Total data points: {len(data.get('history', []))}")

        if data.get('history'):
            print("\nFirst 5 data points:")
            for i, point in enumerate(data['history'][:5]):
                timestamp = datetime.fromtimestamp(point['t'])
                print(f"  [{i+1}] Time: {timestamp.strftime('%Y-%m-%d %H:%M:%S')} | Price: {point['p']:.4f}")

            print("\nLast 5 data points:")
            for i, point in enumerate(data['history'][-5:]):
                timestamp = datetime.fromtimestamp(point['t'])
                print(f"  [{i+1}] Time: {timestamp.strftime('%Y-%m-%d %H:%M:%S')} | Price: {point['p']:.4f}")

            # Calculate price statistics
            prices = [p['p'] for p in data['history']]
            print("\nPrice Statistics:")
            print(f"  Min Price: {min(prices):.4f}")
            print(f"  Max Price: {max(prices):.4f}")
            print(f"  Avg Price: {sum(prices)/len(prices):.4f}")
            print(f"  Price Change: {((prices[-1] - prices[0]) / prices[0] * 100):.2f}%")

    except requests.exceptions.RequestException as e:
        print(f"\nError making request: {e}")

    # Test 2: Custom time range with high fidelity
    print("\n" + "-"*80)
    print("Test 2: Custom time range (last 24 hours), 5-minute resolution")
    print("-" * 80)

    now = int(datetime.now().timestamp())
    day_ago = now - (24 * 60 * 60)

    params_2 = {
        "market": token_id,
        "startTs": day_ago,
        "endTs": now,
        "fidelity": 5  # 5 minutes
    }

    print(f"Request URL: {base_url}")
    print(f"Parameters: {json.dumps(params_2, indent=2)}")
    print(f"Time Range: {datetime.fromtimestamp(day_ago)} to {datetime.fromtimestamp(now)}")

    try:
        response = requests.get(base_url, params=params_2, timeout=10)
        response.raise_for_status()

        data = response.json()

        print(f"\nStatus Code: {response.status_code}")
        print(f"Total data points: {len(data.get('history', []))}")
        print(f"Expected data points (~288 for 24hrs at 5min): {24 * 60 / 5}")

        if data.get('history'):
            print("\nSample data (every 60th point):")
            for i in range(0, len(data['history']), 60):
                point = data['history'][i]
                timestamp = datetime.fromtimestamp(point['t'])
                print(f"  Time: {timestamp.strftime('%Y-%m-%d %H:%M:%S')} | Price: {point['p']:.4f}")

    except requests.exceptions.RequestException as e:
        print(f"\nError making request: {e}")

    # Test 3: Response structure analysis
    print("\n" + "-"*80)
    print("Test 3: Response Structure Analysis")
    print("-" * 80)

    params_3 = {
        "market": token_id,
        "interval": "1d",
        "fidelity": 60
    }

    try:
        response = requests.get(base_url, params=params_3, timeout=10)
        response.raise_for_status()

        data = response.json()

        print("\nResponse Structure:")
        print(f"  Root keys: {list(data.keys())}")

        if data.get('history') and len(data['history']) > 0:
            sample_point = data['history'][0]
            print(f"\n  History point structure:")
            for key, value in sample_point.items():
                print(f"    - {key}: {type(value).__name__} (example: {value})")

        print("\nFull JSON Sample (first 3 points):")
        print(json.dumps({"history": data.get('history', [])[:3]}, indent=2))

    except requests.exceptions.RequestException as e:
        print(f"\nError making request: {e}")

    print_section("TEST COMPLETED")
    print("\nKey Findings:")
    print("  1. Endpoint: https://clob.polymarket.com/prices-history")
    print("  2. Response format: {\"history\": [{\"t\": timestamp, \"p\": price}, ...]}")
    print("  3. Timestamps are Unix timestamps (seconds since epoch)")
    print("  4. Prices are decimals between 0 and 1")
    print("  5. Use 'interval' OR 'startTs/endTs' (mutually exclusive)")
    print("  6. Fidelity controls resolution in minutes")
    print("  7. Rate limit: 100 requests / 10 seconds")

if __name__ == "__main__":
    test_price_history()
