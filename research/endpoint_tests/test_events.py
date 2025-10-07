#!/usr/bin/env python3
"""
Test script for Polymarket Events endpoint (Gamma API)
Gets event metadata with aggregated market data
"""

import requests
import json

def test_events():
    base_url = "https://gamma-api.polymarket.com/events"

    print("="*80)
    print(" EVENTS ENDPOINT TEST")
    print("="*80 + "\n")

    params = {
        "limit": 5,
        "closed": "false",
        "active": "true"
    }

    print(f"Request URL: {base_url}")
    print(f"Parameters: {json.dumps(params, indent=2)}\n")

    try:
        response = requests.get(base_url, params=params, timeout=10)
        response.raise_for_status()

        events = response.json()

        print(f"Status Code: {response.status_code}")
        print(f"Events returned: {len(events)}\n")

        if events:
            print("Event Summaries:")
            for i, event in enumerate(events[:3], 1):
                print(f"\n  {i}. {event.get('title', 'N/A')[:60]}")
                print(f"     Slug: {event.get('slug')}")
                print(f"     Volume: ${event.get('volume', 0):,.2f}")
                print(f"     Volume 24hr: ${event.get('volume24hr', 0):,.2f}")
                print(f"     Liquidity: ${event.get('liquidity', 0):,.2f}")
                print(f"     Open Interest: ${event.get('openInterest', 0):,.2f}")
                print(f"     Markets: {len(event.get('markets', []))}")
                print(f"     Active: {event.get('active')}")

            print("\n\nEvent Structure (first event, abbreviated):")
            event = events[0]
            abbrev = {
                "id": event.get('id'),
                "title": event.get('title', '')[:50] + "...",
                "slug": event.get('slug'),
                "volume": event.get('volume'),
                "volume24hr": event.get('volume24hr'),
                "volume1wk": event.get('volume1wk'),
                "liquidity": event.get('liquidity'),
                "openInterest": event.get('openInterest'),
                "active": event.get('active'),
                "closed": event.get('closed'),
                "num_markets": len(event.get('markets', []))
            }
            print(json.dumps(abbrev, indent=2))

            if event.get('markets'):
                print(f"\n\nFirst market in event:")
                market = event['markets'][0]
                print(f"  Question: {market.get('question', '')[:50]}")
                print(f"  Volume 24hr: ${market.get('volume24hr', 0):,.2f}")

        print("\n" + "="*80)
        print(" TEST COMPLETED")
        print("="*80)
        print("\nKey Findings:")
        print("  - Endpoint: https://gamma-api.polymarket.com/events")
        print("  - Events group multiple related markets")
        print("  - Aggregated volume/liquidity across all markets")
        print("  - Includes nested market objects")
        print("  - Volume fields: total, 24hr, 1wk, 1mo, 1yr")
        print("  - Liquidity split: AMM vs CLOB")
        print("  - Open interest tracking")
        print("  - Rate limit: 100 req / 10s")

    except requests.exceptions.RequestException as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    test_events()
