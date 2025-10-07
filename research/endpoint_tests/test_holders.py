#!/usr/bin/env python3
"""
Test script for Polymarket Top Holders endpoint
Gets the largest position holders for specific markets
"""

import requests
import json

def test_holders():
    base_url = "https://data-api.polymarket.com/holders"
    market_id = "0xdd31ce870876bb410d71f29e3dfea77910bc3effb81a0fb8dc8c282d5968aa5a"

    print("="*80)
    print(" TOP HOLDERS ENDPOINT TEST")
    print("="*80 + "\n")

    params = {
        "market": market_id,
        "limit": 10,
        "minBalance": 100
    }

    print(f"Request URL: {base_url}")
    print(f"Parameters: {json.dumps(params, indent=2)}\n")

    try:
        response = requests.get(base_url, params=params, timeout=10)
        response.raise_for_status()

        data = response.json()

        print(f"Status Code: {response.status_code}")
        print(f"Markets returned: {len(data)}\n")

        if data:
            for market_data in data:
                token_id = market_data.get('token', 'Unknown')[:20] + "..."
                holders = market_data.get('holders', [])

                print(f"Token: {token_id}")
                print(f"Top {len(holders)} holders:\n")

                for i, holder in enumerate(holders[:5], 1):
                    print(f"  {i}. {holder.get('pseudonym', 'Anonymous')}")
                    print(f"     Address: {holder.get('proxyWallet', 'N/A')[:15]}...")
                    print(f"     Amount: {holder.get('amount', 0):.2f} shares")
                    print(f"     Outcome: {holder.get('outcomeIndex')}")

            print("\nStructure:")
            print(json.dumps(data[0], indent=2)[:1000])

        print("\n\nKey Findings:")
        print("  - Endpoint: https://data-api.polymarket.com/holders")
        print("  - Returns top holders by position size")
        print("  - Includes user profiles, pseudonyms, balances")
        print("  - Rate limit: 100 req / 10s")

    except requests.exceptions.RequestException as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    test_holders()
