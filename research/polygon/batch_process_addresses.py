#!/usr/bin/env python3
"""
Batch process all addresses from polymarket_tagged_addresses.json
and add base_address field for each proxy wallet.

Usage:
    export ETHERSCAN_API_KEY=YOUR_KEY
    uv run python batch_process_addresses.py
"""

import json
import os
import sys
from pathlib import Path

# Add the current directory to path to import our mapper
sys.path.insert(0, str(Path(__file__).parent))

from proxy_to_base_mapper_direct import map_proxy_to_base

# Paths
INPUT_FILE = "/Users/lgierhake/Documents/ETH/BT/research/historical_trading/polymarket_tagged_addresses.json"
OUTPUT_FILE = "/Users/lgierhake/Documents/ETH/BT/research/historical_trading/polymarket_tagged_addresses.json"

def main():
    # Check for API key
    if not os.environ.get('ETHERSCAN_API_KEY'):
        print("❌ Error: ETHERSCAN_API_KEY environment variable not set")
        print("Run: export ETHERSCAN_API_KEY=YOUR_KEY")
        sys.exit(1)

    print("="*70)
    print("BATCH PROCESSING POLYMARKET ADDRESSES")
    print("="*70)
    print()

    # Load the JSON file
    print(f"Loading addresses from: {INPUT_FILE}")
    with open(INPUT_FILE, 'r') as f:
        data = json.load(f)

    addresses = data.get('tagged_addresses', [])
    print(f"Found {len(addresses)} addresses to process\n")

    # Process each address
    results = []
    for i, addr_obj in enumerate(addresses, 1):
        proxy_wallet = addr_obj['address']
        name = addr_obj.get('name', '')

        print(f"[{i}/{len(addresses)}] Processing: {proxy_wallet}")
        if name:
            print(f"           Name: {name}")

        # Map proxy to base
        result = map_proxy_to_base(proxy_wallet)

        # Add base_address to the object
        if result['success']:
            addr_obj['base_address'] = result['base_address']
            addr_obj['mapping_method'] = result['method']
            print(f"           ✓ Base Address: {result['base_address']}")
            print(f"           Method: {result['method']}")
        else:
            addr_obj['base_address'] = None
            addr_obj['mapping_method'] = None
            addr_obj['mapping_error'] = result.get('error', 'Unknown error')
            print(f"           ✗ Failed: {result.get('error', 'Unknown error')}")

        print()
        results.append(addr_obj)

    # Update the data
    data['tagged_addresses'] = results

    # Save back to file
    print(f"Saving results to: {OUTPUT_FILE}")
    with open(OUTPUT_FILE, 'w') as f:
        json.dump(data, f, indent=2)

    # Summary
    successful = sum(1 for r in results if r.get('base_address'))
    failed = len(results) - successful

    print()
    print("="*70)
    print("SUMMARY")
    print("="*70)
    print(f"Total addresses: {len(results)}")
    print(f"✓ Successfully mapped: {successful}")
    print(f"✗ Failed: {failed}")
    print()
    print(f"Updated file: {OUTPUT_FILE}")
    print("="*70)

if __name__ == "__main__":
    main()
