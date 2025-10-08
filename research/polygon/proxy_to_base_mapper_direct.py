#!/usr/bin/env python3
"""
Polymarket Proxy Wallet → Base Address Mapper (Direct Blockchain Method)

This script maps Polymarket proxy wallet addresses to their owner's base addresses
using ONLY direct blockchain contract calls.

Polymarket uses two types of proxy wallets:
1. Gnosis Safe wallets (MetaMask users) - call getOwners()
2. Polymarket Proxy wallets (MagicLink users) - call owner()

Usage:
    python proxy_to_base_mapper_direct.py <proxy_wallet_address>

Example:
    python proxy_to_base_mapper_direct.py 0x1234...

Author: BT Research Team
Date: 2025-10-07
"""

import sys
import os
import requests
from typing import Optional
from web3 import Web3

# ============================================================================
# CONFIGURATION
# ============================================================================

# Polygon RPC endpoints (multiple for redundancy)
POLYGON_RPC_URLS = [
    "https://polygon-rpc.com",
    "https://rpc-mainnet.matic.network",
    "https://polygon-mainnet.public.blastapi.io"
]

# Etherscan V2 API (single key works for all chains)
ETHERSCAN_API_URL = "https://api.etherscan.io/v2/api"
POLYGON_CHAIN_ID = 137

# Gnosis Safe ABI - for MetaMask users
GNOSIS_SAFE_ABI = [
    {
        "constant": True,
        "inputs": [],
        "name": "getOwners",
        "outputs": [{"name": "", "type": "address[]"}],
        "payable": False,
        "stateMutability": "view",
        "type": "function"
    }
]

# Polymarket Proxy ABI - for MagicLink users
POLYMARKET_PROXY_ABI = [
    {
        "constant": True,
        "inputs": [],
        "name": "owner",
        "outputs": [{"name": "", "type": "address"}],
        "payable": False,
        "stateMutability": "view",
        "type": "function"
    }
]


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def get_web3_connection() -> Optional[Web3]:
    """
    Try connecting to Polygon via multiple RPC endpoints.

    Returns:
        Connected Web3 instance or None if all connections fail
    """
    for rpc_url in POLYGON_RPC_URLS:
        try:
            w3 = Web3(Web3.HTTPProvider(rpc_url, request_kwargs={'timeout': 10}))
            if w3.is_connected():
                return w3
        except Exception:
            continue
    return None


def is_contract(w3: Web3, address: str) -> bool:
    """
    Check if an address is a smart contract.

    Args:
        w3: Web3 instance
        address: Address to check

    Returns:
        True if address is a contract, False otherwise
    """
    try:
        code = w3.eth.get_code(Web3.to_checksum_address(address))
        return len(code) > 2  # '0x' is empty bytecode
    except Exception:
        return False


# ============================================================================
# MAIN MAPPING LOGIC
# ============================================================================

def get_base_address_from_etherscan(proxy_wallet: str) -> Optional[str]:
    """
    Query Etherscan API to get contract creator (base address).
    Uses Etherscan V2 API which works across all chains with chainid parameter.

    Args:
        proxy_wallet: Proxy wallet address

    Returns:
        Base address (contract creator) if found, None otherwise
    """
    api_key = os.environ.get('ETHERSCAN_API_KEY', '')

    if not api_key:
        return None

    try:
        params = {
            'chainid': POLYGON_CHAIN_ID,
            'module': 'contract',
            'action': 'getcontractcreation',
            'contractaddresses': proxy_wallet,
            'apikey': api_key
        }

        response = requests.get(ETHERSCAN_API_URL, params=params, timeout=10)

        if response.status_code == 200:
            data = response.json()

            if data.get('status') == '1' and data.get('result'):
                results = data['result']
                if isinstance(results, list) and len(results) > 0:
                    creator = results[0].get('contractCreator')
                    if creator:
                        return creator

    except Exception:
        pass

    return None


def get_base_address_from_gnosis_safe(w3: Web3, proxy_wallet: str) -> Optional[str]:
    """
    Try to get base address by calling getOwners() (Gnosis Safe method).

    Args:
        w3: Web3 instance
        proxy_wallet: Proxy wallet address

    Returns:
        First owner address if found, None otherwise
    """
    try:
        contract = w3.eth.contract(
            address=Web3.to_checksum_address(proxy_wallet),
            abi=GNOSIS_SAFE_ABI
        )
        owners = contract.functions.getOwners().call()

        if owners and len(owners) > 0:
            return owners[0]  # Return first owner

    except Exception:
        pass

    return None


def get_base_address_from_polymarket_proxy(w3: Web3, proxy_wallet: str) -> Optional[str]:
    """
    Try to get base address by calling owner() (Polymarket Proxy method).

    Args:
        w3: Web3 instance
        proxy_wallet: Proxy wallet address

    Returns:
        Owner address if found, None otherwise
    """
    try:
        contract = w3.eth.contract(
            address=Web3.to_checksum_address(proxy_wallet),
            abi=POLYMARKET_PROXY_ABI
        )
        owner = contract.functions.owner().call()

        # Check if owner is not zero address
        if owner and owner != "0x0000000000000000000000000000000000000000":
            return owner

    except Exception:
        pass

    return None


def map_proxy_to_base(proxy_wallet: str) -> dict:
    """
    Map a proxy wallet address to its base address using direct blockchain calls.

    Args:
        proxy_wallet: The proxy wallet address to query

    Returns:
        Dictionary with results including success status, addresses, and method used
    """
    print(f"\n{'='*70}")
    print(f"POLYMARKET PROXY → BASE ADDRESS MAPPER")
    print(f"{'='*70}\n")

    # Normalize and validate address
    if not proxy_wallet.startswith('0x'):
        proxy_wallet = '0x' + proxy_wallet

    if not Web3.is_address(proxy_wallet):
        return {
            'success': False,
            'error': 'Invalid Ethereum address format',
            'proxy_wallet': proxy_wallet,
            'base_address': None,
            'method': None
        }

    proxy_wallet = Web3.to_checksum_address(proxy_wallet)
    print(f"Proxy Wallet: {proxy_wallet}\n")

    # Connect to Polygon
    print("Connecting to Polygon...")
    w3 = get_web3_connection()

    if not w3:
        return {
            'success': False,
            'error': 'Failed to connect to Polygon RPC',
            'proxy_wallet': proxy_wallet,
            'base_address': None,
            'method': None
        }

    print("✓ Connected to Polygon\n")

    # Verify it's a contract
    print("Checking if address is a contract...")
    if not is_contract(w3, proxy_wallet):
        return {
            'success': False,
            'error': 'Address is not a contract (EOA detected)',
            'proxy_wallet': proxy_wallet,
            'base_address': None,
            'method': None
        }

    print("✓ Address is a contract\n")

    # Try Method 1: Etherscan API (fastest and most reliable)
    print("[Method 1] Querying Etherscan API for contract creator...")
    base_address = get_base_address_from_etherscan(proxy_wallet)

    if base_address:
        print(f"✓ SUCCESS via Etherscan API!")
        print(f"\n{'='*70}")
        print(f"Proxy Wallet:  {proxy_wallet}")
        print(f"Base Address:  {base_address}")
        print(f"Method:        Etherscan API (contract creator)")
        print(f"{'='*70}\n")

        return {
            'success': True,
            'proxy_wallet': proxy_wallet,
            'base_address': base_address,
            'method': 'Etherscan API (contract creator)',
            'error': None
        }

    print("✗ Etherscan API method failed or no API key\n")

    # Try Method 2: Gnosis Safe (getOwners)
    print("[Method 2] Trying Gnosis Safe getOwners()...")
    base_address = get_base_address_from_gnosis_safe(w3, proxy_wallet)

    if base_address:
        print(f"✓ SUCCESS via Gnosis Safe!")
        print(f"\n{'='*70}")
        print(f"Proxy Wallet:  {proxy_wallet}")
        print(f"Base Address:  {base_address}")
        print(f"Method:        Gnosis Safe (getOwners)")
        print(f"{'='*70}\n")

        return {
            'success': True,
            'proxy_wallet': proxy_wallet,
            'base_address': base_address,
            'method': 'Gnosis Safe (getOwners)',
            'error': None
        }

    print("✗ Gnosis Safe method failed\n")

    # Try Method 3: Polymarket Proxy (owner)
    print("[Method 3] Trying Polymarket Proxy owner()...")
    base_address = get_base_address_from_polymarket_proxy(w3, proxy_wallet)

    if base_address:
        print(f"✓ SUCCESS via Polymarket Proxy!")
        print(f"\n{'='*70}")
        print(f"Proxy Wallet:  {proxy_wallet}")
        print(f"Base Address:  {base_address}")
        print(f"Method:        Polymarket Proxy (owner)")
        print(f"{'='*70}\n")

        return {
            'success': True,
            'proxy_wallet': proxy_wallet,
            'base_address': base_address,
            'method': 'Polymarket Proxy (owner)',
            'error': None
        }

    print("✗ Polymarket Proxy method failed\n")

    # Both methods failed
    print(f"{'='*70}")
    print(f"✗ FAILED - Could not retrieve base address")
    print(f"{'='*70}")
    print(f"\nProxy Wallet: {proxy_wallet}")
    print(f"\nPossible reasons:")
    print(f"  • Contract doesn't have getOwners() or owner() functions")
    print(f"  • Contract is not a Polymarket proxy wallet")
    print(f"  • Contract uses a different proxy pattern")
    print(f"{'='*70}\n")

    return {
        'success': False,
        'proxy_wallet': proxy_wallet,
        'base_address': None,
        'method': None,
        'error': 'Neither Gnosis Safe nor Polymarket Proxy methods worked'
    }


# ============================================================================
# CLI INTERFACE
# ============================================================================

def main():
    """Command-line interface."""
    if len(sys.argv) != 2:
        print("\nUsage: python proxy_to_base_mapper_direct.py <proxy_wallet_address>\n")
        print("Example:")
        print("  python proxy_to_base_mapper_direct.py 0x1234567890abcdef...\n")
        sys.exit(1)

    proxy_wallet = sys.argv[1]
    result = map_proxy_to_base(proxy_wallet)

    # Exit with appropriate status code
    sys.exit(0 if result['success'] else 1)


if __name__ == "__main__":
    main()
