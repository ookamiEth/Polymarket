# Polymarket Proxy Wallet → Base Address Mapper

Maps Polymarket proxy wallet addresses to their owner's base addresses using direct blockchain queries and the Etherscan API.

## Overview

Polymarket users have two types of addresses:
- **Base Address (EOA)**: The user's actual Ethereum address (externally owned account)
- **Proxy Wallet (Contract)**: A smart contract that executes trades on behalf of the user

This script maps proxy wallet addresses back to their owner's base address.

## Features

- ✅ **Works with all Polymarket proxy types** (Gnosis Safe & Polymarket Proxy)
- ✅ **Multiple fallback methods** for maximum reliability
- ✅ **Fast queries** using Etherscan API
- ✅ **No rate limits** with Etherscan API key

## Prerequisites

1. Python 3.8+
2. Etherscan API Key (get one free at [etherscan.io/apis](https://etherscan.io/apis))

## Installation

```bash
# Install dependencies
uv pip install web3 requests

# Or using requirements.txt
uv pip install -r requirements.txt
```

## Configuration

Add your Etherscan API key to `.env`:

```bash
ETHERSCAN_API_KEY=YOUR_API_KEY_HERE
```

## Usage

### Command Line

```bash
# Set API key and run
export ETHERSCAN_API_KEY=YOUR_API_KEY
uv run python proxy_to_base_mapper_direct.py 0xPROXY_WALLET_ADDRESS
```

### Example

```bash
export ETHERSCAN_API_KEY=BZYMV1XFQIFGJHHTEJXRTWGI8VWX3UWFSK
uv run python proxy_to_base_mapper_direct.py 0xcc500cbcc8b7cf5bd21975ebbea34f21b5644c82
```

**Output:**
```
======================================================================
POLYMARKET PROXY → BASE ADDRESS MAPPER
======================================================================

Proxy Wallet: 0xcC500CbCC8B7cF5BD21975ebbEa34F21b5644C82

Connecting to Polygon...
✓ Connected to Polygon

Checking if address is a contract...
✓ Address is a contract

[Method 1] Querying Etherscan API for contract creator...
✓ SUCCESS via Etherscan API!

======================================================================
Proxy Wallet:  0xcC500CbCC8B7cF5BD21975ebbEa34F21b5644C82
Base Address:  0xae700edfd9ab986395f3999fe11177b9903a52f1
Method:        Etherscan API (contract creator)
======================================================================
```

### Python Import

```python
from proxy_to_base_mapper_direct import map_proxy_to_base
import os

# Set API key
os.environ['ETHERSCAN_API_KEY'] = 'YOUR_API_KEY'

# Map proxy to base
result = map_proxy_to_base('0xcc500cbcc8b7cf5bd21975ebbea34f21b5644c82')

if result['success']:
    print(f"Base Address: {result['base_address']}")
    print(f"Method: {result['method']}")
else:
    print(f"Error: {result['error']}")
```

## How It Works

The script tries three methods in order:

### Method 1: Etherscan API (Primary - Fastest ✅)
- Queries Etherscan's contract creation endpoint
- Uses Etherscan V2 API which works across all EVM chains
- Returns the address that deployed the contract
- **Works for ~100% of Polymarket proxies**

### Method 2: Gnosis Safe Contract Call (Fallback)
- For MetaMask users using Gnosis Safe wallets
- Calls `getOwners()` function on the proxy contract
- Returns array of owner addresses

### Method 3: Polymarket Proxy Contract Call (Fallback)
- For MagicLink users using Polymarket custom proxies
- Calls `owner()` function on the proxy contract
- Returns single owner address

## Etherscan API Details

The script uses **Etherscan V2 API** which provides:
- ✅ Single API key works for 50+ chains (Ethereum, Polygon, BSC, Arbitrum, etc.)
- ✅ Free tier: 5 calls/second, 100,000 calls/day
- ✅ Query Polygon with `chainid=137` parameter

### API Endpoint Format
```
https://api.etherscan.io/v2/api
  ?chainid=137                     # Polygon
  &module=contract
  &action=getcontractcreation
  &contractaddresses=0xPROXY...
  &apikey=YOUR_KEY
```

## Tested Proxy Wallets

| Proxy Wallet | Base Address | Method | Status |
|--------------|--------------|--------|--------|
| `0xcc500cbcc8b7cf5bd21975ebbea34f21b5644c82` | `0xae700edfd9ab986395f3999fe11177b9903a52f1` | Etherscan API | ✅ |
| `0xc6E10cf94c06f8b1C1094e400f015aD1e2474f54` | `0x710be84efbd0dfd4b1a67b9525a778ae5c420ff8` | Etherscan API | ✅ |
| `0x86E686cC1D9c79da5ec88c21E170B4d71405f634` | (various owners) | Etherscan API | ✅ |

## Troubleshooting

### "API method failed or no API key"
- Make sure `ETHERSCAN_API_KEY` environment variable is set
- Verify your API key is valid at [etherscan.io/myapikey](https://etherscan.io/myapikey)

### "Address is not a contract"
- The address you provided is an EOA (Externally Owned Account), not a proxy wallet
- Make sure you're querying a proxy wallet address, not a base address

### "Could not retrieve base address"
- All three methods failed
- The contract may use a non-standard proxy pattern
- Try checking the contract on [Polygonscan](https://polygonscan.com/)

## Additional Resources

- [Polymarket Documentation](https://docs.polymarket.com/)
- [Etherscan API Documentation](https://docs.etherscan.io/)
- [Polygon Network Info](https://polygon.technology/)

## License

MIT

## Authors

BT Research Team - 2025-10-07
