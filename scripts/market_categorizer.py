#!/usr/bin/env python3
"""
Market categorizer for consolidating CLOB tick data.

Categorizes markets by type (15min, hourly, daily, etc.) based on slug pattern.
Used by batch collectors to stream data directly into consolidated files.
"""

import re
from typing import Optional

# All market category patterns (from consolidate_clob_ticks.py)
MARKET_PATTERNS = {
    # 15-minute interval markets
    'btc_15min': r'^btc-up-or-down-15m-\d+$',
    'eth_15min': r'^eth-up-or-down-15m-\d+$',

    # Hourly markets (exclude "on-" daily and "-candle" variants, include optional variant numbers)
    'bitcoin_hourly': r'^bitcoin-up-or-down-(?!on-)(?!.*candle).+-et(-\d+)?$',
    'ethereum_hourly': r'^ethereum-up-or-down-(?!on-)(?!.*candle).+-et(-\d+)?$',
    'solana_hourly': r'^solana-up-or-down-(?!on-)(?!.*candle).+-et(-\d+)?$',
    'xrp_hourly': r'^xrp-up-or-down-(?!on-)(?!.*candle).+-et(-\d+)?$',

    # Daily markets (on-DATE format, include optional variant numbers and special words like "noon")
    'bitcoin_daily': r'^bitcoin-up-or-down-on-[a-z]+-\d+(-[a-z]+)?(-\d+)?$',
    'ethereum_daily': r'^ethereum-up-or-down-on-[a-z]+-\d+(-[a-z]+)?(-\d+)?$',
    'solana_daily': r'^solana-up-or-down-on-[a-z]+-\d+(-[a-z]+)?(-\d+)?$',
    'xrp_daily': r'^xrp-up-or-down-on-[a-z]+-\d+(-[a-z]+)?(-\d+)?$',

    # Candle markets (hourly with candle suffix)
    'bitcoin_hourly_candle': r'^bitcoin-up-or-down-.*-candle$',
    'ethereum_hourly_candle': r'^ethereum-up-or-down-.*-candle$',

    # Special markets
    'bitcoin_dominance': r'^bitcoin-dominance-.*$',
    'ethbtc': r'^ethbtc-.*$',
    'soleth': r'^soleth-.*$',
    'bitcoin_aggregated': r'^bitcoin-up-or-down-(in-|this-).*$',
    'ethereum_aggregated': r'^ethereum-up-or-down-(in-|this-).*$',
    'solana_aggregated': r'^solana-up-or-down-(in-|this-).*$',
    'xrp_aggregated': r'^xrp-up-or-down-(in-|this-).*$',
}

# List of all valid categories
ALL_CATEGORIES = list(MARKET_PATTERNS.keys())


def categorize_market(slug: str) -> Optional[str]:
    """
    Categorize a market by its slug.

    Args:
        slug: Market slug (e.g., "btc-up-or-down-15m-1758561300")

    Returns:
        Category string (e.g., "btc_15min") or None if uncategorized
    """
    if not slug:
        return None

    for category, pattern in MARKET_PATTERNS.items():
        if re.match(pattern, slug):
            return category

    return None


def get_consolidated_filename(category: str) -> str:
    """
    Get consolidated parquet filename for a category.

    Args:
        category: Market category (e.g., "btc_15min")

    Returns:
        Filename (e.g., "btc_15min_consolidated.parquet")
    """
    return f"{category}_consolidated.parquet"


def is_valid_category(category: str) -> bool:
    """Check if a category string is valid."""
    return category in MARKET_PATTERNS


# Quick test
if __name__ == '__main__':
    test_cases = [
        ("btc-up-or-down-15m-1758561300", "btc_15min"),
        ("eth-up-or-down-15m-1759297500", "eth_15min"),
        ("bitcoin-up-or-down-june-11-2am-et-833", "bitcoin_hourly"),
        ("bitcoin-up-or-down-on-june-6-717", "bitcoin_daily"),
        ("bitcoin-up-or-down-may-29-2-pm-et-candle", "bitcoin_hourly_candle"),
        ("ethbtc-up-or-down-in-may-267", "ethbtc"),
        ("solana-up-or-down-in-april", "solana_aggregated"),
        ("unknown-market-slug", None),
    ]

    print("Testing market categorizer:")
    print("=" * 80)
    for slug, expected in test_cases:
        result = categorize_market(slug)
        status = "✅" if result == expected else "❌"
        print(f"{status} {slug[:50]:<50} → {result}")

    print("\n" + "=" * 80)
    print(f"Total categories: {len(ALL_CATEGORIES)}")
    print(f"Categories: {', '.join(ALL_CATEGORIES)}")
