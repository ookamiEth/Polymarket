"""
Symbol Discovery for Deribit Options

Queries Deribit API to get available options instruments and filters them
using vectorized Polars operations based on user criteria.

This enables server-side filtering by building a targeted symbol list
before calling the Tardis Raw API.
"""

import httpx
import polars as pl
from datetime import datetime, timedelta
from typing import Optional, List


class DeribitSymbolDiscovery:
    """Discovers and filters Deribit options symbols."""

    DERIBIT_API_BASE = "https://www.deribit.com/api/v2"

    @staticmethod
    async def get_available_instruments(asset: str) -> pl.DataFrame:
        """
        Query Deribit API for all available option instruments for an asset.

        Args:
            asset: 'BTC' or 'ETH'

        Returns:
            Polars DataFrame with columns:
            - instrument_name: e.g., 'BTC-9JUN20-9875-C'
            - strike: strike price (float)
            - expiration_timestamp: unix timestamp in milliseconds
            - option_type: 'call' or 'put'
            - creation_timestamp: when instrument was created
            - is_active: whether instrument is currently active
        """
        url = f"{DeribitSymbolDiscovery.DERIBIT_API_BASE}/public/get_instruments"
        params = {
            "currency": asset,
            "kind": "option",
            "expired": "false"  # Only active instruments
        }

        async with httpx.AsyncClient() as client:
            response = await client.get(url, params=params, timeout=30.0)
            response.raise_for_status()
            data = response.json()

        # Extract relevant fields from response
        instruments = data.get("result", [])

        # Convert to list of dicts for Polars
        rows = []
        for inst in instruments:
            rows.append({
                "instrument_name": inst["instrument_name"],
                "strike": float(inst["strike"]),
                "expiration_timestamp": inst["expiration_timestamp"],
                "option_type": inst["option_type"],
                "creation_timestamp": inst["creation_timestamp"],
                "is_active": inst["is_active"]
            })

        return pl.DataFrame(rows)

    @staticmethod
    def filter_symbols_vectorized(
        df: pl.DataFrame,
        reference_date: datetime,
        min_days: Optional[int] = None,
        max_days: Optional[int] = None,
        option_type: str = "both",
        strike_min: Optional[float] = None,
        strike_max: Optional[float] = None,
    ) -> List[str]:
        """
        Filter symbols using vectorized Polars operations.

        Args:
            df: DataFrame from get_available_instruments()
            reference_date: Reference date for calculating days to expiry
            min_days: Minimum days to expiry (inclusive)
            max_days: Maximum days to expiry (inclusive)
            option_type: 'call', 'put', or 'both'
            strike_min: Minimum strike price (inclusive)
            strike_max: Maximum strike price (inclusive)

        Returns:
            List of filtered symbol names
        """
        # Convert expiration timestamp to datetime and calculate days to expiry
        df = df.with_columns([
            (pl.col("expiration_timestamp") / 1000)  # Convert ms to seconds
            .cast(pl.Int64)
            .cast(pl.Datetime("s"))
            .alias("expiration_datetime")
        ])

        df = df.with_columns([
            ((pl.col("expiration_datetime") - pl.lit(reference_date)).dt.total_days())
            .alias("days_to_expiry")
        ])

        # Apply filters vectorized
        conditions = []

        # Only active instruments
        conditions.append(pl.col("is_active") == True)

        # Days to expiry filter
        if min_days is not None:
            conditions.append(pl.col("days_to_expiry") >= min_days)

        if max_days is not None:
            conditions.append(pl.col("days_to_expiry") <= max_days)

        # Option type filter
        if option_type != "both":
            conditions.append(pl.col("option_type") == option_type)

        # Strike price filter
        if strike_min is not None:
            conditions.append(pl.col("strike") >= strike_min)

        if strike_max is not None:
            conditions.append(pl.col("strike") <= strike_max)

        # Combine all conditions with AND
        if conditions:
            filter_expr = conditions[0]
            for cond in conditions[1:]:
                filter_expr = filter_expr & cond

            df = df.filter(filter_expr)

        # Return list of instrument names
        return df.select("instrument_name").to_series().to_list()


async def build_symbol_list(
    assets: List[str],
    reference_date: datetime,
    min_days: Optional[int] = None,
    max_days: Optional[int] = None,
    option_type: str = "both",
    strike_min: Optional[float] = None,
    strike_max: Optional[float] = None,
) -> List[str]:
    """
    Build complete filtered symbol list for multiple assets.

    This is the main entry point for symbol discovery.

    Args:
        assets: List of assets ('BTC', 'ETH')
        reference_date: Reference date for calculating days to expiry
        min_days: Minimum days to expiry (inclusive)
        max_days: Maximum days to expiry (inclusive)
        option_type: 'call', 'put', or 'both'
        strike_min: Minimum strike price (inclusive)
        strike_max: Maximum strike price (inclusive)

    Returns:
        List of all filtered symbols across all assets
    """
    discovery = DeribitSymbolDiscovery()
    all_symbols = []

    for asset in assets:
        print(f"Discovering symbols for {asset}...")

        # Get all available instruments for this asset
        df = await discovery.get_available_instruments(asset)
        print(f"  Found {df.shape[0]} active {asset} options")

        # Filter using vectorized operations
        filtered_symbols = discovery.filter_symbols_vectorized(
            df,
            reference_date,
            min_days,
            max_days,
            option_type,
            strike_min,
            strike_max
        )

        print(f"  After filtering: {len(filtered_symbols)} symbols")
        all_symbols.extend(filtered_symbols)

    print(f"\nTotal symbols to download: {len(all_symbols)}")
    return all_symbols
