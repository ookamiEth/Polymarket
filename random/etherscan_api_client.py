#!/usr/bin/env python3
"""
Etherscan API client with rate limiting and error handling.

Implements robust API interaction for fetching Ethereum transaction data
with automatic pagination, rate limiting, and exponential backoff.
"""

from __future__ import annotations

import logging
import os
import time
from typing import Any, Literal

import httpx
import polars as pl
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# Constants
API_KEY = os.getenv("ETHERSCAN_API_KEY")
if not API_KEY:
    raise ValueError("ETHERSCAN_API_KEY not found in environment variables")

BASE_URL = "https://api.etherscan.io/v2/api"  # V2 API endpoint
CHAIN_ID = "1"  # Ethereum mainnet
RATE_LIMIT_DELAY = 0.21  # 5 calls/sec with buffer (free tier)
MAX_RETRIES = 5
TIMEOUT_SECONDS = 30.0

# Target address and USDT contract (hardcoded for this analysis)
TARGET_ADDRESS = "0xaB02bf85a7a851b6A379eA3D5bD3B9b4f5Dd8461"
USDT_CONTRACT = "0xdac17f958d2ee523a2206206994597c13d831ec7"


class EtherscanAPIError(Exception):
    """Custom exception for Etherscan API errors."""

    pass


class EtherscanClient:
    """
    Etherscan API client with rate limiting and pagination.

    Handles all API interactions with automatic rate limiting,
    exponential backoff on errors, and proper data conversion to Polars DataFrames.
    """

    def __init__(
        self,
        api_key: str = API_KEY,
        rate_limit_delay: float = RATE_LIMIT_DELAY,
        max_retries: int = MAX_RETRIES,
    ):
        """
        Initialize Etherscan API client.

        Args:
            api_key: Etherscan API key
            rate_limit_delay: Seconds to wait between API calls
            max_retries: Maximum number of retry attempts on errors
        """
        self.api_key = api_key
        self.rate_limit_delay = rate_limit_delay
        self.max_retries = max_retries
        self.client = httpx.Client(timeout=TIMEOUT_SECONDS)
        self.call_count = 0

    def _make_request(self, params: dict[str, str | int], retry_count: int = 0) -> dict:
        """
        Make API request with rate limiting and error handling.

        Args:
            params: API request parameters
            retry_count: Current retry attempt number

        Returns:
            API response as dictionary

        Raises:
            EtherscanAPIError: On API errors after max retries
        """
        params["apikey"] = self.api_key
        params["chainid"] = CHAIN_ID  # Required for V2 API

        try:
            # Rate limiting
            time.sleep(self.rate_limit_delay)

            # Make request
            response = self.client.get(BASE_URL, params=params)
            response.raise_for_status()
            self.call_count += 1

            data = response.json()

            # Check for API-level errors
            if data["status"] == "0":
                if "rate limit" in data.get("message", "").lower():
                    logger.warning("Rate limit reached, backing off...")
                    if retry_count < self.max_retries:
                        backoff_delay = 2**retry_count
                        logger.info(f"Retrying in {backoff_delay}s...")
                        time.sleep(backoff_delay)
                        return self._make_request(params, retry_count + 1)
                    raise EtherscanAPIError(f"Rate limit exceeded: {data['message']}")

                # Empty result is ok (no more data)
                if "no transactions found" in data.get("message", "").lower():
                    return {"status": "1", "message": "OK", "result": []}

                # Other errors
                raise EtherscanAPIError(f"API error: {data.get('message', 'Unknown')}")

            return data

        except httpx.RequestError as e:
            logger.error(f"Request error: {e}")
            if retry_count < self.max_retries:
                backoff_delay = 2**retry_count
                logger.info(f"Retrying in {backoff_delay}s...")
                time.sleep(backoff_delay)
                return self._make_request(params, retry_count + 1)
            raise EtherscanAPIError(f"Request failed after {self.max_retries} retries") from e

    def get_usdt_transfers(
        self,
        address: str = TARGET_ADDRESS,
        contract_address: str = USDT_CONTRACT,
        start_block: int = 0,
        end_block: int = 99999999,
        sort: Literal["asc", "desc"] = "asc",
    ) -> pl.DataFrame:
        """
        Fetch all USDT token transfers for an address (block-range paginated).

        Note: Etherscan limits page * offset <= 10000, so we use block ranges
        to paginate large result sets.

        Args:
            address: Ethereum address to query
            contract_address: USDT token contract address
            start_block: Starting block number
            end_block: Ending block number
            sort: Sort order (asc or desc)

        Returns:
            Polars DataFrame with all token transfers
        """
        logger.info(f"Fetching USDT transfers for {address}")
        logger.info(f"Token contract: {contract_address}")

        all_transfers = []
        current_start_block = start_block
        max_offset = 10000  # Etherscan API limit
        batch_num = 1

        while current_start_block <= end_block:
            logger.info(f"Fetching batch {batch_num} (blocks {current_start_block:,} - {end_block:,})...")

            params = {
                "module": "account",
                "action": "tokentx",
                "address": address,
                "contractaddress": contract_address,
                "startblock": current_start_block,
                "endblock": end_block,
                "page": 1,  # Always use page 1 since we paginate by block range
                "offset": max_offset,
                "sort": sort,
            }

            data = self._make_request(params)

            if len(data["result"]) == 0:
                logger.info(f"No more results after batch {batch_num}")
                break

            all_transfers.extend(data["result"])
            logger.info(f"Batch {batch_num}: Fetched {len(data['result'])} transactions")

            # If we got less than max_offset, we've fetched all remaining data
            if len(data["result"]) < max_offset:
                logger.info("Received fewer than max results, pagination complete")
                break

            # Update start block to continue from last fetched block + 1
            last_block = int(data["result"][-1]["blockNumber"])
            current_start_block = last_block + 1
            batch_num += 1

            # Safety check to prevent infinite loops
            if batch_num > 1000:
                logger.warning("Reached max batch limit (1000), stopping pagination")
                break

        logger.info(f"Total USDT transfers fetched: {len(all_transfers):,}")

        if len(all_transfers) == 0:
            logger.warning("No USDT transfers found")
            return pl.DataFrame(schema=self._get_token_transfer_schema())

        # Convert to Polars DataFrame with proper types
        return self._parse_token_transfers(all_transfers)

    def get_normal_transactions(
        self,
        address: str = TARGET_ADDRESS,
        start_block: int = 0,
        end_block: int = 99999999,
        sort: Literal["asc", "desc"] = "asc",
    ) -> pl.DataFrame:
        """
        Fetch all normal transactions for an address (block-range paginated).

        Args:
            address: Ethereum address to query
            start_block: Starting block number
            end_block: Ending block number
            sort: Sort order (asc or desc)

        Returns:
            Polars DataFrame with all normal transactions
        """
        logger.info(f"Fetching normal transactions for {address}")

        all_txs = []
        current_start_block = start_block
        max_offset = 10000
        batch_num = 1

        while current_start_block <= end_block:
            logger.info(f"Fetching batch {batch_num} (blocks {current_start_block:,} - {end_block:,})...")

            params = {
                "module": "account",
                "action": "txlist",
                "address": address,
                "startblock": current_start_block,
                "endblock": end_block,
                "page": 1,
                "offset": max_offset,
                "sort": sort,
            }

            data = self._make_request(params)

            if len(data["result"]) == 0:
                break

            all_txs.extend(data["result"])
            logger.info(f"Batch {batch_num}: Fetched {len(data['result'])} transactions")

            if len(data["result"]) < max_offset:
                break

            last_block = int(data["result"][-1]["blockNumber"])
            current_start_block = last_block + 1
            batch_num += 1

            if batch_num > 1000:
                logger.warning("Reached max batch limit, stopping")
                break

        logger.info(f"Total normal transactions fetched: {len(all_txs):,}")

        if len(all_txs) == 0:
            return pl.DataFrame(schema=self._get_normal_tx_schema())

        return self._parse_normal_transactions(all_txs)

    def get_internal_transactions(
        self,
        address: str = TARGET_ADDRESS,
        start_block: int = 0,
        end_block: int = 99999999,
        sort: Literal["asc", "desc"] = "asc",
    ) -> pl.DataFrame:
        """
        Fetch all internal transactions for an address (block-range paginated).

        Args:
            address: Ethereum address to query
            start_block: Starting block number
            end_block: Ending block number
            sort: Sort order (asc or desc)

        Returns:
            Polars DataFrame with all internal transactions
        """
        logger.info(f"Fetching internal transactions for {address}")

        all_txs = []
        current_start_block = start_block
        max_offset = 10000
        batch_num = 1

        while current_start_block <= end_block:
            logger.info(f"Fetching batch {batch_num} (blocks {current_start_block:,} - {end_block:,})...")

            params = {
                "module": "account",
                "action": "txlistinternal",
                "address": address,
                "startblock": current_start_block,
                "endblock": end_block,
                "page": 1,
                "offset": max_offset,
                "sort": sort,
            }

            data = self._make_request(params)

            if len(data["result"]) == 0:
                break

            all_txs.extend(data["result"])
            logger.info(f"Batch {batch_num}: Fetched {len(data['result'])} transactions")

            if len(data["result"]) < max_offset:
                break

            last_block = int(data["result"][-1]["blockNumber"])
            current_start_block = last_block + 1
            batch_num += 1

            if batch_num > 1000:
                logger.warning("Reached max batch limit, stopping")
                break

        logger.info(f"Total internal transactions fetched: {len(all_txs):,}")

        if len(all_txs) == 0:
            return pl.DataFrame(schema=self._get_internal_tx_schema())

        return self._parse_internal_transactions(all_txs)

    def _parse_token_transfers(self, raw_data: list[dict]) -> pl.DataFrame:
        """
        Convert raw token transfer data to typed Polars DataFrame.

        Args:
            raw_data: List of raw API response dictionaries

        Returns:
            Typed Polars DataFrame with computed columns
        """
        df = pl.DataFrame(raw_data)

        # Convert types and add computed columns
        df = df.with_columns(
            [
                pl.col("blockNumber").cast(pl.Int64),
                pl.col("timeStamp").cast(pl.Int64),
                pl.col("tokenDecimal").cast(pl.Int32),
                pl.col("gasUsed").cast(pl.Int64),
                pl.col("gasPrice").cast(pl.Int64),
            ]
        )

        # Add computed columns
        df = df.with_columns(
            [
                # Convert Unix timestamp to datetime (UTC)
                pl.from_epoch("timeStamp", time_unit="s").alias("DateTime"),
                # Convert token value (handle decimals)
                (pl.col("value").cast(pl.Float64) / (10 ** pl.col("tokenDecimal"))).alias("TokenValue"),
                # Calculate gas cost in ETH
                ((pl.col("gasUsed").cast(pl.Float64) * pl.col("gasPrice").cast(pl.Float64)) / 10**18).alias(
                    "GasCostETH"
                ),
            ]
        )

        return df

    def _parse_normal_transactions(self, raw_data: list[dict]) -> pl.DataFrame:
        """
        Convert raw normal transaction data to typed Polars DataFrame.

        Args:
            raw_data: List of raw API response dictionaries

        Returns:
            Typed Polars DataFrame
        """
        df = pl.DataFrame(raw_data)

        df = df.with_columns(
            [
                pl.col("blockNumber").cast(pl.Int64),
                pl.col("timeStamp").cast(pl.Int64),
                pl.col("value").cast(pl.Float64),
                pl.col("gasUsed").cast(pl.Int64),
                pl.col("gasPrice").cast(pl.Int64),
                pl.col("isError").cast(pl.Int32),
            ]
        )

        df = df.with_columns(
            [
                pl.from_epoch("timeStamp", time_unit="s").alias("DateTime"),
                (pl.col("value") / 10**18).alias("ValueETH"),
                ((pl.col("gasUsed").cast(pl.Float64) * pl.col("gasPrice").cast(pl.Float64)) / 10**18).alias(
                    "GasCostETH"
                ),
            ]
        )

        return df

    def _parse_internal_transactions(self, raw_data: list[dict]) -> pl.DataFrame:
        """
        Convert raw internal transaction data to typed Polars DataFrame.

        Args:
            raw_data: List of raw API response dictionaries

        Returns:
            Typed Polars DataFrame
        """
        df = pl.DataFrame(raw_data)

        df = df.with_columns(
            [
                pl.col("blockNumber").cast(pl.Int64),
                pl.col("timeStamp").cast(pl.Int64),
                pl.col("value").cast(pl.Float64),
                pl.col("gas").cast(pl.Int64),
                pl.col("gasUsed").cast(pl.Int64),
                pl.col("isError").cast(pl.Int32),
            ]
        )

        df = df.with_columns(
            [
                pl.from_epoch("timeStamp", time_unit="s").alias("DateTime"),
                (pl.col("value") / 10**18).alias("ValueETH"),
            ]
        )

        return df

    def _get_token_transfer_schema(self) -> dict[str, Any]:
        """Return Polars schema for token transfers."""
        return {
            "blockNumber": pl.Int64,
            "timeStamp": pl.Int64,
            "hash": pl.Utf8,
            "from": pl.Utf8,
            "contractAddress": pl.Utf8,
            "to": pl.Utf8,
            "value": pl.Utf8,
            "tokenName": pl.Utf8,
            "tokenSymbol": pl.Utf8,
            "tokenDecimal": pl.Int32,
            "gasUsed": pl.Int64,
            "gasPrice": pl.Int64,
            "DateTime": pl.Datetime,
            "TokenValue": pl.Float64,
            "GasCostETH": pl.Float64,
        }

    def _get_normal_tx_schema(self) -> dict[str, Any]:
        """Return Polars schema for normal transactions."""
        return {
            "blockNumber": pl.Int64,
            "timeStamp": pl.Int64,
            "hash": pl.Utf8,
            "from": pl.Utf8,
            "to": pl.Utf8,
            "value": pl.Float64,
            "gasUsed": pl.Int64,
            "gasPrice": pl.Int64,
            "isError": pl.Int32,
            "DateTime": pl.Datetime,
            "ValueETH": pl.Float64,
            "GasCostETH": pl.Float64,
        }

    def _get_internal_tx_schema(self) -> dict[str, Any]:
        """Return Polars schema for internal transactions."""
        return {
            "blockNumber": pl.Int64,
            "timeStamp": pl.Int64,
            "hash": pl.Utf8,
            "from": pl.Utf8,
            "to": pl.Utf8,
            "value": pl.Float64,
            "gas": pl.Int64,
            "gasUsed": pl.Int64,
            "isError": pl.Int32,
            "DateTime": pl.Datetime,
            "ValueETH": pl.Float64,
        }

    def close(self) -> None:
        """Close HTTP client connection."""
        self.client.close()
        logger.info(f"Total API calls made: {self.call_count}")


def main() -> None:
    """Test the Etherscan API client."""
    logger.info("Testing Etherscan API client...")

    client = EtherscanClient()

    try:
        # Test fetching USDT transfers
        logger.info("=" * 60)
        logger.info("Test 1: Fetch USDT transfers")
        logger.info("=" * 60)
        usdt_df = client.get_usdt_transfers()
        logger.info(f"Fetched {len(usdt_df):,} USDT transfers")
        if len(usdt_df) > 0:
            logger.info(f"\nFirst transfer:\n{usdt_df.head(1)}")
            logger.info(f"\nLast transfer:\n{usdt_df.tail(1)}")

    finally:
        client.close()


if __name__ == "__main__":
    main()
