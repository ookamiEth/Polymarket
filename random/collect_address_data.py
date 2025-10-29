#!/usr/bin/env python3
"""
Collect Ethereum address transaction data using Etherscan API.

Fetches USDT token transfers, normal transactions, and internal transactions
for the target address and saves as Parquet files.
"""

import logging
from pathlib import Path

from etherscan_api_client import TARGET_ADDRESS, USDT_CONTRACT, EtherscanClient

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# Output directory structure
BASE_DIR = Path("/Users/lgierhake/Documents/ETH/BT/random/data/etherscan")
RAW_DIR = BASE_DIR / "raw"
PROCESSED_DIR = BASE_DIR / "processed"

# Create directories
RAW_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

# Short address for filenames (first 10 chars)
SHORT_ADDR = TARGET_ADDRESS[:10]


def collect_usdt_transfers(client: EtherscanClient) -> None:
    """
    Collect all USDT token transfers to target address.

    Args:
        client: Initialized Etherscan API client
    """
    logger.info("=" * 60)
    logger.info("Collecting USDT Token Transfers")
    logger.info("=" * 60)
    logger.info(f"Target Address: {TARGET_ADDRESS}")
    logger.info(f"USDT Contract: {USDT_CONTRACT}")
    logger.info("")

    # Fetch USDT transfers
    usdt_df = client.get_usdt_transfers(
        address=TARGET_ADDRESS,
        contract_address=USDT_CONTRACT,
        sort="asc",  # Chronological order
    )

    if len(usdt_df) == 0:
        logger.warning("No USDT transfers found!")
        return

    # Filter for deposits only (where 'to' = target address)
    deposits_df = usdt_df.filter(usdt_df["to"].str.to_lowercase() == TARGET_ADDRESS.lower())

    logger.info("")
    logger.info(f"Total USDT transactions: {len(usdt_df):,}")
    logger.info(f"USDT deposits (to address): {len(deposits_df):,}")
    logger.info(f"USDT withdrawals (from address): {len(usdt_df) - len(deposits_df):,}")

    # Save raw data (all USDT transfers)
    output_file_all = RAW_DIR / f"usdt_transfers_{SHORT_ADDR}.parquet"
    usdt_df.write_parquet(
        output_file_all,
        compression="snappy",
        statistics=True,
    )
    logger.info(f"✅ Saved all USDT transfers to: {output_file_all}")

    # Save deposits only
    output_file_deposits = RAW_DIR / f"usdt_deposits_{SHORT_ADDR}.parquet"
    deposits_df.write_parquet(
        output_file_deposits,
        compression="snappy",
        statistics=True,
    )
    logger.info(f"✅ Saved USDT deposits to: {output_file_deposits}")

    # Basic statistics
    logger.info("")
    logger.info("USDT Deposit Statistics:")
    logger.info("-" * 60)
    total_usdt = deposits_df["TokenValue"].sum()
    avg_usdt = deposits_df["TokenValue"].mean()
    median_usdt = deposits_df["TokenValue"].median()
    min_usdt = deposits_df["TokenValue"].min()
    max_usdt = deposits_df["TokenValue"].max()

    logger.info(f"  Total Deposited: ${total_usdt:,.2f} USDT")
    logger.info(f"  Average Deposit: ${avg_usdt:,.2f} USDT")
    logger.info(f"  Median Deposit:  ${median_usdt:,.2f} USDT")
    logger.info(f"  Min Deposit:     ${min_usdt:,.2f} USDT")
    logger.info(f"  Max Deposit:     ${max_usdt:,.2f} USDT")
    logger.info("")


def collect_normal_transactions(client: EtherscanClient) -> None:
    """
    Collect all normal transactions involving target address.

    Args:
        client: Initialized Etherscan API client
    """
    logger.info("=" * 60)
    logger.info("Collecting Normal Transactions")
    logger.info("=" * 60)
    logger.info(f"Target Address: {TARGET_ADDRESS}")
    logger.info("")

    # Fetch normal transactions
    txs_df = client.get_normal_transactions(
        address=TARGET_ADDRESS,
        sort="asc",
    )

    if len(txs_df) == 0:
        logger.warning("No normal transactions found!")
        return

    logger.info(f"Total normal transactions: {len(txs_df):,}")

    # Gas statistics
    total_gas_eth = txs_df["GasCostETH"].sum()
    avg_gas_eth = txs_df["GasCostETH"].mean()

    logger.info(f"Total gas spent: {total_gas_eth:.6f} ETH")
    logger.info(f"Average gas per tx: {avg_gas_eth:.6f} ETH")

    # Save raw data
    output_file = RAW_DIR / f"normal_txs_{SHORT_ADDR}.parquet"
    txs_df.write_parquet(
        output_file,
        compression="snappy",
        statistics=True,
    )
    logger.info(f"✅ Saved normal transactions to: {output_file}")
    logger.info("")


def collect_internal_transactions(client: EtherscanClient) -> None:
    """
    Collect all internal transactions involving target address.

    Args:
        client: Initialized Etherscan API client
    """
    logger.info("=" * 60)
    logger.info("Collecting Internal Transactions")
    logger.info("=" * 60)
    logger.info(f"Target Address: {TARGET_ADDRESS}")
    logger.info("")

    # Fetch internal transactions
    txs_df = client.get_internal_transactions(
        address=TARGET_ADDRESS,
        sort="asc",
    )

    if len(txs_df) == 0:
        logger.warning("No internal transactions found!")
        return

    logger.info(f"Total internal transactions: {len(txs_df):,}")

    # Save raw data
    output_file = RAW_DIR / f"internal_txs_{SHORT_ADDR}.parquet"
    txs_df.write_parquet(
        output_file,
        compression="snappy",
        statistics=True,
    )
    logger.info(f"✅ Saved internal transactions to: {output_file}")
    logger.info("")


def generate_summary() -> None:
    """Generate summary of collected data."""
    logger.info("=" * 60)
    logger.info("Collection Summary")
    logger.info("=" * 60)
    logger.info("")
    logger.info("Files saved to:")
    logger.info(f"  {RAW_DIR}")
    logger.info("")
    logger.info("Available files:")

    # List all files in raw directory
    for file in sorted(RAW_DIR.glob("*.parquet")):
        file_size_mb = file.stat().st_size / (1024 * 1024)
        logger.info(f"  - {file.name} ({file_size_mb:.2f} MB)")

    logger.info("")
    logger.info("Next step:")
    logger.info("  Run: uv run python analyze_address_deposits.py")
    logger.info("")
    logger.info("=" * 60)


def main() -> None:
    """Main entry point for data collection."""
    logger.info("")
    logger.info("╔" + "=" * 58 + "╗")
    logger.info("║" + " " * 58 + "║")
    logger.info("║  Ethereum Address Data Collection (Etherscan API)" + " " * 8 + "║")
    logger.info("║" + " " * 58 + "║")
    logger.info("╚" + "=" * 58 + "╝")
    logger.info("")

    # Initialize API client
    client = EtherscanClient()

    try:
        # Collect all data types
        collect_usdt_transfers(client)
        collect_normal_transactions(client)
        collect_internal_transactions(client)

        # Generate summary
        generate_summary()

    except Exception as e:
        logger.error(f"Collection failed: {e}")
        raise

    finally:
        client.close()


if __name__ == "__main__":
    main()
