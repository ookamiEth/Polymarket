#!/usr/bin/env python3
"""
Fetch USDC lending rates from Aave V3 Arbitrum for the same period as BTC funding rates.

Date range: 2023-09-24 to 2025-10-01 (matches btc_funding_rates_2023_2025.parquet)

Data source: Aave V3 Arbitrum subgraph via The Graph
Output: Parquet file with timestamp, supply_rate_apr, utilization_rate
"""

import logging
from datetime import datetime, timezone
from pathlib import Path

import polars as pl

from query_aave_subgraph import AaveV3SubgraphClient

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Constants
DATA_DIR = Path(__file__).parent / "data"
OUTPUT_FILE = DATA_DIR / "aave_usdc_rates_2023_2025.parquet"

# Date range (matches BTC funding rates data)
START_DATE = "2023-09-24"
END_DATE = "2025-10-01"


def fetch_usdc_market_id(client: AaveV3SubgraphClient) -> str:
    """
    Fetch USDC market ID from Aave V3 Arbitrum.

    Args:
        client: Initialized Aave subgraph client

    Returns:
        USDC market ID (contract address)
    """
    logger.info("Fetching USDC market ID...")

    query = """
    {
      markets(where: { inputToken_: { symbol: "USDC" } }, first: 1) {
        id
        name
        inputToken {
          symbol
          name
        }
      }
    }
    """

    result = client.query(query)
    markets = result.get("data", {}).get("markets", [])

    if not markets:
        raise ValueError("USDC market not found in Aave V3 Arbitrum")

    market = markets[0]
    market_id = market["id"]

    logger.info(f"USDC Market ID: {market_id}")
    logger.info(f"Market Name: {market['name']}")

    return market_id


def fetch_historical_snapshots(client: AaveV3SubgraphClient, market_id: str, start_ts: int, end_ts: int) -> pl.DataFrame:
    """
    Fetch daily market snapshots for USDC from Aave V3.

    Args:
        client: Initialized Aave subgraph client
        market_id: USDC market contract address
        start_ts: Start timestamp (Unix seconds)
        end_ts: End timestamp (Unix seconds)

    Returns:
        DataFrame with timestamp, supply_rate_apr, borrow_rate_apr, utilization_rate
    """
    logger.info(f"Fetching daily snapshots for period {START_DATE} to {END_DATE}")
    logger.info(f"Unix timestamps: {start_ts} to {end_ts}")

    # Fetch all daily snapshots in batches
    all_snapshots = []
    batch_size = 1000
    skip = 0

    while True:
        query = f"""
        {{
          marketDailySnapshots(
            first: {batch_size},
            skip: {skip},
            where: {{
              market: "{market_id}",
              timestamp_gte: {start_ts},
              timestamp_lte: {end_ts}
            }},
            orderBy: timestamp,
            orderDirection: asc
          ) {{
            id
            timestamp
            rates {{
              side
              type
              rate
            }}
            totalValueLockedUSD
            totalDepositBalanceUSD
            dailySupplySideRevenueUSD
            dailyBorrowUSD
            inputTokenBalance
            outputTokenSupply
            outputTokenPriceUSD
          }}
        }}
        """

        result = client.query(query)
        snapshots = result.get("data", {}).get("marketDailySnapshots", [])

        if not snapshots:
            break

        all_snapshots.extend(snapshots)
        logger.info(f"Fetched {len(snapshots)} snapshots (total: {len(all_snapshots):,})")

        if len(snapshots) < batch_size:
            break

        skip += batch_size

    logger.info(f"Total snapshots fetched: {len(all_snapshots):,}")

    if not all_snapshots:
        logger.warning("No snapshot data found for the specified period")
        return pl.DataFrame(schema={
            "timestamp": pl.Datetime,
            "supply_rate_apr": pl.Float64,
            "borrow_rate_apr": pl.Float64,
            "total_value_locked_usd": pl.Float64,
        })

    # Convert to DataFrame
    data = []
    for snapshot in all_snapshots:
        timestamp = datetime.fromtimestamp(int(snapshot["timestamp"]), tz=timezone.utc)

        # Extract rates from the rates array
        supply_rate = 0.0
        borrow_rate = 0.0

        for rate in snapshot.get("rates", []):
            if rate["side"] == "LENDER" and rate["type"] == "VARIABLE":
                supply_rate = float(rate["rate"])
            elif rate["side"] == "BORROWER" and rate["type"] == "VARIABLE":
                borrow_rate = float(rate["rate"])

        # Total value locked
        tvl = float(snapshot.get("totalValueLockedUSD", 0))

        data.append({
            "timestamp": timestamp,
            "supply_rate_apr": supply_rate,
            "borrow_rate_apr": borrow_rate,
            "total_value_locked_usd": tvl,
        })

    df = pl.DataFrame(data)
    df = df.sort("timestamp")

    logger.info(f"Converted to DataFrame: {len(df):,} rows")
    logger.info(f"Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")

    return df


def generate_summary_statistics(df: pl.DataFrame) -> None:
    """
    Print summary statistics for USDC lending rates.

    Args:
        df: DataFrame with supply_rate_apr column
    """
    logger.info("\n" + "=" * 80)
    logger.info("SUMMARY STATISTICS")
    logger.info("=" * 80)

    if len(df) == 0:
        logger.warning("No data to summarize")
        return

    logger.info(f"Period: {df['timestamp'].min()} to {df['timestamp'].max()}")
    logger.info(f"Total daily observations: {len(df):,}")

    # Supply rate statistics
    supply_col = df["supply_rate_apr"]
    logger.info("\nSupply Rate (APR %):")
    logger.info(f"  Mean: {float(supply_col.mean() or 0.0):.2f}%")
    logger.info(f"  Median: {float(supply_col.median() or 0.0):.2f}%")
    logger.info(f"  Std Dev: {float(supply_col.std() or 0.0):.2f}%")
    logger.info(f"  Min: {float(supply_col.min() or 0.0):.2f}%")
    logger.info(f"  Max: {float(supply_col.max() or 0.0):.2f}%")

    # Borrow rate statistics
    borrow_col = df["borrow_rate_apr"]
    logger.info("\nBorrow Rate (APR %):")
    logger.info(f"  Mean: {float(borrow_col.mean() or 0.0):.2f}%")
    logger.info(f"  Median: {float(borrow_col.median() or 0.0):.2f}%")
    logger.info(f"  Min: {float(borrow_col.min() or 0.0):.2f}%")
    logger.info(f"  Max: {float(borrow_col.max() or 0.0):.2f}%")

    # TVL statistics
    tvl_col = df["total_value_locked_usd"]
    logger.info("\nTotal Value Locked (USD):")
    logger.info(f"  Mean: ${float(tvl_col.mean() or 0.0):,.0f}")
    logger.info(f"  Median: ${float(tvl_col.median() or 0.0):,.0f}")
    logger.info(f"  Min: ${float(tvl_col.min() or 0.0):,.0f}")
    logger.info(f"  Max: ${float(tvl_col.max() or 0.0):,.0f}")

    logger.info("=" * 80)


def main() -> None:
    """Main execution function."""
    logger.info("=" * 80)
    logger.info("FETCHING AAVE V3 USDC LENDING RATES (ARBITRUM)")
    logger.info(f"Period: {START_DATE} to {END_DATE}")
    logger.info("=" * 80)

    # Initialize Aave subgraph client
    try:
        client = AaveV3SubgraphClient()
    except ValueError as e:
        logger.error(f"Failed to initialize client: {e}")
        logger.error("Make sure GRAPH_API_KEY is set in .env file")
        return

    try:
        # Fetch USDC market ID
        market_id = fetch_usdc_market_id(client)

        # Convert date strings to Unix timestamps
        start_ts = int(datetime.strptime(START_DATE, "%Y-%m-%d").replace(tzinfo=timezone.utc).timestamp())
        end_ts = int(datetime.strptime(END_DATE, "%Y-%m-%d").replace(tzinfo=timezone.utc).timestamp())

        # Fetch historical snapshots
        df = fetch_historical_snapshots(client, market_id, start_ts, end_ts)

        if len(df) > 0:
            # Write to Parquet
            logger.info(f"\nWriting results to {OUTPUT_FILE}")
            df.write_parquet(OUTPUT_FILE, compression="snappy", statistics=True)
            logger.info(f"✅ Wrote {len(df):,} daily observations to Parquet")

            # Generate summary statistics
            generate_summary_statistics(df)

            logger.info("\n✅ Fetch complete!")
        else:
            logger.warning("No data fetched. Skipping output.")

    except Exception as e:
        logger.error(f"Error during execution: {e}")
        raise
    finally:
        client.close()


if __name__ == "__main__":
    main()
