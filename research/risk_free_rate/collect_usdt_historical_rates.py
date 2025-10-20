#!/usr/bin/env python3
"""
USDT Historical Lending Rates Collector
Collects historical USDT lending rates from Aave V3 Arbitrum (Oct 2023 - Oct 2025)
"""

import os
import sys
import json
import time
import logging
from datetime import datetime, date, timezone, timedelta
from pathlib import Path
from typing import List, Dict, Optional, Tuple

import polars as pl
import httpx
from dotenv import load_dotenv

# Load environment variables from project root
env_path = Path(__file__).parent.parent.parent / ".env"
load_dotenv(env_path)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__)

# Constants
GRAPH_API_KEY = os.getenv("GRAPH_API_KEY")
SUBGRAPH_ID = "JCNWRypm7FYwV8fx5HhzZPSFaMxgkPuw4TnR3Gpi81zk"
ENDPOINT = f"https://gateway.thegraph.com/api/{GRAPH_API_KEY}/subgraphs/id/{SUBGRAPH_ID}"

# USDT Market ID on Aave V3 Arbitrum
USDT_MARKET_ID = "0x23878914efe38d27c4d67ab83ed1b93a74d4086a"

# Date range
START_DATE = datetime(2023, 10, 1, tzinfo=timezone.utc)
END_DATE = datetime(2025, 10, 1, tzinfo=timezone.utc)


class USDTHistoricalRatesCollector:
    """Collector for historical USDT lending rates from Aave V3"""

    def __init__(self, output_dir: Path = Path("research/risk_free_rate/data")):
        """
        Initialize the collector

        Args:
            output_dir: Directory to save output files
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.session = httpx.Client(timeout=60.0)
        self.usdt_market_id = USDT_MARKET_ID

    def fetch_historical_snapshots(
        self,
        start_timestamp: int,
        end_timestamp: int,
        first: int = 1000,
        skip: int = 0
    ) -> List[Dict]:
        """
        Fetch historical market snapshots for USDT

        Args:
            start_timestamp: Start time in Unix seconds
            end_timestamp: End time in Unix seconds
            first: Number of records to fetch
            skip: Number of records to skip (pagination)

        Returns:
            List of snapshot dictionaries
        """
        query = """
        query GetUSDTHistoricalRates($marketId: String!, $startTime: Int!, $endTime: Int!, $first: Int!, $skip: Int!) {
            marketDailySnapshots(
                where: {
                    market: $marketId,
                    timestamp_gte: $startTime,
                    timestamp_lte: $endTime
                }
                orderBy: timestamp
                orderDirection: asc
                first: $first
                skip: $skip
            ) {
                id
                timestamp
                blockNumber
                rates {
                    id
                    rate
                    side
                    type
                }
                totalValueLockedUSD
                totalDepositBalanceUSD
                totalBorrowBalanceUSD
                inputTokenBalance
                inputTokenPriceUSD
                outputTokenSupply
                outputTokenPriceUSD
            }
        }
        """

        variables = {
            "marketId": self.usdt_market_id,
            "startTime": start_timestamp,
            "endTime": end_timestamp,
            "first": first,
            "skip": skip
        }

        try:
            response = self.session.post(
                ENDPOINT,
                json={"query": query, "variables": variables}
            )
            response.raise_for_status()
            result = response.json()

            if "errors" in result:
                logger.error(f"GraphQL errors: {result['errors']}")
                return []

            return result.get("data", {}).get("marketDailySnapshots", [])

        except Exception as e:
            logger.error(f"Error fetching data: {e}")
            return []

    def parse_snapshot(self, snapshot: Dict) -> Dict:
        """
        Parse a snapshot to extract key metrics

        Args:
            snapshot: Raw snapshot data from GraphQL

        Returns:
            Parsed snapshot dictionary
        """
        # Extract rates by type
        rates = {}
        for rate_item in snapshot.get("rates", []):
            side = rate_item.get("side", "")
            rate_type = rate_item.get("type", "VARIABLE")
            key = f"{side}_{rate_type}"
            rates[key] = float(rate_item.get("rate", 0))

        # Get timestamp and convert to date
        timestamp = int(snapshot.get("timestamp", 0))
        dt = datetime.fromtimestamp(timestamp, tz=timezone.utc)

        # Parse financial metrics
        tvl = float(snapshot.get("totalValueLockedUSD", 0))
        deposits = float(snapshot.get("totalDepositBalanceUSD", 0))
        borrows = float(snapshot.get("totalBorrowBalanceUSD", 0))

        # Calculate utilization rate
        utilization = (borrows / deposits * 100) if deposits > 0 else 0

        parsed = {
            "date": dt.date(),
            "timestamp": timestamp,
            "datetime": dt,
            "block_number": int(snapshot.get("blockNumber", 0)),

            # Rates (already in percentage)
            "lending_apr": rates.get("LENDER_VARIABLE", 0.0),
            "borrowing_apr": rates.get("BORROWER_VARIABLE", 0.0),
            "stable_borrow_apr": rates.get("BORROWER_STABLE", 0.0),

            # Financial metrics
            "tvl_usd": tvl,
            "total_deposits_usd": deposits,
            "total_borrows_usd": borrows,
            "utilization_rate": utilization,

            # Token metrics
            "token_balance": float(snapshot.get("inputTokenBalance", 0)),
            "token_price_usd": float(snapshot.get("inputTokenPriceUSD", 0)),

            # Supply metrics
            "output_token_supply": float(snapshot.get("outputTokenSupply", 0)),
            "output_token_price": float(snapshot.get("outputTokenPriceUSD", 0)) if snapshot.get("outputTokenPriceUSD") else None,
        }

        return parsed

    def collect_all_data(
        self,
        start_date: datetime = START_DATE,
        end_date: datetime = END_DATE
    ) -> pl.DataFrame:
        """
        Collect all historical data for the date range

        Args:
            start_date: Start date
            end_date: End date

        Returns:
            DataFrame with all historical data
        """
        start_ts = int(start_date.timestamp())
        end_ts = int(end_date.timestamp())

        logger.info(f"Collecting USDT lending rates from {start_date.date()} to {end_date.date()}")
        logger.info(f"Timestamp range: {start_ts} to {end_ts}")

        all_snapshots = []
        batch_size = 1000
        skip = 0
        total_fetched = 0

        while True:
            logger.info(f"Fetching batch: skip={skip}, batch_size={batch_size}")

            # Fetch batch
            snapshots = self.fetch_historical_snapshots(
                start_timestamp=start_ts,
                end_timestamp=end_ts,
                first=batch_size,
                skip=skip
            )

            if not snapshots:
                logger.info("No more snapshots to fetch")
                break

            # Parse snapshots
            for snapshot in snapshots:
                try:
                    parsed = self.parse_snapshot(snapshot)
                    all_snapshots.append(parsed)
                except Exception as e:
                    logger.error(f"Error parsing snapshot: {e}")
                    continue

            batch_count = len(snapshots)
            total_fetched += batch_count
            logger.info(f"Fetched {batch_count} snapshots (total: {total_fetched})")

            # Check if we got less than batch_size (means we're done)
            if batch_count < batch_size:
                break

            # Prepare for next batch
            skip += batch_size

            # Rate limiting
            time.sleep(0.5)

            # Safety limit
            if skip >= 5000:
                logger.warning("Reached safety limit of 5000 records")
                break

        logger.info(f"✓ Collection complete: {len(all_snapshots)} snapshots collected")

        # Convert to DataFrame
        if not all_snapshots:
            logger.warning("No data collected")
            return pl.DataFrame()

        df = pl.DataFrame(all_snapshots)

        # Sort by date
        df = df.sort("date")

        return df

    def fill_missing_dates(self, df: pl.DataFrame) -> Tuple[pl.DataFrame, List[date]]:
        """
        Identify and optionally fill missing dates in the time series

        Args:
            df: DataFrame with historical data

        Returns:
            Tuple of (DataFrame with filled data, list of missing dates)
        """
        if df.is_empty():
            return df, []

        # Get date range
        min_date = df["date"].min()
        max_date = df["date"].max()

        # Generate complete date range
        date_range = pl.date_range(min_date, max_date, interval="1d", eager=True)
        complete_df = pl.DataFrame({"date": date_range})

        # Find missing dates
        existing_dates = set(df["date"].to_list())
        all_dates = set(date_range.to_list())
        missing_dates = sorted(list(all_dates - existing_dates))

        if missing_dates:
            logger.warning(f"Found {len(missing_dates)} missing dates in the data")
            logger.info(f"First few missing dates: {missing_dates[:5]}")

        # Join with complete date range to identify gaps
        df_complete = complete_df.join(df, on="date", how="left")

        # Forward fill rates (rates usually don't change daily)
        df_complete = df_complete.with_columns([
            pl.col("lending_apr").forward_fill(),
            pl.col("borrowing_apr").forward_fill(),
            pl.col("stable_borrow_apr").forward_fill(),
            pl.col("utilization_rate").forward_fill(),
        ])

        return df_complete, missing_dates

    def calculate_statistics(self, df: pl.DataFrame) -> Dict:
        """
        Calculate summary statistics for the dataset

        Args:
            df: DataFrame with historical data

        Returns:
            Dictionary with statistics
        """
        if df.is_empty():
            return {}

        stats = {
            "period": {
                "start_date": str(df["date"].min()),
                "end_date": str(df["date"].max()),
                "total_days": len(df),
                "unique_days": df["date"].n_unique()
            },
            "lending_apr": {
                "mean": float(df["lending_apr"].mean()),
                "median": float(df["lending_apr"].median()),
                "std": float(df["lending_apr"].std()),
                "min": float(df["lending_apr"].min()),
                "max": float(df["lending_apr"].max()),
                "q25": float(df["lending_apr"].quantile(0.25)),
                "q75": float(df["lending_apr"].quantile(0.75))
            },
            "borrowing_apr": {
                "mean": float(df["borrowing_apr"].mean()),
                "median": float(df["borrowing_apr"].median()),
                "std": float(df["borrowing_apr"].std()),
                "min": float(df["borrowing_apr"].min()),
                "max": float(df["borrowing_apr"].max())
            },
            "utilization_rate": {
                "mean": float(df["utilization_rate"].mean()),
                "median": float(df["utilization_rate"].median()),
                "min": float(df["utilization_rate"].min()),
                "max": float(df["utilization_rate"].max())
            },
            "tvl": {
                "mean": float(df["tvl_usd"].mean()),
                "min": float(df["tvl_usd"].min()),
                "max": float(df["tvl_usd"].max()),
                "final": float(df["tvl_usd"].tail(1)[0])
            }
        }

        # Calculate monthly averages
        df_monthly = df.with_columns([
            pl.col("date").dt.year().alias("year"),
            pl.col("date").dt.month().alias("month")
        ]).group_by(["year", "month"]).agg([
            pl.col("lending_apr").mean().alias("avg_lending_apr"),
            pl.col("borrowing_apr").mean().alias("avg_borrowing_apr"),
            pl.col("utilization_rate").mean().alias("avg_utilization"),
            pl.col("tvl_usd").mean().alias("avg_tvl")
        ]).sort(["year", "month"])

        stats["monthly_averages"] = df_monthly.to_dicts()

        # Calculate rate volatility (rolling 30-day std dev)
        df_volatility = df.with_columns([
            pl.col("lending_apr").rolling_std(30).alias("lending_volatility_30d"),
            pl.col("borrowing_apr").rolling_std(30).alias("borrowing_volatility_30d")
        ])

        stats["volatility"] = {
            "lending_30d_avg": float(df_volatility["lending_volatility_30d"].mean()),
            "borrowing_30d_avg": float(df_volatility["borrowing_volatility_30d"].mean())
        }

        return stats

    def save_data(self, df: pl.DataFrame, stats: Dict) -> Tuple[Path, Path]:
        """
        Save data and statistics to files

        Args:
            df: DataFrame with historical data
            stats: Statistics dictionary

        Returns:
            Tuple of (data file path, stats file path)
        """
        # Save main data to Parquet
        data_file = self.output_dir / "usdt_lending_rates_2023_2025.parquet"
        df.write_parquet(
            data_file,
            compression="snappy",
            statistics=True,
            use_pyarrow=True
        )
        logger.info(f"✓ Saved {len(df)} records to {data_file}")

        # Save statistics to JSON
        stats_file = self.output_dir / "usdt_rates_summary.json"
        with open(stats_file, "w") as f:
            json.dump(stats, f, indent=2, default=str)
        logger.info(f"✓ Saved statistics to {stats_file}")

        return data_file, stats_file

    def close(self):
        """Clean up resources"""
        self.session.close()


def main():
    """Main entry point"""
    logger.info("=== USDT Historical Lending Rates Collection ===")

    collector = USDTHistoricalRatesCollector()

    try:
        # Collect all data
        df = collector.collect_all_data()

        if df.is_empty():
            logger.error("No data collected")
            sys.exit(1)

        # Check for missing dates
        df_filled, missing_dates = collector.fill_missing_dates(df)

        if missing_dates:
            logger.info(f"Total missing dates: {len(missing_dates)}")

            # Save list of missing dates
            missing_file = collector.output_dir / "missing_dates.json"
            with open(missing_file, "w") as f:
                json.dump([str(d) for d in missing_dates], f, indent=2)
            logger.info(f"✓ Saved missing dates to {missing_file}")

        # Calculate statistics
        stats = collector.calculate_statistics(df)

        # Display summary
        logger.info("\n=== Data Summary ===")
        logger.info(f"Period: {stats['period']['start_date']} to {stats['period']['end_date']}")
        logger.info(f"Total days: {stats['period']['total_days']}")
        logger.info(f"Unique days: {stats['period']['unique_days']}")

        logger.info("\n=== USDT Lending Rates (APR) ===")
        logger.info(f"Mean: {stats['lending_apr']['mean']:.3f}%")
        logger.info(f"Median: {stats['lending_apr']['median']:.3f}%")
        logger.info(f"Std Dev: {stats['lending_apr']['std']:.3f}%")
        logger.info(f"Range: {stats['lending_apr']['min']:.3f}% - {stats['lending_apr']['max']:.3f}%")
        logger.info(f"IQR (Q25-Q75): {stats['lending_apr']['q25']:.3f}% - {stats['lending_apr']['q75']:.3f}%")

        logger.info("\n=== Utilization Rate ===")
        logger.info(f"Mean: {stats['utilization_rate']['mean']:.1f}%")
        logger.info(f"Range: {stats['utilization_rate']['min']:.1f}% - {stats['utilization_rate']['max']:.1f}%")

        logger.info("\n=== Total Value Locked ===")
        logger.info(f"Mean: ${stats['tvl']['mean']:,.0f}")
        logger.info(f"Final: ${stats['tvl']['final']:,.0f}")

        # Show sample of data
        logger.info("\n=== Sample Data (First 5 Days) ===")
        for row in df.head(5).iter_rows(named=True):
            logger.info(
                f"{row['date']}: Lending {row['lending_apr']:.3f}%, "
                f"Borrowing {row['borrowing_apr']:.3f}%, "
                f"Utilization {row['utilization_rate']:.1f}%, "
                f"TVL ${row['tvl_usd']:,.0f}"
            )

        logger.info("\n=== Sample Data (Last 5 Days) ===")
        for row in df.tail(5).iter_rows(named=True):
            logger.info(
                f"{row['date']}: Lending {row['lending_apr']:.3f}%, "
                f"Borrowing {row['borrowing_apr']:.3f}%, "
                f"Utilization {row['utilization_rate']:.1f}%, "
                f"TVL ${row['tvl_usd']:,.0f}"
            )

        # Save data
        data_file, stats_file = collector.save_data(df, stats)

        # Also save a CSV for easy viewing
        csv_file = collector.output_dir / "usdt_lending_rates_2023_2025.csv"
        df.write_csv(csv_file)
        logger.info(f"✓ Also saved as CSV to {csv_file}")

        logger.info("\n✅ Data collection complete!")
        logger.info(f"Files created:")
        logger.info(f"  - {data_file}")
        logger.info(f"  - {stats_file}")
        logger.info(f"  - {csv_file}")

    except Exception as e:
        logger.error(f"Collection failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    finally:
        collector.close()


if __name__ == "__main__":
    main()