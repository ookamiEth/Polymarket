#!/usr/bin/env python3
"""
Binance BTC Perpetual Funding Rate Collector

Collects historical funding rate data for BTC perpetual futures from Binance
and saves it as a Parquet file using Polars.

Usage:
    uv run python research/risk_free_rate/collect_binance_funding_rates.py
    uv run python research/risk_free_rate/collect_binance_funding_rates.py --test  # Test with 1 day of data
"""

import argparse
import json
import logging
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import httpx
import polars as pl

# Constants
BASE_URL = "https://fapi.binance.com"
FUNDING_ENDPOINT = "/fapi/v1/fundingRate"
SYMBOL = "BTCUSDT"
MAX_LIMIT = 1000  # API max records per request
RATE_LIMIT_DELAY = 0.65  # Conservative: 500 req/5min = 100 req/min = ~1.67 req/sec

# Date ranges (Unix timestamps in milliseconds)
# Start 7 days earlier (Sep 24) to allow 7-day MA calculation from Oct 1
START_DATE = datetime(2023, 9, 24, 0, 0, 0, tzinfo=timezone.utc)
END_DATE = datetime(2025, 10, 1, 0, 0, 0, tzinfo=timezone.utc)
START_TIMESTAMP = int(START_DATE.timestamp() * 1000)
END_TIMESTAMP = int(END_DATE.timestamp() * 1000)

# Funding rate interval (8 hours in milliseconds)
FUNDING_INTERVAL_MS = 8 * 60 * 60 * 1000

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
logger = logging.getLogger(__name__)


class BinanceFundingRateCollector:
    """Collector for Binance perpetual funding rates with proper rate limiting"""

    def __init__(self, output_dir: Path = Path("research/risk_free_rate/data")):
        """
        Initialize the funding rate collector

        Args:
            output_dir: Directory to save output Parquet files
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.session = httpx.Client(timeout=30.0)
        self.all_data: list[dict] = []

    def fetch_funding_rates(
        self, start_time: int, end_time: Optional[int] = None, limit: int = MAX_LIMIT
    ) -> list[dict]:
        """
        Fetch funding rates for a given time range

        Args:
            start_time: Start timestamp in milliseconds (inclusive)
            end_time: End timestamp in milliseconds (inclusive)
            limit: Number of records to fetch (max 1000)

        Returns:
            List of funding rate records
        """
        params = {"symbol": SYMBOL, "startTime": start_time, "limit": limit}

        if end_time:
            params["endTime"] = end_time

        try:
            response = self.session.get(BASE_URL + FUNDING_ENDPOINT, params=params)
            response.raise_for_status()
            return response.json()

        except httpx.HTTPStatusError as e:
            logger.error(f"HTTP error {e.response.status_code}: {e.response.text}")
            raise
        except Exception as e:
            logger.error(f"Error fetching data: {e}")
            raise

    def collect_historical_data(
        self, start_timestamp: int, end_timestamp: int, test_mode: bool = False
    ) -> pl.DataFrame:
        """
        Collect all historical funding rate data for the specified period

        Args:
            start_timestamp: Start timestamp in milliseconds
            end_timestamp: End timestamp in milliseconds
            test_mode: If True, only collect 1 day of data for testing

        Returns:
            DataFrame with all collected funding rate data
        """
        logger.info(
            f"Starting data collection from {datetime.fromtimestamp(start_timestamp / 1000, tz=timezone.utc)} to {datetime.fromtimestamp(end_timestamp / 1000, tz=timezone.utc)}"
        )

        if test_mode:
            # In test mode, only collect 1 day of data
            end_timestamp = start_timestamp + (24 * 60 * 60 * 1000)
            logger.info("TEST MODE: Collecting only 1 day of data")

        current_start = start_timestamp
        batch_count = 0
        total_records = 0

        # Estimate total batches (8-hour intervals, 1000 records per batch)
        total_intervals = (end_timestamp - start_timestamp) // FUNDING_INTERVAL_MS
        estimated_batches = (total_intervals // MAX_LIMIT) + 1

        logger.info(f"Estimated funding periods: {total_intervals:,}")
        logger.info(f"Estimated API calls needed: {estimated_batches}")

        while current_start < end_timestamp:
            batch_count += 1

            # Calculate batch end time (can't exceed overall end time)
            # Each funding rate is 8 hours apart, so 1000 records = 8000 hours
            batch_end = min(current_start + (MAX_LIMIT * FUNDING_INTERVAL_MS), end_timestamp)

            logger.info(
                f"Batch {batch_count}/{estimated_batches}: Fetching from {datetime.fromtimestamp(current_start / 1000, tz=timezone.utc).strftime('%Y-%m-%d %H:%M')}"
            )

            try:
                # Fetch batch
                batch_data = self.fetch_funding_rates(start_time=current_start, end_time=batch_end, limit=MAX_LIMIT)

                if not batch_data:
                    logger.warning(f"No data returned for batch {batch_count}")
                    break

                self.all_data.extend(batch_data)
                batch_size = len(batch_data)
                total_records += batch_size

                logger.info(f"  → Fetched {batch_size} records (total: {total_records:,})")

                # Check if we got less than limit (means we reached the end)
                if batch_size < MAX_LIMIT:
                    logger.info("Received fewer records than limit, collection complete")
                    break

                # Update start time for next batch
                # Use the last funding time from this batch as the next start
                last_funding_time = batch_data[-1]["fundingTime"]
                current_start = last_funding_time + 1  # Add 1ms to avoid duplicates

                # Rate limiting
                if current_start < end_timestamp:
                    time.sleep(RATE_LIMIT_DELAY)

            except Exception as e:
                logger.error(f"Error in batch {batch_count}: {e}")
                # Exponential backoff on error
                wait_time = min(2**batch_count, 60)
                logger.info(f"Waiting {wait_time} seconds before retry...")
                time.sleep(wait_time)
                continue

        logger.info(f"✓ Collection complete: {total_records:,} total records collected")

        # Convert to DataFrame
        return self._create_dataframe(self.all_data)

    def _create_dataframe(self, data: list[dict]) -> pl.DataFrame:
        """
        Convert raw API data to structured Polars DataFrame

        Args:
            data: List of funding rate records from API

        Returns:
            Polars DataFrame with processed funding rate data
        """
        if not data:
            logger.warning("No data to process")
            return pl.DataFrame()

        # Create DataFrame with proper schema
        df = pl.DataFrame(data)

        # Process and enrich the data
        df = df.with_columns(
            [
                # Convert string funding rate to float
                pl.col("fundingRate").cast(pl.Float64).alias("funding_rate"),
                # Convert mark price string to float, handling empty strings
                pl.when(pl.col("markPrice") == "")
                .then(None)
                .otherwise(pl.col("markPrice"))
                .cast(pl.Float64)
                .alias("mark_price"),
                # Keep original funding time as int64
                pl.col("fundingTime").cast(pl.Int64).alias("funding_time_ms"),
                # Convert to datetime (from milliseconds)
                pl.col("fundingTime").cast(pl.Datetime("ms")).alias("funding_time_utc"),
                # Calculate funding rate as percentage
                (pl.col("fundingRate").cast(pl.Float64) * 100).alias("funding_rate_pct"),
                # Calculate annualized rate (8-hour rate * 3 * 365)
                (pl.col("fundingRate").cast(pl.Float64) * 3 * 365).alias("annualized_rate"),
                # Add collection metadata
                pl.lit(datetime.now(timezone.utc)).alias("collected_at"),
                pl.lit(SYMBOL).alias("symbol"),
            ]
        )

        # Select and order columns
        df = df.select(
            [
                "symbol",
                "funding_time_ms",
                "funding_time_utc",
                "funding_rate",
                "funding_rate_pct",
                "annualized_rate",
                "mark_price",
                "collected_at",
            ]
        )

        # Sort by funding time
        df = df.sort("funding_time_ms")

        return df

    def validate_data(self, df: pl.DataFrame) -> dict:
        """
        Validate the collected data and return statistics

        Args:
            df: DataFrame to validate

        Returns:
            Dictionary with validation results and statistics
        """
        stats = {}

        # Basic statistics
        stats["total_records"] = len(df)
        stats["date_range"] = {"start": df["funding_time_utc"].min(), "end": df["funding_time_utc"].max()}

        # Check for duplicates
        duplicates = df.group_by("funding_time_ms").agg(pl.len().alias("count")).filter(pl.col("count") > 1)
        stats["duplicate_timestamps"] = len(duplicates)

        # Check time intervals (should be 8 hours = 28800000 ms)
        time_diffs = df.select((pl.col("funding_time_ms").diff()).alias("interval_ms")).filter(
            pl.col("interval_ms").is_not_null()
        )

        td_min = time_diffs["interval_ms"].min() if len(time_diffs) > 0 else None
        td_max = time_diffs["interval_ms"].max() if len(time_diffs) > 0 else None
        td_median = time_diffs["interval_ms"].median() if len(time_diffs) > 0 else None

        stats["interval_stats"] = {
            "expected_ms": FUNDING_INTERVAL_MS,
            "min_interval_ms": td_min,
            "max_interval_ms": td_max,
            "median_interval_ms": td_median,
        }

        # Check for gaps (intervals > 8 hours)
        gaps = time_diffs.filter(pl.col("interval_ms") > FUNDING_INTERVAL_MS)
        stats["gaps_found"] = len(gaps)

        # Funding rate statistics
        fr_min = df["funding_rate"].min()
        fr_max = df["funding_rate"].max()
        fr_mean = df["funding_rate"].mean()
        fr_median = df["funding_rate"].median()
        fr_std = df["funding_rate"].std()

        stats["funding_rate_stats"] = {
            "min": float(fr_min) if fr_min is not None else 0.0,  # type: ignore
            "max": float(fr_max) if fr_max is not None else 0.0,  # type: ignore
            "mean": float(fr_mean) if fr_mean is not None else 0.0,  # type: ignore
            "median": float(fr_median) if fr_median is not None else 0.0,  # type: ignore
            "std": float(fr_std) if fr_std is not None else 0.0,  # type: ignore
            "positive_count": len(df.filter(pl.col("funding_rate") > 0)),
            "negative_count": len(df.filter(pl.col("funding_rate") < 0)),
            "zero_count": len(df.filter(pl.col("funding_rate") == 0)),
        }

        # Annualized rate statistics
        ar_min = df["annualized_rate"].min()
        ar_max = df["annualized_rate"].max()
        ar_mean = df["annualized_rate"].mean()
        ar_median = df["annualized_rate"].median()

        stats["annualized_rate_stats"] = {
            "min_pct": float(ar_min * 100) if ar_min is not None else 0.0,  # type: ignore
            "max_pct": float(ar_max * 100) if ar_max is not None else 0.0,  # type: ignore
            "mean_pct": float(ar_mean * 100) if ar_mean is not None else 0.0,  # type: ignore
            "median_pct": float(ar_median * 100) if ar_median is not None else 0.0,  # type: ignore
        }

        # Mark price statistics (handle potential null values)
        mark_price_not_null = df.filter(pl.col("mark_price").is_not_null())
        if len(mark_price_not_null) > 0:
            mp_min = mark_price_not_null["mark_price"].min()
            mp_max = mark_price_not_null["mark_price"].max()
            mp_mean = mark_price_not_null["mark_price"].mean()
            stats["mark_price_stats"] = {
                "min": float(mp_min) if mp_min is not None else None,  # type: ignore
                "max": float(mp_max) if mp_max is not None else None,  # type: ignore
                "mean": float(mp_mean) if mp_mean is not None else None,  # type: ignore
                "null_count": len(df) - len(mark_price_not_null),
            }
        else:
            stats["mark_price_stats"] = {"min": None, "max": None, "mean": None, "null_count": len(df)}

        return stats

    def save_to_parquet(self, df: pl.DataFrame, filename: str) -> Path:
        """
        Save DataFrame to Parquet file

        Args:
            df: DataFrame to save
            filename: Output filename

        Returns:
            Path to saved file
        """
        output_path = self.output_dir / filename

        df.write_parquet(output_path, compression="snappy", statistics=True, use_pyarrow=True)

        file_size_mb = output_path.stat().st_size / (1024 * 1024)
        logger.info(f"✓ Saved {len(df):,} records to {output_path} ({file_size_mb:.2f} MB)")

        return output_path

    def close(self):
        """Clean up resources"""
        self.session.close()


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="Collect Binance BTC perpetual funding rates")
    parser.add_argument("--test", action="store_true", help="Test mode: collect only 1 day of data")
    parser.add_argument("--start-date", type=str, help="Start date (YYYY-MM-DD), default: 2023-10-01")
    parser.add_argument("--end-date", type=str, help="End date (YYYY-MM-DD), default: 2025-10-01")

    args = parser.parse_args()

    # Parse custom date range if provided
    if args.start_date:
        start_dt = datetime.strptime(args.start_date, "%Y-%m-%d").replace(tzinfo=timezone.utc)
        start_ts = int(start_dt.timestamp() * 1000)
    else:
        start_ts = START_TIMESTAMP

    if args.end_date:
        end_dt = datetime.strptime(args.end_date, "%Y-%m-%d").replace(tzinfo=timezone.utc)
        end_ts = int(end_dt.timestamp() * 1000)
    else:
        end_ts = END_TIMESTAMP

    collector = BinanceFundingRateCollector()

    try:
        # Collect data
        df = collector.collect_historical_data(start_timestamp=start_ts, end_timestamp=end_ts, test_mode=args.test)

        if df.is_empty():
            logger.error("No data collected")
            sys.exit(1)

        # Validate data
        logger.info("Validating collected data...")
        stats = collector.validate_data(df)

        # Log validation results
        logger.info("=== Data Validation Results ===")
        logger.info(f"Total records: {stats['total_records']:,}")
        logger.info(f"Date range: {stats['date_range']['start']} to {stats['date_range']['end']}")
        logger.info(f"Duplicate timestamps: {stats['duplicate_timestamps']}")
        logger.info(f"Gaps found: {stats['gaps_found']}")
        logger.info(
            f"Funding rate range: {stats['funding_rate_stats']['min']:.6f} to {stats['funding_rate_stats']['max']:.6f}"
        )
        logger.info(f"Mean funding rate: {stats['funding_rate_stats']['mean']:.6f}")
        logger.info(
            f"Positive/Negative/Zero: {stats['funding_rate_stats']['positive_count']}/{stats['funding_rate_stats']['negative_count']}/{stats['funding_rate_stats']['zero_count']}"
        )
        logger.info(f"Mean annualized rate: {stats['annualized_rate_stats']['mean_pct']:.2f}%")

        # Log mark price stats if available
        if stats["mark_price_stats"]["min"] is not None:
            logger.info(
                f"Mark price range: ${stats['mark_price_stats']['min']:,.2f} - ${stats['mark_price_stats']['max']:,.2f}"
            )
            if stats["mark_price_stats"]["null_count"] > 0:
                logger.info(f"Mark price null values: {stats['mark_price_stats']['null_count']}")
        else:
            logger.info(f"Mark price: All values are null ({stats['mark_price_stats']['null_count']} records)")

        # Remove duplicates if any
        if stats["duplicate_timestamps"] > 0:
            logger.warning(f"Removing {stats['duplicate_timestamps']} duplicate timestamps...")
            df = df.unique(subset=["funding_time_ms"], maintain_order=True)

        # Save to Parquet
        filename = "btc_funding_rates_test.parquet" if args.test else "btc_funding_rates_2023_2025.parquet"

        output_path = collector.save_to_parquet(df, filename)

        # Save statistics as JSON
        stats_path = output_path.with_suffix(".stats.json")
        with open(stats_path, "w") as f:
            # Convert datetime objects to strings for JSON serialization
            stats_json = stats.copy()
            stats_json["date_range"]["start"] = str(stats["date_range"]["start"])
            stats_json["date_range"]["end"] = str(stats["date_range"]["end"])
            json.dump(stats_json, f, indent=2)
        logger.info(f"✓ Saved statistics to {stats_path}")

        logger.info("✓ Data collection complete!")

    except KeyboardInterrupt:
        logger.info("\nCollection interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Collection failed: {e}")
        sys.exit(1)
    finally:
        collector.close()


if __name__ == "__main__":
    main()
