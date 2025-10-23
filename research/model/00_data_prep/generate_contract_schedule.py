#!/usr/bin/env python3
"""
Generate 15-minute binary contract schedule.

Creates a calendar of all 15-minute binary option contracts matching
Polymarket's BTC up/down market structure.

Schedule:
- 4 contracts per hour at :00, :15, :30, :45
- Each contract lasts 900 seconds (15 minutes)
- Covers full period: Oct 2023 - Sep 2025

Output schema:
- contract_id: unique identifier (YYYYMMDD_HHMM format)
- open_time: contract open timestamp (seconds)
- close_time: contract close timestamp (seconds)
- hour_of_day: 0-23
- day_of_week: 0-6 (Monday=0)
- date: YYYY-MM-DD
"""

import logging
from datetime import datetime, timezone
from pathlib import Path

import polars as pl

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Constants
OUTPUT_FILE = Path("/Users/lgierhake/Documents/ETH/BT/research/model/results/contract_schedule.parquet")
SECONDS_PER_15MIN = 900
MINUTES_PER_HOUR = 60


def generate_contract_schedule(start_date: str, end_date: str, output_path: Path) -> None:
    """
    Generate 15-minute contract schedule.

    Args:
        start_date: Start date in YYYY-MM-DD format (e.g., "2023-10-01")
        end_date: End date in YYYY-MM-DD format (e.g., "2025-09-30")
        output_path: Path to output parquet file
    """
    logger.info(f"Generating 15-minute contract schedule from {start_date} to {end_date}")

    # Parse dates to UTC midnight timestamps
    start_dt = datetime.strptime(start_date, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    end_dt = datetime.strptime(end_date, "%Y-%m-%d").replace(tzinfo=timezone.utc)

    start_ts = int(start_dt.timestamp())
    end_ts = int(end_dt.timestamp())

    logger.info(f"Start timestamp: {start_ts} ({start_dt})")
    logger.info(f"End timestamp: {end_ts} ({end_dt})")

    # Calculate total duration and number of contracts
    total_seconds = end_ts - start_ts
    total_days = total_seconds // 86400
    total_contracts = (total_seconds // SECONDS_PER_15MIN) + 1  # +1 to include last contract

    logger.info(f"Period: {total_days} days")
    logger.info(f"Expected contracts: ~{total_contracts:,} (4 per hour)")

    # Generate contract open times
    # Start at the first 15-minute boundary at or after start_ts
    first_open = start_ts - (start_ts % SECONDS_PER_15MIN)
    if first_open < start_ts:
        first_open += SECONDS_PER_15MIN

    last_open = end_ts - (end_ts % SECONDS_PER_15MIN)

    # Create range of open times (every 900 seconds)
    open_times = pl.int_range(first_open, last_open + 1, step=SECONDS_PER_15MIN, eager=True)

    logger.info(f"Generating {len(open_times):,} contracts...")

    # Build contract schedule DataFrame
    contracts = pl.DataFrame({"open_time": open_times}).with_columns(
        [
            # Close time (15 minutes after open)
            (pl.col("open_time") + SECONDS_PER_15MIN).alias("close_time"),
            # Convert to datetime for metadata extraction
            pl.from_epoch("open_time", time_unit="s").alias("open_datetime"),
        ]
    )

    # Add metadata columns
    contracts = contracts.with_columns(
        [
            # Contract ID: YYYYMMDD_HHMM
            pl.col("open_datetime").dt.strftime("%Y%m%d_%H%M").alias("contract_id"),
            # Hour of day (0-23)
            pl.col("open_datetime").dt.hour().alias("hour_of_day"),
            # Day of week (0=Monday, 6=Sunday)
            pl.col("open_datetime").dt.weekday().alias("day_of_week"),
            # Date (for joining with daily data like risk-free rates)
            pl.col("open_datetime").cast(pl.Date).alias("date"),
            # Minute within hour (should be 0, 15, 30, or 45)
            pl.col("open_datetime").dt.minute().alias("minute_of_hour"),
        ]
    ).select(
        [
            "contract_id",
            "open_time",
            "close_time",
            "date",
            "hour_of_day",
            "minute_of_hour",
            "day_of_week",
        ]
    )

    # Validation
    logger.info("Validating schedule...")

    # Check that all minutes are 0, 15, 30, or 45
    valid_minutes = contracts.filter(pl.col("minute_of_hour").is_in([0, 15, 30, 45]))
    assert len(valid_minutes) == len(contracts), "Invalid minutes found! Expected all to be 0/15/30/45"

    # Check that close_time = open_time + 900
    duration_check = contracts.with_columns([(pl.col("close_time") - pl.col("open_time")).alias("duration")])
    assert duration_check["duration"].unique().to_list() == [SECONDS_PER_15MIN], (
        "Contract durations not all 900 seconds!"
    )

    # Check for duplicates
    assert contracts["contract_id"].is_unique().all(), "Duplicate contract IDs found!"

    logger.info("✅ Validation passed")

    # Statistics
    contracts_per_day = len(contracts) / total_days
    logger.info(f"Contracts per day: {contracts_per_day:.1f}")
    logger.info(f"First contract: {contracts['contract_id'][0]} (opens: {contracts['open_time'][0]})")
    logger.info(f"Last contract: {contracts['contract_id'][-1]} (opens: {contracts['open_time'][-1]})")

    # Distribution by hour
    hour_dist = contracts.group_by("hour_of_day").agg(pl.len().alias("count")).sort("hour_of_day")
    logger.info("\nContracts by hour of day:")
    print(hour_dist)

    # Distribution by minute
    minute_dist = contracts.group_by("minute_of_hour").agg(pl.len().alias("count")).sort("minute_of_hour")
    logger.info("\nContracts by minute of hour:")
    print(minute_dist)

    # Write output
    logger.info(f"\nWriting schedule to {output_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    contracts.write_parquet(output_path, compression="snappy", statistics=True)

    logger.info(f"Output file size: {output_path.stat().st_size / 1024 / 1024:.2f} MB")

    # Show sample
    logger.info("\nFirst 10 contracts:")
    print(contracts.head(10))

    logger.info("\n✅ Contract schedule generation complete!")


def main() -> None:
    """Main entry point."""
    logger.info("15-Minute Binary Contract Schedule Generator")

    # Use the same period as our perpetual data
    # From the resampling output: 1696118400 (2023-10-01) to 1759276786 (2025-09-30)
    start_date = "2023-10-01"
    end_date = "2025-09-30"

    try:
        generate_contract_schedule(start_date, end_date, OUTPUT_FILE)
    except Exception as e:
        logger.error(f"Error generating schedule: {e}")
        raise


if __name__ == "__main__":
    main()
