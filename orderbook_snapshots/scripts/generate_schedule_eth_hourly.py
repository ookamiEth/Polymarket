#!/usr/bin/env python3
"""
Generate 1 year schedule of Ethereum Up or Down hourly market timestamps

Creates a parquet file with all hourly market periods in ET timezone.
Slug format: ethereum-up-or-down-{month}-{day}-{hour}-et

Usage:
    cd /orderbook_snapshots
    uv run python scripts/generate_schedule_eth_hourly.py
"""

import polars as pl
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo
import time

def generate_eth_hourly_schedule(
    start_timestamp: int = None,
    days: int = 365,
    interval_minutes: int = 60
) -> pl.DataFrame:
    """
    Generate schedule of Ethereum Up or Down hourly markets

    Args:
        start_timestamp: Unix timestamp to start from (default: current time aligned to hour)
        days: Number of days to generate (default: 365)
        interval_minutes: Market interval in minutes (default: 60)

    Returns:
        Polars DataFrame with market schedule
    """

    # ET timezone (handles EST/EDT automatically)
    et_tz = ZoneInfo("America/New_York")

    # If no start timestamp provided, use current time aligned to hour boundary in ET
    if start_timestamp is None:
        now_utc = datetime.now(ZoneInfo("UTC"))
        now_et = now_utc.astimezone(et_tz)
        # Round down to nearest hour in ET
        aligned_et = now_et.replace(minute=0, second=0, microsecond=0)
        start_timestamp = int(aligned_et.timestamp())

    interval_seconds = interval_minutes * 60
    total_periods = (days * 24 * 60) // interval_minutes

    print(f"Generating Ethereum Up or Down hourly market schedule (ET timezone)")
    print(f"="*80)
    print(f"Start timestamp: {start_timestamp}")
    start_dt = datetime.fromtimestamp(start_timestamp, tz=et_tz)
    print(f"Start datetime (ET): {start_dt}")
    print(f"Duration: {days} days")
    print(f"Interval: {interval_minutes} minutes")
    print(f"Total periods: {total_periods:,}\n")

    # Generate timestamps and slugs
    print("Generating timestamps and slugs...")

    timestamps = []
    slugs = []

    for i in range(total_periods):
        ts = start_timestamp + (i * interval_seconds)

        # Convert to ET timezone for slug generation
        dt_et = datetime.fromtimestamp(ts, tz=et_tz)

        # Format: ethereum-up-or-down-october-8-1am-et
        month_name = dt_et.strftime("%B").lower()  # "october"
        day = dt_et.day  # 8
        hour = dt_et.hour  # 0-23

        # Convert 24-hour to 12-hour format with am/pm
        if hour == 0:
            hour_str = "12am"
        elif hour < 12:
            hour_str = f"{hour}am"
        elif hour == 12:
            hour_str = "12pm"
        else:
            hour_str = f"{hour-12}pm"

        slug = f"ethereum-up-or-down-{month_name}-{day}-{hour_str}-et"

        timestamps.append(ts)
        slugs.append(slug)

    print(f"Creating DataFrame with {len(timestamps):,} periods...")

    # Create DataFrame
    df = pl.DataFrame({
        "period_number": list(range(1, total_periods + 1)),
        "start_timestamp": timestamps,
        "end_timestamp": [ts + interval_seconds for ts in timestamps],
        "slug": slugs,
    })

    # Add datetime columns
    print("Adding datetime columns...")
    df = df.with_columns([
        pl.col("start_timestamp").map_elements(
            lambda ts: datetime.fromtimestamp(ts, tz=et_tz),
            return_dtype=pl.Datetime(time_zone="America/New_York")
        ).alias("start_datetime_et"),
        pl.col("end_timestamp").map_elements(
            lambda ts: datetime.fromtimestamp(ts, tz=et_tz),
            return_dtype=pl.Datetime(time_zone="America/New_York")
        ).alias("end_datetime_et"),
    ])

    # Add human-readable columns
    print("Adding human-readable columns...")
    df = df.with_columns([
        pl.col("start_datetime_et").dt.strftime("%Y-%m-%d").alias("date"),
        pl.col("start_datetime_et").dt.strftime("%A").alias("day_of_week"),
        pl.col("start_datetime_et").dt.strftime("%H:%M").alias("time_start_et"),
        pl.col("end_datetime_et").dt.strftime("%H:%M").alias("time_end_et"),
    ])

    print("✓ DataFrame created successfully\n")

    return df


def main():
    """Generate and save schedule"""

    # Generate schedule
    df = generate_eth_hourly_schedule(
        start_timestamp=None,  # Auto-detect current time
        days=365,
        interval_minutes=60
    )

    # Display summary
    print(f"="*80)
    print(f"Generated {len(df):,} market periods\n")
    print("First 10 periods:")
    print(df.head(10).select([
        "period_number",
        "date",
        "time_start_et",
        "time_end_et",
        "start_timestamp",
        "slug"
    ]))

    print("\n" + "="*80)
    print("Last 10 periods:")
    print(df.tail(10).select([
        "period_number",
        "date",
        "time_start_et",
        "time_end_et",
        "start_timestamp",
        "slug"
    ]))

    # Save to parquet
    output_file = "config/eth_hourly_schedule_1year.parquet"
    print(f"\n{'='*80}")
    print(f"Saving to: {output_file}")
    df.write_parquet(output_file)

    # Get file size
    import os
    file_size_kb = os.path.getsize(output_file) / 1024
    file_size_mb = file_size_kb / 1024

    print(f"\n{'='*80}")
    print(f"✓ SUCCESS: Schedule saved!")
    print(f"  File: {output_file}")
    print(f"  Size: {file_size_mb:.2f} MB ({file_size_kb:.1f} KB)")
    print(f"  Total periods: {len(df):,}")
    print(f"  Start: {df['start_datetime_et'][0]}")
    print(f"  End: {df['end_datetime_et'][-1]}")
    print(f"  Duration: 365 days")

    # Show daily breakdown (sample)
    print("\n" + "="*80)
    print("Daily breakdown (first 14 days):")
    daily = df.group_by("date").agg([
        pl.count().alias("periods_per_day"),
        pl.col("start_timestamp").min().alias("first_timestamp"),
        pl.col("start_timestamp").max().alias("last_timestamp")
    ]).sort("date")
    print(daily.head(14))

    print("\n" + "="*80)
    print("Available columns in parquet file:")
    for col in df.columns:
        print(f"  - {col}: {df[col].dtype}")

    print("\n" + "="*80)
    print("Validation:")
    # Check for duplicate slugs
    duplicate_count = len(df) - df['slug'].n_unique()
    if duplicate_count == 0:
        print("  ✓ No duplicate slugs detected")
    else:
        print(f"  ✗ WARNING: {duplicate_count} duplicate slugs found!")
        # Show some duplicates
        dupes = df.filter(pl.col("slug").is_duplicated()).select(["slug", "start_timestamp"]).head(10)
        print("  Sample duplicates:")
        print(dupes)

    # Check timestamp monotonicity
    is_sorted = (df['start_timestamp'].diff().drop_nulls() > 0).all()
    if is_sorted:
        print("  ✓ Timestamps are monotonically increasing")
    else:
        print("  ✗ WARNING: Timestamps are not properly sorted!")

    # Sample slug validation
    print("\n" + "="*80)
    print("Sample slug validation:")
    sample_slugs = df.head(5).select(["slug", "start_datetime_et"])
    print(sample_slugs)

    print("\n" + "="*80)
    print("Schedule generation complete!")
    print("="*80)


if __name__ == "__main__":
    main()
