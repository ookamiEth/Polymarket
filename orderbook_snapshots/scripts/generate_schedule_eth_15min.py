#!/usr/bin/env python3
"""
Generate 1 year schedule of ETH Up or Down 15-minute market timestamps

Creates a parquet file with all 15-minute market periods starting from
current date for the next 365 days.

Usage:
    cd /orderbook_snapshots
    uv run python scripts/generate_schedule_eth_15min.py
"""

import polars as pl
from datetime import datetime, timedelta, timezone
import time

def generate_eth_updown_schedule(
    start_timestamp: int = None,
    days: int = 365,
    interval_minutes: int = 15
) -> pl.DataFrame:
    """
    Generate schedule of ETH Up or Down 15-minute markets

    Args:
        start_timestamp: Unix timestamp to start from (default: current time aligned to 15-min)
        days: Number of days to generate (default: 365)
        interval_minutes: Market interval in minutes (default: 15)

    Returns:
        Polars DataFrame with market schedule
    """

    # If no start timestamp provided, use current time aligned to 15-minute boundary
    if start_timestamp is None:
        now = int(time.time())
        # Round down to nearest 15-minute boundary
        interval_seconds = interval_minutes * 60
        start_timestamp = (now // interval_seconds) * interval_seconds

    interval_seconds = interval_minutes * 60
    total_periods = (days * 24 * 60) // interval_minutes

    print(f"Generating ETH Up or Down 15-minute market schedule")
    print(f"="*80)
    print(f"Start timestamp: {start_timestamp}")
    print(f"Start datetime: {datetime.fromtimestamp(start_timestamp)}")
    print(f"Duration: {days} days")
    print(f"Interval: {interval_minutes} minutes")
    print(f"Total periods: {total_periods:,}\n")

    # Generate timestamps
    print("Generating timestamps...")
    timestamps = []
    for i in range(total_periods):
        ts = start_timestamp + (i * interval_seconds)
        timestamps.append(ts)

    print(f"Creating DataFrame with {len(timestamps):,} periods...")

    # Create DataFrame
    df = pl.DataFrame({
        "period_number": list(range(1, total_periods + 1)),
        "start_timestamp": timestamps,
        "end_timestamp": [ts + interval_seconds for ts in timestamps],
        "start_datetime": [datetime.fromtimestamp(ts) for ts in timestamps],
        "end_datetime": [datetime.fromtimestamp(ts + interval_seconds) for ts in timestamps],
        "slug": [f"eth-updown-15m-{ts}" for ts in timestamps],
    })

    print("Adding human-readable columns...")

    # Add human-readable columns
    df = df.with_columns([
        pl.col("start_datetime").dt.strftime("%Y-%m-%d").alias("date"),
        pl.col("start_datetime").dt.strftime("%A").alias("day_of_week"),
        pl.col("start_datetime").dt.strftime("%H:%M").alias("time_start"),
        pl.col("end_datetime").dt.strftime("%H:%M").alias("time_end"),
    ])

    print("✓ DataFrame created successfully\n")

    return df


def main():
    """Generate and save schedule"""

    # Generate schedule
    df = generate_eth_updown_schedule(
        start_timestamp=None,  # Auto-detect current time
        days=365,
        interval_minutes=15
    )

    # Display summary
    print(f"="*80)
    print(f"Generated {len(df):,} market periods\n")
    print("First 10 periods:")
    print(df.head(10).select([
        "period_number",
        "date",
        "time_start",
        "time_end",
        "start_timestamp",
        "slug"
    ]))

    print("\n" + "="*80)
    print("Last 10 periods:")
    print(df.tail(10).select([
        "period_number",
        "date",
        "time_start",
        "time_end",
        "start_timestamp",
        "slug"
    ]))

    # Save to parquet
    output_file = "config/eth_updown_schedule_1year.parquet"
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
    print(f"  Start: {df['start_datetime'][0]}")
    print(f"  End: {df['end_datetime'][-1]}")
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

    # Check timestamp monotonicity
    is_sorted = (df['start_timestamp'].diff().drop_nulls() > 0).all()
    if is_sorted:
        print("  ✓ Timestamps are monotonically increasing")
    else:
        print("  ✗ WARNING: Timestamps are not properly sorted!")

    print("\n" + "="*80)
    print("Schedule generation complete!")
    print("="*80)


if __name__ == "__main__":
    main()
