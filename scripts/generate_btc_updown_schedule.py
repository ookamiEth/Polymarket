#!/usr/bin/env python3
"""
Generate 1 week schedule of BTC Up or Down 15-minute market timestamps

Creates a parquet file with all 15-minute market periods starting from
a given timestamp for the next 7 days.

Usage:
    uv run python scripts/generate_btc_updown_schedule.py
"""

import polars as pl
from datetime import datetime, timedelta

def generate_btc_updown_schedule(
    start_timestamp: int = 1759753800,
    days: int = 7,
    interval_minutes: int = 15
) -> pl.DataFrame:
    """
    Generate schedule of BTC Up or Down 15-minute markets

    Args:
        start_timestamp: Unix timestamp to start from
        days: Number of days to generate
        interval_minutes: Market interval in minutes (default: 15)

    Returns:
        Polars DataFrame with market schedule
    """

    interval_seconds = interval_minutes * 60
    total_periods = (days * 24 * 60) // interval_minutes

    print(f"Generating BTC Up or Down 15-minute market schedule")
    print(f"Start timestamp: {start_timestamp}")
    print(f"Start datetime: {datetime.fromtimestamp(start_timestamp)}")
    print(f"Duration: {days} days")
    print(f"Interval: {interval_minutes} minutes")
    print(f"Total periods: {total_periods}\n")

    # Generate timestamps
    timestamps = []
    for i in range(total_periods):
        ts = start_timestamp + (i * interval_seconds)
        timestamps.append(ts)

    # Create DataFrame
    df = pl.DataFrame({
        "period_number": list(range(1, total_periods + 1)),
        "start_timestamp": timestamps,
        "end_timestamp": [ts + interval_seconds for ts in timestamps],
        "start_datetime": [datetime.fromtimestamp(ts) for ts in timestamps],
        "end_datetime": [datetime.fromtimestamp(ts + interval_seconds) for ts in timestamps],
        "slug": [f"btc-up-or-down-15m-{ts}" for ts in timestamps],
    })

    # Add human-readable columns
    df = df.with_columns([
        pl.col("start_datetime").dt.strftime("%Y-%m-%d").alias("date"),
        pl.col("start_datetime").dt.strftime("%A").alias("day_of_week"),
        pl.col("start_datetime").dt.strftime("%H:%M").alias("time_start"),
        pl.col("end_datetime").dt.strftime("%H:%M").alias("time_end"),
    ])

    return df


def main():
    """Generate and save schedule"""

    # Generate schedule
    df = generate_btc_updown_schedule(
        start_timestamp=1759753800,
        days=7,
        interval_minutes=15
    )

    # Display summary
    print(f"Generated {len(df)} market periods\n")
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
    output_file = "data/btc_updown_schedule_7days.parquet"
    df.write_parquet(output_file)

    print("\n" + "="*80)
    print(f"✓ Saved to: {output_file}")
    print(f"  Total periods: {len(df)}")
    print(f"  Start: {df['start_datetime'][0]}")
    print(f"  End: {df['end_datetime'][-1]}")
    print(f"  Duration: {days} days")

    # Show daily breakdown
    print("\n" + "="*80)
    print("Daily breakdown:")
    daily = df.group_by("date").agg([
        pl.count().alias("periods_per_day"),
        pl.col("start_timestamp").min().alias("first_timestamp"),
        pl.col("start_timestamp").max().alias("last_timestamp")
    ]).sort("date")
    print(daily)

    print("\n" + "="*80)
    print("Available columns in parquet file:")
    for col in df.columns:
        print(f"  - {col}: {df[col].dtype}")


if __name__ == "__main__":
    days = 7
    main()
