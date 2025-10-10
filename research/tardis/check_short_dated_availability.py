#!/usr/bin/env python3
"""
Check which first-of-month dates have short-dated (0-3 days) BTC call options.
This helps identify which free-tier Tardis dates have the data we need.
"""

import asyncio
import httpx
import polars as pl
from datetime import datetime

async def check_date_for_short_options(date_str: str):
    """Check if a specific date has options expiring within 0-3 days."""
    url = "https://www.deribit.com/api/v2/public/get_instruments"

    all_instruments = []

    # Fetch both active and expired options
    for expired_flag in ["false", "true"]:
        params = {
            "currency": "BTC",
            "kind": "option",
            "expired": expired_flag
        }

        async with httpx.AsyncClient() as client:
            response = await client.get(url, params=params, timeout=30.0)
            response.raise_for_status()
            data = response.json()

        instruments = data.get("result", [])
        all_instruments.extend(instruments)

    # Convert to DataFrame
    rows = []
    for inst in all_instruments:
        rows.append({
            "instrument_name": inst["instrument_name"],
            "strike": float(inst["strike"]),
            "expiration_timestamp": inst["expiration_timestamp"],
            "option_type": inst["option_type"],
        })

    df = pl.DataFrame(rows)

    # Calculate days to expiry
    reference_date = datetime.strptime(date_str, '%Y-%m-%d')
    ref_timestamp = int(reference_date.timestamp() * 1000)

    df = df.with_columns([
        ((pl.col('expiration_timestamp') - ref_timestamp) / (1000 * 60 * 60 * 24))
        .alias('days_to_expiry')
    ])

    # Filter for calls with 0-3 days to expiry
    short_dated = df.filter(
        (pl.col('option_type') == 'call') &
        (pl.col('days_to_expiry') >= 0) &
        (pl.col('days_to_expiry') <= 3)
    )

    return {
        'date': date_str,
        'total_options': df.shape[0],
        'short_dated_calls': short_dated.shape[0],
        'min_days': df['days_to_expiry'].min(),
        'max_days': df['days_to_expiry'].max()
    }

async def main():
    # Check first day of each month for 2024 and recent 2025
    dates_to_check = [
        "2024-01-01", "2024-02-01", "2024-03-01", "2024-04-01",
        "2024-05-01", "2024-06-01", "2024-07-01", "2024-08-01",
        "2024-09-01", "2024-10-01", "2024-11-01", "2024-12-01",
        "2025-01-01", "2025-02-01", "2025-03-01", "2025-04-01",
        "2025-05-01", "2025-06-01", "2025-07-01", "2025-08-01",
        "2025-09-01", "2025-10-01"
    ]

    print("=" * 90)
    print("CHECKING SHORT-DATED OPTIONS AVAILABILITY ON FIRST-OF-MONTH DATES")
    print("=" * 90)
    print(f"{'Date':<12} {'Total Options':<15} {'0-3d Calls':<12} {'Min Days':<10} {'Max Days':<10}")
    print("-" * 90)

    results = []
    for date in dates_to_check:
        try:
            result = await check_date_for_short_options(date)
            results.append(result)
            print(f"{result['date']:<12} {result['total_options']:<15} {result['short_dated_calls']:<12} {result['min_days']:<10.2f} {result['max_days']:<10.2f}")
        except Exception as e:
            print(f"{date:<12} ERROR: {str(e)[:60]}")

    print("\n" + "=" * 90)
    print("SUMMARY: Dates with short-dated (0-3 days) call options:")
    print("=" * 90)

    good_dates = [r for r in results if r['short_dated_calls'] > 0]
    if good_dates:
        for r in good_dates:
            print(f"  {r['date']}: {r['short_dated_calls']} call options with 0-3 days to expiry")
    else:
        print("  No dates found with 0-3 day options!")
        print("\n  Recommendation: Use minimum available days to expiry instead.")
        print("  Most dates have options starting from 7-14 days to expiry.")

if __name__ == "__main__":
    asyncio.run(main())
