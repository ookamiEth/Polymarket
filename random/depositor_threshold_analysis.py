#!/usr/bin/env python3
"""
Analyze unique depositors with specific thresholds.

Min threshold: $2,650 USDT
Max threshold: $186,282 USDT
"""

from pathlib import Path

import polars as pl

from etherscan_api_client import TARGET_ADDRESS

# Directories
BASE_DIR = Path("/Users/lgierhake/Documents/ETH/BT/random/data/etherscan")
RAW_DIR = BASE_DIR / "raw"
SHORT_ADDR = TARGET_ADDRESS[:10]

# Thresholds
MIN_THRESHOLD = 2650.0
MAX_THRESHOLD = 186282.0


def main() -> None:
    """Analyze depositor-level metrics with thresholds."""
    # Load deposit data
    deposit_file = RAW_DIR / f"usdt_deposits_{SHORT_ADDR}.parquet"
    df = pl.read_parquet(deposit_file)

    print()
    print("=" * 80)
    print("UNIQUE DEPOSITOR ANALYSIS")
    print(f"Min Threshold: ${MIN_THRESHOLD:,.2f} | Max Threshold: ${MAX_THRESHOLD:,.2f}")
    print("=" * 80)
    print()

    # Calculate total deposited per unique address
    depositor_totals = (
        df.group_by("from")
        .agg([
            pl.col("TokenValue").sum().alias("total_deposited"),
            pl.len().alias("num_deposits"),
            pl.col("TokenValue").max().alias("largest_single_deposit"),
            pl.col("TokenValue").min().alias("smallest_single_deposit"),
        ])
    )

    total_unique_depositors = len(depositor_totals)

    print("📊 UNIQUE DEPOSITOR METRICS")
    print("-" * 80)
    print(f"Total Unique Depositors: {total_unique_depositors:,}")
    print()

    # Calculate average and median per depositor (based on TOTAL deposited by each address)
    avg_per_depositor = depositor_totals["total_deposited"].mean()
    median_per_depositor = depositor_totals["total_deposited"].median()

    print("Average & Median (based on total deposited per unique address):")
    print(f"  Average Deposit per Depositor: ${avg_per_depositor:,.2f} USDT")
    print(f"  Median Deposit per Depositor:  ${median_per_depositor:,.2f} USDT")
    print()

    # ========================================================================
    # MAXIMUM THRESHOLD ANALYSIS ($186,282)
    # ========================================================================
    print("🔝 MAXIMUM THRESHOLD ANALYSIS ($186,282 USDT)")
    print("-" * 80)

    # Depositors whose TOTAL deposited equals max threshold
    exact_max_total = depositor_totals.filter(pl.col("total_deposited") == MAX_THRESHOLD)
    num_exact_max_total = len(exact_max_total)

    # Depositors whose TOTAL deposited is >= max threshold
    at_least_max_total = depositor_totals.filter(pl.col("total_deposited") >= MAX_THRESHOLD)
    num_at_least_max_total = len(at_least_max_total)

    # Depositors who had at least ONE deposit of max amount
    had_max_deposit = depositor_totals.filter(pl.col("largest_single_deposit") == MAX_THRESHOLD)
    num_had_max_deposit = len(had_max_deposit)

    print(f"Depositors with TOTAL = ${MAX_THRESHOLD:,.2f}:        {num_exact_max_total:,} depositors")
    print(f"Depositors with TOTAL >= ${MAX_THRESHOLD:,.2f}:       {num_at_least_max_total:,} depositors")
    print(f"Depositors with ANY deposit = ${MAX_THRESHOLD:,.2f}:  {num_had_max_deposit:,} depositors")
    print()

    # Show the depositors who deposited MORE than max threshold
    above_max = depositor_totals.filter(pl.col("total_deposited") > MAX_THRESHOLD).sort(
        "total_deposited", descending=True
    )
    if len(above_max) > 0:
        print(f"Depositors who deposited MORE than ${MAX_THRESHOLD:,.2f}:")
        for row in above_max.head(10).iter_rows(named=True):
            addr = row["from"][:10] + "..." + row["from"][-8:]
            total = row["total_deposited"]
            num = row["num_deposits"]
            print(f"  {addr}: ${total:,.2f} ({num} deposits)")
    else:
        print(f"No depositors exceeded ${MAX_THRESHOLD:,.2f}")
    print()

    # ========================================================================
    # MINIMUM THRESHOLD ANALYSIS ($2,650)
    # ========================================================================
    print("🔻 MINIMUM THRESHOLD ANALYSIS ($2,650 USDT)")
    print("-" * 80)

    # Depositors whose TOTAL deposited equals min threshold
    exact_min_total = depositor_totals.filter(pl.col("total_deposited") == MIN_THRESHOLD)
    num_exact_min_total = len(exact_min_total)

    # Depositors whose TOTAL deposited is <= min threshold
    at_most_min_total = depositor_totals.filter(pl.col("total_deposited") <= MIN_THRESHOLD)
    num_at_most_min_total = len(at_most_min_total)

    # Depositors whose TOTAL deposited is >= min threshold
    at_least_min_total = depositor_totals.filter(pl.col("total_deposited") >= MIN_THRESHOLD)
    num_at_least_min_total = len(at_least_min_total)

    # Depositors who had at least ONE deposit of min amount
    had_min_deposit = depositor_totals.filter(pl.col("largest_single_deposit") == MIN_THRESHOLD)
    num_had_min_deposit = len(had_min_deposit)

    print(f"Depositors with TOTAL = ${MIN_THRESHOLD:,.2f}:        {num_exact_min_total:,} depositors")
    print(f"Depositors with TOTAL <= ${MIN_THRESHOLD:,.2f}:       {num_at_most_min_total:,} depositors")
    print(f"Depositors with TOTAL >= ${MIN_THRESHOLD:,.2f}:       {num_at_least_min_total:,} depositors")
    print(f"Depositors with ANY deposit = ${MIN_THRESHOLD:,.2f}:  {num_had_min_deposit:,} depositors")
    print()

    # Show the depositors who deposited LESS than min threshold
    below_min = depositor_totals.filter(pl.col("total_deposited") < MIN_THRESHOLD).sort(
        "total_deposited", descending=True
    )
    if len(below_min) > 0:
        print(f"Top 10 depositors who deposited LESS than ${MIN_THRESHOLD:,.2f}:")
        for row in below_min.head(10).iter_rows(named=True):
            addr = row["from"][:10] + "..." + row["from"][-8:]
            total = row["total_deposited"]
            num = row["num_deposits"]
            print(f"  {addr}: ${total:,.2f} ({num} deposits)")
        print()
        print(f"Total depositors below ${MIN_THRESHOLD:,.2f}: {len(below_min):,}")
    print()

    # ========================================================================
    # DISTRIBUTION BY THRESHOLD RANGES
    # ========================================================================
    print("📈 DISTRIBUTION BY RANGES")
    print("-" * 80)

    below_min = depositor_totals.filter(pl.col("total_deposited") < MIN_THRESHOLD)
    between = depositor_totals.filter(
        (pl.col("total_deposited") >= MIN_THRESHOLD) & (pl.col("total_deposited") < MAX_THRESHOLD)
    )
    at_max_or_above = depositor_totals.filter(pl.col("total_deposited") >= MAX_THRESHOLD)

    num_below_min = len(below_min)
    num_between = len(between)
    num_at_max_or_above = len(at_max_or_above)

    total_volume_below_min = below_min["total_deposited"].sum() if num_below_min > 0 else 0
    total_volume_between = between["total_deposited"].sum() if num_between > 0 else 0
    total_volume_at_max_or_above = (
        at_max_or_above["total_deposited"].sum() if num_at_max_or_above > 0 else 0
    )

    total_volume = depositor_totals["total_deposited"].sum()

    print(f"Below ${MIN_THRESHOLD:,.2f}:")
    print(f"  Depositors: {num_below_min:,} ({num_below_min/total_unique_depositors*100:.1f}%)")
    print(f"  Volume: ${total_volume_below_min:,.2f} ({total_volume_below_min/total_volume*100:.1f}%)")
    print()

    print(f"Between ${MIN_THRESHOLD:,.2f} - ${MAX_THRESHOLD:,.2f}:")
    print(f"  Depositors: {num_between:,} ({num_between/total_unique_depositors*100:.1f}%)")
    print(f"  Volume: ${total_volume_between:,.2f} ({total_volume_between/total_volume*100:.1f}%)")
    print()

    print(f"At or Above ${MAX_THRESHOLD:,.2f}:")
    print(f"  Depositors: {num_at_max_or_above:,} ({num_at_max_or_above/total_unique_depositors*100:.1f}%)")
    print(
        f"  Volume: ${total_volume_at_max_or_above:,.2f} ({total_volume_at_max_or_above/total_volume*100:.1f}%)"
    )
    print()

    # ========================================================================
    # ACTUAL MIN AND MAX IN DATASET
    # ========================================================================
    print("📊 ACTUAL MIN/MAX IN DATASET")
    print("-" * 80)

    actual_min = depositor_totals["total_deposited"].min()
    actual_max = depositor_totals["total_deposited"].max()

    min_depositor = depositor_totals.filter(pl.col("total_deposited") == actual_min)
    max_depositor = depositor_totals.filter(pl.col("total_deposited") == actual_max)

    print(f"Actual Minimum Total per Depositor: ${actual_min:,.2f}")
    if len(min_depositor) > 0:
        print(f"  Number of depositors with this amount: {len(min_depositor)}")
        if len(min_depositor) <= 5:
            for row in min_depositor.iter_rows(named=True):
                addr = row["from"][:10] + "..." + row["from"][-8:]
                print(f"    {addr}: ${row['total_deposited']:,.2f} ({row['num_deposits']} deposits)")
    print()

    print(f"Actual Maximum Total per Depositor: ${actual_max:,.2f}")
    if len(max_depositor) > 0:
        print(f"  Number of depositors with this amount: {len(max_depositor)}")
        for row in max_depositor.head(5).iter_rows(named=True):
            addr = row["from"][:10] + "..." + row["from"][-8:]
            print(f"    {addr}: ${row['total_deposited']:,.2f} ({row['num_deposits']} deposits)")
    print()

    # ========================================================================
    # SUMMARY
    # ========================================================================
    print("=" * 80)
    print("📋 SUMMARY")
    print("=" * 80)
    print()
    print(f"Total Unique Depositors: {total_unique_depositors:,}")
    print(f"Average per Depositor: ${avg_per_depositor:,.2f} USDT")
    print(f"Median per Depositor: ${median_per_depositor:,.2f} USDT")
    print()
    print(f"Depositors who deposited exactly ${MAX_THRESHOLD:,.2f}: {num_exact_max_total:,}")
    print(f"Depositors who deposited exactly ${MIN_THRESHOLD:,.2f}: {num_exact_min_total:,}")
    print()
    print(f"Depositors >= ${MAX_THRESHOLD:,.2f}: {num_at_least_max_total:,}")
    print(f"Depositors <= ${MIN_THRESHOLD:,.2f}: {num_at_most_min_total:,}")
    print()
    print("=" * 80)


if __name__ == "__main__":
    main()
