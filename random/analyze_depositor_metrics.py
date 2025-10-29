#!/usr/bin/env python3
"""
Analyze depositor-level metrics (not transaction-level).

Compares:
1. Transaction-level metrics (current approach)
2. Depositor-level metrics (per unique address)
"""

from pathlib import Path

import polars as pl

from etherscan_api_client import TARGET_ADDRESS

# Directories
BASE_DIR = Path("/Users/lgierhake/Documents/ETH/BT/random/data/etherscan")
RAW_DIR = BASE_DIR / "raw"
SHORT_ADDR = TARGET_ADDRESS[:10]


def main() -> None:
    """Analyze depositor-level vs transaction-level metrics."""
    # Load deposit data
    deposit_file = RAW_DIR / f"usdt_deposits_{SHORT_ADDR}.parquet"
    df = pl.read_parquet(deposit_file)

    print()
    print("=" * 80)
    print("DEPOSITOR METRICS ANALYSIS")
    print("=" * 80)
    print()

    # Basic counts
    total_transactions = len(df)
    unique_depositors = df["from"].n_unique()

    print("📊 Basic Counts:")
    print("-" * 80)
    print(f"  Total Transactions:  {total_transactions:,}")
    print(f"  Unique Depositors:   {unique_depositors:,}")
    print(f"  Avg Txs per Depositor: {total_transactions / unique_depositors:.2f}")
    print()

    # ========================================================================
    # TRANSACTION-LEVEL METRICS (current approach)
    # ========================================================================
    print("📈 TRANSACTION-LEVEL METRICS (Current Approach)")
    print("-" * 80)
    print("  → Treats each transaction separately")
    print("  → If address 0x123 deposited 3 times, counts all 3 transactions")
    print()

    tx_stats = df.select([
        pl.col("TokenValue").sum().alias("total"),
        pl.col("TokenValue").mean().alias("avg"),
        pl.col("TokenValue").median().alias("median"),
        pl.col("TokenValue").min().alias("min"),
        pl.col("TokenValue").max().alias("max"),
    ])

    print(f"  Total Deposited:     ${tx_stats['total'][0]:,.2f} USDT")
    print(f"  Average per Tx:      ${tx_stats['avg'][0]:,.2f} USDT")
    print(f"  Median per Tx:       ${tx_stats['median'][0]:,.2f} USDT")
    print(f"  Min Tx:              ${tx_stats['min'][0]:,.2f} USDT")
    print(f"  Max Tx:              ${tx_stats['max'][0]:,.2f} USDT")
    print()

    # ========================================================================
    # DEPOSITOR-LEVEL METRICS (per unique address)
    # ========================================================================
    print("👤 DEPOSITOR-LEVEL METRICS (Per Unique Address)")
    print("-" * 80)
    print("  → Groups all deposits by the same address")
    print("  → If address 0x123 deposited 3 times, sums them into 1 total")
    print()

    # Group by depositor and calculate their total deposited
    depositor_totals = (
        df.group_by("from")
        .agg([
            pl.col("TokenValue").sum().alias("total_deposited"),
            pl.len().alias("num_deposits"),
        ])
        .with_columns([
            # Average deposit size for each depositor (if they made multiple deposits)
            (pl.col("total_deposited") / pl.col("num_deposits")).alias("avg_deposit_size")
        ])
    )

    # Calculate statistics on depositor totals
    depositor_stats = depositor_totals.select([
        pl.col("total_deposited").mean().alias("avg_total_per_depositor"),
        pl.col("total_deposited").median().alias("median_total_per_depositor"),
        pl.col("total_deposited").min().alias("min_total_per_depositor"),
        pl.col("total_deposited").max().alias("max_total_per_depositor"),
        pl.col("avg_deposit_size").mean().alias("avg_deposit_size_across_depositors"),
        pl.col("avg_deposit_size").median().alias("median_deposit_size_across_depositors"),
    ])

    print("  Metrics based on TOTAL deposited by each unique address:")
    print(f"    Avg Total per Depositor:    ${depositor_stats['avg_total_per_depositor'][0]:,.2f} USDT")
    print(f"    Median Total per Depositor: ${depositor_stats['median_total_per_depositor'][0]:,.2f} USDT")
    print(f"    Min Total per Depositor:    ${depositor_stats['min_total_per_depositor'][0]:,.2f} USDT")
    print(f"    Max Total per Depositor:    ${depositor_stats['max_total_per_depositor'][0]:,.2f} USDT")
    print()

    print("  Metrics based on AVERAGE deposit size per depositor:")
    print(
        f"    Avg of (each depositor's avg): ${depositor_stats['avg_deposit_size_across_depositors'][0]:,.2f} USDT"
    )
    print(
        f"    Median of (each depositor's avg): ${depositor_stats['median_deposit_size_across_depositors'][0]:,.2f} USDT"
    )
    print()

    # ========================================================================
    # DISTRIBUTION ANALYSIS
    # ========================================================================
    print("📊 DISTRIBUTION BREAKDOWN")
    print("-" * 80)

    # Single vs multi depositors
    single_deposit_depositors = depositor_totals.filter(pl.col("num_deposits") == 1)
    multi_deposit_depositors = depositor_totals.filter(pl.col("num_deposits") > 1)

    num_single = len(single_deposit_depositors)
    num_multi = len(multi_deposit_depositors)

    print(f"  Single-Deposit Addresses:  {num_single:,} ({num_single/unique_depositors*100:.1f}%)")
    print(f"  Multi-Deposit Addresses:   {num_multi:,} ({num_multi/unique_depositors*100:.1f}%)")
    print()

    # For single depositors
    if num_single > 0:
        single_stats = single_deposit_depositors.select([
            pl.col("total_deposited").mean().alias("avg"),
            pl.col("total_deposited").median().alias("median"),
        ])
        print("  Single-Deposit Depositors:")
        print(f"    Average deposit:  ${single_stats['avg'][0]:,.2f} USDT")
        print(f"    Median deposit:   ${single_stats['median'][0]:,.2f} USDT")
        print()

    # For multi depositors
    if num_multi > 0:
        multi_stats = multi_deposit_depositors.select([
            pl.col("total_deposited").mean().alias("avg_total"),
            pl.col("total_deposited").median().alias("median_total"),
            pl.col("avg_deposit_size").mean().alias("avg_size"),
            pl.col("avg_deposit_size").median().alias("median_size"),
            pl.col("num_deposits").mean().alias("avg_num_deposits"),
        ])
        print("  Multi-Deposit Depositors:")
        print(f"    Average TOTAL deposited:   ${multi_stats['avg_total'][0]:,.2f} USDT")
        print(f"    Median TOTAL deposited:    ${multi_stats['median_total'][0]:,.2f} USDT")
        print(f"    Average deposit SIZE:      ${multi_stats['avg_size'][0]:,.2f} USDT")
        print(f"    Median deposit SIZE:       ${multi_stats['median_size'][0]:,.2f} USDT")
        print(f"    Average # of deposits:     {multi_stats['avg_num_deposits'][0]:.1f}")
        print()

    # ========================================================================
    # TOP DEPOSITORS
    # ========================================================================
    print("🏆 TOP 10 DEPOSITORS (by total amount)")
    print("-" * 80)

    top_depositors = depositor_totals.sort("total_deposited", descending=True).head(10)

    for i, row in enumerate(top_depositors.iter_rows(named=True), 1):
        addr = row["from"][:10] + "..." + row["from"][-8:]
        total = row["total_deposited"]
        num = row["num_deposits"]
        avg = row["avg_deposit_size"]
        print(f"  {i:2d}. {addr}: ${total:,.2f} USDT ({num} deposits, avg ${avg:,.2f})")

    print()
    print("=" * 80)

    # ========================================================================
    # COMPARISON SUMMARY
    # ========================================================================
    print()
    print("📋 SUMMARY: Transaction-Level vs Depositor-Level")
    print("=" * 80)
    print()
    print("TRANSACTION-LEVEL (current approach):")
    print(f"  → Average per transaction: ${tx_stats['avg'][0]:,.2f}")
    print(f"  → Median per transaction:  ${tx_stats['median'][0]:,.2f}")
    print("  → Useful for: Understanding typical transaction sizes")
    print()
    print("DEPOSITOR-LEVEL (per unique address):")
    print(f"  → Average total per depositor: ${depositor_stats['avg_total_per_depositor'][0]:,.2f}")
    print(f"  → Median total per depositor:  ${depositor_stats['median_total_per_depositor'][0]:,.2f}")
    print("  → Useful for: Understanding how much each depositor contributed")
    print()
    print("=" * 80)


if __name__ == "__main__":
    main()
