#!/usr/bin/env python3
"""
Analyze MegaETH Public Sale deposit transactions.

Calculates deposit statistics including unique depositors, average/median deposits,
and distribution metrics.
"""

import polars as pl


def analyze_deposits(parquet_path: str) -> None:
    """
    Analyze deposit transactions from Parquet file.

    Args:
        parquet_path: Path to the megaeth_deposits.parquet file
    """
    # Read Parquet file with Polars
    df = pl.read_parquet(parquet_path)

    print(f"Loaded {len(df):,} transactions from Parquet file")
    print()

    # Calculate unique depositors (using "From" column)
    unique_depositors = df["From"].n_unique()

    # Total deposits (transaction count)
    total_deposits = len(df)

    # Deposit amount statistics (using TokenValue which is USDT ≈ USD)
    total_deposited = df["TokenValue"].sum()
    avg_deposit = df["TokenValue"].mean()
    median_deposit = df["TokenValue"].median()
    min_deposit = df["TokenValue"].min()
    max_deposit = df["TokenValue"].max()

    # Time range (using parsed DateTime column)
    first_deposit = df["DateTime"].min()
    last_deposit = df["DateTime"].max()

    # Deposits per user distribution
    deposits_per_user = df.group_by("From").agg(pl.len().alias("deposit_count")).sort("deposit_count", descending=True)
    multi_depositors = deposits_per_user.filter(pl.col("deposit_count") > 1)
    num_multi_depositors = len(multi_depositors)

    # Print formatted results
    print("=" * 60)
    print("MegaETH Public Sale Deposit Analysis")
    print("=" * 60)
    print()
    print(f"Unique Depositors: {unique_depositors:,}")
    print(f"Total Deposits: {total_deposits:,}")
    print(f"Multi-Depositors: {num_multi_depositors:,}")
    print()
    print(f"Total Deposited: ${total_deposited:,.2f} USDT (≈USD)")
    print()
    print(f"Average Deposit: ${avg_deposit:,.2f}")
    print(f"Median Deposit: ${median_deposit:,.2f}")
    print(f"Min Deposit: ${min_deposit:,.2f}")
    print(f"Max Deposit: ${max_deposit:,.2f}")
    print()
    print("Time Range:")
    print(f"  First Deposit: {first_deposit}")
    print(f"  Last Deposit:  {last_deposit}")
    print()

    # Show top depositors
    top_depositors = (
        df.group_by("From")
        .agg(
            pl.col("TokenValue").sum().alias("total_usdt"),
            pl.len().alias("num_deposits"),
        )
        .sort("total_usdt", descending=True)
        .head(20)
    )

    print("Top 20 Depositors by Amount:")
    print("-" * 60)
    for row in top_depositors.iter_rows(named=True):
        addr = row["From"][:10] + "..." + row["From"][-8:]
        total = row["total_usdt"]
        count = row["num_deposits"]
        print(f"  {addr}: ${total:,.2f} USDT ({count} deposit{'s' if count > 1 else ''})")

    print()
    print("=" * 60)


def main() -> None:
    """Entry point for the script."""
    parquet_path = "/Users/lgierhake/Downloads/megaeth_deposits.parquet"
    analyze_deposits(parquet_path)


if __name__ == "__main__":
    main()
