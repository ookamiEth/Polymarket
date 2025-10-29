#!/usr/bin/env python3
"""
Analyze Ethereum address deposit transactions from Etherscan API data.

Replicates analyze_deposits.py functionality with enhancements:
- Gas cost analysis
- Temporal patterns (hourly/daily distribution)
- Depositor behavior analysis
- Transaction success rates
"""

import logging
from datetime import datetime
from pathlib import Path

import polars as pl
from etherscan_api_client import TARGET_ADDRESS

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# Directories
BASE_DIR = Path("/Users/lgierhake/Documents/ETH/BT/random/data/etherscan")
RAW_DIR = BASE_DIR / "raw"
REPORTS_DIR = Path("/Users/lgierhake/Documents/ETH/BT/random/reports")

# Create reports directory
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

# Short address for filenames
SHORT_ADDR = TARGET_ADDRESS[:10]


def load_deposit_data() -> pl.DataFrame:
    """
    Load USDT deposit data from Parquet file.

    Returns:
        Polars DataFrame with deposit transactions
    """
    deposit_file = RAW_DIR / f"usdt_deposits_{SHORT_ADDR}.parquet"

    if not deposit_file.exists():
        raise FileNotFoundError(
            f"Deposit data not found: {deposit_file}\nRun collect_address_data.py first to collect the data."
        )

    logger.info(f"Loading deposit data from: {deposit_file}")
    df = pl.read_parquet(deposit_file)
    logger.info(f"Loaded {len(df):,} deposit transactions")

    return df


def analyze_basic_stats(df: pl.DataFrame) -> dict:
    """
    Calculate basic deposit statistics (matching original analyze_deposits.py).

    Args:
        df: DataFrame with deposit transactions

    Returns:
        Dictionary with basic statistics
    """
    # Unique depositors (using "from" column since these are deposits TO the address)
    unique_depositors = df["from"].n_unique()

    # Total deposits
    total_deposits = len(df)

    # Amount statistics
    stats = df.select(
        [
            pl.col("TokenValue").sum().alias("total"),
            pl.col("TokenValue").mean().alias("avg"),
            pl.col("TokenValue").median().alias("median"),
            pl.col("TokenValue").min().alias("min"),
            pl.col("TokenValue").max().alias("max"),
        ]
    )

    # Time range
    first_deposit = df["DateTime"].min()
    last_deposit = df["DateTime"].max()

    # Multi-depositors (addresses with >1 deposit)
    deposits_per_user = df.group_by("from").agg(pl.len().alias("count"))
    multi_depositors = deposits_per_user.filter(pl.col("count") > 1)
    num_multi_depositors = len(multi_depositors)

    # Top depositors
    top_depositors = (
        df.group_by("from")
        .agg(
            [
                pl.col("TokenValue").sum().alias("total_usdt"),
                pl.len().alias("num_deposits"),
            ]
        )
        .sort("total_usdt", descending=True)
        .head(20)
    )

    return {
        "unique_depositors": unique_depositors,
        "total_deposits": total_deposits,
        "multi_depositors": num_multi_depositors,
        "stats": stats,
        "first_deposit": first_deposit,
        "last_deposit": last_deposit,
        "top_depositors": top_depositors,
    }


def analyze_gas_costs(df: pl.DataFrame) -> dict:
    """
    Analyze gas costs for deposit transactions.

    Args:
        df: DataFrame with deposit transactions

    Returns:
        Dictionary with gas cost statistics
    """
    gas_stats = df.select(
        [
            pl.col("GasCostETH").sum().alias("total_gas_eth"),
            pl.col("GasCostETH").mean().alias("avg_gas_eth"),
            pl.col("GasCostETH").median().alias("median_gas_eth"),
            pl.col("GasCostETH").min().alias("min_gas_eth"),
            pl.col("GasCostETH").max().alias("max_gas_eth"),
        ]
    )

    # Assuming ETH = $3000 USD for USD conversion
    eth_price_usd = 3000.0

    total_gas_eth = gas_stats["total_gas_eth"][0]
    avg_gas_eth = gas_stats["avg_gas_eth"][0]
    median_gas_eth = gas_stats["median_gas_eth"][0]

    return {
        "total_gas_eth": total_gas_eth,
        "total_gas_usd": total_gas_eth * eth_price_usd,
        "avg_gas_eth": avg_gas_eth,
        "avg_gas_usd": avg_gas_eth * eth_price_usd,
        "median_gas_eth": median_gas_eth,
        "median_gas_usd": median_gas_eth * eth_price_usd,
        "eth_price_used": eth_price_usd,
    }


def analyze_temporal_patterns(df: pl.DataFrame) -> dict:
    """
    Analyze temporal patterns in deposits.

    Args:
        df: DataFrame with deposit transactions

    Returns:
        Dictionary with temporal statistics
    """
    # Add hour and day of week columns
    df_temporal = df.with_columns(
        [
            pl.col("DateTime").dt.hour().alias("hour"),
            pl.col("DateTime").dt.weekday().alias("weekday"),  # 0=Monday, 6=Sunday
            pl.col("DateTime").dt.date().alias("date"),
        ]
    )

    # Deposits by hour
    hourly_deposits = df_temporal.group_by("hour").agg(pl.len().alias("count")).sort("count", descending=True)

    # Deposits by day of week
    daily_deposits = df_temporal.group_by("weekday").agg(pl.len().alias("count")).sort("count", descending=True)

    # Map weekday numbers to names
    weekday_names = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]

    # Peak hour and day
    peak_hour = hourly_deposits["hour"][0]
    peak_hour_count = hourly_deposits["count"][0]

    peak_weekday = daily_deposits["weekday"][0]
    peak_weekday_count = daily_deposits["count"][0]
    peak_weekday_name = weekday_names[peak_weekday]

    # Deposits per day
    deposits_per_day = df_temporal.group_by("date").agg(pl.len().alias("count")).sort("count", descending=True)

    avg_deposits_per_day = deposits_per_day["count"].mean()
    max_deposits_per_day = deposits_per_day["count"].max()

    return {
        "peak_hour": peak_hour,
        "peak_hour_count": peak_hour_count,
        "peak_weekday": peak_weekday_name,
        "peak_weekday_count": peak_weekday_count,
        "avg_deposits_per_day": avg_deposits_per_day,
        "max_deposits_per_day": max_deposits_per_day,
        "hourly_distribution": hourly_deposits,
        "daily_distribution": daily_deposits,
    }


def analyze_depositor_behavior(df: pl.DataFrame) -> dict:
    """
    Analyze depositor behavior patterns.

    Args:
        df: DataFrame with deposit transactions

    Returns:
        Dictionary with depositor behavior statistics
    """
    # Group by depositor
    depositor_stats = df.group_by("from").agg(
        [
            pl.col("TokenValue").sum().alias("total_deposited"),
            pl.len().alias("num_deposits"),
            pl.col("DateTime").min().alias("first_deposit"),
            pl.col("DateTime").max().alias("last_deposit"),
        ]
    )

    # First-time vs repeat depositors
    first_time_depositors = depositor_stats.filter(pl.col("num_deposits") == 1)
    repeat_depositors = depositor_stats.filter(pl.col("num_deposits") > 1)

    num_first_time = len(first_time_depositors)
    num_repeat = len(repeat_depositors)

    # For repeat depositors, calculate time between deposits
    repeat_with_duration = repeat_depositors.with_columns(
        [((pl.col("last_deposit") - pl.col("first_deposit")).dt.total_seconds() / 86400).alias("days_active")]
    )

    # Average time between deposits for repeat users
    avg_days_active = repeat_with_duration["days_active"].mean()

    # Deposit size progression (are people depositing more over time?)
    # Compare first vs last deposit for repeat depositors
    # This would require matching first and last deposits by user, which is complex
    # For now, we'll skip this metric

    return {
        "num_first_time_depositors": num_first_time,
        "num_repeat_depositors": num_repeat,
        "pct_repeat": (num_repeat / (num_first_time + num_repeat)) * 100,
        "avg_days_active_repeat": avg_days_active if num_repeat > 0 else 0,
    }


def print_analysis_report(
    basic_stats: dict,
    gas_stats: dict,
    temporal_stats: dict,
    behavior_stats: dict,
) -> str:
    """
    Print comprehensive analysis report to console and return as markdown string.

    Args:
        basic_stats: Basic statistics dictionary
        gas_stats: Gas cost statistics dictionary
        temporal_stats: Temporal pattern statistics dictionary
        behavior_stats: Depositor behavior statistics dictionary

    Returns:
        Markdown-formatted report string
    """
    # Console output
    print()
    print("=" * 60)
    print("Ethereum Address Deposit Analysis (Etherscan API)")
    print(f"Address: {TARGET_ADDRESS}")
    print("=" * 60)
    print()

    # Basic statistics (matching original analyze_deposits.py)
    print(f"Unique Depositors: {basic_stats['unique_depositors']:,}")
    print(f"Total Deposits: {basic_stats['total_deposits']:,}")
    print(f"Multi-Depositors: {basic_stats['multi_depositors']:,}")
    print()

    stats = basic_stats["stats"]
    print(f"Total Deposited: ${stats['total'][0]:,.2f} USDT (≈USD)")
    print()
    print(f"Average Deposit: ${stats['avg'][0]:,.2f}")
    print(f"Median Deposit: ${stats['median'][0]:,.2f}")
    print(f"Min Deposit: ${stats['min'][0]:,.2f}")
    print(f"Max Deposit: ${stats['max'][0]:,.2f}")
    print()

    print("Time Range:")
    print(f"  First Deposit: {basic_stats['first_deposit']}")
    print(f"  Last Deposit:  {basic_stats['last_deposit']}")
    print()

    print("Top 20 Depositors by Amount:")
    print("-" * 60)
    for row in basic_stats["top_depositors"].iter_rows(named=True):
        addr = row["from"][:10] + "..." + row["from"][-8:]
        total = row["total_usdt"]
        count = row["num_deposits"]
        plural = "s" if count > 1 else ""
        print(f"  {addr}: ${total:,.2f} USDT ({count} deposit{plural})")

    print()

    # Enhanced analytics
    print("=" * 60)
    print("Enhanced Analytics")
    print("=" * 60)
    print()

    # Gas statistics
    print("Gas Cost Analysis:")
    print("-" * 60)
    print(f"  Total Gas Spent: {gas_stats['total_gas_eth']:.6f} ETH (${gas_stats['total_gas_usd']:,.2f} USD)")
    print(f"  Average Gas/Deposit: {gas_stats['avg_gas_eth']:.6f} ETH (${gas_stats['avg_gas_usd']:.2f} USD)")
    print(f"  Median Gas/Deposit: {gas_stats['median_gas_eth']:.6f} ETH (${gas_stats['median_gas_usd']:.2f} USD)")
    print(f"  (Assumed ETH Price: ${gas_stats['eth_price_used']:,.2f})")
    print()

    # Temporal patterns
    print("Temporal Patterns:")
    print("-" * 60)
    print(f"  Peak Deposit Hour: {temporal_stats['peak_hour']}:00 UTC ({temporal_stats['peak_hour_count']:,} deposits)")
    print(f"  Peak Deposit Day: {temporal_stats['peak_weekday']} ({temporal_stats['peak_weekday_count']:,} deposits)")
    print(f"  Avg Deposits/Day: {temporal_stats['avg_deposits_per_day']:.1f}")
    print(f"  Max Deposits/Day: {temporal_stats['max_deposits_per_day']:,}")
    print()

    # Depositor behavior
    print("Depositor Behavior:")
    print("-" * 60)
    print(
        f"  First-Time Depositors: {behavior_stats['num_first_time_depositors']:,} ({100 - behavior_stats['pct_repeat']:.1f}%)"
    )
    print(f"  Repeat Depositors: {behavior_stats['num_repeat_depositors']:,} ({behavior_stats['pct_repeat']:.1f}%)")
    if behavior_stats["avg_days_active_repeat"] > 0:
        print(f"  Avg Days Active (Repeat): {behavior_stats['avg_days_active_repeat']:.1f} days")
    print()

    print("=" * 60)

    # Generate markdown report
    md_report = f"""# Ethereum Address Deposit Analysis

**Address**: `{TARGET_ADDRESS}`
**Analysis Date**: {datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")} UTC
**Data Source**: Etherscan API

---

## Summary Statistics

| Metric | Value |
|--------|-------|
| Unique Depositors | {basic_stats["unique_depositors"]:,} |
| Total Deposits | {basic_stats["total_deposits"]:,} |
| Multi-Depositors | {basic_stats["multi_depositors"]:,} |
| **Total Deposited** | **${stats["total"][0]:,.2f} USDT** |

---

## Deposit Amount Statistics

| Statistic | Value |
|-----------|-------|
| Average Deposit | ${stats["avg"][0]:,.2f} USDT |
| Median Deposit | ${stats["median"][0]:,.2f} USDT |
| Min Deposit | ${stats["min"][0]:,.2f} USDT |
| Max Deposit | ${stats["max"][0]:,.2f} USDT |

**Time Range**:
- First Deposit: `{basic_stats["first_deposit"]}`
- Last Deposit: `{basic_stats["last_deposit"]}`

---

## Top 20 Depositors

| Address | Total Deposited | Number of Deposits |
|---------|-----------------|-------------------|
"""

    # Add top depositors to markdown
    for row in basic_stats["top_depositors"].iter_rows(named=True):
        addr = row["from"][:10] + "..." + row["from"][-8:]
        total = row["total_usdt"]
        count = row["num_deposits"]
        md_report += f"| `{addr}` | ${total:,.2f} | {count} |\n"

    md_report += f"""
---

## Gas Cost Analysis

| Metric | ETH | USD (@ $3,000/ETH) |
|--------|-----|---------------------|
| Total Gas Spent | {gas_stats["total_gas_eth"]:.6f} ETH | ${gas_stats["total_gas_usd"]:,.2f} |
| Average Gas per Deposit | {gas_stats["avg_gas_eth"]:.6f} ETH | ${gas_stats["avg_gas_usd"]:.2f} |
| Median Gas per Deposit | {gas_stats["median_gas_eth"]:.6f} ETH | ${gas_stats["median_gas_usd"]:.2f} |

---

## Temporal Patterns

| Pattern | Value |
|---------|-------|
| Peak Deposit Hour | {temporal_stats["peak_hour"]}:00 UTC ({temporal_stats["peak_hour_count"]:,} deposits) |
| Peak Deposit Day | {temporal_stats["peak_weekday"]} ({temporal_stats["peak_weekday_count"]:,} deposits) |
| Average Deposits per Day | {temporal_stats["avg_deposits_per_day"]:.1f} |
| Maximum Deposits in One Day | {temporal_stats["max_deposits_per_day"]:,} |

---

## Depositor Behavior

| Behavior Type | Count | Percentage |
|---------------|-------|------------|
| First-Time Depositors | {behavior_stats["num_first_time_depositors"]:,} | {100 - behavior_stats["pct_repeat"]:.1f}% |
| Repeat Depositors | {behavior_stats["num_repeat_depositors"]:,} | {behavior_stats["pct_repeat"]:.1f}% |

**Average Days Active (Repeat Depositors)**: {behavior_stats["avg_days_active_repeat"]:.1f} days

---

## Data Files

- Raw Data: `data/etherscan/raw/usdt_deposits_{SHORT_ADDR}.parquet`
- Report: `reports/address_{SHORT_ADDR}_analysis.md`

---

*Generated by analyze_address_deposits.py using Etherscan API data*
"""

    return md_report


def save_markdown_report(report: str) -> None:
    """
    Save analysis report as markdown file.

    Args:
        report: Markdown-formatted report string
    """
    output_file = REPORTS_DIR / f"address_{SHORT_ADDR}_analysis.md"
    output_file.write_text(report)
    logger.info(f"✅ Saved markdown report to: {output_file}")


def main() -> None:
    """Main entry point for deposit analysis."""
    logger.info("")
    logger.info("╔" + "=" * 58 + "╗")
    logger.info("║" + " " * 58 + "║")
    logger.info("║  Ethereum Address Deposit Analysis" + " " * 23 + "║")
    logger.info("║" + " " * 58 + "║")
    logger.info("╚" + "=" * 58 + "╝")
    logger.info("")

    try:
        # Load deposit data
        df = load_deposit_data()

        # Run all analyses
        logger.info("Running analyses...")
        basic_stats = analyze_basic_stats(df)
        gas_stats = analyze_gas_costs(df)
        temporal_stats = analyze_temporal_patterns(df)
        behavior_stats = analyze_depositor_behavior(df)

        # Print and generate report
        md_report = print_analysis_report(
            basic_stats,
            gas_stats,
            temporal_stats,
            behavior_stats,
        )

        # Save markdown report
        save_markdown_report(md_report)

        logger.info("")
        logger.info("Analysis complete!")
        logger.info("")

    except FileNotFoundError as e:
        logger.error(str(e))
        return

    except Exception as e:
        logger.error(f"Analysis failed: {e}")
        raise


if __name__ == "__main__":
    main()
