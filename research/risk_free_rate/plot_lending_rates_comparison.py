#!/usr/bin/env python3
"""
Compare AAVE USDC and USDT lending rates over time.

This script generates time series plots comparing:
- Supply (lending) rates
- Borrow rates
- TVL/deposits
"""

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import polars as pl

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# File paths
DATA_DIR = Path(__file__).parent / "data"
AAVE_FILE = DATA_DIR / "aave_usdc_rates_2023_2025.parquet"
USDT_FILE = DATA_DIR / "usdt_lending_rates_2023_2025.parquet"
OUTPUT_DIR = Path(__file__).parent / "plots"


def load_data() -> tuple[pl.DataFrame, pl.DataFrame]:
    """Load AAVE USDC and USDT lending rate data.

    Returns:
        Tuple of (aave_df, usdt_df)
    """
    logger.info("Loading AAVE USDC rates...")
    aave_df = pl.read_parquet(AAVE_FILE)
    logger.info(
        f"Loaded {len(aave_df):,} AAVE records from {aave_df['timestamp'].min()} to {aave_df['timestamp'].max()}"
    )

    logger.info("Loading USDT lending rates...")
    usdt_df = pl.read_parquet(USDT_FILE)
    logger.info(f"Loaded {len(usdt_df):,} USDT records from {usdt_df['datetime'].min()} to {usdt_df['datetime'].max()}")

    return aave_df, usdt_df


def prepare_data(aave_df: pl.DataFrame, usdt_df: pl.DataFrame) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Prepare data for plotting by standardizing column names.

    Args:
        aave_df: AAVE USDC data
        usdt_df: USDT lending data

    Returns:
        Tuple of (aave_df, usdt_df) with standardized columns
    """
    # AAVE already has correct column names: timestamp, supply_rate_apr, borrow_rate_apr
    aave_prepared = aave_df.select(
        [
            pl.col("timestamp"),
            pl.col("supply_rate_apr").alias("supply_apr"),
            pl.col("borrow_rate_apr").alias("borrow_apr"),
            pl.col("total_value_locked_usd").alias("tvl_usd"),
        ]
    )

    # USDT needs mapping: datetime -> timestamp, lending_apr -> supply_apr, borrowing_apr -> borrow_apr
    usdt_prepared = usdt_df.select(
        [
            pl.col("datetime").alias("timestamp"),
            pl.col("lending_apr").alias("supply_apr"),
            pl.col("borrowing_apr").alias("borrow_apr"),
            pl.col("total_deposits_usd").alias("tvl_usd"),
        ]
    )

    return aave_prepared, usdt_prepared


def plot_comparison(aave_df: pl.DataFrame, usdt_df: pl.DataFrame, output_dir: Path) -> None:
    """Generate comparison plots.

    Args:
        aave_df: AAVE USDC data
        usdt_df: USDT lending data
        output_dir: Directory to save plots
    """
    output_dir.mkdir(exist_ok=True)

    # Convert to pandas for matplotlib (easier datetime handling)
    aave_pd = aave_df.to_pandas()
    usdt_pd = usdt_df.to_pandas()

    # Create figure with 3 subplots
    fig, axes = plt.subplots(3, 1, figsize=(14, 12), sharex=True)

    # Plot 1: Supply (Lending) Rates
    axes[0].plot(aave_pd["timestamp"], aave_pd["supply_apr"], label="AAVE USDC", linewidth=1.5, alpha=0.8)
    axes[0].plot(usdt_pd["timestamp"], usdt_pd["supply_apr"], label="USDT", linewidth=1.5, alpha=0.8)
    axes[0].set_ylabel("Supply APR (%)", fontsize=12)
    axes[0].set_title("Lending Rate Comparison: AAVE USDC vs USDT (2023-2025)", fontsize=14, fontweight="bold")
    axes[0].legend(loc="best", fontsize=10)
    axes[0].grid(True, alpha=0.3)

    # Plot 2: Borrow Rates
    axes[1].plot(aave_pd["timestamp"], aave_pd["borrow_apr"], label="AAVE USDC", linewidth=1.5, alpha=0.8)
    axes[1].plot(usdt_pd["timestamp"], usdt_pd["borrow_apr"], label="USDT", linewidth=1.5, alpha=0.8)
    axes[1].set_ylabel("Borrow APR (%)", fontsize=12)
    axes[1].set_title("Borrowing Rate Comparison", fontsize=14, fontweight="bold")
    axes[1].legend(loc="best", fontsize=10)
    axes[1].grid(True, alpha=0.3)

    # Plot 3: TVL/Total Deposits (in millions USD)
    axes[2].plot(aave_pd["timestamp"], aave_pd["tvl_usd"] / 1e6, label="AAVE USDC TVL", linewidth=1.5, alpha=0.8)
    axes[2].plot(usdt_pd["timestamp"], usdt_pd["tvl_usd"] / 1e6, label="USDT Deposits", linewidth=1.5, alpha=0.8)
    axes[2].set_ylabel("USD (Millions)", fontsize=12)
    axes[2].set_xlabel("Date", fontsize=12)
    axes[2].set_title("Total Value Locked / Deposits Comparison", fontsize=14, fontweight="bold")
    axes[2].legend(loc="best", fontsize=10)
    axes[2].grid(True, alpha=0.3)

    # Format x-axis
    fig.autofmt_xdate()

    plt.tight_layout()

    # Save plot
    output_file = output_dir / "lending_rates_comparison.png"
    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    logger.info(f"Saved comparison plot to {output_file}")

    plt.close()

    # Create individual plots for better detail
    _plot_individual_metrics(aave_pd, usdt_pd, output_dir)


def _plot_individual_metrics(aave_pd, usdt_pd, output_dir: Path) -> None:
    """Generate individual plots for each metric.

    Args:
        aave_pd: AAVE USDC data (pandas)
        usdt_pd: USDT lending data (pandas)
        output_dir: Directory to save plots
    """
    # Supply rates only
    plt.figure(figsize=(14, 6))
    plt.plot(aave_pd["timestamp"], aave_pd["supply_apr"], label="AAVE USDC", linewidth=1.5, alpha=0.8)
    plt.plot(usdt_pd["timestamp"], usdt_pd["supply_apr"], label="USDT", linewidth=1.5, alpha=0.8)
    plt.ylabel("Supply APR (%)", fontsize=12)
    plt.xlabel("Date", fontsize=12)
    plt.title("Supply (Lending) Rates: AAVE USDC vs USDT", fontsize=14, fontweight="bold")
    plt.legend(loc="best", fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.gcf().autofmt_xdate()
    plt.tight_layout()
    plt.savefig(output_dir / "supply_rates_comparison.png", dpi=300, bbox_inches="tight")
    plt.close()

    # Borrow rates only
    plt.figure(figsize=(14, 6))
    plt.plot(aave_pd["timestamp"], aave_pd["borrow_apr"], label="AAVE USDC", linewidth=1.5, alpha=0.8)
    plt.plot(usdt_pd["timestamp"], usdt_pd["borrow_apr"], label="USDT", linewidth=1.5, alpha=0.8)
    plt.ylabel("Borrow APR (%)", fontsize=12)
    plt.xlabel("Date", fontsize=12)
    plt.title("Borrowing Rates: AAVE USDC vs USDT", fontsize=14, fontweight="bold")
    plt.legend(loc="best", fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.gcf().autofmt_xdate()
    plt.tight_layout()
    plt.savefig(output_dir / "borrow_rates_comparison.png", dpi=300, bbox_inches="tight")
    plt.close()

    # TVL only
    plt.figure(figsize=(14, 6))
    plt.plot(aave_pd["timestamp"], aave_pd["tvl_usd"] / 1e6, label="AAVE USDC TVL", linewidth=1.5, alpha=0.8)
    plt.plot(usdt_pd["timestamp"], usdt_pd["tvl_usd"] / 1e6, label="USDT Deposits", linewidth=1.5, alpha=0.8)
    plt.ylabel("USD (Millions)", fontsize=12)
    plt.xlabel("Date", fontsize=12)
    plt.title("Total Value Locked: AAVE USDC vs USDT Deposits", fontsize=14, fontweight="bold")
    plt.legend(loc="best", fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.gcf().autofmt_xdate()
    plt.tight_layout()
    plt.savefig(output_dir / "tvl_comparison.png", dpi=300, bbox_inches="tight")
    plt.close()

    logger.info("Saved individual metric plots")


def generate_summary_stats(aave_df: pl.DataFrame, usdt_df: pl.DataFrame) -> None:
    """Print summary statistics for both datasets.

    Args:
        aave_df: AAVE USDC data
        usdt_df: USDT lending data
    """
    logger.info("\n" + "=" * 80)
    logger.info("SUMMARY STATISTICS")
    logger.info("=" * 80)

    # AAVE USDC stats
    aave_stats = aave_df.select(
        [
            pl.col("supply_apr").mean().alias("mean_supply_apr"),
            pl.col("supply_apr").median().alias("median_supply_apr"),
            pl.col("supply_apr").std().alias("std_supply_apr"),
            pl.col("borrow_apr").mean().alias("mean_borrow_apr"),
            pl.col("borrow_apr").median().alias("median_borrow_apr"),
            pl.col("borrow_apr").std().alias("std_borrow_apr"),
            pl.col("tvl_usd").mean().alias("mean_tvl_usd"),
        ]
    )

    logger.info("\nAAVE USDC:")
    logger.info(
        f"  Supply APR: {aave_stats['mean_supply_apr'][0]:.2f}% (mean), {aave_stats['median_supply_apr'][0]:.2f}% (median), {aave_stats['std_supply_apr'][0]:.2f}% (std)"
    )
    logger.info(
        f"  Borrow APR: {aave_stats['mean_borrow_apr'][0]:.2f}% (mean), {aave_stats['median_borrow_apr'][0]:.2f}% (median), {aave_stats['std_borrow_apr'][0]:.2f}% (std)"
    )
    logger.info(f"  Mean TVL: ${aave_stats['mean_tvl_usd'][0] / 1e6:.2f}M")

    # USDT stats
    usdt_stats = usdt_df.select(
        [
            pl.col("supply_apr").mean().alias("mean_supply_apr"),
            pl.col("supply_apr").median().alias("median_supply_apr"),
            pl.col("supply_apr").std().alias("std_supply_apr"),
            pl.col("borrow_apr").mean().alias("mean_borrow_apr"),
            pl.col("borrow_apr").median().alias("median_borrow_apr"),
            pl.col("borrow_apr").std().alias("std_borrow_apr"),
            pl.col("tvl_usd").mean().alias("mean_tvl_usd"),
        ]
    )

    logger.info("\nUSDT:")
    logger.info(
        f"  Supply APR: {usdt_stats['mean_supply_apr'][0]:.2f}% (mean), {usdt_stats['median_supply_apr'][0]:.2f}% (median), {usdt_stats['std_supply_apr'][0]:.2f}% (std)"
    )
    logger.info(
        f"  Borrow APR: {usdt_stats['mean_borrow_apr'][0]:.2f}% (mean), {usdt_stats['median_borrow_apr'][0]:.2f}% (median), {usdt_stats['std_borrow_apr'][0]:.2f}% (std)"
    )
    logger.info(f"  Mean Deposits: ${usdt_stats['mean_tvl_usd'][0] / 1e6:.2f}M")

    logger.info("=" * 80)


def main() -> None:
    """Main entry point."""
    logger.info("Starting lending rate comparison...")

    # Load data
    aave_df, usdt_df = load_data()

    # Prepare data (standardize column names)
    aave_prepared, usdt_prepared = prepare_data(aave_df, usdt_df)

    # Generate plots
    plot_comparison(aave_prepared, usdt_prepared, OUTPUT_DIR)

    # Print summary stats
    generate_summary_stats(aave_prepared, usdt_prepared)

    logger.info("\n✅ Comparison complete!")
    logger.info(f"Plots saved to {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
