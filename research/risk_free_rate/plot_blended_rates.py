#!/usr/bin/env python3
"""
Plot blended lending rates with original series.

This script generates time series plots showing:
- Original AAVE USDC supply rate
- Original USDT supply rate
- Blended supply rate (50/50 average)
- Blended supply rate (7-day moving average)
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
INPUT_FILE = DATA_DIR / "blended_lending_rates_2023_2025.parquet"
OUTPUT_DIR = Path(__file__).parent / "plots"


def load_data() -> pl.DataFrame:
    """Load blended lending rates data.

    Returns:
        DataFrame with blended rates and MA
    """
    logger.info(f"Loading blended rates from {INPUT_FILE}...")
    df = pl.read_parquet(INPUT_FILE)
    logger.info(f"Loaded {len(df):,} records from {df['date'].min()} to {df['date'].max()}")
    return df


def plot_blended_rates(df: pl.DataFrame, output_dir: Path) -> None:
    """Generate time series plot showing all rate series.

    Args:
        df: DataFrame with blended rates
        output_dir: Directory to save plots
    """
    output_dir.mkdir(exist_ok=True)

    # Convert to pandas for matplotlib
    df_pd = df.to_pandas()

    # Create main comparison plot
    fig, ax = plt.subplots(figsize=(16, 8))

    # Plot all 4 series
    ax.plot(
        df_pd["timestamp"],
        df_pd["aave_supply_apr"],
        label="AAVE USDC Supply APR",
        linewidth=1.2,
        alpha=0.6,
        color="#1f77b4",
    )

    ax.plot(
        df_pd["timestamp"],
        df_pd["usdt_supply_apr"],
        label="USDT Supply APR",
        linewidth=1.2,
        alpha=0.6,
        color="#ff7f0e",
    )

    ax.plot(
        df_pd["timestamp"],
        df_pd["blended_supply_apr"],
        label="Blended Supply APR (50/50)",
        linewidth=1.5,
        alpha=0.8,
        color="#2ca02c",
        linestyle="--",
    )

    ax.plot(
        df_pd["timestamp"],
        df_pd["blended_supply_apr_ma7"],
        label="Blended Supply APR (7-day MA)",
        linewidth=2.5,
        alpha=1.0,
        color="#d62728",
    )

    ax.set_ylabel("Supply APR (%)", fontsize=13)
    ax.set_xlabel("Date", fontsize=13)
    ax.set_title(
        "Blended Lending Rates: AAVE USDC + USDT (50/50 Average with 7-Day MA)",
        fontsize=15,
        fontweight="bold",
        pad=20,
    )
    ax.legend(loc="best", fontsize=11, framealpha=0.95)
    ax.grid(True, alpha=0.3, linestyle=":")

    fig.autofmt_xdate()
    plt.tight_layout()

    # Save plot
    output_file = output_dir / "blended_rates_comparison.png"
    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    logger.info(f"Saved blended rates plot to {output_file}")

    plt.close()

    # Create additional plot: Focus on blended vs MA
    _plot_blended_vs_ma(df_pd, output_dir)


def _plot_blended_vs_ma(df_pd, output_dir: Path) -> None:
    """Generate plot focusing on blended rate vs smoothed version.

    Args:
        df_pd: DataFrame (pandas) with blended rates
        output_dir: Directory to save plots
    """
    fig, ax = plt.subplots(figsize=(16, 8))

    # Plot blended rate and MA only
    ax.plot(
        df_pd["timestamp"],
        df_pd["blended_supply_apr"],
        label="Blended Supply APR (raw)",
        linewidth=1.5,
        alpha=0.6,
        color="#2ca02c",
    )

    ax.plot(
        df_pd["timestamp"],
        df_pd["blended_supply_apr_ma7"],
        label="Blended Supply APR (7-day MA)",
        linewidth=2.5,
        alpha=1.0,
        color="#d62728",
    )

    ax.set_ylabel("Supply APR (%)", fontsize=13)
    ax.set_xlabel("Date", fontsize=13)
    ax.set_title("Blended Lending Rate: Raw vs 7-Day Moving Average", fontsize=15, fontweight="bold", pad=20)
    ax.legend(loc="best", fontsize=11, framealpha=0.95)
    ax.grid(True, alpha=0.3, linestyle=":")

    fig.autofmt_xdate()
    plt.tight_layout()

    # Save plot
    output_file = output_dir / "blended_rate_smoothing.png"
    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    logger.info(f"Saved blended rate smoothing plot to {output_file}")

    plt.close()


def main() -> None:
    """Main entry point."""
    logger.info("Starting blended rates plotting...")

    # Load data
    df = load_data()

    # Generate plots
    plot_blended_rates(df, OUTPUT_DIR)

    logger.info("\n✅ Plotting complete!")
    logger.info(f"Plots saved to {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
