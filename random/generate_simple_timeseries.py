#!/usr/bin/env python3
"""
Generate simple time series line graph PDF.

Single chart with dual y-axis:
- Primary axis: Daily deposit volume
- Secondary axis: Cumulative unique depositors
"""

from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import polars as pl
from matplotlib.backends.backend_pdf import PdfPages

from etherscan_api_client import TARGET_ADDRESS

# Directories
BASE_DIR = Path("/Users/lgierhake/Documents/ETH/BT/random/data/etherscan")
RAW_DIR = BASE_DIR / "raw"
REPORTS_DIR = Path("/Users/lgierhake/Documents/ETH/BT/random/reports")
SHORT_ADDR = TARGET_ADDRESS[:10]

# Ensure reports directory exists
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

# Plot styling
plt.style.use("dark_background")


def main() -> None:
    """Generate simple time series PDF."""
    print()
    print("=" * 80)
    print("GENERATING TIME SERIES LINE GRAPH PDF")
    print("=" * 80)
    print()

    # Load data
    print("Loading deposit data...")
    deposit_file = RAW_DIR / f"usdt_deposits_{SHORT_ADDR}.parquet"
    df = pl.read_parquet(deposit_file)
    print(f"Loaded {len(df):,} deposits")
    print()

    # Sort by datetime
    df_sorted = df.sort("DateTime")

    # Calculate cumulative deposit volume
    df_cumulative = df_sorted.with_columns([pl.col("TokenValue").cum_sum().alias("CumulativeVolume")])

    dates_all = df_cumulative["DateTime"].to_list()
    cumulative_volume = df_cumulative["CumulativeVolume"].to_list()

    # Calculate cumulative unique depositors
    unique_depositors_list = []
    seen_addresses = set()

    for addr in df_sorted["from"].to_list():
        seen_addresses.add(addr)
        unique_depositors_list.append(len(seen_addresses))

    # Output file
    output_file = REPORTS_DIR / f"deposit_timeseries_{SHORT_ADDR}.pdf"

    print(f"Creating PDF: {output_file}")
    print()

    # Create single-page PDF
    with PdfPages(output_file) as pdf:
        # Create figure with dual y-axis
        fig, ax1 = plt.subplots(figsize=(14, 8))

        # Primary axis: Cumulative deposit volume (LINE)
        color1 = "#00D4FF"
        ax1.set_xlabel("Date", fontsize=14, fontweight="bold")
        ax1.set_ylabel("Cumulative Deposit Volume (USDT)", fontsize=14, fontweight="bold", color=color1)
        ax1.plot(dates_all, cumulative_volume, color=color1, linewidth=3, alpha=0.9)
        ax1.fill_between(dates_all, cumulative_volume, alpha=0.2, color=color1)
        ax1.tick_params(axis="y", labelcolor=color1)
        ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f"${x/1e6:.0f}M"))

        # Secondary axis: Cumulative unique depositors (LINE)
        ax2 = ax1.twinx()
        color2 = "#00FF88"
        ax2.set_ylabel(
            "Cumulative Unique Depositors", fontsize=14, fontweight="bold", color=color2
        )
        ax2.plot(
            dates_all, unique_depositors_list, color=color2, linewidth=3, alpha=0.9
        )
        ax2.fill_between(dates_all, unique_depositors_list, alpha=0.2, color=color2)
        ax2.tick_params(axis="y", labelcolor=color2)

        # Title
        fig.suptitle(
            f"Deposit Time Series Analysis\nAddress: {TARGET_ADDRESS}",
            fontsize=18,
            fontweight="bold",
            y=0.98,
        )

        # Grid
        ax1.grid(alpha=0.2, linestyle="--")

        # Rotate x-axis labels
        plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45, ha="right")

        # Add summary stats box
        total_volume = cumulative_volume[-1]
        final_unique = unique_depositors_list[-1]
        total_deposits = len(df)

        stats_text = (
            f"Total Volume: ${total_volume/1e6:.1f}M USDT\n"
            f"Total Deposits: {total_deposits:,}\n"
            f"Unique Depositors: {final_unique:,}\n"
            f"Avg per Depositor: ${total_volume/final_unique:,.2f}"
        )

        ax1.text(
            0.02,
            0.98,
            stats_text,
            transform=ax1.transAxes,
            fontsize=11,
            verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="black", alpha=0.8, edgecolor="white"),
            fontfamily="monospace",
        )

        # Add legend
        legend_elements = [
            plt.Line2D([0], [0], color=color1, linewidth=3, label="Cumulative Volume"),
            plt.Line2D([0], [0], color=color2, linewidth=3, label="Cumulative Unique Depositors"),
        ]
        ax1.legend(
            handles=legend_elements, loc="upper left", bbox_to_anchor=(0.02, 0.85), framealpha=0.9
        )

        plt.tight_layout()
        pdf.savefig(fig, dpi=150)
        plt.close()

        # Add metadata
        d = pdf.infodict()
        d["Title"] = f"Deposit Time Series - {TARGET_ADDRESS}"
        d["Author"] = "Etherscan API Analysis"
        d["Subject"] = "USDT Deposit Time Series"
        d["CreationDate"] = datetime.now()

    print()
    print("=" * 80)
    print(f"✅ PDF saved to: {output_file}")
    print(f"   File size: {output_file.stat().st_size / 1024:.1f} KB")
    print("=" * 80)
    print()


if __name__ == "__main__":
    main()
