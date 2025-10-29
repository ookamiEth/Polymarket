#!/usr/bin/env python3
"""
Generate time series visualizations of deposit data and save as PDF.

Creates multiple charts:
1. Daily deposit volume (bar chart)
2. Daily deposit count (line chart)
3. Cumulative deposits over time
4. Deposit sizes over time (scatter)
5. Hourly distribution (heatmap style)
"""

from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import polars as pl
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.dates import DateFormatter, DayLocator

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
COLORS = {
    "primary": "#00D4FF",
    "secondary": "#FF00FF",
    "success": "#00FF88",
    "danger": "#FF3366",
    "warning": "#FFB000",
}


def load_data() -> pl.DataFrame:
    """Load deposit data from Parquet file."""
    deposit_file = RAW_DIR / f"usdt_deposits_{SHORT_ADDR}.parquet"
    df = pl.read_parquet(deposit_file)

    # Convert DateTime to date for daily aggregations
    df = df.with_columns([pl.col("DateTime").dt.date().alias("Date")])

    return df


def create_daily_volume_chart(df: pl.DataFrame, ax: plt.Axes) -> None:
    """Create daily deposit volume line chart."""
    # Aggregate by date
    daily_volume = (
        df.group_by("Date").agg(pl.col("TokenValue").sum().alias("Volume")).sort("Date")
    )

    # Convert to pandas for matplotlib
    dates = daily_volume["Date"].to_list()
    volumes = daily_volume["Volume"].to_list()

    # Plot as LINE
    ax.plot(dates, volumes, color=COLORS["primary"], linewidth=2.5, marker='o', markersize=4, alpha=0.9)
    ax.fill_between(dates, volumes, alpha=0.3, color=COLORS["primary"])

    # Format
    ax.set_title("Daily Deposit Volume", fontsize=16, fontweight="bold", pad=20)
    ax.set_xlabel("Date", fontsize=12)
    ax.set_ylabel("Volume (USDT)", fontsize=12)
    ax.grid(alpha=0.2, linestyle="--")

    # Format y-axis with millions
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f"${x/1e6:.1f}M"))

    # Rotate x-axis labels
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha="right")

    # Add stats annotation
    total_volume = sum(volumes)
    avg_volume = total_volume / len(volumes)
    max_volume = max(volumes)

    stats_text = f"Total: ${total_volume/1e6:.1f}M\nAvg: ${avg_volume/1e6:.1f}M\nMax: ${max_volume/1e6:.1f}M"
    ax.text(
        0.02,
        0.98,
        stats_text,
        transform=ax.transAxes,
        fontsize=10,
        verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="black", alpha=0.7),
    )


def create_daily_count_chart(df: pl.DataFrame, ax: plt.Axes) -> None:
    """Create daily deposit count line chart."""
    # Aggregate by date
    daily_count = df.group_by("Date").agg(pl.len().alias("Count")).sort("Date")

    # Convert to pandas for matplotlib
    dates = daily_count["Date"].to_list()
    counts = daily_count["Count"].to_list()

    # Plot
    ax.plot(dates, counts, color=COLORS["success"], linewidth=2, marker="o", markersize=4, alpha=0.8)
    ax.fill_between(dates, counts, alpha=0.3, color=COLORS["success"])

    # Format
    ax.set_title("Daily Deposit Count", fontsize=16, fontweight="bold", pad=20)
    ax.set_xlabel("Date", fontsize=12)
    ax.set_ylabel("Number of Deposits", fontsize=12)
    ax.grid(alpha=0.2, linestyle="--")

    # Rotate x-axis labels
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha="right")

    # Add stats annotation
    total_count = sum(counts)
    avg_count = total_count / len(counts)
    max_count = max(counts)

    stats_text = f"Total: {total_count:,}\nAvg: {avg_count:.0f}\nMax: {max_count:,}"
    ax.text(
        0.02,
        0.98,
        stats_text,
        transform=ax.transAxes,
        fontsize=10,
        verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="black", alpha=0.7),
    )


def create_cumulative_chart(df: pl.DataFrame, ax: plt.Axes) -> None:
    """Create cumulative deposits over time."""
    # Sort by datetime
    df_sorted = df.sort("DateTime")

    # Calculate cumulative sum
    df_cumulative = df_sorted.with_columns(
        [pl.col("TokenValue").cum_sum().alias("CumulativeVolume")]
    )

    # Convert to lists
    dates = df_cumulative["DateTime"].to_list()
    cumulative = df_cumulative["CumulativeVolume"].to_list()

    # Plot
    ax.plot(dates, cumulative, color=COLORS["warning"], linewidth=2.5, alpha=0.9)
    ax.fill_between(dates, cumulative, alpha=0.2, color=COLORS["warning"])

    # Format
    ax.set_title("Cumulative Deposit Volume Over Time", fontsize=16, fontweight="bold", pad=20)
    ax.set_xlabel("Date", fontsize=12)
    ax.set_ylabel("Cumulative Volume (USDT)", fontsize=12)
    ax.grid(alpha=0.2, linestyle="--")

    # Format y-axis with millions
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f"${x/1e6:.0f}M"))

    # Rotate x-axis labels
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha="right")

    # Add final total annotation
    final_total = cumulative[-1]
    ax.text(
        0.98,
        0.98,
        f"Final Total:\n${final_total/1e6:.1f}M",
        transform=ax.transAxes,
        fontsize=11,
        verticalalignment="top",
        horizontalalignment="right",
        bbox=dict(boxstyle="round", facecolor="black", alpha=0.7),
        fontweight="bold",
    )


def create_deposit_size_scatter(df: pl.DataFrame, ax: plt.Axes) -> None:
    """Create scatter plot of deposit sizes over time."""
    # Show ALL deposits (no sampling)
    dates = df["DateTime"].to_list()
    sizes = df["TokenValue"].to_list()

    # Plot with transparency (smaller points since showing all)
    ax.scatter(dates, sizes, alpha=0.4, s=10, c=COLORS["secondary"], edgecolors="none")

    # Format
    ax.set_title("Deposit Sizes Over Time", fontsize=16, fontweight="bold", pad=20)
    ax.set_xlabel("Date", fontsize=12)
    ax.set_ylabel("Deposit Amount (USDT) - Log Scale", fontsize=12)
    ax.grid(alpha=0.2, linestyle="--")

    # Set logarithmic scale on y-axis
    ax.set_yscale("log")

    # Set y-axis limits to start at minimum threshold
    ax.set_ylim(bottom=2650, top=200000)

    # Format y-axis with thousands
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f"${x/1000:.0f}K"))

    # Rotate x-axis labels
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha="right")

    # Add horizontal lines for key thresholds
    ax.axhline(y=2650, color=COLORS["danger"], linestyle="--", alpha=0.5, linewidth=1)
    ax.axhline(y=186282, color=COLORS["success"], linestyle="--", alpha=0.5, linewidth=1)

    # Add threshold labels on the RIGHT side, smaller font
    ax.text(
        0.98,
        0.02,
        "Min: $2,650",
        transform=ax.transAxes,
        fontsize=8,
        color=COLORS["danger"],
        verticalalignment="bottom",
        horizontalalignment="right",
        alpha=0.9,
    )
    ax.text(
        0.98,
        0.98,
        "Max: $186,282",
        transform=ax.transAxes,
        fontsize=8,
        color=COLORS["success"],
        verticalalignment="top",
        horizontalalignment="right",
        alpha=0.9,
    )



def create_hourly_heatmap(df: pl.DataFrame, ax: plt.Axes) -> None:
    """Create hourly deposit pattern visualization."""
    # Add hour and day of week columns
    df_hourly = df.with_columns(
        [
            pl.col("DateTime").dt.hour().alias("Hour"),
            pl.col("DateTime").dt.weekday().alias("Weekday"),
        ]
    )

    # Aggregate by hour and weekday
    hourly_counts = (
        df_hourly.group_by(["Weekday", "Hour"])
        .agg(pl.len().alias("Count"))
        .sort(["Weekday", "Hour"])
    )

    # Create matrix for heatmap
    import numpy as np

    matrix = np.zeros((7, 24))  # 7 days, 24 hours

    for row in hourly_counts.iter_rows(named=True):
        weekday = row["Weekday"]
        hour = row["Hour"]
        count = row["Count"]
        matrix[weekday, hour] = count

    # Plot heatmap
    im = ax.imshow(matrix, cmap="YlOrRd", aspect="auto", interpolation="nearest")

    # Format
    ax.set_title("Deposit Activity by Day and Hour (UTC)", fontsize=16, fontweight="bold", pad=20)
    ax.set_xlabel("Hour of Day (UTC)", fontsize=12)
    ax.set_ylabel("Day of Week", fontsize=12)

    # Set ticks
    ax.set_xticks(range(24))
    ax.set_yticks(range(7))
    ax.set_yticklabels(["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"])

    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label("Number of Deposits", rotation=270, labelpad=20)

    # Add grid
    ax.set_xticks([x - 0.5 for x in range(1, 24)], minor=True)
    ax.set_yticks([y - 0.5 for y in range(1, 7)], minor=True)
    ax.grid(which="minor", color="black", linestyle="-", linewidth=0.5, alpha=0.3)


def create_unique_depositors_over_time(df: pl.DataFrame, ax: plt.Axes) -> None:
    """Create cumulative unique depositors chart."""
    # Sort by datetime
    df_sorted = df.sort("DateTime")

    # Calculate cumulative unique depositors
    unique_depositors_list = []
    seen_addresses = set()

    for addr in df_sorted["from"].to_list():
        seen_addresses.add(addr)
        unique_depositors_list.append(len(seen_addresses))

    dates = df_sorted["DateTime"].to_list()

    # Plot
    ax.plot(dates, unique_depositors_list, color=COLORS["primary"], linewidth=2.5, alpha=0.9)
    ax.fill_between(dates, unique_depositors_list, alpha=0.2, color=COLORS["primary"])

    # Format
    ax.set_title("Cumulative Unique Depositors Over Time", fontsize=16, fontweight="bold", pad=20)
    ax.set_xlabel("Date", fontsize=12)
    ax.set_ylabel("Unique Depositors", fontsize=12)
    ax.grid(alpha=0.2, linestyle="--")

    # Rotate x-axis labels
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha="right")

    # Add final count annotation
    final_count = unique_depositors_list[-1]
    ax.text(
        0.98,
        0.98,
        f"Total Unique:\n{final_count:,}",
        transform=ax.transAxes,
        fontsize=11,
        verticalalignment="top",
        horizontalalignment="right",
        bbox=dict(boxstyle="round", facecolor="black", alpha=0.7),
        fontweight="bold",
    )


def main() -> None:
    """Generate time series PDF report."""
    print()
    print("=" * 80)
    print("GENERATING TIME SERIES PDF REPORT")
    print("=" * 80)
    print()

    # Load data
    print("Loading deposit data...")
    df = load_data()
    print(f"Loaded {len(df):,} deposits")
    print()

    # Output file
    output_file = REPORTS_DIR / f"deposit_timeseries_{SHORT_ADDR}.pdf"

    print(f"Creating PDF: {output_file}")
    print()

    # Create PDF with multiple pages
    with PdfPages(output_file) as pdf:
        # Page 1: Daily volume and count
        print("Creating page 1: Daily volume and count...")
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
        fig.suptitle(
            f"Deposit Time Series Analysis\nAddress: {TARGET_ADDRESS}",
            fontsize=18,
            fontweight="bold",
            y=0.98,
        )

        create_daily_volume_chart(df, ax1)
        create_daily_count_chart(df, ax2)

        plt.tight_layout()
        pdf.savefig(fig, dpi=150)
        plt.close()

        # Page 2: Cumulative charts
        print("Creating page 2: Cumulative charts...")
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
        fig.suptitle(
            f"Cumulative Growth Analysis\nAddress: {TARGET_ADDRESS}",
            fontsize=18,
            fontweight="bold",
            y=0.98,
        )

        create_cumulative_chart(df, ax1)
        create_unique_depositors_over_time(df, ax2)

        plt.tight_layout()
        pdf.savefig(fig, dpi=150)
        plt.close()

        # Page 3: Deposit sizes scatter (standalone)
        print("Creating page 3: Deposit sizes over time...")
        fig, ax = plt.subplots(1, 1, figsize=(14, 8))
        fig.suptitle(
            f"Deposit Sizes Over Time\nAddress: {TARGET_ADDRESS}",
            fontsize=18,
            fontweight="bold",
            y=0.98,
        )

        create_deposit_size_scatter(df, ax)

        plt.tight_layout()
        pdf.savefig(fig, dpi=150)
        plt.close()

        # Page 4: Activity heatmap (standalone)
        print("Creating page 4: Activity patterns...")
        fig, ax = plt.subplots(1, 1, figsize=(14, 8))
        fig.suptitle(
            f"Deposit Activity Patterns\nAddress: {TARGET_ADDRESS}",
            fontsize=18,
            fontweight="bold",
            y=0.98,
        )

        create_hourly_heatmap(df, ax)

        plt.tight_layout()
        pdf.savefig(fig, dpi=150)
        plt.close()

        # Add metadata
        d = pdf.infodict()
        d["Title"] = f"Deposit Time Series Analysis - {TARGET_ADDRESS}"
        d["Author"] = "Etherscan API Analysis"
        d["Subject"] = "USDT Deposit Time Series Visualization"
        d["Keywords"] = "Ethereum, USDT, Deposits, Time Series"
        d["CreationDate"] = datetime.now()

    print()
    print("=" * 80)
    print(f"✅ PDF saved to: {output_file}")
    print(f"   File size: {output_file.stat().st_size / 1024:.1f} KB")
    print("=" * 80)
    print()


if __name__ == "__main__":
    main()
