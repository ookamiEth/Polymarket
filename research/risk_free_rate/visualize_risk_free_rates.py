#!/usr/bin/env python3
"""
Visualize daily risk-free rates for short-dated crypto options.

Generates 5 PNG charts:
1. Time series of all rate components
2. Component contribution (stacked area)
3. Rate distributions (histograms)
4. Correlation heatmap
5. Volatility analysis (30-day rolling)
"""

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import polars as pl
import seaborn as sns

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Configure matplotlib for high-quality output
plt.style.use("seaborn-v0_8-darkgrid")
sns.set_palette("husl")
plt.rcParams["figure.dpi"] = 300
plt.rcParams["savefig.dpi"] = 300
plt.rcParams["font.size"] = 10
plt.rcParams["axes.titlesize"] = 12
plt.rcParams["axes.labelsize"] = 10

# Paths
DATA_DIR = Path(__file__).parent / "data"
INPUT_FILE = DATA_DIR / "daily_risk_free_rates_1d_2d_3d.parquet"
CHARTS_DIR = DATA_DIR / "charts"


def load_data() -> pl.DataFrame:
    """Load calculated risk-free rates."""
    logger.info(f"Loading data from {INPUT_FILE}")
    df = pl.read_parquet(INPUT_FILE)

    # Drop rows with null composite rates (early periods without full MA window)
    df = df.filter(pl.col("composite_3day_rate").is_not_null())

    logger.info(f"Loaded {len(df):,} daily observations")
    return df


def create_charts_directory() -> None:
    """Create charts output directory if it doesn't exist."""
    CHARTS_DIR.mkdir(exist_ok=True)
    logger.info(f"Charts will be saved to {CHARTS_DIR}")


def plot_timeseries(df: pl.DataFrame) -> None:
    """
    Chart 1: Time series of all rate components.

    Shows:
    - Composite 3-day rate (primary)
    - Funding rate (annual)
    - Lending APR
    """
    logger.info("Generating time series chart...")

    fig, ax = plt.subplots(figsize=(14, 6))

    # Convert to pandas for matplotlib compatibility
    df_plot = df.select(["date", "funding_rate_annual", "lending_apr", "composite_3day_rate"]).to_pandas()

    # Plot all components
    ax.plot(
        df_plot["date"],
        df_plot["composite_3day_rate"] * 100,
        label="Composite 3-Day Rate (70/30)",
        linewidth=2,
        color="#1f77b4",
    )
    ax.plot(
        df_plot["date"],
        df_plot["funding_rate_annual"] * 100,
        label="BTC Funding Rate (Annual)",
        linewidth=1.5,
        alpha=0.7,
        color="#ff7f0e",
    )
    ax.plot(
        df_plot["date"],
        df_plot["lending_apr"] * 100,
        label="USDT Lending APR",
        linewidth=1.5,
        alpha=0.7,
        color="#2ca02c",
    )

    # Paper benchmark range
    ax.axhspan(7, 10, alpha=0.1, color="gray", label="Paper Benchmark (7-10%)")

    ax.set_xlabel("Date")
    ax.set_ylabel("Annual Rate (%)")
    ax.set_title(
        "Risk-Free Rate Components for Short-Dated Options (≤3 days)\n"
        "70% BTC Funding + 30% USDT Lending (per SSRN 5231776)"
    )
    ax.legend(loc="upper left", framealpha=0.9)
    ax.grid(True, alpha=0.3)

    # Rotate x-axis labels
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()

    output_path = CHARTS_DIR / "rates_timeseries.png"
    plt.savefig(output_path, bbox_inches="tight")
    logger.info(f"✅ Saved: {output_path}")
    plt.close()


def plot_component_contribution(df: pl.DataFrame) -> None:
    """
    Chart 2: Stacked area showing funding vs lending contribution.

    Shows absolute contribution of each component to the 3-day composite rate.
    """
    logger.info("Generating component contribution chart...")

    fig, ax = plt.subplots(figsize=(14, 6))

    # Calculate contributions
    df_plot = df.select(
        [
            "date",
            (pl.col("funding_rate_annual") * 0.70).alias("funding_contribution"),
            (pl.col("lending_apr") * 0.30).alias("lending_contribution"),
        ]
    ).to_pandas()

    # Stacked area plot
    ax.fill_between(
        df_plot["date"],
        0,
        df_plot["funding_contribution"] * 100,
        label="Funding Contribution (70%)",
        alpha=0.7,
        color="#ff7f0e",
    )
    ax.fill_between(
        df_plot["date"],
        df_plot["funding_contribution"] * 100,
        (df_plot["funding_contribution"] + df_plot["lending_contribution"]) * 100,
        label="Lending Contribution (30%)",
        alpha=0.7,
        color="#2ca02c",
    )

    ax.set_xlabel("Date")
    ax.set_ylabel("Rate Contribution (%)")
    ax.set_title(
        "Component Contribution to 3-Day Composite Risk-Free Rate\nStacked View: Funding (70%) + Lending (30%)"
    )
    ax.legend(loc="upper left", framealpha=0.9)
    ax.grid(True, alpha=0.3)

    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()

    output_path = CHARTS_DIR / "component_contribution.png"
    plt.savefig(output_path, bbox_inches="tight")
    logger.info(f"✅ Saved: {output_path}")
    plt.close()


def plot_distributions(df: pl.DataFrame) -> None:
    """
    Chart 3: Histograms + KDE for rate distributions.

    Since all three maturities are identical (70/30), we just show:
    - Composite rate distribution
    - Funding rate distribution
    - Lending rate distribution
    """
    logger.info("Generating distribution charts...")

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # Convert to pandas
    df_plot = df.select(["composite_3day_rate", "funding_rate_annual", "lending_apr"]).to_pandas()

    # Composite rate
    axes[0].hist(
        df_plot["composite_3day_rate"] * 100,
        bins=40,
        alpha=0.7,
        color="#1f77b4",
        edgecolor="black",
    )
    axes[0].axvline(
        df_plot["composite_3day_rate"].mean() * 100,
        color="red",
        linestyle="--",
        linewidth=2,
        label=f"Mean: {df_plot['composite_3day_rate'].mean() * 100:.2f}%",
    )
    axes[0].axvline(
        df_plot["composite_3day_rate"].median() * 100,
        color="orange",
        linestyle="--",
        linewidth=2,
        label=f"Median: {df_plot['composite_3day_rate'].median() * 100:.2f}%",
    )
    axes[0].set_xlabel("Composite 3-Day Rate (%)")
    axes[0].set_ylabel("Frequency")
    axes[0].set_title("Composite Rate Distribution")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Funding rate
    axes[1].hist(
        df_plot["funding_rate_annual"] * 100,
        bins=40,
        alpha=0.7,
        color="#ff7f0e",
        edgecolor="black",
    )
    axes[1].axvline(
        df_plot["funding_rate_annual"].mean() * 100,
        color="red",
        linestyle="--",
        linewidth=2,
        label=f"Mean: {df_plot['funding_rate_annual'].mean() * 100:.2f}%",
    )
    axes[1].set_xlabel("BTC Funding Rate (Annual %)")
    axes[1].set_ylabel("Frequency")
    axes[1].set_title("Funding Rate Distribution")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    # Lending rate
    axes[2].hist(
        df_plot["lending_apr"] * 100,
        bins=40,
        alpha=0.7,
        color="#2ca02c",
        edgecolor="black",
    )
    axes[2].axvline(
        df_plot["lending_apr"].mean() * 100,
        color="red",
        linestyle="--",
        linewidth=2,
        label=f"Mean: {df_plot['lending_apr'].mean() * 100:.2f}%",
    )
    axes[2].set_xlabel("USDT Lending APR (%)")
    axes[2].set_ylabel("Frequency")
    axes[2].set_title("Lending Rate Distribution")
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()

    output_path = CHARTS_DIR / "rate_distributions.png"
    plt.savefig(output_path, bbox_inches="tight")
    logger.info(f"✅ Saved: {output_path}")
    plt.close()


def plot_correlation_heatmap(df: pl.DataFrame) -> None:
    """
    Chart 4: Correlation heatmap of all rate components.
    """
    logger.info("Generating correlation heatmap...")

    # Select numeric columns for correlation
    df_plot = df.select(
        [
            "funding_rate_annual",
            "lending_apr",
            "composite_1day_rate",
            "composite_2day_rate",
            "composite_3day_rate",
        ]
    ).to_pandas()

    # Rename for readability
    df_plot.columns = [
        "Funding\n(Annual)",
        "Lending\n(APR)",
        "Composite\n1-Day",
        "Composite\n2-Day",
        "Composite\n3-Day",
    ]

    # Calculate correlation matrix
    corr_matrix = df_plot.corr()

    # Plot heatmap
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(
        corr_matrix,
        annot=True,
        fmt=".3f",
        cmap="coolwarm",
        center=0,
        square=True,
        linewidths=1,
        cbar_kws={"shrink": 0.8},
        ax=ax,
    )

    ax.set_title(
        "Correlation Matrix: Risk-Free Rate Components\n(All composites use 70/30 weighting → perfect correlation)",
        pad=20,
    )

    plt.tight_layout()

    output_path = CHARTS_DIR / "correlation_heatmap.png"
    plt.savefig(output_path, bbox_inches="tight")
    logger.info(f"✅ Saved: {output_path}")
    plt.close()


def plot_volatility_analysis(df: pl.DataFrame) -> None:
    """
    Chart 5: 30-day rolling volatility for each rate.

    Shows stability/instability over time.
    """
    logger.info("Generating volatility analysis chart...")

    # Calculate 30-day rolling std
    df_vol = df.select(
        [
            "date",
            pl.col("composite_3day_rate").rolling_std(window_size=30).alias("composite_vol"),
            pl.col("funding_rate_annual").rolling_std(window_size=30).alias("funding_vol"),
            pl.col("lending_apr").rolling_std(window_size=30).alias("lending_vol"),
        ]
    ).to_pandas()

    fig, ax = plt.subplots(figsize=(14, 6))

    # Plot rolling volatilities
    ax.plot(
        df_vol["date"],
        df_vol["composite_vol"] * 100,
        label="Composite 3-Day (30d Std)",
        linewidth=2,
        color="#1f77b4",
    )
    ax.plot(
        df_vol["date"],
        df_vol["funding_vol"] * 100,
        label="Funding Rate (30d Std)",
        linewidth=1.5,
        alpha=0.7,
        color="#ff7f0e",
    )
    ax.plot(
        df_vol["date"],
        df_vol["lending_vol"] * 100,
        label="Lending APR (30d Std)",
        linewidth=1.5,
        alpha=0.7,
        color="#2ca02c",
    )

    ax.set_xlabel("Date")
    ax.set_ylabel("30-Day Rolling Volatility (% points)")
    ax.set_title(
        "Rate Volatility Over Time (30-Day Rolling Standard Deviation)\n"
        "Higher values indicate periods of rate instability"
    )
    ax.legend(loc="upper left", framealpha=0.9)
    ax.grid(True, alpha=0.3)

    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()

    output_path = CHARTS_DIR / "volatility_analysis.png"
    plt.savefig(output_path, bbox_inches="tight")
    logger.info(f"✅ Saved: {output_path}")
    plt.close()


def main() -> None:
    """Main execution function."""
    logger.info("=" * 80)
    logger.info("VISUALIZING RISK-FREE RATES FOR SHORT-DATED CRYPTO OPTIONS")
    logger.info("=" * 80)

    # Load data
    df = load_data()

    # Create output directory
    create_charts_directory()

    # Generate all charts
    plot_timeseries(df)
    plot_component_contribution(df)
    plot_distributions(df)
    plot_correlation_heatmap(df)
    plot_volatility_analysis(df)

    logger.info("\n" + "=" * 80)
    logger.info("✅ All charts generated successfully!")
    logger.info(f"📊 Charts saved to: {CHARTS_DIR}")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
