#!/usr/bin/env python3
"""
Phase 2: Market Microstructure Analysis
=======================================

Deep dive into market microstructure: time decay, jumps, autocorrelation,
and price behavior patterns.

Analysis Goals:
1. Time decay curves (theta) by moneyness
2. Final minute behavior (price evolution in last 60 seconds)
3. Jump detection and frequency analysis
4. Autocorrelation patterns (momentum vs mean reversion)
5. Intraday patterns (time-of-day effects)
"""

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import polars as pl

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Plot styling
plt.style.use("dark_background")
FIGURE_DIR = Path("research/model/03_deep_eda/figures")
TABLE_DIR = Path("research/model/03_deep_eda/tables")


def load_test_data(sample_frac: float = 0.1) -> pl.DataFrame:
    """Load test set with sampling."""
    logger.info(f"Loading test data (sample fraction: {sample_frac})")

    df = pl.read_parquet("research/model/results/xgboost_residual_model_test_2024-07-01_2025-09-30.parquet")
    logger.info(f"Loaded {len(df):,} rows")

    if sample_frac < 1.0:
        df = df.sample(fraction=sample_frac, seed=42)
        logger.info(f"Sampled to {len(df):,} rows")

    # Add derived columns
    df = df.with_columns(
        [
            (pl.col("moneyness") * 100).alias("moneyness_pct"),
            (pl.col("time_remaining") / 60).alias("time_remaining_min"),
            (pl.col("outcome") - pl.col("prob_mid")).alias("prediction_error"),
            # Hour of day from timestamp
            pl.from_epoch("timestamp", time_unit="s").dt.hour().alias("hour_of_day"),
        ]
    )

    return df


def analyze_time_decay(df: pl.DataFrame, output_file: Path) -> None:
    """
    Analyze theta (time decay) across moneyness buckets.

    Theta = dP/dT (change in probability per unit time)
    """
    logger.info("Analyzing time decay (theta)")

    # Define moneyness buckets
    moneyness_buckets = [
        (-np.inf, -1.0, "Deep OTM"),
        (-1.0, -0.5, "OTM"),
        (-0.5, -0.2, "Slight OTM"),
        (-0.2, 0.2, "ATM"),
        (0.2, 0.5, "Slight ITM"),
        (0.5, 1.0, "ITM"),
        (1.0, np.inf, "Deep ITM"),
    ]

    # Time remaining buckets (minutes)
    time_buckets = np.linspace(0, 450 / 60, 20)  # 450s = 7.5 min

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()

    # Fixed colors for each moneyness bucket
    colors = ["#00FF88", "#00D4FF", "#FFB000", "#FF3366", "#FF00FF", "#00FFFF", "#FFFF00"]

    for idx, (m_min, m_max, m_label) in enumerate(moneyness_buckets[:4]):  # Plot first 4
        ax = axes[idx]

        # Filter to moneyness bucket
        df_bucket = df.filter((pl.col("moneyness_pct") >= m_min) & (pl.col("moneyness_pct") < m_max))

        if len(df_bucket) < 100:
            continue

        # Bin by time remaining
        df_binned = (
            df_bucket.with_columns([pl.col("time_remaining_min").cut(time_buckets).alias("time_bin")])
            .group_by("time_bin")
            .agg(
                [
                    pl.len().alias("count"),
                    pl.col("prob_mid").mean().alias("avg_prob"),
                    pl.col("time_remaining_min").mean().alias("avg_time"),
                ]
            )
            .filter(pl.col("count") >= 20)
            .sort("avg_time", descending=True)  # Sort from longest to shortest
        )

        avg_prob = df_binned["avg_prob"].to_numpy()
        avg_time = df_binned["avg_time"].to_numpy()

        # Plot probability vs time remaining
        ax.plot(avg_time, avg_prob, linewidth=2.5, color=colors[idx], marker="o", markersize=6, label=m_label)

        # Calculate theta (slope)
        if len(avg_time) > 2:
            theta, intercept = np.polyfit(avg_time, avg_prob, 1)
            ax.plot(avg_time, theta * avg_time + intercept, linestyle="--", color=colors[idx], alpha=0.5, linewidth=1.5)

            ax.text(
                0.05,
                0.95,
                f"Θ = {theta:.4f} prob/min",
                transform=ax.transAxes,
                fontsize=10,
                verticalalignment="top",
                bbox={"boxstyle": "round", "facecolor": "black", "alpha": 0.8},
            )

        ax.set_xlabel("Time Remaining (minutes)", fontsize=12)
        ax.set_ylabel("Average Probability", fontsize=12)
        ax.set_title(f"Time Decay: {m_label}", fontsize=14)
        ax.grid(alpha=0.2)
        ax.legend(fontsize=10)

    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    logger.info(f"Saved: {output_file}")
    plt.close()


def analyze_final_minute(df: pl.DataFrame, output_file: Path) -> None:
    """Analyze price behavior in the final 60 seconds before expiry."""
    logger.info("Analyzing final minute behavior")

    # Filter to last 60 seconds
    df_final = df.filter(pl.col("time_remaining") <= 60)

    logger.info(f"Final minute samples: {len(df_final):,}")

    # Bin by seconds remaining
    second_bins = np.linspace(0, 60, 30)

    # Analyze by moneyness regime
    moneyness_regimes = [
        (-0.5, -0.1, "OTM"),
        (-0.1, 0.1, "ATM"),
        (0.1, 0.5, "ITM"),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    for idx, (m_min, m_max, m_label) in enumerate(moneyness_regimes):
        ax = axes[idx]

        df_regime = df_final.filter((pl.col("moneyness_pct") >= m_min) & (pl.col("moneyness_pct") < m_max))

        if len(df_regime) < 50:
            continue

        # Bin by time
        df_binned = (
            df_regime.with_columns([pl.col("time_remaining").cut(second_bins).alias("time_bin")])
            .group_by("time_bin")
            .agg(
                [
                    pl.len().alias("count"),
                    pl.col("prob_mid").mean().alias("avg_prob"),
                    pl.col("prob_mid").std().alias("std_prob"),
                    pl.col("time_remaining").mean().alias("avg_time"),
                ]
            )
            .filter(pl.col("count") >= 5)
            .sort("avg_time", descending=True)
        )

        avg_prob = df_binned["avg_prob"].to_numpy()
        std_prob = df_binned["std_prob"].to_numpy()
        avg_time = df_binned["avg_time"].to_numpy()

        # Plot with error bars
        ax.errorbar(
            avg_time,
            avg_prob,
            yerr=std_prob,
            fmt="-o",
            linewidth=2,
            markersize=6,
            capsize=4,
            alpha=0.8,
            label="Mean ± Std",
        )

        ax.set_xlabel("Time Remaining (seconds)", fontsize=12)
        ax.set_ylabel("Probability", fontsize=12)
        ax.set_title(f"Final Minute: {m_label}", fontsize=14)
        ax.grid(alpha=0.2)
        ax.legend(fontsize=10)
        ax.set_ylim(0, 1)

    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    logger.info(f"Saved: {output_file}")
    plt.close()


def analyze_jumps(df: pl.DataFrame, output_file: Path) -> None:
    """Analyze jump frequency and characteristics."""
    logger.info("Analyzing jump patterns")

    # Jump frequency by moneyness
    df_jumps = df.with_columns(
        [
            pl.when(pl.col("jump_detected") == 1).then(pl.lit(1)).otherwise(pl.lit(0)).alias("is_jump"),
        ]
    )

    # Bin by moneyness
    moneyness_bins = np.linspace(-2, 2, 30)

    df_jump_analysis = (
        df_jumps.with_columns([pl.col("moneyness_pct").cut(moneyness_bins).alias("m_bin")])
        .group_by("m_bin")
        .agg(
            [
                pl.len().alias("total_count"),
                pl.col("is_jump").sum().alias("jump_count"),
                pl.col("moneyness_pct").mean().alias("avg_moneyness"),
                pl.col("jump_intensity_300s").mean().alias("avg_jump_intensity"),
            ]
        )
        .filter(pl.col("total_count") >= 100)
        .with_columns([(pl.col("jump_count") / pl.col("total_count") * 100).alias("jump_frequency_pct")])
        .sort("avg_moneyness")
    )

    # Plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # Jump frequency
    ax1.plot(
        df_jump_analysis["avg_moneyness"].to_numpy(),
        df_jump_analysis["jump_frequency_pct"].to_numpy(),
        linewidth=2.5,
        color="#FF3366",
        marker="o",
        markersize=6,
    )
    ax1.fill_between(
        df_jump_analysis["avg_moneyness"].to_numpy(),
        0,
        df_jump_analysis["jump_frequency_pct"].to_numpy(),
        alpha=0.3,
        color="#FF3366",
    )
    ax1.set_xlabel("Moneyness (%)", fontsize=14)
    ax1.set_ylabel("Jump Frequency (%)", fontsize=14)
    ax1.set_title("Jump Frequency by Moneyness", fontsize=16)
    ax1.grid(alpha=0.2)
    ax1.axvline(0, color="gray", linestyle=":", alpha=0.5)

    # Jump intensity
    ax2.plot(
        df_jump_analysis["avg_moneyness"].to_numpy(),
        df_jump_analysis["avg_jump_intensity"].to_numpy(),
        linewidth=2.5,
        color="#00D4FF",
        marker="o",
        markersize=6,
    )
    ax2.set_xlabel("Moneyness (%)", fontsize=14)
    ax2.set_ylabel("Average Jump Intensity", fontsize=14)
    ax2.set_title("Jump Intensity by Moneyness", fontsize=16)
    ax2.grid(alpha=0.2)
    ax2.axvline(0, color="gray", linestyle=":", alpha=0.5)

    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    logger.info(f"Saved: {output_file}")
    plt.close()


def analyze_autocorrelation(df: pl.DataFrame, output_file: Path) -> None:
    """Analyze autocorrelation patterns (momentum vs mean reversion)."""
    logger.info("Analyzing autocorrelation patterns")

    # Focus on autocorr_lag1_300s and autocorr_lag5_300s
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    moneyness_bins = np.linspace(-2, 2, 30)

    # Lag-1 autocorrelation
    df_lag1 = (
        df.with_columns([pl.col("moneyness_pct").cut(moneyness_bins).alias("m_bin")])
        .group_by("m_bin")
        .agg(
            [
                pl.len().alias("count"),
                pl.col("moneyness_pct").mean().alias("avg_moneyness"),
                pl.col("autocorr_lag1_300s").mean().alias("avg_autocorr_lag1"),
                pl.col("autocorr_lag5_300s").mean().alias("avg_autocorr_lag5"),
            ]
        )
        .filter(pl.col("count") >= 100)
        .sort("avg_moneyness")
    )

    # Lag-1
    ax1.plot(
        df_lag1["avg_moneyness"].to_numpy(),
        df_lag1["avg_autocorr_lag1"].to_numpy(),
        linewidth=2.5,
        color="#00FF88",
        marker="o",
        markersize=6,
        label="Lag-1 Autocorr",
    )
    ax1.axhline(0, color="gray", linestyle="--", alpha=0.5)
    ax1.axhline(0.05, color="red", linestyle=":", alpha=0.3, label="Momentum threshold")
    ax1.axhline(-0.05, color="blue", linestyle=":", alpha=0.3, label="Mean reversion threshold")
    ax1.set_xlabel("Moneyness (%)", fontsize=14)
    ax1.set_ylabel("Lag-1 Autocorrelation", fontsize=14)
    ax1.set_title("Lag-1 Autocorrelation (Momentum Detection)", fontsize=16)
    ax1.grid(alpha=0.2)
    ax1.legend(fontsize=10)
    ax1.axvline(0, color="gray", linestyle=":", alpha=0.5)

    # Lag-5
    ax2.plot(
        df_lag1["avg_moneyness"].to_numpy(),
        df_lag1["avg_autocorr_lag5"].to_numpy(),
        linewidth=2.5,
        color="#FFB000",
        marker="o",
        markersize=6,
        label="Lag-5 Autocorr",
    )
    ax2.axhline(0, color="gray", linestyle="--", alpha=0.5)
    ax2.set_xlabel("Moneyness (%)", fontsize=14)
    ax2.set_ylabel("Lag-5 Autocorrelation", fontsize=14)
    ax2.set_title("Lag-5 Autocorrelation", fontsize=16)
    ax2.grid(alpha=0.2)
    ax2.legend(fontsize=10)
    ax2.axvline(0, color="gray", linestyle=":", alpha=0.5)

    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    logger.info(f"Saved: {output_file}")
    plt.close()


def analyze_intraday_patterns(df: pl.DataFrame, output_file: Path) -> None:
    """Analyze time-of-day effects on pricing and outcomes."""
    logger.info("Analyzing intraday patterns")

    # Group by hour of day
    df_hourly = (
        df.group_by("hour_of_day")
        .agg(
            [
                pl.len().alias("count"),
                pl.col("prob_mid").mean().alias("avg_prob"),
                pl.col("outcome").mean().alias("avg_outcome"),
                pl.col("prediction_error").mean().alias("avg_error"),
                pl.col("prediction_error").std().alias("std_error"),
                pl.col("rv_900s").mean().alias("avg_rv"),
            ]
        )
        .sort("hour_of_day")
    )

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()

    hours = df_hourly["hour_of_day"].to_numpy()

    # Volume
    axes[0].bar(hours, df_hourly["count"].to_numpy(), color="#00D4FF", alpha=0.7, edgecolor="white", linewidth=0.5)
    axes[0].set_xlabel("Hour of Day (UTC)", fontsize=12)
    axes[0].set_ylabel("Sample Count", fontsize=12)
    axes[0].set_title("Trading Activity by Hour", fontsize=14)
    axes[0].grid(alpha=0.2, axis="y")

    # Average probability
    axes[1].plot(hours, df_hourly["avg_prob"].to_numpy(), linewidth=2.5, color="#00FF88", marker="o", markersize=6)
    axes[1].set_xlabel("Hour of Day (UTC)", fontsize=12)
    axes[1].set_ylabel("Average Probability", fontsize=12)
    axes[1].set_title("Average Probability by Hour", fontsize=14)
    axes[1].grid(alpha=0.2)

    # Prediction error
    axes[2].errorbar(
        hours,
        df_hourly["avg_error"].to_numpy(),
        yerr=df_hourly["std_error"].to_numpy(),
        fmt="-o",
        linewidth=2,
        markersize=6,
        capsize=4,
        color="#FF3366",
    )
    axes[2].axhline(0, color="gray", linestyle="--", alpha=0.5)
    axes[2].set_xlabel("Hour of Day (UTC)", fontsize=12)
    axes[2].set_ylabel("Mean Prediction Error", fontsize=12)
    axes[2].set_title("Prediction Error by Hour", fontsize=14)
    axes[2].grid(alpha=0.2)

    # Realized volatility
    axes[3].plot(hours, df_hourly["avg_rv"].to_numpy(), linewidth=2.5, color="#FFB000", marker="o", markersize=6)
    axes[3].set_xlabel("Hour of Day (UTC)", fontsize=12)
    axes[3].set_ylabel("Average RV (15min)", fontsize=12)
    axes[3].set_title("Realized Volatility by Hour", fontsize=14)
    axes[3].grid(alpha=0.2)

    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    logger.info(f"Saved: {output_file}")
    plt.close()


def main() -> None:
    """Run Phase 2: Market Microstructure Analysis."""
    logger.info("=" * 80)
    logger.info("PHASE 2: MARKET MICROSTRUCTURE ANALYSIS")
    logger.info("=" * 80)

    # Load data
    df = load_test_data(sample_frac=0.1)

    # Run analyses
    analyze_time_decay(df, FIGURE_DIR / "04_time_decay_theta.png")
    analyze_final_minute(df, FIGURE_DIR / "05_final_minute_behavior.png")
    analyze_jumps(df, FIGURE_DIR / "06_jump_analysis.png")
    analyze_autocorrelation(df, FIGURE_DIR / "07_autocorrelation.png")
    analyze_intraday_patterns(df, FIGURE_DIR / "08_intraday_patterns.png")

    logger.info("=" * 80)
    logger.info("PHASE 2 COMPLETE")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
