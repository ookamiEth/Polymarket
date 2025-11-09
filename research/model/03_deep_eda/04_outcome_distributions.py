#!/usr/bin/env python3
"""
Phase 4: Outcome Distribution Analysis
======================================

Analyze empirical outcome distributions, win rates, asymmetries,
and temporal patterns in binary option settlements.

Analysis Goals:
1. Outcome histograms by moneyness buckets
2. Win rate analysis (actual vs predicted)
3. Asymmetry tests (upside vs downside bias)
4. Temporal patterns (hour, day, month effects)
5. Volatility regime outcomes
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
            pl.from_epoch("timestamp", time_unit="s").dt.hour().alias("hour_of_day"),
            pl.from_epoch("timestamp", time_unit="s").dt.weekday().alias("day_of_week"),
            pl.from_epoch("timestamp", time_unit="s").dt.month().alias("month"),
        ]
    )

    return df


def plot_outcome_distributions_by_moneyness(df: pl.DataFrame, output_file: Path) -> None:
    """Outcome distributions for different moneyness buckets."""
    logger.info("Analyzing outcome distributions by moneyness")

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

    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    axes = axes.flatten()

    for idx, (m_min, m_max, m_label) in enumerate(moneyness_buckets):
        if idx >= len(axes):
            break

        ax = axes[idx]

        # Filter to bucket
        df_bucket = df.filter((pl.col("moneyness_pct") >= m_min) & (pl.col("moneyness_pct") < m_max))

        if len(df_bucket) < 100:
            ax.text(0.5, 0.5, "Insufficient Data", ha="center", va="center", fontsize=14)
            ax.set_title(m_label, fontsize=12)
            continue

        # Calculate statistics
        win_rate = df_bucket["outcome"].mean()
        avg_prob = df_bucket["prob_mid"].mean()
        count = len(df_bucket)

        # Histogram
        outcomes = df_bucket["outcome"].to_numpy()
        ax.hist(
            [1 - outcomes, outcomes],
            bins=2,
            label=["Loss (0)", "Win (1)"],
            color=["#FF3366", "#00FF88"],
            alpha=0.8,
            edgecolor="white",
            linewidth=1,
        )

        ax.set_xlabel("Outcome", fontsize=10)
        ax.set_ylabel("Count", fontsize=10)
        ax.set_title(f"{m_label}\n(n={count:,})", fontsize=12)
        ax.set_xticks([0.25, 0.75])
        ax.set_xticklabels(["Loss", "Win"])

        # Add statistics
        stats_text = f"Win Rate: {win_rate:.1%}\n"
        stats_text += f"Avg Prob: {avg_prob:.1%}"

        ax.text(
            0.95,
            0.95,
            stats_text,
            transform=ax.transAxes,
            fontsize=9,
            verticalalignment="top",
            horizontalalignment="right",
            bbox={"boxstyle": "round", "facecolor": "black", "alpha": 0.8},
        )

    # Remove extra subplot
    if len(moneyness_buckets) < len(axes):
        fig.delaxes(axes[-1])

    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    logger.info(f"Saved: {output_file}")
    plt.close()


def plot_win_rate_analysis(df: pl.DataFrame, output_file: Path) -> None:
    """Win rate vs predicted probability across moneyness range."""
    logger.info("Analyzing win rates vs predictions")

    # Bin by moneyness
    moneyness_bins = np.linspace(-2, 2, 30)

    df_win_rate = (
        df.with_columns([pl.col("moneyness_pct").cut(moneyness_bins).alias("m_bin")])
        .group_by("m_bin")
        .agg(
            [
                pl.len().alias("count"),
                pl.col("moneyness_pct").mean().alias("avg_m"),
                pl.col("outcome").mean().alias("actual_win_rate"),
                pl.col("prob_mid").mean().alias("predicted_prob_bs"),
                pl.col("prob_corrected_xgb").mean().alias("predicted_prob_ml"),
            ]
        )
        .filter(pl.col("count") >= 100)
        .sort("avg_m")
    )

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    moneyness = df_win_rate["avg_m"].to_numpy()
    actual = df_win_rate["actual_win_rate"].to_numpy()
    pred_bs = df_win_rate["predicted_prob_bs"].to_numpy()
    pred_ml = df_win_rate["predicted_prob_ml"].to_numpy()

    # Win rate comparison
    ax1.plot(
        moneyness, actual, linewidth=3, color="#FFFFFF", marker="o", markersize=8, label="Actual Win Rate", zorder=3
    )
    ax1.plot(
        moneyness,
        pred_bs,
        linewidth=2,
        color="#FF3366",
        marker="s",
        markersize=6,
        alpha=0.7,
        label="BS Predicted",
        linestyle="--",
    )
    ax1.plot(
        moneyness,
        pred_ml,
        linewidth=2,
        color="#00FF88",
        marker="^",
        markersize=6,
        alpha=0.7,
        label="ML Predicted",
        linestyle="--",
    )
    ax1.axhline(0.5, color="gray", linestyle=":", alpha=0.5)
    ax1.axvline(0, color="gray", linestyle=":", alpha=0.5)
    ax1.set_xlabel("Moneyness (%)", fontsize=14)
    ax1.set_ylabel("Win Rate / Probability", fontsize=14)
    ax1.set_title("Win Rate vs Predicted Probability", fontsize=16, pad=20)
    ax1.grid(alpha=0.2)
    ax1.legend(fontsize=12)
    ax1.set_ylim(0, 1)

    # Calibration error
    error_bs = actual - pred_bs
    error_ml = actual - pred_ml

    ax2.plot(moneyness, error_bs, linewidth=2.5, color="#FF3366", marker="s", markersize=6, label="BS Error", alpha=0.8)
    ax2.plot(moneyness, error_ml, linewidth=2.5, color="#00FF88", marker="o", markersize=6, label="ML Error", alpha=0.8)
    ax2.axhline(0, color="gray", linestyle="--", alpha=0.5)
    ax2.axvline(0, color="gray", linestyle=":", alpha=0.5)
    ax2.set_xlabel("Moneyness (%)", fontsize=14)
    ax2.set_ylabel("Calibration Error (Actual - Predicted)", fontsize=14)
    ax2.set_title("Model Calibration Error by Moneyness", fontsize=16, pad=20)
    ax2.grid(alpha=0.2)
    ax2.legend(fontsize=12)

    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    logger.info(f"Saved: {output_file}")
    plt.close()


def plot_asymmetry_analysis(df: pl.DataFrame, output_file: Path) -> None:
    """Test for asymmetries in outcomes (upside vs downside)."""
    logger.info("Analyzing outcome asymmetries")

    # Split into upside (ITM) and downside (OTM) calls
    df_otm = df.filter(pl.col("moneyness_pct") < 0)  # Out of the money
    df_itm = df.filter(pl.col("moneyness_pct") > 0)  # In the money

    # Bin by absolute moneyness
    abs_moneyness_bins = np.linspace(0, 2, 20)

    # OTM analysis
    df_otm_binned = (
        df_otm.with_columns([pl.col("moneyness_pct").abs().alias("abs_m")])
        .with_columns([pl.col("abs_m").cut(abs_moneyness_bins).alias("m_bin")])
        .group_by("m_bin")
        .agg(
            [
                pl.len().alias("count"),
                pl.col("abs_m").mean().alias("avg_abs_m"),
                pl.col("outcome").mean().alias("win_rate"),
                pl.col("prob_mid").mean().alias("avg_prob"),
            ]
        )
        .filter(pl.col("count") >= 50)
        .sort("avg_abs_m")
    )

    # ITM analysis
    df_itm_binned = (
        df_itm.with_columns([pl.col("moneyness_pct").abs().alias("abs_m")])
        .with_columns([pl.col("abs_m").cut(abs_moneyness_bins).alias("m_bin")])
        .group_by("m_bin")
        .agg(
            [
                pl.len().alias("count"),
                pl.col("abs_m").mean().alias("avg_abs_m"),
                pl.col("outcome").mean().alias("win_rate"),
                pl.col("prob_mid").mean().alias("avg_prob"),
            ]
        )
        .filter(pl.col("count") >= 50)
        .sort("avg_abs_m")
    )

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # Win rates
    ax1.plot(
        df_otm_binned["avg_abs_m"].to_numpy(),
        df_otm_binned["win_rate"].to_numpy(),
        linewidth=2.5,
        color="#FF3366",
        marker="o",
        markersize=6,
        label="OTM (Downside)",
    )
    ax1.plot(
        df_itm_binned["avg_abs_m"].to_numpy(),
        df_itm_binned["win_rate"].to_numpy(),
        linewidth=2.5,
        color="#00FF88",
        marker="s",
        markersize=6,
        label="ITM (Upside)",
    )
    ax1.set_xlabel("Absolute Moneyness (%)", fontsize=14)
    ax1.set_ylabel("Win Rate", fontsize=14)
    ax1.set_title("Win Rate Asymmetry: OTM vs ITM", fontsize=16, pad=20)
    ax1.grid(alpha=0.2)
    ax1.legend(fontsize=12)

    # Sample distribution
    ax2.bar(
        df_otm_binned["avg_abs_m"].to_numpy(),
        df_otm_binned["count"].to_numpy(),
        width=0.08,
        color="#FF3366",
        alpha=0.7,
        label="OTM",
    )
    ax2.bar(
        df_itm_binned["avg_abs_m"].to_numpy(),
        df_itm_binned["count"].to_numpy(),
        width=0.08,
        color="#00FF88",
        alpha=0.7,
        label="ITM",
    )
    ax2.set_xlabel("Absolute Moneyness (%)", fontsize=14)
    ax2.set_ylabel("Sample Count", fontsize=14)
    ax2.set_title("Sample Distribution", fontsize=16, pad=20)
    ax2.grid(alpha=0.2, axis="y")
    ax2.legend(fontsize=12)

    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    logger.info(f"Saved: {output_file}")
    plt.close()


def plot_temporal_patterns(df: pl.DataFrame, output_file: Path) -> None:
    """Temporal patterns in outcomes (hour, day, month)."""
    logger.info("Analyzing temporal outcome patterns")

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # Hour of day
    df_hourly = (
        df.group_by("hour_of_day")
        .agg(
            [
                pl.len().alias("count"),
                pl.col("outcome").mean().alias("win_rate"),
                pl.col("prob_mid").mean().alias("avg_prob"),
            ]
        )
        .sort("hour_of_day")
    )

    axes[0, 0].plot(
        df_hourly["hour_of_day"].to_numpy(),
        df_hourly["win_rate"].to_numpy(),
        linewidth=2.5,
        color="#00D4FF",
        marker="o",
        markersize=6,
    )
    axes[0, 0].axhline(df["outcome"].mean(), color="gray", linestyle="--", alpha=0.5, label="Overall Mean")
    axes[0, 0].set_xlabel("Hour of Day (UTC)", fontsize=12)
    axes[0, 0].set_ylabel("Win Rate", fontsize=12)
    axes[0, 0].set_title("Win Rate by Hour of Day", fontsize=14)
    axes[0, 0].grid(alpha=0.2)
    axes[0, 0].legend(fontsize=10)

    # Day of week
    df_daily = (
        df.group_by("day_of_week")
        .agg(
            [
                pl.len().alias("count"),
                pl.col("outcome").mean().alias("win_rate"),
                pl.col("prob_mid").mean().alias("avg_prob"),
            ]
        )
        .sort("day_of_week")
    )

    day_names = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    axes[0, 1].bar(
        range(len(df_daily)),
        df_daily["win_rate"].to_numpy(),
        color="#00FF88",
        alpha=0.8,
        edgecolor="white",
        linewidth=0.5,
    )
    axes[0, 1].axhline(df["outcome"].mean(), color="gray", linestyle="--", alpha=0.5, label="Overall Mean")
    axes[0, 1].set_xlabel("Day of Week", fontsize=12)
    axes[0, 1].set_ylabel("Win Rate", fontsize=12)
    axes[0, 1].set_title("Win Rate by Day of Week", fontsize=14)
    axes[0, 1].set_xticks(range(len(df_daily)))
    axes[0, 1].set_xticklabels(day_names[: len(df_daily)])
    axes[0, 1].grid(alpha=0.2, axis="y")
    axes[0, 1].legend(fontsize=10)

    # Month
    df_monthly = (
        df.group_by("month")
        .agg(
            [
                pl.len().alias("count"),
                pl.col("outcome").mean().alias("win_rate"),
                pl.col("prob_mid").mean().alias("avg_prob"),
            ]
        )
        .sort("month")
    )

    month_names = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    axes[1, 0].plot(
        df_monthly["month"].to_numpy(),
        df_monthly["win_rate"].to_numpy(),
        linewidth=2.5,
        color="#FFB000",
        marker="o",
        markersize=6,
    )
    axes[1, 0].axhline(df["outcome"].mean(), color="gray", linestyle="--", alpha=0.5, label="Overall Mean")
    axes[1, 0].set_xlabel("Month", fontsize=12)
    axes[1, 0].set_ylabel("Win Rate", fontsize=12)
    axes[1, 0].set_title("Win Rate by Month", fontsize=14)
    axes[1, 0].set_xticks(df_monthly["month"].to_numpy())
    axes[1, 0].set_xticklabels([month_names[int(m) - 1] for m in df_monthly["month"].to_list()])
    axes[1, 0].grid(alpha=0.2)
    axes[1, 0].legend(fontsize=10)

    # Volatility regime
    vol_terciles = [
        df["rv_900s"].quantile(0.33),
        df["rv_900s"].quantile(0.67),
    ]

    df_vol = df.with_columns(
        [
            pl.when(pl.col("rv_900s") < vol_terciles[0])
            .then(pl.lit("Low Vol"))
            .when(pl.col("rv_900s") < vol_terciles[1])
            .then(pl.lit("Mid Vol"))
            .otherwise(pl.lit("High Vol"))
            .alias("vol_regime")
        ]
    )

    df_vol_agg = (
        df_vol.group_by("vol_regime")
        .agg(
            [
                pl.len().alias("count"),
                pl.col("outcome").mean().alias("win_rate"),
            ]
        )
        .sort("vol_regime")
    )

    colors_vol = {"Low Vol": "#00FF88", "Mid Vol": "#00D4FF", "High Vol": "#FF3366"}
    bar_colors = [colors_vol[regime] for regime in df_vol_agg["vol_regime"].to_list()]

    axes[1, 1].bar(
        range(len(df_vol_agg)),
        df_vol_agg["win_rate"].to_numpy(),
        color=bar_colors,
        alpha=0.8,
        edgecolor="white",
        linewidth=0.5,
    )
    axes[1, 1].axhline(df["outcome"].mean(), color="gray", linestyle="--", alpha=0.5, label="Overall Mean")
    axes[1, 1].set_xlabel("Volatility Regime", fontsize=12)
    axes[1, 1].set_ylabel("Win Rate", fontsize=12)
    axes[1, 1].set_title("Win Rate by Volatility Regime", fontsize=14)
    axes[1, 1].set_xticks(range(len(df_vol_agg)))
    axes[1, 1].set_xticklabels(df_vol_agg["vol_regime"].to_list())
    axes[1, 1].grid(alpha=0.2, axis="y")
    axes[1, 1].legend(fontsize=10)

    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    logger.info(f"Saved: {output_file}")
    plt.close()


def generate_outcome_statistics_table(df: pl.DataFrame) -> None:
    """Generate comprehensive outcome statistics table."""
    logger.info("Generating outcome statistics table")

    # Overall statistics
    overall_stats = {
        "category": "Overall",
        "count": len(df),
        "win_rate": float(df["outcome"].mean() or 0),
        "avg_prob_bs": float(df["prob_mid"].mean() or 0),
        "avg_prob_ml": float(df["prob_corrected_xgb"].mean() or 0),
        "calibration_error_bs": float((df["outcome"].mean() or 0) - (df["prob_mid"].mean() or 0)),
        "calibration_error_ml": float((df["outcome"].mean() or 0) - (df["prob_corrected_xgb"].mean() or 0)),
    }

    stats_list = [overall_stats]

    # By moneyness regime
    moneyness_regimes = [
        ("Deep OTM", (pl.col("moneyness_pct") < -1.0)),
        ("OTM", (pl.col("moneyness_pct") >= -1.0) & (pl.col("moneyness_pct") < -0.2)),
        ("ATM", (pl.col("moneyness_pct") >= -0.2) & (pl.col("moneyness_pct") <= 0.2)),
        ("ITM", (pl.col("moneyness_pct") > 0.2) & (pl.col("moneyness_pct") <= 1.0)),
        ("Deep ITM", (pl.col("moneyness_pct") > 1.0)),
    ]

    for regime_name, condition in moneyness_regimes:
        df_regime = df.filter(condition)
        if len(df_regime) > 0:
            stats_list.append(
                {
                    "category": regime_name,
                    "count": len(df_regime),
                    "win_rate": float(df_regime["outcome"].mean() or 0),
                    "avg_prob_bs": float(df_regime["prob_mid"].mean() or 0),
                    "avg_prob_ml": float(df_regime["prob_corrected_xgb"].mean() or 0),
                    "calibration_error_bs": float(
                        (df_regime["outcome"].mean() or 0) - (df_regime["prob_mid"].mean() or 0)
                    ),
                    "calibration_error_ml": float(
                        (df_regime["outcome"].mean() or 0) - (df_regime["prob_corrected_xgb"].mean() or 0)
                    ),
                }
            )

    df_stats = pl.DataFrame(stats_list)

    # Save
    output_file = TABLE_DIR / "outcome_statistics.csv"
    df_stats.write_csv(output_file)
    logger.info(f"Saved: {output_file}")

    # Display
    logger.info("\n" + str(df_stats))


def main() -> None:
    """Run Phase 4: Outcome Distribution Analysis."""
    logger.info("=" * 80)
    logger.info("PHASE 4: OUTCOME DISTRIBUTION ANALYSIS")
    logger.info("=" * 80)

    # Load data
    df = load_test_data(sample_frac=0.1)

    # Run analyses
    plot_outcome_distributions_by_moneyness(df, FIGURE_DIR / "14_outcome_distributions.png")
    plot_win_rate_analysis(df, FIGURE_DIR / "15_win_rate_analysis.png")
    plot_asymmetry_analysis(df, FIGURE_DIR / "16_asymmetry_analysis.png")
    plot_temporal_patterns(df, FIGURE_DIR / "17_temporal_patterns.png")
    generate_outcome_statistics_table(df)

    logger.info("=" * 80)
    logger.info("PHASE 4 COMPLETE")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
