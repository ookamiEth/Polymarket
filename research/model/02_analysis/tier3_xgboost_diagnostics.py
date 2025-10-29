#!/usr/bin/env python3
"""
Tier 3 XGBoost Diagnostic Visualizations

Creates comprehensive comparison of:
- Baseline (Black-Scholes with IV)
- Tier 2 (Ridge regression residual model)
- Tier 3 (XGBoost residual model)

Visualizations:
1. Three-way calibration comparison
2. Correction magnitude distribution comparison
3. Feature importance comparison (Ridge coef vs XGBoost gain)
4. Performance by regime (Baseline vs Tier 2 vs Tier 3)

Author: BT Research Team
Date: 2025-10-29
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import polars as pl

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

# File paths
MODEL_DIR = Path(__file__).parent.parent
TIER2_FILE = MODEL_DIR / "results/residual_model_pilot.parquet"
TIER3_FILE = MODEL_DIR / "results/xgboost_residual_model_pilot.parquet"
OUTPUT_DIR = MODEL_DIR / "02_analysis/results/tier3"

# Create output directory
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Plotting style
plt.style.use("dark_background")


def brier_score(predictions: np.ndarray, outcomes: np.ndarray) -> float:
    """Calculate Brier score (MSE for probabilities)."""
    return float(np.mean((predictions - outcomes) ** 2))


def calibration_curve(
    predictions: np.ndarray, outcomes: np.ndarray, n_bins: int = 10
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Calculate calibration curve.

    Returns:
        (bin_centers, bin_frequencies, bin_counts)
    """
    bin_edges = np.linspace(0, 1, n_bins + 1)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    bin_frequencies = []
    bin_counts = []

    for i in range(n_bins):
        mask = (predictions >= bin_edges[i]) & (predictions < bin_edges[i + 1])
        if i == n_bins - 1:  # Include upper bound in last bin
            mask = (predictions >= bin_edges[i]) & (predictions <= bin_edges[i + 1])

        count = mask.sum()
        bin_counts.append(count)

        if count > 0:
            freq = outcomes[mask].mean()
            bin_frequencies.append(freq)
        else:
            bin_frequencies.append(np.nan)

    return (
        np.array(bin_centers),
        np.array(bin_frequencies),
        np.array(bin_counts),
    )


def plot_three_way_calibration(
    df_tier2: pl.DataFrame,
    df_tier3: pl.DataFrame,
    output_dir: Path,
) -> None:
    """Create three-way calibration comparison: Baseline vs Tier 2 vs Tier 3."""
    logger.info("Creating three-way calibration comparison...")

    # Extract predictions and outcomes
    baseline_pred = df_tier2["prob_mid"].to_numpy()
    tier2_pred = df_tier2["prob_corrected"].to_numpy()
    tier3_pred = df_tier3["prob_corrected_xgb"].to_numpy()
    outcomes = df_tier2["outcome"].to_numpy()

    # Calculate calibration curves
    baseline_centers, baseline_freq, baseline_counts = calibration_curve(baseline_pred, outcomes, n_bins=10)
    tier2_centers, tier2_freq, tier2_counts = calibration_curve(tier2_pred, outcomes, n_bins=10)
    tier3_centers, tier3_freq, tier3_counts = calibration_curve(tier3_pred, outcomes, n_bins=10)

    # Calculate Brier scores
    baseline_brier = brier_score(baseline_pred, outcomes)
    tier2_brier = brier_score(tier2_pred, outcomes)
    tier3_brier = brier_score(tier3_pred, outcomes)

    tier2_improvement = (baseline_brier - tier2_brier) / baseline_brier * 100
    tier3_improvement = (baseline_brier - tier3_brier) / baseline_brier * 100

    # Create plot
    _fig, ax = plt.subplots(figsize=(12, 10))

    # Scale dot sizes appropriately
    baseline_sizes = np.clip(baseline_counts / 1000, 30, 200)
    tier2_sizes = np.clip(tier2_counts / 1000, 30, 200)
    tier3_sizes = np.clip(tier3_counts / 1000, 30, 200)

    # Plot calibration curves
    ax.scatter(
        baseline_centers,
        baseline_freq,
        s=baseline_sizes,
        alpha=0.7,
        color="#FF6B6B",
        edgecolor="white",
        linewidth=1.5,
        label=f"Baseline (Brier: {baseline_brier:.4f})",
        zorder=3,
    )

    ax.scatter(
        tier2_centers,
        tier2_freq,
        s=tier2_sizes,
        alpha=0.7,
        color="#4ECDC4",
        edgecolor="white",
        linewidth=1.5,
        label=f"Tier 2 Ridge (Brier: {tier2_brier:.4f}, {tier2_improvement:+.2f}%)",
        zorder=4,
    )

    ax.scatter(
        tier3_centers,
        tier3_freq,
        s=tier3_sizes,
        alpha=0.8,
        color="#95E1D3",
        edgecolor="white",
        linewidth=2,
        label=f"Tier 3 XGBoost (Brier: {tier3_brier:.4f}, {tier3_improvement:+.2f}%)",
        zorder=5,
    )

    # Perfect calibration line
    ax.plot([0, 1], [0, 1], "w--", alpha=0.5, linewidth=1.5, label="Perfect Calibration", zorder=2)

    # Styling
    ax.set_xlabel("Predicted Probability", fontsize=14, fontweight="bold")
    ax.set_ylabel("Observed Frequency", fontsize=14, fontweight="bold")
    ax.set_title(
        f"Three-Way Calibration Comparison (October 2023, N={len(outcomes):,})",
        fontsize=16,
        fontweight="bold",
        pad=20,
    )
    ax.legend(loc="upper left", fontsize=11, framealpha=0.9)
    ax.grid(alpha=0.2)
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.05)

    # Add improvement annotation
    improvement_text = (
        f"Tier 2 Improvement: {tier2_improvement:+.2f}%\n"
        f"Tier 3 Improvement: {tier3_improvement:+.2f}%\n"
        f"Tier 3 vs Tier 2: {(tier2_brier - tier3_brier) / tier2_brier * 100:+.2f}%"
    )
    ax.text(
        0.05,
        0.95,
        improvement_text,
        transform=ax.transAxes,
        fontsize=10,
        verticalalignment="top",
        bbox={"boxstyle": "round", "facecolor": "black", "alpha": 0.8},
    )

    plt.tight_layout()
    output_file = output_dir / "three_way_calibration.png"
    plt.savefig(output_file, dpi=150, bbox_inches="tight")
    plt.close()

    logger.info(f"Saved three-way calibration plot to {output_file}")


def plot_correction_distribution_comparison(
    df_tier2: pl.DataFrame,
    df_tier3: pl.DataFrame,
    output_dir: Path,
) -> None:
    """Compare correction distributions: Tier 2 vs Tier 3."""
    logger.info("Creating correction distribution comparison...")

    # Extract corrections
    tier2_corrections = df_tier2["residual_pred"].to_numpy()
    tier3_corrections = df_tier3["residual_pred_xgb"].to_numpy()

    # Create figure
    _fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # 1. Histogram comparison
    ax = axes[0, 0]
    ax.hist(
        tier2_corrections,
        bins=50,
        alpha=0.6,
        color="#4ECDC4",
        edgecolor="white",
        linewidth=0.5,
        label="Tier 2 Ridge",
    )
    ax.hist(
        tier3_corrections,
        bins=50,
        alpha=0.6,
        color="#95E1D3",
        edgecolor="white",
        linewidth=0.5,
        label="Tier 3 XGBoost",
    )
    ax.axvline(0, color="white", linestyle="--", alpha=0.5, linewidth=1)
    ax.set_xlabel("Correction (pp)", fontsize=12, fontweight="bold")
    ax.set_ylabel("Frequency", fontsize=12, fontweight="bold")
    ax.set_title("Correction Distribution", fontsize=14, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(alpha=0.2)

    # Add statistics
    tier2_stats = (
        f"Mean: {tier2_corrections.mean():.4f}\n"
        f"Std: {tier2_corrections.std():.4f}\n"
        f"Range: [{tier2_corrections.min():.3f}, {tier2_corrections.max():.3f}]"
    )
    tier3_stats = (
        f"Mean: {tier3_corrections.mean():.4f}\n"
        f"Std: {tier3_corrections.std():.4f}\n"
        f"Range: [{tier3_corrections.min():.3f}, {tier3_corrections.max():.3f}]"
    )
    ax.text(
        0.02,
        0.98,
        f"Tier 2:\n{tier2_stats}",
        transform=ax.transAxes,
        fontsize=9,
        verticalalignment="top",
        bbox={"boxstyle": "round", "facecolor": "#4ECDC4", "alpha": 0.3},
    )
    ax.text(
        0.65,
        0.98,
        f"Tier 3:\n{tier3_stats}",
        transform=ax.transAxes,
        fontsize=9,
        verticalalignment="top",
        bbox={"boxstyle": "round", "facecolor": "#95E1D3", "alpha": 0.3},
    )

    # 2. Absolute correction comparison
    ax = axes[0, 1]
    tier2_abs = np.abs(tier2_corrections)
    tier3_abs = np.abs(tier3_corrections)
    ax.hist(
        tier2_abs,
        bins=50,
        alpha=0.6,
        color="#4ECDC4",
        edgecolor="white",
        linewidth=0.5,
        label=f"Tier 2 (Mean: {tier2_abs.mean():.4f})",
    )
    ax.hist(
        tier3_abs,
        bins=50,
        alpha=0.6,
        color="#95E1D3",
        edgecolor="white",
        linewidth=0.5,
        label=f"Tier 3 (Mean: {tier3_abs.mean():.4f})",
    )
    ax.set_xlabel("Absolute Correction (pp)", fontsize=12, fontweight="bold")
    ax.set_ylabel("Frequency", fontsize=12, fontweight="bold")
    ax.set_title(
        f"Magnitude Comparison (Tier 3 is {tier3_abs.mean() / tier2_abs.mean():.1f}x larger)",
        fontsize=14,
        fontweight="bold",
    )
    ax.legend(fontsize=10)
    ax.grid(alpha=0.2)

    # 3. Scatter: Tier 2 vs Tier 3 corrections
    ax = axes[1, 0]
    ax.scatter(tier2_corrections, tier3_corrections, s=1, alpha=0.3, color="#95E1D3")
    ax.plot(
        [tier2_corrections.min(), tier2_corrections.max()],
        [tier2_corrections.min(), tier2_corrections.max()],
        "w--",
        alpha=0.5,
        linewidth=1.5,
        label="y=x (Perfect Agreement)",
    )
    ax.axhline(0, color="white", linestyle="--", alpha=0.3, linewidth=1)
    ax.axvline(0, color="white", linestyle="--", alpha=0.3, linewidth=1)
    ax.set_xlabel("Tier 2 Correction (pp)", fontsize=12, fontweight="bold")
    ax.set_ylabel("Tier 3 Correction (pp)", fontsize=12, fontweight="bold")
    ax.set_title("Correction Correlation", fontsize=14, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(alpha=0.2)

    # Calculate correlation
    correlation = np.corrcoef(tier2_corrections, tier3_corrections)[0, 1]
    ax.text(
        0.05,
        0.95,
        f"Correlation: {correlation:.3f}",
        transform=ax.transAxes,
        fontsize=11,
        verticalalignment="top",
        bbox={"boxstyle": "round", "facecolor": "black", "alpha": 0.8},
    )

    # 4. Percentile comparison
    ax = axes[1, 1]
    percentiles = np.arange(0, 101, 5)
    tier2_percentiles = np.percentile(tier2_corrections, percentiles)
    tier3_percentiles = np.percentile(tier3_corrections, percentiles)

    ax.plot(percentiles, tier2_percentiles, linewidth=2, color="#4ECDC4", label="Tier 2 Ridge")
    ax.plot(percentiles, tier3_percentiles, linewidth=2, color="#95E1D3", label="Tier 3 XGBoost")
    ax.axhline(0, color="white", linestyle="--", alpha=0.5, linewidth=1)
    ax.set_xlabel("Percentile", fontsize=12, fontweight="bold")
    ax.set_ylabel("Correction (pp)", fontsize=12, fontweight="bold")
    ax.set_title("Correction Percentiles", fontsize=14, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(alpha=0.2)

    plt.suptitle(
        "Correction Distribution Comparison: Tier 2 vs Tier 3",
        fontsize=16,
        fontweight="bold",
        y=0.995,
    )
    plt.tight_layout()

    output_file = output_dir / "correction_distribution_comparison.png"
    plt.savefig(output_file, dpi=150, bbox_inches="tight")
    plt.close()

    logger.info(f"Saved correction distribution comparison to {output_file}")


def plot_performance_by_regime(
    df_tier2: pl.DataFrame,
    df_tier3: pl.DataFrame,
    output_dir: Path,
) -> None:
    """Compare performance across different market regimes."""
    logger.info("Creating performance by regime comparison...")

    # Create figure
    _fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # 1. Performance by moneyness
    ax = axes[0, 0]
    moneyness = df_tier2["moneyness"].to_numpy()
    moneyness_bins: np.ndarray = np.percentile(moneyness, np.linspace(0, 100, 11))  # type: ignore[assignment]

    baseline_brier_by_moneyness: list[float] = []
    tier2_brier_by_moneyness: list[float] = []
    tier3_brier_by_moneyness: list[float] = []
    moneyness_centers: list[float] = []

    for i in range(1, len(moneyness_bins)):
        mask = (moneyness >= moneyness_bins[i - 1]) & (moneyness < moneyness_bins[i])
        if mask.sum() > 100:
            baseline_brier_by_moneyness.append(
                brier_score(df_tier2["prob_mid"].to_numpy()[mask], df_tier2["outcome"].to_numpy()[mask])
            )
            tier2_brier_by_moneyness.append(
                brier_score(
                    df_tier2["prob_corrected"].to_numpy()[mask],
                    df_tier2["outcome"].to_numpy()[mask],
                )
            )
            tier3_brier_by_moneyness.append(
                brier_score(
                    df_tier3["prob_corrected_xgb"].to_numpy()[mask],
                    df_tier3["outcome"].to_numpy()[mask],
                )
            )
            moneyness_centers.append(float((moneyness_bins[i - 1] + moneyness_bins[i]) / 2))

    ax.plot(
        moneyness_centers,
        baseline_brier_by_moneyness,
        linewidth=2,
        color="#FF6B6B",
        marker="o",
        label="Baseline",
    )
    ax.plot(
        moneyness_centers,
        tier2_brier_by_moneyness,
        linewidth=2,
        color="#4ECDC4",
        marker="s",
        label="Tier 2 Ridge",
    )
    ax.plot(
        moneyness_centers,
        tier3_brier_by_moneyness,
        linewidth=2,
        color="#95E1D3",
        marker="^",
        label="Tier 3 XGBoost",
    )
    ax.set_xlabel("Moneyness (S/K - 1)", fontsize=12, fontweight="bold")
    ax.set_ylabel("Brier Score", fontsize=12, fontweight="bold")
    ax.set_title("Performance by Moneyness", fontsize=14, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(alpha=0.2)

    # 2. Performance by time remaining
    ax = axes[0, 1]
    time_remaining = df_tier2["time_remaining"].to_numpy()
    time_bins: np.ndarray = np.percentile(time_remaining, np.linspace(0, 100, 11))  # type: ignore[assignment]

    baseline_brier_by_time: list[float] = []
    tier2_brier_by_time: list[float] = []
    tier3_brier_by_time: list[float] = []
    time_centers: list[float] = []

    for i in range(1, len(time_bins)):
        mask = (time_remaining >= time_bins[i - 1]) & (time_remaining < time_bins[i])
        if mask.sum() > 100:
            baseline_brier_by_time.append(
                brier_score(df_tier2["prob_mid"].to_numpy()[mask], df_tier2["outcome"].to_numpy()[mask])
            )
            tier2_brier_by_time.append(
                brier_score(
                    df_tier2["prob_corrected"].to_numpy()[mask],
                    df_tier2["outcome"].to_numpy()[mask],
                )
            )
            tier3_brier_by_time.append(
                brier_score(
                    df_tier3["prob_corrected_xgb"].to_numpy()[mask],
                    df_tier3["outcome"].to_numpy()[mask],
                )
            )
            time_centers.append(float((time_bins[i - 1] + time_bins[i]) / 2))

    ax.plot(
        time_centers,
        baseline_brier_by_time,
        linewidth=2,
        color="#FF6B6B",
        marker="o",
        label="Baseline",
    )
    ax.plot(time_centers, tier2_brier_by_time, linewidth=2, color="#4ECDC4", marker="s", label="Tier 2 Ridge")
    ax.plot(
        time_centers,
        tier3_brier_by_time,
        linewidth=2,
        color="#95E1D3",
        marker="^",
        label="Tier 3 XGBoost",
    )
    ax.set_xlabel("Time Remaining (seconds)", fontsize=12, fontweight="bold")
    ax.set_ylabel("Brier Score", fontsize=12, fontweight="bold")
    ax.set_title("Performance by Time to Expiry", fontsize=14, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(alpha=0.2)

    # 3. Performance by realized volatility
    ax = axes[1, 0]
    rv_3600s = df_tier2["rv_3600s"].to_numpy()
    rv_bins: np.ndarray = np.percentile(rv_3600s, np.linspace(0, 100, 11))  # type: ignore[assignment]

    baseline_brier_by_rv: list[float] = []
    tier2_brier_by_rv: list[float] = []
    tier3_brier_by_rv: list[float] = []
    rv_centers: list[float] = []

    for i in range(1, len(rv_bins)):
        mask = (rv_3600s >= rv_bins[i - 1]) & (rv_3600s < rv_bins[i])
        if mask.sum() > 100:
            baseline_brier_by_rv.append(
                brier_score(df_tier2["prob_mid"].to_numpy()[mask], df_tier2["outcome"].to_numpy()[mask])
            )
            tier2_brier_by_rv.append(
                brier_score(
                    df_tier2["prob_corrected"].to_numpy()[mask],
                    df_tier2["outcome"].to_numpy()[mask],
                )
            )
            tier3_brier_by_rv.append(
                brier_score(
                    df_tier3["prob_corrected_xgb"].to_numpy()[mask],
                    df_tier3["outcome"].to_numpy()[mask],
                )
            )
            rv_centers.append(float((rv_bins[i - 1] + rv_bins[i]) / 2))

    ax.plot(rv_centers, baseline_brier_by_rv, linewidth=2, color="#FF6B6B", marker="o", label="Baseline")
    ax.plot(rv_centers, tier2_brier_by_rv, linewidth=2, color="#4ECDC4", marker="s", label="Tier 2 Ridge")
    ax.plot(rv_centers, tier3_brier_by_rv, linewidth=2, color="#95E1D3", marker="^", label="Tier 3 XGBoost")
    ax.set_xlabel("1-Hour Realized Volatility", fontsize=12, fontweight="bold")
    ax.set_ylabel("Brier Score", fontsize=12, fontweight="bold")
    ax.set_title("Performance by Realized Volatility", fontsize=14, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(alpha=0.2)

    # 4. Overall improvement summary
    ax = axes[1, 1]
    categories = ["Overall", "Moneyness", "Time", "RV"]

    baseline_avg = [
        brier_score(df_tier2["prob_mid"].to_numpy(), df_tier2["outcome"].to_numpy()),
        np.mean(baseline_brier_by_moneyness),
        np.mean(baseline_brier_by_time),
        np.mean(baseline_brier_by_rv),
    ]
    tier2_avg = [
        brier_score(df_tier2["prob_corrected"].to_numpy(), df_tier2["outcome"].to_numpy()),
        np.mean(tier2_brier_by_moneyness),
        np.mean(tier2_brier_by_time),
        np.mean(tier2_brier_by_rv),
    ]
    tier3_avg = [
        brier_score(df_tier3["prob_corrected_xgb"].to_numpy(), df_tier3["outcome"].to_numpy()),
        np.mean(tier3_brier_by_moneyness),
        np.mean(tier3_brier_by_time),
        np.mean(tier3_brier_by_rv),
    ]

    x_pos = np.arange(len(categories))
    width = 0.25

    ax.bar(x_pos - width, baseline_avg, width, label="Baseline", color="#FF6B6B", alpha=0.8)
    ax.bar(x_pos, tier2_avg, width, label="Tier 2 Ridge", color="#4ECDC4", alpha=0.8)
    ax.bar(x_pos + width, tier3_avg, width, label="Tier 3 XGBoost", color="#95E1D3", alpha=0.8)

    ax.set_xlabel("Regime", fontsize=12, fontweight="bold")
    ax.set_ylabel("Average Brier Score", fontsize=12, fontweight="bold")
    ax.set_title("Average Performance by Regime", fontsize=14, fontweight="bold")
    ax.set_xticks(x_pos)
    ax.set_xticklabels(categories)
    ax.legend(fontsize=10)
    ax.grid(alpha=0.2, axis="y")

    # Add improvement percentages
    for i, _cat in enumerate(categories):
        tier2_improvement = (baseline_avg[i] - tier2_avg[i]) / baseline_avg[i] * 100
        tier3_improvement = (baseline_avg[i] - tier3_avg[i]) / baseline_avg[i] * 100
        ax.text(
            i,
            max(baseline_avg[i], tier2_avg[i], tier3_avg[i]) * 1.02,
            f"T2: {tier2_improvement:+.1f}%\nT3: {tier3_improvement:+.1f}%",
            ha="center",
            fontsize=8,
        )

    plt.suptitle(
        "Performance Comparison Across Market Regimes",
        fontsize=16,
        fontweight="bold",
        y=0.995,
    )
    plt.tight_layout()

    output_file = output_dir / "performance_by_regime.png"
    plt.savefig(output_file, dpi=150, bbox_inches="tight")
    plt.close()

    logger.info(f"Saved performance by regime comparison to {output_file}")


def main() -> None:
    """Generate all Tier 3 diagnostic visualizations."""
    logger.info("=" * 80)
    logger.info("TIER 3 XGBOOST DIAGNOSTIC VISUALIZATIONS")
    logger.info("=" * 80)

    # Load data
    logger.info(f"Loading Tier 2 results from {TIER2_FILE}...")
    df_tier2 = pl.read_parquet(TIER2_FILE)
    logger.info(f"Loaded {len(df_tier2):,} rows")

    logger.info(f"Loading Tier 3 results from {TIER3_FILE}...")
    df_tier3 = pl.read_parquet(TIER3_FILE)
    logger.info(f"Loaded {len(df_tier3):,} rows")

    # Verify alignment
    assert len(df_tier2) == len(df_tier3), "Tier 2 and Tier 3 datasets must have same length"
    assert (df_tier2["contract_id"] == df_tier3["contract_id"]).all(), "Contract IDs must match"
    assert (df_tier2["timestamp"] == df_tier3["timestamp"]).all(), "Timestamps must match"

    logger.info("Datasets aligned successfully")

    # Generate visualizations
    plot_three_way_calibration(df_tier2, df_tier3, OUTPUT_DIR)
    plot_correction_distribution_comparison(df_tier2, df_tier3, OUTPUT_DIR)
    plot_performance_by_regime(df_tier2, df_tier3, OUTPUT_DIR)

    logger.info("=" * 80)
    logger.info("COMPLETE")
    logger.info("=" * 80)
    logger.info(f"All visualizations saved to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
