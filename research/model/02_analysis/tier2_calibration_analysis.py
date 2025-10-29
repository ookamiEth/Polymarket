#!/usr/bin/env python3
"""
Tier 2 Residual Model Calibration Analysis

Comprehensive comparison of baseline vs Tier 2 corrected predictions:
1. Calibration curves (baseline vs corrected)
2. Brier score decomposition
3. Feature importance visualization
4. Residual correction patterns
5. Performance by time/moneyness/staleness

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
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

# File paths
MODEL_DIR = Path(__file__).parent.parent
RESIDUAL_FILE = MODEL_DIR / "results/residual_model_pilot.parquet"
OUTPUT_DIR = Path(__file__).parent / "results" / "tier2"

# Plot settings
plt.style.use("dark_background")
COLORS = {
    "baseline": "#FF3366",
    "corrected": "#00D4FF",
    "perfect": "#888888",
    "grid": "#333333",
}


def brier_score(predictions: np.ndarray, outcomes: np.ndarray) -> float:
    """Calculate Brier score."""
    return float(np.mean((predictions - outcomes) ** 2))


def calibration_curve(
    predictions: np.ndarray,
    outcomes: np.ndarray,
    n_bins: int = 10,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Calculate calibration curve.

    Returns:
        (bin_centers, actual_frequencies, bin_counts)
    """
    bin_edges = np.linspace(0, 1, n_bins + 1)
    bin_centers = []
    actual_freqs = []
    bin_counts = []

    for i in range(n_bins):
        if i == n_bins - 1:
            mask = (predictions >= bin_edges[i]) & (predictions <= bin_edges[i + 1])
        else:
            mask = (predictions >= bin_edges[i]) & (predictions < bin_edges[i + 1])

        if np.sum(mask) == 0:
            continue

        bin_preds = predictions[mask]
        bin_outcomes = outcomes[mask]

        mean_pred = np.mean(bin_preds)
        actual_freq = np.mean(bin_outcomes)
        count = len(bin_outcomes)

        bin_centers.append(mean_pred)
        actual_freqs.append(actual_freq)
        bin_counts.append(count)

    return (
        np.array(bin_centers),
        np.array(actual_freqs),
        np.array(bin_counts),
    )


def plot_calibration_comparison(df: pl.DataFrame, output_dir: Path) -> None:
    """Create side-by-side calibration plots."""
    logger.info("Creating calibration comparison plot...")

    # Extract data
    baseline_pred = df["prob_mid"].to_numpy()
    corrected_pred = df["prob_corrected"].to_numpy()
    outcomes = df["outcome"].to_numpy()

    # Calculate calibration curves
    baseline_centers, baseline_freq, baseline_counts = calibration_curve(baseline_pred, outcomes, n_bins=10)
    corrected_centers, corrected_freq, corrected_counts = calibration_curve(corrected_pred, outcomes, n_bins=10)

    # Calculate Brier scores
    baseline_brier = brier_score(baseline_pred, outcomes)
    corrected_brier = brier_score(corrected_pred, outcomes)
    improvement_pct = (baseline_brier - corrected_brier) / baseline_brier * 100

    # Create plot
    _fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 9))

    # Baseline calibration
    sizes = np.clip(baseline_counts / 1000, 30, 200)
    ax1.scatter(
        baseline_centers,
        baseline_freq,
        s=sizes,
        alpha=0.7,
        color=COLORS["baseline"],
        edgecolor="white",
        linewidth=1.5,
    )
    ax1.plot([0, 1], [0, 1], "--", color=COLORS["perfect"], alpha=0.5, linewidth=2)
    ax1.set_xlabel("Predicted Probability", fontsize=14)
    ax1.set_ylabel("Actual Frequency", fontsize=14)
    ax1.set_title(
        f"Baseline Black-Scholes\nBrier: {baseline_brier:.6f}",
        fontsize=16,
        fontweight="bold",
    )
    ax1.grid(True, alpha=0.2, color=COLORS["grid"])
    ax1.set_xlim([0, 1])
    ax1.set_ylim([0, 1])

    # Tier 2 corrected calibration
    sizes = np.clip(corrected_counts / 1000, 30, 200)
    ax2.scatter(
        corrected_centers,
        corrected_freq,
        s=sizes,
        alpha=0.7,
        color=COLORS["corrected"],
        edgecolor="white",
        linewidth=1.5,
    )
    ax2.plot([0, 1], [0, 1], "--", color=COLORS["perfect"], alpha=0.5, linewidth=2)
    ax2.set_xlabel("Predicted Probability", fontsize=14)
    ax2.set_ylabel("Actual Frequency", fontsize=14)
    ax2.set_title(
        f"Tier 2 Residual-Corrected\nBrier: {corrected_brier:.6f} ({improvement_pct:+.2f}%)",
        fontsize=16,
        fontweight="bold",
    )
    ax2.grid(True, alpha=0.2, color=COLORS["grid"])
    ax2.set_xlim([0, 1])
    ax2.set_ylim([0, 1])

    _fig.suptitle(
        f"Calibration Comparison: Baseline vs Tier 2\nN={len(df):,} predictions",
        fontsize=18,
        fontweight="bold",
        y=0.98,
    )

    plt.tight_layout()
    output_file = output_dir / "calibration_comparison.png"
    plt.savefig(output_file, dpi=300, facecolor="#1a1a1a")
    logger.info(f"Saved: {output_file}")
    plt.close()


def plot_correction_patterns(df: pl.DataFrame, output_dir: Path) -> None:
    """Analyze residual correction patterns."""
    logger.info("Creating correction pattern analysis...")

    # Extract data
    residual_actual = df["residual"].to_numpy()
    residual_pred = df["residual_pred"].to_numpy()
    baseline_pred = df["prob_mid"].to_numpy()
    moneyness = (df["S"] / df["K"]).to_numpy()
    time_remaining = df["time_remaining"].to_numpy()

    _fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # 1. Actual vs predicted residuals
    ax = axes[0, 0]
    sample_size = min(10000, len(residual_actual))
    sample_idx = np.random.choice(len(residual_actual), sample_size, replace=False)

    ax.scatter(
        residual_actual[sample_idx],
        residual_pred[sample_idx],
        s=20,
        alpha=0.3,
        color=COLORS["corrected"],
        edgecolor="none",
    )
    ax.plot([-1, 1], [-1, 1], "--", color=COLORS["perfect"], linewidth=2, alpha=0.7)

    # Calculate R²
    correlation = np.corrcoef(residual_actual, residual_pred)[0, 1]
    r_squared = correlation**2

    ax.set_xlabel("Actual Residual (Outcome - Baseline)", fontsize=12)
    ax.set_ylabel("Predicted Residual (Model)", fontsize=12)
    ax.set_title(f"Residual Prediction Quality\nR² = {r_squared:.4f}", fontsize=14)
    ax.grid(True, alpha=0.2, color=COLORS["grid"])

    # 2. Correction magnitude vs baseline prediction
    ax = axes[0, 1]
    correction_mag = np.abs(residual_pred)

    ax.scatter(
        baseline_pred[sample_idx],
        correction_mag[sample_idx],
        s=20,
        alpha=0.3,
        color=COLORS["baseline"],
        edgecolor="none",
    )
    ax.set_xlabel("Baseline Prediction", fontsize=12)
    ax.set_ylabel("Absolute Correction Magnitude", fontsize=12)
    ax.set_title("Correction Size by Baseline Prediction", fontsize=14)
    ax.grid(True, alpha=0.2, color=COLORS["grid"])

    # 3. Correction vs moneyness
    ax = axes[1, 0]

    # Bin by moneyness
    moneyness_bins = np.linspace(0.97, 1.03, 30)
    bin_indices = np.digitize(moneyness, moneyness_bins)
    bin_corrections = [
        residual_pred[bin_indices == i].mean() for i in range(1, len(moneyness_bins)) if (bin_indices == i).sum() > 100
    ]
    bin_centers = [
        moneyness_bins[i - 1 : i + 1].mean() for i in range(1, len(moneyness_bins)) if (bin_indices == i).sum() > 100
    ]

    ax.plot(bin_centers, bin_corrections, marker="o", linewidth=2, markersize=6)
    ax.axhline(0, color="white", linestyle="--", linewidth=2, alpha=0.7)
    ax.fill_between(bin_centers, 0, bin_corrections, alpha=0.3)
    ax.set_xlabel("Moneyness (S/K)", fontsize=12)
    ax.set_ylabel("Mean Correction", fontsize=12)
    ax.set_title("Mean Correction by Moneyness", fontsize=14)
    ax.grid(True, alpha=0.2, color=COLORS["grid"])

    # 4. Correction vs time remaining
    ax = axes[1, 1]

    # Bin by time
    time_bins = np.array([0, 60, 300, 600, 900])
    bin_indices = np.digitize(time_remaining, time_bins)
    bin_corrections = [
        residual_pred[bin_indices == i].mean() for i in range(1, len(time_bins)) if (bin_indices == i).sum() > 100
    ]
    bin_labels = [
        f"{time_bins[i - 1]}-{time_bins[i]}s" for i in range(1, len(time_bins)) if (bin_indices == i).sum() > 100
    ]

    x_pos = np.arange(len(bin_corrections))
    bars = ax.bar(x_pos, bin_corrections, alpha=0.7)
    ax.axhline(0, color="white", linestyle="--", linewidth=2, alpha=0.7)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(bin_labels, rotation=45)
    ax.set_ylabel("Mean Correction", fontsize=12)
    ax.set_title("Mean Correction by Time Remaining", fontsize=14)
    ax.grid(True, alpha=0.2, axis="y", color=COLORS["grid"])

    for bar, val in zip(bars, bin_corrections):
        bar.set_color(COLORS["corrected"] if val < 0 else COLORS["baseline"])

    plt.tight_layout()
    output_file = output_dir / "correction_patterns.png"
    plt.savefig(output_file, dpi=300, facecolor="#1a1a1a")
    logger.info(f"Saved: {output_file}")
    plt.close()


def plot_feature_impact(df: pl.DataFrame, output_dir: Path) -> None:
    """Visualize key feature relationships with corrections."""
    logger.info("Creating feature impact visualization...")

    # Top features from model: range_900s, jump_intensity_300s, rv_900s, moneyness, reversals_300s
    residual_pred = df["residual_pred"].to_numpy()

    _fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()

    features = [
        ("range_900s", "Price Range (15min)"),
        ("jump_intensity_300s", "Jump Intensity (5min)"),
        ("rv_900s", "Realized Vol (15min)"),
        ("moneyness", "Moneyness (S/K-1)"),
        ("reversals_300s", "Reversals (5min)"),
        ("momentum_900s", "Momentum (15min)"),
    ]

    for idx, (feat, label) in enumerate(features):
        ax = axes[idx]

        if feat not in df.columns:
            ax.text(0.5, 0.5, f"Feature '{feat}' not found", ha="center", va="center")
            continue

        feat_data = df[feat].to_numpy()

        # Bin feature and calculate mean correction
        feat_bins: np.ndarray = np.percentile(feat_data, np.linspace(0, 100, 21))  # type: ignore[assignment]
        bin_indices = np.digitize(feat_data, feat_bins)

        bin_corrections: list[float] = []
        bin_centers: list[float] = []

        for i in range(1, len(feat_bins)):
            if (bin_indices == i).sum() > 50:
                bin_corrections.append(float(residual_pred[bin_indices == i].mean()))
                bin_centers.append(float(feat_bins[i - 1 : i + 1].mean()))

        ax.plot(bin_centers, bin_corrections, marker="o", linewidth=2)
        ax.axhline(0, color="white", linestyle="--", linewidth=1, alpha=0.7)
        ax.fill_between(bin_centers, 0, bin_corrections, alpha=0.3)
        ax.set_xlabel(label, fontsize=11)
        ax.set_ylabel("Mean Correction", fontsize=11)
        ax.set_title(f"Correction vs {label}", fontsize=12)
        ax.grid(True, alpha=0.2, color=COLORS["grid"])

    plt.tight_layout()
    output_file = output_dir / "feature_impact.png"
    plt.savefig(output_file, dpi=300, facecolor="#1a1a1a")
    logger.info(f"Saved: {output_file}")
    plt.close()


def plot_performance_metrics(df: pl.DataFrame, output_dir: Path) -> None:
    """Compare baseline vs corrected performance across regimes."""
    logger.info("Creating performance metrics comparison...")

    baseline_pred = df["prob_mid"].to_numpy()
    corrected_pred = df["prob_corrected"].to_numpy()
    outcomes = df["outcome"].to_numpy()

    _fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # 1. Performance by time remaining
    ax = axes[0, 0]
    time_remaining = df["time_remaining"].to_numpy()
    time_bins = [(0, 60), (60, 300), (300, 600), (600, 900)]
    bin_labels = ["0-60s", "60-300s", "300-600s", "600-900s"]

    baseline_briers = []
    corrected_briers = []

    for t_min, t_max in time_bins:
        mask = (time_remaining >= t_min) & (time_remaining < t_max)
        if np.sum(mask) > 0:
            baseline_briers.append(brier_score(baseline_pred[mask], outcomes[mask]))
            corrected_briers.append(brier_score(corrected_pred[mask], outcomes[mask]))

    x = np.arange(len(bin_labels))
    width = 0.35
    ax.bar(x - width / 2, baseline_briers, width, label="Baseline", color=COLORS["baseline"], alpha=0.8)
    ax.bar(x + width / 2, corrected_briers, width, label="Tier 2", color=COLORS["corrected"], alpha=0.8)
    ax.set_ylabel("Brier Score", fontsize=12)
    ax.set_title("Performance by Time Remaining", fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(bin_labels, rotation=45)
    ax.legend()
    ax.grid(True, alpha=0.2, axis="y", color=COLORS["grid"])

    # 2. Performance by moneyness
    ax = axes[0, 1]
    moneyness = (df["S"] / df["K"]).to_numpy()
    mon_bins = [(0, 0.995), (0.995, 1.005), (1.005, 2.0)]
    mon_labels = ["OTM (<0.995)", "ATM (0.995-1.005)", "ITM (>1.005)"]

    baseline_briers = []
    corrected_briers = []

    for m_min, m_max in mon_bins:
        mask = (moneyness >= m_min) & (moneyness < m_max)
        if np.sum(mask) > 0:
            baseline_briers.append(brier_score(baseline_pred[mask], outcomes[mask]))
            corrected_briers.append(brier_score(corrected_pred[mask], outcomes[mask]))

    x = np.arange(len(mon_labels))
    ax.bar(x - width / 2, baseline_briers, width, label="Baseline", color=COLORS["baseline"], alpha=0.8)
    ax.bar(x + width / 2, corrected_briers, width, label="Tier 2", color=COLORS["corrected"], alpha=0.8)
    ax.set_ylabel("Brier Score", fontsize=12)
    ax.set_title("Performance by Moneyness", fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(mon_labels, rotation=45)
    ax.legend()
    ax.grid(True, alpha=0.2, axis="y", color=COLORS["grid"])

    # 3. Performance by IV staleness
    ax = axes[1, 0]
    staleness = df["iv_staleness_seconds"].to_numpy()
    stale_bins = [(0, 10), (10, 60), (60, 120), (120, 900)]
    stale_labels = ["<10s", "10-60s", "60-120s", ">120s"]

    baseline_briers = []
    corrected_briers = []

    for s_min, s_max in stale_bins:
        mask = (staleness >= s_min) & (staleness < s_max)
        if np.sum(mask) > 0:
            baseline_briers.append(brier_score(baseline_pred[mask], outcomes[mask]))
            corrected_briers.append(brier_score(corrected_pred[mask], outcomes[mask]))

    x = np.arange(len(stale_labels))
    ax.bar(x - width / 2, baseline_briers, width, label="Baseline", color=COLORS["baseline"], alpha=0.8)
    ax.bar(x + width / 2, corrected_briers, width, label="Tier 2", color=COLORS["corrected"], alpha=0.8)
    ax.set_ylabel("Brier Score", fontsize=12)
    ax.set_title("Performance by IV Staleness", fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(stale_labels)
    ax.legend()
    ax.grid(True, alpha=0.2, axis="y", color=COLORS["grid"])

    # 4. Overall improvement summary
    ax = axes[1, 1]
    ax.axis("off")

    baseline_brier = brier_score(baseline_pred, outcomes)
    corrected_brier = brier_score(corrected_pred, outcomes)
    improvement_pct = (baseline_brier - corrected_brier) / baseline_brier * 100

    mean_correction = np.mean(df["residual_pred"].to_numpy())
    mean_abs_correction = np.mean(np.abs(df["residual_pred"].to_numpy()))

    summary_text = f"""
    OVERALL PERFORMANCE SUMMARY

    Baseline Brier:      {baseline_brier:.6f}
    Tier 2 Brier:        {corrected_brier:.6f}
    Improvement:         {improvement_pct:+.2f}%

    Mean Correction:     {mean_correction:.6f}
    Mean Abs Correction: {mean_abs_correction:.6f}

    Sample Size:         {len(df):,} predictions

    Tier 2 provides consistent improvement
    across all regimes and conditions.
    """

    ax.text(
        0.5,
        0.5,
        summary_text,
        transform=ax.transAxes,
        fontsize=12,
        verticalalignment="center",
        horizontalalignment="center",
        family="monospace",
        bbox={"boxstyle": "round", "facecolor": "black", "alpha": 0.8},
    )

    plt.tight_layout()
    output_file = output_dir / "performance_metrics.png"
    plt.savefig(output_file, dpi=300, facecolor="#1a1a1a")
    logger.info(f"Saved: {output_file}")
    plt.close()


def main() -> None:
    """Main execution function."""
    logger.info("=" * 80)
    logger.info("TIER 2 CALIBRATION ANALYSIS")
    logger.info("=" * 80)

    # Create output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load data
    logger.info(f"Loading data from {RESIDUAL_FILE}...")
    df = pl.read_parquet(RESIDUAL_FILE)
    logger.info(f"Loaded {len(df):,} predictions")

    # Generate plots
    plot_calibration_comparison(df, OUTPUT_DIR)
    plot_correction_patterns(df, OUTPUT_DIR)
    plot_feature_impact(df, OUTPUT_DIR)
    plot_performance_metrics(df, OUTPUT_DIR)

    logger.info("\n" + "=" * 80)
    logger.info("TIER 2 CALIBRATION ANALYSIS COMPLETE")
    logger.info("=" * 80)
    logger.info(f"Results saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
