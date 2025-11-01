#!/usr/bin/env python3
"""
Time-series specific visualizations for LightGBM residual prediction model.

Provides plots for:
- Brier score evolution over time
- Calibration by volatility regime
- Prediction distribution analysis
- Temporal performance breakdown
"""

import logging
from typing import Any, Optional

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import seaborn as sns

from visualization.plot_config import (
    COLORS,
    FONT_SIZES,
    REGIME_COLORS,
    apply_plot_style,
    get_plot_output_path,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def plot_brier_over_time(
    test_df: pl.DataFrame,
    window: str = "1d",
    output_path: Optional[str] = None,
    wandb_log: bool = True,
) -> None:
    """
    Plot Brier score evolution over time to detect regime shifts.

    Args:
        test_df: Test set predictions with columns:
            - date: Date column
            - final_prob: Model predictions
            - outcome: Ground truth (0/1)
            - prob_mid: Baseline predictions
        window: Aggregation window ('1d', '1w', etc.)
        output_path: Output file path (auto-generated if None)
        wandb_log: Whether to upload to W&B
    """
    apply_plot_style()

    logger.info(f"Generating Brier score time series (window={window})...")

    # Convert date to datetime if it's numeric (Unix timestamp)
    if test_df["date"].dtype in [pl.Int32, pl.Int64, pl.UInt32, pl.UInt64]:
        test_df = test_df.with_columns([
            pl.from_epoch(pl.col("date"), time_unit="s").alias("date")
        ])

    # Aggregate by time window
    daily_metrics = (
        test_df.group_by_dynamic("date", every=window)
        .agg(
            [
                ((pl.col("final_prob") - pl.col("outcome")) ** 2).mean().alias("model_brier"),
                ((pl.col("prob_mid") - pl.col("outcome")) ** 2).mean().alias("baseline_brier"),
                pl.len().alias("n_samples"),
            ]
        )
        .sort("date")
    )

    # Extract data
    dates = daily_metrics["date"].to_numpy()
    model_brier = daily_metrics["model_brier"].to_numpy()
    baseline_brier = daily_metrics["baseline_brier"].to_numpy()
    improvement = ((baseline_brier - model_brier) / baseline_brier) * 100

    # Create figure
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), sharex=True)

    # Plot 1: Brier scores
    ax1.plot(
        dates,
        baseline_brier,
        label="Baseline (IV-based)",
        linewidth=2.5,
        alpha=0.8,
        color=COLORS["danger"],
        marker="o",
        markersize=4,
    )
    ax1.plot(
        dates,
        model_brier,
        label="Model (Residual)",
        linewidth=2.5,
        alpha=0.9,
        color=COLORS["primary"],
        marker="s",
        markersize=4,
    )

    # Shade improvement region
    ax1.fill_between(dates, model_brier, baseline_brier, alpha=0.15, color=COLORS["success"])

    ax1.set_ylabel("Brier Score (lower is better)", fontsize=FONT_SIZES["label"])
    ax1.set_title(f"Brier Score Over Time ({window} aggregation)", fontsize=FONT_SIZES["title"])
    ax1.legend(frameon=True, fontsize=FONT_SIZES["legend"])
    ax1.grid(alpha=0.2)

    # Plot 2: Improvement percentage
    ax2.plot(
        dates,
        improvement,
        linewidth=2.5,
        alpha=0.9,
        color=COLORS["success"],
        marker="D",
        markersize=4,
    )
    ax2.axhline(y=0, color=COLORS["danger"], linestyle="--", linewidth=1.5, alpha=0.5, label="No improvement")

    # Shade positive improvement
    ax2.fill_between(dates, 0, improvement, where=(improvement > 0), alpha=0.2, color=COLORS["success"])
    ax2.fill_between(dates, 0, improvement, where=(improvement <= 0), alpha=0.2, color=COLORS["danger"])

    ax2.set_xlabel("Date", fontsize=FONT_SIZES["label"])
    ax2.set_ylabel("Improvement (%)", fontsize=FONT_SIZES["label"])
    ax2.set_title("Model Improvement Over Time", fontsize=FONT_SIZES["title"])
    ax2.legend(frameon=True, fontsize=FONT_SIZES["legend"])
    ax2.grid(alpha=0.2)

    # Add statistics
    stats_text = (
        f"Overall Improvement: {improvement.mean():.2f}%\n"
        f"Best: {improvement.max():.2f}% | Worst: {improvement.min():.2f}%\n"
        f"Std Dev: {improvement.std():.2f}%"
    )
    ax2.text(
        0.02,
        0.98,
        stats_text,
        transform=ax2.transAxes,
        fontsize=FONT_SIZES["annotation"],
        verticalalignment="top",
        bbox={"boxstyle": "round", "alpha": 0.15, "facecolor": "white"},
    )

    plt.tight_layout()

    # Save
    if output_path is None:
        window_str = window.replace("d", "daily").replace("w", "weekly")
        output_path = str(get_plot_output_path("time_series", f"brier_over_time_{window_str}.png"))

    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    logger.info(f"✓ Saved Brier time series to {output_path}")

    # Upload to W&B
    if wandb_log:
        try:
            import wandb

            if wandb.run:
                wandb.log({f"time_series/brier_over_time_{window}": wandb.Image(fig)})
                logger.info("✓ Uploaded to W&B")
        except ImportError:
            logger.warning("wandb not available - skipping upload")

    plt.close(fig)

    # Log statistics
    logger.info(f"Brier Time Series Statistics ({window}):")
    logger.info(f"  Average improvement: {improvement.mean():.2f}%")
    logger.info(f"  Best period: {improvement.max():.2f}%")
    logger.info(f"  Worst period: {improvement.min():.2f}%")
    if improvement.min() < 0:
        logger.warning(f"⚠️  Model WORSE than baseline in some periods (min: {improvement.min():.2f}%)")


def plot_calibration_by_regime(
    test_df: pl.DataFrame,
    regime_col: str = "rv_300s",
    output_path: Optional[str] = None,
    wandb_log: bool = True,
) -> None:
    """
    Plot calibration curves stratified by volatility regime.

    Args:
        test_df: Test set with predictions and outcomes
        regime_col: Column to use for regime stratification (default: rv_300s)
        output_path: Output file path (auto-generated if None)
        wandb_log: Whether to upload to W&B
    """
    apply_plot_style()

    logger.info(f"Generating calibration by regime (regime_col={regime_col})...")

    # Calculate regime quantiles (tertiles)
    vol_q33 = test_df[regime_col].quantile(0.33)
    vol_q67 = test_df[regime_col].quantile(0.67)

    if vol_q33 is None or vol_q67 is None:
        logger.error("Failed to calculate quantiles")
        return

    # Create regime segments
    regimes = {
        "low_vol": test_df.filter(pl.col(regime_col) <= vol_q33),
        "mid_vol": test_df.filter((pl.col(regime_col) > vol_q33) & (pl.col(regime_col) <= vol_q67)),
        "high_vol": test_df.filter(pl.col(regime_col) > vol_q67),
    }

    # Create 3-panel subplot
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    for idx, (regime_name, regime_df) in enumerate(regimes.items()):
        ax = axes[idx]

        if len(regime_df) == 0:
            logger.warning(f"No data for {regime_name}")
            continue

        # Extract predictions and outcomes
        pred = regime_df["final_prob"].to_numpy()
        actual = regime_df["outcome"].to_numpy()

        # Bin predictions
        bins = np.linspace(0, 1, 11)
        bin_centers = (bins[:-1] + bins[1:]) / 2

        calibration = []
        counts = []
        for i in range(len(bins) - 1):
            mask = (pred >= bins[i]) & (pred < bins[i + 1])
            if i == len(bins) - 2:
                mask = (pred >= bins[i]) & (pred <= bins[i + 1])

            if mask.sum() > 0:
                calibration.append(actual[mask].mean())
                counts.append(mask.sum())
            else:
                calibration.append(np.nan)
                counts.append(0)

        # Plot calibration
        sizes = np.clip(np.array(counts) / 1000, 30, 200)
        valid_mask = ~np.isnan(calibration)

        ax.scatter(
            bin_centers[valid_mask],
            np.array(calibration)[valid_mask],
            s=sizes[valid_mask],
            alpha=0.7,
            edgecolor="white",
            linewidth=1,
            color=REGIME_COLORS[regime_name],
            label="Observed frequency",
        )

        # Perfect calibration line
        ax.plot([0, 1], [0, 1], "k--", alpha=0.5, linewidth=1.5, label="Perfect calibration")

        # Calculate Brier score for regime
        brier = float(np.mean((pred - actual) ** 2))

        # Formatting
        regime_label = regime_name.replace("_", " ").title()
        ax.set_xlabel("Predicted Probability", fontsize=FONT_SIZES["label"])
        ax.set_ylabel("Actual Frequency", fontsize=FONT_SIZES["label"])
        ax.set_title(f"{regime_label}\nBrier: {brier:.4f}", fontsize=FONT_SIZES["title"])
        ax.legend(frameon=True, fontsize=FONT_SIZES["legend"] - 1)
        ax.grid(alpha=0.2)
        ax.set_xlim(-0.05, 1.05)
        ax.set_ylim(-0.05, 1.05)
        ax.set_aspect("equal", adjustable="box")

        # Add sample count
        ax.text(
            0.02,
            0.98,
            f"n={len(regime_df):,}",
            transform=ax.transAxes,
            fontsize=FONT_SIZES["annotation"],
            verticalalignment="top",
            bbox={"boxstyle": "round", "alpha": 0.15},
        )

    plt.suptitle(
        f"Calibration by Volatility Regime ({regime_col})",
        fontsize=FONT_SIZES["title"] + 2,
        y=1.02,
    )
    plt.tight_layout()

    # Save
    if output_path is None:
        output_path = str(get_plot_output_path("calibration", "calibration_by_regime.png"))

    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    logger.info(f"✓ Saved calibration by regime to {output_path}")

    # Upload to W&B
    if wandb_log:
        try:
            import wandb

            if wandb.run:
                wandb.log({"calibration/by_regime_comparison": wandb.Image(fig)})
                logger.info("✓ Uploaded to W&B")
        except ImportError:
            logger.warning("wandb not available - skipping upload")

    plt.close(fig)


def plot_prediction_distribution(
    test_df: pl.DataFrame,
    output_path: Optional[str] = None,
    wandb_log: bool = True,
) -> None:
    """
    Plot prediction distribution to detect mode collapse.

    Args:
        test_df: Test set with predictions
        output_path: Output file path
        wandb_log: Whether to upload to W&B
    """
    apply_plot_style()

    logger.info("Generating prediction distribution plot...")

    # Extract predictions
    model_pred = test_df["final_prob"].to_numpy()
    baseline_pred = test_df["prob_mid"].to_numpy()

    # Create figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # Plot 1: Histogram comparison
    bins = np.linspace(0, 1, 51)

    ax1.hist(
        baseline_pred,
        bins=bins,
        alpha=0.6,
        label="Baseline (IV-based)",
        color=COLORS["danger"],
        edgecolor="white",
        linewidth=0.5,
    )
    ax1.hist(
        model_pred,
        bins=bins,
        alpha=0.6,
        label="Model (Residual)",
        color=COLORS["primary"],
        edgecolor="white",
        linewidth=0.5,
    )

    ax1.axvline(
        x=0.5,
        color=COLORS["perfect"],
        linestyle="--",
        linewidth=2,
        alpha=0.5,
        label="Neutral (0.5)",
    )

    ax1.set_xlabel("Predicted Probability", fontsize=FONT_SIZES["label"])
    ax1.set_ylabel("Frequency", fontsize=FONT_SIZES["label"])
    ax1.set_title("Prediction Distribution (Histogram)", fontsize=FONT_SIZES["title"])
    ax1.legend(frameon=True, fontsize=FONT_SIZES["legend"])
    ax1.grid(alpha=0.2)

    # Plot 2: KDE (smoothed)
    sns.kdeplot(
        baseline_pred,
        ax=ax2,
        linewidth=2.5,
        alpha=0.8,
        color=COLORS["danger"],
        label="Baseline (IV-based)",
    )
    sns.kdeplot(
        model_pred,
        ax=ax2,
        linewidth=2.5,
        alpha=0.8,
        color=COLORS["primary"],
        label="Model (Residual)",
    )

    ax2.axvline(x=0.5, color=COLORS["perfect"], linestyle="--", linewidth=2, alpha=0.5, label="Neutral (0.5)")

    ax2.set_xlabel("Predicted Probability", fontsize=FONT_SIZES["label"])
    ax2.set_ylabel("Density", fontsize=FONT_SIZES["label"])
    ax2.set_title("Prediction Distribution (KDE)", fontsize=FONT_SIZES["title"])
    ax2.legend(frameon=True, fontsize=FONT_SIZES["legend"])
    ax2.grid(alpha=0.2)
    ax2.set_xlim(0, 1)

    # Add statistics
    model_std = float(np.std(model_pred))
    baseline_std = float(np.std(baseline_pred))
    model_median = float(np.median(model_pred))

    stats_text = (
        f"Model Stats:\n"
        f"Median: {model_median:.3f}\n"
        f"Std Dev: {model_std:.3f}\n"
        f"Range: [{model_pred.min():.3f}, {model_pred.max():.3f}]\n"
        f"\n"
        f"Baseline Std: {baseline_std:.3f}"
    )

    ax2.text(
        0.98,
        0.98,
        stats_text,
        transform=ax2.transAxes,
        fontsize=FONT_SIZES["annotation"],
        verticalalignment="top",
        horizontalalignment="right",
        bbox={"boxstyle": "round", "alpha": 0.15, "facecolor": "white"},
    )

    plt.suptitle(
        f"Prediction Distribution Analysis (n={len(test_df):,})",
        fontsize=FONT_SIZES["title"] + 2,
        y=1.02,
    )
    plt.tight_layout()

    # Save
    if output_path is None:
        output_path = str(get_plot_output_path("predictions", "distribution_comparison.png"))

    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    logger.info(f"✓ Saved prediction distribution to {output_path}")

    # Upload to W&B
    if wandb_log:
        try:
            import wandb

            if wandb.run:
                wandb.log({"predictions/distribution": wandb.Image(fig)})
                logger.info("✓ Uploaded to W&B")
        except ImportError:
            logger.warning("wandb not available - skipping upload")

    plt.close(fig)

    # Check for mode collapse
    in_middle_range = ((model_pred >= 0.45) & (model_pred <= 0.55)).sum()
    pct_middle = (in_middle_range / len(model_pred)) * 100

    logger.info("Prediction Distribution Statistics:")
    logger.info(f"  Model std dev: {model_std:.4f}")
    logger.info(f"  Baseline std dev: {baseline_std:.4f}")
    logger.info(f"  % in [0.45, 0.55]: {pct_middle:.1f}%")

    if pct_middle > 80:
        logger.warning("⚠️  Possible mode collapse - >80% predictions near 0.5!")
    elif model_std < baseline_std * 0.8:
        logger.warning("⚠️  Model less confident than baseline (lower std dev)")


def generate_time_series_report(
    test_file: str,
    output_dir: str = "results/plots/time_series/",
    wandb_log: bool = True,
) -> dict[str, Any]:
    """
    Generate all time-series plots and summary report.

    Args:
        test_file: Path to test predictions parquet file
        output_dir: Output directory for plots
        wandb_log: Whether to upload to W&B

    Returns:
        Dictionary with summary statistics
    """
    logger.info("=" * 80)
    logger.info("GENERATING TIME-SERIES VISUALIZATION REPORT")
    logger.info("=" * 80)

    # Load test predictions
    test_df = pl.read_parquet(test_file)
    logger.info(f"Loaded {len(test_df):,} test predictions")

    # Generate all plots
    results = {}

    # 1. Brier time series (daily and weekly)
    plot_brier_over_time(test_df, window="1d", wandb_log=wandb_log)
    plot_brier_over_time(test_df, window="1w", wandb_log=wandb_log)

    # 2. Calibration by regime
    if "rv_300s" in test_df.columns:
        plot_calibration_by_regime(test_df, regime_col="rv_300s", wandb_log=wandb_log)
    else:
        logger.warning("rv_300s column not found - skipping regime calibration")

    # 3. Prediction distribution
    plot_prediction_distribution(test_df, wandb_log=wandb_log)

    logger.info("=" * 80)
    logger.info("TIME-SERIES REPORT COMPLETE")
    logger.info("=" * 80)

    return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate time-series visualization plots")
    parser.add_argument(
        "--test-file",
        type=str,
        default="../results/test_features_lgb.parquet",
        help="Path to test predictions file",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results/plots/time_series/",
        help="Output directory",
    )
    parser.add_argument(
        "--no-wandb",
        action="store_true",
        help="Disable W&B upload",
    )

    args = parser.parse_args()

    generate_time_series_report(
        test_file=args.test_file,
        output_dir=args.output_dir,
        wandb_log=not args.no_wandb,
    )
