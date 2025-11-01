#!/usr/bin/env python3
"""
Simulation-based plots for uncertainty quantification.

Provides Monte Carlo bootstrap methods for computing confidence intervals
on model performance metrics.

Functions:
    - plot_bootstrap_ci(): Bootstrap confidence intervals for Brier score
    - generate_simulation_report(): Orchestrates all simulation plots
"""

import logging
from pathlib import Path
from typing import Any, Optional

import matplotlib.pyplot as plt
import numpy as np
import polars as pl

from visualization.plot_config import COLORS, FONT_SIZES, apply_plot_style, get_plot_output_path
from visualization.wandb_integration import upload_plot

logger = logging.getLogger(__name__)


def _calculate_brier_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculate Brier score for binary probabilistic predictions.

    Args:
        y_true: True binary outcomes (0 or 1)
        y_pred: Predicted probabilities [0, 1]

    Returns:
        Brier score (lower is better)
    """
    return float(np.mean((y_pred - y_true) ** 2))


def plot_bootstrap_ci(
    test_df: pl.DataFrame,
    output_path: Optional[str] = None,
    wandb_log: bool = True,
    n_bootstrap: int = 1_000,
    confidence_level: float = 0.95,
    downsample_to: int = 100_000,
) -> dict[str, Any]:
    """
    Plot bootstrap confidence intervals for model Brier score.

    Uses Monte Carlo bootstrap resampling to estimate uncertainty in
    performance metrics. Provides overall CI and regime-stratified CIs.

    Args:
        test_df: Test predictions with columns:
            - final_prob (model predictions)
            - prob_mid (baseline predictions)
            - outcome (binary outcome)
            - volatility (for regime stratification)
        output_path: Path to save plot (optional, auto-generated if None)
        wandb_log: Whether to upload to W&B
        n_bootstrap: Number of bootstrap samples (default: 1,000)
        confidence_level: Confidence level for intervals (default: 0.95)
        downsample_to: Downsample to this many rows for performance

    Returns:
        Dictionary with bootstrap statistics:
            - overall_mean: Mean Brier score across bootstraps
            - overall_ci_lower: Lower CI bound
            - overall_ci_upper: Upper CI bound
            - regime_stats: Dict with stats per regime
    """
    logger.info("Generating bootstrap confidence interval plot...")

    # Downsample if needed
    if len(test_df) > downsample_to:
        logger.info(f"Downsampling from {len(test_df):,} to {downsample_to:,} rows for bootstrap")
        test_df = test_df.sample(n=downsample_to, shuffle=True, seed=42)

    # Extract arrays
    y_true = test_df["outcome"].to_numpy()
    y_pred_model = test_df["final_prob"].to_numpy()
    y_pred_baseline = test_df["prob_mid"].to_numpy()

    # Bootstrap sampling
    logger.info(f"Running {n_bootstrap:,} bootstrap iterations...")
    n_samples = len(y_true)
    brier_model_bootstrap = np.zeros(n_bootstrap)
    brier_baseline_bootstrap = np.zeros(n_bootstrap)

    rng = np.random.RandomState(42)
    for i in range(n_bootstrap):
        # Resample with replacement
        indices = rng.choice(n_samples, size=n_samples, replace=True)
        y_true_boot = y_true[indices]
        y_pred_model_boot = y_pred_model[indices]
        y_pred_baseline_boot = y_pred_baseline[indices]

        # Calculate Brier scores
        brier_model_bootstrap[i] = _calculate_brier_score(y_true_boot, y_pred_model_boot)
        brier_baseline_bootstrap[i] = _calculate_brier_score(y_true_boot, y_pred_baseline_boot)

    # Calculate confidence intervals
    alpha = 1 - confidence_level
    ci_lower = alpha / 2
    ci_upper = 1 - alpha / 2

    model_mean = float(np.mean(brier_model_bootstrap))
    model_ci_lower = float(np.percentile(brier_model_bootstrap, ci_lower * 100))
    model_ci_upper = float(np.percentile(brier_model_bootstrap, ci_upper * 100))

    baseline_mean = float(np.mean(brier_baseline_bootstrap))
    baseline_ci_lower = float(np.percentile(brier_baseline_bootstrap, ci_lower * 100))
    baseline_ci_upper = float(np.percentile(brier_baseline_bootstrap, ci_upper * 100))

    logger.info(f"Model Brier: {model_mean:.4f} [{model_ci_lower:.4f}, {model_ci_upper:.4f}]")
    logger.info(f"Baseline Brier: {baseline_mean:.4f} [{baseline_ci_lower:.4f}, {baseline_ci_upper:.4f}]")

    # Regime-stratified bootstrap
    # Check if volatility is already categorical (string)
    if test_df["volatility"].dtype == pl.Utf8:
        # Volatility is already categorized (low/mid/high)
        volatility = test_df["volatility"].to_numpy()
        regime_labels = ["Low Vol", "Mid Vol", "High Vol"]
        regime_masks = [
            volatility == "low",
            volatility == "mid",
            volatility == "high",
        ]
    else:
        # Volatility is numeric - compute tertiles
        volatility = test_df["volatility"].to_numpy()
        vol_tertiles_list: list[float] = np.percentile(volatility, [33.33, 66.67]).tolist()  # type: ignore
        vol_t1 = vol_tertiles_list[0]
        vol_t2 = vol_tertiles_list[1]
        regime_labels = ["Low Vol", "Mid Vol", "High Vol"]
        regime_masks = [
            volatility <= vol_t1,
            (volatility > vol_t1) & (volatility <= vol_t2),
            volatility > vol_t2,
        ]

    regime_stats = {}
    regime_model_means = []
    regime_model_cis = []

    for regime_name, mask in zip(regime_labels, regime_masks):
        regime_brier = np.zeros(n_bootstrap)
        regime_size = int(np.sum(mask))

        if regime_size < 100:
            logger.warning(f"Regime {regime_name} has only {regime_size} samples - skipping")
            continue

        for i in range(n_bootstrap):
            # Resample within regime
            regime_indices = np.where(mask)[0]
            boot_indices = rng.choice(regime_indices, size=len(regime_indices), replace=True)
            regime_brier[i] = _calculate_brier_score(y_true[boot_indices], y_pred_model[boot_indices])

        regime_mean = float(np.mean(regime_brier))
        regime_ci_lower_val = float(np.percentile(regime_brier, ci_lower * 100))
        regime_ci_upper_val = float(np.percentile(regime_brier, ci_upper * 100))

        regime_stats[regime_name] = {
            "mean": regime_mean,
            "ci_lower": regime_ci_lower_val,
            "ci_upper": regime_ci_upper_val,
            "n_samples": regime_size,
        }

        regime_model_means.append(regime_mean)
        regime_model_cis.append((regime_ci_lower_val, regime_ci_upper_val))

        logger.info(f"{regime_name}: {regime_mean:.4f} [{regime_ci_lower_val:.4f}, {regime_ci_upper_val:.4f}]")

    # Create visualization
    fig = plt.figure(figsize=(16, 10))
    apply_plot_style()

    # 2x2 subplot layout
    gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)

    # 1. Overall distribution (model)
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.hist(brier_model_bootstrap, bins=50, alpha=0.7, color=COLORS["primary"], edgecolor="white", linewidth=0.5)
    ax1.axvline(model_mean, color=COLORS["success"], linewidth=2, linestyle="--", label=f"Mean: {model_mean:.4f}")
    ax1.axvline(
        model_ci_lower, color=COLORS["warning"], linewidth=1.5, linestyle=":", label=f"{confidence_level * 100:.0f}% CI"
    )
    ax1.axvline(model_ci_upper, color=COLORS["warning"], linewidth=1.5, linestyle=":")
    ax1.set_xlabel("Brier Score", fontsize=FONT_SIZES["label"], fontweight="bold")
    ax1.set_ylabel("Frequency", fontsize=FONT_SIZES["label"], fontweight="bold")
    ax1.set_title(f"Model Bootstrap Distribution (n={n_bootstrap:,})", fontsize=FONT_SIZES["title"], fontweight="bold")
    ax1.legend(loc="upper right", fontsize=FONT_SIZES["legend"])
    ax1.grid(True, alpha=0.2, linewidth=0.5)

    # 2. Overall distribution (baseline)
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.hist(brier_baseline_bootstrap, bins=50, alpha=0.7, color=COLORS["danger"], edgecolor="white", linewidth=0.5)
    ax2.axvline(baseline_mean, color=COLORS["success"], linewidth=2, linestyle="--", label=f"Mean: {baseline_mean:.4f}")
    ax2.axvline(
        baseline_ci_lower,
        color=COLORS["warning"],
        linewidth=1.5,
        linestyle=":",
        label=f"{confidence_level * 100:.0f}% CI",
    )
    ax2.axvline(baseline_ci_upper, color=COLORS["warning"], linewidth=1.5, linestyle=":")
    ax2.set_xlabel("Brier Score", fontsize=FONT_SIZES["label"], fontweight="bold")
    ax2.set_ylabel("Frequency", fontsize=FONT_SIZES["label"], fontweight="bold")
    ax2.set_title(
        f"Baseline Bootstrap Distribution (n={n_bootstrap:,})", fontsize=FONT_SIZES["title"], fontweight="bold"
    )
    ax2.legend(loc="upper right", fontsize=FONT_SIZES["legend"])
    ax2.grid(True, alpha=0.2, linewidth=0.5)

    # 3. Improvement distribution
    ax3 = fig.add_subplot(gs[1, 0])
    improvement_bootstrap = brier_baseline_bootstrap - brier_model_bootstrap
    improvement_mean = float(np.mean(improvement_bootstrap))
    improvement_ci_lower_val = float(np.percentile(improvement_bootstrap, ci_lower * 100))
    improvement_ci_upper_val = float(np.percentile(improvement_bootstrap, ci_upper * 100))

    ax3.hist(improvement_bootstrap, bins=50, alpha=0.7, color=COLORS["success"], edgecolor="white", linewidth=0.5)
    ax3.axvline(
        improvement_mean, color=COLORS["primary"], linewidth=2, linestyle="--", label=f"Mean: {improvement_mean:.4f}"
    )
    ax3.axvline(
        improvement_ci_lower_val,
        color=COLORS["warning"],
        linewidth=1.5,
        linestyle=":",
        label=f"{confidence_level * 100:.0f}% CI",
    )
    ax3.axvline(improvement_ci_upper_val, color=COLORS["warning"], linewidth=1.5, linestyle=":")
    ax3.axvline(0, color=COLORS["perfect"], linewidth=1, linestyle="-", alpha=0.5, label="No improvement")
    ax3.set_xlabel("Brier Improvement (Baseline - Model)", fontsize=FONT_SIZES["label"], fontweight="bold")
    ax3.set_ylabel("Frequency", fontsize=FONT_SIZES["label"], fontweight="bold")
    ax3.set_title("Bootstrap Improvement Distribution", fontsize=FONT_SIZES["title"], fontweight="bold")
    ax3.legend(loc="upper right", fontsize=FONT_SIZES["legend"])
    ax3.grid(True, alpha=0.2, linewidth=0.5)

    # 4. Regime-stratified CIs
    ax4 = fig.add_subplot(gs[1, 1])
    if regime_model_means:
        x_pos = np.arange(len(regime_labels))
        errors = [[m - ci[0], ci[1] - m] for m, ci in zip(regime_model_means, regime_model_cis)]
        errors_lower = [e[0] for e in errors]
        errors_upper = [e[1] for e in errors]

        ax4.bar(x_pos, regime_model_means, alpha=0.7, color=COLORS["primary"], edgecolor="white", linewidth=1)
        ax4.errorbar(
            x_pos,
            regime_model_means,
            yerr=[errors_lower, errors_upper],
            fmt="none",
            ecolor=COLORS["warning"],
            elinewidth=2,
            capsize=5,
            capthick=2,
            label=f"{confidence_level * 100:.0f}% CI",
        )
        ax4.axhline(model_mean, color=COLORS["success"], linewidth=1.5, linestyle="--", label="Overall mean")
        ax4.set_xticks(x_pos)
        ax4.set_xticklabels(regime_labels, fontsize=FONT_SIZES["tick"], fontweight="bold")
        ax4.set_ylabel("Brier Score", fontsize=FONT_SIZES["label"], fontweight="bold")
        ax4.set_title("Regime-Stratified Bootstrap CIs", fontsize=FONT_SIZES["title"], fontweight="bold")
        ax4.legend(loc="upper right", fontsize=FONT_SIZES["legend"])
        ax4.grid(True, alpha=0.2, linewidth=0.5, axis="y")

    # Main title
    fig.suptitle(
        "Monte Carlo Bootstrap Confidence Intervals\n(Uncertainty Quantification)",
        fontsize=FONT_SIZES["title"] + 4,
        fontweight="bold",
        y=0.98,
    )

    # Save plot
    if output_path is None:
        output_path = str(get_plot_output_path("simulation", "bootstrap_ci.png"))

    plt.savefig(output_path, dpi=300, bbox_inches="tight", facecolor="white")
    logger.info(f"Saved bootstrap CI plot to {output_path}")

    # Upload to W&B
    if wandb_log and output_path is not None:
        upload_plot(output_path, "bootstrap_ci")

    plt.close()

    # Return statistics
    return {
        "overall_mean": model_mean,
        "overall_ci_lower": model_ci_lower,
        "overall_ci_upper": model_ci_upper,
        "baseline_mean": baseline_mean,
        "baseline_ci_lower": baseline_ci_lower,
        "baseline_ci_upper": baseline_ci_upper,
        "improvement_mean": improvement_mean,
        "improvement_ci_lower": improvement_ci_lower_val,
        "improvement_ci_upper": improvement_ci_upper_val,
        "regime_stats": regime_stats,
        "n_bootstrap": n_bootstrap,
        "confidence_level": confidence_level,
    }


def generate_simulation_report(
    test_file: str,
    output_dir: str = "results/plots/simulation/",
    wandb_log: bool = True,
    n_bootstrap: int = 1_000,
) -> dict[str, Any]:
    """
    Generate comprehensive simulation-based uncertainty quantification report.

    Creates:
    - Bootstrap confidence intervals for Brier score
    - Regime-stratified uncertainty estimates

    Args:
        test_file: Path to test predictions parquet
        output_dir: Output directory for plots
        wandb_log: Whether to upload to W&B
        n_bootstrap: Number of bootstrap samples

    Returns:
        Dictionary with summary statistics
    """
    logger.info("=" * 80)
    logger.info("GENERATING SIMULATION REPORT")
    logger.info("=" * 80)

    output_dir_path = Path(output_dir)
    output_dir_path.mkdir(parents=True, exist_ok=True)

    # Load test data
    logger.info(f"Loading test data from {test_file}...")
    test_df = pl.read_parquet(test_file)
    logger.info(f"Loaded {len(test_df):,} test predictions")

    # Generate bootstrap CI plot
    logger.info("\n[1/1] Generating bootstrap confidence intervals...")
    try:
        bootstrap_results = plot_bootstrap_ci(
            test_df=test_df,
            output_path=str(output_dir_path / "bootstrap_ci.png"),
            wandb_log=wandb_log,
            n_bootstrap=n_bootstrap,
        )

        logger.info("\n" + "=" * 80)
        logger.info("SIMULATION SUMMARY")
        logger.info("=" * 80)
        logger.info(
            f"Model Brier:     {bootstrap_results['overall_mean']:.4f} "
            f"[{bootstrap_results['overall_ci_lower']:.4f}, {bootstrap_results['overall_ci_upper']:.4f}]"
        )
        logger.info(
            f"Baseline Brier:  {bootstrap_results['baseline_mean']:.4f} "
            f"[{bootstrap_results['baseline_ci_lower']:.4f}, {bootstrap_results['baseline_ci_upper']:.4f}]"
        )
        logger.info(
            f"Improvement:     {bootstrap_results['improvement_mean']:.4f} "
            f"[{bootstrap_results['improvement_ci_lower']:.4f}, {bootstrap_results['improvement_ci_upper']:.4f}]"
        )
        logger.info("=" * 80)

        return bootstrap_results

    except Exception as e:
        logger.error(f"Failed to generate bootstrap CI plot: {e}")
        return {}


def main() -> None:
    """CLI entry point for simulation plots."""
    import argparse

    parser = argparse.ArgumentParser(description="Generate simulation-based uncertainty plots")
    parser.add_argument(
        "--test-file",
        type=str,
        required=True,
        help="Path to test predictions parquet",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results/plots/simulation/",
        help="Output directory for plots",
    )
    parser.add_argument(
        "--no-wandb",
        action="store_true",
        help="Disable W&B upload",
    )
    parser.add_argument(
        "--n-bootstrap",
        type=int,
        default=1_000,
        help="Number of bootstrap samples (default: 1,000)",
    )

    args = parser.parse_args()

    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    generate_simulation_report(
        test_file=args.test_file,
        output_dir=args.output_dir,
        wandb_log=not args.no_wandb,
        n_bootstrap=args.n_bootstrap,
    )


if __name__ == "__main__":
    main()
