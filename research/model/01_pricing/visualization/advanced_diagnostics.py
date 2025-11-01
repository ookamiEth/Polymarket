#!/usr/bin/env python3
"""
Advanced Diagnostic Plots for Model Evaluation.

Provides statistical evaluation plots for binary probabilistic predictions:
- ROC Curve & AUC (discrimination)
- Precision-Recall Curve (imbalanced data handling)
- QQ Plot for Residuals (distributional assumptions)
- Lift Chart & Cumulative Gains (targeting efficiency)
- Win Rate Heatmap (regime-specific performance)
- SHAP Dependence Plots (feature interactions)

Usage:
    from visualization.advanced_diagnostics import generate_diagnostics_report
    generate_diagnostics_report(test_file="results/test_predictions.parquet")
"""

import logging
from pathlib import Path
from typing import Any, Optional

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from scipy import stats
from sklearn.metrics import (
    average_precision_score,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)

from visualization.plot_config import (
    COLORS,
    FONT_SIZES,
    REGIME_COLORS,
    apply_plot_style,
    get_plot_output_path,
)
from visualization.wandb_integration import upload_plot

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# Try importing wandb (optional)
try:
    import wandb

    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False


def plot_roc_curve(
    test_df: pl.DataFrame,
    output_path: Optional[str] = None,
    wandb_log: bool = True,
) -> dict[str, Any]:
    """
    Plot ROC curves for model and baseline with regime stratification.

    Creates 2x2 subplot grid:
    - Overall ROC (model vs baseline)
    - Low volatility regime
    - Mid volatility regime
    - High volatility regime

    Args:
        test_df: Test predictions DataFrame with columns:
            - final_prob: Model predictions
            - prob_mid: Baseline predictions
            - outcome: Binary outcomes (0/1)
            - rv_300s: Realized volatility (for regime stratification)
        output_path: Output file path, auto-generated if None
        wandb_log: Whether to upload to W&B

    Returns:
        Dictionary with AUC scores per regime
    """
    apply_plot_style()

    logger.info("Generating ROC curves...")

    # Calculate volatility quantiles for regime stratification
    vol_q33 = test_df["rv_300s"].quantile(0.33)
    vol_q67 = test_df["rv_300s"].quantile(0.67)

    # Create regimes
    regimes = {
        "overall": test_df,
        "low_vol": test_df.filter(pl.col("rv_300s") <= vol_q33),
        "mid_vol": test_df.filter((pl.col("rv_300s") > vol_q33) & (pl.col("rv_300s") <= vol_q67)),
        "high_vol": test_df.filter(pl.col("rv_300s") > vol_q67),
    }

    # Create 2x2 subplot
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    fig.suptitle(
        "ROC Curves: Model vs Baseline by Volatility Regime", fontsize=FONT_SIZES["title"] + 2, fontweight="bold"
    )

    results = {}

    for idx, (regime_name, regime_df) in enumerate(regimes.items()):
        ax = axes.flatten()[idx]

        if len(regime_df) == 0:
            ax.text(0.5, 0.5, "No data", ha="center", va="center")
            ax.set_title(regime_name.replace("_", " ").title())
            continue

        # Extract regime data
        regime_model = regime_df["final_prob"].to_numpy()
        regime_baseline = regime_df["prob_mid"].to_numpy()
        regime_outcome = regime_df["outcome"].to_numpy()

        # Calculate ROC curves
        fpr_model, tpr_model, _ = roc_curve(regime_outcome, regime_model)
        fpr_baseline, tpr_baseline, _ = roc_curve(regime_outcome, regime_baseline)

        # Calculate AUC
        auc_model = roc_auc_score(regime_outcome, regime_model)
        auc_baseline = roc_auc_score(regime_outcome, regime_baseline)

        # Store results
        results[regime_name] = {
            "auc_model": float(auc_model),
            "auc_baseline": float(auc_baseline),
            "auc_improvement": float(auc_model - auc_baseline),
            "n_samples": len(regime_df),
        }

        # Plot ROC curves
        color = REGIME_COLORS.get(regime_name, COLORS["primary"])

        ax.plot(
            fpr_model,
            tpr_model,
            color=color,
            linewidth=2.5,
            alpha=0.9,
            label=f"Model (AUC={auc_model:.3f})",
        )

        ax.plot(
            fpr_baseline,
            tpr_baseline,
            color=COLORS["danger"],
            linewidth=2,
            alpha=0.7,
            linestyle="--",
            label=f"Baseline (AUC={auc_baseline:.3f})",
        )

        # Plot diagonal (random classifier)
        ax.plot(
            [0, 1], [0, 1], color=COLORS["perfect"], linestyle=":", linewidth=1.5, alpha=0.5, label="Random (AUC=0.5)"
        )

        # Styling
        ax.set_xlabel("False Positive Rate", fontsize=FONT_SIZES["label"])
        ax.set_ylabel("True Positive Rate", fontsize=FONT_SIZES["label"])
        ax.set_title(
            f"{regime_name.replace('_', ' ').title()} (n={len(regime_df):,})",
            fontsize=FONT_SIZES["title"],
        )
        ax.legend(loc="lower right", fontsize=FONT_SIZES["legend"] - 1)
        ax.grid(True, alpha=0.2, linewidth=0.5)
        ax.set_xlim((-0.02, 1.02))
        ax.set_ylim((-0.02, 1.02))

        # Annotate improvement
        improvement = auc_model - auc_baseline
        ax.annotate(
            f"Improvement: {improvement:+.3f}",
            xy=(0.6, 0.2),
            xycoords="axes fraction",
            fontsize=FONT_SIZES["annotation"],
            bbox={
                "boxstyle": "round",
                "facecolor": COLORS["success"] if improvement > 0 else COLORS["danger"],
                "alpha": 0.3,
            },
        )

    plt.tight_layout()

    # Save
    if output_path is None:
        output_path = str(get_plot_output_path("diagnostics", "roc_curve.png"))

    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    logger.info(f"Saved ROC curves to {output_path}")

    # Log summary
    logger.info("\nROC AUC Summary:")
    logger.info(f"  Overall Model AUC:    {results['overall']['auc_model']:.3f}")
    logger.info(f"  Overall Baseline AUC: {results['overall']['auc_baseline']:.3f}")
    logger.info(f"  Improvement:          {results['overall']['auc_improvement']:+.3f}")

    # Upload to W&B
    if wandb_log and WANDB_AVAILABLE and wandb.run:
        wandb.log({"diagnostics/roc_curve": wandb.Image(fig)})

    plt.close(fig)

    return results


def plot_precision_recall(
    test_df: pl.DataFrame,
    output_path: Optional[str] = None,
    wandb_log: bool = True,
) -> dict[str, Any]:
    """
    Plot Precision-Recall curves for model and baseline.

    Better than ROC for imbalanced datasets. Shows trade-off between
    precision (positive predictive value) and recall (sensitivity).

    Args:
        test_df: Test predictions DataFrame
        output_path: Output file path
        wandb_log: Whether to upload to W&B

    Returns:
        Dictionary with Average Precision scores
    """
    apply_plot_style()

    logger.info("Generating Precision-Recall curves...")

    # Extract columns
    model_pred = test_df["final_prob"].to_numpy()
    baseline_pred = test_df["prob_mid"].to_numpy()
    outcome = test_df["outcome"].to_numpy()

    # Calculate PR curves
    precision_model, recall_model, _ = precision_recall_curve(outcome, model_pred)
    precision_baseline, recall_baseline, _ = precision_recall_curve(outcome, baseline_pred)

    # Calculate Average Precision
    ap_model = average_precision_score(outcome, model_pred)
    ap_baseline = average_precision_score(outcome, baseline_pred)

    # Create plot
    fig, ax = plt.subplots(figsize=(10, 8))

    ax.plot(
        recall_model,
        precision_model,
        color=COLORS["primary"],
        linewidth=2.5,
        alpha=0.9,
        label=f"Model (AP={ap_model:.3f})",
    )

    ax.plot(
        recall_baseline,
        precision_baseline,
        color=COLORS["danger"],
        linewidth=2,
        alpha=0.7,
        linestyle="--",
        label=f"Baseline (AP={ap_baseline:.3f})",
    )

    # Plot no-skill baseline (proportion of positives)
    no_skill = np.mean(outcome)
    ax.plot(
        [0, 1],
        [no_skill, no_skill],
        color=COLORS["perfect"],
        linestyle=":",
        linewidth=1.5,
        alpha=0.5,
        label=f"No Skill ({no_skill:.3f})",
    )

    # Styling
    ax.set_xlabel("Recall (Sensitivity)", fontsize=FONT_SIZES["label"])
    ax.set_ylabel("Precision (PPV)", fontsize=FONT_SIZES["label"])
    ax.set_title(
        "Precision-Recall Curve: Model vs Baseline", fontsize=FONT_SIZES["title"] + 2, pad=15, fontweight="bold"
    )
    ax.legend(loc="best", fontsize=FONT_SIZES["legend"])
    ax.grid(True, alpha=0.2, linewidth=0.5)
    ax.set_xlim((-0.02, 1.02))
    ax.set_ylim((-0.02, 1.02))

    # Annotate improvement
    improvement = ap_model - ap_baseline
    ax.annotate(
        f"AP Improvement: {improvement:+.3f}\n({improvement / ap_baseline * 100:+.1f}%)",
        xy=(0.05, 0.95),
        xycoords="axes fraction",
        fontsize=FONT_SIZES["annotation"],
        verticalalignment="top",
        bbox={
            "boxstyle": "round",
            "facecolor": COLORS["success"] if improvement > 0 else COLORS["danger"],
            "alpha": 0.3,
        },
    )

    # Save
    if output_path is None:
        output_path = str(get_plot_output_path("diagnostics", "precision_recall.png"))

    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    logger.info(f"Saved Precision-Recall curve to {output_path}")

    # Log summary
    logger.info("\nPrecision-Recall Summary:")
    logger.info(f"  Model AP:      {ap_model:.3f}")
    logger.info(f"  Baseline AP:   {ap_baseline:.3f}")
    logger.info(f"  Improvement:   {improvement:+.3f}")

    # Upload to W&B
    if wandb_log and WANDB_AVAILABLE and wandb.run:
        wandb.log({"diagnostics/precision_recall": wandb.Image(fig)})

    plt.close(fig)

    return {
        "ap_model": float(ap_model),
        "ap_baseline": float(ap_baseline),
        "ap_improvement": float(improvement),
        "no_skill_baseline": float(no_skill),
    }


def plot_qq_residuals(
    test_df: pl.DataFrame,
    output_path: Optional[str] = None,
    wandb_log: bool = True,
) -> dict[str, Any]:
    """
    Plot QQ (Quantile-Quantile) plots for model and baseline residuals.

    Compares residual distributions to normal distribution to validate
    assumptions about error structure. Deviations indicate non-normality.

    Args:
        test_df: Test predictions DataFrame with columns:
            - residual: Model residuals (outcome - final_prob)
            - (or calculated from outcome - final_prob)
        output_path: Output file path
        wandb_log: Whether to upload to W&B

    Returns:
        Dictionary with normality test results
    """
    apply_plot_style()

    logger.info("Generating QQ plots for residuals...")

    # Calculate residuals if not present
    if "residual" in test_df.columns:
        model_residuals = test_df["residual"].to_numpy()
    else:
        model_residuals = (test_df["outcome"] - test_df["final_prob"]).to_numpy()

    # Calculate baseline residuals
    baseline_residuals = (test_df["outcome"] - test_df["prob_mid"]).to_numpy()

    # Create 1x2 subplot
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle("QQ Plots: Residual Normality Check", fontsize=FONT_SIZES["title"] + 2, fontweight="bold")

    results = {}

    # Plot model residuals
    ax = axes[0]
    stats.probplot(model_residuals, dist="norm", plot=ax)
    ax.get_lines()[0].set_markerfacecolor(COLORS["primary"])
    ax.get_lines()[0].set_markeredgecolor("white")
    ax.get_lines()[0].set_markersize(4)
    ax.get_lines()[0].set_alpha(0.6)
    ax.get_lines()[1].set_color(COLORS["danger"])
    ax.get_lines()[1].set_linewidth(2)

    ax.set_title("Model Residuals", fontsize=FONT_SIZES["title"])
    ax.set_xlabel("Theoretical Quantiles", fontsize=FONT_SIZES["label"])
    ax.set_ylabel("Sample Quantiles", fontsize=FONT_SIZES["label"])
    ax.grid(True, alpha=0.2, linewidth=0.5)

    # Calculate statistics
    model_mean = float(np.mean(model_residuals))
    model_std = float(np.std(model_residuals))
    model_skew = float(stats.skew(model_residuals))
    model_kurt = float(stats.kurtosis(model_residuals))

    results["model"] = {
        "mean": model_mean,
        "std": model_std,
        "skewness": model_skew,
        "kurtosis": model_kurt,
    }

    # Annotate
    ax.annotate(
        f"Mean: {model_mean:.4f}\nStd: {model_std:.4f}\nSkew: {model_skew:.2f}\nKurt: {model_kurt:.2f}",
        xy=(0.05, 0.95),
        xycoords="axes fraction",
        fontsize=FONT_SIZES["annotation"],
        verticalalignment="top",
        bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.8},
    )

    # Plot baseline residuals
    ax = axes[1]
    stats.probplot(baseline_residuals, dist="norm", plot=ax)
    ax.get_lines()[0].set_markerfacecolor(COLORS["danger"])
    ax.get_lines()[0].set_markeredgecolor("white")
    ax.get_lines()[0].set_markersize(4)
    ax.get_lines()[0].set_alpha(0.6)
    ax.get_lines()[1].set_color(COLORS["perfect"])
    ax.get_lines()[1].set_linewidth(2)

    ax.set_title("Baseline Residuals", fontsize=FONT_SIZES["title"])
    ax.set_xlabel("Theoretical Quantiles", fontsize=FONT_SIZES["label"])
    ax.set_ylabel("Sample Quantiles", fontsize=FONT_SIZES["label"])
    ax.grid(True, alpha=0.2, linewidth=0.5)

    # Calculate statistics
    baseline_mean = float(np.mean(baseline_residuals))
    baseline_std = float(np.std(baseline_residuals))
    baseline_skew = float(stats.skew(baseline_residuals))
    baseline_kurt = float(stats.kurtosis(baseline_residuals))

    results["baseline"] = {
        "mean": baseline_mean,
        "std": baseline_std,
        "skewness": baseline_skew,
        "kurtosis": baseline_kurt,
    }

    # Annotate
    ax.annotate(
        f"Mean: {baseline_mean:.4f}\nStd: {baseline_std:.4f}\nSkew: {baseline_skew:.2f}\nKurt: {baseline_kurt:.2f}",
        xy=(0.05, 0.95),
        xycoords="axes fraction",
        fontsize=FONT_SIZES["annotation"],
        verticalalignment="top",
        bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.8},
    )

    plt.tight_layout()

    # Save
    if output_path is None:
        output_path = str(get_plot_output_path("diagnostics", "qq_plot_residuals.png"))

    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    logger.info(f"Saved QQ plots to {output_path}")

    # Log summary
    logger.info("\nQQ Plot Summary:")
    logger.info(f"  Model Residuals:    Mean={model_mean:.4f}, Std={model_std:.4f}")
    logger.info(f"  Baseline Residuals: Mean={baseline_mean:.4f}, Std={baseline_std:.4f}")

    # Warnings for non-normality
    if abs(model_skew) > 1.0:
        logger.warning(f"⚠️  Model residuals show high skewness ({model_skew:.2f}). May indicate bias.")
    if abs(model_kurt) > 3.0:
        logger.warning(f"⚠️  Model residuals show high kurtosis ({model_kurt:.2f}). Fat tails detected.")

    # Upload to W&B
    if wandb_log and WANDB_AVAILABLE and wandb.run:
        wandb.log({"diagnostics/qq_plot_residuals": wandb.Image(fig)})

    plt.close(fig)

    return results


def plot_lift_chart(
    test_df: pl.DataFrame,
    output_path: Optional[str] = None,
    wandb_log: bool = True,
    downsample_to: int = 100_000,
) -> dict[str, Any]:
    """
    Plot Lift Chart and Cumulative Gains to show targeting efficiency.

    Measures how much better model is than random targeting. Top decile
    should capture significantly more than 10% of positive outcomes.

    Args:
        test_df: Test predictions DataFrame
        output_path: Output file path
        wandb_log: Whether to upload to W&B
        downsample_to: Maximum rows to use (for performance)

    Returns:
        Dictionary with lift metrics per decile
    """
    apply_plot_style()

    logger.info("Generating Lift Chart and Cumulative Gains...")

    # Downsample if needed
    if len(test_df) > downsample_to:
        logger.info(f"Downsampling from {len(test_df):,} to {downsample_to:,} rows for performance")
        test_df = test_df.sample(n=downsample_to, shuffle=True, seed=42)

    # Extract columns
    model_pred = test_df["final_prob"].to_numpy()
    baseline_pred = test_df["prob_mid"].to_numpy()
    outcome = test_df["outcome"].to_numpy()

    # Create 1x2 subplot
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle(
        "Targeting Efficiency: Lift Chart & Cumulative Gains", fontsize=FONT_SIZES["title"] + 2, fontweight="bold"
    )

    # === LEFT: Cumulative Gains ===
    ax = axes[0]

    # Sort by model predictions (descending)
    sorted_idx_model = np.argsort(model_pred)[::-1]
    sorted_outcome_model = outcome[sorted_idx_model]

    # Sort by baseline predictions (descending)
    sorted_idx_baseline = np.argsort(baseline_pred)[::-1]
    sorted_outcome_baseline = outcome[sorted_idx_baseline]

    # Calculate cumulative gains
    total_positives = np.sum(outcome)
    cumsum_model = np.cumsum(sorted_outcome_model) / total_positives * 100
    cumsum_baseline = np.cumsum(sorted_outcome_baseline) / total_positives * 100
    pct_population = np.arange(1, len(outcome) + 1) / len(outcome) * 100

    # Plot cumulative gains
    ax.plot(
        pct_population,
        cumsum_model,
        color=COLORS["primary"],
        linewidth=2.5,
        alpha=0.9,
        label="Model",
    )

    ax.plot(
        pct_population,
        cumsum_baseline,
        color=COLORS["danger"],
        linewidth=2,
        alpha=0.7,
        linestyle="--",
        label="Baseline",
    )

    # Plot perfect model (diagonal to 100%)
    ax.plot(
        [0, total_positives / len(outcome) * 100, 100],
        [0, 100, 100],
        color=COLORS["success"],
        linestyle=":",
        linewidth=1.5,
        alpha=0.5,
        label="Perfect",
    )

    # Plot random (diagonal)
    ax.plot([0, 100], [0, 100], color=COLORS["perfect"], linestyle=":", linewidth=1.5, alpha=0.5, label="Random")

    ax.set_xlabel("% of Population (sorted by prediction)", fontsize=FONT_SIZES["label"])
    ax.set_ylabel("% of Positives Captured", fontsize=FONT_SIZES["label"])
    ax.set_title("Cumulative Gains Chart", fontsize=FONT_SIZES["title"])
    ax.legend(loc="lower right", fontsize=FONT_SIZES["legend"])
    ax.grid(True, alpha=0.2, linewidth=0.5)
    ax.set_xlim((0, 100))
    ax.set_ylim((0, 105))

    # === RIGHT: Lift Chart (by decile) ===
    ax = axes[1]

    # Calculate lift by decile
    n_deciles = 10
    decile_size = len(outcome) // n_deciles

    deciles = []
    lift_model = []
    lift_baseline = []

    for i in range(n_deciles):
        start_idx = i * decile_size
        end_idx = (i + 1) * decile_size if i < n_deciles - 1 else len(outcome)

        # Model lift
        decile_outcome_model = sorted_outcome_model[start_idx:end_idx]
        response_rate_model = np.mean(decile_outcome_model)
        baseline_rate = np.mean(outcome)  # Random targeting
        lift_m = response_rate_model / baseline_rate if baseline_rate > 0 else 0

        # Baseline lift
        decile_outcome_baseline = sorted_outcome_baseline[start_idx:end_idx]
        response_rate_baseline = np.mean(decile_outcome_baseline)
        lift_b = response_rate_baseline / baseline_rate if baseline_rate > 0 else 0

        deciles.append(i + 1)
        lift_model.append(lift_m)
        lift_baseline.append(lift_b)

    # Plot lift bars
    x = np.arange(n_deciles)
    width = 0.35

    ax.bar(x - width / 2, lift_model, width, color=COLORS["primary"], alpha=0.8, label="Model")
    ax.bar(x + width / 2, lift_baseline, width, color=COLORS["danger"], alpha=0.6, label="Baseline")

    # Baseline line (lift = 1)
    ax.axhline(y=1.0, color=COLORS["perfect"], linestyle="--", linewidth=1.5, alpha=0.7, label="Random (Lift=1)")

    ax.set_xlabel("Decile (1=highest predictions)", fontsize=FONT_SIZES["label"])
    ax.set_ylabel("Lift (vs random targeting)", fontsize=FONT_SIZES["label"])
    ax.set_title("Lift Chart by Decile", fontsize=FONT_SIZES["title"])
    ax.set_xticks(x)
    ax.set_xticklabels(deciles)
    ax.legend(loc="upper right", fontsize=FONT_SIZES["legend"])
    ax.grid(True, alpha=0.2, linewidth=0.5, axis="y")

    # Annotate top decile lift
    ax.annotate(
        f"Top Decile\nLift: {lift_model[0]:.2f}x",
        xy=(0, lift_model[0]),
        xytext=(1, lift_model[0] + 0.3),
        fontsize=FONT_SIZES["annotation"],
        color=COLORS["success"] if lift_model[0] > 1.5 else COLORS["warning"],
        fontweight="bold",
        arrowprops={"arrowstyle": "->", "color": COLORS["success"] if lift_model[0] > 1.5 else COLORS["warning"]},
    )

    plt.tight_layout()

    # Save
    if output_path is None:
        output_path = str(get_plot_output_path("diagnostics", "lift_chart.png"))

    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    logger.info(f"Saved Lift Chart to {output_path}")

    # Calculate summary metrics
    top_10_pct_captured = cumsum_model[len(outcome) // 10]  # % captured in top 10%
    top_20_pct_captured = cumsum_model[len(outcome) // 5]  # % captured in top 20%

    results = {
        "top_decile_lift_model": float(lift_model[0]),
        "top_decile_lift_baseline": float(lift_baseline[0]),
        "top_10_pct_captured": float(top_10_pct_captured),
        "top_20_pct_captured": float(top_20_pct_captured),
        "lift_by_decile": [float(x) for x in lift_model],
    }

    # Log summary
    logger.info("\nLift Chart Summary:")
    logger.info(f"  Top Decile Lift (Model):    {lift_model[0]:.2f}x")
    logger.info(f"  Top Decile Lift (Baseline): {lift_baseline[0]:.2f}x")
    logger.info(f"  Top 10% captures:           {top_10_pct_captured:.1f}% of positives")
    logger.info(f"  Top 20% captures:           {top_20_pct_captured:.1f}% of positives")

    # Warnings
    if lift_model[0] < 1.5:
        logger.warning(f"⚠️  Low top decile lift ({lift_model[0]:.2f}x). Model may not provide strong targeting edge.")
    if lift_model[0] < lift_baseline[0]:
        logger.warning("⚠️  Model lift is worse than baseline in top decile!")

    # Upload to W&B
    if wandb_log and WANDB_AVAILABLE and wandb.run:
        wandb.log({"diagnostics/lift_chart": wandb.Image(fig)})

    plt.close(fig)

    return results


def plot_win_rate_heatmap(
    test_df: pl.DataFrame,
    output_path: Optional[str] = None,
    wandb_log: bool = True,
    threshold: float = 0.5,
) -> dict[str, Any]:
    """
    Plot win rate heatmap by moneyness and time-to-expiry.

    Shows which market regimes (OTM/ATM/ITM × time remaining) have
    strongest/weakest prediction accuracy.

    Args:
        test_df: Test predictions DataFrame with columns:
            - final_prob: Model predictions
            - outcome: Binary outcomes
            - moneyness: Distance from strike
            - time_remaining: Seconds to expiration
        output_path: Output file path
        wandb_log: Whether to upload to W&B
        threshold: Prediction threshold for win/loss (default 0.5)

    Returns:
        Dictionary with win rates per regime
    """
    apply_plot_style()

    logger.info("Generating Win Rate Heatmap...")

    # Check required columns
    required_cols = ["final_prob", "outcome", "moneyness", "time_remaining"]
    missing_cols = [c for c in required_cols if c not in test_df.columns]
    if missing_cols:
        logger.error(f"Missing required columns: {missing_cols}")
        return {"error": f"Missing columns: {missing_cols}"}

    # Create bins
    # Moneyness bins: OTM (<-0.05), ATM (-0.05 to 0.05), ITM (>0.05)
    # Use actual min/max values instead of inf to avoid binning errors
    moneyness_min = float(test_df["moneyness"].min())
    moneyness_max = float(test_df["moneyness"].max())
    moneyness_bins = [moneyness_min - 0.01, -0.05, 0.05, moneyness_max + 0.01]
    moneyness_labels = ["OTM\n(<-0.05)", "ATM\n(-0.05,0.05)", "ITM\n(>0.05)"]

    # Time remaining bins: 0-5min, 5-10min, 10-15min
    # Use actual max value instead of fixed 900
    time_max = float(test_df["time_remaining"].max())
    time_bins = [0, 300, 600, min(900, time_max + 1)]
    time_labels = ["0-5min", "5-10min", "10-15min"]

    # Add bin columns using explicit when/then logic (more reliable than cut with labels)
    df_with_bins = test_df.with_columns(
        [
            # Moneyness bins
            pl.when(pl.col("moneyness") < -0.05)
            .then(pl.lit(moneyness_labels[0]))
            .when(pl.col("moneyness") <= 0.05)
            .then(pl.lit(moneyness_labels[1]))
            .otherwise(pl.lit(moneyness_labels[2]))
            .alias("moneyness_bin"),
            # Time bins
            pl.when(pl.col("time_remaining") < 300)
            .then(pl.lit(time_labels[0]))
            .when(pl.col("time_remaining") < 600)
            .then(pl.lit(time_labels[1]))
            .otherwise(pl.lit(time_labels[2]))
            .alias("time_bin"),
        ]
    )

    # Calculate win rate per bin
    # Win = (predicted > threshold and outcome == 1) or (predicted <= threshold and outcome == 0)
    df_with_wins = df_with_bins.with_columns(
        [
            (
                ((pl.col("final_prob") > threshold) & (pl.col("outcome") == 1))
                | ((pl.col("final_prob") <= threshold) & (pl.col("outcome") == 0))
            ).alias("win")
        ]
    )

    # Group by bins and calculate metrics
    heatmap_data = (
        df_with_wins.group_by(["time_bin", "moneyness_bin"])
        .agg(
            [
                pl.col("win").mean().alias("win_rate"),
                pl.len().alias("count"),
            ]
        )
        .sort(["time_bin", "moneyness_bin"])
    )

    # Pivot to matrix format for heatmap
    # Create matrix manually
    win_rate_matrix = np.zeros((len(time_labels), len(moneyness_labels)))
    count_matrix = np.zeros((len(time_labels), len(moneyness_labels)))

    for row in heatmap_data.iter_rows(named=True):
        time_idx = time_labels.index(row["time_bin"]) if row["time_bin"] in time_labels else -1
        money_idx = moneyness_labels.index(row["moneyness_bin"]) if row["moneyness_bin"] in moneyness_labels else -1

        if time_idx >= 0 and money_idx >= 0:
            win_rate_matrix[time_idx, money_idx] = row["win_rate"]
            count_matrix[time_idx, money_idx] = row["count"]

    # Create figure
    fig, ax = plt.subplots(figsize=(10, 8))

    # Plot heatmap

    im = ax.imshow(win_rate_matrix, cmap="RdYlGn", aspect="auto", vmin=0.4, vmax=0.6)

    # Set ticks and labels
    ax.set_xticks(np.arange(len(moneyness_labels)))
    ax.set_yticks(np.arange(len(time_labels)))
    ax.set_xticklabels(moneyness_labels, fontsize=FONT_SIZES["label"])
    ax.set_yticklabels(time_labels, fontsize=FONT_SIZES["label"])

    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label("Win Rate", fontsize=FONT_SIZES["label"], rotation=270, labelpad=20)

    # Add text annotations (win rate + sample count)
    for i in range(len(time_labels)):
        for j in range(len(moneyness_labels)):
            win_rate = win_rate_matrix[i, j]
            count = int(count_matrix[i, j])

            if count > 0:
                text_color = "white" if win_rate < 0.45 or win_rate > 0.55 else "black"
                ax.text(
                    j,
                    i,
                    f"{win_rate:.1%}\n(n={count:,})",
                    ha="center",
                    va="center",
                    color=text_color,
                    fontsize=FONT_SIZES["annotation"],
                    fontweight="bold" if count > 10000 else "normal",
                )

    # Labels and title
    ax.set_xlabel("Moneyness (Strike Distance)", fontsize=FONT_SIZES["label"], fontweight="bold")
    ax.set_ylabel("Time Remaining to Expiry", fontsize=FONT_SIZES["label"], fontweight="bold")
    ax.set_title(
        f"Win Rate Heatmap by Market Regime\n(Threshold={threshold:.2f})",
        fontsize=FONT_SIZES["title"] + 2,
        pad=15,
        fontweight="bold",
    )

    # Add grid
    ax.set_xticks(np.arange(len(moneyness_labels)) - 0.5, minor=True)
    ax.set_yticks(np.arange(len(time_labels)) - 0.5, minor=True)
    ax.grid(which="minor", color="white", linestyle="-", linewidth=2)

    plt.tight_layout()

    # Save
    if output_path is None:
        output_path = str(get_plot_output_path("diagnostics", "win_rate_heatmap.png"))

    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    logger.info(f"Saved Win Rate Heatmap to {output_path}")

    # Calculate summary statistics
    overall_win_rate = float(df_with_wins["win"].mean())  # type: ignore
    best_regime = heatmap_data.sort("win_rate", descending=True).head(1)
    worst_regime = heatmap_data.sort("win_rate").head(1)

    results = {
        "overall_win_rate": overall_win_rate,
        "best_regime": {
            "time_bin": best_regime["time_bin"][0] if len(best_regime) > 0 else None,
            "moneyness_bin": best_regime["moneyness_bin"][0] if len(best_regime) > 0 else None,
            "win_rate": float(best_regime["win_rate"][0]) if len(best_regime) > 0 else 0.0,
            "count": int(best_regime["count"][0]) if len(best_regime) > 0 else 0,
        },
        "worst_regime": {
            "time_bin": worst_regime["time_bin"][0] if len(worst_regime) > 0 else None,
            "moneyness_bin": worst_regime["moneyness_bin"][0] if len(worst_regime) > 0 else None,
            "win_rate": float(worst_regime["win_rate"][0]) if len(worst_regime) > 0 else 0.0,
            "count": int(worst_regime["count"][0]) if len(worst_regime) > 0 else 0,
        },
        "win_rate_matrix": win_rate_matrix.tolist(),
        "count_matrix": count_matrix.tolist(),
    }

    # Log summary
    logger.info("\nWin Rate Heatmap Summary:")
    logger.info(f"  Overall Win Rate:     {overall_win_rate:.1%}")
    if len(best_regime) > 0:
        logger.info(
            f"  Best Regime:          {best_regime['time_bin'][0]} × {best_regime['moneyness_bin'][0]} "
            f"({best_regime['win_rate'][0]:.1%}, n={best_regime['count'][0]:,})"
        )
    if len(worst_regime) > 0:
        logger.info(
            f"  Worst Regime:         {worst_regime['time_bin'][0]} × {worst_regime['moneyness_bin'][0]} "
            f"({worst_regime['win_rate'][0]:.1%}, n={worst_regime['count'][0]:,})"
        )

    # Warnings
    if overall_win_rate < 0.52:
        logger.warning(f"⚠️  Overall win rate is only {overall_win_rate:.1%}. Model barely beats random (50%).")

    if len(worst_regime) > 0 and worst_regime["win_rate"][0] < 0.48:
        logger.warning(
            f"⚠️  Worst regime ({worst_regime['time_bin'][0]} × {worst_regime['moneyness_bin'][0]}) "
            f"has win rate {worst_regime['win_rate'][0]:.1%}. Consider avoiding this regime."
        )

    # Upload to W&B
    if wandb_log and WANDB_AVAILABLE and wandb.run:
        wandb.log({"diagnostics/win_rate_heatmap": wandb.Image(fig)})

    plt.close(fig)

    return results


def generate_diagnostics_report(
    test_file: str,
    output_dir: str = "results/plots/diagnostics/",
    wandb_log: bool = True,
    model_path: Optional[str] = None,
) -> dict[str, Any]:
    """
    Generate comprehensive diagnostic report.

    Creates:
    - ROC curves (overall + regime stratified)
    - Precision-Recall curves
    - QQ plots for residuals
    - Lift Chart & Cumulative Gains
    - Win Rate Heatmap
    - SHAP dependence plots (if model_path provided)

    Args:
        test_file: Path to test predictions parquet
        output_dir: Output directory for plots
        wandb_log: Whether to upload to W&B
        model_path: Path to trained model for SHAP analysis (optional)

    Returns:
        Dictionary with all diagnostic metrics
    """
    logger.info("=" * 80)
    logger.info("GENERATING ADVANCED DIAGNOSTICS REPORT")
    logger.info("=" * 80)

    output_dir_path = Path(output_dir)
    output_dir_path.mkdir(parents=True, exist_ok=True)

    # Load test data
    logger.info(f"Loading test data from {test_file}")
    test_df = pl.read_parquet(test_file)
    logger.info(f"Loaded {len(test_df):,} test samples")

    results = {}

    # 1. ROC Curve
    logger.info("\n[1/5] Generating ROC curves...")
    try:
        roc_results = plot_roc_curve(
            test_df=test_df,
            output_path=str(output_dir_path / "roc_curve.png"),
            wandb_log=wandb_log,
        )
        results["roc"] = roc_results
    except Exception as e:
        logger.error(f"Failed to generate ROC curves: {e}")
        results["roc"] = {"error": str(e)}

    # 2. Precision-Recall Curve
    logger.info("\n[2/5] Generating Precision-Recall curves...")
    try:
        pr_results = plot_precision_recall(
            test_df=test_df,
            output_path=str(output_dir_path / "precision_recall.png"),
            wandb_log=wandb_log,
        )
        results["precision_recall"] = pr_results
    except Exception as e:
        logger.error(f"Failed to generate Precision-Recall curves: {e}")
        results["precision_recall"] = {"error": str(e)}

    # 3. QQ Plot for Residuals
    logger.info("\n[3/5] Generating QQ plots for residuals...")
    try:
        qq_results = plot_qq_residuals(
            test_df=test_df,
            output_path=str(output_dir_path / "qq_plot_residuals.png"),
            wandb_log=wandb_log,
        )
        results["qq_plot"] = qq_results
    except Exception as e:
        logger.error(f"Failed to generate QQ plots: {e}")
        results["qq_plot"] = {"error": str(e)}

    # 4. Lift Chart & Cumulative Gains
    logger.info("\n[4/5] Generating Lift Chart...")
    try:
        lift_results = plot_lift_chart(
            test_df=test_df,
            output_path=str(output_dir_path / "lift_chart.png"),
            wandb_log=wandb_log,
        )
        results["lift_chart"] = lift_results
    except Exception as e:
        logger.error(f"Failed to generate Lift Chart: {e}")
        results["lift_chart"] = {"error": str(e)}

    # 5. Win Rate Heatmap
    max_steps = 6 if model_path else 5
    logger.info(f"\n[5/{max_steps}] Generating Win Rate Heatmap...")
    try:
        heatmap_results = plot_win_rate_heatmap(
            test_df=test_df,
            output_path=str(output_dir_path / "win_rate_heatmap.png"),
            wandb_log=wandb_log,
        )
        results["win_rate_heatmap"] = heatmap_results
    except Exception as e:
        logger.error(f"Failed to generate Win Rate Heatmap: {e}")
        results["win_rate_heatmap"] = {"error": str(e)}

    # 6. SHAP Dependence Analysis (optional)
    if model_path:
        logger.info(f"\n[6/{max_steps}] Generating SHAP Dependence Plots...")
        try:
            shap_results = plot_shap_dependence_analysis(
                test_df=test_df,
                model_path=model_path,
                top_n_features=10,
                output_dir=str(output_dir_path),
                wandb_log=wandb_log,
                downsample_to=100_000,  # Critical for memory
            )
            results["shap"] = shap_results
        except Exception as e:
            logger.error(f"Failed to generate SHAP plots: {e}")
            results["shap"] = {"error": str(e)}
    else:
        logger.info("\n[6/6] SHAP analysis skipped (no model path provided)")

    logger.info("\n" + "=" * 80)
    logger.info("DIAGNOSTICS REPORT COMPLETE")
    logger.info("=" * 80)

    return results


def plot_shap_dependence_analysis(
    test_df: pl.DataFrame,
    model_path: str,
    top_n_features: int = 10,
    output_dir: Optional[str] = None,
    wandb_log: bool = True,
    downsample_to: int = 100_000,
) -> dict[str, Any]:
    """
    Plot SHAP dependence plots for top features.

    Uses SHAP (SHapley Additive exPlanations) to visualize feature
    interactions and their impact on model predictions.

    Args:
        test_df: Test predictions with feature columns
        model_path: Path to saved LightGBM model file (.txt or .pkl)
        top_n_features: Number of top features to plot (default: 10)
        output_dir: Directory to save plots (optional, auto-generated if None)
        wandb_log: Whether to upload to W&B
        downsample_to: Downsample to this many rows for performance

    Returns:
        Dictionary with SHAP statistics and feature importance
    """
    import lightgbm as lgb
    import shap

    logger.info("Generating SHAP dependence plots...")

    # Downsample if needed
    if len(test_df) > downsample_to:
        logger.info(f"Downsampling from {len(test_df):,} to {downsample_to:,} rows for SHAP analysis")
        test_df = test_df.sample(n=downsample_to, shuffle=True, seed=42)

    # Load model
    logger.info(f"Loading model from {model_path}...")
    if model_path.endswith(".txt"):
        model = lgb.Booster(model_file=model_path)
    else:
        import pickle

        with open(model_path, "rb") as f:
            model = pickle.load(f)

    # Get feature names from model (these are the exact features it was trained on)
    model_features = model.feature_name()

    # Debug: log dataframe columns
    logger.info(f"Test dataframe has {len(test_df.columns)} columns")

    # Filter to only columns that exist in the dataframe and match model features
    feature_cols = [col for col in model_features if col in test_df.columns]

    if len(feature_cols) != len(model_features):
        missing = set(model_features) - set(test_df.columns)
        logger.error(f"Missing {len(missing)} features from test data:")
        for feat in sorted(missing):
            logger.error(f"  - {feat}")
        logger.error(f"This will cause SHAP to fail. Skipping SHAP analysis.")
        return {"error": f"Missing {len(missing)} features: {list(missing)}"}

    logger.info(f"Using {len(feature_cols)} features (model expects {len(model_features)})")
    X = test_df.select(feature_cols).to_pandas()  # noqa: N806

    logger.info(f"Computing SHAP values for {len(X):,} samples and {len(feature_cols)} features...")

    # Create SHAP explainer
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)

    # If binary classification, extract positive class SHAP values
    if isinstance(shap_values, list):
        shap_values = shap_values[1]  # Positive class

    # Calculate feature importance
    feature_importance = np.abs(shap_values).mean(axis=0)
    feature_importance_df = pl.DataFrame(
        {
            "feature": feature_cols,
            "importance": feature_importance,
        }
    ).sort("importance", descending=True)

    logger.info(f"\nTop {top_n_features} features by SHAP importance:")
    for i, row in enumerate(feature_importance_df.head(top_n_features).iter_rows(named=True), 1):
        logger.info(f"  {i}. {row['feature']}: {row['importance']:.4f}")

    # Select top N features
    top_features = feature_importance_df["feature"].head(top_n_features).to_list()

    # Set output directory
    if output_dir is None:
        output_dir = str(get_plot_output_path("diagnostics", "shap"))
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Create dependence plots for each top feature
    logger.info(f"\nGenerating dependence plots for top {top_n_features} features...")
    plot_paths = []

    for i, feature in enumerate(top_features, 1):
        logger.info(f"  [{i}/{top_n_features}] {feature}")

        _fig, ax = plt.subplots(figsize=(10, 6))
        apply_plot_style()

        # Get feature index
        feature_idx = feature_cols.index(feature)

        # SHAP dependence plot
        shap.dependence_plot(
            feature_idx,
            shap_values,
            X,
            feature_names=feature_cols,
            ax=ax,
            show=False,
            dot_size=30,
            alpha=0.6,
        )

        # Styling
        ax.set_xlabel(feature, fontsize=FONT_SIZES["label"] + 2, fontweight="bold")
        ax.set_ylabel(f"SHAP value for {feature}", fontsize=FONT_SIZES["label"] + 2, fontweight="bold")
        ax.set_title(
            f"SHAP Dependence Plot: {feature}\n(Rank #{i} by importance)",
            fontsize=FONT_SIZES["title"] + 2,
            fontweight="bold",
            pad=15,
        )
        ax.grid(True, alpha=0.2, linewidth=0.5)

        # Save plot
        plot_filename = f"shap_dependence_{i:02d}_{feature.replace('/', '_')}.png"
        plot_path = str(Path(output_dir) / plot_filename)
        plt.savefig(plot_path, dpi=300, bbox_inches="tight", facecolor="white")
        plot_paths.append(plot_path)

        # Upload to W&B
        if wandb_log:
            upload_plot(plot_path, f"shap_dependence_{feature}")

        plt.close()

    # Create summary plot
    logger.info("\nGenerating SHAP summary plot...")
    _fig = plt.figure(figsize=(12, 10))
    apply_plot_style()

    shap.summary_plot(
        shap_values,
        X,
        feature_names=feature_cols,
        max_display=top_n_features,
        show=False,
        plot_size=(12, 10),
    )

    plt.title(
        f"SHAP Summary Plot (Top {top_n_features} Features)",
        fontsize=FONT_SIZES["title"] + 4,
        fontweight="bold",
        pad=15,
    )

    # Save summary plot
    summary_path = str(Path(output_dir) / "shap_summary.png")
    plt.savefig(summary_path, dpi=300, bbox_inches="tight", facecolor="white")
    logger.info(f"Saved SHAP summary plot to {summary_path}")

    if wandb_log:
        upload_plot(summary_path, "shap_summary")

    plt.close()

    # Return statistics
    return {
        "top_features": top_features,
        "feature_importance": feature_importance_df.to_dicts(),
        "n_samples": len(X),
        "n_features": len(feature_cols),
        "plot_paths": plot_paths + [summary_path],
    }


def main() -> None:
    """CLI entry point for advanced diagnostics."""
    import argparse

    parser = argparse.ArgumentParser(description="Advanced diagnostic plots for model evaluation")
    parser.add_argument(
        "--test-file",
        type=str,
        required=True,
        help="Path to test predictions parquet file",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results/plots/diagnostics/",
        help="Output directory for plots",
    )
    parser.add_argument(
        "--no-wandb",
        action="store_true",
        help="Disable W&B upload",
    )

    args = parser.parse_args()

    generate_diagnostics_report(
        test_file=args.test_file,
        output_dir=args.output_dir,
        wandb_log=not args.no_wandb,
    )


if __name__ == "__main__":
    main()
