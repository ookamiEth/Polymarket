#!/usr/bin/env python3
"""
Model Analysis and Visualization for LightGBM Residual Predictions.

Provides calibration plots, feature importance, and residual analysis
for evaluating model performance.
"""

import logging
from pathlib import Path
from typing import Any, Optional

import lightgbm as lgb
import matplotlib.pyplot as plt
import numpy as np
import polars as pl

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# Output directory (portable across any CWD)
OUTPUT_DIR = Path(__file__).parent.parent / "results"
OUTPUT_DIR.mkdir(exist_ok=True)


def create_calibration_plot(
    predictions_df: pl.DataFrame,
    output_path: str,
    brier_score: Optional[float] = None,
    title_suffix: str = "",
) -> None:
    """
    Create calibration plot comparing predicted vs actual probabilities.

    Args:
        predictions_df: DataFrame with 'final_prob' and 'outcome' columns
        output_path: Path to save plot
        brier_score: Optional Brier score to display in title
        title_suffix: Optional suffix to add to title (e.g., "Trial 15")
    """
    logger.info(f"Creating calibration plot: {output_path}")

    # Extract predictions and outcomes
    pred = predictions_df["final_prob"].to_numpy()
    actual = predictions_df["outcome"].to_numpy()

    # Bin predictions (10 bins from 0 to 1)
    bins = np.linspace(0, 1, 11)
    bin_centers = (bins[:-1] + bins[1:]) / 2

    calibration = []
    counts = []
    for i in range(len(bins) - 1):
        # Create mask for this bin
        mask = (pred >= bins[i]) & (pred < bins[i + 1])
        if i == len(bins) - 2:
            # Include upper bound in last bin
            mask = (pred >= bins[i]) & (pred <= bins[i + 1])

        if mask.sum() > 0:
            calibration.append(actual[mask].mean())
            counts.append(mask.sum())
        else:
            calibration.append(np.nan)
            counts.append(0)

    # Calculate calibration metrics
    valid_mask = ~np.isnan(calibration)
    if valid_mask.sum() > 0:
        mse_calibration = np.mean((np.array(calibration)[valid_mask] - bin_centers[valid_mask]) ** 2)
        logger.info(f"Calibration MSE: {mse_calibration:.6f}")

    # Plot
    plt.figure(figsize=(10, 8))

    # Scatter plot with size proportional to sample count
    sizes = np.clip(np.array(counts) / 1000, 30, 200)
    plt.scatter(
        bin_centers,
        calibration,
        s=sizes,
        alpha=0.7,
        edgecolor="white",
        linewidth=1,
        color="#00D4FF",
        label="Observed frequency",
    )

    # Perfect calibration line
    plt.plot([0, 1], [0, 1], "k--", alpha=0.5, linewidth=1.5, label="Perfect calibration")

    # Formatting
    plt.xlabel("Predicted Probability", fontsize=12)
    plt.ylabel("Actual Frequency", fontsize=12)

    title = "Calibration Plot"
    if title_suffix:
        title += f" - {title_suffix}"
    if brier_score is not None:
        title += f"\nBrier Score: {brier_score:.4f}"

    plt.title(title, fontsize=14)
    plt.legend(frameon=True, alpha=0.9, fontsize=10)
    plt.grid(alpha=0.3)
    plt.xlim(-0.05, 1.05)
    plt.ylim(-0.05, 1.05)

    # Add annotation about dot sizes
    plt.text(
        0.02,
        0.98,
        f"Dot size ∝ sample count\n(n={len(pred):,} total)",
        transform=plt.gca().transAxes,
        fontsize=9,
        verticalalignment="top",
        bbox={"boxstyle": "round", "alpha": 0.1},
    )

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()

    logger.info(f"✓ Calibration plot saved to {output_path}")


def analyze_residuals(
    predictions_df: pl.DataFrame,
    output_path: Optional[str] = None,
) -> dict[str, Any]:
    """
    Analyze residual distribution and create histogram.

    Args:
        predictions_df: DataFrame with 'residual_pred' column
        output_path: Optional path to save histogram

    Returns:
        Dictionary with residual statistics
    """
    logger.info("Analyzing residual distribution...")

    residuals = predictions_df["residual_pred"].to_numpy()

    # Calculate statistics
    stats = {
        "mean": residuals.mean(),
        "std": residuals.std(),
        "q25": np.percentile(residuals, 25),
        "median": np.percentile(residuals, 50),
        "q75": np.percentile(residuals, 75),
        "min": residuals.min(),
        "max": residuals.max(),
    }

    logger.info("Residual Distribution:")
    logger.info(f"  Mean:   {stats['mean']:.6f}")
    logger.info(f"  Std:    {stats['std']:.6f}")
    logger.info(f"  Q25:    {stats['q25']:.6f}")
    logger.info(f"  Median: {stats['median']:.6f}")
    logger.info(f"  Q75:    {stats['q75']:.6f}")
    logger.info(f"  Min:    {stats['min']:.6f}")
    logger.info(f"  Max:    {stats['max']:.6f}")

    # Create histogram if output path provided
    if output_path:
        plt.figure(figsize=(10, 6))

        plt.hist(
            residuals,
            bins=50,
            alpha=0.7,
            edgecolor="white",
            linewidth=0.5,
            color="#00D4FF",
        )

        # Add mean and median lines
        plt.axvline(
            stats["mean"],
            color="#FF3366",
            linestyle="--",
            linewidth=2,
            label=f"Mean: {stats['mean']:.4f}",
        )
        plt.axvline(
            stats["median"],
            color="#00FF88",
            linestyle="--",
            linewidth=2,
            label=f"Median: {stats['median']:.4f}",
        )

        plt.xlabel("Residual Value", fontsize=12)
        plt.ylabel("Frequency", fontsize=12)
        plt.title("Residual Distribution", fontsize=14)
        plt.legend(frameon=True, alpha=0.9)
        plt.grid(alpha=0.3)

        # Add statistics box
        stats_text = f"n = {len(residuals):,}\nStd = {stats['std']:.4f}"
        plt.text(
            0.98,
            0.98,
            stats_text,
            transform=plt.gca().transAxes,
            fontsize=10,
            verticalalignment="top",
            horizontalalignment="right",
            bbox={"boxstyle": "round", "alpha": 0.1},
        )

        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close()

        logger.info(f"✓ Residual histogram saved to {output_path}")

    return stats


def plot_feature_importance(
    model: lgb.Booster,
    output_path: str,
    top_n: int = 30,
    title_suffix: str = "",
) -> None:
    """
    Plot feature importance from LightGBM model.

    Args:
        model: Trained LightGBM model
        output_path: Path to save plot
        top_n: Number of top features to display
        title_suffix: Optional suffix for plot title
    """
    logger.info(f"Creating feature importance plot: {output_path}")

    # Get importance
    importance = model.feature_importance(importance_type="gain")
    feature_names = model.feature_name()

    # Sort and take top N
    sorted_idx = np.argsort(importance)[::-1][:top_n]
    sorted_features = [feature_names[i] for i in sorted_idx]
    sorted_importance = importance[sorted_idx]

    # Normalize for better visualization
    sorted_importance = sorted_importance / sorted_importance.sum() * 100

    # Plot
    plt.figure(figsize=(10, max(8, top_n * 0.3)))

    bars = plt.barh(range(len(sorted_features)), sorted_importance, color="#00D4FF")

    # Color gradient (darker for more important)
    for i, bar in enumerate(bars):
        alpha = 0.5 + 0.5 * (len(bars) - i) / len(bars)
        bar.set_alpha(alpha)

    plt.yticks(range(len(sorted_features)), sorted_features, fontsize=10)
    plt.xlabel("Importance (% of total gain)", fontsize=12)

    title = f"Top {top_n} Features by Gain"
    if title_suffix:
        title += f" - {title_suffix}"

    plt.title(title, fontsize=14)
    plt.grid(alpha=0.3, axis="x")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()

    logger.info(f"✓ Feature importance plot saved to {output_path}")

    # Log top 10
    logger.info("Top 10 Features:")
    for i, (feat, imp) in enumerate(zip(sorted_features[:10], sorted_importance[:10]), 1):
        logger.info(f"  {i:2d}. {feat:40s}: {imp:6.2f}%")


def analyze_model(
    model_path: str,
    predictions_file: Optional[str] = None,
    output_dir: Optional[str] = None,
    title_suffix: str = "",
) -> dict[str, Any]:
    """
    Comprehensive model analysis with all visualizations.

    Args:
        model_path: Path to saved LightGBM model (.txt or .json)
        predictions_file: Optional path to predictions parquet file
        output_dir: Optional output directory (defaults to ../results)
        title_suffix: Suffix for plot titles (e.g., "Trial 15")

    Returns:
        Dictionary with analysis results
    """
    output_path: Path = Path(output_dir) if output_dir else OUTPUT_DIR
    output_path.mkdir(exist_ok=True)

    results = {}

    # Load model
    logger.info(f"Loading model from {model_path}")
    model = lgb.Booster(model_file=model_path)
    results["num_features"] = model.num_feature()
    results["num_trees"] = model.num_trees()
    results["best_iteration"] = model.best_iteration

    logger.info(f"Model loaded: {results['num_features']} features, {results['num_trees']} trees")

    # Feature importance plot
    feature_importance_path = output_path / f"feature_importance_{title_suffix.lower().replace(' ', '_')}.png"
    plot_feature_importance(
        model,
        str(feature_importance_path),
        top_n=30,
        title_suffix=title_suffix,
    )

    # If predictions file provided, create calibration and residual plots
    if predictions_file and Path(predictions_file).exists():
        logger.info(f"Loading predictions from {predictions_file}")
        predictions_df = pl.read_parquet(predictions_file)

        # Calculate Brier score if possible
        brier_score = None
        if "final_prob" in predictions_df.columns and "outcome" in predictions_df.columns:
            pred = predictions_df["final_prob"].to_numpy()
            actual = predictions_df["outcome"].to_numpy()
            brier_score = np.mean((pred - actual) ** 2)
            results["brier_score"] = brier_score
            logger.info(f"Brier score: {brier_score:.6f}")

            # Calibration plot
            calibration_path = output_path / f"calibration_{title_suffix.lower().replace(' ', '_')}.png"
            create_calibration_plot(
                predictions_df,
                str(calibration_path),
                brier_score=float(brier_score),
                title_suffix=title_suffix,
            )

        # Residual analysis
        if "residual_pred" in predictions_df.columns:
            residual_path = output_path / f"residuals_{title_suffix.lower().replace(' ', '_')}.png"
            residual_stats = analyze_residuals(predictions_df, str(residual_path))
            results["residual_stats"] = residual_stats

    logger.info("=" * 80)
    logger.info("MODEL ANALYSIS COMPLETE")
    logger.info("=" * 80)

    return results


def main() -> None:
    """Run model analysis on best trial."""
    import argparse

    parser = argparse.ArgumentParser(description="Analyze LightGBM model performance")
    parser.add_argument(
        "--model",
        type=str,
        default="../results/lightgbm_model_optimized.txt",
        help="Path to model file",
    )
    parser.add_argument(
        "--predictions",
        type=str,
        help="Path to predictions parquet file (optional)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="../results",
        help="Output directory for plots",
    )
    parser.add_argument(
        "--title-suffix",
        type=str,
        default="Best Model",
        help="Suffix for plot titles",
    )

    args = parser.parse_args()

    analyze_model(
        model_path=args.model,
        predictions_file=args.predictions,
        output_dir=args.output_dir,
        title_suffix=args.title_suffix,
    )


if __name__ == "__main__":
    main()
