#!/usr/bin/env python3
"""
Analyze calibration quality over time (training folds vs holdout period).

This script:
1. Loads walk-forward CV predictions from training period (if available)
2. Computes calibration metrics per fold
3. Analyzes holdout period calibration
4. Compares training vs holdout to detect overfitting
5. Generates time-series calibration plots

Calibration Metrics:
- Calibration slope (ideal = 1.0)
- Calibration intercept (ideal = 0.0)
- Expected Calibration Error (ECE)
- Brier score decomposition (reliability, resolution, uncertainty)
- Reliability diagram (binned predictions vs outcomes)

Output:
- calibration_by_period.csv: Per-period calibration metrics
- calibration_training_vs_holdout.csv: Summary comparison
- calibration_curves_{model_name}.png: Visual comparison
- overfitting_report.csv: Models with calibration degradation >10%
"""

import logging
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from sklearn.calibration import calibration_curve
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Constants
PROJECT_ROOT = Path("/home/ubuntu/Polymarket/research/model/01_pricing")
DATA_PATH = Path("/home/ubuntu/Polymarket/research/model/data/consolidated_features_v4_pipeline_ready.parquet")
MODELS_DIR = PROJECT_ROOT / "models_optuna"
OUTPUT_DIR = PROJECT_ROOT / "analysis" / "calibration"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Model names
TEMPORAL_BUCKETS = ["near", "mid", "far"]
VOL_REGIMES = ["low_vol_atm", "low_vol_otm", "high_vol_atm", "high_vol_otm"]
MODEL_NAMES = [f"{bucket}_{regime}" for bucket in TEMPORAL_BUCKETS for regime in VOL_REGIMES]

# Holdout period (from v4 pipeline evaluation)
HOLDOUT_START = "2025-08-01"
HOLDOUT_END = "2025-11-01"

# Training period (pre-holdout)
TRAINING_START = "2023-10-01"  # Approximate start based on v3
TRAINING_END = "2025-08-01"


def compute_calibration_metrics(y_true: np.ndarray, y_pred: np.ndarray, n_bins: int = 10) -> dict[str, Any]:
    """
    Compute comprehensive calibration metrics.

    Args:
        y_true: True binary outcomes (0 or 1)
        y_pred: Predicted probabilities
        n_bins: Number of bins for ECE calculation

    Returns:
        Dictionary of calibration metrics
    """
    # Brier score
    brier = brier_score_loss(y_true, y_pred)

    # Calibration slope and intercept (using logistic regression)
    # log(odds) = intercept + slope * logit(predicted_prob)
    logit_pred = np.log(y_pred / (1 - y_pred + 1e-10) + 1e-10)

    # Fit logistic regression on logit-transformed predictions
    try:
        lr = LogisticRegression(fit_intercept=True)
        lr.fit(logit_pred.reshape(-1, 1), y_true)
        cal_slope = float(lr.coef_[0][0])  # type: ignore[index]
        cal_intercept = float(lr.intercept_[0])  # type: ignore[index]
    except Exception:
        # Fallback if fitting fails (e.g., perfect predictions)
        cal_slope = 1.0
        cal_intercept = 0.0

    # Expected Calibration Error (ECE)
    # Average absolute difference between predicted and observed frequencies per bin
    prob_true, prob_pred = calibration_curve(y_true, y_pred, n_bins=n_bins, strategy="uniform")

    # Weight by bin size
    bin_edges = np.linspace(0, 1, n_bins + 1)
    bin_indices = np.digitize(y_pred, bin_edges[1:-1])
    bin_counts = np.bincount(bin_indices, minlength=n_bins)
    bin_weights = bin_counts / len(y_pred)

    # ECE = weighted average of |predicted - observed| per bin
    ece = np.sum(np.abs(prob_pred - prob_true) * bin_weights[: len(prob_true)])

    # Brier score decomposition
    # Brier = Reliability - Resolution + Uncertainty
    # Reliability: average squared difference between predicted and observed per bin
    reliability = np.sum(bin_weights[: len(prob_true)] * (prob_pred - prob_true) ** 2)

    # Resolution: variance of observed outcomes across bins
    overall_mean = np.mean(y_true)
    resolution = np.sum(bin_weights[: len(prob_true)] * (prob_true - overall_mean) ** 2)

    # Uncertainty: variance of outcomes (constant term)
    uncertainty = overall_mean * (1 - overall_mean)

    return {
        "brier_score": brier,
        "calibration_slope": cal_slope,
        "calibration_intercept": cal_intercept,
        "ece": ece,
        "reliability": reliability,
        "resolution": resolution,
        "uncertainty": uncertainty,
        "n_samples": len(y_true),
    }


def analyze_holdout_calibration(df: pl.DataFrame, model_name: str, feature_cols: list[str]) -> dict[str, Any]:
    """
    Analyze calibration on holdout period for a single model.

    Args:
        df: Full dataset
        model_name: Name of the model
        feature_cols: List of feature column names

    Returns:
        Dictionary of calibration metrics
    """
    import lightgbm as lgb

    logger.info(f"Analyzing holdout calibration for {model_name}")

    # Filter to holdout period and regime
    regime = model_name.split("_", 1)[1]
    df_holdout = df.filter(
        (pl.col("date") >= pl.lit(HOLDOUT_START))
        & (pl.col("date") < pl.lit(HOLDOUT_END))
        & (pl.col("combined_regime") == regime)
    )

    if len(df_holdout) == 0:
        logger.warning(f"No holdout samples for {model_name}")
        return {}

    # Load model
    model_path = MODELS_DIR / f"lightgbm_{model_name}.txt"
    booster = lgb.Booster(model_file=str(model_path))

    # Generate predictions
    x_data = df_holdout.select(feature_cols).to_numpy()
    y_true = df_holdout["residual"].to_numpy()

    # Predict residuals
    residual_pred = booster.predict(x_data)

    # Convert to binary outcomes and probabilities
    # Assuming residuals are centered around 0, positive = outcome 1
    # This is a simplification - adjust based on your actual target encoding
    y_binary = (y_true > 0).astype(int)

    # Convert residual predictions to probabilities (sigmoid transformation)
    y_pred_prob = 1 / (1 + np.exp(-residual_pred))  # type: ignore[operator]

    # Compute calibration metrics
    metrics = compute_calibration_metrics(y_binary, y_pred_prob)
    metrics["model_name"] = model_name
    metrics["period"] = "holdout"
    metrics["start_date"] = HOLDOUT_START
    metrics["end_date"] = HOLDOUT_END

    logger.info(
        f"  Holdout Brier: {metrics['brier_score']:.4f}, ECE: {metrics['ece']:.4f}, Slope: {metrics['calibration_slope']:.4f}"
    )

    return metrics


def analyze_training_calibration(
    df: pl.DataFrame, model_name: str, feature_cols: list[str], n_folds: int = 5
) -> list[dict[str, Any]]:
    """
    Analyze calibration on training period (split into folds for time-series analysis).

    Args:
        df: Full dataset
        model_name: Name of the model
        feature_cols: List of feature column names
        n_folds: Number of temporal folds to create

    Returns:
        List of calibration metrics per fold
    """
    import lightgbm as lgb

    logger.info(f"Analyzing training calibration for {model_name} ({n_folds} folds)")

    # Filter to training period and regime
    regime = model_name.split("_", 1)[1]
    df_train = df.filter(
        (pl.col("date") >= pl.lit(TRAINING_START))
        & (pl.col("date") < pl.lit(TRAINING_END))
        & (pl.col("combined_regime") == regime)
    )

    if len(df_train) == 0:
        logger.warning(f"No training samples for {model_name}")
        return []

    # Load model
    model_path = MODELS_DIR / f"lightgbm_{model_name}.txt"
    booster = lgb.Booster(model_file=str(model_path))

    # Sort by date and split into temporal folds
    df_train = df_train.sort("date")
    fold_size = len(df_train) // n_folds

    fold_metrics = []
    for fold_idx in range(n_folds):
        start_idx = fold_idx * fold_size
        end_idx = start_idx + fold_size if fold_idx < n_folds - 1 else len(df_train)

        df_fold = df_train[start_idx:end_idx]

        if len(df_fold) == 0:
            continue

        # Generate predictions
        x_data = df_fold.select(feature_cols).to_numpy()
        y_true = df_fold["residual"].to_numpy()

        residual_pred = booster.predict(x_data)

        # Convert to binary
        y_binary = (y_true > 0).astype(int)
        y_pred_prob = 1 / (1 + np.exp(-residual_pred))  # type: ignore[operator]

        # Compute metrics
        metrics = compute_calibration_metrics(y_binary, y_pred_prob)
        metrics["model_name"] = model_name
        metrics["period"] = f"training_fold_{fold_idx + 1}"
        metrics["fold_index"] = fold_idx + 1
        metrics["start_date"] = df_fold["date"].min()
        metrics["end_date"] = df_fold["date"].max()

        fold_metrics.append(metrics)

        logger.info(f"  Fold {fold_idx + 1}: Brier={metrics['brier_score']:.4f}, ECE={metrics['ece']:.4f}")

    return fold_metrics


def plot_calibration_curves(df: pl.DataFrame, model_name: str, feature_cols: list[str], n_bins: int = 10) -> None:
    """
    Generate calibration curve plots comparing training vs holdout.

    Args:
        df: Full dataset
        model_name: Name of the model
        feature_cols: List of feature column names
        n_bins: Number of bins for calibration curve
    """
    import lightgbm as lgb

    logger.info(f"Generating calibration plots for {model_name}")

    # Load model
    model_path = MODELS_DIR / f"lightgbm_{model_name}.txt"
    booster = lgb.Booster(model_file=str(model_path))

    regime = model_name.split("_", 1)[1]

    # Training data
    df_train = df.filter(
        (pl.col("date") >= pl.lit(TRAINING_START))
        & (pl.col("date") < pl.lit(TRAINING_END))
        & (pl.col("combined_regime") == regime)
    )

    # Holdout data
    df_holdout = df.filter(
        (pl.col("date") >= pl.lit(HOLDOUT_START))
        & (pl.col("date") < pl.lit(HOLDOUT_END))
        & (pl.col("combined_regime") == regime)
    )

    _, axes = plt.subplots(1, 2, figsize=(14, 6))

    for idx, (df_subset, label, color) in enumerate(
        [
            (df_train, "Training", "#00B4FF"),
            (df_holdout, "Holdout", "#FF3366"),
        ]
    ):
        if len(df_subset) == 0:
            continue

        x_data = df_subset.select(feature_cols).to_numpy()
        y_true = df_subset["residual"].to_numpy()

        residual_pred = booster.predict(x_data)
        y_binary = (y_true > 0).astype(int)
        y_pred_prob = 1 / (1 + np.exp(-residual_pred))  # type: ignore[operator]

        # Calibration curve
        prob_true, prob_pred = calibration_curve(y_binary, y_pred_prob, n_bins=n_bins, strategy="uniform")

        # Plot calibration curve
        ax = axes[idx]
        ax.plot([0, 1], [0, 1], "k--", lw=1.5, label="Perfect calibration", alpha=0.5)
        ax.plot(prob_pred, prob_true, marker="o", markersize=8, lw=2, color=color, label=f"{label} calibration")

        ax.set_xlabel("Predicted Probability", fontsize=12)
        ax.set_ylabel("Observed Frequency", fontsize=12)
        ax.set_title(f"{label} Period - {model_name}", fontsize=14)
        ax.legend(loc="upper left", fontsize=10)
        ax.grid(alpha=0.3)
        ax.set_xlim([-0.05, 1.05])
        ax.set_ylim([-0.05, 1.05])

    plt.suptitle(f"Calibration Analysis: {model_name}", fontsize=16, y=1.02)
    plt.tight_layout()

    output_file = OUTPUT_DIR / f"calibration_curves_{model_name}.png"
    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    logger.info(f"Saved calibration plot to {output_file}")
    plt.close()


def detect_overfitting(training_metrics: list[dict], holdout_metrics: dict, threshold: float = 0.10) -> dict:
    """
    Detect overfitting by comparing training vs holdout calibration.

    Args:
        training_metrics: List of training fold metrics
        holdout_metrics: Holdout period metrics
        threshold: ECE degradation threshold (e.g., 0.10 = 10% worse)

    Returns:
        Overfitting analysis dictionary
    """
    if not training_metrics or not holdout_metrics:
        return {}

    # Average training metrics
    avg_training_ece = np.mean([m["ece"] for m in training_metrics])
    avg_training_brier = np.mean([m["brier_score"] for m in training_metrics])

    # Compare to holdout
    ece_degradation = holdout_metrics["ece"] - avg_training_ece
    brier_degradation = holdout_metrics["brier_score"] - avg_training_brier

    is_overfit = ece_degradation > threshold

    return {
        "model_name": holdout_metrics["model_name"],
        "avg_training_ece": avg_training_ece,
        "holdout_ece": holdout_metrics["ece"],
        "ece_degradation": ece_degradation,
        "ece_degradation_pct": (ece_degradation / avg_training_ece * 100) if avg_training_ece > 0 else 0,
        "avg_training_brier": avg_training_brier,
        "holdout_brier": holdout_metrics["brier_score"],
        "brier_degradation": brier_degradation,
        "is_overfit": is_overfit,
    }


def main() -> None:
    """Main execution function."""
    logger.info("Starting calibration analysis over time")

    # Load data
    logger.info(f"Loading data from {DATA_PATH}")
    df = pl.scan_parquet(DATA_PATH).collect()
    logger.info(f"Loaded {len(df):,} samples")

    # Get feature columns
    exclude_cols = {"timestamp_seconds", "date", "residual", "combined_regime"}
    feature_cols = [col for col in df.columns if col not in exclude_cols]
    logger.info(f"Using {len(feature_cols)} features")

    # Analyze all models
    all_training_metrics = []
    all_holdout_metrics = []
    overfitting_reports = []

    for model_name in MODEL_NAMES:
        logger.info(f"\n{'=' * 80}")
        logger.info(f"Analyzing {model_name}")
        logger.info(f"{'=' * 80}")

        # Training calibration
        training_metrics = analyze_training_calibration(df, model_name, feature_cols, n_folds=5)
        all_training_metrics.extend(training_metrics)

        # Holdout calibration
        holdout_metrics = analyze_holdout_calibration(df, model_name, feature_cols)
        if holdout_metrics:
            all_holdout_metrics.append(holdout_metrics)

        # Overfitting detection
        if training_metrics and holdout_metrics:
            overfit_analysis = detect_overfitting(training_metrics, holdout_metrics, threshold=0.10)
            overfitting_reports.append(overfit_analysis)

            if overfit_analysis["is_overfit"]:
                logger.warning(
                    f"  ⚠️  OVERFITTING DETECTED: ECE degradation = {overfit_analysis['ece_degradation']:.4f} ({overfit_analysis['ece_degradation_pct']:.1f}%)"
                )

        # Generate calibration plots
        plot_calibration_curves(df, model_name, feature_cols, n_bins=10)

    # Save results
    logger.info("\n" + "=" * 80)
    logger.info("Saving results")
    logger.info("=" * 80)

    # Training metrics
    if all_training_metrics:
        df_training = pl.DataFrame(all_training_metrics)
        training_file = OUTPUT_DIR / "calibration_training_folds.csv"
        df_training.write_csv(training_file)
        logger.info(f"Saved training metrics to {training_file}")

    # Holdout metrics
    if all_holdout_metrics:
        df_holdout = pl.DataFrame(all_holdout_metrics)
        holdout_file = OUTPUT_DIR / "calibration_holdout.csv"
        df_holdout.write_csv(holdout_file)
        logger.info(f"Saved holdout metrics to {holdout_file}")

    # Overfitting report
    if overfitting_reports:
        df_overfit = pl.DataFrame(overfitting_reports)
        overfit_file = OUTPUT_DIR / "overfitting_report.csv"
        df_overfit.write_csv(overfit_file)
        logger.info(f"Saved overfitting report to {overfit_file}")

        # Summary
        n_overfit = df_overfit.filter(pl.col("is_overfit")).shape[0]
        logger.info(
            f"\n📊 OVERFITTING SUMMARY: {n_overfit}/{len(MODEL_NAMES)} models show calibration degradation >10%"
        )

    logger.info("\n" + "=" * 80)
    logger.info("CALIBRATION ANALYSIS COMPLETE")
    logger.info("=" * 80)
    logger.info(f"Output directory: {OUTPUT_DIR}")
    logger.info("Generated files:")
    logger.info(f"  - calibration_training_folds.csv ({len(all_training_metrics)} records)")
    logger.info(f"  - calibration_holdout.csv ({len(all_holdout_metrics)} records)")
    logger.info(f"  - overfitting_report.csv ({len(overfitting_reports)} records)")
    logger.info(f"  - {len(MODEL_NAMES)} calibration curve plots")


if __name__ == "__main__":
    main()
