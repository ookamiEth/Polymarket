#!/usr/bin/env python3
"""
Plot Single Contract Timeline: Baseline vs Model Predictions
=============================================================

Creates a publication-quality timeline plot showing how Black-Scholes baseline
and ML model predictions evolve over a single 15-minute contract's lifetime.

Data source: consolidated_features_v4_pipeline_ready.parquet (holdout period)
"""

import logging
import sys
from datetime import date
from pathlib import Path

import lightgbm as lgb
import matplotlib.pyplot as plt
import numpy as np
import polars as pl

# Import feature columns from training script
from train_multi_horizon_v4 import FEATURE_COLS_V4

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Professional color palette (dark theme)
COLORS = {
    "primary": "#00D4FF",  # Cyan
    "secondary": "#888888",  # Gray
    "success": "#00FF88",  # Green
    "danger": "#FF3366",  # Red
}


def load_models(models_dir: Path) -> dict[str, lgb.Booster]:
    """Load all 12 trained LightGBM models."""
    model_names = [
        "near_low_vol_atm",
        "near_low_vol_otm",
        "near_high_vol_atm",
        "near_high_vol_otm",
        "mid_low_vol_atm",
        "mid_low_vol_otm",
        "mid_high_vol_atm",
        "mid_high_vol_otm",
        "far_low_vol_atm",
        "far_low_vol_otm",
        "far_high_vol_atm",
        "far_high_vol_otm",
    ]

    models = {}
    for model_name in model_names:
        model_path = models_dir / f"lightgbm_{model_name}.txt"
        if not model_path.exists():
            raise FileNotFoundError(f"Model not found: {model_path}")
        booster = lgb.Booster(model_file=str(model_path))
        models[model_name] = booster
        logger.info(f"  Loaded {model_name}")

    return models




def generate_model_predictions(data: pl.DataFrame, models: dict[str, lgb.Booster]) -> np.ndarray:
    """Run model inference to generate predictions."""
    logger.info("Running model inference...")

    # Extract features and routing info
    feature_data = data.select(FEATURE_COLS_V4).to_numpy()
    combined_regime = data.select("combined_regime").to_series().to_numpy()
    prob_mid = data.select("prob_mid").to_series().to_numpy()

    # Initialize predictions
    residual_predictions = np.zeros(len(data))

    # Route to appropriate model
    for model_name, model in models.items():
        mask = combined_regime == model_name
        n_samples = mask.sum()
        if n_samples > 0:
            residual_predictions[mask] = model.predict(feature_data[mask])
            logger.info(f"  Predicted {n_samples} samples for {model_name}")

    # Combine: model_pred = prob_mid + residual_pred
    model_predictions = np.clip(prob_mid + residual_predictions, 0, 1)

    return model_predictions


def create_timeline_plot(data: pl.DataFrame, model_pred: np.ndarray, output_file: Path) -> None:
    """Create publication-quality timeline plot."""
    logger.info("Creating timeline plot...")

    # Extract data
    time_remaining = data["time_remaining"].to_numpy()
    baseline_prob = data["prob_mid"].to_numpy()
    outcome = data["outcome"][0]
    timestamp = data["timestamp_seconds"][0]

    # Convert to elapsed minutes from contract start
    elapsed_minutes = (900 - time_remaining) / 60

    # Calculate Brier scores
    brier_baseline = np.mean((baseline_prob - outcome) ** 2)
    brier_model = np.mean((model_pred - outcome) ** 2)
    improvement_pct = (1 - brier_model / brier_baseline) * 100

    # Create figure with dark theme
    plt.style.use("dark_background")
    fig, ax = plt.subplots(figsize=(14, 8), facecolor="#0e1117")
    ax.set_facecolor("#0e1117")

    # Plot baseline (Black-Scholes)
    ax.plot(
        elapsed_minutes,
        baseline_prob,
        color=COLORS["secondary"],
        linewidth=2,
        alpha=0.9,
        label="Black-Scholes Baseline",
    )

    # Plot model predictions
    ax.plot(
        elapsed_minutes,
        model_pred,
        color=COLORS["primary"],
        linewidth=2.5,
        alpha=0.95,
        label="ML Model (Optuna-optimized)",
    )

    # Plot actual outcome
    outcome_color = COLORS["success"] if outcome == 1 else COLORS["danger"]
    outcome_label = "Actual: UP" if outcome == 1 else "Actual: DOWN"
    ax.axhline(y=outcome, color=outcome_color, linestyle="--", linewidth=2, alpha=0.7, label=outcome_label)

    # Formatting
    ax.set_xlabel("Time from Contract Open (minutes)", fontsize=14, fontweight="bold")
    ax.set_ylabel("Probability (UP)", fontsize=14, fontweight="bold")
    ax.set_title(
        f"Baseline vs Model Predictions Over Contract Lifetime\n" f"Contract: {timestamp}, Outcome={'UP' if outcome == 1 else 'DOWN'}, Observations={len(data):,}",
        fontsize=16,
        fontweight="bold",
        pad=20,
    )

    # Set limits
    ax.set_xlim(0, 15)
    ax.set_ylim(0, 1)

    # Grid
    ax.grid(True, alpha=0.2, linestyle="--")

    # Legend
    ax.legend(loc="best", frameon=True, framealpha=0.9, fontsize=12)

    # Add annotations
    ax.text(
        0.02,
        0.98,
        f"Final Baseline: {baseline_prob[-1]:.1%}\n"
        f"Final Model: {model_pred[-1]:.1%}\n"
        f"Outcome: {outcome_label}\n"
        f"Brier Baseline: {brier_baseline:.4f}\n"
        f"Brier Model: {brier_model:.4f}\n"
        f"Improvement: {improvement_pct:.1f}%",
        transform=ax.transAxes,
        verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="black", alpha=0.7),
        fontsize=11,
        color="white",
    )

    # Tight layout
    plt.tight_layout()

    # Save at 300 DPI
    plt.savefig(output_file, dpi=300, bbox_inches="tight", facecolor="#0e1117")
    logger.info(f"Timeline plot saved to: {output_file}")

    plt.close()


def main():
    """Main execution."""
    # Paths - use consolidated parquet which has full timelines
    consolidated_file = Path("/home/ubuntu/Polymarket/research/model/data/consolidated_features_v4_pipeline_ready.parquet")
    models_dir = Path("models_optuna")
    output_dir = Path("evaluations_optuna")
    output_dir.mkdir(exist_ok=True)

    logger.info("=== Creating Single Contract Timeline Plot ===")

    # Load models
    logger.info(f"Loading models from {models_dir}")
    models = load_models(models_dir)
    logger.info(f"Successfully loaded {len(models)}/12 models")

    # Load consolidated data (lazy scan for memory efficiency)
    logger.info(f"Loading consolidated data from {consolidated_file}...")

    # First, find timestamps with many observations (complete timelines)
    df_lazy = pl.scan_parquet(consolidated_file)

    # Get timestamps with many observations (complete timelines)
    timestamp_counts = (
        df_lazy
        .group_by("timestamp_seconds")
        .agg(pl.len().alias("n_obs"))
        .filter(pl.col("n_obs") >= 10)  # At least 10 observations
        .sort("n_obs", descending=True)
        .collect()
    )

    logger.info(f"Found {len(timestamp_counts):,} timestamps with >=10 observations")
    if len(timestamp_counts) > 0:
        logger.info(f"  Max observations: {timestamp_counts['n_obs'].max()}")
        logger.info(f"  Mean observations: {timestamp_counts['n_obs'].mean():.1f}")

    # Randomly select one timestamp
    import random
    selected_timestamp = timestamp_counts.sample(1)["timestamp_seconds"][0]
    logger.info(f"Selected timestamp: {selected_timestamp}")

    # Load full timeline for this contract
    contract_data = (
        df_lazy
        .filter(pl.col("timestamp_seconds") == selected_timestamp)
        .sort("time_remaining", descending=True)
        .collect()
    )

    logger.info(f"  Observations: {len(contract_data)}")
    logger.info(f"  Time range: {contract_data['time_remaining'].min()}s - {contract_data['time_remaining'].max()}s")
    logger.info(f"  Outcome: {'UP' if contract_data['outcome'][0] == 1 else 'DOWN'}")

    # Generate model predictions
    model_pred = generate_model_predictions(contract_data, models)

    # Create timeline plot
    output_file = output_dir / "timeline_single_contract_v4.png"
    create_timeline_plot(contract_data, model_pred, output_file)

    logger.info("=== Timeline plot complete ===")
    logger.info(f"  Output: {output_file}")


if __name__ == "__main__":
    main()
