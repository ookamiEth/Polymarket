#!/usr/bin/env python3
"""
Plot Single Contract Timeline - Simplified Version
===================================================

Loads ONE random contract from consolidated data, runs inference, and plots timeline.
"""

import logging
import random
from datetime import date
from pathlib import Path

import lightgbm as lgb
import matplotlib.pyplot as plt
import numpy as np
import polars as pl

from train_multi_horizon_v4 import FEATURE_COLS_V4

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Professional color palette
COLORS = {
    "primary": "#00D4FF",  # Cyan
    "secondary": "#888888",  # Gray
    "success": "#00FF88",  # Green
    "danger": "#FF3366",  # Red
}


def load_models(models_dir: Path) -> dict[str, lgb.Booster]:
    """Load all 12 trained LightGBM models."""
    logger.info(f"Loading models from {models_dir}")
    model_names = [
        "near_low_vol_atm", "near_low_vol_otm", "near_high_vol_atm", "near_high_vol_otm",
        "mid_low_vol_atm", "mid_low_vol_otm", "mid_high_vol_atm", "mid_high_vol_otm",
        "far_low_vol_atm", "far_low_vol_otm", "far_high_vol_atm", "far_high_vol_otm",
    ]

    models = {}
    for model_name in model_names:
        model_path = models_dir / f"lightgbm_{model_name}.txt"
        booster = lgb.Booster(model_file=str(model_path))
        models[model_name] = booster
        logger.info(f"  Loaded {model_name}")

    return models


def main():
    # Paths
    consolidated_file = Path("/home/ubuntu/Polymarket/research/model/data/consolidated_features_v4_pipeline_ready.parquet")
    models_dir = Path("models_optuna")
    output_file = Path("evaluations_optuna/single_contract_timeline_v4.png")
    output_file.parent.mkdir(exist_ok=True)

    logger.info("=== Plotting Single Contract Timeline ===")

    # Load models
    models = load_models(models_dir)

    # Lazy scan to find contracts with many observations
    logger.info(f"Scanning {consolidated_file} for contracts...")
    df_lazy = pl.scan_parquet(consolidated_file)

    # Reconstruct contract_open_time and find contracts with >=100 observations and no nulls
    logger.info("Reconstructing contract identifiers...")
    contract_counts = (
        df_lazy
        .filter(pl.col("date") >= date(2024, 10, 1))  # Holdout period
        .filter(pl.col("prob_mid").is_not_null())  # Only non-null prob_mid
        .with_columns([
            (pl.col("timestamp") - (900 - pl.col("time_remaining"))).alias("contract_open_time")
        ])
        .group_by("contract_open_time")
        .agg(pl.len().alias("n_obs"))
        .filter(pl.col("n_obs") >= 800)  # At least 800 non-null observations (allows ~100 nulls)
        .sort("n_obs", descending=True)
        .collect()
    )

    logger.info(f"Found {len(contract_counts):,} contracts with >=100 observations")

    if len(contract_counts) == 0:
        logger.error("No suitable contracts found!")
        return

    # Random selection
    random.seed(42)
    selected_contract_open_time = contract_counts.sample(1)["contract_open_time"][0]

    # Load this ONE contract's data
    logger.info(f"Loading contract with open_time={selected_contract_open_time}...")
    contract_data = (
        df_lazy
        .with_columns([
            (pl.col("timestamp") - (900 - pl.col("time_remaining"))).alias("contract_open_time")
        ])
        .filter(pl.col("contract_open_time") == selected_contract_open_time)
        .sort("time_remaining", descending=True)
        .collect()
    )

    logger.info(f"  Observations: {len(contract_data):,}")
    logger.info(f"  Time range: {contract_data['time_remaining'].min()}s - {contract_data['time_remaining'].max()}s")

    # Generate predictions
    logger.info("Running model inference...")
    feature_data = contract_data.select(FEATURE_COLS_V4).to_numpy()
    combined_regime = contract_data.select("combined_regime").to_series().to_numpy()
    prob_mid = contract_data.select("prob_mid").to_series().to_numpy()

    residual_predictions = np.zeros(len(contract_data))
    for model_name, model in models.items():
        mask = combined_regime == model_name
        if mask.sum() > 0:
            residual_predictions[mask] = model.predict(feature_data[mask])

    model_pred = np.clip(prob_mid + residual_predictions, 0, 1)

    # Extract data for plotting
    time_remaining = contract_data["time_remaining"].to_numpy()
    time_minutes = (900 - time_remaining) / 60
    outcome = contract_data["outcome"][0]

    # Filter out any remaining NaNs for Brier calculation
    valid_mask = ~(np.isnan(prob_mid) | np.isnan(model_pred))
    brier_baseline = np.mean((prob_mid[valid_mask] - outcome) ** 2)
    brier_model = np.mean((model_pred[valid_mask] - outcome) ** 2)

    # Create plot
    logger.info("Creating plot...")
    fig, ax = plt.subplots(figsize=(14, 8), facecolor="#0e1117")
    ax.set_facecolor("#0e1117")

    # Plot baseline (raw predictions)
    ax.plot(time_minutes, prob_mid,
            color=COLORS["secondary"],
            linewidth=1.5,
            alpha=0.7,
            linestyle='-',
            marker='',
            label="Black-Scholes Baseline")

    # Plot model (raw predictions)
    ax.plot(time_minutes, model_pred,
            color=COLORS["primary"],
            linewidth=2,
            alpha=0.9,
            linestyle='-',
            marker='',
            label="ML Model (Optuna-optimized)")

    # Plot outcome
    outcome_color = COLORS["success"] if outcome == 1 else COLORS["danger"]
    ax.axhline(y=outcome,
               color=outcome_color,
               linestyle="--",
               linewidth=2.5,
               alpha=0.8,
               label=f'Actual: {"UP" if outcome == 1 else "DOWN"}')

    # Styling
    ax.set_xlabel("Time from Contract Open (minutes)", fontsize=14, color="white", fontweight="bold")
    ax.set_ylabel("Probability (UP)", fontsize=14, color="white", fontweight="bold")
    ax.set_title(
        f"Single Contract Timeline\n"
        f"Contract {selected_contract_open_time} | Outcome: {'UP' if outcome == 1 else 'DOWN'} | {len(contract_data)} Observations",
        fontsize=16, color="white", fontweight="bold", pad=20
    )

    ax.set_xlim(0, 15)
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.2, linestyle="--")
    ax.tick_params(colors="white", labelsize=10)

    # Legend
    legend = ax.legend(loc="best", frameon=True, framealpha=0.9, fontsize=12,
                       facecolor="#1a1a1a", edgecolor="white")
    plt.setp(legend.get_texts(), color="white")

    # Annotation
    ann_text = f"""Contract Open Time: {selected_contract_open_time}
Observations: {len(contract_data):,}
Final Baseline: {prob_mid[valid_mask][-1]:.1%}
Final Model: {model_pred[valid_mask][-1]:.1%}
Actual Outcome: {outcome:.0f}
Brier (Baseline): {brier_baseline:.4f}
Brier (Model): {brier_model:.4f}
Improvement: {((brier_baseline - brier_model) / brier_baseline * 100):.1f}%"""

    ax.text(0.02, 0.98, ann_text,
            transform=ax.transAxes,
            verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="#1a1a1a", alpha=0.8, edgecolor="white"),
            fontsize=11, color="white")

    plt.tight_layout()
    fig.savefig(output_file, dpi=300, bbox_inches="tight", facecolor="#0e1117")
    logger.info(f"Plot saved to: {output_file}")
    plt.close()

    logger.info("=== Done ===")


if __name__ == "__main__":
    main()
