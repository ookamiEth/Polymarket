#!/usr/bin/env python3
"""
Create Publication-Quality Visualizations for V4 Pricing Model
================================================================

Generates professional plots following CLAUDE.md visualization standards:
1. Calibration plot: Predicted vs actual with proper dot sizing by sample count
2. Timeline plot: Baseline vs model predictions evolution over time_remaining

Uses actual trained LightGBM models to generate predictions.

Author: BT Research Team
Date: 2025-11-17
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import lightgbm as lgb
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from sklearn.calibration import calibration_curve

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

# Feature columns from train_multi_horizon_v4.py (lines 83-240)
FEATURE_COLS_V4 = [
    "K",
    "S",
    "T_years",
    "ask_volume_ratio_1to5",
    "autocorr_decay",
    "autocorr_lag5_300s",
    "bid_ask_imbalance",
    "bid_ask_spread_bps",
    "bid_volume_ratio_1to5",
    "depth_imbalance_5",
    "depth_imbalance_ema_300s",
    "depth_imbalance_ema_3600s",
    "depth_imbalance_ema_60s",
    "depth_imbalance_ema_900s",
    "depth_imbalance_vol_300s",
    "depth_imbalance_vol_3600s",
    "depth_imbalance_vol_60s",
    "depth_imbalance_vol_900s",
    "depth_imbalance_vol_normalized",
    "downside_vol_300",
    "downside_vol_300s",
    "downside_vol_60",
    "downside_vol_900",
    "ema_900s",
    "funding_rate",
    "funding_rate_ema_300s",
    "funding_rate_ema_3600s",
    "funding_rate_ema_60s",
    "funding_rate_ema_900s",
    "garch_forecast_simple",
    "high_15m",
    "hour_cos",
    "hour_of_day_utc",
    "hour_sin",
    "hurst_300s",
    "imbalance_ema_300s",
    "imbalance_ema_3600s",
    "imbalance_ema_60s",
    "imbalance_ema_900s",
    "imbalance_vol_300s",
    "imbalance_vol_3600s",
    "imbalance_vol_60s",
    "imbalance_vol_900s",
    "is_extreme_condition",
    "iv_minus_downside_vol",
    "iv_minus_upside_vol",
    "iv_staleness_seconds",
    "kurtosis_300s",
    "log_moneyness",
    "low_15m",
    "mark_index_basis_bps",
    "mark_index_ema_300s",
    "mark_index_ema_3600s",
    "mark_index_ema_60s",
    "mark_index_ema_900s",
    "momentum_300s",
    "momentum_300s_ema_300s",
    "momentum_300s_ema_3600s",
    "momentum_300s_ema_60s",
    "momentum_300s_ema_900s",
    "momentum_900s",
    "momentum_900s_ema_300s",
    "momentum_900s_ema_3600s",
    "momentum_900s_ema_60s",
    "momentum_900s_ema_900s",
    "moneyness",
    "moneyness_cubed",
    "moneyness_distance",
    "moneyness_percentile",
    "moneyness_squared",
    "moneyness_x_time",
    "moneyness_x_vol",
    "oi_ema_3600s",
    "oi_ema_900s",
    "open_interest",
    "position_scale",
    "price_vs_ema_900",
    "range_300s",
    "range_300s_ema_300s",
    "range_300s_ema_3600s",
    "range_300s_ema_60s",
    "range_300s_ema_900s",
    "range_900s",
    "range_900s_ema_300s",
    "range_900s_ema_3600s",
    "range_900s_ema_60s",
    "range_900s_ema_900s",
    "realized_kurtosis_300",
    "realized_kurtosis_60",
    "realized_kurtosis_900",
    "realized_skewness_300",
    "realized_skewness_60",
    "realized_skewness_900",
    "returns_300s",
    "returns_60s",
    "returns_900s",
    "reversals_300s",
    "rv_300s",
    "rv_300s_ema_300s",
    "rv_300s_ema_3600s",
    "rv_300s_ema_60s",
    "rv_300s_ema_900s",
    "rv_60s",
    "rv_900s",
    "rv_900s_ema_300s",
    "rv_900s_ema_3600s",
    "rv_900s_ema_60s",
    "rv_900s_ema_900s",
    "rv_95th_percentile",
    "rv_ratio",
    "rv_ratio_15m_5m",
    "rv_ratio_1h_15m",
    "rv_ratio_5m_1m",
    "rv_term_structure",
    "sigma_mid",
    "skewness_300s",
    "spread_ema_300s",
    "spread_ema_3600s",
    "spread_ema_60s",
    "spread_ema_900s",
    "spread_vol_300s",
    "spread_vol_3600s",
    "spread_vol_60s",
    "spread_vol_900s",
    "spread_vol_normalized",
    "standardized_moneyness",
    "tail_risk_300s",
    "time_remaining",
    "time_since_high_15m",
    "time_since_low_15m",
    "timestamp_seconds",
    "total_ask_volume_5",
    "total_bid_volume_5",
    "upside_vol_300",
    "upside_vol_300s",
    "upside_vol_60",
    "upside_vol_900",
    "vol_acceleration_300s",
    "vol_asymmetry_300s",
    "vol_asymmetry_ratio_300",
    "vol_asymmetry_ratio_60",
    "vol_asymmetry_ratio_900",
    "vol_high_thresh",
    "vol_low_thresh",
    "vol_of_vol_300",
    "vol_of_vol_300s",
    "vol_persistence_ar1",
    "weighted_mid_ema_300s",
    "weighted_mid_ema_3600s",
    "weighted_mid_ema_60s",
    "weighted_mid_ema_900s",
    "weighted_mid_price_5",
    "weighted_mid_velocity_normalized",
]

# Professional color palette (from CLAUDE.md)
COLORS = {
    "primary": "#00D4FF",  # Cyan
    "secondary": "#FF00FF",  # Magenta
    "success": "#00FF88",  # Green
    "danger": "#FF3366",  # Red
    "warning": "#FFB000",  # Orange
    "info": "#00B4FF",  # Light Blue
    "perfect": "#888888",  # Gray for reference lines
    "grid": "#333333",  # Dark gray for grid
}


def load_models(models_dir: Path) -> dict[str, lgb.Booster]:
    """Load all 12 trained LightGBM models."""
    logger.info(f"Loading models from {models_dir}")
    models = {}

    # All 12 model names (3 temporal × 4 regimes)
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

    for model_name in model_names:
        model_path = models_dir / f"lightgbm_{model_name}.txt"
        if not model_path.exists():
            logger.warning(f"Model not found: {model_path}")
            continue

        try:
            booster = lgb.Booster(model_file=str(model_path))
            models[model_name] = booster
            logger.info(f"  Loaded {model_name}")
        except Exception as e:
            logger.error(f"  Failed to load {model_name}: {e}")

    logger.info(f"Successfully loaded {len(models)}/12 models")
    return models


def generate_predictions(
    data: pl.DataFrame, models: dict[str, lgb.Booster]
) -> pl.DataFrame:
    """Generate model predictions using routing logic."""
    logger.info("Generating predictions for all samples...")

    # Prepare feature matrix
    feature_data = data.select(FEATURE_COLS_V4).to_numpy()

    # Get regime assignments
    combined_regime = data.select("combined_regime").to_series().to_numpy()

    # Initialize prediction array
    residual_predictions = np.zeros(len(data))

    # Route to correct model and predict
    for model_name, model in models.items():
        mask = combined_regime == model_name
        n_samples = mask.sum()

        if n_samples == 0:
            continue

        logger.info(f"  Predicting {n_samples:,} samples for {model_name}")
        residual_predictions[mask] = model.predict(feature_data[mask])

    # Convert to final probabilities: prob_mid + residual_pred
    prob_mid = data.select("prob_mid").to_series().to_numpy()
    final_predictions = np.clip(prob_mid + residual_predictions, 0, 1)

    # Add predictions to dataframe
    result = data.with_columns([pl.Series("model_pred", final_predictions)])

    logger.info("Prediction complete")
    return result


def create_calibration_plot(
    predicted: np.ndarray,
    actual: np.ndarray,
    baseline_predicted: np.ndarray,
    output_file: Path,
) -> None:
    """Create publication-quality calibration plot with proper dot sizing."""
    logger.info("Creating calibration plot...")

    # Compute calibration curves (pos_label=1.0 for float outcomes)
    prob_true_model, prob_pred_model = calibration_curve(
        actual, predicted, n_bins=10, strategy="uniform", pos_label=1.0
    )
    prob_true_baseline, prob_pred_baseline = calibration_curve(
        actual, baseline_predicted, n_bins=10, strategy="uniform", pos_label=1.0
    )

    # Calculate sample counts per bin for dot sizing
    bins = np.linspace(0, 1, 11)
    counts_model = np.histogram(predicted, bins=bins)[0]
    counts_baseline = np.histogram(baseline_predicted, bins=bins)[0]

    # Dot sizing: np.clip(counts / 1000, 30, 200) per CLAUDE.md
    sizes_model = np.clip(counts_model / 1000, 30, 200)
    sizes_baseline = np.clip(counts_baseline / 1000, 30, 200)

    # Calculate Brier scores
    brier_model = np.mean((predicted - actual) ** 2)
    brier_baseline = np.mean((baseline_predicted - actual) ** 2)
    improvement_pct = (1 - brier_model / brier_baseline) * 100

    # Create figure with dark theme
    fig, ax = plt.subplots(figsize=(10, 8), facecolor="#0e1117")
    ax.set_facecolor("#0e1117")

    # Plot calibration curves
    ax.scatter(
        prob_pred_baseline,
        prob_true_baseline,
        s=sizes_baseline,
        alpha=0.7,
        c=COLORS["perfect"],
        edgecolor="white",
        linewidth=1,
        label="Baseline (Black-Scholes)",
        zorder=5,
    )

    ax.scatter(
        prob_pred_model,
        prob_true_model,
        s=sizes_model,
        alpha=0.7,
        c=COLORS["primary"],
        edgecolor="white",
        linewidth=1,
        label="ML Model (Optuna)",
        zorder=10,
    )

    # 45° perfect calibration line
    ax.plot(
        [0, 1],
        [0, 1],
        "--",
        color=COLORS["perfect"],
        alpha=0.5,
        linewidth=1.5,
        label="Perfect Calibration",
        zorder=1,
    )

    # Styling
    ax.set_xlabel("Predicted Probability", fontsize=12, color="white", fontweight="bold")
    ax.set_ylabel("Observed Frequency", fontsize=12, color="white", fontweight="bold")
    ax.set_title(
        f"Calibration Plot - V4 Multi-Horizon Model\n"
        f"Brier Improvement: {improvement_pct:.1f}% "
        f"(Baseline: {brier_baseline:.4f} → Model: {brier_model:.4f})",
        fontsize=14,
        color="white",
        fontweight="bold",
        pad=20,
    )
    ax.grid(True, alpha=0.2, color=COLORS["grid"])
    ax.tick_params(colors="white", labelsize=10)

    # Set equal aspect ratio and limits
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    # Legend
    legend = ax.legend(
        frameon=True,
        facecolor="#1a1a1a",
        edgecolor="white",
        fontsize=10,
        loc="upper left",
    )
    plt.setp(legend.get_texts(), color="white")

    # Annotation box with statistics
    stats_text = f"""Total Samples: {len(predicted):,}
Model Brier: {brier_model:.4f}
Baseline Brier: {brier_baseline:.4f}
Improvement: {improvement_pct:.1f}%"""

    ax.text(
        0.98,
        0.02,
        stats_text,
        transform=ax.transAxes,
        fontsize=10,
        verticalalignment="bottom",
        horizontalalignment="right",
        bbox=dict(boxstyle="round", facecolor="#1a1a1a", alpha=0.8, edgecolor="white"),
        color="white",
    )

    plt.tight_layout()
    fig.savefig(output_file, dpi=300, bbox_inches="tight", facecolor="#0e1117")
    logger.info(f"Calibration plot saved to: {output_file}")
    plt.close()


def create_timeline_plot(data: pl.DataFrame, output_file: Path) -> None:
    """Create timeline plot for a SINGLE random contract."""
    logger.info("Creating single contract timeline plot...")

    # Filter out NaN values from prob_mid and model_pred FIRST
    data_clean = data.filter(
        pl.col("prob_mid").is_not_null() & pl.col("model_pred").is_not_null()
    )
    logger.info(f"Filtered {len(data):,} → {len(data_clean):,} samples (removed NaN prob_mid/model_pred)")

    # Find contracts with many observations (complete timelines)
    timestamp_counts = (
        data_clean
        .group_by("timestamp_seconds")
        .agg(pl.len().alias("n_obs"))
        .filter(pl.col("n_obs") >= 100)  # At least 100 observations
        .sort("n_obs", descending=True)
    )

    logger.info(f"Found {len(timestamp_counts):,} contracts with >=100 observations")

    if len(timestamp_counts) == 0:
        logger.error("No contracts with sufficient observations found!")
        return

    # Randomly select ONE contract
    import random
    random.seed(42)  # For reproducibility
    selected_timestamp = timestamp_counts.sample(1)["timestamp_seconds"][0]

    # Get all data for this specific contract
    contract_data = (
        data_clean
        .filter(pl.col("timestamp_seconds") == selected_timestamp)
        .sort("time_remaining", descending=True)  # Sort by time (newest first)
    )

    logger.info(f"Selected contract timestamp: {selected_timestamp}")
    logger.info(f"  Contract observations: {len(contract_data):,}")
    logger.info(f"  Time range: {contract_data['time_remaining'].min()}s - {contract_data['time_remaining'].max()}s")
    logger.info(f"  Outcome: {'UP' if contract_data['outcome'][0] == 1 else 'DOWN'}")

    # Convert to arrays for plotting (actual second-by-second data for this contract)
    time_remaining = contract_data["time_remaining"].to_numpy()
    time_minutes = (900 - time_remaining) / 60
    prob_mid = contract_data["prob_mid"].to_numpy()
    model_pred = contract_data["model_pred"].to_numpy()
    outcome = contract_data["outcome"][0]

    # Create figure with dark theme
    fig, ax = plt.subplots(figsize=(14, 8), facecolor="#0e1117")
    ax.set_facecolor("#0e1117")

    # Plot baseline (Black-Scholes) - MUCH THICKER
    ax.plot(
        time_minutes,
        prob_mid,
        label="Black-Scholes Baseline",
        color=COLORS["perfect"],
        linewidth=5,  # INCREASED from 2
        alpha=1.0,    # INCREASED from 0.9
        marker="o",
        markersize=14,  # INCREASED from 8
        markeredgewidth=2,
        markeredgecolor='white',
    )

    # Plot model prediction - EVEN THICKER
    ax.plot(
        time_minutes,
        model_pred,
        label="ML Model (Optuna-optimized)",
        color=COLORS["primary"],
        linewidth=6,  # INCREASED from 2.5
        alpha=1.0,    # INCREASED from 0.95
        marker="s",
        markersize=16,  # INCREASED from 8
        markeredgewidth=2,
        markeredgecolor='white',
    )

    # Plot actual outcome as horizontal line
    outcome_color = COLORS["success"] if outcome == 1 else COLORS["danger"]
    outcome_label = f'Actual Outcome: {"UP (1)" if outcome == 1 else "DOWN (0)"}'
    ax.axhline(
        y=outcome,
        color=outcome_color,
        linestyle="--",
        linewidth=4,
        alpha=0.9,
        label=outcome_label,
    )

    # Styling
    ax.set_xlabel(
        "Time from Contract Open (minutes)", fontsize=14, color="white", fontweight="bold"
    )
    ax.set_ylabel("Probability (UP)", fontsize=14, color="white", fontweight="bold")
    ax.set_title(
        f"Single Contract Timeline: Baseline vs Model Predictions\n"
        f"Contract {selected_timestamp} | Outcome: {'UP' if outcome == 1 else 'DOWN'} | {len(contract_data)} Observations",
        fontsize=16,
        color="white",
        fontweight="bold",
        pad=20,
    )

    # Set axis limits
    ax.set_xlim(0, 15)
    ax.set_ylim(0, 1)

    # Add grid
    ax.grid(True, alpha=0.2, linestyle="--", color=COLORS["grid"])

    # Add legend
    legend = ax.legend(
        loc="best", frameon=True, framealpha=0.9, fontsize=12, facecolor="#1a1a1a", edgecolor="white"
    )
    plt.setp(legend.get_texts(), color="white")

    # Tick styling
    ax.tick_params(colors="white", labelsize=10)

    # Calculate Brier scores for this contract
    brier_baseline = np.mean((prob_mid - outcome) ** 2)
    brier_model = np.mean((model_pred - outcome) ** 2)

    # Annotation box
    annotation_text = f"""Contract: {selected_timestamp}
Observations: {len(contract_data):,}
Final Baseline: {prob_mid[-1]:.1%}
Final Model: {model_pred[-1]:.1%}
Actual Outcome: {outcome:.0f}
Brier (Baseline): {brier_baseline:.4f}
Brier (Model): {brier_model:.4f}"""

    ax.text(
        0.02,
        0.98,
        annotation_text,
        transform=ax.transAxes,
        verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="#1a1a1a", alpha=0.8, edgecolor="white"),
        fontsize=11,
        color="white",
    )

    plt.tight_layout()
    fig.savefig(output_file, dpi=300, bbox_inches="tight", facecolor="#0e1117")
    logger.info(f"Timeline plot saved to: {output_file}")
    plt.close()


def main():
    """Main execution."""
    logger.info("=== Creating Publication-Quality Plots for V4 Model ===")

    # Paths
    models_dir = Path("models_optuna")
    stratified_dir = Path("stratified_v4")
    output_dir = Path("evaluations_optuna")

    output_dir.mkdir(exist_ok=True)

    # Load models
    models = load_models(models_dir)
    if len(models) == 0:
        logger.error("No models loaded - cannot generate predictions")
        return

    # Load stratified data (use holdout only for evaluation)
    logger.info("Loading stratified data...")
    all_data = []
    for file in stratified_dir.glob("*_data.parquet"):
        df = pl.read_parquet(file)
        # Filter to holdout period (date >= 2024-10-01)
        from datetime import date

        holdout = df.filter(pl.col("date") >= date(2024, 10, 1))
        all_data.append(holdout)
        logger.info(f"  Loaded {len(holdout):,} holdout samples from {file.name}")

    data = pl.concat(all_data)
    logger.info(f"Total holdout samples: {len(data):,}")

    # Generate predictions
    data = generate_predictions(data, models)

    # Filter out NaN outcomes before plotting
    data_valid = data.filter(pl.col("outcome").is_not_null())
    logger.info(f"Filtered to {len(data_valid):,} samples with valid outcomes")

    # Extract arrays for plotting
    predicted = data_valid.select("model_pred").to_series().to_numpy()
    baseline_predicted = data_valid.select("prob_mid").to_series().to_numpy()
    actual = data_valid.select("outcome").to_series().to_numpy()

    # Create calibration plot
    calibration_output = output_dir / "calibration_plot_v4_publication.png"
    create_calibration_plot(predicted, actual, baseline_predicted, calibration_output)

    # Create timeline plot (use valid data only)
    timeline_output = output_dir / "timeline_plot_v4_publication.png"
    create_timeline_plot(data_valid, timeline_output)

    logger.info("=== Publication plots complete ===")
    logger.info(f"  Calibration: {calibration_output}")
    logger.info(f"  Timeline: {timeline_output}")


if __name__ == "__main__":
    main()
