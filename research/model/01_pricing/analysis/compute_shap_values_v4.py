#!/usr/bin/env python3
"""
Compute SHAP values for all v4 models on holdout period.

This script:
1. Loads holdout period data (Aug-Nov 2025, ~5.7M samples)
2. Loads all 12 trained LightGBM models
3. Computes SHAP values using TreeExplainer (optimized for LightGBM)
4. Generates SHAP summary statistics (mean absolute SHAP per feature)
5. Creates SHAP visualizations (beeswarm, bar plots)
6. Saves SHAP values for future analysis

Output:
- shap_values_v4/{model_name}_shap_values.npz: Compressed SHAP arrays
- shap_importance_summary.csv: Aggregated SHAP importance
- shap_summary_{model_name}.png: SHAP beeswarm plots
- shap_bar_{model_name}.png: SHAP bar plots
"""

import logging
import pickle
from datetime import date
from pathlib import Path
from typing import Optional

import sys

import lightgbm as lgb
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import shap

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import feature columns from training script
from train_multi_horizon_v4 import FEATURE_COLS_V4

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Constants
PROJECT_ROOT = Path("/home/ubuntu/Polymarket/research/model/01_pricing")
MODELS_DIR = PROJECT_ROOT / "models_optuna"
DATA_PATH = Path("/home/ubuntu/Polymarket/research/model/data/consolidated_features_v4_pipeline_ready.parquet")
OUTPUT_DIR = PROJECT_ROOT / "analysis" / "shap_values_v4"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Model names
TEMPORAL_BUCKETS = ["near", "mid", "far"]
VOL_REGIMES = ["low_vol_atm", "low_vol_otm", "high_vol_atm", "high_vol_otm"]
MODEL_NAMES = [f"{bucket}_{regime}" for bucket in TEMPORAL_BUCKETS for regime in VOL_REGIMES]

# Holdout period (from pipeline evaluation)
HOLDOUT_START = date(2025, 8, 1)
HOLDOUT_END = date(2025, 11, 1)


def load_holdout_data(sample_size: Optional[int] = None) -> tuple[pl.DataFrame, list[str]]:
    """
    Load holdout period data.

    Args:
        sample_size: If specified, sample this many rows. If None, use full holdout.

    Returns:
        Tuple of (DataFrame, list of feature names)
    """
    logger.info(f"Loading holdout data from {DATA_PATH}")

    # Load full dataset (lazy)
    df = pl.scan_parquet(DATA_PATH)

    # Filter to holdout period
    df = df.filter((pl.col("date") >= HOLDOUT_START) & (pl.col("date") < HOLDOUT_END))

    # Get row count
    holdout_count = df.select(pl.len()).collect().item()
    logger.info(f"Holdout period contains {holdout_count:,} samples")

    # Sample if requested
    if sample_size is not None and sample_size < holdout_count:
        logger.info(f"Sampling {sample_size:,} rows from holdout period")
        df = df.collect().sample(n=sample_size, seed=42).lazy()
    else:
        logger.info("Using full holdout period")

    # Collect data
    df = df.collect()

    # Use feature columns from training script
    feature_cols = FEATURE_COLS_V4

    logger.info(f"Loaded {len(df):,} samples with {len(feature_cols)} features")

    return df, feature_cols


def compute_shap_for_model(
    model_name: str,
    x_data: np.ndarray,
    feature_names: list[str],
    max_display: int = 20,  # noqa: N803
) -> tuple[np.ndarray, dict[str, float]]:
    """
    Compute SHAP values for a single model.

    Args:
        model_name: Name of the model
        x_data: Feature matrix (numpy array)
        feature_names: List of feature names
        max_display: Number of features to display in plots

    Returns:
        Tuple of (SHAP values array, importance dictionary)
    """
    logger.info(f"Computing SHAP values for {model_name}")

    # Load model
    model_path = MODELS_DIR / f"lightgbm_{model_name}.txt"
    booster = lgb.Booster(model_file=str(model_path))

    # Create TreeExplainer (optimized for tree models)
    logger.info(f"Creating TreeExplainer for {model_name}")
    explainer = shap.TreeExplainer(booster)

    # Compute SHAP values
    logger.info(f"Computing SHAP values ({x_data.shape[0]:,} samples)...")
    shap_values = explainer.shap_values(x_data)

    # Compute mean absolute SHAP (feature importance)
    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    importance = dict(zip(feature_names, mean_abs_shap))

    # Sort by importance
    importance = dict(sorted(importance.items(), key=lambda x: x[1], reverse=True))

    logger.info(f"Top 5 features for {model_name}:")
    for i, (feat, val) in enumerate(list(importance.items())[:5]):
        logger.info(f"  {i + 1}. {feat}: {val:.6f}")

    # Create summary plot (beeswarm)
    logger.info(f"Creating SHAP summary plot for {model_name}")
    plt.figure(figsize=(12, 8))
    shap.summary_plot(shap_values, x_data, feature_names=feature_names, max_display=max_display, show=False)
    plt.title(f"SHAP Summary (Beeswarm) - {model_name}", fontsize=14, pad=20)
    plt.tight_layout()
    summary_file = OUTPUT_DIR / f"shap_summary_{model_name}.png"
    plt.savefig(summary_file, dpi=300, bbox_inches="tight")
    logger.info(f"Saved summary plot to {summary_file}")
    plt.close()

    # Create bar plot (feature importance)
    logger.info(f"Creating SHAP bar plot for {model_name}")
    plt.figure(figsize=(12, 8))
    shap.summary_plot(
        shap_values,
        x_data,
        feature_names=feature_names,
        max_display=max_display,
        plot_type="bar",
        show=False,
    )
    plt.title(f"SHAP Feature Importance - {model_name}", fontsize=14, pad=20)
    plt.tight_layout()
    bar_file = OUTPUT_DIR / f"shap_bar_{model_name}.png"
    plt.savefig(bar_file, dpi=300, bbox_inches="tight")
    logger.info(f"Saved bar plot to {bar_file}")
    plt.close()

    # Save SHAP values
    shap_file = OUTPUT_DIR / f"{model_name}_shap_values.npz"
    np.savez_compressed(shap_file, shap_values=shap_values, feature_names=feature_names)
    logger.info(f"Saved SHAP values to {shap_file}")

    # Save explainer for later use
    explainer_file = OUTPUT_DIR / f"{model_name}_explainer.pkl"
    with open(explainer_file, "wb") as f:
        pickle.dump(explainer, f)
    logger.info(f"Saved explainer to {explainer_file}")

    return shap_values, importance


def compute_all_shap_values(
    df: pl.DataFrame,
    feature_cols: list[str],
    sample_size: Optional[int] = None,
) -> dict[str, dict[str, float]]:
    """
    Compute SHAP values for all 12 models.

    Args:
        df: Holdout data DataFrame
        feature_cols: List of feature column names
        sample_size: If specified, sample this many rows per model

    Returns:
        Dictionary mapping model names to importance dictionaries
    """
    all_importances = {}

    for model_name in MODEL_NAMES:
        # Filter to regime-specific data (combined_regime matches full model name)
        df_regime = df.filter(pl.col("combined_regime") == model_name)

        n_samples = len(df_regime)
        logger.info(f"\n{'=' * 80}")
        logger.info(f"Processing {model_name} ({n_samples:,} samples in regime)")
        logger.info(f"{'=' * 80}")

        if n_samples == 0:
            logger.warning(f"No samples for {model_name}, skipping")
            continue

        # Sample if requested
        if sample_size is not None and sample_size < n_samples:
            logger.info(f"Sampling {sample_size:,} rows from {n_samples:,}")
            df_regime = df_regime.sample(n=sample_size, seed=42)

        # Extract features
        x_data = df_regime.select(feature_cols).to_numpy()

        # Compute SHAP values
        try:
            _, importance = compute_shap_for_model(model_name, x_data, feature_cols, max_display=25)
            all_importances[model_name] = importance
        except Exception as e:
            logger.error(f"Failed to compute SHAP for {model_name}: {e}")
            continue

    return all_importances


def aggregate_shap_importances(all_importances: dict[str, dict[str, float]]) -> pl.DataFrame:
    """
    Aggregate SHAP importances across all models.

    Args:
        all_importances: Dictionary mapping model names to importance dictionaries

    Returns:
        DataFrame with aggregated statistics
    """
    logger.info("\nAggregating SHAP importances across models...")

    # Convert to list of records
    records = []
    for model_name, importance in all_importances.items():
        for feature, value in importance.items():
            records.append({"model_name": model_name, "feature": feature, "mean_abs_shap": value})

    # Create DataFrame
    df = pl.DataFrame(records)

    # Aggregate statistics
    summary = (
        df.group_by("feature")
        .agg(
            [
                pl.col("mean_abs_shap").mean().alias("mean_shap"),
                pl.col("mean_abs_shap").std().alias("std_shap"),
                pl.col("mean_abs_shap").median().alias("median_shap"),
                pl.col("mean_abs_shap").max().alias("max_shap"),
                pl.col("mean_abs_shap").min().alias("min_shap"),
                pl.len().alias("models_using"),
            ]
        )
        .sort("mean_shap", descending=True)
    )

    return summary


def main() -> None:
    """Main execution function."""
    logger.info("Starting SHAP value computation for V4 models")

    # Step 1: Load holdout data
    logger.info("\n" + "=" * 80)
    logger.info("STEP 1: Loading holdout data")
    logger.info("=" * 80)

    # Use full holdout (comment out to use sample)
    df, feature_cols = load_holdout_data(sample_size=None)

    # For faster testing, uncomment to use 1M sample:
    # df, feature_cols = load_holdout_data(sample_size=1_000_000)

    # Step 2: Compute SHAP values for all models
    logger.info("\n" + "=" * 80)
    logger.info("STEP 2: Computing SHAP values for all 12 models")
    logger.info("=" * 80)

    all_importances = compute_all_shap_values(df, feature_cols, sample_size=None)

    # Step 3: Aggregate and save results
    logger.info("\n" + "=" * 80)
    logger.info("STEP 3: Aggregating and saving results")
    logger.info("=" * 80)

    summary = aggregate_shap_importances(all_importances)

    summary_file = OUTPUT_DIR / "shap_importance_summary.csv"
    summary.write_csv(summary_file)
    logger.info(f"Saved SHAP summary to {summary_file}")

    # Display top and bottom features
    logger.info("\n" + "=" * 80)
    logger.info("TOP 20 FEATURES BY MEAN ABSOLUTE SHAP")
    logger.info("=" * 80)
    print(summary.head(20).select(["feature", "mean_shap", "std_shap", "models_using"]))

    logger.info("\n" + "=" * 80)
    logger.info("BOTTOM 20 FEATURES BY MEAN ABSOLUTE SHAP")
    logger.info("=" * 80)
    print(summary.tail(20).select(["feature", "mean_shap", "std_shap", "models_using"]))

    logger.info("\n" + "=" * 80)
    logger.info("ANALYSIS COMPLETE")
    logger.info("=" * 80)
    logger.info(f"Output directory: {OUTPUT_DIR}")
    logger.info("Generated files:")
    logger.info(f"  - {len(all_importances)} model-specific SHAP values (.npz)")
    logger.info(f"  - {len(all_importances)} explainer objects (.pkl)")
    logger.info(f"  - {len(all_importances) * 2} visualization plots (.png)")
    logger.info("  - 1 aggregated summary CSV")


if __name__ == "__main__":
    main()
