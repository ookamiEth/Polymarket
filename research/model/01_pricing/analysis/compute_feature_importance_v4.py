#!/usr/bin/env python3
"""
Extract LightGBM feature importance from all v4 models.

This script:
1. Loads all 12 trained LightGBM models from models_optuna/
2. Extracts feature importance using 'gain' and 'split' methods
3. Generates per-model and aggregated importance rankings
4. Creates heatmap visualization of importance across regimes
5. Identifies top/bottom features globally

Output:
- feature_importance_lgb_v4.csv: Per-model importance rankings
- feature_importance_summary.csv: Aggregated stats across models
- feature_importance_heatmap.png: Visual heatmap
"""

import logging
from pathlib import Path

import lightgbm as lgb
import matplotlib.pyplot as plt
import polars as pl
import seaborn as sns

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Constants
PROJECT_ROOT = Path("/home/ubuntu/Polymarket/research/model/01_pricing")
MODELS_DIR = PROJECT_ROOT / "models_optuna"
OUTPUT_DIR = PROJECT_ROOT / "analysis" / "feature_importance"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Model names (3 temporal buckets × 4 volatility regimes)
TEMPORAL_BUCKETS = ["near", "mid", "far"]
VOL_REGIMES = ["low_vol_atm", "low_vol_otm", "high_vol_atm", "high_vol_otm"]
MODEL_NAMES = [f"{bucket}_{regime}" for bucket in TEMPORAL_BUCKETS for regime in VOL_REGIMES]


def load_model(model_name: str) -> lgb.Booster:
    """Load a trained LightGBM model."""
    model_path = MODELS_DIR / f"lightgbm_{model_name}.txt"
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")

    logger.info(f"Loading model: {model_name}")
    return lgb.Booster(model_file=str(model_path))


def extract_feature_importance(booster: lgb.Booster, importance_type: str = "gain") -> dict[str, float]:
    """
    Extract feature importance from LightGBM booster.

    Args:
        booster: Trained LightGBM booster
        importance_type: 'gain' (total gain) or 'split' (number of splits)

    Returns:
        Dictionary mapping feature names to importance scores
    """
    importance = booster.feature_importance(importance_type=importance_type)
    feature_names = booster.feature_name()

    return dict(zip(feature_names, importance))


def compute_all_importances() -> pl.DataFrame:
    """
    Compute feature importance for all 12 models.

    Returns:
        DataFrame with columns: feature, model_name, importance_gain, importance_split
    """
    all_importances = []

    for model_name in MODEL_NAMES:
        try:
            booster = load_model(model_name)

            # Extract both gain and split importance
            importance_gain = extract_feature_importance(booster, importance_type="gain")
            importance_split = extract_feature_importance(booster, importance_type="split")

            # Combine into records
            for feature in importance_gain:
                all_importances.append(
                    {
                        "feature": feature,
                        "model_name": model_name,
                        "importance_gain": importance_gain[feature],
                        "importance_split": importance_split[feature],
                    }
                )

        except Exception as e:
            logger.error(f"Failed to process model {model_name}: {e}")
            continue

    # Convert to Polars DataFrame
    df = pl.DataFrame(all_importances)

    # Add temporal bucket and regime columns for analysis
    df = df.with_columns(
        [
            pl.col("model_name").str.split("_").list.get(0).alias("temporal_bucket"),
            pl.col("model_name")
            .str.extract(r"_(low_vol_atm|low_vol_otm|high_vol_atm|high_vol_otm)$", 1)
            .alias("vol_regime"),
        ]
    )

    return df


def compute_summary_statistics(df: pl.DataFrame) -> pl.DataFrame:
    """
    Compute aggregated feature importance statistics across all models.

    Returns:
        DataFrame with: feature, mean_gain, std_gain, median_gain, max_gain, min_gain,
                        mean_split, models_using (count)
    """
    summary = (
        df.group_by("feature")
        .agg(
            [
                pl.col("importance_gain").mean().alias("mean_gain"),
                pl.col("importance_gain").std().alias("std_gain"),
                pl.col("importance_gain").median().alias("median_gain"),
                pl.col("importance_gain").max().alias("max_gain"),
                pl.col("importance_gain").min().alias("min_gain"),
                pl.col("importance_split").mean().alias("mean_split"),
                pl.len().alias("models_using"),
            ]
        )
        .sort("mean_gain", descending=True)
    )

    return summary


def create_importance_heatmap(df: pl.DataFrame, output_path: Path, top_n: int = 50) -> None:
    """
    Create heatmap showing top N features across all 12 models.

    Args:
        df: Feature importance DataFrame
        output_path: Path to save the heatmap
        top_n: Number of top features to display
    """
    # Get top N features by mean gain
    summary = compute_summary_statistics(df)
    top_features = summary.head(top_n)["feature"].to_list()

    # Pivot to wide format for heatmap
    heatmap_data = (
        df.filter(pl.col("feature").is_in(top_features))
        .select(["feature", "model_name", "importance_gain"])
        .pivot(on="model_name", index="feature", values="importance_gain")
    )

    # Convert to numpy for seaborn (preserve feature names as index)
    feature_names = heatmap_data["feature"].to_list()
    model_columns = [col for col in heatmap_data.columns if col != "feature"]
    heatmap_array = heatmap_data.select(model_columns).to_numpy()

    # Create figure
    _, ax = plt.subplots(figsize=(16, max(12, top_n * 0.3)))

    # Plot heatmap
    sns.heatmap(
        heatmap_array,
        xticklabels=model_columns,
        yticklabels=feature_names,
        cmap="YlOrRd",
        fmt=".0f",
        linewidths=0.5,
        cbar_kws={"label": "Feature Importance (Gain)"},
        ax=ax,
    )

    ax.set_title(f"Top {top_n} Features by Mean Gain Across 12 V4 Models", fontsize=16, pad=20)
    ax.set_xlabel("Model", fontsize=12)
    ax.set_ylabel("Feature", fontsize=12)

    # Rotate x-axis labels for readability
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    logger.info(f"Saved heatmap to {output_path}")
    plt.close()


def identify_regime_specific_features(df: pl.DataFrame, threshold: float = 0.8) -> pl.DataFrame:
    """
    Identify features that are highly important in specific regimes only.

    Args:
        df: Feature importance DataFrame
        threshold: Gini coefficient threshold (0.8 = concentrated in few models)

    Returns:
        DataFrame with regime-specific features and their concentration metrics
    """

    # Compute Gini coefficient for each feature's importance distribution
    def gini_coefficient(values: list[float]) -> float:
        """Compute Gini coefficient (0=uniform, 1=concentrated)."""
        sorted_values = sorted(values)
        n = len(sorted_values)
        total = sum(sorted_values)
        if total == 0:
            return 0.0  # All zeros, no concentration
        return (2 * sum((i + 1) * val for i, val in enumerate(sorted_values))) / (n * total) - (n + 1) / n

    # Group by feature and compute concentration metrics
    regime_analysis = (
        df.group_by("feature")
        .agg(
            [
                pl.col("importance_gain").alias("gains"),
                pl.col("model_name").alias("models"),
            ]
        )
        .with_columns(
            [
                pl.col("gains")
                .map_elements(lambda x: gini_coefficient(x.to_list()), return_dtype=pl.Float64)
                .alias("gini_coefficient"),
                pl.col("gains").map_elements(lambda x: x.max(), return_dtype=pl.Float64).alias("max_gain"),
                pl.col("gains").map_elements(lambda x: x.mean(), return_dtype=pl.Float64).alias("mean_gain"),
            ]
        )
        .filter(pl.col("gini_coefficient") >= threshold)
        .sort("gini_coefficient", descending=True)
    )

    # Find which model has max importance for each regime-specific feature
    max_model_map = (
        df.sort("importance_gain", descending=True)
        .group_by("feature")
        .agg(pl.col("model_name").first().alias("top_model"))
    )

    regime_analysis = regime_analysis.join(max_model_map, on="feature", how="left").select(
        ["feature", "gini_coefficient", "max_gain", "mean_gain", "top_model"]
    )

    return regime_analysis


def main() -> None:
    """Main execution function."""
    logger.info("Starting LightGBM feature importance extraction for V4 models")

    # Step 1: Compute feature importance for all models
    logger.info("Computing feature importance for all 12 models...")
    df_importance = compute_all_importances()

    # Save detailed per-model importance
    output_file = OUTPUT_DIR / "feature_importance_lgb_v4.csv"
    df_importance.write_csv(output_file)
    logger.info(f"Saved per-model importance to {output_file}")
    logger.info(f"Total records: {len(df_importance):,}")

    # Step 2: Compute summary statistics
    logger.info("Computing summary statistics...")
    df_summary = compute_summary_statistics(df_importance)

    summary_file = OUTPUT_DIR / "feature_importance_summary.csv"
    df_summary.write_csv(summary_file)
    logger.info(f"Saved summary statistics to {summary_file}")

    # Step 3: Identify top and bottom features
    logger.info("\n" + "=" * 80)
    logger.info("TOP 20 FEATURES BY MEAN GAIN")
    logger.info("=" * 80)
    top_20 = df_summary.head(20)
    print(top_20.select(["feature", "mean_gain", "std_gain", "models_using"]))

    logger.info("\n" + "=" * 80)
    logger.info("BOTTOM 20 FEATURES BY MEAN GAIN")
    logger.info("=" * 80)
    bottom_20 = df_summary.tail(20)
    print(bottom_20.select(["feature", "mean_gain", "std_gain", "models_using"]))

    # Step 4: Create heatmap
    logger.info("\nGenerating feature importance heatmap...")
    heatmap_file = OUTPUT_DIR / "feature_importance_heatmap_top50.png"
    create_importance_heatmap(df_importance, heatmap_file, top_n=50)

    # Step 5: Identify regime-specific features
    logger.info("\nIdentifying regime-specific features...")
    regime_specific = identify_regime_specific_features(df_importance, threshold=0.6)

    regime_file = OUTPUT_DIR / "regime_specific_features.csv"
    regime_specific.write_csv(regime_file)
    logger.info(f"Saved regime-specific features to {regime_file}")

    logger.info("\n" + "=" * 80)
    logger.info("REGIME-SPECIFIC FEATURES (Gini >= 0.6)")
    logger.info("=" * 80)
    print(regime_specific.head(20))

    # Step 6: Analyze by temporal bucket
    logger.info("\nAnalyzing feature importance by temporal bucket...")
    bucket_analysis = (
        df_importance.group_by(["feature", "temporal_bucket"])
        .agg([pl.col("importance_gain").mean().alias("mean_gain")])
        .pivot(on="temporal_bucket", index="feature", values="mean_gain")
        .with_columns(
            [
                (pl.col("near").fill_null(0) + pl.col("mid").fill_null(0) + pl.col("far").fill_null(0)).alias(
                    "total_gain"
                )
            ]
        )
        .sort("total_gain", descending=True)
    )

    bucket_file = OUTPUT_DIR / "importance_by_temporal_bucket.csv"
    bucket_analysis.write_csv(bucket_file)
    logger.info(f"Saved temporal bucket analysis to {bucket_file}")

    logger.info("\n" + "=" * 80)
    logger.info("ANALYSIS COMPLETE")
    logger.info("=" * 80)
    logger.info(f"Output directory: {OUTPUT_DIR}")
    logger.info("Generated files:")
    logger.info(f"  1. {output_file.name} - Per-model importance")
    logger.info(f"  2. {summary_file.name} - Summary statistics")
    logger.info(f"  3. {heatmap_file.name} - Heatmap visualization")
    logger.info(f"  4. {regime_file.name} - Regime-specific features")
    logger.info(f"  5. {bucket_file.name} - Temporal bucket analysis")


if __name__ == "__main__":
    main()
