#!/usr/bin/env python3
"""
Phase 5: Feature Attribution & Deployment Readiness Analysis
=============================================================

CRITICAL MISSING ANALYSES:
1. SHAP values & feature importance
2. Feature interaction discovery
3. Regime-conditional performance
4. Feature-based trading signals

This phase answers:
- Which features drive predictions?
- Where does the model work vs fail?
- How should features inform trading decisions?
"""

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Plot styling
plt.style.use("dark_background")
FIGURE_DIR = Path("research/model/03_deep_eda/figures")
TABLE_DIR = Path("research/model/03_deep_eda/tables")


def load_test_data(sample_frac: float = 0.05) -> pl.DataFrame:
    """Load test set with sampling (use 5% for SHAP performance)."""
    logger.info(f"Loading test data (sample fraction: {sample_frac})")

    df = pl.read_parquet("research/model/results/xgboost_residual_model_test_2024-07-01_2025-09-30.parquet")
    logger.info(f"Loaded {len(df):,} rows")

    if sample_frac < 1.0:
        df = df.sample(fraction=sample_frac, seed=42)
        logger.info(f"Sampled to {len(df):,} rows")

    # Add derived columns
    df = df.with_columns(
        [
            (pl.col("moneyness") * 100).alias("moneyness_pct"),
            (pl.col("time_remaining") / 60).alias("time_remaining_min"),
        ]
    )

    # Add error columns
    df = df.with_columns(
        [
            (pl.col("outcome") - pl.col("prob_mid")).alias("error_bs"),
            (pl.col("outcome") - pl.col("prob_corrected_xgb")).alias("error_ml"),
        ]
    )

    df = df.with_columns(
        [
            pl.col("error_bs").abs().alias("abs_error_bs"),
            pl.col("error_ml").abs().alias("abs_error_ml"),
        ]
    )

    return df


def analyze_feature_importance_proxy(df: pl.DataFrame, output_file: Path) -> None:
    """
    Proxy feature importance via correlation with residual corrections.

    Logic: Features that correlate with (residual_pred_xgb) are important.
    """
    logger.info("Analyzing feature importance (correlation proxy)")

    # Define feature categories
    feature_categories = {
        "Realized Volatility": [
            "rv_60s",
            "rv_300s",
            "rv_900s",
            "rv_3600s",
            "rv_ratio_5m_1m",
            "rv_ratio_15m_5m",
            "rv_ratio_1h_15m",
            "rv_term_structure",
        ],
        "Microstructure": [
            "momentum_60s",
            "momentum_300s",
            "momentum_900s",
            "range_60s",
            "range_300s",
            "range_900s",
            "reversals_60s",
            "reversals_300s",
            "autocorr_lag1_300s",
            "autocorr_lag5_300s",
            "hurst_300s",
        ],
        "Context": ["time_remaining", "iv_staleness_seconds", "moneyness"],
        "Jump Detection": ["jump_detected", "jump_intensity_300s"],
    }

    # Flatten feature list
    all_features = []
    for features in feature_categories.values():
        all_features.extend(features)

    # Compute correlations with residual predictions
    correlations = []
    for feature in all_features:
        if feature not in df.columns:
            continue

        # Use pearson correlation
        corr_result = df.select([feature, "residual_pred_xgb"]).drop_nulls().to_pandas().corr().iloc[0, 1]
        corr = corr_result if not np.isnan(corr_result) else 0.0

        # Find category
        category = next((cat for cat, feats in feature_categories.items() if feature in feats), "Other")

        correlations.append(
            {
                "feature": feature,
                "category": category,
                "correlation": abs(corr) if corr is not None else 0.0,
            }
        )

    df_importance = pl.DataFrame(correlations).sort("correlation", descending=True)

    # Plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8))

    # Top 20 features
    df_top20 = df_importance.head(20)

    colors_by_category = {
        "Realized Volatility": "#00FF88",
        "Microstructure": "#00D4FF",
        "Context": "#FFB000",
        "Jump Detection": "#FF3366",
        "Other": "#FFFFFF",
    }

    bar_colors = [colors_by_category.get(cat, "#FFFFFF") for cat in df_top20["category"].to_list()]

    ax1.barh(
        range(len(df_top20)),
        df_top20["correlation"].to_numpy(),
        color=bar_colors,
        alpha=0.8,
        edgecolor="white",
        linewidth=0.5,
    )
    ax1.set_yticks(range(len(df_top20)))
    ax1.set_yticklabels(df_top20["feature"].to_list())
    ax1.set_xlabel("Abs Correlation with Residual Prediction", fontsize=14)
    ax1.set_title("Top 20 Feature Importance (Correlation Proxy)", fontsize=16, pad=20)
    ax1.grid(alpha=0.2, axis="x")
    ax1.invert_yaxis()

    # Category aggregation
    df_category = (
        df_importance.group_by("category")
        .agg(
            [
                pl.len().alias("count"),
                pl.col("correlation").mean().alias("avg_correlation"),
                pl.col("correlation").max().alias("max_correlation"),
            ]
        )
        .sort("avg_correlation", descending=True)
    )

    category_colors = [colors_by_category.get(cat, "#FFFFFF") for cat in df_category["category"].to_list()]

    ax2.bar(
        range(len(df_category)),
        df_category["avg_correlation"].to_numpy(),
        color=category_colors,
        alpha=0.8,
        edgecolor="white",
        linewidth=0.5,
    )
    ax2.set_xticks(range(len(df_category)))
    ax2.set_xticklabels(df_category["category"].to_list(), rotation=45, ha="right")
    ax2.set_ylabel("Average Abs Correlation", fontsize=14)
    ax2.set_title("Feature Category Importance", fontsize=16, pad=20)
    ax2.grid(alpha=0.2, axis="y")

    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    logger.info(f"Saved: {output_file}")
    plt.close()

    # Save table
    df_importance.write_csv(TABLE_DIR / "feature_importance_proxy.csv")
    logger.info(f"Saved: {TABLE_DIR / 'feature_importance_proxy.csv'}")

    # Display top 10
    logger.info("\nTop 10 Most Important Features:")
    logger.info(str(df_importance.head(10)))


def analyze_feature_regimes(df: pl.DataFrame, output_file: Path) -> None:
    """Cluster observations into feature regimes and analyze performance."""
    logger.info("Analyzing feature regimes via clustering")

    # Select key features for clustering
    cluster_features = ["moneyness", "rv_900s", "time_remaining", "jump_intensity_300s", "autocorr_lag1_300s"]

    # Prepare data
    df_cluster = df.select(cluster_features + ["abs_error_ml", "abs_error_bs"]).drop_nulls()

    feature_matrix = df_cluster.select(cluster_features).to_numpy()

    # Standardize
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(feature_matrix)

    # K-means clustering (5 regimes)
    logger.info("Running K-means clustering (k=5)")
    kmeans = KMeans(n_clusters=5, random_state=42)
    labels = kmeans.fit_predict(scaled_features)

    # Add labels to dataframe
    df_labeled = df_cluster.with_columns([pl.Series("regime", labels)])

    # Analyze each regime
    df_regime_stats = (
        df_labeled.group_by("regime")
        .agg(
            [
                pl.len().alias("count"),
                pl.col("moneyness").mean().alias("avg_moneyness"),
                pl.col("rv_900s").mean().alias("avg_rv"),
                pl.col("time_remaining").mean().alias("avg_ttl"),
                pl.col("jump_intensity_300s").mean().alias("avg_jump"),
                pl.col("autocorr_lag1_300s").mean().alias("avg_autocorr"),
                pl.col("abs_error_ml").mean().alias("mae_ml"),
                pl.col("abs_error_bs").mean().alias("mae_bs"),
            ]
        )
        .with_columns(
            [
                ((1 - pl.col("mae_ml") / pl.col("mae_bs")) * 100).alias("improvement_pct"),
            ]
        )
        .sort("mae_ml")
    )

    # Plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    regimes = df_regime_stats["regime"].to_numpy()
    x_pos = np.arange(len(regimes))

    # MAE by regime
    width = 0.35
    ax1.bar(
        x_pos - width / 2,
        df_regime_stats["mae_bs"].to_numpy(),
        width,
        label="BS Baseline",
        color="#FF3366",
        alpha=0.8,
        edgecolor="white",
        linewidth=0.5,
    )
    ax1.bar(
        x_pos + width / 2,
        df_regime_stats["mae_ml"].to_numpy(),
        width,
        label="ML Model",
        color="#00FF88",
        alpha=0.8,
        edgecolor="white",
        linewidth=0.5,
    )
    ax1.set_xlabel("Feature Regime", fontsize=14)
    ax1.set_ylabel("Mean Absolute Error", fontsize=14)
    ax1.set_title("Model Performance by Feature Regime", fontsize=16, pad=20)
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels([f"Regime {r}" for r in regimes])
    ax1.legend(fontsize=12)
    ax1.grid(alpha=0.2, axis="y")

    # Sample distribution
    ax2.bar(x_pos, df_regime_stats["count"].to_numpy(), color="#00D4FF", alpha=0.8, edgecolor="white", linewidth=0.5)
    ax2.set_xlabel("Feature Regime", fontsize=14)
    ax2.set_ylabel("Sample Count", fontsize=14)
    ax2.set_title("Sample Distribution Across Regimes", fontsize=16, pad=20)
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels([f"Regime {r}" for r in regimes])
    ax2.grid(alpha=0.2, axis="y")

    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    logger.info(f"Saved: {output_file}")
    plt.close()

    # Save regime characteristics
    df_regime_stats.write_csv(TABLE_DIR / "feature_regime_analysis.csv")
    logger.info(f"Saved: {TABLE_DIR / 'feature_regime_analysis.csv'}")

    # Display regime characteristics
    logger.info("\nFeature Regime Characteristics:")
    logger.info(str(df_regime_stats))


def analyze_prediction_confidence(df: pl.DataFrame, output_file: Path) -> None:
    """Analyze model confidence via prediction magnitude and error relationship."""
    logger.info("Analyzing prediction confidence")

    # Calculate prediction edge
    df_confidence = df.with_columns(
        [
            (pl.col("prob_corrected_xgb") - pl.col("prob_mid")).alias("edge"),
        ]
    )

    df_confidence = df_confidence.with_columns(
        [
            pl.col("edge").abs().alias("abs_edge"),
        ]
    )

    # Bin by edge magnitude
    edge_bins = np.linspace(0, 0.5, 20)

    df_edge_analysis = (
        df_confidence.with_columns([pl.col("abs_edge").cut(edge_bins).alias("edge_bin")])
        .group_by("edge_bin")
        .agg(
            [
                pl.len().alias("count"),
                pl.col("abs_edge").mean().alias("avg_edge"),
                pl.col("abs_error_ml").mean().alias("mae"),
                pl.col("outcome").mean().alias("actual_outcome"),
                pl.col("prob_corrected_xgb").mean().alias("predicted_prob"),
            ]
        )
        .filter(pl.col("count") >= 100)
        .sort("avg_edge")
    )

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    edge = df_edge_analysis["avg_edge"].to_numpy()
    mae = df_edge_analysis["mae"].to_numpy()
    count = df_edge_analysis["count"].to_numpy()

    # Error vs edge
    ax1.scatter(edge, mae, s=count / 100, c="#00D4FF", alpha=0.7, edgecolor="white", linewidth=0.5)
    ax1.set_xlabel("Prediction Edge (|ML - BS|)", fontsize=14)
    ax1.set_ylabel("Mean Absolute Error", fontsize=14)
    ax1.set_title("Model Error vs Prediction Confidence", fontsize=16, pad=20)
    ax1.grid(alpha=0.2)

    # Calibration by confidence
    ax2.plot(
        edge,
        df_edge_analysis["predicted_prob"].to_numpy(),
        linewidth=2.5,
        color="#00FF88",
        marker="o",
        markersize=6,
        label="Predicted",
    )
    ax2.plot(
        edge,
        df_edge_analysis["actual_outcome"].to_numpy(),
        linewidth=2.5,
        color="#FFFFFF",
        marker="s",
        markersize=6,
        label="Actual",
    )
    ax2.set_xlabel("Prediction Edge (|ML - BS|)", fontsize=14)
    ax2.set_ylabel("Probability", fontsize=14)
    ax2.set_title("Calibration vs Prediction Confidence", fontsize=16, pad=20)
    ax2.grid(alpha=0.2)
    ax2.legend(fontsize=12)

    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    logger.info(f"Saved: {output_file}")
    plt.close()


def analyze_feature_based_signals(df: pl.DataFrame, output_file: Path) -> None:
    """Analyze trading signal quality based on feature thresholds."""
    logger.info("Analyzing feature-based trading signals")

    # Define signal: ML predicts higher than BS by threshold
    df_signals = df.with_columns(
        [
            (pl.col("prob_corrected_xgb") - pl.col("prob_mid")).alias("edge"),
        ]
    )

    # Test different edge thresholds
    thresholds = [0.01, 0.02, 0.03, 0.05, 0.10]

    signal_stats = []
    for threshold in thresholds:
        # Long signal: ML > BS by threshold
        df_long = df_signals.filter(pl.col("edge") > threshold)

        # Short signal: ML < BS by threshold
        df_short = df_signals.filter(pl.col("edge") < -threshold)

        if len(df_long) > 0:
            win_rate_long = df_long["outcome"].mean()
            avg_prob_long = df_long["prob_corrected_xgb"].mean()
            count_long = len(df_long)
        else:
            win_rate_long, avg_prob_long, count_long = 0, 0, 0

        if len(df_short) > 0:
            win_rate_short = 1 - df_short["outcome"].mean()  # Inverse for short
            avg_prob_short = 1 - df_short["prob_corrected_xgb"].mean()
            count_short = len(df_short)
        else:
            win_rate_short, avg_prob_short, count_short = 0, 0, 0

        signal_stats.append(
            {
                "threshold": threshold,
                "long_count": count_long,
                "long_win_rate": float(win_rate_long) if win_rate_long else 0.0,
                "long_avg_prob": float(avg_prob_long) if avg_prob_long else 0.0,
                "short_count": count_short,
                "short_win_rate": float(win_rate_short) if win_rate_short else 0.0,
                "short_avg_prob": float(avg_prob_short) if avg_prob_short else 0.0,
            }
        )

    df_signals_stats = pl.DataFrame(signal_stats)

    # Plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    thresholds_arr = df_signals_stats["threshold"].to_numpy()

    # Win rates
    ax1.plot(
        thresholds_arr,
        df_signals_stats["long_win_rate"].to_numpy(),
        linewidth=2.5,
        color="#00FF88",
        marker="o",
        markersize=8,
        label="Long Signals",
    )
    ax1.plot(
        thresholds_arr,
        df_signals_stats["short_win_rate"].to_numpy(),
        linewidth=2.5,
        color="#FF3366",
        marker="s",
        markersize=8,
        label="Short Signals",
    )
    ax1.axhline(0.5, color="gray", linestyle="--", alpha=0.5, label="Break-even")
    ax1.set_xlabel("Edge Threshold", fontsize=14)
    ax1.set_ylabel("Win Rate", fontsize=14)
    ax1.set_title("Signal Win Rate by Edge Threshold", fontsize=16, pad=20)
    ax1.grid(alpha=0.2)
    ax1.legend(fontsize=12)

    # Sample counts
    ax2.plot(
        thresholds_arr,
        df_signals_stats["long_count"].to_numpy(),
        linewidth=2.5,
        color="#00FF88",
        marker="o",
        markersize=8,
        label="Long",
    )
    ax2.plot(
        thresholds_arr,
        df_signals_stats["short_count"].to_numpy(),
        linewidth=2.5,
        color="#FF3366",
        marker="s",
        markersize=8,
        label="Short",
    )
    ax2.set_xlabel("Edge Threshold", fontsize=14)
    ax2.set_ylabel("Signal Count", fontsize=14)
    ax2.set_title("Trading Opportunity Frequency", fontsize=16, pad=20)
    ax2.grid(alpha=0.2)
    ax2.legend(fontsize=12)
    ax2.set_yscale("log")

    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    logger.info(f"Saved: {output_file}")
    plt.close()

    # Save table
    df_signals_stats.write_csv(TABLE_DIR / "feature_based_signals.csv")
    logger.info(f"Saved: {TABLE_DIR / 'feature_based_signals.csv'}")

    logger.info("\nFeature-Based Signal Analysis:")
    logger.info(str(df_signals_stats))


def main() -> None:
    """Run Phase 5: Feature Attribution & Deployment Readiness."""
    logger.info("=" * 80)
    logger.info("PHASE 5: FEATURE ATTRIBUTION & DEPLOYMENT READINESS")
    logger.info("=" * 80)

    # Load data (use 5% sample for performance)
    df = load_test_data(sample_frac=0.05)

    # Run analyses
    analyze_feature_importance_proxy(df, FIGURE_DIR / "18_feature_importance.png")
    analyze_feature_regimes(df, FIGURE_DIR / "19_feature_regimes.png")
    analyze_prediction_confidence(df, FIGURE_DIR / "20_prediction_confidence.png")
    analyze_feature_based_signals(df, FIGURE_DIR / "21_trading_signals.png")

    logger.info("=" * 80)
    logger.info("PHASE 5 COMPLETE")
    logger.info("=" * 80)
    logger.info("\nKEY INSIGHTS:")
    logger.info("- Feature importance rankings identify critical drivers")
    logger.info("- Feature regimes reveal where model excels vs struggles")
    logger.info("- Prediction confidence analysis guides position sizing")
    logger.info("- Trading signal thresholds provide actionable entry rules")


if __name__ == "__main__":
    main()
