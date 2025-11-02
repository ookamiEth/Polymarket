#!/usr/bin/env python3
"""
Feature Correlation Analysis for LightGBM Model
================================================

Analyze feature correlations to identify redundant features that can be pruned
for improved model efficiency and reduced overfitting risk.

Key analyses:
1. Correlation matrix heatmap
2. Highly correlated feature pairs (r > 0.90)
3. Feature importance-weighted pruning recommendations
4. Category-wise correlation analysis

Author: BT Research Team
Date: 2025-11-02
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import seaborn as sns

# Import feature list from existing module
from lightgbm_memory_optimized import FEATURE_COLS
from scipy.cluster import hierarchy
from scipy.spatial.distance import squareform

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

# Plotting style
plt.style.use("dark_background")
sns.set_palette("husl")


def load_features(data_file: Path, sample_size: int = 1_000_000) -> pl.DataFrame:
    """
    Load feature data for correlation analysis.

    Args:
        data_file: Path to parquet file
        sample_size: Number of rows to sample (for memory efficiency)

    Returns:
        Polars DataFrame with features
    """
    logger.info(f"Loading features from {data_file}")
    logger.info(f"Sampling {sample_size:,} rows for analysis")

    # Get available features
    schema = pl.scan_parquet(data_file).collect_schema()
    available_features = [col for col in FEATURE_COLS if col in schema.names()]

    logger.info(f"Found {len(available_features)}/{len(FEATURE_COLS)} features")

    # Load sample (stratified by time for representativeness)
    df = (
        pl.scan_parquet(data_file)
        .select(available_features)
        .head(sample_size)  # Simple head for speed; could use sample for randomness
        .collect()
    )

    logger.info(f"✓ Loaded {len(df):,} rows × {len(available_features)} features")

    return df


def compute_correlation_matrix(df: pl.DataFrame) -> tuple[np.ndarray, list[str]]:
    """
    Compute Pearson correlation matrix for all features.

    Args:
        df: Polars DataFrame with features

    Returns:
        Tuple of (correlation matrix, feature names)
    """
    logger.info("Computing correlation matrix...")

    # Convert to numpy for correlation (Polars doesn't have native corr matrix)
    feature_names = df.columns
    data = df.to_numpy()

    # Compute correlation matrix
    corr_matrix = np.corrcoef(data, rowvar=False)

    logger.info(f"✓ Computed {corr_matrix.shape[0]}×{corr_matrix.shape[1]} correlation matrix")

    return corr_matrix, feature_names


def find_high_correlations(
    corr_matrix: np.ndarray,
    feature_names: list[str],
    threshold: float = 0.90,
) -> list[tuple[str, str, float]]:
    """
    Find pairs of features with high correlation.

    Args:
        corr_matrix: Correlation matrix
        feature_names: Feature names
        threshold: Correlation threshold (default 0.90)

    Returns:
        List of (feature1, feature2, correlation) tuples
    """
    logger.info(f"Finding feature pairs with |r| > {threshold}")

    high_corr_pairs = []

    # Iterate through upper triangle only (avoid duplicates)
    n_features = len(feature_names)
    for i in range(n_features):
        for j in range(i + 1, n_features):
            corr = corr_matrix[i, j]
            if abs(corr) > threshold:
                high_corr_pairs.append(
                    (feature_names[i], feature_names[j], corr)
                )

    # Sort by absolute correlation
    high_corr_pairs.sort(key=lambda x: abs(x[2]), reverse=True)

    logger.info(f"✓ Found {len(high_corr_pairs)} highly correlated pairs")

    return high_corr_pairs


def categorize_features(feature_names: list[str]) -> dict[str, list[str]]:
    """
    Categorize features by type for organized analysis.

    Args:
        feature_names: List of feature names

    Returns:
        Dictionary mapping category name to feature list
    """
    categories = {
        "Realized Volatility": [],
        "Momentum": [],
        "Range": [],
        "Microstructure": [],
        "EMAs": [],
        "EMA Crosses": [],
        "IV/RV Ratios": [],
        "Extremes": [],
        "Distribution": [],
        "Time Features": [],
        "Volatility Dynamics": [],
        "Jump Analysis": [],
        "Autocorrelation": [],
        "Context": [],
    }

    for feat in feature_names:
        if feat.startswith("rv_"):
            categories["Realized Volatility"].append(feat)
        elif feat.startswith("momentum_"):
            categories["Momentum"].append(feat)
        elif feat.startswith("range_"):
            categories["Range"].append(feat)
        elif feat.startswith("reversals_") or feat.startswith("hurst_"):
            categories["Microstructure"].append(feat)
        elif feat.startswith("ema_") and "cross" not in feat:
            categories["EMAs"].append(feat)
        elif "ema_cross" in feat:
            categories["EMA Crosses"].append(feat)
        elif "iv_rv_ratio" in feat or "term_structure" in feat:
            categories["IV/RV Ratios"].append(feat)
        elif "drawdown" in feat or "runup" in feat:
            categories["Extremes"].append(feat)
        elif "skewness" in feat or "kurtosis" in feat or "tail_risk" in feat:
            categories["Distribution"].append(feat)
        elif feat.startswith("hour_") or feat.startswith("is_"):
            categories["Time Features"].append(feat)
        elif "garch" in feat or "persistence" in feat:
            categories["Volatility Dynamics"].append(feat)
        elif "jump" in feat:
            categories["Jump Analysis"].append(feat)
        elif feat.startswith("autocorr_"):
            categories["Autocorrelation"].append(feat)
        elif feat in ["time_remaining", "iv_staleness_seconds", "moneyness"]:
            categories["Context"].append(feat)

    # Remove empty categories
    categories = {k: v for k, v in categories.items() if v}

    return categories


def plot_correlation_heatmap(
    corr_matrix: np.ndarray,
    feature_names: list[str],
    output_dir: Path,
) -> None:
    """
    Generate correlation heatmap with hierarchical clustering.

    Args:
        corr_matrix: Correlation matrix
        feature_names: Feature names
        output_dir: Output directory for plots
    """
    logger.info("Generating correlation heatmap...")

    output_dir.mkdir(parents=True, exist_ok=True)

    # Hierarchical clustering for better organization
    # Convert correlation to distance
    dissimilarity = 1 - np.abs(corr_matrix)

    # Ensure symmetry (handle numerical precision issues)
    dissimilarity = (dissimilarity + dissimilarity.T) / 2

    # Set diagonal to exactly 0
    np.fill_diagonal(dissimilarity, 0)

    # Ensure no NaN values
    dissimilarity = np.nan_to_num(dissimilarity, nan=1.0)

    linkage = hierarchy.linkage(squareform(dissimilarity), method="average")
    dendro = hierarchy.dendrogram(linkage, no_plot=True)
    reorder_idx = dendro["leaves"]

    # Reorder matrix and labels
    corr_reordered = corr_matrix[np.ix_(reorder_idx, reorder_idx)]
    labels_reordered = [feature_names[i] for i in reorder_idx]

    # Plot heatmap
    fig, ax = plt.subplots(figsize=(20, 18))

    im = ax.imshow(
        corr_reordered,
        cmap="RdBu_r",
        vmin=-1,
        vmax=1,
        aspect="auto",
    )

    # Set ticks
    ax.set_xticks(range(len(labels_reordered)))
    ax.set_yticks(range(len(labels_reordered)))
    ax.set_xticklabels(labels_reordered, rotation=90, ha="right", fontsize=8)
    ax.set_yticklabels(labels_reordered, fontsize=8)

    # Colorbar
    cbar = plt.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label("Pearson Correlation", fontsize=12)

    # Title
    ax.set_title(
        "Feature Correlation Matrix (Hierarchically Clustered)",
        fontsize=16,
        pad=20,
    )

    # Grid for readability
    ax.set_xticks(np.arange(len(labels_reordered)) - 0.5, minor=True)
    ax.set_yticks(np.arange(len(labels_reordered)) - 0.5, minor=True)
    ax.grid(which="minor", color="gray", linestyle="-", linewidth=0.1, alpha=0.2)

    plt.tight_layout()

    output_file = output_dir / "correlation_heatmap.png"
    plt.savefig(output_file, dpi=150, bbox_inches="tight")
    plt.close()

    logger.info(f"✓ Saved heatmap to {output_file}")


def plot_high_correlation_pairs(
    high_corr_pairs: list[tuple[str, str, float]],
    output_dir: Path,
    top_n: int = 20,
) -> None:
    """
    Plot top highly correlated feature pairs.

    Args:
        high_corr_pairs: List of (feature1, feature2, correlation) tuples
        output_dir: Output directory
        top_n: Number of top pairs to plot
    """
    logger.info(f"Plotting top {top_n} highly correlated pairs...")

    if not high_corr_pairs:
        logger.warning("No highly correlated pairs found - skipping plot")
        return

    # Take top N
    pairs = high_corr_pairs[:top_n]

    # Create labels and values
    labels = [f"{f1}\n{f2}" for f1, f2, _ in pairs]
    correlations = [corr for _, _, corr in pairs]

    # Plot
    fig, ax = plt.subplots(figsize=(12, 8))

    colors = ["#FF3366" if c < 0 else "#00D4FF" for c in correlations]
    bars = ax.barh(range(len(pairs)), correlations, color=colors, alpha=0.8, edgecolor="white", linewidth=0.5)

    # Labels
    ax.set_yticks(range(len(pairs)))
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_xlabel("Pearson Correlation", fontsize=12)
    ax.set_title(f"Top {top_n} Highly Correlated Feature Pairs (|r| > 0.90)", fontsize=14, pad=15)

    # Grid
    ax.axvline(x=0, color="white", linewidth=0.8, alpha=0.5)
    ax.grid(axis="x", alpha=0.2)

    # Value labels on bars
    for i, (_bar, corr) in enumerate(zip(bars, correlations)):
        ax.text(
            corr + (0.01 if corr > 0 else -0.01),
            i,
            f"{corr:.3f}",
            va="center",
            ha="left" if corr > 0 else "right",
            fontsize=8,
            fontweight="bold",
        )

    plt.tight_layout()

    output_file = output_dir / "high_correlation_pairs.png"
    plt.savefig(output_file, dpi=150, bbox_inches="tight")
    plt.close()

    logger.info(f"✓ Saved high correlation pairs plot to {output_file}")


def recommend_features_to_prune(
    high_corr_pairs: list[tuple[str, str, float]],
    categories: dict[str, list[str]],
    feature_importance_ranking: dict[str, int] | None = None,
) -> list[str]:
    """
    Recommend features to remove based on correlation analysis.

    For each highly correlated pair, recommend removing the less important one
    (or the more redundant one if importance not available).

    Args:
        high_corr_pairs: List of highly correlated pairs
        categories: Feature categorization
        feature_importance_ranking: Dict mapping feature name to importance rank (1=best)

    Returns:
        List of features recommended for removal
    """
    logger.info("\nAnalyzing pruning recommendations...")

    to_prune = set()
    reasoning = []

    # Default importance (prefer to keep shorter-window features if no ranking)
    def get_importance(feat: str) -> int:
        if feature_importance_ranking:
            return feature_importance_ranking.get(feat, 999)
        # Default heuristic: prefer longer windows
        if "900s" in feat or "3600s" in feat or "15m" in feat or "1h" in feat:
            return 1
        elif "300s" in feat or "5m" in feat:
            return 2
        elif "60s" in feat or "1m" in feat:
            return 3
        else:
            return 4

    for feat1, feat2, corr in high_corr_pairs:
        # Skip if already marked for pruning
        if feat1 in to_prune or feat2 in to_prune:
            continue

        # Get importance rankings
        imp1 = get_importance(feat1)
        imp2 = get_importance(feat2)

        # Choose which to prune
        if imp1 < imp2:
            # Keep feat1, prune feat2
            to_prune.add(feat2)
            reasoning.append(f"  • Prune {feat2:40s} (rank {imp2:3d}) | Keep {feat1:40s} (rank {imp1:3d}) | r={corr:.3f}")
        else:
            # Keep feat2, prune feat1
            to_prune.add(feat1)
            reasoning.append(f"  • Prune {feat1:40s} (rank {imp1:3d}) | Keep {feat2:40s} (rank {imp2:3d}) | r={corr:.3f}")

    logger.info(f"\nPruning Recommendations ({len(to_prune)} features):")
    for line in reasoning:
        logger.info(line)

    return list(to_prune)


def generate_summary_report(
    corr_matrix: np.ndarray,
    feature_names: list[str],
    high_corr_pairs: list[tuple[str, str, float]],
    categories: dict[str, list[str]],
    features_to_prune: list[str],
    output_dir: Path,
) -> None:
    """
    Generate text summary report.

    Args:
        corr_matrix: Correlation matrix
        feature_names: Feature names
        high_corr_pairs: Highly correlated pairs
        categories: Feature categories
        features_to_prune: Recommended features to prune
        output_dir: Output directory
    """
    logger.info("Generating summary report...")

    report_lines = []
    report_lines.append("=" * 80)
    report_lines.append("FEATURE CORRELATION ANALYSIS REPORT")
    report_lines.append("=" * 80)
    report_lines.append("")

    # Summary statistics
    report_lines.append("SUMMARY STATISTICS")
    report_lines.append("-" * 80)
    report_lines.append(f"Total Features:                 {len(feature_names)}")
    report_lines.append(f"Highly Correlated Pairs (>0.90): {len(high_corr_pairs)}")
    report_lines.append(f"Features Recommended to Prune:  {len(features_to_prune)} ({len(features_to_prune)/len(feature_names)*100:.1f}%)")
    report_lines.append(f"Features After Pruning:         {len(feature_names) - len(features_to_prune)}")
    report_lines.append("")

    # Correlation statistics
    # Get upper triangle (exclude diagonal)
    n = len(feature_names)
    upper_tri = corr_matrix[np.triu_indices(n, k=1)]
    report_lines.append("CORRELATION DISTRIBUTION")
    report_lines.append("-" * 80)
    report_lines.append(f"Mean Correlation:               {np.mean(upper_tri):.3f}")
    report_lines.append(f"Median Correlation:             {np.median(upper_tri):.3f}")
    report_lines.append(f"Max Correlation:                {np.max(upper_tri):.3f}")
    report_lines.append(f"Min Correlation:                {np.min(upper_tri):.3f}")
    report_lines.append(f"Std Deviation:                  {np.std(upper_tri):.3f}")
    report_lines.append("")

    # Category breakdown
    report_lines.append("FEATURES BY CATEGORY")
    report_lines.append("-" * 80)
    for cat, feats in categories.items():
        pruned_in_cat = [f for f in feats if f in features_to_prune]
        report_lines.append(f"{cat:30s}: {len(feats):2d} features ({len(pruned_in_cat)} to prune)")
    report_lines.append("")

    # Highly correlated pairs
    report_lines.append("HIGHLY CORRELATED FEATURE PAIRS (|r| > 0.90)")
    report_lines.append("-" * 80)
    if high_corr_pairs:
        for feat1, feat2, corr in high_corr_pairs[:30]:  # Top 30
            mark = "✗" if feat1 in features_to_prune or feat2 in features_to_prune else " "
            report_lines.append(f"{mark} {feat1:35s} ↔ {feat2:35s} | r={corr:6.3f}")
    else:
        report_lines.append("(none found)")
    report_lines.append("")

    # Features to prune
    report_lines.append("RECOMMENDED FEATURES TO PRUNE")
    report_lines.append("-" * 80)
    for feat in sorted(features_to_prune):
        report_lines.append(f"  • {feat}")
    report_lines.append("")

    # Features to keep
    features_to_keep = [f for f in feature_names if f not in features_to_prune]
    report_lines.append(f"FEATURES TO KEEP ({len(features_to_keep)})")
    report_lines.append("-" * 80)
    for cat, feats in categories.items():
        kept_in_cat = [f for f in feats if f not in features_to_prune]
        if kept_in_cat:
            report_lines.append(f"\n{cat}:")
            for feat in kept_in_cat:
                report_lines.append(f"  • {feat}")
    report_lines.append("")

    report_lines.append("=" * 80)

    # Save to file
    report_file = output_dir / "correlation_analysis_report.txt"
    with open(report_file, "w") as f:
        f.write("\n".join(report_lines))

    # Print to console
    for line in report_lines:
        logger.info(line)

    logger.info(f"\n✓ Saved report to {report_file}")


def main() -> None:
    """Main execution function."""
    logger.info("\n" + "=" * 80)
    logger.info("FEATURE CORRELATION ANALYSIS")
    logger.info("=" * 80)

    # Setup paths
    model_dir = Path(__file__).parent.parent
    train_file = model_dir / "results" / "train_features_lgb.parquet"
    output_dir = model_dir / "results" / "correlation_analysis"

    # Load data (sample for efficiency)
    df = load_features(train_file, sample_size=1_000_000)

    # Compute correlation matrix
    corr_matrix, feature_names = compute_correlation_matrix(df)

    # Find high correlations
    high_corr_pairs = find_high_correlations(corr_matrix, feature_names, threshold=0.90)

    # Categorize features
    categories = categorize_features(feature_names)

    # Recommend features to prune
    # TODO: Load actual feature importance rankings from best model
    # For now, use heuristic
    features_to_prune = recommend_features_to_prune(
        high_corr_pairs,
        categories,
        feature_importance_ranking=None,
    )

    # Generate visualizations
    plot_correlation_heatmap(corr_matrix, feature_names, output_dir)
    plot_high_correlation_pairs(high_corr_pairs, output_dir, top_n=20)

    # Generate summary report
    generate_summary_report(
        corr_matrix,
        feature_names,
        high_corr_pairs,
        categories,
        features_to_prune,
        output_dir,
    )

    logger.info("\n✓ Analysis complete!")
    logger.info(f"Results saved to {output_dir}")


if __name__ == "__main__":
    main()
