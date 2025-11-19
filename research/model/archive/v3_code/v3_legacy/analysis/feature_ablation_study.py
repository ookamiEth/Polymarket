#!/usr/bin/env python3
"""
Feature Ablation Study for LightGBM Model
==========================================

Systematic removal of feature categories to measure their individual contribution
to model performance. Addresses the critique: "No ablation studies to confirm if
gains are from params or data luck."

Strategy:
1. Train baseline model with all 62 features
2. Remove each feature category one at a time
3. Retrain and measure performance drop
4. Identify truly essential vs redundant feature groups

Feature Categories:
- Realized Volatility (8 features)
- Momentum (3 features)
- Range (3 features)
- Microstructure (5 features)
- EMAs (4 features)
- EMA Crosses (3 features)
- IV/RV Ratios (4 features)
- Extremes (4 features)
- Distribution (6 features)
- Time Features (6 features)
- Volatility Dynamics (4 features)
- Jump Analysis (3 features)
- Autocorrelation (4 features)
- Context (3 features)

Author: BT Research Team
Date: 2025-11-02
"""

from __future__ import annotations

import gc
import logging
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import lightgbm as lgb
import matplotlib.pyplot as plt
import pandas as pd
import polars as pl
import yaml

# Import from existing modules
from lightgbm_memory_optimized import (
    FEATURE_COLS,
    create_lgb_dataset_from_parquet,
)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

# Plotting style
plt.style.use("dark_background")


@dataclass
class AblationResult:
    """Results from removing a single feature category."""
    category_name: str
    n_features_removed: int
    baseline_brier_improvement: float
    ablated_brier_improvement: float
    performance_drop_pct: float
    performance_drop_abs: float
    training_time_minutes: float
    features_removed: list[str]


def categorize_features(feature_names: list[str]) -> dict[str, list[str]]:
    """
    Categorize features into logical groups.

    Args:
        feature_names: List of all feature names

    Returns:
        Dictionary mapping category name to feature list
    """
    categories = {}

    # Realized Volatility
    categories["Realized Volatility"] = [
        f for f in feature_names
        if f.startswith("rv_") and "ratio" not in f and "term_structure" not in f
    ]

    # Momentum
    categories["Momentum"] = [f for f in feature_names if f.startswith("momentum_")]

    # Range
    categories["Range"] = [f for f in feature_names if f.startswith("range_")]

    # Microstructure (reversals, hurst, autocorr)
    categories["Microstructure"] = [
        f for f in feature_names
        if f.startswith("reversals_") or f.startswith("hurst_")
    ]

    # EMAs (non-cross)
    categories["EMAs"] = [
        f for f in feature_names
        if f.startswith("ema_") and "cross" not in f
    ]

    # EMA Crosses
    categories["EMA Crosses"] = [f for f in feature_names if "ema_cross" in f]

    # IV/RV Ratios
    categories["IV/RV Ratios"] = [
        f for f in feature_names
        if "iv_rv_ratio" in f or f == "rv_term_structure"
    ]

    # Extremes
    categories["Extremes"] = [
        f for f in feature_names
        if "drawdown" in f or "runup" in f
    ]

    # Distribution
    categories["Distribution"] = [
        f for f in feature_names
        if "skewness" in f or "kurtosis" in f or "tail_risk" in f
    ]

    # Time Features
    categories["Time Features"] = [
        f for f in feature_names
        if f.startswith("hour_") or f.startswith("is_") or f.startswith("day_")
    ]

    # Volatility Dynamics
    categories["Volatility Dynamics"] = [
        f for f in feature_names
        if "garch" in f or "persistence" in f or "vol_regime" in f
    ]

    # Jump Analysis
    categories["Jump Analysis"] = [f for f in feature_names if "jump" in f]

    # Autocorrelation
    categories["Autocorrelation"] = [f for f in feature_names if f.startswith("autocorr_")]

    # Context (essential features)
    categories["Context"] = [
        f for f in feature_names
        if f in ["time_remaining", "iv_staleness_seconds", "moneyness"]
    ]

    # RV Ratios (separate from IV/RV)
    categories["RV Ratios"] = [
        f for f in feature_names
        if "rv_ratio" in f
    ]

    # Remove empty categories
    categories = {k: v for k, v in categories.items() if v}

    return categories


def train_with_features(
    train_file: Path,
    val_file: Path,
    features: list[str],
    config: dict[str, Any],
    category_name: str = "Baseline",
) -> tuple[float, float]:
    """
    Train model with specified features.

    Args:
        train_file: Training data file
        val_file: Validation data file
        features: List of features to use
        config: Hyperparameter configuration
        category_name: Name for logging

    Returns:
        Tuple of (Brier improvement %, training time in minutes)
    """
    logger.info(f"\nTraining with {len(features)} features ({category_name})...")

    start_time = time.time()

    # Load datasets
    train_data, baseline_brier_train = create_lgb_dataset_from_parquet(
        str(train_file), features, free_raw_data=True
    )
    val_data, baseline_brier_val = create_lgb_dataset_from_parquet(
        str(val_file), features, reference=train_data, free_raw_data=True
    )

    # Train model
    callbacks = [
        lgb.early_stopping(config.get("early_stopping_rounds", 50)),
        lgb.log_evaluation(period=0),  # Suppress logging
    ]

    model = lgb.train(
        config,
        train_data,
        num_boost_round=config.get("n_estimators", 1000),
        valid_sets=[train_data, val_data],
        valid_names=["train", "val"],
        callbacks=callbacks,
    )

    # Get metrics
    best_score = model.best_score
    val_metrics = best_score.get("val", {})
    residual_mse = val_metrics.get("l2", 0.0)
    brier_improvement_pct = (residual_mse / baseline_brier_val) * 100

    # Training time
    elapsed = (time.time() - start_time) / 60

    logger.info(f"  Brier Improvement: {brier_improvement_pct:.3f}%")
    logger.info(f"  Training Time:     {elapsed:.2f} min")

    # Clean up
    del train_data, val_data, model
    gc.collect()

    return brier_improvement_pct, elapsed


def run_ablation_study(
    train_file: Path,
    val_file: Path,
    all_features: list[str],
    config: dict[str, Any],
) -> tuple[float, list[AblationResult]]:
    """
    Run complete ablation study.

    Args:
        train_file: Training data file
        val_file: Validation data file
        all_features: List of all features
        config: Hyperparameter configuration

    Returns:
        Tuple of (baseline improvement, list of ablation results)
    """
    logger.info("\n" + "=" * 80)
    logger.info("FEATURE ABLATION STUDY")
    logger.info("=" * 80)

    # Categorize features
    categories = categorize_features(all_features)

    logger.info(f"\nFeature Categories ({len(categories)} total):")
    for cat, feats in categories.items():
        logger.info(f"  {cat:30s}: {len(feats):2d} features")

    # Train baseline with all features
    logger.info("\n" + "=" * 80)
    logger.info("BASELINE: Training with ALL features")
    logger.info("=" * 80)

    baseline_improvement, baseline_time = train_with_features(
        train_file, val_file, all_features, config, "All Features"
    )

    logger.info(f"\n✓ Baseline Brier Improvement: {baseline_improvement:.3f}%")

    # Run ablation for each category
    ablation_results = []

    for cat_name, cat_features in categories.items():
        logger.info("\n" + "=" * 80)
        logger.info(f"ABLATION: Removing {cat_name}")
        logger.info("=" * 80)

        # Create feature set without this category
        ablated_features = [f for f in all_features if f not in cat_features]

        logger.info(f"Removing {len(cat_features)} features from {cat_name}:")
        for feat in cat_features:
            logger.info(f"  - {feat}")

        # Train without this category
        ablated_improvement, ablated_time = train_with_features(
            train_file, val_file, ablated_features, config, f"Without {cat_name}"
        )

        # Calculate performance drop
        performance_drop_abs = baseline_improvement - ablated_improvement
        performance_drop_pct = (performance_drop_abs / baseline_improvement) * 100

        result = AblationResult(
            category_name=cat_name,
            n_features_removed=len(cat_features),
            baseline_brier_improvement=baseline_improvement,
            ablated_brier_improvement=ablated_improvement,
            performance_drop_pct=performance_drop_pct,
            performance_drop_abs=performance_drop_abs,
            training_time_minutes=ablated_time,
            features_removed=cat_features,
        )

        ablation_results.append(result)

        logger.info(f"\nResult for {cat_name}:")
        logger.info(f"  Baseline Improvement:   {baseline_improvement:.3f}%")
        logger.info(f"  Ablated Improvement:    {ablated_improvement:.3f}%")
        logger.info(f"  Performance Drop:       {performance_drop_abs:.3f}% ({performance_drop_pct:.1f}% relative)")

    return baseline_improvement, ablation_results


def plot_ablation_results(
    baseline_improvement: float,
    ablation_results: list[AblationResult],
    output_dir: Path,
) -> None:
    """
    Generate visualizations for ablation study.

    Args:
        baseline_improvement: Baseline Brier improvement
        ablation_results: List of ablation results
        output_dir: Output directory
    """
    logger.info("\nGenerating ablation study plots...")

    output_dir.mkdir(parents=True, exist_ok=True)

    # Sort by performance drop (descending)
    results_sorted = sorted(ablation_results, key=lambda x: x.performance_drop_abs, reverse=True)

    # Extract data
    categories = [r.category_name for r in results_sorted]
    performance_drops = [r.performance_drop_abs for r in results_sorted]
    n_features = [r.n_features_removed for r in results_sorted]

    # Plot 1: Performance drop bar chart
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8))

    # Left plot: Absolute performance drop
    colors = ["#FF3366" if drop > 0.5 else "#00D4FF" if drop > 0.1 else "#888888" for drop in performance_drops]
    bars = ax1.barh(range(len(categories)), performance_drops, color=colors, alpha=0.8, edgecolor="white", linewidth=1)

    ax1.set_yticks(range(len(categories)))
    ax1.set_yticklabels(categories, fontsize=10)
    ax1.set_xlabel("Performance Drop (% points)", fontsize=12)
    ax1.set_title("Feature Category Importance\n(Larger drop = more important)", fontsize=14, pad=15)
    ax1.grid(axis="x", alpha=0.2)

    # Add value labels
    for i, (_bar, drop, n) in enumerate(zip(bars, performance_drops, n_features)):
        ax1.text(
            drop + 0.02,
            i,
            f"{drop:.3f}% ({n}f)",
            va="center",
            ha="left",
            fontsize=9,
        )

    # Threshold lines
    ax1.axvline(x=0.5, color="#FF3366", linestyle="--", linewidth=1, alpha=0.5, label="High Impact (>0.5%)")
    ax1.axvline(x=0.1, color="#00D4FF", linestyle="--", linewidth=1, alpha=0.5, label="Medium Impact (>0.1%)")
    ax1.legend(fontsize=9)

    # Right plot: Feature count vs performance drop
    ax2.scatter(n_features, performance_drops, s=200, alpha=0.7, c=colors, edgecolor="white", linewidth=1.5)

    for i, cat in enumerate(categories):
        ax2.annotate(
            cat,
            (n_features[i], performance_drops[i]),
            xytext=(5, 5),
            textcoords="offset points",
            fontsize=8,
            alpha=0.8,
        )

    ax2.set_xlabel("Number of Features Removed", fontsize=12)
    ax2.set_ylabel("Performance Drop (% points)", fontsize=12)
    ax2.set_title("Feature Efficiency\n(More features ≠ better performance)", fontsize=14, pad=15)
    ax2.grid(alpha=0.2)

    plt.tight_layout()
    plt.savefig(output_dir / "ablation_study_results.png", dpi=150, bbox_inches="tight")
    plt.close()

    logger.info("✓ Saved ablation study plot")


def generate_ablation_report(
    baseline_improvement: float,
    ablation_results: list[AblationResult],
    output_dir: Path,
) -> None:
    """
    Generate text report for ablation study.

    Args:
        baseline_improvement: Baseline Brier improvement
        ablation_results: List of ablation results
        output_dir: Output directory
    """
    logger.info("\nGenerating ablation study report...")

    # Sort by performance drop
    results_sorted = sorted(ablation_results, key=lambda x: x.performance_drop_abs, reverse=True)

    # Categorize by impact
    high_impact = [r for r in results_sorted if r.performance_drop_abs > 0.5]
    medium_impact = [r for r in results_sorted if 0.1 < r.performance_drop_abs <= 0.5]
    low_impact = [r for r in results_sorted if r.performance_drop_abs <= 0.1]

    report_lines = []
    report_lines.append("=" * 80)
    report_lines.append("FEATURE ABLATION STUDY REPORT")
    report_lines.append("=" * 80)
    report_lines.append("")

    # Summary
    report_lines.append("SUMMARY")
    report_lines.append("-" * 80)
    report_lines.append(f"Baseline Brier Improvement:     {baseline_improvement:.3f}%")
    report_lines.append(f"Total Feature Categories:       {len(ablation_results)}")
    report_lines.append(f"High Impact Categories (>0.5%): {len(high_impact)}")
    report_lines.append(f"Medium Impact (0.1-0.5%):       {len(medium_impact)}")
    report_lines.append(f"Low Impact (<0.1%):             {len(low_impact)}")
    report_lines.append("")

    # High impact features (ESSENTIAL)
    report_lines.append("HIGH IMPACT CATEGORIES (ESSENTIAL - DO NOT REMOVE)")
    report_lines.append("-" * 80)
    if high_impact:
        for r in high_impact:
            report_lines.append(f"\n{r.category_name} ({r.n_features_removed} features)")
            report_lines.append(f"  Performance Drop: {r.performance_drop_abs:.3f}% ({r.performance_drop_pct:.1f}% relative)")
            report_lines.append("  Features:")
            for feat in r.features_removed:
                report_lines.append(f"    - {feat}")
    else:
        report_lines.append("(none)")
    report_lines.append("")

    # Medium impact
    report_lines.append("MEDIUM IMPACT CATEGORIES (USEFUL)")
    report_lines.append("-" * 80)
    if medium_impact:
        for r in medium_impact:
            report_lines.append(f"{r.category_name:30s}: {r.performance_drop_abs:6.3f}% drop ({r.n_features_removed} features)")
    else:
        report_lines.append("(none)")
    report_lines.append("")

    # Low impact (CANDIDATES FOR REMOVAL)
    report_lines.append("LOW IMPACT CATEGORIES (CANDIDATES FOR REMOVAL)")
    report_lines.append("-" * 80)
    if low_impact:
        for r in low_impact:
            report_lines.append(f"{r.category_name:30s}: {r.performance_drop_abs:6.3f}% drop ({r.n_features_removed} features)")
            report_lines.append("  Features to remove:")
            for feat in r.features_removed:
                report_lines.append(f"    - {feat}")
    else:
        report_lines.append("(none)")
    report_lines.append("")

    # Pruning recommendations
    total_prunable = sum(r.n_features_removed for r in low_impact)
    report_lines.append("PRUNING RECOMMENDATIONS")
    report_lines.append("-" * 80)
    report_lines.append(f"Features that can be safely removed: {total_prunable}")
    report_lines.append("Categories to remove:")
    for r in low_impact:
        report_lines.append(f"  • {r.category_name} ({r.n_features_removed} features, {r.performance_drop_abs:.3f}% impact)")
    report_lines.append("")
    report_lines.append(f"Expected performance after pruning: ~{baseline_improvement - sum(r.performance_drop_abs for r in low_impact):.3f}%")
    report_lines.append(f"Training speedup estimate: ~{total_prunable / 62 * 20:.1f}% faster")
    report_lines.append("")

    report_lines.append("=" * 80)

    # Save to file
    report_file = output_dir / "ablation_study_report.txt"
    with open(report_file, "w") as f:
        f.write("\n".join(report_lines))

    # Print to console
    for line in report_lines:
        logger.info(line)

    logger.info(f"\n✓ Saved report to {report_file}")

    # Save CSV
    results_df = pd.DataFrame([
        {
            "category": r.category_name,
            "n_features": r.n_features_removed,
            "baseline_improvement": r.baseline_brier_improvement,
            "ablated_improvement": r.ablated_brier_improvement,
            "performance_drop_abs": r.performance_drop_abs,
            "performance_drop_pct": r.performance_drop_pct,
            "training_time_min": r.training_time_minutes,
        }
        for r in results_sorted
    ])

    csv_file = output_dir / "ablation_results.csv"
    results_df.to_csv(csv_file, index=False)
    logger.info(f"✓ Saved CSV to {csv_file}")


def main() -> None:
    """Main execution function."""
    logger.info("\n" + "=" * 80)
    logger.info("FEATURE ABLATION STUDY")
    logger.info("=" * 80)

    # Load config
    config_path = Path(__file__).parent / "config" / "best_model_config.yaml"
    logger.info(f"\nLoading configuration from {config_path}")

    with open(config_path) as f:
        config_yaml = yaml.safe_load(f)

    config = config_yaml["hyperparameters"]

    # Setup paths
    model_dir = Path(__file__).parent.parent
    train_file = model_dir / "results" / "train_features_lgb.parquet"
    val_file = model_dir / "results" / "val_features_lgb.parquet"
    output_dir = model_dir / "results" / "ablation_study"

    # Get features
    schema = pl.scan_parquet(train_file).collect_schema()
    all_features = [col for col in FEATURE_COLS if col in schema.names()]
    logger.info(f"\nTotal features: {len(all_features)}")

    # Run ablation study
    baseline_improvement, ablation_results = run_ablation_study(
        train_file, val_file, all_features, config
    )

    # Generate visualizations
    plot_ablation_results(baseline_improvement, ablation_results, output_dir)

    # Generate report
    generate_ablation_report(baseline_improvement, ablation_results, output_dir)

    logger.info("\n✓ Ablation study complete!")
    logger.info(f"Results saved to {output_dir}")


if __name__ == "__main__":
    main()
