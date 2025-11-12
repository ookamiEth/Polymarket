#!/usr/bin/env python3
"""
NaN Prediction Root Cause Analysis
===================================

Investigates why far bucket has 3.1% NaN predictions (81,883 rows) vs <0.5% for near/mid buckets.

Analysis Steps:
1. Generate predictions on holdout data for each bucket
2. Identify rows with NaN predictions
3. Find which features have NaN/inf values in those rows
4. Compare feature distributions: NaN rows vs clean rows
5. Generate diagnostic plots

Outputs:
- nan_analysis_report.txt - Summary statistics
- nan_heatmap.png - Feature missingness by bucket
- nan_correlation_matrix.png - Which features co-occur as NaN?
- feature_validity_ranges.png - Are features out of valid ranges?

Usage:
  uv run python debug_nan_predictions.py
  uv run python debug_nan_predictions.py --bucket far  # Only analyze far bucket

Author: BT Research Team
Date: 2025-11-07
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Any

import lightgbm as lgb
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import seaborn as sns
import yaml

# Import from existing modules
from lightgbm_memory_optimized import FEATURE_COLS, MemoryMonitor

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


def load_config(config_path: Path) -> dict[str, Any]:
    """Load multi-horizon configuration."""
    with open(config_path) as f:
        config = yaml.safe_load(f)
    return config


def load_holdout_data(config: dict[str, Any]) -> pl.DataFrame:
    """
    Load holdout test data for evaluation.

    Uses the same logic as evaluate_multi_horizon.py to load the
    last 3 months of data as holdout test set.
    """
    logger.info("\nLoading holdout test data...")

    model_dir = Path(__file__).parent.parent
    data_dir = model_dir / "results" / "multi_horizon"

    # Load from stratified bucket files (which contain all required columns)
    bucket_files = {
        "near": data_dir / "near_consolidated.parquet",
        "mid": data_dir / "mid_consolidated.parquet",
        "far": data_dir / "far_consolidated.parquet",
    }

    # Check if bucket files exist
    missing_files = [name for name, path in bucket_files.items() if not path.exists()]
    if missing_files:
        raise FileNotFoundError(
            f"Stratified bucket files not found: {missing_files}. "
            "Run train_multi_horizon.py first to generate bucket files."
        )

    # Load and concatenate all buckets
    dfs = []
    for bucket_name, bucket_file in bucket_files.items():
        logger.info(f"  Loading {bucket_name} bucket: {bucket_file}")
        df = pl.scan_parquet(bucket_file)
        dfs.append(df)

    # Concatenate all buckets
    full_df = pl.concat(dfs)

    # Get date range for holdout test
    walk_forward_config = config["walk_forward_validation"]
    holdout_months = walk_forward_config["holdout_months"]

    # Get date stats
    date_stats = full_df.select(
        [pl.col("date").min().alias("min_date"), pl.col("date").max().alias("max_date")]
    ).collect()

    from dateutil.relativedelta import relativedelta

    max_date = date_stats["max_date"][0]
    holdout_start = max_date - relativedelta(months=holdout_months - 1)

    logger.info(f"  Full data range: {date_stats['min_date'][0]} to {max_date}")
    logger.info(f"  Holdout test: {holdout_start} to {max_date}")

    # Filter to holdout period
    holdout_df = full_df.filter((pl.col("date") >= holdout_start) & (pl.col("date") <= max_date))

    # Collect (holdout is small enough to fit in memory)
    holdout_df = holdout_df.collect()

    logger.info(f"  ✓ Loaded {len(holdout_df):,} holdout samples")

    return holdout_df


def generate_bucket_predictions(
    data: pl.DataFrame,
    bucket_name: str,
    model_file: Path,
    features: list[str],
    time_min: int,
    time_max: int,
) -> tuple[np.ndarray, pl.DataFrame]:
    """
    Generate predictions for a specific bucket.

    Returns:
        Tuple of (predictions array, bucket data)
    """
    logger.info(f"\n{'=' * 80}")
    logger.info(f"Generating predictions for {bucket_name.upper()} bucket ({time_min}-{time_max}s)")
    logger.info(f"{'=' * 80}")

    # Filter to bucket time range
    bucket_data = data.filter((pl.col("time_remaining") >= time_min) & (pl.col("time_remaining") < time_max))

    n_samples = len(bucket_data)
    logger.info(f"  Samples: {n_samples:,}")

    if n_samples == 0:
        logger.warning(f"  ⚠ No samples in {bucket_name} bucket")
        return np.array([]), bucket_data

    # Load model
    if not model_file.exists():
        raise FileNotFoundError(f"Model not found: {model_file}")

    logger.info(f"  Loading model: {model_file}")
    model = lgb.Booster(model_file=str(model_file))

    # Generate predictions
    x = bucket_data.select(features).to_numpy()
    predictions = model.predict(x)
    predictions = np.asarray(predictions)

    # Count NaNs
    nan_mask = np.isnan(predictions)
    n_nans = nan_mask.sum()
    nan_pct = (n_nans / n_samples) * 100

    logger.info(f"  Predictions: {n_samples:,}")
    logger.info(f"  NaN count: {n_nans:,} ({nan_pct:.2f}%)")

    if n_nans > 0:
        logger.warning(f"  ⚠ Found {n_nans:,} NaN predictions!")

    return predictions, bucket_data


def analyze_nan_features(
    predictions: np.ndarray,
    bucket_data: pl.DataFrame,
    features: list[str],
    bucket_name: str,
) -> dict[str, Any]:
    """
    Analyze which features cause NaN predictions.

    Returns:
        Dictionary with:
        - nan_mask: Boolean array of NaN predictions
        - feature_nan_counts: Dict of {feature: NaN count in NaN prediction rows}
        - feature_nan_rates: Dict of {feature: NaN rate in NaN prediction rows}
        - clean_vs_nan_stats: Comparison statistics
    """
    logger.info(f"\nAnalyzing NaN features for {bucket_name} bucket...")

    nan_mask = np.isnan(predictions)
    n_nans = nan_mask.sum()

    if n_nans == 0:
        logger.info(f"  ✓ No NaN predictions in {bucket_name} bucket")
        return {
            "nan_mask": nan_mask,
            "n_nans": 0,
            "feature_nan_counts": {},
            "feature_nan_rates": {},
            "clean_vs_nan_stats": {},
        }

    logger.info(f"  NaN predictions: {n_nans:,} / {len(predictions):,} ({n_nans / len(predictions) * 100:.2f}%)")

    # Extract data for NaN prediction rows
    nan_data = bucket_data.filter(pl.Series(nan_mask))
    clean_data = bucket_data.filter(pl.Series(~nan_mask))

    # Check each feature for NaN values
    feature_nan_counts = {}
    feature_nan_rates = {}

    for feature in features:
        if feature not in nan_data.columns:
            continue

        # Count NaNs in NaN prediction rows
        nan_values = nan_data.select(feature).to_numpy().ravel()
        nan_count = np.isnan(nan_values).sum() + np.isinf(nan_values).sum()
        nan_rate = (nan_count / n_nans) * 100

        if nan_count > 0:
            feature_nan_counts[feature] = nan_count
            feature_nan_rates[feature] = nan_rate

    # Sort by count
    sorted_features = sorted(feature_nan_counts.items(), key=lambda x: x[1], reverse=True)

    logger.info("\n  Top 20 features with NaN values in NaN prediction rows:")
    for i, (feature, count) in enumerate(sorted_features[:20], 1):
        rate = feature_nan_rates[feature]
        logger.info(f"    {i:2d}. {feature:40s}: {count:6,} ({rate:5.1f}%)")

    if len(sorted_features) == 0:
        logger.info("    ⚠ No features with NaN values found - predictions may be NaN due to model issues")

    # Compare feature distributions: clean vs NaN rows
    logger.info("\n  Comparing feature distributions (clean vs NaN rows):")

    clean_vs_nan_stats = {}
    for feature in list(dict(sorted_features[:10]).keys()):  # Top 10 features
        if feature not in clean_data.columns or feature not in nan_data.columns:
            continue

        clean_values = clean_data.select(feature).to_numpy().ravel()
        nan_values = nan_data.select(feature).to_numpy().ravel()

        # Remove NaN/inf for stats
        clean_values = clean_values[~(np.isnan(clean_values) | np.isinf(clean_values))]
        nan_values = nan_values[~(np.isnan(nan_values) | np.isinf(nan_values))]

        if len(clean_values) > 0 and len(nan_values) > 0:
            clean_vs_nan_stats[feature] = {
                "clean_mean": float(np.mean(clean_values)),
                "clean_std": float(np.std(clean_values)),
                "nan_mean": float(np.mean(nan_values)),
                "nan_std": float(np.std(nan_values)),
                "clean_min": float(np.min(clean_values)),
                "clean_max": float(np.max(clean_values)),
                "nan_min": float(np.min(nan_values)),
                "nan_max": float(np.max(nan_values)),
            }

    logger.info(f"    Computed distribution stats for {len(clean_vs_nan_stats)} features")

    return {
        "nan_mask": nan_mask,
        "n_nans": n_nans,
        "feature_nan_counts": feature_nan_counts,
        "feature_nan_rates": feature_nan_rates,
        "clean_vs_nan_stats": clean_vs_nan_stats,
        "bucket_data": bucket_data,
        "predictions": predictions,
    }


def plot_nan_heatmap(results: dict[str, Any], output_dir: Path) -> None:
    """
    Create heatmap showing NaN rates by feature and bucket.
    """
    logger.info("\nGenerating NaN heatmap...")

    # Prepare data for heatmap
    all_features = set()
    for _, analysis in results.items():
        all_features.update(analysis["feature_nan_rates"].keys())

    all_features = sorted(all_features)

    if len(all_features) == 0:
        logger.warning("  ⚠ No features with NaN values - skipping heatmap")
        return

    # Limit to top 30 features by total NaN count
    feature_totals = {}
    for feature in all_features:
        total = sum(results[bucket]["feature_nan_counts"].get(feature, 0) for bucket in results)
        feature_totals[feature] = total

    top_features = sorted(feature_totals.items(), key=lambda x: x[1], reverse=True)[:30]
    top_features = [f for f, _ in top_features]

    # Build heatmap matrix
    heatmap_data = []
    for feature in top_features:
        row = [results[bucket]["feature_nan_rates"].get(feature, 0.0) for bucket in ["near", "mid", "far"]]
        heatmap_data.append(row)

    # Convert to numpy array
    heatmap_data = np.array(heatmap_data)

    # Create plot
    plt.figure(figsize=(10, max(8, len(top_features) * 0.3)))
    sns.heatmap(
        heatmap_data,
        xticklabels=["Near (0-5min)", "Mid (5-10min)", "Far (10-15min)"],
        yticklabels=top_features,
        annot=True,
        fmt=".1f",
        cmap="YlOrRd",
        cbar_kws={"label": "NaN Rate (%)"},
    )

    plt.title("Feature NaN Rates by Time Bucket\n(% of NaN prediction rows with NaN feature values)", fontsize=14)
    plt.xlabel("Time Bucket", fontsize=12)
    plt.ylabel("Feature", fontsize=12)
    plt.tight_layout()

    output_file = output_dir / "nan_heatmap.png"
    plt.savefig(output_file, dpi=150)
    logger.info(f"  ✓ Saved: {output_file}")
    plt.close()


def save_report(results: dict[str, Any], output_dir: Path) -> None:
    """
    Save comprehensive NaN analysis report to text file.
    """
    logger.info("\nGenerating NaN analysis report...")

    report_file = output_dir / "nan_analysis_report.txt"

    with open(report_file, "w") as f:
        f.write("=" * 80 + "\n")
        f.write("NaN PREDICTION ROOT CAUSE ANALYSIS\n")
        f.write("=" * 80 + "\n\n")

        # Summary statistics
        f.write("SUMMARY STATISTICS\n")
        f.write("-" * 80 + "\n\n")

        for bucket_name in ["near", "mid", "far"]:
            analysis = results[bucket_name]
            n_nans = analysis["n_nans"]
            n_total = len(analysis.get("predictions", []))
            nan_pct = (n_nans / n_total * 100) if n_total > 0 else 0.0

            f.write(f"{bucket_name.upper()} Bucket:\n")
            f.write(f"  Total predictions: {n_total:,}\n")
            f.write(f"  NaN predictions:   {n_nans:,} ({nan_pct:.2f}%)\n")
            f.write(f"  Features with NaNs: {len(analysis['feature_nan_counts'])}\n")
            f.write("\n")

        # Per-bucket detailed analysis
        for bucket_name in ["near", "mid", "far"]:
            analysis = results[bucket_name]
            n_nans = analysis["n_nans"]

            if n_nans == 0:
                continue

            f.write("\n" + "=" * 80 + "\n")
            f.write(f"{bucket_name.upper()} BUCKET DETAILED ANALYSIS\n")
            f.write("=" * 80 + "\n\n")

            # Top features with NaN values
            f.write("Top 20 Features with NaN Values:\n")
            f.write("-" * 80 + "\n\n")

            sorted_features = sorted(analysis["feature_nan_counts"].items(), key=lambda x: x[1], reverse=True)

            for i, (feature, count) in enumerate(sorted_features[:20], 1):
                rate = analysis["feature_nan_rates"][feature]
                f.write(f"  {i:2d}. {feature:40s}: {count:6,} ({rate:5.1f}%)\n")

            # Distribution comparison
            if analysis["clean_vs_nan_stats"]:
                f.write("\n\nFeature Distribution Comparison (Clean vs NaN Rows):\n")
                f.write("-" * 80 + "\n\n")

                for feature, stats in list(analysis["clean_vs_nan_stats"].items())[:10]:
                    f.write(f"{feature}:\n")
                    f.write(
                        f"  Clean rows: mean={stats['clean_mean']:10.4f}, "
                        f"std={stats['clean_std']:10.4f}, "
                        f"range=[{stats['clean_min']:10.4f}, {stats['clean_max']:10.4f}]\n"
                    )
                    f.write(
                        f"  NaN rows:   mean={stats['nan_mean']:10.4f}, "
                        f"std={stats['nan_std']:10.4f}, "
                        f"range=[{stats['nan_min']:10.4f}, {stats['nan_max']:10.4f}]\n"
                    )
                    f.write("\n")

    logger.info(f"  ✓ Saved: {report_file}")


def main() -> None:
    """Main execution function."""
    # Parse arguments
    parser = argparse.ArgumentParser(description="NaN prediction root cause analysis")
    parser.add_argument(
        "--bucket",
        type=str,
        choices=["near", "mid", "far", "all"],
        default="all",
        help="Which bucket to analyze (default: all)",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).parent / "config" / "multi_horizon_config.yaml",
        help="Path to multi-horizon config file",
    )
    args = parser.parse_args()

    logger.info("=" * 80)
    logger.info("NaN PREDICTION ROOT CAUSE ANALYSIS")
    logger.info("=" * 80)

    monitor = MemoryMonitor()

    # Load configuration
    config = load_config(args.config)

    # Setup paths
    model_dir = Path(__file__).parent.parent
    models_dir = model_dir / Path(config["models"]["output_dir"])
    output_dir = models_dir.parent / "nan_analysis"
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"\nOutput directory: {output_dir}")

    # Load holdout test data
    data = load_holdout_data(config)

    # Get features
    schema = data.collect_schema()
    features = [col for col in FEATURE_COLS if col in schema.names()]
    logger.info(f"\nUsing {len(features)} features")

    monitor.check_memory("After loading data")

    # Determine which buckets to analyze
    buckets_to_analyze = ["near", "mid", "far"] if args.bucket == "all" else [args.bucket]

    # Analyze each bucket
    results = {}

    for bucket_name in buckets_to_analyze:
        bucket_config = config["buckets"][bucket_name]
        model_file = models_dir / f"lightgbm_{bucket_name}.txt"

        # Generate predictions
        predictions, bucket_data = generate_bucket_predictions(
            data,
            bucket_name,
            model_file,
            features,
            bucket_config["time_min"],
            bucket_config["time_max"],
        )

        # Analyze NaN features
        analysis = analyze_nan_features(predictions, bucket_data, features, bucket_name)
        results[bucket_name] = analysis

        monitor.check_memory(f"After analyzing {bucket_name} bucket")

    # Generate visualizations
    if len(buckets_to_analyze) == 3 or args.bucket == "all":
        plot_nan_heatmap(results, output_dir)

    # Save comprehensive report
    save_report(results, output_dir)

    monitor.check_memory("Final")

    logger.info("\n" + "=" * 80)
    logger.info("✓ NaN ANALYSIS COMPLETE")
    logger.info("=" * 80)
    logger.info(f"\nResults saved to: {output_dir}")
    logger.info("  - nan_analysis_report.txt")
    logger.info("  - nan_heatmap.png")


if __name__ == "__main__":
    main()
