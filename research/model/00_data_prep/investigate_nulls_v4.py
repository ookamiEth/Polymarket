#!/usr/bin/env python3
"""
Deep Dive Null Value Investigation
===================================

Analyzes null patterns in consolidated_features_v4.parquet to identify root causes:
1. Temporal distribution of nulls (are they concentrated at specific dates?)
2. Feature category analysis (which feature types have most nulls?)
3. Correlation between null features (do nulls occur together?)
4. Source file validation (check if nulls come from input files)
5. Rolling window edge effects verification

Author: BT Research Team
Date: 2025-11-13
"""

from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path

import polars as pl

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Paths
FEATURES_FILE = Path("/home/ubuntu/Polymarket/research/model/data/consolidated_features_v4.parquet")
BASELINE_FILE = Path("/home/ubuntu/Polymarket/research/model/results/production_backtest_results_v4.parquet")
ADVANCED_FILE = Path("/home/ubuntu/Polymarket/research/model/results/advanced_features.parquet")
MICRO_FILE = Path("/home/ubuntu/Polymarket/research/model/results/microstructure_features.parquet")


def investigate_nulls() -> None:
    """Deep dive into null value patterns."""
    logger.info("=" * 80)
    logger.info("NULL VALUE DEEP DIVE INVESTIGATION")
    logger.info("=" * 80)
    logger.info("")

    # Load consolidated features
    logger.info("Loading consolidated features...")
    df = pl.scan_parquet(FEATURES_FILE)
    schema = df.collect_schema()

    # ============================================================
    # ANALYSIS 1: Temporal Distribution of Nulls
    # ============================================================
    logger.info("")
    logger.info("=" * 80)
    logger.info("ANALYSIS 1: Temporal Distribution of Nulls")
    logger.info("=" * 80)

    # Sample features with high null rates
    high_null_features = [
        "rv_95th_percentile",  # 42.45% nulls (30-day rolling)
        "sigma_mid",           # 6.47% nulls (IV data)
        "reversals_300s",      # 5.44% nulls (300s rolling)
    ]

    logger.info("Analyzing null patterns for key features...")

    for feature in high_null_features:
        logger.info(f"\n{feature}:")

        # Get date of first non-null value
        first_nonnull = (
            df.filter(pl.col(feature).is_not_null())
            .select(pl.col("timestamp_seconds").min())
            .collect()
            .item()
        )

        # Get total null count
        null_count = df.filter(pl.col(feature).is_null()).select(pl.len()).collect().item()
        total_count = df.select(pl.len()).collect().item()

        first_nonnull_date = datetime.fromtimestamp(first_nonnull).strftime("%Y-%m-%d %H:%M:%S UTC")

        logger.info(f"  First non-null: {first_nonnull_date}")
        logger.info(f"  Total nulls: {null_count:,} ({null_count/total_count*100:.2f}%)")

        # Check if nulls are at the beginning (rolling window effect)
        data_start = df.select(pl.col("timestamp_seconds").min()).collect().item()
        days_until_nonnull = (first_nonnull - data_start) / (24 * 3600)
        logger.info(f"  Days from start until first non-null: {days_until_nonnull:.1f}")

    # ============================================================
    # ANALYSIS 2: Feature Category Breakdown
    # ============================================================
    logger.info("")
    logger.info("=" * 80)
    logger.info("ANALYSIS 2: Feature Category Null Breakdown")
    logger.info("=" * 80)

    # Categorize features
    feature_categories = {
        "Baseline": ["time_remaining", "S", "K", "sigma_mid", "T_years", "moneyness", "iv_staleness_seconds"],
        "Funding": [col for col in schema.names() if "funding" in col],
        "Orderbook L0": [col for col in schema.names() if any(x in col for x in ["spread", "imbalance"]) and "depth" not in col],
        "Orderbook 5-level": [col for col in schema.names() if any(x in col for x in ["depth", "weighted_mid", "volume_ratio"])],
        "Price Basis": [col for col in schema.names() if "mark_index" in col],
        "Open Interest": [col for col in schema.names() if col.startswith("oi") or col == "open_interest"],
        "RV/Momentum/Range": [col for col in schema.names() if any(x in col for x in ["rv_", "momentum_", "range_", "ema_900s", "price_vs_ema"])],
        "Advanced (V3)": [col for col in schema.names() if any(x in col for x in ["high_15m", "low_15m", "time_since", "skewness", "kurtosis", "downside_vol", "upside_vol", "vol_asymmetry", "tail_risk", "hour_", "vol_persistence", "vol_acceleration", "vol_of_vol", "garch", "autocorr", "hurst", "iv_rv_ratio"])],
        "V4 Moneyness": [col for col in schema.names() if any(x in col for x in ["log_moneyness", "moneyness_squared", "moneyness_cubed", "standardized_moneyness", "moneyness_x_", "moneyness_distance", "moneyness_percentile"])],
        "V4 Volatility Asymmetry": [col for col in schema.names() if any(x in col for x in ["returns_", "downside_vol_", "upside_vol_", "vol_asymmetry_ratio", "realized_skewness", "realized_kurtosis", "iv_minus_"])],
        "V4 Orderbook Norm": [col for col in schema.names() if any(x in col for x in ["spread_vol_normalized", "depth_imbalance_vol_normalized", "weighted_mid_velocity_normalized"])],
        "V4 Extreme/Regime": [col for col in schema.names() if any(x in col for x in ["rv_ratio", "rv_95th_percentile", "is_extreme", "position_scale", "vol_low_thresh", "vol_high_thresh", "volatility_regime", "market_regime"])],
    }

    logger.info("\nNull counts by feature category:")
    logger.info(f"{'Category':<30} {'Features':<10} {'Avg Null %':<12} {'Max Null %':<12}")
    logger.info("-" * 70)

    category_stats = []

    for category, cols in feature_categories.items():
        if not cols:
            continue

        # Filter to columns that exist in schema
        existing_cols = [c for c in cols if c in schema.names()]
        if not existing_cols:
            continue

        # Compute null percentages
        null_exprs = [pl.col(col).is_null().sum().alias(col) for col in existing_cols]
        null_counts_df = df.select(null_exprs).collect()

        total_rows = df.select(pl.len()).collect().item()

        null_percentages = []
        for col in existing_cols:
            null_pct = (null_counts_df[col][0] / total_rows) * 100
            null_percentages.append(null_pct)

        avg_null_pct = sum(null_percentages) / len(null_percentages) if null_percentages else 0
        max_null_pct = max(null_percentages) if null_percentages else 0

        category_stats.append((category, len(existing_cols), avg_null_pct, max_null_pct))
        logger.info(f"{category:<30} {len(existing_cols):<10} {avg_null_pct:<11.2f}% {max_null_pct:<11.2f}%")

    # ============================================================
    # ANALYSIS 3: Source File Validation
    # ============================================================
    logger.info("")
    logger.info("=" * 80)
    logger.info("ANALYSIS 3: Source File Validation")
    logger.info("=" * 80)

    logger.info("\nChecking if nulls exist in source files...")

    # Check baseline file
    logger.info("\n1. Baseline file (production_backtest_results_v4.parquet):")
    if BASELINE_FILE.exists():
        baseline = pl.scan_parquet(BASELINE_FILE)
        baseline_schema = baseline.collect_schema()

        # Check sigma_mid (known to have nulls)
        if "sigma_mid" in baseline_schema.names():
            sigma_nulls = baseline.filter(pl.col("sigma_mid").is_null()).select(pl.len()).collect().item()
            sigma_total = baseline.select(pl.len()).collect().item()
            logger.info(f"  sigma_mid nulls: {sigma_nulls:,}/{sigma_total:,} ({sigma_nulls/sigma_total*100:.2f}%)")

            # Check temporal distribution
            sigma_first_nonnull = baseline.filter(pl.col("sigma_mid").is_not_null()).select(pl.col("timestamp").min()).collect().item()
            sigma_first_date = datetime.fromtimestamp(sigma_first_nonnull).strftime("%Y-%m-%d %H:%M:%S UTC")
            logger.info(f"  First non-null sigma_mid: {sigma_first_date}")
        else:
            logger.info("  sigma_mid not found in baseline file")
    else:
        logger.info("  Baseline file not found")

    # Check advanced file
    logger.info("\n2. Advanced file (advanced_features.parquet):")
    if ADVANCED_FILE.exists():
        advanced = pl.scan_parquet(ADVANCED_FILE)
        advanced_schema = advanced.collect_schema()

        # Check a few features
        check_features = ["skewness_300s", "vol_of_vol_300s", "autocorr_decay"]
        for feat in check_features:
            if feat in advanced_schema.names():
                feat_nulls = advanced.filter(pl.col(feat).is_null()).select(pl.len()).collect().item()
                feat_total = advanced.select(pl.len()).collect().item()
                logger.info(f"  {feat} nulls: {feat_nulls:,}/{feat_total:,} ({feat_nulls/feat_total*100:.2f}%)")
    else:
        logger.info("  Advanced file not found")

    # Check microstructure file
    logger.info("\n3. Microstructure file (microstructure_features.parquet):")
    if MICRO_FILE.exists():
        micro = pl.scan_parquet(MICRO_FILE)
        micro_schema = micro.collect_schema()

        check_features = ["autocorr_lag5_300s", "hurst_300s", "reversals_300s"]
        for feat in check_features:
            if feat in micro_schema.names():
                feat_nulls = micro.filter(pl.col(feat).is_null()).select(pl.len()).collect().item()
                feat_total = micro.select(pl.len()).collect().item()
                logger.info(f"  {feat} nulls: {feat_nulls:,}/{feat_total:,} ({feat_nulls/feat_total*100:.2f}%)")
    else:
        logger.info("  Microstructure file not found")

    # ============================================================
    # ANALYSIS 4: Rolling Window Edge Effects
    # ============================================================
    logger.info("")
    logger.info("=" * 80)
    logger.info("ANALYSIS 4: Rolling Window Edge Effects Verification")
    logger.info("=" * 80)

    rolling_windows = {
        "30 days (2,592,000s)": ["rv_95th_percentile", "vol_low_thresh", "vol_high_thresh"],
        "3600s (1 hour)": ["rv_900s_ema_3600s", "momentum_900s_ema_3600s", "range_900s_ema_3600s"],
        "900s (15 min)": ["rv_300s_ema_900s", "momentum_300s_ema_900s", "range_300s_ema_900s"],
        "300s (5 min)": ["skewness_300s", "kurtosis_300s", "downside_vol_300s", "vol_of_vol_300s"],
    }

    logger.info("\nExpected null counts for rolling windows (from data start):")
    logger.info(f"{'Window Size':<25} {'Expected Nulls':<20} {'Example Features'}")
    logger.info("-" * 80)

    data_start = df.select(pl.col("timestamp_seconds").min()).collect().item()
    total_rows = df.select(pl.len()).collect().item()

    for window_name, features in rolling_windows.items():
        # Extract window size in seconds
        if "days" in window_name:
            window_seconds = int(window_name.split()[0]) * 24 * 3600
        else:
            window_seconds = int(window_name.split("s")[0])

        # Expected nulls = all rows with timestamp < (data_start + window_seconds)
        expected_null_ts = data_start + window_seconds
        expected_nulls = df.filter(pl.col("timestamp_seconds") < expected_null_ts).select(pl.len()).collect().item()
        expected_null_pct = (expected_nulls / total_rows) * 100

        # Check actual for one feature
        if features and features[0] in schema.names():
            actual_nulls = df.filter(pl.col(features[0]).is_null()).select(pl.len()).collect().item()
            actual_null_pct = (actual_nulls / total_rows) * 100

            match = "✓" if abs(actual_nulls - expected_nulls) < 10000 else "✗"
            logger.info(f"{window_name:<25} {expected_nulls:>12,} ({expected_null_pct:>5.2f}%) {match}  {features[0]}")
            if match == "✗":
                logger.info(f"{'':25} Actual: {actual_nulls:>12,} ({actual_null_pct:>5.2f}%) - MISMATCH!")

    # ============================================================
    # ANALYSIS 5: Date Gap Analysis
    # ============================================================
    logger.info("")
    logger.info("=" * 80)
    logger.info("ANALYSIS 5: Date Gap Analysis (Missing Date Ranges)")
    logger.info("=" * 80)

    logger.info("\nChecking for gaps in timestamp_seconds (should be continuous 1-second intervals)...")

    # Sample check: look for gaps > 1 second
    gaps_df = (
        df.select([
            "timestamp_seconds",
            (pl.col("timestamp_seconds") - pl.col("timestamp_seconds").shift(1)).alias("gap_seconds")
        ])
        .filter(pl.col("gap_seconds") > 1)
        .collect()
    )

    if len(gaps_df) > 0:
        logger.info(f"  Found {len(gaps_df):,} gaps (timestamp jumps > 1 second)")
        logger.info("\n  Largest gaps:")
        largest_gaps = gaps_df.sort("gap_seconds", descending=True).head(10)

        for row in largest_gaps.iter_rows(named=True):
            gap_start_ts = row["timestamp_seconds"] - int(row["gap_seconds"])
            gap_start = datetime.fromtimestamp(gap_start_ts).strftime("%Y-%m-%d %H:%M:%S")
            gap_end = datetime.fromtimestamp(row["timestamp_seconds"]).strftime("%Y-%m-%d %H:%M:%S")
            gap_hours = row["gap_seconds"] / 3600

            logger.info(f"    {gap_start} → {gap_end} ({gap_hours:.1f} hours gap)")
    else:
        logger.info("  ✓ No gaps found - timestamps are continuous")

    logger.info("")
    logger.info("=" * 80)
    logger.info("INVESTIGATION COMPLETE")
    logger.info("=" * 80)


if __name__ == "__main__":
    investigate_nulls()
