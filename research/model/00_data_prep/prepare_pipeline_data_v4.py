#!/usr/bin/env python3
"""
Prepare V4 Data for Multi-Horizon Pipeline
===========================================

Prepares consolidated_features_v4.parquet for training by:
1. Adding residual column (outcome - Black-Scholes baseline)
2. Adding regime columns (temporal + volatility regimes)
3. Cleaning null values (drop first 70 hours of warmup)
4. Validating regime distribution

Output: consolidated_features_v4_pipeline_ready.parquet

Author: BT Research Team
Date: 2025-11-13
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import polars as pl

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent))
from regime_detection_v4 import detect_hierarchical_regime

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Paths
DATA_DIR = Path(__file__).parent.parent / "data"
RESULTS_DIR = Path(__file__).parent.parent / "results"

FEATURES_FILE = DATA_DIR / "consolidated_features_v4.parquet"
BASELINE_FILE = RESULTS_DIR / "production_backtest_results_v4.parquet"
OUTPUT_FILE = DATA_DIR / "consolidated_features_v4_pipeline_ready.parquet"


def load_and_join_data() -> pl.LazyFrame:
    """
    Load features and baseline, join them, and compute residual.

    Returns:
        LazyFrame with features + residual column
    """
    logger.info("Loading consolidated features...")
    df_features = pl.scan_parquet(FEATURES_FILE)

    logger.info("Loading baseline predictions...")
    df_baseline = pl.scan_parquet(BASELINE_FILE)

    # Select key columns from baseline
    # Note: baseline has 'timestamp', features have 'timestamp_seconds'
    baseline_cols = ["timestamp", "outcome", "prob_mid"]
    df_baseline_select = df_baseline.select(baseline_cols)

    # Convert features timestamp_seconds to timestamp for join
    df_features_with_ts = df_features.with_columns([pl.col("timestamp_seconds").alias("timestamp")])

    # Join on timestamp
    logger.info("Joining features with baseline...")
    df = df_features_with_ts.join(df_baseline_select, on="timestamp", how="left", suffix="_baseline")

    # Compute residual = outcome - prob_mid (Black-Scholes baseline)
    logger.info("Computing residual (outcome - Black-Scholes)...")
    df = df.with_columns([(pl.col("outcome") - pl.col("prob_mid")).alias("residual")])

    return df


def clean_nulls(df: pl.LazyFrame, warmup_hours: int = 70) -> pl.LazyFrame:
    """
    Remove warmup period with null values.

    Args:
        df: LazyFrame with data
        warmup_hours: Hours to drop from start (default: 70)

    Returns:
        LazyFrame with warmup period removed
    """
    logger.info(f"Removing first {warmup_hours} hours (warmup period)...")

    # Get min timestamp
    min_timestamp = df.select(pl.col("timestamp").min()).collect().item()

    # Calculate cutoff
    warmup_seconds = warmup_hours * 3600
    cutoff_timestamp = min_timestamp + warmup_seconds

    logger.info(f"Min timestamp: {min_timestamp}")
    logger.info(f"Cutoff timestamp: {cutoff_timestamp}")

    # Filter out warmup period
    df_clean = df.filter(pl.col("timestamp") >= cutoff_timestamp)

    return df_clean


def add_regime_columns(df: pl.DataFrame) -> pl.DataFrame:
    """
    Add regime classification columns.

    Args:
        df: DataFrame with features (must be collected, not lazy)

    Returns:
        DataFrame with regime columns added
    """
    logger.info("Adding regime columns...")

    # Check required columns exist
    required_cols = ["time_remaining", "rv_900s", "moneyness_distance", "timestamp"]
    missing_cols = [col for col in required_cols if col not in df.columns]

    if missing_cols:
        raise ValueError(f"Missing required columns for regime detection: {missing_cols}")

    # Convert timestamp to datetime for regime detection
    # regime_detection_v4.py expects datetime timestamps
    logger.info("Converting timestamp to datetime format...")
    df = df.with_columns([pl.from_epoch(pl.col("timestamp"), time_unit="s").alias("timestamp_dt")])

    # Temporarily rename timestamp_dt to timestamp for regime detection
    df_for_regime = df.drop("timestamp").rename({"timestamp_dt": "timestamp"})

    # Add hierarchical regimes (3 temporal × 4 volatility = 12 combined)
    df_with_regimes = detect_hierarchical_regime(
        df_for_regime, volatility_col="rv_900s", moneyness_col="moneyness_distance", hysteresis=0.1, atm_threshold=0.01
    )

    # Drop the datetime timestamp, keep original int timestamp
    df_final = df_with_regimes.drop("timestamp").with_columns([pl.col("timestamp_seconds").alias("timestamp")])

    return df_final


def validate_regimes(df: pl.DataFrame) -> None:
    """
    Validate regime distribution and sample counts.

    Args:
        df: DataFrame with regime columns
    """
    logger.info("\n" + "=" * 80)
    logger.info("REGIME DISTRIBUTION VALIDATION")
    logger.info("=" * 80)

    # Expected 12 regimes
    expected_regimes = [
        f"{t}_{v}"
        for t in ["near", "mid", "far"]
        for v in ["low_vol_atm", "low_vol_otm", "high_vol_atm", "high_vol_otm"]
    ]

    # Count samples per combined regime
    regime_counts = df.group_by("combined_regime").agg(pl.len().alias("count")).sort("combined_regime")

    logger.info(f"\nRegime distribution ({len(regime_counts)} regimes):")
    for row in regime_counts.iter_rows(named=True):
        regime = row["combined_regime"]
        count = row["count"]
        pct = 100 * count / len(df)
        logger.info(f"  {regime:20s}: {count:>10,} samples ({pct:>5.2f}%)")

    # Check for missing regimes
    actual_regimes = set(regime_counts["combined_regime"].to_list())
    missing_regimes = set(expected_regimes) - actual_regimes

    if missing_regimes:
        logger.warning(f"\n⚠ Missing regimes: {missing_regimes}")

    # Check for regimes below threshold
    min_threshold = 100_000
    low_sample_regimes = regime_counts.filter(pl.col("count") < min_threshold)

    if len(low_sample_regimes) > 0:
        logger.warning(f"\n⚠ Regimes below {min_threshold:,} sample threshold:")
        for row in low_sample_regimes.iter_rows(named=True):
            logger.warning(f"  {row['combined_regime']}: {row['count']:,} samples")

    # Summary statistics
    total_samples = len(df)
    logger.info(f"\nTotal samples: {total_samples:,}")
    logger.info(f"Regimes with >100K samples: {len(regime_counts.filter(pl.col('count') >= 100_000))}/12")
    logger.info("=" * 80 + "\n")


def check_null_rates(df: pl.DataFrame) -> None:
    """
    Check null rates in final dataset.

    Args:
        df: Final DataFrame
    """
    logger.info("\n" + "=" * 80)
    logger.info("NULL VALUE ANALYSIS")
    logger.info("=" * 80)

    total_rows = len(df)
    null_counts = []

    for col in df.columns:
        null_count = df[col].null_count()
        if null_count > 0:
            null_pct = 100 * null_count / total_rows
            null_counts.append((col, null_count, null_pct))

    # Sort by null percentage (descending)
    null_counts.sort(key=lambda x: x[2], reverse=True)

    if null_counts:
        logger.info("\nColumns with nulls (top 20):")
        for col, count, pct in null_counts[:20]:
            logger.info(f"  {col:40s}: {count:>10,} ({pct:>5.2f}%)")

        overall_null_rate = sum(c for _, c, _ in null_counts) / (total_rows * len(df.columns)) * 100
        logger.info(f"\nOverall null rate: {overall_null_rate:.2f}%")
    else:
        logger.info("\n✅ No null values found!")

    logger.info("=" * 80 + "\n")


def main() -> None:
    """Main execution function."""
    logger.info("=" * 80)
    logger.info("V4 PIPELINE DATA PREPARATION")
    logger.info("=" * 80)
    logger.info(f"Features file: {FEATURES_FILE}")
    logger.info(f"Baseline file: {BASELINE_FILE}")
    logger.info(f"Output file:   {OUTPUT_FILE}")
    logger.info("")

    # Step 1: Load and join data
    logger.info("STEP 1: Load and join data...")
    df_lazy = load_and_join_data()

    # Step 2: Clean nulls (drop warmup period)
    logger.info("\nSTEP 2: Clean null values...")
    df_clean = clean_nulls(df_lazy, warmup_hours=70)

    # Step 3: Collect to DataFrame (needed for regime detection)
    logger.info("\nSTEP 3: Collecting data (this may take a few minutes)...")
    df = df_clean.collect()
    logger.info(f"Collected {len(df):,} rows, {len(df.columns)} columns")

    # Step 4: Add regime columns
    logger.info("\nSTEP 4: Add regime columns...")
    df = add_regime_columns(df)
    logger.info(f"Added regime columns, now {len(df.columns)} columns")

    # Step 4b: Add date column (for temporal splits in walk-forward validation)
    logger.info("\nSTEP 4b: Add date column...")
    df = df.with_columns([pl.from_epoch(pl.col("timestamp"), time_unit="s").dt.date().alias("date")])
    logger.info(f"Added date column, now {len(df.columns)} columns")

    # Step 5: Validate regimes
    logger.info("\nSTEP 5: Validate regime distribution...")
    validate_regimes(df)

    # Step 6: Check null rates
    logger.info("\nSTEP 6: Check null rates...")
    check_null_rates(df)

    # Step 7: Write output
    logger.info("\nSTEP 7: Writing pipeline-ready dataset...")
    logger.info(f"Output file: {OUTPUT_FILE}")

    df.write_parquet(
        OUTPUT_FILE,
        compression="snappy",
        statistics=True,
    )

    # Verify output
    output_size_gb = OUTPUT_FILE.stat().st_size / (1024**3)
    logger.info(f"✅ File written: {output_size_gb:.2f} GB")
    logger.info(f"✅ Rows: {len(df):,}")
    logger.info(f"✅ Columns: {len(df.columns)}")

    logger.info("\n" + "=" * 80)
    logger.info("✅ PIPELINE DATA PREPARATION COMPLETE")
    logger.info("=" * 80)
    logger.info(f"\nReady for training: {OUTPUT_FILE}")
    logger.info("Next step: Run train_multi_horizon_v4.py")


if __name__ == "__main__":
    main()
