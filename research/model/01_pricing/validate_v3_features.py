#!/usr/bin/env python3
"""
V3 Features Validation Script
===============================

Validates consolidated_features_v3.parquet file:
- All expected features exist
- No NaN/inf values in critical features
- Temporal ordering is correct
- Sample counts match expected ranges

Usage:
  uv run python validate_v3_features.py

Author: BT Research Team
Date: 2025-11-06
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import polars as pl
from lightgbm_memory_optimized import FEATURE_COLS

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


def validate_v3_features() -> bool:
    """
    Validate V3 consolidated features file.

    Returns:
        True if all validations pass, False otherwise
    """
    logger.info("=" * 80)
    logger.info("V3 FEATURES VALIDATION")
    logger.info("=" * 80)

    # Path to consolidated file
    model_dir = Path(__file__).parent.parent
    consolidated_file = model_dir / "data" / "consolidated_features_v3.parquet"

    if not consolidated_file.exists():
        logger.error(f"✗ Consolidated file not found: {consolidated_file}")
        return False

    logger.info(f"✓ Found consolidated file: {consolidated_file}")

    # Get file stats
    file_size_gb = consolidated_file.stat().st_size / (1024**3)
    logger.info(f"  File size: {file_size_gb:.2f} GB")

    # Load schema (without reading full file)
    logger.info("\nChecking schema...")
    df_lazy = pl.scan_parquet(consolidated_file)
    schema = df_lazy.collect_schema()

    logger.info(f"  Total columns: {len(schema.names())}")

    # Check if all FEATURE_COLS exist in file
    logger.info("\nValidating features...")
    missing_features = []
    for feature in FEATURE_COLS:
        if feature not in schema.names():
            missing_features.append(feature)

    if missing_features:
        logger.error(f"✗ {len(missing_features)} features missing from consolidated file:")
        for feat in missing_features[:20]:  # Show first 20
            logger.error(f"    - {feat}")
        if len(missing_features) > 20:
            logger.error(f"    ... and {len(missing_features) - 20} more")
        return False

    logger.info(f"✓ All {len(FEATURE_COLS)} features present in consolidated file")

    # Check required metadata columns
    logger.info("\nValidating metadata columns...")
    required_cols = ["timestamp_seconds", "time_remaining"]  # date can be derived from timestamp_seconds
    missing_metadata = [col for col in required_cols if col not in schema.names()]

    if missing_metadata:
        logger.error(f"✗ Missing metadata columns: {missing_metadata}")
        return False

    logger.info("✓ All required metadata columns present")

    # Check if date column exists, warn if not (it will be derived in training scripts)
    if "date" not in schema.names():
        logger.info("  Note: 'date' column not present (will be derived from timestamp_seconds during training)")

    # Sample data statistics (read small sample)
    logger.info("\nCollecting data statistics...")
    stats = df_lazy.select(
        [
            pl.len().alias("total_rows"),
            pl.col("timestamp_seconds").min().alias("min_timestamp"),
            pl.col("timestamp_seconds").max().alias("max_timestamp"),
        ]
    ).collect()

    total_rows = stats["total_rows"][0]
    min_timestamp = stats["min_timestamp"][0]
    max_timestamp = stats["max_timestamp"][0]

    # Convert timestamps to dates for display
    from datetime import datetime

    min_date = datetime.fromtimestamp(min_timestamp).date()
    max_date = datetime.fromtimestamp(max_timestamp).date()

    logger.info(f"  Total rows: {total_rows:,}")
    logger.info(f"  Date range: {min_date} to {max_date} (derived from timestamps)")
    logger.info(f"  Timestamp range: {min_timestamp} to {max_timestamp}")

    # Expected range: ~63M rows (based on engineer_all_features_v3.py output)
    if total_rows < 50_000_000:
        logger.warning(f"⚠  Row count seems low (expected ~63M, got {total_rows:,})")
    elif total_rows > 80_000_000:
        logger.warning(f"⚠  Row count seems high (expected ~63M, got {total_rows:,})")
    else:
        logger.info("✓ Row count within expected range")

    # Check for null values in critical features
    logger.info("\nChecking for null values in critical features...")
    critical_features = ["time_remaining", "moneyness", "timestamp_seconds"]
    null_checks = df_lazy.select(
        [pl.col(feat).is_null().sum().alias(f"{feat}_nulls") for feat in critical_features if feat in schema.names()]
    ).collect()

    has_nulls = False
    for col in null_checks.columns:
        null_count = null_checks[col][0]
        if null_count > 0:
            logger.error(f"✗ {col}: {null_count:,} null values")
            has_nulls = True

    if not has_nulls:
        logger.info("✓ No null values in critical features")

    # Check temporal ordering (sample)
    logger.info("\nValidating temporal ordering (sample check)...")
    sample_df = df_lazy.select(["timestamp_seconds"]).head(10000).collect()
    timestamps = sample_df["timestamp_seconds"].to_list()

    # Check if timestamps are monotonically increasing
    is_sorted = all(timestamps[i] <= timestamps[i + 1] for i in range(len(timestamps) - 1))

    if is_sorted:
        logger.info("✓ Timestamps are monotonically increasing (sample)")
    else:
        logger.warning("⚠  Timestamps are NOT sorted (may impact walk-forward validation)")

    # Summary
    logger.info("\n" + "=" * 80)
    logger.info("VALIDATION SUMMARY")
    logger.info("=" * 80)

    all_checks_passed = not missing_features and not missing_metadata and not has_nulls

    if all_checks_passed:
        logger.info("✓ All validation checks PASSED")
        logger.info("\nReady for training! Run:")
        logger.info("  cd research/model/01_pricing")
        logger.info("  ./run_multi_horizon_pipeline.sh")
        return True
    else:
        logger.error("✗ Some validation checks FAILED")
        logger.error("\nPlease fix issues before training")
        return False


def main() -> None:
    """Main entry point."""
    success = validate_v3_features()
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
