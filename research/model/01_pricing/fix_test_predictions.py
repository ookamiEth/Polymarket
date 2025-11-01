#!/usr/bin/env python3
"""
Fix test_predictions.parquet for visualization compatibility.

Fixes:
1. Sort by date for time-series plots
2. Add timestamp column for trading plots
3. Add volatility regime column for simulation plots
4. Handle inf/-inf values for heatmaps
"""

import logging
import sys

import numpy as np
import polars as pl

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

INPUT_FILE = "../results/test_predictions.parquet"
OUTPUT_FILE = "../results/test_predictions_fixed.parquet"


def fix_predictions() -> None:
    """Fix test predictions file."""
    logger.info("=" * 80)
    logger.info("FIXING TEST PREDICTIONS FILE")
    logger.info("=" * 80)

    # Load data
    logger.info(f"\n1. Loading {INPUT_FILE}...")
    df = pl.read_parquet(INPUT_FILE)
    logger.info(f"   Loaded {len(df):,} rows, {len(df.columns)} columns")

    # Check current state
    logger.info(f"\n2. Current state:")
    logger.info(f"   Has 'date' column: {'date' in df.columns}")
    logger.info(f"   Has 'timestamp' column: {'timestamp' in df.columns}")
    logger.info(f"   Has 'volatility' column: {'volatility' in df.columns}")
    if "date" in df.columns:
        logger.info(f"   Date dtype: {df['date'].dtype}")
        logger.info(f"   Date sample: {df['date'].head(3).to_list()}")

    # Fix 1: Sort by date
    logger.info(f"\n3. Sorting by date...")
    df = df.sort("date")
    logger.info(f"   ✓ Sorted")

    # Fix 2: Add timestamp column (duplicate of date)
    if "timestamp" not in df.columns:
        logger.info(f"\n4. Adding 'timestamp' column...")
        df = df.with_columns([pl.col("date").alias("timestamp")])
        logger.info(f"   ✓ Added 'timestamp' column")
    else:
        logger.info(f"\n4. 'timestamp' column already exists")

    # Fix 3: Add volatility regime column
    if "volatility" not in df.columns:
        logger.info(f"\n5. Adding 'volatility' regime column...")
        # Use rv_300s for regime classification
        # Low: <33rd percentile, Mid: 33-66th, High: >66th
        low_threshold = df["rv_300s"].quantile(0.33)
        high_threshold = df["rv_300s"].quantile(0.66)

        df = df.with_columns([
            pl.when(pl.col("rv_300s") < low_threshold)
            .then(pl.lit("low"))
            .when(pl.col("rv_300s") < high_threshold)
            .then(pl.lit("mid"))
            .otherwise(pl.lit("high"))
            .alias("volatility")
        ])
        logger.info(f"   ✓ Added 'volatility' column")
        logger.info(f"   Low threshold: {low_threshold:.6f}")
        logger.info(f"   High threshold: {high_threshold:.6f}")

        # Count distribution
        vol_counts = df.group_by("volatility").agg(pl.len().alias("count"))
        logger.info(f"   Distribution:\n{vol_counts}")
    else:
        logger.info(f"\n5. 'volatility' column already exists")

    # Fix 4: Handle inf/-inf values
    logger.info(f"\n6. Checking for inf/-inf values...")
    inf_cols = []
    for col in df.columns:
        if df[col].dtype in [pl.Float32, pl.Float64]:
            inf_count = df[col].is_infinite().sum()
            if inf_count > 0:
                inf_cols.append((col, inf_count))

    if inf_cols:
        logger.info(f"   Found inf values in {len(inf_cols)} columns:")
        for col, count in inf_cols[:10]:
            logger.info(f"     - {col}: {count} inf values")

        # Replace inf/-inf with None
        logger.info(f"   Replacing inf/-inf with None...")
        for col, _ in inf_cols:
            df = df.with_columns([
                pl.when(pl.col(col).is_infinite())
                .then(None)
                .otherwise(pl.col(col))
                .alias(col)
            ])
        logger.info(f"   ✓ Replaced inf values")
    else:
        logger.info(f"   ✓ No inf values found")

    # Save fixed file
    logger.info(f"\n7. Saving to {OUTPUT_FILE}...")
    df.write_parquet(OUTPUT_FILE, compression="snappy", statistics=True)

    file_size_mb = OUTPUT_FILE.stat().st_size / (1024 * 1024) if OUTPUT_FILE.exists() else 0
    logger.info(f"   ✓ Saved {file_size_mb:.1f} MB")

    # Summary
    logger.info("\n" + "=" * 80)
    logger.info("FIXES APPLIED:")
    logger.info("=" * 80)
    logger.info(f"✓ Sorted by date: {len(df):,} rows")
    logger.info(f"✓ Timestamp column: {'timestamp' in df.columns}")
    logger.info(f"✓ Volatility column: {'volatility' in df.columns}")
    logger.info(f"✓ Inf values handled: {len(inf_cols)} columns cleaned")
    logger.info("=" * 80)
    logger.info(f"\nOutput: {OUTPUT_FILE}")


if __name__ == "__main__":
    from pathlib import Path

    OUTPUT_FILE = Path(OUTPUT_FILE)
    fix_predictions()
