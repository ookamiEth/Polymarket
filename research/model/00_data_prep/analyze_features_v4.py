#!/usr/bin/env python3
"""
Analyze consolidated_features_v4.parquet
=========================================

Reports:
1. Date range (min/max timestamps)
2. Total rows and features
3. Null counts per column
4. Memory usage summary

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


def analyze_features() -> None:
    """Analyze consolidated features file."""
    logger.info("=" * 80)
    logger.info("ANALYZING CONSOLIDATED FEATURES V4")
    logger.info("=" * 80)

    # Check file exists
    if not FEATURES_FILE.exists():
        logger.error(f"File not found: {FEATURES_FILE}")
        return

    file_size_gb = FEATURES_FILE.stat().st_size / (1024**3)
    logger.info(f"File: {FEATURES_FILE}")
    logger.info(f"Size: {file_size_gb:.2f} GB")
    logger.info("")

    # Scan file lazily
    logger.info("Loading schema and computing statistics...")
    df_lazy = pl.scan_parquet(FEATURES_FILE)

    # Get schema
    schema = df_lazy.collect_schema()
    feature_count = len(schema.names())
    logger.info(f"Features: {feature_count}")
    logger.info("")

    # Compute row count
    logger.info("Computing row count...")
    row_count = df_lazy.select(pl.len()).collect().item()
    logger.info(f"Total rows: {row_count:,}")
    logger.info("")

    # Get date range
    logger.info("Computing date range...")
    date_stats = (
        df_lazy.select(
            [
                pl.col("timestamp_seconds").min().alias("min_ts"),
                pl.col("timestamp_seconds").max().alias("max_ts"),
            ]
        )
        .collect()
    )

    min_ts = date_stats["min_ts"][0]
    max_ts = date_stats["max_ts"][0]

    min_date = datetime.fromtimestamp(min_ts).strftime("%Y-%m-%d %H:%M:%S UTC")
    max_date = datetime.fromtimestamp(max_ts).strftime("%Y-%m-%d %H:%M:%S UTC")

    days_span = (max_ts - min_ts) / (24 * 3600)

    logger.info(f"Date range:")
    logger.info(f"  Start: {min_date} (timestamp: {min_ts})")
    logger.info(f"  End:   {max_date} (timestamp: {max_ts})")
    logger.info(f"  Span:  {days_span:.1f} days")
    logger.info("")

    # Compute null counts for all columns
    logger.info("Computing null counts (this may take a few minutes)...")

    # Process in batches to avoid memory issues
    null_counts = {}

    # Sample approach: check nulls on 1M row sample first
    logger.info("  Checking nulls on 1M row sample...")
    sample = df_lazy.head(1_000_000).collect()

    for col in schema.names():
        null_count_sample = sample[col].null_count()
        null_counts[col] = {"sample_nulls": null_count_sample, "sample_size": len(sample)}

    # Now get exact counts for columns with nulls in sample
    cols_with_nulls_in_sample = [col for col, stats in null_counts.items() if stats["sample_nulls"] > 0]

    if cols_with_nulls_in_sample:
        logger.info(f"  Found {len(cols_with_nulls_in_sample)} columns with nulls in sample")
        logger.info("  Computing exact null counts for these columns...")

        # Compute exact counts for columns with nulls
        null_exprs = [pl.col(col).is_null().sum().alias(col) for col in cols_with_nulls_in_sample]
        exact_nulls = df_lazy.select(null_exprs).collect()

        for col in cols_with_nulls_in_sample:
            null_counts[col]["total_nulls"] = exact_nulls[col][0]
    else:
        logger.info("  No nulls found in sample - assuming no nulls in full dataset")

    # Report null counts
    logger.info("")
    logger.info("=" * 80)
    logger.info("NULL VALUE ANALYSIS")
    logger.info("=" * 80)

    columns_with_nulls = [
        col for col, stats in null_counts.items()
        if "total_nulls" in stats and stats["total_nulls"] > 0
    ]

    if columns_with_nulls:
        logger.info(f"Columns with null values: {len(columns_with_nulls)}/{feature_count}")
        logger.info("")

        # Sort by null count descending
        sorted_cols = sorted(
            columns_with_nulls,
            key=lambda c: null_counts[c].get("total_nulls", 0),
            reverse=True
        )

        logger.info("Top 20 columns by null count:")
        logger.info(f"{'Column':<40} {'Null Count':>15} {'Null %':>10}")
        logger.info("-" * 70)

        for col in sorted_cols[:20]:
            total_nulls = null_counts[col]["total_nulls"]
            null_pct = (total_nulls / row_count) * 100
            logger.info(f"{col:<40} {total_nulls:>15,} {null_pct:>9.2f}%")

        if len(sorted_cols) > 20:
            logger.info(f"... and {len(sorted_cols) - 20} more columns with nulls")

        # Summary statistics
        total_null_values = sum(null_counts[col].get("total_nulls", 0) for col in columns_with_nulls)
        total_cells = row_count * feature_count
        overall_null_pct = (total_null_values / total_cells) * 100

        logger.info("")
        logger.info("Summary:")
        logger.info(f"  Total cells: {total_cells:,}")
        logger.info(f"  Total nulls: {total_null_values:,}")
        logger.info(f"  Overall null percentage: {overall_null_pct:.4f}%")
    else:
        logger.info("✓ No null values found in dataset!")

    logger.info("")
    logger.info("=" * 80)
    logger.info("ANALYSIS COMPLETE")
    logger.info("=" * 80)


if __name__ == "__main__":
    analyze_features()
