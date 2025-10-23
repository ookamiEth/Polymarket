#!/usr/bin/env python3

"""
Compare implied volatility calculations using constant vs. daily risk-free rates.

This script performs a comprehensive comparison of IV calculations on the same
options quote dataset using:
1. Constant risk-free rate (4.12%)
2. Daily risk-free rates from blended lending rates (AAVE + USDT)

IMPROVED VERSION with staged processing to handle 204M+ rows without OOM:
- Stage 1: Filter and checkpoint both datasets separately
- Stage 2: Partitioned join by month (30x smaller joins)
- Stage 3: Apply transformations and enrichments
- Stage 4: Generate statistics from checkpointed data

Memory-efficient design:
- Staged processing with intermediate checkpoints
- Partitioned joins to reduce memory pressure
- Checkpoint/recovery system for resilience
- Memory monitoring and proactive cleanup
- Never loads full 204M row dataset into memory
"""

import argparse
import gc
import json
import logging
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

import polars as pl
import psutil

# Constants
DEFAULT_CONSTANT_IV_FILE = "research/tardis/data/archive/btc_options_atm_shortdated_with_iv_constant_rates_2023_2025.parquet"
DEFAULT_DAILY_IV_FILE = "research/tardis/data/consolidated/btc_options_atm_shortdated_with_iv_2023_2025.parquet"
DEFAULT_RATES_FILE = "research/risk_free_rate/data/blended_lending_rates_2023_2025.parquet"
DEFAULT_OUTPUT_DIR = "research/tardis/analysis/output"

# Analysis thresholds
MATERIALITY_THRESHOLDS_ABS = [0.01, 0.05, 0.10]  # Absolute IV difference (vol points)
MATERIALITY_THRESHOLDS_REL = [0.05, 0.10, 0.20]  # Relative difference (%)

# Moneyness bins (using large number instead of inf for Polars compatibility)
# Note: For cut(), number of labels must equal number of bins - 1
MONEYNESS_BINS = [0.0, 0.9, 1.1, 10.0]  # 10 is effectively infinity for moneyness
MONEYNESS_LABELS = ["OTM (<0.9)", "ATM (0.9-1.1)", "ITM (>1.1)"]  # 3 labels for 4 bins

# Time to expiry bins (days) - using 365*2 for max instead of inf
TTL_BINS = [0.0, 7.0, 30.0, 90.0, 730.0]  # 730 days = 2 years max
TTL_LABELS = ["<7d", "7-30d", "30-90d", ">90d"]  # 4 labels for 5 bins

# Memory and processing limits
MAX_MEMORY_GB = 5.0  # Maximum memory usage before cleanup
CHUNK_SIZE_ROWS = 10_000_000  # Process in 10M row chunks for joins
CHECKPOINT_FILE = "research/tardis/analysis/output/.checkpoint.json"

logger = logging.getLogger(__name__)


def setup_logging(verbose: bool = False) -> None:
    """Configure logging with custom format."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )


def get_memory_usage() -> float:
    """Get current memory usage in GB."""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / (1024**3)


def check_memory(operation: str = "") -> None:
    """Check memory usage and log warning if high."""
    memory_gb = get_memory_usage()
    logger.debug(f"Memory usage {operation}: {memory_gb:.2f} GB")

    if memory_gb > MAX_MEMORY_GB:
        logger.warning(f"⚠️ High memory usage: {memory_gb:.2f} GB (limit: {MAX_MEMORY_GB:.1f} GB)")
        logger.info("Triggering garbage collection...")
        gc.collect()
        new_memory = get_memory_usage()
        logger.info(f"Memory after GC: {new_memory:.2f} GB (freed {memory_gb - new_memory:.2f} GB)")


def save_checkpoint(checkpoint_data: dict) -> None:
    """Save checkpoint data to disk."""
    checkpoint_path = Path(CHECKPOINT_FILE)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    with open(checkpoint_path, "w") as f:
        json.dump(checkpoint_data, f, indent=2, default=str)

    logger.debug(f"Checkpoint saved to {checkpoint_path}")


def load_checkpoint() -> Optional[dict]:
    """Load checkpoint data if exists."""
    checkpoint_path = Path(CHECKPOINT_FILE)

    if not checkpoint_path.exists():
        return None

    try:
        with open(checkpoint_path) as f:
            checkpoint = json.load(f)
        logger.info(f"Checkpoint loaded from {checkpoint_path}")
        return checkpoint
    except Exception as e:
        logger.warning(f"Failed to load checkpoint: {e}")
        return None


def clean_checkpoint() -> None:
    """Remove checkpoint file."""
    checkpoint_path = Path(CHECKPOINT_FILE)
    if checkpoint_path.exists():
        checkpoint_path.unlink()
        logger.info("Checkpoint file removed (processing complete)")


def clean_temp_files(temp_dir: Path) -> None:
    """Clean up temporary files from previous runs."""
    if temp_dir.exists():
        for file in temp_dir.glob("*.parquet"):
            file.unlink()
            logger.debug(f"Removed temp file: {file}")
    else:
        temp_dir.mkdir(parents=True, exist_ok=True)


def stage1_filter_and_checkpoint(
    constant_file: str,
    daily_file: str,
    temp_dir: Path,
    checkpoint: Optional[dict] = None,
) -> tuple[Path, Path, dict]:
    """
    Stage 1: Filter successful IVs and checkpoint to disk.

    This separates the filtering from the join to reduce memory pressure.

    Args:
        constant_file: Path to IV file with constant rates
        daily_file: Path to IV file with daily rates
        temp_dir: Directory for temporary files
        checkpoint: Optional checkpoint to resume from

    Returns:
        Tuple of (constant filtered path, daily filtered path, stats dict)
    """
    logger.info("=" * 80)
    logger.info("STAGE 1: FILTER AND CHECKPOINT")
    logger.info("=" * 80)

    stats = {}

    # Output paths
    constant_filtered = temp_dir / "constant_filtered.parquet"
    daily_filtered = temp_dir / "daily_filtered.parquet"

    # Check if already done
    if checkpoint and checkpoint.get("stage1_complete"):
        logger.info("Stage 1 already complete (from checkpoint)")
        return constant_filtered, daily_filtered, checkpoint.get("stage1_stats", {})

    # Process constant rate file
    if not constant_filtered.exists():
        logger.info(f"Filtering constant rate IVs from {constant_file}...")
        check_memory("before constant filter")

        df_constant = pl.scan_parquet(constant_file)
        n_constant_total = df_constant.select(pl.len()).collect().item()
        stats["constant_total"] = n_constant_total

        # Filter and write
        df_constant_success = df_constant.filter(pl.col("iv_calc_status") == "success")

        # Select required columns only
        constant_cols = [
            "timestamp_seconds",
            "symbol",
            "exchange",
            "type",
            "strike_price",
            "expiry_timestamp",
            "spot_price",
            "moneyness",
            "time_to_expiry_days",
            "implied_vol_bid",
            "implied_vol_ask",
        ]

        df_constant_success.select(constant_cols).sink_parquet(
            str(constant_filtered), compression="snappy", statistics=True
        )

        # Count rows in output
        n_constant_success = pl.scan_parquet(constant_filtered).select(pl.len()).collect().item()
        stats["constant_success"] = n_constant_success

        logger.info(
            f"  ✅ Filtered {n_constant_success:,}/{n_constant_total:,} rows "
            f"({n_constant_success / n_constant_total * 100:.1f}% success rate)"
        )
        check_memory("after constant filter")

    # Process daily rate file
    if not daily_filtered.exists():
        logger.info(f"Filtering daily rate IVs from {daily_file}...")
        check_memory("before daily filter")

        df_daily = pl.scan_parquet(daily_file)
        n_daily_total = df_daily.select(pl.len()).collect().item()
        stats["daily_total"] = n_daily_total

        # Filter and write
        df_daily_success = df_daily.filter(pl.col("iv_calc_status") == "success")

        # Select required columns only (fewer than constant)
        daily_cols = [
            "timestamp_seconds",
            "symbol",
            "exchange",
            "type",
            "strike_price",
            "expiry_timestamp",
            "implied_vol_bid",
            "implied_vol_ask",
        ]

        df_daily_success.select(daily_cols).sink_parquet(str(daily_filtered), compression="snappy", statistics=True)

        # Count rows in output
        n_daily_success = pl.scan_parquet(daily_filtered).select(pl.len()).collect().item()
        stats["daily_success"] = n_daily_success

        logger.info(
            f"  ✅ Filtered {n_daily_success:,}/{n_daily_total:,} rows "
            f"({n_daily_success / n_daily_total * 100:.1f}% success rate)"
        )
        check_memory("after daily filter")

    # Update checkpoint
    checkpoint_data = {"stage1_complete": True, "stage1_stats": stats}
    save_checkpoint(checkpoint_data)

    logger.info("✅ Stage 1 complete: Filtered datasets checkpointed to disk")

    return constant_filtered, daily_filtered, stats


def stage2_partitioned_join(
    constant_filtered: Path,
    daily_filtered: Path,
    temp_dir: Path,
    checkpoint: Optional[dict] = None,
) -> Path:
    """
    Stage 2: Perform partitioned join by month to reduce memory usage.

    Instead of joining 198M x 198M rows at once, we partition by date and
    join smaller chunks.

    Args:
        constant_filtered: Path to filtered constant IV file
        daily_filtered: Path to filtered daily IV file
        temp_dir: Directory for temporary files
        checkpoint: Optional checkpoint to resume from

    Returns:
        Path to joined comparison file
    """
    logger.info("=" * 80)
    logger.info("STAGE 2: PARTITIONED JOIN")
    logger.info("=" * 80)

    output_file = temp_dir / "comparison_joined.parquet"

    # Check if already done
    if checkpoint and checkpoint.get("stage2_complete"):
        logger.info("Stage 2 already complete (from checkpoint)")
        return output_file

    # Get date range from data
    logger.info("Analyzing date range...")
    df_constant_lazy = pl.scan_parquet(constant_filtered)

    # Add date column for partitioning
    df_constant_lazy = df_constant_lazy.with_columns(
        [pl.from_epoch("timestamp_seconds", time_unit="s").cast(pl.Date).alias("date")]
    )

    date_range = (
        df_constant_lazy.select([pl.col("date").min().alias("min_date"), pl.col("date").max().alias("max_date")])
        .collect()
        .row(0)
    )
    start_date, end_date = date_range
    logger.info(f"  Date range: {start_date} to {end_date}")

    # Generate monthly partitions
    partitions = []
    current = start_date
    while current <= end_date:
        month_start = current.replace(day=1)
        # Calculate month end
        if month_start.month == 12:
            month_end = month_start.replace(year=month_start.year + 1, month=1, day=1)
        else:
            month_end = month_start.replace(month=month_start.month + 1, day=1)
        partitions.append((month_start, month_end))
        current = month_end

    logger.info(f"  Processing {len(partitions)} monthly partitions")

    # Process each partition
    partition_files = []
    completed_partitions = checkpoint.get("completed_partitions", []) if checkpoint else []

    for i, (month_start, month_end) in enumerate(partitions, 1):
        partition_label = f"{month_start.strftime('%Y-%m')}"
        partition_file = temp_dir / f"partition_{partition_label}.parquet"

        # Skip if already processed
        if partition_label in completed_partitions:
            logger.info(f"  Partition {i}/{len(partitions)}: {partition_label} (already complete)")
            partition_files.append(partition_file)
            continue

        logger.info(f"  Partition {i}/{len(partitions)}: {partition_label}")
        check_memory(f"before partition {partition_label}")

        # Load and filter constant data for this partition
        df_const_partition = (
            pl.scan_parquet(constant_filtered)
            .with_columns([pl.from_epoch("timestamp_seconds", time_unit="s").cast(pl.Date).alias("date")])
            .filter((pl.col("date") >= month_start) & (pl.col("date") < month_end))
            .drop("date")  # Drop date column after filtering
        )

        # Load and filter daily data for this partition
        df_daily_partition = (
            pl.scan_parquet(daily_filtered)
            .with_columns([pl.from_epoch("timestamp_seconds", time_unit="s").cast(pl.Date).alias("date")])
            .filter((pl.col("date") >= month_start) & (pl.col("date") < month_end))
            .drop("date")  # Drop date column after filtering
        )

        # Define join keys
        join_keys = [
            "timestamp_seconds",
            "symbol",
            "exchange",
            "type",
            "strike_price",
            "expiry_timestamp",
        ]

        # Perform join for this partition
        df_joined = df_const_partition.join(
            df_daily_partition,
            on=join_keys,
            how="inner",
            suffix="_daily",
        )

        # Write partition result
        df_joined.sink_parquet(str(partition_file), compression="snappy", statistics=True)

        # Track completion
        completed_partitions.append(partition_label)
        partition_files.append(partition_file)

        # Update checkpoint
        checkpoint_data = {
            "stage1_complete": True,
            "stage1_stats": checkpoint.get("stage1_stats", {}) if checkpoint else {},
            "completed_partitions": completed_partitions,
        }
        save_checkpoint(checkpoint_data)

        check_memory(f"after partition {partition_label}")

    # Combine all partitions into single file
    logger.info("\nCombining all partitions...")
    df_all = pl.scan_parquet(partition_files)
    df_all.sink_parquet(str(output_file), compression="snappy", statistics=True)

    # Clean up partition files
    for pfile in partition_files:
        pfile.unlink()

    # Update checkpoint
    checkpoint_data = {
        "stage1_complete": True,
        "stage2_complete": True,
        "stage1_stats": checkpoint.get("stage1_stats", {}) if checkpoint else {},
    }
    save_checkpoint(checkpoint_data)

    logger.info("✅ Stage 2 complete: Partitioned join written to disk")

    return output_file


def stage3_transform_and_enrich(
    joined_file: Path,
    rates_file: str,
    output_dir: Path,
    checkpoint: Optional[dict] = None,
) -> Path:
    """
    Stage 3: Apply transformations and enrichments to joined data.

    This stage calculates differences, adds bins, filters invalid data,
    and joins with risk-free rates.

    Args:
        joined_file: Path to joined comparison file
        rates_file: Path to risk-free rates file
        output_dir: Output directory
        checkpoint: Optional checkpoint to resume from

    Returns:
        Path to final comparison file
    """
    logger.info("=" * 80)
    logger.info("STAGE 3: TRANSFORM AND ENRICH")
    logger.info("=" * 80)

    output_file = output_dir / "comparison_final.parquet"

    # Check if already done
    if checkpoint and checkpoint.get("stage3_complete"):
        logger.info("Stage 3 already complete (from checkpoint)")
        return output_file

    # Load risk-free rates (small file)
    logger.info(f"Loading risk-free rates from {rates_file}...")
    df_rates = pl.read_parquet(rates_file).select(
        [
            pl.col("date"),
            pl.col("blended_supply_apr_ma7").alias("risk_free_rate_pct"),
        ]
    )
    logger.info(f"  Loaded {len(df_rates):,} days")

    # Load joined data lazily
    logger.info("Loading joined comparison data...")
    df_comparison = pl.scan_parquet(joined_file)

    # Apply all transformations in one chain
    logger.info("Applying transformations...")

    df_comparison = df_comparison.with_columns(
        [
            # Calculate all differences at once
            # Absolute differences (vol points)
            (pl.col("implied_vol_bid_daily") - pl.col("implied_vol_bid")).alias("iv_bid_diff_abs"),
            (pl.col("implied_vol_ask_daily") - pl.col("implied_vol_ask")).alias("iv_ask_diff_abs"),
            # Relative differences (%)
            ((pl.col("implied_vol_bid_daily") - pl.col("implied_vol_bid")) / pl.col("implied_vol_bid") * 100).alias(
                "iv_bid_diff_rel"
            ),
            ((pl.col("implied_vol_ask_daily") - pl.col("implied_vol_ask")) / pl.col("implied_vol_ask") * 100).alias(
                "iv_ask_diff_rel"
            ),
            # Mid IVs
            ((pl.col("implied_vol_bid") + pl.col("implied_vol_ask")) / 2).alias("iv_mid_constant"),
            ((pl.col("implied_vol_bid_daily") + pl.col("implied_vol_ask_daily")) / 2).alias("iv_mid_daily"),
            # Date for joining with rates
            pl.from_epoch("timestamp_seconds", time_unit="s").cast(pl.Date).alias("date"),
        ]
    )

    # Calculate mid differences (separate step for clarity)
    df_comparison = df_comparison.with_columns(
        [
            (pl.col("iv_mid_daily") - pl.col("iv_mid_constant")).alias("iv_mid_diff_abs"),
            ((pl.col("iv_mid_daily") - pl.col("iv_mid_constant")) / pl.col("iv_mid_constant") * 100).alias(
                "iv_mid_diff_rel"
            ),
        ]
    )

    # Add categorical bins using when/then instead of cut() to avoid issues
    logger.info("Adding categorical bins...")
    df_comparison = df_comparison.with_columns(
        [
            # Moneyness binning
            pl.when(pl.col("moneyness") < 0.9)
            .then(pl.lit("OTM (<0.9)"))
            .when(pl.col("moneyness") < 1.1)
            .then(pl.lit("ATM (0.9-1.1)"))
            .otherwise(pl.lit("ITM (>1.1)"))
            .alias("moneyness_bin"),
            # Time to expiry binning
            pl.when(pl.col("time_to_expiry_days") < 7)
            .then(pl.lit("<7d"))
            .when(pl.col("time_to_expiry_days") < 30)
            .then(pl.lit("7-30d"))
            .when(pl.col("time_to_expiry_days") < 90)
            .then(pl.lit("30-90d"))
            .otherwise(pl.lit(">90d"))
            .alias("ttl_bin"),
        ]
    )

    # Filter invalid data
    logger.info("Filtering invalid data...")
    df_comparison = df_comparison.filter(
        # Filter nulls
        pl.col("iv_mid_constant").is_not_null()
        & pl.col("iv_mid_daily").is_not_null()
        # Filter infinities
        & pl.col("iv_mid_constant").is_finite()
        & pl.col("iv_mid_daily").is_finite()
        & pl.col("iv_mid_diff_rel").is_finite()
        & pl.col("iv_bid_diff_rel").is_finite()
        & pl.col("iv_ask_diff_rel").is_finite()
        # Filter extreme outliers
        & (pl.col("iv_mid_constant") > 0)
        & (pl.col("iv_mid_constant") < 20)
        & (pl.col("iv_mid_daily") > 0)
        & (pl.col("iv_mid_daily") < 20)
    )

    # Join with risk-free rates
    logger.info("Joining with risk-free rates...")
    df_comparison = df_comparison.join(pl.LazyFrame(df_rates), on="date", how="left")

    # Write final comparison file
    logger.info("Writing final comparison data...")
    check_memory("before final write")

    df_comparison.sink_parquet(str(output_file), compression="snappy", statistics=True)

    # Report file size
    file_size_mb = output_file.stat().st_size / (1024**2)
    logger.info(f"  ✅ Final comparison written: {file_size_mb:.1f} MB")

    # Update checkpoint
    checkpoint_data = {
        "stage1_complete": True,
        "stage2_complete": True,
        "stage3_complete": True,
        "stage1_stats": checkpoint.get("stage1_stats", {}) if checkpoint else {},
    }
    save_checkpoint(checkpoint_data)

    check_memory("after final write")
    logger.info("✅ Stage 3 complete: Transformations applied and data enriched")

    return output_file


# Note: write_comparison_to_disk_streaming removed - replaced with staged processing


def generate_summary_statistics(df_lazy: pl.LazyFrame) -> dict[str, pl.DataFrame]:
    """
    Generate comprehensive summary statistics using lazy aggregations.

    CRITICAL: Only collects small aggregated results, never the full dataset.

    Args:
        df_lazy: Lazy comparison DataFrame

    Returns:
        Dictionary of summary DataFrames (all small, collected)
    """
    logger.info("\n" + "=" * 80)
    logger.info("SUMMARY STATISTICS (LAZY AGGREGATIONS)")
    logger.info("=" * 80)

    summaries = {}

    # Overall statistics - single aggregation (optimized: 12 quantiles → 6)
    logger.info("\n1. Overall Statistics")
    start_time = time.time()
    overall_stats = df_lazy.select(
        [
            pl.len().alias("n_observations"),
            # Constant IVs
            pl.col("iv_mid_constant").mean().alias("iv_constant_mean"),
            pl.col("iv_mid_constant").median().alias("iv_constant_median"),
            pl.col("iv_mid_constant").std().alias("iv_constant_std"),
            # Daily IVs
            pl.col("iv_mid_daily").mean().alias("iv_daily_mean"),
            pl.col("iv_mid_daily").median().alias("iv_daily_median"),
            pl.col("iv_mid_daily").std().alias("iv_daily_std"),
            # Absolute differences
            pl.col("iv_mid_diff_abs").mean().alias("diff_abs_mean"),
            pl.col("iv_mid_diff_abs").median().alias("diff_abs_median"),
            pl.col("iv_mid_diff_abs").std().alias("diff_abs_std"),
            pl.col("iv_mid_diff_abs").abs().mean().alias("diff_abs_mean_abs"),
            # Quantiles: p05, p50, p95 only (reduced from p01, p05, p25, p75, p95, p99)
            pl.col("iv_mid_diff_abs").quantile(0.05).alias("diff_abs_p05"),
            pl.col("iv_mid_diff_abs").quantile(0.50).alias("diff_abs_p50"),
            pl.col("iv_mid_diff_abs").quantile(0.95).alias("diff_abs_p95"),
            # Relative differences
            pl.col("iv_mid_diff_rel").mean().alias("diff_rel_mean"),
            pl.col("iv_mid_diff_rel").median().alias("diff_rel_median"),
            pl.col("iv_mid_diff_rel").std().alias("diff_rel_std"),
            pl.col("iv_mid_diff_rel").abs().mean().alias("diff_rel_mean_abs"),
            # Quantiles: p05, p50, p95 only (reduced from p01, p05, p25, p75, p95, p99)
            pl.col("iv_mid_diff_rel").quantile(0.05).alias("diff_rel_p05"),
            pl.col("iv_mid_diff_rel").quantile(0.50).alias("diff_rel_p50"),
            pl.col("iv_mid_diff_rel").quantile(0.95).alias("diff_rel_p95"),
            # Correlation
            pl.corr("iv_mid_constant", "iv_mid_daily").alias("correlation"),
        ]
    ).collect()  # Only collects 1 row!
    logger.info(f"  ✅ Completed in {time.time() - start_time:.1f}s")

    summaries["overall"] = overall_stats
    logger.info(f"\n{overall_stats}")

    # Materiality analysis - OPTIMIZED: Single aggregation (6 scans → 1 scan)
    logger.info("\n2. Materiality Analysis")
    start_time = time.time()

    # Get total count once
    total_count = overall_stats["n_observations"][0]

    # Single aggregation for ALL thresholds (6x faster than separate scans)
    materiality_agg = df_lazy.select(
        [
            pl.len().alias("total_count"),
            # Absolute thresholds (3 conditions)
            (pl.col("iv_mid_diff_abs").abs() > 0.01).sum().alias("abs_gt_001"),
            (pl.col("iv_mid_diff_abs").abs() > 0.05).sum().alias("abs_gt_005"),
            (pl.col("iv_mid_diff_abs").abs() > 0.10).sum().alias("abs_gt_010"),
            # Relative thresholds (3 conditions)
            (pl.col("iv_mid_diff_rel").abs() > 5.0).sum().alias("rel_gt_5pct"),
            (pl.col("iv_mid_diff_rel").abs() > 10.0).sum().alias("rel_gt_10pct"),
            (pl.col("iv_mid_diff_rel").abs() > 20.0).sum().alias("rel_gt_20pct"),
        ]
    ).collect()

    # Build results from single aggregation
    materiality_stats = [
        {
            "threshold_type": "absolute",
            "threshold_value": 0.01,
            "n_above_threshold": materiality_agg["abs_gt_001"][0],
            "pct_above_threshold": materiality_agg["abs_gt_001"][0] / total_count * 100,
        },
        {
            "threshold_type": "absolute",
            "threshold_value": 0.05,
            "n_above_threshold": materiality_agg["abs_gt_005"][0],
            "pct_above_threshold": materiality_agg["abs_gt_005"][0] / total_count * 100,
        },
        {
            "threshold_type": "absolute",
            "threshold_value": 0.10,
            "n_above_threshold": materiality_agg["abs_gt_010"][0],
            "pct_above_threshold": materiality_agg["abs_gt_010"][0] / total_count * 100,
        },
        {
            "threshold_type": "relative",
            "threshold_value": 0.05,
            "n_above_threshold": materiality_agg["rel_gt_5pct"][0],
            "pct_above_threshold": materiality_agg["rel_gt_5pct"][0] / total_count * 100,
        },
        {
            "threshold_type": "relative",
            "threshold_value": 0.10,
            "n_above_threshold": materiality_agg["rel_gt_10pct"][0],
            "pct_above_threshold": materiality_agg["rel_gt_10pct"][0] / total_count * 100,
        },
        {
            "threshold_type": "relative",
            "threshold_value": 0.20,
            "n_above_threshold": materiality_agg["rel_gt_20pct"][0],
            "pct_above_threshold": materiality_agg["rel_gt_20pct"][0] / total_count * 100,
        },
    ]

    # Log results
    for stat in materiality_stats:
        if stat["threshold_type"] == "absolute":
            logger.info(
                f"  |diff| > {stat['threshold_value']:.2f} vol points: "
                f"{stat['n_above_threshold']:,} ({stat['pct_above_threshold']:.2f}%)"
            )
        else:
            logger.info(
                f"  |diff| > {stat['threshold_value'] * 100:.0f}%: "
                f"{stat['n_above_threshold']:,} ({stat['pct_above_threshold']:.2f}%)"
            )

    summaries["materiality"] = pl.DataFrame(materiality_stats)
    logger.info(f"  ✅ Completed in {time.time() - start_time:.1f}s")

    return summaries


def generate_segmented_statistics(df_lazy: pl.LazyFrame) -> dict[str, pl.DataFrame]:
    """
    Generate statistics segmented by option characteristics using lazy group_by.

    CRITICAL: Uses lazy group_by operations, only collects small aggregated results.

    Args:
        df_lazy: Lazy comparison DataFrame

    Returns:
        Dictionary of segmented summary DataFrames (all small, collected)
    """
    logger.info("\n" + "=" * 80)
    logger.info("SEGMENTED ANALYSIS (LAZY GROUP BY)")
    logger.info("=" * 80)

    segmented = {}

    # By option type
    logger.info("\n1. By Option Type")
    start_time = time.time()
    by_type = (
        df_lazy.group_by("type")
        .agg(
            [
                pl.len().alias("n_obs"),
                pl.col("iv_mid_diff_abs").mean().alias("diff_abs_mean"),
                pl.col("iv_mid_diff_abs").median().alias("diff_abs_median"),
                pl.col("iv_mid_diff_abs").abs().mean().alias("diff_abs_mean_abs"),
                pl.col("iv_mid_diff_rel").mean().alias("diff_rel_mean"),
                pl.col("iv_mid_diff_rel").abs().mean().alias("diff_rel_mean_abs"),
            ]
        )
        .sort("type")
        .collect()  # Only collects ~2 rows (call/put)
    )
    segmented["by_type"] = by_type
    logger.info(f"\n{by_type}")
    logger.info(f"  ✅ Completed in {time.time() - start_time:.1f}s")

    # By moneyness
    logger.info("\n2. By Moneyness")
    start_time = time.time()
    by_moneyness = (
        df_lazy.group_by("moneyness_bin")
        .agg(
            [
                pl.len().alias("n_obs"),
                pl.col("moneyness").mean().alias("moneyness_mean"),
                pl.col("iv_mid_diff_abs").mean().alias("diff_abs_mean"),
                pl.col("iv_mid_diff_abs").median().alias("diff_abs_median"),
                pl.col("iv_mid_diff_abs").abs().mean().alias("diff_abs_mean_abs"),
                pl.col("iv_mid_diff_rel").mean().alias("diff_rel_mean"),
                pl.col("iv_mid_diff_rel").abs().mean().alias("diff_rel_mean_abs"),
            ]
        )
        .sort("moneyness_bin")
        .collect()  # Only collects ~3 rows (OTM/ATM/ITM)
    )
    segmented["by_moneyness"] = by_moneyness
    logger.info(f"\n{by_moneyness}")
    logger.info(f"  ✅ Completed in {time.time() - start_time:.1f}s")

    # By time to expiry
    logger.info("\n3. By Time to Expiry")
    start_time = time.time()
    by_ttl = (
        df_lazy.group_by("ttl_bin")
        .agg(
            [
                pl.len().alias("n_obs"),
                pl.col("time_to_expiry_days").mean().alias("ttl_mean"),
                pl.col("iv_mid_diff_abs").mean().alias("diff_abs_mean"),
                pl.col("iv_mid_diff_abs").median().alias("diff_abs_median"),
                pl.col("iv_mid_diff_abs").abs().mean().alias("diff_abs_mean_abs"),
                pl.col("iv_mid_diff_rel").mean().alias("diff_rel_mean"),
                pl.col("iv_mid_diff_rel").abs().mean().alias("diff_rel_mean_abs"),
            ]
        )
        .sort("ttl_bin")
        .collect()  # Only collects ~4 rows (<7d, 7-30d, etc.)
    )
    segmented["by_ttl"] = by_ttl
    logger.info(f"\n{by_ttl}")
    logger.info(f"  ✅ Completed in {time.time() - start_time:.1f}s")

    # By type + moneyness
    logger.info("\n4. By Type + Moneyness")
    start_time = time.time()
    by_type_moneyness = (
        df_lazy.group_by(["type", "moneyness_bin"])
        .agg(
            [
                pl.len().alias("n_obs"),
                pl.col("iv_mid_diff_abs").mean().alias("diff_abs_mean"),
                pl.col("iv_mid_diff_abs").abs().mean().alias("diff_abs_mean_abs"),
                pl.col("iv_mid_diff_rel").mean().alias("diff_rel_mean"),
                pl.col("iv_mid_diff_rel").abs().mean().alias("diff_rel_mean_abs"),
            ]
        )
        .sort(["type", "moneyness_bin"])
        .collect()  # Only collects ~6 rows (2 types × 3 moneyness bins)
    )
    segmented["by_type_moneyness"] = by_type_moneyness
    logger.info(f"\n{by_type_moneyness}")
    logger.info(f"  ✅ Completed in {time.time() - start_time:.1f}s")

    return segmented


def generate_time_series_statistics(df_lazy: pl.LazyFrame, df_rates: pl.DataFrame) -> pl.DataFrame:
    """
    Generate daily time series of differences and rates using lazy group_by.

    CRITICAL: Uses lazy group_by by date, only collects ~730 rows (one per day).

    Args:
        df_lazy: Lazy comparison DataFrame
        df_rates: Risk-free rates DataFrame

    Returns:
        Daily time series DataFrame (~730 rows, small enough to collect)
    """
    logger.info("\n" + "=" * 80)
    logger.info("TIME SERIES ANALYSIS (LAZY GROUP BY)")
    logger.info("=" * 80)

    # Daily aggregation - lazy group by, collects ~730 rows
    logger.info("\nGenerating daily time series...")
    start_time = time.time()
    daily_stats = (
        df_lazy.group_by("date")
        .agg(
            [
                pl.len().alias("n_obs"),
                pl.col("iv_mid_constant").mean().alias("iv_constant_mean"),
                pl.col("iv_mid_daily").mean().alias("iv_daily_mean"),
                pl.col("iv_mid_diff_abs").mean().alias("diff_abs_mean"),
                pl.col("iv_mid_diff_abs").median().alias("diff_abs_median"),
                pl.col("iv_mid_diff_abs").abs().mean().alias("diff_abs_mean_abs"),
                pl.col("iv_mid_diff_rel").mean().alias("diff_rel_mean"),
                pl.col("iv_mid_diff_rel").abs().mean().alias("diff_rel_mean_abs"),
                pl.col("risk_free_rate_pct").first().alias("risk_free_rate_pct"),
            ]
        )
        .sort("date")
        .collect()  # Only collects ~730 rows (one per day)
    )
    logger.info(f"  ✅ Completed in {time.time() - start_time:.1f}s")

    # Calculate correlation with risk-free rate changes
    corr_with_rate = daily_stats.select(
        [
            pl.corr("diff_abs_mean", "risk_free_rate_pct").alias("corr_diff_abs_vs_rate"),
            pl.corr("diff_rel_mean", "risk_free_rate_pct").alias("corr_diff_rel_vs_rate"),
        ]
    )

    logger.info("\nCorrelation with risk-free rate:")
    logger.info(f"{corr_with_rate}")

    # Find days with extreme differences
    logger.info("\nTop 10 days with largest absolute differences:")
    top_days = daily_stats.sort("diff_abs_mean_abs", descending=True).head(10)
    logger.info(f"\n{top_days}")

    return daily_stats


def save_results(
    output_dir: str,
    summaries: dict[str, pl.DataFrame],
    segmented: dict[str, pl.DataFrame],
    daily_stats: pl.DataFrame,
) -> None:
    """
    Save all analysis results to output directory.

    Args:
        output_dir: Output directory path
        summaries: Summary statistics DataFrames
        segmented: Segmented analysis DataFrames
        daily_stats: Daily time series DataFrame
    """
    logger.info("\n" + "=" * 80)
    logger.info("SAVING RESULTS")
    logger.info("=" * 80)

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Save all DataFrames as parquet
    for name, df_result in summaries.items():
        file_path = output_path / f"summary_{name}.parquet"
        df_result.write_parquet(file_path)
        logger.info(f"Saved {file_path}")

    for name, df_result in segmented.items():
        file_path = output_path / f"segmented_{name}.parquet"
        df_result.write_parquet(file_path)
        logger.info(f"Saved {file_path}")

    daily_file = output_path / "daily_time_series.parquet"
    daily_stats.write_parquet(daily_file)
    logger.info(f"Saved {daily_file}")

    logger.info(f"\nAll results saved to {output_dir}")


def main() -> None:
    """Main entry point with staged processing."""
    parser = argparse.ArgumentParser(
        description="Compare IV calculations using constant vs. daily risk-free rates (IMPROVED with staged processing)"
    )

    parser.add_argument(
        "--constant-file",
        default=DEFAULT_CONSTANT_IV_FILE,
        help=f"IV file with constant rates (default: {DEFAULT_CONSTANT_IV_FILE})",
    )
    parser.add_argument(
        "--daily-file",
        default=DEFAULT_DAILY_IV_FILE,
        help=f"IV file with daily rates (default: {DEFAULT_DAILY_IV_FILE})",
    )
    parser.add_argument(
        "--rates-file",
        default=DEFAULT_RATES_FILE,
        help=f"Risk-free rates file (default: {DEFAULT_RATES_FILE})",
    )
    parser.add_argument(
        "--output-dir",
        default=DEFAULT_OUTPUT_DIR,
        help=f"Output directory (default: {DEFAULT_OUTPUT_DIR})",
    )
    parser.add_argument(
        "--test-mode",
        action="store_true",
        help="Use test dataset (10M rows) for quick validation",
    )
    parser.add_argument(
        "--clean-start",
        action="store_true",
        help="Clean checkpoints and temp files before starting",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose logging",
    )

    args = parser.parse_args()

    # Setup logging
    setup_logging(args.verbose)

    # Use test files if requested
    if args.test_mode:
        logger.info("🧪 TEST MODE: Using 10M row test dataset")
        args.constant_file = "research/tardis/data/consolidated/quotes_1s_atm_short_dated_with_iv_10m_test.parquet"
        # Assuming test file for daily rates exists with same pattern
        daily_test = args.constant_file.replace("_with_iv_10m_test", "_with_iv_daily_rates_10m_test")
        if Path(daily_test).exists():
            args.daily_file = daily_test
        else:
            logger.warning(f"Daily test file not found: {daily_test}, using full daily file")

    start_time = time.time()
    logger.info("=" * 80)
    logger.info("IV COMPARISON ANALYSIS (STAGED PROCESSING)")
    logger.info("=" * 80)
    logger.info(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"Memory limit: {MAX_MEMORY_GB:.1f} GB")
    logger.info(f"Initial memory: {get_memory_usage():.2f} GB")

    try:
        # Check input files exist
        for file_path in [args.constant_file, args.daily_file, args.rates_file]:
            if not Path(file_path).exists():
                raise FileNotFoundError(f"File not found: {file_path}")

        # Setup directories
        output_path = Path(args.output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        temp_dir = output_path / "temp"

        # Load checkpoint or start fresh
        checkpoint = None
        if args.clean_start:
            logger.info("Cleaning checkpoints and temp files...")
            clean_checkpoint()
            clean_temp_files(temp_dir)
        else:
            checkpoint = load_checkpoint()
            if checkpoint:
                logger.info("📥 Resuming from checkpoint")

        # Clean temp files if no checkpoint
        if not checkpoint:
            clean_temp_files(temp_dir)

        # ==========================================
        # STAGE 1: Filter and checkpoint
        # ==========================================
        constant_filtered, daily_filtered, stage1_stats = stage1_filter_and_checkpoint(
            args.constant_file, args.daily_file, temp_dir, checkpoint
        )
        logger.info(f"Stage 1 statistics: {stage1_stats}")

        # ==========================================
        # STAGE 2: Partitioned join
        # ==========================================
        joined_file = stage2_partitioned_join(constant_filtered, daily_filtered, temp_dir, checkpoint)

        # ==========================================
        # STAGE 3: Transform and enrich
        # ==========================================
        final_comparison_file = stage3_transform_and_enrich(joined_file, args.rates_file, output_path, checkpoint)

        # ==========================================
        # STAGE 4: Generate statistics
        # ==========================================
        logger.info("=" * 80)
        logger.info("STAGE 4: GENERATE STATISTICS")
        logger.info("=" * 80)

        # Load final data lazily for statistics
        df_comparison_lazy = pl.scan_parquet(final_comparison_file)

        # Load rates for time series analysis
        df_rates = pl.read_parquet(args.rates_file).select(
            [
                pl.col("date"),
                pl.col("blended_supply_apr_ma7").alias("risk_free_rate_pct"),
            ]
        )

        # Generate statistics (only small aggregated results are collected)
        logger.info("\nGenerating summary statistics...")
        summaries = generate_summary_statistics(df_comparison_lazy)

        logger.info("\nGenerating segmented statistics...")
        segmented = generate_segmented_statistics(df_comparison_lazy)

        logger.info("\nGenerating time series statistics...")
        daily_stats = generate_time_series_statistics(df_comparison_lazy, df_rates)

        # Save results
        save_results(args.output_dir, summaries, segmented, daily_stats)

        # Clean up temp files and checkpoint
        logger.info("\nCleaning up...")
        clean_temp_files(temp_dir)
        if temp_dir.exists() and not any(temp_dir.iterdir()):
            temp_dir.rmdir()
        clean_checkpoint()

        # Get final count
        total_compared = summaries["overall"]["n_observations"][0]

        # Final summary
        elapsed = time.time() - start_time
        logger.info("\n" + "=" * 80)
        logger.info("🎉 ANALYSIS COMPLETE")
        logger.info("=" * 80)
        logger.info(f"Total time: {elapsed:.1f} seconds ({elapsed / 60:.1f} minutes)")
        logger.info(f"Analyzed {total_compared:,} option quotes")
        logger.info(f"Results saved to {args.output_dir}")
        logger.info(f"Peak memory usage: {get_memory_usage():.2f} GB")
        logger.info("\n✅ Staged processing completed successfully without OOM issues!")

    except KeyboardInterrupt:
        logger.warning("\n⚠️ Process interrupted by user")
        logger.info("Checkpoint saved - run again to resume from last completed stage")
        sys.exit(1)

    except Exception as e:
        logger.error(f"❌ ERROR: {e}")
        import traceback

        traceback.print_exc()
        logger.info("\n💡 Checkpoint saved - you can resume by running the script again")
        sys.exit(1)


if __name__ == "__main__":
    main()
