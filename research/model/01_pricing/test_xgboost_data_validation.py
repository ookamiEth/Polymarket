#!/usr/bin/env python3
"""
XGBoost Data Validation Script

Validates data characteristics and assesses overfitting risk before training.
Analyzes dataset sizes, feature distributions, memory requirements, and data quality.
"""

import logging
from pathlib import Path
from datetime import datetime
import polars as pl
import numpy as np

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def format_bytes(bytes_val: int) -> str:
    """Convert bytes to human-readable format."""
    for unit in ['B', 'KB', 'MB', 'GB']:
        if bytes_val < 1024:
            return f"{bytes_val:.2f} {unit}"
        bytes_val /= 1024
    return f"{bytes_val:.2f} TB"


def analyze_data_characteristics(
    data_file: Path,
    rv_file: Path,
    micro_file: Path,
    advanced_file: Path,
    train_start: str,
    train_end: str,
    val_start: str,
    val_end: str,
    test_start: str,
    test_end: str,
) -> None:
    """Analyze dataset characteristics and memory requirements."""

    logger.info("=" * 80)
    logger.info("XGBOOST DATA VALIDATION REPORT")
    logger.info("=" * 80)
    logger.info(f"Report generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info("")

    # ====================================================================
    # 1. DATASET SIZES
    # ====================================================================
    logger.info("1. DATASET SIZES")
    logger.info("-" * 80)

    # Load main data lazily
    df = pl.scan_parquet(data_file)

    # Get total row count
    total_rows = df.select(pl.len()).collect().item()
    logger.info(f"Total rows in dataset: {total_rows:,}")

    # Get row counts per split
    splits = {
        "Training": (train_start, train_end),
        "Validation": (val_start, val_end),
        "Test": (test_start, test_end),
    }

    split_counts = {}
    for split_name, (start, end) in splits.items():
        count = (
            df.filter((pl.col("date") >= start) & (pl.col("date") < end))
              .select(pl.len())
              .collect()
              .item()
        )
        split_counts[split_name] = count
        pct = (count / total_rows) * 100
        logger.info(f"{split_name:12s}: {count:12,} rows ({pct:5.1f}%)")

    # Check split balance
    train_rows = split_counts["Training"]
    val_rows = split_counts["Validation"]
    test_rows = split_counts["Test"]

    if test_rows > train_rows:
        ratio = test_rows / train_rows
        logger.warning(f"⚠️  Test set is {ratio:.1f}x LARGER than training set!")
        logger.warning("    This increases overfitting risk due to regime shift.")

    if val_rows < train_rows * 0.15:
        pct = (val_rows / train_rows) * 100
        logger.warning(f"⚠️  Validation set is only {pct:.1f}% of training set")
        logger.warning("    Consider increasing validation size to 15-20%")

    logger.info("")

    # ====================================================================
    # 2. FEATURE ANALYSIS
    # ====================================================================
    logger.info("2. FEATURE ANALYSIS")
    logger.info("-" * 80)

    # Count features from each source
    feature_counts = {}

    # Baseline features (from main data file)
    baseline_df = pl.scan_parquet(data_file)
    schema = baseline_df.collect_schema()
    baseline_cols = [c for c in schema.names() if c not in ["date", "symbol", "timestamp_seconds"]]
    feature_counts["Baseline"] = len(baseline_cols)
    logger.info(f"Baseline features:      {feature_counts['Baseline']:3d}")

    # RV features
    if rv_file.exists():
        rv_df = pl.scan_parquet(rv_file)
        schema_rv = rv_df.collect_schema()
        rv_cols = [c for c in schema_rv.names() if c not in ["timestamp_seconds"]]
        feature_counts["RV"] = len(rv_cols)
        logger.info(f"RV features:            {feature_counts['RV']:3d}")
    else:
        logger.warning(f"⚠️  RV features file not found: {rv_file}")
        feature_counts["RV"] = 0

    # Microstructure features
    if micro_file.exists():
        micro_df = pl.scan_parquet(micro_file)
        schema_micro = micro_df.collect_schema()
        micro_cols = [c for c in schema_micro.names() if c not in ["timestamp_seconds"]]
        feature_counts["Microstructure"] = len(micro_cols)
        logger.info(f"Microstructure features: {feature_counts['Microstructure']:3d}")
    else:
        logger.warning(f"⚠️  Microstructure features file not found: {micro_file}")
        feature_counts["Microstructure"] = 0

    # Advanced features
    if advanced_file.exists():
        advanced_df = pl.scan_parquet(advanced_file)
        schema_advanced = advanced_df.collect_schema()
        advanced_cols = [c for c in schema_advanced.names() if c not in ["timestamp_seconds", "timestamp"]]
        feature_counts["Advanced"] = len(advanced_cols)
        logger.info(f"Advanced features:      {feature_counts['Advanced']:3d}")
    else:
        logger.warning(f"⚠️  Advanced features file not found: {advanced_file}")
        feature_counts["Advanced"] = 0

    total_features = sum(feature_counts.values())
    logger.info(f"{'─' * 30}")
    logger.info(f"TOTAL FEATURES:         {total_features:3d}")
    logger.info("")

    # ====================================================================
    # 3. MEMORY ESTIMATION
    # ====================================================================
    logger.info("3. MEMORY ESTIMATION")
    logger.info("-" * 80)

    # Estimate memory requirements
    bytes_per_float = 4  # Float32

    # Training data memory
    train_memory_bytes = train_rows * total_features * bytes_per_float
    logger.info(f"Training data size:     {format_bytes(train_memory_bytes)}")
    logger.info(f"  ({train_rows:,} rows × {total_features} features × {bytes_per_float} bytes)")

    # Validation data memory (if using 20% split from training)
    val_split_rows = int(train_rows * 0.2)
    val_split_memory = val_split_rows * total_features * bytes_per_float
    logger.info(f"Validation split (20%): {format_bytes(val_split_memory)}")
    logger.info(f"  ({val_split_rows:,} rows × {total_features} features × {bytes_per_float} bytes)")

    # Test data memory
    test_memory_bytes = test_rows * total_features * bytes_per_float
    logger.info(f"Test data size:         {format_bytes(test_memory_bytes)}")
    logger.info(f"  ({test_rows:,} rows × {total_features} features × {bytes_per_float} bytes)")

    # XGBoost overhead estimation
    logger.info("")
    logger.info("XGBoost Memory Overhead:")

    # Quantile sketch memory (depends on max_bin)
    bins_default = 256
    bins_reduced = 64
    sketch_mem_default = train_rows * total_features * 8 / bins_default  # Rough estimate
    sketch_mem_reduced = train_rows * total_features * 8 / bins_reduced

    logger.info(f"  Quantile sketches (max_bin=256): {format_bytes(sketch_mem_default)}")
    logger.info(f"  Quantile sketches (max_bin=64):  {format_bytes(sketch_mem_reduced)} ✅ RECOMMENDED")

    # Thread buffers
    threads_default = 4
    threads_reduced = 2
    buffer_mem_default = train_memory_bytes * 0.3 * threads_default / 4
    buffer_mem_reduced = train_memory_bytes * 0.3 * threads_reduced / 4

    logger.info(f"  Thread buffers (nthread=4):      {format_bytes(buffer_mem_default)}")
    logger.info(f"  Thread buffers (nthread=2):      {format_bytes(buffer_mem_reduced)} ✅ RECOMMENDED")

    # Total peak memory estimation
    logger.info("")
    logger.info("Estimated Peak Memory Usage:")

    system_overhead = 5 * 1024**3  # 5GB for OS + Python

    # Current configuration (no memory fixes)
    peak_current = system_overhead + train_memory_bytes + sketch_mem_default + buffer_mem_default
    logger.info(f"  Current config (defaults):  {format_bytes(peak_current)}")

    # Optimized configuration
    peak_optimized = system_overhead + train_memory_bytes + sketch_mem_reduced + buffer_mem_reduced
    logger.info(f"  Optimized config:           {format_bytes(peak_optimized)} ✅")

    # Safety check
    available_ram = 16 * 1024**3  # 16GB
    safe_limit = available_ram * 0.75  # 75% utilization

    logger.info(f"  Available RAM:              {format_bytes(available_ram)}")
    logger.info(f"  Safe limit (75%):           {format_bytes(safe_limit)}")

    if peak_current > safe_limit:
        logger.warning(f"❌ Current config EXCEEDS safe limit by {format_bytes(peak_current - safe_limit)}")
        logger.warning("   Training will likely crash with OOM!")

    if peak_optimized > safe_limit:
        logger.warning(f"⚠️  Optimized config still exceeds safe limit by {format_bytes(peak_optimized - safe_limit)}")
        logger.warning("   May need temporal chunking or cloud instance.")
    else:
        logger.info(f"✅ Optimized config is SAFE (under limit by {format_bytes(safe_limit - peak_optimized)})")

    logger.info("")

    # ====================================================================
    # 4. OVERFITTING RISK ASSESSMENT
    # ====================================================================
    logger.info("4. OVERFITTING RISK ASSESSMENT")
    logger.info("-" * 80)

    # Model complexity estimation
    max_depth = 4
    n_estimators = 100
    max_leaves_per_tree = 2**max_depth
    total_leaf_nodes = n_estimators * max_leaves_per_tree

    logger.info(f"Model complexity:")
    logger.info(f"  Max depth:              {max_depth}")
    logger.info(f"  Number of trees:        {n_estimators}")
    logger.info(f"  Max leaves per tree:    {max_leaves_per_tree}")
    logger.info(f"  Total leaf nodes:       {total_leaf_nodes:,}")

    # Data-to-parameter ratio
    data_param_ratio = train_rows / total_leaf_nodes
    logger.info(f"  Data/parameter ratio:   {data_param_ratio:,.1f}:1")

    if data_param_ratio < 1000:
        logger.warning("⚠️  Data/parameter ratio is LOW (<1000:1)")
        logger.warning("   Higher overfitting risk - use strong regularization!")
    elif data_param_ratio < 5000:
        logger.info("✅ Data/parameter ratio is MODERATE (1000-5000:1)")
        logger.info("   Use standard regularization (gamma, lambda, alpha)")
    else:
        logger.info("✅ Data/parameter ratio is HIGH (>5000:1)")
        logger.info("   Lower overfitting risk, but still use regularization")

    # Feature-to-sample ratio
    feature_sample_ratio = total_features / (train_rows / 1000)
    logger.info(f"  Features per 1K samples: {feature_sample_ratio:.2f}")

    if feature_sample_ratio > 0.1:
        logger.warning("⚠️  High feature-to-sample ratio (>0.1 per 1K)")
        logger.warning("   Consider feature selection or stronger regularization")

    # Time-series considerations
    logger.info("")
    logger.info("Time-series risks:")

    # Convert dates to datetime for calculation if needed
    from datetime import date
    train_start_dt = datetime(train_start.year, train_start.month, train_start.day) if isinstance(train_start, date) else datetime.fromisoformat(train_start)
    train_end_dt = datetime(train_end.year, train_end.month, train_end.day) if isinstance(train_end, date) else datetime.fromisoformat(train_end)
    test_end_dt = datetime(test_end.year, test_end.month, test_end.day) if isinstance(test_end, date) else datetime.fromisoformat(test_end)

    train_months = (train_end_dt - train_start_dt).days // 30
    test_months = (test_end_dt - train_start_dt).days // 30

    logger.info(f"  Training period:        {train_months} months ({train_start} to {train_end})")
    logger.info(f"  Test period:            {test_months} months ({test_start} to {test_end})")

    if test_months > train_months:
        logger.warning("⚠️  Test period is LONGER than training period")
        logger.warning("   Higher risk of regime shift and distribution drift")

    # Validation set recommendation
    logger.info("")
    logger.info("Validation strategy:")
    if val_rows < train_rows * 0.15:
        logger.warning("⚠️  Current validation set is TOO SMALL")
        logger.warning("   Recommended: Split training data 80/20 for train/val")
        logger.warning("   This will provide better early stopping signal")
    else:
        logger.info("✅ Dedicated validation set exists")
        logger.info("   Use for early stopping and hyperparameter tuning")

    logger.info("")

    # ====================================================================
    # 5. DATA QUALITY CHECKS
    # ====================================================================
    logger.info("5. DATA QUALITY CHECKS")
    logger.info("-" * 80)

    # Check for missing values in training data
    train_df = df.filter((pl.col("date") >= train_start) & (pl.col("date") < train_end))

    # Sample 1% to check for nulls (faster than full scan)
    sample_size = max(int(train_rows * 0.01), 1000)
    sample_df = train_df.head(sample_size).collect()

    null_counts = sample_df.null_count()
    cols_with_nulls = [col for col in null_counts.columns if null_counts[col][0] > 0]

    if cols_with_nulls:
        logger.warning(f"⚠️  Found {len(cols_with_nulls)} columns with null values in sample:")
        for col in cols_with_nulls[:10]:  # Show first 10
            null_pct = (null_counts[col][0] / sample_size) * 100
            logger.warning(f"     {col}: {null_pct:.1f}% nulls")
        if len(cols_with_nulls) > 10:
            logger.warning(f"     ... and {len(cols_with_nulls) - 10} more")
    else:
        logger.info("✅ No null values found in sample")

    # Check date range continuity
    date_stats = (
        train_df.select([
            pl.col("date").min().alias("min_date"),
            pl.col("date").max().alias("max_date"),
            pl.col("date").n_unique().alias("unique_dates"),
        ])
        .collect()
    )

    logger.info(f"Date range:")
    logger.info(f"  Min date: {date_stats['min_date'][0]}")
    logger.info(f"  Max date: {date_stats['max_date'][0]}")
    logger.info(f"  Unique dates: {date_stats['unique_dates'][0]:,}")

    logger.info("")

    # ====================================================================
    # 6. RECOMMENDATIONS
    # ====================================================================
    logger.info("6. RECOMMENDATIONS")
    logger.info("=" * 80)

    logger.info("")
    logger.info("✅ REQUIRED FIXES (to prevent OOM crash):")
    logger.info("   1. Set max_bin=64 (reduces quantile sketch memory by 75%)")
    logger.info("   2. Set nthread=2 (reduces buffer memory by 50%)")
    logger.info("   3. Set max_cached_hist_node=32 (limits histogram cache)")
    logger.info("   4. Set ref_resource_aware=True (enables XGBoost memory awareness)")

    logger.info("")
    logger.info("✅ RECOMMENDED REGULARIZATION (to prevent overfitting):")
    logger.info("   1. Add early_stopping_rounds=15 (stop when no improvement)")
    logger.info("   2. Set gamma=25 (minimum loss reduction threshold)")
    logger.info("   3. Set reg_lambda=50 (L2 regularization)")
    logger.info("   4. Set reg_alpha=2 (L1 regularization)")
    logger.info("   5. Set min_child_weight=10 (minimum samples per leaf)")
    logger.info("   6. Split training data 80/20 for train/validation")

    logger.info("")
    logger.info("✅ OPTIONAL IMPROVEMENTS:")
    logger.info("   1. Increase n_estimators to 200 (with early stopping)")
    logger.info("   2. Use YAML config file for hyperparameter management")
    logger.info("   3. Monitor validation metrics during training")
    logger.info("   4. Consider feature selection if memory remains tight")

    logger.info("")
    logger.info("=" * 80)
    logger.info("END OF VALIDATION REPORT")
    logger.info("=" * 80)


def main() -> None:
    """Main entry point."""
    # File paths
    base_dir = Path("/home/ubuntu/Polymarket")
    results_dir = base_dir / "research" / "model" / "results"

    data_file = results_dir / "production_backtest_results.parquet"
    rv_file = results_dir / "realized_volatility_1s.parquet"
    micro_file = results_dir / "microstructure_features.parquet"
    advanced_file = results_dir / "advanced_features.parquet"

    # Date ranges (matching typical training setup)
    from datetime import date
    train_start = date(2023, 10, 1)
    train_end = date(2024, 3, 31)  # 6 months
    val_start = date(2024, 4, 1)
    val_end = date(2024, 6, 30)    # 3 months
    test_start = date(2024, 7, 1)
    test_end = date(2025, 9, 30)   # 15 months

    # Run analysis
    analyze_data_characteristics(
        data_file=data_file,
        rv_file=rv_file,
        micro_file=micro_file,
        advanced_file=advanced_file,
        train_start=train_start,
        train_end=train_end,
        val_start=val_start,
        val_end=val_end,
        test_start=test_start,
        test_end=test_end,
    )


if __name__ == "__main__":
    main()
