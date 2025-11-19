#!/usr/bin/env python3
"""
Memory Profiler for XGBoost Training
=====================================

This script profiles memory usage during XGBoost training to identify
bottlenecks and optimize memory consumption for the 16GB RAM constraint.

Author: BT Research Team
Date: 2025-10-31
"""

from __future__ import annotations

import gc
import logging
import os
import sys
import tracemalloc
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
import psutil
import yaml

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


class MemoryProfiler:
    """Memory profiling utilities for debugging XGBoost training."""

    def __init__(self):
        self.process = psutil.Process(os.getpid())
        self.snapshots = []
        self.tracemalloc_started = False

    def start_tracemalloc(self):
        """Start tracemalloc for detailed memory tracking."""
        if not self.tracemalloc_started:
            tracemalloc.start()
            self.tracemalloc_started = True
            logger.info("Started tracemalloc memory tracking")

    def stop_tracemalloc(self):
        """Stop tracemalloc tracking."""
        if self.tracemalloc_started:
            tracemalloc.stop()
            self.tracemalloc_started = False
            logger.info("Stopped tracemalloc memory tracking")

    def get_memory_gb(self) -> float:
        """Get current process memory usage in GB."""
        return self.process.memory_info().rss / (1024**3)

    def get_system_memory(self) -> dict:
        """Get system memory statistics."""
        vm = psutil.virtual_memory()
        swap = psutil.swap_memory()
        return {
            "total_gb": vm.total / (1024**3),
            "available_gb": vm.available / (1024**3),
            "used_gb": vm.used / (1024**3),
            "percent": vm.percent,
            "swap_used_gb": swap.used / (1024**3),
            "swap_percent": swap.percent,
        }

    def log_memory(self, label: str):
        """Log current memory usage with a label."""
        mem_gb = self.get_memory_gb()
        sys_mem = self.get_system_memory()

        logger.info(f"[MEMORY] {label}")
        logger.info(f"  Process: {mem_gb:.2f} GB")
        logger.info(f"  System: {sys_mem['used_gb']:.1f}/{sys_mem['total_gb']:.1f} GB ({sys_mem['percent']:.1f}%)")
        logger.info(f"  Available: {sys_mem['available_gb']:.1f} GB")

        if sys_mem["swap_percent"] > 10:
            logger.warning(f"  ⚠️  HIGH SWAP: {sys_mem['swap_used_gb']:.1f} GB ({sys_mem['swap_percent']:.1f}%)")

        if mem_gb > 5.0:
            logger.warning(f"  ⚠️  EXCEEDING SAFE LIMIT (5GB): {mem_gb:.2f} GB")

        return mem_gb

    def take_snapshot(self, label: str):
        """Take a tracemalloc snapshot for detailed analysis."""
        if self.tracemalloc_started:
            snapshot = tracemalloc.take_snapshot()
            self.snapshots.append((label, snapshot))
            logger.info(f"Took memory snapshot: {label}")

    def compare_snapshots(self, idx1: int = 0, idx2: int = -1):
        """Compare two memory snapshots to find leaks."""
        if len(self.snapshots) < 2:
            logger.warning("Need at least 2 snapshots to compare")
            return

        label1, snap1 = self.snapshots[idx1]
        label2, snap2 = self.snapshots[idx2]

        logger.info(f"\n[MEMORY DIFF] {label1} → {label2}")
        top_stats = snap2.compare_to(snap1, "lineno")

        for stat in top_stats[:10]:
            if stat.size_diff > 0:
                logger.info(f"  {stat}")

    def get_top_allocations(self, limit: int = 10):
        """Get top memory allocations from most recent snapshot."""
        if not self.snapshots:
            logger.warning("No snapshots available")
            return

        label, snapshot = self.snapshots[-1]
        logger.info(f"\n[TOP ALLOCATIONS] {label}")

        top_stats = snapshot.statistics("lineno")
        for stat in top_stats[:limit]:
            size_mb = stat.size / (1024**2)
            logger.info(f"  {stat}: {size_mb:.1f} MB")

    def force_gc(self):
        """Force garbage collection and log results."""
        mem_before = self.get_memory_gb()
        gc.collect()
        mem_after = self.get_memory_gb()
        freed = mem_before - mem_after

        if freed > 0.1:  # Only log if significant memory freed
            logger.info(f"[GC] Freed {freed:.2f} GB ({mem_before:.2f} → {mem_after:.2f} GB)")


def analyze_dataset_memory(file_path: str, profiler: MemoryProfiler) -> None:
    """Analyze memory usage when loading a dataset."""
    logger.info(f"\n{'='*80}")
    logger.info(f"ANALYZING: {Path(file_path).name}")
    logger.info(f"{'='*80}")

    profiler.log_memory("Before loading")

    # Method 1: Eager loading (baseline)
    logger.info("\nMethod 1: Eager loading with pl.read_parquet()")
    try:
        df = pl.read_parquet(file_path)
        rows, cols = df.shape
        size_mb = df.estimated_size("mb")
        logger.info(f"  Shape: {rows:,} × {cols}")
        logger.info(f"  Estimated size: {size_mb:.1f} MB")

        profiler.log_memory("After eager load")
        profiler.take_snapshot("eager_load")

        # Check data types
        dtypes = df.dtypes
        float64_cols = [col for col, dtype in zip(df.columns, dtypes) if dtype == pl.Float64]
        if float64_cols:
            logger.warning(f"  ⚠️  {len(float64_cols)} Float64 columns (could use Float32)")

        del df
        profiler.force_gc()

    except Exception as e:
        logger.error(f"  Failed to load eagerly: {e}")

    # Method 2: Lazy loading
    logger.info("\nMethod 2: Lazy loading with pl.scan_parquet()")
    try:
        lazy_df = pl.scan_parquet(file_path)
        profiler.log_memory("After lazy scan (no data loaded)")

        # Get schema without loading data
        schema = lazy_df.collect_schema()
        logger.info(f"  Columns: {len(schema)}")

        # Count rows without loading all data
        row_count = lazy_df.select(pl.len()).collect().item()
        logger.info(f"  Rows: {row_count:,}")
        profiler.log_memory("After counting rows")

        # Sample first 1000 rows
        sample = lazy_df.head(1000).collect()
        logger.info(f"  Sample shape: {sample.shape}")
        profiler.log_memory("After loading 1000 rows")

        del sample
        profiler.force_gc()

    except Exception as e:
        logger.error(f"  Failed lazy loading: {e}")

    # Method 3: Column selection
    logger.info("\nMethod 3: Loading only essential columns")
    try:
        lazy_df = pl.scan_parquet(file_path)

        # Identify essential columns
        essential_cols = ["timestamp_seconds", "residual", "predicted_prob", "actual_outcome"]
        available_cols = [col for col in essential_cols if col in lazy_df.collect_schema().names()]

        if available_cols:
            subset_df = lazy_df.select(available_cols).collect()
            logger.info(f"  Loaded {len(available_cols)} columns: {subset_df.shape}")
            profiler.log_memory(f"After loading {len(available_cols)} columns")
            del subset_df
            profiler.force_gc()

    except Exception as e:
        logger.error(f"  Failed column selection: {e}")


def profile_feature_engineering(profiler: MemoryProfiler) -> None:
    """Profile memory usage during feature engineering."""
    logger.info(f"\n{'='*80}")
    logger.info("PROFILING FEATURE ENGINEERING PIPELINE")
    logger.info(f"{'='*80}")

    # File paths
    MODEL_DIR = Path(__file__).parent.parent
    files = {
        "baseline": MODEL_DIR / "results/production_backtest_results.parquet",
        "rv": MODEL_DIR / "results/realized_volatility_1s.parquet",
        "microstructure": MODEL_DIR / "results/microstructure_features.parquet",
        "advanced": MODEL_DIR / "results/advanced_features.parquet",
    }

    profiler.log_memory("Start of feature engineering")

    # Check file sizes
    logger.info("\nFile sizes:")
    for name, path in files.items():
        if path.exists():
            size_gb = path.stat().st_size / (1024**3)
            logger.info(f"  {name}: {size_gb:.2f} GB")
        else:
            logger.warning(f"  {name}: NOT FOUND")

    # Analyze join memory usage
    logger.info("\nSimulating feature joins (October 2023 pilot):")

    try:
        # Load baseline with filter
        from datetime import date
        baseline_df = (
            pl.scan_parquet(str(files["baseline"]))
            .filter(
                (pl.col("date") >= date(2023, 10, 1)) &
                (pl.col("date") <= date(2023, 10, 31))
            )
        )

        profiler.log_memory("After baseline lazy filter")

        # Join features one by one
        for name, path in list(files.items())[1:]:  # Skip baseline
            if not path.exists():
                continue

            logger.info(f"\nJoining {name}...")
            feature_df = pl.scan_parquet(str(path))

            # Simulate join
            baseline_df = baseline_df.join(
                feature_df,
                on="timestamp_seconds",
                how="left"
            )

            # Check query plan
            plan = baseline_df.explain()
            if "STREAMING" in plan:
                logger.info(f"  ✅ Using streaming for {name}")
            else:
                logger.warning(f"  ⚠️  NOT streaming for {name}")

        # Attempt to collect
        logger.info("\nAttempting to collect joined data...")
        profiler.take_snapshot("before_collect")

        # Only collect a sample to avoid OOM
        sample = baseline_df.head(10000).collect()
        logger.info(f"  Sample shape: {sample.shape}")
        profiler.log_memory("After collecting 10K rows")

        # Check memory per row
        mem_per_row = sample.estimated_size("mb") / len(sample) * 1024  # KB per row
        logger.info(f"  Memory per row: {mem_per_row:.2f} KB")

        full_rows = 2_411_938  # October pilot
        estimated_gb = (mem_per_row * full_rows) / (1024**2)
        logger.info(f"  Estimated for {full_rows:,} rows: {estimated_gb:.1f} GB")

        if estimated_gb > 5.0:
            logger.warning(f"  ⚠️  EXCEEDS SAFE LIMIT! Need chunking or streaming")

        del sample
        profiler.force_gc()

    except Exception as e:
        logger.error(f"Feature engineering failed: {e}")
        import traceback
        traceback.print_exc()


def suggest_optimizations(profiler: MemoryProfiler) -> None:
    """Suggest memory optimizations based on profiling."""
    logger.info(f"\n{'='*80}")
    logger.info("MEMORY OPTIMIZATION RECOMMENDATIONS")
    logger.info(f"{'='*80}")

    recommendations = [
        ("Use Float32 instead of Float64", "Reduces memory by 50% for numeric features"),
        ("Stream write large outputs", "Use .sink_parquet(streaming=True) for >10M rows"),
        ("Temporal chunking", "Process data in monthly chunks for 63M rows"),
        ("Column pruning", "Load only required columns, not all 99 features"),
        ("External memory XGBoost", "Use DMatrix with external memory for >5GB datasets"),
        ("Lazy evaluation", "Use pl.scan_parquet() and process in batches"),
        ("Feature selection", "Reduce from 99 to top 30-50 features based on importance"),
        ("Batch predictions", "Process test set in 2M row batches"),
    ]

    for i, (title, desc) in enumerate(recommendations, 1):
        logger.info(f"\n{i}. {title}")
        logger.info(f"   {desc}")

    logger.info(f"\n{'='*80}")
    logger.info("CRITICAL FIXES FOR YOUR SETUP")
    logger.info(f"{'='*80}")

    logger.info("""
1. IMMEDIATE: Fix the 29GB memory spike
   - Don't use .collect() on full joined dataset
   - Use streaming write: df.sink_parquet("output.parquet", streaming=True)

2. FEATURE ENGINEERING: Reduce memory footprint
   - Cast to Float32: .cast({col: pl.Float32 for col in float_cols})
   - Process in chunks: Split by month or week

3. XGBOOST: Enable true external memory
   - Save features to Parquet first
   - Use: xgb.DMatrix("file.parquet?format=parquet#cache")
   - Set max_bin=32 and nthread=1 (as configured)

4. VALIDATION: Use smaller sample
   - 1% of 63M = 630K rows (statistically valid)
   - Or use temporal split (last month only)

5. PREDICTIONS: Batch processing
   - Never load all 39M test rows at once
   - Process in 2M row batches with cleanup
""")


def main():
    """Main profiling workflow."""
    profiler = MemoryProfiler()
    profiler.start_tracemalloc()

    try:
        # Profile dataset loading
        MODEL_DIR = Path(__file__).parent.parent
        test_files = [
            MODEL_DIR / "results/production_backtest_results.parquet",
            MODEL_DIR / "results/realized_volatility_1s.parquet",
        ]

        for file_path in test_files:
            if file_path.exists():
                analyze_dataset_memory(str(file_path), profiler)
            else:
                logger.warning(f"File not found: {file_path}")

        # Profile feature engineering
        profile_feature_engineering(profiler)

        # Show memory allocations
        profiler.get_top_allocations()

        # Compare snapshots if available
        if len(profiler.snapshots) >= 2:
            profiler.compare_snapshots(0, -1)

        # Suggest optimizations
        suggest_optimizations(profiler)

    finally:
        profiler.stop_tracemalloc()
        profiler.log_memory("Final")


if __name__ == "__main__":
    main()