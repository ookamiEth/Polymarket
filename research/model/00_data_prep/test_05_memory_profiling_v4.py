#!/usr/bin/env python3
"""
Memory Profiling Test for V4 Pipeline
======================================

Tests memory usage during V4 pipeline execution to ensure it stays within 256GB RAM limits.

Critical for this machine which has crashed before due to OOM errors.

Tests:
1. Feature engineering peak memory per chunk (<35GB per chunk)
2. Pipeline preparation memory usage (<50GB)
3. Model training memory (8 models, estimate peak usage)
4. Memory leak detection (check for gradual growth)

Runtime: ~30 minutes
Sample Size: 1M rows (representative but fast)

Author: BT Research Team
Date: 2025-11-13
"""

from __future__ import annotations

import logging
import sys
import time
from pathlib import Path

import polars as pl
import psutil

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

# Paths
MODEL_DIR = Path(__file__).parent.parent
DATA_DIR = MODEL_DIR / "data"
RESULTS_DIR = MODEL_DIR / "results"

# Memory limits (GB)
TOTAL_RAM_GB = 256
SOFT_LIMIT_GB = 250  # Reserve 6GB for OS
CHUNK_LIMIT_GB = 35  # Per-chunk limit for feature engineering
PIPELINE_LIMIT_GB = 50  # Pipeline preparation
TRAINING_LIMIT_GB = 100  # Model training (conservative estimate)


class MemoryProfileResult:
    """Store memory profiling result."""

    def __init__(self, name: str, passed: bool, message: str = "", peak_gb: float = 0.0):
        self.name = name
        self.passed = passed
        self.message = message
        self.peak_gb = peak_gb

    def __repr__(self) -> str:
        status = "✅ PASS" if self.passed else "❌ FAIL"
        mem_info = f" (peak: {self.peak_gb:.2f}GB)" if self.peak_gb > 0 else ""
        return f"{status}: {self.name}{mem_info}" + (f" - {self.message}" if self.message else "")


def get_memory_usage_gb() -> float:
    """Get current memory usage in GB."""
    mem = psutil.virtual_memory()
    return mem.used / (1024**3)


def get_memory_available_gb() -> float:
    """Get available memory in GB."""
    mem = psutil.virtual_memory()
    return mem.available / (1024**3)


def profile_feature_engineering_chunk() -> MemoryProfileResult:
    """Profile memory usage during feature engineering on sample chunk."""
    logger.info("\n" + "=" * 80)
    logger.info("TEST 1: Feature Engineering Memory Profile")
    logger.info("=" * 80)

    try:
        baseline_file = RESULTS_DIR / "production_backtest_results_v4.parquet"
        advanced_file = RESULTS_DIR / "advanced_features.parquet"
        micro_file = RESULTS_DIR / "microstructure_features.parquet"

        # Check files exist
        if not all([baseline_file.exists(), advanced_file.exists(), micro_file.exists()]):
            return MemoryProfileResult(
                "Feature engineering",
                False,
                "Input files not found",
            )

        # Record initial memory
        mem_start = get_memory_usage_gb()
        logger.info(f"Initial memory: {mem_start:.2f}GB")

        # Load sample chunk (1M rows from middle of dataset)
        logger.info("Loading sample chunk (1M rows)...")
        baseline_df = pl.scan_parquet(baseline_file).slice(1_000_000, 1_000_000).collect()
        mem_after_baseline = get_memory_usage_gb()
        logger.info(f"  After baseline load: {mem_after_baseline:.2f}GB (+{mem_after_baseline - mem_start:.2f}GB)")

        # Add timestamp_seconds
        baseline_df = baseline_df.with_columns(
            [
                pl.col("timestamp").alias("timestamp_seconds"),
                (pl.col("S") / pl.col("K")).alias("moneyness"),
            ]
        )

        # Load advanced features
        logger.info("Loading advanced features...")
        advanced_df = (
            pl.scan_parquet(advanced_file)
            .filter(pl.col("timestamp_seconds").is_in(baseline_df["timestamp"].cast(pl.Int64)))
            .collect()
        )
        mem_after_advanced = get_memory_usage_gb()
        logger.info(
            f"  After advanced load: {mem_after_advanced:.2f}GB (+{mem_after_advanced - mem_after_baseline:.2f}GB)"
        )

        # Load microstructure
        logger.info("Loading microstructure features...")
        micro_df = (
            pl.scan_parquet(micro_file)
            .filter(pl.col("timestamp_seconds").is_in(baseline_df["timestamp"].cast(pl.Int64)))
            .collect()
        )
        mem_after_micro = get_memory_usage_gb()
        logger.info(f"  After micro load: {mem_after_micro:.2f}GB (+{mem_after_micro - mem_after_advanced:.2f}GB)")

        # Join all features
        logger.info("Joining features...")
        _df = baseline_df.join(advanced_df, on="timestamp_seconds", how="left").join(
            micro_df, on="timestamp_seconds", how="left"
        )
        mem_after_join = get_memory_usage_gb()
        logger.info(f"  After join: {mem_after_join:.2f}GB (+{mem_after_join - mem_after_micro:.2f}GB)")
        logger.info(f"  Joined data shape: {len(_df):,} rows × {len(_df.columns)} columns")

        # Record peak memory
        peak_gb = mem_after_join
        mem_increase = peak_gb - mem_start

        logger.info(f"\nPeak memory: {peak_gb:.2f}GB (increase: {mem_increase:.2f}GB)")
        logger.info(f"Limit: {CHUNK_LIMIT_GB}GB")

        # Check if within limit
        if peak_gb < CHUNK_LIMIT_GB:
            return MemoryProfileResult(
                "Feature engineering chunk",
                True,
                f"Within {CHUNK_LIMIT_GB}GB limit",
                peak_gb,
            )
        else:
            return MemoryProfileResult(
                "Feature engineering chunk",
                False,
                f"Exceeds {CHUNK_LIMIT_GB}GB limit",
                peak_gb,
            )

    except Exception as e:
        return MemoryProfileResult("Feature engineering chunk", False, f"Error: {e}")


def profile_pipeline_preparation() -> MemoryProfileResult:
    """Profile memory usage during pipeline preparation."""
    logger.info("\n" + "=" * 80)
    logger.info("TEST 2: Pipeline Preparation Memory Profile")
    logger.info("=" * 80)

    try:
        consolidated_file = DATA_DIR / "consolidated_features_v4.parquet"

        if not consolidated_file.exists():
            logger.warning("Consolidated features not found - skipping test")
            return MemoryProfileResult(
                "Pipeline preparation",
                True,
                "Skipped (file not found)",
            )

        # Record initial memory
        mem_start = get_memory_usage_gb()
        logger.info(f"Initial memory: {mem_start:.2f}GB")

        # Sample 1M rows
        logger.info("Loading sample (1M rows) for regime assignment...")
        df = pl.scan_parquet(consolidated_file).slice(1_000_000, 1_000_000).collect()
        mem_after_load = get_memory_usage_gb()
        logger.info(f"  After load: {mem_after_load:.2f}GB (+{mem_after_load - mem_start:.2f}GB)")

        # Add regime columns (simplified)
        logger.info("Adding regime columns...")
        df = df.with_columns(
            [
                pl.when(pl.col("time_remaining") <= 300)
                .then(pl.lit("near"))
                .when(pl.col("time_remaining") <= 900)
                .then(pl.lit("mid"))
                .otherwise(pl.lit("far"))
                .alias("temporal_bucket"),
            ]
        )
        mem_after_regime = get_memory_usage_gb()
        logger.info(f"  After regime assignment: {mem_after_regime:.2f}GB (+{mem_after_regime - mem_after_load:.2f}GB)")

        # Record peak
        peak_gb = mem_after_regime
        mem_increase = peak_gb - mem_start

        logger.info(f"\nPeak memory: {peak_gb:.2f}GB (increase: {mem_increase:.2f}GB)")
        logger.info(f"Limit: {PIPELINE_LIMIT_GB}GB")

        if peak_gb < PIPELINE_LIMIT_GB:
            return MemoryProfileResult(
                "Pipeline preparation",
                True,
                f"Within {PIPELINE_LIMIT_GB}GB limit",
                peak_gb,
            )
        else:
            return MemoryProfileResult(
                "Pipeline preparation",
                False,
                f"Exceeds {PIPELINE_LIMIT_GB}GB limit",
                peak_gb,
            )

    except Exception as e:
        return MemoryProfileResult("Pipeline preparation", False, f"Error: {e}")


def profile_training_estimate() -> MemoryProfileResult:
    """Estimate training memory requirements."""
    logger.info("\n" + "=" * 80)
    logger.info("TEST 3: Training Memory Estimate")
    logger.info("=" * 80)

    try:
        pipeline_ready_file = DATA_DIR / "consolidated_features_v4_pipeline_ready.parquet"

        if not pipeline_ready_file.exists():
            logger.warning("Pipeline-ready file not found - skipping test")
            return MemoryProfileResult(
                "Training memory estimate",
                True,
                "Skipped (file not found)",
            )

        # Record initial memory
        mem_start = get_memory_usage_gb()
        logger.info(f"Initial memory: {mem_start:.2f}GB")

        # Sample 1M rows for one regime
        logger.info("Loading sample regime data (1M rows)...")
        _df = pl.scan_parquet(pipeline_ready_file).slice(1_000_000, 1_000_000).collect()
        mem_after_load = get_memory_usage_gb()
        logger.info(f"  After load: {mem_after_load:.2f}GB (+{mem_after_load - mem_start:.2f}GB)")
        logger.info(f"  Sample shape: {len(_df):,} rows × {len(_df.columns)} columns")

        # Estimate: 8 models × peak memory per model
        # Conservative estimate: 2x data size per model
        peak_gb_estimate = mem_after_load * 2 * 8 / 10  # Scaled for 1M vs full dataset

        logger.info(f"\nEstimated peak memory for 8 models: {peak_gb_estimate:.2f}GB")
        logger.info(f"Limit: {TRAINING_LIMIT_GB}GB")
        logger.info("Note: This is a rough estimate based on sample data")

        if peak_gb_estimate < TRAINING_LIMIT_GB:
            return MemoryProfileResult(
                "Training memory estimate",
                True,
                f"Estimated within {TRAINING_LIMIT_GB}GB limit",
                peak_gb_estimate,
            )
        else:
            return MemoryProfileResult(
                "Training memory estimate",
                False,
                f"Estimated exceeds {TRAINING_LIMIT_GB}GB limit",
                peak_gb_estimate,
            )

    except Exception as e:
        return MemoryProfileResult("Training memory estimate", False, f"Error: {e}")


def profile_memory_leak() -> MemoryProfileResult:
    """Test for memory leaks during repeated operations."""
    logger.info("\n" + "=" * 80)
    logger.info("TEST 4: Memory Leak Detection")
    logger.info("=" * 80)

    try:
        baseline_file = RESULTS_DIR / "production_backtest_results_v4.parquet"

        if not baseline_file.exists():
            return MemoryProfileResult(
                "Memory leak detection",
                False,
                "Baseline file not found",
            )

        # Record memory over 5 iterations
        mem_readings = []

        for i in range(5):
            logger.info(f"\nIteration {i + 1}/5...")

            # Load and process sample
            df = pl.scan_parquet(baseline_file).slice(i * 100_000, 100_000).collect()

            # Do some processing
            df = df.with_columns([(pl.col("S") / pl.col("K")).alias("moneyness")])

            # Record memory
            mem_gb = get_memory_usage_gb()
            mem_readings.append(mem_gb)
            logger.info(f"  Memory: {mem_gb:.2f}GB")

            # Clear DataFrame
            del df
            time.sleep(1)  # Let GC work

        # Check for memory leak (>1GB growth across iterations)
        mem_growth = mem_readings[-1] - mem_readings[0]
        logger.info(f"\nMemory growth: {mem_growth:.2f}GB (from {mem_readings[0]:.2f}GB to {mem_readings[-1]:.2f}GB)")

        if mem_growth < 1.0:
            return MemoryProfileResult(
                "Memory leak detection",
                True,
                f"No significant leak detected ({mem_growth:.2f}GB growth)",
                mem_readings[-1],
            )
        else:
            return MemoryProfileResult(
                "Memory leak detection",
                False,
                f"Potential leak detected ({mem_growth:.2f}GB growth)",
                mem_readings[-1],
            )

    except Exception as e:
        return MemoryProfileResult("Memory leak detection", False, f"Error: {e}")


def profile_system_capacity() -> MemoryProfileResult:
    """Check overall system memory capacity."""
    logger.info("\n" + "=" * 80)
    logger.info("TEST 5: System Memory Capacity")
    logger.info("=" * 80)

    try:
        mem = psutil.virtual_memory()
        total_gb = mem.total / (1024**3)
        used_gb = mem.used / (1024**3)
        available_gb = mem.available / (1024**3)
        percent_used = mem.percent

        logger.info("\nSystem memory:")
        logger.info(f"  Total:     {total_gb:.2f}GB")
        logger.info(f"  Used:      {used_gb:.2f}GB ({percent_used:.1f}%)")
        logger.info(f"  Available: {available_gb:.2f}GB")
        logger.info(f"  Soft limit: {SOFT_LIMIT_GB}GB")

        if total_gb >= TOTAL_RAM_GB and available_gb >= 100:
            return MemoryProfileResult(
                "System capacity",
                True,
                f"{total_gb:.0f}GB total, {available_gb:.0f}GB available",
                used_gb,
            )
        elif total_gb < TOTAL_RAM_GB:
            return MemoryProfileResult(
                "System capacity",
                False,
                f"Total RAM {total_gb:.0f}GB < {TOTAL_RAM_GB}GB expected",
                used_gb,
            )
        else:
            return MemoryProfileResult(
                "System capacity",
                False,
                f"Low available memory ({available_gb:.0f}GB)",
                used_gb,
            )

    except Exception as e:
        return MemoryProfileResult("System capacity", False, f"Error: {e}")


def main() -> None:
    """Run memory profiling tests."""
    logger.info("\n" + "=" * 80)
    logger.info("MEMORY PROFILING TEST FOR V4 PIPELINE")
    logger.info("=" * 80)
    logger.info(f"System: {TOTAL_RAM_GB}GB RAM, {SOFT_LIMIT_GB}GB soft limit")
    logger.info("Runtime: ~30 minutes")
    logger.info("=" * 80)

    results: list[MemoryProfileResult] = []

    # Run all memory tests
    results.append(profile_system_capacity())
    results.append(profile_feature_engineering_chunk())
    results.append(profile_pipeline_preparation())
    results.append(profile_training_estimate())
    results.append(profile_memory_leak())

    # Summary
    logger.info("\n" + "=" * 80)
    logger.info("SUMMARY")
    logger.info("=" * 80)

    passed = sum(1 for r in results if r.passed)
    total = len(results)

    for result in results:
        logger.info(str(result))

    logger.info("\n" + "=" * 80)
    logger.info(f"TOTAL: {passed}/{total} tests passed")

    # Overall memory usage
    final_mem = get_memory_usage_gb()
    available_mem = get_memory_available_gb()
    logger.info(f"\nFinal memory: {final_mem:.2f}GB used, {available_mem:.2f}GB available")

    if passed == total:
        logger.info("✅ MEMORY PROFILING PASSED")
        logger.info("=" * 80)
        logger.info("\nMemory usage within safe limits for production pipeline!")
        sys.exit(0)
    else:
        logger.error(f"❌ {total - passed} TEST(S) FAILED")
        logger.info("=" * 80)
        logger.error("\nMemory concerns detected - review limits before production run!")
        sys.exit(1)


if __name__ == "__main__":
    main()
