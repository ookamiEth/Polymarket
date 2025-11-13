#!/usr/bin/env python3
"""
Pre-Flight Validation for V4 Pipeline
======================================

Validates all prerequisites before starting the V4 feature engineering and training pipeline.

Tests:
1. Input file existence (6 required parquet files)
2. Input schema validation (required columns and dtypes)
3. Row count alignment (baseline vs RV vs micro)
4. Disk space check (need 120GB free)
5. Memory check (256GB system)
6. Python environment (package versions)

Runtime: ~5 minutes

Author: BT Research Team
Date: 2025-11-13
"""

from __future__ import annotations

import logging
import shutil
import sys
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
TARDIS_DIR = MODEL_DIR.parent / "tardis/data/consolidated"

# Required input files
REQUIRED_FILES = {
    "baseline": RESULTS_DIR / "production_backtest_results_v4.parquet",
    "rv": RESULTS_DIR / "realized_volatility_1s.parquet",
    "micro": RESULTS_DIR / "microstructure_features.parquet",
    "advanced": RESULTS_DIR / "advanced_features.parquet",
    "funding": TARDIS_DIR / "binance_funding_rates_1s_consolidated.parquet",
    "orderbook": TARDIS_DIR / "binance_orderbook_5_1s_consolidated.parquet",
}

# Required baseline columns
BASELINE_REQUIRED_COLS = {
    "timestamp": pl.Int64,
    "time_remaining": pl.Int64,
    "S": pl.Float64,
    "K": pl.Float64,
    "sigma_mid": pl.Float64,
    "T_years": pl.Float64,
    "outcome": pl.Int8,
    "prob_mid": pl.Float64,
}


class TestResult:
    """Store test result with status and message."""

    def __init__(self, name: str, passed: bool, message: str = ""):
        self.name = name
        self.passed = passed
        self.message = message

    def __repr__(self) -> str:
        status = "✅ PASS" if self.passed else "❌ FAIL"
        return f"{status}: {self.name}" + (f" - {self.message}" if self.message else "")


def test_file_existence() -> list[TestResult]:
    """Test that all required input files exist."""
    logger.info("\n" + "=" * 80)
    logger.info("TEST 1: Input File Existence")
    logger.info("=" * 80)

    results = []

    for name, filepath in REQUIRED_FILES.items():
        exists = filepath.exists()
        size_gb = filepath.stat().st_size / (1024**3) if exists else 0

        if exists:
            results.append(
                TestResult(
                    f"File exists: {name}",
                    True,
                    f"Size: {size_gb:.2f}GB",
                )
            )
            logger.info(f"  ✓ {name}: {filepath} ({size_gb:.2f}GB)")
        else:
            results.append(
                TestResult(
                    f"File exists: {name}",
                    False,
                    f"Missing: {filepath}",
                )
            )
            logger.error(f"  ✗ {name}: {filepath} NOT FOUND")

    return results


def test_schema_validation() -> list[TestResult]:
    """Test that input files have required schemas."""
    logger.info("\n" + "=" * 80)
    logger.info("TEST 2: Schema Validation")
    logger.info("=" * 80)

    results = []

    # Test baseline schema
    logger.info("\nValidating baseline schema...")
    try:
        baseline_schema = pl.scan_parquet(REQUIRED_FILES["baseline"]).collect_schema()

        missing_cols = []
        wrong_dtypes = []

        for col, expected_dtype in BASELINE_REQUIRED_COLS.items():
            if col not in baseline_schema.names():
                missing_cols.append(col)
            elif baseline_schema[col] != expected_dtype:
                wrong_dtypes.append(f"{col} (expected {expected_dtype}, got {baseline_schema[col]})")

        if not missing_cols and not wrong_dtypes:
            results.append(TestResult("Baseline schema", True, "All required columns present"))
            logger.info("  ✓ All required columns present with correct dtypes")
        else:
            msg = ""
            if missing_cols:
                msg += f"Missing: {missing_cols}"
            if wrong_dtypes:
                msg += f" Wrong dtypes: {wrong_dtypes}"
            results.append(TestResult("Baseline schema", False, msg))
            logger.error(f"  ✗ Schema issues: {msg}")

    except Exception as e:
        results.append(TestResult("Baseline schema", False, f"Error reading schema: {e}"))
        logger.error(f"  ✗ Error reading baseline schema: {e}")

    # Test RV schema
    logger.info("\nValidating RV schema...")
    try:
        rv_schema = pl.scan_parquet(REQUIRED_FILES["rv"]).collect_schema()
        required_rv_cols = ["timestamp_seconds", "rv_60s", "rv_300s", "rv_900s", "rv_3600s"]

        missing = [col for col in required_rv_cols if col not in rv_schema.names()]

        if not missing:
            results.append(TestResult("RV schema", True, "All required columns present"))
            logger.info("  ✓ All required RV columns present")
        else:
            results.append(TestResult("RV schema", False, f"Missing: {missing}"))
            logger.error(f"  ✗ Missing RV columns: {missing}")

    except Exception as e:
        results.append(TestResult("RV schema", False, f"Error: {e}"))
        logger.error(f"  ✗ Error reading RV schema: {e}")

    # Test funding schema
    logger.info("\nValidating funding schema...")
    try:
        funding_schema = pl.scan_parquet(REQUIRED_FILES["funding"]).collect_schema()
        required_funding_cols = ["timestamp", "funding_rate", "mark_price", "index_price", "open_interest"]

        missing = [col for col in required_funding_cols if col not in funding_schema.names()]

        if not missing:
            results.append(TestResult("Funding schema", True, "All required columns present"))
            logger.info("  ✓ All required funding columns present")
        else:
            results.append(TestResult("Funding schema", False, f"Missing: {missing}"))
            logger.error(f"  ✗ Missing funding columns: {missing}")

    except Exception as e:
        results.append(TestResult("Funding schema", False, f"Error: {e}"))
        logger.error(f"  ✗ Error reading funding schema: {e}")

    return results


def test_row_count_alignment() -> list[TestResult]:
    """Test that row counts are aligned across input files."""
    logger.info("\n" + "=" * 80)
    logger.info("TEST 3: Row Count Alignment")
    logger.info("=" * 80)

    results = []

    try:
        baseline_rows = pl.scan_parquet(REQUIRED_FILES["baseline"]).select(pl.len()).collect().item()
        rv_rows = pl.scan_parquet(REQUIRED_FILES["rv"]).select(pl.len()).collect().item()
        micro_rows = pl.scan_parquet(REQUIRED_FILES["micro"]).select(pl.len()).collect().item()

        logger.info("\nRow counts:")
        logger.info(f"  Baseline: {baseline_rows:,}")
        logger.info(f"  RV:       {rv_rows:,}")
        logger.info(f"  Micro:    {micro_rows:,}")

        # Check baseline and RV alignment (baseline can have more rows due to warmup periods)
        row_diff = baseline_rows - rv_rows
        diff_pct = 100 * row_diff / baseline_rows if baseline_rows > 0 else 0

        if row_diff >= 0 and diff_pct < 10:  # Allow baseline to have up to 10% more rows
            results.append(
                TestResult(
                    "Baseline-RV alignment",
                    True,
                    f"Baseline: {baseline_rows:,}, RV: {rv_rows:,} ({diff_pct:.1f}% diff)",
                )
            )
            logger.info(f"  ✓ Baseline has {row_diff:,} more rows than RV ({diff_pct:.1f}% - acceptable)")
        else:
            results.append(
                TestResult(
                    "Baseline-RV alignment",
                    False,
                    f"Large mismatch: baseline={baseline_rows:,}, rv={rv_rows:,} ({diff_pct:.1f}%)",
                )
            )
            logger.error(f"  ✗ Row count mismatch too large: {diff_pct:.1f}% (expected <10%)")

        # Check micro and RV match (they should be aligned)
        if micro_rows == rv_rows:
            results.append(TestResult("Micro-RV alignment", True, f"{micro_rows:,} rows"))
            logger.info(f"  ✓ Micro and RV row counts match: {micro_rows:,}")
        else:
            micro_rv_diff = abs(micro_rows - rv_rows)
            results.append(
                TestResult(
                    "Micro-RV alignment",
                    False,
                    f"Mismatch: micro={micro_rows:,}, rv={rv_rows:,} (diff={micro_rv_diff:,})",
                )
            )
            logger.warning(f"  ⚠ Micro-RV mismatch: {micro_rv_diff:,} rows")

        # Check baseline row count is in expected range (updated for V4)
        if 60_000_000 <= baseline_rows <= 70_000_000:
            results.append(TestResult("Baseline row count", True, f"{baseline_rows:,} rows"))
            logger.info(f"  ✓ Baseline row count in expected range: {baseline_rows:,}")
        else:
            results.append(
                TestResult(
                    "Baseline row count",
                    False,
                    f"{baseline_rows:,} rows (expected 60M-70M)",
                )
            )
            logger.error(f"  ✗ Baseline row count outside expected range: {baseline_rows:,}")

    except Exception as e:
        results.append(TestResult("Row count alignment", False, f"Error: {e}"))
        logger.error(f"  ✗ Error checking row counts: {e}")

    return results


def test_disk_space() -> list[TestResult]:
    """Test that sufficient disk space is available."""
    logger.info("\n" + "=" * 80)
    logger.info("TEST 4: Disk Space")
    logger.info("=" * 80)

    results = []

    try:
        stat = shutil.disk_usage(DATA_DIR)
        free_gb = stat.free / (1024**3)
        total_gb = stat.total / (1024**3)
        used_gb = stat.used / (1024**3)

        logger.info(f"\nDisk usage for {DATA_DIR}:")
        logger.info(f"  Total: {total_gb:.1f}GB")
        logger.info(f"  Used:  {used_gb:.1f}GB ({100 * used_gb / total_gb:.1f}%)")
        logger.info(f"  Free:  {free_gb:.1f}GB ({100 * free_gb / total_gb:.1f}%)")

        # Need at least 120GB free (59GB × 2 for intermediate + final)
        if free_gb >= 120:
            results.append(TestResult("Disk space", True, f"{free_gb:.1f}GB free"))
            logger.info(f"  ✓ Sufficient disk space: {free_gb:.1f}GB free")
        else:
            results.append(TestResult("Disk space", False, f"Only {free_gb:.1f}GB free (need 120GB)"))
            logger.error(f"  ✗ Insufficient disk space: {free_gb:.1f}GB free (need 120GB)")

    except Exception as e:
        results.append(TestResult("Disk space", False, f"Error: {e}"))
        logger.error(f"  ✗ Error checking disk space: {e}")

    return results


def test_memory() -> list[TestResult]:
    """Test that system has sufficient memory."""
    logger.info("\n" + "=" * 80)
    logger.info("TEST 5: Memory")
    logger.info("=" * 80)

    results = []

    try:
        mem = psutil.virtual_memory()
        total_gb = mem.total / (1024**3)
        available_gb = mem.available / (1024**3)
        used_gb = mem.used / (1024**3)

        logger.info("\nSystem memory:")
        logger.info(f"  Total:     {total_gb:.1f}GB")
        logger.info(f"  Used:      {used_gb:.1f}GB ({mem.percent:.1f}%)")
        logger.info(f"  Available: {available_gb:.1f}GB")

        # Need at least 200GB total (256GB system)
        if total_gb >= 200:
            results.append(TestResult("Total memory", True, f"{total_gb:.1f}GB total"))
            logger.info(f"  ✓ Sufficient total memory: {total_gb:.1f}GB")
        else:
            results.append(TestResult("Total memory", False, f"Only {total_gb:.1f}GB (need 200GB+)"))
            logger.error(f"  ✗ Insufficient total memory: {total_gb:.1f}GB (need 200GB+)")

        # Should have at least 100GB available to start
        if available_gb >= 100:
            results.append(TestResult("Available memory", True, f"{available_gb:.1f}GB available"))
            logger.info(f"  ✓ Sufficient available memory: {available_gb:.1f}GB")
        else:
            results.append(TestResult("Available memory", False, f"Only {available_gb:.1f}GB available (need 100GB+)"))
            logger.warning(f"  ⚠ Low available memory: {available_gb:.1f}GB (consider freeing memory)")

    except Exception as e:
        results.append(TestResult("Memory check", False, f"Error: {e}"))
        logger.error(f"  ✗ Error checking memory: {e}")

    return results


def test_python_environment() -> list[TestResult]:
    """Test Python environment and package versions."""
    logger.info("\n" + "=" * 80)
    logger.info("TEST 6: Python Environment")
    logger.info("=" * 80)

    results = []

    # Check Python version
    py_version = sys.version_info
    logger.info(f"\nPython version: {py_version.major}.{py_version.minor}.{py_version.micro}")

    if py_version.major == 3 and py_version.minor >= 10:
        results.append(TestResult("Python version", True, f"{py_version.major}.{py_version.minor}"))
        logger.info("  ✓ Python 3.10+ detected")
    else:
        results.append(TestResult("Python version", False, f"{py_version.major}.{py_version.minor} (need 3.10+)"))
        logger.error("  ✗ Need Python 3.10+")

    # Check required packages
    required_packages = ["polars", "lightgbm", "numpy", "yaml", "psutil"]

    for package in required_packages:
        try:
            mod = __import__(package)
            version = getattr(mod, "__version__", "unknown")
            results.append(TestResult(f"Package: {package}", True, f"v{version}"))
            logger.info(f"  ✓ {package}: v{version}")
        except ImportError:
            results.append(TestResult(f"Package: {package}", False, "Not installed"))
            logger.error(f"  ✗ {package}: Not installed")

    return results


def main() -> None:
    """Run all pre-flight tests."""
    logger.info("\n" + "=" * 80)
    logger.info("V4 PIPELINE PRE-FLIGHT VALIDATION")
    logger.info("=" * 80)
    logger.info("Runtime: ~5 minutes")
    logger.info("=" * 80)

    all_results: list[TestResult] = []

    # Run all tests
    all_results.extend(test_file_existence())
    all_results.extend(test_schema_validation())
    all_results.extend(test_row_count_alignment())
    all_results.extend(test_disk_space())
    all_results.extend(test_memory())
    all_results.extend(test_python_environment())

    # Summary
    logger.info("\n" + "=" * 80)
    logger.info("SUMMARY")
    logger.info("=" * 80)

    passed = sum(1 for r in all_results if r.passed)
    total = len(all_results)

    for result in all_results:
        logger.info(str(result))

    logger.info("\n" + "=" * 80)
    logger.info(f"TOTAL: {passed}/{total} tests passed")

    if passed == total:
        logger.info("✅ ALL PRE-FLIGHT TESTS PASSED")
        logger.info("=" * 80)
        logger.info("\nReady to proceed to module unit tests (test_02_modules_v4.py)")
        sys.exit(0)
    else:
        logger.error(f"❌ {total - passed} TEST(S) FAILED")
        logger.info("=" * 80)
        logger.error("\nFix failures before proceeding!")
        sys.exit(1)


if __name__ == "__main__":
    main()
