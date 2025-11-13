#!/usr/bin/env python3
"""
Checkpoint 5.1 Validator: Feature Engineering Output
====================================================

Validates consolidated_features_v4.parquet after feature engineering completes.

Checks:
- Row count (60M-65M expected)
- Feature count (150-160 expected)
- File size (55-65GB expected)
- Sample null rate (<1%)
- No duplicate columns
- Feature quality (no inf/nan values)

Runtime: ~2 minutes

Author: BT Research Team
Date: 2025-11-13
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import polars as pl

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
OUTPUT_FILE = DATA_DIR / "consolidated_features_v4.parquet"


class ValidationResult:
    """Store validation result."""

    def __init__(self, check: str, passed: bool, message: str = ""):
        self.check = check
        self.passed = passed
        self.message = message

    def __repr__(self) -> str:
        status = "✅" if self.passed else "❌"
        return f"{status} {self.check}: {self.message}"


def validate_file_exists() -> ValidationResult:
    """Check that output file exists."""
    logger.info("\n" + "=" * 80)
    logger.info("CHECK 1: File Existence")
    logger.info("=" * 80)

    if not OUTPUT_FILE.exists():
        return ValidationResult("File exists", False, f"File not found: {OUTPUT_FILE}")

    size_gb = OUTPUT_FILE.stat().st_size / (1024**3)
    logger.info(f"File found: {OUTPUT_FILE}")
    logger.info(f"File size: {size_gb:.2f}GB")

    return ValidationResult("File exists", True, f"{size_gb:.2f}GB")


def validate_row_count() -> ValidationResult:
    """Validate row count is in expected range."""
    logger.info("\n" + "=" * 80)
    logger.info("CHECK 2: Row Count")
    logger.info("=" * 80)

    try:
        row_count = pl.scan_parquet(OUTPUT_FILE).select(pl.len()).collect().item()

        logger.info(f"Total rows: {row_count:,}")

        if 60_000_000 <= row_count <= 70_000_000:
            return ValidationResult("Row count", True, f"{row_count:,} rows (expected 60M-70M)")
        else:
            return ValidationResult(
                "Row count",
                False,
                f"{row_count:,} rows (expected 60M-70M)",
            )

    except Exception as e:
        return ValidationResult("Row count", False, f"Error: {e}")


def validate_feature_count() -> ValidationResult:
    """Validate feature count is in expected range."""
    logger.info("\n" + "=" * 80)
    logger.info("CHECK 3: Feature Count")
    logger.info("=" * 80)

    try:
        schema = pl.scan_parquet(OUTPUT_FILE).collect_schema()
        col_count = len(schema.names())

        logger.info(f"Total columns: {col_count}")
        logger.info(f"Feature count (excluding timestamp): {col_count - 1}")

        # List first 20 columns
        logger.info("\nFirst 20 columns:")
        for i, col in enumerate(schema.names()[:20], 1):
            logger.info(f"  {i:2d}. {col} ({schema[col]})")

        if 150 <= col_count <= 165:
            return ValidationResult("Feature count", True, f"{col_count} columns (expected 150-165)")
        else:
            return ValidationResult(
                "Feature count",
                False,
                f"{col_count} columns (expected 150-165)",
            )

    except Exception as e:
        return ValidationResult("Feature count", False, f"Error: {e}")


def validate_no_duplicates() -> ValidationResult:
    """Validate no duplicate columns."""
    logger.info("\n" + "=" * 80)
    logger.info("CHECK 4: Duplicate Columns")
    logger.info("=" * 80)

    try:
        schema = pl.scan_parquet(OUTPUT_FILE).collect_schema()
        col_names = schema.names()
        unique_cols = set(col_names)

        if len(col_names) == len(unique_cols):
            return ValidationResult("No duplicates", True, f"All {len(col_names)} columns unique")
        else:
            # Find duplicates
            from collections import Counter

            counts = Counter(col_names)
            duplicates = {col: count for col, count in counts.items() if count > 1}

            return ValidationResult(
                "No duplicates",
                False,
                f"Found duplicates: {duplicates}",
            )

    except Exception as e:
        return ValidationResult("No duplicates", False, f"Error: {e}")


def validate_null_rate() -> ValidationResult:
    """Validate null rate on sample."""
    logger.info("\n" + "=" * 80)
    logger.info("CHECK 5: Null Rate (Sample)")
    logger.info("=" * 80)

    try:
        # Sample 100K rows from middle (avoid warmup period)
        df = pl.scan_parquet(OUTPUT_FILE).slice(1_000_000, 100_000).collect()

        total_cells = len(df) * len(df.columns)
        null_cells = sum(df[col].null_count() for col in df.columns)
        null_rate = null_cells / total_cells

        logger.info(f"Sample size: {len(df):,} rows × {len(df.columns)} columns")
        logger.info(f"Null cells: {null_cells:,} / {total_cells:,} ({null_rate:.2%})")

        # Check per-column nulls
        high_null_cols = []
        for col in df.columns:
            col_null_rate = df[col].null_count() / len(df)
            if col_null_rate > 0.01:  # >1% nulls
                high_null_cols.append(f"{col} ({col_null_rate:.2%})")

        if high_null_cols:
            logger.warning("\nColumns with >1% nulls:")
            for col_info in high_null_cols[:10]:  # Show first 10
                logger.warning(f"  {col_info}")

        if null_rate < 0.01:
            return ValidationResult("Null rate", True, f"{null_rate:.2%} nulls in sample")
        else:
            return ValidationResult(
                "Null rate",
                False,
                f"{null_rate:.2%} nulls (expected <1%)",
            )

    except Exception as e:
        return ValidationResult("Null rate", False, f"Error: {e}")


def validate_feature_quality() -> ValidationResult:
    """Validate no inf/nan values on sample."""
    logger.info("\n" + "=" * 80)
    logger.info("CHECK 6: Feature Quality (Sample)")
    logger.info("=" * 80)

    try:
        # Sample 100K rows
        df = pl.scan_parquet(OUTPUT_FILE).slice(1_000_000, 100_000).collect()

        # Check inf/nan for numeric columns only
        numeric_cols = [col for col in df.columns if df[col].dtype in [pl.Float32, pl.Float64]]

        logger.info(f"Checking {len(numeric_cols)} numeric columns for inf/nan...")

        inf_cols = []
        nan_cols = []

        for col in numeric_cols:
            inf_count = df[col].is_infinite().sum()
            nan_count = df[col].is_nan().sum()

            if inf_count > 0:
                inf_cols.append(f"{col} ({inf_count})")
            if nan_count > 0:
                nan_cols.append(f"{col} ({nan_count})")

        issues = []
        if inf_cols:
            issues.append(f"Inf values: {len(inf_cols)} columns")
            logger.warning("\nColumns with inf values:")
            for col_info in inf_cols[:5]:
                logger.warning(f"  {col_info}")

        if nan_cols:
            issues.append(f"NaN values: {len(nan_cols)} columns")
            logger.warning("\nColumns with NaN values:")
            for col_info in nan_cols[:5]:
                logger.warning(f"  {col_info}")

        if not issues:
            return ValidationResult("Feature quality", True, "No inf/nan values in sample")
        else:
            return ValidationResult(
                "Feature quality",
                False,
                ", ".join(issues),
            )

    except Exception as e:
        return ValidationResult("Feature quality", False, f"Error: {e}")


def validate_file_size() -> ValidationResult:
    """Validate file size is in expected range."""
    logger.info("\n" + "=" * 80)
    logger.info("CHECK 7: File Size")
    logger.info("=" * 80)

    try:
        size_gb = OUTPUT_FILE.stat().st_size / (1024**3)

        logger.info(f"File size: {size_gb:.2f}GB")

        if 50 <= size_gb <= 70:
            return ValidationResult("File size", True, f"{size_gb:.2f}GB (expected 50-70GB)")
        else:
            return ValidationResult(
                "File size",
                False,
                f"{size_gb:.2f}GB (expected 50-70GB)",
            )

    except Exception as e:
        return ValidationResult("File size", False, f"Error: {e}")


def main() -> None:
    """Run all checkpoint 5.1 validations."""
    logger.info("\n" + "=" * 80)
    logger.info("CHECKPOINT 5.1: Feature Engineering Output Validation")
    logger.info("=" * 80)
    logger.info(f"Validating: {OUTPUT_FILE}")
    logger.info("=" * 80)

    results: list[ValidationResult] = []

    # Run all checks
    results.append(validate_file_exists())

    if not results[0].passed:
        logger.error("\n❌ File not found - cannot continue validation")
        sys.exit(1)

    results.append(validate_row_count())
    results.append(validate_feature_count())
    results.append(validate_no_duplicates())
    results.append(validate_null_rate())
    results.append(validate_feature_quality())
    results.append(validate_file_size())

    # Summary
    logger.info("\n" + "=" * 80)
    logger.info("SUMMARY")
    logger.info("=" * 80)

    for result in results:
        logger.info(str(result))

    passed = sum(1 for r in results if r.passed)
    total = len(results)

    logger.info("\n" + "=" * 80)
    logger.info(f"TOTAL: {passed}/{total} checks passed")

    if passed == total:
        logger.info("✅ CHECKPOINT 5.1 PASSED")
        logger.info("=" * 80)
        logger.info("\nReady to proceed to pipeline preparation (prepare_pipeline_data_v4.py)")
        sys.exit(0)
    else:
        logger.error(f"❌ {total - passed} CHECK(S) FAILED")
        logger.info("=" * 80)
        logger.error("\nFix issues before proceeding!")
        sys.exit(1)


if __name__ == "__main__":
    main()
