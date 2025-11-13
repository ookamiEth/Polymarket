#!/usr/bin/env python3
"""
Checkpoint 5.2 Validator: Pipeline Preparation Output
=====================================================

Validates consolidated_features_v4_pipeline_ready.parquet after pipeline preparation.

Checks:
- File exists
- Residual column added
- Combined_regime column added
- Date column added
- Outcome column preserved
- 8 expected regimes present
- Each regime has >100K samples (minimum threshold)
- Row count matches input

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
OUTPUT_FILE = DATA_DIR / "consolidated_features_v4_pipeline_ready.parquet"

# Expected regimes
EXPECTED_REGIMES = [
    "near_low_vol_atm",
    "near_low_vol_otm",
    "near_high_vol_atm",
    "near_high_vol_otm",
    "mid_low_vol_atm",
    "mid_low_vol_otm",
    "mid_high_vol_atm",
    "mid_high_vol_otm",
]


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


def validate_required_columns() -> list[ValidationResult]:
    """Validate required columns added by pipeline preparation."""
    logger.info("\n" + "=" * 80)
    logger.info("CHECK 2: Required Columns")
    logger.info("=" * 80)

    results = []

    try:
        schema = pl.scan_parquet(OUTPUT_FILE).collect_schema()
        col_names = schema.names()

        # Check residual column
        if "residual" in col_names:
            results.append(ValidationResult("Residual column", True, "Present"))
            logger.info("  ✓ residual column present")
        else:
            results.append(ValidationResult("Residual column", False, "Missing"))
            logger.error("  ✗ residual column missing")

        # Check combined_regime column
        if "combined_regime" in col_names:
            results.append(ValidationResult("Combined_regime column", True, "Present"))
            logger.info("  ✓ combined_regime column present")
        else:
            results.append(ValidationResult("Combined_regime column", False, "Missing"))
            logger.error("  ✗ combined_regime column missing")

        # Check date column
        if "date" in col_names:
            results.append(ValidationResult("Date column", True, "Present"))
            logger.info("  ✓ date column present")
        else:
            results.append(ValidationResult("Date column", False, "Missing"))
            logger.error("  ✗ date column missing")

        # Check outcome column preserved
        if "outcome" in col_names:
            results.append(ValidationResult("Outcome column", True, "Present"))
            logger.info("  ✓ outcome column preserved")
        else:
            results.append(ValidationResult("Outcome column", False, "Missing"))
            logger.error("  ✗ outcome column missing")

    except Exception as e:
        results.append(ValidationResult("Required columns", False, f"Error: {e}"))
        logger.error(f"  ✗ Error checking columns: {e}")

    return results


def validate_regime_distribution() -> list[ValidationResult]:
    """Validate regime distribution."""
    logger.info("\n" + "=" * 80)
    logger.info("CHECK 3: Regime Distribution")
    logger.info("=" * 80)

    results = []

    try:
        # Load regime counts
        df = pl.read_parquet(OUTPUT_FILE, columns=["combined_regime"])
        regime_counts = df.group_by("combined_regime").agg(pl.len().alias("count")).sort("count", descending=True)

        total_rows = len(df)

        logger.info(f"\nTotal rows: {total_rows:,}")
        logger.info("\nRegime distribution:")

        # Display all regimes
        for row in regime_counts.iter_rows(named=True):
            regime = row["combined_regime"]
            count = row["count"]
            pct = 100 * count / total_rows
            logger.info(f"  {regime:25s}: {count:10,} ({pct:5.2f}%)")

        # Check all expected regimes present
        actual_regimes = set(regime_counts["combined_regime"].to_list())
        missing_regimes = set(EXPECTED_REGIMES) - actual_regimes

        if not missing_regimes:
            results.append(ValidationResult("All regimes present", True, "8/8 regimes found"))
            logger.info("\n  ✓ All 8 expected regimes present")
        else:
            results.append(
                ValidationResult(
                    "All regimes present",
                    False,
                    f"Missing: {missing_regimes}",
                )
            )
            logger.error(f"\n  ✗ Missing regimes: {missing_regimes}")

        # Check minimum sample threshold (100K)
        low_sample_regimes = regime_counts.filter(pl.col("count") < 100_000)

        if len(low_sample_regimes) == 0:
            results.append(ValidationResult("Minimum samples", True, "All regimes >100K samples"))
            logger.info("  ✓ All regimes have >100K samples")
        else:
            low_list = []
            for row in low_sample_regimes.iter_rows(named=True):
                low_list.append(f"{row['combined_regime']} ({row['count']:,})")

            results.append(
                ValidationResult(
                    "Minimum samples",
                    False,
                    f"Regimes <100K: {', '.join(low_list)}",
                )
            )
            logger.warning("\n  ⚠ Regimes below 100K threshold:")
            for regime_info in low_list:
                logger.warning(f"    {regime_info}")

        # Check regime balance (no regime should dominate >50%)
        max_count_series = regime_counts["count"]
        if len(max_count_series) > 0:
            max_count = int(max_count_series.to_list()[0]) if len(max_count_series) > 0 else max_count_series.max()
            max_count = int(max_count) if isinstance(max_count, (int, float)) else 0
        else:
            max_count = 0
        max_pct = float(100 * max_count / total_rows) if total_rows > 0 else 0.0

        if max_pct < 50:
            results.append(ValidationResult("Regime balance", True, f"Max regime: {max_pct:.1f}%"))
            logger.info(f"  ✓ Regime balance good (max: {max_pct:.1f}%)")
        else:
            results.append(ValidationResult("Regime balance", False, f"Max regime: {max_pct:.1f}% (>50%)"))
            logger.warning(f"  ⚠ Regime imbalance (max: {max_pct:.1f}%)")

    except Exception as e:
        results.append(ValidationResult("Regime distribution", False, f"Error: {e}"))
        logger.error(f"  ✗ Error checking regimes: {e}")

    return results


def validate_residual_computation() -> ValidationResult:
    """Validate residual computation (outcome - prob_mid)."""
    logger.info("\n" + "=" * 80)
    logger.info("CHECK 4: Residual Computation")
    logger.info("=" * 80)

    try:
        # Sample and check residual = outcome - prob_mid
        df = pl.read_parquet(OUTPUT_FILE, columns=["outcome", "prob_mid", "residual"]).head(10000)

        # Compute expected residual
        df = df.with_columns([(pl.col("outcome") - pl.col("prob_mid")).alias("expected_residual")])

        # Check match (allow small floating point error)
        diff_series = (df["residual"] - df["expected_residual"]).abs()
        max_diff_val = diff_series.max()
        # Ensure we get a numeric value (handle Polars return types)
        max_diff = float(max_diff_val) if isinstance(max_diff_val, (int, float)) else 0.0

        logger.info(f"Max residual difference: {max_diff:.10f}")

        if max_diff < 1e-6:
            return ValidationResult("Residual computation", True, f"Max diff: {max_diff:.2e}")
        else:
            return ValidationResult(
                "Residual computation",
                False,
                f"Max diff: {max_diff:.2e} (expected <1e-6)",
            )

    except Exception as e:
        return ValidationResult("Residual computation", False, f"Error: {e}")


def validate_row_count() -> ValidationResult:
    """Validate row count hasn't changed from consolidated_features_v4.parquet."""
    logger.info("\n" + "=" * 80)
    logger.info("CHECK 5: Row Count")
    logger.info("=" * 80)

    try:
        input_file = DATA_DIR / "consolidated_features_v4.parquet"

        if not input_file.exists():
            return ValidationResult("Row count", False, "Input file not found for comparison")

        input_rows = pl.scan_parquet(input_file).select(pl.len()).collect().item()
        output_rows = pl.scan_parquet(OUTPUT_FILE).select(pl.len()).collect().item()

        logger.info(f"Input rows:  {input_rows:,}")
        logger.info(f"Output rows: {output_rows:,}")

        # Allow small difference due to warmup period drop
        row_diff = abs(output_rows - input_rows)
        pct_diff = 100 * row_diff / input_rows

        if pct_diff < 5:  # Allow up to 5% drop (warmup period)
            return ValidationResult("Row count", True, f"{output_rows:,} rows ({pct_diff:.2f}% diff)")
        else:
            return ValidationResult(
                "Row count",
                False,
                f"{output_rows:,} rows ({pct_diff:.2f}% diff, expected <5%)",
            )

    except Exception as e:
        return ValidationResult("Row count", False, f"Error: {e}")


def main() -> None:
    """Run all checkpoint 5.2 validations."""
    logger.info("\n" + "=" * 80)
    logger.info("CHECKPOINT 5.2: Pipeline Preparation Output Validation")
    logger.info("=" * 80)
    logger.info(f"Validating: {OUTPUT_FILE}")
    logger.info("=" * 80)

    results: list[ValidationResult] = []

    # Run all checks
    results.append(validate_file_exists())

    if not results[0].passed:
        logger.error("\n❌ File not found - cannot continue validation")
        sys.exit(1)

    results.extend(validate_required_columns())
    results.extend(validate_regime_distribution())
    results.append(validate_residual_computation())
    results.append(validate_row_count())

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
        logger.info("✅ CHECKPOINT 5.2 PASSED")
        logger.info("=" * 80)
        logger.info("\nReady to proceed to training (train_multi_horizon_v4.py)")
        sys.exit(0)
    else:
        logger.error(f"❌ {total - passed} CHECK(S) FAILED")
        logger.info("=" * 80)
        logger.error("\nFix issues before proceeding!")
        sys.exit(1)


if __name__ == "__main__":
    main()
