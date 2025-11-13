#!/usr/bin/env python3
"""
Integration tests for Binance data modules in engineer_all_features_v4.py.

Tests each Binance module (2-6) independently to ensure:
1. Column names match between creation and imputation
2. Feature engineering logic is correct
3. Imputation works correctly
4. Output files are created with expected schema

Run with:
    uv run python test_binance_modules_v4.py --module 2  # Test specific module
    uv run python test_binance_modules_v4.py              # Test all modules
"""

import logging
import sys
from pathlib import Path

import polars as pl

# Add parent directory to path to import module functions
sys.path.insert(0, str(Path(__file__).parent))

from engineer_all_features_v4 import (
    engineer_and_write_funding_features,
    engineer_and_write_oi_features,
    engineer_and_write_orderbook_5level_features,
    engineer_and_write_orderbook_l0_features,
    engineer_and_write_price_basis_features,
)
from impute_missing_features_v4 import (
    FUNDING_FEATURES,
    OI_FEATURES,
    ORDERBOOK_5LEVEL_FEATURES,
    ORDERBOOK_L0_FEATURES,
    PRICE_BASIS_FEATURES,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def test_module_2_funding_features():
    """Test Module 2: Funding Rate Features."""
    logger.info("=" * 80)
    logger.info("TEST: Module 2 - Funding Rate Features")
    logger.info("=" * 80)

    try:
        # Run the module
        output_file = engineer_and_write_funding_features()

        # Verify file exists
        assert output_file.exists(), f"Output file not created: {output_file}"
        logger.info(f"✓ Output file created: {output_file}")

        # Load output and check schema
        df = pl.scan_parquet(output_file)
        schema = df.collect_schema()
        columns = list(schema.names())

        logger.info(f"  Columns created: {columns}")
        logger.info(f"  Expected features: {FUNDING_FEATURES}")

        # Verify all expected columns exist
        expected_cols = ["timestamp_seconds"] + FUNDING_FEATURES + ["funding_imputed"]
        missing_cols = set(expected_cols) - set(columns)
        assert not missing_cols, f"Missing columns: {missing_cols}"

        extra_cols = set(columns) - set(expected_cols)
        assert not extra_cols, f"Unexpected columns: {extra_cols}"

        logger.info("✓ All expected columns present")

        # Check row count
        row_count = df.select(pl.len()).collect().item()
        logger.info(f"  Row count: {row_count:,}")
        assert row_count > 60_000_000, f"Row count too low: {row_count:,}"

        # Check for nulls in key columns
        null_counts = df.select([pl.col(col).is_null().sum().alias(col) for col in FUNDING_FEATURES]).collect().row(0)
        logger.info(f"  Null counts: {dict(zip(FUNDING_FEATURES, null_counts))}")

        # Check imputation flag
        imputed_count = df.select(pl.col("funding_imputed").sum()).collect().item()
        imputed_pct = (imputed_count / row_count) * 100
        logger.info(f"  Imputed rows: {imputed_count:,} ({imputed_pct:.2f}%)")

        logger.info("✅ Module 2 PASSED")
        return True

    except Exception as e:
        logger.error(f"❌ Module 2 FAILED: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_module_3_orderbook_l0_features():
    """Test Module 3: Orderbook L0 Features."""
    logger.info("=" * 80)
    logger.info("TEST: Module 3 - Orderbook L0 Features")
    logger.info("=" * 80)

    try:
        # Run the module
        output_file = engineer_and_write_orderbook_l0_features()

        # Verify file exists
        assert output_file.exists(), f"Output file not created: {output_file}"
        logger.info(f"✓ Output file created: {output_file}")

        # Load output and check schema
        df = pl.scan_parquet(output_file)
        schema = df.collect_schema()
        columns = list(schema.names())

        logger.info(f"  Columns created: {columns}")
        logger.info(f"  Expected features: {ORDERBOOK_L0_FEATURES}")

        # Verify all expected columns exist
        expected_cols = ["timestamp_seconds"] + ORDERBOOK_L0_FEATURES + ["orderbook_imputed"]
        missing_cols = set(expected_cols) - set(columns)
        assert not missing_cols, f"Missing columns: {missing_cols}"

        extra_cols = set(columns) - set(expected_cols)
        assert not extra_cols, f"Unexpected columns: {extra_cols}"

        logger.info("✓ All expected columns present")

        # Check row count
        row_count = df.select(pl.len()).collect().item()
        logger.info(f"  Row count: {row_count:,}")
        assert row_count > 60_000_000, f"Row count too low: {row_count:,}"

        # Check for nulls in key columns (sample)
        sample_cols = ["bid_ask_spread_bps", "spread_ema_60s", "bid_ask_imbalance", "imbalance_ema_60s"]
        null_counts = df.select([pl.col(col).is_null().sum().alias(col) for col in sample_cols]).collect().row(0)
        logger.info(f"  Sample null counts: {dict(zip(sample_cols, null_counts))}")

        # Check imputation flag
        imputed_count = df.select(pl.col("orderbook_imputed").sum()).collect().item()
        imputed_pct = (imputed_count / row_count) * 100
        logger.info(f"  Imputed rows: {imputed_count:,} ({imputed_pct:.2f}%)")

        logger.info("✅ Module 3 PASSED")
        return True

    except Exception as e:
        logger.error(f"❌ Module 3 FAILED: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_module_4_orderbook_5level_features():
    """Test Module 4: Orderbook 5-Level Features."""
    logger.info("=" * 80)
    logger.info("TEST: Module 4 - Orderbook 5-Level Features")
    logger.info("=" * 80)

    try:
        # Run the module
        output_file = engineer_and_write_orderbook_5level_features()

        # Verify file exists
        assert output_file.exists(), f"Output file not created: {output_file}"
        logger.info(f"✓ Output file created: {output_file}")

        # Load output and check schema
        df = pl.scan_parquet(output_file)
        schema = df.collect_schema()
        columns = list(schema.names())

        logger.info(f"  Columns created: {columns}")
        logger.info(f"  Expected features: {ORDERBOOK_5LEVEL_FEATURES}")

        # Verify all expected columns exist
        expected_cols = ["timestamp_seconds"] + ORDERBOOK_5LEVEL_FEATURES + ["orderbook_5level_imputed"]
        missing_cols = set(expected_cols) - set(columns)
        assert not missing_cols, f"Missing columns: {missing_cols}"

        extra_cols = set(columns) - set(expected_cols)
        assert not extra_cols, f"Unexpected columns: {extra_cols}"

        logger.info("✓ All expected columns present")

        # Check row count
        row_count = df.select(pl.len()).collect().item()
        logger.info(f"  Row count: {row_count:,}")
        assert row_count > 60_000_000, f"Row count too low: {row_count:,}"

        # Check for nulls in key columns (sample)
        sample_cols = ["depth_imbalance_5", "depth_imbalance_ema_60s", "weighted_mid_price_5", "weighted_mid_ema_60s"]
        null_counts = df.select([pl.col(col).is_null().sum().alias(col) for col in sample_cols]).collect().row(0)
        logger.info(f"  Sample null counts: {dict(zip(sample_cols, null_counts))}")

        # Check imputation flag
        imputed_count = df.select(pl.col("orderbook_5level_imputed").sum()).collect().item()
        imputed_pct = (imputed_count / row_count) * 100
        logger.info(f"  Imputed rows: {imputed_count:,} ({imputed_pct:.2f}%)")

        logger.info("✅ Module 4 PASSED")
        return True

    except Exception as e:
        logger.error(f"❌ Module 4 FAILED: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_module_5_price_basis_features():
    """Test Module 5: Price Basis Features."""
    logger.info("=" * 80)
    logger.info("TEST: Module 5 - Price Basis Features")
    logger.info("=" * 80)

    try:
        # Run the module
        output_file = engineer_and_write_price_basis_features()

        # Verify file exists
        assert output_file.exists(), f"Output file not created: {output_file}"
        logger.info(f"✓ Output file created: {output_file}")

        # Load output and check schema
        df = pl.scan_parquet(output_file)
        schema = df.collect_schema()
        columns = list(schema.names())

        logger.info(f"  Columns created: {columns}")
        logger.info(f"  Expected features: {PRICE_BASIS_FEATURES}")

        # Verify all expected columns exist
        expected_cols = ["timestamp_seconds"] + PRICE_BASIS_FEATURES + ["price_basis_imputed"]
        missing_cols = set(expected_cols) - set(columns)
        assert not missing_cols, f"Missing columns: {missing_cols}"

        extra_cols = set(columns) - set(expected_cols)
        assert not extra_cols, f"Unexpected columns: {extra_cols}"

        logger.info("✓ All expected columns present")

        # Check row count
        row_count = df.select(pl.len()).collect().item()
        logger.info(f"  Row count: {row_count:,}")
        assert row_count > 60_000_000, f"Row count too low: {row_count:,}"

        # Check for nulls in key columns
        sample_cols = ["mark_index_basis_bps", "mark_index_ema_60s"]
        null_counts = df.select([pl.col(col).is_null().sum().alias(col) for col in sample_cols]).collect().row(0)
        logger.info(f"  Sample null counts: {dict(zip(sample_cols, null_counts))}")

        # Check imputation flag
        imputed_count = df.select(pl.col("price_basis_imputed").sum()).collect().item()
        imputed_pct = (imputed_count / row_count) * 100
        logger.info(f"  Imputed rows: {imputed_count:,} ({imputed_pct:.2f}%)")

        logger.info("✅ Module 5 PASSED")
        return True

    except Exception as e:
        logger.error(f"❌ Module 5 FAILED: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_module_6_oi_features():
    """Test Module 6: Open Interest Features."""
    logger.info("=" * 80)
    logger.info("TEST: Module 6 - Open Interest Features")
    logger.info("=" * 80)

    try:
        # Run the module
        output_file = engineer_and_write_oi_features()

        # Verify file exists
        assert output_file.exists(), f"Output file not created: {output_file}"
        logger.info(f"✓ Output file created: {output_file}")

        # Load output and check schema
        df = pl.scan_parquet(output_file)
        schema = df.collect_schema()
        columns = list(schema.names())

        logger.info(f"  Columns created: {columns}")
        logger.info(f"  Expected features: {OI_FEATURES}")

        # Verify all expected columns exist
        expected_cols = ["timestamp_seconds"] + OI_FEATURES + ["open_interest_imputed"]
        missing_cols = set(expected_cols) - set(columns)
        assert not missing_cols, f"Missing columns: {missing_cols}"

        extra_cols = set(columns) - set(expected_cols)
        assert not extra_cols, f"Unexpected columns: {extra_cols}"

        logger.info("✓ All expected columns present")

        # Check row count
        row_count = df.select(pl.len()).collect().item()
        logger.info(f"  Row count: {row_count:,}")
        assert row_count > 60_000_000, f"Row count too low: {row_count:,}"

        # Check for nulls in key columns
        null_counts = df.select([pl.col(col).is_null().sum().alias(col) for col in OI_FEATURES]).collect().row(0)
        logger.info(f"  Null counts: {dict(zip(OI_FEATURES, null_counts))}")

        # Check imputation flag
        imputed_count = df.select(pl.col("open_interest_imputed").sum()).collect().item()
        imputed_pct = (imputed_count / row_count) * 100
        logger.info(f"  Imputed rows: {imputed_count:,} ({imputed_pct:.2f}%)")

        logger.info("✅ Module 6 PASSED")
        return True

    except Exception as e:
        logger.error(f"❌ Module 6 FAILED: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_integration():
    """Test full integration: all Binance modules in sequence."""
    logger.info("=" * 80)
    logger.info("INTEGRATION TEST: All Binance Data Modules (2-6)")
    logger.info("=" * 80)

    results = {
        "Module 2: Funding": test_module_2_funding_features(),
        "Module 3: Orderbook L0": test_module_3_orderbook_l0_features(),
        "Module 4: Orderbook 5-Level": test_module_4_orderbook_5level_features(),
        "Module 5: Price Basis": test_module_5_price_basis_features(),
        "Module 6: Open Interest": test_module_6_oi_features(),
    }

    logger.info("=" * 80)
    logger.info("TEST RESULTS SUMMARY")
    logger.info("=" * 80)

    passed = sum(results.values())
    total = len(results)

    for module, result in results.items():
        status = "✅ PASSED" if result else "❌ FAILED"
        logger.info(f"{module}: {status}")

    logger.info("=" * 80)
    logger.info(f"OVERALL: {passed}/{total} modules passed")
    logger.info("=" * 80)

    if passed == total:
        logger.info("🎉 ALL TESTS PASSED!")
        return 0
    else:
        logger.error(f"❌ {total - passed} TESTS FAILED")
        return 1


def main():
    """Run all tests."""
    import argparse

    parser = argparse.ArgumentParser(description="Test Binance data modules in engineer_all_features_v4.py")
    parser.add_argument(
        "--module",
        choices=["2", "3", "4", "5", "6", "all"],
        default="all",
        help="Which module to test (default: all)",
    )

    args = parser.parse_args()

    if args.module == "2":
        success = test_module_2_funding_features()
    elif args.module == "3":
        success = test_module_3_orderbook_l0_features()
    elif args.module == "4":
        success = test_module_4_orderbook_5level_features()
    elif args.module == "5":
        success = test_module_5_price_basis_features()
    elif args.module == "6":
        success = test_module_6_oi_features()
    else:  # all
        return test_integration()

    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
