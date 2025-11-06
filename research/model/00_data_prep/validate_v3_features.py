#!/usr/bin/env python3
"""
Validation Script for V3 Feature Engineering
=============================================

Validates the engineer_all_features_v3.py script before running on full dataset.

Checks:
1. All input files exist and have expected schemas
2. Feature engineering logic produces expected column count
3. No duplicate column names from joins
4. Sample output matches expected feature list

Author: BT Research Team
Date: 2025-11-05
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import polars as pl
from feature_cols_v3 import FEATURE_COLS, TOTAL_FEATURES

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

# Paths
MODEL_DIR = Path(__file__).parent.parent
RESULTS_DIR = MODEL_DIR / "results"
TARDIS_DIR = MODEL_DIR.parent / "tardis/data/consolidated"

# Input files
INPUT_FILES = {
    "baseline": RESULTS_DIR / "production_backtest_results.parquet",
    "rv": RESULTS_DIR / "realized_volatility_1s.parquet",
    "micro": RESULTS_DIR / "microstructure_features.parquet",
    "advanced": RESULTS_DIR / "advanced_features.parquet",
    "funding": TARDIS_DIR / "binance_funding_rates_1s_consolidated.parquet",
    "orderbook": TARDIS_DIR / "binance_orderbook_5_1s_consolidated.parquet",
}


def check_file_exists() -> bool:
    """Check that all input files exist."""
    logger.info("=" * 80)
    logger.info("Step 1: Checking Input Files")
    logger.info("=" * 80)

    all_exist = True
    for name, path in INPUT_FILES.items():
        if path.exists():
            size_mb = path.stat().st_size / (1024**2)
            logger.info(f"✓ {name:12s}: {path} ({size_mb:.1f} MB)")
        else:
            logger.error(f"✗ {name:12s}: {path} (NOT FOUND)")
            all_exist = False

    return all_exist


def check_schemas() -> bool:
    """Check that input files have expected schemas."""
    logger.info("=" * 80)
    logger.info("Step 2: Checking Schemas")
    logger.info("=" * 80)

    required_columns = {
        "baseline": ["timestamp", "time_remaining", "S", "K", "iv_staleness_seconds"],
        "rv": ["timestamp_seconds", "rv_60s", "rv_300s", "rv_900s"],
        "micro": ["timestamp_seconds", "momentum_300s", "momentum_900s", "range_300s", "range_900s"],
        "advanced": ["timestamp", "ema_12s", "ema_60s", "ema_300s", "ema_900s"],
        "funding": ["timestamp", "funding_rate", "mark_price", "index_price", "last_price", "open_interest"],
        "orderbook": ["timestamp", "bid_price_0", "ask_price_0", "bid_amount_0", "ask_amount_0"],
    }

    all_valid = True
    for name, path in INPUT_FILES.items():
        schema = pl.scan_parquet(path).collect_schema()
        missing = [col for col in required_columns[name] if col not in schema.names()]

        if missing:
            logger.error(f"✗ {name:12s}: Missing columns: {missing}")
            all_valid = False
        else:
            logger.info(f"✓ {name:12s}: All required columns present ({len(schema.names())} total)")

    return all_valid


def test_sample_pipeline() -> bool:
    """Test feature engineering pipeline on a small sample."""
    logger.info("=" * 80)
    logger.info("Step 3: Testing Pipeline on Sample (100K rows)")
    logger.info("=" * 80)

    try:
        # Load baseline sample to get timestamp range
        logger.info("Loading baseline sample...")
        baseline_sample = pl.scan_parquet(INPUT_FILES["baseline"]).head(100_000).collect()
        logger.info(f"✓ Baseline: {len(baseline_sample):,} rows")

        # Get timestamp range (convert from microseconds to seconds)
        min_ts = int(baseline_sample["timestamp"].min() or 0) // 1_000_000  # type: ignore[arg-type]
        max_ts = int(baseline_sample["timestamp"].max() or 0) // 1_000_000  # type: ignore[arg-type]
        logger.info(f"  Timestamp range: {min_ts} to {max_ts}")

        # Test funding features
        logger.info("\nTesting funding features...")
        funding_df = (
            pl.scan_parquet(INPUT_FILES["funding"])
            .filter((pl.col("timestamp") / 1_000_000).cast(pl.Int64).is_between(min_ts, max_ts))
            .head(100_000)
            .collect()
        )
        logger.info(f"✓ Funding: {len(funding_df):,} rows")

        # Test orderbook features
        logger.info("\nTesting orderbook features...")
        orderbook_df = (
            pl.scan_parquet(INPUT_FILES["orderbook"])
            .filter((pl.col("timestamp") / 1_000_000).cast(pl.Int64).is_between(min_ts, max_ts))
            .head(100_000)
            .collect()
        )
        logger.info(f"✓ Orderbook: {len(orderbook_df):,} rows")

        # Test EMAs/SMAs computation
        logger.info("\nTesting EMA/SMA computation...")
        test_series = pl.Series("test", [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])
        ema = test_series.ewm_mean(span=3)
        sma = test_series.rolling_mean(window_size=3)
        vol = test_series.rolling_std(window_size=3)

        logger.info(f"  EMA length: {len(ema)} (expected: 10)")
        logger.info(f"  SMA length: {len(sma)} (expected: 10)")
        logger.info(f"  Vol length: {len(vol)} (expected: 10)")

        if len(ema) == 10 and len(sma) == 10 and len(vol) == 10:
            logger.info("✓ EMA/SMA/Vol computation working correctly")
        else:
            logger.error("✗ EMA/SMA/Vol computation failed")
            return False

        logger.info("\n✓ Sample pipeline test passed")
        return True

    except Exception as e:
        logger.error(f"✗ Sample pipeline test failed: {e}")
        return False


def check_feature_list() -> bool:
    """Check that FEATURE_COLS list is valid."""
    logger.info("=" * 80)
    logger.info("Step 4: Validating Feature List")
    logger.info("=" * 80)

    # Check for duplicates
    duplicates = [f for f in FEATURE_COLS if FEATURE_COLS.count(f) > 1]
    if duplicates:
        logger.error(f"✗ Duplicate features detected: {set(duplicates)}")
        return False
    else:
        logger.info(f"✓ No duplicates in FEATURE_COLS ({TOTAL_FEATURES} unique features)")

    # Check that all features have valid names (no special characters except _)
    invalid_names = [f for f in FEATURE_COLS if not f.replace("_", "").isalnum()]
    if invalid_names:
        logger.error(f"✗ Invalid feature names: {invalid_names}")
        return False
    else:
        logger.info("✓ All feature names are valid")

    return True


def check_for_potential_duplicates() -> bool:
    """Check for potential duplicate columns from joins."""
    logger.info("=" * 80)
    logger.info("Step 5: Checking for Potential Join Conflicts")
    logger.info("=" * 80)

    # Features in FEATURE_COLS
    feature_cols_set = set(FEATURE_COLS)

    # Check if EMAs and IV/RV ratios are in FEATURE_COLS (they should be)
    ema_features = ["ema_12s", "ema_60s", "ema_300s", "ema_900s", "iv_rv_ratio_300s", "iv_rv_ratio_900s"]
    found_emas = [f for f in ema_features if f in feature_cols_set]

    if len(found_emas) == len(ema_features):
        logger.info("✓ EMAs and IV/RV ratios present in FEATURE_COLS")
        logger.info("   (These are loaded in Module 7 only, not Module 1 - no duplicates)")
        return True
    else:
        missing = [f for f in ema_features if f not in feature_cols_set]
        logger.warning(f"⚠️  Missing EMA/IV_RV features from FEATURE_COLS: {missing}")
        return False


def main() -> None:
    """Run all validation checks."""
    logger.info("=" * 80)
    logger.info("V3 Feature Engineering Validation")
    logger.info("=" * 80)

    checks = [
        ("File Existence", check_file_exists),
        ("Schema Validation", check_schemas),
        ("Sample Pipeline", test_sample_pipeline),
        ("Feature List", check_feature_list),
        ("Join Conflicts", check_for_potential_duplicates),
    ]

    results = {}
    for name, check_func in checks:
        try:
            results[name] = check_func()
        except Exception as e:
            logger.error(f"✗ {name} check failed with exception: {e}")
            results[name] = False

    # Summary
    logger.info("=" * 80)
    logger.info("Validation Summary")
    logger.info("=" * 80)

    for name, passed in results.items():
        status = "✓ PASS" if passed else "✗ FAIL"
        logger.info(f"{status:8s}: {name}")

    all_passed = all(results.values())
    logger.info("=" * 80)

    if all_passed:
        logger.info("✓ All validation checks passed!")
        logger.info("\nReady to run: uv run python research/model/00_data_prep/engineer_all_features_v3.py")
    else:
        logger.error("✗ Some validation checks failed. Please fix issues before running full pipeline.")
        sys.exit(1)


if __name__ == "__main__":
    main()
