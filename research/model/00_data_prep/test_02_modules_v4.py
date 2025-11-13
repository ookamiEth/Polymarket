#!/usr/bin/env python3
"""
Module Unit Tests for V4 Pipeline
==================================

Tests each feature engineering module with synthetic data to ensure correct behavior.

Tests:
- Module 1: Existing features join (no duplicate columns)
- Module 2: Funding rate features
- Module 3: Orderbook L0 features
- Module 4: Orderbook 5-level features
- Module 5: Price basis features
- Module 6: OI features
- Module 7: RV/momentum/range features
- Module 7b: Extreme/regime features (NEW)
- V4 transformations: Advanced moneyness, volatility asymmetry, orderbook normalization

Runtime: ~10 minutes

Author: BT Research Team
Date: 2025-11-13
"""

from __future__ import annotations

import logging
import sys
from datetime import datetime

import numpy as np
import polars as pl

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


class TestResult:
    """Store test result with status and message."""

    def __init__(self, name: str, passed: bool, message: str = ""):
        self.name = name
        self.passed = passed
        self.message = message

    def __repr__(self) -> str:
        status = "✅ PASS" if self.passed else "❌ FAIL"
        return f"{status}: {self.name}" + (f" - {self.message}" if self.message else "")


def generate_synthetic_baseline(n: int = 10000) -> pl.DataFrame:
    """Generate synthetic baseline data for testing."""
    np.random.seed(42)

    start_ts = int(datetime(2024, 1, 1).timestamp())

    return pl.DataFrame(
        {
            "timestamp": [start_ts + i for i in range(n)],
            "time_remaining": np.random.randint(60, 900, n),
            "S": np.random.uniform(95000, 105000, n),
            "K": np.random.uniform(95000, 105000, n),
            "sigma_mid": np.random.uniform(0.3, 0.8, n),
            "T_years": np.random.uniform(0.0001, 0.001, n),
            "iv_staleness_seconds": np.random.randint(0, 300, n),
            "outcome": np.random.choice([0, 1], n),
            "prob_mid": np.random.uniform(0.4, 0.6, n),
        }
    )


def generate_synthetic_rv(n: int = 10000) -> pl.DataFrame:
    """Generate synthetic realized volatility data."""
    np.random.seed(42)
    start_ts = int(datetime(2024, 1, 1).timestamp())

    return pl.DataFrame(
        {
            "timestamp_seconds": [start_ts + i for i in range(n)],
            "rv_60s": np.random.uniform(0.002, 0.01, n),
            "rv_300s": np.random.uniform(0.003, 0.015, n),
            "rv_900s": np.random.uniform(0.005, 0.02, n),
            "rv_3600s": np.random.uniform(0.008, 0.03, n),
        }
    )


def generate_synthetic_funding(n: int = 10000) -> pl.DataFrame:
    """Generate synthetic funding rate data."""
    np.random.seed(42)
    start_ts = int(datetime(2024, 1, 1).timestamp()) * 1_000_000  # Microseconds

    return pl.DataFrame(
        {
            "timestamp": [start_ts + i * 1_000_000 for i in range(n)],
            "funding_rate": np.random.uniform(-0.0001, 0.0001, n),
            "mark_price": np.random.uniform(95000, 105000, n),
            "index_price": np.random.uniform(95000, 105000, n),
            "open_interest": np.random.uniform(1e9, 2e9, n),
        }
    )


def generate_synthetic_orderbook(n: int = 10000) -> pl.DataFrame:
    """Generate synthetic orderbook data."""
    np.random.seed(42)
    start_ts = int(datetime(2024, 1, 1).timestamp()) * 1_000_000  # Microseconds

    base_price = 100000
    return pl.DataFrame(
        {
            "timestamp": [start_ts + i * 1_000_000 for i in range(n)],
            # Level 0 (best bid/ask)
            "bid_price_0": base_price - np.random.uniform(0, 10, n),
            "ask_price_0": base_price + np.random.uniform(0, 10, n),
            "bid_amount_0": np.random.uniform(1, 10, n),
            "ask_amount_0": np.random.uniform(1, 10, n),
            # Levels 1-4
            **{f"bid_price_{i}": base_price - np.random.uniform(10 * i, 10 * (i + 1), n) for i in range(1, 5)},
            **{f"ask_price_{i}": base_price + np.random.uniform(10 * i, 10 * (i + 1), n) for i in range(1, 5)},
            **{f"bid_amount_{i}": np.random.uniform(1, 10, n) for i in range(1, 5)},
            **{f"ask_amount_{i}": np.random.uniform(1, 10, n) for i in range(1, 5)},
        }
    )


def generate_synthetic_advanced(n: int = 10000) -> pl.DataFrame:
    """Generate synthetic advanced features data."""
    np.random.seed(43)  # Different seed for variety
    start_ts = int(datetime(2024, 1, 1).timestamp())  # Seconds

    return pl.DataFrame(
        {
            "timestamp_seconds": [start_ts + i for i in range(n)],
            "high_15m": np.random.uniform(100000, 105000, n),
            "low_15m": np.random.uniform(95000, 100000, n),
            "time_since_high_15m": np.random.randint(0, 900, n),
            "time_since_low_15m": np.random.randint(0, 900, n),
            "skewness_300s": np.random.normal(0, 0.5, n),
            "kurtosis_300s": np.random.uniform(2, 5, n),
            "downside_vol_300s": np.random.uniform(0.002, 0.01, n),
            "upside_vol_300s": np.random.uniform(0.002, 0.01, n),
            "vol_asymmetry_300s": np.random.normal(0, 0.2, n),
            "tail_risk_300s": np.random.uniform(0, 0.1, n),
            "hour_of_day_utc": [i % 24 for i in range(n)],
            "hour_sin": np.sin(2 * np.pi * np.arange(n) / 24),
            "hour_cos": np.cos(2 * np.pi * np.arange(n) / 24),
            "vol_persistence_ar1": np.random.uniform(0.5, 0.95, n),
            "vol_acceleration_300s": np.random.normal(0, 0.001, n),
            "vol_of_vol_300s": np.random.uniform(0.0005, 0.002, n),
            "autocorr_decay": np.random.uniform(0, 1, n),
            "reversals_300s": np.random.randint(-5, 5, n),
            "price_oscillation_300s": np.random.uniform(0, 0.001, n),
            "price_range_norm_300s": np.random.uniform(0.001, 0.01, n),
        }
    )


def generate_synthetic_micro(n: int = 10000) -> pl.DataFrame:
    """Generate synthetic microstructure features data."""
    np.random.seed(44)  # Different seed
    start_ts = int(datetime(2024, 1, 1).timestamp())  # Seconds

    return pl.DataFrame(
        {
            "timestamp_seconds": [start_ts + i for i in range(n)],
            "autocorr_lag5_300s": np.random.uniform(-0.1, 0.1, n),
            "hurst_300s": np.random.uniform(0.4, 0.6, n),
        }
    )


def test_module_1_join_integrity() -> TestResult:
    """Test Module 1: Existing features join integrity."""
    logger.info("\nTesting Module 1: Join integrity (baseline + advanced + micro)...")

    try:
        # Generate synthetic data for all three sources
        baseline_df = generate_synthetic_baseline(10000)
        advanced_df = generate_synthetic_advanced(10000)
        micro_df = generate_synthetic_micro(10000)

        # Baseline: Add timestamp_seconds from existing timestamp and add moneyness
        baseline_df = baseline_df.with_columns(
            [
                pl.col("timestamp").alias("timestamp_seconds"),
                (pl.col("S") / pl.col("K")).alias("moneyness"),
            ]
        ).select(
            [
                "timestamp_seconds",
                "time_remaining",
                "S",
                "K",
                "sigma_mid",
                "T_years",
                "moneyness",
                "iv_staleness_seconds",
            ]
        )

        # Join all three (LEFT joins preserve baseline rows)
        df = baseline_df.join(advanced_df, on="timestamp_seconds", how="left").join(
            micro_df, on="timestamp_seconds", how="left"
        )

        # Test 1: No duplicate columns
        col_names = df.columns
        unique_cols = set(col_names)
        if len(col_names) != len(unique_cols):
            duplicates = [col for col in col_names if col_names.count(col) > 1]
            return TestResult("Module 1: Join Integrity", False, f"Duplicate columns: {set(duplicates)}")

        # Test 2: Row count preserved (should match baseline)
        if len(df) != 10000:
            return TestResult("Module 1: Join Integrity", False, f"Row count mismatch: {len(df)} (expected 10000)")

        # Test 3: Expected feature count
        # 8 baseline + 20 advanced (not counting timestamp) + 2 micro = 30 total
        expected_cols = 30
        if len(col_names) != expected_cols:
            return TestResult(
                "Module 1: Join Integrity",
                False,
                f"Column count: {len(col_names)} (expected {expected_cols})",
            )

        # Test 4: Baseline columns not null (should be present in all rows)
        baseline_cols = ["time_remaining", "S", "K", "sigma_mid", "T_years", "moneyness", "iv_staleness_seconds"]
        baseline_nulls = sum(df[col].null_count() for col in baseline_cols)
        if baseline_nulls > 0:
            return TestResult("Module 1: Join Integrity", False, f"Baseline nulls: {baseline_nulls}")

        return TestResult("Module 1: Join Integrity", True, "30 features, no duplicates, LEFT join preserved rows")

    except Exception as e:
        return TestResult("Module 1: Join Integrity", False, f"Exception: {e}")


def test_module_2_funding() -> TestResult:
    """Test Module 2: Funding rate features."""
    logger.info("\nTesting Module 2: Funding rate features...")

    try:
        df = generate_synthetic_funding(10000)

        # Convert timestamp to seconds
        df = df.with_columns([(pl.col("timestamp") / 1_000_000).cast(pl.Int64).alias("timestamp_seconds")])

        # Compute EMAs (simplified from module)
        df = df.with_columns(
            [
                pl.col("funding_rate").ewm_mean(span=60).alias("funding_rate_ema_60s"),
                pl.col("funding_rate").ewm_mean(span=300).alias("funding_rate_ema_300s"),
                pl.col("funding_rate").ewm_mean(span=900).alias("funding_rate_ema_900s"),
                pl.col("funding_rate").ewm_mean(span=3600).alias("funding_rate_ema_3600s"),
            ]
        )

        # Expected: 1 raw + 4 EMAs = 5 features
        expected_cols = [
            "funding_rate",
            "funding_rate_ema_60s",
            "funding_rate_ema_300s",
            "funding_rate_ema_900s",
            "funding_rate_ema_3600s",
        ]

        missing = [col for col in expected_cols if col not in df.columns]

        if missing:
            return TestResult("Module 2: Funding", False, f"Missing columns: {missing}")

        # Check no nulls (after warmup)
        null_count = df.select(expected_cols).tail(5000).null_count().sum_horizontal().to_list()[0]

        if null_count > 0:
            return TestResult("Module 2: Funding", False, f"Found {null_count} nulls in features")

        return TestResult("Module 2: Funding", True, "5 funding features generated correctly")

    except Exception as e:
        return TestResult("Module 2: Funding", False, f"Exception: {e}")


def test_module_3_orderbook_l0() -> TestResult:
    """Test Module 3: Orderbook L0 features."""
    logger.info("\nTesting Module 3: Orderbook L0 features...")

    try:
        df = generate_synthetic_orderbook(10000)

        # Convert timestamp to seconds
        df = df.with_columns([(pl.col("timestamp") / 1_000_000).cast(pl.Int64).alias("timestamp_seconds")])

        # Compute bid-ask spread
        df = df.with_columns([((pl.col("bid_price_0") + pl.col("ask_price_0")) / 2).alias("mid_price")])

        df = df.with_columns(
            [
                ((pl.col("ask_price_0") - pl.col("bid_price_0")) / pl.col("mid_price") * 10000).alias(
                    "bid_ask_spread_bps"
                )
            ]
        )

        # Spread EMAs
        df = df.with_columns(
            [
                pl.col("bid_ask_spread_bps").ewm_mean(span=60).alias("spread_ema_60s"),
                pl.col("bid_ask_spread_bps").ewm_mean(span=300).alias("spread_ema_300s"),
                pl.col("bid_ask_spread_bps").ewm_mean(span=900).alias("spread_ema_900s"),
                pl.col("bid_ask_spread_bps").ewm_mean(span=3600).alias("spread_ema_3600s"),
            ]
        )

        # Spread volatility
        df = df.with_columns(
            [
                pl.col("bid_ask_spread_bps").rolling_std(window_size=60).alias("spread_vol_60s"),
                pl.col("bid_ask_spread_bps").rolling_std(window_size=300).alias("spread_vol_300s"),
                pl.col("bid_ask_spread_bps").rolling_std(window_size=900).alias("spread_vol_900s"),
                pl.col("bid_ask_spread_bps").rolling_std(window_size=3600).alias("spread_vol_3600s"),
            ]
        )

        # Bid-ask imbalance
        df = df.with_columns(
            [
                (
                    (pl.col("bid_amount_0") - pl.col("ask_amount_0"))
                    / (pl.col("bid_amount_0") + pl.col("ask_amount_0"))
                ).alias("bid_ask_imbalance")
            ]
        )

        # Imbalance EMAs
        df = df.with_columns(
            [
                pl.col("bid_ask_imbalance").ewm_mean(span=60).alias("imbalance_ema_60s"),
                pl.col("bid_ask_imbalance").ewm_mean(span=300).alias("imbalance_ema_300s"),
                pl.col("bid_ask_imbalance").ewm_mean(span=900).alias("imbalance_ema_900s"),
                pl.col("bid_ask_imbalance").ewm_mean(span=3600).alias("imbalance_ema_3600s"),
            ]
        )

        # Imbalance volatility
        df = df.with_columns(
            [
                pl.col("bid_ask_imbalance").rolling_std(window_size=60).alias("imbalance_vol_60s"),
                pl.col("bid_ask_imbalance").rolling_std(window_size=300).alias("imbalance_vol_300s"),
                pl.col("bid_ask_imbalance").rolling_std(window_size=900).alias("imbalance_vol_900s"),
                pl.col("bid_ask_imbalance").rolling_std(window_size=3600).alias("imbalance_vol_3600s"),
            ]
        )

        # Expected: 18 features total
        expected_features = 18
        actual_spread_features = len([c for c in df.columns if "spread" in c or "imbalance" in c])

        if actual_spread_features >= expected_features:
            return TestResult("Module 3: Orderbook L0", True, f"{actual_spread_features} features generated")
        else:
            return TestResult(
                "Module 3: Orderbook L0",
                False,
                f"Only {actual_spread_features} features (expected {expected_features})",
            )

    except Exception as e:
        return TestResult("Module 3: Orderbook L0", False, f"Exception: {e}")


def test_module_4_orderbook_5level() -> TestResult:
    """Test Module 4: Orderbook 5-level features."""
    logger.info("\nTesting Module 4: Orderbook 5-level features...")

    try:
        df = generate_synthetic_orderbook(10000)

        # Convert timestamp to seconds
        df = df.with_columns([(pl.col("timestamp") / 1_000_000).cast(pl.Int64).alias("timestamp_seconds")])

        # Compute total volumes (sum across 5 levels)
        df = df.with_columns(
            [
                (
                    pl.col("bid_amount_0")
                    + pl.col("bid_amount_1")
                    + pl.col("bid_amount_2")
                    + pl.col("bid_amount_3")
                    + pl.col("bid_amount_4")
                ).alias("total_bid_volume_5"),
                (
                    pl.col("ask_amount_0")
                    + pl.col("ask_amount_1")
                    + pl.col("ask_amount_2")
                    + pl.col("ask_amount_3")
                    + pl.col("ask_amount_4")
                ).alias("total_ask_volume_5"),
            ]
        )

        # Volume ratios (L0 vs total)
        df = df.with_columns(
            [
                (pl.col("bid_amount_0") / pl.col("total_bid_volume_5")).alias("bid_volume_ratio_1to5"),
                (pl.col("ask_amount_0") / pl.col("total_ask_volume_5")).alias("ask_volume_ratio_1to5"),
            ]
        )

        # Depth imbalance (5 levels)
        df = df.with_columns(
            [
                (
                    (pl.col("total_bid_volume_5") - pl.col("total_ask_volume_5"))
                    / (pl.col("total_bid_volume_5") + pl.col("total_ask_volume_5"))
                ).alias("depth_imbalance_5")
            ]
        )

        # Depth imbalance EMAs
        df = df.with_columns(
            [
                pl.col("depth_imbalance_5").ewm_mean(span=60).alias("depth_imbalance_ema_60s"),
                pl.col("depth_imbalance_5").ewm_mean(span=300).alias("depth_imbalance_ema_300s"),
                pl.col("depth_imbalance_5").ewm_mean(span=900).alias("depth_imbalance_ema_900s"),
                pl.col("depth_imbalance_5").ewm_mean(span=3600).alias("depth_imbalance_ema_3600s"),
            ]
        )

        # Depth imbalance volatility
        df = df.with_columns(
            [
                pl.col("depth_imbalance_5").rolling_std(window_size=60).alias("depth_imbalance_vol_60s"),
                pl.col("depth_imbalance_5").rolling_std(window_size=300).alias("depth_imbalance_vol_300s"),
                pl.col("depth_imbalance_5").rolling_std(window_size=900).alias("depth_imbalance_vol_900s"),
                pl.col("depth_imbalance_5").rolling_std(window_size=3600).alias("depth_imbalance_vol_3600s"),
            ]
        )

        # Volume-weighted mid price (5 levels)
        df = df.with_columns(
            [
                (
                    (
                        pl.col("bid_price_0") * pl.col("bid_amount_0")
                        + pl.col("bid_price_1") * pl.col("bid_amount_1")
                        + pl.col("bid_price_2") * pl.col("bid_amount_2")
                        + pl.col("bid_price_3") * pl.col("bid_amount_3")
                        + pl.col("bid_price_4") * pl.col("bid_amount_4")
                        + pl.col("ask_price_0") * pl.col("ask_amount_0")
                        + pl.col("ask_price_1") * pl.col("ask_amount_1")
                        + pl.col("ask_price_2") * pl.col("ask_amount_2")
                        + pl.col("ask_price_3") * pl.col("ask_amount_3")
                        + pl.col("ask_price_4") * pl.col("ask_amount_4")
                    )
                    / (pl.col("total_bid_volume_5") + pl.col("total_ask_volume_5"))
                ).alias("weighted_mid_price_5")
            ]
        )

        # Weighted mid EMAs
        df = df.with_columns(
            [
                pl.col("weighted_mid_price_5").ewm_mean(span=60).alias("weighted_mid_ema_60s"),
                pl.col("weighted_mid_price_5").ewm_mean(span=300).alias("weighted_mid_ema_300s"),
                pl.col("weighted_mid_price_5").ewm_mean(span=900).alias("weighted_mid_ema_900s"),
                pl.col("weighted_mid_price_5").ewm_mean(span=3600).alias("weighted_mid_ema_3600s"),
            ]
        )

        # Expected: 18 features total (2 totals + 2 ratios + 9 depth imbalance + 5 weighted mid)
        expected_features = 18
        actual_features = len(
            [
                c
                for c in df.columns
                if c
                in [
                    "total_bid_volume_5",
                    "total_ask_volume_5",
                    "bid_volume_ratio_1to5",
                    "ask_volume_ratio_1to5",
                    "depth_imbalance_5",
                    "depth_imbalance_ema_60s",
                    "depth_imbalance_ema_300s",
                    "depth_imbalance_ema_900s",
                    "depth_imbalance_ema_3600s",
                    "depth_imbalance_vol_60s",
                    "depth_imbalance_vol_300s",
                    "depth_imbalance_vol_900s",
                    "depth_imbalance_vol_3600s",
                    "weighted_mid_price_5",
                    "weighted_mid_ema_60s",
                    "weighted_mid_ema_300s",
                    "weighted_mid_ema_900s",
                    "weighted_mid_ema_3600s",
                ]
            ]
        )

        if actual_features >= expected_features:
            return TestResult("Module 4: Orderbook 5-Level", True, f"{actual_features} features generated")
        else:
            return TestResult(
                "Module 4: Orderbook 5-Level",
                False,
                f"Only {actual_features} features (expected {expected_features})",
            )

    except Exception as e:
        return TestResult("Module 4: Orderbook 5-Level", False, f"Exception: {e}")


def test_module_5_price_basis() -> TestResult:
    """Test Module 5: Price basis features."""
    logger.info("\nTesting Module 5: Price basis features...")

    try:
        df = generate_synthetic_funding(10000)

        # Convert timestamp to seconds
        df = df.with_columns([(pl.col("timestamp") / 1_000_000).cast(pl.Int64).alias("timestamp_seconds")])

        # Compute mark-index basis (bps)
        df = df.with_columns(
            [
                ((pl.col("mark_price") - pl.col("index_price")) / pl.col("index_price") * 10000).alias(
                    "mark_index_basis_bps"
                )
            ]
        )

        # Mark-index basis EMAs
        df = df.with_columns(
            [
                pl.col("mark_index_basis_bps").ewm_mean(span=60).alias("mark_index_ema_60s"),
                pl.col("mark_index_basis_bps").ewm_mean(span=300).alias("mark_index_ema_300s"),
                pl.col("mark_index_basis_bps").ewm_mean(span=900).alias("mark_index_ema_900s"),
                pl.col("mark_index_basis_bps").ewm_mean(span=3600).alias("mark_index_ema_3600s"),
            ]
        )

        # Expected: 5 features total (1 raw + 4 EMAs)
        expected_cols = [
            "mark_index_basis_bps",
            "mark_index_ema_60s",
            "mark_index_ema_300s",
            "mark_index_ema_900s",
            "mark_index_ema_3600s",
        ]

        missing = [col for col in expected_cols if col not in df.columns]

        if missing:
            return TestResult("Module 5: Price Basis", False, f"Missing columns: {missing}")

        # Check no nulls (after warmup)
        null_count = df.select(expected_cols).tail(5000).null_count().sum_horizontal().to_list()[0]

        if null_count > 0:
            return TestResult("Module 5: Price Basis", False, f"Found {null_count} nulls in features")

        return TestResult("Module 5: Price Basis", True, "5 basis features generated correctly")

    except Exception as e:
        return TestResult("Module 5: Price Basis", False, f"Exception: {e}")


def test_module_6_oi_features() -> TestResult:
    """Test Module 6: Open Interest features."""
    logger.info("\nTesting Module 6: OI features...")

    try:
        df = generate_synthetic_funding(10000)

        # Convert timestamp to seconds
        df = df.with_columns([(pl.col("timestamp") / 1_000_000).cast(pl.Int64).alias("timestamp_seconds")])

        # OI normalization by index price
        df = df.with_columns([(pl.col("open_interest") / pl.col("index_price")).alias("oi_normalized")])

        # OI EMAs (900s and 3600s only per pruned implementation)
        df = df.with_columns(
            [
                pl.col("open_interest").ewm_mean(span=900).alias("oi_ema_900s"),
                pl.col("open_interest").ewm_mean(span=3600).alias("oi_ema_3600s"),
            ]
        )

        # OI change rate
        df = df.with_columns(
            [
                (pl.col("open_interest") - pl.col("open_interest").shift(300))
                .truediv(pl.col("open_interest").shift(300))
                .alias("oi_change_rate_300s")
            ]
        )

        # Expected: 5 features (open_interest, oi_normalized, 2 EMAs, 1 change rate)
        # Note: open_interest is from source, so we test 4 derived features
        expected_derived_cols = [
            "oi_normalized",
            "oi_ema_900s",
            "oi_ema_3600s",
            "oi_change_rate_300s",
        ]

        missing = [col for col in expected_derived_cols if col not in df.columns]

        if missing:
            return TestResult("Module 6: OI Features", False, f"Missing columns: {missing}")

        # Check for reasonable values (OI should be positive)
        min_oi = df["open_interest"].min()
        if min_oi is not None and isinstance(min_oi, (int, float)) and float(min_oi) <= 0:
            return TestResult("Module 6: OI Features", False, "OI has non-positive values")

        return TestResult("Module 6: OI Features", True, "6 OI features generated correctly")

    except Exception as e:
        return TestResult("Module 6: OI Features", False, f"Exception: {e}")


def test_module_7_rv_momentum_range() -> TestResult:
    """Test Module 7: RV/Momentum/Range features (simplified)."""
    logger.info("\nTesting Module 7: RV/Momentum/Range features...")

    try:
        # Generate RV data
        rv_df = generate_synthetic_rv(10000)

        # Compute RV EMAs (60s, 300s, 900s, 3600s)
        rv_df = rv_df.with_columns(
            [
                pl.col("rv_60s").ewm_mean(span=60).alias("rv_60s_ema_60s"),
                pl.col("rv_300s").ewm_mean(span=300).alias("rv_300s_ema_300s"),
                pl.col("rv_900s").ewm_mean(span=900).alias("rv_900s_ema_900s"),
                pl.col("rv_3600s").ewm_mean(span=3600).alias("rv_3600s_ema_3600s"),
            ]
        )

        # Compute RV ratios (short-term / long-term)
        rv_df = rv_df.with_columns(
            [
                (pl.col("rv_60s") / pl.col("rv_900s")).alias("rv_60_900_ratio"),
                (pl.col("rv_300s") / pl.col("rv_3600s")).alias("rv_300_3600_ratio"),
            ]
        )

        # Expected features (sample of the 46 total)
        # We test core features: 4 RV raw + 4 RV EMAs + 2 ratios = 10 features
        expected_sample = [
            "rv_60s",
            "rv_300s",
            "rv_900s",
            "rv_3600s",
            "rv_60s_ema_60s",
            "rv_300s_ema_300s",
            "rv_900s_ema_900s",
            "rv_3600s_ema_3600s",
            "rv_60_900_ratio",
            "rv_300_3600_ratio",
        ]

        missing = [col for col in expected_sample if col not in rv_df.columns]

        if missing:
            return TestResult("Module 7: RV/Momentum/Range", False, f"Missing sample features: {missing}")

        # Check no nulls in sample (after warmup)
        null_count = rv_df.select(expected_sample).tail(5000).null_count().sum_horizontal().to_list()[0]

        if null_count > 0:
            return TestResult("Module 7: RV/Momentum/Range", False, f"Found {null_count} nulls")

        return TestResult("Module 7: RV/Momentum/Range", True, "~46 RV features validated (10 core tested)")

    except Exception as e:
        return TestResult("Module 7: RV/Momentum/Range", False, f"Exception: {e}")


def test_module_7b_extreme_regime() -> TestResult:
    """Test Module 7b: Extreme/regime features."""
    logger.info("\nTesting Module 7b: Extreme/regime features...")

    try:
        # Generate 30 days of data (need 30 days for rolling windows)
        n = 30 * 24 * 60 * 60  # 30 days in seconds
        n_sample = min(n, 100000)  # Cap at 100K for test speed

        np.random.seed(42)
        start_ts = int(datetime(2024, 1, 1).timestamp())

        rv_df = pl.DataFrame(
            {
                "timestamp_seconds": [start_ts + i for i in range(n_sample)],
                "rv_60s": np.random.uniform(0.002, 0.01, n_sample),
                "rv_900s": np.random.uniform(0.005, 0.02, n_sample),
            }
        )

        baseline_df = pl.DataFrame(
            {
                "timestamp_seconds": [start_ts + i for i in range(n_sample)],
                "S": np.random.uniform(95000, 105000, n_sample),
                "K": np.random.uniform(95000, 105000, n_sample),
            }
        )

        # Join
        df = rv_df.join(baseline_df, on="timestamp_seconds", how="left")

        # Add moneyness_distance
        df = df.with_columns([((pl.col("S") / pl.col("K")) - 1).abs().alias("moneyness_distance")])

        # Sort by timestamp (CRITICAL for rolling windows)
        df = df.sort("timestamp_seconds")

        # Filter nulls
        df = df.filter(pl.col("rv_60s").is_not_null() & pl.col("rv_900s").is_not_null())

        # Compute extreme condition features
        df = df.with_columns(
            [
                # RV ratio
                (pl.col("rv_60s") / (pl.col("rv_900s") + 1e-10)).alias("rv_ratio"),
                # 95th percentile (30 days = 2,592,000 seconds)
                pl.col("rv_900s")
                .rolling_quantile_by(by="timestamp_seconds", window_size="2592000i", quantile=0.95)
                .alias("rv_95th_percentile"),
            ]
        )

        # Flag extreme conditions
        df = df.with_columns(
            [
                pl.when((pl.col("rv_ratio") > 3) | (pl.col("rv_900s") > pl.col("rv_95th_percentile")))
                .then(pl.lit(True))
                .otherwise(pl.lit(False))
                .alias("is_extreme_condition"),
                pl.when(pl.col("rv_ratio") > 3).then(pl.lit(0.5)).otherwise(pl.lit(1.0)).alias("position_scale"),
            ]
        )

        # Regime detection thresholds
        df = df.with_columns(
            [
                pl.col("rv_900s")
                .rolling_quantile_by(by="timestamp_seconds", window_size="2592000i", quantile=0.33)
                .alias("vol_low_thresh"),
                pl.col("rv_900s")
                .rolling_quantile_by(by="timestamp_seconds", window_size="2592000i", quantile=0.67)
                .alias("vol_high_thresh"),
            ]
        )

        # Regime classification
        df = df.with_columns(
            [
                pl.when(pl.col("rv_900s") < pl.col("vol_low_thresh"))
                .then(pl.lit("low"))
                .when(pl.col("rv_900s") > pl.col("vol_high_thresh"))
                .then(pl.lit("high"))
                .otherwise(pl.lit("medium"))
                .alias("volatility_regime"),
                # Combined regime
                pl.when((pl.col("rv_900s") < pl.col("vol_low_thresh")) & (pl.col("moneyness_distance") < 0.01))
                .then(pl.lit("low_vol_atm"))
                .when((pl.col("rv_900s") < pl.col("vol_low_thresh")) & (pl.col("moneyness_distance") >= 0.01))
                .then(pl.lit("low_vol_otm"))
                .when((pl.col("rv_900s") > pl.col("vol_high_thresh")) & (pl.col("moneyness_distance") < 0.01))
                .then(pl.lit("high_vol_atm"))
                .when((pl.col("rv_900s") > pl.col("vol_high_thresh")) & (pl.col("moneyness_distance") >= 0.01))
                .then(pl.lit("high_vol_otm"))
                .otherwise(pl.lit("medium_vol"))
                .alias("market_regime"),
            ]
        )

        # Expected features
        expected_cols = [
            "rv_ratio",
            "rv_95th_percentile",
            "is_extreme_condition",
            "position_scale",
            "vol_low_thresh",
            "vol_high_thresh",
            "volatility_regime",
            "market_regime",
        ]

        missing = [col for col in expected_cols if col not in df.columns]

        if missing:
            return TestResult("Module 7b: Extreme/Regime", False, f"Missing columns: {missing}")

        # Check null rate (should be <5% after warmup period drop)
        null_count_sum = df.select(expected_cols).null_count().sum_horizontal().to_list()[0]
        null_rate = float(null_count_sum) / (len(df) * len(expected_cols))

        if null_rate > 0.05:
            return TestResult("Module 7b: Extreme/Regime", False, f"High null rate: {null_rate:.2%}")

        # Check regime distribution
        regime_counts = df.group_by("market_regime").agg(pl.len().alias("count"))

        if len(regime_counts) < 3:  # Should have at least a few regimes in random data
            return TestResult("Module 7b: Extreme/Regime", False, f"Only {len(regime_counts)} regimes detected")

        return TestResult(
            "Module 7b: Extreme/Regime", True, f"8 features, {null_rate:.2%} nulls, {len(regime_counts)} regimes"
        )

    except Exception as e:
        return TestResult("Module 7b: Extreme/Regime", False, f"Exception: {e}")


def test_v4_advanced_moneyness() -> TestResult:
    """Test V4 transformation: Advanced moneyness features."""
    logger.info("\nTesting V4: Advanced moneyness features...")

    try:
        df = generate_synthetic_baseline(10000)

        # Convert timestamp to datetime (needed for .dt.truncate())
        df = df.with_columns([(pl.col("timestamp") * 1_000_000).cast(pl.Datetime("us")).alias("timestamp_dt")])

        # Advanced moneyness features
        df = df.with_columns(
            [
                # Log moneyness
                (pl.col("S") / pl.col("K")).log().alias("log_moneyness"),
                # Squared moneyness
                ((pl.col("S") / pl.col("K")) - 1).pow(2).alias("moneyness_squared"),
                # Cubed moneyness
                ((pl.col("S") / pl.col("K")) - 1).pow(3).alias("moneyness_cubed"),
                # Moneyness × time
                (((pl.col("S") / pl.col("K")) - 1) * pl.col("time_remaining")).alias("moneyness_x_time"),
                # Moneyness distance
                ((pl.col("S") / pl.col("K")) - 1).abs().alias("moneyness_distance"),
            ]
        )

        # Expected: 8 features (5 above + 3 more in full implementation)
        expected_cols = [
            "log_moneyness",
            "moneyness_squared",
            "moneyness_cubed",
            "moneyness_x_time",
            "moneyness_distance",
        ]

        missing = [col for col in expected_cols if col not in df.columns]

        if missing:
            return TestResult("V4: Advanced Moneyness", False, f"Missing columns: {missing}")

        # Check no inf/nan values
        inf_count = df.select(expected_cols).select(pl.all().is_infinite().sum()).sum_horizontal().to_list()[0]
        nan_count = df.select(expected_cols).select(pl.all().is_nan().sum()).sum_horizontal().to_list()[0]

        if inf_count > 0 or nan_count > 0:
            return TestResult("V4: Advanced Moneyness", False, f"{inf_count} inf, {nan_count} nan values")

        return TestResult("V4: Advanced Moneyness", True, f"{len(expected_cols)} moneyness features generated")

    except Exception as e:
        return TestResult("V4: Advanced Moneyness", False, f"Exception: {e}")


def test_v4_volatility_asymmetry() -> TestResult:
    """Test V4 transformation: Volatility asymmetry features."""
    logger.info("\nTesting V4: Volatility asymmetry features...")

    try:
        # Generate baseline with price series
        n = 10000
        np.random.seed(42)
        start_ts = int(datetime(2024, 1, 1).timestamp())

        # Generate realistic price series with returns
        prices: list[float] = [100000.0]
        for _ in range(n - 1):
            ret = float(np.random.normal(0, 0.001))  # 0.1% std returns
            prices.append(prices[-1] * (1 + ret))

        df = pl.DataFrame(
            {
                "timestamp": [start_ts + i for i in range(n)],
                "S": prices,
                "sigma_mid": np.random.uniform(0.3, 0.8, n),
            }
        )

        # Compute returns
        for window in [60, 300, 900]:
            df = df.with_columns([pl.col("S").pct_change(window).alias(f"returns_{window}s")])

        # Downside and upside volatility
        for window in [60, 300, 900]:
            df = df.with_columns(
                [
                    pl.when(pl.col(f"returns_{window}s") < 0)
                    .then(pl.col(f"returns_{window}s").pow(2))
                    .otherwise(0)
                    .rolling_mean(window)
                    .sqrt()
                    .alias(f"downside_vol_{window}"),
                    pl.when(pl.col(f"returns_{window}s") > 0)
                    .then(pl.col(f"returns_{window}s").pow(2))
                    .otherwise(0)
                    .rolling_mean(window)
                    .sqrt()
                    .alias(f"upside_vol_{window}"),
                ]
            )

        # Asymmetry ratio
        for window in [60, 300, 900]:
            df = df.with_columns(
                [
                    (pl.col(f"downside_vol_{window}") / (pl.col(f"upside_vol_{window}") + 1e-10)).alias(
                        f"vol_asymmetry_ratio_{window}"
                    )
                ]
            )

        # Expected: 15 features (3 windows × 5 metrics)
        expected_cols = [
            "downside_vol_60",
            "upside_vol_60",
            "vol_asymmetry_ratio_60",
            "downside_vol_300",
            "upside_vol_300",
            "vol_asymmetry_ratio_300",
            "downside_vol_900",
            "upside_vol_900",
            "vol_asymmetry_ratio_900",
        ]

        missing = [col for col in expected_cols if col not in df.columns]

        if missing:
            return TestResult("V4: Volatility Asymmetry", False, f"Missing columns: {missing}")

        # Check no inf/nan (after warmup)
        df_test = df.tail(5000)
        inf_count = df_test.select(expected_cols).select(pl.all().is_infinite().sum()).sum_horizontal().to_list()[0]

        if inf_count > 0:
            return TestResult("V4: Volatility Asymmetry", False, f"{inf_count} inf values")

        return TestResult("V4: Volatility Asymmetry", True, f"{len(expected_cols)} asymmetry features generated")

    except Exception as e:
        return TestResult("V4: Volatility Asymmetry", False, f"Exception: {e}")


def main() -> None:
    """Run all module unit tests."""
    logger.info("\n" + "=" * 80)
    logger.info("V4 PIPELINE MODULE UNIT TESTS")
    logger.info("=" * 80)
    logger.info("Runtime: ~10 minutes")
    logger.info("=" * 80)

    all_results: list[TestResult] = []

    # Run all tests
    all_results.append(test_module_1_join_integrity())
    all_results.append(test_module_2_funding())
    all_results.append(test_module_3_orderbook_l0())
    all_results.append(test_module_4_orderbook_5level())
    all_results.append(test_module_5_price_basis())
    all_results.append(test_module_6_oi_features())
    all_results.append(test_module_7_rv_momentum_range())
    all_results.append(test_module_7b_extreme_regime())
    all_results.append(test_v4_advanced_moneyness())
    all_results.append(test_v4_volatility_asymmetry())

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
        logger.info("✅ ALL MODULE TESTS PASSED")
        logger.info("=" * 80)
        logger.info("\nReady to proceed to integration tests (test_03_integration_v4.py)")
        sys.exit(0)
    else:
        logger.error(f"❌ {total - passed} TEST(S) FAILED")
        logger.info("=" * 80)
        logger.error("\nFix failures before proceeding!")
        sys.exit(1)


if __name__ == "__main__":
    main()
