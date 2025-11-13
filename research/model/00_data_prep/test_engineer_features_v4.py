#!/usr/bin/env python3
"""
Unit Tests for Feature Engineering V4 Pipeline
===============================================

Tests each module with synthetic data to catch duplicate column issues
and other bugs before running on real data.

Author: BT Research Team
Date: 2025-11-13
"""

from __future__ import annotations

import logging
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import polars as pl
import pytest

# Import functions from main script
sys.path.insert(0, str(Path(__file__).parent))
from engineer_all_features_v4 import (
    add_advanced_moneyness_features,
    add_volatility_asymmetry_features,
    detect_extreme_conditions,
    detect_regime_with_hysteresis,
    normalize_orderbook_features,
)

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


# ==================== FAKE DATA GENERATORS ====================


def generate_fake_timestamps(n: int, start_date: str = "2024-01-01") -> pl.Series:
    """Generate fake timestamps at 1-second intervals."""
    start = int(datetime.fromisoformat(start_date).timestamp())
    return pl.Series("timestamp_seconds", list(range(start, start + n)), dtype=pl.Int64)


def generate_fake_baseline_data(n: int = 1000) -> pl.DataFrame:
    """
    Generate fake baseline features.

    Columns: timestamp_seconds, time_remaining, S, K, sigma_mid, T_years, moneyness, iv_staleness_seconds
    """
    import numpy as np

    np.random.seed(42)

    timestamps = generate_fake_timestamps(n)

    df = pl.DataFrame(
        {
            "timestamp_seconds": timestamps,
            "time_remaining": np.random.uniform(60, 900, n),
            "S": np.random.uniform(90000, 110000, n),  # BTC spot price
            "K": np.repeat([95000, 100000, 105000], n // 3 + 1)[:n],  # Strike prices
            "sigma_mid": np.random.uniform(0.3, 0.8, n),  # IV
            "T_years": np.random.uniform(0.001, 0.01, n),  # Time to expiry
            "iv_staleness_seconds": np.random.randint(0, 10, n),
        }
    )

    # Compute moneyness
    df = df.with_columns([(pl.col("S") / pl.col("K")).alias("moneyness")])

    return df


def generate_fake_funding_data(n: int = 1000) -> pl.DataFrame:
    """Generate fake funding rate features."""
    import numpy as np

    np.random.seed(43)

    timestamps = generate_fake_timestamps(n)

    df = pl.DataFrame(
        {
            "timestamp_seconds": timestamps,
            "funding_rate": np.random.uniform(-0.001, 0.001, n),
            "funding_rate_ema_60s": np.random.uniform(-0.0008, 0.0008, n),
            "funding_rate_ema_300s": np.random.uniform(-0.0006, 0.0006, n),
            "funding_rate_ema_900s": np.random.uniform(-0.0005, 0.0005, n),
            "funding_rate_ema_3600s": np.random.uniform(-0.0004, 0.0004, n),
        }
    )

    return df


def generate_fake_orderbook_l0_data(n: int = 1000) -> pl.DataFrame:
    """Generate fake orderbook L0 (top-of-book) features."""
    import numpy as np

    np.random.seed(44)

    timestamps = generate_fake_timestamps(n)

    df = pl.DataFrame(
        {
            "timestamp_seconds": timestamps,
            "bid_ask_spread_bps": np.random.uniform(1, 10, n),
            "spread_ema_60s": np.random.uniform(2, 8, n),
            "spread_ema_300s": np.random.uniform(3, 7, n),
            "spread_ema_900s": np.random.uniform(3.5, 6.5, n),
            "spread_ema_3600s": np.random.uniform(4, 6, n),
            "spread_vol_60s": np.random.uniform(0.5, 2, n),
            "spread_vol_300s": np.random.uniform(0.6, 1.8, n),
            "spread_vol_900s": np.random.uniform(0.7, 1.6, n),
            "spread_vol_3600s": np.random.uniform(0.8, 1.4, n),
            "bid_ask_imbalance": np.random.uniform(-0.5, 0.5, n),
            "imbalance_ema_60s": np.random.uniform(-0.3, 0.3, n),
            "imbalance_ema_300s": np.random.uniform(-0.2, 0.2, n),
            "imbalance_ema_900s": np.random.uniform(-0.15, 0.15, n),
            "imbalance_ema_3600s": np.random.uniform(-0.1, 0.1, n),
            "imbalance_vol_60s": np.random.uniform(0.1, 0.4, n),
            "imbalance_vol_300s": np.random.uniform(0.12, 0.35, n),
            "imbalance_vol_900s": np.random.uniform(0.14, 0.3, n),
            "imbalance_vol_3600s": np.random.uniform(0.15, 0.25, n),
        }
    )

    return df


def generate_fake_orderbook_5level_data(n: int = 1000) -> pl.DataFrame:
    """Generate fake 5-level orderbook features."""
    import numpy as np

    np.random.seed(45)

    timestamps = generate_fake_timestamps(n)

    df = pl.DataFrame(
        {
            "timestamp_seconds": timestamps,
            "total_bid_volume_5": np.random.uniform(10, 100, n),
            "total_ask_volume_5": np.random.uniform(10, 100, n),
            "bid_volume_ratio_1to5": np.random.uniform(0.3, 0.7, n),
            "ask_volume_ratio_1to5": np.random.uniform(0.3, 0.7, n),
            "depth_imbalance_5": np.random.uniform(-0.3, 0.3, n),
            "depth_imbalance_ema_60s": np.random.uniform(-0.2, 0.2, n),
            "depth_imbalance_ema_300s": np.random.uniform(-0.15, 0.15, n),
            "depth_imbalance_ema_900s": np.random.uniform(-0.1, 0.1, n),
            "depth_imbalance_ema_3600s": np.random.uniform(-0.08, 0.08, n),
            "depth_imbalance_vol_60s": np.random.uniform(0.05, 0.2, n),
            "depth_imbalance_vol_300s": np.random.uniform(0.06, 0.18, n),
            "depth_imbalance_vol_900s": np.random.uniform(0.07, 0.16, n),
            "depth_imbalance_vol_3600s": np.random.uniform(0.08, 0.14, n),
            "weighted_mid_price_5": np.random.uniform(99000, 101000, n),
            "weighted_mid_ema_60s": np.random.uniform(99200, 100800, n),
            "weighted_mid_ema_300s": np.random.uniform(99400, 100600, n),
            "weighted_mid_ema_900s": np.random.uniform(99500, 100500, n),
            "weighted_mid_ema_3600s": np.random.uniform(99600, 100400, n),
        }
    )

    return df


def generate_fake_basis_data(n: int = 1000) -> pl.DataFrame:
    """Generate fake price basis features."""
    import numpy as np

    np.random.seed(46)

    timestamps = generate_fake_timestamps(n)

    df = pl.DataFrame(
        {
            "timestamp_seconds": timestamps,
            "mark_index_basis_bps": np.random.uniform(-5, 5, n),
            "mark_index_ema_60s": np.random.uniform(-3, 3, n),
            "mark_index_ema_300s": np.random.uniform(-2, 2, n),
            "mark_index_ema_900s": np.random.uniform(-1.5, 1.5, n),
            "mark_index_ema_3600s": np.random.uniform(-1, 1, n),
        }
    )

    return df


def generate_fake_oi_data(n: int = 1000) -> pl.DataFrame:
    """Generate fake open interest features."""
    import numpy as np

    np.random.seed(47)

    timestamps = generate_fake_timestamps(n)

    df = pl.DataFrame(
        {
            "timestamp_seconds": timestamps,
            "open_interest": np.random.uniform(1e9, 2e9, n),
            "oi_ema_900s": np.random.uniform(1.1e9, 1.9e9, n),
            "oi_ema_3600s": np.random.uniform(1.2e9, 1.8e9, n),
        }
    )

    return df


def generate_fake_rv_momentum_range_data(n: int = 1000) -> pl.DataFrame:
    """Generate fake RV/momentum/range features."""
    import numpy as np

    np.random.seed(48)

    timestamps = generate_fake_timestamps(n)

    df = pl.DataFrame(
        {
            "timestamp_seconds": timestamps,
            "rv_60s": np.random.uniform(0.2, 0.8, n),
            "rv_300s": np.random.uniform(0.25, 0.75, n),
            "rv_900s": np.random.uniform(0.3, 0.7, n),
            "rv_300s_ema_60s": np.random.uniform(0.28, 0.72, n),
            "rv_300s_ema_300s": np.random.uniform(0.3, 0.7, n),
            "rv_300s_ema_900s": np.random.uniform(0.32, 0.68, n),
            "rv_300s_ema_3600s": np.random.uniform(0.34, 0.66, n),
            "rv_900s_ema_60s": np.random.uniform(0.32, 0.68, n),
            "rv_900s_ema_300s": np.random.uniform(0.34, 0.66, n),
            "rv_900s_ema_900s": np.random.uniform(0.36, 0.64, n),
            "rv_900s_ema_3600s": np.random.uniform(0.38, 0.62, n),
            "momentum_300s": np.random.uniform(-0.01, 0.01, n),
            "momentum_900s": np.random.uniform(-0.008, 0.008, n),
            "momentum_300s_ema_60s": np.random.uniform(-0.008, 0.008, n),
            "momentum_300s_ema_300s": np.random.uniform(-0.006, 0.006, n),
            "momentum_300s_ema_900s": np.random.uniform(-0.005, 0.005, n),
            "momentum_300s_ema_3600s": np.random.uniform(-0.004, 0.004, n),
            "momentum_900s_ema_60s": np.random.uniform(-0.007, 0.007, n),
            "momentum_900s_ema_300s": np.random.uniform(-0.005, 0.005, n),
            "momentum_900s_ema_900s": np.random.uniform(-0.004, 0.004, n),
            "momentum_900s_ema_3600s": np.random.uniform(-0.003, 0.003, n),
            "range_300s": np.random.uniform(100, 500, n),
            "range_900s": np.random.uniform(200, 800, n),
            "range_300s_ema_60s": np.random.uniform(150, 450, n),
            "range_300s_ema_300s": np.random.uniform(180, 420, n),
            "range_300s_ema_900s": np.random.uniform(200, 400, n),
            "range_300s_ema_3600s": np.random.uniform(220, 380, n),
            "range_900s_ema_60s": np.random.uniform(250, 750, n),
            "range_900s_ema_300s": np.random.uniform(300, 700, n),
            "range_900s_ema_900s": np.random.uniform(350, 650, n),
            "range_900s_ema_3600s": np.random.uniform(400, 600, n),
            "rv_ratio_5m_1m": np.random.uniform(0.8, 1.2, n),
            "rv_ratio_15m_5m": np.random.uniform(0.9, 1.1, n),
            "rv_ratio_1h_15m": np.random.uniform(0.95, 1.05, n),
            "rv_term_structure": np.random.uniform(-0.1, 0.1, n),
            "ema_900s": np.random.uniform(99500, 100500, n),
            "price_vs_ema_900": np.random.uniform(-0.005, 0.005, n),
        }
    )

    return df


# ==================== TEST CASES ====================


def test_module_1_baseline_generation():
    """Test baseline feature generation (no duplicates)."""
    logger.info("TEST: Module 1 - Baseline feature generation")

    df = generate_fake_baseline_data(n=100)

    # Check columns
    expected_cols = ["timestamp_seconds", "time_remaining", "S", "K", "sigma_mid", "T_years", "moneyness", "iv_staleness_seconds"]
    assert all(col in df.columns for col in expected_cols), f"Missing columns in baseline data"

    # Check no duplicates
    assert len(df.columns) == len(set(df.columns)), "Duplicate columns detected in baseline data"

    # Check data types
    assert df["timestamp_seconds"].dtype == pl.Int64
    assert df["S"].dtype == pl.Float64
    assert df["moneyness"].dtype == pl.Float64

    logger.info("✓ Module 1 test passed")


def test_module_2_funding_features():
    """Test funding rate feature generation (no duplicates)."""
    logger.info("TEST: Module 2 - Funding rate features")

    df = generate_fake_funding_data(n=100)

    # Check columns
    expected_cols = ["timestamp_seconds", "funding_rate", "funding_rate_ema_60s", "funding_rate_ema_300s", "funding_rate_ema_900s", "funding_rate_ema_3600s"]
    assert all(col in df.columns for col in expected_cols), "Missing columns in funding data"

    # Check no duplicates
    assert len(df.columns) == len(set(df.columns)), "Duplicate columns detected in funding data"

    logger.info("✓ Module 2 test passed")


def test_module_3_orderbook_l0():
    """Test orderbook L0 features (no duplicates)."""
    logger.info("TEST: Module 3 - Orderbook L0 features")

    df = generate_fake_orderbook_l0_data(n=100)

    # Check no duplicates
    assert len(df.columns) == len(set(df.columns)), "Duplicate columns detected in orderbook L0 data"

    # Check expected feature count
    assert len(df.columns) == 19, f"Expected 19 columns (1 timestamp + 18 features), got {len(df.columns)}"

    logger.info("✓ Module 3 test passed")


def test_module_4_orderbook_5level():
    """Test orderbook 5-level features (no duplicates)."""
    logger.info("TEST: Module 4 - Orderbook 5-level features")

    df = generate_fake_orderbook_5level_data(n=100)

    # Check no duplicates
    assert len(df.columns) == len(set(df.columns)), "Duplicate columns detected in orderbook 5-level data"

    # Check expected feature count
    assert len(df.columns) == 19, f"Expected 19 columns (1 timestamp + 18 features), got {len(df.columns)}"

    logger.info("✓ Module 4 test passed")


def test_module_5_basis_features():
    """Test price basis features (no duplicates)."""
    logger.info("TEST: Module 5 - Price basis features")

    df = generate_fake_basis_data(n=100)

    # Check no duplicates
    assert len(df.columns) == len(set(df.columns)), "Duplicate columns detected in basis data"

    logger.info("✓ Module 5 test passed")


def test_module_6_oi_features():
    """Test open interest features (no duplicates)."""
    logger.info("TEST: Module 6 - Open interest features")

    df = generate_fake_oi_data(n=100)

    # Check no duplicates
    assert len(df.columns) == len(set(df.columns)), "Duplicate columns detected in OI data"

    logger.info("✓ Module 6 test passed")


def test_module_7_rv_momentum_range():
    """Test RV/momentum/range features (no duplicates)."""
    logger.info("TEST: Module 7 - RV/momentum/range features")

    df = generate_fake_rv_momentum_range_data(n=100)

    # Check no duplicates
    assert len(df.columns) == len(set(df.columns)), "Duplicate columns detected in RV/momentum/range data"

    # Check expected feature count (38 features + timestamp_seconds)
    assert len(df.columns) == 38, f"Expected 38 columns, got {len(df.columns)}"

    logger.info("✓ Module 7 test passed")


def test_module_8_join_all_features():
    """Test joining all feature sources (critical test for duplicate columns)."""
    logger.info("TEST: Module 8 - Join all features")

    # Generate all feature sources
    baseline_df = generate_fake_baseline_data(n=100)
    funding_df = generate_fake_funding_data(n=100)
    orderbook_l0_df = generate_fake_orderbook_l0_data(n=100)
    orderbook_5level_df = generate_fake_orderbook_5level_data(n=100)
    basis_df = generate_fake_basis_data(n=100)
    oi_df = generate_fake_oi_data(n=100)
    rv_df = generate_fake_rv_momentum_range_data(n=100)

    # Convert to LazyFrames
    baseline_lazy = baseline_df.lazy()
    funding_lazy = funding_df.lazy()
    orderbook_l0_lazy = orderbook_l0_df.lazy()
    orderbook_5level_lazy = orderbook_5level_df.lazy()
    basis_lazy = basis_df.lazy()
    oi_lazy = oi_df.lazy()
    rv_lazy = rv_df.lazy()

    # Perform joins (same as in main pipeline)
    joined_df = (
        baseline_lazy.join(funding_lazy, on="timestamp_seconds", how="left")
        .join(orderbook_l0_lazy, on="timestamp_seconds", how="left")
        .join(orderbook_5level_lazy, on="timestamp_seconds", how="left")
        .join(basis_lazy, on="timestamp_seconds", how="left")
        .join(oi_lazy, on="timestamp_seconds", how="left")
        .join(rv_lazy, on="timestamp_seconds", how="left")
    )

    # Check schema (should not raise duplicate column error)
    try:
        schema = joined_df.collect_schema()
        logger.info(f"  Joined schema has {len(schema.names())} columns")
    except Exception as e:
        pytest.fail(f"Join failed with error: {e}")

    # Check no duplicate columns
    column_names = schema.names()
    assert len(column_names) == len(set(column_names)), f"Duplicate columns after join: {[c for c in column_names if column_names.count(c) > 1]}"

    # Collect to verify execution
    result_df = joined_df.collect()

    # Check row count (should match baseline)
    assert len(result_df) == 100, f"Expected 100 rows after join, got {len(result_df)}"

    logger.info("✓ Module 8 test passed (no duplicate columns in join)")


def test_v4_advanced_moneyness_features():
    """Test V4 advanced moneyness feature transformations."""
    logger.info("TEST: V4 - Advanced moneyness features")

    # Generate baseline data
    baseline_df = generate_fake_baseline_data(n=100)

    # Convert to LazyFrame and add timestamp column (required for V4 functions)
    df_lazy = baseline_df.lazy().with_columns(
        [(pl.col("timestamp_seconds") * 1_000_000).cast(pl.Datetime("us")).alias("timestamp")]
    )

    # Apply transformation
    try:
        df_transformed = add_advanced_moneyness_features(df_lazy)
        result = df_transformed.collect()
    except Exception as e:
        pytest.fail(f"add_advanced_moneyness_features failed: {e}")

    # Check new features exist
    required_features = [
        "log_moneyness",
        "moneyness_squared",
        "moneyness_cubed",
        "standardized_moneyness",
        "moneyness_x_time",
        "moneyness_distance",
        "moneyness_percentile",
    ]

    for feat in required_features:
        assert feat in result.columns, f"Missing V4 feature: {feat}"

    # moneyness_x_vol is optional (only created if rv_60s is available)
    # In this test it won't be present since we don't have RV data

    # Check no duplicates
    assert len(result.columns) == len(set(result.columns)), "Duplicate columns in advanced moneyness features"

    logger.info("✓ V4 advanced moneyness test passed")


def test_v4_volatility_asymmetry_features():
    """Test V4 volatility asymmetry features."""
    logger.info("TEST: V4 - Volatility asymmetry features")

    # Generate baseline + RV data
    baseline_df = generate_fake_baseline_data(n=1000)  # Need more data for rolling windows
    rv_df = generate_fake_rv_momentum_range_data(n=1000)

    # Join baseline + RV
    df = baseline_df.join(rv_df, on="timestamp_seconds", how="left")

    # Convert to LazyFrame and add timestamp
    df_lazy = df.lazy().with_columns(
        [(pl.col("timestamp_seconds") * 1_000_000).cast(pl.Datetime("us")).alias("timestamp")]
    )

    # Apply transformation
    try:
        df_transformed = add_volatility_asymmetry_features(df_lazy)
        result = df_transformed.collect()
    except Exception as e:
        pytest.fail(f"add_volatility_asymmetry_features failed: {e}")

    # Check expected features exist (windows: 60, 300, 900)
    expected_features = [
        "downside_vol_60",
        "upside_vol_60",
        "vol_asymmetry_ratio_60",
        "realized_skewness_60",
        "realized_kurtosis_60",
        "vol_of_vol_300",
    ]

    for feat in expected_features:
        assert feat in result.columns, f"Missing V4 feature: {feat}"

    # Check no duplicates
    assert len(result.columns) == len(set(result.columns)), "Duplicate columns in volatility asymmetry features"

    logger.info("✓ V4 volatility asymmetry test passed")


def test_v4_normalize_orderbook_features():
    """Test V4 orderbook normalization."""
    logger.info("TEST: V4 - Orderbook normalization")

    # Generate orderbook + RV data
    orderbook_l0_df = generate_fake_orderbook_l0_data(n=100)
    orderbook_5level_df = generate_fake_orderbook_5level_data(n=100)
    rv_df = generate_fake_rv_momentum_range_data(n=100)

    # Join all orderbook sources
    df = orderbook_l0_df.join(orderbook_5level_df, on="timestamp_seconds", how="left").join(
        rv_df, on="timestamp_seconds", how="left"
    )

    # Convert to LazyFrame
    df_lazy = df.lazy()

    # Apply normalization
    try:
        df_normalized = normalize_orderbook_features(df_lazy)
        result = df_normalized.collect()
    except Exception as e:
        pytest.fail(f"normalize_orderbook_features failed: {e}")

    # Check normalized features exist
    expected_features = [
        "spread_vol_normalized",
        "depth_imbalance_vol_normalized",
        "weighted_mid_velocity_normalized",
    ]

    for feat in expected_features:
        assert feat in result.columns, f"Missing V4 feature: {feat}"

    # Check no duplicates
    assert len(result.columns) == len(set(result.columns)), "Duplicate columns in orderbook normalization"

    logger.info("✓ V4 orderbook normalization test passed")


def test_v4_extreme_conditions():
    """Test V4 extreme condition detection."""
    logger.info("TEST: V4 - Extreme condition detection")

    # Generate RV data
    rv_df = generate_fake_rv_momentum_range_data(n=1000)

    # Convert to LazyFrame and add timestamp
    df_lazy = rv_df.lazy().with_columns(
        [(pl.col("timestamp_seconds") * 1_000_000).cast(pl.Datetime("us")).alias("timestamp")]
    )

    # Apply detection
    try:
        df_detected = detect_extreme_conditions(df_lazy)
        result = df_detected.collect()
    except Exception as e:
        pytest.fail(f"detect_extreme_conditions failed: {e}")

    # Check features exist
    expected_features = ["rv_ratio", "rv_95th_percentile", "is_extreme_condition", "position_scale"]

    for feat in expected_features:
        assert feat in result.columns, f"Missing V4 feature: {feat}"

    # Check no duplicates
    assert len(result.columns) == len(set(result.columns)), "Duplicate columns in extreme condition detection"

    logger.info("✓ V4 extreme condition test passed")


def test_v4_regime_detection():
    """Test V4 regime detection with hysteresis."""
    logger.info("TEST: V4 - Regime detection")

    # Generate baseline + RV data
    baseline_df = generate_fake_baseline_data(n=1000)
    rv_df = generate_fake_rv_momentum_range_data(n=1000)

    # Join
    df = baseline_df.join(rv_df, on="timestamp_seconds", how="left")

    # Convert to LazyFrame and add timestamp
    df_lazy = df.lazy().with_columns(
        [
            (pl.col("timestamp_seconds") * 1_000_000).cast(pl.Datetime("us")).alias("timestamp"),
            ((pl.col("S") / pl.col("K")) - 1).abs().alias("moneyness_distance"),  # Required by regime detection
        ]
    )

    # Apply detection
    try:
        df_regime = detect_regime_with_hysteresis(df_lazy)
        result = df_regime.collect()
    except Exception as e:
        pytest.fail(f"detect_regime_with_hysteresis failed: {e}")

    # Check features exist
    expected_features = ["volatility_regime", "market_regime"]

    for feat in expected_features:
        assert feat in result.columns, f"Missing V4 feature: {feat}"

    # Check no duplicates
    assert len(result.columns) == len(set(result.columns)), "Duplicate columns in regime detection"

    logger.info("✓ V4 regime detection test passed")


def test_full_pipeline_integration():
    """Integration test: Full pipeline with all modules and V4 transformations."""
    logger.info("TEST: Full pipeline integration (all modules + V4)")

    # Generate all feature sources
    baseline_df = generate_fake_baseline_data(n=1000)
    funding_df = generate_fake_funding_data(n=1000)
    orderbook_l0_df = generate_fake_orderbook_l0_data(n=1000)
    orderbook_5level_df = generate_fake_orderbook_5level_data(n=1000)
    basis_df = generate_fake_basis_data(n=1000)
    oi_df = generate_fake_oi_data(n=1000)
    rv_df = generate_fake_rv_momentum_range_data(n=1000)

    # Convert to LazyFrames
    baseline_lazy = baseline_df.lazy()
    funding_lazy = funding_df.lazy()
    orderbook_l0_lazy = orderbook_l0_df.lazy()
    orderbook_5level_lazy = orderbook_5level_df.lazy()
    basis_lazy = basis_df.lazy()
    oi_lazy = oi_df.lazy()
    rv_lazy = rv_df.lazy()

    # Perform joins
    joined_df = (
        baseline_lazy.join(funding_lazy, on="timestamp_seconds", how="left")
        .join(orderbook_l0_lazy, on="timestamp_seconds", how="left")
        .join(orderbook_5level_lazy, on="timestamp_seconds", how="left")
        .join(basis_lazy, on="timestamp_seconds", how="left")
        .join(oi_lazy, on="timestamp_seconds", how="left")
        .join(rv_lazy, on="timestamp_seconds", how="left")
    )

    # Check for 'timestamp' column from joins and drop if exists
    schema = joined_df.collect_schema()
    if "timestamp" in schema.names():
        joined_df = joined_df.drop("timestamp")

    # Add timestamp column for V4 transformations
    joined_df = joined_df.with_columns(
        [(pl.col("timestamp_seconds") * 1_000_000).cast(pl.Datetime("us")).alias("timestamp")]
    )

    # Apply V4 transformations
    try:
        df_v4 = add_advanced_moneyness_features(joined_df)
        df_v4 = add_volatility_asymmetry_features(df_v4)
        df_v4 = normalize_orderbook_features(df_v4)
        df_v4 = detect_extreme_conditions(df_v4)
        df_v4 = detect_regime_with_hysteresis(df_v4)

        # Collect final result
        result = df_v4.collect()
    except Exception as e:
        pytest.fail(f"Full pipeline failed: {e}")

    # Check no duplicate columns
    column_names = result.columns
    duplicates = [c for c in column_names if column_names.count(c) > 1]
    assert len(column_names) == len(set(column_names)), f"Duplicate columns in final output: {duplicates}"

    # Check row count
    assert len(result) == 1000, f"Expected 1000 rows, got {len(result)}"

    logger.info(f"✓ Full pipeline integration test passed ({len(result.columns)} total columns)")


# ==================== MAIN TEST RUNNER ====================


def run_all_tests():
    """Run all tests sequentially."""
    logger.info("=" * 80)
    logger.info("RUNNING FEATURE ENGINEERING V4 TESTS")
    logger.info("=" * 80)

    tests = [
        test_module_1_baseline_generation,
        test_module_2_funding_features,
        test_module_3_orderbook_l0,
        test_module_4_orderbook_5level,
        test_module_5_basis_features,
        test_module_6_oi_features,
        test_module_7_rv_momentum_range,
        test_module_8_join_all_features,
        test_v4_advanced_moneyness_features,
        test_v4_volatility_asymmetry_features,
        test_v4_normalize_orderbook_features,
        test_v4_extreme_conditions,
        test_v4_regime_detection,
        test_full_pipeline_integration,
    ]

    failed_tests = []

    for test_func in tests:
        try:
            test_func()
        except AssertionError as e:
            logger.error(f"✗ {test_func.__name__} FAILED: {e}")
            failed_tests.append((test_func.__name__, str(e)))
        except Exception as e:
            logger.error(f"✗ {test_func.__name__} ERROR: {e}")
            failed_tests.append((test_func.__name__, str(e)))

    logger.info("=" * 80)
    if failed_tests:
        logger.error(f"TESTS FAILED: {len(failed_tests)}/{len(tests)}")
        for test_name, error in failed_tests:
            logger.error(f"  - {test_name}: {error}")
        sys.exit(1)
    else:
        logger.info(f"ALL TESTS PASSED: {len(tests)}/{len(tests)} ✓")
        logger.info("=" * 80)


if __name__ == "__main__":
    run_all_tests()
