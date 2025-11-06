#!/usr/bin/env python3
"""
Feature Columns List for V3 Model
==================================

Complete list of 163+ features for consolidated_features_v3.parquet.

This module provides the FEATURE_COLS list for LightGBM model training.
Features are organized by category for clarity.

Usage:
    from feature_cols_v3 import FEATURE_COLS

    # Use in LightGBM training
    X = df[FEATURE_COLS]
    y = df["actual_outcome"]

Author: BT Research Team
Date: 2025-11-05
"""

# Black-Scholes Context (3)
BLACK_SCHOLES_FEATURES = [
    "time_remaining",
    "moneyness",
    "iv_staleness_seconds",
]

# Funding Rate Features (11)
FUNDING_FEATURES = [
    "funding_rate",
    "funding_rate_ema_60s",
    "funding_rate_ema_300s",
    "funding_rate_ema_900s",
    "funding_rate_ema_1800s",
    "funding_rate_ema_3600s",
    "funding_rate_sma_60s",
    "funding_rate_sma_300s",
    "funding_rate_sma_900s",
    "funding_rate_sma_1800s",
    "funding_rate_sma_3600s",
]

# Orderbook L0 (Top of Book) Features (32)
ORDERBOOK_L0_FEATURES = [
    # Spread (16)
    "bid_ask_spread_bps",
    "spread_ema_60s",
    "spread_ema_300s",
    "spread_ema_900s",
    "spread_ema_1800s",
    "spread_ema_3600s",
    "spread_sma_60s",
    "spread_sma_300s",
    "spread_sma_900s",
    "spread_sma_1800s",
    "spread_sma_3600s",
    "spread_vol_60s",
    "spread_vol_300s",
    "spread_vol_900s",
    "spread_vol_1800s",
    "spread_vol_3600s",
    # Imbalance (16)
    "bid_ask_imbalance",
    "imbalance_ema_60s",
    "imbalance_ema_300s",
    "imbalance_ema_900s",
    "imbalance_ema_1800s",
    "imbalance_ema_3600s",
    "imbalance_sma_60s",
    "imbalance_sma_300s",
    "imbalance_sma_900s",
    "imbalance_sma_1800s",
    "imbalance_sma_3600s",
    "imbalance_vol_60s",
    "imbalance_vol_300s",
    "imbalance_vol_900s",
    "imbalance_vol_1800s",
    "imbalance_vol_3600s",
]

# Orderbook 5-Level Depth Features (31)
ORDERBOOK_5LEVEL_FEATURES = [
    # Total volumes (2)
    "total_bid_volume_5",
    "total_ask_volume_5",
    # Volume ratios (2)
    "bid_volume_ratio_1to5",
    "ask_volume_ratio_1to5",
    # Depth imbalance (16)
    "depth_imbalance_5",
    "depth_imbalance_ema_60s",
    "depth_imbalance_ema_300s",
    "depth_imbalance_ema_900s",
    "depth_imbalance_ema_1800s",
    "depth_imbalance_ema_3600s",
    "depth_imbalance_sma_60s",
    "depth_imbalance_sma_300s",
    "depth_imbalance_sma_900s",
    "depth_imbalance_sma_1800s",
    "depth_imbalance_sma_3600s",
    "depth_imbalance_vol_60s",
    "depth_imbalance_vol_300s",
    "depth_imbalance_vol_900s",
    "depth_imbalance_vol_1800s",
    "depth_imbalance_vol_3600s",
    # Weighted mid (11)
    "weighted_mid_price_5",
    "weighted_mid_ema_60s",
    "weighted_mid_ema_300s",
    "weighted_mid_ema_900s",
    "weighted_mid_ema_1800s",
    "weighted_mid_ema_3600s",
    "weighted_mid_sma_60s",
    "weighted_mid_sma_300s",
    "weighted_mid_sma_900s",
    "weighted_mid_sma_1800s",
    "weighted_mid_sma_3600s",
]

# Price Basis Features (34)
PRICE_BASIS_FEATURES = [
    # Mark-index (11)
    "mark_index_basis_bps",
    "mark_index_ema_60s",
    "mark_index_ema_300s",
    "mark_index_ema_900s",
    "mark_index_ema_1800s",
    "mark_index_ema_3600s",
    "mark_index_sma_60s",
    "mark_index_sma_300s",
    "mark_index_sma_900s",
    "mark_index_sma_1800s",
    "mark_index_sma_3600s",
    # Mark-last (9)
    "mark_last_basis_bps",
    "mark_last_ema_300s",
    "mark_last_ema_900s",
    "mark_last_ema_1800s",
    "mark_last_ema_3600s",
    "mark_last_sma_300s",
    "mark_last_sma_900s",
    "mark_last_sma_1800s",
    "mark_last_sma_3600s",
    # Index-last (9)
    "index_last_basis_bps",
    "index_last_ema_300s",
    "index_last_ema_900s",
    "index_last_ema_1800s",
    "index_last_ema_3600s",
    "index_last_sma_300s",
    "index_last_sma_900s",
    "index_last_sma_1800s",
    "index_last_sma_3600s",
    # Ratios (5)
    "mark_index_vs_mark_last",
    "mark_index_vs_index_last",
    "mark_last_vs_index_last",
    "mark_index_dominance",
    "avg_basis_bps",
]

# Open Interest Features (6)
OPEN_INTEREST_FEATURES = [
    "open_interest",
    "oi_ema_60s",
    "oi_ema_300s",
    "oi_ema_900s",
    "oi_ema_1800s",
    "oi_ema_3600s",
]

# Realized Volatility Features (22)
RV_FEATURES = [
    # Raw (2)
    "rv_300s",
    "rv_900s",
    # rv_300s EMAs (5)
    "rv_300s_ema_60s",
    "rv_300s_ema_300s",
    "rv_300s_ema_900s",
    "rv_300s_ema_1800s",
    "rv_300s_ema_3600s",
    # rv_900s EMAs (5)
    "rv_900s_ema_60s",
    "rv_900s_ema_300s",
    "rv_900s_ema_900s",
    "rv_900s_ema_1800s",
    "rv_900s_ema_3600s",
    # rv_300s SMAs (5)
    "rv_300s_sma_60s",
    "rv_300s_sma_300s",
    "rv_300s_sma_900s",
    "rv_300s_sma_1800s",
    "rv_300s_sma_3600s",
    # rv_900s SMAs (5)
    "rv_900s_sma_60s",
    "rv_900s_sma_300s",
    "rv_900s_sma_900s",
    "rv_900s_sma_1800s",
    "rv_900s_sma_3600s",
]

# Momentum Features (22)
MOMENTUM_FEATURES = [
    # Raw (2)
    "momentum_300s",
    "momentum_900s",
    # momentum_300s EMAs (5)
    "momentum_300s_ema_60s",
    "momentum_300s_ema_300s",
    "momentum_300s_ema_900s",
    "momentum_300s_ema_1800s",
    "momentum_300s_ema_3600s",
    # momentum_900s EMAs (5)
    "momentum_900s_ema_60s",
    "momentum_900s_ema_300s",
    "momentum_900s_ema_900s",
    "momentum_900s_ema_1800s",
    "momentum_900s_ema_3600s",
    # momentum_300s SMAs (5)
    "momentum_300s_sma_60s",
    "momentum_300s_sma_300s",
    "momentum_300s_sma_900s",
    "momentum_300s_sma_1800s",
    "momentum_300s_sma_3600s",
    # momentum_900s SMAs (5)
    "momentum_900s_sma_60s",
    "momentum_900s_sma_300s",
    "momentum_900s_sma_900s",
    "momentum_900s_sma_1800s",
    "momentum_900s_sma_3600s",
]

# Range Features (22)
RANGE_FEATURES = [
    # Raw (2)
    "range_300s",
    "range_900s",
    # range_300s EMAs (5)
    "range_300s_ema_60s",
    "range_300s_ema_300s",
    "range_300s_ema_900s",
    "range_300s_ema_1800s",
    "range_300s_ema_3600s",
    # range_900s EMAs (5)
    "range_900s_ema_60s",
    "range_900s_ema_300s",
    "range_900s_ema_900s",
    "range_900s_ema_1800s",
    "range_900s_ema_3600s",
    # range_300s SMAs (5)
    "range_300s_sma_60s",
    "range_300s_sma_300s",
    "range_300s_sma_900s",
    "range_300s_sma_1800s",
    "range_300s_sma_3600s",
    # range_900s SMAs (5)
    "range_900s_sma_60s",
    "range_900s_sma_300s",
    "range_900s_sma_900s",
    "range_900s_sma_1800s",
    "range_900s_sma_3600s",
]

# Derived Ratio Features (4)
RATIO_FEATURES = [
    "rv_ratio_5m_1m",
    "rv_ratio_15m_5m",
    "rv_ratio_1h_15m",
    "rv_term_structure",
]

# EMA Features (4 raw)
EMA_FEATURES = [
    "ema_12s",
    "ema_60s",
    "ema_300s",
    "ema_900s",
]

# EMA Ratio Features (5)
EMA_RATIO_FEATURES = [
    "price_vs_ema_900",
    "ema_12s_vs_60s",
    "ema_60s_vs_300s",
    "ema_300s_vs_900s",
    "ema_spread_12s_900s",
]

# IV/RV Ratio Features (2)
IV_RV_FEATURES = [
    "iv_rv_ratio_300s",
    "iv_rv_ratio_900s",
]

# Drawdown Features (6)
DRAWDOWN_FEATURES = [
    "high_15m",
    "low_15m",
    "drawdown_from_high_15m",
    "runup_from_low_15m",
    "time_since_high_15m",
    "time_since_low_15m",
]

# Higher Moment Features (6)
HIGHER_MOMENT_FEATURES = [
    "skewness_300s",
    "kurtosis_300s",
    "downside_vol_300s",
    "upside_vol_300s",
    "vol_asymmetry_300s",
    "tail_risk_300s",
]

# Time Features (3)
TIME_FEATURES = [
    "hour_of_day_utc",
    "hour_sin",
    "hour_cos",
]

# Volatility Clustering Features (4)
VOL_CLUSTERING_FEATURES = [
    "vol_persistence_ar1",
    "vol_acceleration_300s",
    "vol_of_vol_300s",
    "garch_forecast_simple",
]

# Autocorrelation Features (4)
AUTOCORR_FEATURES = [
    "autocorr_lag5_300s",
    "hurst_300s",
    "autocorr_decay",
    "reversals_300s",
]

# ============================================================================
# MASTER FEATURE LIST (All 226 features)
# ============================================================================

FEATURE_COLS = (
    BLACK_SCHOLES_FEATURES  # 3
    + FUNDING_FEATURES  # 11
    + ORDERBOOK_L0_FEATURES  # 32
    + ORDERBOOK_5LEVEL_FEATURES  # 31
    + PRICE_BASIS_FEATURES  # 34
    + OPEN_INTEREST_FEATURES  # 6
    + RV_FEATURES  # 22
    + MOMENTUM_FEATURES  # 22
    + RANGE_FEATURES  # 22
    + RATIO_FEATURES  # 4
    + EMA_FEATURES  # 4
    + EMA_RATIO_FEATURES  # 5
    + IV_RV_FEATURES  # 2
    + DRAWDOWN_FEATURES  # 6
    + HIGHER_MOMENT_FEATURES  # 6
    + TIME_FEATURES  # 3
    + VOL_CLUSTERING_FEATURES  # 4
    + AUTOCORR_FEATURES  # 4
)

# Feature count
TOTAL_FEATURES = len(FEATURE_COLS)

# Feature categories for analysis
FEATURE_CATEGORIES = {
    "black_scholes": BLACK_SCHOLES_FEATURES,
    "funding": FUNDING_FEATURES,
    "orderbook_l0": ORDERBOOK_L0_FEATURES,
    "orderbook_5level": ORDERBOOK_5LEVEL_FEATURES,
    "price_basis": PRICE_BASIS_FEATURES,
    "open_interest": OPEN_INTEREST_FEATURES,
    "rv": RV_FEATURES,
    "momentum": MOMENTUM_FEATURES,
    "range": RANGE_FEATURES,
    "ratios": RATIO_FEATURES,
    "ema": EMA_FEATURES,
    "ema_ratios": EMA_RATIO_FEATURES,
    "iv_rv": IV_RV_FEATURES,
    "drawdown": DRAWDOWN_FEATURES,
    "higher_moments": HIGHER_MOMENT_FEATURES,
    "time": TIME_FEATURES,
    "vol_clustering": VOL_CLUSTERING_FEATURES,
    "autocorr": AUTOCORR_FEATURES,
}

# ============================================================================
# VALIDATION
# ============================================================================

if __name__ == "__main__":
    print("=" * 80)
    print("Feature Columns V3 - Summary")
    print("=" * 80)
    print(f"\nTotal Features: {TOTAL_FEATURES}")
    print("\nFeature Breakdown:")
    print("-" * 80)
    for category, features in FEATURE_CATEGORIES.items():
        print(f"{category:20s}: {len(features):3d} features")
    print("-" * 80)
    print(f"{'TOTAL':20s}: {TOTAL_FEATURES:3d} features")
    print("\n" + "=" * 80)

    # Check for duplicates
    duplicates = [f for f in FEATURE_COLS if FEATURE_COLS.count(f) > 1]
    if duplicates:
        print(f"\n⚠️  WARNING: Duplicate features detected: {set(duplicates)}")
    else:
        print("\n✓ No duplicate features detected")

    print("\nFirst 10 features:")
    for i, feature in enumerate(FEATURE_COLS[:10], 1):
        print(f"  {i:3d}. {feature}")

    print("\nLast 10 features:")
    for i, feature in enumerate(FEATURE_COLS[-10:], TOTAL_FEATURES - 9):
        print(f"  {i:3d}. {feature}")
