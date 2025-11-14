#!/usr/bin/env python3
"""
Synthetic V4 Data Generator for Testing

Generates realistic synthetic data with all 156 V4 features for testing purposes.
All generators use fixed random seeds for reproducibility.

Author: Test Infrastructure
Date: 2025-01-14
"""

from datetime import datetime
from typing import Literal

import numpy as np
import polars as pl

# Feature list from train_multi_horizon_v4.py (156 features)
FEATURE_COLS_V4 = [
    "K",
    "S",
    "T_years",
    "ask_volume_ratio_1to5",
    "autocorr_decay",
    "autocorr_lag5_300s",
    "bid_ask_imbalance",
    "bid_ask_spread_bps",
    "bid_volume_ratio_1to5",
    "depth_imbalance_5",
    "depth_imbalance_ema_300s",
    "depth_imbalance_ema_3600s",
    "depth_imbalance_ema_60s",
    "depth_imbalance_ema_900s",
    "depth_imbalance_vol_300s",
    "depth_imbalance_vol_3600s",
    "depth_imbalance_vol_60s",
    "depth_imbalance_vol_900s",
    "depth_imbalance_vol_normalized",
    "downside_vol_300",
    "downside_vol_300s",
    "downside_vol_60",
    "downside_vol_900",
    "ema_900s",
    "funding_rate",
    "funding_rate_ema_300s",
    "funding_rate_ema_3600s",
    "funding_rate_ema_60s",
    "funding_rate_ema_900s",
    "garch_forecast_simple",
    "high_15m",
    "hour_cos",
    "hour_of_day_utc",
    "hour_sin",
    "hurst_300s",
    "imbalance_ema_300s",
    "imbalance_ema_3600s",
    "imbalance_ema_60s",
    "imbalance_ema_900s",
    "imbalance_vol_300s",
    "imbalance_vol_3600s",
    "imbalance_vol_60s",
    "imbalance_vol_900s",
    "is_extreme_condition",
    "iv_minus_downside_vol",
    "iv_minus_upside_vol",
    "iv_staleness_seconds",
    "kurtosis_300s",
    "log_moneyness",
    "low_15m",
    "mark_index_basis_bps",
    "mark_index_ema_300s",
    "mark_index_ema_3600s",
    "mark_index_ema_60s",
    "mark_index_ema_900s",
    "market_regime",
    "market_regime_4way",
    "momentum_300s",
    "momentum_300s_ema_300s",
    "momentum_300s_ema_3600s",
    "momentum_300s_ema_60s",
    "momentum_300s_ema_900s",
    "momentum_900s",
    "momentum_900s_ema_300s",
    "momentum_900s_ema_3600s",
    "momentum_900s_ema_60s",
    "momentum_900s_ema_900s",
    "moneyness",
    "moneyness_cubed",
    "moneyness_distance",
    "moneyness_percentile",
    "moneyness_squared",
    "moneyness_x_time",
    "moneyness_x_vol",
    "oi_ema_3600s",
    "oi_ema_900s",
    "open_interest",
    "position_scale",
    "price_vs_ema_900",
    "range_300s",
    "range_300s_ema_300s",
    "range_300s_ema_3600s",
    "range_300s_ema_60s",
    "range_300s_ema_900s",
    "range_900s",
    "range_900s_ema_300s",
    "range_900s_ema_3600s",
    "range_900s_ema_60s",
    "range_900s_ema_900s",
    "realized_kurtosis_300",
    "realized_kurtosis_60",
    "realized_kurtosis_900",
    "realized_skewness_300",
    "realized_skewness_60",
    "realized_skewness_900",
    "returns_300s",
    "returns_60s",
    "returns_900s",
    "reversals_300s",
    "rv_300s",
    "rv_300s_ema_300s",
    "rv_300s_ema_3600s",
    "rv_300s_ema_60s",
    "rv_300s_ema_900s",
    "rv_60s",
    "rv_900s",
    "rv_900s_ema_300s",
    "rv_900s_ema_3600s",
    "rv_900s_ema_60s",
    "rv_900s_ema_900s",
    "rv_95th_percentile",
    "rv_ratio",
    "rv_ratio_15m_5m",
    "rv_ratio_1h_15m",
    "rv_ratio_5m_1m",
    "rv_term_structure",
    "sigma_mid",
    "skewness_300s",
    "spread_ema_300s",
    "spread_ema_3600s",
    "spread_ema_60s",
    "spread_ema_900s",
    "spread_vol_300s",
    "spread_vol_3600s",
    "spread_vol_60s",
    "spread_vol_900s",
    "spread_vol_normalized",
    "standardized_moneyness",
    "tail_risk_300s",
    "temporal_regime",
    "time_remaining",
    "time_since_high_15m",
    "time_since_low_15m",
    "timestamp_seconds",
    "total_ask_volume_5",
    "total_bid_volume_5",
    "upside_vol_300",
    "upside_vol_300s",
    "upside_vol_60",
    "upside_vol_900",
    "vol_acceleration_300s",
    "vol_asymmetry_300s",
    "vol_asymmetry_ratio_300",
    "vol_asymmetry_ratio_60",
    "vol_asymmetry_ratio_900",
    "vol_high_thresh",
    "vol_low_thresh",
    "vol_of_vol_300",
    "vol_of_vol_300s",
    "vol_persistence_ar1",
    "weighted_mid_ema_300s",
    "weighted_mid_ema_3600s",
    "weighted_mid_ema_60s",
    "weighted_mid_ema_900s",
    "weighted_mid_price_5",
    "weighted_mid_velocity_normalized",
]

# 12 combined regimes (3 temporal × 4 volatility)
COMBINED_REGIMES = [
    "near_low_vol_atm",
    "near_low_vol_otm",
    "near_high_vol_atm",
    "near_high_vol_otm",
    "mid_low_vol_atm",
    "mid_low_vol_otm",
    "mid_high_vol_atm",
    "mid_high_vol_otm",
    "far_low_vol_atm",
    "far_low_vol_otm",
    "far_high_vol_atm",
    "far_high_vol_otm",
]


def generate_v4_features(
    n_samples: int = 100_000,
    regime_distribution: Literal["balanced", "realistic", "sparse"] = "balanced",
    seed: int = 42,
) -> pl.DataFrame:
    """
    Generate synthetic V4 feature data with all 156 features.

    Args:
        n_samples: Number of samples to generate
        regime_distribution: Distribution of regimes
            - "balanced": Equal distribution across all 12 regimes
            - "realistic": Realistic imbalanced distribution (ATM > OTM, low_vol > high_vol)
            - "sparse": Some regimes have <10K samples (for testing edge cases)
        seed: Random seed for reproducibility

    Returns:
        DataFrame with 156 features + regime columns + timestamp/date
    """
    np.random.seed(seed)

    # Generate timestamps (Oct 2023 - Sep 2025, ~2 years)
    start_ts = int(datetime(2023, 10, 1).timestamp())
    end_ts = int(datetime(2025, 9, 30).timestamp())
    timestamps = np.sort(np.random.randint(start_ts, end_ts, n_samples))

    # Generate regime assignments based on distribution
    if regime_distribution == "balanced":
        # Equal distribution across 12 regimes
        regime_indices = np.random.choice(len(COMBINED_REGIMES), n_samples)
        combined_regimes = [COMBINED_REGIMES[i] for i in regime_indices]
    elif regime_distribution == "realistic":
        # Realistic imbalanced: ATM 60%, low_vol 70%
        probs = np.array(
            [
                0.21,
                0.14,
                0.21,
                0.14,  # near: ATM=0.42, OTM=0.28 (total 0.35)
                0.18,
                0.12,
                0.06,
                0.04,  # mid: ATM=0.24, OTM=0.16 (total 0.40)
                0.12,
                0.08,
                0.03,
                0.02,  # far: ATM=0.15, OTM=0.10 (total 0.25)
            ]
        )
        probs = probs / probs.sum()  # Normalize
        regime_indices = np.random.choice(len(COMBINED_REGIMES), n_samples, p=probs)
        combined_regimes = [COMBINED_REGIMES[i] for i in regime_indices]
    else:  # sparse
        # Some regimes have <10K samples
        probs = np.array(
            [
                0.30,
                0.25,
                0.20,
                0.15,  # near: dominant
                0.05,
                0.03,
                0.01,
                0.005,  # mid: sparse
                0.003,
                0.002,
                0.003,
                0.002,  # far: very sparse (<1K each)
            ]
        )
        probs = probs / probs.sum()
        regime_indices = np.random.choice(len(COMBINED_REGIMES), n_samples, p=probs)
        combined_regimes = [COMBINED_REGIMES[i] for i in regime_indices]

    # Extract temporal and volatility regimes from combined
    temporal_regimes = [r.split("_")[0] for r in combined_regimes]
    volatility_regimes = ["_".join(r.split("_")[1:]) for r in combined_regimes]  # low_vol_atm, etc.

    # Generate time_remaining based on temporal regime
    time_remaining = np.zeros(n_samples)
    for i, tr in enumerate(temporal_regimes):
        if tr == "near":
            time_remaining[i] = np.random.uniform(0, 300)
        elif tr == "mid":
            time_remaining[i] = np.random.uniform(300, 600)
        else:  # far
            time_remaining[i] = np.random.uniform(600, 900)

    # Generate moneyness based on volatility regime (ATM vs OTM)
    moneyness = np.zeros(n_samples)
    moneyness_distance = np.zeros(n_samples)
    for i, vr in enumerate(volatility_regimes):
        if "atm" in vr:
            moneyness[i] = np.random.uniform(0.95, 1.05)  # ATM: close to 1.0
            moneyness_distance[i] = np.abs(1.0 - moneyness[i])
        else:  # otm
            if np.random.rand() > 0.5:
                moneyness[i] = np.random.uniform(1.10, 1.30)  # OTM call
            else:
                moneyness[i] = np.random.uniform(0.70, 0.90)  # OTM put
            moneyness_distance[i] = np.abs(1.0 - moneyness[i])

    # Generate rv_900s based on volatility regime (low vs high)
    rv_900s = np.zeros(n_samples)
    for i, vr in enumerate(volatility_regimes):
        if "low_vol" in vr:
            rv_900s[i] = np.random.uniform(0.10, 0.30)  # Low vol
        else:  # high_vol
            rv_900s[i] = np.random.uniform(0.50, 0.90)  # High vol

    # Generate vol thresholds (33rd and 67th percentiles, monthly rolling)
    vol_low_thresh = np.percentile(rv_900s, 33) * np.ones(n_samples)
    vol_high_thresh = np.percentile(rv_900s, 67) * np.ones(n_samples)

    # Generate core features
    K = np.random.uniform(80000, 120000, n_samples)  # Strike price (BTC range)  # noqa: N806
    S = K * moneyness  # Spot price  # noqa: N806
    T_years = time_remaining / (365.25 * 24 * 3600)  # Time to expiry in years  # noqa: N806

    # Generate Black-Scholes baseline (simplified)
    sigma_mid = rv_900s + np.random.normal(0, 0.05, n_samples)  # IV ≈ RV + noise
    sigma_mid = np.clip(sigma_mid, 0.05, 2.0)

    # Generate outcome (binary) and prob_mid (BS probability)
    from scipy.stats import norm

    d2 = (np.log(S / K) - 0.5 * sigma_mid**2 * T_years) / (sigma_mid * np.sqrt(T_years + 1e-6))
    prob_mid = norm.cdf(d2)
    prob_mid = np.clip(prob_mid, 0.01, 0.99)  # Clip to valid probability range
    outcome = (np.random.rand(n_samples) < prob_mid).astype(float)

    # Generate residual
    residual = outcome - prob_mid

    # Generate all 156 features with realistic distributions
    data = {
        # Core features
        "K": K,
        "S": S,
        "T_years": T_years,
        "moneyness": moneyness,
        "moneyness_distance": moneyness_distance,
        "time_remaining": time_remaining,
        "sigma_mid": sigma_mid,
        "timestamp_seconds": timestamps,
        # Moneyness transformations
        "log_moneyness": np.log(moneyness),
        "moneyness_squared": moneyness**2,
        "moneyness_cubed": moneyness**3,
        "standardized_moneyness": (moneyness - 1.0) / 0.15,  # Standardized
        "moneyness_percentile": np.random.uniform(0, 1, n_samples),
        "moneyness_x_time": moneyness * time_remaining,
        "moneyness_x_vol": moneyness * rv_900s,
        # Volatility features
        "rv_60s": rv_900s * np.random.uniform(0.8, 1.2, n_samples),
        "rv_300s": rv_900s * np.random.uniform(0.9, 1.1, n_samples),
        "rv_900s": rv_900s,
        "rv_300s_ema_60s": rv_900s * np.random.uniform(0.9, 1.1, n_samples),
        "rv_300s_ema_300s": rv_900s * np.random.uniform(0.9, 1.1, n_samples),
        "rv_300s_ema_900s": rv_900s * np.random.uniform(0.9, 1.1, n_samples),
        "rv_300s_ema_3600s": rv_900s * np.random.uniform(0.9, 1.1, n_samples),
        "rv_900s_ema_60s": rv_900s * np.random.uniform(0.95, 1.05, n_samples),
        "rv_900s_ema_300s": rv_900s * np.random.uniform(0.95, 1.05, n_samples),
        "rv_900s_ema_900s": rv_900s,
        "rv_900s_ema_3600s": rv_900s * np.random.uniform(0.98, 1.02, n_samples),
        "rv_ratio": rv_900s / (rv_900s * 0.9 + 0.01),  # RV ratio
        "rv_ratio_5m_1m": np.random.uniform(0.8, 1.2, n_samples),
        "rv_ratio_15m_5m": np.random.uniform(0.9, 1.1, n_samples),
        "rv_ratio_1h_15m": np.random.uniform(0.95, 1.05, n_samples),
        "rv_term_structure": np.random.uniform(-0.1, 0.1, n_samples),
        "rv_95th_percentile": np.percentile(rv_900s, 95) * np.ones(n_samples),
        # Volatility asymmetry
        "downside_vol_60": rv_900s * np.random.uniform(0.9, 1.3, n_samples),
        "downside_vol_300": rv_900s * np.random.uniform(0.9, 1.3, n_samples),
        "downside_vol_300s": rv_900s * np.random.uniform(0.9, 1.3, n_samples),
        "downside_vol_900": rv_900s * np.random.uniform(0.9, 1.3, n_samples),
        "upside_vol_60": rv_900s * np.random.uniform(0.8, 1.2, n_samples),
        "upside_vol_300": rv_900s * np.random.uniform(0.8, 1.2, n_samples),
        "upside_vol_300s": rv_900s * np.random.uniform(0.8, 1.2, n_samples),
        "upside_vol_900": rv_900s * np.random.uniform(0.8, 1.2, n_samples),
        "vol_asymmetry_ratio_60": np.random.uniform(-0.2, 0.2, n_samples),
        "vol_asymmetry_ratio_300": np.random.uniform(-0.2, 0.2, n_samples),
        "vol_asymmetry_ratio_900": np.random.uniform(-0.2, 0.2, n_samples),
        "realized_skewness_60": np.random.normal(0, 0.5, n_samples),
        "realized_skewness_300": np.random.normal(0, 0.5, n_samples),
        "realized_skewness_900": np.random.normal(0, 0.5, n_samples),
        "realized_kurtosis_60": np.random.uniform(2, 5, n_samples),
        "realized_kurtosis_300": np.random.uniform(2, 5, n_samples),
        "realized_kurtosis_900": np.random.uniform(2, 5, n_samples),
        # Order book features
        "bid_ask_spread_bps": np.random.uniform(5, 50, n_samples),
        "bid_ask_imbalance": np.random.uniform(-0.3, 0.3, n_samples),
        "depth_imbalance_5": np.random.uniform(-0.5, 0.5, n_samples),
        "depth_imbalance_vol_normalized": np.random.uniform(-0.3, 0.3, n_samples),
        "spread_ema_60s": np.random.uniform(10, 40, n_samples),
        "spread_ema_300s": np.random.uniform(10, 40, n_samples),
        "spread_ema_900s": np.random.uniform(10, 40, n_samples),
        "spread_ema_3600s": np.random.uniform(10, 40, n_samples),
        "spread_vol_60s": np.random.uniform(5, 20, n_samples),
        "spread_vol_300s": np.random.uniform(5, 20, n_samples),
        "spread_vol_900s": np.random.uniform(5, 20, n_samples),
        "spread_vol_3600s": np.random.uniform(5, 20, n_samples),
        "spread_vol_normalized": np.random.uniform(0.1, 0.5, n_samples),
        "total_bid_volume_5": np.random.uniform(1, 100, n_samples),
        "total_ask_volume_5": np.random.uniform(1, 100, n_samples),
        "bid_volume_ratio_1to5": np.random.uniform(0.1, 0.5, n_samples),
        "ask_volume_ratio_1to5": np.random.uniform(0.1, 0.5, n_samples),
        "weighted_mid_price_5": S * np.random.uniform(0.99, 1.01, n_samples),
        "weighted_mid_velocity_normalized": np.random.normal(0, 0.1, n_samples),
        "imbalance_ema_60s": np.random.uniform(-0.2, 0.2, n_samples),
        "imbalance_ema_300s": np.random.uniform(-0.2, 0.2, n_samples),
        "imbalance_ema_900s": np.random.uniform(-0.2, 0.2, n_samples),
        "imbalance_ema_3600s": np.random.uniform(-0.2, 0.2, n_samples),
        "imbalance_vol_60s": np.random.uniform(0.05, 0.3, n_samples),
        "imbalance_vol_300s": np.random.uniform(0.05, 0.3, n_samples),
        "imbalance_vol_900s": np.random.uniform(0.05, 0.3, n_samples),
        "imbalance_vol_3600s": np.random.uniform(0.05, 0.3, n_samples),
        "depth_imbalance_ema_60s": np.random.uniform(-0.3, 0.3, n_samples),
        "depth_imbalance_ema_300s": np.random.uniform(-0.3, 0.3, n_samples),
        "depth_imbalance_ema_900s": np.random.uniform(-0.3, 0.3, n_samples),
        "depth_imbalance_ema_3600s": np.random.uniform(-0.3, 0.3, n_samples),
        "depth_imbalance_vol_60s": np.random.uniform(0.1, 0.4, n_samples),
        "depth_imbalance_vol_300s": np.random.uniform(0.1, 0.4, n_samples),
        "depth_imbalance_vol_900s": np.random.uniform(0.1, 0.4, n_samples),
        "depth_imbalance_vol_3600s": np.random.uniform(0.1, 0.4, n_samples),
        "weighted_mid_ema_60s": S * np.random.uniform(0.99, 1.01, n_samples),
        "weighted_mid_ema_300s": S * np.random.uniform(0.99, 1.01, n_samples),
        "weighted_mid_ema_900s": S * np.random.uniform(0.99, 1.01, n_samples),
        "weighted_mid_ema_3600s": S * np.random.uniform(0.99, 1.01, n_samples),
        # Funding rate features
        "funding_rate": np.random.normal(0.0001, 0.0005, n_samples),
        "funding_rate_ema_60s": np.random.normal(0.0001, 0.0004, n_samples),
        "funding_rate_ema_300s": np.random.normal(0.0001, 0.0003, n_samples),
        "funding_rate_ema_900s": np.random.normal(0.0001, 0.0003, n_samples),
        "funding_rate_ema_3600s": np.random.normal(0.0001, 0.0002, n_samples),
        # Price dynamics
        "high_15m": S * np.random.uniform(1.0, 1.01, n_samples),
        "low_15m": S * np.random.uniform(0.99, 1.0, n_samples),
        "ema_900s": S * np.random.uniform(0.995, 1.005, n_samples),
        "price_vs_ema_900": np.random.uniform(-0.01, 0.01, n_samples),
        "time_since_high_15m": np.random.uniform(0, 900, n_samples),
        "time_since_low_15m": np.random.uniform(0, 900, n_samples),
        # Momentum and range
        "momentum_300s": np.random.normal(0, 0.01, n_samples),
        "momentum_900s": np.random.normal(0, 0.015, n_samples),
        "momentum_300s_ema_60s": np.random.normal(0, 0.01, n_samples),
        "momentum_300s_ema_300s": np.random.normal(0, 0.01, n_samples),
        "momentum_300s_ema_900s": np.random.normal(0, 0.01, n_samples),
        "momentum_300s_ema_3600s": np.random.normal(0, 0.01, n_samples),
        "momentum_900s_ema_60s": np.random.normal(0, 0.015, n_samples),
        "momentum_900s_ema_300s": np.random.normal(0, 0.015, n_samples),
        "momentum_900s_ema_900s": np.random.normal(0, 0.015, n_samples),
        "momentum_900s_ema_3600s": np.random.normal(0, 0.015, n_samples),
        "range_300s": S * np.random.uniform(0.005, 0.02, n_samples),
        "range_900s": S * np.random.uniform(0.01, 0.03, n_samples),
        "range_300s_ema_60s": S * np.random.uniform(0.005, 0.02, n_samples),
        "range_300s_ema_300s": S * np.random.uniform(0.005, 0.02, n_samples),
        "range_300s_ema_900s": S * np.random.uniform(0.005, 0.02, n_samples),
        "range_300s_ema_3600s": S * np.random.uniform(0.005, 0.02, n_samples),
        "range_900s_ema_60s": S * np.random.uniform(0.01, 0.03, n_samples),
        "range_900s_ema_300s": S * np.random.uniform(0.01, 0.03, n_samples),
        "range_900s_ema_900s": S * np.random.uniform(0.01, 0.03, n_samples),
        "range_900s_ema_3600s": S * np.random.uniform(0.01, 0.03, n_samples),
        "returns_60s": np.random.normal(0, 0.005, n_samples),
        "returns_300s": np.random.normal(0, 0.01, n_samples),
        "returns_900s": np.random.normal(0, 0.015, n_samples),
        "reversals_300s": np.random.uniform(-0.01, 0.01, n_samples),
        # Advanced microstructure
        "autocorr_lag5_300s": np.random.uniform(-0.1, 0.3, n_samples),
        "autocorr_decay": np.random.uniform(0.5, 0.95, n_samples),
        "hurst_300s": np.random.uniform(0.4, 0.6, n_samples),
        "tail_risk_300s": np.random.uniform(0, 0.05, n_samples),
        "skewness_300s": np.random.normal(0, 0.5, n_samples),
        "kurtosis_300s": np.random.uniform(2, 5, n_samples),
        "vol_of_vol_300": rv_900s * np.random.uniform(0.1, 0.3, n_samples),
        "vol_of_vol_300s": rv_900s * np.random.uniform(0.1, 0.3, n_samples),
        "vol_acceleration_300s": np.random.normal(0, 0.01, n_samples),
        "vol_persistence_ar1": np.random.uniform(0.7, 0.95, n_samples),
        "vol_asymmetry_300s": np.random.uniform(-0.2, 0.2, n_samples),
        # Mark-index basis and open interest
        "mark_index_basis_bps": np.random.uniform(-20, 20, n_samples),
        "mark_index_ema_60s": np.random.uniform(-15, 15, n_samples),
        "mark_index_ema_300s": np.random.uniform(-10, 10, n_samples),
        "mark_index_ema_900s": np.random.uniform(-5, 5, n_samples),
        "mark_index_ema_3600s": np.random.uniform(-3, 3, n_samples),
        "open_interest": np.random.uniform(100, 10000, n_samples),
        "oi_ema_900s": np.random.uniform(100, 10000, n_samples),
        "oi_ema_3600s": np.random.uniform(100, 10000, n_samples),
        # Volatility forecasting
        "garch_forecast_simple": rv_900s * np.random.uniform(0.95, 1.05, n_samples),
        "iv_minus_downside_vol": np.random.uniform(-0.1, 0.1, n_samples),
        "iv_minus_upside_vol": np.random.uniform(-0.1, 0.1, n_samples),
        "iv_staleness_seconds": np.random.uniform(0, 300, n_samples),
        # Time features
        "hour_of_day_utc": np.random.randint(0, 24, n_samples),
        "hour_sin": np.sin(2 * np.pi * np.random.randint(0, 24, n_samples) / 24),
        "hour_cos": np.cos(2 * np.pi * np.random.randint(0, 24, n_samples) / 24),
        # Regime features
        "temporal_regime": temporal_regimes,
        "market_regime": volatility_regimes,  # Store full volatility regime
        "market_regime_4way": volatility_regimes,  # Same as market_regime for V4
        "combined_regime": combined_regimes,
        "vol_low_thresh": vol_low_thresh,
        "vol_high_thresh": vol_high_thresh,
        # Extreme condition and position scale
        "is_extreme_condition": (rv_900s > 3 * np.median(rv_900s)).astype(float),
        "position_scale": np.where(rv_900s > 3 * np.median(rv_900s), 0.5, 1.0),
        # Target variables
        "outcome": outcome,
        "prob_mid": prob_mid,
        "residual": residual,
    }

    # Create DataFrame
    df = pl.DataFrame(data)

    # Add date column
    df = df.with_columns([pl.from_epoch(pl.col("timestamp_seconds"), time_unit="s").cast(pl.Date).alias("date")])

    # Verify all 156 features are present
    missing_features = set(FEATURE_COLS_V4) - set(df.columns)
    if missing_features:
        raise ValueError(f"Missing {len(missing_features)} features: {sorted(missing_features)}")

    return df


def generate_edge_case_data(
    case: Literal["nan_features", "nan_outcomes", "duplicates", "extreme_values"],
) -> pl.DataFrame:
    """Generate edge case datasets for testing error handling."""
    if case == "nan_features":
        # Some features have NaN values
        df = generate_v4_features(n_samples=1000, seed=43)
        df = df.with_columns(
            [pl.when(pl.col("timestamp_seconds") % 10 == 0).then(None).otherwise(pl.col("rv_900s")).alias("rv_900s")]
        )
        return df
    elif case == "nan_outcomes":
        # Some outcomes are NaN (corrupted data)
        df = generate_v4_features(n_samples=1000, seed=44)
        df = df.with_columns(
            [pl.when(pl.col("timestamp_seconds") % 5 == 0).then(None).otherwise(pl.col("outcome")).alias("outcome")]
        )
        return df
    elif case == "duplicates":
        # Duplicate timestamps with different contract_ids
        df1 = generate_v4_features(n_samples=500, seed=45)
        df2 = generate_v4_features(n_samples=500, seed=46)
        # Force same timestamps
        df2 = df2.with_columns([pl.col("timestamp_seconds").alias("timestamp_seconds")])
        return pl.concat([df1, df2])
    elif case == "extreme_values":
        # Extreme values that might break models
        df = generate_v4_features(n_samples=1000, seed=47)
        df = df.with_columns(
            [
                pl.when(pl.col("timestamp_seconds") % 20 == 0)
                .then(10.0)  # Extreme volatility
                .otherwise(pl.col("rv_900s"))
                .alias("rv_900s"),
                pl.when(pl.col("timestamp_seconds") % 30 == 0)
                .then(0.001)  # Near-zero moneyness
                .otherwise(pl.col("moneyness"))
                .alias("moneyness"),
            ]
        )
        return df
    else:
        raise ValueError(f"Unknown edge case: {case}")


if __name__ == "__main__":
    # Test generator
    print("Generating 100K balanced samples...")
    df = generate_v4_features(n_samples=100_000, regime_distribution="balanced", seed=42)
    print(f"✓ Generated {len(df):,} rows with {len(df.columns)} columns")
    print(f"✓ Features: {len([c for c in df.columns if c in FEATURE_COLS_V4])}/156")
    print(f"✓ Regime distribution:\n{df['combined_regime'].value_counts().sort('combined_regime')}")
    print(f"\n✓ Schema:\n{df.schema}")
