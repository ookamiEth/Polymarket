#!/usr/bin/env python3
"""
Impute missing Binance features for dates outside coverage.

Binance data (funding, orderbook) only covers 2023-10-01 to 2025-09-30.
This module imputes missing features for:
- Early gap: 2023-09-26 to 2023-09-30 (5 days, ~432,000 rows)
- Late gap: 2025-10-01 to 2025-11-05 (35 days, ~3,024,000 rows)

Total imputed: ~3,456,000 rows (5.17% of 66.7M total rows)

Imputation strategy:
- Early gap: Forward-fill from first valid date (2023-10-01 00:00:00)
- Late gap: Backward-fill from last valid date (2025-09-30 23:59:59)
"""

import logging
from typing import Literal

import polars as pl

logger = logging.getLogger(__name__)

# Binance data valid range
BINANCE_START_TS = 1696118400  # 2023-10-01 00:00:00 UTC
BINANCE_END_TS = 1759276799  # 2025-09-30 23:59:59 UTC

# Feature groups by source
# NOTE: Column names match actual columns created in engineer_all_features_v4.py
FUNDING_FEATURES = [
    "funding_rate",
    "funding_rate_ema_60s",
    "funding_rate_ema_300s",
    "funding_rate_ema_900s",
    "funding_rate_ema_3600s",
]

ORDERBOOK_L0_FEATURES = [
    "bid_ask_spread_bps",
    "spread_ema_60s",
    "spread_ema_300s",
    "spread_ema_900s",
    "spread_ema_3600s",
    "spread_vol_60s",
    "spread_vol_300s",
    "spread_vol_900s",
    "spread_vol_3600s",
    "bid_ask_imbalance",
    "imbalance_ema_60s",
    "imbalance_ema_300s",
    "imbalance_ema_900s",
    "imbalance_ema_3600s",
    "imbalance_vol_60s",
    "imbalance_vol_300s",
    "imbalance_vol_900s",
    "imbalance_vol_3600s",
]

ORDERBOOK_5LEVEL_FEATURES = [
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

PRICE_BASIS_FEATURES = [
    "mark_index_basis_bps",
    "mark_index_ema_60s",
    "mark_index_ema_300s",
    "mark_index_ema_900s",
    "mark_index_ema_3600s",
]

OI_FEATURES = [
    "open_interest",
    "oi_ema_900s",
    "oi_ema_3600s",
]


def get_features_for_type(
    feature_type: Literal["funding", "orderbook", "orderbook_5level", "price_basis", "open_interest"],
) -> list[str]:
    """Get feature list for a given type."""
    if feature_type == "funding":
        return FUNDING_FEATURES
    elif feature_type == "orderbook":
        return ORDERBOOK_L0_FEATURES
    elif feature_type == "orderbook_5level":
        return ORDERBOOK_5LEVEL_FEATURES
    elif feature_type == "price_basis":
        return PRICE_BASIS_FEATURES
    elif feature_type == "open_interest":
        return OI_FEATURES
    else:
        raise ValueError(f"Unknown feature type: {feature_type}")


def impute_binance_features(
    df: pl.LazyFrame,
    feature_type: Literal["funding", "orderbook", "orderbook_5level", "price_basis", "open_interest"],
    valid_start_ts: int = BINANCE_START_TS,
    valid_end_ts: int = BINANCE_END_TS,
) -> pl.LazyFrame:
    """
    Impute missing Binance features for dates outside valid range.

    Args:
        df: LazyFrame with timestamp_seconds column and features to impute
        feature_type: Type of features (determines which columns to impute)
        valid_start_ts: First valid timestamp (2023-10-01 00:00:00)
        valid_end_ts: Last valid timestamp (2025-09-30 23:59:59)

    Returns:
        LazyFrame with imputed features and features_imputed flag column
    """
    features = get_features_for_type(feature_type)

    logger.info(f"  Imputing {len(features)} {feature_type} features...")
    logger.info(f"    Valid range: {valid_start_ts} to {valid_end_ts}")

    # Add imputation flag column
    df = df.with_columns(
        [
            pl.when((pl.col("timestamp_seconds") < valid_start_ts) | (pl.col("timestamp_seconds") > valid_end_ts))
            .then(pl.lit(True))
            .otherwise(pl.lit(False))
            .alias(f"{feature_type}_imputed")
        ]
    )

    # Get first valid values (for early gap forward-fill)
    # Get last valid values (for late gap backward-fill)
    # These will be computed during lazy execution

    # For each feature, apply conditional imputation
    imputed_features = []
    for feature in features:
        # Early gap: forward-fill from first valid value
        # Late gap: backward-fill from last valid value
        # Valid range: keep original values
        imputed_feature = (
            pl.when(pl.col("timestamp_seconds") < valid_start_ts)
            # Early gap: forward-fill (will get first valid value)
            .then(
                pl.when(pl.col(feature).is_null())
                .then(pl.col(feature).forward_fill())  # Forward-fill from first valid
                .otherwise(pl.col(feature))
            )
            .when(pl.col("timestamp_seconds") > valid_end_ts)
            # Late gap: backward-fill (will get last valid value)
            .then(
                pl.when(pl.col(feature).is_null())
                .then(pl.col(feature).backward_fill())  # Backward-fill from last valid
                .otherwise(pl.col(feature))
            )
            .otherwise(pl.col(feature))  # Keep original in valid range
            .alias(feature)
        )
        imputed_features.append(imputed_feature)

    # Replace features with imputed versions
    df = df.with_columns(imputed_features)

    return df


def add_global_imputation_flag(df: pl.LazyFrame) -> pl.LazyFrame:
    """
    Add global features_imputed flag based on individual imputation flags.

    Args:
        df: LazyFrame with individual imputation flag columns

    Returns:
        LazyFrame with global features_imputed column
    """
    # Check for any imputation flag columns
    imputation_flags = [col for col in df.collect_schema().names() if col.endswith("_imputed")]

    if not imputation_flags:
        # No imputation performed, add False flag
        return df.with_columns([pl.lit(False).alias("features_imputed")])

    # Combine all imputation flags with OR logic
    global_flag = pl.lit(False)
    for flag in imputation_flags:
        global_flag = global_flag | pl.col(flag)

    df = df.with_columns([global_flag.alias("features_imputed")])

    # Drop individual imputation flags (keep only global)
    df = df.drop(imputation_flags)

    return df


def log_imputation_stats(df: pl.LazyFrame, feature_type: str) -> None:
    """Log imputation statistics for a feature type."""
    # Count imputed rows
    stats = df.select(
        [
            pl.len().alias("total_rows"),
            pl.col(f"{feature_type}_imputed").sum().alias("imputed_rows"),
        ]
    ).collect()

    total = stats["total_rows"][0]
    imputed = stats["imputed_rows"][0]
    pct = (imputed / total) * 100 if total > 0 else 0

    logger.info(f"    Imputed {imputed:,} rows ({pct:.2f}%)")
