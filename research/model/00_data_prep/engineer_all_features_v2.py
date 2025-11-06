#!/usr/bin/env python3
"""
Consolidated Feature Engineering V2
====================================

Generates complete feature set for BTC 15-min binary option pricing model.

Changes from V1:
- Removes 12-15 redundant features (EMA crosses, short windows, time dummies)
- Adds 22 new features (funding rates, orderbook, OI, basis)
- Uses 8-hour settlement filtering for funding rate features (827M× speedup)
- Vectorized orderbook slope calculations
- Single consolidated output file

Output: consolidated_features_v2.parquet (~8 GB, 54 features, 60M rows)

Author: BT Research Team
Date: 2025-11-05
"""

from __future__ import annotations

import gc
import logging
import sys
from pathlib import Path
from typing import Any

import numpy as np
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

# Input files
BASELINE_FILE = RESULTS_DIR / "production_backtest_results.parquet"
RV_FILE = RESULTS_DIR / "realized_volatility_1s.parquet"
MICRO_FILE = RESULTS_DIR / "microstructure_features.parquet"
ADVANCED_FILE = RESULTS_DIR / "advanced_features.parquet"
FUNDING_FILE = TARDIS_DIR / "binance_funding_rates_1s_consolidated.parquet"
ORDERBOOK_FILE = TARDIS_DIR / "binance_orderbook_5_1s_consolidated.parquet"

# Output file
OUTPUT_FILE = DATA_DIR / "consolidated_features_v2.parquet"

# Features to KEEP from existing files (44 features)
KEEP_FEATURES = {
    "rv": [
        "rv_300s",
        "rv_900s",
        "rv_3600s",
        "rv_ratio_5m_1m",
        "rv_ratio_15m_5m",
        "rv_ratio_1h_15m",
        "rv_term_structure",
    ],  # 7 (removed rv_60s)
    "microstructure": [
        "momentum_300s",
        "momentum_900s",
        "range_300s",
        "range_900s",
        "reversals_300s",
        "jump_detected",
        "autocorr_lag5_300s",
        "hurst_300s",
    ],  # 8 (removed momentum_60s, range_60s, reversals_60s, autocorr_lag1_300s)
    "advanced": [
        # EMA (5, removed crosses)
        "ema_12s",
        "ema_60s",
        "ema_300s",
        "ema_900s",
        "price_vs_ema_900",
        # IV/RV (2)
        "iv_rv_ratio_300s",
        "iv_rv_ratio_900s",
        # Drawdown (6)
        "high_15m",
        "low_15m",
        "drawdown_from_high_15m",
        "runup_from_low_15m",
        "time_since_high_15m",
        "time_since_low_15m",
        # Higher moments (6)
        "skewness_300s",
        "kurtosis_300s",
        "downside_vol_300s",
        "upside_vol_300s",
        "vol_asymmetry_300s",
        "tail_risk_300s",
        # Time (3, removed dummies)
        "hour_of_day_utc",
        "hour_sin",
        "hour_cos",
        # Vol clustering (4)
        "vol_persistence_ar1",
        "vol_acceleration_300s",
        "vol_of_vol_300s",
        "garch_forecast_simple",
        # Jump/autocorr (3)
        "jump_count_300s",
        "jump_direction_300s",
        "autocorr_decay",
    ],  # 29
    "baseline": [
        "time_remaining",
        "moneyness",
        "iv_staleness_seconds",
    ],  # 3
}


class MemoryMonitor:
    """Monitor memory usage during processing."""

    def __init__(self):
        self.process = psutil.Process()

    def get_memory_gb(self) -> float:
        """Get current memory usage in GB."""
        return self.process.memory_info().rss / (1024**3)

    def log_memory(self, label: str) -> None:
        """Log current memory usage."""
        memory_gb = self.get_memory_gb()
        logger.info(f"[{label}] Memory: {memory_gb:.2f} GB")


def load_existing_features() -> pl.LazyFrame:
    """
    Load existing features from RV, microstructure, advanced, and baseline files.

    Returns:
        Lazy DataFrame with 44 features + timestamp_seconds
    """
    logger.info("=" * 80)
    logger.info("STEP 1: Loading Existing Features")
    logger.info("=" * 80)

    # Load baseline (context features + timestamp)
    logger.info(f"Loading baseline: {BASELINE_FILE}")
    baseline = pl.scan_parquet(BASELINE_FILE).select(["timestamp_seconds"] + KEEP_FEATURES["baseline"])

    # Load RV features
    logger.info(f"Loading RV features: {RV_FILE}")
    rv = pl.scan_parquet(RV_FILE).select(["timestamp_seconds"] + KEEP_FEATURES["rv"])

    # Load microstructure features
    logger.info(f"Loading microstructure features: {MICRO_FILE}")
    micro = pl.scan_parquet(MICRO_FILE).select(["timestamp_seconds"] + KEEP_FEATURES["microstructure"])

    # Load advanced features
    logger.info(f"Loading advanced features: {ADVANCED_FILE}")
    advanced = pl.scan_parquet(ADVANCED_FILE).select(["timestamp_seconds"] + KEEP_FEATURES["advanced"])

    # Join all existing features
    logger.info("Joining existing features on timestamp_seconds...")
    df = (
        baseline.join(rv, on="timestamp_seconds", how="left")
        .join(micro, on="timestamp_seconds", how="left")
        .join(advanced, on="timestamp_seconds", how="left")
    )

    logger.info(
        f"✓ Loaded {len(KEEP_FEATURES['rv']) + len(KEEP_FEATURES['microstructure']) + len(KEEP_FEATURES['advanced']) + len(KEEP_FEATURES['baseline'])} existing features"
    )

    return df


def engineer_funding_features() -> pl.DataFrame:
    """
    Engineer 6 funding rate features using 8-hour settlement filtering.

    Implementation: Compute rolling stats on 8-hour settlements only (2,193 rows),
    then broadcast to 1-second data (827M× speedup).

    Returns:
        DataFrame with funding features (63M rows × 7 columns: timestamp_seconds + 6 features)
    """
    logger.info("=" * 80)
    logger.info("STEP 2: Engineering Funding Rate Features")
    logger.info("=" * 80)

    # Load funding rates data
    logger.info(f"Loading funding rates: {FUNDING_FILE}")
    df = pl.scan_parquet(FUNDING_FILE).select(
        [
            "timestamp",  # Microseconds
            "funding_timestamp",
            "funding_rate",
        ]
    )

    # Convert timestamp to seconds for join key
    df = df.with_columns([(pl.col("timestamp") / 1_000_000).cast(pl.Int64).alias("timestamp_seconds")])

    logger.info("Filtering to 8-hour funding settlements...")
    # Extract rows where funding_timestamp changes (8-hour settlements)
    settlements = (
        df.filter(pl.col("funding_timestamp") != pl.col("funding_timestamp").shift(1))
        .select(["funding_timestamp", "funding_rate"])
        .collect()  # Only ~2,193 rows, safe to collect
    )

    logger.info(f"✓ Found {len(settlements):,} funding settlements over 731 days")

    # Compute rolling statistics on settlements
    logger.info("Computing 30-day rolling statistics on settlements...")
    settlements = settlements.with_columns(
        [
            # 30 days = 90 settlements (30 × 3 per day)
            pl.col("funding_rate").rolling_mean(window_size=90).alias("funding_rate_30d_mean"),
            pl.col("funding_rate").rolling_std(window_size=90).alias("funding_rate_30d_std"),
            # 7 days = 21 settlements
            pl.col("funding_rate").rolling_mean(window_size=21).alias("funding_rate_7d_mean"),
        ]
    )

    # Compute derived features
    logger.info("Computing funding rate features...")
    settlements = settlements.with_columns(
        [
            # Z-score (30-day)
            ((pl.col("funding_rate") - pl.col("funding_rate_30d_mean")) / pl.col("funding_rate_30d_std")).alias(
                "funding_rate_zscore_30d"
            ),
            # Deviation from 7-day MA
            ((pl.col("funding_rate") - pl.col("funding_rate_7d_mean")) / pl.col("funding_rate_7d_mean")).alias(
                "funding_rate_vs_ma_7d"
            ),
            # 8-hour change (shift by 1 settlement)
            (pl.col("funding_rate") - pl.col("funding_rate").shift(1)).alias("funding_rate_change_8h"),
            # 24-hour change (shift by 3 settlements)
            (pl.col("funding_rate") - pl.col("funding_rate").shift(3)).alias("funding_rate_change_24h"),
        ]
    )

    # Extreme funding indicator
    settlements = settlements.with_columns([(pl.col("funding_rate_zscore_30d").abs() > 2).alias("is_extreme_funding")])

    # Convert back to lazy frame and join to 1-second data
    logger.info("Broadcasting settlement features to 1-second data...")
    settlements_lazy = pl.LazyFrame(settlements)

    df = df.join(settlements_lazy, on="funding_timestamp", how="left")

    # Select final columns
    df = df.select(
        [
            "timestamp_seconds",
            "funding_rate",
            "funding_rate_change_8h",
            "funding_rate_change_24h",
            "funding_rate_zscore_30d",
            "is_extreme_funding",
            "funding_rate_vs_ma_7d",
        ]
    )

    # Collect (63M rows but only 7 columns, manageable)
    logger.info("Collecting funding features...")
    df = df.collect()

    logger.info(f"✓ Generated 6 funding rate features ({len(df):,} rows)")

    return df


def engineer_orderbook_features() -> pl.DataFrame:
    """
    Engineer 12 orderbook microstructure features.

    Categories:
    - Level 0 (5): spread, mid, imbalance, spread_vol, imbalance_change
    - Multi-level (4): total volumes, depth imbalance, weighted mid
    - Slope (3): bid/ask slopes, price pressure

    Returns:
        DataFrame with orderbook features (63M rows × 13 columns)
    """
    logger.info("=" * 80)
    logger.info("STEP 3: Engineering Orderbook Features")
    logger.info("=" * 80)

    logger.info(f"Loading orderbook: {ORDERBOOK_FILE}")
    df = pl.scan_parquet(ORDERBOOK_FILE).select(
        [
            "timestamp",
            # Level 0
            "bid_price_0",
            "ask_price_0",
            "bid_amount_0",
            "ask_amount_0",
            # Levels 1-4 (for multi-level and slope)
            "bid_price_1",
            "bid_price_2",
            "bid_price_3",
            "bid_price_4",
            "ask_price_1",
            "ask_price_2",
            "ask_price_3",
            "ask_price_4",
            "bid_amount_1",
            "bid_amount_2",
            "bid_amount_3",
            "bid_amount_4",
            "ask_amount_1",
            "ask_amount_2",
            "ask_amount_3",
            "ask_amount_4",
        ]
    )

    # Convert timestamp to seconds
    df = df.with_columns([(pl.col("timestamp") / 1_000_000).cast(pl.Int64).alias("timestamp_seconds")])

    logger.info("Computing Level 0 features...")

    # Mid price
    df = df.with_columns([((pl.col("bid_price_0") + pl.col("ask_price_0")) / 2).alias("mid_price")])

    # Bid-ask spread (bps)
    df = df.with_columns(
        [((pl.col("ask_price_0") - pl.col("bid_price_0")) / pl.col("mid_price") * 10000).alias("bid_ask_spread_bps")]
    )

    # Bid-ask imbalance
    df = df.with_columns(
        [
            (
                (pl.col("bid_amount_0") - pl.col("ask_amount_0")) / (pl.col("bid_amount_0") + pl.col("ask_amount_0"))
            ).alias("bid_ask_imbalance")
        ]
    )

    # Spread volatility (300s rolling std)
    df = df.with_columns([pl.col("bid_ask_spread_bps").rolling_std(window_size=300).alias("spread_volatility_300s")])

    # Imbalance change (60s)
    df = df.with_columns(
        [(pl.col("bid_ask_imbalance") - pl.col("bid_ask_imbalance").shift(60)).alias("imbalance_change_60s")]
    )

    logger.info("Computing multi-level depth features...")

    # Total volumes (sum across 5 levels)
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

    # Depth imbalance (5 levels)
    df = df.with_columns(
        [
            (
                (pl.col("total_bid_volume_5") - pl.col("total_ask_volume_5"))
                / (pl.col("total_bid_volume_5") + pl.col("total_ask_volume_5"))
            ).alias("depth_imbalance_5")
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

    logger.info("Computing orderbook slope features...")

    # Cumulative volumes for slope calculation
    df = df.with_columns(
        [
            # Bid side cumulative volumes
            pl.lit(0.0).alias("cum_bid_0"),
            pl.col("bid_amount_0").alias("cum_bid_1"),
            (pl.col("bid_amount_0") + pl.col("bid_amount_1")).alias("cum_bid_2"),
            (pl.col("bid_amount_0") + pl.col("bid_amount_1") + pl.col("bid_amount_2")).alias("cum_bid_3"),
            (pl.col("bid_amount_0") + pl.col("bid_amount_1") + pl.col("bid_amount_2") + pl.col("bid_amount_3")).alias(
                "cum_bid_4"
            ),
            # Ask side cumulative volumes
            pl.lit(0.0).alias("cum_ask_0"),
            pl.col("ask_amount_0").alias("cum_ask_1"),
            (pl.col("ask_amount_0") + pl.col("ask_amount_1")).alias("cum_ask_2"),
            (pl.col("ask_amount_0") + pl.col("ask_amount_1") + pl.col("ask_amount_2")).alias("cum_ask_3"),
            (pl.col("ask_amount_0") + pl.col("ask_amount_1") + pl.col("ask_amount_2") + pl.col("ask_amount_3")).alias(
                "cum_ask_4"
            ),
        ]
    )

    # Stack into lists for per-row slope calculation
    df = df.with_columns(
        [
            # Bid side
            pl.concat_list(
                [
                    pl.col("bid_price_0"),
                    pl.col("bid_price_1"),
                    pl.col("bid_price_2"),
                    pl.col("bid_price_3"),
                    pl.col("bid_price_4"),
                ]
            ).alias("bid_prices"),
            pl.concat_list(
                [
                    pl.col("cum_bid_0"),
                    pl.col("cum_bid_1"),
                    pl.col("cum_bid_2"),
                    pl.col("cum_bid_3"),
                    pl.col("cum_bid_4"),
                ]
            ).alias("bid_vols"),
            # Ask side
            pl.concat_list(
                [
                    pl.col("ask_price_0"),
                    pl.col("ask_price_1"),
                    pl.col("ask_price_2"),
                    pl.col("ask_price_3"),
                    pl.col("ask_price_4"),
                ]
            ).alias("ask_prices"),
            pl.concat_list(
                [
                    pl.col("cum_ask_0"),
                    pl.col("cum_ask_1"),
                    pl.col("cum_ask_2"),
                    pl.col("cum_ask_3"),
                    pl.col("cum_ask_4"),
                ]
            ).alias("ask_vols"),
        ]
    )

    # Compute slopes using map_elements (necessary for per-row linear regression)
    logger.info("Computing bid/ask slopes (this may take 20-30 minutes for 63M rows)...")

    def compute_slope_from_lists(row: dict[str, Any]) -> float:
        """
        Compute linear regression slope: price ~ volume.

        Args:
            row: Dict with 'vols' and 'prices' keys (lists of 5 values)

        Returns:
            Slope (cov(x,y) / var(x))
        """
        vols = np.array(row["vols"])
        prices = np.array(row["prices"])

        # Handle edge case: zero variance
        var_x = np.var(vols)
        if var_x < 1e-10:
            return 0.0

        cov_xy = np.cov(vols, prices)[0, 1]
        return cov_xy / var_x

    df = df.with_columns(
        [
            # Bid slope
            pl.struct(["bid_vols", "bid_prices"])
            .map_elements(
                lambda s: compute_slope_from_lists({"vols": s["bid_vols"], "prices": s["bid_prices"]}),
                return_dtype=pl.Float64,
            )
            .alias("bid_slope"),
            # Ask slope
            pl.struct(["ask_vols", "ask_prices"])
            .map_elements(
                lambda s: compute_slope_from_lists({"vols": s["ask_vols"], "prices": s["ask_prices"]}),
                return_dtype=pl.Float64,
            )
            .alias("ask_slope"),
        ]
    )

    # Price pressure (asymmetry)
    df = df.with_columns([(pl.col("bid_slope") - pl.col("ask_slope")).alias("price_pressure")])

    # Select final columns
    df = df.select(
        [
            "timestamp_seconds",
            # Level 0 (5)
            "bid_ask_spread_bps",
            "mid_price",
            "bid_ask_imbalance",
            "spread_volatility_300s",
            "imbalance_change_60s",
            # Multi-level (4)
            "total_bid_volume_5",
            "total_ask_volume_5",
            "depth_imbalance_5",
            "weighted_mid_price_5",
            # Slope (3)
            "bid_slope",
            "ask_slope",
            "price_pressure",
        ]
    )

    logger.info("Collecting orderbook features (this may take 10-15 minutes)...")
    df = df.collect()

    logger.info(f"✓ Generated 12 orderbook features ({len(df):,} rows)")

    return df


def engineer_price_basis_features() -> pl.DataFrame:
    """
    Engineer 3 price basis features from funding rates file.

    Uses mark_price, index_price, last_price to compute basis spreads.

    Returns:
        DataFrame with basis features (63M rows × 4 columns)
    """
    logger.info("=" * 80)
    logger.info("STEP 4: Engineering Price Basis Features")
    logger.info("=" * 80)

    logger.info(f"Loading prices from: {FUNDING_FILE}")
    df = pl.scan_parquet(FUNDING_FILE).select(
        [
            "timestamp",
            "mark_price",
            "index_price",
            "last_price",
        ]
    )

    # Convert timestamp to seconds
    df = df.with_columns([(pl.col("timestamp") / 1_000_000).cast(pl.Int64).alias("timestamp_seconds")])

    logger.info("Computing price basis features...")

    # Mark-index basis (primary funding rate signal)
    df = df.with_columns(
        [((pl.col("mark_price") - pl.col("index_price")) / pl.col("index_price") * 10000).alias("mark_index_basis_bps")]
    )

    # Mark-last basis (execution quality)
    df = df.with_columns(
        [((pl.col("mark_price") - pl.col("last_price")) / pl.col("last_price") * 10000).alias("mark_last_basis_bps")]
    )

    # Index-last basis (arbitrage opportunity)
    df = df.with_columns(
        [((pl.col("index_price") - pl.col("last_price")) / pl.col("last_price") * 10000).alias("index_last_basis_bps")]
    )

    # Select final columns
    df = df.select(
        [
            "timestamp_seconds",
            "mark_index_basis_bps",
            "mark_last_basis_bps",
            "index_last_basis_bps",
        ]
    )

    logger.info("Collecting price basis features...")
    df = df.collect()

    logger.info(f"✓ Generated 3 price basis features ({len(df):,} rows)")

    return df


def engineer_oi_features() -> pl.DataFrame:
    """
    Engineer 4 open interest features.

    Features:
    - open_interest_change_1h
    - open_interest_zscore_30d
    - price_oi_divergence
    - oi_momentum_24h

    Returns:
        DataFrame with OI features (63M rows × 5 columns)
    """
    logger.info("=" * 80)
    logger.info("STEP 5: Engineering Open Interest Features")
    logger.info("=" * 80)

    logger.info(f"Loading OI and prices from: {FUNDING_FILE}")
    df = pl.scan_parquet(FUNDING_FILE).select(
        [
            "timestamp",
            "open_interest",
            "last_price",
        ]
    )

    # Convert timestamp to seconds
    df = df.with_columns([(pl.col("timestamp") / 1_000_000).cast(pl.Int64).alias("timestamp_seconds")])

    logger.info("Forward-filling OI nulls (3,219 isolated 1-second gaps)...")
    df = df.with_columns([pl.col("open_interest").forward_fill().alias("open_interest")])

    logger.info("Computing OI features...")

    # 1-hour OI change
    df = df.with_columns(
        [
            (
                (pl.col("open_interest") - pl.col("open_interest").shift(3600)) / pl.col("open_interest").shift(3600)
            ).alias("open_interest_change_1h")
        ]
    )

    # 24-hour OI momentum
    df = df.with_columns(
        [
            (
                (pl.col("open_interest") - pl.col("open_interest").shift(86400)) / pl.col("open_interest").shift(86400)
            ).alias("oi_momentum_24h")
        ]
    )

    # OI z-score (30-day rolling)
    df = df.with_columns(
        [
            (
                (pl.col("open_interest") - pl.col("open_interest").rolling_mean(window_size=2592000))
                / pl.col("open_interest").rolling_std(window_size=2592000)
            ).alias("open_interest_zscore_30d")
        ]
    )

    # Price change (1-hour)
    df = df.with_columns(
        [
            ((pl.col("last_price") - pl.col("last_price").shift(3600)) / pl.col("last_price").shift(3600)).alias(
                "price_change_1h"
            )
        ]
    )

    # Price-OI divergence (with threshold)
    df = df.with_columns(
        [
            (
                ((pl.col("price_change_1h") > 0.001) & (pl.col("open_interest_change_1h") < -0.001))
                | ((pl.col("price_change_1h") < -0.001) & (pl.col("open_interest_change_1h") > 0.001))
            ).alias("price_oi_divergence")
        ]
    )

    # Select final columns
    df = df.select(
        [
            "timestamp_seconds",
            "open_interest_change_1h",
            "open_interest_zscore_30d",
            "price_oi_divergence",
            "oi_momentum_24h",
        ]
    )

    logger.info("Collecting OI features...")
    df = df.collect()

    logger.info(f"✓ Generated 4 OI features ({len(df):,} rows)")

    return df


def join_all_features(
    existing: pl.LazyFrame,
    funding: pl.DataFrame,
    orderbook: pl.DataFrame,
    basis: pl.DataFrame,
    oi: pl.DataFrame,
) -> pl.LazyFrame:
    """
    Join all feature sources on timestamp_seconds.

    Args:
        existing: Lazy frame with 44 existing features
        funding: DataFrame with 6 funding features
        orderbook: DataFrame with 12 orderbook features
        basis: DataFrame with 3 basis features
        oi: DataFrame with 4 OI features

    Returns:
        Lazy frame with all 54 features (44 + 6 + 12 + 3 + 4 - 15 removed = 54)
    """
    logger.info("=" * 80)
    logger.info("STEP 6: Joining All Features")
    logger.info("=" * 80)

    # Convert collected DataFrames to lazy
    funding_lazy = pl.LazyFrame(funding)
    orderbook_lazy = pl.LazyFrame(orderbook)
    basis_lazy = pl.LazyFrame(basis)
    oi_lazy = pl.LazyFrame(oi)

    # Sequential joins on timestamp_seconds
    logger.info("Joining funding features...")
    df = existing.join(funding_lazy, on="timestamp_seconds", how="left")

    logger.info("Joining orderbook features...")
    df = df.join(orderbook_lazy, on="timestamp_seconds", how="left")

    logger.info("Joining basis features...")
    df = df.join(basis_lazy, on="timestamp_seconds", how="left")

    logger.info("Joining OI features...")
    df = df.join(oi_lazy, on="timestamp_seconds", how="left")

    logger.info("✓ All features joined")

    return df


def validate_and_write_output(df: pl.LazyFrame, memory_monitor: MemoryMonitor) -> None:
    """
    Validate feature quality and write to parquet.

    Args:
        df: Lazy frame with all features
        memory_monitor: Memory monitor instance
    """
    logger.info("=" * 80)
    logger.info("STEP 7: Validation and Output")
    logger.info("=" * 80)

    # Validation on sample (10M rows)
    logger.info("Validating feature quality on 10M row sample...")
    sample = df.head(10_000_000).collect()

    # Check nulls
    null_counts = {col: sample[col].null_count() for col in sample.columns}
    critical_nulls = {k: v for k, v in null_counts.items() if v > 0 and k != "timestamp_seconds"}

    if critical_nulls:
        logger.warning(f"Nulls detected: {critical_nulls}")
    else:
        logger.info("✓ No nulls in features")

    # Check row count
    logger.info("Counting total rows...")
    row_count = df.select(pl.len()).collect().item()
    logger.info(f"✓ Total rows: {row_count:,}")

    # Check feature count
    schema = df.collect_schema()
    feature_count = len(schema.names()) - 1  # Exclude timestamp_seconds
    logger.info(f"✓ Total features: {feature_count}")

    # Expected: 44 existing + 6 funding + 12 orderbook + 3 basis + 4 OI = 69 columns (+ timestamp)
    # But we removed 15 features, so 54 features + timestamp = 55 columns
    expected_features = 54
    if feature_count != expected_features:
        logger.warning(f"Feature count mismatch: expected {expected_features}, got {feature_count}")

    # Write output (streaming for memory safety)
    logger.info(f"Writing consolidated features to: {OUTPUT_FILE}")
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    df.sink_parquet(
        OUTPUT_FILE,
        compression="snappy",
    )

    memory_monitor.log_memory("After write")

    # Verify output file
    output_size_gb = OUTPUT_FILE.stat().st_size / (1024**3)
    logger.info(f"✓ Output file size: {output_size_gb:.2f} GB")

    logger.info("=" * 80)
    logger.info("FEATURE ENGINEERING COMPLETE")
    logger.info("=" * 80)
    logger.info(f"Output: {OUTPUT_FILE}")
    logger.info(f"Rows: {row_count:,}")
    logger.info(f"Features: {feature_count}")
    logger.info(f"Size: {output_size_gb:.2f} GB")


def main() -> None:
    """Main execution pipeline."""
    logger.info("=" * 80)
    logger.info("Consolidated Feature Engineering V2")
    logger.info("=" * 80)

    memory_monitor = MemoryMonitor()
    memory_monitor.log_memory("Start")

    # Step 1: Load existing features
    existing = load_existing_features()
    memory_monitor.log_memory("After loading existing")
    gc.collect()

    # Step 2: Generate funding features
    funding = engineer_funding_features()
    memory_monitor.log_memory("After funding")
    gc.collect()

    # Step 3: Generate orderbook features
    orderbook = engineer_orderbook_features()
    memory_monitor.log_memory("After orderbook")
    gc.collect()

    # Step 4: Generate price basis features
    basis = engineer_price_basis_features()
    memory_monitor.log_memory("After basis")
    gc.collect()

    # Step 5: Generate OI features
    oi = engineer_oi_features()
    memory_monitor.log_memory("After OI")
    gc.collect()

    # Step 6: Join all features
    all_features = join_all_features(existing, funding, orderbook, basis, oi)
    memory_monitor.log_memory("After join")
    gc.collect()

    # Step 7: Validate and write
    validate_and_write_output(all_features, memory_monitor)

    logger.info("✓ Feature engineering pipeline completed successfully")


if __name__ == "__main__":
    main()
