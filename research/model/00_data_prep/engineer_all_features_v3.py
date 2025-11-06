#!/usr/bin/env python3
"""
Consolidated Feature Engineering V3
====================================

Generates complete feature set for BTC 15-min binary option pricing model.

Philosophy Shift (V2 → V3):
- Remove ALL manual feature engineering (z-scores, changes, boolean indicators)
- Use raw metrics + EMAs/SMAs at standard time windows
- Let GBT model learn optimal relationships and thresholds
- Focus on orderbook spread/imbalance dynamics (not slopes)
- Emphasize mark-index basis (funding rate signal, settlement-relevant)

Standard Time Windows: 60s, 300s, 900s, 1800s, 3600s

Feature Count: 196 total
- Existing features (kept as-is): 26
- Funding rate: 11 (1 raw + 5 EMAs + 5 SMAs)
- Orderbook L0: 32 (spread + imbalance with EMAs/SMAs/volatility)
- Orderbook 5-levels: 31 (depth, volumes, ratios)
- Price basis: 11 (mark-index only: 1 raw + 5 EMAs + 5 SMAs)
- Open interest: 6 (1 raw + 5 EMAs)
- RV/Momentum/Range/EMA: 79 (raw + EMAs/SMAs + EMA ratios)

Memory Architecture: TEMPORAL CHUNKING
- Module 8 uses 3-month chunking optimized for 256GB RAM
- Processes 63M rows in ~12 quarterly chunks (~5.25M rows each)
- Joins all 6 feature sources per chunk, then concatenates
- Proven pattern from tardis/analysis/compare_iv_chunked.py

Output: consolidated_features_v3.parquet (~10-12 GB, 196 features, 63M rows)

Author: BT Research Team
Date: 2025-01-06 (Updated to 3-month chunks for 256GB RAM)
"""

from __future__ import annotations

import logging
import sys
import time
from datetime import timedelta
from pathlib import Path

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

# Output files
OUTPUT_FILE = DATA_DIR / "consolidated_features_v3.parquet"
INTERMEDIATE_DIR = DATA_DIR / "features_v3" / "intermediate"

# Standard time windows (seconds)
WINDOWS = [60, 300, 900, 1800, 3600]

# Features to KEEP from existing files (33 features)
KEEP_FEATURES = {
    "baseline": [
        "time_remaining",
        "moneyness",
        "iv_staleness_seconds",
    ],  # 3
    "advanced": [
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
        # Time (3)
        "hour_of_day_utc",
        "hour_sin",
        "hour_cos",
        # Vol clustering (4)
        "vol_persistence_ar1",
        "vol_acceleration_300s",
        "vol_of_vol_300s",
        "garch_forecast_simple",
        # EMA (4 raw)
        "ema_12s",
        "ema_60s",
        "ema_300s",
        "ema_900s",
        # IV/RV ratios (2)
        "iv_rv_ratio_300s",
        "iv_rv_ratio_900s",
    ],  # 25
    "microstructure": [
        "autocorr_lag5_300s",
        "hurst_300s",
        "autocorr_decay",
        "reversals_300s",
    ],  # 4
    "rv": [
        "rv_60s",  # Keep for new EMAs/SMAs
    ],  # 1
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


def load_and_write_existing_features(memory_monitor: MemoryMonitor) -> Path:
    """
    Load existing features from baseline, microstructure, and advanced files,
    then write to intermediate parquet file.

    TWO-STAGE APPROACH (avoids OOM on 128GB system):
    Stage 1: Write each source separately with timestamp conversion
    Stage 2: Join the intermediate files (much smaller memory footprint)

    Note: Baseline and advanced files use 'timestamp' (microseconds), while micro
    uses 'timestamp_seconds'. We convert timestamp to timestamp_seconds for consistency.

    Also: EMAs and IV/RV ratios are NOT loaded here - they're computed in Module 7
    to avoid duplicate columns.

    Known Issue: Microstructure file has 85,487 extra rows vs baseline (63,158,387 vs 63,072,900).
    LEFT JOIN strategy preserves all baseline rows (correct behavior).

    Args:
        memory_monitor: Memory monitor for tracking usage

    Returns:
        Path to intermediate parquet file with 26 features + timestamp_seconds
        - Baseline: 3 (time_remaining, moneyness, iv_staleness_seconds)
        - Advanced: 21 (drawdown, higher moments, time, vol clustering, autocorr_decay, reversals_300s)
        - Micro: 2 (autocorr_lag5_300s, hurst_300s)
    """
    logger.info("=" * 80)
    logger.info("MODULE 1: Loading Existing Features (Two-Stage Approach)")
    logger.info("=" * 80)

    memory_monitor.log_memory("Module 1 start")

    INTERMEDIATE_DIR.mkdir(parents=True, exist_ok=True)

    # =========================
    # STAGE 1: Write each source separately
    # =========================

    # Step 1a: Process and write baseline
    logger.info("\n[Stage 1a] Processing baseline features...")
    logger.info(f"  Input: {BASELINE_FILE}")

    baseline = pl.scan_parquet(BASELINE_FILE).select(
        [
            "timestamp",
            "time_remaining",
            "S",  # Spot price
            "K",  # Strike price
            "iv_staleness_seconds",
        ]
    )

    baseline = baseline.with_columns(
        [
            pl.col("timestamp").alias("timestamp_seconds"),
            (pl.col("S") / pl.col("K")).alias("moneyness"),
        ]
    ).select(
        [
            "timestamp_seconds",
            "time_remaining",
            "moneyness",
            "iv_staleness_seconds",
        ]
    )

    baseline_temp = INTERMEDIATE_DIR / "01a_baseline_temp.parquet"
    logger.info(f"  Writing to: {baseline_temp.name}")
    baseline.sink_parquet(baseline_temp, compression="snappy")
    memory_monitor.log_memory("After baseline write")
    logger.info(f"  ✓ Baseline written ({baseline_temp.stat().st_size / (1024**3):.2f} GB)")

    # Step 1b: Process and write advanced
    logger.info("\n[Stage 1b] Processing advanced features...")
    logger.info(f"  Input: {ADVANCED_FILE}")

    advanced_keep = [
        "high_15m",
        "low_15m",
        "drawdown_from_high_15m",
        "runup_from_low_15m",
        "time_since_high_15m",
        "time_since_low_15m",
        "skewness_300s",
        "kurtosis_300s",
        "downside_vol_300s",
        "upside_vol_300s",
        "vol_asymmetry_300s",
        "tail_risk_300s",
        "hour_of_day_utc",
        "hour_sin",
        "hour_cos",
        "vol_persistence_ar1",
        "vol_acceleration_300s",
        "vol_of_vol_300s",
        "garch_forecast_simple",
        "autocorr_decay",
        "reversals_300s",
    ]

    advanced = pl.scan_parquet(ADVANCED_FILE).select(["timestamp"] + advanced_keep)
    advanced = advanced.with_columns([pl.col("timestamp").alias("timestamp_seconds")]).select(
        ["timestamp_seconds"] + advanced_keep
    )

    advanced_temp = INTERMEDIATE_DIR / "01b_advanced_temp.parquet"
    logger.info(f"  Writing to: {advanced_temp.name}")
    advanced.sink_parquet(advanced_temp, compression="snappy")
    memory_monitor.log_memory("After advanced write")
    logger.info(f"  ✓ Advanced written ({advanced_temp.stat().st_size / (1024**3):.2f} GB)")

    # Step 1c: Copy microstructure (already has timestamp_seconds)
    logger.info("\n[Stage 1c] Processing microstructure features...")
    logger.info(f"  Input: {MICRO_FILE}")

    micro_keep = ["autocorr_lag5_300s", "hurst_300s"]
    micro = pl.scan_parquet(MICRO_FILE).select(["timestamp_seconds"] + micro_keep)

    micro_temp = INTERMEDIATE_DIR / "01c_micro_temp.parquet"
    logger.info(f"  Writing to: {micro_temp.name}")
    micro.sink_parquet(micro_temp, compression="snappy")
    memory_monitor.log_memory("After micro write")
    logger.info(f"  ✓ Microstructure written ({micro_temp.stat().st_size / (1024**3):.2f} GB)")

    # =========================
    # STAGE 2: Single lazy join chain (no intermediate writes)
    # =========================

    logger.info("\n[Stage 2] Joining all sources (single lazy chain)...")

    baseline_df = pl.scan_parquet(baseline_temp)
    advanced_df = pl.scan_parquet(advanced_temp)
    micro_df = pl.scan_parquet(micro_temp)

    # Chain all joins together (lazy - no materialization yet)
    logger.info("  Building join chain: baseline → advanced → micro...")
    df = baseline_df.join(advanced_df, on="timestamp_seconds", how="left").join(
        micro_df, on="timestamp_seconds", how="left"
    )
    memory_monitor.log_memory("After join chain (lazy)")

    # Single streaming write (materializes and writes in one pass)
    output_file = INTERMEDIATE_DIR / "01_existing_features.parquet"
    logger.info(f"  Writing joined output: {output_file.name}")
    logger.info("  (Streaming write - materializing join chain...)")

    start_write = time.time()
    df.sink_parquet(output_file, compression="snappy")
    elapsed_write = time.time() - start_write

    memory_monitor.log_memory("After write")
    logger.info(
        f"  ✓ Joined features written in {elapsed_write:.1f}s ({output_file.stat().st_size / (1024**3):.2f} GB)"
    )

    # Cleanup temp files
    logger.info("\n  Cleaning up temporary files...")
    baseline_temp.unlink()
    advanced_temp.unlink()
    micro_temp.unlink()
    logger.info("  ✓ All temporary files removed")

    total_stage2_time = elapsed_write

    # Verify output
    logger.info("\nValidating output...")
    output_rows = pl.scan_parquet(output_file).select(pl.len()).collect().item()
    output_cols = len(pl.scan_parquet(output_file).collect_schema().names())

    logger.info("✓ Module 1 completed")
    logger.info(f"  Stage 2 total time: {total_stage2_time:.1f}s")
    logger.info(f"  Output: {output_rows:,} rows × {output_cols} columns")
    logger.info(f"  File size: {output_file.stat().st_size / (1024**3):.2f} GB")
    logger.info("  Expected: ~63M rows, 27 columns")

    # Check output quality on sample
    logger.info("\nChecking data quality on sample (10K rows)...")
    sample = pl.scan_parquet(output_file).head(10_000).collect()
    null_counts = {col: sample[col].null_count() for col in sample.columns}
    critical_nulls = {k: v for k, v in null_counts.items() if v > 0 and k != "timestamp_seconds"}

    if critical_nulls:
        logger.warning(f"  Nulls detected in sample: {list(critical_nulls.keys())[:5]}...")
    else:
        logger.info("  ✓ No nulls in sample")

    logger.info("✓ Wrote 26 existing features (3 baseline + 21 advanced + 2 micro)")

    return output_file


def engineer_and_write_funding_features() -> Path:
    """
    Engineer 11 funding rate features: 1 raw + 5 EMAs + 5 SMAs.
    Write to intermediate parquet file.

    Standard windows: 60s, 300s, 900s, 1800s, 3600s

    Returns:
        Path to intermediate parquet file (63M rows × 12 columns: timestamp_seconds + 11 features)
    """
    logger.info("=" * 80)
    logger.info("MODULE 2: Engineering Funding Rate Features")
    logger.info("=" * 80)

    # Load funding rates data
    logger.info(f"Loading funding rates: {FUNDING_FILE}")
    df = pl.scan_parquet(FUNDING_FILE).select(
        [
            "timestamp",  # Microseconds
            "funding_rate",
        ]
    )

    # Deduplicate source data (9 duplicate timestamps in source)
    logger.info("  Deduplicating source data...")
    df = df.unique(subset=["timestamp"], maintain_order=True)

    # Convert timestamp to seconds for join key
    df = df.with_columns([(pl.col("timestamp") / 1_000_000).cast(pl.Int64).alias("timestamp_seconds")])

    logger.info("Computing funding rate EMAs...")
    df = df.with_columns(
        [
            pl.col("funding_rate").ewm_mean(span=60).alias("funding_rate_ema_60s"),
            pl.col("funding_rate").ewm_mean(span=300).alias("funding_rate_ema_300s"),
            pl.col("funding_rate").ewm_mean(span=900).alias("funding_rate_ema_900s"),
            pl.col("funding_rate").ewm_mean(span=1800).alias("funding_rate_ema_1800s"),
            pl.col("funding_rate").ewm_mean(span=3600).alias("funding_rate_ema_3600s"),
        ]
    )

    logger.info("Computing funding rate SMAs...")
    df = df.with_columns(
        [
            pl.col("funding_rate").rolling_mean(window_size=60).alias("funding_rate_sma_60s"),
            pl.col("funding_rate").rolling_mean(window_size=300).alias("funding_rate_sma_300s"),
            pl.col("funding_rate").rolling_mean(window_size=900).alias("funding_rate_sma_900s"),
            pl.col("funding_rate").rolling_mean(window_size=1800).alias("funding_rate_sma_1800s"),
            pl.col("funding_rate").rolling_mean(window_size=3600).alias("funding_rate_sma_3600s"),
        ]
    )

    # Select final columns
    df = df.select(
        [
            "timestamp_seconds",
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
    )

    # Write to intermediate file (streaming write - no memory spike)
    output_file = INTERMEDIATE_DIR / "02_funding_features.parquet"
    logger.info(f"Writing funding features to {output_file}...")
    df.sink_parquet(output_file, compression="snappy")

    logger.info("✓ Wrote 11 funding rate features")

    return output_file


def engineer_and_write_orderbook_l0_features() -> Path:
    """
    Engineer 32 orderbook L0 (top-of-book) features.
    Write to intermediate parquet file.

    Categories:
    - Bid-ask spread: 1 raw + 5 EMAs + 5 SMAs + 5 volatility = 16 features
    - Bid-ask imbalance: 1 raw + 5 EMAs + 5 SMAs + 5 volatility = 16 features

    Returns:
        Path to intermediate parquet file (63M rows × 33 columns)
    """
    logger.info("=" * 80)
    logger.info("MODULE 3: Engineering Orderbook L0 Features")
    logger.info("=" * 80)

    logger.info(f"Loading orderbook: {ORDERBOOK_FILE}")
    df = pl.scan_parquet(ORDERBOOK_FILE).select(
        [
            "timestamp",
            "bid_price_0",
            "ask_price_0",
            "bid_amount_0",
            "ask_amount_0",
        ]
    )

    # Convert timestamp to seconds
    df = df.with_columns([(pl.col("timestamp") / 1_000_000).cast(pl.Int64).alias("timestamp_seconds")])

    logger.info("Computing bid-ask spread features...")

    # Mid price
    df = df.with_columns([((pl.col("bid_price_0") + pl.col("ask_price_0")) / 2).alias("mid_price")])

    # Bid-ask spread (bps)
    df = df.with_columns(
        [((pl.col("ask_price_0") - pl.col("bid_price_0")) / pl.col("mid_price") * 10000).alias("bid_ask_spread_bps")]
    )

    # Spread EMAs
    df = df.with_columns(
        [
            pl.col("bid_ask_spread_bps").ewm_mean(span=60).alias("spread_ema_60s"),
            pl.col("bid_ask_spread_bps").ewm_mean(span=300).alias("spread_ema_300s"),
            pl.col("bid_ask_spread_bps").ewm_mean(span=900).alias("spread_ema_900s"),
            pl.col("bid_ask_spread_bps").ewm_mean(span=1800).alias("spread_ema_1800s"),
            pl.col("bid_ask_spread_bps").ewm_mean(span=3600).alias("spread_ema_3600s"),
        ]
    )

    # Spread SMAs
    df = df.with_columns(
        [
            pl.col("bid_ask_spread_bps").rolling_mean(window_size=60).alias("spread_sma_60s"),
            pl.col("bid_ask_spread_bps").rolling_mean(window_size=300).alias("spread_sma_300s"),
            pl.col("bid_ask_spread_bps").rolling_mean(window_size=900).alias("spread_sma_900s"),
            pl.col("bid_ask_spread_bps").rolling_mean(window_size=1800).alias("spread_sma_1800s"),
            pl.col("bid_ask_spread_bps").rolling_mean(window_size=3600).alias("spread_sma_3600s"),
        ]
    )

    # Spread volatility
    df = df.with_columns(
        [
            pl.col("bid_ask_spread_bps").rolling_std(window_size=60).alias("spread_vol_60s"),
            pl.col("bid_ask_spread_bps").rolling_std(window_size=300).alias("spread_vol_300s"),
            pl.col("bid_ask_spread_bps").rolling_std(window_size=900).alias("spread_vol_900s"),
            pl.col("bid_ask_spread_bps").rolling_std(window_size=1800).alias("spread_vol_1800s"),
            pl.col("bid_ask_spread_bps").rolling_std(window_size=3600).alias("spread_vol_3600s"),
        ]
    )

    logger.info("Computing bid-ask imbalance features...")

    # Bid-ask imbalance
    df = df.with_columns(
        [
            (
                (pl.col("bid_amount_0") - pl.col("ask_amount_0")) / (pl.col("bid_amount_0") + pl.col("ask_amount_0"))
            ).alias("bid_ask_imbalance")
        ]
    )

    # Imbalance EMAs
    df = df.with_columns(
        [
            pl.col("bid_ask_imbalance").ewm_mean(span=60).alias("imbalance_ema_60s"),
            pl.col("bid_ask_imbalance").ewm_mean(span=300).alias("imbalance_ema_300s"),
            pl.col("bid_ask_imbalance").ewm_mean(span=900).alias("imbalance_ema_900s"),
            pl.col("bid_ask_imbalance").ewm_mean(span=1800).alias("imbalance_ema_1800s"),
            pl.col("bid_ask_imbalance").ewm_mean(span=3600).alias("imbalance_ema_3600s"),
        ]
    )

    # Imbalance SMAs
    df = df.with_columns(
        [
            pl.col("bid_ask_imbalance").rolling_mean(window_size=60).alias("imbalance_sma_60s"),
            pl.col("bid_ask_imbalance").rolling_mean(window_size=300).alias("imbalance_sma_300s"),
            pl.col("bid_ask_imbalance").rolling_mean(window_size=900).alias("imbalance_sma_900s"),
            pl.col("bid_ask_imbalance").rolling_mean(window_size=1800).alias("imbalance_sma_1800s"),
            pl.col("bid_ask_imbalance").rolling_mean(window_size=3600).alias("imbalance_sma_3600s"),
        ]
    )

    # Imbalance volatility
    df = df.with_columns(
        [
            pl.col("bid_ask_imbalance").rolling_std(window_size=60).alias("imbalance_vol_60s"),
            pl.col("bid_ask_imbalance").rolling_std(window_size=300).alias("imbalance_vol_300s"),
            pl.col("bid_ask_imbalance").rolling_std(window_size=900).alias("imbalance_vol_900s"),
            pl.col("bid_ask_imbalance").rolling_std(window_size=1800).alias("imbalance_vol_1800s"),
            pl.col("bid_ask_imbalance").rolling_std(window_size=3600).alias("imbalance_vol_3600s"),
        ]
    )

    # Select final columns
    df = df.select(
        [
            "timestamp_seconds",
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
    )

    # Write to intermediate file (streaming write - no memory spike)
    output_file = INTERMEDIATE_DIR / "03_orderbook_l0_features.parquet"
    logger.info(f"Writing orderbook L0 features to {output_file}...")
    df.sink_parquet(output_file, compression="snappy")

    logger.info("✓ Wrote 32 orderbook L0 features")

    return output_file


def engineer_and_write_orderbook_5level_features() -> Path:
    """
    Engineer 31 orderbook 5-level depth features.
    Write to intermediate parquet file.

    Categories:
    - Total volumes: total_bid_volume_5, total_ask_volume_5 (2)
    - Volume ratios: bid_volume_ratio_1to5, ask_volume_ratio_1to5 (2)
    - Depth imbalance: 1 raw + 5 EMAs + 5 SMAs + 5 volatility = 16 features
    - Weighted mid: 1 raw + 5 EMAs + 5 SMAs = 11 features

    Returns:
        Path to intermediate parquet file (63M rows × 32 columns)
    """
    logger.info("=" * 80)
    logger.info("MODULE 4: Engineering Orderbook 5-Level Features")
    logger.info("=" * 80)

    logger.info(f"Loading orderbook: {ORDERBOOK_FILE}")
    df = pl.scan_parquet(ORDERBOOK_FILE).select(
        [
            "timestamp",
            # Bid prices and amounts (levels 0-4)
            "bid_price_0",
            "bid_price_1",
            "bid_price_2",
            "bid_price_3",
            "bid_price_4",
            "bid_amount_0",
            "bid_amount_1",
            "bid_amount_2",
            "bid_amount_3",
            "bid_amount_4",
            # Ask prices and amounts (levels 0-4)
            "ask_price_0",
            "ask_price_1",
            "ask_price_2",
            "ask_price_3",
            "ask_price_4",
            "ask_amount_0",
            "ask_amount_1",
            "ask_amount_2",
            "ask_amount_3",
            "ask_amount_4",
        ]
    )

    # Convert timestamp to seconds
    df = df.with_columns([(pl.col("timestamp") / 1_000_000).cast(pl.Int64).alias("timestamp_seconds")])

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
            pl.col("depth_imbalance_5").ewm_mean(span=1800).alias("depth_imbalance_ema_1800s"),
            pl.col("depth_imbalance_5").ewm_mean(span=3600).alias("depth_imbalance_ema_3600s"),
        ]
    )

    # Depth imbalance SMAs
    df = df.with_columns(
        [
            pl.col("depth_imbalance_5").rolling_mean(window_size=60).alias("depth_imbalance_sma_60s"),
            pl.col("depth_imbalance_5").rolling_mean(window_size=300).alias("depth_imbalance_sma_300s"),
            pl.col("depth_imbalance_5").rolling_mean(window_size=900).alias("depth_imbalance_sma_900s"),
            pl.col("depth_imbalance_5").rolling_mean(window_size=1800).alias("depth_imbalance_sma_1800s"),
            pl.col("depth_imbalance_5").rolling_mean(window_size=3600).alias("depth_imbalance_sma_3600s"),
        ]
    )

    # Depth imbalance volatility
    df = df.with_columns(
        [
            pl.col("depth_imbalance_5").rolling_std(window_size=60).alias("depth_imbalance_vol_60s"),
            pl.col("depth_imbalance_5").rolling_std(window_size=300).alias("depth_imbalance_vol_300s"),
            pl.col("depth_imbalance_5").rolling_std(window_size=900).alias("depth_imbalance_vol_900s"),
            pl.col("depth_imbalance_5").rolling_std(window_size=1800).alias("depth_imbalance_vol_1800s"),
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
            pl.col("weighted_mid_price_5").ewm_mean(span=1800).alias("weighted_mid_ema_1800s"),
            pl.col("weighted_mid_price_5").ewm_mean(span=3600).alias("weighted_mid_ema_3600s"),
        ]
    )

    # Weighted mid SMAs
    df = df.with_columns(
        [
            pl.col("weighted_mid_price_5").rolling_mean(window_size=60).alias("weighted_mid_sma_60s"),
            pl.col("weighted_mid_price_5").rolling_mean(window_size=300).alias("weighted_mid_sma_300s"),
            pl.col("weighted_mid_price_5").rolling_mean(window_size=900).alias("weighted_mid_sma_900s"),
            pl.col("weighted_mid_price_5").rolling_mean(window_size=1800).alias("weighted_mid_sma_1800s"),
            pl.col("weighted_mid_price_5").rolling_mean(window_size=3600).alias("weighted_mid_sma_3600s"),
        ]
    )

    # Select final columns
    df = df.select(
        [
            "timestamp_seconds",
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
    )

    # Write to intermediate file (streaming write - no memory spike)
    output_file = INTERMEDIATE_DIR / "04_orderbook_5level_features.parquet"
    logger.info(f"Writing orderbook 5-level features to {output_file}...")
    df.sink_parquet(output_file, compression="snappy")

    logger.info("✓ Wrote 31 orderbook 5-level features")

    return output_file


def engineer_and_write_price_basis_features() -> Path:
    """
    Engineer 34 price basis features.
    Write to intermediate parquet file.

    Categories:
    - Mark-index basis: 1 raw + 5 EMAs + 5 SMAs = 11 features
    - Mark-last basis: 1 raw + 4 EMAs + 4 SMAs = 9 features
    - Index-last basis: 1 raw + 4 EMAs + 4 SMAs = 9 features
    - Basis ratios: 5 features (mark_index / mark_last, etc.)

    Returns:
        Path to intermediate parquet file (63M rows × 35 columns)
    """
    logger.info("=" * 80)
    logger.info("MODULE 5: Engineering Price Basis Features")
    logger.info("=" * 80)

    logger.info(f"Loading prices from: {FUNDING_FILE}")
    df = pl.scan_parquet(FUNDING_FILE).select(
        [
            "timestamp",
            "mark_price",
            "index_price",
        ]
    )

    # Deduplicate source data (9 duplicate timestamps in source)
    logger.info("  Deduplicating source data...")
    df = df.unique(subset=["timestamp"], maintain_order=True)

    # Convert timestamp to seconds
    df = df.with_columns([(pl.col("timestamp") / 1_000_000).cast(pl.Int64).alias("timestamp_seconds")])

    logger.info("Computing mark-index basis features...")

    # Mark-index basis (primary funding rate signal)
    df = df.with_columns(
        [((pl.col("mark_price") - pl.col("index_price")) / pl.col("index_price") * 10000).alias("mark_index_basis_bps")]
    )

    # Mark-index EMAs
    df = df.with_columns(
        [
            pl.col("mark_index_basis_bps").ewm_mean(span=60).alias("mark_index_ema_60s"),
            pl.col("mark_index_basis_bps").ewm_mean(span=300).alias("mark_index_ema_300s"),
            pl.col("mark_index_basis_bps").ewm_mean(span=900).alias("mark_index_ema_900s"),
            pl.col("mark_index_basis_bps").ewm_mean(span=1800).alias("mark_index_ema_1800s"),
            pl.col("mark_index_basis_bps").ewm_mean(span=3600).alias("mark_index_ema_3600s"),
        ]
    )

    # Mark-index SMAs
    df = df.with_columns(
        [
            pl.col("mark_index_basis_bps").rolling_mean(window_size=60).alias("mark_index_sma_60s"),
            pl.col("mark_index_basis_bps").rolling_mean(window_size=300).alias("mark_index_sma_300s"),
            pl.col("mark_index_basis_bps").rolling_mean(window_size=900).alias("mark_index_sma_900s"),
            pl.col("mark_index_basis_bps").rolling_mean(window_size=1800).alias("mark_index_sma_1800s"),
            pl.col("mark_index_basis_bps").rolling_mean(window_size=3600).alias("mark_index_sma_3600s"),
        ]
    )

    # Select final columns (mark-index only)
    df = df.select(
        [
            "timestamp_seconds",
            # Mark-index (11 features)
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
        ]
    )

    # Write to intermediate file (streaming write - no memory spike)
    output_file = INTERMEDIATE_DIR / "05_price_basis_features.parquet"
    logger.info(f"Writing price basis features to {output_file}...")
    df.sink_parquet(output_file, compression="snappy")

    logger.info("✓ Wrote 11 price basis features (mark-index only)")

    return output_file


def engineer_and_write_oi_features() -> Path:
    """
    Engineer 6 open interest features: 1 raw + 5 EMAs.
    Write to intermediate parquet file.

    Returns:
        Path to intermediate parquet file (63M rows × 7 columns)
    """
    logger.info("=" * 80)
    logger.info("MODULE 6: Engineering Open Interest Features")
    logger.info("=" * 80)

    logger.info(f"Loading OI from: {FUNDING_FILE}")
    df = pl.scan_parquet(FUNDING_FILE).select(
        [
            "timestamp",
            "open_interest",
        ]
    )

    # Deduplicate source data (9 duplicate timestamps in source)
    logger.info("  Deduplicating source data...")
    df = df.unique(subset=["timestamp"], maintain_order=True)

    # Convert timestamp to seconds
    df = df.with_columns([(pl.col("timestamp") / 1_000_000).cast(pl.Int64).alias("timestamp_seconds")])

    logger.info("Forward-filling OI nulls (3,219 isolated 1-second gaps)...")
    df = df.with_columns([pl.col("open_interest").forward_fill().alias("open_interest")])

    logger.info("Computing OI EMAs...")
    df = df.with_columns(
        [
            pl.col("open_interest").ewm_mean(span=60).alias("oi_ema_60s"),
            pl.col("open_interest").ewm_mean(span=300).alias("oi_ema_300s"),
            pl.col("open_interest").ewm_mean(span=900).alias("oi_ema_900s"),
            pl.col("open_interest").ewm_mean(span=1800).alias("oi_ema_1800s"),
            pl.col("open_interest").ewm_mean(span=3600).alias("oi_ema_3600s"),
        ]
    )

    # Select final columns
    df = df.select(
        [
            "timestamp_seconds",
            "open_interest",
            "oi_ema_60s",
            "oi_ema_300s",
            "oi_ema_900s",
            "oi_ema_1800s",
            "oi_ema_3600s",
        ]
    )

    # Write to intermediate file (streaming write - no memory spike)
    output_file = INTERMEDIATE_DIR / "06_oi_features.parquet"
    logger.info(f"Writing OI features to {output_file}...")
    df.sink_parquet(output_file, compression="snappy")

    logger.info("✓ Wrote 6 OI features")

    return output_file


def engineer_and_write_rv_momentum_range_features() -> Path:
    """
    Engineer 79 RV/momentum/range/EMA features from existing files.
    Write to intermediate parquet file.

    Categories:
    - RV: rv_300s, rv_900s (2 raw) + 10 EMAs/SMAs = 22 features
    - Momentum: momentum_300s, momentum_900s (2 raw) + 10 EMAs/SMAs = 22 features
    - Range: range_300s, range_900s (2 raw) + 10 EMAs/SMAs = 22 features
    - Derived ratios: 4 features (rv_ratio_5m_1m, etc.)
    - EMAs: 4 raw EMAs (ema_12s, ema_60s, ema_300s, ema_900s)
    - EMA ratios: 5 features (price_vs_ema_900, etc.)

    Total: 79 features

    Returns:
        Path to intermediate parquet file (63M rows × 80 columns)
    """
    logger.info("=" * 80)
    logger.info("MODULE 7: Engineering RV/Momentum/Range Features")
    logger.info("=" * 80)

    logger.info(f"Loading RV features: {RV_FILE}")
    df = pl.scan_parquet(RV_FILE).select(
        [
            "timestamp_seconds",
            "rv_300s",
            "rv_900s",
        ]
    )

    logger.info("Loading microstructure for momentum/range: {MICRO_FILE}")
    micro = pl.scan_parquet(MICRO_FILE).select(
        [
            "timestamp_seconds",
            "momentum_300s",
            "momentum_900s",
            "range_300s",
            "range_900s",
        ]
    )

    # Join
    df = df.join(micro, on="timestamp_seconds", how="left")

    logger.info("Computing RV EMAs/SMAs...")

    # RV EMAs (rv_300s and rv_900s)
    df = df.with_columns(
        [
            pl.col("rv_300s").ewm_mean(span=60).alias("rv_300s_ema_60s"),
            pl.col("rv_300s").ewm_mean(span=300).alias("rv_300s_ema_300s"),
            pl.col("rv_300s").ewm_mean(span=900).alias("rv_300s_ema_900s"),
            pl.col("rv_300s").ewm_mean(span=1800).alias("rv_300s_ema_1800s"),
            pl.col("rv_300s").ewm_mean(span=3600).alias("rv_300s_ema_3600s"),
            pl.col("rv_900s").ewm_mean(span=60).alias("rv_900s_ema_60s"),
            pl.col("rv_900s").ewm_mean(span=300).alias("rv_900s_ema_300s"),
            pl.col("rv_900s").ewm_mean(span=900).alias("rv_900s_ema_900s"),
            pl.col("rv_900s").ewm_mean(span=1800).alias("rv_900s_ema_1800s"),
            pl.col("rv_900s").ewm_mean(span=3600).alias("rv_900s_ema_3600s"),
        ]
    )

    # RV SMAs
    df = df.with_columns(
        [
            pl.col("rv_300s").rolling_mean(window_size=60).alias("rv_300s_sma_60s"),
            pl.col("rv_300s").rolling_mean(window_size=300).alias("rv_300s_sma_300s"),
            pl.col("rv_300s").rolling_mean(window_size=900).alias("rv_300s_sma_900s"),
            pl.col("rv_300s").rolling_mean(window_size=1800).alias("rv_300s_sma_1800s"),
            pl.col("rv_300s").rolling_mean(window_size=3600).alias("rv_300s_sma_3600s"),
            pl.col("rv_900s").rolling_mean(window_size=60).alias("rv_900s_sma_60s"),
            pl.col("rv_900s").rolling_mean(window_size=300).alias("rv_900s_sma_300s"),
            pl.col("rv_900s").rolling_mean(window_size=900).alias("rv_900s_sma_900s"),
            pl.col("rv_900s").rolling_mean(window_size=1800).alias("rv_900s_sma_1800s"),
            pl.col("rv_900s").rolling_mean(window_size=3600).alias("rv_900s_sma_3600s"),
        ]
    )

    logger.info("Computing momentum EMAs/SMAs...")

    # Momentum EMAs
    df = df.with_columns(
        [
            pl.col("momentum_300s").ewm_mean(span=60).alias("momentum_300s_ema_60s"),
            pl.col("momentum_300s").ewm_mean(span=300).alias("momentum_300s_ema_300s"),
            pl.col("momentum_300s").ewm_mean(span=900).alias("momentum_300s_ema_900s"),
            pl.col("momentum_300s").ewm_mean(span=1800).alias("momentum_300s_ema_1800s"),
            pl.col("momentum_300s").ewm_mean(span=3600).alias("momentum_300s_ema_3600s"),
            pl.col("momentum_900s").ewm_mean(span=60).alias("momentum_900s_ema_60s"),
            pl.col("momentum_900s").ewm_mean(span=300).alias("momentum_900s_ema_300s"),
            pl.col("momentum_900s").ewm_mean(span=900).alias("momentum_900s_ema_900s"),
            pl.col("momentum_900s").ewm_mean(span=1800).alias("momentum_900s_ema_1800s"),
            pl.col("momentum_900s").ewm_mean(span=3600).alias("momentum_900s_ema_3600s"),
        ]
    )

    # Momentum SMAs
    df = df.with_columns(
        [
            pl.col("momentum_300s").rolling_mean(window_size=60).alias("momentum_300s_sma_60s"),
            pl.col("momentum_300s").rolling_mean(window_size=300).alias("momentum_300s_sma_300s"),
            pl.col("momentum_300s").rolling_mean(window_size=900).alias("momentum_300s_sma_900s"),
            pl.col("momentum_300s").rolling_mean(window_size=1800).alias("momentum_300s_sma_1800s"),
            pl.col("momentum_300s").rolling_mean(window_size=3600).alias("momentum_300s_sma_3600s"),
            pl.col("momentum_900s").rolling_mean(window_size=60).alias("momentum_900s_sma_60s"),
            pl.col("momentum_900s").rolling_mean(window_size=300).alias("momentum_900s_sma_300s"),
            pl.col("momentum_900s").rolling_mean(window_size=900).alias("momentum_900s_sma_900s"),
            pl.col("momentum_900s").rolling_mean(window_size=1800).alias("momentum_900s_sma_1800s"),
            pl.col("momentum_900s").rolling_mean(window_size=3600).alias("momentum_900s_sma_3600s"),
        ]
    )

    logger.info("Computing range EMAs/SMAs...")

    # Range EMAs
    df = df.with_columns(
        [
            pl.col("range_300s").ewm_mean(span=60).alias("range_300s_ema_60s"),
            pl.col("range_300s").ewm_mean(span=300).alias("range_300s_ema_300s"),
            pl.col("range_300s").ewm_mean(span=900).alias("range_300s_ema_900s"),
            pl.col("range_300s").ewm_mean(span=1800).alias("range_300s_ema_1800s"),
            pl.col("range_300s").ewm_mean(span=3600).alias("range_300s_ema_3600s"),
            pl.col("range_900s").ewm_mean(span=60).alias("range_900s_ema_60s"),
            pl.col("range_900s").ewm_mean(span=300).alias("range_900s_ema_300s"),
            pl.col("range_900s").ewm_mean(span=900).alias("range_900s_ema_900s"),
            pl.col("range_900s").ewm_mean(span=1800).alias("range_900s_ema_1800s"),
            pl.col("range_900s").ewm_mean(span=3600).alias("range_900s_ema_3600s"),
        ]
    )

    # Range SMAs
    df = df.with_columns(
        [
            pl.col("range_300s").rolling_mean(window_size=60).alias("range_300s_sma_60s"),
            pl.col("range_300s").rolling_mean(window_size=300).alias("range_300s_sma_300s"),
            pl.col("range_300s").rolling_mean(window_size=900).alias("range_300s_sma_900s"),
            pl.col("range_300s").rolling_mean(window_size=1800).alias("range_300s_sma_1800s"),
            pl.col("range_300s").rolling_mean(window_size=3600).alias("range_300s_sma_3600s"),
            pl.col("range_900s").rolling_mean(window_size=60).alias("range_900s_sma_60s"),
            pl.col("range_900s").rolling_mean(window_size=300).alias("range_900s_sma_300s"),
            pl.col("range_900s").rolling_mean(window_size=900).alias("range_900s_sma_900s"),
            pl.col("range_900s").rolling_mean(window_size=1800).alias("range_900s_sma_1800s"),
            pl.col("range_900s").rolling_mean(window_size=3600).alias("range_900s_sma_3600s"),
        ]
    )

    logger.info("Computing derived ratios...")

    # RV ratios (kept from V2 for continuity)
    df = df.with_columns(
        [
            (pl.col("rv_300s") / (pl.col("rv_300s").rolling_mean(window_size=60) + 1e-8)).alias("rv_ratio_5m_1m"),
            (pl.col("rv_900s") / (pl.col("rv_300s") + 1e-8)).alias("rv_ratio_15m_5m"),
            (pl.col("rv_900s").rolling_mean(window_size=3600) / (pl.col("rv_900s") + 1e-8)).alias("rv_ratio_1h_15m"),
            ((pl.col("rv_900s") - pl.col("rv_300s")) / (pl.col("rv_300s") + 1e-8)).alias("rv_term_structure"),
        ]
    )

    logger.info("Computing EMA ratios...")

    # Load advanced file for EMA ratios
    logger.info(f"Loading advanced for EMAs: {ADVANCED_FILE}")
    advanced = pl.scan_parquet(ADVANCED_FILE).select(
        [
            "timestamp",
            "ema_12s",
            "ema_60s",
            "ema_300s",
            "ema_900s",
        ]
    )

    # Convert timestamp to seconds
    advanced = advanced.with_columns([pl.col("timestamp").alias("timestamp_seconds")]).select(
        [
            "timestamp_seconds",
            "ema_12s",
            "ema_60s",
            "ema_300s",
            "ema_900s",
        ]
    )

    # Join EMAs
    df = df.join(advanced, on="timestamp_seconds", how="left")

    # Load baseline for price
    logger.info(f"Loading baseline for price: {BASELINE_FILE}")
    baseline = pl.scan_parquet(BASELINE_FILE).select(
        [
            "timestamp",
            "S",  # Spot price
        ]
    )

    # Convert timestamp to seconds
    baseline = baseline.with_columns([pl.col("timestamp").alias("timestamp_seconds")]).select(
        ["timestamp_seconds", "S"]
    )

    df = df.join(baseline, on="timestamp_seconds", how="left")

    # EMA ratios (price normalized by EMAs)
    df = df.with_columns(
        [
            ((pl.col("S") - pl.col("ema_900s")) / pl.col("ema_900s")).alias("price_vs_ema_900"),
            (pl.col("ema_12s") / pl.col("ema_60s")).alias("ema_12s_vs_60s"),
            (pl.col("ema_60s") / pl.col("ema_300s")).alias("ema_60s_vs_300s"),
            (pl.col("ema_300s") / pl.col("ema_900s")).alias("ema_300s_vs_900s"),
            ((pl.col("ema_12s") - pl.col("ema_900s")) / pl.col("ema_900s")).alias("ema_spread_12s_900s"),
        ]
    )

    # Select final columns (exclude S - only used for ratios)
    df = df.select(
        [
            "timestamp_seconds",
            # RV (22)
            "rv_300s",
            "rv_900s",
            "rv_300s_ema_60s",
            "rv_300s_ema_300s",
            "rv_300s_ema_900s",
            "rv_300s_ema_1800s",
            "rv_300s_ema_3600s",
            "rv_900s_ema_60s",
            "rv_900s_ema_300s",
            "rv_900s_ema_900s",
            "rv_900s_ema_1800s",
            "rv_900s_ema_3600s",
            "rv_300s_sma_60s",
            "rv_300s_sma_300s",
            "rv_300s_sma_900s",
            "rv_300s_sma_1800s",
            "rv_300s_sma_3600s",
            "rv_900s_sma_60s",
            "rv_900s_sma_300s",
            "rv_900s_sma_900s",
            "rv_900s_sma_1800s",
            "rv_900s_sma_3600s",
            # Momentum (22)
            "momentum_300s",
            "momentum_900s",
            "momentum_300s_ema_60s",
            "momentum_300s_ema_300s",
            "momentum_300s_ema_900s",
            "momentum_300s_ema_1800s",
            "momentum_300s_ema_3600s",
            "momentum_900s_ema_60s",
            "momentum_900s_ema_300s",
            "momentum_900s_ema_900s",
            "momentum_900s_ema_1800s",
            "momentum_900s_ema_3600s",
            "momentum_300s_sma_60s",
            "momentum_300s_sma_300s",
            "momentum_300s_sma_900s",
            "momentum_300s_sma_1800s",
            "momentum_300s_sma_3600s",
            "momentum_900s_sma_60s",
            "momentum_900s_sma_300s",
            "momentum_900s_sma_900s",
            "momentum_900s_sma_1800s",
            "momentum_900s_sma_3600s",
            # Range (22)
            "range_300s",
            "range_900s",
            "range_300s_ema_60s",
            "range_300s_ema_300s",
            "range_300s_ema_900s",
            "range_300s_ema_1800s",
            "range_300s_ema_3600s",
            "range_900s_ema_60s",
            "range_900s_ema_300s",
            "range_900s_ema_900s",
            "range_900s_ema_1800s",
            "range_900s_ema_3600s",
            "range_300s_sma_60s",
            "range_300s_sma_300s",
            "range_300s_sma_900s",
            "range_300s_sma_1800s",
            "range_300s_sma_3600s",
            "range_900s_sma_60s",
            "range_900s_sma_300s",
            "range_900s_sma_900s",
            "range_900s_sma_1800s",
            "range_900s_sma_3600s",
            # Ratios (4)
            "rv_ratio_5m_1m",
            "rv_ratio_15m_5m",
            "rv_ratio_1h_15m",
            "rv_term_structure",
            # EMAs (4 raw)
            "ema_12s",
            "ema_60s",
            "ema_300s",
            "ema_900s",
            # EMA ratios (5)
            "price_vs_ema_900",
            "ema_12s_vs_60s",
            "ema_60s_vs_300s",
            "ema_300s_vs_900s",
            "ema_spread_12s_900s",
        ]
    )

    # Write to intermediate file (streaming write - no memory spike)
    output_file = INTERMEDIATE_DIR / "07_rv_momentum_range_features.parquet"
    logger.info(f"Writing RV/momentum/range features to {output_file}...")
    df.sink_parquet(output_file, compression="snappy")

    logger.info("✓ Wrote 79 RV/momentum/range/EMA features")

    return output_file


def join_all_features_temporal_chunking(
    existing_file: Path,
    funding_file: Path,
    orderbook_l0_file: Path,
    orderbook_5level_file: Path,
    basis_file: Path,
    oi_file: Path,
    rv_momentum_range_file: Path,
    memory_monitor: MemoryMonitor,
) -> Path:
    """
    Join all feature sources using temporal (3-month) chunking optimized for 256GB RAM.

    MEMORY-OPTIMIZED APPROACH:
    - Processes data in 3-month chunks (63M rows → ~5.25M rows/chunk, ~12 chunks total)
    - Joins all 6 sources within each chunk
    - Writes temporal chunks to disk
    - Concatenates chunks lazily at end

    This avoids Polars hash table memory explosion (100GB+) from chaining
    6 joins on 63M rows. 3-month chunking reduces hash table size by 12x while
    maintaining good performance (fewer disk I/O operations than daily chunking).

    Args:
        existing_file: Path to existing features (26 features)
        funding_file: Path to funding features (11 features)
        orderbook_l0_file: Path to L0 orderbook features (32 features)
        orderbook_5level_file: Path to 5-level orderbook features (31 features)
        basis_file: Path to basis features (34 features)
        oi_file: Path to OI features (6 features)
        rv_momentum_range_file: Path to RV/momentum/range features (79 features)
        memory_monitor: Memory monitor instance

    Returns:
        Path to final consolidated features file
    """
    logger.info("=" * 80)
    logger.info("MODULE 8: Joining All Features (Temporal Chunking)")
    logger.info("=" * 80)

    # Get date range from existing features file
    logger.info("Determining date range from existing features...")
    existing_df = pl.scan_parquet(existing_file)
    date_range_df = (
        existing_df.select([pl.from_epoch("timestamp_seconds", time_unit="s").cast(pl.Date).alias("date")])
        .unique()
        .sort("date")
        .collect()
    )

    dates = date_range_df["date"].to_list()
    start_date = dates[0]
    end_date = dates[-1]
    logger.info(f"  Date range: {start_date} to {end_date} ({len(dates)} unique days)")

    # Generate 3-month chunks
    chunk_ranges = []
    current_start = start_date
    while current_start <= end_date:
        # Add 3 months (approximately 90 days)
        current_end = current_start + timedelta(days=90)
        chunk_ranges.append((current_start, min(current_end, end_date + timedelta(days=1))))
        current_start = current_end

    logger.info(f"  Processing in {len(chunk_ranges)} chunks of ~3 months each")
    memory_monitor.log_memory("After date range computation")

    # Create temporary directory for temporal chunks
    temporal_chunks_dir = INTERMEDIATE_DIR / "temporal_chunks"
    temporal_chunks_dir.mkdir(exist_ok=True)

    # Process each 3-month chunk
    chunk_files = []
    logger.info(f"\nProcessing {len(chunk_ranges)} chunks (~3 months each)...")

    for idx, (chunk_start, chunk_end) in enumerate(chunk_ranges, 1):
        # Filter each source to this 3-month period
        # Create date filter expression (must be repeated for each scan due to Polars lazy evaluation)
        existing_chunk = pl.scan_parquet(existing_file).filter(
            (pl.from_epoch("timestamp_seconds", time_unit="s").cast(pl.Date) >= chunk_start)
            & (pl.from_epoch("timestamp_seconds", time_unit="s").cast(pl.Date) < chunk_end)
        )
        funding_chunk = pl.scan_parquet(funding_file).filter(
            (pl.from_epoch("timestamp_seconds", time_unit="s").cast(pl.Date) >= chunk_start)
            & (pl.from_epoch("timestamp_seconds", time_unit="s").cast(pl.Date) < chunk_end)
        )
        orderbook_l0_chunk = pl.scan_parquet(orderbook_l0_file).filter(
            (pl.from_epoch("timestamp_seconds", time_unit="s").cast(pl.Date) >= chunk_start)
            & (pl.from_epoch("timestamp_seconds", time_unit="s").cast(pl.Date) < chunk_end)
        )
        orderbook_5level_chunk = pl.scan_parquet(orderbook_5level_file).filter(
            (pl.from_epoch("timestamp_seconds", time_unit="s").cast(pl.Date) >= chunk_start)
            & (pl.from_epoch("timestamp_seconds", time_unit="s").cast(pl.Date) < chunk_end)
        )
        basis_chunk = pl.scan_parquet(basis_file).filter(
            (pl.from_epoch("timestamp_seconds", time_unit="s").cast(pl.Date) >= chunk_start)
            & (pl.from_epoch("timestamp_seconds", time_unit="s").cast(pl.Date) < chunk_end)
        )
        oi_chunk = pl.scan_parquet(oi_file).filter(
            (pl.from_epoch("timestamp_seconds", time_unit="s").cast(pl.Date) >= chunk_start)
            & (pl.from_epoch("timestamp_seconds", time_unit="s").cast(pl.Date) < chunk_end)
        )
        rv_chunk = pl.scan_parquet(rv_momentum_range_file).filter(
            (pl.from_epoch("timestamp_seconds", time_unit="s").cast(pl.Date) >= chunk_start)
            & (pl.from_epoch("timestamp_seconds", time_unit="s").cast(pl.Date) < chunk_end)
        )

        # Chain joins for this chunk (fits in memory with 256GB)
        chunk_df = (
            existing_chunk.join(funding_chunk, on="timestamp_seconds", how="left")
            .join(orderbook_l0_chunk, on="timestamp_seconds", how="left")
            .join(orderbook_5level_chunk, on="timestamp_seconds", how="left")
            .join(basis_chunk, on="timestamp_seconds", how="left")
            .join(oi_chunk, on="timestamp_seconds", how="left")
            .join(rv_chunk, on="timestamp_seconds", how="left")
        )

        # Collect and write chunk
        chunk_result = chunk_df.collect()
        chunk_file = (
            temporal_chunks_dir
            / f"features_v3_chunk_{idx:03d}_{chunk_start.isoformat()}_to_{chunk_end.isoformat()}.parquet"
        )
        chunk_result.write_parquet(chunk_file, compression="snappy")
        chunk_files.append(chunk_file)

        # Progress logging
        logger.info(
            f"  Processed chunk {idx}/{len(chunk_ranges)} ({chunk_start} to {chunk_end}): {len(chunk_result):,} rows"
        )
        memory_monitor.log_memory(f"After chunk {idx}")

    logger.info(f"\n✓ All {len(chunk_ranges)} temporal chunks written")
    logger.info(f"  Total chunks: {len(chunk_files)}")
    logger.info(f"  Chunk dir size: {sum(f.stat().st_size for f in chunk_files) / (1024**3):.2f} GB")

    # Concatenate all temporal chunks lazily
    logger.info("\nConcatenating all temporal chunks...")
    output_file = DATA_DIR / "consolidated_features_v3.parquet"

    lazy_chunks = [pl.scan_parquet(f) for f in chunk_files]
    combined = pl.concat(lazy_chunks)

    logger.info(f"Writing final consolidated features to {output_file}...")
    combined.sink_parquet(output_file, compression="snappy")

    memory_monitor.log_memory("After concatenation")

    # Cleanup temporal chunks
    logger.info("\nCleaning up temporal chunk files...")
    for f in chunk_files:
        f.unlink()
    temporal_chunks_dir.rmdir()
    logger.info("✓ Temporal chunks cleaned up")

    return output_file


def validate_final_output(output_file: Path, memory_monitor: MemoryMonitor) -> None:
    """
    Validate final consolidated features file.

    Args:
        output_file: Path to consolidated features parquet
        memory_monitor: Memory monitor instance
    """
    logger.info("=" * 80)
    logger.info("MODULE 9: Validation")
    logger.info("=" * 80)

    logger.info(f"Validating output file: {output_file}")

    # Scan output lazily
    df = pl.scan_parquet(output_file)

    # Validation on sample (1M rows - reduced for memory safety)
    logger.info("Validating feature quality on 1M row sample...")
    sample = df.head(1_000_000).collect()

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

    # Expected: 196 features + timestamp_seconds = 197 columns
    expected_features = 196
    if feature_count != expected_features:
        logger.warning(f"Feature count mismatch: expected {expected_features}, got {feature_count}")
        logger.warning("This may indicate missing or duplicate features.")

    memory_monitor.log_memory("After validation")

    # Verify output file
    output_size_gb = output_file.stat().st_size / (1024**3)
    logger.info(f"✓ Output file size: {output_size_gb:.2f} GB")

    logger.info("=" * 80)
    logger.info("FEATURE ENGINEERING V3 COMPLETE")
    logger.info("=" * 80)
    logger.info(f"Output: {output_file}")
    logger.info(f"Rows: {row_count:,}")
    logger.info(f"Features: {feature_count}")
    logger.info(f"Size: {output_size_gb:.2f} GB")


def main() -> None:
    """
    Main execution pipeline (memory-optimized 3-month chunking for 256GB RAM).

    Pipeline:
    1. Modules 1-7: Write intermediate feature files to disk (~13GB total)
    2. Module 8: Join all features using 3-month temporal chunking
       - Processes 63M rows in ~12 quarterly chunks (~5.25M rows/chunk)
       - Joins all 6 feature sources within each chunk
       - Writes temporal chunks, then concatenates
       - Avoids Polars hash table memory explosion (100GB+ → ~25GB peak per chunk)
    3. Module 9: Validates final consolidated output

    Peak memory usage: ~25-30 GB per chunk (safe for 256GB system)
    Runtime: ~10-15 minutes (optimized for 256GB RAM)
    """
    logger.info("=" * 80)
    logger.info("Consolidated Feature Engineering V3 (3-Month Chunking for 256GB RAM)")
    logger.info("=" * 80)

    memory_monitor = MemoryMonitor()
    memory_monitor.log_memory("Start")

    # Module 1: Load and write existing features
    existing_file = load_and_write_existing_features(memory_monitor)
    memory_monitor.log_memory("After Module 1 (existing)")

    # Module 2: Generate and write funding features
    funding_file = engineer_and_write_funding_features()
    memory_monitor.log_memory("After Module 2 (funding)")

    # Module 3: Generate and write orderbook L0 features
    orderbook_l0_file = engineer_and_write_orderbook_l0_features()
    memory_monitor.log_memory("After Module 3 (orderbook L0)")

    # Module 4: Generate and write orderbook 5-level features
    orderbook_5level_file = engineer_and_write_orderbook_5level_features()
    memory_monitor.log_memory("After Module 4 (orderbook 5-level)")

    # Module 5: Generate and write price basis features
    basis_file = engineer_and_write_price_basis_features()
    memory_monitor.log_memory("After Module 5 (basis)")

    # Module 6: Generate and write OI features
    oi_file = engineer_and_write_oi_features()
    memory_monitor.log_memory("After Module 6 (OI)")

    # Module 7: Generate and write RV/momentum/range features
    rv_momentum_range_file = engineer_and_write_rv_momentum_range_features()
    memory_monitor.log_memory("After Module 7 (RV/momentum/range)")

    # Module 8: Join all features using temporal chunking (writes output directly)
    output_file = join_all_features_temporal_chunking(
        existing_file,
        funding_file,
        orderbook_l0_file,
        orderbook_5level_file,
        basis_file,
        oi_file,
        rv_momentum_range_file,
        memory_monitor,
    )
    memory_monitor.log_memory("After Module 8 (temporal chunking)")

    # Module 9: Validate final output
    validate_final_output(output_file, memory_monitor)

    logger.info("✓ Feature engineering V3 pipeline completed successfully")


if __name__ == "__main__":
    main()
