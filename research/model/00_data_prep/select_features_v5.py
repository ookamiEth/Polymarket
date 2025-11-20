#!/usr/bin/env python3
"""
Feature Selection for V5 - Reduce to <50 Features
==================================================

Filters consolidated_features_v5_pipeline_ready.parquet (193 columns) to keep
only 43 features per V5_FEATURE_SPECIFICATION.md:
- 9 moneyness features (from V4)
- 4 volatility features (from V4)
- 30 Greek features (from engineer_greeks_v5.py)

Plus 11 metadata columns (date, timestamp, S, K, outcome, prob_mid, etc.)
and 4 regime columns (combined_regime, temporal_regime, etc.)

Input:  consolidated_features_v5_pipeline_ready.parquet (193 columns, ~66M rows)
Output: consolidated_features_v5_selected.parquet (58 columns, ~66M rows)

Runtime: ~1-2 minutes on 32 vCPU machine

Author: BT Research Team
Date: 2025-11-20
"""

from __future__ import annotations

import logging
from pathlib import Path

import polars as pl

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def select_minimalist_features_v5(df: pl.LazyFrame) -> pl.LazyFrame:
    """
    Select minimalist feature set per V5_FEATURE_SPECIFICATION.md.

    Keeps:
    - 9 moneyness features (from V4)
    - 4 volatility features (from V4)
    - 30 Greek features (from engineer_greeks_v5.py)
    - 11 metadata columns (includes date, outcome, prob_mid, etc.)
    - 4 regime columns (if present)

    Drops:
    - 150 V4 features (microstructure, market, redundant vol windows, etc.)

    Args:
        df: LazyFrame with 193 columns (V4 + Greeks)

    Returns:
        LazyFrame with 58 columns (43 features + 11 metadata + 4 regime columns if present)
    """
    logger.info("Selecting minimalist V5 feature set...")

    # Moneyness features from V4 (9 features)
    moneyness_features = [
        "moneyness",
        "log_moneyness",
        "moneyness_squared",
        "moneyness_cubed",
        "moneyness_distance",
        "moneyness_percentile",
        "standardized_moneyness",
        "moneyness_x_vol",
        "moneyness_x_time",
    ]

    # Volatility features from V4 (4 features)
    volatility_features = [
        "rv_900s",
        "rv_ratio",  # rv_300s / rv_900s
        "vol_asymmetry_ratio_300",
        "realized_skewness_300",
    ]

    # Greek features from engineer_greeks_v5.py (30 features)
    greek_features = [
        # Risk-free rate (1)
        "r",
        # Core Greeks (10)
        "delta",
        "gamma",
        "vega",
        "theta",
        "vanna",
        "volga",
        "charm",
        "d1",
        "d2",
        "phi_d2",
        # First-order transformations (11)
        "log_abs_delta",
        "log_abs_gamma",
        "log_abs_vega",
        "sqrt_abs_gamma",
        "sqrt_abs_vega",
        "delta_squared",
        "theta_per_day",
        "vega_time_weighted",
        "inverse_gamma",
        "delta_sign",
        "theta_sign",
        # Second-order transformations (7)
        "gamma_delta_ratio",
        "vega_delta_ratio",
        "vanna_vega_ratio",
        "delta_vega_product",
        "gamma_vega_product",
        "greek_stability",
        "convexity_measure",
        # Categorical (1)
        "moneyness_bin",
    ]

    # Metadata columns (preserve, not counted as features)
    metadata_columns = [
        "residual",  # Target variable
        "timestamp",
        "timestamp_seconds",  # Unix timestamp for debugging/joins
        "date",  # CRITICAL: Required for walk-forward validation
        "time_remaining",
        "S",  # Spot price (underlying price)
        "K",  # Strike price (target price)
        "sigma_mid",  # Implied volatility
        "T_years",  # Time to expiry in years
        "outcome",  # Binary outcome (0 or 1) - needed for evaluation
        "prob_mid",  # Mid probability - needed for evaluation
    ]

    # Regime columns (added by regime_detection_v5.py, may not exist yet)
    regime_columns = [
        "combined_regime",
        "temporal_regime",
        "market_regime",
        "volatility_regime",
    ]

    # Combine all features to keep
    features_to_keep = moneyness_features + volatility_features + greek_features + metadata_columns

    # Check which columns exist in the DataFrame
    available_columns = df.collect_schema().names()
    missing_features = [col for col in features_to_keep if col not in available_columns]

    if missing_features:
        logger.warning(f"Missing {len(missing_features)} expected features: {missing_features[:5]}...")
        # Filter to only existing columns
        features_to_keep = [col for col in features_to_keep if col in available_columns]

    # Add regime columns if they exist (they won't exist before regime_detection_v5.py runs)
    regime_cols_present = [col for col in regime_columns if col in available_columns]
    if regime_cols_present:
        logger.info(f"Found {len(regime_cols_present)} regime columns, keeping them")
        features_to_keep.extend(regime_cols_present)

    # Select only these columns
    df_selected = df.select(features_to_keep)

    # Log summary
    logger.info(f"Selected {len(features_to_keep)} columns:")
    logger.info(f"  - Moneyness features: {len(moneyness_features)}")
    logger.info(f"  - Volatility features: {len(volatility_features)}")
    logger.info(f"  - Greek features: {len(greek_features)}")
    logger.info(f"  - Metadata columns: {len(metadata_columns)}")
    if regime_cols_present:
        logger.info(f"  - Regime columns: {len(regime_cols_present)}")

    # Calculate feature count (excluding metadata)
    feature_count = len(moneyness_features) + len(volatility_features) + len(greek_features)
    logger.info(f"Total feature count (excludes metadata): {feature_count}")

    if feature_count >= 50:
        logger.warning(f"Feature count {feature_count} exceeds target <50!")

    return df_selected


def main():
    """Main feature selection pipeline."""

    logger.info("=" * 80)
    logger.info("V5 FEATURE SELECTION")
    logger.info("=" * 80)

    # Get project root (search up from script location)
    script_dir = Path(__file__).parent.resolve()  # research/model/00_data_prep
    model_dir = script_dir.parent  # research/model
    project_root = model_dir.parent  # /home/ubuntu/Polymarket

    # Paths (all absolute)
    input_path = model_dir / "data" / "consolidated_features_v5_pipeline_ready.parquet"
    output_path = model_dir / "data" / "consolidated_features_v5_selected.parquet"

    # Log resolved paths
    logger.info(f"Project root: {project_root}")
    logger.info(f"Input: {input_path}")
    logger.info(f"Output: {output_path}")
    logger.info("")

    # Check input exists
    if not input_path.exists():
        raise FileNotFoundError(
            f"Input file not found: {input_path}\nRun engineer_greeks_v5.py first to generate V5 features"
        )

    # Load V5 features with Greeks (lazily)
    logger.info(f"Loading V5 features with Greeks: {input_path}")
    df = pl.scan_parquet(input_path)

    # Get schema info
    schema_check = df.select([pl.len()]).collect()
    row_count = schema_check[0, 0]
    input_col_count = len(df.collect_schema().names())
    logger.info(f"Input: {row_count:,} rows, {input_col_count} columns")

    # Select minimalist features
    df_selected = select_minimalist_features_v5(df)

    # Write output (streaming for memory efficiency)
    logger.info(f"Writing selected features to {output_path}...")
    df_selected.sink_parquet(
        path=str(output_path),
        compression="snappy",
        statistics=True,
    )

    logger.info("✓ V5 feature selection complete")

    # Verify output
    df_verify = pl.scan_parquet(output_path)
    output_col_count = len(df_verify.collect_schema().names())
    verify_row_count = df_verify.select([pl.len()]).collect()[0, 0]

    logger.info("")
    logger.info("=" * 80)
    logger.info("FEATURE SELECTION COMPLETE")
    logger.info("=" * 80)
    logger.info(f"Input:  {input_path.name} ({input_col_count} columns)")
    logger.info(f"Output: {output_path.name} ({output_col_count} columns)")
    logger.info(f"Rows: {verify_row_count:,}")
    logger.info(f"Columns reduced: {input_col_count} → {output_col_count} ({output_col_count - input_col_count:+d})")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
