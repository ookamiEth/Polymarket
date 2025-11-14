#!/usr/bin/env python3
"""
Edge Case Fixtures for V4 Pipeline Testing

Provides edge case datasets to test error handling, validation, and robustness.

Author: Test Infrastructure
Date: 2025-01-14
"""

from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import polars as pl
from synthetic_v4_data_generator import generate_v4_features


def create_missing_features_data(n_samples: int = 1000, missing_features: Optional[list[str]] = None) -> pl.DataFrame:
    """
    Create dataset with missing features (simulates schema changes).

    Args:
        n_samples: Number of samples
        missing_features: List of features to remove (default: remove critical features)

    Returns:
        DataFrame with missing features
    """
    df = generate_v4_features(n_samples=n_samples, seed=50)

    if missing_features is None:
        # Remove critical features that would break training
        missing_features = ["rv_900s", "moneyness", "time_remaining", "sigma_mid"]

    return df.drop(missing_features)


def create_nan_features_data(n_samples: int = 1000, nan_fraction: float = 0.1) -> pl.DataFrame:
    """
    Create dataset with NaN values in features.

    Args:
        n_samples: Number of samples
        nan_fraction: Fraction of rows with NaN (0.0 to 1.0)

    Returns:
        DataFrame with NaN in multiple features
    """
    df = generate_v4_features(n_samples=n_samples, seed=51)

    # Inject NaN into multiple features
    features_to_corrupt = ["rv_900s", "moneyness", "bid_ask_spread_bps", "funding_rate"]

    for feature in features_to_corrupt:
        df = df.with_columns(
            [
                pl.when(pl.col("timestamp_seconds") % int(1 / nan_fraction) == 0)
                .then(None)
                .otherwise(pl.col(feature))
                .alias(feature)
            ]
        )

    return df


def create_nan_outcomes_data(n_samples: int = 1000, nan_fraction: float = 0.2) -> pl.DataFrame:
    """
    Create dataset with NaN outcomes (corrupted labels).

    Args:
        n_samples: Number of samples
        nan_fraction: Fraction of rows with NaN outcomes

    Returns:
        DataFrame with NaN in outcome column
    """
    df = generate_v4_features(n_samples=n_samples, seed=52)

    df = df.with_columns(
        [
            pl.when(pl.col("timestamp_seconds") % int(1 / nan_fraction) == 0)
            .then(None)
            .otherwise(pl.col("outcome"))
            .alias("outcome")
        ]
    )

    return df


def create_inf_values_data(n_samples: int = 1000) -> pl.DataFrame:
    """
    Create dataset with infinite values (division by zero scenarios).

    Returns:
        DataFrame with inf values in features
    """
    df = generate_v4_features(n_samples=n_samples, seed=53)

    # Inject inf values
    df = df.with_columns(
        [
            pl.when(pl.col("timestamp_seconds") % 50 == 0)
            .then(float("inf"))
            .otherwise(pl.col("rv_900s"))
            .alias("rv_900s"),
            pl.when(pl.col("timestamp_seconds") % 75 == 0)
            .then(float("-inf"))
            .otherwise(pl.col("funding_rate"))
            .alias("funding_rate"),
        ]
    )

    return df


def create_duplicate_timestamps_data(n_samples: int = 1000) -> pl.DataFrame:
    """
    Create dataset with duplicate timestamps (simulates contract_id collision).

    Returns:
        DataFrame with duplicate timestamps
    """
    # Create two datasets with same timestamps but different data
    df1 = generate_v4_features(n_samples=n_samples // 2, seed=54)
    df2 = generate_v4_features(n_samples=n_samples // 2, seed=55)

    # Force df2 to have same timestamps as df1
    timestamps_to_duplicate = df1["timestamp_seconds"].to_numpy()
    df2 = df2.with_columns([pl.Series("timestamp_seconds", timestamps_to_duplicate)])

    # Different outcomes/features (simulates different contracts)
    df2 = df2.with_columns(
        [
            (pl.col("outcome") * 0 + 1 - pl.col("outcome")).alias("outcome"),  # Flip outcomes
            (pl.col("moneyness") * 1.1).alias("moneyness"),  # Different strikes
        ]
    )

    return pl.concat([df1, df2])


def create_extreme_values_data(n_samples: int = 1000) -> pl.DataFrame:
    """
    Create dataset with extreme but valid values.

    Returns:
        DataFrame with extreme values (high vol, deep OTM, etc.)
    """
    df = generate_v4_features(n_samples=n_samples, seed=56)

    # Inject extreme values
    df = df.with_columns(
        [
            # Extreme volatility (market crash scenarios)
            pl.when(pl.col("timestamp_seconds") % 100 == 0)
            .then(5.0)  # 500% annualized vol
            .otherwise(pl.col("rv_900s"))
            .alias("rv_900s"),
            # Deep OTM (near-zero moneyness)
            pl.when(pl.col("timestamp_seconds") % 120 == 0)
            .then(0.01)  # 99% OTM
            .otherwise(pl.col("moneyness"))
            .alias("moneyness"),
            # Near-expiry (seconds to expiry)
            pl.when(pl.col("timestamp_seconds") % 80 == 0)
            .then(1.0)  # 1 second to expiry
            .otherwise(pl.col("time_remaining"))
            .alias("time_remaining"),
            # Extreme spread (illiquid market)
            pl.when(pl.col("timestamp_seconds") % 90 == 0)
            .then(500.0)  # 500 bps spread
            .otherwise(pl.col("bid_ask_spread_bps"))
            .alias("bid_ask_spread_bps"),
        ]
    )

    return df


def create_boundary_values_data(n_samples: int = 1000) -> pl.DataFrame:
    """
    Create dataset with values at regime boundaries (tests hysteresis).

    Returns:
        DataFrame with values exactly at regime thresholds
    """
    df = generate_v4_features(n_samples=n_samples, seed=57)

    # Calculate actual percentiles for thresholds
    vol_values = df["rv_900s"].to_numpy()
    vol_low_thresh = np.percentile(vol_values, 33)
    vol_high_thresh = np.percentile(vol_values, 67)

    # Place samples exactly at boundaries
    df = df.with_columns(
        [
            # Exactly at temporal boundaries
            pl.when(pl.col("timestamp_seconds") % 3 == 0)
            .then(300.0)  # Exactly at near/mid boundary
            .when(pl.col("timestamp_seconds") % 3 == 1)
            .then(600.0)  # Exactly at mid/far boundary
            .otherwise(pl.col("time_remaining"))
            .alias("time_remaining"),
            # Exactly at volatility thresholds
            pl.when(pl.col("timestamp_seconds") % 2 == 0)
            .then(pl.lit(vol_low_thresh))  # Exactly at low/medium boundary
            .otherwise(pl.lit(vol_high_thresh))  # Exactly at medium/high boundary
            .alias("rv_900s"),
            # Exactly at ATM threshold (moneyness_distance = 0.01)
            pl.when(pl.col("timestamp_seconds") % 4 == 0)
            .then(1.01)  # Exactly 0.01 away from ATM
            .otherwise(pl.col("moneyness"))
            .alias("moneyness"),
        ]
    )

    # Recalculate moneyness_distance
    df = df.with_columns([(pl.col("moneyness") - 1.0).abs().alias("moneyness_distance")])

    return df


def create_imbalanced_regime_data(n_samples: int = 10000) -> pl.DataFrame:
    """
    Create dataset with severely imbalanced regime distribution.

    Returns:
        DataFrame where some regimes have <100 samples
    """
    return generate_v4_features(n_samples=n_samples, regime_distribution="sparse", seed=58)


def create_temporal_leakage_data(n_samples: int = 5000) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    """
    Create train/val/test data with overlapping dates (tests leakage detection).

    Returns:
        Tuple of (train_df, val_df, test_df) with overlapping dates
    """
    # Generate data spanning 12 months
    df = generate_v4_features(n_samples=n_samples, seed=59)

    # Split with intentional overlap
    dates = df["date"].to_numpy()
    cutoff1 = int(np.percentile(dates.astype("int64"), 60))  # 60th percentile
    cutoff2 = int(np.percentile(dates.astype("int64"), 70))  # 70th percentile (OVERLAP!)

    train_df = df.filter(pl.col("date").cast(pl.Int64) <= pl.lit(cutoff1))
    val_df = df.filter(
        (pl.col("date").cast(pl.Int64) >= pl.lit(cutoff1 - 7 * 86400))  # Overlap: 7 days before cutoff1
        & (pl.col("date").cast(pl.Int64) <= pl.lit(cutoff2))
    )
    test_df = df.filter(pl.col("date").cast(pl.Int64) >= pl.lit(cutoff2))

    return train_df, val_df, test_df


def create_empty_regime_data(n_samples: int = 10000) -> pl.DataFrame:
    """
    Create dataset where one regime has zero samples.

    Returns:
        DataFrame missing one of the 12 regimes
    """
    df = generate_v4_features(n_samples=n_samples, regime_distribution="balanced", seed=60)

    # Remove all far_high_vol_otm samples
    df = df.filter(pl.col("combined_regime") != "far_high_vol_otm")

    return df


def create_incorrect_schema_data(n_samples: int = 1000) -> pl.DataFrame:
    """
    Create dataset with incorrect data types (tests schema validation).

    Returns:
        DataFrame with wrong dtypes
    """
    df = generate_v4_features(n_samples=n_samples, seed=61)

    # Convert some float columns to strings
    df = df.with_columns(
        [
            pl.col("rv_900s").cast(pl.Utf8).alias("rv_900s"),
            pl.col("moneyness").cast(pl.Utf8).alias("moneyness"),
        ]
    )

    return df


def create_mid_month_holdout_data(max_date: str = "2024-12-15") -> pl.DataFrame:
    """
    Create dataset ending mid-month (tests holdout calculation bug).

    Args:
        max_date: Maximum date in dataset (YYYY-MM-DD)

    Returns:
        DataFrame ending on a non-month-end date
    """
    n_samples = 50000
    df = generate_v4_features(n_samples=n_samples, seed=62)

    # Filter to end on max_date
    max_dt = datetime.fromisoformat(max_date)
    max_ts = int(max_dt.timestamp())

    df = df.filter(pl.col("timestamp_seconds") <= max_ts)

    return df


def create_zero_variance_features_data(n_samples: int = 1000) -> pl.DataFrame:
    """
    Create dataset where some features have zero variance (constant values).

    Returns:
        DataFrame with constant features
    """
    df = generate_v4_features(n_samples=n_samples, seed=63)

    # Set some features to constant values
    df = df.with_columns(
        [
            pl.lit(0.5).alias("rv_900s"),  # All samples have same vol
            pl.lit(1.0).alias("moneyness"),  # All ATM
            pl.lit(0.0).alias("funding_rate"),  # Zero funding
        ]
    )

    return df


def save_edge_case_fixtures(output_dir: Path) -> None:
    """Save all edge case fixtures to parquet files for testing."""
    output_dir.mkdir(parents=True, exist_ok=True)

    fixtures = {
        "missing_features.parquet": create_missing_features_data(),
        "nan_features.parquet": create_nan_features_data(),
        "nan_outcomes.parquet": create_nan_outcomes_data(),
        "inf_values.parquet": create_inf_values_data(),
        "duplicates.parquet": create_duplicate_timestamps_data(),
        "extreme_values.parquet": create_extreme_values_data(),
        "boundary_values.parquet": create_boundary_values_data(),
        "imbalanced_regimes.parquet": create_imbalanced_regime_data(),
        "empty_regime.parquet": create_empty_regime_data(),
        "incorrect_schema.parquet": create_incorrect_schema_data(),
        "mid_month_holdout.parquet": create_mid_month_holdout_data(),
        "zero_variance.parquet": create_zero_variance_features_data(),
    }

    # Save temporal leakage as 3 separate files
    train_df, val_df, test_df = create_temporal_leakage_data()
    train_df.write_parquet(output_dir / "temporal_leakage_train.parquet")
    val_df.write_parquet(output_dir / "temporal_leakage_val.parquet")
    test_df.write_parquet(output_dir / "temporal_leakage_test.parquet")

    # Save other fixtures
    for filename, df in fixtures.items():
        filepath = output_dir / filename
        df.write_parquet(filepath)
        print(f"✓ Saved {filename} ({len(df):,} rows)")

    print(f"\n✓ All edge case fixtures saved to {output_dir}")


if __name__ == "__main__":
    # Generate and save fixtures
    fixtures_dir = Path(__file__).parent
    save_edge_case_fixtures(fixtures_dir)
