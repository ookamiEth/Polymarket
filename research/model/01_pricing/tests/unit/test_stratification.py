#!/usr/bin/env python3
"""
Unit Tests: Bucket Stratification Logic
========================================

Tests for stratify_data_by_time() function to ensure correct filtering
by time_remaining buckets with no data leakage across boundaries.

Critical for multi-horizon model correctness - incorrect stratification
would route wrong data to wrong models, invalidating all results.
"""

from __future__ import annotations

from pathlib import Path

import polars as pl
import pytest


@pytest.mark.unit
def test_stratification_near_bucket(test_train_file: Path, test_output_dir: Path) -> None:
    """Test near bucket (0-300s) filters correctly."""
    from train_multi_horizon import stratify_data_by_time

    output_file = stratify_data_by_time(
        input_file=test_train_file,
        output_dir=test_output_dir,
        bucket_name="near",
        time_min=0,
        time_max=300,
    )

    # Validate output
    df = pl.read_parquet(output_file)

    # Should have data
    assert len(df) > 0

    # All rows should be in [0, 300) range
    time_min_val = df["time_remaining"].min()
    time_max_val = df["time_remaining"].max()
    assert time_min_val is not None and float(time_min_val) >= 0  # type: ignore[arg-type]
    assert time_max_val is not None and float(time_max_val) < 300  # type: ignore[arg-type]

    # No rows with time_remaining >= 300
    assert (df["time_remaining"] >= 300).sum() == 0


@pytest.mark.unit
def test_stratification_mid_bucket(test_train_file: Path, test_output_dir: Path) -> None:
    """Test mid bucket (300-600s) filters correctly."""
    from train_multi_horizon import stratify_data_by_time

    output_file = stratify_data_by_time(
        input_file=test_train_file,
        output_dir=test_output_dir,
        bucket_name="mid",
        time_min=300,
        time_max=600,
    )

    # Validate output
    df = pl.read_parquet(output_file)

    # Should have data
    assert len(df) > 0

    # All rows should be in [300, 600) range
    time_min_val = df["time_remaining"].min()
    time_max_val = df["time_remaining"].max()
    assert time_min_val is not None and float(time_min_val) >= 300  # type: ignore[arg-type]
    assert time_max_val is not None and float(time_max_val) < 600  # type: ignore[arg-type]

    # No rows outside [300, 600)
    assert (df["time_remaining"] < 300).sum() == 0
    assert (df["time_remaining"] >= 600).sum() == 0


@pytest.mark.unit
def test_stratification_far_bucket(test_train_file: Path, test_output_dir: Path) -> None:
    """Test far bucket (600-900s) filters correctly."""
    from train_multi_horizon import stratify_data_by_time

    output_file = stratify_data_by_time(
        input_file=test_train_file,
        output_dir=test_output_dir,
        bucket_name="far",
        time_min=600,
        time_max=900,
    )

    # Validate output
    df = pl.read_parquet(output_file)

    # Should have data
    assert len(df) > 0

    # All rows should be in [600, 900) range
    time_min_val = df["time_remaining"].min()
    time_max_val = df["time_remaining"].max()
    assert time_min_val is not None and float(time_min_val) >= 600  # type: ignore[arg-type]
    assert time_max_val is not None and float(time_max_val) < 900  # type: ignore[arg-type]

    # No rows with time_remaining < 600
    assert (df["time_remaining"] < 600).sum() == 0


@pytest.mark.unit
def test_stratification_boundary_300s(test_output_dir: Path) -> None:
    """Test boundary at exactly 300s - should be in mid bucket."""
    from train_multi_horizon import stratify_data_by_time

    # Create test data with boundary values
    test_data = pl.DataFrame(
        {
            "time_remaining": [299, 300, 301],
            "outcome": [0, 1, 0],
            "prob_mid": [0.5, 0.5, 0.5],
            "date": pl.date_range(start=pl.date(2024, 1, 1), end=pl.date(2024, 1, 3), interval="1d", eager=True),
        }
    )

    test_file = test_output_dir / "boundary_300s_test.parquet"
    test_data.write_parquet(test_file)

    # Filter for near bucket (0-300s)
    near_output = stratify_data_by_time(
        input_file=test_file,
        output_dir=test_output_dir / "near",
        bucket_name="near",
        time_min=0,
        time_max=300,
    )

    # Filter for mid bucket (300-600s)
    mid_output = stratify_data_by_time(
        input_file=test_file,
        output_dir=test_output_dir / "mid",
        bucket_name="mid",
        time_min=300,
        time_max=600,
    )

    near_df = pl.read_parquet(near_output)
    mid_df = pl.read_parquet(mid_output)

    # 299 should be in near
    assert 299 in near_df["time_remaining"].to_list()

    # 300 and 301 should be in mid
    assert 300 in mid_df["time_remaining"].to_list()
    assert 301 in mid_df["time_remaining"].to_list()

    # 300 should NOT be in near
    assert 300 not in near_df["time_remaining"].to_list()


@pytest.mark.unit
def test_stratification_boundary_600s(test_output_dir: Path) -> None:
    """Test boundary at exactly 600s - should be in far bucket."""
    from train_multi_horizon import stratify_data_by_time

    # Create test data with boundary values
    test_data = pl.DataFrame(
        {
            "time_remaining": [599, 600, 601],
            "outcome": [0, 1, 0],
            "prob_mid": [0.5, 0.5, 0.5],
            "date": pl.date_range(start=pl.date(2024, 1, 1), end=pl.date(2024, 1, 3), interval="1d", eager=True),
        }
    )

    test_file = test_output_dir / "boundary_600s_test.parquet"
    test_data.write_parquet(test_file)

    # Filter for mid bucket (300-600s)
    mid_output = stratify_data_by_time(
        input_file=test_file,
        output_dir=test_output_dir / "mid",
        bucket_name="mid",
        time_min=300,
        time_max=600,
    )

    # Filter for far bucket (600-900s)
    far_output = stratify_data_by_time(
        input_file=test_file,
        output_dir=test_output_dir / "far",
        bucket_name="far",
        time_min=600,
        time_max=900,
    )

    mid_df = pl.read_parquet(mid_output)
    far_df = pl.read_parquet(far_output)

    # 599 should be in mid
    assert 599 in mid_df["time_remaining"].to_list()

    # 600 and 601 should be in far
    assert 600 in far_df["time_remaining"].to_list()
    assert 601 in far_df["time_remaining"].to_list()

    # 600 should NOT be in mid
    assert 600 not in mid_df["time_remaining"].to_list()


@pytest.mark.unit
def test_stratification_no_leakage(test_train_file: Path, test_output_dir: Path) -> None:
    """Test no data leakage across buckets - all rows accounted for exactly once."""
    from train_multi_horizon import stratify_data_by_time

    # Stratify into all 3 buckets
    near_output = stratify_data_by_time(
        input_file=test_train_file,
        output_dir=test_output_dir / "near",
        bucket_name="near",
        time_min=0,
        time_max=300,
    )

    mid_output = stratify_data_by_time(
        input_file=test_train_file,
        output_dir=test_output_dir / "mid",
        bucket_name="mid",
        time_min=300,
        time_max=600,
    )

    far_output = stratify_data_by_time(
        input_file=test_train_file,
        output_dir=test_output_dir / "far",
        bucket_name="far",
        time_min=600,
        time_max=900,
    )

    # Load all buckets
    df_original = pl.read_parquet(test_train_file)
    df_near = pl.read_parquet(near_output)
    df_mid = pl.read_parquet(mid_output)
    df_far = pl.read_parquet(far_output)

    # Sum of bucket row counts should equal original row count
    total_bucket_rows = len(df_near) + len(df_mid) + len(df_far)
    assert total_bucket_rows == len(df_original)

    # Check for overlaps - concatenate all bucket time_remaining values
    all_bucket_times = pl.concat(
        [
            df_near.select("time_remaining"),
            df_mid.select("time_remaining"),
            df_far.select("time_remaining"),
        ]
    )

    # Should have same count as original (no duplicates)
    assert len(all_bucket_times) == len(df_original)


@pytest.mark.unit
def test_stratification_preserves_columns(test_train_file: Path, test_output_dir: Path) -> None:
    """Test stratification preserves all original columns."""
    from train_multi_horizon import stratify_data_by_time

    df_original = pl.read_parquet(test_train_file)

    output_file = stratify_data_by_time(
        input_file=test_train_file,
        output_dir=test_output_dir,
        bucket_name="near",
        time_min=0,
        time_max=300,
    )

    df_stratified = pl.read_parquet(output_file)

    # All original columns should be present
    assert set(df_stratified.columns) == set(df_original.columns)


@pytest.mark.unit
def test_stratification_distribution(test_train_file: Path, test_output_dir: Path) -> None:
    """Test bucket distribution matches expected percentages (~33% each)."""
    from train_multi_horizon import stratify_data_by_time

    # Stratify into all 3 buckets
    near_output = stratify_data_by_time(
        input_file=test_train_file,
        output_dir=test_output_dir / "near",
        bucket_name="near",
        time_min=0,
        time_max=300,
    )

    mid_output = stratify_data_by_time(
        input_file=test_train_file,
        output_dir=test_output_dir / "mid",
        bucket_name="mid",
        time_min=300,
        time_max=600,
    )

    far_output = stratify_data_by_time(
        input_file=test_train_file,
        output_dir=test_output_dir / "far",
        bucket_name="far",
        time_min=600,
        time_max=900,
    )

    # Load all buckets
    df_original = pl.read_parquet(test_train_file)
    df_near = pl.read_parquet(near_output)
    df_mid = pl.read_parquet(mid_output)
    df_far = pl.read_parquet(far_output)

    total = len(df_original)
    near_pct = (len(df_near) / total) * 100
    mid_pct = (len(df_mid) / total) * 100
    far_pct = (len(df_far) / total) * 100

    # Test data is stratified to ~33% each (allow 5% tolerance)
    assert 28 <= near_pct <= 38, f"Near bucket: {near_pct:.1f}% (expected ~33%)"
    assert 28 <= mid_pct <= 38, f"Mid bucket: {mid_pct:.1f}% (expected ~33%)"
    assert 28 <= far_pct <= 38, f"Far bucket: {far_pct:.1f}% (expected ~33%)"
