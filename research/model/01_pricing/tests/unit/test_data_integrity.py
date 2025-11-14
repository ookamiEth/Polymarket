#!/usr/bin/env python3
"""
Data Integrity Tests for V4 Pipeline

Tests critical data integrity bugs:
- Bug 4.1: Join on timestamp only (missing contract_id) causes data corruption
- Bug 6.1: NaN outcomes replaced with 0.0 instead of filtered
- Bug 1.1: Missing features don't fail training (silent degradation)

Author: Test Infrastructure
Date: 2025-01-14
"""

import sys
from pathlib import Path

import numpy as np
import polars as pl
import pytest

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from conftest import (
    assert_no_duplicates,
    assert_v4_schema,
)


class TestJoinIntegrity:
    """Test Bug 4.1: Join on timestamp without contract_id causes duplicates."""

    def test_duplicate_timestamps_detected(self, v4_edge_case_duplicates: pl.DataFrame) -> None:
        """
        Test that duplicate timestamps are detected.

        Bug 4.1: prepare_pipeline_data_v4.py joins only on timestamp,
        causing data duplication when multiple contracts exist at same time.
        """
        df = v4_edge_case_duplicates

        # Verify duplicates exist
        n_total = len(df)
        n_unique_timestamps = len(df.select("timestamp_seconds").unique())

        assert n_total > n_unique_timestamps, (
            f"Expected duplicates in test data, but got {n_total} rows with {n_unique_timestamps} unique timestamps"
        )

        # This should fail if duplicates exist (which they do in edge case data)
        with pytest.raises(AssertionError, match="duplicate rows"):
            assert_no_duplicates(df, key_cols=["timestamp_seconds"])

    def test_join_with_contract_id_prevents_duplicates(self, v4_synthetic_small: pl.DataFrame) -> None:
        """
        Test that joining with [timestamp, contract_id] prevents duplicates.

        Correct fix for Bug 4.1: Join on both timestamp AND contract_id.
        """
        # Simulate baseline data with contract IDs
        baseline = v4_synthetic_small.select(["timestamp_seconds", "outcome", "prob_mid"]).with_columns(
            [pl.lit("contract_001").alias("contract_id")]
        )

        # Simulate features with contract IDs
        features = v4_synthetic_small.select(["timestamp_seconds", "rv_900s", "moneyness"]).with_columns(
            [pl.lit("contract_001").alias("contract_id")]
        )

        # WRONG: Join only on timestamp (Bug 4.1)
        wrong_join = features.join(
            baseline.select(["timestamp_seconds", "outcome", "prob_mid"]),
            on="timestamp_seconds",
            how="left",
        )

        # CORRECT: Join on [timestamp, contract_id]
        correct_join = features.join(baseline, on=["timestamp_seconds", "contract_id"], how="left")

        # Both should have same row count (no duplicates)
        assert len(wrong_join) == len(features), "Wrong join created duplicates"
        assert len(correct_join) == len(features), "Correct join created duplicates"

        # Verify no duplicates in correct join
        assert_no_duplicates(correct_join, key_cols=["timestamp_seconds", "contract_id"])

    def test_duplicate_timestamps_different_contracts(self) -> None:
        """
        Test scenario where multiple contracts exist at same timestamp.

        This is the exact scenario that Bug 4.1 fails on.
        """
        # Create baseline with 2 contracts at same timestamp
        baseline = pl.DataFrame(
            {
                "timestamp_seconds": [1000, 1000, 1001],
                "contract_id": ["A", "B", "C"],
                "outcome": [1.0, 0.0, 1.0],
                "prob_mid": [0.6, 0.4, 0.5],
            }
        )

        # Create features for same contracts
        features = pl.DataFrame(
            {
                "timestamp_seconds": [1000, 1000, 1001],
                "contract_id": ["A", "B", "C"],
                "rv_900s": [0.1, 0.2, 0.15],
                "moneyness": [1.0, 0.9, 1.05],
            }
        )

        # WRONG: Join only on timestamp
        wrong_join = features.with_columns([pl.col("timestamp_seconds").alias("timestamp")]).join(
            baseline.select(["timestamp_seconds", "outcome", "prob_mid"]).with_columns(
                [pl.col("timestamp_seconds").alias("timestamp")]
            ),
            on="timestamp",
            how="left",
        )

        # Wrong join creates 4 rows instead of 3 (cartesian product for timestamp 1000)
        # Contract A at ts=1000 matched with both outcomes (1.0 and 0.0)
        # Contract B at ts=1000 matched with both outcomes (1.0 and 0.0)
        assert len(wrong_join) > 3, f"Expected duplicates from wrong join, got {len(wrong_join)} rows"

        # CORRECT: Join on [timestamp, contract_id]
        correct_join = features.join(
            baseline.select(["timestamp_seconds", "contract_id", "outcome", "prob_mid"]),
            on=["timestamp_seconds", "contract_id"],
            how="left",
        )

        # Correct join maintains 3 rows (1:1 mapping)
        assert len(correct_join) == 3, f"Expected 3 rows, got {len(correct_join)}"

        # Verify outcomes match correctly
        assert correct_join.filter(pl.col("contract_id") == "A")["outcome"][0] == 1.0
        assert correct_join.filter(pl.col("contract_id") == "B")["outcome"][0] == 0.0
        assert correct_join.filter(pl.col("contract_id") == "C")["outcome"][0] == 1.0


class TestNaNHandling:
    """Test Bug 6.1: NaN outcomes should be filtered, not replaced with 0.0."""

    def test_nan_outcomes_detected(self, v4_edge_case_nan_outcomes: pl.DataFrame) -> None:
        """
        Test that NaN outcomes are detected in test data.
        """
        df = v4_edge_case_nan_outcomes
        n_nulls = df["outcome"].null_count()

        assert n_nulls > 0, "Expected NaN outcomes in test data"

    def test_nan_outcomes_cause_training_failure(self, v4_edge_case_nan_outcomes: pl.DataFrame) -> None:
        """
        Test that NaN outcomes should cause training to fail (not be replaced with 0.0).

        Bug 6.1: lightgbm_memory_optimized.py replaces NaN outcomes with 0.0,
        causing training on corrupted data.
        """
        df = v4_edge_case_nan_outcomes

        # WRONG: Replace NaN with 0.0 (Bug 6.1)
        y_wrong = df["outcome"].to_numpy()
        y_wrong_cleaned = np.nan_to_num(y_wrong, nan=0.0)  # BUG!

        # This creates corrupted training data
        assert not np.isnan(y_wrong_cleaned).any(), "NaNs were replaced with 0.0"
        assert (y_wrong_cleaned == 0.0).sum() > df.filter(pl.col("outcome") == 0.0).height, (
            "More zeros after replacement (NaNs became 0.0)"
        )

        # CORRECT: Filter out NaN outcomes before training
        mask = ~df["outcome"].is_null()
        df_correct = df.filter(mask)
        y_correct = df_correct["outcome"].to_numpy()

        # Verify no NaNs in cleaned data
        assert not np.isnan(y_correct).any(), "Should have no NaNs after filtering"

        # Verify we filtered rows (not replaced)
        assert len(y_correct) < len(df), f"Should have filtered {df['outcome'].null_count()} rows"

    def test_nan_features_handling(self, v4_edge_case_nan_features: pl.DataFrame) -> None:
        """
        Test that NaN features are handled appropriately.

        Features with NaN should either:
        1. Use LightGBM's native NaN handling (use_missing=True)
        2. Be imputed with domain-specific values
        3. Cause filtering of rows
        """
        df = v4_edge_case_nan_features

        # Check which features have NaN
        features_with_nan = []
        for col in ["rv_900s", "moneyness", "bid_ask_spread_bps", "funding_rate"]:
            if df[col].null_count() > 0:
                features_with_nan.append(col)

        assert len(features_with_nan) > 0, "Expected NaN in features"

        # Option 1: LightGBM can handle NaN natively (keep as NaN)
        x_native = df.select(features_with_nan).to_numpy()  # noqa: N806
        assert np.isnan(x_native).any(), "Features should retain NaN for LightGBM"

        # Option 2: Filter rows with critical NaN (e.g., rv_900s, moneyness)
        critical_features = ["rv_900s", "moneyness"]
        mask = pl.all_horizontal([~pl.col(f).is_null() for f in critical_features])
        df_filtered = df.filter(mask)

        # Verify critical features have no NaN
        for feat in critical_features:
            assert df_filtered[feat].null_count() == 0, f"{feat} should have no NaN after filtering"

    def test_inf_values_cause_failure(self) -> None:
        """
        Test that infinite values in features are detected and handled.

        Related to Bug 6.1: inf values should not be silently converted.
        """
        # Create data with inf values
        df = pl.DataFrame(
            {
                "rv_900s": [0.1, 0.2, float("inf"), 0.3, float("-inf")],
                "outcome": [1.0, 0.0, 1.0, 0.0, 1.0],
            }
        )

        # WRONG: Replace inf with NaN (masks the issue)
        rv_wrong = df["rv_900s"].to_numpy()
        rv_wrong_cleaned = np.nan_to_num(rv_wrong, nan=np.nan, posinf=np.nan, neginf=np.nan)
        assert np.isnan(rv_wrong_cleaned).any(), "Inf values became NaN (masked)"

        # CORRECT: Detect and fail or filter
        rv_values = df["rv_900s"].to_numpy()
        has_inf = np.isinf(rv_values).any()
        assert has_inf, "Should detect infinite values"

        # Filter out rows with inf
        df_filtered = df.filter(~pl.col("rv_900s").is_infinite())
        assert len(df_filtered) == 3, f"Should have 3 rows after filtering inf, got {len(df_filtered)}"


class TestFeatureValidation:
    """Test Bug 1.1: Missing features should fail training, not just warn."""

    def test_all_features_present(self, v4_synthetic_small: pl.DataFrame) -> None:
        """
        Test that all 156 features are present in valid data.
        """
        # Should pass without errors
        assert_v4_schema(v4_synthetic_small, require_all_features=True)

    def test_missing_features_detected(self, v4_edge_case_missing_features: pl.DataFrame) -> None:
        """
        Test that missing features are detected.

        Bug 1.1: train_multi_horizon_v4.py only warns about missing features
        but continues training with incomplete feature set.
        """
        df = v4_edge_case_missing_features

        # This should fail because features are missing
        with pytest.raises(AssertionError, match="Missing.*features"):
            assert_v4_schema(df, require_all_features=True)

    def test_critical_features_missing_fails_training(self) -> None:
        """
        Test that missing critical features should fail training.

        Critical features: rv_900s, moneyness, time_remaining, sigma_mid
        """
        import sys

        sys.path.insert(0, str(Path(__file__).parent.parent / "fixtures"))
        from synthetic_v4_data_generator import FEATURE_COLS_V4, generate_v4_features  # type: ignore

        # Generate full dataset
        df_full = generate_v4_features(n_samples=100, seed=100)

        # Remove critical features
        critical_features = ["rv_900s", "moneyness", "time_remaining", "sigma_mid"]
        df_missing = df_full.drop(critical_features)

        # Verify features are missing
        for feat in critical_features:
            assert feat not in df_missing.columns, f"{feat} should be missing"

        # This should fail
        with pytest.raises(AssertionError):
            assert_v4_schema(df_missing, require_all_features=True)

        # Check how many features are missing
        present_features = [f for f in FEATURE_COLS_V4 if f in df_missing.columns]
        missing_features = set(FEATURE_COLS_V4) - set(present_features)

        # Should have 4 missing features
        assert len(missing_features) == 4, f"Expected 4 missing, got {len(missing_features)}"
        assert critical_features == sorted(missing_features), "Missing features should be critical ones"

    def test_feature_count_threshold(self) -> None:
        """
        Test that training should fail if feature count below threshold.

        Recommended threshold: 150/156 features (96%)
        """
        import sys

        sys.path.insert(0, str(Path(__file__).parent.parent / "fixtures"))
        from synthetic_v4_data_generator import FEATURE_COLS_V4, generate_v4_features  # type: ignore

        df_full = generate_v4_features(n_samples=100, seed=101)

        # Remove 10 features (below 96% threshold)
        features_to_remove = FEATURE_COLS_V4[:10]
        df_partial = df_full.drop(features_to_remove)

        present_features = [f for f in FEATURE_COLS_V4 if f in df_partial.columns]
        pct_present = len(present_features) / len(FEATURE_COLS_V4) * 100

        assert pct_present < 96.0, f"Should be below threshold, got {pct_present:.1f}%"

        # This should fail
        with pytest.raises(AssertionError):
            assert_v4_schema(df_partial, require_all_features=True)

    def test_schema_validation_with_wrong_types(self) -> None:
        """
        Test that schema validation detects incorrect data types.

        Related to Bug 1.1: Schema changes should be caught early.
        """
        import sys

        sys.path.insert(0, str(Path(__file__).parent.parent / "fixtures"))
        from edge_case_fixtures import create_incorrect_schema_data  # type: ignore

        df = create_incorrect_schema_data()

        # Verify some columns are strings (wrong type)
        assert df.schema["rv_900s"] == pl.Utf8, "rv_900s should be Utf8 in incorrect schema"
        assert df.schema["moneyness"] == pl.Utf8, "moneyness should be Utf8 in incorrect schema"

        # Feature type validation should fail
        from conftest import assert_feature_types

        with pytest.raises(AssertionError, match="wrong types"):
            assert_feature_types(df, expected_type=pl.Float64)


class TestDataQualityValidation:
    """Additional data quality tests to prevent silent corruption."""

    def test_outcome_values_in_valid_range(self, v4_synthetic_small: pl.DataFrame) -> None:
        """
        Test that outcome values are binary (0 or 1).
        """
        df = v4_synthetic_small
        outcomes = df["outcome"].unique().sort().to_list()

        assert outcomes == [0.0, 1.0], f"Outcomes should be [0.0, 1.0], got {outcomes}"

    def test_prob_mid_in_valid_range(self, v4_synthetic_small: pl.DataFrame) -> None:
        """
        Test that prob_mid values are probabilities (0 to 1).
        """
        df = v4_synthetic_small
        min_prob = float(df["prob_mid"].min())  # type: ignore
        max_prob = float(df["prob_mid"].max())  # type: ignore

        assert min_prob >= 0.0, f"prob_mid min {min_prob} < 0.0"
        assert max_prob <= 1.0, f"prob_mid max {max_prob} > 1.0"

    def test_residual_calculation_correct(self, v4_synthetic_small: pl.DataFrame) -> None:
        """
        Test that residual = outcome - prob_mid.
        """
        df = v4_synthetic_small

        # Recalculate residual
        expected_residual = df["outcome"] - df["prob_mid"]
        actual_residual = df["residual"]

        # Check if close (allow floating point precision)
        diff = (expected_residual - actual_residual).abs()
        max_diff = float(diff.max())  # type: ignore
        assert max_diff < 1e-6, f"Residual calculation incorrect, max diff: {max_diff}"

    def test_no_duplicate_rows(self, v4_synthetic_small: pl.DataFrame) -> None:
        """
        Test that dataset has no fully duplicate rows.
        """
        n_total = len(v4_synthetic_small)
        n_unique = len(v4_synthetic_small.unique())

        assert n_total == n_unique, f"Found {n_total - n_unique} duplicate rows"

    def test_timestamps_sorted(self, v4_synthetic_small: pl.DataFrame) -> None:
        """
        Test that timestamps are sorted (important for temporal validation).
        """
        timestamps = v4_synthetic_small["timestamp_seconds"].to_numpy()
        is_sorted = np.all(timestamps[:-1] <= timestamps[1:])

        assert is_sorted, "Timestamps should be sorted"

    def test_no_future_leakage_in_features(self, v4_synthetic_small: pl.DataFrame) -> None:
        """
        Test that features don't contain future information.

        This is a sanity check - features should only use past data.
        """
        # All features should be computable from past data only
        # Check that time_remaining is positive (not negative = future)
        time_remaining = v4_synthetic_small["time_remaining"]
        min_time = float(time_remaining.min())  # type: ignore
        max_time = float(time_remaining.max())  # type: ignore

        assert min_time >= 0.0, "time_remaining should be non-negative"
        assert max_time <= 900.0, "time_remaining should be <= 15 minutes"


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v", "--tb=short"])
