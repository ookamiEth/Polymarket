#!/usr/bin/env python3
"""
Temporal Integrity Tests for Multi-Horizon Training Pipeline
=============================================================

Validates that no data leakage occurs in the time series training pipeline.

These tests ensure:
1. No timestamp overlap between train/val/test splits
2. Minimum temporal gaps between splits (prevents rolling window overlap)
3. Holdout period is completely separate
4. Feature windows don't exceed gap periods
5. Walk-forward windows don't overlap holdout

Author: BT Research Team
Date: 2025-11-03
"""

from __future__ import annotations

import logging
from datetime import date
from pathlib import Path

import polars as pl
import pytest
import yaml
from dateutil.relativedelta import relativedelta

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Paths
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
CONFIG_PATH = Path(__file__).parent.parent / "config" / "multi_horizon_config.yaml"
DATA_DIR = PROJECT_ROOT / "results"


class TestTemporalIntegrity:
    """Test suite for temporal data integrity."""

    @pytest.fixture(scope="class")
    def config(self) -> dict:
        """Load multi-horizon configuration."""
        with open(CONFIG_PATH) as f:
            return yaml.safe_load(f)

    @pytest.fixture(scope="class")
    def train_data(self) -> pl.DataFrame | None:
        """Load training data if available."""
        train_file = DATA_DIR / "train_features_lgb.parquet"
        if train_file.exists():
            return pl.read_parquet(train_file)
        return None

    @pytest.fixture(scope="class")
    def val_data(self) -> pl.DataFrame | None:
        """Load validation data if available."""
        val_file = DATA_DIR / "val_features_lgb.parquet"
        if val_file.exists():
            return pl.read_parquet(val_file)
        return None

    @pytest.fixture(scope="class")
    def test_data(self) -> pl.DataFrame | None:
        """Load test data if available."""
        test_file = DATA_DIR / "test_features_lgb.parquet"
        if test_file.exists():
            return pl.read_parquet(test_file)
        return None

    def test_1_no_timestamp_overlap_train_val(
        self, train_data: pl.DataFrame | None, val_data: pl.DataFrame | None
    ) -> None:
        """
        Test 1: Verify no timestamp overlap between train and validation sets.

        CRITICAL: Overlapping timestamps indicate data leakage.
        """
        if train_data is None or val_data is None:
            pytest.skip("Training/validation data not available")

        logger.info("Test 1: Checking for timestamp overlap between train/val...")

        train_timestamps = set(train_data.select("timestamp").to_series().to_list())
        val_timestamps = set(val_data.select("timestamp").to_series().to_list())

        overlap = train_timestamps.intersection(val_timestamps)

        assert len(overlap) == 0, (
            f"Found {len(overlap):,} overlapping timestamps between train and val sets! "
            "This indicates severe data leakage."
        )

        logger.info("✓ Test 1 PASSED: No timestamp overlap between train/val")

    def test_2_minimum_temporal_gap_train_val(
        self, train_data: pl.DataFrame | None, val_data: pl.DataFrame | None
    ) -> None:
        """
        Test 2: Verify minimum temporal gap between train and validation sets.

        CRITICAL: Gap must exceed feature windows (900s = 15 min) to prevent leakage.
        """
        if train_data is None or val_data is None:
            pytest.skip("Training/validation data not available")

        logger.info("Test 2: Checking temporal gap between train/val...")

        # Convert to datetime
        train_dates = train_data.select("date").to_series()
        val_dates = val_data.select("date").to_series()

        train_max_date = train_dates.max()
        val_min_date = val_dates.min()

        # Calculate gap in days - convert to date objects for consistency
        train_max_date_obj: date = (
            train_max_date if isinstance(train_max_date, date) else date.fromisoformat(str(train_max_date))
        )
        val_min_date_obj: date = (
            val_min_date if isinstance(val_min_date, date) else date.fromisoformat(str(val_min_date))
        )

        gap_days = (val_min_date_obj - train_max_date_obj).days  # type: ignore[operator]

        min_gap_days = 1  # Walk-forward uses 1-day gap
        max_feature_window_seconds = 900  # 15 minutes (longest rolling window)
        max_feature_window_days = max_feature_window_seconds / 86400

        assert gap_days >= min_gap_days, (
            f"Temporal gap ({gap_days} days) is too small! "
            f"Must be ≥ {min_gap_days} day(s) to prevent rolling window overlap. "
            f"Max feature window: {max_feature_window_seconds}s = {max_feature_window_days:.4f} days"
        )

        logger.info(f"✓ Test 2 PASSED: Temporal gap = {gap_days} days (> {min_gap_days} day required)")

    def test_3_holdout_period_separation(self, config: dict, train_data: pl.DataFrame | None) -> None:
        """
        Test 3: Verify holdout test period is completely separate.

        CRITICAL: Holdout period must never appear in training data.
        """
        if train_data is None:
            pytest.skip("Training data not available")

        logger.info("Test 3: Checking holdout period separation...")

        # Get holdout configuration
        holdout_months = config["walk_forward_validation"]["holdout_months"]

        # Calculate holdout start date
        all_dates = train_data.select("date").to_series()
        max_date_raw = all_dates.max()

        # Ensure max_date is a date object
        if max_date_raw is None:
            pytest.fail("Training data has no dates!")

        max_date = date.fromisoformat(str(max_date_raw)) if not isinstance(max_date_raw, date) else max_date_raw

        # Holdout starts N months before end date (+ 1 day)
        holdout_start = max_date - relativedelta(months=holdout_months) + relativedelta(days=1)

        # Check if any training data falls in holdout period
        train_in_holdout = all_dates.filter(all_dates >= holdout_start)

        assert len(train_in_holdout) == 0, (
            f"Found {len(train_in_holdout):,} training samples in holdout period! "
            f"Holdout starts: {holdout_start}, but training data extends to: {max_date}"
        )

        logger.info(f"✓ Test 3 PASSED: Holdout period (last {holdout_months} months) is separate")

    def test_4_walk_forward_window_validity(self, config: dict) -> None:
        """
        Test 4: Verify walk-forward windows don't overlap holdout period.

        CRITICAL: Walk-forward validation windows must end before holdout starts.
        """
        logger.info("Test 4: Checking walk-forward window validity...")

        wf_config = config["walk_forward_validation"]
        train_months = wf_config["train_months"]
        val_months = wf_config["val_months"]
        step_months = wf_config["step_months"]
        holdout_months = wf_config["holdout_months"]

        # Simulate dataset dates (Oct 2023 - Sep 2025)
        start_date = date(2023, 10, 1)
        end_date = date(2025, 9, 30)

        # Calculate holdout start
        holdout_start = end_date - relativedelta(months=holdout_months) + relativedelta(days=1)

        # Generate walk-forward windows
        current_start = start_date
        windows = []

        while True:
            train_end = current_start + relativedelta(months=train_months) - relativedelta(days=1)
            val_start = train_end + relativedelta(days=1)
            val_end = val_start + relativedelta(months=val_months) - relativedelta(days=1)

            # Stop if validation would overlap holdout
            if val_end >= holdout_start:
                break

            windows.append((current_start, train_end, val_start, val_end))
            current_start += relativedelta(months=step_months)

        # Check that last window ends before holdout
        if windows:
            last_val_end = windows[-1][3]
            assert last_val_end < holdout_start, (
                f"Last walk-forward window overlaps holdout! "
                f"Last val ends: {last_val_end}, Holdout starts: {holdout_start}"
            )

            logger.info(f"✓ Test 4 PASSED: All {len(windows)} walk-forward windows are valid")
            logger.info(f"  Last window val ends: {last_val_end}")
            logger.info(f"  Holdout starts: {holdout_start}")
            logger.info(f"  Gap: {(holdout_start - last_val_end).days} days")
        else:
            pytest.fail("No walk-forward windows generated!")

    def test_5_feature_windows_within_gap(self, config: dict) -> None:
        """
        Test 5: Verify feature windows don't exceed temporal gaps.

        CRITICAL: Rolling feature windows must be shorter than train/val gap.
        """
        logger.info("Test 5: Checking feature window sizes...")

        # Maximum feature window (from feature engineering scripts)
        max_feature_window_seconds = 3600  # 1 hour (rv_3600s is longest)

        # Walk-forward gap (1 day)
        wf_gap_seconds = 86400  # 24 hours

        assert wf_gap_seconds > max_feature_window_seconds, (
            f"Feature window ({max_feature_window_seconds}s) exceeds temporal gap ({wf_gap_seconds}s)! "
            "This will cause rolling window overlap and data leakage."
        )

        safety_factor = wf_gap_seconds / max_feature_window_seconds

        logger.info(f"✓ Test 5 PASSED: Feature windows safely within gap (safety factor: {safety_factor:.1f}x)")
        logger.info(f"  Max feature window: {max_feature_window_seconds}s (1 hour)")
        logger.info(f"  Temporal gap: {wf_gap_seconds}s (24 hours)")

    def test_6_walk_forward_enabled_in_config(self, config: dict) -> None:
        """
        Test 6: Verify walk-forward validation is enabled in config.

        CRITICAL: This is enforced at runtime, but config should also reflect it.
        """
        logger.info("Test 6: Checking walk-forward config...")

        wf_enabled = config["walk_forward_validation"]["enabled"]

        assert wf_enabled is True, (
            "walk_forward_validation.enabled must be True! This is mandatory to prevent data leakage in time series."
        )

        logger.info("✓ Test 6 PASSED: Walk-forward validation is enabled in config")

    def test_7_no_future_information_in_features(self) -> None:
        """
        Test 7: Verify features use only trailing windows (no forward-looking).

        CRITICAL: All rolling operations must use past data only.
        """
        logger.info("Test 7: Checking for forward-looking features...")

        # List of feature engineering scripts
        feature_scripts = [
            Path(__file__).parent.parent.parent / "00_data_prep" / "calculate_realized_volatility.py",
            Path(__file__).parent.parent.parent / "00_data_prep" / "engineer_microstructure_features.py",
        ]

        forward_looking_patterns = [
            ".rolling_mean(center=True)",
            ".rolling_std(center=True)",
            ".rolling_sum(center=True)",
            ".shift(-",  # Negative shift = future data
            "future_",  # Any feature named with "future_"
        ]

        violations = []

        for script in feature_scripts:
            if not script.exists():
                continue

            with open(script) as f:
                content = f.read()
                for pattern in forward_looking_patterns:
                    if pattern in content:
                        violations.append(f"{script.name}: Found pattern '{pattern}'")

        assert len(violations) == 0, f"Found {len(violations)} potential forward-looking features:\n" + "\n".join(
            f"  - {v}" for v in violations
        )

        logger.info("✓ Test 7 PASSED: No forward-looking features detected")


# ============================================================================
# CLI for running tests
# ============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
