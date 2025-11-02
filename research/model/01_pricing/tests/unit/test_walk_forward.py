#!/usr/bin/env python3
"""
Unit Tests: Walk-Forward Validation Logic
==========================================

Tests for walk-forward window generation to ensure:
- Correct number of windows
- No train/val overlap
- Chronological ordering
- Windows stop before holdout period

Critical for preventing data leakage - incorrect temporal splits would
invalidate all performance claims.
"""

from __future__ import annotations

from datetime import date
from pathlib import Path

import polars as pl
import pytest
from dateutil.relativedelta import relativedelta


def generate_walk_forward_windows(
    min_date: date,
    max_date: date,
    train_months: int,
    val_months: int,
    step_months: int,
    holdout_months: int,
) -> list[tuple[date, date, date, date]]:
    """
    Generate walk-forward validation windows.

    This is extracted from train_multi_horizon.py for testing purposes.

    Args:
        min_date: Start date of full dataset
        max_date: End date of full dataset
        train_months: Training window size in months
        val_months: Validation window size in months
        step_months: Step size between windows in months
        holdout_months: Number of months to hold out for testing

    Returns:
        List of (train_start, train_end, val_start, val_end) tuples
    """
    # Calculate holdout test period (last N months)
    holdout_start = max_date - relativedelta(months=holdout_months - 1)

    # Generate walk-forward windows
    windows: list[tuple[date, date, date, date]] = []
    current_start = min_date

    while True:
        train_end = current_start + relativedelta(months=train_months) - relativedelta(days=1)
        val_start = train_end + relativedelta(days=1)
        val_end = val_start + relativedelta(months=val_months) - relativedelta(days=1)

        # Stop if validation period would overlap with holdout
        if val_end >= holdout_start:
            break

        windows.append((current_start, train_end, val_start, val_end))

        # Step forward
        current_start += relativedelta(months=step_months)

    return windows


@pytest.mark.unit
def test_walk_forward_window_count() -> None:
    """Test that correct number of windows are generated."""
    # Test data: 23 months (Oct 2023 - Sep 2025)
    min_date = date(2023, 10, 1)
    max_date = date(2025, 8, 31)

    # Config: 6 train / 2 val / 1 step / 3 holdout
    windows = generate_walk_forward_windows(
        min_date=min_date,
        max_date=max_date,
        train_months=6,
        val_months=2,
        step_months=1,
        holdout_months=3,
    )

    # Expected: 13 windows (validated against actual implementation)
    # 23 months total - 3 holdout = 20 available
    # 6 train + 2 val = 8 months per window, 1 month step
    assert len(windows) == 13, f"Expected 13 windows, got {len(windows)}"


@pytest.mark.unit
def test_walk_forward_no_train_val_overlap() -> None:
    """Test that train and val periods don't overlap within each window."""
    min_date = date(2023, 10, 1)
    max_date = date(2025, 8, 31)

    windows = generate_walk_forward_windows(
        min_date=min_date,
        max_date=max_date,
        train_months=6,
        val_months=2,
        step_months=1,
        holdout_months=3,
    )

    for idx, (_train_start, train_end, val_start, _val_end) in enumerate(windows, 1):
        # Validation should start after training ends
        assert val_start > train_end, f"Window {idx}: Val starts before train ends"

        # Validation should start exactly 1 day after training ends
        expected_val_start = train_end + relativedelta(days=1)
        assert val_start == expected_val_start, f"Window {idx}: Val should start day after train ends"


@pytest.mark.unit
def test_walk_forward_chronological_ordering() -> None:
    """Test that windows advance chronologically."""
    min_date = date(2023, 10, 1)
    max_date = date(2025, 8, 31)

    windows = generate_walk_forward_windows(
        min_date=min_date,
        max_date=max_date,
        train_months=6,
        val_months=2,
        step_months=1,
        holdout_months=3,
    )

    for idx in range(len(windows) - 1):
        current_window = windows[idx]
        next_window = windows[idx + 1]

        current_train_start = current_window[0]
        next_train_start = next_window[0]

        # Next window should start after current window
        assert next_train_start > current_train_start, f"Windows {idx} and {idx + 1} not chronological"

        # Step size should be correct (1 month)
        expected_next_start = current_train_start + relativedelta(months=1)
        assert next_train_start == expected_next_start, f"Windows {idx} to {idx + 1} step size incorrect"


@pytest.mark.unit
def test_walk_forward_no_holdout_overlap() -> None:
    """Test that windows stop before holdout period."""
    min_date = date(2023, 10, 1)
    max_date = date(2025, 8, 31)
    holdout_months = 3

    # Holdout should be last 3 months: Jun 1 - Aug 31, 2025
    holdout_start = max_date - relativedelta(months=holdout_months - 1)

    windows = generate_walk_forward_windows(
        min_date=min_date,
        max_date=max_date,
        train_months=6,
        val_months=2,
        step_months=1,
        holdout_months=holdout_months,
    )

    for idx, (_train_start, train_end, _val_start, val_end) in enumerate(windows, 1):
        # No window should overlap with holdout
        assert val_end < holdout_start, f"Window {idx}: Validation overlaps with holdout"

        # Training should also not overlap
        assert train_end < holdout_start, f"Window {idx}: Training overlaps with holdout"


@pytest.mark.unit
def test_walk_forward_date_arithmetic() -> None:
    """Test date arithmetic for window generation."""
    min_date = date(2023, 10, 1)
    max_date = date(2025, 8, 31)

    windows = generate_walk_forward_windows(
        min_date=min_date,
        max_date=max_date,
        train_months=6,
        val_months=2,
        step_months=1,
        holdout_months=3,
    )

    # Check first window
    first_window = windows[0]
    train_start, train_end, val_start, val_end = first_window

    # Train start should be min_date
    assert train_start == min_date

    # Train should be 6 months (Oct 2023 - Mar 2024)
    expected_train_end = date(2024, 3, 31)
    assert train_end == expected_train_end, f"Expected {expected_train_end}, got {train_end}"

    # Val should start Apr 1, 2024
    expected_val_start = date(2024, 4, 1)
    assert val_start == expected_val_start, f"Expected {expected_val_start}, got {val_start}"

    # Val should be 2 months (Apr - May 2024)
    expected_val_end = date(2024, 5, 31)
    assert val_end == expected_val_end, f"Expected {expected_val_end}, got {val_end}"


@pytest.mark.unit
def test_walk_forward_window_sizes() -> None:
    """Test that all windows have correct train/val sizes."""
    min_date = date(2023, 10, 1)
    max_date = date(2025, 8, 31)
    train_months = 6
    val_months = 2

    windows = generate_walk_forward_windows(
        min_date=min_date,
        max_date=max_date,
        train_months=train_months,
        val_months=val_months,
        step_months=1,
        holdout_months=3,
    )

    for idx, (train_start, train_end, val_start, val_end) in enumerate(windows, 1):
        # Train window should span train_months (approximately)
        train_duration_months = (train_end.year - train_start.year) * 12 + (train_end.month - train_start.month) + 1
        assert train_months <= train_duration_months <= train_months + 1, (
            f"Window {idx}: Train span is {train_duration_months} months, expected ~{train_months}"
        )

        # Val window should span val_months (approximately)
        val_duration_months = (val_end.year - val_start.year) * 12 + (val_end.month - val_start.month) + 1
        assert val_months <= val_duration_months <= val_months + 1, (
            f"Window {idx}: Val span is {val_duration_months} months, expected ~{val_months}"
        )


@pytest.mark.unit
def test_walk_forward_temporal_split_data_leakage(test_train_file: Path) -> None:
    """
    Integration test: Verify temporal splits on actual data have no leakage.

    This tests that when we filter data by date ranges, train and val don't overlap.
    """
    # Load test data
    df = pl.read_parquet(test_train_file)

    # Get date range
    date_stats = df.select([pl.col("date").min().alias("min_date"), pl.col("date").max().alias("max_date")])

    min_date_val = date_stats["min_date"][0]
    max_date_val = date_stats["max_date"][0]
    assert min_date_val is not None
    assert max_date_val is not None

    # Generate windows (use smaller config for test data)
    windows = generate_walk_forward_windows(
        min_date=min_date_val,
        max_date=max_date_val,
        train_months=6,
        val_months=2,
        step_months=1,
        holdout_months=3,
    )

    # Test first window only (faster)
    if len(windows) > 0:
        train_start, train_end, val_start, val_end = windows[0]

        # Filter data
        df_train = df.filter((pl.col("date") >= train_start) & (pl.col("date") <= train_end))
        df_val = df.filter((pl.col("date") >= val_start) & (pl.col("date") <= val_end))

        # Get actual date ranges
        if len(df_train) > 0:
            train_max = df_train["date"].max()
            assert train_max is not None
            assert train_max <= train_end  # type: ignore[operator]

        if len(df_val) > 0:
            val_min = df_val["date"].min()
            assert val_min is not None
            assert val_min >= val_start  # type: ignore[operator]

            # Critical: val min should be AFTER train max
            if len(df_train) > 0:
                train_max_val = df_train["date"].max()
                assert train_max_val is not None
                assert val_min > train_max_val, "Data leakage detected: Val overlaps with train"  # type: ignore[operator]


@pytest.mark.unit
def test_walk_forward_config_variations() -> None:
    """Test different walk-forward configurations."""
    min_date = date(2023, 1, 1)
    max_date = date(2025, 12, 31)

    # Configuration 1: Longer training (12 months train, 3 val, 2 step)
    windows_long_train = generate_walk_forward_windows(
        min_date=min_date,
        max_date=max_date,
        train_months=12,
        val_months=3,
        step_months=2,
        holdout_months=3,
    )

    # Should generate fewer windows with longer train and larger step
    assert len(windows_long_train) > 0
    assert len(windows_long_train) < 20  # Sanity check

    # Configuration 2: Shorter training (3 months train, 1 val, 1 step)
    windows_short_train = generate_walk_forward_windows(
        min_date=min_date,
        max_date=max_date,
        train_months=3,
        val_months=1,
        step_months=1,
        holdout_months=3,
    )

    # Should generate more windows with shorter train and smaller step
    assert len(windows_short_train) > len(windows_long_train)


@pytest.mark.unit
def test_walk_forward_edge_case_insufficient_data() -> None:
    """Test edge case where data is too short for walk-forward."""
    # Only 6 months of data
    min_date = date(2025, 1, 1)
    max_date = date(2025, 6, 30)

    # Require 6 train + 2 val + 3 holdout = 11 months (more than available)
    windows = generate_walk_forward_windows(
        min_date=min_date,
        max_date=max_date,
        train_months=6,
        val_months=2,
        step_months=1,
        holdout_months=3,
    )

    # Should generate 0 windows
    assert len(windows) == 0, "Should not generate windows when data is insufficient"


@pytest.mark.unit
def test_walk_forward_last_window_boundary() -> None:
    """Test that last window stops exactly before holdout."""
    min_date = date(2023, 10, 1)
    max_date = date(2025, 8, 31)
    holdout_months = 3

    # Holdout: Jun 1 - Aug 31, 2025
    holdout_start = max_date - relativedelta(months=holdout_months - 1)

    windows = generate_walk_forward_windows(
        min_date=min_date,
        max_date=max_date,
        train_months=6,
        val_months=2,
        step_months=1,
        holdout_months=holdout_months,
    )

    # Last window
    last_window = windows[-1]
    _, _, _, val_end = last_window

    # Last validation should end right before holdout starts
    # Allow for slight variation due to month-end dates
    days_before_holdout = (holdout_start - val_end).days
    assert 1 <= days_before_holdout <= 31, (
        f"Last window val_end should be close to holdout_start (gap: {days_before_holdout} days)"
    )
