"""
Pytest configuration and shared fixtures for multi-horizon pipeline tests.

This module provides:
- Shared test fixtures (paths, configs, test data)
- Helper assertion functions
- Test cleanup utilities
"""

import shutil
from pathlib import Path
from typing import Optional

import polars as pl
import pytest

# ============================================================================
# DIRECTORY PATHS
# ============================================================================


@pytest.fixture(scope="session")
def project_root() -> Path:
    """Return project root directory."""
    return Path(__file__).parent.parent


@pytest.fixture(scope="session")
def fixtures_dir() -> Path:
    """Return test fixtures directory."""
    return Path(__file__).parent / "fixtures"


@pytest.fixture(scope="session")
def test_output_dir(tmp_path_factory) -> Path:
    """Create temporary directory for test outputs."""
    return tmp_path_factory.mktemp("test_outputs")


# ============================================================================
# TEST DATA FIXTURES
# ============================================================================


@pytest.fixture(scope="session")
def test_train_file(fixtures_dir: Path) -> Path:
    """Return path to test training data (60K rows)."""
    path = fixtures_dir / "test_train.parquet"
    if not path.exists():
        pytest.fail(
            f"Test training data not found: {path}\nRun: cd tests/fixtures && uv run python generate_test_data.py"
        )
    return path


@pytest.fixture(scope="session")
def test_val_file(fixtures_dir: Path) -> Path:
    """Return path to test validation data (20K rows)."""
    path = fixtures_dir / "test_val.parquet"
    if not path.exists():
        pytest.fail(
            f"Test validation data not found: {path}\nRun: cd tests/fixtures && uv run python generate_test_data.py"
        )
    return path


@pytest.fixture(scope="session")
def test_test_file(fixtures_dir: Path) -> Path:
    """Return path to test holdout data (20K rows)."""
    path = fixtures_dir / "test_test.parquet"
    if not path.exists():
        pytest.fail(
            f"Test holdout data not found: {path}\nRun: cd tests/fixtures && uv run python generate_test_data.py"
        )
    return path


@pytest.fixture(scope="function")
def test_train_df(test_train_file: Path) -> pl.DataFrame:
    """Load test training data as DataFrame."""
    return pl.read_parquet(test_train_file)


@pytest.fixture(scope="function")
def test_val_df(test_val_file: Path) -> pl.DataFrame:
    """Load test validation data as DataFrame."""
    return pl.read_parquet(test_val_file)


@pytest.fixture(scope="function")
def test_test_df(test_test_file: Path) -> pl.DataFrame:
    """Load test holdout data as DataFrame."""
    return pl.read_parquet(test_test_file)


# ============================================================================
# SYNTHETIC DATA FIXTURES
# ============================================================================


@pytest.fixture
def synthetic_predictions_df() -> pl.DataFrame:
    """
    Create synthetic prediction data for testing metrics.

    Returns DataFrame with:
    - outcome: binary 0/1
    - prob_mid: baseline probability
    - prediction: residual predictions
    - time_remaining: for bucket routing
    """
    return pl.DataFrame(
        {
            "outcome": [0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
            "prob_mid": [0.5, 0.5, 0.6, 0.4, 0.5, 0.5, 0.7, 0.3, 0.5, 0.5],
            "prediction": [-0.1, 0.2, -0.2, 0.3, -0.1, 0.1, -0.3, 0.4, 0.0, 0.0],
            "time_remaining": [100, 150, 200, 250, 350, 400, 450, 500, 700, 800],
        }
    )


# ============================================================================
# V4 SYNTHETIC DATA FIXTURES
# ============================================================================


@pytest.fixture
def v4_synthetic_small() -> pl.DataFrame:
    """
    Generate small V4 dataset (1K rows) for fast tests.

    Returns:
        DataFrame with all 156 V4 features + regime columns
    """
    import sys

    sys.path.insert(0, str(Path(__file__).parent / "fixtures"))
    from synthetic_v4_data_generator import generate_v4_features  # type: ignore

    return generate_v4_features(n_samples=1000, regime_distribution="balanced", seed=42)


@pytest.fixture
def v4_synthetic_medium() -> pl.DataFrame:
    """
    Generate medium V4 dataset (10K rows) for moderate tests.

    Returns:
        DataFrame with all 156 V4 features + regime columns
    """
    import sys

    sys.path.insert(0, str(Path(__file__).parent / "fixtures"))
    from synthetic_v4_data_generator import generate_v4_features  # type: ignore

    return generate_v4_features(n_samples=10_000, regime_distribution="realistic", seed=42)


@pytest.fixture
def v4_synthetic_large() -> pl.DataFrame:
    """
    Generate large V4 dataset (100K rows) for integration tests.

    Returns:
        DataFrame with all 156 V4 features + regime columns
    """
    import sys

    sys.path.insert(0, str(Path(__file__).parent / "fixtures"))
    from synthetic_v4_data_generator import generate_v4_features  # type: ignore

    return generate_v4_features(n_samples=100_000, regime_distribution="realistic", seed=42)


@pytest.fixture
def v4_edge_case_missing_features() -> pl.DataFrame:
    """Load edge case: missing features."""
    import sys

    sys.path.insert(0, str(Path(__file__).parent / "fixtures"))
    from edge_case_fixtures import create_missing_features_data  # type: ignore

    return create_missing_features_data()


@pytest.fixture
def v4_edge_case_nan_features() -> pl.DataFrame:
    """Load edge case: NaN in features."""
    import sys

    sys.path.insert(0, str(Path(__file__).parent / "fixtures"))
    from edge_case_fixtures import create_nan_features_data  # type: ignore

    return create_nan_features_data()


@pytest.fixture
def v4_edge_case_nan_outcomes() -> pl.DataFrame:
    """Load edge case: NaN in outcomes."""
    import sys

    sys.path.insert(0, str(Path(__file__).parent / "fixtures"))
    from edge_case_fixtures import create_nan_outcomes_data  # type: ignore

    return create_nan_outcomes_data()


@pytest.fixture
def v4_edge_case_duplicates() -> pl.DataFrame:
    """Load edge case: duplicate timestamps."""
    import sys

    sys.path.insert(0, str(Path(__file__).parent / "fixtures"))
    from edge_case_fixtures import create_duplicate_timestamps_data  # type: ignore

    return create_duplicate_timestamps_data()


# ============================================================================
# CONFIG FIXTURES
# ============================================================================


@pytest.fixture(scope="session")
def config_dir(project_root: Path) -> Path:
    """Return config directory."""
    return project_root / "config"


@pytest.fixture(scope="session")
def multi_horizon_config_file(config_dir: Path) -> Path:
    """Return path to multi-horizon config YAML."""
    return config_dir / "multi_horizon_config.yaml"


# ============================================================================
# CLEANUP UTILITIES
# ============================================================================


@pytest.fixture(autouse=True)
def cleanup_test_outputs(test_output_dir: Path):
    """Automatically cleanup test outputs after each test."""
    yield
    # Cleanup after test
    if test_output_dir.exists():
        for item in test_output_dir.iterdir():
            if item.is_file():
                item.unlink()
            elif item.is_dir():
                shutil.rmtree(item)


# ============================================================================
# ASSERTION HELPERS
# ============================================================================


def assert_valid_parquet(file_path: Path, min_rows: int = 1) -> None:
    """
    Assert that a Parquet file exists and is valid.

    Args:
        file_path: Path to Parquet file
        min_rows: Minimum expected row count

    Raises:
        AssertionError: If file invalid
    """
    assert file_path.exists(), f"Parquet file does not exist: {file_path}"
    assert file_path.suffix == ".parquet", f"Not a Parquet file: {file_path}"

    # Try reading
    df = pl.read_parquet(file_path)
    assert len(df) >= min_rows, f"Expected at least {min_rows} rows, got {len(df)}"


def assert_metrics_valid(metrics: dict, check_improvement: bool = True) -> None:
    """
    Assert that metrics dictionary contains valid values.

    Args:
        metrics: Dictionary with metric keys
        check_improvement: Whether to check for positive improvement

    Raises:
        AssertionError: If metrics invalid
    """
    # Required keys
    required_keys = [
        "brier_baseline",
        "brier_model",
        "brier_improvement_pct",
        "residual_mse",
        "residual_rmse",
        "residual_mae",
    ]

    for key in required_keys:
        assert key in metrics, f"Missing required metric: {key}"

    # Value ranges
    assert 0 <= metrics["brier_baseline"] <= 1, "brier_baseline must be in [0, 1]"
    assert 0 <= metrics["brier_model"] <= 1, "brier_model must be in [0, 1]"
    assert metrics["residual_mse"] >= 0, "residual_mse must be non-negative"
    assert metrics["residual_rmse"] >= 0, "residual_rmse must be non-negative"
    assert metrics["residual_mae"] >= 0, "residual_mae must be non-negative"

    # Check improvement if requested
    if check_improvement:
        assert metrics["brier_improvement_pct"] > 0, "Expected positive Brier improvement"


def assert_no_data_leakage(train_dates: list, val_dates: list) -> None:
    """
    Assert no temporal data leakage between train and validation sets.

    Args:
        train_dates: List of training dates
        val_dates: List of validation dates

    Raises:
        AssertionError: If validation dates overlap with training
    """
    max_train_date = max(train_dates)
    min_val_date = min(val_dates)

    assert min_val_date > max_train_date, (
        f"Data leakage detected: min_val_date ({min_val_date}) <= max_train_date ({max_train_date})"
    )


def assert_bucket_distribution(
    df: pl.DataFrame,
    expected_near_pct: float = 33.0,
    expected_mid_pct: float = 33.0,
    expected_far_pct: float = 34.0,
    tolerance: float = 5.0,
) -> None:
    """
    Assert bucket distribution is within tolerance.

    Args:
        df: DataFrame with time_remaining column
        expected_near_pct: Expected % in near bucket
        expected_mid_pct: Expected % in mid bucket
        expected_far_pct: Expected % in far bucket
        tolerance: Allowed deviation in percentage points

    Raises:
        AssertionError: If distribution outside tolerance
    """
    total = len(df)

    near_count = len(df.filter(pl.col("time_remaining") < 300))
    mid_count = len(df.filter((pl.col("time_remaining") >= 300) & (pl.col("time_remaining") < 600)))
    far_count = len(df.filter(pl.col("time_remaining") >= 600))

    near_pct = (near_count / total) * 100
    mid_pct = (mid_count / total) * 100
    far_pct = (far_count / total) * 100

    assert abs(near_pct - expected_near_pct) <= tolerance, (
        f"Near bucket: {near_pct:.1f}% != {expected_near_pct:.1f}% (tolerance: {tolerance}pp)"
    )
    assert abs(mid_pct - expected_mid_pct) <= tolerance, (
        f"Mid bucket: {mid_pct:.1f}% != {expected_mid_pct:.1f}% (tolerance: {tolerance}pp)"
    )
    assert abs(far_pct - expected_far_pct) <= tolerance, (
        f"Far bucket: {far_pct:.1f}% != {expected_far_pct:.1f}% (tolerance: {tolerance}pp)"
    )


# ============================================================================
# PYTEST CONFIGURATION
# ============================================================================


def pytest_configure(config):
    """Register custom markers."""
    config.addinivalue_line("markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')")
    config.addinivalue_line("markers", "integration: marks tests as integration tests")
    config.addinivalue_line("markers", "unit: marks tests as unit tests")
    config.addinivalue_line("markers", "v4: marks tests specific to V4 pipeline")


# ============================================================================
# V4 ASSERTION HELPERS
# ============================================================================


def assert_v4_schema(df: pl.DataFrame, require_all_features: bool = True) -> None:
    """
    Assert DataFrame has valid V4 schema (156 features + regime columns).

    Args:
        df: DataFrame to validate
        require_all_features: If True, require all 156 features present

    Raises:
        AssertionError: If schema invalid
    """
    import sys

    sys.path.insert(0, str(Path(__file__).parent / "fixtures"))
    from synthetic_v4_data_generator import FEATURE_COLS_V4  # type: ignore

    # Check required columns
    required_cols = ["timestamp_seconds", "outcome", "prob_mid", "residual", "combined_regime", "date"]
    for col in required_cols:
        assert col in df.columns, f"Missing required column: {col}"

    # Check features
    if require_all_features:
        missing_features = set(FEATURE_COLS_V4) - set(df.columns)
        assert not missing_features, f"Missing {len(missing_features)} features: {sorted(missing_features)[:10]}..."
    else:
        # At least some features present
        features_present = [f for f in FEATURE_COLS_V4 if f in df.columns]
        assert len(features_present) > 0, "No V4 features found in DataFrame"


def assert_no_duplicates(df: pl.DataFrame, key_cols: list[str]) -> None:
    """
    Assert no duplicate rows based on key columns.

    Args:
        df: DataFrame to check
        key_cols: Columns that should be unique together

    Raises:
        AssertionError: If duplicates found
    """
    n_total = len(df)
    n_unique = len(df.select(key_cols).unique())

    assert n_total == n_unique, f"Found {n_total - n_unique} duplicate rows based on {key_cols}"


def assert_regime_distribution(
    df: pl.DataFrame,
    expected_regimes: Optional[list[str]] = None,
    min_samples_per_regime: int = 100,
) -> None:
    """
    Assert regime distribution is valid.

    Args:
        df: DataFrame with combined_regime column
        expected_regimes: List of expected regimes (default: all 12)
        min_samples_per_regime: Minimum samples required per regime

    Raises:
        AssertionError: If distribution invalid
    """
    if expected_regimes is None:
        import sys

        sys.path.insert(0, str(Path(__file__).parent / "fixtures"))
        from synthetic_v4_data_generator import COMBINED_REGIMES  # type: ignore

        expected_regimes = COMBINED_REGIMES

    assert "combined_regime" in df.columns, "Missing combined_regime column"

    regime_counts = df["combined_regime"].value_counts()
    present_regimes = set(regime_counts["combined_regime"].to_list())

    # Handle expected_regimes being None after conditional assignment
    regimes_list = expected_regimes if expected_regimes is not None else []
    expected_set = set(regimes_list)

    missing_regimes = expected_set - present_regimes
    if missing_regimes:
        raise AssertionError(f"Missing regimes: {sorted(missing_regimes)}")

    # Check minimum samples
    for regime in regimes_list:
        count = (
            regime_counts.filter(pl.col("combined_regime") == regime)["count"][0] if regime in present_regimes else 0
        )
        assert count >= min_samples_per_regime, (
            f"Regime {regime} has only {count} samples (min: {min_samples_per_regime})"
        )


def assert_no_nan_in_features(df: pl.DataFrame, feature_cols: list[str]) -> None:
    """
    Assert no NaN values in specified feature columns.

    Args:
        df: DataFrame to check
        feature_cols: List of feature columns to check

    Raises:
        AssertionError: If NaN found
    """
    for col in feature_cols:
        if col not in df.columns:
            continue
        n_nulls = df[col].null_count()
        assert n_nulls == 0, f"Found {n_nulls} NaN values in {col}"


def assert_feature_types(df: pl.DataFrame, expected_type: type[pl.DataType] = pl.Float64) -> None:
    """
    Assert feature columns have correct data types.

    Args:
        df: DataFrame to check
        expected_type: Expected Polars data type for features

    Raises:
        AssertionError: If types incorrect
    """
    import sys

    sys.path.insert(0, str(Path(__file__).parent / "fixtures"))
    from synthetic_v4_data_generator import FEATURE_COLS_V4  # type: ignore

    wrong_types = []
    for col in FEATURE_COLS_V4:
        if col not in df.columns:
            continue
        if df.schema[col] != expected_type:
            wrong_types.append(f"{col}: {df.schema[col]} != {expected_type}")

    assert not wrong_types, f"Found {len(wrong_types)} features with wrong types:\n" + "\n".join(wrong_types[:10])


def assert_temporal_boundaries(df: pl.DataFrame, bucket: str, expected_min: float, expected_max: float) -> None:
    """
    Assert temporal regime boundaries are correct.

    Args:
        df: DataFrame with time_remaining column
        bucket: Temporal bucket name (near/mid/far)
        expected_min: Minimum time_remaining for bucket
        expected_max: Maximum time_remaining for bucket

    Raises:
        AssertionError: If boundaries violated
    """
    assert "time_remaining" in df.columns, "Missing time_remaining column"
    assert "temporal_regime" in df.columns, "Missing temporal_regime column"

    bucket_df = df.filter(pl.col("temporal_regime") == bucket)
    if len(bucket_df) == 0:
        return  # No samples in this bucket

    min_val = float(bucket_df["time_remaining"].min())  # type: ignore
    max_val = float(bucket_df["time_remaining"].max())  # type: ignore

    assert min_val >= expected_min, f"{bucket}: min time_remaining {min_val} < expected {expected_min}"
    assert max_val <= expected_max, f"{bucket}: max time_remaining {max_val} > expected {expected_max}"
