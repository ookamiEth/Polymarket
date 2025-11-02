"""
Pytest configuration and shared fixtures for multi-horizon pipeline tests.

This module provides:
- Shared test fixtures (paths, configs, test data)
- Helper assertion functions
- Test cleanup utilities
"""

import shutil
from pathlib import Path

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
