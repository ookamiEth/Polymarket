"""
Unit tests for metrics calculation functions.

Tests the core performance metrics used to evaluate multi-horizon models:
- Brier score calculation
- Brier improvement percentage
- Residual metrics (MSE, RMSE, MAE)
- Probability clipping
- Calibration error
"""

import sys
from pathlib import Path

import numpy as np
import pytest

# Add parent directory to path to import modules
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from evaluate_multi_horizon import calculate_metrics  # noqa: E402


@pytest.mark.unit
class TestBrierScoreCalculation:
    """Test Brier score calculation correctness."""

    def test_brier_baseline_calculation(self):
        """Test baseline Brier score from prob_mid."""
        # Perfect baseline (prob_mid = outcomes)
        outcomes = np.array([0.0, 1.0, 0.0, 1.0, 0.0])
        prob_baseline = np.array([0.0, 1.0, 0.0, 1.0, 0.0])
        predictions = np.zeros(5)

        metrics = calculate_metrics(predictions, outcomes, prob_baseline, "test")

        # Perfect baseline should have 0 Brier score
        assert np.isclose(metrics["brier_baseline"], 0.0), f"Expected 0.0, got {metrics['brier_baseline']}"

    def test_brier_model_calculation(self):
        """Test model Brier score calculation."""
        # Constant baseline, perfect model predictions
        outcomes = np.array([0.0, 1.0, 0.0, 1.0, 0.0])
        prob_baseline = np.array([0.5, 0.5, 0.5, 0.5, 0.5])
        predictions = np.array([-0.5, 0.5, -0.5, 0.5, -0.5])  # Corrects to outcomes

        metrics = calculate_metrics(predictions, outcomes, prob_baseline, "test")

        # Perfect model should have 0 Brier score
        assert np.isclose(metrics["brier_model"], 0.0), f"Expected 0.0, got {metrics['brier_model']}"

    def test_brier_improvement_positive(self):
        """Test that improvement is positive when model beats baseline."""
        outcomes = np.array([0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0])
        prob_baseline = np.array([0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5])
        predictions = np.array([-0.1, 0.1, -0.1, 0.1, -0.1, 0.1, -0.1, 0.1])  # Moves toward outcomes

        metrics = calculate_metrics(predictions, outcomes, prob_baseline, "test")

        # Model should improve over baseline
        assert metrics["brier_improvement_pct"] > 0, "Model should improve over baseline"

    def test_brier_improvement_negative(self):
        """Test that improvement is negative when model worse than baseline."""
        outcomes = np.array([0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0])
        prob_baseline = np.array([0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5])
        predictions = np.array([0.1, -0.1, 0.1, -0.1, 0.1, -0.1, 0.1, -0.1])  # Moves away from outcomes

        metrics = calculate_metrics(predictions, outcomes, prob_baseline, "test")

        # Model should be worse than baseline
        assert metrics["brier_improvement_pct"] < 0, "Model should be worse than baseline"


@pytest.mark.unit
class TestResidualMetrics:
    """Test residual metrics (MSE, RMSE, MAE)."""

    def test_residual_mse_zero_for_perfect_predictions(self):
        """Test that perfect residual predictions give MSE=0."""
        outcomes = np.array([0.0, 1.0, 0.0, 1.0, 0.0])
        prob_baseline = np.array([0.3, 0.7, 0.4, 0.6, 0.5])

        # Perfect residual predictions
        residual_true = outcomes - prob_baseline
        predictions = residual_true  # Exact match

        metrics = calculate_metrics(predictions, outcomes, prob_baseline, "test")

        assert np.isclose(metrics["residual_mse"], 0.0), f"Expected MSE=0, got {metrics['residual_mse']}"
        assert np.isclose(metrics["residual_rmse"], 0.0), f"Expected RMSE=0, got {metrics['residual_rmse']}"
        assert np.isclose(metrics["residual_mae"], 0.0), f"Expected MAE=0, got {metrics['residual_mae']}"

    def test_residual_rmse_is_sqrt_of_mse(self):
        """Test that RMSE = sqrt(MSE)."""
        outcomes = np.array([0.0, 1.0, 0.0, 1.0, 0.0])
        prob_baseline = np.array([0.5, 0.5, 0.5, 0.5, 0.5])
        predictions = np.array([-0.2, 0.3, -0.1, 0.2, -0.15])

        metrics = calculate_metrics(predictions, outcomes, prob_baseline, "test")

        expected_rmse = np.sqrt(metrics["residual_mse"])
        assert np.isclose(metrics["residual_rmse"], expected_rmse), (
            f"RMSE should be sqrt(MSE): {metrics['residual_rmse']} != sqrt({metrics['residual_mse']})"
        )

    def test_residual_mae_calculation(self):
        """Test MAE calculation correctness."""
        outcomes = np.array([0.0, 1.0, 0.0, 1.0])
        prob_baseline = np.array([0.5, 0.5, 0.5, 0.5])
        predictions = np.array([0.0, 0.0, 0.0, 0.0])  # No correction

        # Residual true: [-0.5, 0.5, -0.5, 0.5]
        # Residual pred: [0, 0, 0, 0]
        # Error: [0.5, 0.5, 0.5, 0.5]
        # MAE: 0.5

        metrics = calculate_metrics(predictions, outcomes, prob_baseline, "test")

        assert np.isclose(metrics["residual_mae"], 0.5), f"Expected MAE=0.5, got {metrics['residual_mae']}"


@pytest.mark.unit
class TestProbabilityClipping:
    """Test that probabilities are clipped to [0, 1]."""

    def test_prob_clipping_below_zero(self):
        """Test that probabilities < 0 are clipped to 0."""
        outcomes = np.array([0.0, 0.0, 0.0])
        prob_baseline = np.array([0.1, 0.2, 0.3])
        predictions = np.array([-0.5, -0.8, -1.0])  # Would result in negative probabilities

        metrics = calculate_metrics(predictions, outcomes, prob_baseline, "test")

        # Model Brier should be calculated with clipped probabilities
        # prob_pred = [0.0, 0.0, 0.0] after clipping
        # Brier = mean((outcomes - prob_pred)^2) = mean([0, 0, 0]) = 0
        assert np.isclose(metrics["brier_model"], 0.0), f"Expected clipped Brier=0.0, got {metrics['brier_model']}"

    def test_prob_clipping_above_one(self):
        """Test that probabilities > 1 are clipped to 1."""
        outcomes = np.array([1.0, 1.0, 1.0])
        prob_baseline = np.array([0.7, 0.8, 0.9])
        predictions = np.array([0.5, 0.6, 0.7])  # Would result in probabilities > 1

        metrics = calculate_metrics(predictions, outcomes, prob_baseline, "test")

        # prob_pred = [1.0, 1.0, 1.0] after clipping
        # Brier = mean((outcomes - prob_pred)^2) = mean([0, 0, 0]) = 0
        assert np.isclose(metrics["brier_model"], 0.0), f"Expected clipped Brier=0.0, got {metrics['brier_model']}"


@pytest.mark.unit
class TestEdgeCases:
    """Test edge cases for metrics calculation."""

    def test_perfect_baseline(self):
        """Test metrics when baseline is already perfect."""
        outcomes = np.array([0.0, 1.0, 0.0, 1.0, 0.0])
        prob_baseline = outcomes.copy()  # Perfect baseline
        predictions = np.array([0.1, -0.1, 0.1, -0.1, 0.1])  # Worse than baseline

        metrics = calculate_metrics(predictions, outcomes, prob_baseline, "test")

        # Baseline Brier should be 0
        assert np.isclose(metrics["brier_baseline"], 0.0)

        # Model should be worse (negative improvement)
        assert metrics["brier_improvement_pct"] < 0

    def test_random_baseline(self):
        """Test metrics with random 50/50 baseline."""
        np.random.seed(42)
        outcomes = np.random.randint(0, 2, 1000).astype(float)
        prob_baseline = np.full(1000, 0.5)
        predictions = np.zeros(1000)  # No improvement

        metrics = calculate_metrics(predictions, outcomes, prob_baseline, "test")

        # Baseline Brier should be ~0.25 for random 50/50
        assert 0.23 < metrics["brier_baseline"] < 0.27

        # Model Brier should match baseline (no improvement)
        assert np.isclose(metrics["brier_model"], metrics["brier_baseline"], atol=1e-6)

        # Improvement should be ~0%
        assert np.abs(metrics["brier_improvement_pct"]) < 0.01

    def test_known_improvement_percentage(self):
        """Test Brier improvement with known ground truth."""
        # Construct example with known improvement
        outcomes = np.array([0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0])
        prob_baseline = np.array([0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5])

        # Baseline Brier = mean((outcomes - 0.5)^2) = mean([0.25, 0.25, ...]) = 0.25

        # Move predictions halfway to outcomes
        predictions = np.array([-0.25, 0.25, -0.25, 0.25, -0.25, 0.25, -0.25, 0.25])
        # prob_pred = [0.25, 0.75, 0.25, 0.75, ...]
        # Model Brier = mean([0.0625, 0.0625, ...]) = 0.0625

        # Improvement = (0.25 - 0.0625) / 0.25 * 100 = 75%

        metrics = calculate_metrics(predictions, outcomes, prob_baseline, "test")

        assert np.isclose(metrics["brier_baseline"], 0.25), f"Expected baseline=0.25, got {metrics['brier_baseline']}"
        assert np.isclose(metrics["brier_model"], 0.0625), f"Expected model=0.0625, got {metrics['brier_model']}"
        assert np.isclose(metrics["brier_improvement_pct"], 75.0), (
            f"Expected 75% improvement, got {metrics['brier_improvement_pct']:.2f}%"
        )


@pytest.mark.unit
class TestCalibrationError:
    """Test calibration error calculation."""

    def test_calibration_error_perfect_calibration(self):
        """Test that perfect calibration gives low error."""
        np.random.seed(42)

        # Create perfectly calibrated predictions
        n_samples = 10000
        prob_pred_values = [0.1, 0.3, 0.5, 0.7, 0.9]
        outcomes_list = []
        prob_baseline_list = []
        predictions_list = []

        for prob in prob_pred_values:
            # Generate outcomes with true probability = prob
            n_in_bin = n_samples // len(prob_pred_values)
            outcomes_bin = (np.random.rand(n_in_bin) < prob).astype(float)
            outcomes_list.append(outcomes_bin)

            # Set baseline to 0.5, predictions correct to prob
            prob_baseline_list.append(np.full(n_in_bin, 0.5))
            predictions_list.append(np.full(n_in_bin, prob - 0.5))

        outcomes = np.concatenate(outcomes_list)
        prob_baseline = np.concatenate(prob_baseline_list)
        predictions = np.concatenate(predictions_list)

        metrics = calculate_metrics(predictions, outcomes, prob_baseline, "test")

        # Perfect calibration should have low error (< 0.05)
        # Note: Some error expected due to random sampling
        assert metrics["calibration_error"] < 0.05, (
            f"Expected calibration_error < 0.05, got {metrics['calibration_error']}"
        )

    def test_calibration_error_miscalibrated(self):
        """Test that miscalibrated predictions give high error."""
        # All predictions 0.9, but outcomes are 50/50
        outcomes = np.array([0, 1] * 500)  # 50% positive
        prob_baseline = np.full(1000, 0.4)
        predictions = np.full(1000, 0.5)  # prob_pred = 0.9

        metrics = calculate_metrics(predictions, outcomes, prob_baseline, "test")

        # Miscalibration: predict 0.9, observe 0.5 -> large error
        assert metrics["calibration_error"] > 0.1, (
            f"Expected calibration_error > 0.1, got {metrics['calibration_error']}"
        )


@pytest.mark.unit
class TestMetricsDictionary:
    """Test that metrics dictionary contains all required fields."""

    def test_all_fields_present(self):
        """Test that all expected fields are in metrics dictionary."""
        outcomes = np.array([0.0, 1.0, 0.0, 1.0, 0.0])
        prob_baseline = np.array([0.5, 0.5, 0.5, 0.5, 0.5])
        predictions = np.array([-0.1, 0.1, -0.1, 0.1, -0.1])

        metrics = calculate_metrics(predictions, outcomes, prob_baseline, "test_label")

        required_fields = [
            "label",
            "n_samples",
            "brier_baseline",
            "brier_model",
            "brier_improvement_pct",
            "residual_mse",
            "residual_rmse",
            "residual_mae",
            "calibration_error",
        ]

        for field in required_fields:
            assert field in metrics, f"Missing required field: {field}"

    def test_n_samples_correct(self):
        """Test that n_samples matches input length."""
        outcomes = np.array([0.0, 1.0, 0.0, 1.0, 0.0])
        prob_baseline = np.array([0.5, 0.5, 0.5, 0.5, 0.5])
        predictions = np.array([-0.1, 0.1, -0.1, 0.1, -0.1])

        metrics = calculate_metrics(predictions, outcomes, prob_baseline, "test")

        assert metrics["n_samples"] == 5, f"Expected n_samples=5, got {metrics['n_samples']}"

    def test_label_preserved(self):
        """Test that label is preserved in metrics."""
        outcomes = np.array([0.0, 1.0])
        prob_baseline = np.array([0.5, 0.5])
        predictions = np.array([0.0, 0.0])

        metrics = calculate_metrics(predictions, outcomes, prob_baseline, "my_test_label")

        assert metrics["label"] == "my_test_label", f"Expected label='my_test_label', got '{metrics['label']}'"


@pytest.mark.unit
def test_metrics_with_synthetic_fixture(synthetic_predictions_df):
    """Test metrics calculation with synthetic fixture from conftest."""
    from tests.conftest import assert_metrics_valid

    # Extract arrays from DataFrame
    outcomes = synthetic_predictions_df["outcome"].to_numpy()
    prob_baseline = synthetic_predictions_df["prob_mid"].to_numpy()
    predictions = synthetic_predictions_df["prediction"].to_numpy()

    metrics = calculate_metrics(predictions, outcomes, prob_baseline, "synthetic_test")

    # Validate metrics using helper
    assert_metrics_valid(metrics, check_improvement=False)  # Don't require positive improvement
