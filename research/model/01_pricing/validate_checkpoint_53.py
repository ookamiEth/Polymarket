#!/usr/bin/env python3
"""
Checkpoint 5.3 Validator: Training Output
=========================================

Validates trained models after train_multi_horizon_v4.py completes.

Checks:
- 8 model files exist (.txt format)
- 8 config files exist (.yaml format)
- Reasonable MSE (0.01-0.10 range)
- Reasonable Brier (0.10-0.25 range)
- Walk-forward metrics present (13 windows expected)
- IC statistics present
- Training time reasonable

Runtime: ~1 minute

Author: BT Research Team
Date: 2025-11-13
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import lightgbm as lgb
import yaml

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

# Paths
MODEL_DIR = Path(__file__).parent.parent
MODELS_DIR = MODEL_DIR / "results" / "models_v4"

# Expected models
EXPECTED_MODELS = [
    "near_low_vol_atm",
    "near_low_vol_otm",
    "near_high_vol_atm",
    "near_high_vol_otm",
    "mid_low_vol_atm",
    "mid_low_vol_otm",
    "mid_high_vol_atm",
    "mid_high_vol_otm",
]


class ValidationResult:
    """Store validation result."""

    def __init__(self, check: str, passed: bool, message: str = ""):
        self.check = check
        self.passed = passed
        self.message = message

    def __repr__(self) -> str:
        status = "✅" if self.passed else "❌"
        return f"{status} {self.check}: {self.message}"


def validate_files_exist() -> list[ValidationResult]:
    """Check that all model and config files exist."""
    logger.info("\n" + "=" * 80)
    logger.info("CHECK 1: File Existence")
    logger.info("=" * 80)

    results = []

    if not MODELS_DIR.exists():
        results.append(ValidationResult("Models directory", False, f"Directory not found: {MODELS_DIR}"))
        return results

    results.append(ValidationResult("Models directory", True, str(MODELS_DIR)))
    logger.info(f"Models directory found: {MODELS_DIR}")

    for model_name in EXPECTED_MODELS:
        model_file = MODELS_DIR / f"lightgbm_{model_name}.txt"
        config_file = MODELS_DIR / f"config_{model_name}.yaml"

        # Check model file
        if model_file.exists():
            size_mb = model_file.stat().st_size / (1024**2)
            results.append(ValidationResult(f"Model: {model_name}", True, f"{size_mb:.2f}MB"))
            logger.info(f"  ✓ {model_name}: model file ({size_mb:.2f}MB)")
        else:
            results.append(ValidationResult(f"Model: {model_name}", False, "Model file missing"))
            logger.error(f"  ✗ {model_name}: model file missing")

        # Check config file
        if config_file.exists():
            size_kb = config_file.stat().st_size / 1024
            results.append(ValidationResult(f"Config: {model_name}", True, f"{size_kb:.1f}KB"))
            logger.info(f"  ✓ {model_name}: config file ({size_kb:.1f}KB)")
        else:
            results.append(ValidationResult(f"Config: {model_name}", False, "Config file missing"))
            logger.error(f"  ✗ {model_name}: config file missing")

    return results


def validate_model_metrics() -> list[ValidationResult]:
    """Validate model performance metrics."""
    logger.info("\n" + "=" * 80)
    logger.info("CHECK 2: Model Metrics")
    logger.info("=" * 80)

    results = []

    for model_name in EXPECTED_MODELS:
        config_file = MODELS_DIR / f"config_{model_name}.yaml"

        if not config_file.exists():
            results.append(ValidationResult(f"Metrics: {model_name}", False, "Config file missing"))
            continue

        try:
            with open(config_file) as f:
                config = yaml.safe_load(f)

            perf = config.get("performance", {})
            wf = config.get("walk_forward", {})

            residual_mse = perf.get("residual_mse", None)
            baseline_brier = perf.get("baseline_brier", None)
            training_time = perf.get("training_time_minutes", None)
            n_windows = wf.get("n_windows", None)
            mean_ic = wf.get("mean_ic", None)

            logger.info(f"\n{model_name}:")
            logger.info(f"  Residual MSE: {residual_mse:.6f}" if residual_mse else "  Residual MSE: N/A")
            logger.info(f"  Baseline Brier: {baseline_brier:.6f}" if baseline_brier else "  Baseline Brier: N/A")
            logger.info(f"  Walk-forward windows: {n_windows}" if n_windows else "  Walk-forward windows: N/A")
            logger.info(f"  Mean IC: {mean_ic:.4f}" if mean_ic else "  Mean IC: N/A")
            logger.info(f"  Training time: {training_time:.1f} min" if training_time else "  Training time: N/A")

            # Validate MSE range
            if residual_mse is not None and 0.01 <= residual_mse <= 0.10:
                results.append(ValidationResult(f"MSE: {model_name}", True, f"{residual_mse:.6f}"))
            elif residual_mse is not None:
                results.append(
                    ValidationResult(
                        f"MSE: {model_name}",
                        False,
                        f"{residual_mse:.6f} (expected 0.01-0.10)",
                    )
                )
            else:
                results.append(ValidationResult(f"MSE: {model_name}", False, "MSE missing from config"))

            # Validate Brier range
            if baseline_brier is not None and 0.10 <= baseline_brier <= 0.25:
                results.append(ValidationResult(f"Brier: {model_name}", True, f"{baseline_brier:.6f}"))
            elif baseline_brier is not None:
                results.append(
                    ValidationResult(
                        f"Brier: {model_name}",
                        False,
                        f"{baseline_brier:.6f} (expected 0.10-0.25)",
                    )
                )
            else:
                results.append(ValidationResult(f"Brier: {model_name}", False, "Brier missing from config"))

            # Validate walk-forward windows
            if n_windows is not None and n_windows >= 10:
                results.append(ValidationResult(f"Windows: {model_name}", True, f"{n_windows} windows"))
            elif n_windows is not None:
                results.append(ValidationResult(f"Windows: {model_name}", False, f"{n_windows} windows (expected ≥10)"))
            else:
                results.append(ValidationResult(f"Windows: {model_name}", False, "Window count missing"))

            # Check IC (optional - can be negative for poor models)
            if mean_ic is not None:
                results.append(ValidationResult(f"IC: {model_name}", True, f"{mean_ic:.4f}"))
            else:
                results.append(ValidationResult(f"IC: {model_name}", False, "IC missing"))

        except Exception as e:
            results.append(ValidationResult(f"Metrics: {model_name}", False, f"Error: {e}"))
            logger.error(f"  ✗ Error reading config: {e}")

    return results


def validate_model_loadable() -> list[ValidationResult]:
    """Validate that models can be loaded."""
    logger.info("\n" + "=" * 80)
    logger.info("CHECK 3: Model Loadability")
    logger.info("=" * 80)

    results = []

    for model_name in EXPECTED_MODELS:
        model_file = MODELS_DIR / f"lightgbm_{model_name}.txt"

        if not model_file.exists():
            results.append(ValidationResult(f"Load: {model_name}", False, "Model file missing"))
            continue

        try:
            model = lgb.Booster(model_file=str(model_file))

            # Check basic model properties
            n_features = model.num_feature()
            n_trees = model.num_trees()

            logger.info(f"  ✓ {model_name}: {n_trees} trees, {n_features} features")
            results.append(ValidationResult(f"Load: {model_name}", True, f"{n_trees} trees, {n_features} features"))

        except Exception as e:
            results.append(ValidationResult(f"Load: {model_name}", False, f"Error: {e}"))
            logger.error(f"  ✗ {model_name}: Error loading model: {e}")

    return results


def main() -> None:
    """Run all checkpoint 5.3 validations."""
    logger.info("\n" + "=" * 80)
    logger.info("CHECKPOINT 5.3: Training Output Validation")
    logger.info("=" * 80)
    logger.info(f"Validating models in: {MODELS_DIR}")
    logger.info("=" * 80)

    results: list[ValidationResult] = []

    # Run all checks
    results.extend(validate_files_exist())
    results.extend(validate_model_metrics())
    results.extend(validate_model_loadable())

    # Summary
    logger.info("\n" + "=" * 80)
    logger.info("SUMMARY")
    logger.info("=" * 80)

    # Group results by type for better readability
    file_results = [r for r in results if "file" in r.check.lower() or "directory" in r.check.lower() or ":" in r.check]
    metric_results = [
        r for r in results if any(x in r.check.lower() for x in ["mse", "brier", "windows", "ic", "metrics"])
    ]
    load_results = [r for r in results if "load" in r.check.lower()]

    if file_results:
        logger.info("\nFile Existence:")
        for result in file_results[:10]:  # Show first 10
            logger.info(f"  {result}")

    if metric_results:
        logger.info("\nModel Metrics:")
        for result in metric_results:
            logger.info(f"  {result}")

    if load_results:
        logger.info("\nModel Loadability:")
        for result in load_results:
            logger.info(f"  {result}")

    passed = sum(1 for r in results if r.passed)
    total = len(results)

    logger.info("\n" + "=" * 80)
    logger.info(f"TOTAL: {passed}/{total} checks passed")

    if passed == total:
        logger.info("✅ CHECKPOINT 5.3 PASSED - All 8 models validated")
        logger.info("=" * 80)
        logger.info("\n🎉 V4 PIPELINE COMPLETE!")
        logger.info("\nNext steps:")
        logger.info("  1. Run evaluate_multi_horizon_v4.py (if exists)")
        logger.info("  2. Review model configs for IC and MSE improvements")
        logger.info("  3. Consider Optuna hyperparameter tuning")
        sys.exit(0)
    else:
        logger.error(f"❌ {total - passed} CHECK(S) FAILED")
        logger.info("=" * 80)
        logger.error("\nFix issues before deploying models!")
        sys.exit(1)


if __name__ == "__main__":
    main()
