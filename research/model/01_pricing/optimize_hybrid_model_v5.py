#!/usr/bin/env python3
"""
Multi-Horizon + Regime Hybrid Model Optimization (V4)
=======================================================

Per-model Bayesian hyperparameter optimization using Optuna with walk-forward validation:
- Optimize 8-12 hybrid models individually
- Regime-specific search spaces (ATM vs OTM, low vol vs high vol)
- Walk-forward objective: Mean validation MSE across 13 folds
- Priority-based trial counts (100/50/25 trials)

This is Phase 2 (optional) of the V4 implementation.

Usage:
  # Optimize specific model (high priority: 100 trials)
  uv run python optimize_hybrid_model_v4.py --model near_low_vol_atm --n_trials 100

  # Optimize medium priority model (50 trials)
  uv run python optimize_hybrid_model_v4.py --model mid_high_vol_atm --n_trials 50

  # Resume from existing study
  uv run python optimize_hybrid_model_v4.py --model near_low_vol_atm --resume

Author: BT Research Team
Date: 2025-11-14
"""

from __future__ import annotations

import argparse
import gc
import logging
import sys
import time
from pathlib import Path
from typing import Any

import lightgbm as lgb
import numpy as np
import optuna

# W&B integration (optional)
try:
    import wandb
    from wandb.integration.optuna import WandbCallback  # type: ignore[import-not-found]

    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    wandb = None  # type: ignore[assignment]
    WandbCallback = None  # type: ignore[assignment,misc]
import polars as pl
import yaml

# Import from V4 training module
from train_multi_horizon_v4 import FEATURE_COLS_V4

# Import from V3 legacy
_v3_core_path = str(Path(__file__).parent / "v3_legacy" / "core")
if _v3_core_path not in sys.path:
    sys.path.insert(0, _v3_core_path)

# ruff: noqa: E402
from lightgbm_memory_optimized import (  # type: ignore[import-not-found]
    MemoryMonitor,
    create_lgb_dataset_from_parquet,
)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


def load_config(config_path: Path) -> dict[str, Any]:
    """Load multi-horizon + regime hybrid configuration."""
    logger.info(f"Loading configuration from {config_path}")
    with open(config_path) as f:
        config = yaml.safe_load(f)
    return config


def get_hybrid_search_space(
    bucket: str,
    regime: str,
    config: dict[str, Any],
) -> dict[str, tuple]:
    """
    Get Optuna search space for hybrid model with regime-specific adjustments.

    Args:
        bucket: Temporal bucket (near/mid/far)
        regime: Volatility regime (low_vol_atm, low_vol_otm, high_vol_atm, high_vol_otm)
        config: Multi-horizon + regime configuration

    Returns:
        Dictionary of parameter ranges
    """
    # Base search space from bucket
    if "optuna" not in config or "search_spaces" not in config["optuna"]:
        raise ValueError("Optuna search spaces not defined in config")

    base_space = config["optuna"]["search_spaces"][bucket].copy()

    # Regime-specific adjustments
    if "atm" in regime:
        # ATM regimes: Tighter regularization (less overfitting at the money)
        base_space["lambda_l2"] = (base_space["lambda_l2"][0] * 1.5, base_space["lambda_l2"][1] * 1.5)
        logger.info("  ATM adjustment: L2 regularization increased by 1.5x")

    if "high_vol" in regime:
        # High vol regimes: More regularization, fewer leaves (reduce overfitting in noisy conditions)
        base_space["num_leaves"] = (base_space["num_leaves"][0], base_space["num_leaves"][1] // 2)
        base_space["lambda_l2"] = (base_space["lambda_l2"][0] * 1.2, base_space["lambda_l2"][1] * 1.2)
        logger.info("  High vol adjustment: num_leaves halved, L2 increased by 1.2x")

    logger.info(f"\nSearch space for {bucket}_{regime}:")
    logger.info(f"  Learning rate:      {base_space['learning_rate']}")
    logger.info(f"  Num leaves:         {base_space['num_leaves']}")
    logger.info(f"  Min data in leaf:   {base_space['min_data_in_leaf']}")
    logger.info(f"  L2 regularization:  {base_space['lambda_l2']}")
    logger.info(f"  Max depth:          {base_space['max_depth']}")
    logger.info(f"  L1 regularization:  {base_space['lambda_l1']}")

    return base_space


def generate_walk_forward_windows(config: dict[str, Any], data_file: Path) -> list[tuple]:
    """
    Generate walk-forward validation windows for optimization.

    Args:
        config: Multi-horizon + regime configuration
        data_file: Input data file (model-specific stratified file)

    Returns:
        List of (train_start, train_end, val_start, val_end) tuples
    """

    from dateutil.relativedelta import relativedelta

    walk_forward_config = config["walk_forward_validation"]

    # Get date range from data
    df_lazy = pl.scan_parquet(data_file)
    date_stats = df_lazy.select(
        [pl.col("date").min().alias("min_date"), pl.col("date").max().alias("max_date")]
    ).collect()

    min_date = date_stats["min_date"][0]
    max_date = date_stats["max_date"][0]

    # Calculate holdout
    holdout_months = walk_forward_config["holdout_months"]
    holdout_start = max_date.replace(day=1) - relativedelta(months=holdout_months - 1)
    walk_forward_end = holdout_start - relativedelta(days=1)

    # Generate windows
    train_months = walk_forward_config["train_months"]
    val_months = walk_forward_config["val_months"]
    step_months = walk_forward_config["step_months"]
    embargo_days = walk_forward_config.get("embargo_days", 0)

    windows = []
    current_start = min_date

    while True:
        train_end = current_start + relativedelta(months=train_months) - relativedelta(days=1)

        # Add embargo period
        if embargo_days > 0:
            embargo_start = train_end + relativedelta(days=1)
            embargo_end = embargo_start + relativedelta(days=embargo_days - 1)
            val_start = embargo_end + relativedelta(days=1)
        else:
            val_start = train_end + relativedelta(days=1)

        val_end = val_start + relativedelta(months=val_months) - relativedelta(days=1)

        if val_end >= holdout_start:
            break

        windows.append((current_start, train_end, val_start, val_end))
        current_start += relativedelta(months=step_months)

    logger.info(f"Generated {len(windows)} walk-forward windows")
    logger.info(f"  Walk-forward range: {min_date} to {walk_forward_end}")
    logger.info(f"  Holdout test: {holdout_start} to {max_date}")

    return windows


class V4OptunaObjective:
    """Optuna objective function for V4 hybrid model with walk-forward validation."""

    def __init__(
        self,
        model_name: str,
        search_space: dict[str, Any],
        data_file: Path,
        features: list[str],
        fixed_params: dict[str, Any],
        windows: list[tuple],
    ):
        self.model_name = model_name
        self.search_space = search_space
        self.data_file = data_file
        self.features = features
        self.fixed_params = fixed_params
        self.windows = windows

        # Load full dataset lazily
        self.df_lazy = pl.scan_parquet(data_file)

        logger.info(f"\nOptimization objective initialized for {model_name}")
        logger.info(f"  Data file: {data_file}")
        logger.info(f"  Features: {len(features)}")
        logger.info(f"  Windows: {len(windows)}")

    def __call__(self, trial: optuna.Trial) -> float:
        """
        Objective function for a single Optuna trial with walk-forward validation.

        Args:
            trial: Optuna trial object

        Returns:
            Mean validation residual MSE across all windows (lower is better)
        """
        # Sample hyperparameters from search space
        params = self.fixed_params.copy()

        params["learning_rate"] = trial.suggest_float(
            "learning_rate",
            self.search_space["learning_rate"][0],
            self.search_space["learning_rate"][1],
            log=True,
        )

        params["num_leaves"] = trial.suggest_int(
            "num_leaves",
            self.search_space["num_leaves"][0],
            self.search_space["num_leaves"][1],
        )

        params["min_data_in_leaf"] = trial.suggest_int(
            "min_data_in_leaf",
            self.search_space["min_data_in_leaf"][0],
            self.search_space["min_data_in_leaf"][1],
        )

        params["lambda_l2"] = trial.suggest_float(
            "lambda_l2",
            self.search_space["lambda_l2"][0],
            self.search_space["lambda_l2"][1],
        )

        params["max_depth"] = trial.suggest_int(
            "max_depth",
            self.search_space["max_depth"][0],
            self.search_space["max_depth"][1],
        )

        params["lambda_l1"] = trial.suggest_float(
            "lambda_l1",
            self.search_space["lambda_l1"][0],
            self.search_space["lambda_l1"][1],
        )

        params["feature_fraction"] = trial.suggest_float("feature_fraction", 0.5, 1.0)

        # Walk-forward validation
        window_mses = []
        # Use tempfile for cross-platform compatibility
        import tempfile

        temp_base = Path(tempfile.gettempdir())
        output_dir = temp_base / f"optuna_v4_{self.model_name}"
        output_dir.mkdir(parents=True, exist_ok=True)

        for window_idx, (train_start, train_end, val_start, val_end) in enumerate(self.windows):
            # Create temporal splits
            df_train = self.df_lazy.filter((pl.col("date") >= train_start) & (pl.col("date") <= train_end))
            df_val = self.df_lazy.filter((pl.col("date") >= val_start) & (pl.col("date") <= val_end))

            # Check sample counts
            n_train = df_train.select(pl.len()).collect().item()
            n_val = df_val.select(pl.len()).collect().item()

            if n_train < 1000 or n_val < 100:
                continue  # Skip insufficient windows

            # Write temporary files
            temp_train = output_dir / f"trial{trial.number}_w{window_idx}_train.parquet"
            temp_val = output_dir / f"trial{trial.number}_w{window_idx}_val.parquet"

            df_train.sink_parquet(str(temp_train))
            df_val.sink_parquet(str(temp_val))

            try:
                # Create datasets
                train_data, _ = create_lgb_dataset_from_parquet(str(temp_train), self.features, free_raw_data=True)
                val_data, _ = create_lgb_dataset_from_parquet(
                    str(temp_val), self.features, reference=train_data, free_raw_data=True
                )

                # Train
                model = lgb.train(
                    params,
                    train_data,
                    num_boost_round=params.get("n_estimators", 1000),
                    valid_sets=[val_data],
                    valid_names=["val"],
                    callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)],  # Silent
                )

                # Get validation metric (RMSE) and convert to MSE
                best_score = model.best_score
                if "val" not in best_score or "rmse" not in best_score["val"]:
                    logger.warning(
                        f"  Trial {trial.number} window {window_idx}: "
                        f"Validation metric not found in best_score: {best_score}"
                    )
                    continue  # Skip this window

                val_rmse = best_score["val"]["rmse"]
                val_mse = val_rmse**2  # Convert RMSE to MSE

                window_mses.append(val_mse)

                # Clean up
                del train_data, val_data, model
                gc.collect()

            except Exception as e:
                logger.warning(f"  Trial {trial.number} window {window_idx} failed: {e}")
                continue

            finally:
                # Remove temporary files
                temp_train.unlink(missing_ok=True)
                temp_val.unlink(missing_ok=True)

        # Check if we got any valid results
        if len(window_mses) == 0:
            logger.warning(f"  Trial {trial.number}: No valid windows")
            return float("inf")  # Worst possible score

        # Return mean MSE across windows
        mean_mse = float(np.mean(window_mses))
        logger.info(
            f"  Trial {trial.number}: Mean MSE = {mean_mse:.6f} "
            f"(std={np.std(window_mses):.6f}, windows={len(window_mses)})"
        )

        return mean_mse


def optimize_model(
    model_name: str,
    config: dict[str, Any],
    n_trials: int,
    n_jobs: int,
    output_dir: Path,
    resume: bool = False,
) -> dict[str, Any]:
    """
    Optimize hyperparameters for a single hybrid model.

    Args:
        model_name: Model identifier (e.g., "near_low_vol_atm")
        config: Multi-horizon + regime configuration
        n_trials: Number of Optuna trials
        n_jobs: Number of parallel jobs
        output_dir: Output directory for optimized models
        resume: Whether to resume from existing study

    Returns:
        Dictionary with best parameters and metrics
    """
    logger.info("\n" + "=" * 80)
    logger.info(f"OPTIMIZING MODEL: {model_name}")
    logger.info("=" * 80)

    monitor = MemoryMonitor()

    # Get model config
    if model_name not in config["hybrid_models"]:
        raise ValueError(f"Model {model_name} not found in config")

    model_config = config["hybrid_models"][model_name]
    bucket = model_config["bucket"]
    regime = model_config["regime"]

    logger.info(f"  Bucket: {bucket}")
    logger.info(f"  Regime: {regime}")

    # Get search space
    search_space = get_hybrid_search_space(bucket, regime, config)

    # Get stratified data file
    model_dir = Path(__file__).resolve().parent.parent
    data_dir = model_dir / Path(config["training"]["output_dir"])
    data_file = data_dir / f"{model_name}_data.parquet"

    if not data_file.exists():
        raise FileNotFoundError(f"Stratified data not found: {data_file}. Run train_multi_horizon_v4.py first.")

    logger.info(f"\nData file: {data_file}")

    # Get features
    schema = pl.scan_parquet(data_file).collect_schema()
    features = [col for col in FEATURE_COLS_V4 if col in schema.names()]
    logger.info(f"Features: {len(features)}")

    # Generate walk-forward windows
    windows = generate_walk_forward_windows(config, data_file)

    # Fixed parameters
    fixed_params = config["training"]["base_params"].copy()

    # Create Optuna study
    study_name = f"v4_{model_name}"
    storage_file = output_dir / f"optuna_study_{model_name}.db"
    storage = f"sqlite:///{storage_file}"

    if resume and storage_file.exists():
        logger.info(f"\n✓ Resuming study from {storage_file}")
        study = optuna.load_study(study_name=study_name, storage=storage)
        logger.info(f"  Previous trials: {len(study.trials)}")
    else:
        logger.info(f"\n✓ Creating new study: {study_name}")
        study = optuna.create_study(
            study_name=study_name,
            storage=storage,
            direction="minimize",  # Minimize validation MSE
            load_if_exists=False,
        )

    # Create objective
    objective = V4OptunaObjective(
        model_name=model_name,
        search_space=search_space,
        data_file=data_file,
        features=features,
        fixed_params=fixed_params,
        windows=windows,
    )

    # Initialize W&B for Optuna tracking (if enabled)
    wandb_run = None
    wandb_callback = None
    if config.get("wandb", {}).get("enabled", False) and WANDB_AVAILABLE:
        try:
            wandb_run = wandb.init(  # type: ignore[union-attr]
                project=config["wandb"]["project"] + "-optuna",
                name=f"optimize_{model_name}",
                config={
                    "model": model_name,
                    "n_trials": n_trials,
                    "n_jobs": n_jobs,
                    "search_space": search_space,
                },
                tags=config["wandb"]["tags"] + ["optuna-optimization"],
            )
            wandb_callback = WandbCallback(metric_name="mean_validation_mse")  # type: ignore[misc]
            logger.info("✓ W&B Optuna tracking enabled")
        except Exception as e:
            logger.warning(f"Failed to initialize W&B Optuna tracking: {e}")
            wandb_run = None
            wandb_callback = None

    # Optimize
    logger.info(f"\n{'=' * 80}")
    logger.info(f"RUNNING OPTIMIZATION ({n_trials} trials, {n_jobs} parallel jobs)")
    logger.info(f"{'=' * 80}\n")

    start_time = time.time()

    study.optimize(
        objective,
        n_trials=n_trials,
        n_jobs=n_jobs,
        show_progress_bar=True,
        callbacks=[wandb_callback] if wandb_callback else [],
    )

    optimization_time = (time.time() - start_time) / 60

    # Results
    logger.info("\n" + "=" * 80)
    logger.info("OPTIMIZATION RESULTS")
    logger.info("=" * 80)

    logger.info(f"\nBest trial: {study.best_trial.number}")
    logger.info(f"  Best validation MSE: {study.best_value:.6f}")
    logger.info("  Parameters:")

    best_params = study.best_params
    for param, value in best_params.items():
        logger.info(f"    {param:20s}: {value}")

    logger.info(f"\nOptimization time: {optimization_time:.2f} minutes")

    # Save best parameters
    output_dir.mkdir(parents=True, exist_ok=True)
    best_config_file = output_dir / f"config_{model_name}_optimized.yaml"

    best_config = {
        "model": model_config,
        "hyperparameters": {**fixed_params, **best_params},
        "optimization": {
            "n_trials": n_trials,
            "best_trial": study.best_trial.number,
            "best_mse": float(study.best_value),
            "optimization_time_minutes": optimization_time,
        },
    }

    with open(best_config_file, "w") as f:
        yaml.dump(best_config, f, default_flow_style=False, sort_keys=False)

    logger.info(f"\n✓ Saved optimized config to {best_config_file}")

    # =========================================================================
    # TRAIN FINAL MODEL WITH OPTIMIZED HYPERPARAMETERS
    # =========================================================================
    logger.info(f"\n{'=' * 80}")
    logger.info("TRAINING FINAL MODEL WITH OPTIMIZED HYPERPARAMETERS")
    logger.info(f"{'=' * 80}")

    # Import training utilities
    from train_multi_horizon_v4 import train_model_walk_forward

    try:
        # Get pipeline-ready data file
        model_dir = Path(__file__).parent.parent
        pipeline_ready_file = model_dir / "data" / "consolidated_features_v5_pipeline_ready.parquet"

        if not pipeline_ready_file.exists():
            raise FileNotFoundError(f"Pipeline-ready file not found: {pipeline_ready_file}")

        # Train model on full walk-forward validation using optimized hyperparameters
        final_model, metrics = train_model_walk_forward(
            model_name=model_name,
            model_config=model_config,
            data_file=pipeline_ready_file,
            hyperparameters={**fixed_params, **best_params},
            features=features,
            output_dir=output_dir,  # Temp files
            walk_forward_config=config["walk_forward_validation"],
            wandb_run=None,
        )

        # Save final model to models_optuna directory
        final_model_path = output_dir / f"lightgbm_{model_name}.txt"
        final_model.save_model(str(final_model_path))

        logger.info(f"\n✓ Final optimized model saved to: {final_model_path}")

        # Format MSE safely (handle None/string cases)
        mean_mse = metrics.get("mean_mse", None)
        if mean_mse is not None and isinstance(mean_mse, (int, float)):
            logger.info(f"  Walk-forward validation MSE: {mean_mse:.6f}")
        else:
            logger.info(f"  Walk-forward validation MSE: {mean_mse}")

        logger.info("  This model can now be used in evaluation and production")

    except Exception as e:
        logger.error(f"\n✗ Failed to train final model: {e}")
        logger.error("  Optimized hyperparameters were saved, but model training failed")
        logger.error("  You can manually train the model by running:")
        logger.error(f"    cd {model_dir}")
        logger.error(f"    uv run python train_multi_horizon_v4.py --model {model_name} \\")
        logger.error("      --config ../config/multi_horizon_regime_config_v5.yaml")
        import traceback

        traceback.print_exc()

    # Clean up temporary directory
    import shutil

    temp_base_dir = Path(f"/tmp/optuna_v4_{model_name}")
    if temp_base_dir.exists():
        try:
            shutil.rmtree(temp_base_dir)
            logger.info(f"✓ Cleaned up temp directory: {temp_base_dir}")
        except Exception as e:
            logger.warning(f"Failed to clean up temp directory {temp_base_dir}: {e}")

    monitor.check_memory("After optimization")

    # Finish W&B run
    if wandb_run is not None:
        wandb_run.finish()
        logger.info("✓ W&B Optuna run finished")

    return best_config


def main() -> None:
    """Main execution function."""
    # Parse arguments
    parser = argparse.ArgumentParser(description="V4 Multi-horizon + regime hybrid model optimization")
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        choices=[
            "near_low_vol_atm",
            "near_low_vol_otm",
            "near_high_vol_atm",
            "near_high_vol_otm",
            "mid_low_vol_atm",
            "mid_low_vol_otm",
            "mid_high_vol_atm",
            "mid_high_vol_otm",
            "far_low_vol_atm",
            "far_low_vol_otm",
            "far_high_vol_atm",
            "far_high_vol_otm",
        ],
        help="Model to optimize",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).resolve().parent.parent / "config" / "multi_horizon_regime_config_v5.yaml",
        help="Path to multi-horizon + regime config file",
    )
    parser.add_argument(
        "--n_trials",
        type=int,
        default=100,
        help="Number of Optuna trials (default: 100 for high priority, 50 for medium)",
    )
    parser.add_argument(
        "--n_jobs",
        type=int,
        default=4,
        help="Number of parallel jobs (default: 4 for 32 vCPU machine)",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=None,
        help="Output directory for optimized models (default: from config)",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from existing Optuna study",
    )
    args = parser.parse_args()

    logger.info("=" * 80)
    logger.info("V4 MULTI-HORIZON + REGIME HYBRID MODEL OPTIMIZATION")
    logger.info("=" * 80)

    # Load configuration
    config = load_config(args.config)

    # Determine output directory
    if args.output_dir:
        output_dir = args.output_dir
    else:
        model_dir = Path(__file__).resolve().parent.parent
        output_dir = model_dir / "01_pricing" / "models_optuna"

    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Output directory: {output_dir}")

    # Optimize model
    best_config = optimize_model(
        model_name=args.model,
        config=config,
        n_trials=args.n_trials,
        n_jobs=args.n_jobs,
        output_dir=output_dir,
        resume=args.resume,
    )

    logger.info("\n✓ Optimization complete!")
    logger.info(f"Best MSE: {best_config['optimization']['best_mse']:.6f}")
    logger.info(f"Config saved to: {output_dir / f'config_{args.model}_optimized.yaml'}")


if __name__ == "__main__":
    main()
