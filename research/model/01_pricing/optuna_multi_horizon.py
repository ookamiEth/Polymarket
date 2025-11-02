#!/usr/bin/env python3
"""
Multi-Horizon Optuna Hyperparameter Optimization
=================================================

Per-bucket Bayesian hyperparameter optimization with specialized search spaces:
- Near bucket: Higher LR, shallower trees (fast convergence)
- Mid bucket: Balanced parameters
- Far bucket: Lower LR, deeper trees (capture complexity)

This is Phase 2 of the multi-horizon implementation.

Usage:
  # Optimize all buckets (150 trials total)
  uv run python optuna_multi_horizon.py

  # Optimize specific bucket
  uv run python optuna_multi_horizon.py --bucket near --n-trials 100

  # Resume from existing study
  uv run python optuna_multi_horizon.py --bucket near --resume

Author: BT Research Team
Date: 2025-11-02
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
import optuna
import polars as pl
import yaml

# Import from existing modules
from lightgbm_memory_optimized import (
    FEATURE_COLS,
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

# W&B integration
try:
    import wandb

    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    logger.warning("W&B not available - continuing without experiment tracking")


def load_config(config_path: Path) -> dict[str, Any]:
    """Load multi-horizon configuration."""
    logger.info(f"Loading configuration from {config_path}")
    with open(config_path) as f:
        config = yaml.safe_load(f)
    return config


def get_search_space(bucket_name: str, config: dict[str, Any]) -> dict[str, tuple]:
    """
    Get Optuna search space for specific bucket.

    Args:
        bucket_name: Bucket identifier (near/mid/far)
        config: Multi-horizon configuration

    Returns:
        Dictionary of parameter ranges
    """
    if not config["training"]["per_bucket_optimization"]["enabled"]:
        raise ValueError("Per-bucket optimization not enabled in config")

    search_spaces = config["training"]["per_bucket_optimization"]
    bucket_space = search_spaces[bucket_name]

    logger.info(f"\n{bucket_name.upper()} bucket search space:")
    logger.info(f"  Learning rate:      {bucket_space['learning_rate']}")
    logger.info(f"  Num leaves:         {bucket_space['num_leaves']}")
    logger.info(f"  Min data in leaf:   {bucket_space['min_data_in_leaf']}")
    logger.info(f"  L2 regularization:  {bucket_space['lambda_l2']}")

    return bucket_space


class OptunaObjective:
    """Optuna objective function for bucket-specific optimization."""

    def __init__(
        self,
        bucket_name: str,
        search_space: dict[str, Any],
        train_file: Path,
        val_file: Path,
        features: list[str],
        fixed_params: dict[str, Any],
        baseline_brier: float,
    ):
        self.bucket_name = bucket_name
        self.search_space = search_space
        self.train_file = train_file
        self.val_file = val_file
        self.features = features
        self.fixed_params = fixed_params
        self.baseline_brier = baseline_brier

        # Load datasets once (reuse across trials)
        logger.info(f"Loading datasets for {bucket_name} bucket...")
        self.train_data, _ = create_lgb_dataset_from_parquet(str(train_file), features, free_raw_data=True)
        self.val_data, _ = create_lgb_dataset_from_parquet(
            str(val_file), features, reference=self.train_data, free_raw_data=True
        )
        logger.info(f"  Train: {self.train_data.num_data():,} samples, Val: {self.val_data.num_data():,} samples")

    def __call__(self, trial: optuna.Trial) -> float:
        """
        Objective function for a single Optuna trial.

        Args:
            trial: Optuna trial object

        Returns:
            Validation residual MSE (lower is better)
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
            log=True,
        )

        # Train model
        callbacks = [
            lgb.early_stopping(50),
            lgb.log_evaluation(0),  # Silent
        ]

        model = lgb.train(
            params,
            self.train_data,
            num_boost_round=1000,
            valid_sets=[self.val_data],
            valid_names=["val"],
            callbacks=callbacks,
        )

        # Get validation metrics
        val_metrics = model.best_score.get("val", {})
        residual_mse = val_metrics.get("l2", 0.0)
        brier_improvement_pct = (residual_mse / self.baseline_brier) * 100

        # Log intermediate values
        trial.set_user_attr("residual_mse", residual_mse)
        trial.set_user_attr("brier_improvement_pct", brier_improvement_pct)
        trial.set_user_attr("best_iteration", model.best_iteration)

        logger.info(
            f"  Trial {trial.number:3d}: MSE={residual_mse:.6f}, Improvement={brier_improvement_pct:.2f}%, "
            f"Iter={model.best_iteration}"
        )

        # Clean up
        del model
        gc.collect()

        return residual_mse  # Minimize residual MSE

    def cleanup(self):
        """Clean up datasets after optimization."""
        del self.train_data, self.val_data
        gc.collect()


def optimize_bucket(
    bucket_name: str,
    config: dict[str, Any],
    n_trials: int,
    resume: bool,
    output_dir: Path,
    wandb_run: Any | None = None,
) -> dict[str, Any]:
    """
    Run Optuna optimization for a specific bucket.

    Args:
        bucket_name: Bucket identifier (near/mid/far)
        config: Multi-horizon configuration
        n_trials: Number of Optuna trials
        resume: Whether to resume from existing study
        output_dir: Output directory for results
        wandb_run: W&B run (optional)

    Returns:
        Best hyperparameters dictionary
    """
    logger.info(f"\n{'=' * 80}")
    logger.info(f"OPTIMIZING {bucket_name.upper()} BUCKET")
    logger.info(f"{'=' * 80}")

    monitor = MemoryMonitor()
    start_time = time.time()

    bucket_config = config["buckets"][bucket_name]

    # Get search space
    search_space = get_search_space(bucket_name, config)

    # Setup paths
    model_dir = Path(__file__).parent.parent
    data_dir = model_dir / Path(config["data"]["output_dir"])
    train_file = data_dir / config["data"]["bucket_files"][bucket_name]["train"]
    val_file = data_dir / config["data"]["bucket_files"][bucket_name]["val"]

    # Verify files exist
    if not train_file.exists():
        raise FileNotFoundError(f"Train file not found: {train_file}. Run train_multi_horizon.py first.")
    if not val_file.exists():
        raise FileNotFoundError(f"Val file not found: {val_file}. Run train_multi_horizon.py first.")

    # Get features
    schema = pl.scan_parquet(train_file).collect_schema()
    features = [col for col in FEATURE_COLS if col in schema.names()]
    logger.info(f"Using {len(features)} features")

    # Calculate baseline Brier
    _, baseline_brier_raw = create_lgb_dataset_from_parquet(str(val_file), features)
    baseline_brier: float = float(baseline_brier_raw)
    logger.info(f"Baseline Brier score: {baseline_brier:.6f}")

    # Fixed parameters (from best single model config)
    fixed_params = {
        "objective": "regression",
        "metric": ["l2", "mae"],
        "verbosity": -1,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq": 1,
        "n_jobs": -1,
        "force_col_wise": True,
        "early_stopping_rounds": 50,
    }

    # Create Optuna study
    study_name = f"multi_horizon_{bucket_name}"
    storage_path = output_dir / f"optuna_{bucket_name}.db"
    storage = f"sqlite:///{storage_path}"

    if resume and storage_path.exists():
        logger.info(f"Resuming study from {storage_path}")
        study = optuna.load_study(study_name=study_name, storage=storage, sampler=optuna.samplers.TPESampler())
        logger.info(f"Loaded {len(study.trials)} existing trials")
    else:
        logger.info(f"Creating new study: {study_name}")
        study = optuna.create_study(
            study_name=study_name,
            storage=storage,
            direction="minimize",
            sampler=optuna.samplers.TPESampler(),
            load_if_exists=False,
        )

    # Create objective
    objective = OptunaObjective(
        bucket_name=bucket_name,
        search_space=search_space,
        train_file=train_file,
        val_file=val_file,
        features=features,
        fixed_params=fixed_params,
        baseline_brier=baseline_brier,
    )

    # Run optimization
    logger.info(f"\nRunning {n_trials} trials...")
    try:
        study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
    finally:
        objective.cleanup()

    # Get best results
    best_trial = study.best_trial
    best_params = best_trial.params
    best_mse = best_trial.value
    best_improvement = best_trial.user_attrs["brier_improvement_pct"]
    best_iteration = best_trial.user_attrs["best_iteration"]

    optimization_time = (time.time() - start_time) / 60

    logger.info(f"\n{bucket_name.upper()} Optimization Results:")
    logger.info(f"  Best trial:          {best_trial.number}")
    logger.info(f"  Best MSE:            {best_mse:.6f}")
    logger.info(f"  Best improvement:    {best_improvement:.2f}%")
    logger.info(f"  Best iteration:      {best_iteration}")
    logger.info(f"  Optimization time:   {optimization_time:.2f} min")
    logger.info("\nBest hyperparameters:")
    for param, value in best_params.items():
        logger.info(f"    {param:20s}: {value}")

    # Check against target
    target_min = float(bucket_config["target_improvement"].split("-")[0])
    current_improvement = bucket_config["current_improvement"]

    if best_improvement >= target_min:
        logger.info(f"  ✓ Target achieved ({target_min}%+)")
    else:
        logger.warning(f"  ⚠ Below target (expected {target_min}%+)")

    improvement_gain = best_improvement - current_improvement
    logger.info(
        f"  Gain over baseline:  +{improvement_gain:.2f}pp (from {current_improvement:.1f}% to {best_improvement:.2f}%)"
    )

    # Save best config
    output_dir.mkdir(parents=True, exist_ok=True)
    config_file = output_dir / f"config_{bucket_name}_optimized.yaml"

    best_config = {
        "bucket": bucket_config,
        "hyperparameters": {**fixed_params, **best_params, "n_estimators": 1000},
        "optimization": {
            "study_name": study_name,
            "n_trials": len(study.trials),
            "best_trial": best_trial.number,
            "optimization_time_minutes": optimization_time,
        },
        "performance": {
            "baseline_brier": baseline_brier,
            "residual_mse": best_mse,
            "brier_improvement_pct": best_improvement,
            "best_iteration": best_iteration,
            "improvement_gain_pp": improvement_gain,
        },
    }

    with open(config_file, "w") as f:
        yaml.dump(best_config, f, default_flow_style=False, sort_keys=False)

    logger.info(f"\n✓ Saved optimized config to {config_file}")

    # Log to W&B
    if wandb_run is not None:
        wandb_run.log(
            {
                f"optuna_{bucket_name}/best_improvement": best_improvement,
                f"optuna_{bucket_name}/best_mse": best_mse,
                f"optuna_{bucket_name}/optimization_time_min": optimization_time,
                f"optuna_{bucket_name}/improvement_gain_pp": improvement_gain,
            }
        )

    monitor.check_memory("After optimization")

    return best_config["hyperparameters"]


def main() -> None:
    """Main execution function."""
    # Parse arguments
    parser = argparse.ArgumentParser(description="Multi-horizon Optuna hyperparameter optimization")
    parser.add_argument(
        "--bucket",
        type=str,
        choices=["near", "mid", "far", "all"],
        default="all",
        help="Which bucket to optimize (default: all)",
    )
    parser.add_argument(
        "--n-trials",
        type=int,
        default=50,
        help="Number of Optuna trials per bucket (default: 50)",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).parent / "config" / "multi_horizon_config.yaml",
        help="Path to multi-horizon config file",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from existing Optuna study",
    )
    args = parser.parse_args()

    logger.info("\n" + "=" * 80)
    logger.info("MULTI-HORIZON OPTUNA OPTIMIZATION")
    logger.info("=" * 80)

    # Load configuration
    config = load_config(args.config)

    # Enable per-bucket optimization in config
    if not config["training"]["per_bucket_optimization"]["enabled"]:
        logger.warning("Enabling per_bucket_optimization in config")
        config["training"]["per_bucket_optimization"]["enabled"] = True

    # Initialize W&B if enabled
    wandb_run = None
    if config["wandb"]["enabled"] and WANDB_AVAILABLE:
        wandb_run = wandb.init(
            project=config["wandb"]["project"],
            entity=config["wandb"]["entity"],
            config={"multi_horizon_optuna": config, "n_trials": args.n_trials},
            tags=config["wandb"]["tags"] + ["optuna", "phase-2"],
            notes="Phase 2: Per-bucket Bayesian hyperparameter optimization",
        )
        logger.info("✓ W&B run initialized")

    # Setup paths
    model_dir = Path(__file__).parent.parent
    output_dir = model_dir / Path(config["models"]["output_dir"])

    # Determine which buckets to optimize
    buckets_to_optimize = ["near", "mid", "far"] if args.bucket == "all" else [args.bucket]

    logger.info(f"\nOptimizing buckets: {', '.join(buckets_to_optimize)}")
    logger.info(f"Trials per bucket: {args.n_trials}")
    logger.info(
        f"Total trials: {len(buckets_to_optimize) * args.n_trials} (~{len(buckets_to_optimize) * args.n_trials * 2 / 60:.1f} hours)\n"
    )

    # Optimize each bucket
    all_results = {}

    for bucket_name in buckets_to_optimize:
        try:
            best_params = optimize_bucket(
                bucket_name=bucket_name,
                config=config,
                n_trials=args.n_trials,
                resume=args.resume,
                output_dir=output_dir,
                wandb_run=wandb_run,
            )
            all_results[bucket_name] = best_params
        except Exception as e:
            logger.error(f"Error optimizing {bucket_name}: {e}")
            raise

        gc.collect()

    # Summary
    logger.info("\n" + "=" * 80)
    logger.info("OPTIMIZATION SUMMARY")
    logger.info("=" * 80)

    for bucket_name in buckets_to_optimize:
        config_file = output_dir / f"config_{bucket_name}_optimized.yaml"
        if config_file.exists():
            with open(config_file) as f:
                bucket_results = yaml.safe_load(f)
            improvement = bucket_results["performance"]["brier_improvement_pct"]
            gain = bucket_results["performance"]["improvement_gain_pp"]
            target_min = float(bucket_results["bucket"]["target_improvement"].split("-")[0])

            status = "✓" if improvement >= target_min else "⚠"
            logger.info(
                f"{status} {bucket_results['bucket']['name']:25s}: {improvement:6.2f}% "
                f"(+{gain:4.2f}pp gain, target {target_min:4.1f}%+)"
            )

    # Finish W&B
    if wandb_run is not None:
        wandb_run.finish()

    logger.info("\n✓ Multi-horizon Optuna optimization complete!")
    logger.info(f"Optimized configs saved to: {output_dir}")
    logger.info("\nNext steps:")
    logger.info("  1. Review optimized configs")
    logger.info("  2. Retrain models with optimized hyperparameters")
    logger.info("  3. Run evaluation: uv run python evaluate_multi_horizon.py")


if __name__ == "__main__":
    main()
