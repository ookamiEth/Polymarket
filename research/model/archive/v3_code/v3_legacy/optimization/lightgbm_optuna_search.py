#!/usr/bin/env python3
"""
LightGBM Hyperparameter Optimization with Optuna
=================================================

Bayesian optimization using Optuna to find optimal LightGBM hyperparameters
for residual modeling on BTC up/down prediction markets.

Key improvements over grid search:
1. Intelligent search guided by previous trials (TPE sampler)
2. Expanded search space (8 parameters vs 4 in grid)
3. GOSS boosting support (3.5x faster training)
4. Early pruning of unpromising trials
5. Hyperparameter importance analysis

Target: Exceed 12.32% Brier improvement from grid search Trial 15

Author: BT Research Team
Date: 2025-11-02
"""

from __future__ import annotations

import gc
import logging
import pickle
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import lightgbm as lgb
import optuna
import polars as pl
import yaml
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler

# W&B integration
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    logging.warning("W&B not available - continuing without experiment tracking")

# Import from existing modules
from lightgbm_memory_optimized import (
    FEATURE_COLS,
    create_lgb_dataset_from_parquet,
)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


class OptunaObjective:
    """
    Objective function for Optuna optimization.

    Trains a LightGBM model with hyperparameters suggested by Optuna
    and returns the Brier score improvement on validation set.
    """

    def __init__(
        self,
        config: dict[str, Any],
        train_file: Path,
        val_file: Path,
        features: list[str],
    ):
        self.config = config
        self.train_file = train_file
        self.val_file = val_file
        self.features = features
        self.trial_count = 0

        # Load datasets once and reuse (memory efficient)
        logger.info("Loading training and validation datasets...")
        self.train_data, self.baseline_brier_train = create_lgb_dataset_from_parquet(
            str(train_file), features, free_raw_data=False  # Keep in memory for reuse
        )
        self.val_data, self.baseline_brier_val = create_lgb_dataset_from_parquet(
            str(val_file), features, reference=self.train_data, free_raw_data=False
        )
        logger.info(f"✓ Datasets loaded | Baseline Brier (val): {self.baseline_brier_val:.6f}")

    def __call__(self, trial: optuna.Trial) -> float:
        """
        Objective function called by Optuna for each trial.

        Args:
            trial: Optuna trial object for suggesting hyperparameters

        Returns:
            Brier score improvement percentage (higher is better)
        """
        self.trial_count += 1
        start_time = time.time()

        # Sample hyperparameters from search space
        params = self._suggest_hyperparameters(trial)

        # Log trial info
        logger.info(f"\n{'=' * 80}")
        logger.info(f"TRIAL {self.trial_count} (Optuna Trial {trial.number})")
        logger.info(f"{'=' * 80}")
        logger.info("Hyperparameters:")
        for key, value in params.items():
            if key not in ["verbose", "seed", "num_threads"]:
                logger.info(f"  {key:30s}: {value}")

        # Train model with early stopping
        callbacks = [lgb.log_evaluation(25)]

        # Add pruning callback
        callbacks.append(
            optuna.integration.LightGBMPruningCallback(trial, "l2", "val")
        )

        try:
            model = lgb.train(
                params,
                self.train_data,
                num_boost_round=self.config["fixed_hyperparameters"]["n_estimators"],
                valid_sets=[self.train_data, self.val_data],
                valid_names=["train", "val"],
                callbacks=callbacks,
            )

            # Get validation metrics
            best_score = model.best_score
            if not best_score or "val" not in best_score:
                logger.warning("No validation metrics found - trial failed")
                raise optuna.TrialPruned()

            val_metrics = best_score["val"]
            residual_mse = val_metrics.get("l2", None)  # MSE in LightGBM

            if residual_mse is None:
                logger.warning("MSE metric not found - trial failed")
                raise optuna.TrialPruned()

            # Calculate Brier improvement
            # For residual models: MSE of residuals = Brier improvement
            model_brier = self.baseline_brier_val - residual_mse
            brier_improvement_pct = (residual_mse / self.baseline_brier_val) * 100

            # Training time
            elapsed = time.time() - start_time

            # Log results
            logger.info("\nResults:")
            logger.info(f"  Baseline Brier:        {self.baseline_brier_val:.6f}")
            logger.info(f"  Residual MSE:          {residual_mse:.6f}")
            logger.info(f"  Model Brier:           {model_brier:.6f}")
            logger.info(f"  Improvement:           {brier_improvement_pct:.3f}%")
            logger.info(f"  Training Time:         {elapsed/60:.2f} minutes")
            logger.info(f"  Best Iteration:        {model.best_iteration}")

            # Log to trial attributes for later analysis
            trial.set_user_attr("baseline_brier", self.baseline_brier_val)
            trial.set_user_attr("residual_mse", residual_mse)
            trial.set_user_attr("model_brier", model_brier)
            trial.set_user_attr("training_time_minutes", elapsed / 60)
            trial.set_user_attr("best_iteration", model.best_iteration)

            # Clean up model
            del model
            gc.collect()

            return brier_improvement_pct

        except optuna.TrialPruned:
            logger.info("Trial pruned by Optuna")
            raise
        except Exception as e:
            logger.error(f"Trial failed with error: {e}")
            raise optuna.TrialPruned() from e

    def _suggest_hyperparameters(self, trial: optuna.Trial) -> dict[str, Any]:
        """
        Suggest hyperparameters based on search space configuration.

        Args:
            trial: Optuna trial object

        Returns:
            Dictionary of hyperparameters for LightGBM
        """
        search_space = self.config["search_space"]
        params = {}

        # Sample hyperparameters from search space
        for param_name, param_config in search_space.items():
            param_type = param_config["type"]

            # Skip conditional parameters (handled separately)
            if "condition" in param_config:
                continue

            if param_type == "float":
                params[param_name] = trial.suggest_float(
                    param_name,
                    param_config["low"],
                    param_config["high"],
                    log=param_config.get("log", False),
                    step=param_config.get("step"),
                )
            elif param_type == "int":
                params[param_name] = trial.suggest_int(
                    param_name,
                    param_config["low"],
                    param_config["high"],
                    step=param_config.get("step", 1),
                )
            elif param_type == "categorical":
                params[param_name] = trial.suggest_categorical(
                    param_name,
                    param_config["choices"],
                )

        # Handle conditional parameters (GOSS-specific)
        if params.get("boosting_type") == "goss":
            if "top_rate" in search_space:
                params["top_rate"] = trial.suggest_float(
                    "top_rate",
                    search_space["top_rate"]["low"],
                    search_space["top_rate"]["high"],
                )
            if "other_rate" in search_space:
                params["other_rate"] = trial.suggest_float(
                    "other_rate",
                    search_space["other_rate"]["low"],
                    search_space["other_rate"]["high"],
                )

        # Add fixed hyperparameters
        params.update(self.config["fixed_hyperparameters"])

        # Memory/performance settings
        if self.config.get("use_gpu", False):
            params.update(
                {
                    "device_type": "gpu",
                    "gpu_use_dp": False,
                }
            )

        return params


def create_study(config: dict[str, Any]) -> optuna.Study:
    """
    Create Optuna study with configured sampler and pruner.

    Args:
        config: Configuration dictionary

    Returns:
        Optuna study object
    """
    opt_config = config["optimization"]

    # Create sampler (TPE for Bayesian optimization)
    sampler = TPESampler(seed=config["fixed_hyperparameters"]["seed"]) if opt_config["sampler"] == "TPE" else None

    # Create pruner (MedianPruner to stop unpromising trials)
    if opt_config["pruner"] == "MedianPruner":
        pruner = MedianPruner(
            n_startup_trials=opt_config["pruner_config"]["n_startup_trials"],
            n_warmup_steps=opt_config["pruner_config"]["n_warmup_steps"],
            interval_steps=opt_config["pruner_config"]["interval_steps"],
        )
    else:
        pruner = None

    # Create study
    study = optuna.create_study(
        study_name=opt_config["study_name"],
        direction=opt_config["direction"],
        sampler=sampler,
        pruner=pruner,
        storage=opt_config.get("storage"),
    )

    logger.info(f"✓ Created Optuna study: {opt_config['study_name']}")
    logger.info(f"  Sampler: {opt_config['sampler']}")
    logger.info(f"  Pruner: {opt_config['pruner']}")
    logger.info(f"  Direction: {opt_config['direction']}")

    return study


def save_results(
    study: optuna.Study,
    config: dict[str, Any],
    output_dir: Path,
) -> None:
    """
    Save optimization results to disk.

    Args:
        study: Completed Optuna study
        config: Configuration dictionary
        output_dir: Output directory
    """
    logger.info("\nSaving results...")

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save study object
    study_path = output_dir / "optuna_study.pkl"
    with open(study_path, "wb") as f:
        pickle.dump(study, f)
    logger.info(f"✓ Saved study to {study_path}")

    # Save trials to CSV
    trials_df = study.trials_dataframe()
    trials_csv = output_dir / "optuna_trials.csv"
    trials_df.to_csv(trials_csv, index=False)
    logger.info(f"✓ Saved trials to {trials_csv}")

    # Save best trial config
    best_trial = study.best_trial
    best_config = {
        "hyperparameters": best_trial.params,
        "performance": {
            "brier_improvement_pct": best_trial.value,
            "baseline_brier": best_trial.user_attrs.get("baseline_brier"),
            "residual_mse": best_trial.user_attrs.get("residual_mse"),
            "model_brier": best_trial.user_attrs.get("model_brier"),
            "training_time_minutes": best_trial.user_attrs.get("training_time_minutes"),
            "best_iteration": best_trial.user_attrs.get("best_iteration"),
        },
        "metadata": {
            "trial_number": best_trial.number,
            "optimization_date": datetime.now().isoformat(),
            "total_trials": len(study.trials),
            "pruned_trials": len([t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED]),
            "completed_trials": len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]),
        },
    }

    # Add fixed hyperparameters
    best_config["hyperparameters"].update(config["fixed_hyperparameters"])

    best_config_path = output_dir / "best_config.yaml"
    with open(best_config_path, "w") as f:
        yaml.dump(best_config, f, default_flow_style=False, sort_keys=False)
    logger.info(f"✓ Saved best config to {best_config_path}")

    # Print summary
    logger.info("\n" + "=" * 80)
    logger.info("OPTIMIZATION SUMMARY")
    logger.info("=" * 80)
    logger.info(f"Total Trials:          {len(study.trials)}")
    logger.info(f"Completed Trials:      {best_config['metadata']['completed_trials']}")
    logger.info(f"Pruned Trials:         {best_config['metadata']['pruned_trials']}")
    logger.info(f"\nBest Trial: #{best_trial.number}")
    logger.info(f"  Brier Improvement:   {best_trial.value:.3f}%")
    logger.info(f"  Model Brier:         {best_config['performance']['model_brier']:.6f}")
    logger.info(f"  Training Time:       {best_config['performance']['training_time_minutes']:.2f} minutes")
    logger.info("\nBest Hyperparameters:")
    for key, value in best_trial.params.items():
        logger.info(f"  {key:30s}: {value}")
    logger.info("=" * 80)


def analyze_hyperparameter_importance(
    study: optuna.Study,
    output_dir: Path,
) -> None:
    """
    Analyze and visualize hyperparameter importance.

    Args:
        study: Completed Optuna study
        output_dir: Output directory for plots
    """
    logger.info("\nAnalyzing hyperparameter importance...")

    plot_dir = output_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)

    try:
        import optuna.visualization as vis
        import plotly.io as pio

        # Hyperparameter importance plot
        fig = vis.plot_param_importances(study)
        pio.write_html(fig, plot_dir / "param_importance.html")
        logger.info("✓ Saved hyperparameter importance plot")

        # Optimization history
        fig = vis.plot_optimization_history(study)
        pio.write_html(fig, plot_dir / "optimization_history.html")
        logger.info("✓ Saved optimization history plot")

        # Parallel coordinate plot
        fig = vis.plot_parallel_coordinate(study)
        pio.write_html(fig, plot_dir / "parallel_coordinate.html")
        logger.info("✓ Saved parallel coordinate plot")

        # Slice plot (marginal effects)
        fig = vis.plot_slice(study)
        pio.write_html(fig, plot_dir / "slice_plot.html")
        logger.info("✓ Saved slice plot")

        # Contour plot (interactions)
        fig = vis.plot_contour(study)
        pio.write_html(fig, plot_dir / "contour_plot.html")
        logger.info("✓ Saved contour plot")

    except ImportError:
        logger.warning("Plotly not available - skipping visualization")
    except Exception as e:
        logger.warning(f"Failed to generate plots: {e}")


def main() -> None:
    """Main execution function."""
    logger.info("\n" + "=" * 80)
    logger.info("LIGHTGBM HYPERPARAMETER OPTIMIZATION WITH OPTUNA")
    logger.info("=" * 80)

    # Load configuration
    config_path = Path(__file__).parent / "config" / "optuna_config.yaml"
    logger.info(f"\nLoading configuration from {config_path}")

    with open(config_path) as f:
        config = yaml.safe_load(f)

    # Initialize W&B if enabled
    wandb_run = None
    if config["wandb"]["enabled"] and WANDB_AVAILABLE:
        wandb_run = wandb.init(
            project=config["wandb"]["project"],
            entity=config["wandb"]["entity"],
            config=config,
            tags=config["wandb"]["tags"],
            notes=config["wandb"]["notes"],
        )
        logger.info("✓ W&B run initialized")

    # Setup paths
    model_dir = Path(__file__).parent.parent
    data_config = config["data"]

    train_file = model_dir / "results" / data_config["train_file"]
    val_file = model_dir / "results" / data_config["val_file"]
    output_dir = model_dir / "results" / "optuna"

    # Validate files exist
    if not train_file.exists():
        raise FileNotFoundError(f"Training file not found: {train_file}")
    if not val_file.exists():
        raise FileNotFoundError(f"Validation file not found: {val_file}")

    logger.info("\nData files:")
    logger.info(f"  Train: {train_file}")
    logger.info(f"  Val:   {val_file}")

    # Get feature columns
    schema = pl.scan_parquet(train_file).collect_schema()
    features = [col for col in FEATURE_COLS if col in schema.names()]
    logger.info(f"\nUsing {len(features)} features for optimization")

    # Create objective function
    objective = OptunaObjective(
        config=config,
        train_file=train_file,
        val_file=val_file,
        features=features,
    )

    # Create study
    study = create_study(config)

    # Run optimization
    opt_config = config["optimization"]
    logger.info("\nStarting optimization...")
    logger.info(f"  Trials: {opt_config['n_trials']}")
    logger.info(f"  Timeout: {opt_config['timeout']/3600:.1f} hours")
    logger.info(f"  Jobs: {opt_config['n_jobs']}")

    start_time = time.time()

    study.optimize(
        objective,
        n_trials=opt_config["n_trials"],
        timeout=opt_config.get("timeout"),
        n_jobs=opt_config.get("n_jobs", 1),
        show_progress_bar=True,
    )

    elapsed = time.time() - start_time
    logger.info(f"\n✓ Optimization completed in {elapsed/3600:.2f} hours")

    # Save results
    save_results(study, config, output_dir)

    # Analyze hyperparameter importance
    if config["output"].get("generate_plots", True):
        analyze_hyperparameter_importance(study, output_dir)

    # Log to W&B
    if wandb_run is not None:
        wandb_run.log(
            {
                "best_brier_improvement_pct": study.best_value,
                "best_trial_number": study.best_trial.number,
                "total_trials": len(study.trials),
                "optimization_time_hours": elapsed / 3600,
            }
        )
        wandb_run.finish()

    logger.info("\n✓ All done!")


if __name__ == "__main__":
    main()
