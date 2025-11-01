#!/usr/bin/env python3
"""
LightGBM Grid Search with Weights & Biases Tracking
====================================================

Exhaustive grid search for hyperparameter tuning with W&B experiment tracking.

Features:
- Exhaustive search over discrete hyperparameter grid
- Weights & Biases integration for visualization
- Checkpoint/resume capability
- Progress tracking and best model selection
- Full dataset training (63M rows) with memory safety

Usage:
    # Phase 1 (coarse grid, 32 combinations)
    uv run python lightgbm_grid_search.py --config config/lightgbm_grid_search_config.yaml

    # Resume from checkpoint
    uv run python lightgbm_grid_search.py --config config/lightgbm_grid_search_config.yaml --resume

    # Pilot test on October 2023
    uv run python lightgbm_grid_search.py --config config/lightgbm_grid_search_config.yaml --pilot

Author: BT Research Team
Date: 2025-11-01
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from datetime import date
from itertools import product
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

# Import from existing lightgbm script
from lightgbm_memory_optimized import (
    OUTPUT_DIR,
    evaluate_brier_score,
    train_lightgbm_memory_optimized,
    train_temporal_chunks,
)

try:
    import wandb

    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    print("⚠️  WARNING: wandb not installed. Install with: pip install wandb")
    print("Continuing without W&B tracking (results will only be saved locally)")

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


class GridSearchManager:
    """Manages grid search execution, checkpointing, and W&B tracking."""

    def __init__(self, config: dict[str, Any], resume: bool = False):
        self.config = config
        self.resume = resume

        # Extract config sections
        self.grid_params = config["grid_search"]
        self.fixed_params = config["fixed_params"]
        self.goss_params = config.get("goss_params", {})
        self.dataset_config = config["dataset"]
        self.wandb_config = config["wandb"]
        self.execution_config = config["execution"]
        self.optimization_config = config["optimization"]

        # Setup paths
        self.checkpoint_file = Path(self.execution_config["checkpoint_file"])
        self.results_csv = Path(self.execution_config["results_csv"])
        self.best_model_path = Path(self.execution_config["best_model_path"])
        self.summary_file = Path(self.execution_config["summary_file"])

        # Ensure output directory exists
        for file_path in [self.checkpoint_file, self.results_csv, self.best_model_path, self.summary_file]:
            file_path.parent.mkdir(parents=True, exist_ok=True)

        # State tracking
        self.results: list[dict[str, Any]] = []
        self.best_result: dict[str, Any] | None = None
        self.completed_trials: set[str] = set()

        # Initialize W&B
        self.wandb_enabled = WANDB_AVAILABLE and self.wandb_config.get("project")

        # Load checkpoint if resuming
        if self.resume and self.checkpoint_file.exists():
            self._load_checkpoint()

    def _generate_grid_combinations(self) -> list[dict[str, Any]]:
        """Generate all hyperparameter combinations from grid."""
        param_names = list(self.grid_params.keys())
        param_values = [self.grid_params[name] for name in param_names]

        combinations = []
        for values in product(*param_values):
            combo = dict(zip(param_names, values))
            combinations.append(combo)

        logger.info(f"Generated {len(combinations)} hyperparameter combinations")
        return combinations

    def _get_trial_id(self, params: dict[str, Any]) -> str:
        """Generate unique trial ID from hyperparameters."""
        # Sort for consistency
        param_str = "_".join(f"{k}={v}" for k, v in sorted(params.items()))
        return param_str

    def _get_run_name(self, params: dict[str, Any], trial_num: int = 0) -> str:
        """Generate W&B run name from hyperparameters."""
        template = self.wandb_config.get("run_name_template", "{boosting_type}-lr{learning_rate}")

        # Add trial_num to params for template formatting
        format_params = {**params, "trial_num": trial_num}

        # Simple template formatting
        name = template.format(**format_params)
        return name

    def _merge_params(self, grid_params: dict[str, Any]) -> dict[str, Any]:
        """Merge grid params with fixed params to create full config."""
        full_config = {
            "hyperparameters": {**self.fixed_params, **grid_params},
            "memory": {
                "max_bin": self.fixed_params.get("max_bin", 255),
                "num_threads": self.fixed_params.get("num_threads", 12),
                "min_data_per_group": self.fixed_params.get("min_data_per_group", 100),
                "max_cat_threshold": self.fixed_params.get("max_cat_threshold", 32),
                "histogram_pool_size": self.fixed_params.get("histogram_pool_size", -1),
            },
            "use_gpu": False,
        }

        # Add GOSS-specific params if using GOSS boosting
        if grid_params.get("boosting_type") == "goss":
            full_config["hyperparameters"].update(self.goss_params)

        return full_config

    def _train_single_trial(
        self,
        trial_num: int,
        total_trials: int,
        grid_params: dict[str, Any],
    ) -> dict[str, Any]:
        """Train a single trial with given hyperparameters."""
        trial_id = self._get_trial_id(grid_params)
        run_name = self._get_run_name(grid_params, trial_num)

        logger.info("=" * 80)
        logger.info(f"TRIAL {trial_num}/{total_trials}: {run_name}")
        logger.info("=" * 80)
        logger.info(f"Hyperparameters: {grid_params}")

        # Initialize W&B run if enabled
        wandb_run = None
        if self.wandb_enabled:
            wandb_run = wandb.init(
                project=self.wandb_config["project"],
                entity=self.wandb_config.get("entity"),
                name=run_name,
                tags=self.wandb_config.get("tags", []),
                notes=self.wandb_config.get("notes", ""),
                config=grid_params,
                reinit=True,  # Allow multiple runs in same process
            )

        start_time = time.time()

        try:
            # Merge grid params with fixed params
            full_config = self._merge_params(grid_params)

            # Create temporary config file for this trial
            trial_config_path = OUTPUT_DIR / f"trial_config_{trial_num}.yaml"
            with open(trial_config_path, "w") as f:
                yaml.dump(full_config, f)

            # Train model using existing infrastructure
            # This returns a dictionary with test set metrics
            test_metrics = train_temporal_chunks(
                start_date=date.fromisoformat(self.dataset_config["start_date"]),
                end_date=date.fromisoformat(self.dataset_config["end_date"]),
                config_file=str(trial_config_path),
                chunk_months=self.dataset_config["chunk_months"],
                train_ratio=self.dataset_config["train_ratio"],
                val_ratio=self.dataset_config["val_ratio"],
                test_ratio=self.dataset_config["test_ratio"],
                evaluate_test=True,
                wandb_run=wandb_run,  # Pass W&B run for training curve logging
            )

            # Calculate training time
            training_time = (time.time() - start_time) / 60  # minutes

            # Combine grid params with metrics
            result = {
                "trial_num": trial_num,
                "trial_id": trial_id,
                "run_name": run_name,
                **grid_params,
                "training_time_minutes": training_time,
                "status": "completed",
                **test_metrics,  # Add all test metrics (Brier scores, residual metrics)
            }

            # Log to W&B if enabled
            if wandb_run:
                wandb.log(
                    {
                        "training_time_minutes": training_time,
                        **test_metrics,  # Log all metrics to W&B
                    }
                )

                # Save model artifact
                model_artifact = wandb.Artifact(
                    name=f"model-{run_name}",
                    type="model",
                    description=f"LightGBM model for trial {trial_num}",
                )
                model_file = OUTPUT_DIR / "lightgbm_model_optimized.txt"
                if model_file.exists():
                    model_artifact.add_file(str(model_file))
                    wandb_run.log_artifact(model_artifact)

            # Clean up temporary config
            trial_config_path.unlink(missing_ok=True)

            logger.info(f"✅ Trial {trial_num} completed in {training_time:.2f} minutes")

            return result

        except Exception as e:
            logger.error(f"❌ Trial {trial_num} failed: {e}")
            training_time = (time.time() - start_time) / 60

            result = {
                "trial_num": trial_num,
                "trial_id": trial_id,
                "run_name": run_name,
                **grid_params,
                "training_time_minutes": training_time,
                "error": str(e),
                "status": "failed",
            }

            if wandb_run:
                wandb.log({"error": str(e), "status": "failed"})

            return result

        finally:
            # Finish W&B run
            if wandb_run:
                wandb.finish()

    def _save_checkpoint(self) -> None:
        """Save current progress to checkpoint file."""
        checkpoint = {
            "completed_trials": list(self.completed_trials),
            "results": self.results,
            "best_result": self.best_result,
        }

        with open(self.checkpoint_file, "w") as f:
            json.dump(checkpoint, f, indent=2)

        logger.info(f"Checkpoint saved: {self.checkpoint_file}")

    def _load_checkpoint(self) -> None:
        """Load progress from checkpoint file."""
        with open(self.checkpoint_file) as f:
            checkpoint = json.load(f)

        self.completed_trials = set(checkpoint["completed_trials"])
        self.results = checkpoint["results"]
        self.best_result = checkpoint.get("best_result")

        logger.info(f"Loaded checkpoint: {len(self.completed_trials)} trials completed")

    def _save_results_csv(self) -> None:
        """Save all results to CSV file."""
        if not self.results:
            return

        df = pd.DataFrame(self.results)
        df.to_csv(self.results_csv, index=False)
        logger.info(f"Results saved: {self.results_csv}")

    def _save_summary(self) -> None:
        """Save summary of grid search results."""
        if not self.best_result:
            logger.warning("No best result to save")
            return

        summary = f"""
{'=' * 80}
GRID SEARCH SUMMARY
{'=' * 80}

Total Trials: {len(self.results)}
Completed: {len([r for r in self.results if r.get('status') != 'failed'])}
Failed: {len([r for r in self.results if r.get('status') == 'failed'])}

Best Configuration:
{'-' * 80}
Trial: {self.best_result['trial_num']}
Run Name: {self.best_result['run_name']}

Hyperparameters:
"""

        # Add hyperparameters
        for param in self.grid_params.keys():
            if param in self.best_result:
                summary += f"  {param}: {self.best_result[param]}\n"

        summary += f"""
Performance Metrics:
{'-' * 80}
"""

        # Add metrics (when available)
        metric_keys = ["brier_improvement_pct", "model_brier", "residual_mse", "residual_rmse", "training_time_minutes"]
        for key in metric_keys:
            if key in self.best_result:
                summary += f"  {key}: {self.best_result[key]}\n"

        summary += f"""
{'=' * 80}

Best model saved to: {self.best_model_path}
All results saved to: {self.results_csv}
W&B project: {self.wandb_config['project']}

{'=' * 80}
"""

        with open(self.summary_file, "w") as f:
            f.write(summary)

        print(summary)
        logger.info(f"Summary saved: {self.summary_file}")

    def run_grid_search(self) -> None:
        """Execute the complete grid search."""
        logger.info("=" * 80)
        logger.info("STARTING LIGHTGBM GRID SEARCH WITH W&B TRACKING")
        logger.info("=" * 80)

        if self.wandb_enabled:
            logger.info(f"W&B Project: {self.wandb_config['project']}")
            logger.info(f"W&B Entity: {self.wandb_config.get('entity', 'default')}")
        else:
            logger.warning("W&B tracking disabled (not installed or no project configured)")

        # Generate all combinations
        combinations = self._generate_grid_combinations()
        total_trials = len(combinations)

        logger.info(f"Total combinations: {total_trials}")
        logger.info(f"Dataset: {self.dataset_config['start_date']} to {self.dataset_config['end_date']}")
        logger.info(f"Estimated time: {total_trials * 15:.0f} - {total_trials * 20:.0f} minutes")

        # Execute trials
        for trial_num, grid_params in enumerate(combinations, start=1):
            trial_id = self._get_trial_id(grid_params)

            # Skip if already completed (from checkpoint)
            if trial_id in self.completed_trials:
                logger.info(f"Skipping trial {trial_num}/{total_trials} (already completed): {trial_id}")
                continue

            # Train trial
            result = self._train_single_trial(trial_num, total_trials, grid_params)

            # Update results
            self.results.append(result)
            self.completed_trials.add(trial_id)

            # Update best result (if metrics available)
            optimization_metric = self.optimization_config["metric"]
            if optimization_metric in result and result.get("status") != "failed":
                if self.best_result is None or result[optimization_metric] > self.best_result.get(
                    optimization_metric, float("-inf")
                ):
                    self.best_result = result
                    logger.info(f"🏆 New best result! {optimization_metric}: {result[optimization_metric]}")

            # Save checkpoint periodically
            if trial_num % self.execution_config["checkpoint_every"] == 0:
                self._save_checkpoint()
                self._save_results_csv()

        # Final save
        self._save_checkpoint()
        self._save_results_csv()
        self._save_summary()

        logger.info("=" * 80)
        logger.info("GRID SEARCH COMPLETED")
        logger.info("=" * 80)


def main() -> None:
    """Main entry point."""
    parser = argparse.ArgumentParser(description="LightGBM Grid Search with W&B Tracking")
    parser.add_argument(
        "--config",
        type=str,
        default="config/lightgbm_grid_search_config.yaml",
        help="Grid search configuration file",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from checkpoint",
    )
    parser.add_argument(
        "--pilot",
        action="store_true",
        help="Run pilot test on October 2023 only",
    )

    args = parser.parse_args()

    # Load configuration
    config_path = Path(__file__).parent / args.config
    with open(config_path) as f:
        config = yaml.safe_load(f)

    # Override with pilot settings if requested
    if args.pilot:
        config["dataset"]["start_date"] = "2023-10-01"
        config["dataset"]["end_date"] = "2023-10-31"
        config["dataset"]["chunk_months"] = 0  # No chunking
        logger.info("🧪 PILOT MODE: Running on October 2023 only")

    # Create grid search manager and run
    manager = GridSearchManager(config, resume=args.resume)
    manager.run_grid_search()


if __name__ == "__main__":
    main()
