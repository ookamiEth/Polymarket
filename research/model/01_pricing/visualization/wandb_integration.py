#!/usr/bin/env python3
"""
W&B Integration Helpers.

Utilities for uploading plots, fetching run histories, and managing
Weights & Biases integration across visualization modules.

Usage:
    from visualization.wandb_integration import upload_plots_batch
    upload_plots_batch({"calibration": "path/to/plot.png", ...})
"""

import logging
from pathlib import Path
from typing import Any, Optional, Union

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# Try importing wandb, but make it optional
try:
    import wandb

    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    logger.warning("W&B not installed - integration features disabled")


def check_wandb_available() -> bool:
    """Check if W&B is available and initialized."""
    if not WANDB_AVAILABLE:
        logger.warning("W&B not available - install with: uv pip install wandb")
        return False

    if wandb.run is None:
        logger.warning("No active W&B run - call wandb.init() first")
        return False

    return True


def upload_plot(
    plot_path: Union[str, Path],
    wandb_key: str,
    caption: Optional[str] = None,
) -> bool:
    """
    Upload a single plot to W&B.

    Args:
        plot_path: Path to plot image file
        wandb_key: Key for W&B logging (e.g., "time_series/brier_over_time")
        caption: Optional caption for the plot

    Returns:
        True if upload succeeded, False otherwise
    """
    if not check_wandb_available():
        return False

    plot_path = Path(plot_path)

    if not plot_path.exists():
        logger.error(f"Plot file not found: {plot_path}")
        return False

    try:
        wandb.log({wandb_key: wandb.Image(str(plot_path), caption=caption)})
        logger.info(f"Uploaded plot to W&B: {wandb_key}")
        return True
    except Exception as e:
        logger.error(f"Failed to upload plot {wandb_key}: {e}")
        return False


def upload_plots_batch(
    plots: dict[str, Union[str, Path]],
    prefix: Optional[str] = None,
) -> dict[str, bool]:
    """
    Upload multiple plots to W&B in a single batch.

    Args:
        plots: Dictionary mapping W&B keys to plot file paths
               e.g., {"brier_time_series": "path/to/plot.png", ...}
        prefix: Optional prefix to prepend to all keys (e.g., "grid_search/")

    Returns:
        Dictionary mapping keys to success status (True/False)
    """
    if not check_wandb_available():
        return dict.fromkeys(plots, False)

    results = {}

    wandb_dict = {}
    for key, plot_path in plots.items():
        plot_path = Path(plot_path)

        if not plot_path.exists():
            logger.error(f"Plot file not found: {plot_path}")
            results[key] = False
            continue

        # Add prefix if provided
        wandb_key = f"{prefix}{key}" if prefix else key

        try:
            wandb_dict[wandb_key] = wandb.Image(str(plot_path))
            results[key] = True
        except Exception as e:
            logger.error(f"Failed to prepare plot {key}: {e}")
            results[key] = False

    # Batch upload
    if wandb_dict:
        try:
            wandb.log(wandb_dict)
            logger.info(f"Uploaded {len(wandb_dict)} plots to W&B")
        except Exception as e:
            logger.error(f"Batch upload failed: {e}")
            # Mark all as failed
            for key in wandb_dict:
                results[key] = False

    return results


def upload_table(
    data: list[dict[str, Any]],
    columns: list[str],
    wandb_key: str,
) -> bool:
    """
    Upload a data table to W&B.

    Args:
        data: List of dictionaries with row data
        columns: Column names
        wandb_key: Key for W&B logging (e.g., "grid_search/top_trials")

    Returns:
        True if upload succeeded, False otherwise
    """
    if not check_wandb_available():
        return False

    try:
        table = wandb.Table(columns=columns, data=[list(row.values()) for row in data])
        wandb.log({wandb_key: table})
        logger.info(f"Uploaded table to W&B: {wandb_key}")
        return True
    except Exception as e:
        logger.error(f"Failed to upload table {wandb_key}: {e}")
        return False


def fetch_run_history(
    run_id: str,
    project: str,
    entity: Optional[str] = None,
    keys: Optional[list[str]] = None,
) -> dict[str, Any]:
    """
    Fetch history from a specific W&B run.

    Args:
        run_id: W&B run ID
        project: W&B project name
        entity: W&B entity (username/team), None for default
        keys: Optional list of metric keys to fetch (all if None)

    Returns:
        Dictionary with run metadata and history dataframe
    """
    if not WANDB_AVAILABLE:
        logger.error("W&B not available")
        return {}

    try:
        api = wandb.Api()
        run = api.run(f"{entity}/{project}/{run_id}" if entity else f"{project}/{run_id}")

        history = run.history(keys=keys) if keys else run.history()

        return {
            "name": run.name,
            "id": run.id,
            "config": run.config,
            "summary": run.summary._json_dict,
            "history": history,
        }
    except Exception as e:
        logger.error(f"Failed to fetch run {run_id}: {e}")
        return {}


def fetch_multiple_runs(
    project: str,
    entity: Optional[str] = None,
    filters: Optional[dict[str, Any]] = None,
    limit: int = 100,
) -> list[dict[str, Any]]:
    """
    Fetch multiple runs from a W&B project.

    Args:
        project: W&B project name
        entity: W&B entity (username/team), None for default
        filters: Optional W&B filters dict (e.g., {"config.learning_rate": 0.01})
        limit: Maximum number of runs to fetch

    Returns:
        List of run dictionaries with metadata
    """
    if not WANDB_AVAILABLE:
        logger.error("W&B not available")
        return []

    try:
        api = wandb.Api()
        runs = api.runs(
            f"{entity}/{project}" if entity else project,
            filters=filters,
        )

        results = []
        for run in runs[:limit]:
            results.append(
                {
                    "name": run.name,
                    "id": run.id,
                    "config": run.config,
                    "summary": run.summary._json_dict,
                    "state": run.state,
                    "created_at": run.created_at,
                }
            )

        logger.info(f"Fetched {len(results)} runs from {project}")
        return results
    except Exception as e:
        logger.error(f"Failed to fetch runs from {project}: {e}")
        return []


def compare_runs(
    run_ids: list[str],
    project: str,
    entity: Optional[str] = None,
    metric: str = "validation/brier",
) -> dict[str, Any]:
    """
    Compare multiple runs by a specific metric.

    Args:
        run_ids: List of W&B run IDs
        project: W&B project name
        entity: W&B entity (username/team)
        metric: Metric key to compare

    Returns:
        Dictionary with comparison data and best run info
    """
    if not WANDB_AVAILABLE:
        logger.error("W&B not available")
        return {}

    try:
        api = wandb.Api()

        comparison = []
        for run_id in run_ids:
            run = api.run(f"{entity}/{project}/{run_id}" if entity else f"{project}/{run_id}")

            metric_value = run.summary.get(metric)

            comparison.append(
                {
                    "run_id": run_id,
                    "name": run.name,
                    "metric_value": metric_value,
                    "config": run.config,
                }
            )

        # Sort by metric (assuming lower is better for Brier)
        comparison.sort(key=lambda x: x["metric_value"] if x["metric_value"] is not None else float("inf"))

        best_run = comparison[0] if comparison else None

        logger.info(f"Compared {len(comparison)} runs by {metric}")

        return {
            "metric": metric,
            "comparison": comparison,
            "best_run": best_run,
        }
    except Exception as e:
        logger.error(f"Failed to compare runs: {e}")
        return {}


def log_model_summary(
    model_name: str,
    model_type: str,
    metrics: dict[str, float],
    config: dict[str, Any],
    artifacts_dir: Optional[Union[str, Path]] = None,
) -> bool:
    """
    Log comprehensive model summary to W&B.

    Args:
        model_name: Name of the model
        model_type: Model type (e.g., "lightgbm", "xgboost")
        metrics: Dictionary of evaluation metrics
        config: Model configuration/hyperparameters
        artifacts_dir: Optional directory containing model artifacts to upload

    Returns:
        True if logging succeeded, False otherwise
    """
    if not check_wandb_available():
        return False

    try:
        # Log config
        wandb.config.update(
            {
                "model_name": model_name,
                "model_type": model_type,
                **config,
            }
        )

        # Log metrics
        wandb.log(metrics)

        # Log artifacts if provided
        if artifacts_dir:
            artifacts_dir = Path(artifacts_dir)
            if artifacts_dir.exists():
                artifact = wandb.Artifact(f"{model_name}_artifacts", type="model")
                artifact.add_dir(str(artifacts_dir))
                wandb.log_artifact(artifact)

        logger.info(f"Logged model summary for {model_name}")
        return True
    except Exception as e:
        logger.error(f"Failed to log model summary: {e}")
        return False


def init_wandb_run(
    project: str,
    name: str,
    config: dict[str, Any],
    entity: Optional[str] = None,
    tags: Optional[list[str]] = None,
    notes: Optional[str] = None,
) -> bool:
    """
    Initialize a W&B run with standardized configuration.

    Args:
        project: W&B project name
        name: Run name
        config: Configuration dictionary
        entity: W&B entity (username/team)
        tags: Optional list of tags
        notes: Optional run notes

    Returns:
        True if initialization succeeded, False otherwise
    """
    if not WANDB_AVAILABLE:
        logger.error("W&B not available")
        return False

    try:
        wandb.init(
            project=project,
            entity=entity,
            name=name,
            config=config,
            tags=tags,
            notes=notes,
        )

        logger.info(f"Initialized W&B run: {name}")
        return True
    except Exception as e:
        logger.error(f"Failed to initialize W&B run: {e}")
        return False


def finish_wandb_run() -> bool:
    """
    Finish the current W&B run.

    Returns:
        True if finished successfully, False otherwise
    """
    if not check_wandb_available():
        return False

    try:
        wandb.finish()
        logger.info("Finished W&B run")
        return True
    except Exception as e:
        logger.error(f"Failed to finish W&B run: {e}")
        return False
