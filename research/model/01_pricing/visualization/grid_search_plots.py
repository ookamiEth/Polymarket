#!/usr/bin/env python3
"""
Grid Search Visualization Module.

Provides plotting capabilities for comparing multiple grid search trials:
- Training curves across trials (from W&B)
- Grid search summary dashboard
- Hyperparameter sensitivity analysis
- Trial comparison tables

Usage:
    from visualization.grid_search_plots import plot_training_curves_from_wandb
    plot_training_curves_from_wandb(project="lightgbm-residual-tuning", trials=[1,2,3])
"""

import logging
from pathlib import Path
from typing import Any, Optional

import matplotlib.pyplot as plt
import numpy as np
import polars as pl

from visualization.plot_config import (
    COLORS,
    FONT_SIZES,
    apply_plot_style,
    get_plot_output_path,
    get_trial_color,
)

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
    logger.warning("W&B not available - training curves will not work")


def plot_training_curves_from_wandb(
    project: str = "lightgbm-residual-tuning",
    entity: Optional[str] = None,
    trial_prefix: str = "trial_",
    metric_name: str = "validation/brier",
    output_path: Optional[str] = None,
    wandb_log: bool = False,
) -> None:
    """
    Plot training curves (validation Brier) across all grid search trials.

    Fetches run histories from W&B and overlays validation curves to compare
    convergence speed, stability, and final performance.

    Args:
        project: W&B project name
        entity: W&B entity (username/team), None for default
        trial_prefix: Prefix for trial run names (e.g., "trial_")
        metric_name: Metric to plot (e.g., "validation/brier")
        output_path: Output file path, auto-generated if None
        wandb_log: Whether to upload result to W&B

    Output:
        - Single plot with overlaid training curves (one line per trial)
        - Color-coded by trial number
        - Legend showing trial number + final Brier score
        - Saves to results/plots/grid_search/training_curves.png
    """
    if not WANDB_AVAILABLE:
        logger.error("W&B not installed - cannot fetch training curves")
        return

    apply_plot_style()

    logger.info(f"Fetching training curves from W&B project: {project}")

    # Fetch all runs matching trial prefix
    api = wandb.Api()
    runs = api.runs(f"{entity}/{project}" if entity else project)

    trial_runs = [r for r in runs if r.name.startswith(trial_prefix)]
    logger.info(f"Found {len(trial_runs)} trial runs")

    if not trial_runs:
        logger.warning(f"No runs found with prefix '{trial_prefix}'")
        return

    # Create figure
    fig, ax = plt.subplots(figsize=(12, 7))

    trial_data = []

    for run in sorted(trial_runs, key=lambda r: r.name):
        # Extract trial number from name
        trial_name = run.name
        try:
            trial_num = int(trial_name.replace(trial_prefix, "").split("_")[0])
        except (ValueError, IndexError):
            logger.warning(f"Cannot parse trial number from: {trial_name}")
            continue

        # Fetch history
        history = run.history(keys=[metric_name, "_step"])

        if history.empty or metric_name not in history.columns:
            logger.warning(f"No {metric_name} data for {trial_name}")
            continue

        # Extract metrics
        steps = history["_step"].values
        brier = history[metric_name].values

        # Remove NaN values
        valid_mask = ~np.isnan(brier)
        steps = steps[valid_mask]
        brier = brier[valid_mask]

        if len(steps) == 0:
            continue

        # Plot curve
        color = get_trial_color(trial_num)
        final_brier = brier[-1]

        ax.plot(
            steps,
            brier,
            color=color,
            linewidth=1.5,
            alpha=0.8,
            label=f"Trial {trial_num}: {final_brier:.4f}",
        )

        trial_data.append(
            {
                "trial_num": trial_num,
                "final_brier": final_brier,
                "steps": len(steps),
                "min_brier": np.min(brier),
            }
        )

    # Styling
    ax.set_xlabel("Training Iteration", fontsize=FONT_SIZES["label"])
    ax.set_ylabel("Validation Brier Score", fontsize=FONT_SIZES["label"])
    ax.set_title(
        f"Training Curves: Grid Search Comparison ({len(trial_data)} trials)",
        fontsize=FONT_SIZES["title"],
        pad=15,
    )

    # Legend (outside plot to avoid occlusion)
    ax.legend(
        loc="center left",
        bbox_to_anchor=(1.02, 0.5),
        fontsize=FONT_SIZES["legend"] - 1,
        frameon=True,
        framealpha=0.9,
    )

    # Grid
    ax.grid(True, alpha=0.2, linewidth=0.5)

    # Annotations
    best_trial = min(trial_data, key=lambda x: x["final_brier"])
    ax.annotate(
        f"Best: Trial {best_trial['trial_num']} ({best_trial['final_brier']:.4f})",
        xy=(0.02, 0.98),
        xycoords="axes fraction",
        fontsize=FONT_SIZES["annotation"],
        verticalalignment="top",
        bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.8},
    )

    # Save
    if output_path is None:
        output_path = str(get_plot_output_path("grid_search", "training_curves.png"))

    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    logger.info(f"Saved training curves to {output_path}")

    # Upload to W&B
    if wandb_log and wandb.run:
        wandb.log({"grid_search/training_curves": wandb.Image(fig)})

    plt.close(fig)


def plot_grid_search_summary(
    results_file: str,
    output_path: Optional[str] = None,
    wandb_log: bool = True,
) -> None:
    """
    Create comprehensive grid search summary dashboard.

    Creates 2x2 subplot grid:
    - Top-left: Bar chart of Brier improvement by trial
    - Top-right: Scatter of runtime vs performance
    - Bottom-left: Hyperparameter sensitivity (learning rate vs Brier)
    - Bottom-right: Table of top 5 trials with hyperparameters

    Args:
        results_file: Path to grid search results parquet with columns:
            - trial_num
            - test_brier
            - baseline_brier
            - improvement_pct
            - runtime_seconds
            - learning_rate, num_leaves, max_depth, min_child_samples
        output_path: Output file path, auto-generated if None
        wandb_log: Whether to upload result to W&B

    Output:
        - 2x2 dashboard PNG
        - Saves to results/plots/grid_search/summary_dashboard.png
    """
    apply_plot_style()

    logger.info(f"Creating grid search summary from {results_file}")

    # Load results
    if not Path(results_file).exists():
        logger.error(f"Results file not found: {results_file}")
        return

    df = pl.read_parquet(results_file)
    logger.info(f"Loaded {len(df)} trial results")

    # Check required columns
    required_cols = [
        "trial_num",
        "test_brier",
        "baseline_brier",
        "improvement_pct",
        "runtime_seconds",
    ]
    missing_cols = [c for c in required_cols if c not in df.columns]
    if missing_cols:
        logger.error(f"Missing required columns: {missing_cols}")
        return

    # Sort by improvement
    df = df.sort("improvement_pct", descending=True)

    # Create 2x2 subplot
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle(
        f"Grid Search Summary: {len(df)} Trials",
        fontsize=FONT_SIZES["title"] + 2,
        fontweight="bold",
    )

    # --- Top-left: Bar chart of improvements ---
    ax = axes[0, 0]

    trial_nums = df["trial_num"].to_list()
    improvements = df["improvement_pct"].to_list()

    colors = [get_trial_color(t) for t in trial_nums]

    bars = ax.bar(trial_nums, improvements, color=colors, alpha=0.8, edgecolor="white", linewidth=1)

    # Highlight best trial
    best_idx = np.argmax(improvements)
    bars[best_idx].set_edgecolor(COLORS["success"])
    bars[best_idx].set_linewidth(3)

    ax.set_xlabel("Trial Number", fontsize=FONT_SIZES["label"])
    ax.set_ylabel("Brier Improvement (%)", fontsize=FONT_SIZES["label"])
    ax.set_title("Performance by Trial", fontsize=FONT_SIZES["title"])
    ax.grid(True, alpha=0.2, axis="y")

    # Annotate best
    ax.annotate(
        f"Best: {improvements[best_idx]:.2f}%",
        xy=(trial_nums[best_idx], improvements[best_idx]),
        xytext=(0, 10),
        textcoords="offset points",
        ha="center",
        fontsize=FONT_SIZES["annotation"],
        color=COLORS["success"],
        fontweight="bold",
    )

    # --- Top-right: Runtime vs Performance scatter ---
    ax = axes[0, 1]

    runtimes = df["runtime_seconds"].to_numpy() / 60  # Convert to minutes
    improvements_arr = df["improvement_pct"].to_numpy()

    scatter = ax.scatter(
        runtimes,
        improvements_arr,
        c=[get_trial_color(t) for t in trial_nums],
        s=150,
        alpha=0.7,
        edgecolor="white",
        linewidth=1.5,
    )

    # Annotate trials
    for i, trial_num in enumerate(trial_nums):
        ax.annotate(
            str(trial_num),
            (runtimes[i], improvements_arr[i]),
            fontsize=FONT_SIZES["annotation"] - 1,
            ha="center",
            va="center",
            color="white",
            fontweight="bold",
        )

    ax.set_xlabel("Runtime (minutes)", fontsize=FONT_SIZES["label"])
    ax.set_ylabel("Brier Improvement (%)", fontsize=FONT_SIZES["label"])
    ax.set_title("Runtime vs Performance", fontsize=FONT_SIZES["title"])
    ax.grid(True, alpha=0.2)

    # --- Bottom-left: Hyperparameter sensitivity ---
    ax = axes[1, 0]

    # Extract hyperparameters (if available)
    if "learning_rate" in df.columns and "num_leaves" in df.columns:
        learning_rates = df["learning_rate"].to_numpy()
        num_leaves = df["num_leaves"].to_numpy()

        # Scatter colored by improvement
        scatter = ax.scatter(
            learning_rates,
            improvements_arr,
            c=num_leaves,
            s=150,
            alpha=0.7,
            cmap="viridis",
            edgecolor="white",
            linewidth=1.5,
        )

        # Colorbar
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label("num_leaves", fontsize=FONT_SIZES["label"])

        ax.set_xlabel("Learning Rate", fontsize=FONT_SIZES["label"])
        ax.set_ylabel("Brier Improvement (%)", fontsize=FONT_SIZES["label"])
        ax.set_title("Hyperparameter Sensitivity", fontsize=FONT_SIZES["title"])
        ax.grid(True, alpha=0.2)
    else:
        # No hyperparameter data - show improvement distribution
        ax.hist(improvements_arr, bins=15, color=COLORS["primary"], alpha=0.7, edgecolor="white")
        ax.axvline(
            np.mean(improvements_arr),
            color=COLORS["danger"],
            linestyle="--",
            linewidth=2,
            label=f"Mean: {np.mean(improvements_arr):.2f}%",
        )
        ax.set_xlabel("Brier Improvement (%)", fontsize=FONT_SIZES["label"])
        ax.set_ylabel("Count", fontsize=FONT_SIZES["label"])
        ax.set_title("Improvement Distribution", fontsize=FONT_SIZES["title"])
        ax.legend(fontsize=FONT_SIZES["legend"])
        ax.grid(True, alpha=0.2, axis="y")

    # --- Bottom-right: Top trials table ---
    ax = axes[1, 1]
    ax.axis("off")

    # Prepare table data (top 5 trials)
    top_trials = df.head(5)

    table_data = []
    headers = ["Trial", "Brier", "Improv%", "Runtime"]

    for row in top_trials.iter_rows(named=True):
        table_data.append(
            [
                str(row["trial_num"]),
                f"{row['test_brier']:.4f}",
                f"{row['improvement_pct']:+.2f}%",
                f"{row['runtime_seconds'] / 60:.1f}m",
            ]
        )

    # Add hyperparameters if available
    if "learning_rate" in df.columns:
        headers.extend(["LR", "Leaves"])
        for i, row in enumerate(top_trials.iter_rows(named=True)):
            table_data[i].extend(
                [
                    f"{row['learning_rate']:.3f}",
                    str(row.get("num_leaves", "N/A")),
                ]
            )

    # Create table
    table = ax.table(
        cellText=table_data,
        colLabels=headers,
        cellLoc="center",
        loc="center",
        bbox=[0, 0, 1, 1],
    )

    table.auto_set_font_size(False)
    table.set_fontsize(FONT_SIZES["annotation"])
    table.scale(1, 2.5)

    # Style header
    for i in range(len(headers)):
        cell = table[(0, i)]
        cell.set_facecolor(COLORS["primary"])
        cell.set_text_props(weight="bold", color="white")

    # Color rows by rank
    for i in range(len(table_data)):
        for j in range(len(headers)):
            cell = table[(i + 1, j)]
            if i == 0:  # Best trial
                cell.set_facecolor("#003300")  # Dark green
            elif i == 1:
                cell.set_facecolor("#002200")
            else:
                cell.set_facecolor("#001100")

    ax.set_title("Top 5 Trials", fontsize=FONT_SIZES["title"], pad=20)

    # Adjust layout
    plt.tight_layout()

    # Save
    if output_path is None:
        output_path = str(get_plot_output_path("grid_search", "summary_dashboard.png"))

    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    logger.info(f"Saved grid search summary to {output_path}")

    # Upload to W&B
    if wandb_log and wandb.run:
        wandb.log({"grid_search/summary_dashboard": wandb.Image(fig)})

    plt.close(fig)


def plot_model_complexity(
    results_file: str,
    output_path: Optional[str] = None,
    wandb_log: bool = True,
) -> dict[str, Any]:
    """
    Plot model complexity vs performance to show bias-variance tradeoff.

    Shows training and test performance across different model complexities
    to identify the sweet spot and detect overfitting.

    Args:
        results_file: Path to grid search results parquet with columns:
            - trial_num
            - num_leaves, max_depth (hyperparameters)
            - test_brier, train_brier (optional)
        output_path: Output file path
        wandb_log: Whether to upload to W&B

    Returns:
        Dictionary with complexity analysis results
    """
    apply_plot_style()

    logger.info("Generating Model Complexity Graph...")

    # Load results
    if not Path(results_file).exists():
        logger.error(f"Results file not found: {results_file}")
        return {"error": "File not found"}

    df = pl.read_parquet(results_file)
    logger.info(f"Loaded {len(df)} trial results")

    # Check required columns
    required_cols = ["trial_num", "test_brier"]
    missing_cols = [c for c in required_cols if c not in df.columns]
    if missing_cols:
        logger.error(f"Missing required columns: {missing_cols}")
        return {"error": f"Missing columns: {missing_cols}"}

    # Calculate complexity metric
    if "num_leaves" in df.columns and "max_depth" in df.columns:
        # Complexity = num_leaves * max_depth (rough proxy for model capacity)
        df = df.with_columns([(pl.col("num_leaves") * pl.col("max_depth")).alias("complexity")])
    elif "num_leaves" in df.columns:
        df = df.with_columns([pl.col("num_leaves").alias("complexity")])
    else:
        logger.error("Neither num_leaves nor max_depth found in results")
        return {"error": "Missing hyperparameter columns"}

    # Sort by complexity
    df = df.sort("complexity")

    # Extract data
    complexity = df["complexity"].to_numpy()
    test_brier = df["test_brier"].to_numpy()
    trial_nums = df["trial_num"].to_numpy()

    # Check if train_brier is available
    has_train = "train_brier" in df.columns
    if has_train:
        train_brier = df["train_brier"].to_numpy()

    # Create plot
    fig, ax = plt.subplots(figsize=(12, 7))

    # Plot test error
    ax.plot(
        complexity,
        test_brier,
        marker="o",
        markersize=8,
        linewidth=2.5,
        alpha=0.9,
        color=COLORS["danger"],
        label="Test Brier (generalization)",
    )

    # Plot train error if available
    if has_train:
        ax.plot(
            complexity,
            train_brier,
            marker="s",
            markersize=7,
            linewidth=2,
            alpha=0.7,
            linestyle="--",
            color=COLORS["primary"],
            label="Train Brier (fit)",
        )

    # Annotate trial numbers
    for _i, (comp, test_b, trial_num) in enumerate(zip(complexity, test_brier, trial_nums)):
        ax.annotate(
            str(trial_num),
            (comp, test_b),
            textcoords="offset points",
            xytext=(0, 10),
            ha="center",
            fontsize=FONT_SIZES["annotation"] - 1,
            color=COLORS["danger"],
            alpha=0.7,
        )

    # Find optimal complexity (lowest test error)
    optimal_idx = np.argmin(test_brier)
    optimal_complexity = complexity[optimal_idx]
    optimal_brier = test_brier[optimal_idx]
    optimal_trial = trial_nums[optimal_idx]

    # Highlight optimal point
    ax.plot(
        optimal_complexity,
        optimal_brier,
        marker="*",
        markersize=20,
        color=COLORS["success"],
        markeredgecolor="white",
        markeredgewidth=2,
        label=f"Optimal (Trial {optimal_trial})",
        zorder=10,
    )

    # Styling
    ax.set_xlabel("Model Complexity (num_leaves × max_depth)", fontsize=FONT_SIZES["label"], fontweight="bold")
    ax.set_ylabel("Brier Score (lower is better)", fontsize=FONT_SIZES["label"], fontweight="bold")
    ax.set_title(
        "Model Complexity vs Performance\n(Bias-Variance Tradeoff)",
        fontsize=FONT_SIZES["title"] + 2,
        pad=15,
        fontweight="bold",
    )
    ax.legend(loc="best", fontsize=FONT_SIZES["legend"])
    ax.grid(True, alpha=0.2, linewidth=0.5)

    # Add shaded regions
    if len(complexity) > 3:
        # Underfit region (low complexity)
        ax.axvspan(
            float(complexity.min()),
            float(complexity[len(complexity) // 3]),
            alpha=0.1,
            color=COLORS["warning"],
            label="Underfit Risk",
        )

        # Overfit region (high complexity)
        ax.axvspan(
            float(complexity[2 * len(complexity) // 3]),
            float(complexity.max()),
            alpha=0.1,
            color=COLORS["danger"],
            label="Overfit Risk",
        )

        # Recreate legend to include regions
        ax.legend(loc="best", fontsize=FONT_SIZES["legend"])

    # Annotations
    if has_train:
        # Calculate gap (train-test) at optimal point
        train_test_gap = abs(test_brier[optimal_idx] - train_brier[optimal_idx])
        gap_pct = train_test_gap / test_brier[optimal_idx] * 100

        ax.annotate(
            f"Optimal Complexity: {optimal_complexity:.0f}\n"
            f"Test Brier: {optimal_brier:.4f}\n"
            f"Train-Test Gap: {gap_pct:.1f}%",
            xy=(0.05, 0.95),
            xycoords="axes fraction",
            fontsize=FONT_SIZES["annotation"],
            verticalalignment="top",
            bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.8},
        )
    else:
        ax.annotate(
            f"Optimal Complexity: {optimal_complexity:.0f}\nTest Brier: {optimal_brier:.4f}",
            xy=(0.05, 0.95),
            xycoords="axes fraction",
            fontsize=FONT_SIZES["annotation"],
            verticalalignment="top",
            bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.8},
        )

    plt.tight_layout()

    # Save
    if output_path is None:
        output_path = str(get_plot_output_path("grid_search", "complexity_graph.png"))

    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    logger.info(f"Saved Model Complexity Graph to {output_path}")

    # Summary statistics
    results = {
        "optimal_complexity": float(optimal_complexity),
        "optimal_test_brier": float(optimal_brier),
        "optimal_trial": int(optimal_trial),
        "complexity_range": [float(complexity.min()), float(complexity.max())],
    }

    if has_train:
        results["optimal_train_brier"] = float(train_brier[optimal_idx])
        results["train_test_gap"] = float(train_test_gap)

    # Log summary
    logger.info("\nModel Complexity Analysis:")
    logger.info(f"  Complexity range:     {complexity.min():.0f} to {complexity.max():.0f}")
    logger.info(f"  Optimal complexity:   {optimal_complexity:.0f} (Trial {optimal_trial})")
    logger.info(f"  Optimal test Brier:   {optimal_brier:.4f}")
    if has_train:
        logger.info(f"  Train-test gap:       {gap_pct:.1f}%")

    # Warnings
    if has_train and gap_pct > 10:
        logger.warning(f"⚠️  Large train-test gap ({gap_pct:.1f}%). Model may be overfitting.")

    # Upload to W&B
    if wandb_log and wandb.run:
        wandb.log({"grid_search/complexity_graph": wandb.Image(fig)})

    plt.close(fig)

    return results


def generate_grid_search_report(
    results_file: str,
    project: str = "lightgbm-residual-tuning",
    entity: Optional[str] = None,
    output_dir: str = "results/plots/grid_search/",
    wandb_log: bool = True,
) -> dict[str, Any]:
    """
    Generate complete grid search visualization report.

    Creates:
    - Training curves from W&B
    - Grid search summary dashboard
    - Model complexity graph (bias-variance tradeoff)

    Args:
        results_file: Path to grid search results parquet
        project: W&B project name
        entity: W&B entity (username/team)
        output_dir: Output directory for plots
        wandb_log: Whether to upload to W&B

    Returns:
        Dictionary with summary statistics
    """
    logger.info("=" * 80)
    logger.info("GENERATING GRID SEARCH REPORT")
    logger.info("=" * 80)

    output_dir_path = Path(output_dir)
    output_dir_path.mkdir(parents=True, exist_ok=True)

    # 1. Training curves from W&B
    logger.info("\n[1/3] Plotting training curves from W&B...")
    try:
        plot_training_curves_from_wandb(
            project=project,
            entity=entity,
            output_path=str(output_dir_path / "training_curves.png"),
            wandb_log=wandb_log,
        )
    except Exception as e:
        logger.error(f"Failed to plot training curves: {e}")

    # 2. Grid search summary
    logger.info("\n[2/3] Creating grid search summary dashboard...")
    try:
        plot_grid_search_summary(
            results_file=results_file,
            output_path=str(output_dir_path / "summary_dashboard.png"),
            wandb_log=wandb_log,
        )
    except Exception as e:
        logger.error(f"Failed to create summary dashboard: {e}")

    # 3. Model complexity graph
    logger.info("\n[3/3] Generating model complexity graph...")
    try:
        complexity_results = plot_model_complexity(
            results_file=results_file,
            output_path=str(output_dir_path / "complexity_graph.png"),
            wandb_log=wandb_log,
        )
        logger.info(f"Optimal complexity: {complexity_results.get('optimal_complexity', 'N/A')}")
        logger.info(f"Optimal Brier: {complexity_results.get('optimal_brier', 'N/A'):.4f}")
    except Exception as e:
        logger.error(f"Failed to generate complexity graph: {e}")

    # Load results for summary stats
    if Path(results_file).exists():
        df = pl.read_parquet(results_file)

        # Extract metrics with None checks
        # Polars returns numeric types for these columns, but Pyright sees PythonLiteral union
        best_brier_val = df["test_brier"].min()
        best_improvement_val = df["improvement_pct"].max()
        mean_improvement_val = df["improvement_pct"].mean()
        total_runtime_val = df["runtime_seconds"].sum()

        summary = {
            "total_trials": len(df),
            "best_brier": float(best_brier_val) if best_brier_val is not None else 0.0,  # type: ignore
            "best_improvement": float(best_improvement_val) if best_improvement_val is not None else 0.0,  # type: ignore
            "mean_improvement": float(mean_improvement_val) if mean_improvement_val is not None else 0.0,  # type: ignore
            "total_runtime_hours": float(total_runtime_val / 3600) if total_runtime_val is not None else 0.0,  # type: ignore
        }

        logger.info("\n" + "=" * 80)
        logger.info("GRID SEARCH SUMMARY")
        logger.info("=" * 80)
        logger.info(f"Total trials:        {summary['total_trials']}")
        logger.info(f"Best Brier:          {summary['best_brier']:.4f}")
        logger.info(f"Best improvement:    {summary['best_improvement']:+.2f}%")
        logger.info(f"Mean improvement:    {summary['mean_improvement']:+.2f}%")
        logger.info(f"Total runtime:       {summary['total_runtime_hours']:.1f} hours")
        logger.info("=" * 80)

        return summary

    return {}


def main() -> None:
    """CLI entry point for grid search visualizations."""
    import argparse

    parser = argparse.ArgumentParser(description="Grid search visualization tools")
    parser.add_argument(
        "--results",
        type=str,
        required=True,
        help="Path to grid search results parquet file",
    )
    parser.add_argument(
        "--project",
        type=str,
        default="lightgbm-residual-tuning",
        help="W&B project name",
    )
    parser.add_argument(
        "--entity",
        type=str,
        help="W&B entity (username/team)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results/plots/grid_search/",
        help="Output directory for plots",
    )
    parser.add_argument(
        "--no-wandb",
        action="store_true",
        help="Disable W&B upload",
    )

    args = parser.parse_args()

    generate_grid_search_report(
        results_file=args.results,
        project=args.project,
        entity=args.entity,
        output_dir=args.output_dir,
        wandb_log=not args.no_wandb,
    )


if __name__ == "__main__":
    main()
