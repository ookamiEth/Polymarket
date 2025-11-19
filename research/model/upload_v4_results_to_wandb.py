#!/usr/bin/env python3
"""
Upload existing V4 model results retroactively to Weights & Biases.

This script uploads all V4 hybrid regime model artifacts, metrics, and visualizations
to W&B for interactive analysis and tracking.
"""

import os
from pathlib import Path
from typing import Any

import polars as pl
import wandb
import yaml
from dotenv import load_dotenv

# Type alias for W&B run
WandbRun = Any

# Load environment variables
load_dotenv("/home/ubuntu/Polymarket/.env")


def load_config() -> dict[str, Any]:
    """Load the V4 config file."""
    config_path = Path("config/multi_horizon_regime_config.yaml")
    with open(config_path) as f:
        return yaml.safe_load(f)


def upload_evaluation_metrics(run: WandbRun) -> None:
    """Upload per-model evaluation metrics."""
    print("\n📊 Uploading evaluation metrics...")

    eval_csv = Path("results/multi_horizon_hybrid_v4/evaluations/evaluation_per_model.csv")
    if not eval_csv.exists():
        print(f"  ⚠️  Evaluation CSV not found: {eval_csv}")
        return

    # Read evaluation data
    df = pl.read_csv(eval_csv)
    print(f"  ✅ Loaded {len(df)} model evaluations")

    # Log as W&B table
    run.log({"evaluation_per_model": wandb.Table(dataframe=df.to_pandas())})

    # Log summary metrics
    summary_yaml = Path("results/multi_horizon_hybrid_v4/evaluations/evaluation_summary.yaml")
    if summary_yaml.exists():
        with open(summary_yaml) as f:
            summary = yaml.safe_load(f)
        run.log(summary)
        print("  ✅ Uploaded summary metrics")


def upload_feature_importance(run: WandbRun) -> None:
    """Upload feature importance analysis."""
    print("\n🔍 Uploading feature importance...")

    feat_csv = Path("02_analysis/feature_importance_analysis_v4.csv")
    if not feat_csv.exists():
        print(f"  ⚠️  Feature importance CSV not found: {feat_csv}")
        return

    df = pl.read_csv(feat_csv)
    print(f"  ✅ Loaded {len(df)} feature importance entries")

    # Log as W&B table
    run.log({"feature_importance_v4": wandb.Table(dataframe=df.to_pandas())})

    # Log top 20 features as bar chart
    top_20 = df.head(20).to_pandas()
    table = wandb.Table(dataframe=top_20)
    run.log({"top_20_features": wandb.plot.bar(table, "feature", "importance", title="Top 20 Feature Importances")})


def upload_model_artifacts(run: WandbRun, config: dict[str, Any]) -> None:
    """Upload model files and configs as W&B artifacts."""
    print("\n🤖 Uploading model artifacts...")

    models_dir = Path("01_pricing/models_v4")
    if not models_dir.exists():
        print(f"  ⚠️  Models directory not found: {models_dir}")
        return

    # Create artifact
    artifact = wandb.Artifact("v4-hybrid-models", type="model", description="12 LightGBM hybrid models")

    model_count = 0
    config_count = 0

    # Add all model files
    for model_file in models_dir.glob("*.txt"):
        artifact.add_file(str(model_file), name=f"models/{model_file.name}")
        model_count += 1

    # Add all config files
    for config_file in models_dir.glob("*.yaml"):
        artifact.add_file(str(config_file), name=f"configs/{config_file.name}")
        config_count += 1

    run.log_artifact(artifact)
    print(f"  ✅ Uploaded {model_count} model files and {config_count} config files")


def upload_optuna_studies(run: WandbRun) -> None:
    """Upload Optuna hyperparameter optimization results."""
    print("\n🔬 Uploading Optuna study results...")

    optuna_dir = Path("01_pricing/models_optuna")
    if not optuna_dir.exists():
        print(f"  ⚠️  Optuna directory not found: {optuna_dir}")
        return

    # Create artifact for Optuna databases
    artifact = wandb.Artifact("v4-optuna-studies", type="optuna", description="Hyperparameter optimization studies")

    db_count = 0
    for db_file in optuna_dir.glob("*.db"):
        artifact.add_file(str(db_file), name=f"studies/{db_file.name}")
        db_count += 1

    run.log_artifact(artifact)
    print(f"  ✅ Uploaded {db_count} Optuna study databases")


def upload_visualizations(run: WandbRun) -> None:
    """Upload SHAP values and other visualizations."""
    print("\n📈 Uploading visualizations...")

    shap_dir = Path("02_analysis/pricing/shap_values_v4")
    if not shap_dir.exists():
        print(f"  ⚠️  SHAP directory not found: {shap_dir}")
        return

    # Upload SHAP plots
    plot_count = 0
    for plot_file in shap_dir.glob("*.png"):
        run.log({f"shap/{plot_file.stem}": wandb.Image(str(plot_file))})
        plot_count += 1

    print(f"  ✅ Uploaded {plot_count} SHAP visualizations")

    # Upload other result plots
    results_dir = Path("results/multi_horizon_hybrid_v4")
    if results_dir.exists():
        for plot_file in results_dir.glob("*.png"):
            run.log({f"results/{plot_file.stem}": wandb.Image(str(plot_file))})
            plot_count += 1

    print(f"  ✅ Total visualizations uploaded: {plot_count}")


def upload_backtest_summary(run: WandbRun) -> None:
    """Upload aggregated backtest results (not full 66M rows)."""
    print("\n💰 Uploading backtest summary...")

    backtest_file = Path("results/production_backtest_results_v4.parquet")
    if not backtest_file.exists():
        print(f"  ⚠️  Backtest file not found: {backtest_file}")
        return

    # Load and aggregate (don't upload full 66M rows!)
    df = pl.scan_parquet(backtest_file)

    # Compute summary statistics (use prob_mid and price_mid from actual columns)
    summary_stats = (
        df.select(
            [
                pl.len().alias("total_predictions"),
                pl.col("prob_mid").mean().alias("mean_probability"),
                pl.col("prob_mid").std().alias("std_probability"),
                pl.col("price_mid").mean().alias("mean_market_price"),
                (pl.col("prob_mid") - pl.col("price_mid")).abs().mean().alias("mean_abs_error"),
            ]
        )
        .collect()
        .to_dicts()[0]
    )

    run.log({"backtest_summary": summary_stats})
    print(f"  ✅ Uploaded backtest summary ({summary_stats['total_predictions']:,} predictions)")


def upload_training_logs(run: WandbRun) -> None:
    """Upload training logs as artifacts."""
    print("\n📝 Uploading training logs...")

    logs_dir = Path("01_pricing/logs_v4")
    if not logs_dir.exists():
        print(f"  ⚠️  Logs directory not found: {logs_dir}")
        return

    artifact = wandb.Artifact("v4-training-logs", type="logs", description="Training and optimization logs")

    log_count = 0
    for log_file in logs_dir.glob("*.log"):
        artifact.add_file(str(log_file), name=f"logs/{log_file.name}")
        log_count += 1

    run.log_artifact(artifact)
    print(f"  ✅ Uploaded {log_count} log files")


def main() -> None:
    """Main upload function."""
    print("=" * 80)
    print("🚀 V4 Hybrid Model Results Upload to Weights & Biases")
    print("=" * 80)

    # Change to model directory
    os.chdir("/home/ubuntu/Polymarket/research/model")

    # Load config
    config = load_config()
    wandb_config = config["wandb"]

    # Initialize W&B run
    print(f"\n🔑 Initializing W&B (project: {wandb_config['project']})...")
    run = wandb.init(
        project=wandb_config["project"],
        entity=wandb_config["entity"],
        name="v4-baseline-retroactive-upload",
        tags=wandb_config["tags"] + ["retroactive", "baseline"],
        notes=wandb_config["notes"] + " [Retroactive upload of existing results]",
        config=config,
    )

    print(f"  ✅ W&B run initialized: {run.url}")

    # Upload all components
    try:
        upload_evaluation_metrics(run)
        upload_feature_importance(run)
        upload_model_artifacts(run, config)
        upload_optuna_studies(run)
        upload_visualizations(run)
        upload_backtest_summary(run)
        upload_training_logs(run)

        print("\n" + "=" * 80)
        print("✅ All V4 results uploaded successfully!")
        print(f"🔗 View your results at: {run.url}")
        print("=" * 80)

    except Exception as e:
        print(f"\n❌ Error during upload: {e}")
        raise

    finally:
        run.finish()
        print("\n👋 W&B run finished")


if __name__ == "__main__":
    main()
