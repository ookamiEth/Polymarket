#!/usr/bin/env python3
"""
Master Visualization Orchestration Script.

Generates all critical plots for LightGBM residual prediction model:
- Time-series analysis (Brier evolution, calibration by regime, prediction distribution)
- Grid search comparison (training curves, summary dashboard)
- Diagnostics & SHAP (ROC, PR, QQ, Lift, Heatmaps, feature importance)
- Trading simulation (equity curve, P&L, Sharpe ratio)
- Uncertainty quantification (bootstrap confidence intervals)

Coordinates W&B uploads and generates comprehensive HTML report.

Usage:
    # Generate all plots with SHAP analysis
    uv run python visualization/generate_all_plots.py \
        --test-file ../results/test_predictions.parquet \
        --model-path ../results/lightgbm_model_optimized.txt \
        --grid-search-file ../results/grid_search_results.parquet \
        --wandb-project lightgbm-residual-tuning

    # Skip W&B upload
    uv run python visualization/generate_all_plots.py \
        --test-file ../results/test_predictions.parquet \
        --model-path ../results/lightgbm_model_optimized.txt \
        --no-wandb

    # Minimal run (no grid search or SHAP)
    uv run python visualization/generate_all_plots.py \
        --test-file ../results/test_predictions.parquet \
        --no-wandb
"""

import argparse
import logging
from pathlib import Path
from typing import Any

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# Import visualization modules
try:
    from visualization.advanced_diagnostics import generate_diagnostics_report
    from visualization.grid_search_plots import generate_grid_search_report
    from visualization.simulation_plots import generate_simulation_report
    from visualization.time_series_plots import generate_time_series_report
    from visualization.trading_plots import generate_trading_report
    from visualization.wandb_integration import check_wandb_available

    MODULES_AVAILABLE = True
except ImportError as e:
    logger.error(f"Failed to import visualization modules: {e}")
    logger.error("Make sure you're running from the correct directory")
    MODULES_AVAILABLE = False


def generate_html_report(
    output_file: str,
    plots_summary: dict[str, Any],
) -> None:
    """
    Generate comprehensive HTML report with all plots.

    Args:
        output_file: Path to output HTML file
        plots_summary: Dictionary with plot metadata and paths
    """
    logger.info(f"Generating HTML report: {output_file}")

    html = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>LightGBM Model Visualization Report</title>
        <style>
            body {
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                max-width: 1400px;
                margin: 0 auto;
                padding: 20px;
                background-color: #0a0a0a;
                color: #e0e0e0;
            }
            h1 {
                color: #00D4FF;
                border-bottom: 3px solid #00D4FF;
                padding-bottom: 10px;
            }
            h2 {
                color: #FF00FF;
                margin-top: 40px;
                border-bottom: 2px solid #FF00FF;
                padding-bottom: 8px;
            }
            h3 {
                color: #00FF88;
                margin-top: 30px;
            }
            .plot-container {
                margin: 20px 0;
                background-color: #1a1a1a;
                padding: 20px;
                border-radius: 8px;
                box-shadow: 0 2px 4px rgba(0, 212, 255, 0.1);
            }
            .plot-container img {
                width: 100%;
                height: auto;
                border-radius: 4px;
            }
            .metrics {
                background-color: #1a1a1a;
                padding: 15px;
                border-radius: 8px;
                margin: 20px 0;
                border-left: 4px solid #00D4FF;
            }
            .metric-row {
                display: flex;
                justify-content: space-between;
                padding: 8px 0;
                border-bottom: 1px solid #333;
            }
            .metric-label {
                font-weight: bold;
                color: #00D4FF;
            }
            .metric-value {
                color: #00FF88;
                font-family: 'Courier New', monospace;
            }
            .warning {
                background-color: #2a1a00;
                border-left: 4px solid #FFB000;
                padding: 15px;
                margin: 20px 0;
                border-radius: 4px;
            }
            .success {
                background-color: #002a00;
                border-left: 4px solid #00FF88;
                padding: 15px;
                margin: 20px 0;
                border-radius: 4px;
            }
            footer {
                margin-top: 50px;
                padding-top: 20px;
                border-top: 2px solid #333;
                text-align: center;
                color: #666;
            }
        </style>
    </head>
    <body>
        <h1>🤖 LightGBM Residual Prediction Model - Visualization Report</h1>
        <p style="color: #888; font-size: 14px;">
            Generated: {timestamp}<br>
            Project: BTC 15-minute Options Pricing
        </p>
    """

    # Time-series section
    if "time_series" in plots_summary:
        html += """
        <h2>📈 Time-Series Analysis</h2>
        <p>Temporal analysis of model performance to detect degradation and regime-specific failures.</p>
        """

        ts_summary = plots_summary["time_series"]

        if ts_summary.get("plots"):
            for plot_name, plot_path in ts_summary["plots"].items():
                plot_path = Path(plot_path)
                if plot_path.exists():
                    html += f"""
                    <div class="plot-container">
                        <h3>{plot_name.replace("_", " ").title()}</h3>
                        <img src="{plot_path.relative_to(Path(output_file).parent)}" alt="{plot_name}">
                    </div>
                    """

    # Grid search section
    if "grid_search" in plots_summary:
        html += """
        <h2>🔍 Grid Search Analysis</h2>
        <p>Comparison of {num_trials} hyperparameter configurations with training dynamics.</p>
        """.format(num_trials=plots_summary["grid_search"].get("total_trials", "N/A"))

        gs_summary = plots_summary["grid_search"]

        # Metrics
        if gs_summary.get("metrics"):
            html += '<div class="metrics">'
            for key, value in gs_summary["metrics"].items():
                html += f"""
                <div class="metric-row">
                    <span class="metric-label">{key.replace("_", " ").title()}</span>
                    <span class="metric-value">{value}</span>
                </div>
                """
            html += "</div>"

        if gs_summary.get("plots"):
            for plot_name, plot_path in gs_summary["plots"].items():
                plot_path = Path(plot_path)
                if plot_path.exists():
                    html += f"""
                    <div class="plot-container">
                        <h3>{plot_name.replace("_", " ").title()}</h3>
                        <img src="{plot_path.relative_to(Path(output_file).parent)}" alt="{plot_name}">
                    </div>
                    """

    # Warnings section
    warnings = plots_summary.get("warnings", [])
    if warnings:
        html += """
        <h2>⚠️ Warnings & Recommendations</h2>
        """
        for warning in warnings:
            html += f'<div class="warning">{warning}</div>'

    # Success messages
    successes = plots_summary.get("successes", [])
    if successes:
        html += """
        <h2>✅ Validation Checks</h2>
        """
        for success in successes:
            html += f'<div class="success">{success}</div>'

    html += """
        <footer>
            <p>Generated by LightGBM Visualization Suite</p>
            <p>For questions, contact the quant research team</p>
        </footer>
    </body>
    </html>
    """

    # Write HTML
    with open(output_file, "w") as f:
        f.write(html)

    logger.info(f"HTML report saved to {output_file}")


def main() -> None:
    """Main orchestration function."""
    parser = argparse.ArgumentParser(description="Generate all visualization plots for LightGBM model evaluation")
    parser.add_argument(
        "--test-file",
        type=str,
        required=True,
        help="Path to test predictions parquet file (test_predictions.parquet)",
    )
    parser.add_argument(
        "--grid-search-file",
        type=str,
        help="Path to grid search results parquet (optional)",
    )
    parser.add_argument(
        "--model-path",
        type=str,
        help="Path to trained model file (.pkl or .txt) for SHAP analysis (optional)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results/plots/",
        help="Output directory for all plots",
    )
    parser.add_argument(
        "--no-wandb",
        action="store_true",
        help="Disable W&B upload",
    )
    parser.add_argument(
        "--wandb-project",
        type=str,
        default="lightgbm-residual-tuning",
        help="W&B project name",
    )
    parser.add_argument(
        "--wandb-entity",
        type=str,
        help="W&B entity (username/team)",
    )
    parser.add_argument(
        "--html-report",
        type=str,
        default="results/plots/report.html",
        help="Path to HTML report output",
    )

    args = parser.parse_args()

    if not MODULES_AVAILABLE:
        logger.error("Visualization modules not available - exiting")
        return

    logger.info("=" * 80)
    logger.info("LIGHTGBM MODEL VISUALIZATION SUITE")
    logger.info("=" * 80)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    plots_summary: dict[str, Any] = {
        "timestamp": "",
        "warnings": [],
        "successes": [],
    }

    # Import datetime here to add timestamp
    from datetime import datetime

    plots_summary["timestamp"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Check W&B availability
    wandb_log = not args.no_wandb
    if wandb_log:
        if check_wandb_available():
            logger.info("✓ W&B available - plots will be uploaded")
        else:
            logger.warning("W&B not available - skipping uploads")
            wandb_log = False

    # ========================================================================
    # 1. TIME-SERIES ANALYSIS
    # ========================================================================
    logger.info("\n" + "=" * 80)
    logger.info("[1/5] GENERATING TIME-SERIES PLOTS")
    logger.info("=" * 80)

    try:
        ts_results = generate_time_series_report(
            test_file=args.test_file,
            output_dir=str(output_dir / "time_series"),
            wandb_log=wandb_log,
        )

        plots_summary["time_series"] = ts_results
        logger.info("✓ Time-series plots generated successfully")

        # Check for warnings
        if ts_results.get("warnings"):
            plots_summary["warnings"].extend(ts_results["warnings"])

    except Exception as e:
        logger.error(f"Failed to generate time-series plots: {e}")
        plots_summary["warnings"].append(f"Time-series plot generation failed: {e}")

    # ========================================================================
    # 2. GRID SEARCH ANALYSIS
    # ========================================================================
    if args.grid_search_file:
        logger.info("\n" + "=" * 80)
        logger.info("[2/5] GENERATING GRID SEARCH PLOTS")
        logger.info("=" * 80)

        try:
            gs_results = generate_grid_search_report(
                results_file=args.grid_search_file,
                project=args.wandb_project,
                entity=args.wandb_entity,
                output_dir=str(output_dir / "grid_search"),
                wandb_log=wandb_log,
            )

            plots_summary["grid_search"] = {
                "metrics": gs_results,
                "plots": {
                    "training_curves": output_dir / "grid_search" / "training_curves.png",
                    "summary_dashboard": output_dir / "grid_search" / "summary_dashboard.png",
                },
            }

            logger.info("✓ Grid search plots generated successfully")

        except Exception as e:
            logger.error(f"Failed to generate grid search plots: {e}")
            plots_summary["warnings"].append(f"Grid search plot generation failed: {e}")
    else:
        logger.info("\n[2/5] Grid search file not provided - skipping")

    # ========================================================================
    # 3. DIAGNOSTICS & SHAP ANALYSIS
    # ========================================================================
    logger.info("\n" + "=" * 80)
    logger.info("[3/5] GENERATING DIAGNOSTICS PLOTS")
    logger.info("=" * 80)

    try:
        diag_results = generate_diagnostics_report(
            test_file=args.test_file,
            model_path=args.model_path,  # Optional - SHAP will be skipped if None
            output_dir=str(output_dir / "diagnostics"),
            wandb_log=wandb_log,
        )

        plots_summary["diagnostics"] = diag_results
        logger.info("✓ Diagnostics plots generated successfully")

        # Check for warnings
        if diag_results.get("warnings"):
            plots_summary["warnings"].extend(diag_results["warnings"])

        if not args.model_path:
            plots_summary["warnings"].append("Model path not provided - SHAP analysis skipped")

    except Exception as e:
        logger.error(f"Failed to generate diagnostics plots: {e}")
        plots_summary["warnings"].append(f"Diagnostics plot generation failed: {e}")

    # ========================================================================
    # 4. TRADING SIMULATION
    # ========================================================================
    logger.info("\n" + "=" * 80)
    logger.info("[4/5] GENERATING TRADING SIMULATION PLOTS")
    logger.info("=" * 80)

    try:
        trading_results = generate_trading_report(
            test_file=args.test_file,
            output_dir=str(output_dir / "trading"),
            wandb_log=wandb_log,
        )

        plots_summary["trading"] = trading_results
        logger.info("✓ Trading simulation plots generated successfully")

    except Exception as e:
        logger.error(f"Failed to generate trading simulation plots: {e}")
        plots_summary["warnings"].append(f"Trading simulation plot generation failed: {e}")

    # ========================================================================
    # 5. UNCERTAINTY QUANTIFICATION (BOOTSTRAP)
    # ========================================================================
    logger.info("\n" + "=" * 80)
    logger.info("[5/5] GENERATING UNCERTAINTY QUANTIFICATION PLOTS")
    logger.info("=" * 80)

    try:
        sim_results = generate_simulation_report(
            test_file=args.test_file,
            output_dir=str(output_dir / "simulation"),
            wandb_log=wandb_log,
        )

        plots_summary["simulation"] = sim_results
        logger.info("✓ Uncertainty quantification plots generated successfully")

    except Exception as e:
        logger.error(f"Failed to generate uncertainty quantification plots: {e}")
        plots_summary["warnings"].append(f"Uncertainty quantification plot generation failed: {e}")

    # ========================================================================
    # 6. GENERATE HTML REPORT
    # ========================================================================
    logger.info("\n" + "=" * 80)
    logger.info("GENERATING HTML REPORT")
    logger.info("=" * 80)

    try:
        generate_html_report(
            output_file=args.html_report,
            plots_summary=plots_summary,
        )
        logger.info(f"✓ HTML report generated: {args.html_report}")
    except Exception as e:
        logger.error(f"Failed to generate HTML report: {e}")

    # ========================================================================
    # 7. SUMMARY
    # ========================================================================
    logger.info("\n" + "=" * 80)
    logger.info("VISUALIZATION SUMMARY")
    logger.info("=" * 80)

    logger.info(f"\nOutput directory: {output_dir}")
    logger.info(f"HTML report: {args.html_report}")

    if plots_summary["warnings"]:
        logger.warning(f"\n⚠️  {len(plots_summary['warnings'])} warnings detected:")
        for warning in plots_summary["warnings"]:
            logger.warning(f"  - {warning}")

    if plots_summary["successes"]:
        logger.info(f"\n✅ {len(plots_summary['successes'])} validation checks passed")

    logger.info("\n" + "=" * 80)
    logger.info("VISUALIZATION COMPLETE")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
