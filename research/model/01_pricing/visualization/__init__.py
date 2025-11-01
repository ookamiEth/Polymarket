"""
Visualization module for LightGBM residual prediction model.

Provides comprehensive plotting capabilities for:
- Time-series analysis (Brier evolution, calibration by regime, prediction distribution)
- Grid search comparison (training curves, summary dashboard, complexity graph)
- Advanced diagnostics (ROC, Precision-Recall, QQ plots, Lift charts, Win rate heatmaps, SHAP)
- Simulation-based uncertainty (Bootstrap confidence intervals)
- Trading simulation (Equity curves, P&L analysis)
- W&B integration (batch uploads, run fetching, metrics logging)

Complete Feature Set:
    - Time-series plots to detect model degradation
    - Calibration by volatility regime
    - Grid search training curves from W&B
    - Statistical evaluation (ROC/AUC, Precision-Recall, QQ plots)
    - Targeting efficiency (Lift charts, Cumulative gains)
    - Regime-specific performance (Win rate heatmaps)
    - Model validation (Complexity graph, Bootstrap CI)
    - Feature explainability (SHAP dependence plots)
    - Trading backtests (Equity curves with threshold strategies)

Usage:
    # Generate all plots
    from visualization.generate_all_plots import main
    main()

    # Individual plot modules
    from visualization.time_series_plots import plot_brier_over_time
    from visualization.grid_search_plots import plot_training_curves_from_wandb, plot_model_complexity
    from visualization.advanced_diagnostics import plot_roc_curve, plot_shap_dependence_analysis
    from visualization.simulation_plots import plot_bootstrap_ci
    from visualization.trading_plots import plot_equity_curve
    from visualization.wandb_integration import upload_plots_batch
"""

# Configuration
# Advanced diagnostic plots
from visualization.advanced_diagnostics import (
    generate_diagnostics_report,
    plot_lift_chart,
    plot_precision_recall,
    plot_qq_residuals,
    plot_roc_curve,
    plot_shap_dependence_analysis,
    plot_win_rate_heatmap,
)

# Grid search plotting functions
from visualization.grid_search_plots import (
    generate_grid_search_report,
    plot_grid_search_summary,
    plot_model_complexity,
    plot_training_curves_from_wandb,
)
from visualization.plot_config import (
    COLORS,
    FONT_SIZES,
    PLOTS_DIR,
    REGIME_COLORS,
    TRIAL_COLORS,
    apply_plot_style,
    get_plot_output_path,
    get_trial_color,
)

# Simulation-based plots
from visualization.simulation_plots import (
    generate_simulation_report,
    plot_bootstrap_ci,
)

# Time-series plotting functions
from visualization.time_series_plots import (
    generate_time_series_report,
    plot_brier_over_time,
    plot_calibration_by_regime,
    plot_prediction_distribution,
)

# Trading simulation plots
from visualization.trading_plots import (
    generate_trading_report,
    plot_equity_curve,
)

# W&B integration helpers
from visualization.wandb_integration import (
    check_wandb_available,
    upload_plot,
    upload_plots_batch,
    upload_table,
)

__all__ = [
    # Configuration
    "COLORS",
    "FONT_SIZES",
    "PLOTS_DIR",
    "REGIME_COLORS",
    "TRIAL_COLORS",
    "apply_plot_style",
    "get_plot_output_path",
    "get_trial_color",
    # Time-series plots
    "plot_brier_over_time",
    "plot_calibration_by_regime",
    "plot_prediction_distribution",
    "generate_time_series_report",
    # Grid search plots
    "plot_training_curves_from_wandb",
    "plot_grid_search_summary",
    "plot_model_complexity",
    "generate_grid_search_report",
    # Advanced diagnostic plots
    "plot_roc_curve",
    "plot_precision_recall",
    "plot_qq_residuals",
    "plot_lift_chart",
    "plot_win_rate_heatmap",
    "plot_shap_dependence_analysis",
    "generate_diagnostics_report",
    # Simulation plots
    "plot_bootstrap_ci",
    "generate_simulation_report",
    # Trading plots
    "plot_equity_curve",
    "generate_trading_report",
    # W&B integration
    "check_wandb_available",
    "upload_plot",
    "upload_plots_batch",
    "upload_table",
]
