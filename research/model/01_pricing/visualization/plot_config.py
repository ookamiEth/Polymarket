"""
Centralized plot configuration and styling for LightGBM model visualizations.

Follows CLAUDE.md visualization standards:
- Dot sizing: 30-200 range
- Font sizes: Title 14-16pt, Labels 12pt, Ticks 10pt
- Colors: Professional quant palette
- DPI: 150-300 for publication
- Grid alpha: 0.2-0.3
"""

from pathlib import Path

# Color palette (from CLAUDE.md)
COLORS = {
    "primary": "#00D4FF",  # Cyan - main data
    "secondary": "#FF00FF",  # Magenta - secondary data
    "success": "#00FF88",  # Green - positive/improvement
    "danger": "#FF3366",  # Red - negative/baseline
    "warning": "#FFB000",  # Orange - warnings
    "info": "#00B4FF",  # Light Blue - informational
    "perfect": "#888888",  # Gray - reference lines
    "grid": "#333333",  # Dark gray - grid lines
}

# Font sizes (following standards)
FONT_SIZES = {
    "title": 14,
    "subtitle": 13,
    "label": 12,
    "tick": 10,
    "legend": 10,
    "annotation": 9,
}

# Plot styling
PLOT_STYLE = {
    "figure.dpi": 150,
    "savefig.dpi": 150,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "axes.grid": True,
    "grid.alpha": 0.3,
    "grid.linewidth": 0.5,
    "axes.linewidth": 1.0,
    "lines.linewidth": 2.0,
    "patch.linewidth": 0.5,
    "xtick.labelsize": FONT_SIZES["tick"],
    "ytick.labelsize": FONT_SIZES["tick"],
    "axes.labelsize": FONT_SIZES["label"],
    "axes.titlesize": FONT_SIZES["title"],
    "legend.fontsize": FONT_SIZES["legend"],
}

# Dot sizing for scatter plots
DOT_SIZE_RANGE = (30, 200)  # Min, max

# Output paths
RESULTS_DIR = Path(__file__).parent.parent / "results"
PLOTS_DIR = RESULTS_DIR / "plots"

# W&B project name (for uploading plots)
WANDB_PROJECT = "lightgbm-residual-tuning"


def get_plot_output_path(category: str, filename: str) -> Path:
    """
    Get standardized output path for plots.

    Args:
        category: Plot category (time_series, calibration, grid_search, etc.)
        filename: Plot filename (e.g., 'brier_over_time.png')

    Returns:
        Full path to output file
    """
    output_dir = PLOTS_DIR / category
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir / filename


def apply_plot_style() -> None:
    """Apply standardized plot style to matplotlib."""
    import matplotlib.pyplot as plt

    plt.style.use("default")  # Reset to default first
    plt.rcParams.update(PLOT_STYLE)


# Trial colors for grid search (cycle through palette)
TRIAL_COLORS = [
    COLORS["primary"],
    COLORS["secondary"],
    COLORS["success"],
    COLORS["danger"],
    COLORS["warning"],
    COLORS["info"],
]


def get_trial_color(trial_num: int) -> str:
    """Get color for a specific trial number."""
    return TRIAL_COLORS[trial_num % len(TRIAL_COLORS)]


# Regime colors
REGIME_COLORS = {
    "low_vol": COLORS["success"],  # Green - calm
    "mid_vol": COLORS["primary"],  # Cyan - normal
    "high_vol": COLORS["danger"],  # Red - volatile
    "overall": COLORS["info"],  # Blue - aggregate
}
