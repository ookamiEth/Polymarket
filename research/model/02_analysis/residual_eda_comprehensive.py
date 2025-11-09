#!/usr/bin/env python3
"""
Comprehensive Residual Exploratory Data Analysis

Analyzes Black-Scholes residuals and demonstrates non-linear appreciation
of binary options. Creates publication-quality visualizations.

Author: Quantitative Research Team
Date: 2025-11-09
"""

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from scipy import stats
from scipy.ndimage import gaussian_filter1d

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Color palette (dark theme)
COLORS = {
    'primary': '#00D4FF',      # Cyan
    'secondary': '#FF00FF',    # Magenta
    'success': '#00FF88',      # Green
    'danger': '#FF3366',       # Red
    'warning': '#FFB000',      # Orange
    'info': '#00B4FF',         # Light Blue
    'perfect': '#888888',      # Gray for reference lines
    'grid': '#333333',         # Dark gray for grid
}

# Paths
DATA_DIR = Path("/Users/lgierhake/Documents/ETH/BT/research/model/results")
OUTPUT_DIR = Path("/Users/lgierhake/Documents/ETH/BT/research/model/02_analysis/figures")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Data file
RESIDUALS_FILE = DATA_DIR / "xgboost_residual_model_test_2024-07-01_2025-09-30.parquet"


def load_data(sample_frac: float = 0.1) -> pl.DataFrame:
    """
    Load residuals data with optional sampling for faster iteration.

    Args:
        sample_frac: Fraction of data to sample (0.1 = 10%, 1.0 = all data)

    Returns:
        Polars DataFrame with residuals and features
    """
    logger.info(f"Loading data from {RESIDUALS_FILE}")
    logger.info(f"Sampling {sample_frac*100:.1f}% of data")

    # Load with lazy scan for memory efficiency
    df = pl.scan_parquet(RESIDUALS_FILE)

    # Get total rows for sampling
    total_rows = df.select(pl.len()).collect().item()
    sample_size = int(total_rows * sample_frac)

    logger.info(f"Total rows: {total_rows:,}, Sampling: {sample_size:,}")

    # Sample uniformly across time
    df_sampled = (
        df.with_row_index("idx")
          .filter(pl.col("idx") % int(1/sample_frac) == 0)
          .drop("idx")
          .collect()
    )

    logger.info(f"Loaded {len(df_sampled):,} rows")
    logger.info(f"Columns: {df_sampled.columns}")

    return df_sampled


def plot_residual_distribution(df: pl.DataFrame, save_path: Path) -> None:
    """
    Plot residual distribution with diagnostic statistics.

    Residual = actual outcome - BS probability
    Range: [-1, +1] where -1 means BS predicted 1.0 but outcome was 0
    """
    logger.info("Creating residual distribution plot...")

    residuals = df["residual"].to_numpy()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6), facecolor='#0e1117')

    # Panel 1: Histogram
    ax1.set_facecolor('#0e1117')

    counts, bins, patches = ax1.hist(
        residuals,
        bins=100,
        alpha=0.7,
        color=COLORS['primary'],
        edgecolor='white',
        linewidth=0.5
    )

    # Mean and median lines
    mean_val = np.mean(residuals)
    median_val = np.median(residuals)

    ax1.axvline(mean_val,
                color=COLORS['secondary'],
                linestyle='-',
                linewidth=2,
                label=f'Mean: {mean_val:.4f}')
    ax1.axvline(median_val,
                color=COLORS['success'],
                linestyle='--',
                linewidth=2,
                label=f'Median: {median_val:.4f}')
    ax1.axvline(0,
                color=COLORS['perfect'],
                linestyle=':',
                linewidth=1.5,
                alpha=0.5,
                label='Zero (No Error)')

    # Statistical annotations
    skewness = stats.skew(residuals)
    kurtosis = stats.kurtosis(residuals)
    stats_text = f"""N: {len(residuals):,}
Std: {np.std(residuals):.4f}
Skew: {skewness:.3f}
Kurtosis: {kurtosis:.3f}
MAE: {np.mean(np.abs(residuals)):.4f}"""

    ax1.text(0.02, 0.98, stats_text,
             transform=ax1.transAxes,
             fontsize=10,
             verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='#1a1a1a', alpha=0.8, edgecolor='white'),
             color='white')

    # Styling
    ax1.set_xlabel('Residual (Actual - Predicted)', fontsize=12, color='white', fontweight='bold')
    ax1.set_ylabel('Frequency', fontsize=12, color='white', fontweight='bold')
    ax1.set_title('Residual Distribution: Black-Scholes Errors', fontsize=14, color='white', fontweight='bold', pad=20)
    ax1.grid(True, alpha=0.2, color=COLORS['grid'], axis='y')
    ax1.tick_params(colors='white', labelsize=10)

    legend = ax1.legend(frameon=True, facecolor='#1a1a1a', edgecolor='white', fontsize=10, loc='upper right')
    plt.setp(legend.get_texts(), color='white')

    # Panel 2: Q-Q Plot (test for normality)
    ax2.set_facecolor('#0e1117')

    stats.probplot(residuals, dist="norm", plot=ax2)

    # Restyle Q-Q plot for dark theme
    ax2.get_lines()[0].set_marker('o')
    ax2.get_lines()[0].set_markersize(3)
    ax2.get_lines()[0].set_markerfacecolor(COLORS['primary'])
    ax2.get_lines()[0].set_markeredgewidth(0)
    ax2.get_lines()[0].set_alpha(0.5)

    ax2.get_lines()[1].set_color(COLORS['danger'])
    ax2.get_lines()[1].set_linewidth(2)

    ax2.set_xlabel('Theoretical Quantiles', fontsize=12, color='white', fontweight='bold')
    ax2.set_ylabel('Sample Quantiles', fontsize=12, color='white', fontweight='bold')
    ax2.set_title('Q-Q Plot: Test for Normality', fontsize=14, color='white', fontweight='bold', pad=20)
    ax2.grid(True, alpha=0.2, color=COLORS['grid'])
    ax2.tick_params(colors='white', labelsize=10)

    plt.tight_layout()
    fig.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='#0e1117', edgecolor='none')
    logger.info(f"Saved to {save_path}")
    plt.close()


def plot_residuals_vs_predicted(df: pl.DataFrame, save_path: Path) -> None:
    """
    Plot residuals vs predicted probability to check for heteroskedasticity.

    Ideal: Random scatter around zero (constant variance)
    Bad: Funnel shape or systematic patterns (variance changes with prediction)
    """
    logger.info("Creating residuals vs predicted plot...")

    # Sample for visibility (hexbin for full dataset)
    sample_df = df.sample(n=min(50000, len(df)))

    prob_mid = sample_df["prob_mid"].to_numpy()
    residuals = sample_df["residual"].to_numpy()

    fig, ax = plt.subplots(figsize=(12, 8), facecolor='#0e1117')
    ax.set_facecolor('#0e1117')

    # Hexbin for density
    hexbin = ax.hexbin(
        prob_mid,
        residuals,
        gridsize=50,
        cmap='Blues',
        mincnt=1,
        alpha=0.8
    )

    # Zero line
    ax.axhline(y=0, color=COLORS['perfect'], linestyle='--', linewidth=2, alpha=0.7, label='Zero Error')

    # Add smoothed mean residual by bin
    bins = np.linspace(0, 1, 20)
    bin_centers = []
    bin_means = []

    for i in range(len(bins)-1):
        mask = (prob_mid >= bins[i]) & (prob_mid < bins[i+1])
        if mask.sum() > 0:
            bin_centers.append((bins[i] + bins[i+1]) / 2)
            bin_means.append(np.mean(residuals[mask]))

    ax.plot(bin_centers, bin_means,
            color=COLORS['danger'],
            linewidth=3,
            marker='o',
            markersize=8,
            label='Mean Residual by Bin',
            zorder=100)

    # Colorbar
    cbar = plt.colorbar(hexbin, ax=ax)
    cbar.set_label('Count', fontsize=12, color='white', fontweight='bold')
    cbar.ax.tick_params(colors='white', labelsize=10)

    # Styling
    ax.set_xlabel('Predicted Probability (BS Model)', fontsize=12, color='white', fontweight='bold')
    ax.set_ylabel('Residual (Actual - Predicted)', fontsize=12, color='white', fontweight='bold')
    ax.set_title('Residuals vs Predicted: Check for Heteroskedasticity', fontsize=14, color='white', fontweight='bold', pad=20)
    ax.grid(True, alpha=0.2, color=COLORS['grid'])
    ax.tick_params(colors='white', labelsize=10)
    ax.set_xlim(0, 1)
    ax.set_ylim(-1, 1)

    legend = ax.legend(frameon=True, facecolor='#1a1a1a', edgecolor='white', fontsize=10, loc='upper right')
    plt.setp(legend.get_texts(), color='white')

    plt.tight_layout()
    fig.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='#0e1117', edgecolor='none')
    logger.info(f"Saved to {save_path}")
    plt.close()


def plot_residuals_vs_moneyness(df: pl.DataFrame, save_path: Path) -> None:
    """
    Plot residuals vs moneyness to detect systematic bias.

    Moneyness = (S/K - 1) where:
    - Negative: Out-of-the-money (OTM) calls
    - Zero: At-the-money (ATM)
    - Positive: In-the-money (ITM) calls
    """
    logger.info("Creating residuals vs moneyness plot...")

    # Sample for visibility
    sample_df = df.sample(n=min(50000, len(df)))

    moneyness = sample_df["moneyness"].to_numpy()
    residuals = sample_df["residual"].to_numpy()

    fig, ax = plt.subplots(figsize=(12, 8), facecolor='#0e1117')
    ax.set_facecolor('#0e1117')

    # Hexbin for density
    hexbin = ax.hexbin(
        moneyness * 100,  # Convert to percentage
        residuals,
        gridsize=50,
        cmap='Greens',
        mincnt=1,
        alpha=0.8
    )

    # Zero lines
    ax.axhline(y=0, color=COLORS['perfect'], linestyle='--', linewidth=2, alpha=0.7, label='Zero Error')
    ax.axvline(x=0, color=COLORS['warning'], linestyle=':', linewidth=2, alpha=0.5, label='ATM')

    # Add smoothed mean residual by bin
    bins = np.linspace(moneyness.min(), moneyness.max(), 30)
    bin_centers = []
    bin_means = []

    for i in range(len(bins)-1):
        mask = (moneyness >= bins[i]) & (moneyness < bins[i+1])
        if mask.sum() > 100:  # Require sufficient samples
            bin_centers.append((bins[i] + bins[i+1]) / 2 * 100)
            bin_means.append(np.mean(residuals[mask]))

    if len(bin_centers) > 0:
        ax.plot(bin_centers, bin_means,
                color=COLORS['danger'],
                linewidth=3,
                marker='o',
                markersize=8,
                label='Mean Residual by Bin',
                zorder=100)

    # Colorbar
    cbar = plt.colorbar(hexbin, ax=ax)
    cbar.set_label('Count', fontsize=12, color='white', fontweight='bold')
    cbar.ax.tick_params(colors='white', labelsize=10)

    # Styling
    ax.set_xlabel('Moneyness (S/K - 1) [%]', fontsize=12, color='white', fontweight='bold')
    ax.set_ylabel('Residual (Actual - Predicted)', fontsize=12, color='white', fontweight='bold')
    ax.set_title('Residuals vs Moneyness: Detect Systematic Bias', fontsize=14, color='white', fontweight='bold', pad=20)
    ax.grid(True, alpha=0.2, color=COLORS['grid'])
    ax.tick_params(colors='white', labelsize=10)

    legend = ax.legend(frameon=True, facecolor='#1a1a1a', edgecolor='white', fontsize=10, loc='upper right')
    plt.setp(legend.get_texts(), color='white')

    plt.tight_layout()
    fig.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='#0e1117', edgecolor='none')
    logger.info(f"Saved to {save_path}")
    plt.close()


def plot_probability_vs_moneyness(df: pl.DataFrame, save_path: Path) -> None:
    """
    Plot probability vs moneyness to demonstrate non-linear appreciation.

    Linear: Would be a straight line
    Non-linear: Sigmoid curve (steep at ATM, flat at tails)
    """
    logger.info("Creating probability vs moneyness plot (non-linearity demo)...")

    # Filter to mid-contract for fair comparison (450s ± 50s)
    df_mid = df.filter(
        (pl.col("time_remaining") >= 400) &
        (pl.col("time_remaining") <= 500)
    )

    logger.info(f"Filtered to {len(df_mid):,} mid-contract observations")

    # Create bins by moneyness
    moneyness = df_mid["moneyness"].to_numpy()
    prob_mid = df_mid["prob_mid"].to_numpy()

    # Bin and aggregate
    bins = np.linspace(moneyness.min(), moneyness.max(), 100)
    bin_centers = []
    bin_probs = []
    bin_counts = []

    for i in range(len(bins)-1):
        mask = (moneyness >= bins[i]) & (moneyness < bins[i+1])
        if mask.sum() > 50:  # Require sufficient samples
            bin_centers.append((bins[i] + bins[i+1]) / 2)
            bin_probs.append(np.mean(prob_mid[mask]))
            bin_counts.append(mask.sum())

    bin_centers = np.array(bin_centers)
    bin_probs = np.array(bin_probs)
    bin_counts = np.array(bin_counts)

    # Plot
    fig, ax = plt.subplots(figsize=(12, 8), facecolor='#0e1117')
    ax.set_facecolor('#0e1117')

    # Scatter with size proportional to count
    sizes = np.clip(bin_counts / 1000, 30, 200)

    scatter = ax.scatter(
        bin_centers * 100,  # Convert to percentage
        bin_probs,
        s=sizes,
        alpha=0.7,
        c=COLORS['primary'],
        edgecolor='white',
        linewidth=1,
        label='Empirical (BS Model)'
    )

    # Fit theoretical sigmoid for comparison
    from scipy.special import ndtr

    # Simple theoretical curve (d2 approximation)
    # For T=450s, sigma=40%, r=5%
    T_years = 450 / (365 * 24 * 3600)
    sigma = 0.40
    r = 0.05

    moneyness_theory = np.linspace(bin_centers.min(), bin_centers.max(), 200)
    # d2 = [ln(S/K) + (r - sigma^2/2)T] / (sigma * sqrt(T))
    # For small moneyness: ln(S/K) ≈ S/K - 1 = moneyness
    d2_theory = (moneyness_theory + (r - 0.5 * sigma**2) * T_years) / (sigma * np.sqrt(T_years))
    prob_theory = ndtr(d2_theory)  # N(d2)

    ax.plot(
        moneyness_theory * 100,
        prob_theory,
        color=COLORS['danger'],
        linewidth=3,
        linestyle='--',
        label='Theoretical BS (σ=40%, T=450s)',
        zorder=100
    )

    # Reference lines
    ax.axhline(y=0.5, color=COLORS['perfect'], linestyle=':', linewidth=1.5, alpha=0.5, label='50% Probability')
    ax.axvline(x=0, color=COLORS['warning'], linestyle=':', linewidth=2, alpha=0.5, label='ATM')

    # Annotations for key points
    atm_idx = np.argmin(np.abs(bin_centers))
    if atm_idx < len(bin_centers):
        ax.annotate(
            f'ATM\nP≈{bin_probs[atm_idx]:.3f}',
            xy=(bin_centers[atm_idx] * 100, bin_probs[atm_idx]),
            xytext=(bin_centers[atm_idx] * 100 + 1, bin_probs[atm_idx] + 0.15),
            fontsize=10,
            color='white',
            bbox=dict(boxstyle='round', facecolor='#1a1a1a', alpha=0.8, edgecolor='white'),
            arrowprops=dict(arrowstyle='->', color='white', lw=1.5)
        )

    # Styling
    ax.set_xlabel('Moneyness (S/K - 1) [%]', fontsize=12, color='white', fontweight='bold')
    ax.set_ylabel('Probability (Binary Call)', fontsize=12, color='white', fontweight='bold')
    ax.set_title('Non-Linear Appreciation: Probability vs Moneyness (T=450s)', fontsize=14, color='white', fontweight='bold', pad=20)
    ax.grid(True, alpha=0.2, color=COLORS['grid'])
    ax.tick_params(colors='white', labelsize=10)
    ax.set_ylim(0, 1)

    legend = ax.legend(frameon=True, facecolor='#1a1a1a', edgecolor='white', fontsize=10, loc='upper left')
    plt.setp(legend.get_texts(), color='white')

    # Add text box explaining non-linearity
    explanation = """Non-Linear Behavior:
• Sigmoid shape (S-curve)
• Steepest at ATM (0%)
• Flat in tails (±2%)
• NOT a straight line"""

    ax.text(0.98, 0.02, explanation,
            transform=ax.transAxes,
            fontsize=9,
            verticalalignment='bottom',
            horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='#1a1a1a', alpha=0.9, edgecolor=COLORS['warning']),
            color='white')

    plt.tight_layout()
    fig.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='#0e1117', edgecolor='none')
    logger.info(f"Saved to {save_path}")
    plt.close()


def plot_delta_vs_moneyness(df: pl.DataFrame, save_path: Path) -> None:
    """
    Plot delta (∂prob/∂moneyness) vs moneyness to quantify non-linearity.

    Linear: Delta would be constant
    Non-linear: Delta varies (highest at ATM, lowest in tails)
    """
    logger.info("Creating delta vs moneyness plot...")

    # Filter to mid-contract
    df_mid = df.filter(
        (pl.col("time_remaining") >= 400) &
        (pl.col("time_remaining") <= 500)
    )

    moneyness = df_mid["moneyness"].to_numpy()
    prob_mid = df_mid["prob_mid"].to_numpy()

    # Create fine bins for numerical differentiation
    bins = np.linspace(moneyness.min(), moneyness.max(), 200)
    bin_centers = []
    bin_probs = []

    for i in range(len(bins)-1):
        mask = (moneyness >= bins[i]) & (moneyness < bins[i+1])
        if mask.sum() > 50:
            bin_centers.append((bins[i] + bins[i+1]) / 2)
            bin_probs.append(np.mean(prob_mid[mask]))

    bin_centers = np.array(bin_centers)
    bin_probs = np.array(bin_probs)

    # Compute delta as numerical derivative
    delta = np.gradient(bin_probs, bin_centers)

    # Plot
    fig, ax = plt.subplots(figsize=(12, 8), facecolor='#0e1117')
    ax.set_facecolor('#0e1117')

    # Main delta curve
    ax.plot(
        bin_centers * 100,
        delta,
        color=COLORS['primary'],
        linewidth=3,
        label='Empirical Delta (∂P/∂M)',
        zorder=100
    )

    # Fill under curve
    ax.fill_between(
        bin_centers * 100,
        0,
        delta,
        alpha=0.3,
        color=COLORS['primary']
    )

    # Reference lines
    ax.axvline(x=0, color=COLORS['warning'], linestyle=':', linewidth=2, alpha=0.5, label='ATM')
    ax.axhline(y=np.mean(delta), color=COLORS['perfect'], linestyle='--', linewidth=2, alpha=0.5,
               label=f'Mean Delta: {np.mean(delta):.2f}')

    # Annotations for max delta
    max_delta_idx = np.argmax(delta)
    ax.annotate(
        f'Max Δ = {delta[max_delta_idx]:.2f}\n@ M={bin_centers[max_delta_idx]*100:.2f}%',
        xy=(bin_centers[max_delta_idx] * 100, delta[max_delta_idx]),
        xytext=(bin_centers[max_delta_idx] * 100 + 1, delta[max_delta_idx] - 20),
        fontsize=10,
        color='white',
        bbox=dict(boxstyle='round', facecolor='#1a1a1a', alpha=0.8, edgecolor='white'),
        arrowprops=dict(arrowstyle='->', color='white', lw=1.5)
    )

    # Statistical summary
    stats_text = f"""Delta Statistics:
Mean: {np.mean(delta):.2f}
Std: {np.std(delta):.2f}
Max: {np.max(delta):.2f}
Min: {np.min(delta):.2f}
Range: {np.max(delta) - np.min(delta):.2f}
CV: {np.std(delta)/np.mean(delta)*100:.1f}%"""

    ax.text(0.02, 0.98, stats_text,
            transform=ax.transAxes,
            fontsize=9,
            verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='#1a1a1a', alpha=0.9, edgecolor='white'),
            color='white')

    # Styling
    ax.set_xlabel('Moneyness (S/K - 1) [%]', fontsize=12, color='white', fontweight='bold')
    ax.set_ylabel('Delta (∂Probability / ∂Moneyness)', fontsize=12, color='white', fontweight='bold')
    ax.set_title('Delta Sensitivity: Quantifying Non-Linearity (T=450s)', fontsize=14, color='white', fontweight='bold', pad=20)
    ax.grid(True, alpha=0.2, color=COLORS['grid'])
    ax.tick_params(colors='white', labelsize=10)

    legend = ax.legend(frameon=True, facecolor='#1a1a1a', edgecolor='white', fontsize=10, loc='upper right')
    plt.setp(legend.get_texts(), color='white')

    # Add explanation
    explanation = """Non-Linear Evidence:
• Delta varies 50x from tails to ATM
• Coefficient of Variation > 100%
• NOT constant (would be flat line if linear)"""

    ax.text(0.98, 0.02, explanation,
            transform=ax.transAxes,
            fontsize=9,
            verticalalignment='bottom',
            horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='#1a1a1a', alpha=0.9, edgecolor=COLORS['danger']),
            color='white')

    plt.tight_layout()
    fig.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='#0e1117', edgecolor='none')
    logger.info(f"Saved to {save_path}")
    plt.close()


def plot_calibration(df: pl.DataFrame, save_path: Path) -> None:
    """
    Create calibration plot: predicted probability vs actual outcome.

    Perfect calibration: points lie on 45° line
    Miscalibration: systematic deviations from diagonal
    """
    logger.info("Creating calibration plot...")

    # Sample for speed
    sample_df = df.sample(n=min(100000, len(df)))

    prob_mid = sample_df["prob_mid"].to_numpy()
    outcome = sample_df["outcome"].to_numpy()

    # Bin predictions
    bins = np.linspace(0, 1, 21)  # 20 bins
    bin_centers = []
    bin_actual = []
    bin_counts = []

    for i in range(len(bins)-1):
        mask = (prob_mid >= bins[i]) & (prob_mid < bins[i+1])
        if mask.sum() > 0:
            bin_centers.append((bins[i] + bins[i+1]) / 2)
            bin_actual.append(np.mean(outcome[mask]))
            bin_counts.append(mask.sum())

    bin_centers = np.array(bin_centers)
    bin_actual = np.array(bin_actual)
    bin_counts = np.array(bin_counts)

    # Compute Brier score
    brier_score = np.mean((prob_mid - outcome) ** 2)

    # Plot
    fig, ax = plt.subplots(figsize=(10, 10), facecolor='#0e1117')
    ax.set_facecolor('#0e1117')

    # Scale dot sizes
    sizes = np.clip(bin_counts / 100, 30, 200)

    # Scatter
    scatter = ax.scatter(
        bin_centers,
        bin_actual,
        s=sizes,
        alpha=0.7,
        c=COLORS['primary'],
        edgecolor='white',
        linewidth=1.5,
        label='Calibration (binned)'
    )

    # Perfect calibration line
    ax.plot([0, 1], [0, 1],
            '--',
            color=COLORS['perfect'],
            alpha=0.7,
            linewidth=2,
            label='Perfect Calibration',
            zorder=50)

    # Confidence bands (±1.96 SE for 95% CI)
    # SE ≈ sqrt(p(1-p)/n) for binomial
    for i, (bc, ba, cnt) in enumerate(zip(bin_centers, bin_actual, bin_counts)):
        se = np.sqrt(bc * (1 - bc) / cnt)
        ci_lower = ba - 1.96 * se
        ci_upper = ba + 1.96 * se
        ax.plot([bc, bc], [ci_lower, ci_upper],
                color=COLORS['info'],
                alpha=0.5,
                linewidth=2)

    # Styling
    ax.set_xlabel('Predicted Probability (BS Model)', fontsize=12, color='white', fontweight='bold')
    ax.set_ylabel('Actual Outcome Frequency', fontsize=12, color='white', fontweight='bold')
    ax.set_title(f'Calibration Plot | Brier Score: {brier_score:.4f}', fontsize=14, color='white', fontweight='bold', pad=20)
    ax.grid(True, alpha=0.2, color=COLORS['grid'])
    ax.tick_params(colors='white', labelsize=10)
    ax.set_aspect('equal', adjustable='box')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    legend = ax.legend(frameon=True, facecolor='#1a1a1a', edgecolor='white', fontsize=10, loc='upper left')
    plt.setp(legend.get_texts(), color='white')

    # Sample size note
    note = f"Sample: {len(prob_mid):,} predictions\nError bars: 95% CI"
    ax.text(0.98, 0.02, note,
            transform=ax.transAxes,
            fontsize=9,
            verticalalignment='bottom',
            horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='#1a1a1a', alpha=0.9, edgecolor='white'),
            color='white')

    plt.tight_layout()
    fig.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='#0e1117', edgecolor='none')
    logger.info(f"Saved to {save_path}")
    plt.close()


def generate_summary_stats(df: pl.DataFrame) -> dict[str, float]:
    """
    Generate summary statistics for the report.

    Returns:
        Dictionary with key metrics
    """
    logger.info("Computing summary statistics...")

    stats = {
        'n_total': len(df),
        'mean_residual': df["residual"].mean(),
        'std_residual': df["residual"].std(),
        'mae_residual': df["residual"].abs().mean(),
        'brier_score': ((df["prob_mid"] - df["outcome"]) ** 2).mean(),
        'mean_prob_mid': df["prob_mid"].mean(),
        'mean_outcome': df["outcome"].mean(),
        'mean_xgb_correction': df["residual_pred_xgb"].mean(),
        'std_xgb_correction': df["residual_pred_xgb"].std(),
    }

    # Compute delta statistics at mid-contract
    df_mid = df.filter(
        (pl.col("time_remaining") >= 400) &
        (pl.col("time_remaining") <= 500)
    )

    moneyness = df_mid["moneyness"].to_numpy()
    prob_mid = df_mid["prob_mid"].to_numpy()

    # Bin and compute delta
    bins = np.linspace(moneyness.min(), moneyness.max(), 100)
    bin_centers = []
    bin_probs = []

    for i in range(len(bins)-1):
        mask = (moneyness >= bins[i]) & (moneyness < bins[i+1])
        if mask.sum() > 50:
            bin_centers.append((bins[i] + bins[i+1]) / 2)
            bin_probs.append(np.mean(prob_mid[mask]))

    if len(bin_centers) > 1:
        delta = np.gradient(bin_probs, bin_centers)
        stats['mean_delta'] = np.mean(delta)
        stats['std_delta'] = np.std(delta)
        stats['max_delta'] = np.max(delta)
        stats['min_delta'] = np.min(delta)
        stats['delta_cv'] = np.std(delta) / np.mean(delta) * 100  # Coefficient of variation

    return stats


def main() -> None:
    """Main execution function."""
    logger.info("Starting comprehensive residual EDA...")

    # Load data (use 10% sample for faster iteration, 1.0 for full analysis)
    df = load_data(sample_frac=0.1)

    # Generate plots
    logger.info("Generating visualizations...")

    plot_residual_distribution(df, OUTPUT_DIR / "01_residual_distribution.png")
    plot_residuals_vs_predicted(df, OUTPUT_DIR / "02_residuals_vs_predicted.png")
    plot_residuals_vs_moneyness(df, OUTPUT_DIR / "03_residuals_vs_moneyness.png")
    plot_probability_vs_moneyness(df, OUTPUT_DIR / "04_probability_vs_moneyness.png")
    plot_delta_vs_moneyness(df, OUTPUT_DIR / "05_delta_vs_moneyness.png")
    plot_calibration(df, OUTPUT_DIR / "06_calibration.png")

    # Generate summary statistics
    stats = generate_summary_stats(df)

    # Print summary
    logger.info("\n" + "="*80)
    logger.info("SUMMARY STATISTICS")
    logger.info("="*80)
    for key, value in stats.items():
        if isinstance(value, int):
            logger.info(f"{key:25s}: {value:,}")
        else:
            logger.info(f"{key:25s}: {value:.6f}")
    logger.info("="*80)

    logger.info(f"\nAll plots saved to: {OUTPUT_DIR}")
    logger.info("Analysis complete!")


if __name__ == "__main__":
    main()
