#!/usr/bin/env python3
"""
Create comparison plot: Linear vs Non-Linear appreciation

This creates a side-by-side comparison showing why linear models fail.
"""

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from scipy.special import ndtr

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Color palette
COLORS = {
    'primary': '#00D4FF',
    'secondary': '#FF00FF',
    'success': '#00FF88',
    'danger': '#FF3366',
    'warning': '#FFB000',
    'perfect': '#888888',
    'grid': '#333333',
}

# Paths
DATA_DIR = Path("/Users/lgierhake/Documents/ETH/BT/research/model/results")
OUTPUT_DIR = Path("/Users/lgierhake/Documents/ETH/BT/research/model/02_analysis/figures")
RESIDUALS_FILE = DATA_DIR / "xgboost_residual_model_test_2024-07-01_2025-09-30.parquet"


def main() -> None:
    """Create linear vs non-linear comparison plot."""
    logger.info("Creating linear vs non-linear comparison...")

    # Load data (mid-contract only)
    df = (
        pl.scan_parquet(RESIDUALS_FILE)
          .filter(
              (pl.col("time_remaining") >= 400) &
              (pl.col("time_remaining") <= 500)
          )
          .select(["moneyness", "prob_mid"])
          .with_row_index("idx")
          .filter(pl.col("idx") % 10 == 0)  # 10% sample
          .drop("idx")
          .collect()
    )

    logger.info(f"Loaded {len(df):,} mid-contract observations")

    moneyness = df["moneyness"].to_numpy()
    prob_mid = df["prob_mid"].to_numpy()

    # Bin the data
    bins = np.linspace(moneyness.min(), moneyness.max(), 100)
    bin_centers = []
    bin_probs = []

    for i in range(len(bins)-1):
        mask = (moneyness >= bins[i]) & (moneyness < bins[i+1])
        if mask.sum() > 50:
            bin_centers.append((bins[i] + bins[i+1]) / 2)
            bin_probs.append(np.mean(prob_mid[mask]))

    bin_centers = np.array(bin_centers)
    bin_probs = np.array(bin_probs)

    # Fit linear model (for comparison)
    from sklearn.linear_model import LinearRegression
    lr = LinearRegression()
    lr.fit(bin_centers.reshape(-1, 1), bin_probs)
    prob_linear = lr.predict(bin_centers.reshape(-1, 1))

    # Theoretical BS curve
    T_years = 450 / (365 * 24 * 3600)
    sigma = 0.40
    r = 0.05

    moneyness_theory = np.linspace(bin_centers.min(), bin_centers.max(), 200)
    d2_theory = (moneyness_theory + (r - 0.5 * sigma**2) * T_years) / (sigma * np.sqrt(T_years))
    prob_theory = ndtr(d2_theory)

    # Create comparison plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8), facecolor='#0e1117')

    # Panel 1: Linear Model (WRONG)
    ax1.set_facecolor('#0e1117')

    ax1.scatter(bin_centers * 100, bin_probs,
                alpha=0.6, s=50, color=COLORS['primary'],
                edgecolor='white', linewidth=0.5,
                label='Actual Data', zorder=10)

    ax1.plot(bin_centers * 100, prob_linear,
             color=COLORS['danger'], linewidth=4,
             linestyle='-', label='LINEAR FIT (WRONG)',
             zorder=20, alpha=0.9)

    # Compute R² for linear model
    ss_res = np.sum((bin_probs - prob_linear) ** 2)
    ss_tot = np.sum((bin_probs - np.mean(bin_probs)) ** 2)
    r2_linear = 1 - (ss_res / ss_tot)

    # Add error bars
    errors = bin_probs - prob_linear
    mae_linear = np.mean(np.abs(errors))

    ax1.axhline(y=0.5, color=COLORS['perfect'], linestyle=':', linewidth=1.5, alpha=0.5)
    ax1.axvline(x=0, color=COLORS['warning'], linestyle=':', linewidth=2, alpha=0.5)

    # Annotations
    stats_text = f"""❌ LINEAR MODEL FAILS:
R² = {r2_linear:.3f}
MAE = {mae_linear:.3f}
Max Error = {np.max(np.abs(errors)):.3f}

Problems:
• Assumes constant delta
• Misses convexity
• Poor at tails & ATM"""

    ax1.text(0.02, 0.98, stats_text,
             transform=ax1.transAxes,
             fontsize=10,
             verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='#1a1a1a', alpha=0.9, edgecolor=COLORS['danger']),
             color='white')

    ax1.set_xlabel('Moneyness (S/K - 1) [%]', fontsize=12, color='white', fontweight='bold')
    ax1.set_ylabel('Probability', fontsize=12, color='white', fontweight='bold')
    ax1.set_title('❌ LINEAR MODEL (INCORRECT)', fontsize=16, color=COLORS['danger'], fontweight='bold', pad=20)
    ax1.grid(True, alpha=0.2, color=COLORS['grid'])
    ax1.tick_params(colors='white', labelsize=10)
    ax1.set_ylim(0, 1)

    legend1 = ax1.legend(frameon=True, facecolor='#1a1a1a', edgecolor='white', fontsize=10, loc='upper left')
    plt.setp(legend1.get_texts(), color='white')

    # Panel 2: Non-Linear Model (CORRECT)
    ax2.set_facecolor('#0e1117')

    ax2.scatter(bin_centers * 100, bin_probs,
                alpha=0.6, s=50, color=COLORS['primary'],
                edgecolor='white', linewidth=0.5,
                label='Actual Data', zorder=10)

    ax2.plot(moneyness_theory * 100, prob_theory,
             color=COLORS['success'], linewidth=4,
             linestyle='-', label='BLACK-SCHOLES (CORRECT)',
             zorder=20, alpha=0.9)

    # Compute R² for BS model
    prob_bs_binned = np.interp(bin_centers, moneyness_theory, prob_theory)
    ss_res_bs = np.sum((bin_probs - prob_bs_binned) ** 2)
    r2_bs = 1 - (ss_res_bs / ss_tot)
    mae_bs = np.mean(np.abs(bin_probs - prob_bs_binned))

    ax2.axhline(y=0.5, color=COLORS['perfect'], linestyle=':', linewidth=1.5, alpha=0.5)
    ax2.axvline(x=0, color=COLORS['warning'], linestyle=':', linewidth=2, alpha=0.5)

    # Annotations
    stats_text2 = f"""✓ NON-LINEAR MODEL WORKS:
R² = {r2_bs:.3f}
MAE = {mae_bs:.3f}
Max Error = {np.max(np.abs(bin_probs - prob_bs_binned)):.3f}

Features:
• Sigmoid (S-curve)
• Steep at ATM
• Flat at tails
• Captures convexity"""

    ax2.text(0.02, 0.98, stats_text2,
             transform=ax2.transAxes,
             fontsize=10,
             verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='#1a1a1a', alpha=0.9, edgecolor=COLORS['success']),
             color='white')

    ax2.set_xlabel('Moneyness (S/K - 1) [%]', fontsize=12, color='white', fontweight='bold')
    ax2.set_ylabel('Probability', fontsize=12, color='white', fontweight='bold')
    ax2.set_title('✓ NON-LINEAR MODEL (CORRECT)', fontsize=16, color=COLORS['success'], fontweight='bold', pad=20)
    ax2.grid(True, alpha=0.2, color=COLORS['grid'])
    ax2.tick_params(colors='white', labelsize=10)
    ax2.set_ylim(0, 1)

    legend2 = ax2.legend(frameon=True, facecolor='#1a1a1a', edgecolor='white', fontsize=10, loc='upper left')
    plt.setp(legend2.get_texts(), color='white')

    # Overall title
    fig.suptitle('Binary Options: Linear vs Non-Linear Appreciation (T=450s)',
                 fontsize=18, color='white', fontweight='bold', y=0.98)

    plt.tight_layout(rect=[0, 0, 1, 0.96])

    output_path = OUTPUT_DIR / "00_linear_vs_nonlinear_comparison.png"
    fig.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='#0e1117', edgecolor='none')
    logger.info(f"Saved to {output_path}")

    # Print summary
    logger.info("\n" + "="*80)
    logger.info("COMPARISON RESULTS")
    logger.info("="*80)
    logger.info(f"LINEAR MODEL:")
    logger.info(f"  R² = {r2_linear:.4f}")
    logger.info(f"  MAE = {mae_linear:.4f}")
    logger.info(f"  Max Error = {np.max(np.abs(errors)):.4f}")
    logger.info("")
    logger.info(f"NON-LINEAR MODEL (Black-Scholes):")
    logger.info(f"  R² = {r2_bs:.4f}")
    logger.info(f"  MAE = {mae_bs:.4f}")
    logger.info(f"  Max Error = {np.max(np.abs(bin_probs - prob_bs_binned)):.4f}")
    logger.info("")
    logger.info(f"IMPROVEMENT:")
    logger.info(f"  R² improvement: {(r2_bs - r2_linear):.4f} ({(r2_bs/r2_linear - 1)*100:.1f}%)")
    logger.info(f"  MAE improvement: {(mae_linear - mae_bs):.4f} ({(1 - mae_bs/mae_linear)*100:.1f}%)")
    logger.info("="*80)


if __name__ == "__main__":
    main()
