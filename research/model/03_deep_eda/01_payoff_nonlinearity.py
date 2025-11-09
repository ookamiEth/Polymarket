#!/usr/bin/env python3
"""
Phase 1: Payoff & Non-Linearity Analysis
========================================

Comprehensive analysis of binary option payoff structures, delta surfaces,
gamma/convexity, and empirical vs theoretical comparisons.

Analysis Goals:
1. Map delta surface across (moneyness, time_remaining) space
2. Calculate gamma (second derivative) to find convexity hotspots
3. Compute vega sensitivity by strike
4. Compare empirical behavior to Black-Scholes theoretical expectations
5. Identify regime changes in payoff structure (high vol vs low vol)
"""

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from scipy import stats

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Plot styling
plt.style.use("dark_background")
FIGURE_DIR = Path("research/model/03_deep_eda/figures")
TABLE_DIR = Path("research/model/03_deep_eda/tables")
FIGURE_DIR.mkdir(parents=True, exist_ok=True)
TABLE_DIR.mkdir(parents=True, exist_ok=True)


def load_test_data(sample_frac: float = 0.1) -> pl.DataFrame:
    """Load test set predictions with sampling for memory efficiency."""
    logger.info(f"Loading test data (sample fraction: {sample_frac})")

    # Load full test set
    df = pl.read_parquet("research/model/results/xgboost_residual_model_test_2024-07-01_2025-09-30.parquet")

    logger.info(f"Loaded {len(df):,} rows")

    # Sample if needed
    if sample_frac < 1.0:
        df = df.sample(fraction=sample_frac, seed=42)
        logger.info(f"Sampled to {len(df):,} rows")

    # Add derived columns
    df = df.with_columns(
        [
            # Moneyness in percent (moneyness is already decimal in data)
            (pl.col("moneyness") * 100).alias("moneyness_pct"),
            # Time remaining in minutes
            (pl.col("time_remaining") / 60).alias("time_remaining_min"),
            # Prediction error
            (pl.col("outcome") - pl.col("prob_mid")).alias("prediction_error"),
        ]
    )

    return df


def calculate_empirical_delta(
    df: pl.DataFrame,
    moneyness_bins: int = 50,
    min_samples: int = 100,
) -> pl.DataFrame:
    """
    Calculate empirical delta (dP/dM) via finite differences.

    Delta = ΔProbability / ΔMoneyness

    For each moneyness bin:
    - Compute average probability
    - Compute delta as slope between adjacent bins
    """
    logger.info("Calculating empirical delta surface")

    # Create moneyness bins
    moneyness_range = df.select(
        [
            pl.col("moneyness_pct").min().alias("min_m"),
            pl.col("moneyness_pct").max().alias("max_m"),
        ]
    ).row(0)

    bins = np.linspace(moneyness_range[0], moneyness_range[1], moneyness_bins)

    # Bin data and aggregate
    df_binned = (
        df.with_columns(
            [
                pl.col("moneyness_pct").cut(bins).alias("moneyness_bin"),
            ]
        )
        .group_by("moneyness_bin")
        .agg(
            [
                pl.len().alias("count"),
                pl.col("prob_mid").mean().alias("avg_prob"),
                pl.col("outcome").mean().alias("avg_outcome"),
                pl.col("moneyness_pct").mean().alias("avg_moneyness"),
            ]
        )
        .filter(pl.col("count") >= min_samples)
        .sort("avg_moneyness")
    )

    # Calculate delta via finite differences
    probs = df_binned["avg_prob"].to_numpy()
    moneyness_vals = df_binned["avg_moneyness"].to_numpy()

    # Forward difference for delta
    deltas = np.diff(probs) / np.diff(moneyness_vals)

    # Assign deltas (use midpoint for delta location)
    delta_df = pl.DataFrame(
        {
            "moneyness": moneyness_vals[:-1],
            "probability": probs[:-1],
            "delta": deltas,
            "count": df_binned["count"].to_numpy()[:-1],
        }
    )

    logger.info(f"Calculated delta for {len(delta_df)} bins")
    logger.info(f"Delta range: [{delta_df['delta'].min():.2f}, {delta_df['delta'].max():.2f}]")

    return delta_df


def calculate_gamma(delta_df: pl.DataFrame) -> pl.DataFrame:
    """
    Calculate gamma (second derivative) from delta.

    Gamma = ΔDelta / ΔMoneyness

    Indicates where convexity is highest (option sensitivity changes fastest).
    """
    logger.info("Calculating gamma (convexity)")

    deltas = delta_df["delta"].to_numpy()
    moneyness_vals = delta_df["moneyness"].to_numpy()

    # Second derivative via finite differences
    gammas = np.diff(deltas) / np.diff(moneyness_vals)

    gamma_df = pl.DataFrame(
        {
            "moneyness": moneyness_vals[:-1],
            "delta": deltas[:-1],
            "gamma": gammas,
        }
    )

    logger.info(f"Gamma range: [{gamma_df['gamma'].min():.2f}, {gamma_df['gamma'].max():.2f}]")

    return gamma_df


def calculate_bs_theoretical_delta(
    moneyness: np.ndarray, vol: float = 0.50, time_to_expiry: float = 450 / 86400
) -> np.ndarray:
    """
    Calculate theoretical Black-Scholes delta for binary options.

    For binary call: Delta = e^(-rT) × φ(d2) / (S × σ × √T)
    where φ is the standard normal PDF.

    Approximation for r=0: Delta ≈ φ(d2) / (σ × √T)
    """
    # Convert moneyness % to decimal
    moneyness_decimal = moneyness / 100.0

    # d2 calculation (assuming S=K*(1+M), r≈0)
    # d2 = [ln(S/K) - 0.5*σ²*T] / (σ√T)
    # d2 ≈ [ln(1+M) - 0.5*σ²*T] / (σ√T)

    d2 = (np.log(1 + moneyness_decimal) - 0.5 * vol**2 * time_to_expiry) / (vol * np.sqrt(time_to_expiry))

    # Binary delta = φ(d2) / (S × σ × √T)
    # Normalized by S, this becomes: φ(d2) / (σ × √T)
    phi_d2 = stats.norm.pdf(d2)
    delta_bs = phi_d2 / (vol * np.sqrt(time_to_expiry))

    return delta_bs


def plot_delta_surface_2d(delta_df: pl.DataFrame, output_file: Path) -> None:
    """Plot empirical delta vs moneyness with theoretical comparison."""
    logger.info("Generating 2D delta surface plot")

    fig, ax = plt.subplots(figsize=(14, 8))

    # Empirical delta
    moneyness = delta_df["moneyness"].to_numpy()
    delta_emp = delta_df["delta"].to_numpy()

    # Theoretical delta (multiple volatilities)
    moneyness_theory = np.linspace(moneyness.min(), moneyness.max(), 200)

    # Plot empirical
    scatter = ax.scatter(
        moneyness,
        delta_emp,
        s=50,
        c=delta_df["count"].to_numpy(),
        cmap="viridis",
        alpha=0.7,
        edgecolor="white",
        linewidth=0.5,
        label="Empirical Delta",
    )

    # Plot theoretical (multiple vol scenarios)
    for vol in [0.3, 0.5, 0.7]:
        delta_theory = calculate_bs_theoretical_delta(moneyness_theory, vol=vol)
        ax.plot(
            moneyness_theory,
            delta_theory,
            linestyle="--",
            linewidth=2,
            alpha=0.6,
            label=f"BS Theory (σ={vol:.0%})",
        )

    # Styling
    ax.set_xlabel("Moneyness (%)", fontsize=14)
    ax.set_ylabel("Delta (ΔP / ΔM%)", fontsize=14)
    ax.set_title("Binary Option Delta Surface: Empirical vs Theoretical", fontsize=16, pad=20)
    ax.grid(alpha=0.2)
    ax.legend(fontsize=11, frameon=True, framealpha=0.9)
    ax.axvline(0, color="gray", linestyle=":", alpha=0.5, label="ATM")

    # Colorbar for sample counts
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label("Sample Count", fontsize=12)

    # Statistics annotation
    delta_stats = delta_df.select(
        [
            pl.col("delta").min().alias("min_delta"),
            pl.col("delta").max().alias("max_delta"),
            pl.col("delta").mean().alias("mean_delta"),
            pl.col("delta").std().alias("std_delta"),
        ]
    ).row(0)

    cv = (delta_stats[3] / delta_stats[2]) * 100 if delta_stats[2] != 0 else 0

    stats_text = "Empirical Delta Statistics:\n"
    stats_text += f"Range: [{delta_stats[0]:.1f}, {delta_stats[1]:.1f}]\n"
    stats_text += f"Mean: {delta_stats[2]:.1f}\n"
    stats_text += f"CV: {cv:.1f}%"

    ax.text(
        0.02,
        0.98,
        stats_text,
        transform=ax.transAxes,
        fontsize=10,
        verticalalignment="top",
        bbox={"boxstyle": "round", "facecolor": "black", "alpha": 0.8},
    )

    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    logger.info(f"Saved: {output_file}")
    plt.close()


def plot_gamma_surface(gamma_df: pl.DataFrame, output_file: Path) -> None:
    """Plot gamma (convexity) surface."""
    logger.info("Generating gamma surface plot")

    fig, ax = plt.subplots(figsize=(14, 8))

    moneyness = gamma_df["moneyness"].to_numpy()
    gamma = gamma_df["gamma"].to_numpy()

    # Plot gamma
    ax.plot(moneyness, gamma, linewidth=2, color="#00D4FF", label="Gamma (Δ²P / ΔM²)")
    ax.fill_between(moneyness, 0, gamma, alpha=0.3, color="#00D4FF")

    # Mark peak gamma
    peak_idx = np.argmax(np.abs(gamma))
    peak_moneyness = moneyness[peak_idx]
    peak_gamma = gamma[peak_idx]

    ax.scatter(
        [peak_moneyness],
        [peak_gamma],
        s=200,
        color="#FF3366",
        marker="*",
        zorder=5,
        label=f"Peak Gamma at M={peak_moneyness:.2f}%",
    )

    # Styling
    ax.set_xlabel("Moneyness (%)", fontsize=14)
    ax.set_ylabel("Gamma (Convexity)", fontsize=14)
    ax.set_title("Binary Option Gamma Surface (Convexity Hotspots)", fontsize=16, pad=20)
    ax.grid(alpha=0.2)
    ax.legend(fontsize=12, frameon=True, framealpha=0.9)
    ax.axhline(0, color="gray", linestyle="--", alpha=0.5)
    ax.axvline(0, color="gray", linestyle=":", alpha=0.5, label="ATM")

    # Statistics
    gamma_stats = gamma_df.select(
        [
            pl.col("gamma").min().alias("min_gamma"),
            pl.col("gamma").max().alias("max_gamma"),
            pl.col("gamma").mean().alias("mean_gamma"),
        ]
    ).row(0)

    stats_text = "Gamma Statistics:\n"
    stats_text += f"Range: [{gamma_stats[0]:.2f}, {gamma_stats[1]:.2f}]\n"
    stats_text += f"Peak: {peak_gamma:.2f} at M={peak_moneyness:.2f}%\n"
    stats_text += f"Mean: {gamma_stats[2]:.2f}"

    ax.text(
        0.02,
        0.98,
        stats_text,
        transform=ax.transAxes,
        fontsize=10,
        verticalalignment="top",
        bbox={"boxstyle": "round", "facecolor": "black", "alpha": 0.8},
    )

    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    logger.info(f"Saved: {output_file}")
    plt.close()


def plot_probability_vs_moneyness_by_regime(df: pl.DataFrame, output_file: Path) -> None:
    """Plot probability vs moneyness split by volatility regime."""
    logger.info("Generating probability vs moneyness by regime")

    # Define volatility regimes (terciles)
    vol_quantiles = df.select(
        [
            pl.col("rv_900s").quantile(0.33).alias("q33"),
            pl.col("rv_900s").quantile(0.67).alias("q67"),
        ]
    ).row(0)

    # Assign regimes
    df_regimes = df.with_columns(
        [
            pl.when(pl.col("rv_900s") < vol_quantiles[0])
            .then(pl.lit("Low Vol"))
            .when(pl.col("rv_900s") < vol_quantiles[1])
            .then(pl.lit("Mid Vol"))
            .otherwise(pl.lit("High Vol"))
            .alias("vol_regime"),
        ]
    )

    # Bin by moneyness for each regime
    moneyness_bins = np.linspace(-2, 2, 40)

    fig, ax = plt.subplots(figsize=(14, 8))

    colors = {"Low Vol": "#00FF88", "Mid Vol": "#00D4FF", "High Vol": "#FF3366"}

    for regime in ["Low Vol", "Mid Vol", "High Vol"]:
        df_regime = df_regimes.filter(pl.col("vol_regime") == regime)

        # Bin and aggregate
        df_binned = (
            df_regime.with_columns(
                [
                    pl.col("moneyness_pct").cut(moneyness_bins).alias("m_bin"),
                ]
            )
            .group_by("m_bin")
            .agg(
                [
                    pl.len().alias("count"),
                    pl.col("prob_mid").mean().alias("avg_prob"),
                    pl.col("moneyness_pct").mean().alias("avg_m"),
                ]
            )
            .filter(pl.col("count") >= 50)
            .sort("avg_m")
        )

        ax.plot(
            df_binned["avg_m"].to_numpy(),
            df_binned["avg_prob"].to_numpy(),
            linewidth=2.5,
            label=regime,
            color=colors[regime],
            alpha=0.9,
        )

    # Styling
    ax.set_xlabel("Moneyness (%)", fontsize=14)
    ax.set_ylabel("Average Probability", fontsize=14)
    ax.set_title("Binary Option Payoff Curve by Volatility Regime", fontsize=16, pad=20)
    ax.grid(alpha=0.2)
    ax.legend(fontsize=12, frameon=True, framealpha=0.9)
    ax.axhline(0.5, color="gray", linestyle="--", alpha=0.5, label="50%")
    ax.axvline(0, color="gray", linestyle=":", alpha=0.5, label="ATM")
    ax.set_ylim(0, 1)

    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    logger.info(f"Saved: {output_file}")
    plt.close()


def generate_delta_statistics_table(delta_df: pl.DataFrame, gamma_df: pl.DataFrame) -> None:
    """Generate comprehensive delta/gamma statistics table."""
    logger.info("Generating delta/gamma statistics table")

    # Delta statistics
    delta_stats = delta_df.select(
        [
            pl.lit("Delta").alias("metric"),
            pl.col("delta").min().alias("min"),
            pl.col("delta").quantile(0.25).alias("q25"),
            pl.col("delta").median().alias("median"),
            pl.col("delta").quantile(0.75).alias("q75"),
            pl.col("delta").max().alias("max"),
            pl.col("delta").mean().alias("mean"),
            pl.col("delta").std().alias("std"),
        ]
    )

    # Gamma statistics
    gamma_stats = gamma_df.select(
        [
            pl.lit("Gamma").alias("metric"),
            pl.col("gamma").min().alias("min"),
            pl.col("gamma").quantile(0.25).alias("q25"),
            pl.col("gamma").median().alias("median"),
            pl.col("gamma").quantile(0.75).alias("q75"),
            pl.col("gamma").max().alias("max"),
            pl.col("gamma").mean().alias("mean"),
            pl.col("gamma").std().alias("std"),
        ]
    )

    # Combine
    stats_table = pl.concat([delta_stats, gamma_stats])

    # Save
    output_file = TABLE_DIR / "delta_gamma_statistics.csv"
    stats_table.write_csv(output_file)
    logger.info(f"Saved: {output_file}")

    # Display
    logger.info("\n" + str(stats_table))


def main() -> None:
    """Run Phase 1: Payoff & Non-Linearity Analysis."""
    logger.info("=" * 80)
    logger.info("PHASE 1: PAYOFF & NON-LINEARITY ANALYSIS")
    logger.info("=" * 80)

    # Load data (sample 10% for speed, increase to 1.0 for full analysis)
    df = load_test_data(sample_frac=0.1)

    # Calculate delta surface
    delta_df = calculate_empirical_delta(df, moneyness_bins=50, min_samples=100)

    # Calculate gamma
    gamma_df = calculate_gamma(delta_df)

    # Generate plots
    plot_delta_surface_2d(delta_df, FIGURE_DIR / "01_delta_surface_2d.png")
    plot_gamma_surface(gamma_df, FIGURE_DIR / "02_gamma_surface.png")
    plot_probability_vs_moneyness_by_regime(df, FIGURE_DIR / "03_payoff_by_vol_regime.png")

    # Generate statistics table
    generate_delta_statistics_table(delta_df, gamma_df)

    logger.info("=" * 80)
    logger.info("PHASE 1 COMPLETE")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
