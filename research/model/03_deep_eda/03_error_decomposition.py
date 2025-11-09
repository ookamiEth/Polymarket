#!/usr/bin/env python3
"""
Phase 3: Prediction Error Decomposition Analysis
================================================

Deep analysis of where and when the model makes errors.

Analysis Goals:
1. Error decomposition by moneyness, volatility, time-of-day
2. Conditional error analysis (errors during jumps, high vol, etc.)
3. Temporal clustering (are bad predictions clustered in time?)
4. Feature impact on errors
5. Tail event analysis (performance during extreme moves)
"""

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import polars as pl

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


def load_test_data(sample_frac: float = 0.1) -> pl.DataFrame:
    """Load test set with sampling."""
    logger.info(f"Loading test data (sample fraction: {sample_frac})")

    df = pl.read_parquet("research/model/results/xgboost_residual_model_test_2024-07-01_2025-09-30.parquet")
    logger.info(f"Loaded {len(df):,} rows")

    if sample_frac < 1.0:
        df = df.sample(fraction=sample_frac, seed=42)
        logger.info(f"Sampled to {len(df):,} rows")

    # Add derived columns
    df = df.with_columns(
        [
            (pl.col("moneyness") * 100).alias("moneyness_pct"),
            (pl.col("time_remaining") / 60).alias("time_remaining_min"),
            pl.from_epoch("timestamp", time_unit="s").dt.hour().alias("hour_of_day"),
            pl.from_epoch("timestamp", time_unit="s").dt.date().alias("date"),
        ]
    )

    # Add error columns (need to compute in separate step to reference)
    df = df.with_columns(
        [
            (pl.col("outcome") - pl.col("prob_mid")).alias("error_bs"),
            (pl.col("outcome") - pl.col("prob_corrected_xgb")).alias("error_ml"),
        ]
    )

    df = df.with_columns(
        [
            pl.col("error_bs").abs().alias("abs_error_bs"),
            pl.col("error_ml").abs().alias("abs_error_ml"),
        ]
    )

    return df


def plot_error_heatmap_2d(df: pl.DataFrame, output_file: Path) -> None:
    """2D heatmap: Error magnitude by (moneyness, volatility)."""
    logger.info("Generating 2D error heatmap (moneyness × volatility)")

    # Create bins
    moneyness_bins = np.linspace(-2, 2, 25).tolist()
    vol_quantiles_array = np.linspace(0, 100, 25)
    vol_bins_array = np.percentile(df["rv_900s"].to_numpy(), vol_quantiles_array)
    vol_bins = vol_bins_array.tolist()

    # Aggregate errors
    df_grid = (
        df.with_columns(
            [
                pl.col("moneyness_pct").cut(moneyness_bins).alias("m_bin"),
                pl.col("rv_900s").cut(vol_bins).alias("v_bin"),
            ]
        )
        .group_by(["m_bin", "v_bin"])
        .agg(
            [
                pl.len().alias("count"),
                pl.col("moneyness_pct").mean().alias("avg_m"),
                pl.col("rv_900s").mean().alias("avg_vol"),
                pl.col("abs_error_ml").mean().alias("mae_ml"),
                pl.col("abs_error_bs").mean().alias("mae_bs"),
            ]
        )
        .filter(pl.col("count") >= 50)
    )

    # Pivot for heatmap
    moneyness_vals = sorted(df_grid["avg_m"].unique().to_list())
    vol_vals = sorted(df_grid["avg_vol"].unique().to_list())

    # Create grid
    grid_ml = np.full((len(vol_vals), len(moneyness_vals)), np.nan)

    for row in df_grid.iter_rows(named=True):
        try:
            m_idx = moneyness_vals.index(row["avg_m"])
            v_idx = vol_vals.index(row["avg_vol"])
            grid_ml[v_idx, m_idx] = row["mae_ml"]
        except (ValueError, IndexError):
            continue

    # Plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 7))

    # ML model errors
    im1 = ax1.imshow(grid_ml, aspect="auto", cmap="YlOrRd", origin="lower", interpolation="nearest")
    ax1.set_xlabel("Moneyness (%)", fontsize=14)
    ax1.set_ylabel("Realized Volatility (15min)", fontsize=14)
    ax1.set_title("ML Model: Mean Absolute Error Heatmap", fontsize=16, pad=20)

    # Set tick labels
    ax1.set_xticks(np.linspace(0, len(moneyness_vals) - 1, 5))
    ax1.set_xticklabels([f"{moneyness_vals[int(i)]:.1f}" for i in np.linspace(0, len(moneyness_vals) - 1, 5)])
    ax1.set_yticks(np.linspace(0, len(vol_vals) - 1, 5))
    ax1.set_yticklabels([f"{vol_vals[int(i)]:.3f}" for i in np.linspace(0, len(vol_vals) - 1, 5)])

    cbar1 = plt.colorbar(im1, ax=ax1)
    cbar1.set_label("MAE", fontsize=12)

    # BS model errors (for comparison)
    grid_bs = np.full((len(vol_vals), len(moneyness_vals)), np.nan)
    for row in df_grid.iter_rows(named=True):
        try:
            m_idx = moneyness_vals.index(row["avg_m"])
            v_idx = vol_vals.index(row["avg_vol"])
            grid_bs[v_idx, m_idx] = row["mae_bs"]
        except (ValueError, IndexError):
            continue

    im2 = ax2.imshow(grid_bs, aspect="auto", cmap="YlOrRd", origin="lower", interpolation="nearest")
    ax2.set_xlabel("Moneyness (%)", fontsize=14)
    ax2.set_ylabel("Realized Volatility (15min)", fontsize=14)
    ax2.set_title("BS Baseline: Mean Absolute Error Heatmap", fontsize=16, pad=20)

    ax2.set_xticks(np.linspace(0, len(moneyness_vals) - 1, 5))
    ax2.set_xticklabels([f"{moneyness_vals[int(i)]:.1f}" for i in np.linspace(0, len(moneyness_vals) - 1, 5)])
    ax2.set_yticks(np.linspace(0, len(vol_vals) - 1, 5))
    ax2.set_yticklabels([f"{vol_vals[int(i)]:.3f}" for i in np.linspace(0, len(vol_vals) - 1, 5)])

    cbar2 = plt.colorbar(im2, ax=ax2)
    cbar2.set_label("MAE", fontsize=12)

    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    logger.info(f"Saved: {output_file}")
    plt.close()


def plot_errors_by_moneyness(df: pl.DataFrame, output_file: Path) -> None:
    """Error distribution by moneyness buckets."""
    logger.info("Analyzing errors by moneyness")

    moneyness_bins = np.linspace(-2, 2, 30)

    df_errors = (
        df.with_columns([pl.col("moneyness_pct").cut(moneyness_bins).alias("m_bin")])
        .group_by("m_bin")
        .agg(
            [
                pl.len().alias("count"),
                pl.col("moneyness_pct").mean().alias("avg_m"),
                pl.col("error_ml").mean().alias("mean_error_ml"),
                pl.col("error_ml").std().alias("std_error_ml"),
                pl.col("abs_error_ml").mean().alias("mae_ml"),
                pl.col("abs_error_bs").mean().alias("mae_bs"),
            ]
        )
        .filter(pl.col("count") >= 100)
        .sort("avg_m")
    )

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    moneyness = df_errors["avg_m"].to_numpy()

    # Mean error (bias)
    ax1.plot(
        moneyness,
        df_errors["mean_error_ml"].to_numpy(),
        linewidth=2.5,
        color="#00D4FF",
        marker="o",
        markersize=6,
        label="ML Model",
    )
    ax1.fill_between(
        moneyness,
        df_errors["mean_error_ml"].to_numpy() - df_errors["std_error_ml"].to_numpy(),
        df_errors["mean_error_ml"].to_numpy() + df_errors["std_error_ml"].to_numpy(),
        alpha=0.3,
        color="#00D4FF",
    )
    ax1.axhline(0, color="gray", linestyle="--", alpha=0.5)
    ax1.axvline(0, color="gray", linestyle=":", alpha=0.5)
    ax1.set_xlabel("Moneyness (%)", fontsize=14)
    ax1.set_ylabel("Mean Prediction Error", fontsize=14)
    ax1.set_title("Prediction Bias by Moneyness", fontsize=16, pad=20)
    ax1.grid(alpha=0.2)
    ax1.legend(fontsize=12)

    # MAE comparison
    ax2.plot(
        moneyness,
        df_errors["mae_bs"].to_numpy(),
        linewidth=2.5,
        color="#FF3366",
        marker="s",
        markersize=6,
        label="BS Baseline",
        alpha=0.7,
    )
    ax2.plot(
        moneyness,
        df_errors["mae_ml"].to_numpy(),
        linewidth=2.5,
        color="#00FF88",
        marker="o",
        markersize=6,
        label="ML Model",
    )
    ax2.axvline(0, color="gray", linestyle=":", alpha=0.5)
    ax2.set_xlabel("Moneyness (%)", fontsize=14)
    ax2.set_ylabel("Mean Absolute Error", fontsize=14)
    ax2.set_title("Model Accuracy by Moneyness", fontsize=16, pad=20)
    ax2.grid(alpha=0.2)
    ax2.legend(fontsize=12)

    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    logger.info(f"Saved: {output_file}")
    plt.close()


def plot_conditional_errors(df: pl.DataFrame, output_file: Path) -> None:
    """Errors conditioned on market events (jumps, high vol, etc.)."""
    logger.info("Analyzing conditional errors")

    # Define conditions
    conditions = {
        "No Jump": df.filter(pl.col("jump_detected") == 0),
        "Jump": df.filter(pl.col("jump_detected") == 1),
        "Low Vol": df.filter(pl.col("rv_900s") < df["rv_900s"].quantile(0.33)),
        "High Vol": df.filter(pl.col("rv_900s") > df["rv_900s"].quantile(0.67)),
        "Fresh IV": df.filter(pl.col("iv_staleness_seconds") < 60),
        "Stale IV": df.filter(pl.col("iv_staleness_seconds") > 300),
    }

    # Calculate stats for each condition
    stats = []
    for condition_name, df_cond in conditions.items():
        mae_bs_val = df_cond["abs_error_bs"].mean()
        mae_ml_val = df_cond["abs_error_ml"].mean()

        # Type narrowing and conversion
        mae_bs_float = float(mae_bs_val) if mae_bs_val is not None else 0.0
        mae_ml_float = float(mae_ml_val) if mae_ml_val is not None else 0.0

        improvement_val = (1 - mae_ml_float / mae_bs_float) * 100 if mae_bs_float > 0 else 0.0

        stats.append(
            {
                "condition": condition_name,
                "count": len(df_cond),
                "mae_bs": mae_bs_float,
                "mae_ml": mae_ml_float,
                "improvement": improvement_val,
            }
        )

    df_stats = pl.DataFrame(stats)

    # Plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    x = np.arange(len(df_stats))
    width = 0.35

    # MAE comparison
    ax1.bar(
        x - width / 2,
        df_stats["mae_bs"].to_numpy(),
        width,
        label="BS Baseline",
        color="#FF3366",
        alpha=0.8,
        edgecolor="white",
        linewidth=0.5,
    )
    ax1.bar(
        x + width / 2,
        df_stats["mae_ml"].to_numpy(),
        width,
        label="ML Model",
        color="#00FF88",
        alpha=0.8,
        edgecolor="white",
        linewidth=0.5,
    )
    ax1.set_xlabel("Condition", fontsize=14)
    ax1.set_ylabel("Mean Absolute Error", fontsize=14)
    ax1.set_title("Error by Market Condition", fontsize=16, pad=20)
    ax1.set_xticks(x)
    ax1.set_xticklabels(df_stats["condition"].to_list(), rotation=45, ha="right")
    ax1.legend(fontsize=12)
    ax1.grid(alpha=0.2, axis="y")

    # Improvement percentage
    colors_improvement = ["#00FF88" if val > 0 else "#FF3366" for val in df_stats["improvement"].to_list()]
    ax2.bar(
        x, df_stats["improvement"].to_numpy(), color=colors_improvement, alpha=0.8, edgecolor="white", linewidth=0.5
    )
    ax2.axhline(0, color="gray", linestyle="--", alpha=0.5)
    ax2.set_xlabel("Condition", fontsize=14)
    ax2.set_ylabel("ML Improvement (%)", fontsize=14)
    ax2.set_title("ML Model Improvement over BS Baseline", fontsize=16, pad=20)
    ax2.set_xticks(x)
    ax2.set_xticklabels(df_stats["condition"].to_list(), rotation=45, ha="right")
    ax2.grid(alpha=0.2, axis="y")

    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    logger.info(f"Saved: {output_file}")
    plt.close()

    # Save table
    df_stats.write_csv(TABLE_DIR / "conditional_error_analysis.csv")
    logger.info(f"Saved: {TABLE_DIR / 'conditional_error_analysis.csv'}")


def plot_temporal_error_clustering(df: pl.DataFrame, output_file: Path) -> None:
    """Analyze if errors cluster in time."""
    logger.info("Analyzing temporal error clustering")

    # Aggregate by date
    df_daily = (
        df.group_by("date")
        .agg(
            [
                pl.len().alias("count"),
                pl.col("abs_error_ml").mean().alias("mae_ml"),
                pl.col("abs_error_bs").mean().alias("mae_bs"),
                pl.col("rv_900s").mean().alias("avg_vol"),
            ]
        )
        .sort("date")
    )

    dates = df_daily["date"].to_numpy()
    mae_ml = df_daily["mae_ml"].to_numpy()
    mae_bs = df_daily["mae_bs"].to_numpy()
    avg_vol = df_daily["avg_vol"].to_numpy()

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 10), sharex=True)

    # Daily MAE
    ax1.plot(dates, mae_bs, linewidth=2, color="#FF3366", alpha=0.7, label="BS Baseline")
    ax1.plot(dates, mae_ml, linewidth=2, color="#00FF88", label="ML Model")
    ax1.fill_between(dates, mae_bs, mae_ml, where=(mae_ml < mae_bs), color="#00FF88", alpha=0.3, label="ML Improvement")
    ax1.fill_between(dates, mae_bs, mae_ml, where=(mae_ml >= mae_bs), color="#FF3366", alpha=0.3, label="ML Worse")
    ax1.set_ylabel("Mean Absolute Error", fontsize=14)
    ax1.set_title("Daily Model Performance Over Time", fontsize=16, pad=20)
    ax1.legend(fontsize=11)
    ax1.grid(alpha=0.2)

    # Daily volatility (for context)
    ax2.plot(dates, avg_vol, linewidth=2, color="#FFB000")
    ax2.fill_between(dates, 0, avg_vol, alpha=0.3, color="#FFB000")
    ax2.set_xlabel("Date", fontsize=14)
    ax2.set_ylabel("Average RV (15min)", fontsize=14)
    ax2.set_title("Realized Volatility Context", fontsize=16, pad=20)
    ax2.grid(alpha=0.2)

    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    logger.info(f"Saved: {output_file}")
    plt.close()


def plot_tail_event_performance(df: pl.DataFrame, output_file: Path) -> None:
    """Performance during extreme moves (tail events)."""
    logger.info("Analyzing tail event performance")

    # Define tail events based on absolute moneyness
    df_tail = df.with_columns(
        [
            pl.col("moneyness_pct").abs().alias("abs_moneyness"),
        ]
    )

    # Percentiles of absolute moneyness
    percentiles = [50, 75, 90, 95, 99]
    quantile_vals = [df_tail["abs_moneyness"].quantile(p / 100) for p in percentiles]

    # Calculate errors for each tail group
    tail_stats = []
    for i, (p, q) in enumerate(zip(percentiles, quantile_vals)):
        if i == 0:
            df_group = df_tail.filter(pl.col("abs_moneyness") <= q)
            label = f"<{p}th pct"
        else:
            df_group = df_tail.filter((pl.col("abs_moneyness") > quantile_vals[i - 1]) & (pl.col("abs_moneyness") <= q))
            label = f"{percentiles[i - 1]}-{p}th pct"

        tail_stats.append(
            {
                "percentile": label,
                "count": len(df_group),
                "mae_bs": df_group["abs_error_bs"].mean(),
                "mae_ml": df_group["abs_error_ml"].mean(),
                "median_abs_m": df_group["abs_moneyness"].median(),
            }
        )

    # Add extreme tail (>99th)
    df_extreme = df_tail.filter(pl.col("abs_moneyness") > quantile_vals[-1])
    tail_stats.append(
        {
            "percentile": ">99th pct",
            "count": len(df_extreme),
            "mae_bs": df_extreme["abs_error_bs"].mean(),
            "mae_ml": df_extreme["abs_error_ml"].mean(),
            "median_abs_m": df_extreme["abs_moneyness"].median(),
        }
    )

    df_tail_stats = pl.DataFrame(tail_stats)

    # Plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    x = np.arange(len(df_tail_stats))
    width = 0.35

    # MAE by tail group
    ax1.bar(
        x - width / 2,
        df_tail_stats["mae_bs"].to_numpy(),
        width,
        label="BS Baseline",
        color="#FF3366",
        alpha=0.8,
        edgecolor="white",
        linewidth=0.5,
    )
    ax1.bar(
        x + width / 2,
        df_tail_stats["mae_ml"].to_numpy(),
        width,
        label="ML Model",
        color="#00FF88",
        alpha=0.8,
        edgecolor="white",
        linewidth=0.5,
    )
    ax1.set_xlabel("Moneyness Percentile Group", fontsize=14)
    ax1.set_ylabel("Mean Absolute Error", fontsize=14)
    ax1.set_title("Tail Event Performance", fontsize=16, pad=20)
    ax1.set_xticks(x)
    ax1.set_xticklabels(df_tail_stats["percentile"].to_list(), rotation=45, ha="right")
    ax1.legend(fontsize=12)
    ax1.grid(alpha=0.2, axis="y")

    # Sample size
    ax2.bar(x, df_tail_stats["count"].to_numpy(), color="#00D4FF", alpha=0.8, edgecolor="white", linewidth=0.5)
    ax2.set_xlabel("Moneyness Percentile Group", fontsize=14)
    ax2.set_ylabel("Sample Count", fontsize=14)
    ax2.set_title("Sample Distribution by Tail Group", fontsize=16, pad=20)
    ax2.set_xticks(x)
    ax2.set_xticklabels(df_tail_stats["percentile"].to_list(), rotation=45, ha="right")
    ax2.grid(alpha=0.2, axis="y")
    ax2.set_yscale("log")

    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    logger.info(f"Saved: {output_file}")
    plt.close()

    # Save table
    df_tail_stats.write_csv(TABLE_DIR / "tail_event_performance.csv")
    logger.info(f"Saved: {TABLE_DIR / 'tail_event_performance.csv'}")


def main() -> None:
    """Run Phase 3: Prediction Error Decomposition Analysis."""
    logger.info("=" * 80)
    logger.info("PHASE 3: PREDICTION ERROR DECOMPOSITION ANALYSIS")
    logger.info("=" * 80)

    # Load data
    df = load_test_data(sample_frac=0.1)

    # Run analyses
    plot_error_heatmap_2d(df, FIGURE_DIR / "09_error_heatmap_2d.png")
    plot_errors_by_moneyness(df, FIGURE_DIR / "10_errors_by_moneyness.png")
    plot_conditional_errors(df, FIGURE_DIR / "11_conditional_errors.png")
    plot_temporal_error_clustering(df, FIGURE_DIR / "12_temporal_error_clustering.png")
    plot_tail_event_performance(df, FIGURE_DIR / "13_tail_event_performance.png")

    logger.info("=" * 80)
    logger.info("PHASE 3 COMPLETE")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
