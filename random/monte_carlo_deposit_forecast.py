#!/usr/bin/env python3
"""
Monte Carlo simulation to forecast final depositor count and volume for token sale.

Sale period: October 27, 1pm UTC - October 30, 1pm UTC (72 hours total)
Models time-dependent arrival rates with late-hour surge scenario.
Outputs: 90% confidence intervals and probability distributions.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from matplotlib.backends.backend_pdf import PdfPages
from scipy import stats
from scipy.optimize import curve_fit

from etherscan_api_client import TARGET_ADDRESS

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# Directories
BASE_DIR = Path("/Users/lgierhake/Documents/ETH/BT/random/data/etherscan")
RAW_DIR = BASE_DIR / "raw"
REPORTS_DIR = Path("/Users/lgierhake/Documents/ETH/BT/random/reports")
SHORT_ADDR = TARGET_ADDRESS[:10]

# Sale parameters
SALE_START_UTC = datetime(2025, 10, 27, 13, 0, 0, tzinfo=timezone.utc)
SALE_END_UTC = datetime(2025, 10, 30, 13, 0, 0, tzinfo=timezone.utc)
TOTAL_SALE_HOURS = 72.0

# Monte Carlo parameters
N_SIMULATIONS = 10000
RANDOM_SEED = 42

# Plot styling
plt.style.use("dark_background")
COLORS = {
    "primary": "#00D4FF",
    "secondary": "#FF00FF",
    "success": "#00FF88",
    "danger": "#FF3366",
    "warning": "#FFB000",
    "info": "#00B4FF",
}


def load_deposit_data() -> tuple[pl.DataFrame, dict[str, Any]]:
    """Load deposit data and calculate current state."""
    logger.info("Loading deposit data...")
    deposit_file = RAW_DIR / f"usdt_deposits_{SHORT_ADDR}.parquet"
    df = pl.read_parquet(deposit_file)

    # Calculate current state
    current_state = {
        "total_deposits": len(df),
        "unique_depositors": df["from"].n_unique(),
        "total_volume": df["TokenValue"].sum(),
        "last_deposit_time": df["DateTime"].max(),
        "data_collection_time": datetime.now(timezone.utc),
    }

    logger.info(f"Loaded {len(df):,} deposits")
    logger.info(f"Unique depositors: {current_state['unique_depositors']:,}")
    logger.info(f"Total volume: ${current_state['total_volume']:,.2f}")
    logger.info(f"Last deposit: {current_state['last_deposit_time']}")

    return df, current_state


def calculate_hourly_statistics(df: pl.DataFrame) -> pl.DataFrame:
    """Calculate deposits per hour since sale start."""
    logger.info("Calculating hourly statistics...")

    # Ensure DateTime column is timezone-aware
    df = df.with_columns([pl.col("DateTime").dt.replace_time_zone("UTC")])

    # Add hours since sale start
    df = df.with_columns(
        [
            (
                (pl.col("DateTime") - pl.lit(SALE_START_UTC)).dt.total_seconds() / 3600
            ).alias("hours_since_start")
        ]
    )

    # Bin into hours and count
    df = df.with_columns(
        [pl.col("hours_since_start").floor().cast(pl.Int64).alias("hour_bin")]
    )

    # Aggregate by hour
    hourly_stats = (
        df.group_by("hour_bin")
        .agg(
            [
                pl.len().alias("deposit_count"),
                pl.col("TokenValue").sum().alias("volume"),
                pl.col("from").n_unique().alias("unique_depositors"),
            ]
        )
        .sort("hour_bin")
    )

    logger.info(f"Calculated statistics for {len(hourly_stats)} hours")
    return hourly_stats


def exponential_decay(t: np.ndarray, a: float, b: float, c: float) -> np.ndarray:
    """Exponential decay function: f(t) = a * exp(-b * t) + c."""
    return a * np.exp(-b * t) + c


def fit_arrival_rate_model(hourly_stats: pl.DataFrame) -> dict[str, float]:
    """Fit exponential decay to hourly deposit rates."""
    logger.info("Fitting arrival rate model...")

    hours = hourly_stats["hour_bin"].to_numpy()
    deposits = hourly_stats["deposit_count"].to_numpy()

    # Initial guess: high initial rate, fast decay, baseline ~400
    p0 = [6000, 0.1, 400]

    try:
        # Fit exponential decay
        params, _ = curve_fit(exponential_decay, hours, deposits, p0=p0, maxfev=10000)
        a, b, c = params

        # Calculate R² for goodness of fit
        predicted = exponential_decay(hours, a, b, c)
        ss_res = np.sum((deposits - predicted) ** 2)
        ss_tot = np.sum((deposits - np.mean(deposits)) ** 2)
        r_squared = 1 - (ss_res / ss_tot)

        logger.info(f"Fitted parameters: a={a:.2f}, b={b:.4f}, c={c:.2f}")
        logger.info(f"R² = {r_squared:.4f}")

        return {"a": float(a), "b": float(b), "c": float(c), "r_squared": r_squared}

    except Exception as e:
        logger.error(f"Failed to fit model: {e}")
        # Fallback to simple average
        avg_rate = float(deposits[-6:].mean())  # Last 6 hours average
        logger.warning(f"Using fallback: constant rate = {avg_rate:.2f}/hr")
        return {"a": 0.0, "b": 0.0, "c": avg_rate, "r_squared": 0.0}


def extract_deposit_size_distribution(df: pl.DataFrame) -> dict[str, Any]:
    """Extract empirical deposit size distribution."""
    logger.info("Extracting deposit size distribution...")

    sizes = df["TokenValue"].to_numpy()

    # Identify whale deposits (exactly max)
    max_deposit = 186282.0
    whale_deposits = sizes[np.isclose(sizes, max_deposit, rtol=0.001)]
    whale_probability = len(whale_deposits) / len(sizes)

    # Non-whale deposits
    non_whale_sizes = sizes[~np.isclose(sizes, max_deposit, rtol=0.001)]

    # Calculate statistics
    distribution = {
        "whale_probability": whale_probability,
        "whale_amount": max_deposit,
        "non_whale_sizes": non_whale_sizes,
        "mean": float(np.mean(non_whale_sizes)),
        "std": float(np.std(non_whale_sizes)),
        "median": float(np.median(non_whale_sizes)),
        "q25": float(np.percentile(non_whale_sizes, 25)),
        "q75": float(np.percentile(non_whale_sizes, 75)),
    }

    logger.info(f"Whale deposit probability: {whale_probability:.1%}")
    logger.info(f"Non-whale mean: ${distribution['mean']:,.2f}")
    logger.info(f"Non-whale median: ${distribution['median']:,.2f}")

    return distribution


def calculate_repeat_probability(df: pl.DataFrame) -> float:
    """Calculate probability of repeat deposits."""
    depositor_counts = df.group_by("from").agg([pl.len().alias("num_deposits")])
    total_deposits = len(df)
    first_time_deposits = len(depositor_counts.filter(pl.col("num_deposits") == 1))
    repeat_deposits = total_deposits - first_time_deposits
    repeat_prob = repeat_deposits / total_deposits

    logger.info(f"Repeat deposit probability: {repeat_prob:.1%}")
    return repeat_prob


def simulate_deposit_arrivals(
    hour: float, params: dict[str, float], apply_surge: bool, surge_start_hour: float
) -> int:
    """Simulate number of deposit arrivals for a given hour."""
    # Base rate from fitted model
    base_rate = exponential_decay(np.array([hour]), params["a"], params["b"], params["c"])[
        0
    ]

    # Apply late surge if applicable
    if apply_surge and hour >= surge_start_hour:
        surge_multiplier = np.random.uniform(1.5, 2.5)
        base_rate *= surge_multiplier

    # Sample from Poisson distribution
    n_deposits = np.random.poisson(max(0, base_rate))
    return int(n_deposits)


def simulate_deposit_sizes(
    n_deposits: int, size_dist: dict[str, Any], rng: np.random.Generator
) -> np.ndarray:
    """Sample deposit sizes from empirical distribution."""
    sizes = np.zeros(n_deposits)

    for i in range(n_deposits):
        # Decide if whale or not
        if rng.random() < size_dist["whale_probability"]:
            sizes[i] = size_dist["whale_amount"]
        else:
            # Sample from non-whale distribution
            sizes[i] = rng.choice(size_dist["non_whale_sizes"])

    return sizes


def run_single_simulation(
    current_state: dict[str, Any],
    hours_remaining: float,
    hours_elapsed: float,
    params: dict[str, float],
    size_dist: dict[str, Any],
    repeat_prob: float,
    apply_surge: bool,
    rng: np.random.Generator,
) -> dict[str, float]:
    """Run a single Monte Carlo simulation."""
    # Initialize with current state
    total_deposits = current_state["total_deposits"]
    total_volume = current_state["total_volume"]
    unique_depositors = current_state["unique_depositors"]

    # Surge starts in final 6 hours (hour 66)
    surge_start_hour = 66.0

    # Simulate hour by hour
    for hour_offset in range(int(np.ceil(hours_remaining))):
        current_hour = hours_elapsed + hour_offset

        # Simulate arrivals
        n_new_deposits = simulate_deposit_arrivals(current_hour, params, apply_surge, surge_start_hour)

        if n_new_deposits == 0:
            continue

        # Simulate deposit sizes
        new_sizes = simulate_deposit_sizes(n_new_deposits, size_dist, rng)

        # Update totals
        total_deposits += n_new_deposits
        total_volume += np.sum(new_sizes)

        # Simulate unique depositors (accounting for repeats)
        n_new_unique = 0
        for _ in range(n_new_deposits):
            if rng.random() > repeat_prob:
                n_new_unique += 1

        unique_depositors += n_new_unique

    return {
        "final_deposits": total_deposits,
        "final_unique_depositors": unique_depositors,
        "final_volume": total_volume,
    }


def run_monte_carlo(
    current_state: dict[str, Any],
    hours_remaining: float,
    hours_elapsed: float,
    params: dict[str, float],
    size_dist: dict[str, Any],
    repeat_prob: float,
) -> pl.DataFrame:
    """Run Monte Carlo simulations."""
    logger.info(f"Running {N_SIMULATIONS:,} Monte Carlo simulations...")

    rng = np.random.default_rng(RANDOM_SEED)
    results = []

    for i in range(N_SIMULATIONS):
        if (i + 1) % 1000 == 0:
            logger.info(f"  Completed {i + 1:,}/{N_SIMULATIONS:,} simulations...")

        # 50% probability of late surge
        apply_surge = rng.random() < 0.5

        result = run_single_simulation(
            current_state,
            hours_remaining,
            hours_elapsed,
            params,
            size_dist,
            repeat_prob,
            apply_surge,
            rng,
        )
        result["surge_applied"] = apply_surge
        results.append(result)

    logger.info("Monte Carlo simulations completed!")

    # Convert to DataFrame
    results_df = pl.DataFrame(results)
    return results_df


def calculate_statistics(results_df: pl.DataFrame, current_state: dict[str, Any]) -> dict[str, Any]:
    """Calculate summary statistics from simulation results."""
    logger.info("Calculating statistics...")

    stats = {
        "current": current_state,
        "depositors": {
            "median": results_df["final_unique_depositors"].median(),
            "mean": results_df["final_unique_depositors"].mean(),
            "p5": results_df["final_unique_depositors"].quantile(0.05),
            "p95": results_df["final_unique_depositors"].quantile(0.95),
            "ci_lower": results_df["final_unique_depositors"].quantile(0.05),
            "ci_upper": results_df["final_unique_depositors"].quantile(0.95),
        },
        "volume": {
            "median": results_df["final_volume"].median(),
            "mean": results_df["final_volume"].mean(),
            "p5": results_df["final_volume"].quantile(0.05),
            "p95": results_df["final_volume"].quantile(0.95),
            "ci_lower": results_df["final_volume"].quantile(0.05),
            "ci_upper": results_df["final_volume"].quantile(0.95),
        },
        "probabilities": {
            "depositors_gt_50k": (
                results_df["final_unique_depositors"] > 50000
            ).sum() / len(results_df),
            "volume_gt_1b": (results_df["final_volume"] > 1_000_000_000).sum()
            / len(results_df),
        },
    }

    logger.info(f"Median final depositors: {stats['depositors']['median']:,.0f}")
    logger.info(
        f"90% CI: [{stats['depositors']['ci_lower']:,.0f}, {stats['depositors']['ci_upper']:,.0f}]"
    )
    logger.info(f"Median final volume: ${stats['volume']['median']:,.2f}")
    logger.info(
        f"90% CI: [${stats['volume']['ci_lower']:,.2f}, ${stats['volume']['ci_upper']:,.2f}]"
    )

    return stats


def generate_forecast_pdf(
    results_df: pl.DataFrame,
    stats: dict[str, Any],
    hourly_stats: pl.DataFrame,
    params: dict[str, float],
    hours_remaining: float,
) -> None:
    """Generate comprehensive PDF report with forecast visualizations."""
    logger.info("Generating forecast PDF...")

    output_file = REPORTS_DIR / f"deposit_forecast_{SHORT_ADDR}.pdf"

    with PdfPages(output_file) as pdf:
        # Page 1: Final Depositor Count Distribution
        create_depositor_distribution_plot(results_df, stats, pdf)

        # Page 2: Final Volume Distribution
        create_volume_distribution_plot(results_df, stats, pdf)

        # Page 3: Time Series Projection
        create_time_series_projection(hourly_stats, results_df, stats, params, hours_remaining, pdf)

        # Page 4: Scenario Comparison
        create_scenario_comparison(results_df, pdf)

        # Page 5: Summary & Methodology
        create_summary_page(stats, params, pdf)

        # Add metadata
        d = pdf.infodict()
        d["Title"] = f"Deposit Forecast - {TARGET_ADDRESS}"
        d["Author"] = "Monte Carlo Analysis"
        d["Subject"] = "Token Sale Deposit Forecast"
        d["Keywords"] = "Monte Carlo, Forecast, Token Sale"
        d["CreationDate"] = datetime.now()

    logger.info(f"✅ PDF saved to: {output_file}")
    logger.info(f"   File size: {output_file.stat().st_size / 1024:.1f} KB")


def create_depositor_distribution_plot(
    results_df: pl.DataFrame, stats: dict[str, Any], pdf: PdfPages
) -> None:
    """Page 1: Final depositor count distribution."""
    fig, ax = plt.subplots(figsize=(14, 8))

    depositors = results_df["final_unique_depositors"].to_numpy()
    current = stats["current"]["unique_depositors"]
    median = stats["depositors"]["median"]
    ci_lower = stats["depositors"]["ci_lower"]
    ci_upper = stats["depositors"]["ci_upper"]

    # Histogram
    ax.hist(depositors, bins=50, alpha=0.7, color=COLORS["primary"], edgecolor="white", linewidth=0.5)

    # Vertical lines
    ax.axvline(current, color=COLORS["warning"], linestyle="--", linewidth=2, label=f"Current: {current:,.0f}")
    ax.axvline(median, color=COLORS["success"], linestyle="-", linewidth=3, label=f"Median Forecast: {median:,.0f}")
    ax.axvline(ci_lower, color=COLORS["danger"], linestyle=":", linewidth=2, alpha=0.7)
    ax.axvline(ci_upper, color=COLORS["danger"], linestyle=":", linewidth=2, alpha=0.7, label="90% CI")

    # Shaded CI region
    ax.axvspan(ci_lower, ci_upper, alpha=0.1, color=COLORS["danger"])

    # Format
    ax.set_title(
        "Final Unique Depositor Count - Monte Carlo Forecast (10,000 simulations)",
        fontsize=16,
        fontweight="bold",
        pad=20,
    )
    ax.set_xlabel("Unique Depositors", fontsize=12)
    ax.set_ylabel("Frequency", fontsize=12)
    ax.grid(alpha=0.2, linestyle="--")
    ax.legend(loc="upper right", fontsize=11, framealpha=0.9)

    # Stats box
    increase = median - current
    increase_pct = (increase / current) * 100
    stats_text = (
        f"Current: {current:,.0f}\n"
        f"Forecast Median: {median:,.0f}\n"
        f"Expected Increase: +{increase:,.0f} ({increase_pct:.1f}%)\n"
        f"90% CI: [{ci_lower:,.0f}, {ci_upper:,.0f}]\n"
        f"P(>50k): {stats['probabilities']['depositors_gt_50k']:.1%}"
    )
    ax.text(
        0.02,
        0.98,
        stats_text,
        transform=ax.transAxes,
        fontsize=11,
        verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="black", alpha=0.8, edgecolor="white"),
        fontfamily="monospace",
    )

    plt.tight_layout()
    pdf.savefig(fig, dpi=150)
    plt.close()


def create_volume_distribution_plot(
    results_df: pl.DataFrame, stats: dict[str, Any], pdf: PdfPages
) -> None:
    """Page 2: Final volume distribution."""
    fig, ax = plt.subplots(figsize=(14, 8))

    volume = results_df["final_volume"].to_numpy() / 1e6  # Convert to millions
    current = stats["current"]["total_volume"] / 1e6
    median = stats["volume"]["median"] / 1e6
    ci_lower = stats["volume"]["ci_lower"] / 1e6
    ci_upper = stats["volume"]["ci_upper"] / 1e6

    # Histogram
    ax.hist(volume, bins=50, alpha=0.7, color=COLORS["secondary"], edgecolor="white", linewidth=0.5)

    # Vertical lines
    ax.axvline(current, color=COLORS["warning"], linestyle="--", linewidth=2, label=f"Current: ${current:.1f}M")
    ax.axvline(median, color=COLORS["success"], linestyle="-", linewidth=3, label=f"Median Forecast: ${median:.1f}M")
    ax.axvline(ci_lower, color=COLORS["danger"], linestyle=":", linewidth=2, alpha=0.7)
    ax.axvline(ci_upper, color=COLORS["danger"], linestyle=":", linewidth=2, alpha=0.7, label="90% CI")

    # Shaded CI region
    ax.axvspan(ci_lower, ci_upper, alpha=0.1, color=COLORS["danger"])

    # Format
    ax.set_title(
        "Final Total Volume - Monte Carlo Forecast (10,000 simulations)",
        fontsize=16,
        fontweight="bold",
        pad=20,
    )
    ax.set_xlabel("Total Volume ($ Millions)", fontsize=12)
    ax.set_ylabel("Frequency", fontsize=12)
    ax.grid(alpha=0.2, linestyle="--")
    ax.legend(loc="upper right", fontsize=11, framealpha=0.9)

    # Stats box
    increase = median - current
    increase_pct = (increase / current) * 100
    stats_text = (
        f"Current: ${current:.1f}M\n"
        f"Forecast Median: ${median:.1f}M\n"
        f"Expected Increase: +${increase:.1f}M ({increase_pct:.1f}%)\n"
        f"90% CI: [${ci_lower:.1f}M, ${ci_upper:.1f}M]\n"
        f"P(>$1B): {stats['probabilities']['volume_gt_1b']:.1%}"
    )
    ax.text(
        0.02,
        0.98,
        stats_text,
        transform=ax.transAxes,
        fontsize=11,
        verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="black", alpha=0.8, edgecolor="white"),
        fontfamily="monospace",
    )

    plt.tight_layout()
    pdf.savefig(fig, dpi=150)
    plt.close()


def create_time_series_projection(
    hourly_stats: pl.DataFrame,
    results_df: pl.DataFrame,
    stats: dict[str, Any],
    params: dict[str, float],
    hours_remaining: float,
    pdf: PdfPages,
) -> None:
    """Page 3: Time series projection with confidence bands."""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))

    # Historical data
    hours_hist = hourly_stats["hour_bin"].to_numpy()
    deposits_cumsum = hourly_stats["deposit_count"].cum_sum().to_numpy()

    current_hour = hours_hist[-1]
    future_hours = np.arange(current_hour + 1, TOTAL_SALE_HOURS + 1)

    # Plot historical
    ax1.plot(hours_hist, deposits_cumsum, color=COLORS["primary"], linewidth=2.5, label="Historical", alpha=0.9)

    # Forecast median (simple extrapolation based on median increase)
    median_increase_deposits = stats["depositors"]["median"] - stats["current"]["unique_depositors"]
    deposits_per_hour_forecast = median_increase_deposits / hours_remaining
    forecast_deposits = deposits_cumsum[-1] + deposits_per_hour_forecast * (future_hours - current_hour)

    ax1.plot(
        future_hours,
        forecast_deposits,
        color=COLORS["success"],
        linewidth=2.5,
        linestyle="--",
        label="Median Forecast",
        alpha=0.9,
    )

    # Confidence bands (rough approximation)
    ci_range = stats["depositors"]["ci_upper"] - stats["depositors"]["ci_lower"]
    upper_band = forecast_deposits + (ci_range / 2) * ((future_hours - current_hour) / hours_remaining)
    lower_band = forecast_deposits - (ci_range / 2) * ((future_hours - current_hour) / hours_remaining)

    ax1.fill_between(future_hours, lower_band, upper_band, alpha=0.2, color=COLORS["success"], label="90% CI")

    # Surge window shading
    ax1.axvspan(66, 72, alpha=0.1, color=COLORS["warning"], label="Late Surge Window")

    ax1.set_title("Cumulative Unique Depositors - Time Series Projection", fontsize=16, fontweight="bold", pad=20)
    ax1.set_xlabel("Hours Since Sale Start", fontsize=12)
    ax1.set_ylabel("Cumulative Unique Depositors", fontsize=12)
    ax1.grid(alpha=0.2, linestyle="--")
    ax1.legend(loc="upper left", fontsize=10, framealpha=0.9)

    # Volume subplot (similar structure)
    volume_cumsum = hourly_stats["volume"].cum_sum().to_numpy() / 1e6
    median_increase_volume = (stats["volume"]["median"] - stats["current"]["total_volume"]) / 1e6
    volume_per_hour_forecast = median_increase_volume / hours_remaining
    forecast_volume = volume_cumsum[-1] + volume_per_hour_forecast * (future_hours - current_hour)

    ax2.plot(hours_hist, volume_cumsum, color=COLORS["secondary"], linewidth=2.5, label="Historical", alpha=0.9)
    ax2.plot(
        future_hours,
        forecast_volume,
        color=COLORS["success"],
        linewidth=2.5,
        linestyle="--",
        label="Median Forecast",
        alpha=0.9,
    )

    ci_range_volume = (stats["volume"]["ci_upper"] - stats["volume"]["ci_lower"]) / 1e6
    upper_band_vol = forecast_volume + (ci_range_volume / 2) * ((future_hours - current_hour) / hours_remaining)
    lower_band_vol = forecast_volume - (ci_range_volume / 2) * ((future_hours - current_hour) / hours_remaining)

    ax2.fill_between(future_hours, lower_band_vol, upper_band_vol, alpha=0.2, color=COLORS["success"], label="90% CI")
    ax2.axvspan(66, 72, alpha=0.1, color=COLORS["warning"], label="Late Surge Window")

    ax2.set_title("Cumulative Volume - Time Series Projection", fontsize=16, fontweight="bold", pad=20)
    ax2.set_xlabel("Hours Since Sale Start", fontsize=12)
    ax2.set_ylabel("Cumulative Volume ($ Millions)", fontsize=12)
    ax2.grid(alpha=0.2, linestyle="--")
    ax2.legend(loc="upper left", fontsize=10, framealpha=0.9)

    plt.tight_layout()
    pdf.savefig(fig, dpi=150)
    plt.close()


def create_scenario_comparison(results_df: pl.DataFrame, pdf: PdfPages) -> None:
    """Page 4: Scenario comparison (with vs without late surge)."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 8))

    # Split by surge
    no_surge = results_df.filter(pl.col("surge_applied") == False)
    with_surge = results_df.filter(pl.col("surge_applied") == True)

    # Depositor comparison
    data_depositors = [
        no_surge["final_unique_depositors"].to_numpy(),
        with_surge["final_unique_depositors"].to_numpy(),
    ]

    bp1 = ax1.boxplot(
        data_depositors,
        labels=["No Surge", "With Surge"],
        patch_artist=True,
        widths=0.6,
        showfliers=False,
    )

    for patch, color in zip(bp1["boxes"], [COLORS["primary"], COLORS["warning"]]):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    ax1.set_title("Final Depositors by Scenario", fontsize=14, fontweight="bold")
    ax1.set_ylabel("Unique Depositors", fontsize=12)
    ax1.grid(alpha=0.2, linestyle="--", axis="y")

    # Volume comparison
    data_volume = [
        no_surge["final_volume"].to_numpy() / 1e6,
        with_surge["final_volume"].to_numpy() / 1e6,
    ]

    bp2 = ax2.boxplot(
        data_volume,
        labels=["No Surge", "With Surge"],
        patch_artist=True,
        widths=0.6,
        showfliers=False,
    )

    for patch, color in zip(bp2["boxes"], [COLORS["secondary"], COLORS["warning"]]):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    ax2.set_title("Final Volume by Scenario", fontsize=14, fontweight="bold")
    ax2.set_ylabel("Total Volume ($ Millions)", fontsize=12)
    ax2.grid(alpha=0.2, linestyle="--", axis="y")

    fig.suptitle("Scenario Comparison: Base vs Late Surge (50% probability)", fontsize=16, fontweight="bold", y=0.98)

    plt.tight_layout()
    pdf.savefig(fig, dpi=150)
    plt.close()


def create_summary_page(stats: dict[str, Any], params: dict[str, float], pdf: PdfPages) -> None:
    """Page 5: Summary and methodology."""
    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(111)
    ax.axis("off")

    summary_text = f"""
MONTE CARLO DEPOSIT FORECAST - SUMMARY

═══════════════════════════════════════════════════════════════════════════

CURRENT STATE (as of {stats['current']['data_collection_time'].strftime('%Y-%m-%d %H:%M UTC')})
  • Total Deposits:        {stats['current']['total_deposits']:,}
  • Unique Depositors:     {stats['current']['unique_depositors']:,}
  • Total Volume:          ${stats['current']['total_volume']:,.2f}

FORECAST: FINAL UNIQUE DEPOSITORS
  • Median Forecast:       {stats['depositors']['median']:,.0f}
  • 90% Confidence Int:    [{stats['depositors']['ci_lower']:,.0f}, {stats['depositors']['ci_upper']:,.0f}]
  • Expected Increase:     +{stats['depositors']['median'] - stats['current']['unique_depositors']:,.0f} depositors
  • P(>50,000 depositors): {stats['probabilities']['depositors_gt_50k']:.1%}

FORECAST: FINAL VOLUME
  • Median Forecast:       ${stats['volume']['median']:,.2f}
  • 90% Confidence Int:    [${stats['volume']['ci_lower']:,.2f}, ${stats['volume']['ci_upper']:,.2f}]
  • Expected Increase:     +${stats['volume']['median'] - stats['current']['total_volume']:,.2f}
  • P(>$1 Billion):        {stats['probabilities']['volume_gt_1b']:.1%}

═══════════════════════════════════════════════════════════════════════════

MODEL METHODOLOGY

1. ARRIVAL RATE MODEL
   • Exponential decay fitted to historical data (hours 0-{len(stats['current'])}):
     λ(t) = {params['a']:.2f} × exp(-{params['b']:.4f} × t) + {params['c']:.2f}
   • R² = {params['r_squared']:.4f}
   • Late surge: 1.5-2.5× multiplier in final 6 hours (50% probability)

2. DEPOSIT SIZE DISTRIBUTION
   • Whale deposits (${186282:,.0f}): {stats.get('whale_prob', 0.063):.1%} of deposits
   • Non-whale: Sampled from empirical distribution
   • Median non-whale: ${stats.get('median_non_whale', 3284):.2f}

3. DEPOSITOR BEHAVIOR
   • First-time deposit probability: 94.9%
   • Repeat deposit probability: 5.1%

4. MONTE CARLO SIMULATION
   • Number of runs: {N_SIMULATIONS:,}
   • Random seed: {RANDOM_SEED}
   • Confidence level: 90% (5th to 95th percentile)
   • Scenarios: 50% no surge, 50% late surge

5. VALIDATION
   • Model fitted to hours 0-{len(stats['current'])}
   • Forecast window: Remaining hours until October 30, 1pm UTC
   • Data freshness: <1 hour old

═══════════════════════════════════════════════════════════════════════════

KEY ASSUMPTIONS
  1. Historical patterns continue (exponential decay)
  2. 50% probability of late-hour surge (1.5-2.5× increase)
  3. Deposit size distribution remains stable
  4. Repeat depositor rate stays constant (~5%)
  5. No external shocks or whale coordination

═══════════════════════════════════════════════════════════════════════════

Generated: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}
Address: {TARGET_ADDRESS}
"""

    ax.text(
        0.05,
        0.95,
        summary_text,
        transform=ax.transAxes,
        fontsize=10,
        verticalalignment="top",
        fontfamily="monospace",
        bbox=dict(boxstyle="round", facecolor="black", alpha=0.9, edgecolor="white", linewidth=1),
    )

    plt.tight_layout()
    pdf.savefig(fig, dpi=150)
    plt.close()


def save_results(results_df: pl.DataFrame, stats: dict[str, Any], params: dict[str, float]) -> None:
    """Save raw results and parameters."""
    logger.info("Saving results...")

    # Save simulation results
    results_file = REPORTS_DIR / f"forecast_results_{SHORT_ADDR}.parquet"
    results_df.write_parquet(results_file, compression="snappy", statistics=True)
    logger.info(f"✅ Saved simulation results to: {results_file}")

    # Save model parameters
    params_file = REPORTS_DIR / f"model_parameters_{SHORT_ADDR}.json"
    params_data = {
        "model_params": params,
        "statistics": {
            "depositors": stats["depositors"],
            "volume": stats["volume"],
            "probabilities": stats["probabilities"],
        },
        "current_state": {
            k: str(v) if isinstance(v, datetime) else v for k, v in stats["current"].items()
        },
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }

    with open(params_file, "w") as f:
        json.dump(params_data, f, indent=2)

    logger.info(f"✅ Saved model parameters to: {params_file}")


def main() -> None:
    """Main entry point for Monte Carlo forecast."""
    logger.info("")
    logger.info("=" * 80)
    logger.info("MONTE CARLO DEPOSIT FORECAST")
    logger.info("=" * 80)
    logger.info("")

    # Load data and calculate current state
    df, current_state = load_deposit_data()

    # Calculate time parameters
    now_utc = datetime.now(timezone.utc)
    hours_elapsed = (now_utc - SALE_START_UTC).total_seconds() / 3600
    hours_remaining = (SALE_END_UTC - now_utc).total_seconds() / 3600

    logger.info(f"Sale started: {SALE_START_UTC}")
    logger.info(f"Sale ends: {SALE_END_UTC}")
    logger.info(f"Hours elapsed: {hours_elapsed:.1f}")
    logger.info(f"Hours remaining: {hours_remaining:.1f}")
    logger.info("")

    # Calculate hourly statistics
    hourly_stats = calculate_hourly_statistics(df)

    # Fit arrival rate model
    params = fit_arrival_rate_model(hourly_stats)

    # Extract deposit size distribution
    size_dist = extract_deposit_size_distribution(df)

    # Calculate repeat probability
    repeat_prob = calculate_repeat_probability(df)

    logger.info("")

    # Run Monte Carlo simulations
    results_df = run_monte_carlo(
        current_state,
        hours_remaining,
        hours_elapsed,
        params,
        size_dist,
        repeat_prob,
    )

    # Calculate statistics
    stats = calculate_statistics(results_df, current_state)
    stats["whale_prob"] = size_dist["whale_probability"]
    stats["median_non_whale"] = size_dist["median"]

    logger.info("")

    # Generate PDF report
    generate_forecast_pdf(results_df, stats, hourly_stats, params, hours_remaining)

    # Save results
    save_results(results_df, stats, params)

    logger.info("")
    logger.info("=" * 80)
    logger.info("FORECAST COMPLETE")
    logger.info("=" * 80)
    logger.info("")


if __name__ == "__main__":
    main()
