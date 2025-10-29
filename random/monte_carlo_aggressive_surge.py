#!/usr/bin/env python3
"""
Monte Carlo simulation with AGGRESSIVE late surge (matching opening hour intensity).

Modifications from base model:
- Surge multiplier: 17-20× (vs. 1.5-2.5×) to match opening hour intensity
- Target: 6,000-7,000 deposits/hour in final 6 hours
- Multiprocessing: Parallel simulation for faster execution
- Configurable surge probability: Run with 50% or 75% probability
"""

from __future__ import annotations

import json
import logging
import multiprocessing as mp
from datetime import datetime, timezone
from functools import partial
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

# AGGRESSIVE SURGE PARAMETERS
AGGRESSIVE_SURGE_MIN = 17.0  # Minimum multiplier (vs 1.5 in base model)
AGGRESSIVE_SURGE_MAX = 20.0  # Maximum multiplier (vs 2.5 in base model)

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

    # Initial guess
    p0 = [6000, 0.1, 400]

    try:
        params, _ = curve_fit(exponential_decay, hours, deposits, p0=p0, maxfev=10000)
        a, b, c = params

        predicted = exponential_decay(hours, a, b, c)
        ss_res = np.sum((deposits - predicted) ** 2)
        ss_tot = np.sum((deposits - np.mean(deposits)) ** 2)
        r_squared = 1 - (ss_res / ss_tot)

        logger.info(f"Fitted parameters: a={a:.2f}, b={b:.4f}, c={c:.2f}")
        logger.info(f"R² = {r_squared:.4f}")

        return {"a": float(a), "b": float(b), "c": float(c), "r_squared": r_squared}

    except Exception as e:
        logger.error(f"Failed to fit model: {e}")
        avg_rate = float(deposits[-6:].mean())
        logger.warning(f"Using fallback: constant rate = {avg_rate:.2f}/hr")
        return {"a": 0.0, "b": 0.0, "c": avg_rate, "r_squared": 0.0}


def extract_deposit_size_distribution(df: pl.DataFrame) -> dict[str, Any]:
    """Extract empirical deposit size distribution."""
    logger.info("Extracting deposit size distribution...")

    sizes = df["TokenValue"].to_numpy()
    max_deposit = 186282.0
    whale_deposits = sizes[np.isclose(sizes, max_deposit, rtol=0.001)]
    whale_probability = len(whale_deposits) / len(sizes)
    non_whale_sizes = sizes[~np.isclose(sizes, max_deposit, rtol=0.001)]

    distribution = {
        "whale_probability": whale_probability,
        "whale_amount": max_deposit,
        "non_whale_sizes": non_whale_sizes,
        "mean": float(np.mean(non_whale_sizes)),
        "std": float(np.std(non_whale_sizes)),
        "median": float(np.median(non_whale_sizes)),
    }

    logger.info(f"Whale deposit probability: {whale_probability:.1%}")
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
    hour: float,
    params: dict[str, float],
    apply_surge: bool,
    surge_start_hour: float,
    surge_min: float,
    surge_max: float,
) -> int:
    """Simulate number of deposit arrivals for a given hour."""
    base_rate = exponential_decay(np.array([hour]), params["a"], params["b"], params["c"])[0]

    # Apply AGGRESSIVE late surge if applicable
    if apply_surge and hour >= surge_start_hour:
        surge_multiplier = np.random.uniform(surge_min, surge_max)
        base_rate *= surge_multiplier

    n_deposits = np.random.poisson(max(0, base_rate))
    return int(n_deposits)


def simulate_deposit_sizes(
    n_deposits: int, size_dist: dict[str, Any], rng: np.random.Generator
) -> np.ndarray:
    """Sample deposit sizes from empirical distribution."""
    sizes = np.zeros(n_deposits)

    for i in range(n_deposits):
        if rng.random() < size_dist["whale_probability"]:
            sizes[i] = size_dist["whale_amount"]
        else:
            sizes[i] = rng.choice(size_dist["non_whale_sizes"])

    return sizes


def run_single_simulation_worker(
    sim_id: int,
    current_state: dict[str, Any],
    hours_remaining: float,
    hours_elapsed: float,
    params: dict[str, float],
    size_dist: dict[str, Any],
    repeat_prob: float,
    surge_probability: float,
    surge_min: float,
    surge_max: float,
    random_seed: int,
) -> dict[str, float]:
    """Run a single Monte Carlo simulation (worker function for multiprocessing)."""
    # Create RNG with unique seed for this simulation
    rng = np.random.default_rng(random_seed + sim_id)

    # Initialize with current state
    total_deposits = current_state["total_deposits"]
    total_volume = current_state["total_volume"]
    unique_depositors = current_state["unique_depositors"]

    # Decide if surge applies for this simulation
    apply_surge = rng.random() < surge_probability

    # Surge starts in final 6 hours (hour 66)
    surge_start_hour = 66.0

    # Simulate hour by hour
    for hour_offset in range(int(np.ceil(hours_remaining))):
        current_hour = hours_elapsed + hour_offset

        # Simulate arrivals
        n_new_deposits = simulate_deposit_arrivals(
            current_hour, params, apply_surge, surge_start_hour, surge_min, surge_max
        )

        if n_new_deposits == 0:
            continue

        # Simulate deposit sizes
        new_sizes = simulate_deposit_sizes(n_new_deposits, size_dist, rng)

        # Update totals
        total_deposits += n_new_deposits
        total_volume += np.sum(new_sizes)

        # Simulate unique depositors
        n_new_unique = sum(1 for _ in range(n_new_deposits) if rng.random() > repeat_prob)
        unique_depositors += n_new_unique

    return {
        "final_deposits": total_deposits,
        "final_unique_depositors": unique_depositors,
        "final_volume": total_volume,
        "surge_applied": apply_surge,
    }


def run_monte_carlo_parallel(
    current_state: dict[str, Any],
    hours_remaining: float,
    hours_elapsed: float,
    params: dict[str, float],
    size_dist: dict[str, Any],
    repeat_prob: float,
    surge_probability: float,
    surge_min: float = AGGRESSIVE_SURGE_MIN,
    surge_max: float = AGGRESSIVE_SURGE_MAX,
) -> pl.DataFrame:
    """Run Monte Carlo simulations in parallel using multiprocessing."""
    logger.info(f"Running {N_SIMULATIONS:,} Monte Carlo simulations (PARALLEL)...")
    logger.info(f"Surge probability: {surge_probability:.0%}")
    logger.info(f"Aggressive surge multiplier: {surge_min:.1f}×-{surge_max:.1f}×")

    # Number of CPU cores to use
    n_cores = mp.cpu_count()
    logger.info(f"Using {n_cores} CPU cores")

    # Create worker function with fixed parameters
    worker_func = partial(
        run_single_simulation_worker,
        current_state=current_state,
        hours_remaining=hours_remaining,
        hours_elapsed=hours_elapsed,
        params=params,
        size_dist=size_dist,
        repeat_prob=repeat_prob,
        surge_probability=surge_probability,
        surge_min=surge_min,
        surge_max=surge_max,
        random_seed=RANDOM_SEED,
    )

    # Run simulations in parallel
    with mp.Pool(n_cores) as pool:
        results = list(pool.map(worker_func, range(N_SIMULATIONS)))

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
            "depositors_gt_50k": (results_df["final_unique_depositors"] > 50000).sum() / len(results_df),
            "depositors_gt_60k": (results_df["final_unique_depositors"] > 60000).sum() / len(results_df),
            "volume_gt_1b": (results_df["final_volume"] > 1_000_000_000).sum() / len(results_df),
            "volume_gt_1_5b": (results_df["final_volume"] > 1_500_000_000).sum() / len(results_df),
        },
    }

    logger.info(f"Median final depositors: {stats['depositors']['median']:,.0f}")
    logger.info(f"90% CI: [{stats['depositors']['ci_lower']:,.0f}, {stats['depositors']['ci_upper']:,.0f}]")
    logger.info(f"Median final volume: ${stats['volume']['median']:,.2f}")
    logger.info(f"90% CI: [${stats['volume']['ci_lower']:,.2f}, ${stats['volume']['ci_upper']:,.2f}]")
    logger.info(f"P(>$1B): {stats['probabilities']['volume_gt_1b']:.1%}")
    logger.info(f"P(>$1.5B): {stats['probabilities']['volume_gt_1_5b']:.1%}")

    return stats


def save_summary(
    stats: dict[str, Any],
    params: dict[str, float],
    surge_probability: float,
    output_prefix: str,
) -> None:
    """Save markdown summary report."""
    output_file = REPORTS_DIR / f"{output_prefix}_summary.md"

    surge_pct = int(surge_probability * 100)

    content = f"""# Aggressive Surge Forecast - {surge_pct}% Surge Probability

**Analysis Date:** {stats['current']['data_collection_time'].strftime('%Y-%m-%d %H:%M UTC')}
**Sale End:** October 30, 2025 at 1:00 PM UTC

**Model:** AGGRESSIVE late surge (17-20× multiplier matching opening hour intensity)
**Surge Probability:** {surge_probability:.0%} (vs 50% baseline)

---

## 90% Confidence Intervals

### Final Unique Depositors
- **Current:** {stats['current']['unique_depositors']:,} depositors
- **Forecast:** **{stats['depositors']['ci_lower']:,.0f} to {stats['depositors']['ci_upper']:,.0f} depositors**
- **Median:** {stats['depositors']['median']:,.0f} depositors
- **Expected Increase:** +{stats['depositors']['median'] - stats['current']['unique_depositors']:,.0f} depositors (+{((stats['depositors']['median'] - stats['current']['unique_depositors']) / stats['current']['unique_depositors'] * 100):.1f}%)

### Final Total Volume
- **Current:** ${stats['current']['total_volume'] / 1e6:.1f} million
- **Forecast:** **${stats['volume']['ci_lower'] / 1e6:.1f}M to ${stats['volume']['ci_upper'] / 1e6:.1f}M**
- **Median:** ${stats['volume']['median'] / 1e6:.1f} million
- **Expected Increase:** +${(stats['volume']['median'] - stats['current']['total_volume']) / 1e6:.1f}M (+{((stats['volume']['median'] - stats['current']['total_volume']) / stats['current']['total_volume'] * 100):.1f}%)

### Key Probabilities
- **P(>50k depositors):** {stats['probabilities']['depositors_gt_50k']:.1%}
- **P(>60k depositors):** {stats['probabilities']['depositors_gt_60k']:.1%}
- **P(>$1 Billion):** {stats['probabilities']['volume_gt_1b']:.1%}
- **P(>$1.5 Billion):** {stats['probabilities']['volume_gt_1_5b']:.1%}

---

## Model Parameters

### Aggressive Surge
- **Multiplier:** 17-20× (vs 1.5-2.5× baseline)
- **Target Rate:** 6,000-7,000 deposits/hour (matching opening hour)
- **Surge Window:** Final 6 hours (hours 66-72)
- **Surge Probability:** {surge_probability:.0%}

### Base Model
- Decay curve: λ(t) = {params['a']:.2f} × e^(-{params['b']:.4f}t) + {params['c']:.2f}
- R² = {params['r_squared']:.4f}
- Simulations: {N_SIMULATIONS:,}

---

**Generated:** {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}
"""

    with open(output_file, "w") as f:
        f.write(content)

    logger.info(f"✅ Summary saved to: {output_file}")


def main(surge_probability: float, output_prefix: str) -> None:
    """Main entry point for aggressive surge Monte Carlo forecast."""
    logger.info("")
    logger.info("=" * 80)
    logger.info(f"AGGRESSIVE SURGE MONTE CARLO FORECAST ({surge_probability:.0%} probability)")
    logger.info("=" * 80)
    logger.info("")

    # Load data
    df, current_state = load_deposit_data()

    # Calculate time parameters
    now_utc = datetime.now(timezone.utc)
    hours_elapsed = (now_utc - SALE_START_UTC).total_seconds() / 3600
    hours_remaining = (SALE_END_UTC - now_utc).total_seconds() / 3600

    logger.info(f"Hours elapsed: {hours_elapsed:.1f}")
    logger.info(f"Hours remaining: {hours_remaining:.1f}")
    logger.info("")

    # Prepare model
    hourly_stats = calculate_hourly_statistics(df)
    params = fit_arrival_rate_model(hourly_stats)
    size_dist = extract_deposit_size_distribution(df)
    repeat_prob = calculate_repeat_probability(df)

    logger.info("")

    # Run Monte Carlo with specified surge probability
    results_df = run_monte_carlo_parallel(
        current_state,
        hours_remaining,
        hours_elapsed,
        params,
        size_dist,
        repeat_prob,
        surge_probability,
    )

    # Calculate statistics
    stats = calculate_statistics(results_df, current_state)

    logger.info("")

    # Save results
    save_summary(stats, params, surge_probability, output_prefix)

    # Save raw data
    results_file = REPORTS_DIR / f"{output_prefix}_results.parquet"
    results_df.write_parquet(results_file, compression="snappy", statistics=True)
    logger.info(f"✅ Saved results to: {results_file}")

    params_file = REPORTS_DIR / f"{output_prefix}_parameters.json"
    params_data = {
        "model_params": params,
        "surge_probability": surge_probability,
        "surge_min": AGGRESSIVE_SURGE_MIN,
        "surge_max": AGGRESSIVE_SURGE_MAX,
        "statistics": {k: v for k, v in stats.items() if k != "current"},
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    with open(params_file, "w") as f:
        json.dump(params_data, f, indent=2)
    logger.info(f"✅ Saved parameters to: {params_file}")

    logger.info("")
    logger.info("=" * 80)
    logger.info("FORECAST COMPLETE")
    logger.info("=" * 80)
    logger.info("")


if __name__ == "__main__":
    import sys

    # Run Forecast A: 50% surge probability
    logger.info("\n" + "="*80)
    logger.info("FORECAST A: 50% AGGRESSIVE SURGE PROBABILITY")
    logger.info("="*80 + "\n")
    main(surge_probability=0.50, output_prefix="aggressive_forecast_50pct")

    # Run Forecast B: 75% surge probability
    logger.info("\n" + "="*80)
    logger.info("FORECAST B: 75% AGGRESSIVE SURGE PROBABILITY")
    logger.info("="*80 + "\n")
    main(surge_probability=0.75, output_prefix="aggressive_forecast_75pct")
