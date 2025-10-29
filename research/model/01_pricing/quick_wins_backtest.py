#!/usr/bin/env python3
"""
Quick Wins Implementation for Binary Option Pricing Model
=========================================================

This script implements all "quick win" improvements:
1. Consistent IV source (calls only)
2. Volatility term structure interpolation
3. Volatility smile adjustment
4. Robust strike price setting

Author: Senior Quant Researcher
Date: October 23, 2025
"""

import polars as pl
import numpy as np
from pathlib import Path
import logging
from datetime import datetime
import time
from scipy.stats import norm
from typing import Tuple, Optional, Dict
import json

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Constants
SECONDS_PER_YEAR = 31_557_600  # 365.25 * 24 * 60 * 60
SECONDS_PER_CONTRACT = 900  # 15 minutes

# File paths
BASE_DIR = Path("/Users/lgierhake/Documents/ETH/BT/research/model")
OPTIONS_FILE = Path("/Users/lgierhake/Documents/ETH/BT/research/tardis/data/consolidated/btc_options_atm_shortdated_with_iv_2023_2025.parquet")
PERPETUAL_FILE = BASE_DIR / "results" / "btc_perpetual_1s_resampled.parquet"
RISK_FREE_FILE = Path("/Users/lgierhake/Documents/ETH/BT/research/risk_free_rate/data/blended_lending_rates_2023_2025.parquet")

# Output paths
OUTPUT_DIR = BASE_DIR / "results" / "quick_wins"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Quick wins configuration
QUICK_WINS_CONFIG = {
    "use_calls_only": True,  # Quick Win #1
    "interpolate_iv": True,  # Quick Win #2
    "apply_smile": True,     # Quick Win #3
    "smooth_strike": True,   # Quick Win #4
    "strike_smoothing_window": 5,  # seconds
    "smile_params": {
        "skew": -0.15,      # Negative for put skew
        "kurtosis": 0.05,   # Excess kurtosis
        "term_skew": -0.02, # Skew decay with time
    }
}


def generate_contracts(start_date: datetime, end_date: datetime) -> pl.DataFrame:
    """Generate 15-minute binary option contracts."""
    logger.info("Generating contracts...")

    start_ts = int(start_date.timestamp())
    end_ts = int(end_date.timestamp())

    contract_starts = []
    current = start_ts

    while current < end_ts:
        contract_starts.append(current)
        current += SECONDS_PER_CONTRACT

    contracts = pl.DataFrame({
        "contract_id": [f"btc_binary_{ts}" for ts in contract_starts],
        "open_time": contract_starts,
    })

    contracts = contracts.with_columns([
        (pl.col("open_time") + SECONDS_PER_CONTRACT).alias("close_time")
    ])

    logger.info(f"Generated {len(contracts):,} contracts")
    return contracts


def create_pricing_grid(contracts: pl.DataFrame) -> pl.DataFrame:
    """Create second-by-second pricing grid."""
    logger.info("Creating pricing grid...")

    seconds_range = pl.DataFrame({
        "seconds_offset": range(SECONDS_PER_CONTRACT)
    })

    grid = contracts.join(seconds_range, how="cross")

    grid = grid.with_columns([
        (pl.col("open_time") + pl.col("seconds_offset")).alias("timestamp"),
        (SECONDS_PER_CONTRACT - pl.col("seconds_offset")).alias("time_remaining"),
    ])

    logger.info(f"Grid size: {len(grid):,} rows")
    return grid


def calculate_robust_strike(
    spot_df: pl.DataFrame,
    contract_open: int,
    window: int = 5
) -> float:
    """
    QUICK WIN #4: Calculate robust strike using exponentially weighted average.

    Reduces noise from single outlier trades at contract open.
    """
    # Get prices around contract open
    prices = spot_df.filter(
        (pl.col("timestamp_seconds") >= contract_open - window) &
        (pl.col("timestamp_seconds") <= contract_open)
    ).sort("timestamp_seconds")

    if len(prices) == 0:
        # Fallback to nearest price
        nearest = spot_df.filter(
            pl.col("timestamp_seconds") <= contract_open
        ).sort("timestamp_seconds").tail(1)

        if len(nearest) == 0:
            # If still no data, try after the contract open
            nearest = spot_df.filter(
                pl.col("timestamp_seconds") >= contract_open
            ).sort("timestamp_seconds").head(1)

        if len(nearest) > 0:
            return nearest["vwap"][0]
        else:
            # No data at all - this shouldn't happen
            logger.warning(f"No spot data found for contract at {contract_open}")
            return None

    # Exponential weights (more recent = higher weight)
    prices_np = prices["vwap"].to_numpy()
    n = len(prices_np)
    weights = np.exp(np.linspace(-1, 0, n))
    weights /= weights.sum()

    return float(np.average(prices_np, weights=weights))


def join_spot_prices_with_robust_strike(
    grid: pl.DataFrame,
    perpetual_df: pl.DataFrame
) -> pl.DataFrame:
    """Join spot prices and calculate robust strikes."""
    logger.info("Joining spot prices with robust strike calculation...")

    # Join current spot prices
    grid = grid.join(
        perpetual_df,
        left_on="timestamp",
        right_on="timestamp_seconds",
        how="left"
    )
    grid = grid.rename({"vwap": "S"})

    if QUICK_WINS_CONFIG["smooth_strike"]:
        logger.info("Calculating robust strikes (Quick Win #4)...")

        # Get unique contracts
        contracts = grid.select(["contract_id", "open_time"]).unique()

        # Calculate robust strike for each contract
        strikes = []
        for row in contracts.iter_rows():
            contract_id, open_time = row
            strike = calculate_robust_strike(
                perpetual_df,
                open_time,
                QUICK_WINS_CONFIG["strike_smoothing_window"]
            )
            if strike is not None:
                strikes.append({"contract_id": contract_id, "K": strike})

        strikes_df = pl.DataFrame(strikes)
        grid = grid.join(strikes_df, on="contract_id", how="left")
    else:
        # Original method: spot at open
        strikes = grid.filter(pl.col("seconds_offset") == 0).select(["contract_id", "S"])
        strikes = strikes.rename({"S": "K"})
        grid = grid.join(strikes, on="contract_id", how="left")

    coverage = grid["S"].is_not_null().sum() / len(grid)
    logger.info(f"Spot price coverage: {coverage*100:.1f}%")

    return grid


def interpolate_iv_to_target_expiry(
    options_df: pl.DataFrame,
    timestamp: int,
    target_seconds: int = SECONDS_PER_CONTRACT
) -> Optional[float]:
    """
    QUICK WIN #2: Interpolate IV to exact 15-minute expiry.

    Uses variance interpolation between bracketing expiries.
    """
    # Get options at this timestamp
    options_at_time = options_df.filter(
        pl.col("timestamp_seconds") == timestamp
    ).sort("time_to_expiry_seconds")

    if len(options_at_time) < 2:
        return None

    # Find bracketing expiries
    expiries = options_at_time["time_to_expiry_seconds"].to_numpy()
    ivs = options_at_time["implied_vol_mid"].to_numpy()

    # Find nearest expiries
    idx_after = np.searchsorted(expiries, target_seconds)

    if idx_after == 0:
        # Target is before first expiry, use first
        return float(ivs[0])
    elif idx_after >= len(expiries):
        # Target is after last expiry, use last
        return float(ivs[-1])
    else:
        # Interpolate between bracketing expiries
        t1, t2 = expiries[idx_after - 1], expiries[idx_after]
        iv1, iv2 = ivs[idx_after - 1], ivs[idx_after]

        # Variance interpolation
        var1 = iv1 ** 2 * t1
        var2 = iv2 ** 2 * t2

        # Linear interpolation in variance space
        target_var = var1 + (var2 - var1) * (target_seconds - t1) / (t2 - t1)

        # Convert back to IV
        return float(np.sqrt(target_var / target_seconds))


def calculate_smile_adjusted_iv(
    base_iv: float,
    moneyness: float,
    time_to_expiry_years: float
) -> float:
    """
    QUICK WIN #3: Adjust IV for volatility smile.

    Accounts for higher IV for OTM options (especially puts).
    """
    if not QUICK_WINS_CONFIG["apply_smile"]:
        return base_iv

    params = QUICK_WINS_CONFIG["smile_params"]

    # Log-moneyness for smile
    log_m = np.log(moneyness) if moneyness > 0 else 0
    sqrt_t = np.sqrt(time_to_expiry_years)

    # Smile adjustment
    smile_mult = 1 + (
        params["skew"] * log_m * sqrt_t +
        params["kurtosis"] * log_m**2 +
        params["term_skew"] * log_m * time_to_expiry_years
    )

    # Ensure reasonable bounds
    smile_mult = np.clip(smile_mult, 0.8, 1.3)

    return base_iv * smile_mult


def join_options_iv_improved(grid: pl.DataFrame) -> pl.DataFrame:
    """
    QUICK WIN #1: Join options IV using CALLS ONLY.
    Plus interpolation and smile adjustment.
    """
    logger.info("Loading and filtering options data (CALLS ONLY - Quick Win #1)...")

    # Get timestamp range
    min_ts = grid["timestamp"].min()
    max_ts = grid["timestamp"].max()

    # Load options - CALLS ONLY!
    options = pl.scan_parquet(OPTIONS_FILE).filter(
        (pl.col("timestamp_seconds") >= min_ts) &
        (pl.col("timestamp_seconds") <= max_ts) &
        (pl.col("type") == "call") &  # QUICK WIN #1: CALLS ONLY
        (pl.col("moneyness") >= 0.95) &  # Slightly wider for smile
        (pl.col("moneyness") <= 1.05) &
        (pl.col("iv_calc_status") == "success") &
        (pl.col("implied_vol_bid").is_not_null()) &
        (pl.col("implied_vol_ask").is_not_null()) &
        (pl.col("implied_vol_bid") > 0.01) &
        (pl.col("implied_vol_bid") < 5.0)
    ).collect()

    logger.info(f"Loaded {len(options):,} CALL options (was mixing before)")

    # Add mid IV
    options = options.with_columns([
        ((pl.col("implied_vol_bid") + pl.col("implied_vol_ask")) / 2).alias("implied_vol_mid")
    ])

    # Get unique (timestamp, close_time) pairs
    unique_pairs = grid.select(["timestamp", "close_time"]).unique()
    logger.info(f"Processing {len(unique_pairs):,} unique timestamp pairs...")

    if QUICK_WINS_CONFIG["interpolate_iv"]:
        logger.info("Applying IV interpolation to 15-min expiry (Quick Win #2)...")

        # Process each unique timestamp
        iv_results = []
        for row in unique_pairs.iter_rows():
            timestamp, close_time = row

            # Get options that expire after contract closes
            valid_options = options.filter(
                (pl.col("timestamp_seconds") == timestamp) &
                (pl.col("expiry_timestamp") > close_time)
            )

            if len(valid_options) > 0:
                # Interpolate to exact 15-minute IV
                time_to_close = close_time - timestamp
                interpolated_iv = interpolate_iv_to_target_expiry(
                    valid_options,
                    timestamp,
                    time_to_close
                )

                if interpolated_iv:
                    iv_results.append({
                        "timestamp": timestamp,
                        "close_time": close_time,
                        "implied_vol_mid": interpolated_iv
                    })

        iv_df = pl.DataFrame(iv_results)
    else:
        # Original method: closest expiry
        pairs_with_options = unique_pairs.join(
            options,
            left_on="timestamp",
            right_on="timestamp_seconds",
            how="left"
        )

        pairs_with_options = pairs_with_options.filter(
            pl.col("expiry_timestamp") > pl.col("close_time")
        )

        iv_df = (
            pairs_with_options
            .sort(["timestamp", "close_time", "time_to_expiry_seconds"])
            .group_by(["timestamp", "close_time"])
            .first()
            .select(["timestamp", "close_time", "implied_vol_mid"])
        )

    # Join IVs to grid
    grid = grid.join(iv_df, on=["timestamp", "close_time"], how="left")

    # Report coverage before forward-fill
    raw_coverage = grid["implied_vol_mid"].is_not_null().sum() / len(grid)
    logger.info(f"IV coverage (before forward-fill): {raw_coverage*100:.1f}%")

    # Forward-fill within contracts
    grid = grid.sort(["contract_id", "timestamp"])
    grid = grid.with_columns([
        pl.col("implied_vol_mid").forward_fill().over("contract_id")
    ])

    final_coverage = grid["implied_vol_mid"].is_not_null().sum() / len(grid)
    logger.info(f"IV coverage (after forward-fill): {final_coverage*100:.1f}%")

    # Add moneyness
    grid = grid.with_columns([
        (pl.col("S") / pl.col("K")).alias("moneyness")
    ])

    # Apply smile adjustment if enabled
    if QUICK_WINS_CONFIG["apply_smile"]:
        logger.info("Applying volatility smile adjustment (Quick Win #3)...")

        grid = grid.with_columns([
            (pl.col("time_remaining") / SECONDS_PER_YEAR).alias("T_years")
        ])

        # Apply smile adjustment
        smile_adjusted = []
        for row in grid.iter_rows(named=True):
            if row["implied_vol_mid"] and row["moneyness"]:
                adjusted_iv = calculate_smile_adjusted_iv(
                    row["implied_vol_mid"],
                    row["moneyness"],
                    row["T_years"]
                )
                smile_adjusted.append(adjusted_iv)
            else:
                smile_adjusted.append(row["implied_vol_mid"])

        grid = grid.with_columns([
            pl.Series("implied_vol_adjusted", smile_adjusted)
        ])
    else:
        grid = grid.with_columns([
            pl.col("implied_vol_mid").alias("implied_vol_adjusted")
        ])

    return grid


def calculate_binary_prices(grid: pl.DataFrame) -> pl.DataFrame:
    """Calculate Black-Scholes binary option prices."""
    logger.info("Calculating binary option prices...")

    # Add risk-free rate (October 2023 average)
    grid = grid.with_columns([
        pl.lit(0.0427).alias("r")  # From risk-free rate analysis
    ])

    # Time in years
    grid = grid.with_columns([
        (pl.col("time_remaining") / SECONDS_PER_YEAR).alias("T_years")
    ])

    # Calculate d2
    grid = grid.with_columns([
        pl.when(
            (pl.col("T_years") > 0) &
            (pl.col("implied_vol_adjusted").is_not_null()) &
            (pl.col("implied_vol_adjusted") > 0)
        ).then(
            ((pl.col("S") / pl.col("K")).log() +
             (pl.col("r") - 0.5 * pl.col("implied_vol_adjusted") ** 2) * pl.col("T_years")) /
            (pl.col("implied_vol_adjusted") * pl.col("T_years").sqrt())
        ).otherwise(None).alias("d2")
    ])

    # Apply normal CDF
    grid = grid.with_columns([
        pl.col("d2").map_batches(
            lambda s: norm.cdf(s.to_numpy()) if s.null_count() < len(s) else [None] * len(s)
        ).alias("prob")
    ])

    # Calculate discounted price
    grid = grid.with_columns([
        (pl.col("prob") * ((-pl.col("r") * pl.col("T_years")).exp())).alias("price")
    ])

    return grid


def add_outcomes(grid: pl.DataFrame) -> pl.DataFrame:
    """Add actual binary outcomes."""
    logger.info("Adding outcomes...")

    # Get closing prices
    closes = grid.filter(pl.col("seconds_offset") == SECONDS_PER_CONTRACT - 1).select(["contract_id", "S"])
    closes = closes.rename({"S": "S_close"})

    grid = grid.join(closes, on="contract_id", how="left")

    # Binary outcome
    grid = grid.with_columns([
        (pl.col("S_close") > pl.col("K")).cast(pl.Int8).alias("outcome")
    ])

    return grid


def calculate_metrics(df: pl.DataFrame) -> Dict:
    """Calculate comprehensive performance metrics."""
    logger.info("Calculating performance metrics...")

    # Filter to valid predictions
    valid = df.filter(
        pl.col("price").is_not_null() &
        pl.col("outcome").is_not_null()
    )

    if len(valid) == 0:
        return {}

    predictions = valid["price"].to_numpy()
    outcomes = valid["outcome"].to_numpy()

    # Brier score
    brier = np.mean((predictions - outcomes) ** 2)

    # Calibration analysis
    n_bins = 10
    bins = np.linspace(0, 1, n_bins + 1)
    calibration_data = []

    for i in range(n_bins):
        mask = (predictions >= bins[i]) & (predictions < bins[i + 1])
        if i == n_bins - 1:
            mask = (predictions >= bins[i]) & (predictions <= bins[i + 1])

        if mask.sum() > 0:
            avg_pred = predictions[mask].mean()
            actual_rate = outcomes[mask].mean()
            count = mask.sum()
            calibration_data.append({
                "bin": f"{bins[i]:.1f}-{bins[i+1]:.1f}",
                "avg_prediction": avg_pred,
                "actual_rate": actual_rate,
                "count": count,
                "error": abs(avg_pred - actual_rate)
            })

    # Expected calibration error
    ece = sum(row["error"] * row["count"] for row in calibration_data) / len(valid)
    mce = max(row["error"] for row in calibration_data) if calibration_data else 0

    # Trading performance (confidence threshold)
    confident = valid.filter(
        (pl.col("price") > 0.6) | (pl.col("price") < 0.4)
    )

    if len(confident) > 0:
        win_rate = confident.filter(
            ((pl.col("price") > 0.6) & (pl.col("outcome") == 1)) |
            ((pl.col("price") < 0.4) & (pl.col("outcome") == 0))
        ).height / len(confident)
    else:
        win_rate = 0

    metrics = {
        "total_predictions": len(valid),
        "coverage": len(valid) / len(df),
        "brier_score": brier,
        "ece": ece,
        "mce": mce,
        "mean_prediction": predictions.mean(),
        "mean_outcome": outcomes.mean(),
        "win_rate_confident": win_rate,
        "confident_trades": len(confident),
        "calibration_data": calibration_data
    }

    return metrics


def main():
    """Run improved backtest with all quick wins."""
    start_time = time.time()

    logger.info("="*80)
    logger.info("QUICK WINS IMPLEMENTATION BACKTEST")
    logger.info("="*80)
    logger.info(f"Configuration: {json.dumps(QUICK_WINS_CONFIG, indent=2)}")

    # Test period: October 2023
    start_date = datetime(2023, 10, 1)
    end_date = datetime(2023, 10, 31, 23, 59, 59)

    # Generate contracts
    contracts = generate_contracts(start_date, end_date)

    # Create pricing grid
    grid = create_pricing_grid(contracts)

    # Load perpetual data
    logger.info("Loading perpetual data...")
    perpetual_df = pl.read_parquet(PERPETUAL_FILE)

    # Join spot prices with robust strike calculation
    grid = join_spot_prices_with_robust_strike(grid, perpetual_df)

    # Join options IV with all improvements
    grid = join_options_iv_improved(grid)

    # Calculate binary prices
    grid = calculate_binary_prices(grid)

    # Add outcomes
    grid = add_outcomes(grid)

    # Calculate metrics
    metrics = calculate_metrics(grid)

    # Save results
    output_file = OUTPUT_DIR / "quick_wins_backtest_results.parquet"
    logger.info(f"Saving results to {output_file}...")
    grid.write_parquet(output_file)

    # Save metrics
    metrics_file = OUTPUT_DIR / "quick_wins_metrics.json"
    with open(metrics_file, 'w') as f:
        # Convert numpy types to native Python types
        metrics_json = {
            k: v if not isinstance(v, np.ndarray) else v.tolist()
            for k, v in metrics.items()
        }
        json.dump(metrics_json, f, indent=2, default=str)

    # Print summary
    logger.info("\n" + "="*80)
    logger.info("RESULTS SUMMARY")
    logger.info("="*80)
    logger.info(f"Total predictions: {metrics['total_predictions']:,}")
    logger.info(f"Coverage: {metrics['coverage']*100:.1f}%")
    logger.info(f"Brier Score: {metrics['brier_score']:.4f}")
    logger.info(f"ECE: {metrics['ece']:.4f}")
    logger.info(f"MCE: {metrics['mce']:.4f}")
    logger.info(f"Win Rate (confident): {metrics['win_rate_confident']*100:.1f}%")

    logger.info("\nCalibration Analysis:")
    for cal in metrics.get("calibration_data", []):
        logger.info(f"  {cal['bin']}: Pred={cal['avg_prediction']:.3f}, "
                   f"Actual={cal['actual_rate']:.3f}, Error={cal['error']:.3f}")

    elapsed = time.time() - start_time
    logger.info(f"\nTotal runtime: {elapsed:.1f} seconds")

    return metrics


if __name__ == "__main__":
    metrics = main()