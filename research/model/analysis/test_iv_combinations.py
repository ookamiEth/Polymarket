#!/usr/bin/env python3
"""
Test Different IV Source Combinations for Binary Option Pricing

This script tests all combinations of:
- Option Type: Call vs Put
- Quote Side: Bid vs Ask vs Mid
- Filters: ATM only (current moneyness)

Goal: Find which IV source produces the best calibration and lowest Brier score.
"""

import polars as pl
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple
import logging
from scipy.stats import norm
from datetime import datetime
import json

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# File paths
RESULTS_FILE = Path("/Users/lgierhake/Documents/ETH/BT/research/model/results/pilot_backtest_results.parquet")
OPTIONS_FILE = Path("/Users/lgierhake/Documents/ETH/BT/research/tardis/data/consolidated/btc_options_atm_shortdated_with_iv_2023_2025.parquet")
PERPETUAL_FILE = Path("/Users/lgierhake/Documents/ETH/BT/research/model/results/btc_perpetual_1s_resampled.parquet")

# Constants
SECONDS_PER_CONTRACT = 900  # 15 minutes

def create_test_grid(num_contracts: int = 100) -> pl.DataFrame:
    """Create a small test grid for IV comparison."""
    logger.info(f"Creating test grid with {num_contracts} contracts...")

    # Extract unique contracts from existing results
    results = pl.read_parquet(RESULTS_FILE)
    contracts = results.select(["contract_id", "open_time", "close_time"]).unique().head(num_contracts)

    # Create time grid
    seconds_range = pl.DataFrame({
        "seconds_offset": range(SECONDS_PER_CONTRACT)
    })

    # Cross join
    grid = contracts.join(seconds_range, how="cross")

    # Add computed columns
    grid = grid.with_columns([
        (pl.col("open_time") + pl.col("seconds_offset")).alias("timestamp"),
        (SECONDS_PER_CONTRACT - pl.col("seconds_offset")).alias("time_remaining"),
    ])

    return grid

def join_spot_and_rates(grid: pl.DataFrame) -> pl.DataFrame:
    """Add spot prices and risk-free rates."""
    logger.info("Joining spot prices...")

    # Load perpetual data
    perpetual = pl.read_parquet(PERPETUAL_FILE).select(["timestamp_seconds", "vwap"])
    grid = grid.join(perpetual, left_on="timestamp", right_on="timestamp_seconds", how="left")
    grid = grid.rename({"vwap": "S"})

    # Use fixed rate for October 2023 (from previous analysis)
    # October 2023 average risk-free rate was approximately 4.27%
    grid = grid.with_columns([
        pl.lit(0.0427).alias("r")
    ])

    return grid

def get_iv_combinations(
    grid: pl.DataFrame,
    options_file: Path
) -> Dict[str, pl.DataFrame]:
    """Get all IV combinations: call/put × bid/ask/mid."""
    logger.info("Loading options data...")

    # Load options for October 2023
    options = pl.scan_parquet(options_file).filter(
        (pl.col("timestamp_seconds") >= 1696118400) &  # Oct 1, 2023
        (pl.col("timestamp_seconds") < 1698796800) &   # Nov 1, 2023
        (pl.col("moneyness") >= 0.98) &  # ATM only
        (pl.col("moneyness") <= 1.02) &
        (pl.col("iv_calc_status") == "success") &
        (pl.col("implied_vol_bid").is_not_null()) &
        (pl.col("implied_vol_ask").is_not_null())
    ).collect()

    logger.info(f"Loaded {len(options):,} ATM option quotes")

    # Split by type
    calls = options.filter(pl.col("type") == "call")
    puts = options.filter(pl.col("type") == "put")

    logger.info(f"Calls: {len(calls):,}, Puts: {len(puts):,}")

    results = {}

    # Test each combination
    combinations = [
        ("call_bid", calls, "implied_vol_bid"),
        ("call_ask", calls, "implied_vol_ask"),
        ("call_mid", calls, None),  # Will calculate
        ("put_bid", puts, "implied_vol_bid"),
        ("put_ask", puts, "implied_vol_ask"),
        ("put_mid", puts, None),  # Will calculate
    ]

    for name, option_df, vol_col in combinations:
        logger.info(f"Processing {name}...")

        # Add mid if needed
        if vol_col is None:
            option_df = option_df.with_columns([
                ((pl.col("implied_vol_bid") + pl.col("implied_vol_ask")) / 2).alias("implied_vol_mid")
            ])
            vol_col = "implied_vol_mid"

        # Get unique timestamp-close_time pairs from grid
        unique_pairs = grid.select(["timestamp", "close_time"]).unique()

        # Join options to pairs
        pairs_with_options = unique_pairs.join(
            option_df.select(["timestamp_seconds", "expiry_timestamp", vol_col, "time_to_expiry_seconds"]),
            left_on="timestamp",
            right_on="timestamp_seconds",
            how="left"
        )

        # Filter valid expiries
        pairs_with_options = pairs_with_options.filter(
            pl.col("expiry_timestamp") > pl.col("close_time")
        )

        # Get closest expiry for each pair
        closest_options = (
            pairs_with_options
            .sort(["timestamp", "close_time", "time_to_expiry_seconds"])
            .group_by(["timestamp", "close_time"])
            .first()
        )

        # Join back to grid
        grid_with_iv = grid.join(
            closest_options.select(["timestamp", "close_time", vol_col]),
            on=["timestamp", "close_time"],
            how="left"
        ).rename({vol_col: "sigma"})

        results[name] = grid_with_iv

        # Report coverage
        coverage = grid_with_iv["sigma"].is_not_null().sum() / len(grid_with_iv)
        logger.info(f"  Coverage: {coverage*100:.1f}%")

    return results

def calculate_binary_price(S, K, r, sigma, T_seconds):
    """Calculate binary option price using Black-Scholes."""
    T_years = T_seconds / 31_557_600

    # Handle edge cases
    if T_years <= 0 or sigma <= 0:
        return 1.0 if S > K else 0.0

    # Calculate d2
    d2 = (np.log(S / K) + (r - 0.5 * sigma ** 2) * T_years) / (sigma * np.sqrt(T_years))

    # Probability
    prob = norm.cdf(d2)

    # Discount
    discount = np.exp(-r * T_years)

    return discount * prob

def add_prices_and_outcomes(df: pl.DataFrame) -> pl.DataFrame:
    """Add binary option prices and actual outcomes."""

    # Add strike prices (spot at open)
    strikes = df.filter(pl.col("seconds_offset") == 0).select(["contract_id", "S"])
    strikes = strikes.rename({"S": "K"})
    df = df.join(strikes, on="contract_id", how="left")

    # Add closing prices (spot at close)
    closes = df.filter(pl.col("seconds_offset") == SECONDS_PER_CONTRACT - 1).select(["contract_id", "S"])
    closes = closes.rename({"S": "S_close"})
    df = df.join(closes, on="contract_id", how="left")

    # Add outcome
    df = df.with_columns([
        (pl.col("S_close") > pl.col("K")).cast(pl.Int8).alias("outcome")
    ])

    # Calculate prices for valid rows
    valid_mask = df["sigma"].is_not_null() & (df["S"] > 0) & (df["K"] > 0)
    valid_df = df.filter(valid_mask)

    if len(valid_df) > 0:
        prices = []
        for row in valid_df.iter_rows(named=True):
            price = calculate_binary_price(
                row["S"], row["K"], row["r"], row["sigma"], row["time_remaining"]
            )
            prices.append(price)

        valid_df = valid_df.with_columns([
            pl.Series("price", prices)
        ])

        # Join back
        df = df.join(
            valid_df.select(["contract_id", "seconds_offset", "price"]),
            on=["contract_id", "seconds_offset"],
            how="left"
        )

    return df

def evaluate_performance(df: pl.DataFrame, name: str) -> Dict:
    """Calculate performance metrics for an IV source."""

    # Filter to valid predictions
    valid = df.filter(pl.col("price").is_not_null())

    if len(valid) == 0:
        return {"name": name, "error": "No valid predictions"}

    # Calculate Brier score
    errors = (valid["price"] - valid["outcome"]).to_numpy()
    brier = np.mean(errors ** 2)

    # Calculate calibration (ECE)
    buckets = [(i/10, (i+1)/10) for i in range(10)]
    ece = 0
    mce = 0

    for low, high in buckets:
        bucket = valid.filter((pl.col("price") >= low) & (pl.col("price") < high))
        if len(bucket) > 0:
            avg_pred = bucket["price"].mean()
            actual_rate = bucket["outcome"].mean()
            error = abs(avg_pred - actual_rate)
            weight = len(bucket) / len(valid)
            ece += weight * error
            mce = max(mce, error)

    # Trading performance (10% threshold)
    trades = valid.filter((pl.col("price") > 0.6) | (pl.col("price") < 0.4))
    if len(trades) > 0:
        wins = trades.filter(
            ((pl.col("price") > 0.6) & (pl.col("outcome") == 1)) |
            ((pl.col("price") < 0.4) & (pl.col("outcome") == 0))
        ).height
        win_rate = wins / len(trades)
    else:
        win_rate = 0

    # Moneyness breakdown
    df_with_moneyness = valid.with_columns([
        (pl.col("S") / pl.col("K")).alias("moneyness")
    ])

    atm = df_with_moneyness.filter(
        (pl.col("moneyness") >= 0.995) & (pl.col("moneyness") <= 1.005)
    )
    if len(atm) > 0:
        atm_brier = np.mean((atm["price"] - atm["outcome"]).to_numpy() ** 2)
    else:
        atm_brier = np.nan

    return {
        "name": name,
        "total_predictions": len(valid),
        "brier_score": brier,
        "ece": ece,
        "mce": mce,
        "win_rate": win_rate,
        "trade_count": len(trades) if len(trades) > 0 else 0,
        "atm_brier": atm_brier,
    }

def main():
    """Run IV combination testing."""
    logger.info("=" * 80)
    logger.info("IV SOURCE COMBINATION TESTING")
    logger.info("=" * 80)

    # Create test grid
    grid = create_test_grid(num_contracts=100)  # Use 100 contracts for testing
    logger.info(f"Grid size: {len(grid):,} rows")

    # Add spot and rates
    grid = join_spot_and_rates(grid)

    # Get all IV combinations
    iv_combinations = get_iv_combinations(grid, OPTIONS_FILE)

    # Also test mixed (current approach)
    logger.info("Creating mixed (random) combination...")
    options_all = pl.scan_parquet(OPTIONS_FILE).filter(
        (pl.col("timestamp_seconds") >= 1696118400) &
        (pl.col("timestamp_seconds") < 1698796800) &
        (pl.col("moneyness") >= 0.98) &
        (pl.col("moneyness") <= 1.02) &
        (pl.col("iv_calc_status") == "success")
    ).collect()

    # Don't filter by type - get random mix
    unique_pairs = grid.select(["timestamp", "close_time"]).unique()
    pairs_with_options = unique_pairs.join(
        options_all.select(["timestamp_seconds", "expiry_timestamp", "implied_vol_bid", "implied_vol_ask", "time_to_expiry_seconds"]),
        left_on="timestamp",
        right_on="timestamp_seconds",
        how="left"
    )
    pairs_with_options = pairs_with_options.filter(pl.col("expiry_timestamp") > pl.col("close_time"))
    closest_options = (
        pairs_with_options
        .sort(["timestamp", "close_time", "time_to_expiry_seconds"])
        .group_by(["timestamp", "close_time"])
        .first()
    )
    grid_mixed = grid.join(
        closest_options.select(["timestamp", "close_time", "implied_vol_bid", "implied_vol_ask"]),
        on=["timestamp", "close_time"],
        how="left"
    )
    grid_mixed = grid_mixed.with_columns([
        ((pl.col("implied_vol_bid") + pl.col("implied_vol_ask")) / 2).alias("sigma")
    ])
    iv_combinations["mixed_current"] = grid_mixed

    # Evaluate each combination
    results = []
    for name, df in iv_combinations.items():
        logger.info(f"\nEvaluating {name}...")

        # Add prices and outcomes
        df_with_prices = add_prices_and_outcomes(df)

        # Evaluate
        metrics = evaluate_performance(df_with_prices, name)
        results.append(metrics)

    # Create results table
    logger.info("\n" + "=" * 80)
    logger.info("RESULTS SUMMARY")
    logger.info("=" * 80)

    # Sort by Brier score
    results = sorted(results, key=lambda x: x.get("brier_score", float("inf")))

    print("\n📊 Performance Comparison:")
    print("-" * 100)
    print(f"{'IV Source':<20} {'Brier↓':<10} {'ECE':<10} {'MCE':<10} {'Win Rate':<10} {'ATM Brier':<10} {'Coverage':<10}")
    print("-" * 100)

    for r in results:
        if "error" in r:
            print(f"{r['name']:<20} {r['error']}")
        else:
            coverage = f"{r['total_predictions']/90000*100:.1f}%"
            print(f"{r['name']:<20} {r['brier_score']:<10.4f} {r['ece']:<10.4f} {r['mce']:<10.4f} "
                  f"{r['win_rate']*100:<10.1f}% {r['atm_brier']:<10.4f} {coverage:<10}")

    print("-" * 100)

    # Find winner
    best = results[0]
    print(f"\n🏆 WINNER: {best['name']}")
    print(f"   Brier Score: {best['brier_score']:.4f}")
    print(f"   Calibration Error: {best['ece']:.4f}")
    print(f"   Win Rate: {best['win_rate']*100:.1f}%")

    # Compare to mixed
    mixed = next((r for r in results if r["name"] == "mixed_current"), None)
    if mixed and mixed != best:
        improvement = (mixed["brier_score"] - best["brier_score"]) / mixed["brier_score"] * 100
        print(f"\n📈 Improvement over mixed: {improvement:.1f}%")

    # Save detailed results
    output_file = Path("/Users/lgierhake/Documents/ETH/BT/research/model/results/iv_combination_test_results.json")
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2, default=str)
    logger.info(f"\nDetailed results saved to {output_file}")

    # Recommendations
    print("\n" + "=" * 80)
    print("📝 RECOMMENDATIONS")
    print("=" * 80)

    if "call" in best["name"]:
        print("✅ Use CALL options for IV")
    elif "put" in best["name"]:
        print("✅ Use PUT options for IV")
    else:
        print("⚠️ Current mixed approach might be optimal")

    if "bid" in best["name"]:
        print("✅ Use BID implied volatility")
    elif "ask" in best["name"]:
        print("✅ Use ASK implied volatility")
    else:
        print("✅ Use MID implied volatility")

    print("\n🔬 Additional Testing Needed:")
    print("1. Run on full dataset (not just 100 contracts)")
    print("2. Test put-call average: (call_IV + put_IV) / 2")
    print("3. Test dynamic selection based on moneyness")
    print("4. Test time-weighted average of multiple expiries")

if __name__ == "__main__":
    main()