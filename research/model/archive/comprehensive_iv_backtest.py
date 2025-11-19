#!/usr/bin/env python3
"""
Comprehensive IV Testing Backtest

This enhanced version of the pilot backtest collects BOTH call and put IVs
separately for every second, allowing us to test all combinations:
- Call Bid, Call Ask, Call Mid
- Put Bid, Put Ask, Put Mid
- Synthetic (Call + Put average)

Key improvements:
1. Separate call/put IV tracking
2. Updates IV every second (not just once)
3. Calculates 7 different prices for comparison
4. Maintains full traceability of IV sources
"""

from pathlib import Path
import polars as pl
import numpy as np
import logging
from datetime import datetime
import time
from black_scholes_binary import SECONDS_PER_YEAR
from scipy.stats import norm

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")
logger = logging.getLogger(__name__)

# File paths
OUTPUT_FILE = Path("/Users/lgierhake/Documents/ETH/BT/research/model/results/comprehensive_iv_backtest_results.parquet")
PERPETUAL_FILE = Path("/Users/lgierhake/Documents/ETH/BT/research/model/results/btc_perpetual_1s_resampled.parquet")
OPTIONS_FILE = Path("/Users/lgierhake/Documents/ETH/BT/research/tardis/data/consolidated/btc_options_atm_shortdated_with_iv_2023_2025.parquet")

# Constants
SECONDS_PER_CONTRACT = 900  # 15 minutes
START_DATE = datetime(2023, 10, 1)
END_DATE = datetime(2023, 10, 31, 23, 59, 59)
RISK_FREE_RATE = 0.0427  # October 2023 average


def generate_contracts() -> pl.DataFrame:
    """Generate 15-minute binary option contracts for October 2023."""
    logger.info("Generating contracts...")

    # Generate contract start times (every 15 minutes)
    start_ts = int(START_DATE.timestamp())
    end_ts = int(END_DATE.timestamp())

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
    """Create a grid with one row per second per contract."""
    logger.info("Creating pricing grid...")

    # Create seconds range
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

    logger.info(f"Grid size: {len(grid):,} rows")
    return grid


def join_spot_prices(grid: pl.DataFrame) -> pl.DataFrame:
    """Join spot prices from perpetual data."""
    logger.info("Joining spot prices...")

    perpetual = pl.read_parquet(PERPETUAL_FILE).select(["timestamp_seconds", "vwap"])
    grid = grid.join(perpetual, left_on="timestamp", right_on="timestamp_seconds", how="left")
    grid = grid.rename({"vwap": "S"})

    null_count = grid["S"].null_count()
    logger.info(f"Spot price coverage: {(len(grid) - null_count) / len(grid) * 100:.1f}%")

    return grid


def join_options_iv_comprehensive(grid: pl.DataFrame) -> pl.DataFrame:
    """
    Join options IV data for BOTH calls and puts separately.
    This ensures we have call and put IVs for every second.
    """
    logger.info("Loading and filtering options data...")

    # Get timestamp range
    min_ts = grid["timestamp"].min()
    max_ts = grid["timestamp"].max()

    # Load and filter options
    options = pl.scan_parquet(OPTIONS_FILE).filter(
        (pl.col("timestamp_seconds") >= min_ts)
        & (pl.col("timestamp_seconds") <= max_ts)
        & (pl.col("moneyness") >= 0.98)  # ATM only
        & (pl.col("moneyness") <= 1.02)
        & (pl.col("iv_calc_status") == "success")
        & (pl.col("implied_vol_bid").is_not_null())
        & (pl.col("implied_vol_ask").is_not_null())
        & (pl.col("implied_vol_bid") > 0.01)
        & (pl.col("implied_vol_bid") < 5.0)
    ).collect()

    logger.info(f"Loaded {len(options):,} ATM option quotes")

    # Split into calls and puts
    calls = options.filter(pl.col("type") == "call").select([
        "timestamp_seconds",
        "expiry_timestamp",
        "time_to_expiry_seconds",
        "implied_vol_bid",
        "implied_vol_ask",
    ])

    puts = options.filter(pl.col("type") == "put").select([
        "timestamp_seconds",
        "expiry_timestamp",
        "time_to_expiry_seconds",
        "implied_vol_bid",
        "implied_vol_ask",
    ])

    logger.info(f"Calls: {len(calls):,}, Puts: {len(puts):,}")

    # Get unique timestamp-close_time pairs
    unique_pairs = grid.select(["timestamp", "close_time"]).unique()
    logger.info(f"Unique (timestamp, close_time) pairs: {len(unique_pairs):,}")

    # Process CALLS
    logger.info("Processing call options...")
    pairs_with_calls = unique_pairs.join(
        calls,
        left_on="timestamp",
        right_on="timestamp_seconds",
        how="left"
    )

    # Filter valid expiries
    pairs_with_calls = pairs_with_calls.filter(
        pl.col("expiry_timestamp") > pl.col("close_time")
    )

    # Get closest expiry for each pair
    closest_calls = (
        pairs_with_calls
        .sort(["timestamp", "close_time", "time_to_expiry_seconds"])
        .group_by(["timestamp", "close_time"])
        .first()
        .select([
            "timestamp",
            "close_time",
            pl.col("implied_vol_bid").alias("call_vol_bid"),
            pl.col("implied_vol_ask").alias("call_vol_ask"),
        ])
    )

    # Process PUTS
    logger.info("Processing put options...")
    pairs_with_puts = unique_pairs.join(
        puts,
        left_on="timestamp",
        right_on="timestamp_seconds",
        how="left"
    )

    # Filter valid expiries
    pairs_with_puts = pairs_with_puts.filter(
        pl.col("expiry_timestamp") > pl.col("close_time")
    )

    # Get closest expiry for each pair
    closest_puts = (
        pairs_with_puts
        .sort(["timestamp", "close_time", "time_to_expiry_seconds"])
        .group_by(["timestamp", "close_time"])
        .first()
        .select([
            "timestamp",
            "close_time",
            pl.col("implied_vol_bid").alias("put_vol_bid"),
            pl.col("implied_vol_ask").alias("put_vol_ask"),
        ])
    )

    # Join both calls and puts to grid
    logger.info("Joining calls and puts to grid...")
    grid = grid.join(closest_calls, on=["timestamp", "close_time"], how="left")
    grid = grid.join(closest_puts, on=["timestamp", "close_time"], how="left")

    # Report coverage before forward-fill
    call_coverage = grid["call_vol_bid"].is_not_null().sum() / len(grid)
    put_coverage = grid["put_vol_bid"].is_not_null().sum() / len(grid)
    logger.info(f"Coverage before forward-fill - Calls: {call_coverage*100:.1f}%, Puts: {put_coverage*100:.1f}%")

    # Forward-fill within each contract
    logger.info("Applying forward-fill...")
    grid = grid.sort(["contract_id", "timestamp"])

    grid = grid.with_columns([
        pl.col("call_vol_bid").forward_fill().over("contract_id"),
        pl.col("call_vol_ask").forward_fill().over("contract_id"),
        pl.col("put_vol_bid").forward_fill().over("contract_id"),
        pl.col("put_vol_ask").forward_fill().over("contract_id"),
    ])

    # Calculate mid and synthetic IVs
    grid = grid.with_columns([
        ((pl.col("call_vol_bid") + pl.col("call_vol_ask")) / 2).alias("call_vol_mid"),
        ((pl.col("put_vol_bid") + pl.col("put_vol_ask")) / 2).alias("put_vol_mid"),
    ])

    grid = grid.with_columns([
        ((pl.col("call_vol_mid") + pl.col("put_vol_mid")) / 2).alias("synthetic_vol_mid")
    ])

    # Report final coverage
    call_coverage_final = grid["call_vol_bid"].is_not_null().sum() / len(grid)
    put_coverage_final = grid["put_vol_bid"].is_not_null().sum() / len(grid)
    logger.info(f"Coverage after forward-fill - Calls: {call_coverage_final*100:.1f}%, Puts: {put_coverage_final*100:.1f}%")

    return grid


def calculate_comprehensive_prices(grid: pl.DataFrame) -> pl.DataFrame:
    """Calculate binary option prices using all 7 IV sources."""
    logger.info("Calculating comprehensive binary option prices...")

    # Add strike prices (spot at open)
    strikes = grid.filter(pl.col("seconds_offset") == 0).select(["contract_id", "S"])
    strikes = strikes.rename({"S": "K"})
    grid = grid.join(strikes, on="contract_id", how="left")

    # Add risk-free rate
    grid = grid.with_columns([
        pl.lit(RISK_FREE_RATE).alias("r")
    ])

    # Convert time to years
    grid = grid.with_columns([
        (pl.col("time_remaining") / SECONDS_PER_YEAR).alias("T_years")
    ])

    # Define IV sources to test
    iv_sources = [
        ("call_bid", "call_vol_bid"),
        ("call_ask", "call_vol_ask"),
        ("call_mid", "call_vol_mid"),
        ("put_bid", "put_vol_bid"),
        ("put_ask", "put_vol_ask"),
        ("put_mid", "put_vol_mid"),
        ("synthetic", "synthetic_vol_mid"),
    ]

    # Calculate prices for each IV source
    for name, iv_col in iv_sources:
        logger.info(f"  Calculating prices using {name}...")

        # Calculate d2
        grid = grid.with_columns([
            pl.when((pl.col("T_years") > 0) & (pl.col(iv_col).is_not_null()) & (pl.col(iv_col) > 0))
            .then(
                ((pl.col("S") / pl.col("K")).log() +
                 (pl.col("r") - 0.5 * pl.col(iv_col) ** 2) * pl.col("T_years")) /
                (pl.col(iv_col) * pl.col("T_years").sqrt())
            )
            .otherwise(None)
            .alias(f"d2_{name}")
        ])

        # Apply normal CDF
        grid = grid.with_columns([
            pl.col(f"d2_{name}").map_batches(
                lambda s: norm.cdf(s.to_numpy()) if s.null_count() < len(s) else [None] * len(s)
            ).alias(f"prob_{name}")
        ])

        # Calculate discounted price
        grid = grid.with_columns([
            (pl.col(f"prob_{name}") * ((-pl.col("r") * pl.col("T_years")).exp())).alias(f"price_{name}")
        ])

    logger.info("Comprehensive pricing complete")
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


def calculate_performance_metrics(grid: pl.DataFrame) -> pl.DataFrame:
    """Calculate performance metrics for each IV source."""
    logger.info("Calculating performance metrics...")

    iv_sources = ["call_bid", "call_ask", "call_mid", "put_bid", "put_ask", "put_mid", "synthetic"]
    metrics = []

    for source in iv_sources:
        price_col = f"price_{source}"

        # Filter to valid predictions
        valid = grid.filter(pl.col(price_col).is_not_null())

        if len(valid) > 0:
            # Calculate Brier score
            errors = (valid[price_col] - valid["outcome"]).to_numpy()
            brier = np.mean(errors ** 2)

            # Calculate calibration
            buckets = [(i/10, (i+1)/10) for i in range(10)]
            ece = 0
            mce = 0

            for low, high in buckets:
                bucket = valid.filter((pl.col(price_col) >= low) & (pl.col(price_col) < high))
                if len(bucket) > 0:
                    avg_pred = bucket[price_col].mean()
                    actual_rate = bucket["outcome"].mean()
                    error = abs(avg_pred - actual_rate)
                    weight = len(bucket) / len(valid)
                    ece += weight * error
                    mce = max(mce, error)

            # Trading performance
            trades = valid.filter((pl.col(price_col) > 0.6) | (pl.col(price_col) < 0.4))
            if len(trades) > 0:
                wins = trades.filter(
                    ((pl.col(price_col) > 0.6) & (pl.col("outcome") == 1)) |
                    ((pl.col(price_col) < 0.4) & (pl.col("outcome") == 0))
                ).height
                win_rate = wins / len(trades)
            else:
                win_rate = 0

            metrics.append({
                "iv_source": source,
                "predictions": len(valid),
                "coverage": len(valid) / len(grid),
                "brier_score": brier,
                "ece": ece,
                "mce": mce,
                "win_rate": win_rate,
                "trade_count": len(trades) if len(trades) > 0 else 0,
            })

    return pl.DataFrame(metrics)


def main():
    """Run comprehensive IV backtest."""
    start_time = time.time()

    logger.info("="*80)
    logger.info("COMPREHENSIVE IV TESTING BACKTEST")
    logger.info("="*80)

    # Generate contracts
    contracts = generate_contracts()

    # Create pricing grid
    grid = create_pricing_grid(contracts)

    # Join data
    grid = join_spot_prices(grid)
    grid = join_options_iv_comprehensive(grid)

    # Calculate all prices
    grid = calculate_comprehensive_prices(grid)

    # Add outcomes
    grid = add_outcomes(grid)

    # Calculate metrics
    metrics = calculate_performance_metrics(grid)

    # Display results
    logger.info("\n" + "="*80)
    logger.info("PERFORMANCE COMPARISON")
    logger.info("="*80)

    # Sort by Brier score
    metrics = metrics.sort("brier_score")

    print("\n📊 IV Source Performance Ranking:")
    print("-"*80)
    print(f"{'Rank':<6} {'IV Source':<15} {'Brier↓':<10} {'ECE':<10} {'Win Rate':<10} {'Coverage':<10}")
    print("-"*80)

    for i, row in enumerate(metrics.iter_rows(named=True), 1):
        print(f"{i:<6} {row['iv_source']:<15} {row['brier_score']:<10.4f} {row['ece']:<10.4f} "
              f"{row['win_rate']*100:<10.1f}% {row['coverage']*100:<10.1f}%")

    print("-"*80)

    # Find winner
    best = metrics.row(0, named=True)
    print(f"\n🏆 WINNER: {best['iv_source']}")
    print(f"   Brier Score: {best['brier_score']:.4f}")
    print(f"   Calibration Error: {best['ece']:.4f}")
    print(f"   Win Rate: {best['win_rate']*100:.1f}%")

    # Save results
    logger.info(f"\nSaving results to {OUTPUT_FILE}...")

    # Select relevant columns for output
    output_cols = [
        "contract_id", "timestamp", "seconds_offset", "time_remaining",
        "S", "K", "S_close", "outcome",
        "call_vol_bid", "call_vol_ask", "call_vol_mid",
        "put_vol_bid", "put_vol_ask", "put_vol_mid",
        "synthetic_vol_mid",
        "price_call_bid", "price_call_ask", "price_call_mid",
        "price_put_bid", "price_put_ask", "price_put_mid",
        "price_synthetic"
    ]

    # Filter to columns that exist
    existing_cols = [col for col in output_cols if col in grid.columns]

    grid.select(existing_cols).write_parquet(OUTPUT_FILE)

    # Summary
    elapsed_time = time.time() - start_time
    file_size_mb = OUTPUT_FILE.stat().st_size / 1024 / 1024

    logger.info("="*80)
    logger.info("BACKTEST COMPLETE")
    logger.info("="*80)
    logger.info(f"Elapsed time: {elapsed_time/60:.1f} minutes")
    logger.info(f"Output file: {file_size_mb:.1f} MB")
    logger.info(f"Total rows: {len(grid):,}")
    logger.info(f"Contracts processed: {len(contracts):,}")

    # Save metrics
    metrics_file = OUTPUT_FILE.parent / "comprehensive_iv_metrics.parquet"
    metrics.write_parquet(metrics_file)
    logger.info(f"Metrics saved to {metrics_file}")

    print("\n📝 Next Steps:")
    print("1. Review the winning IV source")
    print("2. Update production backtest to use optimal source")
    print("3. Re-run full backtest with consistent IV")


if __name__ == "__main__":
    main()