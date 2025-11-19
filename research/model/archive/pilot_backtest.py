#!/usr/bin/env python3
"""
Pilot Backtest: First Month (October 2023)

Tests the full research pipeline on a small sample:
- ~2,900 contracts (Oct 1-31, 2023)
- ~2.6M pricing calculations (2,900 contracts × 900 seconds)
- Validates all data joins work correctly
- Generates calibration data for analysis

Pipeline:
1. Load October 2023 contracts
2. Generate pricing grid (every second for each contract)
3. Join spot prices (perpetual data)
4. Join risk-free rates (daily lending rates)
5. Join options IV (closest expiry after contract close)
6. Calculate BS binary prices (bid/ask/mid)
7. Record outcomes (did BTC close above strike?)
8. Write results to parquet

Expected runtime: 5-15 minutes
Expected output: ~2.6M rows, ~200-300 MB
"""

import logging

# Import our pricing modules
import sys
from pathlib import Path

import polars as pl

sys.path.append(str(Path(__file__).parent))
from black_scholes_binary import add_binary_pricing_bid_ask_mid

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# File paths
CONTRACTS_FILE = Path("/Users/lgierhake/Documents/ETH/BT/research/model/results/contract_schedule.parquet")
PERPETUAL_FILE = Path("/Users/lgierhake/Documents/ETH/BT/research/model/results/btc_perpetual_1s_resampled.parquet")
OPTIONS_FILE = Path(
    "/Users/lgierhake/Documents/ETH/BT/research/tardis/data/consolidated/btc_options_atm_shortdated_with_iv_2023_2025.parquet"
)
RATES_FILE = Path(
    "/Users/lgierhake/Documents/ETH/BT/research/risk_free_rate/data/blended_lending_rates_2023_2025.parquet"
)
OUTPUT_FILE = Path("/Users/lgierhake/Documents/ETH/BT/research/model/results/pilot_backtest_results.parquet")

# Constants
SECONDS_PER_CONTRACT = 900  # 15 minutes


def load_pilot_contracts(start_date: str = "2023-10-01", end_date: str = "2023-10-31") -> pl.DataFrame:
    """Load contracts for pilot period."""
    logger.info(f"Loading contracts from {start_date} to {end_date}")

    contracts = pl.read_parquet(CONTRACTS_FILE)

    # Filter to pilot period - convert strings to date objects for comparison
    from datetime import date

    start = date.fromisoformat(start_date)
    end = date.fromisoformat(end_date)

    contracts_pilot = contracts.filter((pl.col("date") >= start) & (pl.col("date") <= end))

    logger.info(f"Loaded {len(contracts_pilot):,} contracts")
    logger.info(f"First contract: {contracts_pilot['contract_id'][0]}")
    logger.info(f"Last contract: {contracts_pilot['contract_id'][-1]}")

    return contracts_pilot


def generate_pricing_grid(contracts: pl.DataFrame) -> pl.DataFrame:
    """
    Generate pricing grid: one row per second per contract.

    For each contract, creates 900 rows (one per second from open to close).
    """
    logger.info("Generating pricing grid...")
    logger.info(f"Input: {len(contracts):,} contracts")
    logger.info(f"Output: {len(contracts) * SECONDS_PER_CONTRACT:,} rows expected")

    # For each contract, create 900 rows (one per second)
    # Use cross join with a range DataFrame
    seconds_range = pl.DataFrame({"seconds_offset": list(range(SECONDS_PER_CONTRACT))})

    # Cross join each contract with all 900 seconds
    grid = contracts.join(seconds_range, how="cross")

    # Calculate actual timestamp for each second
    grid = grid.with_columns([(pl.col("open_time") + pl.col("seconds_offset")).alias("timestamp")])

    # Calculate time remaining
    grid = grid.with_columns([(pl.col("close_time") - pl.col("timestamp")).alias("time_remaining")])

    logger.info(f"Generated pricing grid: {len(grid):,} rows")

    return grid


def join_spot_prices(grid: pl.DataFrame, perpetual_file: Path) -> pl.DataFrame:
    """Join spot prices from perpetual data."""
    logger.info("Joining spot prices...")

    # Load perpetual data
    perpetual = pl.read_parquet(perpetual_file).select(["timestamp_seconds", "vwap"])

    # Join on timestamp
    grid = grid.join(perpetual, left_on="timestamp", right_on="timestamp_seconds", how="left")

    # Rename vwap to spot_price (S)
    grid = grid.rename({"vwap": "S"})

    # Check for nulls
    null_count = grid["S"].null_count()
    logger.info(f"Spot price nulls: {null_count:,} / {len(grid):,} ({null_count / len(grid) * 100:.2f}%)")

    return grid


def join_risk_free_rates(grid: pl.DataFrame, rates_file: Path) -> pl.DataFrame:
    """Join risk-free rates from daily lending rates."""
    logger.info("Joining risk-free rates...")

    # Load rates
    rates = pl.read_parquet(rates_file).select(["date", "blended_supply_apr"])

    # Join on date
    grid = grid.join(rates, on="date", how="left")

    # Convert APR to decimal
    grid = grid.with_columns([(pl.col("blended_supply_apr") / 100).alias("r")])

    # Check for nulls
    null_count = grid["r"].null_count()
    logger.info(f"Risk-free rate nulls: {null_count:,} / {len(grid):,} ({null_count / len(grid) * 100:.2f}%)")

    return grid


def join_options_iv(grid: pl.DataFrame, options_file: Path) -> pl.DataFrame:
    """
    Join options IV using closest-expiry strategy.

    Strategy:
    1. Load options for relevant time window
    2. Filter to valid options (expiry > contract close, valid IV)
    3. For each timestamp, find option with minimum time_to_expiry
    4. Join to grid

    This is the most complex join due to the expiry constraint.
    """
    logger.info("Joining options IV...")

    # Get time window
    min_ts_val = grid.select(pl.col("timestamp").min()).item()
    max_ts_val = grid.select(pl.col("timestamp").max()).item()

    if min_ts_val is None or max_ts_val is None:
        raise ValueError("Grid has no timestamps!")

    min_ts = int(min_ts_val)
    max_ts = int(max_ts_val)

    logger.info(f"Time window: {min_ts} to {max_ts}")

    # Load options for this time window
    logger.info("Loading options data...")
    options = pl.scan_parquet(options_file)

    # Filter to relevant time window and valid options
    options_filtered = options.filter(
        (pl.col("timestamp_seconds") >= int(min_ts))
        & (pl.col("timestamp_seconds") <= int(max_ts))
        & (pl.col("iv_calc_status") == "success")
        & (pl.col("implied_vol_bid").is_not_null())
        & (pl.col("implied_vol_ask").is_not_null())
        & (pl.col("implied_vol_bid") > 0.01)
        & (pl.col("implied_vol_bid") < 5.0)
    ).select(
        [
            "timestamp_seconds",
            "expiry_timestamp",
            "time_to_expiry_seconds",
            "implied_vol_bid",
            "implied_vol_ask",
        ]
    )

    logger.info("Collecting options...")
    options_df = options_filtered.collect()
    logger.info(f"Loaded {len(options_df):,} option quotes")

    # Strategy: For each unique (timestamp, close_time) pair, find closest expiry option
    # This is more efficient than row-by-row

    # Get unique timestamp-close_time pairs
    unique_pairs = grid.select(["timestamp", "close_time"]).unique()
    logger.info(f"Unique (timestamp, close_time) pairs: {len(unique_pairs):,}")

    # For each pair, find closest expiry option that expires after close_time
    logger.info("Finding closest expiry options...")

    # Join options to unique pairs
    pairs_with_options = unique_pairs.join(options_df, left_on="timestamp", right_on="timestamp_seconds", how="left")

    # Filter to valid expiries (option expires after contract close)
    pairs_with_options = pairs_with_options.filter(pl.col("expiry_timestamp") > pl.col("close_time"))

    # For each (timestamp, close_time), find minimum time_to_expiry
    closest_options = (
        pairs_with_options.sort(["timestamp", "close_time", "time_to_expiry_seconds"])
        .group_by(["timestamp", "close_time"])
        .first()
    )

    logger.info(f"Found closest options for {len(closest_options):,} pairs")
    logger.info(f"Coverage: {len(closest_options) / len(unique_pairs) * 100:.1f}%")

    # Join back to grid
    grid = grid.join(
        closest_options.select(
            [
                "timestamp",
                "close_time",
                "implied_vol_bid",
                "implied_vol_ask",
                "expiry_timestamp",
                "time_to_expiry_seconds",
            ]
        ),
        on=["timestamp", "close_time"],
        how="left",
    )

    # Check coverage before forward-fill
    null_count_before = grid["implied_vol_bid"].null_count()
    logger.info(
        f"Options IV nulls (before forward-fill): {null_count_before:,} / {len(grid):,} ({null_count_before / len(grid) * 100:.2f}%)"
    )

    # Forward-fill IV within each contract to improve coverage
    # This maintains no-lookahead since we only fill within contracts,
    # and initial join already enforced expiry > close_time constraint
    logger.info("Applying forward-fill to improve coverage...")

    grid = grid.sort(["contract_id", "timestamp"])

    grid = grid.with_columns(
        [
            pl.col("implied_vol_bid").forward_fill().over("contract_id"),
            pl.col("implied_vol_ask").forward_fill().over("contract_id"),
            pl.col("expiry_timestamp").forward_fill().over("contract_id"),
            pl.col("time_to_expiry_seconds").forward_fill().over("contract_id"),
        ]
    )

    # Check coverage after forward-fill
    null_count_after = grid["implied_vol_bid"].null_count()
    logger.info(
        f"Options IV nulls (after forward-fill): {null_count_after:,} / {len(grid):,} ({null_count_after / len(grid) * 100:.2f}%)"
    )
    logger.info(
        f"Forward-fill improvement: {null_count_before - null_count_after:,} additional rows ({(null_count_before - null_count_after) / len(grid) * 100:.1f}%)"
    )

    return grid


def add_strike_prices(grid: pl.DataFrame) -> pl.DataFrame:
    """
    Add strike price (K) for each contract.

    Strike = spot price at contract open (first second of contract).
    """
    logger.info("Adding strike prices...")

    # For each contract, get spot price at open_time (first second)
    strikes = grid.filter(pl.col("seconds_offset") == 0).select(["contract_id", "S"])

    strikes = strikes.rename({"S": "K"})

    # Join back to grid
    grid = grid.join(strikes, on="contract_id", how="left")

    logger.info("✅ Strike prices added")

    return grid


def add_outcomes(grid: pl.DataFrame) -> pl.DataFrame:
    """
    Add binary outcome: 1 if S(close) > K, else 0.
    """
    logger.info("Adding outcomes...")

    # For each contract, get spot price at close_time (last second)
    close_prices = grid.filter(pl.col("seconds_offset") == SECONDS_PER_CONTRACT - 1).select(["contract_id", "S"])

    close_prices = close_prices.rename({"S": "S_close"})

    # Join back to grid
    grid = grid.join(close_prices, on="contract_id", how="left")

    # Binary outcome
    grid = grid.with_columns([(pl.col("S_close") > pl.col("K")).cast(pl.Int8).alias("outcome")])

    # Summary
    total_contracts = grid["contract_id"].n_unique()
    wins = grid.filter(pl.col("seconds_offset") == 0).filter(pl.col("outcome") == 1).height
    logger.info(f"Outcomes: {wins:,} wins / {total_contracts:,} contracts ({wins / total_contracts * 100:.1f}%)")

    return grid


def calculate_prices(grid: pl.DataFrame) -> pl.DataFrame:
    """
    Calculate BS binary option prices using bid/ask/mid IV.
    """
    logger.info("Calculating binary option prices...")

    # Filter to rows with all required data
    valid_rows = grid.filter(
        pl.col("S").is_not_null()
        & pl.col("K").is_not_null()
        & pl.col("r").is_not_null()
        & pl.col("implied_vol_bid").is_not_null()
        & pl.col("implied_vol_ask").is_not_null()
        & (pl.col("time_remaining") > 0)
    )

    logger.info(
        f"Valid rows for pricing: {len(valid_rows):,} / {len(grid):,} ({len(valid_rows) / len(grid) * 100:.1f}%)"
    )

    if len(valid_rows) == 0:
        logger.error("No valid rows for pricing!")
        return grid

    # Calculate prices
    priced = add_binary_pricing_bid_ask_mid(
        valid_rows,
        spot_col="S",
        strike_col="K",
        rate_col="r",
        sigma_bid_col="implied_vol_bid",
        sigma_ask_col="implied_vol_ask",
        time_seconds_col="time_remaining",
    )

    # Merge back to grid (left join to preserve all rows)
    # Select key columns from priced
    price_cols = priced.select(
        [
            "contract_id",
            "timestamp",
            "sigma_mid",
            "T_years",
            "d2_bid",
            "d2_ask",
            "d2_mid",
            "prob_bid",
            "prob_ask",
            "prob_mid",
            "discount",
            "price_bid",
            "price_ask",
            "price_mid",
        ]
    )

    grid = grid.join(price_cols, on=["contract_id", "timestamp"], how="left")

    logger.info("✅ Prices calculated")

    return grid


def run_pilot_backtest() -> None:
    """Main pilot backtest execution."""
    logger.info("=" * 80)
    logger.info("PILOT BACKTEST: October 2023")
    logger.info("=" * 80)

    # 1. Load contracts
    contracts = load_pilot_contracts()

    # 2. Generate pricing grid
    grid = generate_pricing_grid(contracts)

    # 3. Join spot prices
    grid = join_spot_prices(grid, PERPETUAL_FILE)

    # 4. Add strike prices (spot at contract open)
    grid = add_strike_prices(grid)

    # 5. Join risk-free rates
    grid = join_risk_free_rates(grid, RATES_FILE)

    # 6. Join options IV
    grid = join_options_iv(grid, OPTIONS_FILE)

    # 7. Add outcomes
    grid = add_outcomes(grid)

    # 8. Calculate prices
    grid = calculate_prices(grid)

    # 9. Write results
    logger.info(f"\nWriting results to {OUTPUT_FILE}")
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

    grid.write_parquet(OUTPUT_FILE, compression="snappy", statistics=True)

    file_size_mb = OUTPUT_FILE.stat().st_size / 1024 / 1024
    logger.info(f"Output file: {file_size_mb:.1f} MB")

    # Summary statistics
    logger.info("\n" + "=" * 80)
    logger.info("PILOT BACKTEST COMPLETE")
    logger.info("=" * 80)
    logger.info(f"Total rows: {len(grid):,}")
    logger.info(f"Contracts: {grid['contract_id'].n_unique():,}")

    # Pricing coverage
    priced_rows = grid.filter(pl.col("price_mid").is_not_null())
    logger.info(f"Priced rows: {len(priced_rows):,} ({len(priced_rows) / len(grid) * 100:.1f}%)")

    # Sample results
    logger.info("\nSample results (first contract, selected seconds):")
    sample = (
        grid.filter(pl.col("contract_id") == grid["contract_id"][0])
        .filter(pl.col("seconds_offset").is_in([0, 300, 600, 899]))
        .select(
            [
                "contract_id",
                "seconds_offset",
                "S",
                "K",
                "time_remaining",
                "sigma_mid",
                "price_mid",
                "outcome",
            ]
        )
    )
    print(sample)

    logger.info(f"\n✅ Results saved to: {OUTPUT_FILE}")


def main() -> None:
    """Main entry point."""
    try:
        run_pilot_backtest()
    except Exception as e:
        logger.error(f"Pilot backtest failed: {e}")
        raise


if __name__ == "__main__":
    main()
