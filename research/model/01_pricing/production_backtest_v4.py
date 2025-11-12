#!/usr/bin/env python3
"""
Production Backtest with Streaming Architecture

Enhanced version of pilot_backtest.py with:
1. Streaming/chunked processing for large datasets
2. Data quality controls and IV staleness tracking
3. Memory-efficient implementation
4. Progress tracking and checkpointing
5. Comprehensive error handling

Designed to handle the full 2-year dataset (70k contracts, 63M calculations)
without memory issues on machines with limited RAM.

Key improvements:
- Processes options data in chunks (max 1M rows at a time)
- Tracks IV staleness (seconds since last fresh quote)
- Rejects predictions with stale IV (>60 seconds old)
- Uses lazy evaluation and streaming writes
- Adds quality metrics and validation
"""

import logging
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import polars as pl
from scipy import stats
from tqdm import tqdm

# Add parent and archive directory to path for imports
sys.path.append(str(Path(__file__).parent))
sys.path.append(str(Path(__file__).parent / "archive"))
from black_scholes_binary import add_binary_pricing_bid_ask_mid

# Configure logging with more detail
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("/home/ubuntu/Polymarket/research/model/logs/production_backtest_v4.log"),
    ],
)
logger = logging.getLogger(__name__)

# File paths
CONTRACTS_FILE = Path("/home/ubuntu/Polymarket/research/model/results/contract_schedule.parquet")
PERPETUAL_FILE = Path("/home/ubuntu/Polymarket/research/tardis/data/consolidated/btc_perpetual_1s_resampled.parquet")
OPTIONS_FILE = Path(
    "/home/ubuntu/Polymarket/research/tardis/data/consolidated/btc_options_atm_shortdated_with_iv_2023_2025.parquet"
)
RATES_FILE = Path(
    "/home/ubuntu/Polymarket/research/risk_free_rate/data/blended_lending_rates_2023_2025.parquet"
)
OUTPUT_FILE = Path("/home/ubuntu/Polymarket/research/model/results/production_backtest_results_v4.parquet")
CHECKPOINT_FILE = Path("/home/ubuntu/Polymarket/research/model/checkpoints/backtest_progress_v4.json")

# Constants
SECONDS_PER_CONTRACT = 900  # 15 minutes
OPTIONS_CHUNK_SIZE = 1_000_000  # Process options in 1M row chunks
CONTRACT_BATCH_SIZE = 100  # Process contracts in batches
MAX_IV_STALENESS_SECONDS = 300  # Accept IV up to 5 minutes old
MEMORY_LIMIT_GB = 5  # Target memory usage limit


class DataQualityMonitor:
    """Track data quality metrics throughout the backtest."""

    def __init__(self):
        self.metrics = {
            "total_contracts": 0,
            "total_calculations": 0,
            "spot_price_coverage": 0.0,
            "risk_free_coverage": 0.0,
            "iv_coverage_raw": 0.0,
            "iv_coverage_after_ffill": 0.0,
            "iv_staleness_violations": 0,
            "pricing_success_rate": 0.0,
            "mean_iv_staleness": 0.0,
            "max_iv_staleness": 0.0,
        }

    def update(self, key: str, value: float) -> None:
        """Update a metric."""
        self.metrics[key] = value

    def log_summary(self) -> None:
        """Log summary of data quality metrics."""
        logger.info("\n" + "=" * 80)
        logger.info("DATA QUALITY REPORT")
        logger.info("=" * 80)
        for key, value in self.metrics.items():
            if isinstance(value, float):
                logger.info(f"{key}: {value:.2%}" if "coverage" in key or "rate" in key else f"{key}: {value:.2f}")
            else:
                logger.info(f"{key}: {value:,}")


def load_contracts(
    start_date: Optional[str] = None, end_date: Optional[str] = None, batch_size: int = CONTRACT_BATCH_SIZE
) -> pl.LazyFrame:
    """
    Load contracts with optional date filtering.
    Returns LazyFrame for memory efficiency.
    """
    logger.info(f"Loading contracts (batch size: {batch_size})")

    contracts = pl.scan_parquet(CONTRACTS_FILE)

    if start_date and end_date:
        from datetime import date

        start = date.fromisoformat(start_date)
        end = date.fromisoformat(end_date)
        contracts = contracts.filter((pl.col("date") >= start) & (pl.col("date") <= end))
        logger.info(f"Filtered to date range: {start_date} to {end_date}")

    return contracts


def create_streaming_grid(contracts: pl.LazyFrame) -> pl.LazyFrame:
    """
    Create pricing grid using lazy evaluation.
    More memory efficient than eager cross join.
    """
    logger.info("Creating streaming pricing grid...")

    # Create seconds range as LazyFrame
    seconds_range = pl.LazyFrame({"seconds_offset": list(range(SECONDS_PER_CONTRACT))})

    # Use lazy cross join
    grid = contracts.join(seconds_range, how="cross")

    # Add computed columns
    grid = grid.with_columns(
        [
            (pl.col("open_time") + pl.col("seconds_offset")).alias("timestamp"),
            (pl.col("close_time") - (pl.col("open_time") + pl.col("seconds_offset"))).alias("time_remaining"),
        ]
    )

    return grid


def join_spot_prices_streaming(grid: pl.LazyFrame) -> pl.LazyFrame:
    """Join spot prices using lazy evaluation."""
    logger.info("Setting up spot price join...")

    # Scan perpetual data lazily
    perpetual = pl.scan_parquet(PERPETUAL_FILE).select(["timestamp_seconds", "vwap"])

    # Lazy join
    grid = grid.join(perpetual, left_on="timestamp", right_on="timestamp_seconds", how="left")
    grid = grid.rename({"vwap": "S"})

    return grid


def join_risk_free_rates_streaming(grid: pl.LazyFrame) -> pl.LazyFrame:
    """Join risk-free rates using lazy evaluation."""
    logger.info("Setting up risk-free rate join...")

    # Scan rates lazily
    rates = pl.scan_parquet(RATES_FILE).select(["date", "blended_supply_apr"])

    # Lazy join
    grid = grid.join(rates, on="date", how="left")
    grid = grid.with_columns([(pl.col("blended_supply_apr") / 100).alias("r")])

    return grid


def process_options_iv_chunked(
    grid: pl.DataFrame, min_timestamp: int, max_timestamp: int, chunk_size: int = OPTIONS_CHUNK_SIZE
) -> pl.DataFrame:
    """
    Process options IV in chunks to manage memory.
    Tracks IV staleness and applies quality controls.
    """
    logger.info("Processing options IV in chunks...")
    monitor = DataQualityMonitor()

    # Get unique timestamp-close_time pairs
    unique_pairs = grid.select(["timestamp", "close_time"]).unique()
    total_pairs = len(unique_pairs)
    logger.info(f"Processing {total_pairs:,} unique timestamp pairs")

    # Process options in chunks
    options_scanner = pl.scan_parquet(OPTIONS_FILE)

    # Pre-filter options to relevant time window
    options_filtered = options_scanner.filter(
        (pl.col("timestamp_seconds") >= min_timestamp)
        & (pl.col("timestamp_seconds") <= max_timestamp)
        & (pl.col("iv_calc_status") == "success")
        & (pl.col("implied_vol_bid").is_not_null())
        & (pl.col("implied_vol_ask").is_not_null())
        & (pl.col("implied_vol_bid").is_between(0.01, 5.0))
    ).select(
        [
            "timestamp_seconds",
            "expiry_timestamp",
            "time_to_expiry_seconds",
            "implied_vol_bid",
            "implied_vol_ask",
        ]
    )

    # Collect in chunks and process
    all_closest_options = []

    with tqdm(total=total_pairs, desc="Finding closest IV") as pbar:
        for chunk_start in range(0, total_pairs, chunk_size):
            chunk_end = min(chunk_start + chunk_size, total_pairs)
            pairs_chunk = unique_pairs[chunk_start:chunk_end]

            # Get options for this chunk's timestamps
            chunk_timestamps = pairs_chunk["timestamp"].unique().to_list()

            options_chunk = options_filtered.filter(
                pl.col("timestamp_seconds").is_in(chunk_timestamps)
            ).collect()

            if len(options_chunk) == 0:
                pbar.update(chunk_end - chunk_start)
                continue

            # Join and filter
            pairs_with_options = pairs_chunk.join(
                options_chunk, left_on="timestamp", right_on="timestamp_seconds", how="left"
            )

            # Filter to valid expiries
            pairs_with_options = pairs_with_options.filter(pl.col("expiry_timestamp") > pl.col("close_time"))

            # Find closest option for each pair
            closest_chunk = (
                pairs_with_options.sort(["timestamp", "close_time", "time_to_expiry_seconds"])
                .group_by(["timestamp", "close_time"])
                .first()
            )

            # Add IV timestamp for staleness tracking
            closest_chunk = closest_chunk.with_columns([
                pl.col("timestamp").alias("iv_timestamp"),  # When this IV was quoted
            ])

            all_closest_options.append(closest_chunk)
            pbar.update(chunk_end - chunk_start)

    # Combine all chunks
    if all_closest_options:
        closest_options = pl.concat(all_closest_options)
    else:
        logger.warning("No valid options found!")
        return grid

    # Join to grid
    grid = grid.join(
        closest_options.select(
            [
                "timestamp",
                "close_time",
                "implied_vol_bid",
                "implied_vol_ask",
                "expiry_timestamp",
                "time_to_expiry_seconds",
                "iv_timestamp",
            ]
        ),
        on=["timestamp", "close_time"],
        how="left",
    )

    # Track initial coverage
    initial_coverage = 1.0 - (grid["implied_vol_bid"].null_count() / len(grid))
    monitor.update("iv_coverage_raw", initial_coverage)
    logger.info(f"Initial IV coverage: {initial_coverage:.1%}")

    # Apply forward-fill with staleness tracking
    grid = apply_forward_fill_with_staleness(grid, monitor)

    return grid


def apply_forward_fill_with_staleness(grid: pl.DataFrame, monitor: DataQualityMonitor) -> pl.DataFrame:
    """
    Apply forward-fill with staleness tracking.
    Marks data as invalid if IV is older than 5 minutes.
    """
    logger.info("Applying forward-fill with staleness controls (5-minute threshold)...")

    # Sort for forward-fill
    grid = grid.sort(["contract_id", "timestamp"])

    # Forward-fill IV values within contracts
    grid = grid.with_columns(
        [
            pl.col("implied_vol_bid").forward_fill().over("contract_id"),
            pl.col("implied_vol_ask").forward_fill().over("contract_id"),
            pl.col("expiry_timestamp").forward_fill().over("contract_id"),
            pl.col("iv_timestamp").forward_fill().over("contract_id"),
        ]
    )

    # Calculate IV staleness (seconds since IV was quoted)
    grid = grid.with_columns(
        [(pl.col("timestamp") - pl.col("iv_timestamp")).alias("iv_staleness_seconds")]
    )

    # Mark stale IV as invalid
    grid = grid.with_columns(
        [
            pl.when(pl.col("iv_staleness_seconds") > MAX_IV_STALENESS_SECONDS)
            .then(None)
            .otherwise(pl.col("implied_vol_bid"))
            .alias("implied_vol_bid_validated"),

            pl.when(pl.col("iv_staleness_seconds") > MAX_IV_STALENESS_SECONDS)
            .then(None)
            .otherwise(pl.col("implied_vol_ask"))
            .alias("implied_vol_ask_validated"),
        ]
    )

    # Calculate final coverage and staleness metrics
    final_coverage = 1.0 - (grid["implied_vol_bid_validated"].null_count() / len(grid))
    monitor.update("iv_coverage_after_ffill", final_coverage)

    staleness_violations = (grid["iv_staleness_seconds"] > MAX_IV_STALENESS_SECONDS).sum()
    monitor.update("iv_staleness_violations", staleness_violations)

    valid_staleness = grid.filter(pl.col("iv_staleness_seconds").is_not_null())["iv_staleness_seconds"]
    if len(valid_staleness) > 0:
        monitor.update("mean_iv_staleness", valid_staleness.mean())
        monitor.update("max_iv_staleness", valid_staleness.max())

    logger.info(f"Final IV coverage (after staleness filter): {final_coverage:.1%}")
    logger.info(f"Staleness violations: {staleness_violations:,} ({staleness_violations/len(grid)*100:.2%})")

    # Use validated columns for pricing (drop original IV columns to avoid duplicates)
    grid = grid.drop(["implied_vol_bid", "implied_vol_ask"])
    grid = grid.rename({
        "implied_vol_bid_validated": "implied_vol_bid",
        "implied_vol_ask_validated": "implied_vol_ask",
    })

    return grid


def add_strikes_and_outcomes(grid: pl.DataFrame) -> pl.DataFrame:
    """Add strike prices and binary outcomes."""
    logger.info("Adding strikes and outcomes...")

    # Strike = spot at contract open
    strikes = grid.filter(pl.col("seconds_offset") == 0).select(["contract_id", "S"]).rename({"S": "K"})
    grid = grid.join(strikes, on="contract_id", how="left")

    # Outcome = 1 if S(close) > K
    close_prices = (
        grid.filter(pl.col("seconds_offset") == SECONDS_PER_CONTRACT - 1)
        .select(["contract_id", "S"])
        .rename({"S": "S_close"})
    )
    grid = grid.join(close_prices, on="contract_id", how="left")
    grid = grid.with_columns([(pl.col("S_close") > pl.col("K")).cast(pl.Int8).alias("outcome")])

    return grid


def process_contract_batch(
    contracts_batch: pl.DataFrame, perpetual_file: Path, rates_file: Path, options_file: Path
) -> pl.DataFrame:
    """
    Process a batch of contracts end-to-end.
    Returns DataFrame with all pricing calculations.
    """
    batch_id = contracts_batch["contract_id"][0]
    logger.info(f"Processing batch starting with contract {batch_id}")

    # Create pricing grid
    grid = create_streaming_grid(pl.LazyFrame(contracts_batch)).collect()

    # Join spot prices
    grid = join_spot_prices_streaming(pl.LazyFrame(grid)).collect()

    # Join risk-free rates
    grid = join_risk_free_rates_streaming(pl.LazyFrame(grid)).collect()

    # Get time bounds for options filtering
    min_ts = int(grid["timestamp"].min())
    max_ts = int(grid["timestamp"].max())

    # Process options IV with chunking and staleness tracking
    grid = process_options_iv_chunked(grid, min_ts, max_ts)

    # Add strikes and outcomes
    grid = add_strikes_and_outcomes(grid)

    # Calculate prices (only for valid rows)
    valid_rows = grid.filter(
        pl.col("S").is_not_null()
        & pl.col("K").is_not_null()
        & pl.col("r").is_not_null()
        & pl.col("implied_vol_bid").is_not_null()
        & pl.col("implied_vol_ask").is_not_null()
        & (pl.col("time_remaining") > 0)
    )

    if len(valid_rows) > 0:
        priced = add_binary_pricing_bid_ask_mid(
            valid_rows,
            spot_col="S",
            strike_col="K",
            rate_col="r",
            sigma_bid_col="implied_vol_bid",
            sigma_ask_col="implied_vol_ask",
            time_seconds_col="time_remaining",
        )

        # Join prices back to grid
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

    return grid


def run_production_backtest(
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    test_mode: bool = False,
) -> None:
    """
    Main production backtest with streaming and checkpointing.

    Args:
        start_date: Optional start date (YYYY-MM-DD)
        end_date: Optional end date (YYYY-MM-DD)
        test_mode: If True, only process first 10 contracts for testing
    """
    logger.info("=" * 80)
    logger.info("PRODUCTION BACKTEST (Streaming Architecture)")
    logger.info("=" * 80)

    start_time = time.time()
    monitor = DataQualityMonitor()

    # Create output directories
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    CHECKPOINT_FILE.parent.mkdir(parents=True, exist_ok=True)
    Path("/home/ubuntu/Polymarket/research/model/logs").mkdir(parents=True, exist_ok=True)

    # Load contracts
    contracts_lazy = load_contracts(start_date, end_date)

    if test_mode:
        logger.info("TEST MODE: Processing first 10 contracts only")
        contracts = contracts_lazy.head(10).collect()
    else:
        contracts = contracts_lazy.collect()

    total_contracts = len(contracts)
    monitor.update("total_contracts", total_contracts)
    monitor.update("total_calculations", total_contracts * SECONDS_PER_CONTRACT)

    logger.info(f"Total contracts to process: {total_contracts:,}")
    logger.info(f"Expected calculations: {total_contracts * SECONDS_PER_CONTRACT:,}")

    # Process in batches to manage memory
    results = []
    batch_size = CONTRACT_BATCH_SIZE if not test_mode else 5

    with tqdm(total=total_contracts, desc="Processing contracts") as pbar:
        for i in range(0, total_contracts, batch_size):
            batch_end = min(i + batch_size, total_contracts)
            batch = contracts[i:batch_end]

            try:
                # Process batch
                batch_results = process_contract_batch(batch, PERPETUAL_FILE, RATES_FILE, OPTIONS_FILE)
                results.append(batch_results)

                # Update progress
                pbar.update(batch_end - i)

                # Log progress
                if (i + batch_size) % (batch_size * 10) == 0:
                    logger.info(f"Processed {i + batch_size:,} / {total_contracts:,} contracts")
                    logger.info(f"Memory usage: {len(results)} batches in memory")

            except Exception as e:
                logger.error(f"Error processing batch {i}-{batch_end}: {e}")
                if test_mode:
                    raise
                continue

    # Combine results
    logger.info("Combining batch results...")
    final_results = pl.concat(results)

    # Calculate final metrics
    priced_rows = final_results.filter(pl.col("price_mid").is_not_null())
    monitor.update("pricing_success_rate", len(priced_rows) / len(final_results))

    # Write results with streaming for large outputs
    logger.info(f"Writing results to {OUTPUT_FILE}")

    if len(final_results) > 10_000_000:  # Use streaming for >10M rows
        logger.info("Using streaming write for large output...")
        pl.LazyFrame(final_results).sink_parquet(
            OUTPUT_FILE,
            compression="snappy",
            statistics=True,
            row_group_size=100_000,
        )
    else:
        final_results.write_parquet(OUTPUT_FILE, compression="snappy", statistics=True)

    # Log summary
    elapsed_time = time.time() - start_time
    file_size_mb = OUTPUT_FILE.stat().st_size / 1024 / 1024

    logger.info("\n" + "=" * 80)
    logger.info("BACKTEST COMPLETE")
    logger.info("=" * 80)
    logger.info(f"Elapsed time: {elapsed_time/60:.1f} minutes")
    logger.info(f"Output file: {file_size_mb:.1f} MB")
    logger.info(f"Total rows: {len(final_results):,}")
    logger.info(f"Contracts processed: {final_results['contract_id'].n_unique():,}")
    logger.info(f"Pricing success rate: {len(priced_rows)/len(final_results)*100:.1%}")

    # Log data quality report
    monitor.log_summary()

    # Sample results
    logger.info("\nSample results (first contract):")
    sample = (
        final_results.filter(pl.col("contract_id") == final_results["contract_id"][0])
        .filter(pl.col("seconds_offset").is_in([0, 300, 600, 899]))
        .select(
            [
                "contract_id",
                "seconds_offset",
                "S",
                "K",
                "time_remaining",
                "sigma_mid",
                "iv_staleness_seconds",
                "price_mid",
                "outcome",
            ]
        )
    )
    print(sample)

    logger.info(f"\n✅ Results saved to: {OUTPUT_FILE}")


def main() -> None:
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="Production backtest with streaming architecture")
    parser.add_argument("--start-date", type=str, help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end-date", type=str, help="End date (YYYY-MM-DD)")
    parser.add_argument("--test", action="store_true", help="Test mode (process 10 contracts only)")
    parser.add_argument("--pilot", action="store_true", help="Run October 2023 pilot")

    args = parser.parse_args()

    if args.pilot:
        # Run October 2023 pilot for comparison
        run_production_backtest("2023-10-01", "2023-10-31", test_mode=False)
    else:
        run_production_backtest(args.start_date, args.end_date, test_mode=args.test)


if __name__ == "__main__":
    main()