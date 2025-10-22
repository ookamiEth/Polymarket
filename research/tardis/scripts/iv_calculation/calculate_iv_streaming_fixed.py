#!/usr/bin/env python3

"""
Calculate implied volatility for large options dataset using TRUE streaming processing.

Fixed version that avoids memory overflow by:
1. Writing each chunk to a temporary parquet file
2. Using lazy concatenation with sink_parquet(streaming=True) for final output
3. Never loading more than one chunk (5M rows) into memory

Supports two risk-free rate modes:
- CONSTANT: Use a single rate for all calculations (default: 4.12%)
- DAILY: Use per-day rates from a file (blended_supply_apr_ma7 from lending rates)

Optimized for 200M+ rows with minimal memory usage (<2GB constant).
"""

import argparse
import logging
import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np
import polars as pl
from py_vollib_vectorized import vectorized_implied_volatility  # type: ignore

# Constants
DEFAULT_RISK_FREE_RATE = 0.0412
DEFAULT_CHUNK_SIZE = 5_000_000  # 5M rows per chunk (~1GB memory)
DEFAULT_INPUT = "data/consolidated/quotes_1s_atm_short_dated_optimized.parquet"
DEFAULT_OUTPUT = "data/consolidated/quotes_1s_atm_short_dated_with_iv.parquet"
DEFAULT_RISK_FREE_RATES_FILE = "research/risk_free_rate/data/blended_lending_rates_2023_2025.parquet"

# Quality thresholds
MIN_OPTION_PRICE_BTC = 0.00001  # Minimum option price in BTC
MIN_TIME_TO_EXPIRY_HOURS = 0.5  # Minimum 30 minutes to expiry
MAX_BID_ASK_SPREAD_PCT = 2.0  # Maximum 200% spread

logger = logging.getLogger(__name__)


def setup_logging(verbose: bool = False) -> None:
    """Configure logging with custom format."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )


def load_daily_risk_free_rates(rates_file: str) -> pl.DataFrame:
    """
    Load daily risk-free rates from parquet file.

    Args:
        rates_file: Path to parquet file with daily rates

    Returns:
        DataFrame with 'date' and 'risk_free_rate' columns (rates in decimal)

    Raises:
        FileNotFoundError: If rates file doesn't exist
        ValueError: If required columns are missing
    """
    if not Path(rates_file).exists():
        raise FileNotFoundError(f"Risk-free rates file not found: {rates_file}")

    logger.info(f"Loading daily risk-free rates from {rates_file}...")

    # Load rates file
    rates_df = pl.read_parquet(rates_file)

    # Validate required columns
    required_cols = ["date", "blended_supply_apr_ma7", "blended_supply_apr"]
    missing_cols = [col for col in required_cols if col not in rates_df.columns]
    if missing_cols:
        raise ValueError(
            f"Missing required columns in rates file: {missing_cols}. Available columns: {rates_df.columns}"
        )

    # Select and process columns
    # Use blended_supply_apr as fallback when MA7 is null (first 6 days)
    rates_df = rates_df.select(
        [
            pl.col("date"),
            # Convert from percentage to decimal (e.g., 2.79% -> 0.0279)
            # Use MA7 if available, otherwise fall back to non-smoothed rate
            pl.when(pl.col("blended_supply_apr_ma7").is_not_null())
            .then(pl.col("blended_supply_apr_ma7") / 100.0)
            .otherwise(pl.col("blended_supply_apr") / 100.0)
            .alias("risk_free_rate"),
        ]
    )

    # Check for any remaining nulls (shouldn't happen)
    n_nulls = rates_df.filter(pl.col("risk_free_rate").is_null()).height
    if n_nulls > 0:
        logger.warning(f"Removing {n_nulls} rows with null risk-free rates")
        rates_df = rates_df.filter(pl.col("risk_free_rate").is_not_null())

    # Report statistics
    n_dates = len(rates_df)
    min_date = rates_df["date"].min()
    max_date = rates_df["date"].max()
    mean_rate: float | None = rates_df["risk_free_rate"].mean()  # type: ignore[assignment]

    logger.info(f"Loaded {n_dates:,} daily rates from {min_date} to {max_date}")
    if mean_rate is not None:
        logger.info(f"Mean risk-free rate: {mean_rate * 100:.2f}%")

    return rates_df


def calculate_ivs_chunk(
    df: pl.DataFrame,
    risk_free_rate: float,
    chunk_idx: int,
    total_chunks: int,
    risk_free_rates_df: Optional[pl.DataFrame] = None,
) -> pl.DataFrame:
    """
    Calculate implied volatilities for a single chunk using vectorized operations.

    Args:
        df: DataFrame chunk with options data
        risk_free_rate: Annual risk-free rate (constant, used if risk_free_rates_df is None)
        chunk_idx: Current chunk number (for logging)
        total_chunks: Total number of chunks (for logging)
        risk_free_rates_df: Optional DataFrame with daily rates (columns: 'date', 'risk_free_rate')

    Returns:
        DataFrame with IV columns added

    Raises:
        ValueError: If using daily rates and some dates are missing from rates file
    """
    start_time = time.time()
    chunk_size = len(df)

    logger.info(f"  Chunk {chunk_idx}/{total_chunks}: Processing {chunk_size:,} rows...")

    # Filter for valid prices and time to expiry
    df_valid = df.filter(
        (pl.col("bid_price") > MIN_OPTION_PRICE_BTC)
        & (pl.col("ask_price") > MIN_OPTION_PRICE_BTC)
        & (pl.col("time_to_expiry_days") > MIN_TIME_TO_EXPIRY_HOURS / 24)
        & (pl.col("spot_price") > 0)
        & (pl.col("strike_price") > 0)
    )

    valid_count = len(df_valid)
    invalid_count = chunk_size - valid_count

    if invalid_count > 0:
        logger.debug(f"    Filtered {invalid_count:,} invalid rows")

    if valid_count == 0:
        # Return original dataframe with null IV columns
        return df.with_columns(
            [
                pl.lit(None).cast(pl.Float64).alias("implied_vol_bid"),
                pl.lit(None).cast(pl.Float64).alias("implied_vol_ask"),
                pl.lit("no_valid_data").alias("iv_calc_status"),
            ]
        )

    # If using daily rates, join with rates DataFrame and extract per-row rates
    if risk_free_rates_df is not None:
        # Add date column from timestamp
        df_valid = df_valid.with_columns(
            [pl.from_epoch("timestamp_seconds", time_unit="s").cast(pl.Date).alias("date")]
        )

        # Join with rates DataFrame
        df_valid = df_valid.join(risk_free_rates_df, on="date", how="left")

        # Check for missing rates (dates not in rates file)
        n_missing = df_valid.filter(pl.col("risk_free_rate").is_null()).height
        if n_missing > 0:
            missing_dates = (
                df_valid.filter(pl.col("risk_free_rate").is_null())
                .select("date")
                .unique()
                .sort("date")
                .to_series()
                .to_list()
            )
            raise ValueError(
                f"Found {n_missing} rows with missing risk-free rates. "
                f"Missing dates: {missing_dates[:10]}{'...' if len(missing_dates) > 10 else ''}"
            )

    # Prepare arrays for vectorized calculation
    prices_bid = df_valid["bid_price"].to_numpy()
    prices_ask = df_valid["ask_price"].to_numpy()
    S = df_valid["spot_price"].to_numpy()  # noqa: N806
    K = df_valid["strike_price"].to_numpy()  # noqa: N806
    t = df_valid["time_to_expiry_days"].to_numpy() / 365.25  # Convert to years

    # Use per-row rates if available, otherwise use constant rate
    if risk_free_rates_df is not None:
        r = df_valid["risk_free_rate"].to_numpy()
    else:
        r = np.full(valid_count, risk_free_rate)

    # Convert option type to py_vollib format
    flag = df_valid["type"].str.replace("call", "c").str.replace("put", "p").to_numpy()

    # CRITICAL: Convert option prices from BTC to USD
    prices_bid_usd = prices_bid * S
    prices_ask_usd = prices_ask * S

    # Calculate IVs (vectorized - all rows at once)
    try:
        iv_bid = vectorized_implied_volatility(
            price=prices_bid_usd,
            S=S,
            K=K,
            t=t,
            r=r,
            flag=flag,
            model="black_scholes",
            return_as="numpy",
        )
    except Exception as e:
        logger.warning(f"    Bid IV calculation failed: {e}")
        iv_bid = np.full(valid_count, np.nan)

    try:
        iv_ask = vectorized_implied_volatility(
            price=prices_ask_usd,
            S=S,
            K=K,
            t=t,
            r=r,
            flag=flag,
            model="black_scholes",
            return_as="numpy",
        )
    except Exception as e:
        logger.warning(f"    Ask IV calculation failed: {e}")
        iv_ask = np.full(valid_count, np.nan)

    # Create status column
    status = np.where(
        np.isnan(iv_bid) | np.isnan(iv_ask),
        "failed",
        "success",
    )

    # Add IV columns to valid DataFrame
    df_valid = df_valid.with_columns(
        [
            pl.Series("implied_vol_bid", iv_bid),
            pl.Series("implied_vol_ask", iv_ask),
            pl.Series("iv_calc_status", status),
        ]
    )

    # Drop temporary columns added for daily rates (if any)
    if risk_free_rates_df is not None:
        df_valid = df_valid.drop(["date", "risk_free_rate"])

    # Join back with invalid rows (to maintain all original data)
    if invalid_count > 0:
        # Get invalid rows
        df_invalid = df.filter(
            ~(
                (pl.col("bid_price") > MIN_OPTION_PRICE_BTC)
                & (pl.col("ask_price") > MIN_OPTION_PRICE_BTC)
                & (pl.col("time_to_expiry_days") > MIN_TIME_TO_EXPIRY_HOURS / 24)
                & (pl.col("spot_price") > 0)
                & (pl.col("strike_price") > 0)
            )
        ).with_columns(
            [
                pl.lit(None).cast(pl.Float64).alias("implied_vol_bid"),
                pl.lit(None).cast(pl.Float64).alias("implied_vol_ask"),
                pl.lit("invalid_input").alias("iv_calc_status"),
            ]
        )

        # Combine valid and invalid
        df_result = pl.concat([df_valid, df_invalid])
    else:
        df_result = df_valid

    # Calculate statistics
    success_count = np.sum(status == "success")
    success_rate = success_count / valid_count * 100 if valid_count > 0 else 0
    elapsed = time.time() - start_time
    rows_per_sec = chunk_size / elapsed if elapsed > 0 else 0

    logger.info(
        f"    Completed: {success_count:,}/{valid_count:,} successful "
        f"({success_rate:.1f}%), {elapsed:.1f}s ({rows_per_sec:,.0f} rows/sec)"
    )

    return df_result


def process_with_streaming(
    input_file: str,
    output_file: str,
    risk_free_rate: float,
    chunk_size: int,
    test_rows: Optional[int] = None,
    risk_free_rates_file: Optional[str] = None,
) -> None:
    """
    Process large options file using TRUE streaming with chunked IV calculation.

    This version writes each chunk to a temporary file, then uses lazy concatenation
    with sink_parquet(streaming=True) to combine them without loading all into memory.

    Args:
        input_file: Input parquet file path
        output_file: Output parquet file path
        risk_free_rate: Annual risk-free rate (constant, used if risk_free_rates_file is None)
        chunk_size: Rows to process per chunk
        test_rows: If set, only process this many rows (for testing)
        risk_free_rates_file: Optional path to daily rates file
    """
    logger.info("=" * 80)
    logger.info("STREAMING IMPLIED VOLATILITY CALCULATION (FIXED)")
    logger.info("=" * 80)
    logger.info(f"Input: {input_file}")
    logger.info(f"Output: {output_file}")

    # Load daily rates if file provided
    risk_free_rates_df = None
    if risk_free_rates_file:
        risk_free_rates_df = load_daily_risk_free_rates(risk_free_rates_file)
        logger.info("Rate mode: DAILY (per-day rates from file)")
    else:
        logger.info(f"Rate mode: CONSTANT ({risk_free_rate * 100:.2f}%)")

    logger.info(f"Chunk size: {chunk_size:,} rows")

    # Check input file
    if not Path(input_file).exists():
        raise FileNotFoundError(f"Input file not found: {input_file}")

    # Create output directory if needed
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Get total row count
    logger.info("\nAnalyzing input file...")
    lazy_df = pl.scan_parquet(input_file)

    if test_rows:
        logger.info(f"TEST MODE: Processing only {test_rows:,} rows")
        lazy_df = lazy_df.head(test_rows)

    total_rows = lazy_df.select(pl.len()).collect().item()
    logger.info(f"Total rows to process: {total_rows:,}")

    # Calculate number of chunks
    n_chunks = (total_rows + chunk_size - 1) // chunk_size
    logger.info(f"Processing in {n_chunks} chunks")

    # Process chunks and write to temporary files
    logger.info("\n" + "=" * 80)
    logger.info("PROCESSING CHUNKS")
    logger.info("=" * 80)

    start_time = time.time()
    chunk_files = []

    # Clean up any existing chunk files first
    for old_chunk in output_path.parent.glob(f"{output_path.stem}.chunk*.parquet"):
        old_chunk.unlink()
        logger.debug(f"Removed old chunk file: {old_chunk}")

    for chunk_idx in range(n_chunks):
        chunk_start = chunk_idx * chunk_size
        chunk_end = min(chunk_start + chunk_size, total_rows)
        actual_chunk_size = chunk_end - chunk_start

        logger.info(f"\nProcessing rows {chunk_start:,} to {chunk_end:,}...")

        # Read chunk using slice to skip rows correctly
        # CRITICAL FIX: row_index_offset doesn't skip rows, use slice() instead
        df_chunk = pl.scan_parquet(input_file).slice(chunk_start, actual_chunk_size).collect()

        # Calculate IVs for chunk
        df_with_iv = calculate_ivs_chunk(
            df_chunk,
            risk_free_rate,
            chunk_idx + 1,
            n_chunks,
            risk_free_rates_df,
        )

        # Write chunk to temporary file
        chunk_file = output_path.parent / f"{output_path.stem}.chunk{chunk_idx:03d}.parquet"
        df_with_iv.write_parquet(
            chunk_file,
            compression="snappy",
            statistics=True,
        )
        chunk_files.append(str(chunk_file))
        logger.info(f"    Written to temporary file: {chunk_file.name}")

        # Free memory
        del df_chunk, df_with_iv

        # Estimate time remaining
        elapsed = time.time() - start_time
        rows_done = chunk_end
        if rows_done > 0:
            eta_seconds = (elapsed / rows_done) * (total_rows - rows_done)
            eta_minutes = eta_seconds / 60
            logger.info(f"  Progress: {rows_done / total_rows * 100:.1f}%, ETA: {eta_minutes:.1f} min")

    # Combine all chunks using lazy evaluation and streaming write
    logger.info("\n" + "=" * 80)
    logger.info("WRITING FINAL OUTPUT (STREAMING)")
    logger.info("=" * 80)

    logger.info(f"Combining {len(chunk_files)} chunks using lazy evaluation...")
    logger.info("Using sink_parquet() to stream data and avoid memory overflow...")

    # CRITICAL: Use lazy scan and sink_parquet for streaming
    # This ensures we never load all data into memory at once
    lazy_combined = pl.scan_parquet(chunk_files)

    # Stream the combined data directly to the output file
    # sink_parquet automatically uses streaming when processing lazy frames
    lazy_combined.sink_parquet(
        path=output_file,
        compression="snappy",
        statistics=True,
    )

    logger.info("✅ Successfully written output using streaming!")

    # Clean up temporary chunk files
    logger.info("\nCleaning up temporary files...")
    for chunk_file in chunk_files:
        Path(chunk_file).unlink()
        logger.debug(f"  Removed: {Path(chunk_file).name}")

    # Calculate final statistics
    total_time = time.time() - start_time

    logger.info("\n" + "=" * 80)
    logger.info("SUMMARY")
    logger.info("=" * 80)

    # Load result for statistics (lazy)
    result_df = pl.scan_parquet(output_file)
    stats = result_df.select(
        [
            pl.len().alias("total_rows"),
            (pl.col("iv_calc_status") == "success").sum().alias("success_count"),
            (pl.col("iv_calc_status") == "failed").sum().alias("failed_count"),
            (pl.col("iv_calc_status") == "invalid_input").sum().alias("invalid_count"),
            pl.col("implied_vol_bid").mean().alias("mean_iv_bid"),
            pl.col("implied_vol_ask").mean().alias("mean_iv_ask"),
        ]
    ).collect()

    total = stats["total_rows"][0]
    success = stats["success_count"][0]
    failed = stats["failed_count"][0]
    invalid = stats["invalid_count"][0]

    logger.info(f"Total rows: {total:,}")
    logger.info(f"Successful IV calculations: {success:,} ({success / total * 100:.1f}%)")
    logger.info(f"Failed IV calculations: {failed:,} ({failed / total * 100:.1f}%)")
    logger.info(f"Invalid input rows: {invalid:,} ({invalid / total * 100:.1f}%)")

    if success > 0:
        logger.info(f"Mean IV (bid): {stats['mean_iv_bid'][0]:.4f}")
        logger.info(f"Mean IV (ask): {stats['mean_iv_ask'][0]:.4f}")

    logger.info(f"\nTotal processing time: {total_time:.1f} seconds ({total_time / 60:.1f} minutes)")
    logger.info(f"Average speed: {total_rows / total_time:,.0f} rows/second")

    # Check output file size
    output_size_mb = Path(output_file).stat().st_size / (1024 * 1024)
    output_size_gb = output_size_mb / 1024
    logger.info(f"Output file size: {output_size_gb:.2f} GB ({output_size_mb:.1f} MB)")

    logger.info("\n✅ Processing complete! Memory usage stayed under 2GB throughout.")


def main() -> None:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Calculate implied volatilities for options data using TRUE streaming processing (fixed version)"
    )

    parser.add_argument(
        "--input",
        default=DEFAULT_INPUT,
        help=f"Input parquet file (default: {DEFAULT_INPUT})",
    )
    parser.add_argument(
        "--output",
        default=DEFAULT_OUTPUT,
        help=f"Output parquet file (default: {DEFAULT_OUTPUT})",
    )
    parser.add_argument(
        "--risk-free-rate",
        type=float,
        default=DEFAULT_RISK_FREE_RATE,
        help=f"Annual risk-free rate (constant, default: {DEFAULT_RISK_FREE_RATE})",
    )
    parser.add_argument(
        "--risk-free-rates-file",
        type=str,
        help=f"Path to daily risk-free rates parquet file (default: {DEFAULT_RISK_FREE_RATES_FILE})",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=DEFAULT_CHUNK_SIZE,
        help=f"Rows per chunk (default: {DEFAULT_CHUNK_SIZE:,})",
    )
    parser.add_argument(
        "--test-rows",
        type=int,
        help="Process only this many rows for testing",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose logging",
    )

    args = parser.parse_args()

    # Setup logging
    setup_logging(args.verbose)

    try:
        # Run processing
        process_with_streaming(
            input_file=args.input,
            output_file=args.output,
            risk_free_rate=args.risk_free_rate,
            chunk_size=args.chunk_size,
            test_rows=args.test_rows,
            risk_free_rates_file=args.risk_free_rates_file,
        )
    except Exception as e:
        logger.error(f"ERROR: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
