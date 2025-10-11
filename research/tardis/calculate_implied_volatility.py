#!/usr/bin/env python3

import argparse
import logging
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np
import polars as pl
from scipy import optimize
from scipy.stats import norm

# Constants
DEFAULT_RISK_FREE_RATE = 0.05
DEFAULT_SPOT_COLUMN = "mid_px"
DEFAULT_OUTPUT_SUFFIX = "_with_iv"
DEFAULT_IV_BOUNDS = (0.001, 5.0)
DEFAULT_TOLERANCE = 1e-6
DERIBIT_EXPIRY_HOUR_UTC = 8
SECONDS_PER_YEAR = 365.25 * 24 * 3600
MERGE_TOLERANCE_SECONDS = 60

logger = logging.getLogger(__name__)


def _empty_dataframe() -> pl.DataFrame:
    """Return empty DataFrame with expected schema."""
    schema = {
        "exchange": pl.Utf8,
        "symbol": pl.Utf8,
        "timestamp": pl.Int64,
        "local_timestamp": pl.Int64,
        "type": pl.Utf8,
        "strike_price": pl.Float64,
        "underlying": pl.Utf8,
        "expiry_str": pl.Utf8,
        "bid_price": pl.Float64,
        "bid_amount": pl.Float64,
        "ask_price": pl.Float64,
        "ask_amount": pl.Float64,
        "spot_price": pl.Float64,
        "time_to_expiry": pl.Float64,
        "implied_vol_bid": pl.Float64,
        "implied_vol_ask": pl.Float64,
        "iv_calc_status": pl.Utf8,
    }
    return pl.DataFrame(schema=schema)


def black_scholes_price(
    S: float,  # noqa: N803
    K: float,  # noqa: N803
    T: float,  # noqa: N803
    r: float,
    sigma: float,
    option_type: str,
) -> float:
    """
    Calculate Black-Scholes option price.

    Args:
        S: Spot price
        K: Strike price
        T: Time to expiry (years)
        r: Risk-free rate
        sigma: Volatility
        option_type: "call" or "put"

    Returns:
        Option price
    """
    if T <= 0 or sigma <= 0:
        # Handle edge cases
        return max(S - K, 0) if option_type == "call" else max(K - S, 0)

    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)

    if option_type == "call":
        price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    else:
        price = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)

    return float(price)


def calculate_implied_volatility(
    market_price: float,
    S: float,  # noqa: N803
    K: float,  # noqa: N803
    T: float,  # noqa: N803
    r: float,
    option_type: str,
    bounds: tuple[float, float] = DEFAULT_IV_BOUNDS,
    tolerance: float = DEFAULT_TOLERANCE,
) -> tuple[Optional[float], str]:
    """
    Calculate implied volatility using Brent's method.

    Args:
        market_price: Observed market price
        S: Spot price
        K: Strike price
        T: Time to expiry (years)
        r: Risk-free rate
        option_type: "call" or "put"
        bounds: (min_vol, max_vol) search bounds
        tolerance: Convergence tolerance

    Returns:
        (implied_vol, status) tuple where status is:
        - "success": IV calculated successfully
        - "failed_bounds": No solution within bounds
        - "failed_convergence": Solver did not converge
        - "invalid_inputs": Invalid input parameters
    """
    # Validate inputs
    if market_price <= 0 or S <= 0 or K <= 0 or T <= 0:
        return (None, "invalid_inputs")

    # Check intrinsic value
    intrinsic = max(S - K, 0) if option_type == "call" else max(K - S, 0)

    if market_price < intrinsic * 0.99:  # Allow small numerical tolerance
        return (None, "invalid_inputs")

    def objective(sigma: float) -> float:
        """Objective function: BS_price(sigma) - market_price"""
        try:
            return black_scholes_price(S, K, T, r, sigma, option_type) - market_price
        except Exception:
            return float("inf")

    try:
        # Check if solution exists within bounds
        f_low = objective(bounds[0])
        f_high = objective(bounds[1])

        if f_low * f_high > 0:
            # Try wider bounds
            wider_bounds = (0.0001, 10.0)
            f_low_wide = objective(wider_bounds[0])
            f_high_wide = objective(wider_bounds[1])

            if f_low_wide * f_high_wide > 0:
                return (None, "failed_bounds")

            bounds = wider_bounds

        # Solve for IV using Brent's method
        iv_result = optimize.brentq(objective, bounds[0], bounds[1], xtol=tolerance, rtol=np.float64(tolerance))

        # Type narrowing: brentq returns float when full_output=False (default)
        assert isinstance(iv_result, (float, np.floating))

        return (float(iv_result), "success")

    except ValueError:
        return (None, "failed_bounds")
    except RuntimeError:
        return (None, "failed_convergence")
    except Exception:
        return (None, "failed_convergence")


def parse_expiry_datetime(expiry_str: str, base_year: Optional[int] = None) -> Optional[datetime]:
    """
    Parse Deribit expiry string to datetime.

    Format: "1JAN25" → datetime(2025, 1, 1, 8, 0, 0, UTC)
    Deribit expiry time is 08:00 UTC.

    Args:
        expiry_str: Expiry string like "1JAN25"
        base_year: Reference year for 2-digit year parsing (default: current year)

    Returns:
        Expiry datetime in UTC or None if parsing fails
    """
    try:
        # Expected format: "1JAN25" or "31DEC24"
        import re

        match = re.match(r"(\d{1,2})([A-Z]{3})(\d{2})", expiry_str.upper())
        if not match:
            return None

        day = int(match.group(1))
        month_str = match.group(2)
        year_two_digit = int(match.group(3))

        # Convert 2-digit year to 4-digit
        if base_year is None:
            base_year = datetime.now(timezone.utc).year

        century = (base_year // 100) * 100
        year = century + year_two_digit

        # If year is more than 20 years in the past, assume next century
        if year < base_year - 20:
            year += 100

        # Parse month
        month_map = {
            "JAN": 1,
            "FEB": 2,
            "MAR": 3,
            "APR": 4,
            "MAY": 5,
            "JUN": 6,
            "JUL": 7,
            "AUG": 8,
            "SEP": 9,
            "OCT": 10,
            "NOV": 11,
            "DEC": 12,
        }

        month = month_map.get(month_str)
        if month is None:
            return None

        return datetime(year, month, day, DERIBIT_EXPIRY_HOUR_UTC, 0, 0, tzinfo=timezone.utc)

    except Exception:
        return None


def load_spot_data(
    spot_data_dir: str, underlying: str, date: str, spot_column: str = DEFAULT_SPOT_COLUMN
) -> pl.DataFrame:
    """
    Load Hyperliquid spot price data for given date and underlying.

    Args:
        spot_data_dir: Base directory with Hyperliquid data
        underlying: Asset symbol (BTC, ETH)
        date: Date string (YYYY-MM-DD)
        spot_column: Column to use as spot price

    Returns:
        DataFrame with columns: time (Datetime), spot_price (Float64)
    """
    # Path format: {spot_data_dir}/YYYY/MM/YYYYMMDD_{UNDERLYING}.parquet
    dt = datetime.strptime(date, "%Y-%m-%d")
    year = dt.strftime("%Y")
    month = dt.strftime("%m")
    filename = f"{dt.strftime('%Y%m%d')}_{underlying}.parquet"

    spot_file_path = os.path.join(spot_data_dir, year, month, filename)

    if not os.path.exists(spot_file_path):
        raise FileNotFoundError(f"Spot data not found: {spot_file_path}")

    logger.info(f"Loading spot data: {spot_file_path}")

    df = pl.read_parquet(spot_file_path)

    if spot_column not in df.columns:
        raise ValueError(f"Spot column '{spot_column}' not found. Available: {df.columns}")

    # Select and rename
    df = df.select([pl.col("time"), pl.col(spot_column).alias("spot_price")])

    logger.info(f"Loaded {len(df):,} spot price records for {underlying} on {date}")

    return df


def merge_with_spot_prices(options_df: pl.DataFrame, spot_df: pl.DataFrame) -> pl.DataFrame:
    """
    Merge options data with spot prices using nearest timestamp.

    Args:
        options_df: Options DataFrame with timestamp column (microseconds)
        spot_df: Spot DataFrame with time column (Datetime UTC)

    Returns:
        Merged DataFrame with spot_price column
    """
    # Convert options timestamp (microseconds) to datetime with UTC timezone
    options_df = options_df.with_columns(
        [pl.from_epoch(pl.col("timestamp"), time_unit="us").dt.replace_time_zone("UTC").alias("timestamp_dt")]
    )

    # Sort both dataframes by timestamp (required for asof join)
    options_df = options_df.sort("timestamp_dt")
    spot_df = spot_df.sort("time")

    # Perform asof join (nearest timestamp within tolerance)
    merged_df = options_df.join_asof(
        spot_df,
        left_on="timestamp_dt",
        right_on="time",
        strategy="nearest",
        tolerance=f"{MERGE_TOLERANCE_SECONDS}s",
    )

    # Drop temporary column
    merged_df = merged_df.drop("timestamp_dt")

    # Check for missing spot prices
    missing_count = merged_df.filter(pl.col("spot_price").is_null()).shape[0]
    if missing_count > 0:
        logger.warning(f"{missing_count:,} rows missing spot price (outside {MERGE_TOLERANCE_SECONDS}s tolerance)")

    return merged_df


def add_time_to_expiry(df: pl.DataFrame) -> pl.DataFrame:
    """
    Parse expiry_str and calculate time_to_expiry in years.

    Args:
        df: DataFrame with expiry_str and timestamp columns

    Returns:
        DataFrame with time_to_expiry column added
    """
    logger.info("Parsing expiry dates and calculating time to expiry...")

    # Parse expiry_str to datetime using a Python UDF
    def parse_expiry(expiry_str: str) -> Optional[int]:
        """Parse expiry string to timestamp (microseconds)."""
        expiry_dt = parse_expiry_datetime(expiry_str)
        if expiry_dt is None:
            return None
        return int(expiry_dt.timestamp() * 1_000_000)

    # Apply parsing function
    df = df.with_columns(
        [pl.col("expiry_str").map_elements(parse_expiry, return_dtype=pl.Int64).alias("expiry_timestamp")]
    )

    # Convert to datetime
    df = df.with_columns([pl.from_epoch(pl.col("expiry_timestamp"), time_unit="us").alias("expiry_datetime")])

    # Convert timestamp (microseconds) to datetime
    df = df.with_columns([pl.from_epoch(pl.col("timestamp"), time_unit="us").alias("timestamp_dt")])

    # Calculate time to expiry in years
    df = df.with_columns(
        [
            (
                (pl.col("expiry_datetime").cast(pl.Int64) - pl.col("timestamp_dt").cast(pl.Int64))
                / (SECONDS_PER_YEAR * 1_000_000)
            ).alias("time_to_expiry")
        ]
    )

    # Drop temporary columns
    df = df.drop(["expiry_timestamp", "expiry_datetime", "timestamp_dt"])

    # Filter out invalid time_to_expiry
    invalid_count = df.filter(pl.col("time_to_expiry").is_null() | (pl.col("time_to_expiry") <= 0)).shape[0]

    if invalid_count > 0:
        logger.warning(f"Filtering {invalid_count:,} rows with invalid time_to_expiry")
        df = df.filter(pl.col("time_to_expiry") > 0)

    return df


def calculate_ivs_batch(
    df: pl.DataFrame,
    risk_free_rate: float,
    iv_bounds: tuple[float, float],
    tolerance: float,
) -> pl.DataFrame:
    """
    Calculate implied volatilities for bid and ask prices.

    Args:
        df: DataFrame with required columns
        risk_free_rate: Annual risk-free rate
        iv_bounds: (min_vol, max_vol) search bounds
        tolerance: Convergence tolerance

    Returns:
        DataFrame with implied_vol_bid, implied_vol_ask, iv_calc_status columns
    """
    logger.info(f"Calculating implied volatilities for {len(df):,} rows...")

    # Filter out rows with null values in required columns
    df_valid = df.filter(
        pl.col("bid_price").is_not_null()
        & pl.col("ask_price").is_not_null()
        & pl.col("spot_price").is_not_null()
        & pl.col("strike_price").is_not_null()
        & pl.col("time_to_expiry").is_not_null()
    )

    filtered_count = len(df) - len(df_valid)
    if filtered_count > 0:
        logger.warning(f"Filtered {filtered_count:,} rows with null values in required columns")

    # Initialize result columns
    iv_bid_list = []
    iv_ask_list = []
    status_list = []

    # Convert to row dictionaries for iteration
    rows = df_valid.select(
        [
            "bid_price",
            "ask_price",
            "spot_price",
            "strike_price",
            "time_to_expiry",
            "type",
        ]
    ).to_dicts()

    success_count = 0
    failed_count = 0

    for idx, row in enumerate(rows):
        if idx % 10000 == 0 and idx > 0:
            logger.info(f"  Processed {idx:,}/{len(rows):,} rows...")

        S = row["spot_price"]  # noqa: N806
        K = row["strike_price"]  # noqa: N806
        T = row["time_to_expiry"]  # noqa: N806
        option_type = row["type"]
        bid_price = row["bid_price"]
        ask_price = row["ask_price"]

        # Calculate IV for bid
        iv_bid, status_bid = calculate_implied_volatility(
            bid_price, S, K, T, risk_free_rate, option_type, iv_bounds, tolerance
        )

        # Calculate IV for ask
        iv_ask, status_ask = calculate_implied_volatility(
            ask_price, S, K, T, risk_free_rate, option_type, iv_bounds, tolerance
        )

        # Determine overall status
        if status_bid == "success" and status_ask == "success":
            status = "success"
            success_count += 1
        else:
            status = f"bid:{status_bid},ask:{status_ask}"
            failed_count += 1

        iv_bid_list.append(iv_bid)
        iv_ask_list.append(iv_ask)
        status_list.append(status)

    # Add columns to DataFrame
    df_valid = df_valid.with_columns(
        [
            pl.Series("implied_vol_bid", iv_bid_list),
            pl.Series("implied_vol_ask", iv_ask_list),
            pl.Series("iv_calc_status", status_list),
        ]
    )

    success_rate = success_count / max(len(df_valid), 1) * 100
    logger.info(f"IV calculation complete: {success_count:,} success, {failed_count:,} failed ({success_rate:.1f}%)")

    return df_valid


def process_options_file(
    input_file: str,
    spot_data_dir: str,
    risk_free_rate: float,
    spot_column: str,
    iv_bounds: tuple[float, float],
    tolerance: float,
) -> pl.DataFrame:
    """
    Process Deribit options file and calculate implied volatilities.

    Args:
        input_file: Path to Deribit options Parquet file
        spot_data_dir: Directory with Hyperliquid spot data
        risk_free_rate: Annual risk-free rate
        spot_column: Column to use as spot price
        iv_bounds: (min_vol, max_vol) search bounds
        tolerance: Convergence tolerance

    Returns:
        DataFrame with IV columns added
    """
    logger.info("=" * 80)
    logger.info("LOAD OPTIONS DATA")
    logger.info("=" * 80)

    if not os.path.exists(input_file):
        raise FileNotFoundError(f"Input file not found: {input_file}")

    options_df = pl.read_parquet(input_file)
    logger.info(f"Loaded {len(options_df):,} options records from {input_file}")

    # Extract date and underlying from filename
    # Expected format: deribit_options_2025-09-01_BTC_1s.parquet
    filename = os.path.basename(input_file)
    parts = filename.replace("deribit_options_", "").replace(".parquet", "").split("_")

    if len(parts) < 2:
        raise ValueError(f"Cannot parse filename: {filename}")

    date = parts[0]
    underlying = parts[1]

    logger.info(f"Detected: {underlying} on {date}")

    logger.info("=" * 80)
    logger.info("LOAD SPOT PRICES")
    logger.info("=" * 80)

    spot_df = load_spot_data(spot_data_dir, underlying, date, spot_column)

    logger.info("=" * 80)
    logger.info("MERGE WITH SPOT PRICES")
    logger.info("=" * 80)

    options_df = merge_with_spot_prices(options_df, spot_df)

    logger.info("=" * 80)
    logger.info("CALCULATE TIME TO EXPIRY")
    logger.info("=" * 80)

    options_df = add_time_to_expiry(options_df)

    logger.info("=" * 80)
    logger.info("CALCULATE IMPLIED VOLATILITIES")
    logger.info("=" * 80)

    options_df = calculate_ivs_batch(options_df, risk_free_rate, iv_bounds, tolerance)

    return options_df


def save_output(df: pl.DataFrame, output_path: str) -> None:
    """Save DataFrame to Parquet file."""
    df.write_parquet(output_path)
    size_mb = os.path.getsize(output_path) / 1_000_000
    logger.info(f"Saved: {output_path} ({size_mb:.2f} MB)")


def main() -> None:
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Calculate implied volatilities for Deribit options data")

    parser.add_argument("--input-file", required=True, help="Input Parquet file (Deribit options)")
    parser.add_argument(
        "--spot-data-dir",
        required=True,
        help="Directory with Hyperliquid spot data (e.g., /path/to/asset_ctxs)",
    )
    parser.add_argument(
        "--risk-free-rate",
        type=float,
        default=DEFAULT_RISK_FREE_RATE,
        help=f"Annual risk-free rate (default: {DEFAULT_RISK_FREE_RATE})",
    )
    parser.add_argument(
        "--spot-column",
        default=DEFAULT_SPOT_COLUMN,
        help=f"Spot price column to use (default: {DEFAULT_SPOT_COLUMN})",
    )
    parser.add_argument(
        "--output-suffix",
        default=DEFAULT_OUTPUT_SUFFIX,
        help=f"Suffix for output file (default: {DEFAULT_OUTPUT_SUFFIX})",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Also overwrite the original input file",
    )
    parser.add_argument(
        "--iv-bounds",
        default=f"{DEFAULT_IV_BOUNDS[0]},{DEFAULT_IV_BOUNDS[1]}",
        help=f"IV search bounds as 'min,max' (default: {DEFAULT_IV_BOUNDS[0]},{DEFAULT_IV_BOUNDS[1]})",
    )
    parser.add_argument(
        "--tolerance",
        type=float,
        default=DEFAULT_TOLERANCE,
        help=f"Solver convergence tolerance (default: {DEFAULT_TOLERANCE})",
    )
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable debug logging")

    args = parser.parse_args()

    # Configure logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    # Parse IV bounds
    try:
        bounds_parts = args.iv_bounds.split(",")
        if len(bounds_parts) != 2:
            raise ValueError("iv_bounds must be in format 'min,max'")
        iv_bounds = (float(bounds_parts[0]), float(bounds_parts[1]))
        if iv_bounds[0] >= iv_bounds[1]:
            raise ValueError("min_bound must be < max_bound")
        if iv_bounds[0] <= 0:
            raise ValueError("min_bound must be > 0")
    except ValueError as e:
        logger.error(f"ERROR: Invalid iv_bounds: {e}")
        sys.exit(1)

    # Validate inputs
    if not os.path.exists(args.input_file):
        logger.error(f"ERROR: Input file not found: {args.input_file}")
        sys.exit(1)

    if not os.path.isdir(args.spot_data_dir):
        logger.error(f"ERROR: Spot data directory not found: {args.spot_data_dir}")
        sys.exit(1)

    logger.info("=" * 80)
    logger.info("IMPLIED VOLATILITY CALCULATOR")
    logger.info("=" * 80)
    logger.info(f"Input: {args.input_file}")
    logger.info(f"Spot data: {args.spot_data_dir} (column: {args.spot_column})")
    logger.info(f"Risk-free rate: {args.risk_free_rate * 100:.2f}%")
    logger.info(f"IV bounds: [{iv_bounds[0]}, {iv_bounds[1]}]")
    logger.info(f"Tolerance: {args.tolerance}")

    try:
        # Process file
        df = process_options_file(
            input_file=args.input_file,
            spot_data_dir=args.spot_data_dir,
            risk_free_rate=args.risk_free_rate,
            spot_column=args.spot_column,
            iv_bounds=iv_bounds,
            tolerance=args.tolerance,
        )

        logger.info("=" * 80)
        logger.info("SAVE OUTPUT")
        logger.info("=" * 80)

        # Generate output path with suffix
        input_path = Path(args.input_file)
        output_filename = input_path.stem + args.output_suffix + input_path.suffix
        output_path = str(input_path.parent / output_filename)

        save_output(df, output_path)

        # Optionally overwrite original
        if args.overwrite:
            logger.info(f"Overwriting original: {args.input_file}")
            save_output(df, args.input_file)

        logger.info("=" * 80)
        logger.info("SUCCESS")
        logger.info("=" * 80)
        logger.info(f"Output: {output_path}")
        logger.info(f"Rows: {len(df):,}")
        logger.info(f"Columns: {len(df.columns)}")

        # Summary statistics
        success_count = df.filter(pl.col("iv_calc_status") == "success").shape[0]
        success_rate = success_count / max(len(df), 1) * 100
        logger.info(f"IV success rate: {success_count:,}/{len(df):,} ({success_rate:.1f}%)")

        if success_count > 0:
            iv_bid_mean = (
                df.filter(pl.col("implied_vol_bid").is_not_null()).select(pl.col("implied_vol_bid").mean()).item()
            )
            iv_ask_mean = (
                df.filter(pl.col("implied_vol_ask").is_not_null()).select(pl.col("implied_vol_ask").mean()).item()
            )
            logger.info(f"Mean IV (bid): {iv_bid_mean:.4f}")
            logger.info(f"Mean IV (ask): {iv_ask_mean:.4f}")

        logger.info("=" * 80)

    except Exception as e:
        logger.error(f"ERROR: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
