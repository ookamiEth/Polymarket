#!/usr/bin/env python3

import argparse
import logging
import os
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np
import polars as pl

# Constants
DEFAULT_RISK_FREE_RATE = 0.0412
DEFAULT_SPOT_COLUMN = "mid_px"
DEFAULT_OUTPUT_SUFFIX = "_with_iv"
DERIBIT_EXPIRY_HOUR_UTC = 8
SECONDS_PER_YEAR = 365.25 * 24 * 3600
MERGE_TOLERANCE_SECONDS = 60

# Quality filter thresholds
MIN_OPTION_PRICE_USD = 10.0  # Minimum $10 for bid/ask to ensure liquidity
MAX_BID_ASK_SPREAD_PCT = 0.50  # Max 50% spread (ask/bid - 1) for data quality
MIN_TIME_TO_EXPIRY_HOURS = 1.0  # Minimum 1 hour to avoid numerical instability
MONEYNESS_MIN = 0.50  # Minimum strike/spot ratio
MONEYNESS_MAX = 2.00  # Maximum strike/spot ratio

logger = logging.getLogger(__name__)


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
    # Drop existing columns from previous runs to allow re-processing
    cols_to_drop = [
        "time",
        "spot_price",
        "time_to_expiry",
        "implied_vol_bid",
        "implied_vol_ask",
        "iv_calc_status",
        "time_right",
        "spot_price_right",
    ]
    for col in cols_to_drop:
        if col in options_df.columns:
            options_df = options_df.drop(col)

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
    # NOTE: map_elements required here - Polars lacks native "1JAN25" date parser
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
            ((pl.col("expiry_datetime") - pl.col("timestamp_dt")).dt.total_seconds() / SECONDS_PER_YEAR).alias(
                "time_to_expiry"
            )
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


def apply_crypto_quality_filters(df: pl.DataFrame) -> pl.DataFrame:
    """
    Apply quality filters for liquid, tradable crypto options.

    Filters (USD-based for consistency):
    - Minimum option price: >= $10 USD for bid/ask (ensures liquidity)
    - Bid-ask spread: <= 50% (ask/bid - 1, filters wide/stale quotes)
    - Moneyness range: 0.50-2.00 strike/spot (includes 2x OTM "lottery tickets")
    - Time to expiry: >= 1 hour (avoids numerical instability)

    Note: Arbitrage validation (below-intrinsic) handled by py_vollib during IV calculation.

    Args:
        df: DataFrame with bid_price, ask_price, strike_price, spot_price, time_to_expiry (BTC)

    Returns:
        Filtered DataFrame with liquid, tradable options
    """
    logger.info("Applying crypto quality filters...")

    rows_before = len(df)

    # Convert BTC prices to USD for filtering
    df_filtered = df.with_columns(
        [
            (pl.col("bid_price") * pl.col("spot_price")).alias("bid_price_usd"),
            (pl.col("ask_price") * pl.col("spot_price")).alias("ask_price_usd"),
        ]
    ).filter(
        # USD price filters (meaningful across all spot price levels)
        (pl.col("bid_price_usd") >= MIN_OPTION_PRICE_USD)
        & (pl.col("ask_price_usd") >= MIN_OPTION_PRICE_USD)
        # Bid-ask spread quality: (ask - bid) / bid <= 50%
        & ((pl.col("ask_price") / pl.col("bid_price") - 1) <= MAX_BID_ASK_SPREAD_PCT)
        # Moneyness: 0.50 to 2.00 (crypto traders love 2x OTM calls)
        & (pl.col("strike_price") / pl.col("spot_price")).is_between(MONEYNESS_MIN, MONEYNESS_MAX)
        # Time floor: avoid near-expiry numerical issues
        & (pl.col("time_to_expiry") >= MIN_TIME_TO_EXPIRY_HOURS / 8760)
    )

    # Drop temporary USD columns
    df_filtered = df_filtered.drop(["bid_price_usd", "ask_price_usd"])

    rows_after = len(df_filtered)
    filtered_count = rows_before - rows_after
    filtered_pct = (filtered_count / rows_before * 100) if rows_before > 0 else 0

    logger.info(f"Filtered {filtered_count:,} rows ({filtered_pct:.1f}%)")
    logger.info(f"Remaining: {rows_after:,} liquid, tradable options")

    # Log filter breakdown if verbose
    if logger.isEnabledFor(logging.DEBUG):
        with_columns = df.with_columns(
            [
                (pl.col("bid_price") * pl.col("spot_price")).alias("bid_usd"),
                (pl.col("ask_price") * pl.col("spot_price")).alias("ask_usd"),
            ]
        )
        price_fails = with_columns.filter(
            (pl.col("bid_usd") < MIN_OPTION_PRICE_USD) | (pl.col("ask_usd") < MIN_OPTION_PRICE_USD)
        ).shape[0]
        spread_fails = df.filter((pl.col("ask_price") / pl.col("bid_price") - 1) > MAX_BID_ASK_SPREAD_PCT).shape[0]
        moneyness_fails = df.filter(
            ~(pl.col("strike_price") / pl.col("spot_price")).is_between(MONEYNESS_MIN, MONEYNESS_MAX)
        ).shape[0]
        time_fails = df.filter(pl.col("time_to_expiry") < MIN_TIME_TO_EXPIRY_HOURS / 8760).shape[0]

        logger.debug(f"  Price < ${MIN_OPTION_PRICE_USD}: {price_fails:,}")
        logger.debug(f"  Spread > {MAX_BID_ASK_SPREAD_PCT:.0%}: {spread_fails:,}")
        logger.debug(f"  Moneyness outside [{MONEYNESS_MIN}, {MONEYNESS_MAX}]: {moneyness_fails:,}")
        logger.debug(f"  Time < {MIN_TIME_TO_EXPIRY_HOURS}h: {time_fails:,}")

    return df_filtered


def calculate_ivs_batch(
    df: pl.DataFrame,
    risk_free_rate: float,
) -> pl.DataFrame:
    """
    Calculate implied volatilities using vectorized Jaeckel's method.

    Uses py_vollib_vectorized with Jaeckel's "Let's Be Rational" algorithm
    for 25-50x speedup vs row-by-row scipy.brentq. Achieves machine precision
    (10⁻¹⁵) in exactly 2 iterations using fourth-order Householder convergence.

    Args:
        df: DataFrame with required columns (bid_price, ask_price, spot_price, strike_price, time_to_expiry, type)
        risk_free_rate: Annual risk-free rate (e.g., 0.05 for 5%)

    Returns:
        DataFrame with implied_vol_bid, implied_vol_ask, iv_calc_status columns added
    """
    from py_vollib_vectorized import vectorized_implied_volatility  # type: ignore

    logger.info(f"Calculating implied volatilities for {len(df):,} rows (vectorized Jaeckel's method)...")

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

    if len(df_valid) == 0:
        logger.warning("No valid rows to process")
        return df_valid.with_columns(
            [
                pl.Series("implied_vol_bid", [], dtype=pl.Float64),
                pl.Series("implied_vol_ask", [], dtype=pl.Float64),
                pl.Series("iv_calc_status", [], dtype=pl.Utf8),
            ]
        )

    # Prepare numpy arrays for vectorized calculation
    prices_bid = df_valid["bid_price"].to_numpy()
    prices_ask = df_valid["ask_price"].to_numpy()
    S = df_valid["spot_price"].to_numpy()  # noqa: N806
    K = df_valid["strike_price"].to_numpy()  # noqa: N806
    t = df_valid["time_to_expiry"].to_numpy()
    r = np.full(len(df_valid), risk_free_rate)

    # Convert "call"/"put" to "c"/"p" for py_vollib
    flag = df_valid["type"].str.replace("call", "c").str.replace("put", "p").to_numpy()

    # CRITICAL: Deribit quotes option prices in BTC, but strikes/spots are in USD
    # Convert option prices from BTC to USD before IV calculation
    prices_bid_usd = prices_bid * S
    prices_ask_usd = prices_ask * S

    logger.info(
        f"Sample (first row): bid={prices_bid[0]:.6f} BTC (${prices_bid_usd[0]:.2f} USD), "
        f"spot=${S[0]:.0f}, strike=${K[0]:.0f}, T={t[0]:.6f}y ({t[0] * 365:.1f}d), flag={flag[0]}"
    )

    # VECTORIZED IV CALCULATION (ALL rows at once - NO LOOP!)
    logger.info("Computing bid IVs (vectorized)...")
    try:
        iv_bid = vectorized_implied_volatility(
            price=prices_bid_usd, S=S, K=K, t=t, r=r, flag=flag, model="black_scholes", return_as="numpy"
        )
    except Exception as e:
        logger.warning(f"Vectorized bid calculation failed: {e}")
        # Fallback to NaN array
        iv_bid = np.full(len(df_valid), np.nan)

    logger.info("Computing ask IVs (vectorized)...")
    try:
        iv_ask = vectorized_implied_volatility(
            price=prices_ask_usd, S=S, K=K, t=t, r=r, flag=flag, model="black_scholes", return_as="numpy"
        )
    except Exception as e:
        logger.warning(f"Vectorized ask calculation failed: {e}")
        # Fallback to NaN array
        iv_ask = np.full(len(df_valid), np.nan)

    # Create status column (success if both bid and ask IVs are valid)
    status = np.where(
        np.isnan(iv_bid) | np.isnan(iv_ask),
        "failed",  # Either bid or ask failed
        "success",  # Both succeeded
    )

    # Add columns to DataFrame
    df_valid = df_valid.with_columns(
        [
            pl.Series("implied_vol_bid", iv_bid),
            pl.Series("implied_vol_ask", iv_ask),
            pl.Series("iv_calc_status", status),
        ]
    )

    success_count = np.sum(status == "success")
    failed_count = len(status) - success_count
    success_rate = success_count / max(len(df_valid), 1) * 100

    logger.info(f"IV calculation complete: {success_count:,} success, {failed_count:,} failed ({success_rate:.1f}%)")

    return df_valid


def process_options_file(
    input_file: str,
    spot_data_dir: str,
    risk_free_rate: float,
    spot_column: str,
) -> pl.DataFrame:
    """
    Process Deribit options file and calculate implied volatilities.

    Args:
        input_file: Path to Deribit options Parquet file
        spot_data_dir: Directory with Hyperliquid spot data
        risk_free_rate: Annual risk-free rate
        spot_column: Column to use as spot price

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
    logger.info("APPLY CRYPTO QUALITY FILTERS")
    logger.info("=" * 80)

    options_df = apply_crypto_quality_filters(options_df)

    logger.info("=" * 80)
    logger.info("CALCULATE IMPLIED VOLATILITIES")
    logger.info("=" * 80)

    options_df = calculate_ivs_batch(options_df, risk_free_rate)

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
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable debug logging")

    args = parser.parse_args()

    # Configure logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

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

    try:
        # Process file
        df = process_options_file(
            input_file=args.input_file,
            spot_data_dir=args.spot_data_dir,
            risk_free_rate=args.risk_free_rate,
            spot_column=args.spot_column,
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
