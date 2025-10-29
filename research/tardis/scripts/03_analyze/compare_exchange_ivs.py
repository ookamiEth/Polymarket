#!/usr/bin/env python3
"""
Compare our calculated IVs against Deribit's exchange-provided IVs.

This validates that our IV calculation methodology is correct by comparing
against the ground truth (exchange IVs) on the same options data.
"""

import logging

import numpy as np
import polars as pl
from py_vollib_vectorized import vectorized_implied_volatility

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Constants
RISK_FREE_RATE = 0.0412  # 4.12% annual rate
OPTIONS_CHAIN_FILE = "datasets_deribit_options/deribit_options_chain_2025-10-01_OPTIONS.parquet"
SPOT_DATA_FILE = "deribit_btc_perpetual_1s.parquet"
TARGET_DATE = "2025-09-30"  # Date within both datasets


def load_spot_data(spot_file: str, target_date: str) -> pl.DataFrame:
    """
    Load spot price data for given date.

    Args:
        spot_file: Path to spot price parquet file
        target_date: Date string in YYYY-MM-DD format

    Returns:
        DataFrame with columns: time (Datetime UTC), price (Float64)
    """
    logger.info(f"Loading spot data from {spot_file} for {target_date}...")

    df = pl.read_parquet(spot_file)

    # Convert timestamp to datetime with UTC timezone
    df = df.with_columns([pl.from_epoch(pl.col("timestamp"), time_unit="us").dt.replace_time_zone("UTC").alias("time")])

    # Filter to target date
    df = df.filter(pl.col("time").dt.date() == pl.date(*[int(x) for x in target_date.split("-")]))

    # Select relevant columns
    df = df.select(["time", "price"])

    logger.info(f"  Loaded {len(df):,} spot price records")
    return df


def load_options_with_exchange_ivs(options_file: str, target_date: str) -> pl.DataFrame:
    """
    Load options chain data with exchange-provided IVs.

    Args:
        options_file: Path to options chain parquet file
        target_date: Date string in YYYY-MM-DD format

    Returns:
        DataFrame with options data including exchange IVs
    """
    logger.info(f"Loading options chain from {options_file} for {target_date}...")

    df = pl.scan_parquet(options_file)

    # Filter to target date and BTC options only
    df = df.filter(
        (
            pl.from_epoch(pl.col("timestamp"), time_unit="us").dt.date()
            == pl.date(*[int(x) for x in target_date.split("-")])
        )
        & (pl.col("underlying") == "BTC")
    )

    # Convert timestamp to datetime with UTC
    df = df.with_columns([pl.from_epoch(pl.col("timestamp"), time_unit="us").dt.replace_time_zone("UTC").alias("time")])

    # Filter to options with valid exchange IVs and prices
    df = df.filter(
        pl.col("bid_iv").is_not_null()
        & pl.col("ask_iv").is_not_null()
        & pl.col("bid_price").is_not_null()
        & pl.col("ask_price").is_not_null()
        & (pl.col("bid_price") > 0)
        & (pl.col("ask_price") > 0)
        & (pl.col("days_to_expiry") > 0)
    )

    df = df.collect()

    logger.info(f"  Loaded {len(df):,} options with valid exchange IVs")
    return df


def calculate_our_ivs(df: pl.DataFrame, spot_df: pl.DataFrame) -> pl.DataFrame:
    """
    Calculate our IVs using the same method as calculate_implied_volatility.py

    Args:
        df: Options data with bid/ask prices
        spot_df: Spot price data

    Returns:
        DataFrame with calculated IVs added
    """
    logger.info("Calculating our IVs...")

    # Join with spot data (asof join to match closest spot price)
    df = df.sort("time")
    spot_df = spot_df.sort("time")

    df = df.join_asof(spot_df, on="time", suffix="_spot")

    # Convert time to expiry (days_to_expiry → years)
    # Using 365.25 to account for leap years (crypto trades 24/7/365)
    df = df.with_columns([(pl.col("days_to_expiry") / 365.25).alias("time_to_expiry")])

    # Parse option type from type column (already has "call"/"put")
    df = df.with_columns(
        [pl.when(pl.col("type").str.to_lowercase() == "call").then(pl.lit("c")).otherwise(pl.lit("p")).alias("flag")]
    )

    # CRITICAL: Convert BTC-denominated option prices to USD
    # Deribit quotes option prices in BTC, but strikes/spots are in USD
    df = df.with_columns(
        [
            (pl.col("bid_price") * pl.col("price")).alias("bid_price_usd"),
            (pl.col("ask_price") * pl.col("price")).alias("ask_price_usd"),
        ]
    )

    # Filter out invalid data
    df = df.filter(
        pl.col("time_to_expiry").is_not_null()
        & (pl.col("time_to_expiry") > 0)
        & pl.col("price").is_not_null()
        & (pl.col("price") > 0)
    )

    logger.info(f"  {len(df):,} options ready for IV calculation")

    # Vectorized IV calculation
    logger.info("  Running vectorized IV calculation (bid)...")

    try:
        iv_bid = vectorized_implied_volatility(
            price=df["bid_price_usd"].to_numpy(),
            S=df["price"].to_numpy(),
            K=df["strike_price"].to_numpy(),
            t=df["time_to_expiry"].to_numpy(),
            r=np.full(len(df), RISK_FREE_RATE),
            flag=df["flag"].to_numpy(),
            model="black_scholes",
            return_as="numpy",
        )

        df = df.with_columns([pl.Series("calc_iv_bid", iv_bid)])

    except Exception as e:
        logger.error(f"IV calculation failed: {e}")
        raise

    logger.info("  Running vectorized IV calculation (ask)...")

    try:
        iv_ask = vectorized_implied_volatility(
            price=df["ask_price_usd"].to_numpy(),
            S=df["price"].to_numpy(),
            K=df["strike_price"].to_numpy(),
            t=df["time_to_expiry"].to_numpy(),
            r=np.full(len(df), RISK_FREE_RATE),
            flag=df["flag"].to_numpy(),
            model="black_scholes",
            return_as="numpy",
        )

        df = df.with_columns([pl.Series("calc_iv_ask", iv_ask)])

    except Exception as e:
        logger.error(f"IV calculation failed: {e}")
        raise

    # Filter to successful IV calculations
    df = df.filter(
        pl.col("calc_iv_bid").is_not_null()
        & pl.col("calc_iv_ask").is_not_null()
        & pl.col("calc_iv_bid").is_finite()
        & pl.col("calc_iv_ask").is_finite()
        & (pl.col("calc_iv_bid") > 0)
        & (pl.col("calc_iv_ask") > 0)
    )

    logger.info(f"  Successfully calculated IVs for {len(df):,} options")

    return df


def generate_comparison_metrics(df: pl.DataFrame) -> None:
    """
    Generate validation metrics comparing our IVs vs exchange IVs.

    Args:
        df: DataFrame with both calculated and exchange IVs
    """
    logger.info("\n" + "=" * 80)
    logger.info("IV COMPARISON: Calculated vs Exchange")
    logger.info("=" * 80)

    # Convert IVs to percentages for display
    df = df.with_columns(
        [
            (pl.col("calc_iv_bid") * 100).alias("calc_iv_bid_pct"),
            (pl.col("calc_iv_ask") * 100).alias("calc_iv_ask_pct"),
            (pl.col("bid_iv")).alias("exch_iv_bid_pct"),  # Already in percentage
            (pl.col("ask_iv")).alias("exch_iv_ask_pct"),  # Already in percentage
        ]
    )

    # Calculate differences
    df = df.with_columns(
        [
            (pl.col("calc_iv_bid_pct") - pl.col("exch_iv_bid_pct")).alias("diff_bid"),
            (pl.col("calc_iv_ask_pct") - pl.col("exch_iv_ask_pct")).alias("diff_ask"),
        ]
    )

    # Summary statistics
    logger.info("\nBID IV Statistics:")
    logger.info(f"  Total options: {len(df):,}")
    logger.info(
        f"  Calculated IV:  mean={df['calc_iv_bid_pct'].mean():.2f}%, "
        f"median={df['calc_iv_bid_pct'].median():.2f}%, "
        f"std={df['calc_iv_bid_pct'].std():.2f}%"
    )
    logger.info(
        f"  Exchange IV:    mean={df['exch_iv_bid_pct'].mean():.2f}%, "
        f"median={df['exch_iv_bid_pct'].median():.2f}%, "
        f"std={df['exch_iv_bid_pct'].std():.2f}%"
    )
    logger.info(
        f"  Difference:     mean={df['diff_bid'].mean():.2f}%, "
        f"median={df['diff_bid'].median():.2f}%, "
        f"MAE={df['diff_bid'].abs().mean():.2f}%"
    )

    # Correlation
    corr_bid = df.select([pl.corr("calc_iv_bid_pct", "exch_iv_bid_pct").alias("corr")]).item(0, 0)
    logger.info(f"  Correlation: {corr_bid:.4f}")

    logger.info("\nASK IV Statistics:")
    logger.info(
        f"  Calculated IV:  mean={df['calc_iv_ask_pct'].mean():.2f}%, "
        f"median={df['calc_iv_ask_pct'].median():.2f}%, "
        f"std={df['calc_iv_ask_pct'].std():.2f}%"
    )
    logger.info(
        f"  Exchange IV:    mean={df['exch_iv_ask_pct'].mean():.2f}%, "
        f"median={df['exch_iv_ask_pct'].median():.2f}%, "
        f"std={df['exch_iv_ask_pct'].std():.2f}%"
    )
    logger.info(
        f"  Difference:     mean={df['diff_ask'].mean():.2f}%, "
        f"median={df['diff_ask'].median():.2f}%, "
        f"MAE={df['diff_ask'].abs().mean():.2f}%"
    )

    corr_ask = df.select([pl.corr("calc_iv_ask_pct", "exch_iv_ask_pct").alias("corr")]).item(0, 0)
    logger.info(f"  Correlation: {corr_ask:.4f}")

    # Show sample comparisons
    logger.info("\n" + "=" * 80)
    logger.info("SAMPLE COMPARISONS (10 examples)")
    logger.info("=" * 80)

    sample = df.select(
        [
            "symbol",
            "strike_price",
            "flag",
            "days_to_expiry",
            "calc_iv_bid_pct",
            "exch_iv_bid_pct",
            "diff_bid",
        ]
    ).head(10)

    print()
    print(sample)
    print()

    # Distribution of differences
    logger.info("=" * 80)
    logger.info("DIFFERENCE DISTRIBUTION (Calculated - Exchange)")
    logger.info("=" * 80)

    bins = [-float("inf"), -5, -2, -1, 1, 2, 5, float("inf")]
    labels = ["< -5%", "-5% to -2%", "-2% to -1%", "-1% to +1%", "+1% to +2%", "+2% to +5%", "> +5%"]

    for i in range(len(labels)):
        if i == 0:
            count = df.filter(pl.col("diff_bid") < bins[i + 1]).height
        elif i == len(labels) - 1:
            count = df.filter(pl.col("diff_bid") >= bins[i]).height
        else:
            count = df.filter((pl.col("diff_bid") >= bins[i]) & (pl.col("diff_bid") < bins[i + 1])).height

        pct = count / len(df) * 100
        logger.info(f"  {labels[i]:<15}: {count:>7,} ({pct:>5.1f}%)")

    # Validation
    logger.info("\n" + "=" * 80)
    logger.info("VALIDATION RESULT")
    logger.info("=" * 80)

    mae_bid = df["diff_bid"].abs().mean()
    mae_ask = df["diff_ask"].abs().mean()

    if mae_bid <= 5.0 and mae_ask <= 5.0 and corr_bid >= 0.90 and corr_ask >= 0.90:
        logger.info("✓ SUCCESS: Our IV calculations closely match exchange IVs!")
        logger.info(f"  MAE (bid): {mae_bid:.2f}%")
        logger.info(f"  MAE (ask): {mae_ask:.2f}%")
        logger.info(f"  Correlation (bid): {corr_bid:.4f}")
        logger.info(f"  Correlation (ask): {corr_ask:.4f}")
    else:
        logger.warning("✗ WARNING: Significant discrepancy detected")
        logger.warning(f"  MAE (bid): {mae_bid:.2f}% (expected: <5%)")
        logger.warning(f"  MAE (ask): {mae_ask:.2f}% (expected: <5%)")
        logger.warning(f"  Correlation (bid): {corr_bid:.4f} (expected: >0.90)")
        logger.warning(f"  Correlation (ask): {corr_ask:.4f} (expected: >0.90)")


def main() -> None:
    """Main execution."""
    logger.info("Starting IV comparison: Calculated vs Exchange")
    logger.info(f"Target date: {TARGET_DATE}")

    # Load data
    spot_df = load_spot_data(SPOT_DATA_FILE, TARGET_DATE)
    options_df = load_options_with_exchange_ivs(OPTIONS_CHAIN_FILE, TARGET_DATE)

    # Calculate our IVs
    results_df = calculate_our_ivs(options_df, spot_df)

    # Generate comparison metrics
    generate_comparison_metrics(results_df)

    # Save results
    output_file = f"iv_comparison_{TARGET_DATE}.parquet"
    results_df.write_parquet(output_file)
    logger.info(f"\nResults saved to: {output_file}")


if __name__ == "__main__":
    main()
