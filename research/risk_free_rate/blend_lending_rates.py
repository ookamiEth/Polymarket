#!/usr/bin/env python3
"""
Blend AAVE USDC and USDT lending rates with 7-day moving average.

This script:
- Loads AAVE USDC and USDT lending rate data
- Aligns timestamps (intersection only)
- Blends supply rates using 50/50 simple average
- Calculates 7-day trailing moving average on blended rate
- Saves to parquet file
"""

import logging
from pathlib import Path

import polars as pl

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# File paths
DATA_DIR = Path(__file__).parent / "data"
AAVE_FILE = DATA_DIR / "aave_usdc_rates_2023_2025.parquet"
USDT_FILE = DATA_DIR / "usdt_lending_rates_2023_2025.parquet"
OUTPUT_FILE = DATA_DIR / "blended_lending_rates_2023_2025.parquet"


def load_and_prepare_data() -> tuple[pl.DataFrame, pl.DataFrame]:
    """Load and prepare both datasets for blending.

    Returns:
        Tuple of (aave_df, usdt_df) with standardized date columns
    """
    logger.info("Loading AAVE USDC rates...")
    aave_df = pl.read_parquet(AAVE_FILE)

    logger.info("Loading USDT lending rates...")
    usdt_df = pl.read_parquet(USDT_FILE)

    # Extract date from timestamp for joining
    aave_prepared = aave_df.with_columns([pl.col("timestamp").dt.date().alias("date")])

    # USDT uses "datetime" column, need to extract date
    # Drop the int64 timestamp column and use datetime as timestamp
    usdt_prepared = (
        usdt_df.drop("timestamp")
        .with_columns([pl.col("datetime").dt.date().alias("date")])
        .rename({"datetime": "timestamp"})
    )

    logger.info(
        f"AAVE: {len(aave_prepared):,} records from {aave_prepared['date'].min()} to {aave_prepared['date'].max()}"
    )
    logger.info(
        f"USDT: {len(usdt_prepared):,} records from {usdt_prepared['date'].min()} to {usdt_prepared['date'].max()}"
    )

    return aave_prepared, usdt_prepared


def blend_rates(aave_df: pl.DataFrame, usdt_df: pl.DataFrame) -> pl.DataFrame:
    """Blend supply rates from both datasets using 50/50 average.

    Args:
        aave_df: AAVE USDC data with date column
        usdt_df: USDT data with date column

    Returns:
        DataFrame with blended rates (intersection only)
    """
    logger.info("Blending rates using 50/50 simple average (intersection only)...")

    # Select relevant columns for joining
    aave_selected = aave_df.select(
        [
            pl.col("date"),
            pl.col("timestamp").alias("aave_timestamp"),
            pl.col("supply_rate_apr").alias("aave_supply_apr"),
        ]
    )

    usdt_selected = usdt_df.select(
        [
            pl.col("date"),
            pl.col("timestamp").alias("usdt_timestamp"),
            pl.col("lending_apr").alias("usdt_supply_apr"),
        ]
    )

    # Inner join on date (intersection only)
    blended_df = aave_selected.join(usdt_selected, on="date", how="inner")

    # Calculate 50/50 blended rate
    blended_df = blended_df.with_columns(
        [((pl.col("aave_supply_apr") + pl.col("usdt_supply_apr")) / 2.0).alias("blended_supply_apr")]
    )

    # Use AAVE timestamp as primary timestamp (both should be close on same date)
    blended_df = blended_df.with_columns([pl.col("aave_timestamp").alias("timestamp")])

    logger.info(f"Blended {len(blended_df):,} records (intersection of both datasets)")
    logger.info(f"Date range: {blended_df['date'].min()} to {blended_df['date'].max()}")

    return blended_df


def calculate_moving_average(df: pl.DataFrame) -> pl.DataFrame:
    """Calculate 7-day trailing moving average on blended rate.

    Args:
        df: DataFrame with blended_supply_apr column

    Returns:
        DataFrame with ma7 column added
    """
    logger.info("Calculating 7-day trailing moving average...")

    # Sort by date to ensure correct MA calculation
    df_sorted = df.sort("date")

    # Calculate 7-day trailing MA using rolling window
    df_with_ma = df_sorted.with_columns(
        [pl.col("blended_supply_apr").rolling_mean(window_size=7).alias("blended_supply_apr_ma7")]
    )

    return df_with_ma


def save_output(df: pl.DataFrame, output_file: Path) -> None:
    """Save blended rates to parquet file.

    Args:
        df: DataFrame with blended rates and MA
        output_file: Output parquet file path
    """
    # Select final columns in logical order
    output_df = df.select(
        [
            "date",
            "timestamp",
            "aave_supply_apr",
            "usdt_supply_apr",
            "blended_supply_apr",
            "blended_supply_apr_ma7",
        ]
    )

    logger.info(f"Writing output to {output_file}...")
    output_df.write_parquet(output_file, compression="snappy", statistics=True)

    logger.info(f"✅ Saved {len(output_df):,} records to {output_file}")


def print_summary_statistics(df: pl.DataFrame) -> None:
    """Print summary statistics for blended rates.

    Args:
        df: DataFrame with blended rates
    """
    logger.info("\n" + "=" * 80)
    logger.info("SUMMARY STATISTICS")
    logger.info("=" * 80)

    stats = df.select(
        [
            pl.col("aave_supply_apr").mean().alias("mean_aave"),
            pl.col("usdt_supply_apr").mean().alias("mean_usdt"),
            pl.col("blended_supply_apr").mean().alias("mean_blended"),
            pl.col("blended_supply_apr").median().alias("median_blended"),
            pl.col("blended_supply_apr").std().alias("std_blended"),
            pl.col("blended_supply_apr_ma7").mean().alias("mean_ma7"),
            pl.col("blended_supply_apr_ma7").std().alias("std_ma7"),
        ]
    )

    logger.info("\nOriginal rates (mean):")
    logger.info(f"  AAVE USDC: {stats['mean_aave'][0]:.2f}%")
    logger.info(f"  USDT:      {stats['mean_usdt'][0]:.2f}%")

    logger.info("\nBlended rate (raw):")
    logger.info(
        f"  Mean:   {stats['mean_blended'][0]:.2f}% (median: {stats['median_blended'][0]:.2f}%, std: {stats['std_blended'][0]:.2f}%)"
    )

    logger.info("\nBlended rate (7-day MA):")
    logger.info(f"  Mean:   {stats['mean_ma7'][0]:.2f}% (std: {stats['std_ma7'][0]:.2f}%)")

    # Smoothing effect
    volatility_reduction = ((stats["std_blended"][0] - stats["std_ma7"][0]) / stats["std_blended"][0]) * 100
    logger.info(f"\nSmoothing effect: {volatility_reduction:.1f}% reduction in volatility")

    logger.info("=" * 80)


def main() -> None:
    """Main entry point."""
    logger.info("Starting lending rate blending process...")

    # Load and prepare data
    aave_df, usdt_df = load_and_prepare_data()

    # Blend rates (intersection only)
    blended_df = blend_rates(aave_df, usdt_df)

    # Calculate 7-day moving average
    blended_with_ma = calculate_moving_average(blended_df)

    # Save to parquet
    save_output(blended_with_ma, OUTPUT_FILE)

    # Print summary statistics
    print_summary_statistics(blended_with_ma)

    logger.info("\n✅ Blending complete!")


if __name__ == "__main__":
    main()
