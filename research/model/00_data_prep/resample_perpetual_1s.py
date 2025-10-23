#!/usr/bin/env python3
"""
Resample BTC perpetual trade data to 1-second VWAP bars.

Input: btc_perpetual_1s_2023_2025.parquet (trade-by-trade data)
Output: btc_perpetual_1s_resampled.parquet (1-second bars with VWAP)

The input file contains individual trades with microsecond timestamps.
We need to aggregate these to 1-second intervals using Volume-Weighted Average Price (VWAP).

VWAP formula: sum(price × amount) / sum(amount) for each 1-second bin
"""

import logging
from pathlib import Path

import polars as pl

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Constants
INPUT_FILE = Path(
    "/Users/lgierhake/Documents/ETH/BT/research/tardis/data/consolidated/btc_perpetual_1s_2023_2025.parquet"
)
OUTPUT_FILE = Path("/Users/lgierhake/Documents/ETH/BT/research/model/results/btc_perpetual_1s_resampled.parquet")
MICROSECONDS_PER_SECOND = 1_000_000


def resample_to_1s_vwap(input_path: Path, output_path: Path) -> None:
    """
    Resample trade data to 1-second VWAP bars.

    Strategy:
    1. Read trade data (timestamp in microseconds)
    2. Convert timestamp to seconds (integer division)
    3. Group by timestamp_seconds
    4. Calculate VWAP: sum(price * amount) / sum(amount)
    5. Also track: total_volume, num_trades, first/last price, high/low
    6. Fill gaps with forward-fill (carry last known price)
    7. Write to parquet

    Args:
        input_path: Path to trade data parquet
        output_path: Path to output resampled parquet
    """
    logger.info(f"Reading perpetual trade data from {input_path}")
    logger.info("This may take a moment - loading 13.5M trades...")

    # Read trade data
    df = pl.read_parquet(input_path)

    logger.info(f"Loaded {len(df):,} trades")
    logger.info(f"Date range: {df['timestamp'].min()} to {df['timestamp'].max()} (microseconds)")

    # Convert timestamp from microseconds to seconds
    logger.info("Converting timestamps to seconds...")
    df = df.with_columns([(pl.col("timestamp") // MICROSECONDS_PER_SECOND).alias("timestamp_seconds")])

    logger.info("Resampling to 1-second VWAP bars...")

    # Calculate VWAP and other metrics per second
    # VWAP = sum(price * amount) / sum(amount)
    df_resampled = (
        df.group_by("timestamp_seconds")
        .agg(
            [
                # VWAP calculation
                (pl.col("price") * pl.col("amount")).sum().alias("price_times_volume"),
                pl.col("amount").sum().alias("total_volume"),
                # Additional price metrics
                pl.col("price").first().alias("open"),
                pl.col("price").max().alias("high"),
                pl.col("price").min().alias("low"),
                pl.col("price").last().alias("close"),
                # Trade count
                pl.len().alias("num_trades"),
            ]
        )
        .with_columns(
            [
                # Calculate VWAP
                (pl.col("price_times_volume") / pl.col("total_volume")).alias("vwap")
            ]
        )
        .select(
            [
                "timestamp_seconds",
                "vwap",
                "open",
                "high",
                "low",
                "close",
                "total_volume",
                "num_trades",
            ]
        )
        .sort("timestamp_seconds")
    )

    logger.info(f"Resampled to {len(df_resampled):,} 1-second bars")

    # Check for gaps
    min_ts_series = df_resampled.select(pl.col("timestamp_seconds").min())
    max_ts_series = df_resampled.select(pl.col("timestamp_seconds").max())

    if len(min_ts_series) == 0 or len(max_ts_series) == 0:
        logger.error("Empty dataframe - cannot determine timestamp range")
        raise ValueError("No data in resampled dataframe")

    min_ts: int = int(min_ts_series.item())
    max_ts: int = int(max_ts_series.item())
    expected_rows: int = max_ts - min_ts + 1
    actual_rows: int = len(df_resampled)
    missing_seconds: int = expected_rows - actual_rows

    logger.info(f"Expected {expected_rows:,} seconds from {min_ts} to {max_ts}")
    logger.info(f"Found {actual_rows:,} seconds with trades")
    logger.info(f"Missing {missing_seconds:,} seconds ({missing_seconds / expected_rows * 100:.2f}%)")

    if missing_seconds > 0:
        logger.info("Filling gaps with forward-fill (carry last known price)...")

        # Create complete timestamp range
        all_seconds = pl.DataFrame({"timestamp_seconds": pl.int_range(min_ts, max_ts + 1, eager=True)})

        # Left join to preserve all seconds
        df_complete = all_seconds.join(df_resampled, on="timestamp_seconds", how="left")

        # Forward-fill VWAP and OHLC (carry last known values)
        df_complete = df_complete.with_columns(
            [
                pl.col("vwap").forward_fill().alias("vwap"),
                pl.col("open").forward_fill().alias("open"),
                pl.col("high").forward_fill().alias("high"),
                pl.col("low").forward_fill().alias("low"),
                pl.col("close").forward_fill().alias("close"),
            ]
        )

        # Fill volume/trades with 0 for missing seconds
        df_complete = df_complete.with_columns(
            [
                pl.col("total_volume").fill_null(0.0),
                pl.col("num_trades").fill_null(0),
            ]
        )

        df_resampled = df_complete

        logger.info(f"After gap-filling: {len(df_resampled):,} seconds")

    # Add metadata columns
    df_resampled = df_resampled.with_columns(
        [
            pl.lit("deribit").alias("exchange"),
            pl.lit("BTC-PERPETUAL").alias("symbol"),
        ]
    ).select(
        [
            "exchange",
            "symbol",
            "timestamp_seconds",
            "vwap",
            "open",
            "high",
            "low",
            "close",
            "total_volume",
            "num_trades",
        ]
    )

    # Validate output
    logger.info("Validating output...")
    assert df_resampled["timestamp_seconds"].is_sorted(), "Timestamps not sorted!"
    assert df_resampled["vwap"].null_count() == 0, "VWAP contains nulls!"

    logger.info("✅ Validation passed")

    # Write output
    logger.info(f"Writing resampled data to {output_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    df_resampled.write_parquet(
        output_path,
        compression="snappy",
        statistics=True,
    )

    # Summary statistics
    logger.info("\n" + "=" * 60)
    logger.info("SUMMARY")
    logger.info("=" * 60)
    logger.info(f"Input trades: {len(df):,}")
    logger.info(f"Output 1s bars: {len(df_resampled):,}")
    logger.info(f"Output file size: {output_path.stat().st_size / 1024 / 1024:.2f} MB")
    logger.info(f"Date range: {df_resampled['timestamp_seconds'].min()} to {df_resampled['timestamp_seconds'].max()}")
    logger.info(f"Average trades per second: {df_resampled['num_trades'].mean():.2f}")
    logger.info(f"Max trades per second: {df_resampled['num_trades'].max():,}")
    logger.info(f"Seconds with 0 trades: {(df_resampled['num_trades'] == 0).sum():,}")
    logger.info("=" * 60)

    # Show sample
    logger.info("\nFirst 5 rows:")
    print(df_resampled.head(5))

    logger.info("\n✅ Resampling complete!")


def main() -> None:
    """Main entry point."""
    logger.info("BTC Perpetual Data Resampling")
    logger.info("Converting trade data to 1-second VWAP bars")

    try:
        resample_to_1s_vwap(INPUT_FILE, OUTPUT_FILE)
    except Exception as e:
        logger.error(f"Error during resampling: {e}")
        raise


if __name__ == "__main__":
    main()
