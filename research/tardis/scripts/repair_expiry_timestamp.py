#!/usr/bin/env python3
"""
Repair script to add missing expiry_timestamp column to IV-enriched options data.

The IV calculation script accidentally dropped the expiry_timestamp column,
which is required by the V4 backtest. This script adds it back by parsing
the expiry_str column.
"""

import polars as pl
import logging
from pathlib import Path

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def repair_expiry_timestamp():
    """Add expiry_timestamp column back to the IV-enriched options file."""

    # File path
    data_dir = Path("/home/ubuntu/Polymarket/research/tardis/data/consolidated")
    input_file = data_dir / "btc_options_atm_shortdated_with_iv_2023_2025.parquet"
    backup_file = data_dir / "btc_options_atm_shortdated_with_iv_2023_2025.parquet.before_repair"

    logger.info(f"Loading data from {input_file}")

    # Load the parquet file
    df = pl.read_parquet(input_file)
    logger.info(f"Loaded {len(df):,} rows")

    # Check if expiry_timestamp already exists
    if "expiry_timestamp" in df.columns:
        logger.warning("Column 'expiry_timestamp' already exists. No repair needed.")
        return

    # Check that expiry_str exists
    if "expiry_str" not in df.columns:
        logger.error("Column 'expiry_str' not found. Cannot repair.")
        raise ValueError("Missing required column: expiry_str")

    logger.info("Creating backup before repair...")
    df.write_parquet(backup_file)
    logger.info(f"Backup saved to {backup_file}")

    logger.info("Adding expiry_timestamp column by parsing expiry_str...")

    # Convert expiry_str (format: '29DEC23') to expiry_timestamp
    # The expiry_str is in format DDMMMYY (e.g., '29DEC23')
    df = df.with_columns([
        pl.col("expiry_str")
        .str.strptime(pl.Datetime, "%d%b%y", strict=False)
        .dt.epoch("s")
        .alias("expiry_timestamp")
    ])

    # Verify the column was added
    if "expiry_timestamp" not in df.columns:
        logger.error("Failed to add expiry_timestamp column")
        raise RuntimeError("Column creation failed")

    # Check for nulls in the new column
    null_count = df["expiry_timestamp"].null_count()
    if null_count > 0:
        logger.warning(f"Found {null_count:,} null values in expiry_timestamp")

    # Get some statistics
    logger.info("Column statistics:")
    logger.info(f"  Total rows: {len(df):,}")
    logger.info(f"  Unique expiry_timestamps: {df['expiry_timestamp'].n_unique():,}")
    logger.info(f"  Min expiry_timestamp: {df['expiry_timestamp'].min()}")
    logger.info(f"  Max expiry_timestamp: {df['expiry_timestamp'].max()}")

    # Save the repaired file
    logger.info(f"Saving repaired data to {input_file}")
    df.write_parquet(input_file)

    # Verify file was written
    file_size = input_file.stat().st_size / (1024**3)  # Size in GB
    logger.info(f"Successfully saved {file_size:.2f} GB file")

    # Show the columns in the repaired file
    logger.info(f"Final columns: {df.columns}")

    logger.info("✅ Repair completed successfully!")
    logger.info(f"The file now contains the 'expiry_timestamp' column required by the V4 backtest.")

    return df

if __name__ == "__main__":
    try:
        repair_expiry_timestamp()
    except Exception as e:
        logger.error(f"Repair failed: {e}")
        raise