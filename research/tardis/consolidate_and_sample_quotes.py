#!/usr/bin/env python3

import argparse
import logging
import os
import sys
import time
from pathlib import Path
from typing import Optional

import polars as pl

logger = logging.getLogger(__name__)


def consolidate_csv_files(
    input_dir: str,
    output_path: str,
) -> pl.DataFrame:
    """Consolidate all CSV.gz files into single sorted Parquet.

    Args:
        input_dir: Directory containing CSV.gz files
        output_path: Output path for consolidated Parquet

    Returns:
        Consolidated DataFrame
    """
    logger.info("=" * 80)
    logger.info("STAGE 1: CONSOLIDATING CSV.GZ FILES")
    logger.info("=" * 80)

    # Find all CSV.gz files
    csv_files = sorted(Path(input_dir).glob("*.csv.gz"))
    logger.info(f"Found {len(csv_files)} CSV.gz files")

    if len(csv_files) == 0:
        logger.error(f"No CSV.gz files found in {input_dir}")
        sys.exit(1)

    # Schema overrides to handle Tardis data type inconsistencies
    # Matches download_and_filter_options_quotes.py:92-98
    schema_overrides = {
        "strike_price": pl.Float64,
        "bid_price": pl.Float64,
        "bid_amount": pl.Float64,
        "ask_price": pl.Float64,
        "ask_amount": pl.Float64,
    }

    # Read all files with lazy evaluation
    logger.info("Reading all CSV.gz files (lazy)...")
    lazy_dfs = []
    for csv_file in csv_files:
        try:
            lazy_df = pl.scan_csv(csv_file, schema_overrides=schema_overrides)
            lazy_dfs.append(lazy_df)
        except Exception as e:
            logger.warning(f"Failed to read {csv_file.name}: {e}")

    if len(lazy_dfs) == 0:
        logger.error("No valid CSV files could be read")
        sys.exit(1)

    logger.info(f"Successfully scanned {len(lazy_dfs)} files")

    # Concatenate all lazy frames
    logger.info("Concatenating all files...")
    combined = pl.concat(lazy_dfs)

    # Sort by timestamp and collect
    logger.info("Sorting by timestamp and collecting (this may take time)...")
    start_time = time.time()
    df = combined.sort("timestamp").collect()
    elapsed = time.time() - start_time

    original_count = len(df)
    logger.info(f"Collected {original_count:,} rows in {elapsed:.1f}s")

    # Deduplicate exact duplicates
    logger.info("Deduplicating...")
    df = df.unique()
    dedup_count = original_count - len(df)
    logger.info(f"Removed {dedup_count:,} duplicate rows ({len(df):,} remaining)")

    # Write consolidated Parquet
    logger.info(f"Writing consolidated Parquet to {output_path}...")
    df.write_parquet(output_path, compression="snappy")

    file_size_mb = os.path.getsize(output_path) / (1024 * 1024)
    logger.info(f"Consolidated file size: {file_size_mb:.1f} MB")
    logger.info(f"Timestamp range: {df['timestamp'].min()} to {df['timestamp'].max()}")

    return df


def sample_to_1s(
    df: pl.DataFrame,
) -> pl.DataFrame:
    """Aggregate data to 1-second granularity.

    Args:
        df: Consolidated DataFrame with microsecond timestamps

    Returns:
        DataFrame aggregated to 1-second granularity
    """
    logger.info("=" * 80)
    logger.info("STAGE 2: SAMPLING TO 1-SECOND GRANULARITY")
    logger.info("=" * 80)

    original_count = len(df)
    logger.info(f"Starting with {original_count:,} rows")

    # Add timestamp_seconds column
    logger.info("Converting timestamps to seconds...")
    df = df.with_columns([(pl.col("timestamp") // 1_000_000).cast(pl.Int64).alias("timestamp_seconds")])

    # Group by (second, symbol) and aggregate
    logger.info("Aggregating to 1-second granularity...")
    df_1s = df.group_by(["timestamp_seconds", "symbol"]).agg(
        [
            pl.col("exchange").first(),
            pl.col("type").first(),
            pl.col("strike_price").first(),
            pl.col("underlying").first(),
            pl.col("expiry_str").first(),
            pl.col("bid_price").filter(pl.col("bid_price").is_not_null()).last().alias("bid_price"),
            pl.col("bid_amount").filter(pl.col("bid_amount").is_not_null()).last().alias("bid_amount"),
            pl.col("ask_price").filter(pl.col("ask_price").is_not_null()).last().alias("ask_price"),
            pl.col("ask_amount").filter(pl.col("ask_amount").is_not_null()).last().alias("ask_amount"),
            pl.len().alias("quote_count"),
        ]
    )

    # Sort for forward-fill step
    df_1s = df_1s.sort(["symbol", "timestamp_seconds"])

    sampled_count = len(df_1s)
    reduction_pct = (1 - sampled_count / original_count) * 100
    logger.info(f"Sampled to {sampled_count:,} rows ({reduction_pct:.1f}% reduction)")

    return df_1s


def forward_fill_gaps(
    df_1s: pl.DataFrame,
    max_fill_gap: Optional[int] = None,
) -> pl.DataFrame:
    """Forward-fill gaps to create continuous second-by-second time series.

    Args:
        df_1s: DataFrame with 1-second granularity
        max_fill_gap: Maximum gap (in seconds) to forward-fill (None = unlimited)

    Returns:
        DataFrame with forward-filled gaps
    """
    logger.info("=" * 80)
    logger.info("STAGE 3: FORWARD-FILLING GAPS")
    logger.info("=" * 80)

    if max_fill_gap is not None:
        logger.info(f"Maximum forward-fill gap: {max_fill_gap} seconds")
    else:
        logger.info("Maximum forward-fill gap: unlimited")

    original_count = len(df_1s)
    logger.info(f"Starting with {original_count:,} rows")

    # 1. Create global time grid
    min_ts_val = df_1s["timestamp_seconds"].min()
    max_ts_val = df_1s["timestamp_seconds"].max()

    # Type narrowing: ensure we have valid int values
    if min_ts_val is None or max_ts_val is None:
        logger.error("ERROR: Unable to determine timestamp range")
        raise ValueError("timestamp_seconds column contains only null values")

    # Cast to int - Polars min/max returns Python scalar
    if isinstance(min_ts_val, int) and isinstance(max_ts_val, int):
        min_ts = min_ts_val
        max_ts = max_ts_val
    else:
        # Handle case where Polars returns other numeric types
        min_ts = int(min_ts_val)  # type: ignore[arg-type]
        max_ts = int(max_ts_val)  # type: ignore[arg-type]

    total_seconds = max_ts - min_ts + 1

    logger.info(f"Time range: {min_ts} to {max_ts} ({total_seconds:,} seconds)")
    logger.info("Creating complete time grid...")

    time_grid = pl.DataFrame({"timestamp_seconds": pl.arange(min_ts, max_ts + 1, 1, eager=True)})

    # 2. Get unique symbols and their lifecycles
    logger.info("Determining symbol lifecycles...")
    symbol_ranges = df_1s.group_by("symbol").agg(
        [
            pl.col("timestamp_seconds").min().alias("first_seen"),
            pl.col("timestamp_seconds").max().alias("last_seen"),
            pl.col("exchange").first(),
            pl.col("type").first(),
            pl.col("strike_price").first(),
            pl.col("underlying").first(),
            pl.col("expiry_str").first(),
        ]
    )

    num_symbols = len(symbol_ranges)
    logger.info(f"Found {num_symbols:,} unique symbols")

    # 3. Cross join: time_grid × symbols, filtered to each symbol's lifecycle
    logger.info("Creating complete (timestamp, symbol) grid (this may take time)...")
    start_time = time.time()

    complete_grid = time_grid.join(symbol_ranges, how="cross")
    complete_grid = complete_grid.filter(
        (pl.col("timestamp_seconds") >= pl.col("first_seen")) & (pl.col("timestamp_seconds") <= pl.col("last_seen"))
    )

    grid_size = len(complete_grid)
    elapsed = time.time() - start_time
    logger.info(f"Created grid with {grid_size:,} rows in {elapsed:.1f}s")

    # 4. Left join with actual data
    logger.info("Joining with actual quote data...")
    filled = complete_grid.join(
        df_1s.select(
            [
                "timestamp_seconds",
                "symbol",
                "bid_price",
                "bid_amount",
                "ask_price",
                "ask_amount",
                "quote_count",
            ]
        ),
        on=["timestamp_seconds", "symbol"],
        how="left",
    )

    # 5. Forward-fill prices within each symbol group
    logger.info("Forward-filling bid/ask prices...")
    filled = filled.sort(["symbol", "timestamp_seconds"])

    filled = filled.with_columns(
        [
            pl.col("bid_price").forward_fill().over("symbol"),
            pl.col("bid_amount").forward_fill().over("symbol"),
            pl.col("ask_price").forward_fill().over("symbol"),
            pl.col("ask_amount").forward_fill().over("symbol"),
        ]
    )

    # 6. Apply max_fill_gap constraint if specified
    if max_fill_gap is not None:
        logger.info(f"Applying max forward-fill gap of {max_fill_gap} seconds...")

        # Calculate seconds since last real update
        filled = filled.with_columns(
            [
                pl.when(pl.col("quote_count").is_not_null())
                .then(pl.col("timestamp_seconds"))
                .otherwise(None)
                .forward_fill()
                .over("symbol")
                .alias("last_update_ts")
            ]
        )

        filled = filled.with_columns(
            [(pl.col("timestamp_seconds") - pl.col("last_update_ts")).alias("seconds_since_last_update")]
        )

        # Nullify forward-filled values beyond max_fill_gap
        filled = filled.with_columns(
            [
                pl.when(pl.col("seconds_since_last_update") <= max_fill_gap)
                .then(pl.col("bid_price"))
                .otherwise(None)
                .alias("bid_price"),
                pl.when(pl.col("seconds_since_last_update") <= max_fill_gap)
                .then(pl.col("bid_amount"))
                .otherwise(None)
                .alias("bid_amount"),
                pl.when(pl.col("seconds_since_last_update") <= max_fill_gap)
                .then(pl.col("ask_price"))
                .otherwise(None)
                .alias("ask_price"),
                pl.when(pl.col("seconds_since_last_update") <= max_fill_gap)
                .then(pl.col("ask_amount"))
                .otherwise(None)
                .alias("ask_amount"),
            ]
        )

        # Keep seconds_since_last_update column
        filled = filled.drop("last_update_ts")
    else:
        # Add placeholder column
        filled = filled.with_columns([pl.lit(None).cast(pl.Int64).alias("seconds_since_last_update")])

    # 7. Add metadata columns
    filled = filled.with_columns(
        [
            pl.when(pl.col("quote_count").is_null())
            .then(pl.lit(True))
            .otherwise(pl.lit(False))
            .alias("is_forward_filled"),
            pl.col("quote_count").fill_null(0),
        ]
    )

    # 8. Calculate seconds_since_last_update if not already done
    if max_fill_gap is None:
        filled = filled.with_columns(
            [
                pl.when(pl.col("quote_count").is_not_null())
                .then(pl.col("timestamp_seconds"))
                .otherwise(None)
                .forward_fill()
                .over("symbol")
                .alias("last_update_ts")
            ]
        )

        filled = filled.with_columns(
            [(pl.col("timestamp_seconds") - pl.col("last_update_ts")).alias("seconds_since_last_update")]
        )

        filled = filled.drop("last_update_ts")

    # Drop lifecycle columns (first_seen, last_seen)
    filled = filled.drop(["first_seen", "last_seen"])

    # Reorder columns for clarity
    filled = filled.select(
        [
            "timestamp_seconds",
            "symbol",
            "exchange",
            "type",
            "strike_price",
            "underlying",
            "expiry_str",
            "bid_price",
            "bid_amount",
            "ask_price",
            "ask_amount",
            "quote_count",
            "is_forward_filled",
            "seconds_since_last_update",
        ]
    )

    final_count = len(filled)
    filled_count = final_count - original_count
    logger.info(f"Final row count: {final_count:,} ({filled_count:,} forward-filled)")

    # Statistics: Forward-fill
    num_forward_filled = filled.filter(pl.col("is_forward_filled")).shape[0]
    forward_fill_pct = (num_forward_filled / final_count) * 100
    logger.info(f"Forward-filled rows: {num_forward_filled:,} ({forward_fill_pct:.1f}%)")

    # Statistics: NULL bid/ask
    logger.info("\nData Quality Statistics:")
    null_bid = filled.filter(pl.col("bid_price").is_null()).shape[0]
    null_ask = filled.filter(pl.col("ask_price").is_null()).shape[0]
    both_null = filled.filter((pl.col("bid_price").is_null()) & (pl.col("ask_price").is_null())).shape[0]
    either_null = filled.filter((pl.col("bid_price").is_null()) | (pl.col("ask_price").is_null())).shape[0]
    both_present = filled.filter((pl.col("bid_price").is_not_null()) & (pl.col("ask_price").is_not_null())).shape[0]

    logger.info(f"  Rows with NULL bid_price:        {null_bid:>10,} ({(null_bid / final_count) * 100:>5.2f}%)")
    logger.info(f"  Rows with NULL ask_price:        {null_ask:>10,} ({(null_ask / final_count) * 100:>5.2f}%)")
    logger.info(f"  Rows with both NULL:             {both_null:>10,} ({(both_null / final_count) * 100:>5.2f}%)")
    logger.info(f"  Rows with either NULL:           {either_null:>10,} ({(either_null / final_count) * 100:>5.2f}%)")
    logger.info(f"  Rows with both bid & ask:        {both_present:>10,} ({(both_present / final_count) * 100:>5.2f}%)")

    return filled


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Consolidate Deribit options quote CSV.gz files and sample to 1-second granularity with forward-fill"
    )

    parser.add_argument(
        "--input-dir",
        required=True,
        help="Input directory containing CSV.gz files",
    )
    parser.add_argument(
        "--output-consolidated",
        required=True,
        help="Output path for consolidated Parquet file",
    )
    parser.add_argument(
        "--output-sampled",
        required=True,
        help="Output path for 1-second sampled Parquet file",
    )
    parser.add_argument(
        "--max-fill-gap",
        type=int,
        default=None,
        help="Maximum gap (seconds) to forward-fill (default: unlimited)",
    )
    parser.add_argument(
        "--skip-consolidation",
        action="store_true",
        help="Skip consolidation (use existing consolidated file)",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable debug logging",
    )

    args = parser.parse_args()

    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(level=log_level, format="%(message)s", handlers=[logging.StreamHandler(sys.stdout)])

    # Validate input directory
    if not os.path.isdir(args.input_dir):
        logger.error(f"ERROR: Input directory does not exist: {args.input_dir}")
        sys.exit(1)

    # Validate max_fill_gap
    if args.max_fill_gap is not None and args.max_fill_gap < 1:
        logger.error("ERROR: --max-fill-gap must be >= 1")
        sys.exit(1)

    logger.info("=" * 80)
    logger.info("CONSOLIDATE & SAMPLE OPTIONS QUOTES")
    logger.info("=" * 80)
    logger.info(f"Input directory: {args.input_dir}")
    logger.info(f"Output consolidated: {args.output_consolidated}")
    logger.info(f"Output sampled: {args.output_sampled}")
    logger.info(f"Max fill gap: {args.max_fill_gap if args.max_fill_gap else 'unlimited'}")
    logger.info("=" * 80)

    start_time = time.time()

    # Stage 1: Consolidate
    if args.skip_consolidation:
        logger.info("Skipping consolidation, reading existing file...")
        if not os.path.exists(args.output_consolidated):
            logger.error(f"ERROR: Consolidated file does not exist: {args.output_consolidated}")
            sys.exit(1)
        df = pl.read_parquet(args.output_consolidated)
        logger.info(f"Loaded {len(df):,} rows from {args.output_consolidated}")
    else:
        df = consolidate_csv_files(args.input_dir, args.output_consolidated)

    # Stage 2: Sample to 1-second
    df_1s = sample_to_1s(df)

    # Stage 3: Forward-fill gaps
    df_filled = forward_fill_gaps(df_1s, args.max_fill_gap)

    # Write final output
    logger.info("=" * 80)
    logger.info("WRITING FINAL OUTPUT")
    logger.info("=" * 80)
    logger.info(f"Writing sampled data to {args.output_sampled}...")
    df_filled.write_parquet(args.output_sampled, compression="snappy")

    file_size_mb = os.path.getsize(args.output_sampled) / (1024 * 1024)
    logger.info(f"Sampled file size: {file_size_mb:.1f} MB")

    # Summary
    elapsed = time.time() - start_time
    logger.info("=" * 80)
    logger.info("COMPLETE!")
    logger.info(f"Total time: {elapsed:.1f}s")
    logger.info(f"Consolidated file: {args.output_consolidated}")
    logger.info(f"Sampled file: {args.output_sampled}")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
