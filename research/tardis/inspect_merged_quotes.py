#!/usr/bin/env python3
"""
Inspect and validate the consolidated 1s-sampled options quotes data.

Provides comprehensive analysis of the merged Parquet file including:
- Basic statistics and data quality checks
- Timestamp coverage and gaps
- Symbol distribution and activity
- Validation of sorting and deduplication
"""

import argparse
import logging
from datetime import datetime
from pathlib import Path

import polars as pl

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def inspect_file_metadata(file_path: Path) -> None:
    """Display file-level metadata."""
    logger.info("=" * 80)
    logger.info("FILE METADATA")
    logger.info("=" * 80)

    file_size_mb = file_path.stat().st_size / (1024 * 1024)
    logger.info(f"File: {file_path}")
    logger.info(f"Size: {file_size_mb:,.1f} MB ({file_path.stat().st_size:,} bytes)")


def inspect_schema_and_sample(file_path: Path, n_rows: int = 1000) -> pl.DataFrame:
    """Load sample and display schema."""
    logger.info("=" * 80)
    logger.info("SCHEMA AND SAMPLE DATA")
    logger.info("=" * 80)

    # Load sample
    logger.info(f"Loading first {n_rows:,} rows...")
    df_sample = pl.read_parquet(file_path, n_rows=n_rows)

    # Display schema
    logger.info("\nSchema:")
    for col, dtype in zip(df_sample.columns, df_sample.dtypes):
        logger.info(f"  {col:20s} {dtype}")

    # Display sample
    logger.info("\nFirst 5 rows:")
    print(df_sample.head(5))

    logger.info("\nLast 5 rows of sample:")
    print(df_sample.tail(5))

    return df_sample


def get_basic_stats(file_path: Path) -> dict:
    """Get basic statistics without loading full file."""
    logger.info("=" * 80)
    logger.info("BASIC STATISTICS")
    logger.info("=" * 80)

    # Use lazy scan for efficient statistics
    logger.info("Scanning file (this may take a moment)...")

    lazy_df = pl.scan_parquet(file_path)

    # Count total rows
    total_rows = lazy_df.select(pl.len()).collect().item()
    logger.info(f"Total rows: {total_rows:,}")

    # Get timestamp range
    ts_stats = lazy_df.select(
        [
            pl.col("timestamp_seconds").min().alias("min_ts"),
            pl.col("timestamp_seconds").max().alias("max_ts"),
        ]
    ).collect()

    min_ts = ts_stats["min_ts"][0]
    max_ts = ts_stats["max_ts"][0]

    min_date = datetime.fromtimestamp(min_ts).strftime("%Y-%m-%d %H:%M:%S")
    max_date = datetime.fromtimestamp(max_ts).strftime("%Y-%m-%d %H:%M:%S")

    duration_days = (max_ts - min_ts) / 86400

    logger.info("\nTimestamp range:")
    logger.info(f"  Start: {min_ts} ({min_date})")
    logger.info(f"  End:   {max_ts} ({max_date})")
    logger.info(f"  Duration: {duration_days:.1f} days")

    # Count unique symbols
    unique_symbols = lazy_df.select(pl.col("symbol").n_unique()).collect().item()
    logger.info(f"\nUnique symbols: {unique_symbols:,}")

    # Exchange and type distribution
    exchange_counts = lazy_df.group_by("exchange").len().collect()
    logger.info("\nExchange distribution:")
    for row in exchange_counts.iter_rows():
        logger.info(f"  {row[0]:10s}: {row[1]:,} rows")

    type_counts = lazy_df.group_by("type").len().collect()
    logger.info("\nOption type distribution:")
    for row in type_counts.iter_rows():
        logger.info(f"  {row[0]:10s}: {row[1]:,} rows")

    return {
        "total_rows": total_rows,
        "min_ts": min_ts,
        "max_ts": max_ts,
        "duration_days": duration_days,
        "unique_symbols": unique_symbols,
    }


def validate_sorting(file_path: Path, sample_size: int = 1_000_000) -> None:
    """Validate that data is properly sorted."""
    logger.info("=" * 80)
    logger.info("SORTING VALIDATION")
    logger.info("=" * 80)

    logger.info(f"Checking sorting on sample of {sample_size:,} rows...")

    # Load sample
    df_sample = pl.read_parquet(file_path, n_rows=sample_size)

    # Check if sorted by comparing with sorted version
    ts_col = df_sample["timestamp_seconds"]
    is_sorted = (ts_col == ts_col.sort()).all()

    if is_sorted:
        logger.info("✓ Data is sorted by timestamp_seconds")
    else:
        logger.warning("✗ Data is NOT properly sorted by timestamp_seconds")

    # Check for sorting within timestamp groups
    logger.info("Checking symbol sorting within timestamps...")

    # Sample a few timestamps and check symbol ordering
    sample_timestamps = df_sample["timestamp_seconds"].unique().head(10)

    all_symbol_sorted = True
    for ts in sample_timestamps:
        ts_group = df_sample.filter(pl.col("timestamp_seconds") == ts)
        if len(ts_group) > 1:
            symbols = ts_group["symbol"].to_list()
            is_symbol_sorted = symbols == sorted(symbols)
            if not is_symbol_sorted:
                all_symbol_sorted = False
                logger.warning(f"  ✗ Symbols not sorted at timestamp {ts}")
                break

    if all_symbol_sorted:
        logger.info("✓ Symbols are sorted within timestamps")


def check_duplicates(file_path: Path) -> None:
    """Check for duplicate (timestamp, symbol) keys."""
    logger.info("=" * 80)
    logger.info("DUPLICATE CHECK")
    logger.info("=" * 80)

    logger.info("Checking for duplicate (timestamp_seconds, symbol) keys...")

    lazy_df = pl.scan_parquet(file_path)

    # Count total rows
    total_rows = lazy_df.select(pl.len()).collect().item()

    # Count unique (timestamp, symbol) combinations
    unique_keys = lazy_df.select(["timestamp_seconds", "symbol"]).unique().select(pl.len()).collect().item()

    duplicates = total_rows - unique_keys

    if duplicates == 0:
        logger.info("✓ No duplicates found")
        logger.info(f"  Total rows: {total_rows:,}")
        logger.info(f"  Unique keys: {unique_keys:,}")
    else:
        logger.warning(f"✗ Found {duplicates:,} duplicate keys!")
        logger.warning(f"  Total rows: {total_rows:,}")
        logger.warning(f"  Unique keys: {unique_keys:,}")


def analyze_symbols(file_path: Path, top_n: int = 20) -> None:
    """Analyze symbol distribution and activity."""
    logger.info("=" * 80)
    logger.info("SYMBOL ANALYSIS")
    logger.info("=" * 80)

    logger.info(f"Analyzing top {top_n} most active symbols...")

    lazy_df = pl.scan_parquet(file_path)

    # Top symbols by quote count
    top_symbols = (
        lazy_df.group_by("symbol")
        .agg(
            [
                pl.len().alias("quote_count"),
                pl.col("timestamp_seconds").min().alias("first_ts"),
                pl.col("timestamp_seconds").max().alias("last_ts"),
            ]
        )
        .with_columns([((pl.col("last_ts") - pl.col("first_ts")) / 86400).alias("duration_days")])
        .sort("quote_count", descending=True)
        .head(top_n)
        .collect()
    )

    logger.info(f"\nTop {top_n} symbols by quote count:")
    print(top_symbols)

    # Underlying distribution
    logger.info("\nAnalyzing by underlying asset...")
    underlying_stats = (
        lazy_df.group_by("underlying")
        .agg(
            [
                pl.len().alias("total_quotes"),
                pl.col("symbol").n_unique().alias("unique_contracts"),
            ]
        )
        .sort("total_quotes", descending=True)
        .collect()
    )

    logger.info("\nUnderlying asset distribution:")
    print(underlying_stats)


def analyze_spreads(file_path: Path, sample_size: int = 100_000) -> None:
    """Analyze bid-ask spreads."""
    logger.info("=" * 80)
    logger.info("BID-ASK SPREAD ANALYSIS")
    logger.info("=" * 80)

    logger.info(f"Analyzing spreads on sample of {sample_size:,} rows...")

    df_sample = pl.read_parquet(file_path, n_rows=sample_size)

    # Calculate spreads
    df_spreads = df_sample.with_columns(
        [
            (pl.col("ask_price") - pl.col("bid_price")).alias("spread"),
            ((pl.col("ask_price") - pl.col("bid_price")) / pl.col("bid_price") * 100).alias("spread_pct"),
        ]
    )

    # Spread statistics
    spread_stats = df_spreads.select(
        [
            pl.col("spread").min().alias("min_spread"),
            pl.col("spread").mean().alias("avg_spread"),
            pl.col("spread").median().alias("median_spread"),
            pl.col("spread").max().alias("max_spread"),
            pl.col("spread_pct").mean().alias("avg_spread_pct"),
            pl.col("spread_pct").median().alias("median_spread_pct"),
        ]
    )

    logger.info("\nSpread statistics:")
    print(spread_stats)

    # Count null prices
    null_counts = df_sample.select(
        [
            pl.col("bid_price").is_null().sum().alias("null_bid"),
            pl.col("ask_price").is_null().sum().alias("null_ask"),
        ]
    )

    logger.info("\nNull price counts (in sample):")
    print(null_counts)


def main() -> None:
    """Main inspection routine."""
    parser = argparse.ArgumentParser(description="Inspect and validate consolidated options quotes data")
    parser.add_argument(
        "file_path",
        type=str,
        help="Path to consolidated Parquet file",
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=1_000_000,
        help="Sample size for validation checks (default: 1,000,000)",
    )
    parser.add_argument(
        "--top-symbols",
        type=int,
        default=20,
        help="Number of top symbols to display (default: 20)",
    )

    args = parser.parse_args()

    file_path = Path(args.file_path)

    if not file_path.exists():
        logger.error(f"File not found: {file_path}")
        return

    logger.info("=" * 80)
    logger.info("CONSOLIDATED OPTIONS QUOTES - DATA INSPECTION")
    logger.info("=" * 80)

    # Run all inspections
    inspect_file_metadata(file_path)
    inspect_schema_and_sample(file_path)
    stats = get_basic_stats(file_path)
    validate_sorting(file_path, args.sample_size)
    check_duplicates(file_path)
    analyze_symbols(file_path, args.top_symbols)
    analyze_spreads(file_path, min(args.sample_size, 100_000))

    # Final summary
    logger.info("=" * 80)
    logger.info("INSPECTION COMPLETE")
    logger.info("=" * 80)
    logger.info(f"File: {file_path}")
    logger.info(f"Total rows: {stats['total_rows']:,}")
    logger.info(f"Date range: {stats['duration_days']:.1f} days")
    logger.info(f"Unique symbols: {stats['unique_symbols']:,}")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
