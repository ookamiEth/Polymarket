#!/usr/bin/env python3
"""
Consolidate multiple Parquet files into a single file.

Simple memory-efficient consolidation for daily Parquet files.
Uses Polars lazy evaluation to minimize memory usage.
"""

import argparse
import logging
import sys
import time
from pathlib import Path

import polars as pl

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def consolidate_parquet_files(
    input_dir: str,
    output_file: str,
    pattern: str = "**/*.parquet",
    streaming: bool = True,
) -> None:
    """Consolidate multiple Parquet files into a single file.

    Args:
        input_dir: Directory containing Parquet files
        output_file: Output path for consolidated Parquet file
        pattern: Glob pattern for finding Parquet files (default: **/*.parquet)
        streaming: Use streaming write for large datasets (default: True)
    """
    logger.info("=" * 80)
    logger.info("PARQUET CONSOLIDATION")
    logger.info("=" * 80)
    logger.info(f"Input directory: {input_dir}")
    logger.info(f"Output file: {output_file}")
    logger.info(f"Pattern: {pattern}")
    logger.info(f"Streaming: {streaming}")
    logger.info("")

    # Find all Parquet files
    input_path = Path(input_dir)
    if not input_path.exists():
        logger.error(f"Input directory not found: {input_dir}")
        sys.exit(1)

    parquet_files = sorted(input_path.glob(pattern))
    logger.info(f"Found {len(parquet_files)} Parquet files")

    if len(parquet_files) == 0:
        logger.error(f"No Parquet files found matching pattern: {pattern}")
        sys.exit(1)

    # Display sample files
    logger.info("Sample files:")
    for i, file in enumerate(parquet_files[:5]):
        logger.info(f"  {i+1}. {file.relative_to(input_path)}")
    if len(parquet_files) > 5:
        logger.info(f"  ... and {len(parquet_files) - 5} more")
    logger.info("")

    # Create output directory if needed
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Check if output already exists
    if output_path.exists():
        logger.warning(f"Output file already exists: {output_file}")
        response = input("Overwrite? (y/N): ").strip().lower()
        if response != "y":
            logger.info("Consolidation cancelled")
            sys.exit(0)
        logger.info("Overwriting existing file...")

    # Consolidate using Polars lazy evaluation
    start_time = time.time()
    logger.info("Loading and consolidating files...")

    try:
        # Use lazy scan for all files
        lazy_dfs = [pl.scan_parquet(str(f)) for f in parquet_files]

        # Concatenate lazily
        combined = pl.concat(lazy_dfs)

        # Sort by timestamp (if column exists)
        # Try common timestamp column names
        sample_df = pl.read_parquet(parquet_files[0], n_rows=1)
        timestamp_col = None
        for col in ["timestamp", "timestamp_seconds", "local_timestamp"]:
            if col in sample_df.columns:
                timestamp_col = col
                break

        if timestamp_col:
            logger.info(f"Sorting by {timestamp_col}...")
            combined = combined.sort(timestamp_col)

        # Write consolidated file
        logger.info("Writing consolidated file...")
        if streaming:
            # Streaming write for large datasets (>100M rows)
            combined.sink_parquet(
                output_file,
                compression="snappy",
                statistics=True,
                row_group_size=1_000_000,  # 1M rows per row group
            )
            logger.info("✅ Consolidation complete (streaming mode)")
            logger.info("   Note: Use separate command to get row count")
        else:
            # Eager collect for smaller datasets
            df_consolidated = combined.collect()
            df_consolidated.write_parquet(
                output_file,
                compression="snappy",
                statistics=True,
                row_group_size=1_000_000,
            )
            logger.info(f"✅ Consolidation complete: {len(df_consolidated):,} rows")

    except Exception as e:
        logger.error(f"Consolidation failed: {e}")
        sys.exit(1)

    elapsed = time.time() - start_time
    logger.info(f"Elapsed time: {elapsed / 60:.1f} minutes")

    # Get file size
    file_size_mb = output_path.stat().st_size / (1024 * 1024)
    logger.info(f"Output file size: {file_size_mb:.1f} MB")

    # Get row count (lazy aggregation)
    logger.info("Calculating row count...")
    row_count = pl.scan_parquet(output_file).select(pl.len()).collect().item()
    logger.info(f"Total rows: {row_count:,}")

    # Calculate compression ratio
    logger.info("")
    logger.info("Summary:")
    logger.info(f"  Input files: {len(parquet_files)}")
    logger.info(f"  Output file: {output_file}")
    logger.info(f"  Total rows: {row_count:,}")
    logger.info(f"  File size: {file_size_mb:.1f} MB")
    logger.info(f"  Avg rows/file: {row_count // len(parquet_files):,}")
    logger.info(f"  Processing time: {elapsed / 60:.1f} minutes")


def main() -> None:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Consolidate multiple Parquet files into a single file"
    )
    parser.add_argument(
        "--input-dir",
        required=True,
        help="Input directory containing Parquet files",
    )
    parser.add_argument(
        "--output-file",
        required=True,
        help="Output path for consolidated Parquet file",
    )
    parser.add_argument(
        "--pattern",
        default="**/*.parquet",
        help="Glob pattern for finding files (default: **/*.parquet)",
    )
    parser.add_argument(
        "--no-streaming",
        action="store_true",
        help="Disable streaming write (use for smaller datasets <100M rows)",
    )

    args = parser.parse_args()

    consolidate_parquet_files(
        input_dir=args.input_dir,
        output_file=args.output_file,
        pattern=args.pattern,
        streaming=not args.no_streaming,
    )


if __name__ == "__main__":
    main()
