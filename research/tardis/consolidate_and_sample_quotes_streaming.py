#!/usr/bin/env python3
"""
Space-efficient options quote consolidation pipeline.

Streams CSV.gz → 1s sampled directly, avoiding large intermediate files.
Optimized for systems with limited disk space.
"""

import argparse
import heapq
import logging
import os
import sys
import time
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import polars as pl
import pyarrow.parquet as pq

logger = logging.getLogger(__name__)


@dataclass(order=True)
class HeapEntry:
    """Entry for k-way merge heap, ordered by (timestamp_seconds, symbol)."""

    timestamp_seconds: int
    symbol: str = field(compare=True)
    row_data: dict[str, Any] = field(compare=False, repr=False)
    batch_idx: int = field(compare=False, repr=False)
    reader: Any = field(compare=False, repr=False)


class ParquetBatchReader:
    """Memory-efficient iterator over Parquet file in batches."""

    def __init__(self, file_path: Path, chunk_size: int = 10_000):
        """Initialize batch reader for Parquet file.

        Args:
            file_path: Path to Parquet file
            chunk_size: Number of rows to read per batch (default: 10,000)
        """
        self.file_path = file_path
        self.chunk_size = chunk_size
        self.current_chunk: list[dict[str, Any]] = []
        self.chunk_idx = 0
        self.total_rows_read = 0
        self.exhausted = False

        # Read schema to know column names
        schema_df = pl.read_parquet(file_path, n_rows=1)
        self.columns = schema_df.columns

    def __iter__(self) -> Iterator[tuple[int, str, dict[str, Any]]]:
        """Iterate over rows in file.

        Yields:
            Tuple of (timestamp_seconds, symbol, row_dict)
        """
        return self

    def __next__(self) -> tuple[int, str, dict[str, Any]]:
        """Get next row from file.

        Returns:
            Tuple of (timestamp_seconds, symbol, row_dict)

        Raises:
            StopIteration: When file is exhausted
        """
        # Load next chunk if current is exhausted
        if self.chunk_idx >= len(self.current_chunk):
            if self.exhausted:
                raise StopIteration

            self._load_next_chunk()

            if len(self.current_chunk) == 0:
                self.exhausted = True
                raise StopIteration

        # Get row from current chunk
        row_dict = self.current_chunk[self.chunk_idx]
        self.chunk_idx += 1

        timestamp_seconds = row_dict["timestamp_seconds"]
        symbol = row_dict["symbol"]

        return (timestamp_seconds, symbol, row_dict)

    def _load_next_chunk(self) -> None:
        """Load next chunk of rows from Parquet file."""
        try:
            # Use lazy scan with slice to efficiently read chunk
            df_chunk = pl.scan_parquet(self.file_path).slice(self.total_rows_read, self.chunk_size).collect()

            if len(df_chunk) == 0:
                self.current_chunk = []
                self.exhausted = True
                return

            # Convert to list of dicts
            self.current_chunk = df_chunk.to_dicts()
            self.chunk_idx = 0
            self.total_rows_read += len(self.current_chunk)

        except Exception as e:
            logger.warning(f"Error loading chunk from {self.file_path.name}: {e}")
            self.current_chunk = []
            self.exhausted = True


def sample_batch_to_1s(df: pl.DataFrame) -> pl.DataFrame:
    """Sample a batch DataFrame to 1-second granularity.

    Args:
        df: DataFrame with microsecond timestamps

    Returns:
        DataFrame sampled to 1-second granularity
    """
    # Add timestamp_seconds column
    df = df.with_columns([(pl.col("timestamp") // 1_000_000).cast(pl.Int64).alias("timestamp_seconds")])

    # Group by (second, symbol) and aggregate
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

    # Sort for efficient merging later
    df_1s = df_1s.sort(["timestamp_seconds", "symbol"])

    return df_1s


def kway_merge_batches_streaming(
    batch_files: list[Path],
    output_path: Path,
    chunk_size: int = 100_000,
    reader_chunk_size: int = 10_000,
) -> int:
    """Merge k sorted Parquet files using heap-based streaming merge with deduplication.

    Args:
        batch_files: List of sorted Parquet batch files to merge
        output_path: Path for merged output Parquet file
        chunk_size: Number of rows to buffer before writing (default: 100,000)
        reader_chunk_size: Chunk size for reading each batch file (default: 10,000)

    Returns:
        Total number of rows written to output file
    """
    logger.info(f"Initializing k-way merge for {len(batch_files)} batch files...")
    logger.info(f"Reader chunk size: {reader_chunk_size:,} rows")
    logger.info(f"Output buffer size: {chunk_size:,} rows")

    # Initialize readers for all batch files
    readers: list[ParquetBatchReader] = []
    for batch_file in batch_files:
        try:
            reader = ParquetBatchReader(batch_file, chunk_size=reader_chunk_size)
            readers.append(reader)
        except Exception as e:
            logger.warning(f"Failed to open {batch_file.name}: {e}")

    if len(readers) == 0:
        logger.error("No valid batch files to merge")
        raise ValueError("No valid batch files found")

    logger.info(f"Successfully opened {len(readers)} batch files")

    # Build initial heap with first row from each reader
    heap: list[HeapEntry] = []
    for idx, reader in enumerate(readers):
        try:
            timestamp_seconds, symbol, row_data = next(reader)
            entry = HeapEntry(
                timestamp_seconds=timestamp_seconds,
                symbol=symbol,
                row_data=row_data,
                batch_idx=idx,
                reader=reader,
            )
            heapq.heappush(heap, entry)
        except StopIteration:
            # Empty batch file
            logger.warning(f"Batch file {batch_files[idx].name} is empty")
            continue

    logger.info(f"Initial heap size: {len(heap)} entries")

    # Initialize PyArrow streaming writer
    # We'll create the schema from the first row
    first_entry = heap[0] if heap else None
    if first_entry is None:
        raise ValueError("No data to merge")

    # Create PyArrow schema from first row
    first_df = pl.DataFrame([first_entry.row_data])
    schema = first_df.to_arrow().schema

    # Open PyArrow Parquet writer for streaming writes
    writer = pq.ParquetWriter(str(output_path), schema, compression="snappy")

    # Main merge loop
    output_buffer: list[dict[str, Any]] = []
    prev_key: Optional[tuple[int, str]] = None
    total_rows_written = 0
    duplicates_skipped = 0
    rows_processed = 0

    merge_start = time.time()

    while heap:
        # Pop minimum entry
        entry = heapq.heappop(heap)
        current_key = (entry.timestamp_seconds, entry.symbol)
        rows_processed += 1

        # Deduplication: skip if same key as previous
        if current_key == prev_key:
            duplicates_skipped += 1
        else:
            output_buffer.append(entry.row_data)
            prev_key = current_key

        # Write buffer to disk when full
        if len(output_buffer) >= chunk_size:
            df_chunk = pl.DataFrame(output_buffer)
            table_chunk = df_chunk.to_arrow()
            writer.write_table(table_chunk)
            total_rows_written += len(output_buffer)
            output_buffer = []

            # Progress logging every 10M rows
            if total_rows_written % 10_000_000 == 0:
                elapsed = time.time() - merge_start
                throughput = total_rows_written / elapsed
                logger.info(
                    f"Progress: {total_rows_written:,} rows written, "
                    f"{duplicates_skipped:,} duplicates skipped, "
                    f"{throughput:,.0f} rows/sec"
                )

        # Refill heap from same reader
        try:
            timestamp_seconds, symbol, row_data = next(entry.reader)
            new_entry = HeapEntry(
                timestamp_seconds=timestamp_seconds,
                symbol=symbol,
                row_data=row_data,
                batch_idx=entry.batch_idx,
                reader=entry.reader,
            )
            heapq.heappush(heap, new_entry)
        except StopIteration:
            # This reader is exhausted
            pass

    # Write final buffer to disk
    if output_buffer:
        df_chunk = pl.DataFrame(output_buffer)
        table_chunk = df_chunk.to_arrow()
        writer.write_table(table_chunk)
        total_rows_written += len(output_buffer)

    # Close writer
    writer.close()

    merge_elapsed = time.time() - merge_start
    throughput = total_rows_written / merge_elapsed if merge_elapsed > 0 else 0

    logger.info("=" * 80)
    logger.info("K-way merge complete:")
    logger.info(f"  Total rows processed: {rows_processed:,}")
    logger.info(f"  Total rows written: {total_rows_written:,}")
    logger.info(f"  Duplicates removed: {duplicates_skipped:,}")
    logger.info(f"  Merge time: {merge_elapsed:.1f}s")
    logger.info(f"  Average throughput: {throughput:,.0f} rows/sec")
    logger.info("=" * 80)

    return total_rows_written


def consolidate_and_sample_batched(
    input_dir: str,
    batch_size: int = 30,
    checkpoint_dir: str = "checkpoints_1s_sampled",
) -> Path:
    """Consolidate CSV.gz files and sample to 1-second, using batch processing.

    Args:
        input_dir: Directory containing CSV.gz files
        batch_size: Number of files to process per batch (default: 30)
        checkpoint_dir: Directory for intermediate 1s-sampled batch files

    Returns:
        Path to the consolidated 1-second sampled Parquet file
    """
    logger.info("=" * 80)
    logger.info("STAGE 1: BATCH PROCESSING & SAMPLING TO 1-SECOND")
    logger.info("=" * 80)

    # Find all CSV.gz files
    csv_files = sorted(Path(input_dir).glob("*.csv.gz"))
    logger.info(f"Found {len(csv_files)} CSV.gz files")
    logger.info(f"Batch size: {batch_size} files")

    if len(csv_files) == 0:
        logger.error(f"No CSV.gz files found in {input_dir}")
        sys.exit(1)

    # Create checkpoint directory
    checkpoint_path = Path(checkpoint_dir)
    checkpoint_path.mkdir(exist_ok=True)
    logger.info(f"Intermediate 1s-sampled batches will be stored in: {checkpoint_dir}")

    # Schema overrides
    schema_overrides = {
        "strike_price": pl.Float64,
        "bid_price": pl.Float64,
        "bid_amount": pl.Float64,
        "ask_price": pl.Float64,
        "ask_amount": pl.Float64,
    }

    # Process files in batches
    num_batches = (len(csv_files) + batch_size - 1) // batch_size
    logger.info(f"Processing {num_batches} batches...")

    batch_files: list[Path] = []
    total_raw_rows = 0
    total_sampled_rows = 0

    for batch_idx in range(num_batches):
        start_idx = batch_idx * batch_size
        end_idx = min(start_idx + batch_size, len(csv_files))
        batch_csv_files = csv_files[start_idx:end_idx]

        batch_file_path = checkpoint_path / f"batch_1s_{batch_idx:04d}.parquet"

        # Skip if batch already processed
        if batch_file_path.exists():
            logger.info(f"Batch {batch_idx + 1}/{num_batches}: Using cached {batch_file_path.name}")
            batch_files.append(batch_file_path)
            # Read to get row counts for summary
            cached_df = pl.read_parquet(batch_file_path)
            total_sampled_rows += len(cached_df)
            continue

        logger.info(
            f"Batch {batch_idx + 1}/{num_batches}: Processing files {start_idx + 1}-{end_idx} ({len(batch_csv_files)} files)"
        )

        # Read batch files lazily
        lazy_dfs = []
        for csv_file in batch_csv_files:
            try:
                lazy_df = pl.scan_csv(csv_file, schema_overrides=schema_overrides)
                lazy_dfs.append(lazy_df)
            except Exception as e:
                logger.warning(f"  Failed to read {csv_file.name}: {e}")

        if len(lazy_dfs) == 0:
            logger.warning(f"  No valid files in batch {batch_idx + 1}, skipping")
            continue

        # Combine batch
        logger.info(f"  Combining {len(lazy_dfs)} files...")
        batch_combined = pl.concat(lazy_dfs)

        # Collect batch (load into memory)
        logger.info("  Collecting batch...")
        batch_start = time.time()
        batch_df = batch_combined.collect()
        collect_elapsed = time.time() - batch_start

        batch_raw_rows = len(batch_df)
        total_raw_rows += batch_raw_rows
        logger.info(f"  Collected {batch_raw_rows:,} rows in {collect_elapsed:.1f}s")

        # Sample to 1-second immediately (reduces size ~50%)
        logger.info("  Sampling to 1-second granularity...")
        sample_start = time.time()
        batch_1s = sample_batch_to_1s(batch_df)
        sample_elapsed = time.time() - sample_start

        batch_sampled_rows = len(batch_1s)
        total_sampled_rows += batch_sampled_rows
        reduction_pct = (1 - batch_sampled_rows / batch_raw_rows) * 100

        logger.info(
            f"  Sampled to {batch_sampled_rows:,} rows in {sample_elapsed:.1f}s ({reduction_pct:.1f}% reduction)"
        )

        # Free memory
        del batch_df

        # Write sampled batch
        logger.info(f"  Writing {batch_file_path.name}...")
        batch_1s.write_parquet(batch_file_path, compression="snappy")

        batch_size_mb = batch_file_path.stat().st_size / (1024 * 1024)
        logger.info(f"  Batch file size: {batch_size_mb:.1f} MB")

        batch_files.append(batch_file_path)

    logger.info("=" * 80)
    logger.info("Batch processing complete:")
    logger.info(f"  Total raw rows processed: {total_raw_rows:,}")
    logger.info(f"  Total 1s-sampled rows: {total_sampled_rows:,}")
    if total_raw_rows > 0:
        logger.info(f"  Overall reduction: {(1 - total_sampled_rows / total_raw_rows) * 100:.1f}%")
    else:
        logger.info("  Overall reduction: (all batches cached)")

    # Merge sampled batches using k-way merge
    logger.info("=" * 80)
    logger.info("STAGE 2: MERGING 1S-SAMPLED BATCHES (K-WAY MERGE)")
    logger.info("=" * 80)

    merge_start = time.time()

    # Create a temporary output file for merged data
    merged_output = checkpoint_path / "merged_sorted_temp.parquet"

    # Use k-way merge with streaming and deduplication
    total_merged_rows = kway_merge_batches_streaming(
        batch_files=batch_files,
        output_path=merged_output,
        chunk_size=100_000,
        reader_chunk_size=10_000,
    )

    merge_elapsed = time.time() - merge_start
    throughput = total_merged_rows / merge_elapsed if merge_elapsed > 0 else 0
    logger.info(f"Total merge time: {merge_elapsed:.1f}s ({throughput:,.0f} rows/sec)")

    # Get file size
    merged_size_mb = merged_output.stat().st_size / (1024 * 1024)
    logger.info(f"Merged file size: {merged_size_mb:.1f} MB")
    logger.info(f"Merged file path: {merged_output}")

    # Cleanup intermediate batch files
    logger.info("Cleaning up intermediate batch files...")
    for batch_file in batch_files:
        try:
            batch_file.unlink()
        except Exception as e:
            logger.warning(f"Failed to delete {batch_file}: {e}")

    # Return path to merged file (do NOT load into memory)
    return merged_output


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

    # Type narrowing
    if min_ts_val is None or max_ts_val is None:
        logger.error("ERROR: Unable to determine timestamp range")
        raise ValueError("timestamp_seconds column contains only null values")

    if isinstance(min_ts_val, int) and isinstance(max_ts_val, int):
        min_ts = min_ts_val
        max_ts = max_ts_val
    else:
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
    logger.info("Creating complete (timestamp, symbol) grid...")
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

        filled = filled.drop("last_update_ts")
    else:
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
                pl.when(pl.col("quote_count") > 0)
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

    # Drop lifecycle columns
    filled = filled.drop(["first_seen", "last_seen"])

    # Reorder columns
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

    # Statistics
    num_forward_filled = filled.filter(pl.col("is_forward_filled")).shape[0]
    forward_fill_pct = (num_forward_filled / final_count) * 100
    logger.info(f"Forward-filled rows: {num_forward_filled:,} ({forward_fill_pct:.1f}%)")

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
        description="Space-efficient consolidation: CSV.gz → 1s sampled with forward-fill (no large intermediate files)"
    )

    parser.add_argument(
        "--input-dir",
        required=True,
        help="Input directory containing CSV.gz files",
    )
    parser.add_argument(
        "--output-sampled",
        required=True,
        help="Output path for 1-second sampled Parquet file",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=30,
        help="Number of CSV.gz files to process per batch (default: 30)",
    )
    parser.add_argument(
        "--checkpoint-dir",
        default="checkpoints_1s_sampled",
        help="Directory for intermediate 1s-sampled batch files (default: checkpoints_1s_sampled)",
    )
    parser.add_argument(
        "--max-fill-gap",
        type=int,
        default=None,
        help="Maximum gap (seconds) to forward-fill (default: unlimited)",
    )
    parser.add_argument(
        "--skip-forward-fill",
        action="store_true",
        help="Skip forward-fill stage (output 1s-sampled data only)",
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

    # Validate batch_size
    if args.batch_size < 1:
        logger.error("ERROR: --batch-size must be >= 1")
        sys.exit(1)

    # Validate max_fill_gap
    if args.max_fill_gap is not None and args.max_fill_gap < 1:
        logger.error("ERROR: --max-fill-gap must be >= 1")
        sys.exit(1)

    logger.info("=" * 80)
    logger.info("SPACE-EFFICIENT CONSOLIDATION & SAMPLING")
    logger.info("=" * 80)
    logger.info(f"Input directory: {args.input_dir}")
    logger.info(f"Output sampled: {args.output_sampled}")
    logger.info(f"Batch size: {args.batch_size} files")
    logger.info(f"Checkpoint directory: {args.checkpoint_dir}")
    logger.info(f"Max fill gap: {args.max_fill_gap if args.max_fill_gap else 'unlimited'}")
    logger.info(f"Skip forward-fill: {args.skip_forward_fill}")
    logger.info("=" * 80)

    start_time = time.time()

    # Stage 1 & 2: Consolidate and sample (returns Path to merged file)
    merged_file_path = consolidate_and_sample_batched(
        args.input_dir,
        batch_size=args.batch_size,
        checkpoint_dir=args.checkpoint_dir,
    )

    # Stage 3: Forward-fill gaps (optional)
    if args.skip_forward_fill:
        logger.info("=" * 80)
        logger.info("Skipping forward-fill as requested")
        logger.info("=" * 80)
        # Simply copy/rename merged file to output path
        import shutil

        shutil.copy2(merged_file_path, args.output_sampled)
        logger.info(f"Copied merged file to {args.output_sampled}")
        file_size_mb = os.path.getsize(args.output_sampled) / (1024 * 1024)
        logger.info(f"Output file size: {file_size_mb:.1f} MB")

        # Clean up checkpoint directory
        checkpoint_path = merged_file_path.parent
        try:
            merged_file_path.unlink()
            checkpoint_path.rmdir()
            logger.info(f"Removed checkpoint directory: {checkpoint_path}")
        except Exception as e:
            logger.warning(f"Failed to cleanup checkpoint: {e}")
    else:
        # Load merged file and apply forward-fill
        logger.info("=" * 80)
        logger.info("Loading merged file for forward-fill stage...")
        df_1s = pl.read_parquet(merged_file_path)
        logger.info(f"Loaded {len(df_1s):,} rows")

        df_final = forward_fill_gaps(df_1s, args.max_fill_gap)

        # Write final output
        logger.info("=" * 80)
        logger.info("WRITING FINAL OUTPUT")
        logger.info("=" * 80)
        logger.info(f"Writing forward-filled data to {args.output_sampled}...")
        df_final.write_parquet(args.output_sampled, compression="snappy")

        file_size_mb = os.path.getsize(args.output_sampled) / (1024 * 1024)
        logger.info(f"Output file size: {file_size_mb:.1f} MB")

        # Clean up checkpoint directory
        checkpoint_path = merged_file_path.parent
        try:
            merged_file_path.unlink()
            checkpoint_path.rmdir()
            logger.info(f"Removed checkpoint directory: {checkpoint_path}")
        except Exception as e:
            logger.warning(f"Failed to cleanup checkpoint: {e}")

    # Summary
    elapsed = time.time() - start_time
    logger.info("=" * 80)
    logger.info("COMPLETE!")
    logger.info(f"Total time: {elapsed:.1f}s ({elapsed / 60:.1f} minutes)")
    logger.info(f"Output file: {args.output_sampled}")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
