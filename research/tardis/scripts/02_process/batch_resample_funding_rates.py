#!/usr/bin/env python3
"""
Batch resample Binance funding rate files to 1-second intervals.

Automatically discovers and processes all funding rate files from raw data directory.
Supports parallel processing and checkpointing for large-scale resampling.

Usage:
    # Process all files
    uv run python batch_resample_funding_rates.py \\
        --input-dir data/raw/binance_funding_rates \\
        --output-dir data/processed/binance_funding_rates_1s \\
        --workers 5

    # With forward-fill
    uv run python batch_resample_funding_rates.py \\
        --input-dir data/raw/binance_funding_rates \\
        --output-dir data/processed/binance_funding_rates_1s \\
        --method forward_fill \\
        --max-fill-gap 60 \\
        --workers 5
"""

import argparse
import json
import logging
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Optional

from resample_funding_rates_to_1s import resample_file

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def discover_input_files(input_dir: str) -> list[dict]:
    """Discover all Parquet files in input directory.

    Args:
        input_dir: Root directory containing funding rate files

    Returns:
        List of dicts with file metadata
    """
    files = []
    input_path = Path(input_dir)

    for parquet_file in input_path.glob("**/*.parquet"):
        # Parse path: {exchange}/{symbol}/{date}.parquet
        parts = parquet_file.relative_to(input_path).parts
        if len(parts) == 3:
            exchange, symbol, filename = parts
            date = filename.replace(".parquet", "")

            files.append(
                {
                    "path": str(parquet_file),
                    "exchange": exchange,
                    "symbol": symbol,
                    "date": date,
                }
            )

    logger.info(f"Discovered {len(files)} files in {input_dir}")
    return files


def generate_output_path(input_metadata: dict, output_dir: str) -> str:
    """Generate output file path.

    Args:
        input_metadata: Dict with exchange, symbol, date
        output_dir: Output root directory

    Returns:
        Output file path
    """
    exchange = input_metadata["exchange"]
    symbol = input_metadata["symbol"]
    date = input_metadata["date"]

    output_file = os.path.join(output_dir, exchange, symbol, f"{date}_1s.parquet")
    return output_file


def save_checkpoint(checkpoint_file: str, completed_files: set[str], stats: dict) -> None:
    """Save progress checkpoint."""
    checkpoint_data = {
        "completed_files": sorted(completed_files),
        "total_processed": stats.get("total_processed", 0),
        "total_input_rows": stats.get("total_input_rows", 0),
        "total_output_rows": stats.get("total_output_rows", 0),
        "last_updated": datetime.now().isoformat(),
    }
    with open(checkpoint_file, "w") as f:
        json.dump(checkpoint_data, f, indent=2)
    logger.debug(f"Checkpoint saved: {len(completed_files)} files completed")


def load_checkpoint(checkpoint_file: str) -> tuple[set[str], dict]:
    """Load progress checkpoint."""
    if not os.path.exists(checkpoint_file):
        return set(), {}

    try:
        with open(checkpoint_file) as f:
            data = json.load(f)
        completed_files = set(data.get("completed_files", []))
        stats = {
            "total_processed": data.get("total_processed", 0),
            "total_input_rows": data.get("total_input_rows", 0),
            "total_output_rows": data.get("total_output_rows", 0),
        }
        logger.info(f"Loaded checkpoint: {len(completed_files)} files already completed")
        return completed_files, stats
    except Exception as e:
        logger.warning(f"Failed to load checkpoint: {e}")
        return set(), {}


def process_single_file(
    file_metadata: dict,
    output_dir: str,
    method: str,
    max_fill_gap: Optional[int],
) -> dict:
    """Process a single file (worker function for parallel execution).

    Args:
        file_metadata: Dict with file info
        output_dir: Output directory
        method: Resampling method
        max_fill_gap: Max gap for forward-fill

    Returns:
        Processing statistics
    """
    input_file = file_metadata["path"]
    output_file = generate_output_path(file_metadata, output_dir)

    # Check if already exists
    if os.path.exists(output_file):
        logger.info(f"✓ {output_file} already exists, skipping")
        return {
            "input_file": input_file,
            "output_file": output_file,
            "skipped": True,
        }

    try:
        stats = resample_file(
            input_file=input_file,
            output_file=output_file,
            method=method,
            max_fill_gap=max_fill_gap,
            verbose=False,
        )
        stats["skipped"] = False
        return stats
    except Exception as e:
        logger.error(f"✗ Failed to process {input_file}: {e}")
        return {
            "input_file": input_file,
            "error": str(e),
            "skipped": False,
        }


def main() -> None:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Batch resample Binance funding rate data to 1-second intervals",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process all files with 5 workers
  uv run python batch_resample_funding_rates.py \\
      --input-dir data/raw/binance_funding_rates \\
      --output-dir data/processed/binance_funding_rates_1s \\
      --workers 5

  # With forward-fill and resume
  uv run python batch_resample_funding_rates.py \\
      --input-dir data/raw/binance_funding_rates \\
      --output-dir data/processed/binance_funding_rates_1s \\
      --method forward_fill \\
      --max-fill-gap 60 \\
      --workers 5 \\
      --resume
        """,
    )

    parser.add_argument(
        "--input-dir",
        required=True,
        help="Input directory containing raw funding rate data",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Output directory for 1-second resampled data",
    )
    parser.add_argument(
        "--method",
        choices=["last", "forward_fill"],
        default="last",
        help="Resampling method (default: last)",
    )
    parser.add_argument(
        "--max-fill-gap",
        type=int,
        default=None,
        help="Maximum gap to forward-fill in seconds (only with forward_fill)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=5,
        help="Number of parallel workers (default: 5, use 1 for sequential)",
    )
    parser.add_argument(
        "--checkpoint-dir",
        default="./checkpoints",
        help="Checkpoint directory (default: ./checkpoints)",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from checkpoint (skip already processed files)",
    )

    args = parser.parse_args()

    # Validate input directory
    if not os.path.exists(args.input_dir):
        logger.error(f"Input directory not found: {args.input_dir}")
        sys.exit(1)

    # Create output and checkpoint directories
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.checkpoint_dir, exist_ok=True)

    # Discover input files
    files = discover_input_files(args.input_dir)
    if not files:
        logger.error("No Parquet files found in input directory")
        sys.exit(1)

    # Load checkpoint if resuming
    checkpoint_file = os.path.join(args.checkpoint_dir, "batch_resample_progress.json")
    completed_files = set()
    stats = {"total_processed": 0, "total_input_rows": 0, "total_output_rows": 0}

    if args.resume:
        completed_files, stats = load_checkpoint(checkpoint_file)

    # Filter out already completed files
    files_to_process = [f for f in files if f["path"] not in completed_files]
    logger.info(f"Files to process: {len(files_to_process)} (skipping {len(completed_files)} already done)")

    if not files_to_process:
        logger.info("All files already processed!")
        return

    # Process files
    start_time = time.time()
    completed_tasks = 0

    if args.workers == 1:
        # Sequential processing
        for file_metadata in files_to_process:
            result = process_single_file(
                file_metadata,
                args.output_dir,
                args.method,
                args.max_fill_gap,
            )

            # Update stats
            if not result.get("skipped", False) and "error" not in result:
                stats["total_processed"] += 1
                stats["total_input_rows"] += result.get("input_rows", 0)
                stats["total_output_rows"] += result.get("output_rows", 0)

            # Mark as completed
            completed_files.add(file_metadata["path"])
            completed_tasks += 1

            # Save checkpoint every 10 files
            if completed_tasks % 10 == 0:
                save_checkpoint(checkpoint_file, completed_files, stats)

            # Progress update
            progress = completed_tasks / len(files_to_process) * 100
            elapsed = time.time() - start_time
            rate = completed_tasks / elapsed if elapsed > 0 else 0
            eta = (len(files_to_process) - completed_tasks) / rate if rate > 0 else 0

            logger.info(
                f"Progress: {completed_tasks}/{len(files_to_process)} ({progress:.1f}%) | "
                f"Rate: {rate:.2f} files/sec | ETA: {eta / 60:.1f} min"
            )

    else:
        # Parallel processing
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            futures = {}
            for file_metadata in files_to_process:
                future = executor.submit(
                    process_single_file,
                    file_metadata,
                    args.output_dir,
                    args.method,
                    args.max_fill_gap,
                )
                futures[future] = file_metadata

            for future in as_completed(futures):
                file_metadata = futures[future]
                try:
                    result = future.result()

                    # Update stats
                    if not result.get("skipped", False) and "error" not in result:
                        stats["total_processed"] += 1
                        stats["total_input_rows"] += result.get("input_rows", 0)
                        stats["total_output_rows"] += result.get("output_rows", 0)

                    # Mark as completed
                    completed_files.add(file_metadata["path"])
                    completed_tasks += 1

                    # Save checkpoint every 10 files
                    if completed_tasks % 10 == 0:
                        save_checkpoint(checkpoint_file, completed_files, stats)

                    # Progress update
                    progress = completed_tasks / len(files_to_process) * 100
                    elapsed = time.time() - start_time
                    rate = completed_tasks / elapsed if elapsed > 0 else 0
                    eta = (len(files_to_process) - completed_tasks) / rate if rate > 0 else 0

                    logger.info(
                        f"Progress: {completed_tasks}/{len(files_to_process)} ({progress:.1f}%) | "
                        f"Rate: {rate:.2f} files/sec | ETA: {eta / 60:.1f} min"
                    )

                except Exception as e:
                    logger.error(f"Task failed for {file_metadata['path']}: {e}")

    # Final checkpoint
    save_checkpoint(checkpoint_file, completed_files, stats)

    # Summary
    elapsed_total = time.time() - start_time
    logger.info("\n" + "=" * 60)
    logger.info("BATCH RESAMPLING COMPLETE")
    logger.info("=" * 60)
    logger.info(f"Total files processed: {stats['total_processed']}")
    logger.info(f"Total input rows: {stats['total_input_rows']:,}")
    logger.info(f"Total output rows: {stats['total_output_rows']:,}")
    logger.info(f"Compression: {stats['total_output_rows'] / stats['total_input_rows'] * 100:.1f}% of original")
    logger.info(f"Total time: {elapsed_total / 60:.1f} minutes")
    logger.info(f"Output directory: {args.output_dir}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
