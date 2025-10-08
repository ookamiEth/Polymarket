#!/usr/bin/env python3
"""
Download Deribit Options Data and Convert to Parquet

Complete pipeline:
1. Download CSV from Tardis (with proper completion check)
2. Convert to Parquet with vectorized operations
3. Ready for fast analysis

FULLY VECTORIZED - NO FOR LOOPS
"""

from tardis_dev import datasets
import polars as pl
from datetime import datetime
import os
import time


def download_deribit_options_safe(date="2025-10-01", download_dir="./datasets_deribit_options"):
    """
    Download with proper completion checking.
    """
    print("=" * 80)
    print(f"DOWNLOADING DERIBIT OPTIONS DATA FOR {date}")
    print("=" * 80)
    print()

    filename = f"deribit_options_chain_{date}_OPTIONS.csv.gz"
    filepath = os.path.join(download_dir, filename)

    # Remove any incomplete files
    for file in os.listdir(download_dir) if os.path.exists(download_dir) else []:
        if 'unconfirmed' in file or file.startswith('.'):
            old_file = os.path.join(download_dir, file)
            print(f"Removing incomplete/temp file: {old_file}")
            try:
                os.remove(old_file)
            except:
                pass

    # Check if complete file exists
    if os.path.exists(filepath):
        size_gb = os.path.getsize(filepath) / 1e9
        print(f"✓ File already exists: {filepath}")
        print(f"  Size: {size_gb:.2f} GB")

        # Quick validation
        try:
            df_test = pl.scan_csv(filepath).head(10).collect()
            print(f"  ✓ File appears valid (tested first 10 rows)")
            print()
            return filepath
        except Exception as e:
            print(f"  ✗ File appears corrupted: {e}")
            print(f"  Removing and re-downloading...")
            os.remove(filepath)

    print(f"Downloading from Tardis.dev...")
    print(f"Target: {download_dir}/{filename}")
    print(f"Using FREE ACCESS (October 1st = first day of month)")
    print()
    print("This may take 5-10 minutes for ~3GB of data...")
    print()

    start_time = time.time()

    # Download
    datasets.download(
        exchange="deribit",
        data_types=["options_chain"],
        from_date=date,
        to_date=date,
        symbols=["OPTIONS"],
        api_key=None,
        download_dir=download_dir
    )

    elapsed = time.time() - start_time

    # Verify download completed
    if not os.path.exists(filepath):
        print(f"\n✗ ERROR: Download did not complete properly")
        print(f"  Expected file: {filepath}")
        return None

    size_gb = os.path.getsize(filepath) / 1e9
    print(f"\n✓ Download complete!")
    print(f"  File: {filepath}")
    print(f"  Size: {size_gb:.2f} GB")
    print(f"  Time: {elapsed:.1f} seconds ({size_gb/(elapsed/60):.1f} MB/min)")
    print()

    return filepath


def convert_to_parquet(csv_path: str, target_date: datetime):
    """
    Convert CSV to Parquet with vectorized operations.
    Time Complexity: O(n) single pass
    """
    parquet_path = csv_path.replace('.csv.gz', '.parquet')

    print("=" * 80)
    print("CSV → PARQUET CONVERSION (VECTORIZED)")
    print("=" * 80)
    print(f"Input:  {csv_path}")
    print(f"Output: {parquet_path}")
    print()

    if os.path.exists(parquet_path):
        size_gb = os.path.getsize(parquet_path) / 1e9
        print(f"✓ Parquet file already exists: {size_gb:.2f} GB")
        print()
        return parquet_path

    csv_size_gb = os.path.getsize(csv_path) / 1e9
    print(f"Input CSV size: {csv_size_gb:.2f} GB")
    print()
    print("Converting with streaming pipeline (O(n) single pass)...")
    print("All operations are vectorized (no for loops)")
    print()

    start_time = time.time()

    # VECTORIZED PIPELINE
    df = (
        pl.scan_csv(csv_path)
        # Parse symbol components vectorially
        .with_columns([
            pl.col('symbol').str.split('-').alias('parts')
        ])
        .with_columns([
            pl.col('parts').list.get(0).alias('underlying'),
            pl.col('parts').list.get(1).alias('expiry_str'),
            pl.col('parts').list.get(2).cast(pl.Float64, strict=False).alias('parsed_strike'),
            pl.when(pl.col('parts').list.len() >= 4)
              .then(
                  pl.when(pl.col('parts').list.get(3) == 'C')
                    .then(pl.lit('call'))
                    .otherwise(pl.lit('put'))
              )
              .alias('parsed_option_type')
        ])
        .with_columns([
            pl.col('expiry_str')
              .str.to_uppercase()
              .str.strptime(pl.Date, '%d%b%y', strict=False)
              .alias('expiry_date')
        ])
        .with_columns([
            (pl.col('expiry_date').cast(pl.Datetime('us')) - pl.lit(target_date))
              .dt.total_days()
              .alias('days_to_expiry')
        ])
        .drop('parts')
    )

    # Write to Parquet (this executes the lazy pipeline)
    df.sink_parquet(
        parquet_path,
        compression='zstd',
        compression_level=3
    )

    elapsed = time.time() - start_time
    parquet_size_gb = os.path.getsize(parquet_path) / 1e9

    print(f"✓ Conversion complete!")
    print(f"  Time: {elapsed:.1f} seconds")
    print(f"  Output size: {parquet_size_gb:.2f} GB")
    print(f"  Compression: {csv_size_gb / parquet_size_gb:.1f}×")
    print(f"  Space saved: {csv_size_gb - parquet_size_gb:.2f} GB")
    print()

    return parquet_path


def main():
    """
    Complete download and conversion pipeline.
    """
    print("\n")
    print("╔" + "═" * 78 + "╗")
    print("║" + " " * 15 + "DERIBIT OPTIONS - DOWNLOAD & CONVERT PIPELINE" + " " * 18 + "║")
    print("║" + " " * 20 + "FULLY VECTORIZED - O(n) SINGLE PASS" + " " * 23 + "║")
    print("╚" + "═" * 78 + "╝")
    print("\n")

    DATE = "2025-10-01"
    TARGET_DATE = datetime(2025, 10, 1)
    DOWNLOAD_DIR = "./datasets_deribit_options"

    # Ensure directory exists
    os.makedirs(DOWNLOAD_DIR, exist_ok=True)

    # Step 1: Download
    print("STEP 1: DOWNLOAD")
    print("=" * 80)
    csv_path = download_deribit_options_safe(date=DATE, download_dir=DOWNLOAD_DIR)

    if not csv_path:
        print("\n✗ FAILED: Could not download data")
        return

    # Step 2: Convert
    print("\nSTEP 2: CONVERT TO PARQUET")
    print("=" * 80)
    parquet_path = convert_to_parquet(csv_path, TARGET_DATE)

    # Summary
    print("\n" + "=" * 80)
    print("PIPELINE COMPLETE!")
    print("=" * 80)
    print(f"""
✓ CSV file:     {csv_path}
✓ Parquet file: {parquet_path}

Next step: Run the analysis script with the Parquet file!
  uv run python get_btc_options_oct1_2025_optimized.py
    """)
    print("=" * 80)


if __name__ == "__main__":
    main()
