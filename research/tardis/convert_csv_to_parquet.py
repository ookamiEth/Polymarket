#!/usr/bin/env python3
"""
Convert Deribit Options CSV to Parquet Format

This script converts the large CSV.gz file to Parquet format for:
- 10-50× faster loading times
- Better compression
- Columnar storage (efficient for analytics)

FULLY VECTORIZED - NO FOR LOOPS
Time Complexity: O(n) single pass with streaming
"""

import polars as pl
from datetime import datetime
import os


def convert_csv_to_parquet(
    csv_path: str,
    parquet_path: str = None,
    target_date: datetime = None
):
    """
    Convert CSV to Parquet with vectorized symbol parsing.

    Time Complexity: O(n) - single pass through data
    Space Complexity: O(1) - uses streaming mode

    Args:
        csv_path: Path to input CSV.gz file
        parquet_path: Path for output Parquet file (auto-generated if None)
        target_date: Date to calculate days_to_expiry (default: 2025-10-01)
    """

    if parquet_path is None:
        parquet_path = csv_path.replace('.csv.gz', '.parquet')

    if target_date is None:
        target_date = datetime(2025, 10, 1)

    print("=" * 80)
    print("CSV → PARQUET CONVERSION (VECTORIZED)")
    print("=" * 80)
    print(f"Input:  {csv_path}")
    print(f"Output: {parquet_path}")
    print(f"Target date for days_to_expiry: {target_date.date()}")
    print()

    # Check if input exists
    if not os.path.exists(csv_path):
        print(f"ERROR: Input file not found: {csv_path}")
        return None

    csv_size_gb = os.path.getsize(csv_path) / 1e9
    print(f"Input CSV size: {csv_size_gb:.2f} GB")
    print()

    # Check if output already exists
    if os.path.exists(parquet_path):
        parquet_size_gb = os.path.getsize(parquet_path) / 1e9
        print(f"Output Parquet already exists: {parquet_path}")
        print(f"Parquet size: {parquet_size_gb:.2f} GB")
        print(f"Compression ratio: {csv_size_gb / parquet_size_gb:.1f}×")
        print("\nSkipping conversion. Delete the Parquet file to reconvert.")
        return parquet_path

    print("[STEP 1] Scanning CSV with lazy evaluation...")
    print("This doesn't load data into memory yet.")

    # Use scan_csv for lazy evaluation - O(1) operation
    df = pl.scan_csv(csv_path)

    print("\n[STEP 2] Applying vectorized transformations...")
    print("All operations are lazy and will execute in a single streaming pass.")
    print()

    # VECTORIZED: Split symbol into parts
    # Time: O(n) with compiled string operations
    print("  → Parsing symbols (vectorized string split)")
    df = df.with_columns([
        pl.col('symbol').str.split('-').alias('symbol_parts')
    ])

    # VECTORIZED: Extract components
    # Time: O(n) with list operations
    print("  → Extracting components (vectorized list indexing)")
    df = df.with_columns([
        pl.col('symbol_parts').list.get(0).alias('underlying'),
        pl.col('symbol_parts').list.get(1).alias('expiry_str'),
        pl.col('symbol_parts').list.get(2).cast(pl.Float64, strict=False).alias('parsed_strike'),
        pl.when(pl.col('symbol_parts').list.len() >= 4)
          .then(
              pl.when(pl.col('symbol_parts').list.get(3) == 'C')
                .then(pl.lit('call'))
                .otherwise(pl.lit('put'))
          )
          .otherwise(pl.lit(None))
          .alias('parsed_option_type')
    ])

    # VECTORIZED: Parse expiry date
    # Time: O(n) with compiled strptime
    print("  → Parsing expiry dates (vectorized strptime)")
    df = df.with_columns([
        pl.col('expiry_str')
          .str.to_uppercase()
          .str.strptime(pl.Date, '%d%b%y', strict=False)
          .alias('expiry_date')
    ])

    # VECTORIZED: Calculate days to expiry
    # Time: O(n) with vectorized datetime operations
    print("  → Calculating days to expiry (vectorized date arithmetic)")
    df = df.with_columns([
        (pl.col('expiry_date').cast(pl.Datetime('us')) - pl.lit(target_date))
          .dt.total_days()
          .alias('days_to_expiry')
    ])

    # Drop intermediate column
    df = df.drop('symbol_parts')

    print("\n[STEP 3] Writing to Parquet format (streaming)...")
    print("This executes all lazy operations in a single pass.")
    print("Using streaming sink for memory efficiency...")
    print()

    # Sink to parquet with streaming - executes the entire lazy query
    # Time: O(n) single pass through data
    # This is THE operation that actually processes the data
    df.sink_parquet(
        parquet_path,
        compression='zstd',  # Good compression ratio and speed
        compression_level=3   # Balanced compression (3 is fast, 22 is max)
    )

    print("✓ Conversion complete!")
    print()

    # Check output size
    parquet_size_gb = os.path.getsize(parquet_path) / 1e9
    print(f"Output Parquet size: {parquet_size_gb:.2f} GB")
    print(f"Compression ratio: {csv_size_gb / parquet_size_gb:.1f}×")
    print(f"Space saved: {csv_size_gb - parquet_size_gb:.2f} GB")
    print()

    # Verify by reading schema
    print("[VERIFICATION] Checking Parquet schema...")
    schema = pl.scan_parquet(parquet_path).collect_schema()
    print(f"Columns: {len(schema)}")
    print(f"Schema: {list(schema.keys())[:10]}... (showing first 10)")
    print()

    print("=" * 80)
    print("CONVERSION SUMMARY")
    print("=" * 80)
    print(f"""
✓ Input:  {csv_path} ({csv_size_gb:.2f} GB)
✓ Output: {parquet_path} ({parquet_size_gb:.2f} GB)
✓ Compression: {csv_size_gb / parquet_size_gb:.1f}× smaller
✓ Time complexity: O(n) single pass
✓ All operations vectorized (no for loops)

Next step: Use the Parquet file in your analysis script for 10-50× faster loading!
    """)
    print("=" * 80)

    return parquet_path


def main():
    """
    Main conversion function.
    """
    print("\n")
    print("╔" + "═" * 78 + "╗")
    print("║" + " " * 20 + "CSV → PARQUET CONVERTER [OPTIMIZED]" + " " * 23 + "║")
    print("║" + " " * 15 + "FULLY VECTORIZED - O(n) SINGLE PASS STREAMING" + " " * 18 + "║")
    print("╚" + "═" * 78 + "╝")
    print()

    # Configuration
    CSV_PATH = "./datasets_deribit_options/deribit_options_chain_2025-10-01_OPTIONS.csv.gz"
    PARQUET_PATH = "./datasets_deribit_options/deribit_options_chain_2025-10-01_OPTIONS.parquet"
    TARGET_DATE = datetime(2025, 10, 1)

    # Convert
    result = convert_csv_to_parquet(
        csv_path=CSV_PATH,
        parquet_path=PARQUET_PATH,
        target_date=TARGET_DATE
    )

    if result:
        print("\n✓ SUCCESS! You can now use the Parquet file for fast analysis.")
        print(f"\nParquet file: {result}")
    else:
        print("\n✗ FAILED! Check error messages above.")


if __name__ == "__main__":
    main()
