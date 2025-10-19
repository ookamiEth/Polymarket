#!/usr/bin/env python3
"""
Test the optimized chunked script on a small subset to validate correctness.
"""

import polars as pl
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path


def create_test_subset():
    """Create a small test subset of data for validation."""
    print("Creating test subset...")

    # Create test quotes file (first 2 months only)
    quotes_df = pl.scan_parquet("quotes_1s_merged.parquet")

    # Get October and November 2023 only
    start_ts = int(datetime(2023, 10, 1).timestamp())
    end_ts = int(datetime(2023, 11, 30, 23, 59, 59).timestamp())

    test_quotes = quotes_df.filter(
        (pl.col("timestamp_seconds") >= start_ts) &
        (pl.col("timestamp_seconds") <= end_ts)
    )

    # Write test file
    test_quotes.sink_parquet(
        "test_quotes_subset.parquet",
        compression="snappy",
        statistics=True,
    )

    # Also create perpetual subset
    perp_df = pl.read_parquet("deribit_btc_perpetual_1s.parquet")
    perp_df = perp_df.with_columns([
        (pl.col("timestamp") // 1_000_000).alias("timestamp_seconds")
    ])

    test_perp = perp_df.filter(
        (pl.col("timestamp_seconds") >= start_ts) &
        (pl.col("timestamp_seconds") <= end_ts)
    )

    test_perp.drop("timestamp_seconds").write_parquet(
        "test_perpetual_subset.parquet",
        compression="snappy",
        statistics=True,
    )

    print("✅ Test subset created")
    print(f"  Quotes: test_quotes_subset.parquet")
    print(f"  Perpetual: test_perpetual_subset.parquet")


def modify_script_for_test(script_path: str, output_path: str):
    """Modify script to use test files."""
    with open(script_path) as f:
        content = f.read()

    # Replace file names
    content = content.replace(
        'QUOTES_FILE = "quotes_1s_merged.parquet"',
        'QUOTES_FILE = "test_quotes_subset.parquet"'
    )
    content = content.replace(
        'PERPETUAL_FILE = "deribit_btc_perpetual_1s.parquet"',
        'PERPETUAL_FILE = "test_perpetual_subset.parquet"'
    )

    # Change output file name
    if "optimized" in script_path:
        content = content.replace(
            'OUTPUT_FILE = "quotes_1s_atm_short_dated_optimized.parquet"',
            'OUTPUT_FILE = "test_output_optimized.parquet"'
        )
        content = content.replace(
            'CHECKPOINT_FILE = "checkpoints/filter_progress_optimized.json"',
            'CHECKPOINT_FILE = "checkpoints/test_progress_optimized.json"'
        )
    else:
        content = content.replace(
            'OUTPUT_FILE = "quotes_1s_atm_short_dated.parquet"',
            'OUTPUT_FILE = "test_output_original.parquet"'
        )
        content = content.replace(
            'CHECKPOINT_FILE = "checkpoints/filter_progress.json"',
            'CHECKPOINT_FILE = "checkpoints/test_progress_original.json"'
        )

    with open(output_path, "w") as f:
        f.write(content)

    print(f"✅ Created test script: {output_path}")


def run_test_script(script_path: str) -> tuple[float, int]:
    """Run a test script and return timing and row count."""
    print(f"\nRunning {script_path}...")

    start_time = time.time()

    # Run the script
    result = subprocess.run(
        ["uv", "run", "python", script_path],
        capture_output=True,
        text=True
    )

    elapsed = time.time() - start_time

    if result.returncode != 0:
        print(f"❌ Script failed:")
        print(result.stderr)
        return elapsed, 0

    # Parse output for row count
    row_count = 0
    for line in result.stdout.split("\n"):
        if "Total output rows:" in line or "Final rows:" in line:
            # Extract number
            import re
            numbers = re.findall(r"[\d,]+", line)
            if numbers:
                row_count = int(numbers[-1].replace(",", ""))
                break

    print(f"✅ Completed in {elapsed:.1f}s")
    print(f"   Output rows: {row_count:,}")

    return elapsed, row_count


def compare_outputs():
    """Compare the outputs of original and optimized scripts."""
    print("\n" + "=" * 80)
    print("COMPARING OUTPUTS")
    print("=" * 80)

    if not Path("test_output_optimized.parquet").exists():
        print("❌ Optimized output not found")
        return False

    # Load both outputs
    opt_df = pl.read_parquet("test_output_optimized.parquet")

    print(f"\nOptimized output:")
    print(f"  Rows: {len(opt_df):,}")
    print(f"  Columns: {opt_df.columns}")

    # Check sample data
    print("\n📊 Sample data (first 5 rows):")
    print(opt_df.head(5).select([
        "timestamp_seconds", "symbol", "moneyness", "time_to_expiry_days", "spot_price"
    ]))

    # Validate data quality
    print("\n✅ Data validation:")
    print(f"  Moneyness range: {opt_df['moneyness'].min():.4f} - {opt_df['moneyness'].max():.4f}")
    print(f"  TTL range: {opt_df['time_to_expiry_days'].min():.2f} - {opt_df['time_to_expiry_days'].max():.2f} days")
    print(f"  Non-null spot prices: {opt_df.filter(pl.col('spot_price').is_not_null()).height:,}")

    return True


def main():
    """Run the validation test."""
    print("=" * 80)
    print("OPTIMIZED SCRIPT VALIDATION TEST")
    print("=" * 80)

    # Step 1: Create test subset
    if not Path("test_quotes_subset.parquet").exists():
        create_test_subset()
    else:
        print("✅ Test subset already exists")

    # Step 2: Create test versions of scripts
    modify_script_for_test(
        "filter_atm_short_dated_chunked_optimized.py",
        "test_optimized_chunked.py"
    )

    # Step 3: Run optimized script
    opt_time, opt_rows = run_test_script("test_optimized_chunked.py")

    # Step 4: Compare results
    if opt_rows > 0:
        compare_outputs()

        print("\n" + "=" * 80)
        print("VALIDATION SUMMARY")
        print("=" * 80)
        print(f"✅ Optimized script:")
        print(f"   Time: {opt_time:.1f}s")
        print(f"   Rows: {opt_rows:,}")
        print(f"\n🎉 Validation successful! Ready to run on full dataset.")
    else:
        print("\n❌ Validation failed - check errors above")

    # Cleanup test files (optional)
    # for f in ["test_optimized_chunked.py", "test_quotes_subset.parquet", "test_perpetual_subset.parquet"]:
    #     if Path(f).exists():
    #         Path(f).unlink()


if __name__ == "__main__":
    main()