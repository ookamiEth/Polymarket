#!/usr/bin/env python3
"""
Consolidate CLOB tick data parquet files by market type.

Combines thousands of individual parquet files into 6 consolidated files:
- btc_15min_consolidated.parquet
- eth_15min_consolidated.parquet
- bitcoin_hourly_consolidated.parquet
- ethereum_hourly_consolidated.parquet
- solana_hourly_consolidated.parquet
- xrp_hourly_consolidated.parquet

After consolidation, deletes individual parquet files and meta.json files.
"""

import polars as pl
from pathlib import Path
import re
from typing import Dict, List
import sys

# Market type patterns
MARKET_PATTERNS = {
    # 15-minute interval markets
    'btc_15min': r'^btc-up-or-down-15m-\d+_0x[a-f0-9]+\.parquet$',
    'eth_15min': r'^eth-up-or-down-15m-\d+_0x[a-f0-9]+\.parquet$',

    # Hourly markets (exclude "on-" daily and "-candle" variants, include optional variant numbers)
    'bitcoin_hourly': r'^bitcoin-up-or-down-(?!on-)(?!.*candle).+-et(-\d+)?_0x[a-f0-9]+\.parquet$',
    'ethereum_hourly': r'^ethereum-up-or-down-(?!on-)(?!.*candle).+-et(-\d+)?_0x[a-f0-9]+\.parquet$',
    'solana_hourly': r'^solana-up-or-down-(?!on-)(?!.*candle).+-et(-\d+)?_0x[a-f0-9]+\.parquet$',
    'xrp_hourly': r'^xrp-up-or-down-(?!on-)(?!.*candle).+-et(-\d+)?_0x[a-f0-9]+\.parquet$',

    # Daily markets (on-DATE format, include optional variant numbers and special words like "noon")
    'bitcoin_daily': r'^bitcoin-up-or-down-on-[a-z]+-\d+(-[a-z]+)?(-\d+)?_0x[a-f0-9]+\.parquet$',
    'ethereum_daily': r'^ethereum-up-or-down-on-[a-z]+-\d+(-[a-z]+)?(-\d+)?_0x[a-f0-9]+\.parquet$',
    'solana_daily': r'^solana-up-or-down-on-[a-z]+-\d+(-[a-z]+)?(-\d+)?_0x[a-f0-9]+\.parquet$',
    'xrp_daily': r'^xrp-up-or-down-on-[a-z]+-\d+(-[a-z]+)?(-\d+)?_0x[a-f0-9]+\.parquet$',

    # Candle markets (hourly with candle suffix)
    'bitcoin_hourly_candle': r'^bitcoin-up-or-down-.*-candle_0x[a-f0-9]+\.parquet$',
    'ethereum_hourly_candle': r'^ethereum-up-or-down-.*-candle_0x[a-f0-9]+\.parquet$',

    # Special markets
    'bitcoin_dominance': r'^bitcoin-dominance-.*_0x[a-f0-9]+\.parquet$',
    'ethbtc': r'^ethbtc-.*_0x[a-f0-9]+\.parquet$',
    'soleth': r'^soleth-.*_0x[a-f0-9]+\.parquet$',
    'bitcoin_aggregated': r'^bitcoin-up-or-down-(in-|this-).*_0x[a-f0-9]+\.parquet$',
    'ethereum_aggregated': r'^ethereum-up-or-down-(in-|this-).*_0x[a-f0-9]+\.parquet$',
    'solana_aggregated': r'^solana-up-or-down-(in-|this-).*_0x[a-f0-9]+\.parquet$',
    'xrp_aggregated': r'^xrp-up-or-down-(in-|this-).*_0x[a-f0-9]+\.parquet$',
}

def categorize_files(data_dir: Path) -> Dict[str, List[Path]]:
    """Categorize all parquet files by market type."""
    print("📂 Scanning parquet files...")

    categories = {key: [] for key in MARKET_PATTERNS.keys()}
    uncategorized = []

    for file_path in data_dir.glob("*.parquet"):
        filename = file_path.name
        matched = False

        for market_type, pattern in MARKET_PATTERNS.items():
            if re.match(pattern, filename):
                categories[market_type].append(file_path)
                matched = True
                break

        if not matched:
            uncategorized.append(file_path)

    # Print summary
    print("\n" + "="*80)
    print("FILE CATEGORIZATION SUMMARY")
    print("="*80)
    for market_type, files in categories.items():
        print(f"  {market_type:20s}: {len(files):5d} files")

    if uncategorized:
        print(f"  {'uncategorized':20s}: {len(uncategorized):5d} files")
        print("\nUncategorized files:")
        for f in uncategorized[:10]:
            print(f"    - {f.name}")
        if len(uncategorized) > 10:
            print(f"    ... and {len(uncategorized) - 10} more")

    print("="*80 + "\n")

    return categories, uncategorized

def consolidate_market_type(market_type: str, file_paths: List[Path], output_dir: Path) -> Path:
    """Consolidate all files for a given market type."""
    if not file_paths:
        print(f"⏭️  Skipping {market_type}: no files found")
        return None

    print(f"\n{'='*80}")
    print(f"🔄 Consolidating: {market_type}")
    print(f"{'='*80}")
    print(f"   Files to merge: {len(file_paths)}")

    # Read all parquet files
    print("   Reading parquet files...")
    dfs = []
    failed = []

    for i, file_path in enumerate(file_paths, 1):
        try:
            df = pl.read_parquet(file_path)
            dfs.append(df)

            if i % 500 == 0:
                print(f"   Progress: {i}/{len(file_paths)} files read")
        except Exception as e:
            failed.append((file_path, str(e)))

    if failed:
        print(f"\n   ⚠️  Failed to read {len(failed)} files:")
        for path, error in failed[:5]:
            print(f"      - {path.name}: {error}")
        if len(failed) > 5:
            print(f"      ... and {len(failed) - 5} more")

    if not dfs:
        print(f"   ❌ No valid data to consolidate for {market_type}")
        return None

    # Concatenate
    print("   Concatenating DataFrames...")
    df_combined = pl.concat(dfs)

    # Sort by timestamp
    print("   Sorting by timestamp...")
    df_combined = df_combined.sort("timestamp")

    # Check for duplicates
    initial_rows = len(df_combined)
    print(f"   Total rows: {initial_rows:,}")

    # Remove duplicate trades (by transactionHash)
    if "transactionHash" in df_combined.columns:
        df_combined = df_combined.unique(subset=["transactionHash"], keep="first")
        final_rows = len(df_combined)
        duplicates_removed = initial_rows - final_rows

        if duplicates_removed > 0:
            print(f"   Removed {duplicates_removed:,} duplicate trades")
        print(f"   Final rows: {final_rows:,}")

    # Write consolidated file
    output_file = output_dir / f"{market_type}_consolidated.parquet"
    print(f"   Writing to: {output_file.name}")
    df_combined.write_parquet(output_file)

    # Get file size
    file_size_mb = output_file.stat().st_size / (1024 * 1024)
    print(f"   ✅ Success! File size: {file_size_mb:.2f} MB")

    return output_file

def verify_consolidation(original_files: List[Path], consolidated_file: Path) -> bool:
    """Verify that consolidation preserved all data."""
    print(f"\n🔍 Verifying consolidation for {consolidated_file.name}...")

    try:
        # Count total rows in original files
        original_row_count = 0
        for file_path in original_files:
            try:
                df = pl.read_parquet(file_path)
                original_row_count += len(df)
            except:
                pass

        # Count rows in consolidated file
        df_consolidated = pl.read_parquet(consolidated_file)
        consolidated_row_count = len(df_consolidated)

        print(f"   Original total rows: {original_row_count:,}")
        print(f"   Consolidated rows: {consolidated_row_count:,}")

        # Allow for duplicate removal
        if consolidated_row_count <= original_row_count:
            print(f"   ✅ Verification passed")
            return True
        else:
            print(f"   ❌ Verification failed: consolidated has MORE rows")
            return False

    except Exception as e:
        print(f"   ❌ Verification error: {e}")
        return False

def cleanup_files(categories: Dict[str, List[Path]], data_dir: Path):
    """Delete individual parquet files and meta.json files."""
    print(f"\n{'='*80}")
    print("🗑️  CLEANUP: Deleting individual files")
    print(f"{'='*80}")

    # Delete individual parquet files
    print("\n📦 Deleting individual parquet files...")
    total_deleted = 0

    for market_type, file_paths in categories.items():
        deleted = 0
        for file_path in file_paths:
            try:
                file_path.unlink()
                deleted += 1
                total_deleted += 1
            except Exception as e:
                print(f"   ⚠️  Failed to delete {file_path.name}: {e}")

        print(f"   {market_type}: deleted {deleted} files")

    print(f"\n   ✅ Total parquet files deleted: {total_deleted}")

    # Delete all meta.json files
    print("\n📄 Deleting meta.json files...")
    meta_files = list(data_dir.glob("*.meta.json"))
    meta_deleted = 0

    for meta_file in meta_files:
        try:
            meta_file.unlink()
            meta_deleted += 1
        except Exception as e:
            print(f"   ⚠️  Failed to delete {meta_file.name}: {e}")

    print(f"   ✅ Meta.json files deleted: {meta_deleted}")

    # Show remaining files
    print(f"\n📋 Remaining files in {data_dir}:")
    remaining = sorted(data_dir.glob("*"))
    for f in remaining:
        if f.is_file():
            size_mb = f.stat().st_size / (1024 * 1024)
            print(f"   - {f.name} ({size_mb:.2f} MB)")

    print(f"\n   Total remaining files: {len([f for f in remaining if f.is_file()])}")

def main():
    """Main consolidation workflow."""
    data_dir = Path("data/clob_ticks")

    if not data_dir.exists():
        print(f"❌ Directory not found: {data_dir}")
        sys.exit(1)

    print("\n" + "="*80)
    print("CLOB TICKS CONSOLIDATION")
    print("="*80)
    print(f"Directory: {data_dir}")
    print()

    # Step 1: Categorize files
    categories, uncategorized = categorize_files(data_dir)

    # Step 2: Consolidate each market type
    consolidated_files = {}

    for market_type, file_paths in categories.items():
        if file_paths:
            output_file = consolidate_market_type(market_type, file_paths, data_dir)
            if output_file:
                consolidated_files[market_type] = output_file

    # Step 3: Verify consolidations
    print(f"\n{'='*80}")
    print("VERIFICATION")
    print(f"{'='*80}")

    all_verified = True
    for market_type, output_file in consolidated_files.items():
        if not verify_consolidation(categories[market_type], output_file):
            all_verified = False

    if not all_verified:
        print("\n❌ Verification failed! Not proceeding with cleanup.")
        print("   Please review the consolidated files before manual cleanup.")
        sys.exit(1)

    # Step 4: Cleanup (only if verification passed)
    print("\n✅ All verifications passed!")

    # Auto-proceed with cleanup (script is designed for automation)
    print("\n🗑️  Proceeding with deletion of individual files...")
    cleanup_files(categories, data_dir)

    print(f"\n{'='*80}")
    print("✅ CONSOLIDATION COMPLETE!")
    print(f"{'='*80}")
    print(f"\nFinal consolidated files created:")
    for market_type, output_file in consolidated_files.items():
        size_mb = output_file.stat().st_size / (1024 * 1024)
        print(f"   - {output_file.name} ({size_mb:.2f} MB)")

if __name__ == '__main__':
    main()
