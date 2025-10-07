#!/usr/bin/env python3
"""
Validate Parquet files - check structure, data quality, and contents
"""

import polars as pl
from pathlib import Path

def validate_parquet_files():
    """Validate all generated parquet files"""

    data_dir = Path("/Users/lgierhake/Documents/ETH/BT/top_traders/data/parquet")

    print("="*80)
    print(" PARQUET FILE VALIDATION")
    print("="*80)

    periods = ['day', 'week', 'month', 'all']

    for period in periods:
        print(f"\n{period.upper()}")
        print("-"*80)

        period_dir = data_dir / period
        if not period_dir.exists():
            print(f"✗ Directory not found: {period_dir}")
            continue

        parquet_files = list(period_dir.glob("*.parquet"))
        if not parquet_files:
            print(f"✗ No parquet files found")
            continue

        # Read the most recent file
        parquet_file = sorted(parquet_files)[-1]
        print(f"File: {parquet_file.name}")

        df = pl.read_parquet(parquet_file)

        # Basic stats
        print(f"\nBasic Statistics:")
        print(f"  Rows: {df.height:,}")
        print(f"  Columns: {df.width}")
        print(f"  File size: {parquet_file.stat().st_size / 1024:.1f} KB")

        # Schema
        print(f"\nSchema:")
        for col in df.columns:
            dtype = df[col].dtype
            null_count = df[col].null_count()
            print(f"  {col:30} {str(dtype):20} (nulls: {null_count})")

        # Data quality checks
        print(f"\nData Quality:")

        # Check for nulls in critical fields
        critical_fields = ['user_address', 'username', 'volume_usd', 'pnl_usd']
        for field in critical_fields:
            null_count = df[field].null_count()
            if null_count > 0:
                print(f"  ⚠ {field}: {null_count} nulls")
            else:
                print(f"  ✓ {field}: no nulls")

        # Check data ranges
        print(f"\nData Ranges:")
        print(f"  Volume: ${df['volume_usd'].min():,.2f} to ${df['volume_usd'].max():,.2f}")
        print(f"  P&L: ${df['pnl_usd'].min():,.2f} to ${df['pnl_usd'].max():,.2f}")

        # Check rankings
        print(f"\nRankings:")
        print(f"  Rank by P&L range: {df['rank_by_pnl'].min()} to {df['rank_by_pnl'].max()}")
        print(f"  Rank by Volume range: {df['rank_by_volume'].min()} to {df['rank_by_volume'].max()}")

        # Top 5 preview
        print(f"\nTop 5 by Volume:")
        top_5 = df.sort("rank_by_volume").head(5)
        for row in top_5.iter_rows(named=True):
            print(f"  {row['rank_by_volume']:3}. {row['username'][:30]:30} ${row['volume_usd']:>15,.2f}")

        print(f"\nTop 5 by P&L:")
        top_5_pnl = df.sort("rank_by_pnl").head(5)
        for row in top_5_pnl.iter_rows(named=True):
            print(f"  {row['rank_by_pnl']:3}. {row['username'][:30]:30} ${row['pnl_usd']:>15,.2f}")

    print(f"\n{'='*80}")
    print(" VALIDATION COMPLETE")
    print(f"{'='*80}")

if __name__ == "__main__":
    validate_parquet_files()
