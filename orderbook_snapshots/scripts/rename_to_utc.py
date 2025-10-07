#!/usr/bin/env python3
"""
Rename Parquet Files from CEST to UTC

This script renames all parquet files in data/raw/ from CEST (UTC+2) timestamps
to UTC timestamps. It handles directory moves when date boundaries are crossed.

The timestamps INSIDE the parquet files are already in UTC - this script only
fixes the filenames to match.

Usage:
    uv run python scripts/rename_to_utc.py [--dry-run]
"""

import argparse
from pathlib import Path
from datetime import datetime, timedelta
import sys


def parse_filename(filename: str) -> datetime | None:
    """
    Parse filename like 'orderbook_20251007_1700.parquet'
    Returns datetime in CEST (local time as written in filename)
    """
    try:
        stem = filename.replace('.parquet', '')
        parts = stem.split('_')
        if len(parts) != 3 or parts[0] != 'orderbook':
            return None

        date_str = parts[1]  # YYYYMMDD
        time_str = parts[2]  # HHMM

        year = int(date_str[0:4])
        month = int(date_str[4:6])
        day = int(date_str[6:8])
        hour = int(time_str[0:2])
        minute = int(time_str[2:4])

        return datetime(year, month, day, hour, minute)
    except (ValueError, IndexError):
        return None


def convert_cest_to_utc(cest_dt: datetime) -> datetime:
    """Convert CEST datetime to UTC by subtracting 2 hours"""
    return cest_dt - timedelta(hours=2)


def generate_utc_path(utc_dt: datetime, base_dir: Path) -> Path:
    """Generate the new UTC-based file path"""
    year = utc_dt.strftime("%Y")
    month = utc_dt.strftime("%m")
    day = utc_dt.strftime("%d")
    filename = f"orderbook_{utc_dt.strftime('%Y%m%d_%H%M')}.parquet"

    return base_dir / year / month / day / filename


def find_all_parquet_files(base_dir: Path) -> list[Path]:
    """Find all parquet files in data/raw/ hierarchy"""
    return sorted(base_dir.glob("**/orderbook_*.parquet"))


def main():
    parser = argparse.ArgumentParser(description="Rename parquet files from CEST to UTC")
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Show what would be renamed without actually renaming'
    )
    args = parser.parse_args()

    # Configuration
    base_dir = Path("data/raw")
    if not base_dir.exists():
        print(f"Error: {base_dir} does not exist")
        sys.exit(1)

    # Find all files
    files = find_all_parquet_files(base_dir)
    print(f"Found {len(files)} parquet files")
    print()

    if args.dry_run:
        print("=" * 80)
        print("DRY RUN MODE - No files will be renamed")
        print("=" * 80)
        print()

    # Process each file
    renamed_count = 0
    error_count = 0

    for old_path in files:
        filename = old_path.name

        # Parse CEST timestamp from filename
        cest_dt = parse_filename(filename)
        if cest_dt is None:
            print(f"⚠️  Skipping {old_path} (cannot parse filename)")
            error_count += 1
            continue

        # Convert to UTC
        utc_dt = convert_cest_to_utc(cest_dt)

        # Generate new path
        new_path = generate_utc_path(utc_dt, base_dir)

        # Check if rename is needed
        if old_path == new_path:
            print(f"✓ {old_path.relative_to(base_dir)} (already correct)")
            continue

        # Show the rename operation
        print(f"{'[DRY RUN] ' if args.dry_run else ''}Renaming:")
        print(f"  From: {old_path.relative_to(base_dir)}")
        print(f"        {cest_dt.strftime('%Y-%m-%d %H:%M')} CEST")
        print(f"  To:   {new_path.relative_to(base_dir)}")
        print(f"        {utc_dt.strftime('%Y-%m-%d %H:%M')} UTC")

        if not args.dry_run:
            try:
                # Create target directory if needed
                new_path.parent.mkdir(parents=True, exist_ok=True)

                # Rename/move the file
                old_path.rename(new_path)
                print(f"  ✓ Success")
                renamed_count += 1
            except Exception as e:
                print(f"  ✗ Error: {e}")
                error_count += 1
        else:
            renamed_count += 1

        print()

    # Summary
    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"Total files found: {len(files)}")
    print(f"Files {'to be ' if args.dry_run else ''}renamed: {renamed_count}")
    print(f"Errors: {error_count}")

    if args.dry_run:
        print()
        print("Run without --dry-run to actually rename the files")


if __name__ == "__main__":
    main()
