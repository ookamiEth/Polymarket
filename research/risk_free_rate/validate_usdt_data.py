#!/usr/bin/env python3
"""
Validate USDT Historical Lending Rates Data
Checks data completeness, quality, and identifies any anomalies
"""

import sys
from pathlib import Path
from datetime import datetime, timedelta, timezone

import polars as pl


def validate_usdt_data(file_path: str) -> None:
    """
    Comprehensive validation of USDT lending rates data

    Args:
        file_path: Path to the Parquet file
    """
    print("=" * 80)
    print("USDT HISTORICAL LENDING RATES - DATA VALIDATION")
    print("=" * 80)
    print()

    # Load the Parquet file
    print(f"Loading data from: {file_path}")
    df = pl.read_parquet(file_path)

    print(f"✓ Successfully loaded {len(df):,} records")
    print(f"  Schema: {list(df.columns)}")
    print(f"  Data types: {df.dtypes}")
    print()

    # 1. Check date range and completeness
    print("1. DATE RANGE VALIDATION")
    print("-" * 40)

    min_date = df["date"].min()
    max_date = df["date"].max()

    print(f"  First date: {min_date}")
    print(f"  Last date: {max_date}")

    # Check for expected date range
    expected_start = datetime(2023, 10, 1).date()
    expected_end = datetime(2025, 9, 30).date()

    if min_date == expected_start:
        print(f"  ✓ Start date matches expected: {expected_start}")
    else:
        print(f"  ⚠ Start date mismatch! Expected: {expected_start}, Got: {min_date}")

    if max_date == expected_end:
        print(f"  ✓ End date matches expected: {expected_end}")
    else:
        print(f"  ⚠ End date mismatch! Expected: {expected_end}, Got: {max_date}")

    # Check for missing dates
    date_range = pl.date_range(min_date, max_date, interval="1d", eager=True)
    expected_days = len(date_range)
    actual_days = len(df)

    print(f"\n  Expected days: {expected_days}")
    print(f"  Actual days: {actual_days}")

    if expected_days == actual_days:
        print("  ✓ No missing dates - data is complete!")
    else:
        missing = expected_days - actual_days
        print(f"  ⚠ Missing {missing} days of data")

        # Find specific missing dates
        actual_dates = set(df["date"].to_list())
        expected_dates = set(date_range.to_list())
        missing_dates = sorted(expected_dates - actual_dates)

        if missing_dates:
            print(f"  Missing dates: {missing_dates[:10]}")
            if len(missing_dates) > 10:
                print(f"  ... and {len(missing_dates) - 10} more")

    print()

    # 2. Check for duplicates
    print("2. DUPLICATE CHECK")
    print("-" * 40)

    duplicates = df.filter(pl.col("date").is_duplicated())
    if len(duplicates) == 0:
        print("  ✓ No duplicate dates found")
    else:
        print(f"  ⚠ Found {len(duplicates)} duplicate dates:")
        print(duplicates.select("date").head())

    print()

    # 3. Data quality checks
    print("3. DATA QUALITY VALIDATION")
    print("-" * 40)

    # Check for null values
    null_counts = df.null_count()
    has_nulls = False
    for col in df.columns:
        null_count = null_counts[col][0]
        if null_count > 0:
            has_nulls = True
            print(f"  ⚠ Column '{col}' has {null_count} null values")

    if not has_nulls:
        print("  ✓ No null values in any column")

    # Check rate ranges
    print("\n  Rate Range Validation:")
    print(f"    Lending APR: {df['lending_apr'].min():.3f}% - {df['lending_apr'].max():.3f}%")
    print(f"    Borrowing APR: {df['borrowing_apr'].min():.3f}% - {df['borrowing_apr'].max():.3f}%")

    # Flag potential outliers (rates > 20%)
    high_supply_rates = df.filter(pl.col("lending_apr") > 20)
    high_borrow_rates = df.filter(pl.col("borrowing_apr") > 30)

    if len(high_supply_rates) > 0:
        print(f"\n  ⚠ Found {len(high_supply_rates)} days with lending APR > 20%:")
        print(high_supply_rates.select(["date", "lending_apr", "utilization_rate"]).head())

    if len(high_borrow_rates) > 0:
        print(f"\n  ⚠ Found {len(high_borrow_rates)} days with borrowing APR > 30%:")
        print(high_borrow_rates.select(["date", "borrowing_apr", "utilization_rate"]).head())

    # Check utilization rate
    print("\n  Utilization Rate Validation:")
    print(f"    Range: {df['utilization_rate'].min():.1f}% - {df['utilization_rate'].max():.1f}%")

    invalid_utilization = df.filter(
        (pl.col("utilization_rate") < 0) | (pl.col("utilization_rate") > 100)
    )
    if len(invalid_utilization) == 0:
        print("    ✓ All utilization rates within valid range (0-100%)")
    else:
        print(f"    ⚠ Found {len(invalid_utilization)} invalid utilization rates")

    # Check TVL consistency
    print("\n  TVL Validation:")
    tvl_min = df["tvl_usd"].min()
    tvl_max = df["tvl_usd"].max()
    print(f"    Range: ${tvl_min:,.0f} - ${tvl_max:,.0f}")

    # Check for sudden TVL drops (> 50% in one day)
    df_with_prev = df.with_columns([
        pl.col("tvl_usd").shift(1).alias("prev_tvl")
    ])

    df_with_change = df_with_prev.with_columns([
        ((pl.col("tvl_usd") - pl.col("prev_tvl")) / pl.col("prev_tvl") * 100)
        .alias("tvl_change_pct")
    ])

    large_drops = df_with_change.filter(pl.col("tvl_change_pct") < -50)
    large_increases = df_with_change.filter(pl.col("tvl_change_pct") > 100)

    if len(large_drops) > 0:
        print(f"\n    ⚠ Found {len(large_drops)} days with TVL drops > 50%:")
        print(large_drops.select(["date", "tvl_usd", "prev_tvl", "tvl_change_pct"]).head())

    if len(large_increases) > 0:
        print(f"\n    ⚠ Found {len(large_increases)} days with TVL increases > 100%:")
        print(large_increases.select(["date", "tvl_usd", "prev_tvl", "tvl_change_pct"]).head())

    print()

    # 4. Statistical summary
    print("4. STATISTICAL SUMMARY")
    print("-" * 40)

    stats_df = df.select([
        pl.col("lending_apr").mean().alias("lending_apr_mean"),
        pl.col("lending_apr").median().alias("lending_apr_median"),
        pl.col("lending_apr").std().alias("lending_apr_std"),
        pl.col("borrowing_apr").mean().alias("borrowing_apr_mean"),
        pl.col("borrowing_apr").median().alias("borrowing_apr_median"),
        pl.col("borrowing_apr").std().alias("borrowing_apr_std"),
        pl.col("utilization_rate").mean().alias("util_mean"),
        pl.col("utilization_rate").median().alias("util_median"),
        pl.col("tvl_usd").mean().alias("tvl_mean"),
    ])

    stats = stats_df.row(0, named=True)

    print("  Lending APR Statistics:")
    print(f"    Mean: {stats['lending_apr_mean']:.3f}%")
    print(f"    Median: {stats['lending_apr_median']:.3f}%")
    print(f"    Std Dev: {stats['lending_apr_std']:.3f}%")

    print("\n  Borrowing APR Statistics:")
    print(f"    Mean: {stats['borrowing_apr_mean']:.3f}%")
    print(f"    Median: {stats['borrowing_apr_median']:.3f}%")
    print(f"    Std Dev: {stats['borrowing_apr_std']:.3f}%")

    print("\n  Utilization Rate:")
    print(f"    Mean: {stats['util_mean']:.1f}%")
    print(f"    Median: {stats['util_median']:.1f}%")

    print("\n  Total Value Locked:")
    print(f"    Mean: ${stats['tvl_mean']:,.0f}")

    print()

    # 5. Data consistency checks
    print("5. DATA CONSISTENCY CHECKS")
    print("-" * 40)

    # Check if dates are chronologically ordered
    is_sorted = df["date"].is_sorted()
    if is_sorted:
        print("  ✓ Dates are properly sorted chronologically")
    else:
        print("  ⚠ Dates are NOT properly sorted!")

    # Check relationship between lending and borrowing rates
    # Borrowing rate should generally be higher than lending rate
    df_rate_check = df.with_columns([
        (pl.col("borrowing_apr") > pl.col("lending_apr")).alias("borrow_higher")
    ])

    correct_rate_relationship = df_rate_check.filter(pl.col("borrow_higher")).height
    total = len(df_rate_check)
    pct = (correct_rate_relationship / total) * 100

    print(f"\n  Rate Relationship Check:")
    print(f"    Days with borrowing > lending: {correct_rate_relationship}/{total} ({pct:.1f}%)")

    if pct > 95:
        print("    ✓ Rate relationship is consistent")
    else:
        print("    ⚠ Some days have unusual rate relationships")
        unusual = df_rate_check.filter(~pl.col("borrow_higher"))
        print(f"    Found {len(unusual)} days with lending >= borrowing")

    # Check spread (borrowing - lending)
    df_spread = df.with_columns([
        (pl.col("borrowing_apr") - pl.col("lending_apr")).alias("spread")
    ])

    mean_spread = df_spread["spread"].mean()
    min_spread = df_spread["spread"].min()
    max_spread = df_spread["spread"].max()

    print(f"\n  Rate Spread (Borrowing - Lending):")
    print(f"    Mean: {mean_spread:.3f}%")  # type: ignore
    print(f"    Range: {min_spread:.3f}% to {max_spread:.3f}%")  # type: ignore

    print()

    # 6. Final validation summary
    print("=" * 80)
    print("VALIDATION SUMMARY")
    print("=" * 80)

    issues = []

    if expected_days != actual_days:
        issues.append(f"Missing {expected_days - actual_days} days of data")

    if len(duplicates) > 0:
        issues.append(f"Found {len(duplicates)} duplicate dates")

    if has_nulls:
        issues.append("Contains null values")

    if len(high_supply_rates) > 0:
        issues.append(f"{len(high_supply_rates)} days with unusually high supply rates (>20%)")

    if len(high_borrow_rates) > 0:
        issues.append(f"{len(high_borrow_rates)} days with unusually high borrow rates (>30%)")

    if len(invalid_utilization) > 0:
        issues.append(f"{len(invalid_utilization)} invalid utilization rates")

    if not is_sorted:
        issues.append("Data not properly sorted by date")

    if len(issues) == 0:
        print("✅ DATA VALIDATION PASSED!")
        print("   All checks completed successfully.")
        print("   Data is complete, consistent, and ready for analysis.")
    else:
        print("⚠️  VALIDATION ISSUES FOUND:")
        for i, issue in enumerate(issues, 1):
            print(f"   {i}. {issue}")
        print("\n   Note: Some issues may be expected (e.g., rate spikes during high demand)")

    print()
    print("Data file:", file_path)
    print("Total records:", f"{len(df):,}")
    print("Date range:", f"{min_date} to {max_date}")
    print()


def main():
    """Main entry point"""
    # Path to the USDT lending rates Parquet file
    data_dir = Path("data")
    parquet_file = data_dir / "usdt_lending_rates_2023_2025.parquet"

    if not parquet_file.exists():
        print(f"Error: Data file not found at {parquet_file}")
        sys.exit(1)

    validate_usdt_data(str(parquet_file))


if __name__ == "__main__":
    main()