#!/usr/bin/env python3
"""
Query Examples for Polymarket Leaderboard Parquet Files

Demonstrates common queries and analysis patterns using Polars.
"""

import polars as pl
from pathlib import Path

# Base directory
DATA_DIR = Path("/Users/lgierhake/Documents/ETH/BT/top_traders/data/parquet")

def example_1_load_and_filter():
    """Example 1: Load and basic filtering"""
    print("="*80)
    print(" EXAMPLE 1: Load and Filter")
    print("="*80)

    # Load week leaderboard
    df = pl.read_parquet(DATA_DIR / "week" / "leaderboard_week_20251001.parquet")

    print(f"Total traders: {df.height:,}\n")

    # Filter active traders (volume > 0)
    active = df.filter(pl.col("volume_usd") > 0)
    print(f"Active traders (volume > 0): {active.height:,}")

    # Filter named traders
    named = df.filter(pl.col("has_custom_username") == True)
    print(f"Named traders: {named.height:,}")

    # High volume traders (> $1M)
    high_vol = df.filter(pl.col("volume_usd") > 1_000_000)
    print(f"High volume traders (>$1M): {high_vol.height:,}")

    # Profitable traders
    profitable = df.filter(pl.col("pnl_usd") > 0)
    print(f"Profitable traders: {profitable.height:,}")

    print()

def example_2_top_performers():
    """Example 2: Find top performers"""
    print("="*80)
    print(" EXAMPLE 2: Top Performers")
    print("="*80)

    df = pl.read_parquet(DATA_DIR / "all" / "leaderboard_all_20251001.parquet")

    # Top 10 by volume
    print("\nTop 10 by Volume (All-Time):")
    top_vol = df.sort("rank_by_volume").head(10)
    for row in top_vol.select(["rank_by_volume", "username", "volume_usd"]).iter_rows(named=True):
        print(f"  {row['rank_by_volume']:3}. {row['username'][:30]:30} ${row['volume_usd']:>15,.2f}")

    # Top 10 by P&L
    print("\nTop 10 by P&L (All-Time):")
    top_pnl = df.sort("rank_by_pnl").head(10)
    for row in top_pnl.select(["rank_by_pnl", "username", "pnl_usd"]).iter_rows(named=True):
        print(f"  {row['rank_by_pnl']:3}. {row['username'][:30]:30} ${row['pnl_usd']:>15,.2f}")

    # Top 10 by ROI
    print("\nTop 10 by ROI% (All-Time, min $10k volume):")
    top_roi = (df
        .filter(pl.col("volume_usd") > 10_000)
        .filter(pl.col("pnl_roi_percentage").is_not_null())
        .sort("pnl_roi_percentage", descending=True)
        .head(10)
    )
    for row in top_roi.select(["username", "pnl_roi_percentage", "volume_usd", "pnl_usd"]).iter_rows(named=True):
        print(f"  {row['username'][:30]:30} {row['pnl_roi_percentage']:>6.2f}% "
              f"(Vol: ${row['volume_usd']:>12,.0f}, P&L: ${row['pnl_usd']:>10,.0f})")

    print()

def example_3_aggregations():
    """Example 3: Aggregate statistics"""
    print("="*80)
    print(" EXAMPLE 3: Aggregate Statistics")
    print("="*80)

    df = pl.read_parquet(DATA_DIR / "week" / "leaderboard_week_20251001.parquet")

    # Overall stats
    stats = df.select([
        pl.col("volume_usd").sum().alias("total_volume"),
        pl.col("pnl_usd").sum().alias("total_pnl"),
        pl.col("volume_usd").mean().alias("avg_volume"),
        pl.col("pnl_usd").mean().alias("avg_pnl"),
        pl.col("volume_usd").median().alias("median_volume"),
        (pl.col("pnl_usd") > 0).sum().alias("profitable_count"),
    ])

    print("\nWeekly Statistics:")
    for key, value in stats.to_dicts()[0].items():
        if "count" in key:
            print(f"  {key}: {value:,}")
        else:
            print(f"  {key}: ${value:,.2f}")

    # Volume distribution
    print("\nVolume Distribution:")
    buckets = df.select([
        (pl.col("volume_usd") == 0).sum().alias("zero_volume"),
        ((pl.col("volume_usd") > 0) & (pl.col("volume_usd") <= 1_000)).sum().alias("0-1k"),
        ((pl.col("volume_usd") > 1_000) & (pl.col("volume_usd") <= 10_000)).sum().alias("1k-10k"),
        ((pl.col("volume_usd") > 10_000) & (pl.col("volume_usd") <= 100_000)).sum().alias("10k-100k"),
        ((pl.col("volume_usd") > 100_000) & (pl.col("volume_usd") <= 1_000_000)).sum().alias("100k-1M"),
        (pl.col("volume_usd") > 1_000_000).sum().alias("1M+"),
    ])

    for key, value in buckets.to_dicts()[0].items():
        print(f"  {key}: {value:,} traders")

    print()

def example_4_compare_periods():
    """Example 4: Compare different time periods"""
    print("="*80)
    print(" EXAMPLE 4: Compare Time Periods")
    print("="*80)

    # Load all periods
    day = pl.read_parquet(DATA_DIR / "day" / "leaderboard_day_20251001.parquet")
    week = pl.read_parquet(DATA_DIR / "week" / "leaderboard_week_20251001.parquet")
    all_time = pl.read_parquet(DATA_DIR / "all" / "leaderboard_all_20251001.parquet")

    print("\nTop Trader by Volume (Each Period):")
    print(f"  Day:      {day.sort('rank_by_volume').row(0, named=True)['username']:30} ${day.sort('rank_by_volume').row(0, named=True)['volume_usd']:>15,.2f}")
    print(f"  Week:     {week.sort('rank_by_volume').row(0, named=True)['username']:30} ${week.sort('rank_by_volume').row(0, named=True)['volume_usd']:>15,.2f}")
    print(f"  All-Time: {all_time.sort('rank_by_volume').row(0, named=True)['username']:30} ${all_time.sort('rank_by_volume').row(0, named=True)['volume_usd']:>15,.2f}")

    print("\nTotal Volume by Period:")
    print(f"  Day:      ${day['volume_usd'].sum():>15,.2f}")
    print(f"  Week:     ${week['volume_usd'].sum():>15,.2f}")
    print(f"  All-Time: ${all_time['volume_usd'].sum():>15,.2f}")

    print("\nActive Traders by Period:")
    print(f"  Day:      {day.filter(pl.col('volume_usd') > 0).height:,}")
    print(f"  Week:     {week.filter(pl.col('volume_usd') > 0).height:,}")
    print(f"  All-Time: {all_time.filter(pl.col('volume_usd') > 0).height:,}")

    print()

def example_5_specific_trader():
    """Example 5: Track specific trader"""
    print("="*80)
    print(" EXAMPLE 5: Track Specific Trader")
    print("="*80)

    # Pick a known active trader
    trader_address = "0x53757615de1c42b83f893b79d4241a009dc2aeea"

    periods = ['day', 'week', 'all']
    print(f"\nTrader: {trader_address}\n")

    for period in periods:
        df = pl.read_parquet(DATA_DIR / period / f"leaderboard_{period}_20251001.parquet")

        trader = df.filter(pl.col("user_address") == trader_address)

        if trader.height > 0:
            row = trader.row(0, named=True)
            print(f"{period.upper():8}: Rank by Vol: {row['rank_by_volume']:4} | "
                  f"Volume: ${row['volume_usd']:>12,.2f} | P&L: ${row['pnl_usd']:>10,.2f}")
        else:
            print(f"{period.upper():8}: Not in top 1000")

    print()

def example_6_advanced_queries():
    """Example 6: Advanced queries and analysis"""
    print("="*80)
    print(" EXAMPLE 6: Advanced Queries")
    print("="*80)

    df = pl.read_parquet(DATA_DIR / "week" / "leaderboard_week_20251001.parquet")

    # Consistent performers (good volume + good P&L)
    print("\nTop Consistent Performers (High Volume + Positive P&L):")
    consistent = (df
        .filter(pl.col("volume_usd") > 1_000_000)
        .filter(pl.col("pnl_usd") > 10_000)
        .sort("volume_usd", descending=True)
        .head(5)
    )

    for row in consistent.select(["username", "volume_usd", "pnl_usd", "pnl_roi_percentage"]).iter_rows(named=True):
        print(f"  {row['username'][:30]:30} Vol: ${row['volume_usd']:>12,.0f} | "
              f"P&L: ${row['pnl_usd']:>10,.0f} | ROI: {row['pnl_roi_percentage']:>6.2f}%")

    # Find traders who rank differently by volume vs P&L
    print("\nBiggest Rank Differences (Volume vs P&L):")
    rank_diff = (df
        .with_columns([
            (pl.col("rank_by_pnl") - pl.col("rank_by_volume")).abs().alias("rank_diff")
        ])
        .filter(pl.col("volume_usd") > 100_000)
        .sort("rank_diff", descending=True)
        .head(5)
    )

    for row in rank_diff.select(["username", "rank_by_volume", "rank_by_pnl", "rank_diff"]).iter_rows(named=True):
        print(f"  {row['username'][:30]:30} Vol Rank: {row['rank_by_volume']:4} | "
              f"P&L Rank: {row['rank_by_pnl']:4} | Diff: {row['rank_diff']}")

    print()

def main():
    """Run all examples"""
    print("\n" + "="*80)
    print(" POLYMARKET LEADERBOARD QUERY EXAMPLES")
    print("="*80 + "\n")

    example_1_load_and_filter()
    example_2_top_performers()
    example_3_aggregations()
    example_4_compare_periods()
    example_5_specific_trader()
    example_6_advanced_queries()

    print("="*80)
    print(" ALL EXAMPLES COMPLETE")
    print("="*80 + "\n")

if __name__ == "__main__":
    main()
