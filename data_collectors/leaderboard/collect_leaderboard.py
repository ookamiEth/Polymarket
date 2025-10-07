#!/usr/bin/env python3
"""
Polymarket Leaderboard Parquet Collector

Collects leaderboard data for all time periods (day/week/month/all)
and saves as comprehensive Parquet files using Polars.

Usage:
    uv run python collect_leaderboard.py
"""

import polars as pl
import requests
from datetime import datetime, date
from pathlib import Path
import time
import json

class LeaderboardCollector:
    def __init__(self, base_dir="/Users/lgierhake/Documents/ETH/BT/top_traders/data"):
        self.base_url = "https://data-api.polymarket.com/leaderboard"
        self.base_dir = Path(base_dir)
        self.rate_limit_delay = 0.11  # 100 req/10s = ~9 req/sec safe

    def collect_leaderboard(self, period: str, limit_per_page: int = 50, max_traders: int = 1000) -> list[dict]:
        """
        Collect complete leaderboard for a given period

        Args:
            period: 'day', 'week', 'month', or 'all'
            limit_per_page: API limit (max 50)
            max_traders: Maximum to collect (1000+)

        Returns:
            List of all trader records
        """
        all_data = []
        offset = 0

        print(f"Collecting {period} leaderboard...")

        while offset < max_traders:
            params = {
                "timePeriod": period,
                "limit": limit_per_page,
                "offset": offset
            }

            try:
                response = requests.get(self.base_url, params=params, timeout=10)
                response.raise_for_status()
                data = response.json()

                if not data:
                    print(f"  No more data at offset {offset}")
                    break

                all_data.extend(data)
                print(f"  Offset {offset:4}: +{len(data)} traders (total: {len(all_data):,})")

                if len(data) < limit_per_page:
                    print(f"  Received fewer than {limit_per_page}, stopping")
                    break

                offset += limit_per_page
                time.sleep(self.rate_limit_delay)

            except requests.exceptions.RequestException as e:
                print(f"  ✗ Error at offset {offset}: {e}")
                break

        print(f"✓ Collected {len(all_data):,} traders for period={period}")
        return all_data

    def create_dataframe(self, data: list[dict], period: str, snapshot_dt: datetime) -> pl.DataFrame:
        """Convert raw API data to structured DataFrame with all fields"""

        # Convert to Polars
        df = pl.DataFrame(data)

        # Add metadata and transform
        df = df.with_columns([
            # Metadata
            pl.lit(snapshot_dt).alias("snapshot_timestamp"),
            pl.lit(snapshot_dt.date()).alias("snapshot_date"),
            pl.lit(period).cast(pl.Categorical).alias("period_type"),

            # Rename API fields
            pl.col("rank").cast(pl.Int32).alias("rank_api"),
            pl.col("rank").cast(pl.Int32).alias("rank_by_pnl"),
            pl.col("user_id").alias("user_address"),
            pl.col("user_name").alias("username"),
            pl.col("vol").cast(pl.Float64).alias("volume_usd"),
            pl.col("pnl").cast(pl.Float64).alias("pnl_usd"),
            pl.col("profile_image").alias("profile_image_url"),

            # Calculated fields
            (pl.col("user_name") != pl.col("user_id")).alias("has_custom_username"),
        ])

        # Add profile URL
        df = df.with_columns([
            pl.when(pl.col("has_custom_username"))
              .then(pl.lit("https://polymarket.com/@") + pl.col("username"))
              .otherwise(pl.lit(""))
              .alias("profile_url")
        ])

        # Calculate rank by volume
        df = df.with_columns([
            pl.col("volume_usd")
              .rank(method="ordinal", descending=True)
              .cast(pl.Int32)
              .alias("rank_by_volume")
        ])

        # Calculate ROI percentage
        df = df.with_columns([
            pl.when(pl.col("volume_usd") > 0)
              .then((pl.col("pnl_usd") / pl.col("volume_usd")) * 100)
              .otherwise(None)
              .alias("pnl_roi_percentage")
        ])

        # Data quality flag
        df = df.with_columns([
            (pl.col("user_address").is_not_null() &
             pl.col("username").is_not_null())
              .alias("is_complete_record")
        ])

        # Select final columns in order
        df = df.select([
            "snapshot_timestamp",
            "snapshot_date",
            "period_type",
            "rank_api",
            "rank_by_pnl",
            "rank_by_volume",
            "user_address",
            "username",
            "has_custom_username",
            "volume_usd",
            "pnl_usd",
            "pnl_roi_percentage",
            "profile_image_url",
            "profile_url",
            "is_complete_record",
        ])

        return df

    def save_parquet(self, df: pl.DataFrame, period: str, snapshot_date: date) -> Path:
        """Save DataFrame as Parquet file"""

        period_dir = self.base_dir / "parquet" / period
        period_dir.mkdir(parents=True, exist_ok=True)

        filename = f"leaderboard_{period}_{snapshot_date.strftime('%Y%m%d')}.parquet"
        output_file = period_dir / filename

        df.write_parquet(
            output_file,
            compression="snappy",
            statistics=True,
            row_group_size=100  # Good for time-series queries
        )

        file_size_kb = output_file.stat().st_size / 1024
        print(f"✓ Saved: {output_file.name}")
        print(f"  Rows: {df.height:,} | Columns: {df.width} | Size: {file_size_kb:.1f} KB")

        return output_file

    def collect_all_periods(self) -> dict:
        """Collect all 4 time periods in one run"""

        snapshot_dt = datetime.now()
        snapshot_date = snapshot_dt.date()

        periods = ['day', 'week', 'month', 'all']
        results = {}

        print("="*80)
        print(" POLYMARKET LEADERBOARD COLLECTION")
        print("="*80)
        print(f"Timestamp: {snapshot_dt}")
        print(f"Date: {snapshot_date}")
        print(f"Periods: {', '.join(periods)}")
        print("="*80)

        for period in periods:
            print(f"\n{'='*80}")
            print(f" COLLECTING: {period.upper()}")
            print(f"{'='*80}")

            # Collect data
            data = self.collect_leaderboard(period, max_traders=1000)

            if not data:
                print(f"⚠ No data collected for {period}, skipping")
                continue

            # Create DataFrame
            df = self.create_dataframe(data, period, snapshot_dt)

            # Save Parquet
            file_path = self.save_parquet(df, period, snapshot_date)

            # Gather stats
            top_by_volume = df.sort("rank_by_volume").head(3)
            top_by_pnl = df.sort("rank_by_pnl").head(3)

            results[period] = {
                "file": str(file_path),
                "traders": df.height,
                "top_by_volume": top_by_volume.select(["username", "volume_usd"]).to_dicts(),
                "top_by_pnl": top_by_pnl.select(["username", "pnl_usd"]).to_dicts(),
                "total_volume": df["volume_usd"].sum(),
                "total_pnl": df["pnl_usd"].sum(),
            }

        # Save metadata
        self._save_metadata(results, snapshot_dt)

        # Summary
        self._print_summary(results)

        return results

    def _save_metadata(self, results: dict, snapshot_dt: datetime):
        """Save collection metadata"""
        metadata_dir = self.base_dir / "metadata"
        metadata_dir.mkdir(parents=True, exist_ok=True)

        # Metadata file
        metadata = {
            "collection_timestamp": snapshot_dt.isoformat(),
            "collection_date": snapshot_dt.date().isoformat(),
            "periods_collected": list(results.keys()),
            "total_files": len(results),
            "results": results
        }

        metadata_file = metadata_dir / f"collection_{snapshot_dt.strftime('%Y%m%d_%H%M%S')}.json"
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2, default=str)

        print(f"\n✓ Metadata saved: {metadata_file}")

        # Append to log
        log_file = metadata_dir / "collection_log.jsonl"
        with open(log_file, 'a') as f:
            f.write(json.dumps(metadata, default=str) + '\n')

    def _print_summary(self, results: dict):
        """Print collection summary"""
        print(f"\n{'='*80}")
        print(" COLLECTION COMPLETE")
        print(f"{'='*80}")

        for period, info in results.items():
            print(f"\n{period.upper()}:")
            print(f"  Traders: {info['traders']:,}")
            print(f"  Total Volume: ${info['total_volume']:,.2f}")
            print(f"  Total P&L: ${info['total_pnl']:,.2f}")

            print(f"  Top 3 by Volume:")
            for i, trader in enumerate(info['top_by_volume'], 1):
                print(f"    {i}. {trader['username'][:30]:30} ${trader['volume_usd']:>12,.2f}")

            print(f"  Top 3 by P&L:")
            for i, trader in enumerate(info['top_by_pnl'], 1):
                print(f"    {i}. {trader['username'][:30]:30} ${trader['pnl_usd']:>12,.2f}")

            print(f"  File: {Path(info['file']).name}")

        print(f"\n{'='*80}")

def main():
    """Main entry point"""
    collector = LeaderboardCollector()
    results = collector.collect_all_periods()
    return results

if __name__ == "__main__":
    main()
