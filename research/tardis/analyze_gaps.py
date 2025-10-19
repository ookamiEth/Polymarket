#!/usr/bin/env python3
"""
Analyze timestamp gaps in 1s-sampled options quotes data.

Shows how continuous vs sparse the data is per symbol.
"""

import argparse
import logging
from pathlib import Path

import polars as pl

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def analyze_gaps_sample(file_path: Path, n_symbols: int = 10) -> None:
    """Analyze gaps for a sample of symbols."""
    logger.info("=" * 80)
    logger.info("GAP ANALYSIS")
    logger.info("=" * 80)

    # Get top N symbols by quote count
    logger.info(f"Finding top {n_symbols} most active symbols...")
    lazy_df = pl.scan_parquet(file_path)

    top_symbols = (
        lazy_df.group_by("symbol")
        .agg([pl.len().alias("quote_count")])
        .sort("quote_count", descending=True)
        .head(n_symbols)
        .collect()
    )

    logger.info(f"\nTop {n_symbols} symbols by quote count:")
    print(top_symbols)

    # Analyze each symbol
    logger.info(f"\n{'=' * 80}")
    logger.info("ANALYZING GAPS FOR EACH SYMBOL")
    logger.info(f"{'=' * 80}\n")

    results = []

    for i, row in enumerate(top_symbols.iter_rows(named=True)):
        symbol = row["symbol"]
        total_quotes = row["quote_count"]

        logger.info(f"[{i + 1}/{n_symbols}] Analyzing: {symbol} ({total_quotes:,} quotes)")

        # Load this symbol's data
        symbol_data = (
            lazy_df.filter(pl.col("symbol") == symbol)
            .select(["timestamp_seconds", "quote_count"])
            .sort("timestamp_seconds")
            .collect()
        )

        if len(symbol_data) == 0:
            continue

        # Calculate gaps
        timestamps = symbol_data["timestamp_seconds"]
        min_ts = timestamps.min()
        max_ts = timestamps.max()
        duration_seconds = max_ts - min_ts

        # Calculate consecutive timestamp differences
        gaps = timestamps.diff().drop_nulls()

        # Gap statistics
        gap_1s = (gaps == 1).sum()  # Consecutive seconds
        gap_2to10s = ((gaps >= 2) & (gaps <= 10)).sum()
        gap_11to60s = ((gaps >= 11) & (gaps <= 60)).sum()
        gap_1to10min = ((gaps >= 61) & (gaps <= 600)).sum()
        gap_over10min = (gaps > 600).sum()

        avg_gap = gaps.mean()
        median_gap = gaps.median()
        max_gap = gaps.max()

        # Calculate coverage (what % of time had quotes)
        coverage_pct = (len(symbol_data) / duration_seconds * 100) if duration_seconds > 0 else 0

        logger.info(f"  Time span: {duration_seconds:,} seconds ({duration_seconds / 86400:.1f} days)")
        logger.info(f"  Data points: {len(symbol_data):,}")
        logger.info(f"  Coverage: {coverage_pct:.2f}% (has quotes)")
        logger.info(f"  Average gap: {avg_gap:.1f}s")
        logger.info(f"  Median gap: {median_gap:.0f}s")
        logger.info(f"  Max gap: {max_gap:,}s ({max_gap / 3600:.1f} hours)")
        logger.info("  Gap distribution:")
        logger.info(f"    1s (consecutive):   {gap_1s:,} ({gap_1s / len(gaps) * 100:.1f}%)")
        logger.info(f"    2-10s:              {gap_2to10s:,} ({gap_2to10s / len(gaps) * 100:.1f}%)")
        logger.info(f"    11-60s:             {gap_11to60s:,} ({gap_11to60s / len(gaps) * 100:.1f}%)")
        logger.info(f"    1-10 min:           {gap_1to10min:,} ({gap_1to10min / len(gaps) * 100:.1f}%)")
        logger.info(f"    >10 min:            {gap_over10min:,} ({gap_over10min / len(gaps) * 100:.1f}%)")
        logger.info("")

        results.append(
            {
                "symbol": symbol,
                "total_quotes": total_quotes,
                "duration_days": duration_seconds / 86400,
                "coverage_pct": coverage_pct,
                "avg_gap_sec": avg_gap,
                "median_gap_sec": median_gap,
                "max_gap_sec": max_gap,
                "consecutive_pct": gap_1s / len(gaps) * 100 if len(gaps) > 0 else 0,
            }
        )

    # Summary table
    logger.info("=" * 80)
    logger.info("SUMMARY TABLE")
    logger.info("=" * 80)
    summary_df = pl.DataFrame(results)
    print(summary_df)


def analyze_overall_gaps(file_path: Path, sample_size: int = 10_000_000) -> None:
    """Analyze gaps across all data (using sample)."""
    logger.info("=" * 80)
    logger.info("OVERALL GAP ANALYSIS (Sample)")
    logger.info("=" * 80)

    logger.info(f"Loading sample of {sample_size:,} rows...")

    df = pl.read_parquet(file_path, n_rows=sample_size)

    # Group by symbol and analyze gaps within each
    logger.info("Calculating gaps per symbol...")

    # For each symbol, calculate time gaps
    gap_analysis = (
        df.sort(["symbol", "timestamp_seconds"])
        .with_columns([pl.col("timestamp_seconds").diff().over("symbol").alias("gap")])
        .filter(pl.col("gap").is_not_null())
    )

    # Overall gap statistics
    logger.info("\nOverall gap statistics (across all symbols in sample):")
    logger.info(f"  Mean gap: {gap_analysis['gap'].mean():.1f} seconds")
    logger.info(f"  Median gap: {gap_analysis['gap'].median():.0f} seconds")
    logger.info(f"  Min gap: {gap_analysis['gap'].min():.0f} seconds")
    logger.info(f"  Max gap: {gap_analysis['gap'].max():,.0f} seconds ({gap_analysis['gap'].max() / 3600:.1f} hours)")

    # Gap percentiles
    p25 = gap_analysis["gap"].quantile(0.25)
    p50 = gap_analysis["gap"].quantile(0.50)
    p75 = gap_analysis["gap"].quantile(0.75)
    p90 = gap_analysis["gap"].quantile(0.90)
    p95 = gap_analysis["gap"].quantile(0.95)
    p99 = gap_analysis["gap"].quantile(0.99)

    logger.info(f"  25th percentile: {p25:.0f}s")
    logger.info(f"  50th percentile: {p50:.0f}s")
    logger.info(f"  75th percentile: {p75:.0f}s")
    logger.info(f"  90th percentile: {p90:.0f}s")
    logger.info(f"  95th percentile: {p95:.0f}s")
    logger.info(f"  99th percentile: {p99:.0f}s")

    # Count consecutive seconds (gap = 1)
    consecutive = (gap_analysis["gap"] == 1).sum()
    total_gaps = len(gap_analysis)
    logger.info(f"\nConsecutive seconds: {consecutive:,} ({consecutive / total_gaps * 100:.1f}%)")


def main() -> None:
    """Main analysis routine."""
    parser = argparse.ArgumentParser(description="Analyze timestamp gaps in consolidated options quotes")
    parser.add_argument(
        "file_path",
        type=str,
        help="Path to consolidated Parquet file",
    )
    parser.add_argument(
        "--n-symbols",
        type=int,
        default=10,
        help="Number of symbols to analyze in detail (default: 10)",
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=10_000_000,
        help="Sample size for overall gap analysis (default: 10,000,000)",
    )

    args = parser.parse_args()

    file_path = Path(args.file_path)

    if not file_path.exists():
        logger.error(f"File not found: {file_path}")
        return

    logger.info("=" * 80)
    logger.info("TIMESTAMP GAP ANALYSIS")
    logger.info("=" * 80)
    logger.info(f"File: {file_path}")
    logger.info("=" * 80)

    # Run analyses
    analyze_overall_gaps(file_path, args.sample_size)
    analyze_gaps_sample(file_path, args.n_symbols)

    logger.info("=" * 80)
    logger.info("ANALYSIS COMPLETE")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
