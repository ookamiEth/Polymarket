#!/usr/bin/env python3
"""
Calculate daily risk-free rates for 1-day, 2-day, and 3-day crypto options.

Methodology from SSRN 5231776: "Option Returns and the Risk-Free Rate in Crypto Asset Markets"
by Winkel, Härdle, Zhou, and Chen (2025)

Combines:
- BTC perpetual funding rates (Binance) - 8-hour periods, 7-day MA
- USDT lending rates (Aave V3 Arbitrum) - daily snapshots

Weighting scheme (per paper):
- 1-day options: 85% funding / 15% lending
- 2-day options: 75% funding / 25% lending
- 3-day options: 70% funding / 30% lending
"""

import json
import logging
from datetime import datetime
from pathlib import Path

import polars as pl

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Constants
DATA_DIR = Path(__file__).parent / "data"
FUNDING_FILE = DATA_DIR / "btc_funding_rates_2023_2025.parquet"
LENDING_FILE = DATA_DIR / "usdt_lending_rates_2023_2025.parquet"
OUTPUT_FILE = DATA_DIR / "daily_risk_free_rates_1d_2d_3d.parquet"
SUMMARY_FILE = DATA_DIR / "risk_free_rates_summary.json"

# Rate adjustment parameters (per paper)
MAX_FUNDING_RATE = 0.50  # 50% annual cap
EXTREME_FUNDING_THRESHOLD = 0.30  # 30% annual threshold for dampening
FUNDING_MA_PERIODS = 21  # 7 days * 3 periods per day

# Weighting scheme per SSRN 5231776 paper
# For options ≤3 days: 70% funding / 30% stablecoin lending
WEIGHTS = {
    1: {"funding": 0.70, "lending": 0.30},
    2: {"funding": 0.70, "lending": 0.30},
    3: {"funding": 0.70, "lending": 0.30},
}


def load_funding_rates() -> pl.DataFrame:
    """
    Load BTC funding rates and prepare for daily aggregation.

    Returns:
        DataFrame with funding_time_utc, funding_rate columns
    """
    logger.info(f"Loading funding rates from {FUNDING_FILE}")

    df = pl.read_parquet(FUNDING_FILE)

    logger.info(f"Loaded {len(df):,} funding rate records")
    logger.info(f"Date range: {df['funding_time_utc'].min()} to {df['funding_time_utc'].max()}")

    return df.select(["funding_time_utc", "funding_rate"])


def load_lending_rates() -> pl.DataFrame:
    """
    Load USDT lending rates from Aave V3.

    Returns:
        DataFrame with date, lending_apr columns
    """
    logger.info(f"Loading USDT lending rates from {LENDING_FILE}")

    df = pl.read_parquet(LENDING_FILE)

    logger.info(f"Loaded {len(df):,} daily lending rate records")
    logger.info(f"Date range: {df['date'].min()} to {df['date'].max()}")

    # Select only needed columns and convert APR from percentage to decimal
    # (lending_apr is already in percentage form: 5.49 means 5.49%)
    return df.select(["date", "lending_apr"])


def calculate_funding_rate_daily(funding_df: pl.DataFrame) -> pl.DataFrame:
    """
    Calculate daily annualized funding rates with 7-day moving average.

    Steps:
    1. Calculate 7-day (21-period) moving average of 8-hour rates
    2. Annualize: (1 + r_8h)^(3*365) - 1
    3. Apply extreme rate adjustments
    4. Aggregate to daily (take value at end of day)

    Args:
        funding_df: DataFrame with funding_time_utc, funding_rate

    Returns:
        DataFrame with date, funding_rate_8h_avg, funding_rate_annual
    """
    logger.info("Calculating daily funding rates with 7-day MA and annualization")

    # Calculate 7-day moving average
    df = funding_df.sort("funding_time_utc").with_columns(
        [pl.col("funding_rate").rolling_mean(window_size=FUNDING_MA_PERIODS).alias("funding_rate_8h_avg")]
    )

    # Annualize funding rates: (1 + r)^(3*365) - 1
    # For small rates, use approximation: r * (3*365)
    # For larger rates, use exact formula
    df = df.with_columns(
        [
            pl.when(pl.col("funding_rate_8h_avg").abs() < 0.001)
            .then(pl.col("funding_rate_8h_avg") * (3 * 365))
            .otherwise((1 + pl.col("funding_rate_8h_avg")).pow(3 * 365) - 1)
            .alias("funding_rate_annual_raw")
        ]
    )

    # Apply extreme rate adjustments
    df = df.with_columns([_adjust_extreme_funding(pl.col("funding_rate_annual_raw")).alias("funding_rate_annual")])

    # Extract date and aggregate to daily (use last value of each day)
    df = df.with_columns([pl.col("funding_time_utc").dt.date().alias("date")])

    # Group by date and take last value (end of day)
    daily_df = (
        df.group_by("date")
        .agg(
            [
                pl.col("funding_rate_8h_avg").last(),
                pl.col("funding_rate_annual").last(),
            ]
        )
        .sort("date")
    )

    logger.info(f"Calculated {len(daily_df):,} daily funding rates")

    return daily_df


def _adjust_extreme_funding(rate_col: pl.Expr) -> pl.Expr:
    """
    Apply extreme funding rate adjustments per paper recommendations.

    Adjustments:
    - Cap at MAX_FUNDING_RATE (50%)
    - Dampen rates > EXTREME_FUNDING_THRESHOLD (30%) logarithmically
    - Halve negative rates (reduce impact of negative funding)

    Args:
        rate_col: Polars expression for annual funding rate

    Returns:
        Polars expression with adjusted rates
    """
    # Handle rates above threshold
    adjusted = (
        pl.when(rate_col > MAX_FUNDING_RATE)
        .then(pl.lit(MAX_FUNDING_RATE))
        .when(rate_col > EXTREME_FUNDING_THRESHOLD)
        .then(pl.lit(EXTREME_FUNDING_THRESHOLD) + (rate_col - EXTREME_FUNDING_THRESHOLD).log1p() / 100)
        .when(rate_col < 0)
        .then(rate_col * 0.5)  # Halve negative rates
        .otherwise(rate_col)
    )

    return adjusted


def merge_rates(funding_df: pl.DataFrame, lending_df: pl.DataFrame) -> pl.DataFrame:
    """
    Merge funding and lending rates on date.

    Handles missing data:
    - Forward-fill lending rates if funding data exists but lending missing
    - Mark data quality for transparency

    Args:
        funding_df: Daily funding rates
        lending_df: Daily lending rates

    Returns:
        Merged DataFrame with all rate components
    """
    logger.info("Merging funding and lending rates")

    # Left join on date (keep all funding dates)
    merged = funding_df.join(lending_df, on="date", how="left")

    # Count missing lending data
    missing_count = merged.filter(pl.col("lending_apr").is_null()).height

    if missing_count > 0:
        logger.warning(f"Found {missing_count} dates with missing lending data - will forward-fill")

        # Forward-fill missing lending rates
        merged = merged.with_columns([pl.col("lending_apr").forward_fill().alias("lending_apr")])

    # Create data quality flag
    merged = merged.with_columns(
        [
            pl.when(pl.col("lending_apr").is_null())
            .then(pl.lit("missing_lending"))
            .otherwise(pl.lit("complete"))
            .alias("data_quality")
        ]
    )

    # Convert lending_apr from percentage to decimal (5.49 -> 0.0549)
    merged = merged.with_columns([(pl.col("lending_apr") / 100).alias("lending_apr")])

    logger.info(f"Merged dataset has {len(merged):,} daily observations")

    return merged


def calculate_composite_rates(df: pl.DataFrame) -> pl.DataFrame:
    """
    Calculate composite risk-free rates for 1-day, 2-day, and 3-day maturities.

    Formula:
        composite_rate = w_funding * funding_rate_annual + w_lending * lending_apr

    Apply floor at 0% (never negative).

    Args:
        df: DataFrame with funding_rate_annual, lending_apr

    Returns:
        DataFrame with additional composite_1day_rate, composite_2day_rate, composite_3day_rate columns
    """
    logger.info("Calculating composite rates for 1-day, 2-day, and 3-day maturities")

    # Calculate each maturity's composite rate
    df = df.with_columns(
        [
            (WEIGHTS[1]["funding"] * pl.col("funding_rate_annual") + WEIGHTS[1]["lending"] * pl.col("lending_apr"))
            .clip(lower_bound=0.0)
            .alias("composite_1day_rate"),
            (WEIGHTS[2]["funding"] * pl.col("funding_rate_annual") + WEIGHTS[2]["lending"] * pl.col("lending_apr"))
            .clip(lower_bound=0.0)
            .alias("composite_2day_rate"),
            (WEIGHTS[3]["funding"] * pl.col("funding_rate_annual") + WEIGHTS[3]["lending"] * pl.col("lending_apr"))
            .clip(lower_bound=0.0)
            .alias("composite_3day_rate"),
        ]
    )

    # Log summary statistics
    for maturity in [1, 2, 3]:
        col_name = f"composite_{maturity}day_rate"
        mean_rate = float(df[col_name].mean() or 0.0)  # type: ignore[arg-type]
        median_rate = float(df[col_name].median() or 0.0)  # type: ignore[arg-type]
        logger.info(
            f"{maturity}-day rate: mean={mean_rate:.4f} ({mean_rate * 100:.2f}%), "
            f"median={median_rate:.4f} ({median_rate * 100:.2f}%)"
        )

    return df


def generate_summary_statistics(df: pl.DataFrame) -> dict:
    """
    Generate comprehensive summary statistics for all rate components.

    Args:
        df: DataFrame with all calculated rates

    Returns:
        Dictionary with summary statistics
    """
    logger.info("Generating summary statistics")

    def calc_stats(col_name: str) -> dict[str, float]:
        """Calculate statistics for a single column."""
        col = df[col_name]

        return {
            "mean": float(col.mean() or 0.0),  # type: ignore[arg-type]
            "median": float(col.median() or 0.0),  # type: ignore[arg-type]
            "std": float(col.std() or 0.0),  # type: ignore[arg-type]
            "min": float(col.min() or 0.0),  # type: ignore[arg-type]
            "max": float(col.max() or 0.0),  # type: ignore[arg-type]
            "p10": float(col.quantile(0.10) or 0.0),  # type: ignore[arg-type]
            "p25": float(col.quantile(0.25) or 0.0),  # type: ignore[arg-type]
            "p75": float(col.quantile(0.75) or 0.0),  # type: ignore[arg-type]
            "p90": float(col.quantile(0.90) or 0.0),  # type: ignore[arg-type]
        }

    summary = {
        "metadata": {
            "generated_at": datetime.utcnow().isoformat() + "Z",
            "period_start": str(df["date"].min()),
            "period_end": str(df["date"].max()),
            "total_days": len(df),
            "data_quality": {
                "complete": int(df.filter(pl.col("data_quality") == "complete").height),
                "missing_lending": int(df.filter(pl.col("data_quality") == "missing_lending").height),
            },
        },
        "funding_rate_annual": calc_stats("funding_rate_annual"),
        "lending_apr": {k: v * 100 for k, v in calc_stats("lending_apr").items()},  # Convert to percentage
        "composite_1day_rate": {k: v * 100 for k, v in calc_stats("composite_1day_rate").items()},
        "composite_2day_rate": {k: v * 100 for k, v in calc_stats("composite_2day_rate").items()},
        "composite_3day_rate": {k: v * 100 for k, v in calc_stats("composite_3day_rate").items()},
        "correlations": {
            "1day_vs_2day": float(df.select(pl.corr("composite_1day_rate", "composite_2day_rate")).item()),
            "2day_vs_3day": float(df.select(pl.corr("composite_2day_rate", "composite_3day_rate")).item()),
            "1day_vs_3day": float(df.select(pl.corr("composite_1day_rate", "composite_3day_rate")).item()),
            "funding_vs_lending": float(df.select(pl.corr("funding_rate_annual", "lending_apr")).item()),
        },
        "weights_used": WEIGHTS,
    }

    return summary


def main() -> None:
    """Main execution function."""
    logger.info("=" * 80)
    logger.info("CALCULATING DAILY RISK-FREE RATES FOR SHORT-DATED CRYPTO OPTIONS")
    logger.info("=" * 80)

    # Load data
    funding_df = load_funding_rates()
    lending_df = load_lending_rates()

    # Calculate daily funding rates
    funding_daily = calculate_funding_rate_daily(funding_df)

    # Merge datasets
    merged_df = merge_rates(funding_daily, lending_df)

    # Calculate composite rates
    result_df = calculate_composite_rates(merged_df)

    # Select final columns for output
    output_df = result_df.select(
        [
            "date",
            "funding_rate_8h_avg",
            "funding_rate_annual",
            "lending_apr",
            "composite_1day_rate",
            "composite_2day_rate",
            "composite_3day_rate",
            "data_quality",
        ]
    )

    # Write to Parquet
    logger.info(f"Writing results to {OUTPUT_FILE}")
    output_df.write_parquet(OUTPUT_FILE, compression="snappy", statistics=True)
    logger.info(f"✅ Wrote {len(output_df):,} daily observations to Parquet")

    # Generate and save summary statistics
    summary = generate_summary_statistics(output_df)

    logger.info(f"Writing summary statistics to {SUMMARY_FILE}")
    with open(SUMMARY_FILE, "w") as f:
        json.dump(summary, f, indent=2)

    logger.info("✅ Summary statistics saved")

    # Print key results
    logger.info("\n" + "=" * 80)
    logger.info("KEY RESULTS")
    logger.info("=" * 80)
    logger.info(f"Period: {summary['metadata']['period_start']} to {summary['metadata']['period_end']}")
    logger.info(f"Total days: {summary['metadata']['total_days']:,}")
    logger.info("\n3-Day Composite Rate (primary metric):")
    logger.info(f"  Mean: {summary['composite_3day_rate']['mean']:.2f}% (paper benchmark: 7-10%)")
    logger.info(f"  Median: {summary['composite_3day_rate']['median']:.2f}%")
    logger.info(f"  Range: {summary['composite_3day_rate']['min']:.2f}% - {summary['composite_3day_rate']['max']:.2f}%")
    logger.info(f"\nCorrelation (1-day vs 3-day): {summary['correlations']['1day_vs_3day']:.4f}")
    logger.info("=" * 80)

    logger.info("\n✅ Calculation complete! Next step: Run visualize_risk_free_rates.py")


if __name__ == "__main__":
    main()
