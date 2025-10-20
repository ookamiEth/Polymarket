#!/usr/bin/env python3
"""
Calculate daily risk-free rate for crypto options using perpetual futures funding rates.

Methodology from SSRN 5231776: "Option Returns and the Risk-Free Rate in Crypto Asset Markets"
by Winkel, Härdle, Zhou, and Chen (2025)

Paper methodology (Page 3, Section 3.1):
"Specifically, we propose the use of perpetual futures contracts to approximate the risk-free rate"

Data source:
- BTC perpetual funding rates (Binance) - 8-hour settlement periods
- Annualized using simple multiplication: r_annual = r_8h × 3 × 365

This simplified implementation matches the paper's actual methodology, which uses ONLY
perpetual funding rates (no stablecoin blending, no moving averages, no adjustments).
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
OUTPUT_FILE = DATA_DIR / "daily_risk_free_rates.parquet"
SUMMARY_FILE = DATA_DIR / "risk_free_rates_summary.json"


def load_funding_rates() -> pl.DataFrame:
    """
    Load BTC funding rates from Binance.

    Returns:
        DataFrame with funding_time_utc, funding_rate columns
    """
    logger.info(f"Loading funding rates from {FUNDING_FILE}")

    df = pl.read_parquet(FUNDING_FILE)

    logger.info(f"Loaded {len(df):,} funding rate records")
    logger.info(f"Date range: {df['funding_time_utc'].min()} to {df['funding_time_utc'].max()}")

    return df.select(["funding_time_utc", "funding_rate"])


def calculate_daily_risk_free_rate(funding_df: pl.DataFrame) -> pl.DataFrame:
    """
    Calculate daily annualized risk-free rate from 8-hour funding rates.

    Methodology per SSRN 5231776 paper:
    1. Aggregate 8-hour funding rates to daily (mean of 3 periods per day)
    2. Annualize using simple multiplication: r_annual = r_8h × 3 × 365

    Args:
        funding_df: DataFrame with funding_time_utc, funding_rate

    Returns:
        DataFrame with date, funding_rate_8h_mean, risk_free_rate_annual
    """
    logger.info("Calculating daily risk-free rates")

    # Extract date and aggregate to daily mean
    daily_df = (
        funding_df.with_columns([pl.col("funding_time_utc").dt.date().alias("date")])
        .group_by("date")
        .agg([pl.col("funding_rate").mean().alias("funding_rate_8h_mean")])
        .sort("date")
    )

    # Annualize using simple multiplication (standard practice)
    # r_annual = r_8h × 3 periods/day × 365 days/year
    daily_df = daily_df.with_columns([(pl.col("funding_rate_8h_mean") * 3 * 365).alias("risk_free_rate_annual")])

    logger.info(f"Calculated {len(daily_df):,} daily risk-free rates")

    # Log summary statistics
    mean_rate = float(daily_df["risk_free_rate_annual"].mean() or 0.0)  # type: ignore[arg-type]
    median_rate = float(daily_df["risk_free_rate_annual"].median() or 0.0)  # type: ignore[arg-type]
    logger.info(f"Mean annual rate: {mean_rate:.4f} ({mean_rate * 100:.2f}%)")
    logger.info(f"Median annual rate: {median_rate:.4f} ({median_rate * 100:.2f}%)")

    return daily_df


def generate_summary_statistics(df: pl.DataFrame) -> dict:
    """
    Generate comprehensive summary statistics for risk-free rates.

    Args:
        df: DataFrame with risk_free_rate_annual column

    Returns:
        Dictionary with summary statistics
    """
    logger.info("Generating summary statistics")

    col = df["risk_free_rate_annual"]

    stats = {
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
            "methodology": "Perpetual futures funding rates only (per SSRN 5231776)",
        },
        "risk_free_rate_annual": {k: v * 100 for k, v in stats.items()},  # Convert to percentage
    }

    return summary


def main() -> None:
    """Main execution function."""
    logger.info("=" * 80)
    logger.info("CALCULATING DAILY RISK-FREE RATE FOR CRYPTO OPTIONS")
    logger.info("Methodology: SSRN 5231776 (Perpetual Funding Rates)")
    logger.info("=" * 80)

    # Load data
    funding_df = load_funding_rates()

    # Calculate daily risk-free rate
    result_df = calculate_daily_risk_free_rate(funding_df)

    # Write to Parquet
    logger.info(f"Writing results to {OUTPUT_FILE}")
    result_df.write_parquet(OUTPUT_FILE, compression="snappy", statistics=True)
    logger.info(f"✅ Wrote {len(result_df):,} daily observations to Parquet")

    # Generate and save summary statistics
    summary = generate_summary_statistics(result_df)

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
    logger.info("\nRisk-Free Rate (Annual):")
    logger.info(f"  Mean: {summary['risk_free_rate_annual']['mean']:.2f}%")
    logger.info(f"  Median: {summary['risk_free_rate_annual']['median']:.2f}%")
    logger.info(
        f"  Range: {summary['risk_free_rate_annual']['min']:.2f}% - {summary['risk_free_rate_annual']['max']:.2f}%"
    )
    logger.info(f"  Std Dev: {summary['risk_free_rate_annual']['std']:.2f}%")
    logger.info("=" * 80)

    logger.info("\n✅ Calculation complete!")


if __name__ == "__main__":
    main()
