#!/usr/bin/env python3
"""
Phase 1: Data Availability Check for Advanced Feature Engineering

Checks for:
1. BTC 1-second spot price series
2. Hour-of-day timestamps
3. Multi-expiry ATM IVs (today/tomorrow/day after)
4. Contract metadata (strike, expiry)

Outputs:
- /tmp/data_availability_report.txt
- Console report with recommendations
"""

import logging
from pathlib import Path

import polars as pl

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Paths
BASE_PATH = Path("/Users/lgierhake/Documents/ETH/BT/research/model")
RESULTS_PATH = BASE_PATH / "results"
OUTPUT_REPORT = Path("/tmp/data_availability_report.txt")


def check_btc_spot_prices() -> dict:
    """Check if we have BTC 1-second spot price data."""
    logger.info("Checking BTC 1-second spot price data...")

    results = {
        "available": False,
        "file_path": None,
        "row_count": 0,
        "date_range": None,
        "columns": [],
        "sample_data": None,
    }

    # Check for BTC perpetual resampled data
    btc_file = RESULTS_PATH / "btc_perpetual_1s_resampled.parquet"

    if btc_file.exists():
        try:
            df = pl.scan_parquet(str(btc_file))

            # Get metadata without loading all data
            schema = df.schema
            row_count = df.select(pl.len()).collect()[0, 0]

            # Get date range
            date_stats = df.select([
                pl.col("timestamp").min().alias("min_date"),
                pl.col("timestamp").max().alias("max_date"),
            ]).collect()

            results["available"] = True
            results["file_path"] = str(btc_file)
            results["row_count"] = row_count
            results["date_range"] = (
                str(date_stats["min_date"][0]),
                str(date_stats["max_date"][0]),
            )
            results["columns"] = list(schema.keys())

            # Get sample
            sample = df.head(5).collect()
            results["sample_data"] = sample

            logger.info(f"✅ BTC spot price data found: {row_count:,} rows")
            logger.info(f"   Date range: {results['date_range'][0]} to {results['date_range'][1]}")
            logger.info(f"   Columns: {', '.join(results['columns'])}")

        except Exception as e:
            logger.error(f"❌ Error reading BTC spot price data: {e}")
            results["error"] = str(e)
    else:
        logger.warning(f"❌ BTC spot price file not found: {btc_file}")

    return results


def check_production_backtest_data() -> dict:
    """Check production backtest results for timestamps and contract metadata."""
    logger.info("Checking production backtest data...")

    results = {
        "available": False,
        "file_path": None,
        "row_count": 0,
        "has_timestamps": False,
        "has_strike": False,
        "has_expiry": False,
        "has_iv": False,
        "columns": [],
        "sample_data": None,
    }

    backtest_file = RESULTS_PATH / "production_backtest_results.parquet"

    if backtest_file.exists():
        try:
            df = pl.scan_parquet(str(backtest_file))

            schema = df.schema
            row_count = df.select(pl.len()).collect()[0, 0]

            results["available"] = True
            results["file_path"] = str(backtest_file)
            results["row_count"] = row_count
            results["columns"] = list(schema.keys())

            # Check for key columns
            results["has_timestamps"] = "timestamp" in schema
            results["has_strike"] = "strike" in schema
            results["has_expiry"] = "expiry" in schema
            results["has_iv"] = "iv" in schema or "implied_volatility" in schema

            # Get sample
            sample = df.head(5).collect()
            results["sample_data"] = sample

            logger.info(f"✅ Production backtest data found: {row_count:,} rows")
            logger.info(f"   Has timestamps: {results['has_timestamps']}")
            logger.info(f"   Has strike: {results['has_strike']}")
            logger.info(f"   Has expiry: {results['has_expiry']}")
            logger.info(f"   Has IV: {results['has_iv']}")
            logger.info(f"   Columns: {', '.join(results['columns'][:10])}...")

        except Exception as e:
            logger.error(f"❌ Error reading production backtest data: {e}")
            results["error"] = str(e)
    else:
        logger.warning(f"❌ Production backtest file not found: {backtest_file}")

    return results


def check_iv_data() -> dict:
    """Check if we have multi-expiry ATM IV data."""
    logger.info("Checking multi-expiry ATM IV data...")

    results = {
        "available": False,
        "source": None,
        "has_multiple_expiries": False,
        "expiry_count": 0,
        "notes": "",
    }

    # Check if IV data exists in production backtest
    backtest_file = RESULTS_PATH / "production_backtest_results.parquet"

    if backtest_file.exists():
        try:
            df = pl.scan_parquet(str(backtest_file))
            schema = df.schema

            if "iv" in schema or "implied_volatility" in schema:
                results["available"] = True
                results["source"] = "production_backtest (single expiry)"
                results["has_multiple_expiries"] = False
                results["notes"] = "IV available but only for current contract expiry"

                logger.warning("⚠️  IV data available but only for current contract")
                logger.warning("    Need: iv_atm_today, iv_atm_tomorrow, iv_atm_day_after")
                logger.warning("    Have: iv (current contract only)")

            else:
                logger.warning("❌ No IV data found in production backtest")
                results["notes"] = "No IV data found - Category 2 features unavailable"

        except Exception as e:
            logger.error(f"❌ Error checking IV data: {e}")
            results["error"] = str(e)

    # TODO: Check for Deribit options data with multiple expiries
    # This would require additional data collection

    return results


def generate_report(btc_data: dict, backtest_data: dict, iv_data: dict) -> None:
    """Generate comprehensive data availability report."""
    logger.info("\n" + "="*80)
    logger.info("DATA AVAILABILITY REPORT")
    logger.info("="*80)

    report_lines = []
    report_lines.append("="*80)
    report_lines.append("DATA AVAILABILITY REPORT FOR ADVANCED FEATURE ENGINEERING")
    report_lines.append("="*80)
    report_lines.append("")

    # BTC Spot Prices
    report_lines.append("1. BTC 1-SECOND SPOT PRICE DATA")
    report_lines.append("-" * 40)
    if btc_data["available"]:
        report_lines.append(f"✅ AVAILABLE: {btc_data['row_count']:,} rows")
        report_lines.append(f"   File: {btc_data['file_path']}")
        report_lines.append(f"   Date range: {btc_data['date_range'][0]} to {btc_data['date_range'][1]}")
        report_lines.append(f"   Columns: {', '.join(btc_data['columns'])}")
        report_lines.append("")
        report_lines.append("   ✅ Can compute: EMA, drawdown, higher moments, autocorrelation")
    else:
        report_lines.append("❌ NOT AVAILABLE")
        report_lines.append("   Action required: Need BTC 1-second price data")
    report_lines.append("")

    # Production Backtest Data
    report_lines.append("2. PRODUCTION BACKTEST DATA (Timestamps, Contract Metadata)")
    report_lines.append("-" * 40)
    if backtest_data["available"]:
        report_lines.append(f"✅ AVAILABLE: {backtest_data['row_count']:,} rows")
        report_lines.append(f"   File: {backtest_data['file_path']}")
        report_lines.append(f"   Has timestamps: {backtest_data['has_timestamps']}")
        report_lines.append(f"   Has strike: {backtest_data['has_strike']}")
        report_lines.append(f"   Has expiry: {backtest_data['has_expiry']}")
        report_lines.append("")
        if backtest_data["has_timestamps"]:
            report_lines.append("   ✅ Can compute: Time-of-day features (hour, is_us_hours, etc.)")
        if backtest_data["has_strike"] and backtest_data["has_expiry"]:
            report_lines.append("   ✅ Can compute: Drawdown features (distance to strike, strike vs range)")
    else:
        report_lines.append("❌ NOT AVAILABLE")
        report_lines.append("   Action required: Run production backtest first")
    report_lines.append("")

    # IV Data
    report_lines.append("3. MULTI-EXPIRY ATM IMPLIED VOLATILITY DATA")
    report_lines.append("-" * 40)
    if iv_data["available"] and iv_data["has_multiple_expiries"]:
        report_lines.append(f"✅ AVAILABLE: {iv_data['expiry_count']} expiries")
        report_lines.append(f"   Source: {iv_data['source']}")
        report_lines.append("")
        report_lines.append("   ✅ Can compute: IV term structure, IV/RV ratios, GARCH vs IV")
    elif iv_data["available"] and not iv_data["has_multiple_expiries"]:
        report_lines.append(f"⚠️  PARTIAL: Single expiry only")
        report_lines.append(f"   Source: {iv_data['source']}")
        report_lines.append(f"   Note: {iv_data['notes']}")
        report_lines.append("")
        report_lines.append("   ❌ CANNOT compute: IV term structure (need multiple expiries)")
        report_lines.append("   ⚠️  CAN compute: IV/RV ratios (using current IV)")
        report_lines.append("   ⚠️  CAN compute: GARCH vs IV (using current IV)")
        report_lines.append("")
        report_lines.append("   RECOMMENDATION: Skip Category 2 features requiring multiple expiries")
        report_lines.append("                   Keep features #13-14 (IV/RV ratios) if useful")
    else:
        report_lines.append("❌ NOT AVAILABLE")
        report_lines.append(f"   Note: {iv_data['notes']}")
        report_lines.append("")
        report_lines.append("   ❌ CANNOT compute: All Category 2 features (6 features)")
        report_lines.append("   ❌ CANNOT compute: Feature #39 (GARCH vs IV)")
        report_lines.append("")
        report_lines.append("   RECOMMENDATION: Proceed without Category 2 (lose 7 features total)")
    report_lines.append("")

    # Summary
    report_lines.append("="*80)
    report_lines.append("SUMMARY")
    report_lines.append("="*80)

    feature_count = 45  # Total proposed features
    available_count = 0
    unavailable_categories = []

    if btc_data["available"]:
        available_count += 8  # Category 1: EMAs
        available_count += 8  # Category 3: Drawdown
        available_count += 6  # Category 4: Higher moments
        available_count += 6  # Category 5: Time-of-day
        available_count += 5  # Category 6: Vol clustering (minus GARCH vs IV if no IV)
        available_count += 6  # Category 7: Jump/autocorr

    if not iv_data["available"] or not iv_data["has_multiple_expiries"]:
        available_count -= 6  # Category 2: IV features
        if not iv_data["available"]:
            available_count -= 1  # Feature #39: GARCH vs IV
        unavailable_categories.append("Category 2: IV-Based Features (6 features)")
        if not iv_data["available"]:
            unavailable_categories.append("Feature #39: GARCH vs IV")

    report_lines.append(f"Total proposed features: {feature_count}")
    report_lines.append(f"Available features: {available_count}")
    report_lines.append(f"Unavailable features: {feature_count - available_count}")
    report_lines.append("")

    if unavailable_categories:
        report_lines.append("Unavailable categories:")
        for cat in unavailable_categories:
            report_lines.append(f"  - {cat}")
        report_lines.append("")

    report_lines.append("RECOMMENDATION:")
    if available_count >= 38:  # At least 38 of 45 features
        report_lines.append("  ✅ Proceed with feature engineering")
        report_lines.append(f"  ✅ {available_count} features available (sufficient for improvement)")
        if unavailable_categories:
            report_lines.append(f"  ⚠️  Skip unavailable categories (adjust expected improvement by -1-2%)")
    else:
        report_lines.append("  ❌ Critical data missing - address before proceeding")

    report_lines.append("")
    report_lines.append("="*80)
    report_lines.append("END OF REPORT")
    report_lines.append("="*80)

    # Write to file
    with open(OUTPUT_REPORT, "w") as f:
        f.write("\n".join(report_lines))

    logger.info(f"Report saved to: {OUTPUT_REPORT}")

    # Print to console
    print("\n" + "\n".join(report_lines))


def main() -> None:
    """Main execution."""
    logger.info("Starting data availability check...")
    logger.info("")

    # Check each data source
    btc_data = check_btc_spot_prices()
    backtest_data = check_production_backtest_data()
    iv_data = check_iv_data()

    # Generate report
    generate_report(btc_data, backtest_data, iv_data)

    logger.info("")
    logger.info("✅ Data availability check complete")
    logger.info(f"   Report: {OUTPUT_REPORT}")


if __name__ == "__main__":
    main()
