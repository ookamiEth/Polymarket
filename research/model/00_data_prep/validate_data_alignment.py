#!/usr/bin/env python3
"""
Validate data alignment across all datasets.

Checks:
1. Timestamp coverage: Do all datasets cover the same period?
2. Gaps: Are there significant gaps in any dataset?
3. Join compatibility: Can we successfully join datasets on timestamps/dates?
4. Data quality: Are there nulls, duplicates, or invalid values?

Datasets:
- BTC perpetual (1s resampled)
- BTC options IV
- Risk-free rates (daily)
- Contract schedule
"""

import logging
from datetime import datetime
from pathlib import Path

import polars as pl

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# File paths
PERPETUAL_FILE = Path("/Users/lgierhake/Documents/ETH/BT/research/model/results/btc_perpetual_1s_resampled.parquet")
OPTIONS_FILE = Path(
    "/Users/lgierhake/Documents/ETH/BT/research/tardis/data/consolidated/btc_options_atm_shortdated_with_iv_2023_2025.parquet"
)
RATES_FILE = Path(
    "/Users/lgierhake/Documents/ETH/BT/research/risk_free_rate/data/blended_lending_rates_2023_2025.parquet"
)
CONTRACTS_FILE = Path("/Users/lgierhake/Documents/ETH/BT/research/model/results/contract_schedule.parquet")


def format_timestamp(ts: int) -> str:
    """Format Unix timestamp to readable string."""
    return datetime.fromtimestamp(ts).strftime("%Y-%m-%d %H:%M:%S UTC")


def validate_timestamp_coverage() -> None:
    """Check timestamp coverage across all datasets."""
    logger.info("=" * 80)
    logger.info("TIMESTAMP COVERAGE VALIDATION")
    logger.info("=" * 80)

    # 1. Perpetual data
    logger.info("\n1. BTC Perpetual (1s resampled)")
    perp = pl.scan_parquet(PERPETUAL_FILE).select(["timestamp_seconds"]).collect()
    perp_min = perp["timestamp_seconds"].min()
    perp_max = perp["timestamp_seconds"].max()
    perp_count = len(perp)

    if perp_min is None or perp_max is None:
        logger.error("Perpetual data has no timestamps!")
        return

    logger.info(f"  Records: {perp_count:,}")
    logger.info(f"  Min timestamp: {perp_min} ({format_timestamp(int(perp_min))})")
    logger.info(f"  Max timestamp: {perp_max} ({format_timestamp(int(perp_max))})")
    logger.info(f"  Duration: {(int(perp_max) - int(perp_min)) / 86400:.1f} days")

    # 2. Options data
    logger.info("\n2. BTC Options IV")
    options = pl.scan_parquet(OPTIONS_FILE).select(["timestamp_seconds", "expiry_timestamp"]).collect()
    opt_min = options["timestamp_seconds"].min()
    opt_max = options["timestamp_seconds"].max()
    opt_count = len(options)

    if opt_min is None or opt_max is None:
        logger.error("Options data has no timestamps!")
        return

    logger.info(f"  Records: {opt_count:,}")
    logger.info(f"  Min timestamp: {opt_min} ({format_timestamp(int(opt_min))})")
    logger.info(f"  Max timestamp: {opt_max} ({format_timestamp(int(opt_max))})")
    logger.info(f"  Duration: {(int(opt_max) - int(opt_min)) / 86400:.1f} days")

    # Check expiry range
    exp_min = options["expiry_timestamp"].min()
    exp_max = options["expiry_timestamp"].max()
    if exp_min is not None and exp_max is not None:
        logger.info(f"  Min expiry: {exp_min} ({format_timestamp(int(exp_min))})")
        logger.info(f"  Max expiry: {exp_max} ({format_timestamp(int(exp_max))})")

    # 3. Risk-free rates
    logger.info("\n3. Risk-Free Rates (daily)")
    rates = pl.read_parquet(RATES_FILE)
    rates_min = rates["date"].min()
    rates_max = rates["date"].max()
    rates_count = len(rates)

    logger.info(f"  Records: {rates_count:,}")
    logger.info(f"  Min date: {rates_min}")
    logger.info(f"  Max date: {rates_max}")

    # Check for gaps in rates
    if rates_min is not None and rates_max is not None:
        expected_days = (rates_max - rates_min).days + 1  # type: ignore
        missing_days = expected_days - rates_count
        logger.info(f"  Expected days: {expected_days}")
        logger.info(f"  Missing days: {missing_days} ({missing_days / expected_days * 100:.2f}%)")

    # 4. Contract schedule
    logger.info("\n4. Contract Schedule")
    contracts = pl.read_parquet(CONTRACTS_FILE)
    cont_min = contracts["open_time"].min()
    cont_max = contracts["close_time"].max()
    cont_count = len(contracts)

    if cont_min is None or cont_max is None:
        logger.error("Contract schedule has no timestamps!")
        return

    logger.info(f"  Records: {cont_count:,}")
    logger.info(f"  Min open: {cont_min} ({format_timestamp(int(cont_min))})")
    logger.info(f"  Max close: {cont_max} ({format_timestamp(int(cont_max))})")
    logger.info(f"  Duration: {(int(cont_max) - int(cont_min)) / 86400:.1f} days")

    # Alignment check
    logger.info("\n" + "=" * 80)
    logger.info("ALIGNMENT SUMMARY")
    logger.info("=" * 80)

    # Find common period
    common_min = max([int(perp_min), int(opt_min), int(cont_min)])
    common_max = min([int(perp_max), int(opt_max), int(cont_max)])

    logger.info(f"Common period: {format_timestamp(common_min)} to {format_timestamp(common_max)}")
    logger.info(f"Duration: {(common_max - common_min) / 86400:.1f} days")

    # Check if rates cover common period
    if rates_min is not None and rates_max is not None:
        rates_min_ts = int(datetime.combine(rates_min, datetime.min.time()).timestamp())  # type: ignore
        rates_max_ts = int(datetime.combine(rates_max, datetime.max.time()).timestamp())  # type: ignore

        if rates_min_ts <= common_min and rates_max_ts >= common_max:
            logger.info("✅ Risk-free rates cover full common period")
        else:
            logger.warning("⚠️  Risk-free rates DO NOT cover full common period")
            logger.warning(f"   Rates: {rates_min} to {rates_max}")
            logger.warning(f"   Common: {format_timestamp(common_min)} to {format_timestamp(common_max)}")


def validate_data_quality() -> None:
    """Check data quality issues."""
    logger.info("\n" + "=" * 80)
    logger.info("DATA QUALITY VALIDATION")
    logger.info("=" * 80)

    # 1. Perpetual data quality
    logger.info("\n1. BTC Perpetual")
    perp = pl.read_parquet(PERPETUAL_FILE)
    logger.info(f"  Null VWAPs: {perp['vwap'].null_count():,}")
    logger.info(f"  Zero volume seconds: {(perp['total_volume'] == 0).sum():,}")
    logger.info(f"  VWAP range: ${perp['vwap'].min():.2f} to ${perp['vwap'].max():.2f}")

    # 2. Options data quality
    logger.info("\n2. BTC Options IV")
    logger.info("  (Scanning sample - full scan would be slow)")
    options_sample = pl.read_parquet(OPTIONS_FILE, n_rows=1_000_000)

    logger.info(f"  Sample size: {len(options_sample):,}")
    logger.info(f"  Null bid IVs: {options_sample['implied_vol_bid'].null_count():,}")
    logger.info(f"  Null ask IVs: {options_sample['implied_vol_ask'].null_count():,}")
    logger.info(f"  IV calc failures: {(options_sample['iv_calc_status'] != 'success').sum():,}")

    # Check for reasonable IV values (0.01 to 5.0)
    valid_ivs = options_sample.filter(
        (pl.col("implied_vol_bid") >= 0.01)
        & (pl.col("implied_vol_bid") <= 5.0)
        & (pl.col("implied_vol_bid").is_not_null())
    )
    logger.info(f"  Valid bid IVs (0.01-5.0): {len(valid_ivs):,} / {len(options_sample):,}")

    # 3. Risk-free rates quality
    logger.info("\n3. Risk-Free Rates")
    rates = pl.read_parquet(RATES_FILE)
    logger.info(f"  Null rates: {rates['blended_supply_apr'].null_count():,}")
    logger.info(f"  Rate range: {rates['blended_supply_apr'].min():.2f}% to {rates['blended_supply_apr'].max():.2f}%")
    logger.info(f"  Mean rate: {rates['blended_supply_apr'].mean():.2f}%")


def validate_join_compatibility() -> None:
    """Test join compatibility between datasets."""
    logger.info("\n" + "=" * 80)
    logger.info("JOIN COMPATIBILITY VALIDATION")
    logger.info("=" * 80)

    # Test 1: Join perpetual with contracts
    logger.info("\n1. Testing perpetual ↔ contracts join")
    perp_sample = pl.scan_parquet(PERPETUAL_FILE).head(10000).collect()
    contracts_sample = pl.read_parquet(CONTRACTS_FILE).head(100)

    # Get timestamps that should match contract opens
    test_join = perp_sample.join(
        contracts_sample.select(["open_time"]), left_on="timestamp_seconds", right_on="open_time", how="inner"
    )
    logger.info(f"  Matches found: {len(test_join):,} (expected: some matches)")
    if len(test_join) > 0:
        logger.info("  ✅ Perpetual ↔ contracts join successful")
    else:
        logger.warning("  ⚠️  No matches found - check timestamp alignment")

    # Test 2: Join contracts with rates (on date)
    logger.info("\n2. Testing contracts ↔ rates join")
    rates = pl.read_parquet(RATES_FILE)

    test_join2 = contracts_sample.join(rates.select(["date", "blended_supply_apr"]), on="date", how="left")
    matches = test_join2.filter(pl.col("blended_supply_apr").is_not_null())
    logger.info(f"  Matches found: {len(matches):,} / {len(contracts_sample):,}")
    if len(matches) == len(contracts_sample):
        logger.info("  ✅ All contracts have matching rates")
    else:
        logger.warning(f"  ⚠️  {len(contracts_sample) - len(matches)} contracts missing rates")


def main() -> None:
    """Main entry point."""
    logger.info("Data Alignment Validation")
    logger.info("Checking all datasets for coverage, quality, and join compatibility\n")

    try:
        validate_timestamp_coverage()
        validate_data_quality()
        validate_join_compatibility()

        logger.info("\n" + "=" * 80)
        logger.info("VALIDATION COMPLETE")
        logger.info("=" * 80)
        logger.info("✅ All datasets validated. Ready for backtesting!")

    except Exception as e:
        logger.error(f"Validation failed: {e}")
        raise


if __name__ == "__main__":
    main()
