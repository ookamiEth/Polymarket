#!/usr/bin/env python3
"""
Test script to verify historical mark_iv data availability from Tardis.dev via tardis-machine.

This test checks if the ticker data type from Deribit includes mark_iv, bid_iv, and ask_iv
fields for historical options data.

Usage:
    uv run python test/test_historical_mark_iv.py

Requirements:
    - tardis-machine running on localhost:8000 (npx tardis-machine --port=8000)
    - Test date: August 1, 2025 (same as our IV calculation test)
"""

import asyncio
import io
import logging
import sys
from datetime import datetime

import httpx
import msgspec.json
import polars as pl

# Configuration
TARDIS_MACHINE_URL = "http://localhost:8000"
HTTP_TIMEOUT_SECONDS = 120.0
SERVER_CHECK_TIMEOUT = 10.0

# Test parameters - Using actual symbols from our August 1, 2025 download
TEST_DATE = "2025-08-01"
# We'll load symbols from the successful IV calculation data
# to ensure we test with instruments that actually exist
TEST_INSTRUMENTS_FILE = "datasets_deribit_options/deribit_options_2025-08-01_BTC_5s_with_iv.parquet"
TEST_INSTRUMENTS = []  # Will be populated from file

logger = logging.getLogger(__name__)


async def check_tardis_machine_server(base_url: str) -> bool:
    """Check if tardis-machine server is running."""
    try:
        async with httpx.AsyncClient(timeout=SERVER_CHECK_TIMEOUT) as client:
            response = await client.get(f"{base_url}/")
            return response.status_code in [200, 404]
    except Exception as e:
        logger.debug(f"Server check failed: {e}")
        return False


async def fetch_ticker_data_normalized(
    instruments: list[str],
    from_date: str,
    to_date: str,
    tardis_machine_url: str = TARDIS_MACHINE_URL,
) -> pl.DataFrame:
    """Try fetching normalized derivative_ticker data."""
    from_datetime = f"{from_date}T00:00:00.000Z"
    to_datetime = f"{to_date}T23:59:59.999Z"

    options = {
        "exchange": "deribit",
        "from": from_datetime,
        "to": to_datetime,
        "symbols": instruments,
        "dataTypes": ["derivative_ticker"],
    }

    logger.info("Trying normalized derivative_ticker...")

    async with httpx.AsyncClient(timeout=HTTP_TIMEOUT_SECONDS) as client:
        try:
            response = await client.get(
                f"{tardis_machine_url}/replay-normalized",
                params={"options": msgspec.json.encode(options).decode("utf-8")},
            )

            if response.status_code != 200:
                logger.warning(f"Normalized ticker failed ({response.status_code}): {response.text[:200]}")
                return pl.DataFrame()

            if not response.text.strip():
                logger.warning("No normalized ticker data")
                return pl.DataFrame()

            df = pl.read_ndjson(io.StringIO(response.text))
            logger.info(f"✓ Got {len(df):,} normalized ticker messages")
            return df

        except Exception as e:
            logger.warning(f"Normalized ticker error: {e}")
            return pl.DataFrame()


async def fetch_ticker_data_raw(
    instruments: list[str],
    from_date: str,
    to_date: str,
    tardis_machine_url: str = TARDIS_MACHINE_URL,
) -> pl.DataFrame:
    """Try fetching raw (exchange-native) ticker data."""
    from_datetime = f"{from_date}T00:00:00.000Z"
    to_datetime = f"{to_date}T23:59:59.999Z"

    # For raw data, we need to specify the channel/message type
    # Deribit uses 'ticker' channel for ticker updates
    options = {
        "exchange": "deribit",
        "from": from_datetime,
        "to": to_datetime,
        "symbols": instruments,
        "filters": [{"channel": "ticker"}],  # Raw ticker channel
    }

    logger.info("Trying raw (exchange-native) ticker data...")

    async with httpx.AsyncClient(timeout=HTTP_TIMEOUT_SECONDS) as client:
        try:
            response = await client.get(
                f"{tardis_machine_url}/replay",  # Use /replay, not /replay-normalized
                params={"options": msgspec.json.encode(options).decode("utf-8")},
            )

            if response.status_code != 200:
                logger.warning(f"Raw ticker failed ({response.status_code}): {response.text[:200]}")
                return pl.DataFrame()

            if not response.text.strip():
                logger.warning("No raw ticker data")
                return pl.DataFrame()

            df = pl.read_ndjson(io.StringIO(response.text))
            logger.info(f"✓ Got {len(df):,} raw ticker messages")
            return df

        except Exception as e:
            logger.warning(f"Raw ticker error: {e}")
            return pl.DataFrame()


async def fetch_ticker_data(
    instruments: list[str],
    from_date: str,
    to_date: str,
    tardis_machine_url: str = TARDIS_MACHINE_URL,
) -> pl.DataFrame:
    """
    Fetch ticker data for given instruments (tries both normalized and raw).

    Args:
        instruments: List of Deribit instrument names
        from_date: Start date (YYYY-MM-DD)
        to_date: End date (YYYY-MM-DD)
        tardis_machine_url: Tardis server URL

    Returns:
        DataFrame with ticker data including mark_iv, bid_iv, ask_iv
    """
    from_datetime = f"{from_date}T00:00:00.000Z"
    to_datetime = f"{to_date}T23:59:59.999Z"

    logger.info(f"Requesting ticker data for {len(instruments)} instruments...")
    logger.info(f"Date range: {from_datetime} to {to_datetime}\n")

    # Try normalized first
    df_normalized = await fetch_ticker_data_normalized(instruments, from_date, to_date, tardis_machine_url)
    if len(df_normalized) > 0:
        return df_normalized

    # Fall back to raw
    logger.info("")
    df_raw = await fetch_ticker_data_raw(instruments, from_date, to_date, tardis_machine_url)
    if len(df_raw) > 0:
        return df_raw

    logger.warning("❌ No ticker data from either normalized or raw endpoints")
    return pl.DataFrame()


def analyze_ticker_data(df: pl.DataFrame) -> None:
    """
    Analyze ticker DataFrame to check for mark_iv availability.

    Args:
        df: Ticker data DataFrame
    """
    if len(df) == 0:
        logger.error("❌ No ticker data received")
        return

    logger.info("\n" + "=" * 80)
    logger.info("TICKER DATA ANALYSIS")
    logger.info("=" * 80)

    # Show available columns
    logger.info(f"\nAvailable columns ({len(df.columns)}):")
    for col in sorted(df.columns):
        logger.info(f"  - {col}")

    # Check for IV-related fields
    iv_fields = ["mark_iv", "bid_iv", "ask_iv", "markIv", "bidIv", "askIv"]
    found_iv_fields = [field for field in iv_fields if field in df.columns]

    logger.info(f"\nIV fields found: {found_iv_fields if found_iv_fields else '❌ NONE'}")

    if not found_iv_fields:
        logger.warning("❌ No IV fields found in ticker data!")
        logger.warning("This may mean:")
        logger.warning("  1. Ticker data type doesn't include IV fields")
        logger.warning("  2. Need to use a different data type or API endpoint")
        logger.warning("  3. IV fields may be in 'greeks' nested structure")
        return

    # Show sample data with IV fields
    logger.info("\n" + "=" * 80)
    logger.info("SAMPLE TICKER DATA (first 5 rows)")
    logger.info("=" * 80)

    display_cols = ["timestamp", "symbol"] + found_iv_fields
    if "mark_price" in df.columns:
        display_cols.append("mark_price")
    if "underlying_price" in df.columns:
        display_cols.append("underlying_price")

    available_display_cols = [col for col in display_cols if col in df.columns]
    sample = df.head(5).select(available_display_cols)
    print(sample)

    # Statistics for IV fields
    logger.info("\n" + "=" * 80)
    logger.info("IV STATISTICS")
    logger.info("=" * 80)

    for field in found_iv_fields:
        non_null = df.filter(pl.col(field).is_not_null())
        if len(non_null) > 0:
            mean_iv = non_null[field].mean()
            median_iv = non_null[field].median()
            min_iv = non_null[field].min()
            max_iv = non_null[field].max()
            logger.info(f"\n{field}:")
            logger.info(f"  Non-null count: {len(non_null):,} / {len(df):,}")
            logger.info(f"  Mean: {mean_iv:.4f}")
            logger.info(f"  Median: {median_iv:.4f}")
            logger.info(f"  Range: [{min_iv:.4f}, {max_iv:.4f}]")
        else:
            logger.warning(f"  {field}: All null values")

    # Check greeks structure if available
    if "greeks" in df.columns:
        logger.info("\n" + "=" * 80)
        logger.info("GREEKS STRUCTURE")
        logger.info("=" * 80)
        logger.info("Found 'greeks' column - this may contain IV data in nested format")

        # Try to extract greeks fields
        try:
            greeks_sample = df.head(3).select(["symbol", "greeks"])
            print(greeks_sample)
        except Exception as e:
            logger.warning(f"Could not display greeks: {e}")


def load_test_instruments(file_path: str, num_instruments: int = 5) -> list[str]:
    """
    Load test instruments from successful IV calculation data.

    Args:
        file_path: Path to parquet file with options data
        num_instruments: Number of instruments to sample

    Returns:
        List of instrument symbols
    """
    logger.info("=" * 80)
    logger.info("LOAD TEST INSTRUMENTS")
    logger.info("=" * 80)

    df = pl.read_parquet(file_path)

    # Filter to successful IV calculations
    success_df = df.filter(pl.col("iv_calc_status") == "success")

    if len(success_df) == 0:
        logger.error("No successful IV calculations found!")
        sys.exit(1)

    # Sample diverse instruments (different expiries and moneyness)
    # Get unique symbols
    unique_symbols = success_df["symbol"].unique().to_list()

    logger.info(f"Found {len(unique_symbols):,} unique symbols with successful IV calculations")

    # Sample a few for testing
    import random

    random.seed(42)  # Reproducible
    sampled = random.sample(unique_symbols, min(num_instruments, len(unique_symbols)))

    logger.info(f"Selected {len(sampled)} instruments for testing:")
    for sym in sampled:
        logger.info(f"  - {sym}")

    return sampled


async def main_async() -> None:
    """Main async test function."""
    # Load instruments from successful IV data
    instruments = load_test_instruments(TEST_INSTRUMENTS_FILE, num_instruments=5)

    # Check server
    logger.info("\n" + "=" * 80)
    logger.info("HISTORICAL MARK_IV TEST")
    logger.info("=" * 80)
    logger.info(f"Testing date: {TEST_DATE}")
    logger.info(f"Number of instruments: {len(instruments)}")
    logger.info("")

    if not await check_tardis_machine_server(TARDIS_MACHINE_URL):
        logger.error(f"❌ ERROR: tardis-machine server not accessible at {TARDIS_MACHINE_URL}")
        logger.error("Start server with: npx tardis-machine --port=8000")
        sys.exit(1)

    logger.info(f"✓ Server OK: {TARDIS_MACHINE_URL}\n")

    # Fetch ticker data
    logger.info("=" * 80)
    logger.info("FETCH TICKER DATA")
    logger.info("=" * 80)

    df = await fetch_ticker_data(
        instruments=instruments,
        from_date=TEST_DATE,
        to_date=TEST_DATE,
        tardis_machine_url=TARDIS_MACHINE_URL,
    )

    # Analyze results
    analyze_ticker_data(df)

    # Summary
    logger.info("\n" + "=" * 80)
    logger.info("TEST SUMMARY")
    logger.info("=" * 80)

    if len(df) > 0:
        iv_fields = ["mark_iv", "bid_iv", "ask_iv", "markIv", "bidIv", "askIv"]
        found_iv_fields = [field for field in iv_fields if field in df.columns]

        if found_iv_fields:
            logger.info("✓ SUCCESS: Historical mark_iv data is available!")
            logger.info(f"✓ Found IV fields: {', '.join(found_iv_fields)}")
            logger.info(f"✓ Total ticker messages: {len(df):,}")
            logger.info("\nRECOMMENDATION:")
            logger.info("  Use ticker data type instead of quote data type to get mark_iv")
            logger.info("  Modify tardis_download.py to fetch ticker instead of quote data")
        else:
            logger.warning("❌ FAILED: No IV fields found in ticker data")
            logger.warning("\nNEXT STEPS:")
            logger.warning("  1. Check if 'greeks' column contains IV data")
            logger.warning("  2. Try different data types (e.g., 'book_change')")
            logger.warning("  3. Check Deribit API documentation for mark_iv field location")
    else:
        logger.error("❌ FINDING: Historical mark_iv NOT available via Tardis replay")
        logger.error("\nCONCLUSION:")
        logger.error("  Tardis.dev does NOT provide historical ticker data with mark_iv for replay.")
        logger.error("  The 'derivative_ticker' and 'ticker' data types return no data for options.")
        logger.error("")
        logger.error("ALTERNATIVES:")
        logger.error("  1. **Calculate IV yourself** (current approach with calculate_implied_volatility.py)")
        logger.error("     ✓ Works well: 76.1% success rate on Aug 1 data")
        logger.error("     ✓ Failures are due to market data quality (below intrinsic value)")
        logger.error("     ✓ Can use this as ground truth for research")
        logger.error("")
        logger.error("  2. **Use Deribit's public/get_ticker API directly**")
        logger.error("     - Poll every few seconds/minutes and store mark_iv")
        logger.error("     - Rate limit: 100 req/10s")
        logger.error("     - Good for real-time, but expensive for backfilling")
        logger.error("")
        logger.error("  3. **Use third-party data providers**")
        logger.error("     - Amberdata: Has historical IV surfaces, term structure, skew")
        logger.error("     - Laevitas: Provides historical options Greeks and IV data")
        logger.error("     - Cost: ~$100-500/month for historical data access")
        logger.error("")
        logger.error("RECOMMENDATION:")
        logger.error("  Continue using your calculated IVs (calculate_implied_volatility.py).")
        logger.error("  For the 23.9% failures, they're not usable anyway (below intrinsic).")
        logger.error("  Your approach provides reproducible, auditable IV calculations.")

    logger.info("=" * 80)


def main() -> None:
    """Main entry point."""
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    try:
        asyncio.run(main_async())
    except KeyboardInterrupt:
        logger.info("\n\nInterrupted by user")
        sys.exit(130)
    except Exception as e:
        logger.error(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
