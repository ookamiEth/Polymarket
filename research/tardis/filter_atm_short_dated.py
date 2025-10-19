#!/usr/bin/env python3
"""
Filter Deribit options quotes to only ATM (±3%) short-dated (≤3 days) BTC options.

This script:
1. Parses expiry dates from option symbols (e.g., "BTC-29DEC23-27000-C")
2. Joins options quotes with BTC perpetual spot prices (exact timestamp match)
3. Calculates moneyness (spot/strike) and time-to-expiry
4. Filters for ATM (±3%) contracts with ≤3 days until expiry
5. Adds derived fields (mid_price, spreads, flags)
6. Outputs filtered dataset to quotes_1s_atm_short_dated.parquet

Input files:
- quotes_1s_merged.parquet (1.1B rows, 8.2 GB)
- deribit_btc_perpetual_1s.parquet (13.5M rows)

Output:
- quotes_1s_atm_short_dated.parquet (estimated 2-10M rows)
"""

import logging
import sys
from datetime import datetime
from pathlib import Path

import polars as pl

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# Constants
QUOTES_FILE = "quotes_1s_merged.parquet"
PERPETUAL_FILE = "deribit_btc_perpetual_1s.parquet"
OUTPUT_FILE = "quotes_1s_atm_short_dated.parquet"

ATM_THRESHOLD = 0.03  # ±3% moneyness
MAX_TTL_SECONDS = 3 * 24 * 60 * 60  # 3 days in seconds
MAX_TTL_DAYS = 3.0

# Month name to number mapping for expiry parsing
MONTH_MAP = {
    "JAN": "01",
    "FEB": "02",
    "MAR": "03",
    "APR": "04",
    "MAY": "05",
    "JUN": "06",
    "JUL": "07",
    "AUG": "08",
    "SEP": "09",
    "OCT": "10",
    "NOV": "11",
    "DEC": "12",
}


def parse_expiry_dates(df: pl.LazyFrame) -> pl.LazyFrame:
    """
    Parse expiry dates from symbol strings and add expiry_timestamp column.

    Symbol format: BTC-29DEC23-27000-C
    Expiry format: DDMMMYY (e.g., 29DEC23)

    Uses vectorized Polars string operations (no row loops).

    Args:
        df: LazyFrame with 'expiry_str' column

    Returns:
        LazyFrame with added 'expiry_timestamp' (Int64, Unix seconds)
    """
    logger.info("Parsing expiry dates from symbols...")

    # Extract day, month, year from expiry_str
    # Format: DDMMMYY (e.g., "29DEC23") or DMMMYY (e.g., "2OCT23")
    # Use regex to handle variable-length day field

    # Pattern: (\d{1,2})([A-Z]{3})(\d{2})
    # Group 1: 1-2 digit day
    # Group 2: 3-letter month
    # Group 3: 2-digit year

    df = df.with_columns(
        [
            # Extract using regex groups
            pl.col("expiry_str").str.extract(r"^(\d{1,2})", 1).alias("expiry_day"),
            pl.col("expiry_str").str.extract(r"([A-Z]{3})", 1).alias("expiry_month_str"),
            pl.col("expiry_str").str.extract(r"(\d{2})$", 1).alias("expiry_year_short"),
        ]
    )

    # Replace month names with numbers using when-then chains
    month_expr = pl.col("expiry_month_str")
    for month_name, month_num in MONTH_MAP.items():
        month_expr = pl.when(pl.col("expiry_month_str") == month_name).then(pl.lit(month_num)).otherwise(month_expr)

    df = df.with_columns([month_expr.alias("expiry_month")])

    # Zero-pad day to 2 digits (e.g., "2" -> "02")
    df = df.with_columns([pl.col("expiry_day").str.zfill(2).alias("expiry_day")])

    # Build ISO date string: "20YY-MM-DD"
    df = df.with_columns(
        [
            (
                pl.lit("20")
                + pl.col("expiry_year_short")
                + pl.lit("-")
                + pl.col("expiry_month")
                + pl.lit("-")
                + pl.col("expiry_day")
            ).alias("expiry_date_iso")
        ]
    )

    # Parse to datetime, then convert to Unix timestamp (seconds)
    # Expiry is typically at 08:00 UTC on Deribit, but we'll use midnight for simplicity
    # This is conservative (slightly underestimates TTL)
    # Handle malformed dates by setting them to far future (filtered out later)
    df = df.with_columns(
        [
            pl.when(pl.col("expiry_date_iso").str.contains(r"^\d{4}-\d{2}-\d{2}$"))
            .then(
                pl.col("expiry_date_iso")
                .str.strptime(pl.Date, "%Y-%m-%d", strict=False)
                .cast(pl.Datetime)
                .dt.epoch("s")
            )
            .otherwise(pl.lit(2147483647))  # Unix timestamp max (year 2038) for invalid dates
            .alias("expiry_timestamp")
        ]
    )

    # Drop intermediate columns
    df = df.drop(
        [
            "expiry_day",
            "expiry_month_str",
            "expiry_year_short",
            "expiry_month",
            "expiry_date_iso",
        ]
    )

    logger.info("✅ Expiry dates parsed successfully")
    return df


def prepare_spot_prices(perpetual_file: str) -> tuple[pl.DataFrame, int, int]:
    """
    Prepare BTC perpetual spot prices for joining with options quotes.

    Converts microsecond timestamps to seconds and creates lookup table.
    LOADS EAGERLY into memory (287MB file fits comfortably in 16GB RAM).
    This allows Polars optimizer to use exact join statistics.

    Args:
        perpetual_file: Path to perpetual trades Parquet file

    Returns:
        Tuple of (DataFrame with timestamp_seconds & spot_price, min_timestamp, max_timestamp)
    """
    logger.info(f"Loading perpetual data from {perpetual_file}...")

    # ✅ EAGER load - 287MB file fits in memory, gives optimizer exact stats
    df = pl.read_parquet(perpetual_file)

    # Convert microsecond timestamp to seconds
    df = df.with_columns(
        [
            (pl.col("timestamp") // 1_000_000).alias("timestamp_seconds"),
            pl.col("price").alias("spot_price"),
        ]
    )

    # Select only needed columns
    df = df.select(["timestamp_seconds", "spot_price"])

    # Handle duplicates: use last price per second (consistent with 1s sampling)
    # Group by timestamp_seconds and take last spot_price
    df = df.group_by("timestamp_seconds").agg(pl.col("spot_price").last()).sort("timestamp_seconds")

    # Get timestamp bounds for temporal filtering
    min_ts = int(df["timestamp_seconds"].min())  # type: ignore[arg-type]
    max_ts = int(df["timestamp_seconds"].max())  # type: ignore[arg-type]

    # Log memory usage for transparency
    memory_mb = df.estimated_size("mb")
    logger.info(f"✅ Loaded {len(df):,} spot prices ({memory_mb:.1f} MB in memory)")
    logger.info(
        f"   Timestamp range: {min_ts} to {max_ts} ({datetime.fromtimestamp(min_ts)} to {datetime.fromtimestamp(max_ts)})"
    )
    return df, min_ts, max_ts


def apply_temporal_prefilter(quotes_df: pl.LazyFrame, min_ts: int, max_ts: int) -> pl.LazyFrame:
    """
    Apply temporal pre-filter to reduce dataset before expensive operations.

    CRITICAL OPTIMIZATION: This reduces 1.1B rows → ~50M rows using a cheap Int64 filter
    BEFORE joining with spot prices or parsing dates.

    Filters for quotes within the perpetual data timestamp range, which eliminates
    all quotes outside the available spot price window (cannot be enriched anyway).

    Args:
        quotes_df: LazyFrame of options quotes
        min_ts: Minimum timestamp from perpetual data (Unix seconds)
        max_ts: Maximum timestamp from perpetual data (Unix seconds)

    Returns:
        Filtered LazyFrame (only rows with timestamps in [min_ts, max_ts])
    """
    logger.info(
        f"Applying temporal pre-filter: {datetime.fromtimestamp(min_ts)} to {datetime.fromtimestamp(max_ts)}..."
    )

    df = quotes_df.filter((pl.col("timestamp_seconds") >= min_ts) & (pl.col("timestamp_seconds") <= max_ts))

    # DO NOT collect() here - breaks lazy execution!
    # Statistics will be gathered after pipeline completes
    logger.info("✅ Temporal filter applied (timestamp range filter)")

    return df


def apply_cheap_filters(quotes_df: pl.LazyFrame) -> pl.LazyFrame:
    """
    Apply cheap non-join filters before expensive operations.

    CRITICAL OPTIMIZATION: Reduces row count BEFORE regex parsing and joins.

    Filters applied (all cheap operations):
    1. BTC underlying only (string equality)
    2. Has at least one quote side (null checks)

    This is done BEFORE:
    - Parsing expiry dates (expensive regex)
    - Joining with spot prices (expensive join)
    - Calculating moneyness (requires spot prices)

    Args:
        quotes_df: Temporally filtered LazyFrame

    Returns:
        LazyFrame with cheap filters applied
    """
    logger.info("Applying cheap filters: BTC only, has quotes...")

    df = quotes_df.filter(
        (pl.col("underlying") == "BTC") & ((pl.col("bid_price").is_not_null()) | (pl.col("ask_price").is_not_null()))
    )

    # DO NOT collect() here - breaks lazy execution!
    logger.info("✅ Cheap filters applied (BTC underlying, non-null quotes)")

    return df


def enrich_quotes(quotes_df: pl.LazyFrame, spot_df: pl.DataFrame) -> pl.LazyFrame:
    """
    Enrich options quotes with spot prices.

    Performs inner join on timestamp_seconds (exact match).
    Adds: spot_price, moneyness

    NOTE: time_to_expiry calculations moved to after date parsing.

    Args:
        quotes_df: LazyFrame of options quotes
        spot_df: DataFrame (EAGER) of spot prices - allows optimizer to use exact join stats

    Returns:
        Enriched LazyFrame with added columns
    """
    logger.info("Joining options quotes with spot prices (exact timestamp match)...")

    # Inner join on timestamp_seconds
    # Polars will efficiently broadcast the small eager DataFrame
    df = quotes_df.join(spot_df.lazy(), on="timestamp_seconds", how="inner")

    # Calculate moneyness (doesn't need expiry date)
    df = df.with_columns(
        [
            # Moneyness: spot / strike
            (pl.col("spot_price") / pl.col("strike_price")).alias("moneyness"),
        ]
    )

    logger.info("✅ Quotes enriched with spot prices")
    return df


def calculate_time_to_expiry(df: pl.LazyFrame) -> pl.LazyFrame:
    """
    Calculate time-to-expiry fields after date parsing.

    Must be called AFTER parse_expiry_dates() as it needs expiry_timestamp.

    Args:
        df: LazyFrame with expiry_timestamp column

    Returns:
        LazyFrame with time_to_expiry_seconds and time_to_expiry_days
    """
    logger.info("Calculating time to expiry...")

    df = df.with_columns(
        [
            # Time to expiry in seconds
            (pl.col("expiry_timestamp") - pl.col("timestamp_seconds")).alias("time_to_expiry_seconds"),
        ]
    )

    # Convert TTL to days (float)
    df = df.with_columns([(pl.col("time_to_expiry_seconds") / 86400.0).alias("time_to_expiry_days")])

    logger.info("✅ Time to expiry calculated")
    return df


def apply_filters(df: pl.LazyFrame) -> pl.LazyFrame:
    """
    Apply ATM and short-dated filters to enriched quotes.

    NOTE: BTC filter and null checks have been applied earlier in pipeline.
    This function applies the FINAL filters that require enrichment:
    - Moneyness: 0.97 to 1.03 (±3% ATM)
    - Time to expiry: 0 to 3 days (not expired, ≤3 days)

    Args:
        df: Enriched LazyFrame with moneyness and time_to_expiry_seconds

    Returns:
        Filtered LazyFrame
    """
    logger.info("Applying final filters: ATM ±3%, TTL ≤3 days...")

    df = df.filter(
        (pl.col("moneyness") >= 1.0 - ATM_THRESHOLD)
        & (pl.col("moneyness") <= 1.0 + ATM_THRESHOLD)
        & (pl.col("time_to_expiry_seconds") > 0)  # Not expired
        & (pl.col("time_to_expiry_seconds") <= MAX_TTL_SECONDS)  # ≤3 days
    )

    # DO NOT collect() here - breaks lazy execution!
    logger.info("✅ Final filters applied (ATM ±3%, TTL ≤3 days)")

    return df


def add_derived_fields(df: pl.LazyFrame) -> pl.LazyFrame:
    """
    Add derived fields: mid_price, spread_abs, has_bid, has_ask.

    Args:
        df: Filtered LazyFrame

    Returns:
        LazyFrame with additional derived columns
    """
    logger.info("Adding derived fields (mid_price, spreads, flags)...")

    df = df.with_columns(
        [
            # Mid-price: average of bid and ask (handle nulls)
            (
                pl.when(pl.col("bid_price").is_not_null() & pl.col("ask_price").is_not_null())
                .then((pl.col("bid_price") + pl.col("ask_price")) / 2.0)
                .when(pl.col("bid_price").is_not_null())
                .then(pl.col("bid_price"))
                .when(pl.col("ask_price").is_not_null())
                .then(pl.col("ask_price"))
                .otherwise(None)
            ).alias("mid_price"),
            # Spread (absolute): ask - bid
            (pl.col("ask_price") - pl.col("bid_price")).alias("spread_abs"),
            # Boolean flags
            pl.col("bid_price").is_not_null().alias("has_bid"),
            pl.col("ask_price").is_not_null().alias("has_ask"),
        ]
    )

    logger.info("✅ Derived fields added")
    return df


def write_output(df: pl.LazyFrame, output_file: str) -> None:
    """
    Write filtered DataFrame to Parquet file using streaming engine.

    Uses Polars streaming sink for memory-efficient processing of large datasets.
    Processes data in batches without loading entire dataset into memory.

    Args:
        df: Filtered LazyFrame to write
        output_file: Output file path
    """
    logger.info(f"Writing filtered data to {output_file} (streaming mode)...")
    logger.info("⏱️  Estimated time: 5-15 minutes (optimized pipeline)...")

    # Streaming sink - processes in batches, minimal memory usage
    # CRITICAL: Polars uses streaming execution automatically with sink_parquet
    # The row_group_size controls memory usage during writes
    df.sink_parquet(
        output_file,
        compression="snappy",
        statistics=True,
        row_group_size=50_000,  # ✅ 50K rows per batch (safer for 16GB RAM machines)
        maintain_order=False,  # ✅ Allow parallel writes across row groups
    )

    # Get file size
    file_size_mb = Path(output_file).stat().st_size / (1024 * 1024)
    logger.info(f"✅ Wrote data to {output_file} ({file_size_mb:.1f} MB)")


def generate_report(output_file: str) -> None:
    """
    Generate validation report for filtered dataset.

    Loads data lazily and uses aggregations to avoid loading full dataset into memory.

    Args:
        output_file: Output file path
    """
    logger.info("\n" + "=" * 80)
    logger.info("FILTERING REPORT")
    logger.info("=" * 80)

    # Load lazily for memory-efficient stats
    df_lazy = pl.scan_parquet(output_file)

    # Get row count (single scan, minimal memory)
    row_count = df_lazy.select(pl.len()).collect().item()
    unique_symbols = df_lazy.select(pl.col("symbol").n_unique()).collect().item()

    # Basic stats
    logger.info("\n📊 Dataset Statistics:")
    logger.info(f"  Total rows: {row_count:,}")
    logger.info(f"  Unique symbols: {unique_symbols:,}")
    logger.info(f"  Output file: {output_file}")
    logger.info(f"  File size: {Path(output_file).stat().st_size / (1024**2):.1f} MB")

    # Time range - collect aggregated stats only
    time_stats = (
        df_lazy.select(
            [
                pl.col("timestamp_seconds").min().alias("min_ts"),
                pl.col("timestamp_seconds").max().alias("max_ts"),
            ]
        )
        .collect()
        .row(0, named=True)
    )

    min_ts = time_stats["min_ts"]
    max_ts = time_stats["max_ts"]

    if min_ts is None or max_ts is None:
        logger.warning("⚠️  Unable to determine time range (empty timestamp column)")
    else:
        # Convert Polars values to Python int for type safety
        min_ts_int = int(min_ts)  # type: ignore[arg-type]
        max_ts_int = int(max_ts)  # type: ignore[arg-type]

        logger.info("\n📅 Time Range:")
        logger.info(f"  Start: {datetime.fromtimestamp(min_ts_int)} (ts: {min_ts_int})")
        logger.info(f"  End: {datetime.fromtimestamp(max_ts_int)} (ts: {max_ts_int})")
        duration_days = (max_ts_int - min_ts_int) / 86400.0
        logger.info(f"  Duration: {duration_days:.1f} days")

    # Moneyness distribution - collect only aggregates
    moneyness_stats = (
        df_lazy.select(
            [
                pl.col("moneyness").min().alias("min"),
                pl.col("moneyness").quantile(0.25).alias("p25"),
                pl.col("moneyness").quantile(0.5).alias("p50"),
                pl.col("moneyness").quantile(0.75).alias("p75"),
                pl.col("moneyness").max().alias("max"),
            ]
        )
        .collect()
        .row(0, named=True)
    )

    logger.info("\n💰 Moneyness Distribution:")
    logger.info(f"  Min: {moneyness_stats['min']:.4f}")
    logger.info(f"  25th percentile: {moneyness_stats['p25']:.4f}")
    logger.info(f"  Median: {moneyness_stats['p50']:.4f}")
    logger.info(f"  75th percentile: {moneyness_stats['p75']:.4f}")
    logger.info(f"  Max: {moneyness_stats['max']:.4f}")

    # Time to expiry distribution
    tte_stats = (
        df_lazy.select(
            [
                pl.col("time_to_expiry_days").min().alias("min"),
                pl.col("time_to_expiry_days").quantile(0.25).alias("p25"),
                pl.col("time_to_expiry_days").quantile(0.5).alias("p50"),
                pl.col("time_to_expiry_days").quantile(0.75).alias("p75"),
                pl.col("time_to_expiry_days").max().alias("max"),
            ]
        )
        .collect()
        .row(0, named=True)
    )

    logger.info("\n⏰ Time to Expiry Distribution (days):")
    logger.info(f"  Min: {tte_stats['min']:.3f}")
    logger.info(f"  25th percentile: {tte_stats['p25']:.3f}")
    logger.info(f"  Median: {tte_stats['p50']:.3f}")
    logger.info(f"  75th percentile: {tte_stats['p75']:.3f}")
    logger.info(f"  Max: {tte_stats['max']:.3f}")

    # Option type distribution
    type_counts = df_lazy.group_by("type").agg(pl.len().alias("count")).sort("count", descending=True).collect()
    logger.info("\n📈 Option Type Distribution:")
    for row in type_counts.iter_rows(named=True):
        pct = row["count"] / row_count * 100
        logger.info(f"  {row['type']}: {row['count']:,} ({pct:.1f}%)")

    # Quote completeness - use aggregations
    completeness_stats = (
        df_lazy.select(
            [
                (pl.col("has_bid") & pl.col("has_ask")).sum().alias("has_both"),
                (pl.col("has_bid") & ~pl.col("has_ask")).sum().alias("has_bid_only"),
                (~pl.col("has_bid") & pl.col("has_ask")).sum().alias("has_ask_only"),
            ]
        )
        .collect()
        .row(0, named=True)
    )

    logger.info("\n📋 Quote Completeness:")
    logger.info(
        f"  Both bid & ask: {completeness_stats['has_both']:,} ({completeness_stats['has_both'] / row_count * 100:.1f}%)"
    )
    logger.info(
        f"  Bid only: {completeness_stats['has_bid_only']:,} ({completeness_stats['has_bid_only'] / row_count * 100:.1f}%)"
    )
    logger.info(
        f"  Ask only: {completeness_stats['has_ask_only']:,} ({completeness_stats['has_ask_only'] / row_count * 100:.1f}%)"
    )

    # Top symbols by quote count
    logger.info("\n🏆 Top 10 Symbols by Quote Count:")
    top_symbols = (
        df_lazy.group_by("symbol").agg(pl.len().alias("count")).sort("count", descending=True).head(10).collect()
    )
    for row in top_symbols.iter_rows(named=True):
        logger.info(f"  {row['symbol']}: {row['count']:,}")

    # Sample rows - only collect 5 rows
    logger.info("\n🔍 Sample Rows (first 5):")
    sample = (
        df_lazy.head(5)
        .select(
            [
                "timestamp_seconds",
                "symbol",
                "spot_price",
                "strike_price",
                "moneyness",
                "time_to_expiry_days",
                "mid_price",
            ]
        )
        .collect()
    )
    logger.info(f"\n{sample}")

    logger.info("\n" + "=" * 80)
    logger.info("✅ FILTERING COMPLETE")
    logger.info("=" * 80)


def main() -> None:
    """Main execution function with optimized filter-first pipeline."""
    start_time = datetime.now()
    logger.info("=" * 80)
    logger.info("🚀 OPTIMIZED ATM SHORT-DATED OPTIONS FILTERING")
    logger.info("=" * 80)
    logger.info(f"Start time: {start_time}")
    logger.info("Pipeline: Temporal filter → Cheap filters → Parse dates → Join → ATM filter → Write")
    logger.info("Expected runtime: 5-15 minutes for 1.1B input rows\n")

    try:
        # ==================== STAGE 1: LOAD DATA ====================
        logger.info("📂 STAGE 1: Loading data (lazy)...")

        # Check if input files exist
        if not Path(QUOTES_FILE).exists():
            logger.error(f"❌ Input file not found: {QUOTES_FILE}")
            sys.exit(1)
        if not Path(PERPETUAL_FILE).exists():
            logger.error(f"❌ Input file not found: {PERPETUAL_FILE}")
            sys.exit(1)

        quotes_df = pl.scan_parquet(QUOTES_FILE)
        quotes_count = quotes_df.select(pl.len()).collect().item()
        logger.info(f"✅ Loaded {quotes_count:,} option quotes (lazy)\n")

        # ==================== STAGE 2: PREPARE SPOT PRICES ====================
        logger.info("📊 STAGE 2: Preparing spot prices...")
        spot_df, min_ts, max_ts = prepare_spot_prices(PERPETUAL_FILE)

        # Check if we have spot prices
        if len(spot_df) == 0:
            logger.error("❌ No spot prices found in perpetual data file")
            sys.exit(1)

        logger.info("")

        # ==================== STAGE 3: TEMPORAL PRE-FILTER ====================
        logger.info("⏰ STAGE 3: Temporal pre-filter (cheap Int64 filter)...")
        quotes_df = apply_temporal_prefilter(quotes_df, min_ts, max_ts)
        logger.info("")

        # ==================== STAGE 4: CHEAP FILTERS ====================
        logger.info("🔍 STAGE 4: Applying cheap filters (BTC, has quotes)...")
        quotes_df = apply_cheap_filters(quotes_df)
        logger.info("")

        # ==================== STAGE 5: ENRICH WITH SPOT PRICES ====================
        logger.info("💰 STAGE 5: Enriching with spot prices (join on reduced dataset)...")
        enriched_df = enrich_quotes(quotes_df, spot_df)
        logger.info("")

        # ==================== STAGE 6: PARSE EXPIRY DATES ====================
        logger.info("📅 STAGE 6: Parsing expiry dates (regex AFTER join - fewer rows)...")
        enriched_df = parse_expiry_dates(enriched_df)
        logger.info("")

        # ==================== STAGE 7: CALCULATE TIME TO EXPIRY ====================
        logger.info("⏳ STAGE 7: Calculating time to expiry...")
        enriched_df = calculate_time_to_expiry(enriched_df)
        logger.info("")

        # ==================== STAGE 8: FINAL FILTERS ====================
        logger.info("🎯 STAGE 8: Applying final filters (ATM ±3%, TTL ≤3 days)...")
        filtered_df = apply_filters(enriched_df)
        logger.info("")

        # ==================== STAGE 9: ADD DERIVED FIELDS ====================
        logger.info("📈 STAGE 9: Adding derived fields (mid_price, spreads)...")
        final_df = add_derived_fields(filtered_df)
        logger.info("")

        # ==================== STAGE 10: WRITE OUTPUT ====================
        logger.info("💾 STAGE 10: Writing output (streaming mode)...")
        write_output(final_df, OUTPUT_FILE)
        logger.info("")

        # ==================== STAGE 11: GENERATE REPORT ====================
        logger.info("📋 STAGE 11: Generating validation report...")
        generate_report(OUTPUT_FILE)

        # Timing and performance summary
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()

        # Get actual output statistics (lazy count, safe)
        output_count = pl.scan_parquet(OUTPUT_FILE).select(pl.len()).collect().item()
        reduction_factor = quotes_count / output_count if output_count > 0 else float("inf")

        logger.info("\n" + "=" * 80)
        logger.info("🎯 PERFORMANCE SUMMARY")
        logger.info("=" * 80)
        logger.info(f"⏱️  Total runtime: {duration:.1f} seconds ({duration / 60:.1f} minutes)")
        logger.info(f"📊 Data reduction: {quotes_count:,} → {output_count:,} rows ({reduction_factor:.1f}x reduction)")
        logger.info(f"💾 Output file size: {Path(OUTPUT_FILE).stat().st_size / (1024**2):.1f} MB")
        logger.info(f"⚡ Processing rate: {quotes_count / duration:,.0f} rows/second")

    except Exception as e:
        logger.error(f"❌ Error during filtering: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
