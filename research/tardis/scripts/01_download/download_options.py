#!/usr/bin/env python3

import argparse
import asyncio
import io
import logging
import os
import subprocess
import sys
import time
from datetime import datetime, timedelta
from typing import Optional

import httpx
import msgspec.json
import polars as pl
from dotenv import load_dotenv

DEFAULT_BATCH_SIZE = 50
DEFAULT_MAX_WORKERS = 16  # Optimized for 32 vCPUs (async I/O bound)
DEFAULT_TARDIS_MACHINE_URL = "http://localhost:8000"
HTTP_TIMEOUT_SECONDS = 1800.0
SERVER_CHECK_TIMEOUT = 10.0
DEFAULT_RESAMPLE_INTERVAL = "1s"

BTC_STRIKES = range(40000, 130001, 1000)
ETH_STRIKES = range(1500, 5001, 100)

logger = logging.getLogger(__name__)


def _empty_dataframe() -> pl.DataFrame:
    schema = {
        "exchange": pl.Utf8,
        "symbol": pl.Utf8,
        "timestamp": pl.Int64,
        "local_timestamp": pl.Int64,
        "type": pl.Utf8,
        "strike_price": pl.Float64,
        "underlying": pl.Utf8,
        "expiry_str": pl.Utf8,
        "bid_price": pl.Float64,
        "bid_amount": pl.Float64,
        "ask_price": pl.Float64,
        "ask_amount": pl.Float64,
    }
    return pl.DataFrame(schema=schema)


def start_tardis_machine(api_key: str, port: int = 8000) -> subprocess.Popen:
    """Start tardis-machine server as background process."""
    logger.info(f"Starting tardis-machine server on port {port}...")

    cmd = ["npx", "tardis-machine", f"--port={port}", f"--api-key={api_key}"]

    try:
        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    except FileNotFoundError as e:
        raise Exception("npx not found. Install Node.js first: https://nodejs.org/") from e

    # Give server time to start
    time.sleep(3)

    if process.poll() is not None:
        # Process died
        stderr = process.stderr.read() if process.stderr else ""
        raise Exception(f"tardis-machine failed to start: {stderr}")

    logger.info("tardis-machine server started successfully")
    return process


async def check_tardis_machine_server(base_url: str) -> bool:
    try:
        async with httpx.AsyncClient(timeout=SERVER_CHECK_TIMEOUT) as client:
            response = await client.get(f"{base_url}/")
            return response.status_code in [200, 404]
    except Exception as e:
        logger.debug(f"Server check failed: {e}")
        return False


async def _fetch_single_batch(
    client: httpx.AsyncClient,
    semaphore: asyncio.Semaphore,
    batch_idx: int,
    total_batches: int,
    symbol_batch: list[str],
    from_datetime: str,
    to_datetime: str,
    data_types: list[str],
    tardis_machine_url: str,
) -> tuple[int, pl.DataFrame]:
    async with semaphore:
        logger.info(f"Batch {batch_idx}/{total_batches}: {len(symbol_batch)} symbols...")

        options = {
            "exchange": "deribit",
            "from": from_datetime,
            "to": to_datetime,
            "symbols": symbol_batch,
            "dataTypes": data_types,
        }

        try:
            batch_start = time.time()

            response = await client.get(
                f"{tardis_machine_url}/replay-normalized",
                params={"options": msgspec.json.encode(options).decode("utf-8")},
            )

            if response.status_code != 200:
                raise Exception(f"tardis-machine error ({response.status_code}): {response.text[:500]}")

            batch_elapsed = time.time() - batch_start
            size_mb = len(response.content) / 1_000_000

            if not response.text.strip():
                logger.warning(f"Batch {batch_idx}: No data received")
                return (batch_idx, _empty_dataframe())

            try:
                df = pl.read_ndjson(io.StringIO(response.text))
            except Exception as e:
                logger.error(f"Batch {batch_idx}: JSON parse error: {e}")
                return (batch_idx, _empty_dataframe())

            original_rows = len(df)
            logger.info(
                f"Batch {batch_idx}: Received {original_rows:,} messages ({size_mb:.2f} MB) in {batch_elapsed:.1f}s"
            )

            if original_rows == 0:
                return (batch_idx, _empty_dataframe())

            df = df.filter(pl.col("type").str.contains("quote|book_snapshot"))

            if len(df) == 0:
                logger.warning(f"Batch {batch_idx}: No quote/book_snapshot messages found")
                return (batch_idx, _empty_dataframe())

            df = df.filter(pl.col("symbol").str.count_matches("-") == 3)

            if len(df) == 0:
                logger.warning(f"Batch {batch_idx}: No valid symbols after filtering")
                return (batch_idx, _empty_dataframe())

            df = df.with_columns([pl.col("symbol").str.split("-").alias("symbol_parts")])

            df = df.with_columns(
                [
                    pl.lit("deribit").alias("exchange"),
                    pl.col("symbol_parts").list.get(0).alias("underlying"),
                    pl.col("symbol_parts").list.get(1).alias("expiry_str"),
                    pl.col("symbol_parts").list.get(2).cast(pl.Float64, strict=False).alias("strike_price"),
                    pl.when(pl.col("symbol_parts").list.get(3) == "C")
                    .then(pl.lit("call"))
                    .otherwise(pl.lit("put"))
                    .alias("type"),
                ]
            )

            df = df.drop("symbol_parts")

            df = df.filter(pl.col("strike_price").is_not_null())

            if len(df) == 0:
                logger.warning(f"Batch {batch_idx}: No valid rows after symbol parsing")
                return (batch_idx, _empty_dataframe())

            df = df.with_columns(
                [
                    pl.col("timestamp")
                    .str.strptime(pl.Datetime, format="%Y-%m-%dT%H:%M:%S%.fZ", strict=False)
                    .dt.epoch("us")
                    .fill_null(0)
                    .alias("timestamp"),
                ]
            )

            df = df.with_columns([pl.col("timestamp").alias("local_timestamp")])

            # Extract bid/ask prices and amounts
            # Filter to only rows with non-empty bids AND asks
            df = df.filter((pl.col("bids").list.len() > 0) & (pl.col("asks").list.len() > 0))

            if len(df) == 0:
                logger.warning(f"Batch {batch_idx}: No rows with both bids and asks")
                return (batch_idx, _empty_dataframe())

            # Extract first bid/ask - bids/asks are List[Struct[2 fields]]
            df = df.with_columns(
                [
                    pl.col("bids").list.first().alias("best_bid"),
                    pl.col("asks").list.first().alias("best_ask"),
                ]
            )

            # Drop the original bids/asks columns to avoid column name conflicts
            df = df.drop(["bids", "asks"])

            # Unnest best_bid struct
            cols_before = set(df.columns)
            df = df.unnest("best_bid")
            cols_after = set(df.columns)
            new_bid_cols = sorted(list(cols_after - cols_before))  # Get the 2 new columns

            # Rename them to bid_price and bid_amount (first field is price, second is amount)
            if len(new_bid_cols) == 2:
                df = df.rename({new_bid_cols[0]: "bid_price", new_bid_cols[1]: "bid_amount"})
            else:
                logger.error(f"Batch {batch_idx}: Expected 2 bid columns, got {new_bid_cols}")
                return (batch_idx, _empty_dataframe())

            # Unnest best_ask struct
            cols_before = set(df.columns)
            df = df.unnest("best_ask")
            cols_after = set(df.columns)
            new_ask_cols = sorted(list(cols_after - cols_before))

            # Rename to ask_price and ask_amount
            if len(new_ask_cols) == 2:
                df = df.rename({new_ask_cols[0]: "ask_price", new_ask_cols[1]: "ask_amount"})
            else:
                logger.error(f"Batch {batch_idx}: Expected 2 ask columns, got {new_ask_cols}")
                return (batch_idx, _empty_dataframe())

            df = df.filter(
                (pl.col("bid_price").is_null() | (pl.col("bid_price") >= 0))
                & (pl.col("ask_price").is_null() | (pl.col("ask_price") >= 0))
                & (pl.col("bid_amount").is_null() | (pl.col("bid_amount") >= 0))
                & (pl.col("ask_amount").is_null() | (pl.col("ask_amount") >= 0))
                & (
                    pl.col("bid_price").is_null()
                    | pl.col("ask_price").is_null()
                    | (pl.col("bid_price") <= pl.col("ask_price"))
                )
            )

            df = df.select(
                [
                    "exchange",
                    "symbol",
                    "timestamp",
                    "local_timestamp",
                    "type",
                    "strike_price",
                    "underlying",
                    "expiry_str",
                    "bid_price",
                    "bid_amount",
                    "ask_price",
                    "ask_amount",
                ]
            )

            filtered_count = original_rows - len(df)
            logger.info(f"Batch {batch_idx}: Parsed {len(df):,} records ({filtered_count} filtered)")

            return (batch_idx, df)

        except httpx.HTTPError as e:
            logger.error(f"HTTP error on batch {batch_idx}: {e}")
            raise Exception(f"HTTP error connecting to tardis-machine: {e}") from e


async def fetch_sampled_quotes(
    symbols: list[str],
    from_date: str,
    to_date: str,
    interval: str = DEFAULT_RESAMPLE_INTERVAL,
    tardis_machine_url: str = DEFAULT_TARDIS_MACHINE_URL,
    include_book: bool = False,
    batch_size: int = DEFAULT_BATCH_SIZE,
    max_workers: int = DEFAULT_MAX_WORKERS,
) -> pl.DataFrame:
    book_info = f", book_snapshot_1_{interval}" if include_book else ""
    logger.info(f"Fetching {len(symbols)} symbols ({from_date} to {to_date}): quote_{interval}{book_info}")

    data_types = [f"quote_{interval}"]
    if include_book:
        data_types.append(f"book_snapshot_1_{interval}")

    from_datetime = f"{from_date}T00:00:00.000Z"
    to_datetime = f"{to_date}T23:59:59.999Z"

    symbol_batches = (
        [symbols[i : i + batch_size] for i in range(0, len(symbols), batch_size)]
        if len(symbols) > batch_size
        else [symbols]
    )
    num_batches = len(symbol_batches)
    if num_batches > 1:
        logger.info(f"Batching {len(symbols)} symbols into {num_batches} batches of {batch_size}")
        logger.info(f"Using {max_workers} concurrent workers for parallel fetching")

    start_time = time.time()
    logger.info(f"Requesting {len(symbol_batches)} batch(es) with data types: {', '.join(data_types)}")

    semaphore = asyncio.Semaphore(max_workers)

    async with httpx.AsyncClient(timeout=HTTP_TIMEOUT_SECONDS) as client:
        tasks = [
            _fetch_single_batch(
                client=client,
                semaphore=semaphore,
                batch_idx=batch_idx,
                total_batches=num_batches,
                symbol_batch=symbol_batch,
                from_datetime=from_datetime,
                to_datetime=to_datetime,
                data_types=data_types,
                tardis_machine_url=tardis_machine_url,
            )
            for batch_idx, symbol_batch in enumerate(symbol_batches, 1)
        ]

        results = await asyncio.gather(*tasks, return_exceptions=True)

        all_dfs = []
        for result in results:
            if isinstance(result, Exception):
                logger.error(f"Batch failed with error: {result}")
                raise result
            # After isinstance check, result is tuple[int, pl.DataFrame]
            assert not isinstance(result, BaseException)  # Type narrowing for pyright
            _batch_idx, batch_df = result
            all_dfs.append(batch_df)

    df = pl.concat(all_dfs, how="vertical") if all_dfs else _empty_dataframe()

    total_elapsed = time.time() - start_time
    rate = len(df) / max(total_elapsed, 0.001)
    logger.info(
        f"Complete: {len(df):,} rows from {len(symbol_batches)} batches in {total_elapsed:.1f}s ({rate:.0f} rows/s)"
    )

    if len(df) == 0:
        logger.warning("No data received - check symbols exist and date range has data")

    return df


def build_symbol_list(
    assets: list[str],
    reference_date: datetime,
    min_days: Optional[int],
    max_days: Optional[int],
    option_type: str,
) -> list[str]:
    suffixes = ["C", "P"] if option_type == "both" else ["C"] if option_type == "call" else ["P"]

    expiry_dates = [reference_date + timedelta(days=d) for d in range(min_days or 0, (max_days or 0) + 1)]

    all_symbols = []
    for asset in assets:
        strikes = BTC_STRIKES if asset == "BTC" else ETH_STRIKES
        symbols = [
            f"{asset}-{d.day}{d.strftime('%b').upper()}{d.strftime('%y')}-{strike}-{suffix}"
            for d in expiry_dates
            for strike in strikes
            for suffix in suffixes
        ]
        logger.info(f"{asset}: {len(symbols):,} symbols (strikes {min(strikes)}-{max(strikes)})")
        all_symbols.extend(symbols)

    logger.info(f"Total: {len(all_symbols):,} symbols ({len(assets)} assets, {len(expiry_dates)} dates)")
    return all_symbols


def save_output(df: pl.DataFrame, output_dir: str, output_format: str, filename_suffix: str) -> str:
    ext = "parquet" if output_format == "parquet" else "csv"
    filepath = os.path.join(output_dir, f"deribit_options_{filename_suffix}.{ext}")

    if output_format == "parquet":
        df.write_parquet(filepath)
    else:
        df.write_csv(filepath)

    size_mb = os.path.getsize(filepath) / 1_000_000
    logger.info(f"Saved {output_format}: {filepath} ({size_mb:.2f} MB)")
    return filepath


async def main_async(
    from_date: str,
    to_date: str,
    assets: list[str],
    min_days: Optional[int],
    max_days: Optional[int],
    option_type: str,
    resample_interval: str,
    output_dir: str,
    output_format: str,
    tardis_machine_url: str,
    include_book: bool,
    max_workers: int,
    tardis_api_key: Optional[str] = None,
):
    tardis_process = None

    try:
        # Check if server is already running
        if not await check_tardis_machine_server(tardis_machine_url):
            if not tardis_api_key:
                logger.error(f"ERROR: tardis-machine server not accessible at {tardis_machine_url}")
                logger.error("Either:")
                logger.error("  1. Set TARDIS_API_KEY in .env file, OR")
                logger.error("  2. Start server manually: npx tardis-machine --port=8000 --api-key=YOUR_KEY")
                sys.exit(1)

            # Auto-start tardis-machine
            tardis_process = start_tardis_machine(tardis_api_key, port=8000)

            # Wait for server to be ready (up to 30 seconds)
            for _i in range(30):
                await asyncio.sleep(1)
                if await check_tardis_machine_server(tardis_machine_url):
                    break
            else:
                if tardis_process:
                    tardis_process.terminate()
                raise Exception("tardis-machine server did not become ready in 30 seconds")

        logger.info(f"Server OK: {tardis_machine_url}")

        reference_date = datetime.strptime(from_date, "%Y-%m-%d")
        logger.info("=" * 80)
        logger.info("SYMBOL GENERATION")
        logger.info("=" * 80)

        symbols = build_symbol_list(
            assets=assets,
            reference_date=reference_date,
            min_days=min_days,
            max_days=max_days,
            option_type=option_type,
        )

        if not symbols:
            logger.error("ERROR: No symbols generated - check date range parameters")
            sys.exit(1)

        logger.info("=" * 80)
        logger.info("FETCH SAMPLED DATA")
        logger.info("=" * 80)

        df = await fetch_sampled_quotes(
            symbols=symbols,
            from_date=from_date,
            to_date=to_date,
            interval=resample_interval,
            tardis_machine_url=tardis_machine_url,
            include_book=include_book,
            max_workers=max_workers,
        )

        if df.shape[0] == 0:
            logger.error("ERROR: No data received from tardis-machine")
            sys.exit(1)

        logger.info("=" * 80)
        logger.info("SAVE OUTPUT")
        logger.info("=" * 80)

        filename_suffix = f"{from_date}_{'_'.join(assets)}_{resample_interval}"
        output_path = save_output(df, output_dir, output_format, filename_suffix)

        logger.info("=" * 80)
        logger.info(f"SUCCESS: {output_path}")
        logger.info(f"{df.shape[0]:,} rows, {df['symbol'].n_unique()} unique symbols, interval {resample_interval}")
        logger.info("=" * 80)

    finally:
        # Cleanup: stop tardis-machine if we started it
        if tardis_process:
            logger.info("Stopping tardis-machine server...")
            tardis_process.terminate()
            try:
                tardis_process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                logger.warning("tardis-machine did not stop gracefully, killing...")
                tardis_process.kill()


def main():
    # Load environment variables from .env
    load_dotenv()
    tardis_api_key = os.getenv("TARDIS_API_KEY")

    parser = argparse.ArgumentParser(description="Download Deribit options data using tardis-machine")

    parser.add_argument("--from-date", required=True, help="Start date (YYYY-MM-DD)")
    parser.add_argument("--to-date", required=True, help="End date (YYYY-MM-DD)")
    parser.add_argument("--assets", required=True, help="Comma-separated assets (BTC, ETH)")
    parser.add_argument("--min-days", type=int, help="Minimum days to expiry")
    parser.add_argument("--max-days", type=int, help="Maximum days to expiry")
    parser.add_argument("--option-type", default="both", help="call, put, or both (default: both)")
    parser.add_argument(
        "--resample-interval",
        default=DEFAULT_RESAMPLE_INTERVAL,
        help=f"Sampling interval (default: {DEFAULT_RESAMPLE_INTERVAL})",
    )
    parser.add_argument("--include-book", action="store_true", help="Include orderbook snapshots")
    parser.add_argument("--output-dir", default="./datasets_deribit_options", help="Output directory")
    parser.add_argument("--output-format", default="parquet", choices=["csv", "parquet"], help="Output format")
    parser.add_argument("--tardis-machine-url", default=DEFAULT_TARDIS_MACHINE_URL, help="Tardis server URL")
    parser.add_argument(
        "--max-workers",
        type=int,
        default=DEFAULT_MAX_WORKERS,
        help=f"Number of concurrent workers (default: {DEFAULT_MAX_WORKERS})",
    )
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable debug logging")

    args = parser.parse_args()

    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(level=log_level, format="%(message)s", handlers=[logging.StreamHandler(sys.stdout)])

    try:
        from_date = datetime.strptime(args.from_date, "%Y-%m-%d")
        to_date = datetime.strptime(args.to_date, "%Y-%m-%d")
        if from_date > to_date:
            raise ValueError("from_date must be <= to_date")

        assets = [a.strip().upper() for a in args.assets.split(",")]
        if invalid := set(assets) - {"BTC", "ETH"}:
            raise ValueError(f"Invalid assets: {invalid}")

        if args.min_days is not None and args.min_days < 0:
            raise ValueError("min_days must be >= 0")
        if args.max_days is not None and args.max_days < 0:
            raise ValueError("max_days must be >= 0")
        if args.min_days is not None and args.max_days is not None and args.min_days > args.max_days:
            raise ValueError("min_days must be <= max_days")

        option_type = args.option_type.lower()
        if option_type not in {"call", "put", "both"}:
            raise ValueError(f"Invalid option_type: {option_type}")

        if args.resample_interval[-1].lower() not in {"s", "m", "h", "d"}:
            raise ValueError(f"Invalid interval suffix: {args.resample_interval[-1]}")
        if int(args.resample_interval[:-1]) <= 0:
            raise ValueError("Interval value must be > 0")

        if args.max_workers < 1:
            raise ValueError("max_workers must be >= 1")
        if args.max_workers > 20:
            raise ValueError("max_workers must be <= 20 to avoid overwhelming the server")

    except ValueError as e:
        logger.error(f"ERROR: {e}")
        sys.exit(1)

    os.makedirs(args.output_dir, exist_ok=True)

    logger.info("=" * 80)
    logger.info("DERIBIT OPTIONS DOWNLOAD")
    logger.info("=" * 80)
    days_range = f"{args.min_days or 0}-{args.max_days or 'any'}"
    book_info = f", book={args.include_book}" if args.include_book else ""
    logger.info(f"{', '.join(assets)} {option_type} | {args.from_date} to {args.to_date} | Days: {days_range}")
    logger.info(
        f"Interval: {args.resample_interval} | Format: {args.output_format} | Dir: {args.output_dir}{book_info}"
    )

    try:
        asyncio.run(
            main_async(
                from_date=args.from_date,
                to_date=args.to_date,
                assets=assets,
                min_days=args.min_days,
                max_days=args.max_days,
                option_type=option_type,
                resample_interval=args.resample_interval,
                output_dir=args.output_dir,
                output_format=args.output_format,
                tardis_machine_url=args.tardis_machine_url,
                include_book=args.include_book,
                max_workers=args.max_workers,
                tardis_api_key=tardis_api_key,
            )
        )

    except Exception as e:
        logger.error(f"ERROR: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
