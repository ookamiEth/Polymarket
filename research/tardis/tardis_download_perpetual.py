#!/usr/bin/env python3

import argparse
import asyncio
import io
import logging
import os
import subprocess
import sys
import time
from typing import Optional

import httpx
import msgspec.json
import polars as pl
from dotenv import load_dotenv

DEFAULT_TARDIS_MACHINE_URL = "http://localhost:8000"
HTTP_TIMEOUT_SECONDS = 1800.0
SERVER_CHECK_TIMEOUT = 10.0
DEFAULT_RESAMPLE_INTERVAL = "1s"

logger = logging.getLogger(__name__)


def _empty_dataframe() -> pl.DataFrame:
    schema = {
        "exchange": pl.Utf8,
        "symbol": pl.Utf8,
        "timestamp": pl.Int64,
        "local_timestamp": pl.Int64,
        "price": pl.Float64,
        "amount": pl.Float64,
        "side": pl.Utf8,
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


async def fetch_perpetual_data(
    symbol: str,
    from_date: str,
    to_date: str,
    interval: str = DEFAULT_RESAMPLE_INTERVAL,
    tardis_machine_url: str = DEFAULT_TARDIS_MACHINE_URL,
    data_types: Optional[list[str]] = None,
) -> pl.DataFrame:
    """Fetch perpetual futures data from Tardis."""
    if data_types is None:
        data_types = [f"trade_{interval}"]

    logger.info(f"Fetching {symbol} data ({from_date} to {to_date}): {', '.join(data_types)}")

    from_datetime = f"{from_date}T00:00:00.000Z"
    to_datetime = f"{to_date}T23:59:59.999Z"

    options = {
        "exchange": "deribit",
        "from": from_datetime,
        "to": to_datetime,
        "symbols": [symbol],
        "dataTypes": data_types,
    }

    start_time = time.time()

    try:
        async with httpx.AsyncClient(timeout=HTTP_TIMEOUT_SECONDS) as client:
            logger.info("Requesting data from tardis-machine...")

            response = await client.get(
                f"{tardis_machine_url}/replay-normalized",
                params={"options": msgspec.json.encode(options).decode("utf-8")},
            )

            if response.status_code != 200:
                raise Exception(f"tardis-machine error ({response.status_code}): {response.text[:500]}")

            elapsed = time.time() - start_time
            size_mb = len(response.content) / 1_000_000

            if not response.text.strip():
                logger.warning("No data received")
                return _empty_dataframe()

            try:
                df = pl.read_ndjson(io.StringIO(response.text))
            except Exception as e:
                logger.error(f"JSON parse error: {e}")
                return _empty_dataframe()

            original_rows = len(df)
            logger.info(f"Received {original_rows:,} messages ({size_mb:.2f} MB) in {elapsed:.1f}s")

            if original_rows == 0:
                return _empty_dataframe()

            # Process based on data type
            if "trade" in data_types[0]:
                # Trade data processing
                df = df.with_columns(
                    [
                        pl.lit("deribit").alias("exchange"),
                        pl.col("timestamp")
                        .str.strptime(pl.Datetime, format="%Y-%m-%dT%H:%M:%S%.fZ", strict=False)
                        .dt.epoch("us")
                        .fill_null(0)
                        .alias("timestamp"),
                    ]
                )

                df = df.with_columns([pl.col("timestamp").alias("local_timestamp")])

                df = df.select(
                    [
                        "exchange",
                        "symbol",
                        "timestamp",
                        "local_timestamp",
                        "price",
                        "amount",
                        "side",
                    ]
                )

            elif "book_snapshot" in data_types[0]:
                # Orderbook snapshot processing
                df = df.with_columns(
                    [
                        pl.lit("deribit").alias("exchange"),
                        pl.col("timestamp")
                        .str.strptime(pl.Datetime, format="%Y-%m-%dT%H:%M:%S%.fZ", strict=False)
                        .dt.epoch("us")
                        .fill_null(0)
                        .alias("timestamp"),
                    ]
                )

                df = df.with_columns([pl.col("timestamp").alias("local_timestamp")])

                # Extract best bid/ask
                df = df.with_columns(
                    [
                        pl.col("bids").list.first().struct.field("price").alias("bid_price"),
                        pl.col("bids").list.first().struct.field("amount").alias("bid_amount"),
                        pl.col("asks").list.first().struct.field("price").alias("ask_price"),
                        pl.col("asks").list.first().struct.field("amount").alias("ask_amount"),
                    ]
                )

                df = df.select(
                    [
                        "exchange",
                        "symbol",
                        "timestamp",
                        "local_timestamp",
                        "bid_price",
                        "bid_amount",
                        "ask_price",
                        "ask_amount",
                    ]
                )

            logger.info(f"Parsed {len(df):,} records")

            return df

    except httpx.HTTPError as e:
        logger.error(f"HTTP error: {e}")
        raise Exception(f"HTTP error connecting to tardis-machine: {e}") from e


def save_output(df: pl.DataFrame, output_dir: str, output_format: str, filename_suffix: str) -> str:
    ext = "parquet" if output_format == "parquet" else "csv"
    filepath = os.path.join(output_dir, f"deribit_perpetual_{filename_suffix}.{ext}")

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
    symbol: str,
    resample_interval: str,
    data_type: str,
    output_dir: str,
    output_format: str,
    tardis_machine_url: str,
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

        logger.info("=" * 80)
        logger.info("FETCH PERPETUAL DATA")
        logger.info("=" * 80)

        # Determine data types based on data_type parameter
        # Note: Tardis normalizers use specific data type names
        if data_type == "trades":
            data_types = ["trade"]
        elif data_type == "book":
            data_types = [f"book_snapshot_5_{resample_interval}"]
        else:  # both
            data_types = ["trade", f"book_snapshot_5_{resample_interval}"]

        df = await fetch_perpetual_data(
            symbol=symbol,
            from_date=from_date,
            to_date=to_date,
            interval=resample_interval,
            tardis_machine_url=tardis_machine_url,
            data_types=data_types,
        )

        if df.shape[0] == 0:
            logger.error("ERROR: No data received from tardis-machine")
            sys.exit(1)

        logger.info("=" * 80)
        logger.info("SAVE OUTPUT")
        logger.info("=" * 80)

        filename_suffix = f"{from_date}_{to_date}_{symbol.replace('-', '_')}_{resample_interval}"
        output_path = save_output(df, output_dir, output_format, filename_suffix)

        logger.info("=" * 80)
        logger.info(f"SUCCESS: {output_path}")
        logger.info(f"{df.shape[0]:,} rows, interval {resample_interval}")
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

    parser = argparse.ArgumentParser(description="Download Deribit perpetual futures data using tardis-machine")

    parser.add_argument("--from-date", required=True, help="Start date (YYYY-MM-DD)")
    parser.add_argument("--to-date", required=True, help="End date (YYYY-MM-DD)")
    parser.add_argument("--symbol", default="BTC-PERPETUAL", help="Perpetual symbol (default: BTC-PERPETUAL)")
    parser.add_argument(
        "--resample-interval",
        default=DEFAULT_RESAMPLE_INTERVAL,
        help=f"Sampling interval (default: {DEFAULT_RESAMPLE_INTERVAL})",
    )
    parser.add_argument(
        "--data-type",
        default="trades",
        choices=["trades", "book", "both"],
        help="Data type to fetch (default: trades)",
    )
    parser.add_argument("--output-dir", default="./datasets_deribit_perpetual", help="Output directory")
    parser.add_argument("--output-format", default="parquet", choices=["csv", "parquet"], help="Output format")
    parser.add_argument("--tardis-machine-url", default=DEFAULT_TARDIS_MACHINE_URL, help="Tardis server URL")
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable debug logging")

    args = parser.parse_args()

    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(level=log_level, format="%(message)s", handlers=[logging.StreamHandler(sys.stdout)])

    try:
        from datetime import datetime

        from_date = datetime.strptime(args.from_date, "%Y-%m-%d")
        to_date = datetime.strptime(args.to_date, "%Y-%m-%d")
        if from_date > to_date:
            raise ValueError("from_date must be <= to_date")

        if args.resample_interval[-1].lower() not in {"s", "m", "h", "d"}:
            raise ValueError(f"Invalid interval suffix: {args.resample_interval[-1]}")
        if int(args.resample_interval[:-1]) <= 0:
            raise ValueError("Interval value must be > 0")

    except ValueError as e:
        logger.error(f"ERROR: {e}")
        sys.exit(1)

    os.makedirs(args.output_dir, exist_ok=True)

    logger.info("=" * 80)
    logger.info("DERIBIT PERPETUAL DOWNLOAD")
    logger.info("=" * 80)
    logger.info(f"{args.symbol} | {args.from_date} to {args.to_date}")
    logger.info(f"Data: {args.data_type} | Interval: {args.resample_interval} | Format: {args.output_format}")

    try:
        asyncio.run(
            main_async(
                from_date=args.from_date,
                to_date=args.to_date,
                symbol=args.symbol,
                resample_interval=args.resample_interval,
                data_type=args.data_type,
                output_dir=args.output_dir,
                output_format=args.output_format,
                tardis_machine_url=args.tardis_machine_url,
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
