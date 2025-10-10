#!/usr/bin/env python3
"""
Tardis Machine Data Downloader for Deribit Options

VALID SAMPLING INTERVALS (--resample-interval):
  Format: <positive_integer><suffix>
  Valid suffixes:
    s = seconds   (e.g., 1s, 5s, 30s)
    m = minutes   (e.g., 1m, 5m, 15m)
    h = hours     (e.g., 1h, 4h, 24h)
    d = days      (e.g., 1d, 7d)

  Examples: 1s, 5s, 30s, 1m, 5m, 15m, 1h, 4h, 1d
"""

import argparse
import asyncio
import sys
import os
import time
import msgspec.json
import logging
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Tuple, TypedDict
import httpx
import polars as pl


DEFAULT_BATCH_SIZE = 50
DEFAULT_MAX_WORKERS = 5
DEFAULT_TARDIS_MACHINE_URL = "http://localhost:8000"
HTTP_TIMEOUT_SECONDS = 1800.0
SERVER_CHECK_TIMEOUT = 10.0
DEFAULT_RESAMPLE_INTERVAL = "1m"
DEFAULT_BOOK_LEVELS = 3
MICROSECONDS_PER_SECOND = 1_000_000
MEGABYTES_DIVISOR = 1_000_000

BTC_STRIKES = range(40000, 130001, 1000)
ETH_STRIKES = range(1500, 5001, 100)


class QuoteData(TypedDict):
    exchange: str
    symbol: str
    timestamp: int
    local_timestamp: int
    type: str
    strike_price: float
    underlying: str
    expiry_str: str
    bid_price: Optional[float]
    bid_amount: Optional[float]
    ask_price: Optional[float]
    ask_amount: Optional[float]


logger = logging.getLogger(__name__)


def _parse_timestamp(timestamp_str: str) -> int:
    try:
        dt = datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
        return int(dt.timestamp() * MICROSECONDS_PER_SECOND)
    except (ValueError, AttributeError):
        return 0


def _validate_quote_data(quote: QuoteData) -> bool:
    for key in ['bid_price', 'ask_price', 'bid_amount', 'ask_amount']:
        if quote[key] is not None and quote[key] < 0:
            return False
    bid, ask = quote['bid_price'], quote['ask_price']
    return not (bid is not None and ask is not None and bid > ask)


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
    symbol_batch: List[str],
    from_datetime: str,
    to_datetime: str,
    data_types: List[str],
    tardis_machine_url: str,
) -> Tuple[int, List[QuoteData]]:
    """Fetch a single batch of symbols with semaphore-based concurrency control."""
    async with semaphore:
        logger.info(f"Batch {batch_idx}/{total_batches}: {len(symbol_batch)} symbols...")

        options = {
            "exchange": "deribit",
            "from": from_datetime,
            "to": to_datetime,
            "symbols": symbol_batch,
            "dataTypes": data_types
        }

        try:
            batch_start = time.time()

            response = await client.get(
                f"{tardis_machine_url}/replay-normalized",
                params={"options": msgspec.json.encode(options).decode('utf-8')}
            )

            if response.status_code != 200:
                raise Exception(
                    f"tardis-machine error ({response.status_code}): {response.text[:500]}"
                )

            lines = response.text.strip().split('\n') if response.text.strip() else []
            batch_elapsed = time.time() - batch_start
            size_mb = len(response.content) / MEGABYTES_DIVISOR

            if not lines:
                logger.warning(f"Batch {batch_idx}: No data received")
                return (batch_idx, [])

            logger.info(f"Batch {batch_idx}: Received {len(lines):,} messages ({size_mb:.2f} MB) in {batch_elapsed:.1f}s")

            batch_rows = []
            parse_errors = validation_failures = 0

            for line in lines:
                try:
                    msg = msgspec.json.decode(line)
                    if 'quote' in msg.get('type', '') or 'book_snapshot' in msg.get('type', ''):
                        quote_row = _parse_quote_message(msg)
                        if quote_row and _validate_quote_data(quote_row):
                            batch_rows.append(quote_row)
                        elif quote_row:
                            validation_failures += 1
                except msgspec.DecodeError:
                    parse_errors += 1
                except Exception:
                    parse_errors += 1

            errors_msg = f" ({parse_errors} parse errors, {validation_failures} validation failures)" if parse_errors or validation_failures else ""
            logger.info(f"Batch {batch_idx}: Parsed {len(batch_rows):,} records{errors_msg}")

            return (batch_idx, batch_rows)

        except httpx.HTTPError as e:
            logger.error(f"HTTP error on batch {batch_idx}: {e}")
            raise Exception(f"HTTP error connecting to tardis-machine: {e}") from e


async def fetch_sampled_quotes(
    symbols: List[str],
    from_date: str,
    to_date: str,
    interval: str = DEFAULT_RESAMPLE_INTERVAL,
    tardis_machine_url: str = DEFAULT_TARDIS_MACHINE_URL,
    include_book: bool = False,
    book_levels: int = DEFAULT_BOOK_LEVELS,
    batch_size: int = DEFAULT_BATCH_SIZE,
    max_workers: int = DEFAULT_MAX_WORKERS,
) -> pl.DataFrame:

    book_info = f", book_snapshot_{book_levels}_{interval}" if include_book else ""
    logger.info(f"Fetching {len(symbols)} symbols ({from_date} to {to_date}): quote_{interval}{book_info}")

    data_types = [f"quote_{interval}"]
    if include_book:
        data_types.append(f"book_snapshot_{book_levels}_{interval}")

    from_datetime = f"{from_date}T00:00:00.000Z"
    to_datetime = f"{to_date}T23:59:59.999Z"

    symbol_batches = [symbols[i:i + batch_size] for i in range(0, len(symbols), batch_size)] if len(symbols) > batch_size else [symbols]
    num_batches = len(symbol_batches)
    if num_batches > 1:
        logger.info(f"Batching {len(symbols)} symbols into {num_batches} batches of {batch_size}")
        logger.info(f"Using {max_workers} concurrent workers for parallel fetching")

    start_time = time.time()
    logger.info(f"Requesting {len(symbol_batches)} batch(es) with data types: {', '.join(data_types)}")

    # Create semaphore to limit concurrent requests
    semaphore = asyncio.Semaphore(max_workers)

    async with httpx.AsyncClient(timeout=HTTP_TIMEOUT_SECONDS) as client:
        # Create tasks for all batches
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

        # Execute all batches concurrently with max_workers limit
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Process results
        all_rows = []
        for result in results:
            if isinstance(result, Exception):
                logger.error(f"Batch failed with error: {result}")
                raise result
            batch_idx, batch_rows = result
            all_rows.extend(batch_rows)

    total_elapsed = time.time() - start_time
    rate = len(all_rows) / max(total_elapsed, 0.001)
    logger.info(f"Complete: {len(all_rows):,} rows from {len(symbol_batches)} batches in {total_elapsed:.1f}s ({rate:.0f} rows/s)")

    if not all_rows:
        logger.warning("No data received - check symbols exist and date range has data")
        schema = {
            "exchange": pl.Utf8, "symbol": pl.Utf8, "timestamp": pl.Int64, "local_timestamp": pl.Int64,
            "type": pl.Utf8, "strike_price": pl.Float64, "underlying": pl.Utf8, "expiry_str": pl.Utf8,
            "bid_price": pl.Float64, "bid_amount": pl.Float64, "ask_price": pl.Float64, "ask_amount": pl.Float64,
        }
        return pl.DataFrame(schema=schema)

    return pl.DataFrame(all_rows)


def _parse_quote_message(msg: dict) -> Optional[QuoteData]:
    try:
        parts = msg.get('symbol', '').split('-')
        if len(parts) != 4:
            return None

        bids, asks = msg.get('bids', []), msg.get('asks', [])
        def extract(book): return (book[0]['price'], book[0]['amount']) if book and isinstance(book[0], dict) else (None, None)
        bid_price, bid_amount = extract(bids)
        ask_price, ask_amount = extract(asks)

        return QuoteData(
            exchange="deribit", symbol=msg.get('symbol', ''),
            timestamp=_parse_timestamp(msg.get('timestamp', '')),
            local_timestamp=_parse_timestamp(msg.get('timestamp', '')),
            type="call" if parts[3] == "C" else "put",
            strike_price=float(parts[2]), underlying=parts[0], expiry_str=parts[1],
            bid_price=bid_price, bid_amount=bid_amount,
            ask_price=ask_price, ask_amount=ask_amount,
        )
    except (ValueError, KeyError, IndexError, TypeError):
        return None


def build_symbol_list(
    assets: List[str], reference_date: datetime,
    min_days: Optional[int], max_days: Optional[int], option_type: str,
) -> List[str]:
    suffixes = (['C', 'P'] if option_type == 'both' else
                ['C'] if option_type == 'call' else ['P'])

    expiry_dates = [reference_date + timedelta(days=d)
                    for d in range(min_days or 0, (max_days or 0) + 1)]

    all_symbols = []
    for asset in assets:
        strikes = BTC_STRIKES if asset == 'BTC' else ETH_STRIKES
        symbols = [f"{asset}-{d.day}{d.strftime('%b').upper()}{d.strftime('%y')}-{strike}-{suffix}"
                   for d in expiry_dates for strike in strikes for suffix in suffixes]
        logger.info(f"{asset}: {len(symbols):,} symbols (strikes {min(strikes)}-{max(strikes)})")
        all_symbols.extend(symbols)

    logger.info(f"Total: {len(all_symbols):,} symbols ({len(assets)} assets, {len(expiry_dates)} dates)")
    return all_symbols


def save_output(df: pl.DataFrame, output_dir: str, output_format: str, filename_suffix: str) -> str:
    ext = 'parquet' if output_format == 'parquet' else 'csv'
    filepath = os.path.join(output_dir, f"deribit_options_{filename_suffix}.{ext}")

    if output_format == 'parquet':
        df.write_parquet(filepath)
    else:
        df.write_csv(filepath)

    size_mb = os.path.getsize(filepath) / MEGABYTES_DIVISOR
    logger.info(f"Saved {output_format}: {filepath} ({size_mb:.2f} MB)")
    return filepath


async def main_async(
    from_date: str,
    to_date: str,
    assets: List[str],
    min_days: Optional[int],
    max_days: Optional[int],
    option_type: str,
    resample_interval: str,
    output_dir: str,
    output_format: str,
    tardis_machine_url: str,
    include_book: bool,
    book_levels: int,
    max_workers: int,
):

    if not await check_tardis_machine_server(tardis_machine_url):
        logger.error(f"ERROR: tardis-machine server not accessible at {tardis_machine_url}")
        logger.error("Start server: npx tardis-machine --port=8000")
        sys.exit(1)

    logger.info(f"Server OK: {tardis_machine_url}")
    reference_date = datetime.strptime(from_date, '%Y-%m-%d')
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
        symbols=symbols, from_date=from_date, to_date=to_date,
        interval=resample_interval, tardis_machine_url=tardis_machine_url,
        include_book=include_book, book_levels=book_levels,
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


def main():
    parser = argparse.ArgumentParser(description="Download Deribit options data using tardis-machine")

    parser.add_argument('--from-date', required=True, help='Start date (YYYY-MM-DD)')
    parser.add_argument('--to-date', required=True, help='End date (YYYY-MM-DD)')
    parser.add_argument('--assets', required=True, help='Comma-separated assets (BTC, ETH)')
    parser.add_argument('--min-days', type=int, help='Minimum days to expiry')
    parser.add_argument('--max-days', type=int, help='Maximum days to expiry')
    parser.add_argument('--option-type', default='both', help='call, put, or both (default: both)')
    parser.add_argument('--resample-interval', default='5s', help='Sampling interval (default: 5s)')
    parser.add_argument('--include-book', action='store_true', help='Include orderbook snapshots')
    parser.add_argument('--book-levels', type=int, default=25, help='Orderbook levels (default: 25)')
    parser.add_argument('--output-dir', default='./datasets_deribit_options', help='Output directory')
    parser.add_argument('--output-format', default='parquet', choices=['csv', 'parquet'], help='Output format')
    parser.add_argument('--tardis-machine-url', default=DEFAULT_TARDIS_MACHINE_URL, help='Tardis server URL')
    parser.add_argument('--max-workers', type=int, default=DEFAULT_MAX_WORKERS, help=f'Number of concurrent workers (default: {DEFAULT_MAX_WORKERS})')
    parser.add_argument('--verbose', '-v', action='store_true', help='Enable debug logging')

    args = parser.parse_args()

    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format='%(message)s',
        handlers=[logging.StreamHandler(sys.stdout)]
    )

    try:
        from_date = datetime.strptime(args.from_date, '%Y-%m-%d')
        to_date = datetime.strptime(args.to_date, '%Y-%m-%d')
        if from_date > to_date:
            raise ValueError(f"from_date must be <= to_date")

        assets = [a.strip().upper() for a in args.assets.split(',')]
        if invalid := set(assets) - {'BTC', 'ETH'}:
            raise ValueError(f"Invalid assets: {invalid}")

        if args.min_days is not None and args.min_days < 0:
            raise ValueError(f"min_days must be >= 0")
        if args.max_days is not None and args.max_days < 0:
            raise ValueError(f"max_days must be >= 0")
        if args.min_days is not None and args.max_days is not None and args.min_days > args.max_days:
            raise ValueError(f"min_days must be <= max_days")

        option_type = args.option_type.lower()
        if option_type not in {'call', 'put', 'both'}:
            raise ValueError(f"Invalid option_type: {option_type}")

        if args.resample_interval[-1].lower() not in {'s', 'm', 'h', 'd'}:
            raise ValueError(f"Invalid interval suffix: {args.resample_interval[-1]}")
        if int(args.resample_interval[:-1]) <= 0:
            raise ValueError(f"Interval value must be > 0")

        if args.max_workers < 1:
            raise ValueError(f"max_workers must be >= 1")
        if args.max_workers > 20:
            raise ValueError(f"max_workers must be <= 20 to avoid overwhelming the server")

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
    logger.info(f"Interval: {args.resample_interval} | Format: {args.output_format} | Dir: {args.output_dir}{book_info}")

    try:
        asyncio.run(main_async(
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
            book_levels=args.book_levels,
            max_workers=args.max_workers,
        ))

    except Exception as e:
        logger.error(f"ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
