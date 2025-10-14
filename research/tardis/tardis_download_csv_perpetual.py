#!/usr/bin/env python3

import argparse
import logging
import os
import sys
from datetime import datetime

from dotenv import load_dotenv
from tardis_dev import datasets

logger = logging.getLogger(__name__)


def main():
    """Download Deribit perpetual futures CSV data from Tardis datasets API."""
    # Load environment variables from .env
    load_dotenv()
    tardis_api_key = os.getenv("TARDIS_API_KEY")

    parser = argparse.ArgumentParser(description="Download Deribit perpetual futures CSV data from Tardis datasets API")

    parser.add_argument("--from-date", required=True, help="Start date (YYYY-MM-DD)")
    parser.add_argument("--to-date", required=True, help="End date (YYYY-MM-DD)")
    parser.add_argument("--symbol", default="BTC-PERPETUAL", help="Perpetual symbol (default: BTC-PERPETUAL)")
    parser.add_argument(
        "--data-types",
        default="trades",
        help="Comma-separated data types (trades, incremental_book_L2, book_snapshot_25, derivative_ticker, quotes)",
    )
    parser.add_argument("--output-dir", default="./datasets_deribit_perpetual", help="Output directory")
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable debug logging")
    parser.add_argument("--concurrent", type=int, default=10, help="Number of concurrent downloads (default: 10)")

    args = parser.parse_args()

    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(level=log_level, format="%(message)s", handlers=[logging.StreamHandler(sys.stdout)])

    # Validate dates
    try:
        from_date = datetime.strptime(args.from_date, "%Y-%m-%d")
        to_date = datetime.strptime(args.to_date, "%Y-%m-%d")
        if from_date > to_date:
            raise ValueError("from_date must be <= to_date")
    except ValueError as e:
        logger.error(f"ERROR: {e}")
        sys.exit(1)

    # Parse data types
    data_types = [dt.strip() for dt in args.data_types.split(",")]
    valid_types = [
        "trades",
        "incremental_book_L2",
        "book_snapshot_25",
        "book_snapshot_5",
        "derivative_ticker",
        "quotes",
        "liquidations",
    ]
    for dt in data_types:
        if dt not in valid_types:
            logger.error(f"ERROR: Invalid data type '{dt}'. Valid types: {', '.join(valid_types)}")
            sys.exit(1)

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    logger.info("=" * 80)
    logger.info("TARDIS CSV DOWNLOAD - DERIBIT PERPETUAL")
    logger.info("=" * 80)
    logger.info(f"Symbol: {args.symbol}")
    logger.info(f"Date range: {args.from_date} to {args.to_date}")
    logger.info(f"Data types: {', '.join(data_types)}")
    logger.info(f"Output dir: {args.output_dir}")
    logger.info(f"Concurrent downloads: {args.concurrent}")
    logger.info("=" * 80)

    if not tardis_api_key:
        logger.warning(
            "WARNING: TARDIS_API_KEY not found in .env - will only download publicly available data "
            "(first day of each month)"
        )

    try:
        # Download datasets
        datasets.download(
            exchange="deribit",
            data_types=data_types,
            from_date=args.from_date,
            to_date=args.to_date,
            symbols=[args.symbol],
            api_key=tardis_api_key if tardis_api_key else "",
            download_dir=args.output_dir,
            concurrency=args.concurrent,
        )

        logger.info("=" * 80)
        logger.info(f"SUCCESS: Downloaded to {args.output_dir}")
        logger.info("=" * 80)

    except Exception as e:
        logger.error(f"ERROR: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
