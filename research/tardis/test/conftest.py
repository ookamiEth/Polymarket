"""
Pytest configuration and fixtures for Deribit options download tests.

Provides sample data that mimics the structure of Tardis.dev options_chain CSV format.
"""

import pytest
import polars as pl
from datetime import datetime, timedelta
import tempfile
import os


@pytest.fixture
def sample_options_data():
    """
    Create sample options_chain data that mimics Deribit format.

    Schema matches Tardis.dev options_chain CSV:
    - exchange, symbol, timestamp, local_timestamp
    - type (call/put), strike_price, expiration
    - open_interest, last_price, bid_price, bid_amount, bid_iv
    - ask_price, ask_amount, ask_iv, mark_price, mark_iv
    - underlying_index, underlying_price
    - delta, gamma, vega, theta, rho
    """
    # Base timestamp: Oct 1, 2025 00:00:00 UTC
    base_ts = int(datetime(2025, 10, 1).timestamp() * 1_000_000)

    # Create sample data with multiple strikes, expiries, and option types
    # Expiries: 1 week (Oct 8), 2 weeks (Oct 15), 1 month (Nov 1)
    expiry_dates = [
        datetime(2025, 10, 8),   # 7 days out
        datetime(2025, 10, 15),  # 14 days out
        datetime(2025, 11, 1),   # 31 days out
    ]

    # Strikes around $60k for BTC, $2.5k for ETH
    btc_strikes = [58000, 60000, 62000, 64000]
    eth_strikes = [2400, 2500, 2600, 2700]

    data = []

    for underlying in ['BTC', 'ETH']:
        strikes = btc_strikes if underlying == 'BTC' else eth_strikes
        underlying_price = 60000 if underlying == 'BTC' else 2500

        for expiry_date in expiry_dates:
            expiry_str = expiry_date.strftime('%d%b%y').upper()
            expiry_ts = int(expiry_date.timestamp() * 1_000_000)

            for strike in strikes:
                for option_type in ['call', 'put']:
                    # Generate symbol: BTC-08OCT25-60000-C
                    symbol = f"{underlying}-{expiry_str}-{strike}-{'C' if option_type == 'call' else 'P'}"

                    # Calculate approximate delta based on moneyness
                    moneyness = strike / underlying_price
                    if option_type == 'call':
                        delta = max(0.05, min(0.95, 1.5 - moneyness))
                    else:
                        delta = -max(0.05, min(0.95, moneyness - 0.5))

                    # Add some tick-level data points (simulate updates)
                    for i in range(10):
                        timestamp = base_ts + i * 1_000_000  # 1 second apart

                        data.append({
                            'exchange': 'deribit',
                            'symbol': symbol,
                            'timestamp': timestamp,
                            'local_timestamp': timestamp + 50000,  # 50ms delay
                            'type': option_type,
                            'strike_price': float(strike),
                            'expiration': expiry_ts,
                            'open_interest': 100.5 + i,
                            'last_price': 0.05 + i * 0.001,
                            'bid_price': 0.048 + i * 0.001,
                            'bid_amount': 10.0,
                            'bid_iv': 65.5 + i * 0.1,
                            'ask_price': 0.052 + i * 0.001,
                            'ask_amount': 15.0,
                            'ask_iv': 66.5 + i * 0.1,
                            'mark_price': 0.05 + i * 0.001,
                            'mark_iv': 66.0 + i * 0.1,
                            'underlying_index': f'SYN.{underlying}-{expiry_str}',
                            'underlying_price': underlying_price + i * 10,
                            'delta': delta + i * 0.001,
                            'gamma': 0.0001 + i * 0.00001,
                            'vega': 2.0 + i * 0.1,
                            'theta': -50.0 - i * 0.5,
                            'rho': 0.25 + i * 0.01,
                        })

    return pl.DataFrame(data)


@pytest.fixture
def sample_csv_path(sample_options_data, tmp_path):
    """
    Write sample data to a temporary CSV file.
    """
    csv_path = tmp_path / "test_options.csv"
    sample_options_data.write_csv(csv_path)
    return str(csv_path)


@pytest.fixture
def sample_parquet_path(sample_options_data, tmp_path):
    """
    Write sample data to a temporary Parquet file.
    """
    parquet_path = tmp_path / "test_options.parquet"
    sample_options_data.write_parquet(parquet_path)
    return str(parquet_path)


@pytest.fixture
def temp_output_dir(tmp_path):
    """
    Create a temporary directory for output files.
    """
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    return str(output_dir)


@pytest.fixture
def base_date():
    """
    Base date for testing (Oct 1, 2025).
    """
    return datetime(2025, 10, 1)


@pytest.fixture
def valid_cli_args():
    """
    Example of valid CLI arguments.
    """
    return {
        'from_date': '2025-10-01',
        'to_date': '2025-10-01',
        'assets': 'BTC,ETH',
        'min_days': 7,
        'max_days': 30,
        'option_type': 'both',
        'strike_min': None,
        'strike_max': None,
        'resample_interval': None,
        'api_key': None,
        'output_dir': './datasets',
        'output_format': 'parquet',
    }
