"""
Tests for command-line argument parsing and validation.

Tests all CLI arguments for the Deribit options download script:
- Date validation (from_date, to_date)
- Asset selection (BTC, ETH, both)
- Days to expiry filtering (min_days, max_days)
- Option type filtering (call, put, both)
- Strike price filtering (strike_min, strike_max)
- Resampling interval validation
- API key handling
- Output directory and format
"""

import pytest
import argparse
from datetime import datetime, timedelta
import os


# We'll test the argument parser that will be in the main script
# For now, we'll define what the parser should do


class CLIArgumentParser:
    """
    Mock CLI argument parser for testing.
    This will be implemented in the actual script.
    """

    @staticmethod
    def parse_date(date_str):
        """Parse date string to datetime object."""
        try:
            return datetime.strptime(date_str, '%Y-%m-%d')
        except ValueError as e:
            raise ValueError(f"Invalid date format: {date_str}. Expected YYYY-MM-DD") from e

    @staticmethod
    def validate_date_range(from_date, to_date):
        """Validate that from_date <= to_date."""
        if from_date > to_date:
            raise ValueError(f"from_date ({from_date}) must be <= to_date ({to_date})")
        return True

    @staticmethod
    def parse_assets(assets_str):
        """Parse comma-separated assets string."""
        if not assets_str:
            raise ValueError("Assets cannot be empty")

        assets = [a.strip().upper() for a in assets_str.split(',')]
        valid_assets = {'BTC', 'ETH'}

        invalid = set(assets) - valid_assets
        if invalid:
            raise ValueError(f"Invalid assets: {invalid}. Valid options: {valid_assets}")

        return assets

    @staticmethod
    def validate_days_range(min_days, max_days):
        """Validate days to expiry range."""
        if min_days is not None and min_days < 0:
            raise ValueError(f"min_days must be >= 0, got {min_days}")

        if max_days is not None and max_days < 0:
            raise ValueError(f"max_days must be >= 0, got {max_days}")

        if min_days is not None and max_days is not None and min_days > max_days:
            raise ValueError(f"min_days ({min_days}) must be <= max_days ({max_days})")

        return True

    @staticmethod
    def parse_option_type(option_type_str):
        """Parse and validate option type."""
        if not option_type_str:
            return 'both'

        option_type = option_type_str.lower()
        valid_types = {'call', 'put', 'both'}

        if option_type not in valid_types:
            raise ValueError(f"Invalid option_type: {option_type}. Valid options: {valid_types}")

        return option_type

    @staticmethod
    def validate_strike_range(strike_min, strike_max):
        """Validate strike price range."""
        if strike_min is not None and strike_min < 0:
            raise ValueError(f"strike_min must be >= 0, got {strike_min}")

        if strike_max is not None and strike_max < 0:
            raise ValueError(f"strike_max must be >= 0, got {strike_max}")

        if strike_min is not None and strike_max is not None and strike_min > strike_max:
            raise ValueError(f"strike_min ({strike_min}) must be <= strike_max ({strike_max})")

        return True

    @staticmethod
    def parse_resample_interval(interval_str):
        """Parse and validate resample interval."""
        if not interval_str:
            return None

        # Valid formats: 1s, 5s, 1m, 5m, 15m, 1h, etc.
        valid_suffixes = {'s', 'm', 'h', 'd'}

        if len(interval_str) < 2:
            raise ValueError(f"Invalid interval format: {interval_str}")

        suffix = interval_str[-1].lower()
        if suffix not in valid_suffixes:
            raise ValueError(f"Invalid interval suffix: {suffix}. Valid: {valid_suffixes}")

        try:
            value = int(interval_str[:-1])
            if value <= 0:
                raise ValueError(f"Interval value must be > 0, got {value}")
        except ValueError as e:
            raise ValueError(f"Invalid interval value: {interval_str}") from e

        return interval_str

    @staticmethod
    def validate_output_format(format_str):
        """Validate output format."""
        if not format_str:
            return 'parquet'

        format_str = format_str.lower()
        valid_formats = {'csv', 'parquet'}

        if format_str not in valid_formats:
            raise ValueError(f"Invalid output_format: {format_str}. Valid options: {valid_formats}")

        return format_str


# ============================================================================
# TESTS: Date Parsing and Validation
# ============================================================================


class TestDateParsing:
    """Test date parsing and validation."""

    def test_parse_valid_date(self):
        """Test parsing valid date strings."""
        parser = CLIArgumentParser()

        date1 = parser.parse_date('2025-10-01')
        assert date1 == datetime(2025, 10, 1)

        date2 = parser.parse_date('2025-01-15')
        assert date2 == datetime(2025, 1, 15)

    def test_parse_invalid_date_format(self):
        """Test parsing invalid date formats."""
        parser = CLIArgumentParser()

        with pytest.raises(ValueError, match="Invalid date format"):
            parser.parse_date('10/01/2025')  # Wrong format

        with pytest.raises(ValueError, match="Invalid date format"):
            parser.parse_date('2025-13-01')  # Invalid month

        with pytest.raises(ValueError, match="Invalid date format"):
            parser.parse_date('not-a-date')

    def test_validate_date_range_valid(self):
        """Test valid date ranges."""
        parser = CLIArgumentParser()

        from_date = datetime(2025, 10, 1)
        to_date = datetime(2025, 10, 31)

        assert parser.validate_date_range(from_date, to_date) is True

        # Same date should be valid
        assert parser.validate_date_range(from_date, from_date) is True

    def test_validate_date_range_invalid(self):
        """Test invalid date ranges (from > to)."""
        parser = CLIArgumentParser()

        from_date = datetime(2025, 10, 31)
        to_date = datetime(2025, 10, 1)

        with pytest.raises(ValueError, match="must be <="):
            parser.validate_date_range(from_date, to_date)


# ============================================================================
# TESTS: Asset Parsing
# ============================================================================


class TestAssetParsing:
    """Test asset parsing and validation."""

    def test_parse_single_asset(self):
        """Test parsing single asset."""
        parser = CLIArgumentParser()

        assert parser.parse_assets('BTC') == ['BTC']
        assert parser.parse_assets('ETH') == ['ETH']
        assert parser.parse_assets('btc') == ['BTC']  # Case insensitive

    def test_parse_multiple_assets(self):
        """Test parsing multiple assets."""
        parser = CLIArgumentParser()

        assets = parser.parse_assets('BTC,ETH')
        assert set(assets) == {'BTC', 'ETH'}

        assets = parser.parse_assets('btc, eth')  # Spaces and case
        assert set(assets) == {'BTC', 'ETH'}

    def test_parse_invalid_assets(self):
        """Test parsing invalid assets."""
        parser = CLIArgumentParser()

        with pytest.raises(ValueError, match="Invalid assets"):
            parser.parse_assets('BTC,INVALID')

        with pytest.raises(ValueError, match="Invalid assets"):
            parser.parse_assets('SOL')

    def test_parse_empty_assets(self):
        """Test parsing empty assets string."""
        parser = CLIArgumentParser()

        with pytest.raises(ValueError, match="cannot be empty"):
            parser.parse_assets('')


# ============================================================================
# TESTS: Days to Expiry Validation
# ============================================================================


class TestDaysValidation:
    """Test days to expiry range validation."""

    def test_validate_valid_days_range(self):
        """Test valid days ranges."""
        parser = CLIArgumentParser()

        assert parser.validate_days_range(7, 30) is True
        assert parser.validate_days_range(0, 365) is True
        assert parser.validate_days_range(None, None) is True  # No filtering
        assert parser.validate_days_range(7, None) is True  # Min only
        assert parser.validate_days_range(None, 30) is True  # Max only

    def test_validate_invalid_days_range(self):
        """Test invalid days ranges."""
        parser = CLIArgumentParser()

        # Negative values
        with pytest.raises(ValueError, match="must be >= 0"):
            parser.validate_days_range(-1, 30)

        with pytest.raises(ValueError, match="must be >= 0"):
            parser.validate_days_range(7, -1)

        # Min > Max
        with pytest.raises(ValueError, match="must be <="):
            parser.validate_days_range(30, 7)


# ============================================================================
# TESTS: Option Type Parsing
# ============================================================================


class TestOptionTypeParsing:
    """Test option type parsing and validation."""

    def test_parse_valid_option_types(self):
        """Test parsing valid option types."""
        parser = CLIArgumentParser()

        assert parser.parse_option_type('call') == 'call'
        assert parser.parse_option_type('put') == 'put'
        assert parser.parse_option_type('both') == 'both'
        assert parser.parse_option_type('CALL') == 'call'  # Case insensitive
        assert parser.parse_option_type(None) == 'both'  # Default
        assert parser.parse_option_type('') == 'both'  # Default

    def test_parse_invalid_option_types(self):
        """Test parsing invalid option types."""
        parser = CLIArgumentParser()

        with pytest.raises(ValueError, match="Invalid option_type"):
            parser.parse_option_type('invalid')

        with pytest.raises(ValueError, match="Invalid option_type"):
            parser.parse_option_type('calls')


# ============================================================================
# TESTS: Strike Price Validation
# ============================================================================


class TestStrikeValidation:
    """Test strike price range validation."""

    def test_validate_valid_strike_range(self):
        """Test valid strike ranges."""
        parser = CLIArgumentParser()

        assert parser.validate_strike_range(50000, 70000) is True
        assert parser.validate_strike_range(0, 100000) is True
        assert parser.validate_strike_range(None, None) is True  # No filtering
        assert parser.validate_strike_range(50000, None) is True  # Min only
        assert parser.validate_strike_range(None, 70000) is True  # Max only

    def test_validate_invalid_strike_range(self):
        """Test invalid strike ranges."""
        parser = CLIArgumentParser()

        # Negative values
        with pytest.raises(ValueError, match="must be >= 0"):
            parser.validate_strike_range(-1000, 70000)

        with pytest.raises(ValueError, match="must be >= 0"):
            parser.validate_strike_range(50000, -1000)

        # Min > Max
        with pytest.raises(ValueError, match="must be <="):
            parser.validate_strike_range(70000, 50000)


# ============================================================================
# TESTS: Resample Interval Parsing
# ============================================================================


class TestResampleIntervalParsing:
    """Test resample interval parsing and validation."""

    def test_parse_valid_intervals(self):
        """Test parsing valid interval strings."""
        parser = CLIArgumentParser()

        assert parser.parse_resample_interval('1s') == '1s'
        assert parser.parse_resample_interval('5s') == '5s'
        assert parser.parse_resample_interval('1m') == '1m'
        assert parser.parse_resample_interval('5m') == '5m'
        assert parser.parse_resample_interval('15m') == '15m'
        assert parser.parse_resample_interval('1h') == '1h'
        assert parser.parse_resample_interval('1d') == '1d'
        assert parser.parse_resample_interval(None) is None  # No resampling
        assert parser.parse_resample_interval('') is None  # No resampling

    def test_parse_invalid_intervals(self):
        """Test parsing invalid interval strings."""
        parser = CLIArgumentParser()

        with pytest.raises(ValueError):
            parser.parse_resample_interval('5x')  # Invalid suffix

        with pytest.raises(ValueError):
            parser.parse_resample_interval('abcs')  # Non-numeric value

        with pytest.raises(ValueError):
            parser.parse_resample_interval('0s')  # Zero value

        with pytest.raises(ValueError):
            parser.parse_resample_interval('s')  # No value


# ============================================================================
# TESTS: Output Format Validation
# ============================================================================


class TestOutputFormatValidation:
    """Test output format validation."""

    def test_validate_valid_formats(self):
        """Test valid output formats."""
        parser = CLIArgumentParser()

        assert parser.validate_output_format('csv') == 'csv'
        assert parser.validate_output_format('parquet') == 'parquet'
        assert parser.validate_output_format('CSV') == 'csv'  # Case insensitive
        assert parser.validate_output_format('PARQUET') == 'parquet'
        assert parser.validate_output_format(None) == 'parquet'  # Default
        assert parser.validate_output_format('') == 'parquet'  # Default

    def test_validate_invalid_formats(self):
        """Test invalid output formats."""
        parser = CLIArgumentParser()

        with pytest.raises(ValueError, match="Invalid output_format"):
            parser.validate_output_format('json')

        with pytest.raises(ValueError, match="Invalid output_format"):
            parser.validate_output_format('txt')


# ============================================================================
# TESTS: Integration - Full Argument Set
# ============================================================================


class TestFullArgumentValidation:
    """Test validation of complete argument sets."""

    def test_valid_full_argument_set(self, valid_cli_args):
        """Test a complete valid argument set."""
        parser = CLIArgumentParser()

        # Parse and validate all arguments
        from_date = parser.parse_date(valid_cli_args['from_date'])
        to_date = parser.parse_date(valid_cli_args['to_date'])
        parser.validate_date_range(from_date, to_date)

        assets = parser.parse_assets(valid_cli_args['assets'])
        assert len(assets) == 2

        parser.validate_days_range(
            valid_cli_args['min_days'],
            valid_cli_args['max_days']
        )

        option_type = parser.parse_option_type(valid_cli_args['option_type'])
        assert option_type == 'both'

        parser.validate_strike_range(
            valid_cli_args['strike_min'],
            valid_cli_args['strike_max']
        )

        interval = parser.parse_resample_interval(valid_cli_args['resample_interval'])
        assert interval is None

        output_format = parser.validate_output_format(valid_cli_args['output_format'])
        assert output_format == 'parquet'

    def test_minimal_argument_set(self):
        """Test minimal required arguments."""
        parser = CLIArgumentParser()

        # Only required: dates and assets
        from_date = parser.parse_date('2025-10-01')
        to_date = parser.parse_date('2025-10-01')
        assets = parser.parse_assets('BTC')

        # All others can be None/default
        assert parser.validate_days_range(None, None) is True
        assert parser.parse_option_type(None) == 'both'
        assert parser.validate_strike_range(None, None) is True
        assert parser.parse_resample_interval(None) is None
        assert parser.validate_output_format(None) == 'parquet'
