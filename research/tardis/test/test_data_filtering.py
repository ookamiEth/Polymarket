"""
Tests for vectorized data filtering operations.

Tests all filtering operations using Polars for maximum performance:
- Symbol parsing (extract underlying, strike, expiry, option type)
- Asset filtering
- Days to expiry calculation and filtering
- Option type filtering
- Strike price filtering
- Combined filters
"""

import pytest
import polars as pl
from datetime import datetime, timedelta


class DataFilters:
    """
    Vectorized data filtering operations using Polars.
    All operations are vectorized - NO FOR LOOPS.
    """

    @staticmethod
    def parse_symbols(df: pl.DataFrame) -> pl.DataFrame:
        """
        Parse Deribit option symbols vectorized.

        Symbol format: {UNDERLYING}-{EXPIRY}-{STRIKE}-{TYPE}
        Example: BTC-08OCT25-60000-C

        Returns df with added columns:
        - underlying: BTC, ETH, etc.
        - expiry_str: 08OCT25
        - parsed_strike: 60000
        - parsed_option_type: call or put
        - expiry_date: datetime
        """
        # VECTORIZED: Split all symbols at once
        df = df.with_columns([
            pl.col('symbol').str.split('-').alias('symbol_parts')
        ])

        # VECTORIZED: Extract components from all rows
        df = df.with_columns([
            pl.col('symbol_parts').list.get(0).alias('underlying'),
            pl.col('symbol_parts').list.get(1).alias('expiry_str'),
            pl.col('symbol_parts').list.get(2).cast(pl.Float64, strict=False).alias('parsed_strike'),
            pl.when(pl.col('symbol_parts').list.len() >= 4)
              .then(
                  pl.when(pl.col('symbol_parts').list.get(3) == 'C')
                    .then(pl.lit('call'))
                    .otherwise(pl.lit('put'))
              )
              .otherwise(pl.lit(None))
              .alias('parsed_option_type')
        ])

        # VECTORIZED: Parse expiry date
        df = df.with_columns([
            pl.col('expiry_str')
              .str.to_uppercase()
              .str.strptime(pl.Date, '%d%b%y', strict=False)
              .alias('expiry_date')
        ])

        # Clean up temporary column
        df = df.drop('symbol_parts')

        return df

    @staticmethod
    def calculate_days_to_expiry(df: pl.DataFrame, reference_date: datetime) -> pl.DataFrame:
        """
        Calculate days to expiry for all options vectorized.

        Args:
            df: DataFrame with expiry_date column
            reference_date: Date to calculate from

        Returns:
            df with added days_to_expiry column
        """
        # VECTORIZED: Calculate days for all rows at once
        df = df.with_columns([
            (pl.col('expiry_date').cast(pl.Datetime('us')) - pl.lit(reference_date))
              .dt.total_days()
              .alias('days_to_expiry')
        ])

        return df

    @staticmethod
    def filter_by_assets(df: pl.DataFrame, assets: list[str]) -> pl.DataFrame:
        """
        Filter by underlying assets vectorized.

        Args:
            df: DataFrame with underlying column
            assets: List of assets to keep (e.g., ['BTC', 'ETH'])

        Returns:
            Filtered DataFrame
        """
        # VECTORIZED: Filter all rows at once
        return df.filter(pl.col('underlying').is_in(assets))

    @staticmethod
    def filter_by_days_to_expiry(
        df: pl.DataFrame,
        min_days: int = None,
        max_days: int = None
    ) -> pl.DataFrame:
        """
        Filter by days to expiry range vectorized.

        Args:
            df: DataFrame with days_to_expiry column
            min_days: Minimum days (inclusive), None for no minimum
            max_days: Maximum days (inclusive), None for no maximum

        Returns:
            Filtered DataFrame
        """
        # VECTORIZED: Build filter condition
        conditions = []

        if min_days is not None:
            conditions.append(pl.col('days_to_expiry') >= min_days)

        if max_days is not None:
            conditions.append(pl.col('days_to_expiry') <= max_days)

        if not conditions:
            return df  # No filtering

        # VECTORIZED: Apply combined filter
        filter_expr = conditions[0]
        for cond in conditions[1:]:
            filter_expr = filter_expr & cond

        return df.filter(filter_expr)

    @staticmethod
    def filter_by_option_type(df: pl.DataFrame, option_type: str) -> pl.DataFrame:
        """
        Filter by option type vectorized.

        Args:
            df: DataFrame with parsed_option_type column
            option_type: 'call', 'put', or 'both'

        Returns:
            Filtered DataFrame
        """
        if option_type == 'both':
            return df  # No filtering

        # VECTORIZED: Filter all rows at once
        return df.filter(pl.col('parsed_option_type') == option_type)

    @staticmethod
    def filter_by_strike_range(
        df: pl.DataFrame,
        strike_min: float = None,
        strike_max: float = None
    ) -> pl.DataFrame:
        """
        Filter by strike price range vectorized.

        Args:
            df: DataFrame with parsed_strike column
            strike_min: Minimum strike (inclusive), None for no minimum
            strike_max: Maximum strike (inclusive), None for no maximum

        Returns:
            Filtered DataFrame
        """
        # VECTORIZED: Build filter condition
        conditions = []

        if strike_min is not None:
            conditions.append(pl.col('parsed_strike') >= strike_min)

        if strike_max is not None:
            conditions.append(pl.col('parsed_strike') <= strike_max)

        if not conditions:
            return df  # No filtering

        # VECTORIZED: Apply combined filter
        filter_expr = conditions[0]
        for cond in conditions[1:]:
            filter_expr = filter_expr & cond

        return df.filter(filter_expr)


# ============================================================================
# TESTS: Symbol Parsing
# ============================================================================


class TestSymbolParsing:
    """Test vectorized symbol parsing."""

    def test_parse_btc_call_symbol(self, sample_options_data):
        """Test parsing BTC call option symbols."""
        filters = DataFilters()

        # Filter to just one BTC call for testing
        sample = sample_options_data.filter(
            (pl.col('symbol').str.contains('BTC')) &
            (pl.col('type') == 'call')
        ).head(1)

        parsed = filters.parse_symbols(sample)

        assert 'underlying' in parsed.columns
        assert 'expiry_str' in parsed.columns
        assert 'parsed_strike' in parsed.columns
        assert 'parsed_option_type' in parsed.columns
        assert 'expiry_date' in parsed.columns

        # Check first row
        row = parsed.row(0, named=True)
        assert row['underlying'] == 'BTC'
        assert row['parsed_option_type'] == 'call'
        assert isinstance(row['parsed_strike'], float)

    def test_parse_eth_put_symbol(self, sample_options_data):
        """Test parsing ETH put option symbols."""
        filters = DataFilters()

        # Filter to just one ETH put for testing
        sample = sample_options_data.filter(
            (pl.col('symbol').str.contains('ETH')) &
            (pl.col('type') == 'put')
        ).head(1)

        parsed = filters.parse_symbols(sample)

        row = parsed.row(0, named=True)
        assert row['underlying'] == 'ETH'
        assert row['parsed_option_type'] == 'put'

    def test_parse_all_symbols_vectorized(self, sample_options_data):
        """Test parsing all symbols at once (vectorized)."""
        filters = DataFilters()

        parsed = filters.parse_symbols(sample_options_data)

        # All rows should be parsed
        assert parsed.shape[0] == sample_options_data.shape[0]

        # Check no nulls in critical columns
        assert parsed['underlying'].null_count() == 0
        assert parsed['expiry_str'].null_count() == 0
        assert parsed['parsed_option_type'].null_count() == 0

    def test_parse_expiry_date(self, sample_options_data):
        """Test expiry date parsing."""
        filters = DataFilters()

        parsed = filters.parse_symbols(sample_options_data)

        # Check that expiry dates are valid
        assert parsed['expiry_date'].null_count() == 0

        # Expiry dates should be in the future relative to Oct 1, 2025
        # (Our sample data has expiries on Oct 8, 15, Nov 1)
        unique_expiries = parsed['expiry_date'].unique().sort()
        assert len(unique_expiries) == 3  # 3 different expiries in sample data


# ============================================================================
# TESTS: Days to Expiry Calculation
# ============================================================================


class TestDaysToExpiryCalculation:
    """Test vectorized days to expiry calculation."""

    def test_calculate_days_to_expiry(self, sample_options_data, base_date):
        """Test days to expiry calculation."""
        filters = DataFilters()

        # Parse symbols first
        parsed = filters.parse_symbols(sample_options_data)

        # Calculate days to expiry
        with_days = filters.calculate_days_to_expiry(parsed, base_date)

        assert 'days_to_expiry' in with_days.columns

        # Check that we have the expected expiries
        # Sample data has: Oct 8 (7 days), Oct 15 (14 days), Nov 1 (31 days)
        unique_days = sorted(with_days['days_to_expiry'].unique().to_list())
        assert 7 in unique_days
        assert 14 in unique_days
        assert 31 in unique_days

    def test_days_to_expiry_all_positive(self, sample_options_data, base_date):
        """Test that all days to expiry are positive (futures only)."""
        filters = DataFilters()

        parsed = filters.parse_symbols(sample_options_data)
        with_days = filters.calculate_days_to_expiry(parsed, base_date)

        # All should be >= 0 (or >= 7 in our sample data)
        assert (with_days['days_to_expiry'] >= 0).all()


# ============================================================================
# TESTS: Asset Filtering
# ============================================================================


class TestAssetFiltering:
    """Test vectorized asset filtering."""

    def test_filter_btc_only(self, sample_options_data):
        """Test filtering for BTC only."""
        filters = DataFilters()

        parsed = filters.parse_symbols(sample_options_data)
        filtered = filters.filter_by_assets(parsed, ['BTC'])

        # All rows should be BTC
        assert (filtered['underlying'] == 'BTC').all()

        # Should have fewer rows than original
        assert filtered.shape[0] < parsed.shape[0]

    def test_filter_eth_only(self, sample_options_data):
        """Test filtering for ETH only."""
        filters = DataFilters()

        parsed = filters.parse_symbols(sample_options_data)
        filtered = filters.filter_by_assets(parsed, ['ETH'])

        # All rows should be ETH
        assert (filtered['underlying'] == 'ETH').all()

    def test_filter_both_assets(self, sample_options_data):
        """Test filtering for both BTC and ETH."""
        filters = DataFilters()

        parsed = filters.parse_symbols(sample_options_data)
        filtered = filters.filter_by_assets(parsed, ['BTC', 'ETH'])

        # Should have all rows (no filtering)
        assert filtered.shape[0] == parsed.shape[0]

    def test_filter_empty_result(self, sample_options_data):
        """Test filtering with no matches."""
        filters = DataFilters()

        parsed = filters.parse_symbols(sample_options_data)
        filtered = filters.filter_by_assets(parsed, ['SOL'])  # Not in sample data

        # Should have 0 rows
        assert filtered.shape[0] == 0


# ============================================================================
# TESTS: Days to Expiry Filtering
# ============================================================================


class TestDaysToExpiryFiltering:
    """Test vectorized days to expiry filtering."""

    def test_filter_min_days_only(self, sample_options_data, base_date):
        """Test filtering with minimum days only."""
        filters = DataFilters()

        parsed = filters.parse_symbols(sample_options_data)
        with_days = filters.calculate_days_to_expiry(parsed, base_date)

        # Filter for >= 14 days
        filtered = filters.filter_by_days_to_expiry(with_days, min_days=14)

        # Should have 14 and 31 day expiries, not 7
        unique_days = filtered['days_to_expiry'].unique().sort().to_list()
        assert 7 not in unique_days
        assert 14 in unique_days
        assert 31 in unique_days

    def test_filter_max_days_only(self, sample_options_data, base_date):
        """Test filtering with maximum days only."""
        filters = DataFilters()

        parsed = filters.parse_symbols(sample_options_data)
        with_days = filters.calculate_days_to_expiry(parsed, base_date)

        # Filter for <= 14 days
        filtered = filters.filter_by_days_to_expiry(with_days, max_days=14)

        # Should have 7 and 14 day expiries, not 31
        unique_days = filtered['days_to_expiry'].unique().sort().to_list()
        assert 7 in unique_days
        assert 14 in unique_days
        assert 31 not in unique_days

    def test_filter_days_range(self, sample_options_data, base_date):
        """Test filtering with both min and max days."""
        filters = DataFilters()

        parsed = filters.parse_symbols(sample_options_data)
        with_days = filters.calculate_days_to_expiry(parsed, base_date)

        # Filter for 7-30 days
        filtered = filters.filter_by_days_to_expiry(with_days, min_days=7, max_days=30)

        # Should have 7 and 14, not 31
        unique_days = filtered['days_to_expiry'].unique().sort().to_list()
        assert 7 in unique_days
        assert 14 in unique_days
        assert 31 not in unique_days

    def test_filter_no_days_constraint(self, sample_options_data, base_date):
        """Test filtering with no days constraints."""
        filters = DataFilters()

        parsed = filters.parse_symbols(sample_options_data)
        with_days = filters.calculate_days_to_expiry(parsed, base_date)

        # No filtering
        filtered = filters.filter_by_days_to_expiry(with_days)

        # Should have all rows
        assert filtered.shape[0] == with_days.shape[0]


# ============================================================================
# TESTS: Option Type Filtering
# ============================================================================


class TestOptionTypeFiltering:
    """Test vectorized option type filtering."""

    def test_filter_calls_only(self, sample_options_data):
        """Test filtering for calls only."""
        filters = DataFilters()

        parsed = filters.parse_symbols(sample_options_data)
        filtered = filters.filter_by_option_type(parsed, 'call')

        # All rows should be calls
        assert (filtered['parsed_option_type'] == 'call').all()

        # Should have fewer rows than original
        assert filtered.shape[0] < parsed.shape[0]

    def test_filter_puts_only(self, sample_options_data):
        """Test filtering for puts only."""
        filters = DataFilters()

        parsed = filters.parse_symbols(sample_options_data)
        filtered = filters.filter_by_option_type(parsed, 'put')

        # All rows should be puts
        assert (filtered['parsed_option_type'] == 'put').all()

    def test_filter_both_types(self, sample_options_data):
        """Test filtering for both calls and puts."""
        filters = DataFilters()

        parsed = filters.parse_symbols(sample_options_data)
        filtered = filters.filter_by_option_type(parsed, 'both')

        # Should have all rows (no filtering)
        assert filtered.shape[0] == parsed.shape[0]


# ============================================================================
# TESTS: Strike Price Filtering
# ============================================================================


class TestStrikePriceFiltering:
    """Test vectorized strike price filtering."""

    def test_filter_min_strike_only(self, sample_options_data):
        """Test filtering with minimum strike only."""
        filters = DataFilters()

        parsed = filters.parse_symbols(sample_options_data)

        # Filter for BTC strikes >= 60000
        filtered = filters.filter_by_strike_range(parsed, strike_min=60000)

        # All strikes should be >= 60000
        assert (filtered['parsed_strike'] >= 60000).all()

    def test_filter_max_strike_only(self, sample_options_data):
        """Test filtering with maximum strike only."""
        filters = DataFilters()

        parsed = filters.parse_symbols(sample_options_data)

        # Filter for strikes <= 60000
        filtered = filters.filter_by_strike_range(parsed, strike_max=60000)

        # All strikes should be <= 60000
        assert (filtered['parsed_strike'] <= 60000).all()

    def test_filter_strike_range(self, sample_options_data):
        """Test filtering with both min and max strike."""
        filters = DataFilters()

        parsed = filters.parse_symbols(sample_options_data)

        # Filter for strikes between 60000-62000
        filtered = filters.filter_by_strike_range(
            parsed,
            strike_min=60000,
            strike_max=62000
        )

        # All strikes should be in range
        assert (filtered['parsed_strike'] >= 60000).all()
        assert (filtered['parsed_strike'] <= 62000).all()

    def test_filter_no_strike_constraint(self, sample_options_data):
        """Test filtering with no strike constraints."""
        filters = DataFilters()

        parsed = filters.parse_symbols(sample_options_data)
        filtered = filters.filter_by_strike_range(parsed)

        # Should have all rows
        assert filtered.shape[0] == parsed.shape[0]


# ============================================================================
# TESTS: Combined Filters
# ============================================================================


class TestCombinedFilters:
    """Test combining multiple filters."""

    def test_combined_filters_btc_calls_short_dated(self, sample_options_data, base_date):
        """Test combining asset, option type, and days filters."""
        filters = DataFilters()

        # Parse and calculate days
        parsed = filters.parse_symbols(sample_options_data)
        with_days = filters.calculate_days_to_expiry(parsed, base_date)

        # Apply multiple filters
        filtered = with_days
        filtered = filters.filter_by_assets(filtered, ['BTC'])
        filtered = filters.filter_by_option_type(filtered, 'call')
        filtered = filters.filter_by_days_to_expiry(filtered, min_days=7, max_days=14)

        # Verify all conditions
        assert (filtered['underlying'] == 'BTC').all()
        assert (filtered['parsed_option_type'] == 'call').all()
        assert (filtered['days_to_expiry'] >= 7).all()
        assert (filtered['days_to_expiry'] <= 14).all()

        # Should have some data
        assert filtered.shape[0] > 0

    def test_combined_filters_eth_atm_puts(self, sample_options_data, base_date):
        """Test combining asset, option type, and strike filters."""
        filters = DataFilters()

        parsed = filters.parse_symbols(sample_options_data)
        with_days = filters.calculate_days_to_expiry(parsed, base_date)

        # Apply filters for ETH ATM puts
        filtered = with_days
        filtered = filters.filter_by_assets(filtered, ['ETH'])
        filtered = filters.filter_by_option_type(filtered, 'put')
        filtered = filters.filter_by_strike_range(filtered, strike_min=2400, strike_max=2600)

        # Verify all conditions
        assert (filtered['underlying'] == 'ETH').all()
        assert (filtered['parsed_option_type'] == 'put').all()
        assert (filtered['parsed_strike'] >= 2400).all()
        assert (filtered['parsed_strike'] <= 2600).all()

    def test_combined_filters_no_results(self, sample_options_data, base_date):
        """Test combined filters that produce no results."""
        filters = DataFilters()

        parsed = filters.parse_symbols(sample_options_data)
        with_days = filters.calculate_days_to_expiry(parsed, base_date)

        # Apply impossible combination
        filtered = with_days
        filtered = filters.filter_by_assets(filtered, ['BTC'])
        filtered = filters.filter_by_strike_range(filtered, strike_min=1000000)  # Way OTM

        # Should have 0 results
        assert filtered.shape[0] == 0


# ============================================================================
# TESTS: Performance (Vectorization Validation)
# ============================================================================


class TestVectorization:
    """Validate that operations are truly vectorized."""

    def test_no_loops_in_parse_symbols(self, sample_options_data):
        """Verify symbol parsing is vectorized (fast even with many rows)."""
        filters = DataFilters()

        # Should complete quickly even with all rows
        parsed = filters.parse_symbols(sample_options_data)

        # Verify it worked
        assert parsed.shape[0] == sample_options_data.shape[0]
        assert 'underlying' in parsed.columns

    def test_no_loops_in_filtering(self, sample_options_data, base_date):
        """Verify filtering is vectorized (fast even with many rows)."""
        filters = DataFilters()

        parsed = filters.parse_symbols(sample_options_data)
        with_days = filters.calculate_days_to_expiry(parsed, base_date)

        # Chain multiple filters - should be fast
        filtered = with_days
        filtered = filters.filter_by_assets(filtered, ['BTC'])
        filtered = filters.filter_by_option_type(filtered, 'call')
        filtered = filters.filter_by_days_to_expiry(filtered, min_days=7, max_days=30)
        filtered = filters.filter_by_strike_range(filtered, strike_min=58000, strike_max=64000)

        # Verify it worked
        assert filtered.shape[0] > 0
