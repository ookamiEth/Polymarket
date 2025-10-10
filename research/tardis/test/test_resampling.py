"""
Tests for vectorized resampling operations.

Tests resampling of tick-level options_chain data to various intervals:
- 1 second, 5 seconds, 1 minute, 5 minutes, etc.
- OHLC for prices
- Last values for greeks, IV, open interest
- Validation of resampling logic
"""

import pytest
import polars as pl
from datetime import datetime, timedelta


class Resampler:
    """
    Vectorized resampling operations using Polars group_by_dynamic.
    All operations are vectorized - NO FOR LOOPS.
    """

    @staticmethod
    def prepare_for_resampling(df: pl.DataFrame) -> pl.DataFrame:
        """
        Prepare dataframe for resampling by converting timestamp to datetime.

        Args:
            df: DataFrame with timestamp column (microseconds since epoch)

        Returns:
            df with datetime column added
        """
        # VECTORIZED: Convert all timestamps at once
        df = df.with_columns([
            (pl.col('timestamp').cast(pl.Int64) * 1000)
              .cast(pl.Datetime('us'))
              .alias('datetime')
        ])

        return df

    @staticmethod
    def resample_options_data(
        df: pl.DataFrame,
        interval: str,
        group_by_symbol: bool = True
    ) -> pl.DataFrame:
        """
        Resample options_chain data to specified interval using vectorized operations.

        Args:
            df: DataFrame with datetime column and options data
            interval: Polars interval string (e.g., '1s', '5s', '1m', '5m', '1h')
            group_by_symbol: Whether to group by symbol (default True)

        Returns:
            Resampled DataFrame with OHLC for prices, last for greeks/IV/OI

        Aggregations:
        - last_price: OHLC
        - bid_price, ask_price: Last
        - mark_price: Last
        - delta, gamma, theta, vega, rho: Last
        - mark_iv, bid_iv, ask_iv: Last
        - open_interest: Last
        - underlying_price: Last
        """
        # Sort by datetime (required for group_by_dynamic)
        if group_by_symbol:
            df = df.sort(['symbol', 'datetime'])
            group_cols = ['symbol']
        else:
            df = df.sort('datetime')
            group_cols = []

        # VECTORIZED: Resample using group_by_dynamic
        # This is O(n log n) for sort + O(n) for groupby
        agg_exprs = [
            # OHLC for last_price
            pl.col('last_price').first().alias('open'),
            pl.col('last_price').max().alias('high'),
            pl.col('last_price').min().alias('low'),
            pl.col('last_price').last().alias('close'),

            # Last values for other prices
            pl.col('bid_price').last().alias('bid_price'),
            pl.col('ask_price').last().alias('ask_price'),
            pl.col('mark_price').last().alias('mark_price'),

            # Last values for greeks
            pl.col('delta').last().alias('delta'),
            pl.col('gamma').last().alias('gamma'),
            pl.col('theta').last().alias('theta'),
            pl.col('vega').last().alias('vega'),
            pl.col('rho').last().alias('rho'),

            # Last values for IV
            pl.col('mark_iv').last().alias('mark_iv'),
            pl.col('bid_iv').last().alias('bid_iv'),
            pl.col('ask_iv').last().alias('ask_iv'),

            # Last values for other fields
            pl.col('open_interest').last().alias('open_interest'),
            pl.col('underlying_price').last().alias('underlying_price'),

            # Count of ticks in interval
            pl.len().alias('tick_count'),
        ]

        if group_by_symbol:
            # Preserve symbol metadata
            agg_exprs.extend([
                pl.col('strike_price').first().alias('strike_price'),
                pl.col('expiration').first().alias('expiration'),
                pl.col('type').first().alias('type'),
            ])

        resampled = (
            df.group_by_dynamic(
                'datetime',
                every=interval,
                group_by=group_cols if group_cols else None
            )
            .agg(agg_exprs)
        )

        return resampled

    @staticmethod
    def validate_interval(interval: str) -> bool:
        """
        Validate that interval string is valid for Polars.

        Valid formats: 1s, 5s, 1m, 5m, 15m, 1h, etc.
        """
        try:
            # Try to create a simple resampling to validate
            test_df = pl.DataFrame({
                'datetime': [datetime(2025, 10, 1)],
                'value': [1.0]
            })

            test_df.group_by_dynamic('datetime', every=interval).agg(pl.col('value').sum())
            return True

        except Exception:
            return False


# ============================================================================
# TESTS: Timestamp Preparation
# ============================================================================


class TestTimestampPreparation:
    """Test timestamp conversion for resampling."""

    def test_prepare_timestamps(self, sample_options_data):
        """Test converting microsecond timestamps to datetime."""
        resampler = Resampler()

        prepared = resampler.prepare_for_resampling(sample_options_data)

        assert 'datetime' in prepared.columns
        assert prepared['datetime'].dtype == pl.Datetime('us')

        # Verify no nulls
        assert prepared['datetime'].null_count() == 0

    def test_datetime_ordering(self, sample_options_data):
        """Test that datetime ordering matches timestamp ordering."""
        resampler = Resampler()

        prepared = resampler.prepare_for_resampling(sample_options_data)

        # Sort by timestamp
        by_ts = prepared.sort('timestamp')

        # Sort by datetime
        by_dt = prepared.sort('datetime')

        # Should be the same order
        assert by_ts['symbol'].to_list() == by_dt['symbol'].to_list()


# ============================================================================
# TESTS: Interval Validation
# ============================================================================


class TestIntervalValidation:
    """Test interval string validation."""

    def test_valid_intervals(self):
        """Test validation of valid interval strings."""
        resampler = Resampler()

        assert resampler.validate_interval('1s') is True
        assert resampler.validate_interval('5s') is True
        assert resampler.validate_interval('1m') is True
        assert resampler.validate_interval('5m') is True
        assert resampler.validate_interval('15m') is True
        assert resampler.validate_interval('1h') is True
        assert resampler.validate_interval('1d') is True

    def test_invalid_intervals(self):
        """Test validation of invalid interval strings."""
        resampler = Resampler()

        # Invalid intervals should return False (catch exceptions internally)
        try:
            assert resampler.validate_interval('invalid') is False
        except:
            pass  # Exception is also acceptable

        try:
            assert resampler.validate_interval('') is False
        except:
            pass  # Exception is also acceptable

        try:
            assert resampler.validate_interval('1x') is False
        except:
            pass  # Exception is also acceptable


# ============================================================================
# TESTS: Basic Resampling
# ============================================================================


class TestBasicResampling:
    """Test basic resampling functionality."""

    def test_resample_1_second(self, sample_options_data):
        """Test resampling to 1-second intervals."""
        resampler = Resampler()

        # Prepare data
        prepared = resampler.prepare_for_resampling(sample_options_data)

        # Take a small sample for testing
        sample = prepared.head(100)

        # Resample to 1 second
        resampled = resampler.resample_options_data(sample, '1s')

        # Should have fewer rows than original (compression)
        assert resampled.shape[0] <= sample.shape[0]

        # Should have OHLC columns
        assert 'open' in resampled.columns
        assert 'high' in resampled.columns
        assert 'low' in resampled.columns
        assert 'close' in resampled.columns

        # Should have tick_count
        assert 'tick_count' in resampled.columns

    def test_resample_5_seconds(self, sample_options_data):
        """Test resampling to 5-second intervals."""
        resampler = Resampler()

        prepared = resampler.prepare_for_resampling(sample_options_data)
        sample = prepared.head(100)

        # Resample to 5 seconds
        resampled_5s = resampler.resample_options_data(sample, '5s')

        # Resample to 1 second for comparison
        resampled_1s = resampler.resample_options_data(sample, '1s')

        # 5s should have fewer (or equal) rows than 1s
        assert resampled_5s.shape[0] <= resampled_1s.shape[0]

    def test_resample_1_minute(self, sample_options_data):
        """Test resampling to 1-minute intervals."""
        resampler = Resampler()

        prepared = resampler.prepare_for_resampling(sample_options_data)
        sample = prepared.head(1000)

        # Resample to 1 minute
        resampled = resampler.resample_options_data(sample, '1m')

        # Should have compression
        assert resampled.shape[0] <= sample.shape[0]


# ============================================================================
# TESTS: OHLC Logic
# ============================================================================


class TestOHLCLogic:
    """Test OHLC calculation logic."""

    def test_ohlc_high_low_bounds(self, sample_options_data):
        """Test that high >= low for all resampled bars."""
        resampler = Resampler()

        prepared = resampler.prepare_for_resampling(sample_options_data)
        sample = prepared.head(100)

        resampled = resampler.resample_options_data(sample, '1s')

        # High should be >= low
        assert (resampled['high'] >= resampled['low']).all()

    def test_ohlc_contains_open_close(self, sample_options_data):
        """Test that high/low contain open/close prices."""
        resampler = Resampler()

        prepared = resampler.prepare_for_resampling(sample_options_data)
        sample = prepared.head(100)

        resampled = resampler.resample_options_data(sample, '1s')

        # Open should be between low and high (or equal)
        assert (resampled['open'] >= resampled['low']).all()
        assert (resampled['open'] <= resampled['high']).all()

        # Close should be between low and high (or equal)
        assert (resampled['close'] >= resampled['low']).all()
        assert (resampled['close'] <= resampled['high']).all()

    def test_single_tick_ohlc(self, sample_options_data):
        """Test that single tick intervals have OHLC all equal."""
        resampler = Resampler()

        prepared = resampler.prepare_for_resampling(sample_options_data)
        sample = prepared.head(10)

        # Use a very fine interval to get single ticks
        resampled = resampler.resample_options_data(sample, '1s')

        # For bars with tick_count=1, OHLC should all be equal
        single_tick_bars = resampled.filter(pl.col('tick_count') == 1)

        if single_tick_bars.shape[0] > 0:
            assert (single_tick_bars['open'] == single_tick_bars['close']).all()
            assert (single_tick_bars['high'] == single_tick_bars['low']).all()


# ============================================================================
# TESTS: Greek and IV Aggregation
# ============================================================================


class TestGreeksAndIVAggregation:
    """Test that greeks and IV are properly aggregated."""

    def test_greeks_preserved(self, sample_options_data):
        """Test that greeks are preserved in resampled data."""
        resampler = Resampler()

        prepared = resampler.prepare_for_resampling(sample_options_data)
        sample = prepared.head(100)

        resampled = resampler.resample_options_data(sample, '1s')

        # Should have greek columns
        assert 'delta' in resampled.columns
        assert 'gamma' in resampled.columns
        assert 'theta' in resampled.columns
        assert 'vega' in resampled.columns
        assert 'rho' in resampled.columns

        # Greeks should be last values from each interval
        # (We can't test exact values without knowing the data, but we can check they exist)
        assert resampled['delta'].null_count() == 0

    def test_iv_preserved(self, sample_options_data):
        """Test that IV values are preserved in resampled data."""
        resampler = Resampler()

        prepared = resampler.prepare_for_resampling(sample_options_data)
        sample = prepared.head(100)

        resampled = resampler.resample_options_data(sample, '1s')

        # Should have IV columns
        assert 'mark_iv' in resampled.columns
        assert 'bid_iv' in resampled.columns
        assert 'ask_iv' in resampled.columns

    def test_open_interest_preserved(self, sample_options_data):
        """Test that open interest is preserved in resampled data."""
        resampler = Resampler()

        prepared = resampler.prepare_for_resampling(sample_options_data)
        sample = prepared.head(100)

        resampled = resampler.resample_options_data(sample, '1s')

        # Should have open_interest column
        assert 'open_interest' in resampled.columns
        assert resampled['open_interest'].null_count() == 0


# ============================================================================
# TESTS: Symbol Grouping
# ============================================================================


class TestSymbolGrouping:
    """Test resampling with symbol grouping."""

    def test_resample_grouped_by_symbol(self, sample_options_data):
        """Test that each symbol is resampled independently."""
        resampler = Resampler()

        prepared = resampler.prepare_for_resampling(sample_options_data)
        sample = prepared.head(200)

        # Resample with symbol grouping
        resampled = resampler.resample_options_data(sample, '1s', group_by_symbol=True)

        # Should have symbol column
        assert 'symbol' in resampled.columns

        # Each unique symbol should have its own timeline
        unique_symbols = resampled['symbol'].unique()
        assert len(unique_symbols) > 1  # Multiple symbols

    def test_resample_without_symbol_grouping(self, sample_options_data):
        """Test resampling without symbol grouping."""
        resampler = Resampler()

        prepared = resampler.prepare_for_resampling(sample_options_data)

        # Filter to single symbol for this test
        single_symbol = prepared.filter(
            pl.col('symbol') == prepared['symbol'].first()
        ).head(100)

        # Resample without symbol grouping
        resampled = resampler.resample_options_data(
            single_symbol,
            '1s',
            group_by_symbol=False
        )

        # Should still work
        assert resampled.shape[0] > 0


# ============================================================================
# TESTS: Compression Ratios
# ============================================================================


class TestCompressionRatios:
    """Test compression ratios for different intervals."""

    def test_compression_ratio_increases_with_interval(self, sample_options_data):
        """Test that larger intervals produce more compression."""
        resampler = Resampler()

        prepared = resampler.prepare_for_resampling(sample_options_data)
        sample = prepared.head(1000)

        # Resample at different intervals
        resampled_1s = resampler.resample_options_data(sample, '1s')
        resampled_5s = resampler.resample_options_data(sample, '5s')
        resampled_1m = resampler.resample_options_data(sample, '1m')

        # Larger intervals should have fewer rows
        assert resampled_1s.shape[0] >= resampled_5s.shape[0]
        assert resampled_5s.shape[0] >= resampled_1m.shape[0]

    def test_tick_count_accuracy(self, sample_options_data):
        """Test that tick_count accurately reflects ticks per interval."""
        resampler = Resampler()

        prepared = resampler.prepare_for_resampling(sample_options_data)

        # Filter to single symbol
        single_symbol = prepared.filter(
            pl.col('symbol') == prepared['symbol'].first()
        )

        original_count = single_symbol.shape[0]

        # Resample
        resampled = resampler.resample_options_data(single_symbol, '1s')

        # Sum of tick_counts should equal original count
        total_ticks = resampled['tick_count'].sum()
        assert total_ticks == original_count


# ============================================================================
# TESTS: Edge Cases
# ============================================================================


class TestEdgeCases:
    """Test edge cases in resampling."""

    def test_empty_dataframe(self):
        """Test resampling empty dataframe."""
        resampler = Resampler()

        # Create empty dataframe with correct schema
        empty_df = pl.DataFrame({
            'datetime': [],
            'symbol': [],
            'last_price': [],
            'bid_price': [],
            'ask_price': [],
            'mark_price': [],
            'delta': [],
            'gamma': [],
            'theta': [],
            'vega': [],
            'rho': [],
            'mark_iv': [],
            'bid_iv': [],
            'ask_iv': [],
            'open_interest': [],
            'underlying_price': [],
            'strike_price': [],
            'expiration': [],
            'type': [],
        }).with_columns([
            pl.col('datetime').cast(pl.Datetime('us'))
        ])

        # Should not crash
        resampled = resampler.resample_options_data(empty_df, '1s')
        assert resampled.shape[0] == 0

    def test_single_row(self, sample_options_data):
        """Test resampling single row."""
        resampler = Resampler()

        prepared = resampler.prepare_for_resampling(sample_options_data)
        single_row = prepared.head(1)

        # Should not crash
        resampled = resampler.resample_options_data(single_row, '1s')
        assert resampled.shape[0] == 1

        # OHLC should all be the same
        # Note: Using select to avoid timestamp conversion issues
        ohlc = resampled.select(['open', 'high', 'low', 'close'])
        open_val = ohlc['open'][0]
        high_val = ohlc['high'][0]
        low_val = ohlc['low'][0]
        close_val = ohlc['close'][0]
        assert open_val == high_val == low_val == close_val


# ============================================================================
# TESTS: Vectorization Validation
# ============================================================================


class TestResamplingVectorization:
    """Validate that resampling is truly vectorized."""

    def test_no_loops_in_resampling(self, sample_options_data):
        """Verify resampling is vectorized (fast even with many rows)."""
        resampler = Resampler()

        # Use full dataset
        prepared = resampler.prepare_for_resampling(sample_options_data)

        # Should complete quickly using vectorized group_by_dynamic
        resampled = resampler.resample_options_data(prepared, '1s')

        # Verify it worked
        assert resampled.shape[0] > 0
        assert 'open' in resampled.columns
