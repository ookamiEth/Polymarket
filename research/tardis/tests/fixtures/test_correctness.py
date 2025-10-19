#!/usr/bin/env python3
"""
Correctness Test: Verify Original vs Optimized produce identical results

Ensures the optimized implementation produces the same output as the original
across all test fixtures, including edge cases.
"""

import sys
import os
import msgspec.json
import polars as pl
import io

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tardis_download import _parse_quote_message, _validate_quote_data

# Import optimized version
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
from optimized_tardis_download import _empty_dataframe


def load_fixture(filename: str) -> str:
    """Load fixture file contents."""
    filepath = os.path.join(os.path.dirname(__file__), 'fixtures', filename)
    with open(filepath, 'r') as f:
        return f.read()


def process_original(response_text: str) -> pl.DataFrame:
    """Process data using original implementation."""
    lines = response_text.strip().split('\n') if response_text.strip() else []
    batch_rows = []

    for line in lines:
        try:
            msg = msgspec.json.decode(line)
            if 'quote' in msg.get('type', '') or 'book_snapshot' in msg.get('type', ''):
                quote_row = _parse_quote_message(msg)
                if quote_row and _validate_quote_data(quote_row):
                    batch_rows.append(quote_row)
        except (msgspec.DecodeError, Exception):
            pass

    if batch_rows:
        df = pl.DataFrame(batch_rows)
        # Sort for consistent comparison
        df = df.sort(['timestamp', 'symbol'])
        return df
    else:
        return _empty_dataframe()


def process_optimized(response_text: str) -> pl.DataFrame:
    """Process data using optimized implementation."""
    if not response_text.strip():
        return _empty_dataframe()

    try:
        df = pl.read_ndjson(io.StringIO(response_text))

        if len(df) > 0:
            # Vectorized filtering
            df = df.filter(pl.col('type').str.contains('quote|book_snapshot'))

            if len(df) > 0:
                df = df.filter(pl.col('symbol').str.count_matches('-') == 3)

                if len(df) > 0:
                    df = df.with_columns([
                        pl.col('symbol').str.split('-').alias('symbol_parts')
                    ])

                    df = df.with_columns([
                        pl.lit('deribit').alias('exchange'),
                        pl.col('symbol_parts').list.get(0).alias('underlying'),
                        pl.col('symbol_parts').list.get(1).alias('expiry_str'),
                        pl.col('symbol_parts').list.get(2).cast(pl.Float64, strict=False).alias('strike_price'),
                        pl.when(pl.col('symbol_parts').list.get(3) == 'C')
                          .then(pl.lit('call'))
                          .otherwise(pl.lit('put'))
                          .alias('type'),
                    ])

                    df = df.drop('symbol_parts')
                    df = df.filter(pl.col('strike_price').is_not_null())

                    if len(df) > 0:
                        # Vectorized timestamp parsing
                        df = df.with_columns([
                            pl.col('timestamp')
                              .str.strptime(pl.Datetime, format='%Y-%m-%dT%H:%M:%S%.fZ', strict=False)
                              .dt.epoch('us')
                              .fill_null(0)
                              .alias('timestamp'),
                        ])

                        df = df.with_columns([
                            pl.col('timestamp').alias('local_timestamp')
                        ])

                        # Vectorized book data extraction
                        df = df.with_columns([
                            pl.col('bids').list.first().struct.field('price').alias('bid_price'),
                            pl.col('bids').list.first().struct.field('amount').alias('bid_amount'),
                            pl.col('asks').list.first().struct.field('price').alias('ask_price'),
                            pl.col('asks').list.first().struct.field('amount').alias('ask_amount'),
                        ])

                        # Vectorized validation
                        df = df.filter(
                            (pl.col('bid_price').is_null() | (pl.col('bid_price') >= 0)) &
                            (pl.col('ask_price').is_null() | (pl.col('ask_price') >= 0)) &
                            (pl.col('bid_amount').is_null() | (pl.col('bid_amount') >= 0)) &
                            (pl.col('ask_amount').is_null() | (pl.col('ask_amount') >= 0)) &
                            (pl.col('bid_price').is_null() |
                             pl.col('ask_price').is_null() |
                             (pl.col('bid_price') <= pl.col('ask_price')))
                        )

                        df = df.select([
                            'exchange', 'symbol', 'timestamp', 'local_timestamp', 'type',
                            'strike_price', 'underlying', 'expiry_str',
                            'bid_price', 'bid_amount', 'ask_price', 'ask_amount'
                        ])

                        # Sort for consistent comparison
                        df = df.sort(['timestamp', 'symbol'])
                        return df

    except Exception as e:
        print(f"  Error in optimized: {e}")
        import traceback
        traceback.print_exc()

    return _empty_dataframe()


def compare_dataframes(df1: pl.DataFrame, df2: pl.DataFrame, tolerance: float = 1e-6) -> bool:
    """Compare two DataFrames with tolerance for floating point errors."""
    if len(df1) != len(df2):
        print(f"  ❌ Row count mismatch: {len(df1)} vs {len(df2)}")
        return False

    if df1.columns != df2.columns:
        print(f"  ❌ Column mismatch: {df1.columns} vs {df2.columns}")
        return False

    if len(df1) == 0:
        return True  # Both empty

    # Compare column by column
    for col in df1.columns:
        if df1[col].dtype == pl.Float64 or df1[col].dtype == pl.Float32:
            # Use tolerance for floating point comparison
            diff = (df1[col] - df2[col]).abs()
            if diff.max() > tolerance:
                print(f"  ❌ Column '{col}' has differences > {tolerance}")
                print(f"     Max diff: {diff.max()}")
                bad_idx = diff.arg_max()
                print(f"     At index {bad_idx}: {df1[col][bad_idx]} vs {df2[col][bad_idx]}")
                return False
        else:
            # Exact comparison for other types
            if not df1[col].equals(df2[col]):
                print(f"  ❌ Column '{col}' has differences")
                # Find first difference
                for i in range(len(df1)):
                    if df1[col][i] != df2[col][i]:
                        print(f"     At index {i}: '{df1[col][i]}' vs '{df2[col][i]}'")
                        break
                return False

    return True


def test_correctness():
    """Run correctness tests on all fixtures."""
    fixtures = [
        ('small_1k.json', '1K'),
        ('medium_10k.json', '10K'),
        ('large_100k.json', '100K'),
        ('with_book_5k.json', '5K w/book'),
        ('edge_cases.json', 'Edge Cases'),
        ('single_symbol_1k.json', 'Single Symbol'),
        ('empty.json', 'Empty'),
    ]

    print("=" * 80)
    print("CORRECTNESS TEST: Original vs Optimized")
    print("=" * 80)
    print()

    all_passed = True

    for fixture_file, label in fixtures:
        print(f"Testing: {label} ({fixture_file})")
        print("-" * 80)

        try:
            response_text = load_fixture(fixture_file)

            # Process with both implementations
            df_original = process_original(response_text)
            df_optimized = process_optimized(response_text)

            print(f"  Original:  {len(df_original):,} rows")
            print(f"  Optimized: {len(df_optimized):,} rows")

            # Compare
            if compare_dataframes(df_original, df_optimized):
                print("  ✅ PASS: Outputs match")
            else:
                print("  ❌ FAIL: Outputs differ")
                all_passed = False

                # Debug: Show sample of each
                if len(df_original) > 0:
                    print("\n  Sample from original:")
                    print(df_original.head(3))
                if len(df_optimized) > 0:
                    print("\n  Sample from optimized:")
                    print(df_optimized.head(3))

            print()

        except Exception as e:
            print(f"  ❌ ERROR: {e}")
            import traceback
            traceback.print_exc()
            all_passed = False
            print()

    print("=" * 80)
    if all_passed:
        print("✅ SUCCESS: All correctness tests passed!")
    else:
        print("❌ FAILURE: Some tests failed")
    print("=" * 80)
    print()

    return all_passed


def main():
    """Main entry point."""
    print()
    passed = test_correctness()
    sys.exit(0 if passed else 1)


if __name__ == "__main__":
    main()
