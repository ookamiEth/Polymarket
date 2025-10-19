#!/usr/bin/env python3
"""
Performance Benchmark: Original vs Optimized Tardis Downloader

Measures execution time, memory usage, and throughput for both implementations.
Uses synthetic fixture data to simulate real API responses.
"""

import sys
import os
import time
import tracemalloc
from typing import Dict, List, Tuple
import asyncio

# Add parent directory to path to import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import both versions
from tardis_download import _parse_quote_message, _validate_quote_data
import tardis_download as original

# Import optimized version
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
from optimized_tardis_download import _fetch_single_batch_optimized, _empty_dataframe
import optimized_tardis_download as optimized

import msgspec.json
import polars as pl


class BenchmarkResult:
    def __init__(self, name: str, num_rows: int, exec_time: float, memory_mb: float, throughput: float):
        self.name = name
        self.num_rows = num_rows
        self.exec_time = exec_time
        self.memory_mb = memory_mb
        self.throughput = throughput


def load_fixture(filename: str) -> str:
    """Load fixture file contents."""
    filepath = os.path.join(os.path.dirname(__file__), 'fixtures', filename)
    with open(filepath, 'r') as f:
        return f.read()


def simulate_response(text: str):
    """Create a mock response object."""
    class MockResponse:
        def __init__(self, text: str):
            self.text = text
            self.content = text.encode('utf-8')
            self.status_code = 200
    return MockResponse(text)


def benchmark_original(response_text: str) -> Tuple[int, float, float]:
    """
    Benchmark the original implementation (with Python loops).
    Simulates the core processing logic from _fetch_single_batch.
    """
    tracemalloc.start()
    start_time = time.perf_counter()

    # Simulate original processing
    lines = response_text.strip().split('\n') if response_text.strip() else []
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

    # Convert to DataFrame (as done in original)
    if batch_rows:
        df = pl.DataFrame(batch_rows)
    else:
        df = pl.DataFrame()

    elapsed = time.perf_counter() - start_time
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    memory_mb = peak / (1024 * 1024)
    return len(df), elapsed, memory_mb


def benchmark_optimized(response_text: str) -> Tuple[int, float, float]:
    """
    Benchmark the optimized implementation (vectorized Polars).
    Simulates the core processing logic from _fetch_single_batch_optimized.
    """
    import io

    tracemalloc.start()
    start_time = time.perf_counter()

    if not response_text.strip():
        df = _empty_dataframe()
    else:
        try:
            # Parse all JSON at once
            df = pl.read_ndjson(io.StringIO(response_text))

            if len(df) > 0:
                # Vectorized filtering
                df = df.filter(pl.col('type').str.contains('quote|book_snapshot'))

                if len(df) > 0:
                    # Vectorized symbol parsing
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
        except Exception:
            df = _empty_dataframe()

    elapsed = time.perf_counter() - start_time
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    memory_mb = peak / (1024 * 1024)
    return len(df), elapsed, memory_mb


def run_benchmark_suite():
    """Run complete benchmark suite on all fixtures."""
    fixtures = [
        ('small_1k.json', '1K'),
        ('medium_10k.json', '10K'),
        ('large_100k.json', '100K'),
        ('with_book_5k.json', '5K w/book'),
        ('edge_cases.json', 'Edge Cases'),
    ]

    print("=" * 80)
    print("PERFORMANCE BENCHMARK: Original vs Optimized")
    print("=" * 80)
    print()

    all_results = []

    for fixture_file, label in fixtures:
        print(f"Testing: {label} ({fixture_file})")
        print("-" * 80)

        try:
            response_text = load_fixture(fixture_file)
            num_lines = len(response_text.strip().split('\n')) if response_text.strip() else 0
            print(f"Input: {num_lines:,} messages")

            # Warmup runs
            benchmark_original(response_text[:1000])  # Small warmup
            benchmark_optimized(response_text[:1000])

            # Run original
            print("  Running original (with Python loops)...", end='', flush=True)
            orig_rows, orig_time, orig_mem = benchmark_original(response_text)
            orig_throughput = orig_rows / max(orig_time, 0.001)
            print(f" ✓")

            # Run optimized
            print("  Running optimized (vectorized)...", end='', flush=True)
            opt_rows, opt_time, opt_mem = benchmark_optimized(response_text)
            opt_throughput = opt_rows / max(opt_time, 0.001)
            print(f" ✓")

            # Calculate improvements
            speedup = orig_time / max(opt_time, 0.001)
            mem_reduction = ((orig_mem - opt_mem) / max(orig_mem, 0.001)) * 100

            print()
            print(f"Results:")
            print(f"  Original:  {orig_time:.3f}s | {orig_mem:.1f} MB | {orig_throughput:,.0f} rows/s | {orig_rows:,} rows")
            print(f"  Optimized: {opt_time:.3f}s | {opt_mem:.1f} MB | {opt_throughput:,.0f} rows/s | {opt_rows:,} rows")
            print(f"  Improvement: {speedup:.1f}x faster | {mem_reduction:+.1f}% memory | {opt_throughput/max(orig_throughput, 1):.1f}x throughput")
            print()

            all_results.append({
                'fixture': label,
                'orig_time': orig_time,
                'opt_time': opt_time,
                'speedup': speedup,
                'orig_mem': orig_mem,
                'opt_mem': opt_mem,
                'orig_throughput': orig_throughput,
                'opt_throughput': opt_throughput,
                'rows': orig_rows,
            })

        except Exception as e:
            print(f"  ❌ Error: {e}")
            import traceback
            traceback.print_exc()
            print()

    # Summary table
    print("=" * 80)
    print("SUMMARY TABLE")
    print("=" * 80)
    print()
    print(f"{'Dataset':<15} {'Original':<12} {'Optimized':<12} {'Speedup':<10} {'Rows':<12}")
    print("-" * 80)
    for r in all_results:
        print(f"{r['fixture']:<15} {r['orig_time']:.3f}s{'':<6} {r['opt_time']:.3f}s{'':<6} {r['speedup']:.1f}x{'':<6} {r['rows']:>10,}")

    print()
    avg_speedup = sum(r['speedup'] for r in all_results) / len(all_results) if all_results else 0
    print(f"Average Speedup: {avg_speedup:.1f}x")
    print()

    # Check if optimized version is consistently faster
    all_faster = all(r['speedup'] > 1.0 for r in all_results)
    if all_faster:
        print("✅ SUCCESS: Optimized version is consistently faster across all datasets!")
    else:
        print("⚠️  WARNING: Optimized version is not faster in all cases")

    print()
    print("=" * 80)

    return all_results


def main():
    """Main entry point."""
    print()
    results = run_benchmark_suite()

    # Save results to file
    output_file = 'test/benchmark_results.txt'
    with open(output_file, 'w') as f:
        f.write("BENCHMARK RESULTS\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"{'Dataset':<15} {'Original':<12} {'Optimized':<12} {'Speedup':<10} {'Rows':<12}\n")
        f.write("-" * 80 + "\n")
        for r in results:
            f.write(f"{r['fixture']:<15} {r['orig_time']:.3f}s{'':<6} {r['opt_time']:.3f}s{'':<6} {r['speedup']:.1f}x{'':<6} {r['rows']:>10,}\n")
        avg_speedup = sum(r['speedup'] for r in results) / len(results) if results else 0
        f.write(f"\nAverage Speedup: {avg_speedup:.1f}x\n")

    print(f"Results saved to: {output_file}")
    print()


if __name__ == "__main__":
    main()
