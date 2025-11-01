#!/usr/bin/env python3
"""
Memory profiling script for XGBoost training pipeline.

Identifies memory hotspots and inefficiencies in the data loading,
join operations, and streaming processes.
"""

from __future__ import annotations

import gc
import logging
import sys
from pathlib import Path

import polars as pl
import psutil

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

# File paths
MODEL_DIR = Path(__file__).parent.parent
BASELINE_FILE = MODEL_DIR / "results/production_backtest_results.parquet"
RV_FILE = MODEL_DIR / "results/realized_volatility_1s.parquet"
MICROSTRUCTURE_FILE = MODEL_DIR / "results/microstructure_features.parquet"
ADVANCED_FILE = MODEL_DIR / "results/advanced_features.parquet"


def log_memory(label: str) -> float:
    """Log current memory usage and return in GB."""
    process = psutil.Process()
    mem_info = process.memory_info()
    mem_gb = mem_info.rss / (1024**3)
    logger.info(f"[MEMORY] {label}: {mem_gb:.2f} GB")
    return mem_gb


def get_file_size_gb(file_path: Path) -> float:
    """Get file size in GB."""
    if not file_path.exists():
        return 0.0
    return file_path.stat().st_size / (1024**3)


def analyze_schema_resolution() -> None:
    """Test schema resolution overhead."""
    logger.info("=" * 80)
    logger.info("SCHEMA RESOLUTION ANALYSIS")
    logger.info("=" * 80)

    # Baseline file
    logger.info("\n1. Baseline file schema access:")
    mem_start = log_memory("Before schema access")

    baseline_lf = pl.scan_parquet(BASELINE_FILE)

    # Method 1: Direct .columns (triggers warning)
    logger.info("  Testing: baseline_lf.columns (triggers schema resolution)")
    try:
        cols1 = baseline_lf.columns
        logger.info(f"  Columns count: {len(cols1)}")
    except Exception as e:
        logger.error(f"  Error: {e}")

    mem_after_cols = log_memory("After .columns access")
    logger.info(f"  Memory increase: {(mem_after_cols - mem_start) * 1024:.1f} MB")

    # Method 2: Recommended collect_schema()
    logger.info("\n  Testing: baseline_lf.collect_schema().names()")
    mem_before_schema = log_memory("Before collect_schema()")
    schema = baseline_lf.collect_schema()
    cols2 = schema.names()
    logger.info(f"  Columns count: {len(cols2)}")
    mem_after_schema = log_memory("After collect_schema()")
    logger.info(f"  Memory increase: {(mem_after_schema - mem_before_schema) * 1024:.1f} MB")

    # RV file
    logger.info("\n2. RV file schema access:")
    rv_lf = pl.scan_parquet(RV_FILE)
    schema_rv = rv_lf.collect_schema()
    rv_cols = [c for c in schema_rv.names() if c != "timestamp_seconds"]
    logger.info(f"  RV features: {len(rv_cols)}")

    # Microstructure file
    logger.info("\n3. Microstructure file schema access:")
    micro_lf = pl.scan_parquet(MICROSTRUCTURE_FILE)
    schema_micro = micro_lf.collect_schema()
    micro_cols = [c for c in schema_micro.names() if c != "timestamp_seconds"]
    logger.info(f"  Microstructure features: {len(micro_cols)}")

    # Advanced file
    logger.info("\n4. Advanced file schema access:")
    adv_lf = pl.scan_parquet(ADVANCED_FILE)
    schema_adv = adv_lf.collect_schema()
    adv_cols = [c for c in schema_adv.names() if c not in ["timestamp", "timestamp_seconds"]]
    logger.info(f"  Advanced features: {len(adv_cols)}")

    log_memory("After all schema accesses")


def analyze_join_operations() -> None:
    """Analyze memory impact of join operations."""
    logger.info("\n" + "=" * 80)
    logger.info("JOIN OPERATIONS ANALYSIS")
    logger.info("=" * 80)

    # Start with baseline (October 2023 only for speed)
    from datetime import date

    start_date = date(2023, 10, 1)
    end_date = date(2023, 10, 31)

    logger.info(f"\nFiltering to October 2023: {start_date} to {end_date}")
    mem_start = log_memory("Initial")

    # Load baseline
    logger.info("\n1. Loading baseline (lazy)...")
    baseline_lf = pl.scan_parquet(BASELINE_FILE)
    baseline_lf = baseline_lf.filter((pl.col("date") >= start_date) & (pl.col("date") <= end_date))
    n_baseline = baseline_lf.select(pl.len()).collect().item()
    logger.info(f"  Baseline rows: {n_baseline:,}")
    mem_after_baseline = log_memory("After baseline filter")
    logger.info(f"  Memory increase: {(mem_after_baseline - mem_start) * 1024:.1f} MB")

    # Add moneyness and residual
    logger.info("\n2. Adding computed columns (moneyness, residual)...")
    baseline_lf = baseline_lf.with_columns([
        (pl.col("outcome") - pl.col("prob_mid")).alias("residual"),
        (pl.col("S") / pl.col("K") - 1.0).alias("moneyness"),
    ])
    mem_after_cols = log_memory("After adding columns (lazy)")
    logger.info(f"  Memory increase: {(mem_after_cols - mem_after_baseline) * 1024:.1f} MB")

    # Load RV features
    logger.info("\n3. Loading RV features (lazy)...")
    rv_lf = pl.scan_parquet(RV_FILE)

    # Get timestamp range for filtering
    import datetime
    start_ts = int(datetime.datetime(2023, 10, 1).timestamp())
    end_ts = int(datetime.datetime(2023, 11, 1).timestamp())

    logger.info(f"  Filtering RV to timestamp range: {start_ts} to {end_ts}")
    rv_lf = rv_lf.filter((pl.col("timestamp_seconds") >= start_ts) & (pl.col("timestamp_seconds") < end_ts))
    n_rv = rv_lf.select(pl.len()).collect().item()
    logger.info(f"  RV rows after filter: {n_rv:,}")
    mem_after_rv = log_memory("After RV filter")
    logger.info(f"  Memory increase: {(mem_after_rv - mem_after_cols) * 1024:.1f} MB")

    # Join RV
    logger.info("\n4. Joining RV features (lazy)...")
    schema_rv = rv_lf.collect_schema()
    rv_features = [c for c in schema_rv.names() if c != "timestamp_seconds"]
    logger.info(f"  RV features to join: {len(rv_features)}")

    joined_lf = baseline_lf.join(
        rv_lf.select(["timestamp_seconds"] + rv_features),
        left_on="timestamp",
        right_on="timestamp_seconds",
        how="left"
    )
    mem_after_join_rv = log_memory("After RV join (lazy)")
    logger.info(f"  Memory increase: {(mem_after_join_rv - mem_after_rv) * 1024:.1f} MB")

    # Check result count
    n_after_rv_join = joined_lf.select(pl.len()).collect().item()
    logger.info(f"  Rows after RV join: {n_after_rv_join:,}")

    # Load microstructure features
    logger.info("\n5. Joining microstructure features (lazy)...")
    micro_lf = pl.scan_parquet(MICROSTRUCTURE_FILE)
    micro_lf = micro_lf.filter((pl.col("timestamp_seconds") >= start_ts) & (pl.col("timestamp_seconds") < end_ts))
    n_micro = micro_lf.select(pl.len()).collect().item()
    logger.info(f"  Microstructure rows after filter: {n_micro:,}")

    schema_micro = micro_lf.collect_schema()
    micro_features = [c for c in schema_micro.names() if c != "timestamp_seconds"]
    logger.info(f"  Microstructure features to join: {len(micro_features)}")

    joined_lf = joined_lf.join(
        micro_lf.select(["timestamp_seconds"] + micro_features),
        left_on="timestamp",
        right_on="timestamp_seconds",
        how="left"
    )
    mem_after_join_micro = log_memory("After microstructure join (lazy)")
    logger.info(f"  Memory increase: {(mem_after_join_micro - mem_after_join_rv) * 1024:.1f} MB")

    # Load advanced features
    logger.info("\n6. Joining advanced features (lazy)...")
    adv_lf = pl.scan_parquet(ADVANCED_FILE)
    adv_lf = adv_lf.filter((pl.col("timestamp") >= start_ts) & (pl.col("timestamp") < end_ts))
    n_adv = adv_lf.select(pl.len()).collect().item()
    logger.info(f"  Advanced rows after filter: {n_adv:,}")

    schema_adv = adv_lf.collect_schema()
    adv_features = [c for c in schema_adv.names() if c not in ["timestamp", "timestamp_seconds"]]
    logger.info(f"  Advanced features to join: {len(adv_features)}")

    joined_lf = joined_lf.join(
        adv_lf.select(["timestamp"] + adv_features),
        on="timestamp",
        how="left"
    )
    mem_after_join_adv = log_memory("After advanced join (lazy)")
    logger.info(f"  Memory increase: {(mem_after_join_adv - mem_after_join_micro) * 1024:.1f} MB")

    # Final count
    n_final = joined_lf.select(pl.len()).collect().item()
    logger.info(f"  Rows after all joins: {n_final:,}")

    logger.info("\n7. Collecting small sample to test materialization...")
    sample_size = 10000
    sample_df = joined_lf.head(sample_size).collect()
    mem_after_sample = log_memory("After collecting 10k sample")
    logger.info(f"  Sample size: {len(sample_df):,} rows")
    logger.info(f"  Sample columns: {len(sample_df.columns)}")
    logger.info(f"  Memory increase: {(mem_after_sample - mem_after_join_adv) * 1024:.1f} MB")

    # Check for nulls in sample
    null_cols = [col for col in sample_df.columns if sample_df[col].null_count() > 0]
    logger.info(f"  Columns with nulls in sample: {len(null_cols)}")
    if null_cols:
        logger.info(f"  Example null columns: {null_cols[:5]}")

    del sample_df
    gc.collect()


def analyze_casting_overhead() -> None:
    """Analyze memory impact of type casting operations."""
    logger.info("\n" + "=" * 80)
    logger.info("CASTING OPERATIONS ANALYSIS")
    logger.info("=" * 80)

    # Load small sample
    from datetime import date

    start_date = date(2023, 10, 1)
    end_date = date(2023, 10, 2)  # Just 1 day

    logger.info(f"\nLoading 1 day of data: {start_date}")
    baseline_lf = pl.scan_parquet(BASELINE_FILE)
    baseline_lf = baseline_lf.filter((pl.col("date") >= start_date) & (pl.col("date") <= end_date))

    logger.info("\n1. Collecting without casting...")
    mem_before = log_memory("Before collect (no cast)")
    df_no_cast = baseline_lf.collect()
    mem_after_no_cast = log_memory("After collect (no cast)")
    logger.info(f"  Rows: {len(df_no_cast):,}")
    logger.info(f"  Memory: {(mem_after_no_cast - mem_before) * 1024:.1f} MB")

    # Check dtypes
    dtypes = df_no_cast.dtypes
    float64_cols = [col for col, dtype in zip(df_no_cast.columns, dtypes) if str(dtype) == "Float64"]
    logger.info(f"  Float64 columns: {len(float64_cols)}")

    logger.info("\n2. Collecting with Float32 cast...")
    mem_before_cast = log_memory("Before collect (with cast)")
    df_cast = baseline_lf.collect()
    # Cast only numeric columns
    numeric_cols = [col for col, dtype in zip(df_cast.columns, df_cast.dtypes) if str(dtype) in ["Float64", "Int64"]]
    logger.info(f"  Numeric columns to cast: {len(numeric_cols)}")

    # Cast to Float32
    df_cast_f32 = df_cast.select([
        pl.col(col).cast(pl.Float32) if col in numeric_cols else pl.col(col)
        for col in df_cast.columns
    ])
    mem_after_cast = log_memory("After Float32 cast")
    logger.info(f"  Memory after cast: {(mem_after_cast - mem_before_cast) * 1024:.1f} MB")
    logger.info(f"  Memory reduction: {(mem_after_no_cast - mem_after_cast) * 1024:.1f} MB")

    del df_no_cast, df_cast, df_cast_f32
    gc.collect()


def analyze_streaming_sink() -> None:
    """Analyze streaming sink behavior."""
    logger.info("\n" + "=" * 80)
    logger.info("STREAMING SINK ANALYSIS")
    logger.info("=" * 80)

    from datetime import date
    import tempfile

    start_date = date(2023, 10, 1)
    end_date = date(2023, 10, 7)  # 1 week

    logger.info(f"\nTesting streaming sink with 1 week of data: {start_date} to {end_date}")

    baseline_lf = pl.scan_parquet(BASELINE_FILE)
    baseline_lf = baseline_lf.filter((pl.col("date") >= start_date) & (pl.col("date") <= end_date))
    n_rows = baseline_lf.select(pl.len()).collect().item()
    logger.info(f"  Rows to write: {n_rows:,}")

    # Test 1: Regular collect + write
    logger.info("\n1. Testing collect() + write_parquet()...")
    mem_before_eager = log_memory("Before eager collect")
    df_eager = baseline_lf.collect()
    mem_after_eager = log_memory("After eager collect")
    logger.info(f"  Memory increase: {(mem_after_eager - mem_before_eager) * 1024:.1f} MB")

    with tempfile.NamedTemporaryFile(suffix=".parquet", delete=False) as tmp:
        tmp_path_eager = tmp.name

    df_eager.write_parquet(tmp_path_eager, compression="snappy")
    mem_after_write = log_memory("After write_parquet")
    file_size = Path(tmp_path_eager).stat().st_size / (1024**2)
    logger.info(f"  File size: {file_size:.1f} MB")

    del df_eager
    gc.collect()
    Path(tmp_path_eager).unlink()

    # Test 2: Streaming sink
    logger.info("\n2. Testing sink_parquet(streaming=True)...")
    mem_before_stream = log_memory("Before streaming sink")

    with tempfile.NamedTemporaryFile(suffix=".parquet", delete=False) as tmp:
        tmp_path_stream = tmp.name

    baseline_lf.sink_parquet(tmp_path_stream, compression="snappy")
    mem_after_stream = log_memory("After streaming sink")
    logger.info(f"  Memory increase: {(mem_after_stream - mem_before_stream) * 1024:.1f} MB")

    file_size_stream = Path(tmp_path_stream).stat().st_size / (1024**2)
    logger.info(f"  File size: {file_size_stream:.1f} MB")

    Path(tmp_path_stream).unlink()

    logger.info("\n3. Memory comparison:")
    eager_mem = (mem_after_eager - mem_before_eager) * 1024
    stream_mem = (mem_after_stream - mem_before_stream) * 1024
    logger.info(f"  Eager (collect + write):  {eager_mem:.1f} MB")
    logger.info(f"  Streaming (sink):         {stream_mem:.1f} MB")
    logger.info(f"  Memory saved:             {eager_mem - stream_mem:.1f} MB ({(1 - stream_mem/eager_mem) * 100:.1f}%)")


def main() -> None:
    """Run all memory profiling analyses."""
    logger.info("=" * 80)
    logger.info("XGBOOST MEMORY PROFILING")
    logger.info("=" * 80)

    # Log initial state
    log_memory("Initial")

    # File sizes
    logger.info("\nFile sizes:")
    logger.info(f"  Baseline:       {get_file_size_gb(BASELINE_FILE):.2f} GB")
    logger.info(f"  RV:             {get_file_size_gb(RV_FILE):.2f} GB")
    logger.info(f"  Microstructure: {get_file_size_gb(MICROSTRUCTURE_FILE):.2f} GB")
    logger.info(f"  Advanced:       {get_file_size_gb(ADVANCED_FILE):.2f} GB")

    # Run analyses
    analyze_schema_resolution()
    gc.collect()

    analyze_join_operations()
    gc.collect()

    analyze_casting_overhead()
    gc.collect()

    analyze_streaming_sink()
    gc.collect()

    # Final summary
    logger.info("\n" + "=" * 80)
    logger.info("PROFILING COMPLETE")
    logger.info("=" * 80)
    log_memory("Final")


if __name__ == "__main__":
    main()
