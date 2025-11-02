#!/usr/bin/env python3
"""
Generate stratified test fixtures for multi-horizon ML pipeline testing.

Creates 100K-row test datasets sampled from production data with:
- Stratified sampling by bucket (near/mid/far)
- Temporal coverage for walk-forward validation
- Train/val/test splits (60K/20K/20K)
- Preserved feature distributions

Usage:
    uv run python generate_test_data.py
"""

import logging
from datetime import date
from pathlib import Path

import polars as pl

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# Constants
RESULTS_DIR = Path("/home/ubuntu/Polymarket/research/model/results")
FIXTURES_DIR = Path(__file__).parent
TEST_SAMPLES = 100_000
TRAIN_RATIO = 0.60
VAL_RATIO = 0.20
TEST_RATIO = 0.20

# Bucket definitions (from config)
BUCKETS = {
    "near": {"time_min": 0, "time_max": 300, "target_pct": 0.33},
    "mid": {"time_min": 300, "time_max": 600, "target_pct": 0.33},
    "far": {"time_min": 600, "time_max": 900, "target_pct": 0.34},
}


def sample_stratified_by_bucket(df: pl.LazyFrame, n_samples: int, seed: int = 42) -> pl.DataFrame:
    """
    Sample data stratified by time_remaining bucket.

    Args:
        df: Lazy DataFrame with full dataset
        n_samples: Total samples to draw
        seed: Random seed for reproducibility

    Returns:
        Stratified sample DataFrame
    """
    logger.info(f"Sampling {n_samples:,} rows stratified by bucket...")

    samples_per_bucket = {
        "near": int(n_samples * BUCKETS["near"]["target_pct"]),
        "mid": int(n_samples * BUCKETS["mid"]["target_pct"]),
        "far": int(n_samples * BUCKETS["far"]["target_pct"]),
    }

    sampled_dfs = []

    for bucket_name, bucket_config in BUCKETS.items():
        time_min = bucket_config["time_min"]
        time_max = bucket_config["time_max"]
        n_bucket = samples_per_bucket[bucket_name]

        logger.info(f"  {bucket_name.upper()}: {time_min}-{time_max}s, {n_bucket:,} samples")

        # Filter by bucket and sample
        bucket_df = (
            df.filter((pl.col("time_remaining") >= time_min) & (pl.col("time_remaining") < time_max))
            .collect()
            .sample(n=n_bucket, seed=seed, shuffle=True)
        )

        sampled_dfs.append(bucket_df)

    # Combine all buckets
    combined = pl.concat(sampled_dfs)
    logger.info(f"  Total sampled: {len(combined):,} rows")

    return combined


def ensure_temporal_coverage(df: pl.DataFrame, min_months: int = 8) -> pl.DataFrame:
    """
    Ensure dataset has sufficient temporal coverage for walk-forward testing.

    Args:
        df: Sampled DataFrame
        min_months: Minimum months required for walk-forward

    Returns:
        DataFrame with validated temporal coverage
    """
    logger.info("Validating temporal coverage...")

    min_date = df.select(pl.col("date").min()).item()
    max_date = df.select(pl.col("date").max()).item()

    if isinstance(min_date, str):
        min_date = date.fromisoformat(min_date)
    if isinstance(max_date, str):
        max_date = date.fromisoformat(max_date)

    months_coverage = (max_date.year - min_date.year) * 12 + (max_date.month - min_date.month)

    logger.info(f"  Date range: {min_date} to {max_date}")
    logger.info(f"  Coverage: {months_coverage} months")

    if months_coverage < min_months:
        logger.warning(f"  ⚠️  Coverage {months_coverage} months < required {min_months} months")
        logger.warning("  Walk-forward tests may have fewer windows")
    else:
        logger.info("  ✓ Coverage sufficient for walk-forward testing")

    return df


def create_train_val_test_splits(
    df: pl.DataFrame,
) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    """
    Create temporal train/val/test splits.

    Args:
        df: Full sampled DataFrame

    Returns:
        Tuple of (train_df, val_df, test_df)
    """
    logger.info("Creating train/val/test splits...")

    # Sort by date for temporal split
    df = df.sort("date")

    n_total = len(df)
    n_train = int(n_total * TRAIN_RATIO)
    n_val = int(n_total * VAL_RATIO)

    # Temporal split (train earliest, test latest)
    train_df = df.head(n_train)
    val_df = df.slice(n_train, n_val)
    test_df = df.slice(n_train + n_val, n_total - n_train - n_val)

    logger.info(f"  Train: {len(train_df):,} rows")
    logger.info(f"  Val:   {len(val_df):,} rows")
    logger.info(f"  Test:  {len(test_df):,} rows")

    # Verify no overlap
    train_max_date = train_df.select(pl.col("date").max()).item()
    val_min_date = val_df.select(pl.col("date").min()).item()
    val_max_date = val_df.select(pl.col("date").max()).item()
    test_min_date = test_df.select(pl.col("date").min()).item()

    logger.info(f"  Train dates: ... to {train_max_date}")
    logger.info(f"  Val dates:   {val_min_date} to {val_max_date}")
    logger.info(f"  Test dates:  {test_min_date} to ...")

    return train_df, val_df, test_df


def validate_test_data(df: pl.DataFrame, name: str) -> None:
    """
    Validate test data quality.

    Args:
        df: DataFrame to validate
        name: Dataset name for logging
    """
    logger.info(f"Validating {name}...")

    # Check for required columns
    required_cols = ["outcome", "prob_mid", "time_remaining", "date"]
    missing_cols = [col for col in required_cols if col not in df.columns]

    if missing_cols:
        logger.error(f"  ❌ Missing required columns: {missing_cols}")
        raise ValueError(f"Missing columns: {missing_cols}")

    # Check outcome range
    outcome_min = df.select(pl.col("outcome").min()).item()
    outcome_max = df.select(pl.col("outcome").max()).item()

    if outcome_min < 0 or outcome_max > 1:
        logger.error(f"  ❌ Outcome range invalid: [{outcome_min}, {outcome_max}]")
        raise ValueError("Outcome must be in [0, 1]")

    # Check prob_mid range
    prob_min = df.select(pl.col("prob_mid").min()).item()
    prob_max = df.select(pl.col("prob_mid").max()).item()

    if prob_min < 0 or prob_max > 1:
        logger.error(f"  ❌ prob_mid range invalid: [{prob_min}, {prob_max}]")
        raise ValueError("prob_mid must be in [0, 1]")

    # Check time_remaining range
    time_min = df.select(pl.col("time_remaining").min()).item()
    time_max = df.select(pl.col("time_remaining").max()).item()

    if time_min < 0 or time_max > 900:
        logger.error(f"  ❌ time_remaining range invalid: [{time_min}, {time_max}]")
        raise ValueError("time_remaining must be in [0, 900]")

    # Check for nulls
    null_counts = df.null_count()
    total_nulls = null_counts.select(pl.sum_horizontal(pl.all())).item()

    if total_nulls > 0:
        logger.warning(f"  ⚠️  Found {total_nulls} null values")

    # Check bucket distribution
    near_pct = len(df.filter(pl.col("time_remaining") < 300)) / len(df) * 100
    mid_pct = len(df.filter((pl.col("time_remaining") >= 300) & (pl.col("time_remaining") < 600))) / len(df) * 100
    far_pct = len(df.filter(pl.col("time_remaining") >= 600)) / len(df) * 100

    logger.info("  Bucket distribution:")
    logger.info(f"    Near (0-300s):   {near_pct:.1f}%")
    logger.info(f"    Mid (300-600s):  {mid_pct:.1f}%")
    logger.info(f"    Far (600-900s):  {far_pct:.1f}%")

    logger.info(f"  ✓ {name} validated successfully")


def main() -> None:
    """Generate test fixtures."""
    logger.info("=" * 80)
    logger.info("GENERATING TEST FIXTURES FOR MULTI-HORIZON PIPELINE")
    logger.info("=" * 80)
    logger.info(f"Target samples: {TEST_SAMPLES:,}")
    logger.info(f"Output directory: {FIXTURES_DIR}")
    logger.info("")

    # Load full training dataset
    train_file = RESULTS_DIR / "train_features_lgb.parquet"

    if not train_file.exists():
        logger.error(f"Training data not found: {train_file}")
        raise FileNotFoundError(f"Missing: {train_file}")

    logger.info(f"Loading full training data: {train_file}")
    df_lazy = pl.scan_parquet(train_file)

    # Sample stratified by bucket
    df_sampled = sample_stratified_by_bucket(df_lazy, TEST_SAMPLES)

    # Ensure temporal coverage
    df_sampled = ensure_temporal_coverage(df_sampled, min_months=8)

    # Create train/val/test splits
    train_df, val_df, test_df = create_train_val_test_splits(df_sampled)

    # Validate each split
    validate_test_data(train_df, "train")
    validate_test_data(val_df, "val")
    validate_test_data(test_df, "test")

    # Save test fixtures
    logger.info("")
    logger.info("Saving test fixtures...")

    train_output = FIXTURES_DIR / "test_train.parquet"
    val_output = FIXTURES_DIR / "test_val.parquet"
    test_output = FIXTURES_DIR / "test_test.parquet"

    train_df.write_parquet(train_output, compression="snappy", statistics=True)
    logger.info(f"  ✓ Saved: {train_output} ({len(train_df):,} rows)")

    val_df.write_parquet(val_output, compression="snappy", statistics=True)
    logger.info(f"  ✓ Saved: {val_output} ({len(val_df):,} rows)")

    test_df.write_parquet(test_output, compression="snappy", statistics=True)
    logger.info(f"  ✓ Saved: {test_output} ({len(test_df):,} rows)")

    # Summary
    total_size_mb = (train_output.stat().st_size + val_output.stat().st_size + test_output.stat().st_size) / (1024**2)

    logger.info("")
    logger.info("=" * 80)
    logger.info("TEST FIXTURES GENERATED SUCCESSFULLY")
    logger.info("=" * 80)
    logger.info(f"Total rows: {len(train_df) + len(val_df) + len(test_df):,}")
    logger.info(f"Total size: {total_size_mb:.2f} MB")
    logger.info(f"Location: {FIXTURES_DIR}/")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
