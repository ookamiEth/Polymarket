#!/usr/bin/env python3
"""
Simple XGBoost baseline model using only production_backtest_results data.
Designed to work within memory constraints.
"""

import argparse
import gc
import logging
from datetime import date
from pathlib import Path

import polars as pl
import xgboost as xgb

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Paths
MODEL_DIR = Path(__file__).parent.parent
BASELINE_FILE = MODEL_DIR / "results/production_backtest_results.parquet"
OUTPUT_DIR = MODEL_DIR / "results"

# Simple feature list from baseline file only
FEATURE_COLS = [
    "K", "S", "T_years", "time_remaining",
    "iv_staleness_seconds", "sigma_mid",
    "d2_bid", "d2_ask", "d2_mid",
    "prob_bid", "prob_ask", "prob_mid",
    "price_bid", "price_ask", "price_mid",
    "blended_supply_apr", "r",
]


def prepare_simple_features(
    start_date: date,
    end_date: date,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """
    Prepare simple features from baseline file only.
    Returns train and validation DataFrames.
    """
    logger.info(f"Loading data from {start_date} to {end_date}")

    # Load only necessary columns
    df = (
        pl.scan_parquet(str(BASELINE_FILE))
        .filter(
            (pl.col("date") >= start_date) &
            (pl.col("date") <= end_date)
        )
        .select(
            FEATURE_COLS + ["outcome", "timestamp"]
        )
        .with_columns([
            # Calculate residual
            (pl.col("outcome") - pl.col("prob_mid")).alias("residual"),
            # Calculate moneyness
            (pl.col("K") / pl.col("S")).alias("moneyness"),
        ])
        .drop_nulls()
        .collect()
    )

    logger.info(f"Loaded {len(df):,} rows")

    # Add moneyness to feature list
    feature_cols = FEATURE_COLS + ["moneyness"]

    # Split 80/20
    n_rows = len(df)
    train_rows = int(n_rows * 0.8)

    # Sort by timestamp for temporal split
    df = df.sort("timestamp")

    train_df = df.head(train_rows)
    val_df = df.tail(n_rows - train_rows)

    logger.info(f"Train: {len(train_df):,} rows, Val: {len(val_df):,} rows")

    return train_df, val_df


def train_simple_xgboost(
    train_df: pl.DataFrame,
    val_df: pl.DataFrame,
) -> xgb.Booster:
    """Train simple XGBoost model."""

    # Extract features and target
    feature_cols = FEATURE_COLS + ["moneyness"]

    X_train = train_df.select(feature_cols).to_numpy()
    y_train = train_df.select("residual").to_numpy().ravel()

    X_val = val_df.select(feature_cols).to_numpy()
    y_val = val_df.select("residual").to_numpy().ravel()

    # Create DMatrix
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dval = xgb.DMatrix(X_val, label=y_val)

    # Clean up arrays
    del X_train, y_train, X_val, y_val
    gc.collect()

    # Simple parameters
    params = {
        "objective": "reg:squarederror",
        "eval_metric": "rmse",
        "tree_method": "hist",  # Fast histogram method
        "max_depth": 6,
        "learning_rate": 0.05,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "nthread": 4,
        "seed": 42,
    }

    logger.info("Training XGBoost model...")
    evals = [(dtrain, "train"), (dval, "val")]

    model = xgb.train(
        params,
        dtrain,
        num_boost_round=100,
        evals=evals,
        early_stopping_rounds=10,
        verbose_eval=10,
    )

    logger.info(f"Best iteration: {model.best_iteration}")
    logger.info(f"Best score: {model.best_score:.6f}")

    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--pilot",
        action="store_true",
        help="Run pilot on Oct 2023",
    )
    parser.add_argument(
        "--start-date",
        type=str,
        default="2023-10-01",
        help="Start date (YYYY-MM-DD)",
    )
    parser.add_argument(
        "--end-date",
        type=str,
        default="2023-10-31",
        help="End date (YYYY-MM-DD)",
    )
    args = parser.parse_args()

    if args.pilot:
        start_date = date(2023, 10, 1)
        end_date = date(2023, 10, 31)
    else:
        start_date = date.fromisoformat(args.start_date)
        end_date = date.fromisoformat(args.end_date)

    logger.info("=" * 80)
    logger.info("SIMPLE XGBOOST BASELINE")
    logger.info("=" * 80)
    logger.info(f"Date range: {start_date} to {end_date}")
    logger.info(f"Using {len(FEATURE_COLS) + 1} features")

    # Prepare features
    train_df, val_df = prepare_simple_features(start_date, end_date)

    # Train model
    model = train_simple_xgboost(train_df, val_df)

    # Save model
    model_file = OUTPUT_DIR / f"xgboost_simple_{start_date}_{end_date}.json"
    model.save_model(str(model_file))
    logger.info(f"Model saved to {model_file}")

    # Clean up
    del train_df, val_df, model
    gc.collect()

    logger.info("=" * 80)
    logger.info("TRAINING COMPLETE")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()