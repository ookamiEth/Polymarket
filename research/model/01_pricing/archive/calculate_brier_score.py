#!/usr/bin/env python3
"""
Calculate and compare Brier scores for the original and quick wins implementations.
"""

import polars as pl
import numpy as np
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

def calculate_brier_score(predictions, outcomes):
    """Calculate Brier score, handling NaN values properly."""
    # Remove any NaN or infinite values
    mask = np.isfinite(predictions) & np.isfinite(outcomes)
    clean_pred = predictions[mask]
    clean_out = outcomes[mask]

    if len(clean_pred) == 0:
        return np.nan

    # Brier score = mean((prediction - outcome)^2)
    brier = np.mean((clean_pred - clean_out) ** 2)
    return brier

def analyze_results(file_path: Path, name: str):
    """Analyze results from a backtest file."""
    logger.info(f"\nAnalyzing {name}...")
    logger.info(f"File: {file_path}")

    # Load results
    df = pl.read_parquet(file_path)

    # Check which price column exists
    price_col = "price"
    if "price" not in df.columns:
        if "price_mid" in df.columns:
            price_col = "price_mid"
        elif "price_ask" in df.columns:
            price_col = "price_ask"
        else:
            logger.error(f"No price column found. Available columns: {df.columns[:10]}")
            return None

    # Get valid predictions (non-null price and outcome)
    valid = df.filter(
        pl.col(price_col).is_not_null() &
        pl.col("outcome").is_not_null()
    )

    # Check for NaN/infinite values
    predictions = valid[price_col].to_numpy()
    outcomes = valid["outcome"].to_numpy()

    # Count problematic values
    nan_count = np.isnan(predictions).sum()
    inf_count = np.isinf(predictions).sum()

    logger.info(f"Total predictions: {len(df):,}")
    logger.info(f"Valid (non-null): {len(valid):,}")
    logger.info(f"NaN predictions: {nan_count:,}")
    logger.info(f"Infinite predictions: {inf_count:,}")

    # Calculate Brier score
    brier = calculate_brier_score(predictions, outcomes)

    # Calculate other metrics for comparison
    clean_mask = np.isfinite(predictions)
    clean_pred = predictions[clean_mask]
    clean_out = outcomes[clean_mask]

    if len(clean_pred) > 0:
        mean_pred = np.mean(clean_pred)
        mean_out = np.mean(clean_out)

        # Calculate calibration
        bins = np.linspace(0, 1, 11)
        ece = 0
        mce = 0

        for i in range(len(bins)-1):
            mask = (clean_pred >= bins[i]) & (clean_pred < bins[i+1])
            if i == len(bins)-2:
                mask = (clean_pred >= bins[i]) & (clean_pred <= bins[i+1])

            if mask.sum() > 0:
                avg_pred = clean_pred[mask].mean()
                avg_out = clean_out[mask].mean()
                error = abs(avg_pred - avg_out)
                weight = mask.sum() / len(clean_pred)
                ece += weight * error
                mce = max(mce, error)

        logger.info(f"\nMetrics for {name}:")
        logger.info(f"  Brier Score: {brier:.4f}")
        logger.info(f"  ECE: {ece:.4f}")
        logger.info(f"  MCE: {mce:.4f}")
        logger.info(f"  Mean Prediction: {mean_pred:.4f}")
        logger.info(f"  Mean Outcome: {mean_out:.4f}")
        logger.info(f"  Bias: {mean_pred - mean_out:.4f}")

        # Check distribution of predictions
        logger.info(f"\nPrediction Distribution:")
        logger.info(f"  Min: {clean_pred.min():.4f}")
        logger.info(f"  25%: {np.percentile(clean_pred, 25):.4f}")
        logger.info(f"  50%: {np.percentile(clean_pred, 50):.4f}")
        logger.info(f"  75%: {np.percentile(clean_pred, 75):.4f}")
        logger.info(f"  Max: {clean_pred.max():.4f}")

        # Investigate NaN sources if present
        if nan_count > 0:
            logger.info(f"\nInvestigating NaN sources...")

            # Check where NaNs occur
            nan_df = valid.filter(pl.col(price_col).is_nan())
            if len(nan_df) > 0:
                sample = nan_df.head(5)
                logger.info(f"Sample of NaN predictions:")
                for row in sample.iter_rows(named=True):
                    logger.info(f"  Time remaining: {row['time_remaining']}, "
                              f"IV: {row.get('implied_vol_adjusted', 'N/A')}, "
                              f"d2: {row.get('d2', 'N/A')}")

    return {
        "name": name,
        "total": len(df),
        "valid": len(valid),
        "brier": brier,
        "ece": ece if 'ece' in locals() else None,
        "mce": mce if 'mce' in locals() else None,
        "nan_count": nan_count,
        "inf_count": inf_count
    }

def main():
    logger.info("="*80)
    logger.info("BRIER SCORE ANALYSIS")
    logger.info("="*80)

    # Check for original pilot backtest results
    original_file = Path("/Users/lgierhake/Documents/ETH/BT/research/model/results/pilot_backtest_results.parquet")
    quick_wins_file = Path("/Users/lgierhake/Documents/ETH/BT/research/model/results/quick_wins/quick_wins_optimized_results.parquet")

    results = []

    # Analyze original if exists
    if original_file.exists():
        results.append(analyze_results(original_file, "Original Pilot Backtest"))
    else:
        logger.warning(f"Original file not found: {original_file}")
        logger.info("Running original pilot backtest to get baseline...")
        import subprocess
        try:
            subprocess.run(["uv", "run", "python", "01_pricing/pilot_backtest.py"],
                         cwd="/Users/lgierhake/Documents/ETH/BT/research/model",
                         capture_output=True, timeout=180)
            if original_file.exists():
                results.append(analyze_results(original_file, "Original Pilot Backtest"))
        except Exception as e:
            logger.error(f"Could not run original backtest: {e}")

    # Analyze quick wins
    if quick_wins_file.exists():
        results.append(analyze_results(quick_wins_file, "Quick Wins Implementation"))
    else:
        logger.error(f"Quick wins file not found: {quick_wins_file}")

    # Compare results
    if len(results) == 2:
        logger.info("\n" + "="*80)
        logger.info("COMPARISON SUMMARY")
        logger.info("="*80)

        orig = results[0]
        quick = results[1]

        if orig["brier"] and quick["brier"]:
            brier_change = ((quick["brier"] - orig["brier"]) / orig["brier"]) * 100
            logger.info(f"\nBrier Score:")
            logger.info(f"  Original:   {orig['brier']:.4f}")
            logger.info(f"  Quick Wins: {quick['brier']:.4f}")
            logger.info(f"  Change:     {brier_change:+.1f}%")

            if quick["brier"] < orig["brier"]:
                logger.info("  ✅ IMPROVED (lower is better)")
            else:
                logger.info("  ❌ WORSE (lower is better)")

        if orig["ece"] and quick["ece"]:
            ece_change = ((quick["ece"] - orig["ece"]) / orig["ece"]) * 100
            logger.info(f"\nExpected Calibration Error:")
            logger.info(f"  Original:   {orig['ece']:.4f}")
            logger.info(f"  Quick Wins: {quick['ece']:.4f}")
            logger.info(f"  Change:     {ece_change:+.1f}%")

            if quick["ece"] < orig["ece"]:
                logger.info("  ✅ IMPROVED (lower is better)")

    logger.info("\n" + "="*80)
    logger.info("ANALYSIS COMPLETE")
    logger.info("="*80)

if __name__ == "__main__":
    main()