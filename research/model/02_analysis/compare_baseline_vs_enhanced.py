#!/usr/bin/env python3
"""
Compare Baseline vs Enhanced Model Performance

Analyzes the improvement from adaptive volatility integration by comparing:
1. Brier scores (baseline vs enhanced)
2. Calibration errors
3. Performance by time remaining
4. Performance by IV staleness
5. Statistical significance tests

Expected improvement: 15-25% Brier reduction (0.162 → 0.120-0.135)

Author: BT Research Team
Date: 2025-10-29
"""

import logging
import sys
from pathlib import Path

import numpy as np
import polars as pl
from scipy import stats

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

# File paths
BASELINE_FILE = Path("/Users/lgierhake/Documents/ETH/BT/research/model/results/production_backtest_results.parquet")
ENHANCED_FILE = Path("/Users/lgierhake/Documents/ETH/BT/research/model/results/production_backtest_enhanced.parquet")
OUTPUT_DIR = Path("/Users/lgierhake/Documents/ETH/BT/research/model/02_analysis/results")


def brier_score(predictions: pl.Series, outcomes: pl.Series) -> float:
    """
    Calculate Brier score (mean squared error for probabilities).

    Lower is better. Perfect score = 0, random = 0.25.

    Args:
        predictions: Predicted probabilities (0-1)
        outcomes: Actual outcomes (0 or 1)

    Returns:
        Brier score (float)
    """
    squared_errors = (predictions - outcomes) ** 2
    mean_val = squared_errors.mean()
    return float(mean_val) if mean_val is not None else 0.0  # type: ignore[arg-type]


def expected_calibration_error(
    predictions: pl.Series, outcomes: pl.Series, n_bins: int = 10
) -> tuple[float, pl.DataFrame]:
    """
    Calculate Expected Calibration Error (ECE).

    ECE measures the average difference between predicted probabilities
    and actual frequencies across probability bins.

    Args:
        predictions: Predicted probabilities
        outcomes: Actual outcomes
        n_bins: Number of bins for calibration

    Returns:
        (ECE score, calibration DataFrame)
    """
    # Create bins
    bin_edges = np.linspace(0, 1, n_bins + 1)
    df = pl.DataFrame({"pred": predictions, "outcome": outcomes})

    calibration_data = []
    total_samples = len(df)

    for i in range(n_bins):
        bin_start = bin_edges[i]
        bin_end = bin_edges[i + 1]

        # Filter to bin
        if i == n_bins - 1:  # Last bin includes upper edge
            bin_df = df.filter((pl.col("pred") >= bin_start) & (pl.col("pred") <= bin_end))
        else:
            bin_df = df.filter((pl.col("pred") >= bin_start) & (pl.col("pred") < bin_end))

        if len(bin_df) == 0:
            continue

        # Calculate bin statistics
        bin_count = len(bin_df)
        bin_weight = bin_count / total_samples
        mean_pred = bin_df["pred"].mean()
        actual_freq = bin_df["outcome"].mean()

        calibration_data.append(
            {
                "bin_start": bin_start,
                "bin_end": bin_end,
                "count": bin_count,
                "weight": bin_weight,
                "mean_predicted": mean_pred,
                "actual_frequency": actual_freq,
                "calibration_error": abs(mean_pred - actual_freq)  # type: ignore[operator]
                if mean_pred is not None and actual_freq is not None
                else 0.0,
            }
        )

    calibration_df = pl.DataFrame(calibration_data)

    # Calculate weighted ECE
    ece = (calibration_df["weight"] * calibration_df["calibration_error"]).sum()

    return float(ece), calibration_df


def performance_by_time_bucket(df: pl.DataFrame, prob_col: str) -> pl.DataFrame:
    """
    Calculate performance metrics by time remaining bucket.

    Buckets: 0-300s, 300-600s, 600-900s

    Args:
        df: DataFrame with predictions and outcomes
        prob_col: Column name for probabilities

    Returns:
        DataFrame with metrics by bucket
    """
    # Define buckets
    df = df.with_columns(
        [
            pl.when(pl.col("time_remaining") < 300)
            .then(pl.lit("0-300s"))
            .when(pl.col("time_remaining") < 600)
            .then(pl.lit("300-600s"))
            .otherwise(pl.lit("600-900s"))
            .alias("time_bucket")
        ]
    )

    # Calculate metrics per bucket
    results = []
    for bucket in ["0-300s", "300-600s", "600-900s"]:
        bucket_df = df.filter(pl.col("time_bucket") == bucket)

        if len(bucket_df) == 0:
            continue

        brier = brier_score(bucket_df[prob_col], bucket_df["outcome"])
        ece, _ = expected_calibration_error(bucket_df[prob_col], bucket_df["outcome"])

        results.append(
            {
                "time_bucket": bucket,
                "count": len(bucket_df),
                "brier_score": brier,
                "ece": ece,
                "mean_prob": bucket_df[prob_col].mean(),
                "actual_freq": bucket_df["outcome"].mean(),
            }
        )

    return pl.DataFrame(results)


def performance_by_staleness(df: pl.DataFrame, prob_col: str) -> pl.DataFrame:
    """
    Calculate performance metrics by IV staleness bucket.

    Buckets: <10s, 10-60s, 60-120s, >120s

    Args:
        df: DataFrame with predictions and outcomes
        prob_col: Column name for probabilities

    Returns:
        DataFrame with metrics by staleness bucket
    """
    # Define staleness buckets
    df = df.with_columns(
        [
            pl.when(pl.col("iv_staleness_seconds") < 10)
            .then(pl.lit("<10s (fresh)"))
            .when(pl.col("iv_staleness_seconds") < 60)
            .then(pl.lit("10-60s (moderate)"))
            .when(pl.col("iv_staleness_seconds") < 120)
            .then(pl.lit("60-120s (stale)"))
            .otherwise(pl.lit(">120s (very stale)"))
            .alias("staleness_bucket")
        ]
    )

    # Calculate metrics per bucket
    results = []
    for bucket in ["<10s (fresh)", "10-60s (moderate)", "60-120s (stale)", ">120s (very stale)"]:
        bucket_df = df.filter(pl.col("staleness_bucket") == bucket)

        if len(bucket_df) == 0:
            continue

        brier = brier_score(bucket_df[prob_col], bucket_df["outcome"])
        ece, _ = expected_calibration_error(bucket_df[prob_col], bucket_df["outcome"])

        results.append(
            {
                "staleness_bucket": bucket,
                "count": len(bucket_df),
                "brier_score": brier,
                "ece": ece,
                "mean_prob": bucket_df[prob_col].mean(),
                "actual_freq": bucket_df["outcome"].mean(),
            }
        )

    return pl.DataFrame(results)


def statistical_significance_test(baseline_preds: pl.Series, enhanced_preds: pl.Series, outcomes: pl.Series) -> dict:
    """
    Test statistical significance of improvement using paired t-test.

    Compares squared errors (Brier components) between models.

    Args:
        baseline_preds: Baseline predictions
        enhanced_preds: Enhanced predictions
        outcomes: Actual outcomes

    Returns:
        Dictionary with test results
    """
    # Calculate squared errors for each prediction
    baseline_errors = (baseline_preds - outcomes) ** 2
    enhanced_errors = (enhanced_preds - outcomes) ** 2

    # Paired t-test (same observations, different models)
    t_stat, p_value = stats.ttest_rel(baseline_errors.to_numpy(), enhanced_errors.to_numpy())

    # Effect size (Cohen's d)
    diff = baseline_errors - enhanced_errors
    diff_mean = diff.mean()
    diff_std = diff.std()
    cohens_d = float(diff_mean / diff_std) if diff_std is not None and diff_std != 0 else 0.0  # type: ignore[operator]

    return {
        "t_statistic": t_stat,
        "p_value": p_value,
        "significant": p_value < 0.05,
        "cohens_d": cohens_d,
        "mean_error_reduction": diff.mean(),
    }


def main() -> None:
    """Main execution function."""
    logger.info("=" * 80)
    logger.info("BASELINE VS ENHANCED MODEL COMPARISON")
    logger.info("=" * 80)

    # Create output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load data
    logger.info(f"Loading baseline results: {BASELINE_FILE}")
    baseline = pl.read_parquet(BASELINE_FILE)
    logger.info(f"Loaded {len(baseline):,} rows")

    logger.info(f"Loading enhanced results: {ENHANCED_FILE}")
    enhanced = pl.read_parquet(ENHANCED_FILE)
    logger.info(f"Loaded {len(enhanced):,} rows")

    # Filter to valid predictions only
    baseline_valid = baseline.filter(pl.col("prob_mid").is_not_null() & pl.col("outcome").is_not_null())
    enhanced_valid = enhanced.filter(pl.col("price_adaptive").is_not_null() & pl.col("outcome").is_not_null())

    logger.info(f"Baseline valid predictions: {len(baseline_valid):,}")
    logger.info(f"Enhanced valid predictions: {len(enhanced_valid):,}")

    # Overall performance
    logger.info("\n" + "=" * 80)
    logger.info("OVERALL PERFORMANCE")
    logger.info("=" * 80)

    baseline_brier = brier_score(baseline_valid["prob_mid"], baseline_valid["outcome"])
    enhanced_brier = brier_score(enhanced_valid["price_adaptive"], enhanced_valid["outcome"])
    improvement_pct = (baseline_brier - enhanced_brier) / baseline_brier * 100

    baseline_ece, _ = expected_calibration_error(baseline_valid["prob_mid"], baseline_valid["outcome"])
    enhanced_ece, _ = expected_calibration_error(enhanced_valid["price_adaptive"], enhanced_valid["outcome"])

    logger.info(f"Baseline Brier Score:  {baseline_brier:.6f}")
    logger.info(f"Enhanced Brier Score:  {enhanced_brier:.6f}")
    logger.info(f"Improvement:           {improvement_pct:+.2f}% ({baseline_brier - enhanced_brier:.6f})")
    logger.info("")
    logger.info(f"Baseline ECE:          {baseline_ece:.6f}")
    logger.info(f"Enhanced ECE:          {enhanced_ece:.6f}")
    logger.info(f"ECE Improvement:       {(baseline_ece - enhanced_ece) / baseline_ece * 100:+.2f}%")

    # Statistical significance
    logger.info("\n" + "=" * 80)
    logger.info("STATISTICAL SIGNIFICANCE")
    logger.info("=" * 80)

    # Align datasets for paired comparison
    baseline_aligned = baseline_valid.select(["contract_id", "timestamp", "prob_mid", "outcome"]).rename(
        {"prob_mid": "baseline_prob"}
    )
    enhanced_aligned = enhanced_valid.select(["contract_id", "timestamp", "price_adaptive", "outcome"]).rename(
        {"price_adaptive": "enhanced_prob"}
    )

    paired = baseline_aligned.join(enhanced_aligned, on=["contract_id", "timestamp", "outcome"], how="inner")

    sig_test = statistical_significance_test(paired["baseline_prob"], paired["enhanced_prob"], paired["outcome"])

    logger.info(f"Paired t-test statistic:  {sig_test['t_statistic']:.4f}")
    logger.info(f"P-value:                  {sig_test['p_value']:.6f}")
    logger.info(f"Significant (p < 0.05):   {sig_test['significant']}")
    logger.info(f"Cohen's d (effect size):  {sig_test['cohens_d']:.4f}")
    logger.info(f"Mean error reduction:     {sig_test['mean_error_reduction']:.6f}")

    # Performance by time bucket
    logger.info("\n" + "=" * 80)
    logger.info("PERFORMANCE BY TIME REMAINING")
    logger.info("=" * 80)

    baseline_time = performance_by_time_bucket(baseline_valid, "prob_mid")
    enhanced_time = performance_by_time_bucket(enhanced_valid, "price_adaptive")

    logger.info("\nBaseline:")
    print(baseline_time)

    logger.info("\nEnhanced:")
    print(enhanced_time)

    # Save comparison
    baseline_time.write_parquet(OUTPUT_DIR / "baseline_by_time.parquet")
    enhanced_time.write_parquet(OUTPUT_DIR / "enhanced_by_time.parquet")

    # Performance by staleness (enhanced only)
    logger.info("\n" + "=" * 80)
    logger.info("PERFORMANCE BY IV STALENESS (Enhanced Model)")
    logger.info("=" * 80)

    enhanced_staleness = performance_by_staleness(enhanced_valid, "price_adaptive")
    logger.info("\nEnhanced by Staleness:")
    print(enhanced_staleness)

    enhanced_staleness.write_parquet(OUTPUT_DIR / "enhanced_by_staleness.parquet")

    # Summary report
    summary = {
        "baseline_brier": baseline_brier,
        "enhanced_brier": enhanced_brier,
        "improvement_pct": improvement_pct,
        "baseline_ece": baseline_ece,
        "enhanced_ece": enhanced_ece,
        "p_value": sig_test["p_value"],
        "significant": sig_test["significant"],
        "cohens_d": sig_test["cohens_d"],
    }

    summary_df = pl.DataFrame([summary])
    summary_df.write_parquet(OUTPUT_DIR / "overall_summary.parquet")

    logger.info("\n" + "=" * 80)
    logger.info("ANALYSIS COMPLETE")
    logger.info("=" * 80)
    logger.info(f"Results saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
