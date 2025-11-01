#!/usr/bin/env python3
"""
Regime-Stratified Model Evaluation.

Evaluates model performance across different market regimes to detect
regime-specific failures and non-stationarity.

Addresses critique: "Evaluation misses regime shifts - Brier is averaged
over 2 years, but BTC has phases (e.g., 2023 bull vs. 2024 sideways).
A flat 0.16 might hide poor performance in high-vol regimes."
"""

import logging
from typing import Any

import numpy as np
import polars as pl

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def calculate_brier_score(predictions: np.ndarray, outcomes: np.ndarray) -> float:
    """
    Calculate Brier score.

    Args:
        predictions: Predicted probabilities
        outcomes: Actual outcomes (0 or 1)

    Returns:
        Brier score (lower is better)
    """
    return float(np.mean((predictions - outcomes) ** 2))


def evaluate_single_regime(
    predictions_df: pl.DataFrame,
    regime_name: str,
) -> dict[str, Any]:
    """
    Evaluate Brier score for a single regime.

    Args:
        predictions_df: DataFrame with predictions and outcomes
        regime_name: Name of the regime for logging

    Returns:
        Dictionary with metrics
    """
    if len(predictions_df) == 0:
        logger.warning(f"  {regime_name}: No data (skipping)")
        return {}

    # Extract predictions and outcomes
    final_prob = predictions_df["final_prob"].to_numpy()
    outcome = predictions_df["outcome"].to_numpy()
    prob_mid = predictions_df["prob_mid"].to_numpy() if "prob_mid" in predictions_df.columns else None

    # Calculate model Brier
    model_brier = calculate_brier_score(final_prob, outcome)

    # Calculate baseline Brier (if available)
    baseline_brier = calculate_brier_score(prob_mid, outcome) if prob_mid is not None else None

    # Calculate improvement
    improvement_pct = ((baseline_brier - model_brier) / baseline_brier * 100) if baseline_brier is not None else None

    return {
        "regime": regime_name,
        "n_samples": len(predictions_df),
        "model_brier": model_brier,
        "baseline_brier": baseline_brier,
        "improvement_pct": improvement_pct,
    }


def evaluate_by_volatility_regime(predictions_df: pl.DataFrame) -> dict[str, dict[str, Any]]:
    """
    Stratify evaluation by volatility regime (low/mid/high).

    Uses realized volatility (rv_300s) to define regimes.

    Args:
        predictions_df: DataFrame with predictions, outcomes, and rv_300s column

    Returns:
        Dictionary of regime metrics
    """
    logger.info("\n" + "=" * 80)
    logger.info("VOLATILITY REGIME ANALYSIS")
    logger.info("=" * 80)

    if "rv_300s" not in predictions_df.columns:
        logger.warning("rv_300s column not found - skipping volatility regime analysis")
        return {}

    # Calculate volatility quantiles (tertiles)
    vol_q33 = predictions_df["rv_300s"].quantile(0.33)
    vol_q67 = predictions_df["rv_300s"].quantile(0.67)
    vol_quantiles = [vol_q33, vol_q67] if vol_q33 is not None and vol_q67 is not None else []

    logger.info(f"Volatility quantiles: {vol_quantiles}")
    logger.info(f"  Low vol:  rv_300s <= {vol_quantiles[0]:.6f}")
    logger.info(f"  Mid vol:  {vol_quantiles[0]:.6f} < rv_300s <= {vol_quantiles[1]:.6f}")
    logger.info(f"  High vol: rv_300s > {vol_quantiles[1]:.6f}")

    # Create regime segments
    regimes = {
        "low_vol": predictions_df.filter(pl.col("rv_300s") <= vol_quantiles[0]),
        "mid_vol": predictions_df.filter(
            (pl.col("rv_300s") > vol_quantiles[0]) & (pl.col("rv_300s") <= vol_quantiles[1])
        ),
        "high_vol": predictions_df.filter(pl.col("rv_300s") > vol_quantiles[1]),
    }

    # Evaluate each regime
    results = {}
    for regime_name, regime_df in regimes.items():
        results[regime_name] = evaluate_single_regime(regime_df, regime_name)

        # Log results
        if results[regime_name]:
            r = results[regime_name]
            logger.info(
                f"  {regime_name:12s}: n={r['n_samples']:7,} | "
                f"Brier={r['model_brier']:.4f} | "
                f"Improvement={r['improvement_pct']:+.2f}%"
                if r["improvement_pct"] is not None
                else ""
            )

    return results


def evaluate_by_time_to_expiry(predictions_df: pl.DataFrame) -> dict[str, dict[str, Any]]:
    """
    Stratify evaluation by time remaining to expiration.

    Args:
        predictions_df: DataFrame with predictions, outcomes, and time_remaining column

    Returns:
        Dictionary of regime metrics
    """
    logger.info("\n" + "=" * 80)
    logger.info("TIME-TO-EXPIRY REGIME ANALYSIS")
    logger.info("=" * 80)

    if "time_remaining" not in predictions_df.columns:
        logger.warning("time_remaining column not found - skipping time-to-expiry analysis")
        return {}

    # Define time buckets (in seconds)
    # 0-5min, 5-10min, 10-15min
    time_bins = [(0, 300), (300, 600), (600, 900)]

    results = {}
    for low, high in time_bins:
        regime_name = f"ttl_{low // 60}-{high // 60}min"

        regime_df = predictions_df.filter((pl.col("time_remaining") > low) & (pl.col("time_remaining") <= high))

        results[regime_name] = evaluate_single_regime(regime_df, regime_name)

        # Log results
        if results[regime_name]:
            r = results[regime_name]
            logger.info(
                f"  {regime_name:15s}: n={r['n_samples']:7,} | "
                f"Brier={r['model_brier']:.4f} | "
                f"Improvement={r['improvement_pct']:+.2f}%"
                if r["improvement_pct"] is not None
                else ""
            )

    return results


def evaluate_by_temporal_period(predictions_df: pl.DataFrame) -> dict[str, dict[str, Any]]:
    """
    Stratify evaluation by temporal period to detect model drift.

    Splits data into quartiles by date to assess whether performance
    degrades over time (non-stationarity).

    Args:
        predictions_df: DataFrame with predictions, outcomes, and date column

    Returns:
        Dictionary of regime metrics
    """
    logger.info("\n" + "=" * 80)
    logger.info("TEMPORAL DRIFT ANALYSIS")
    logger.info("=" * 80)

    if "date" not in predictions_df.columns:
        logger.warning("date column not found - skipping temporal analysis")
        return {}

    # Calculate date quantiles (quartiles)
    date_q25 = predictions_df["date"].quantile(0.25)
    date_q50 = predictions_df["date"].quantile(0.5)
    date_q75 = predictions_df["date"].quantile(0.75)
    date_quantiles = [date_q25, date_q50, date_q75] if all([date_q25, date_q50, date_q75]) else []

    logger.info("Temporal periods:")
    logger.info(f"  Q1 (oldest):  date <= {date_quantiles[0]}")
    logger.info(f"  Q2:           {date_quantiles[0]} < date <= {date_quantiles[1]}")
    logger.info(f"  Q3:           {date_quantiles[1]} < date <= {date_quantiles[2]}")
    logger.info(f"  Q4 (newest):  date > {date_quantiles[2]}")

    # Create temporal segments
    periods = {
        "Q1_oldest": predictions_df.filter(pl.col("date") <= date_quantiles[0]),
        "Q2": predictions_df.filter((pl.col("date") > date_quantiles[0]) & (pl.col("date") <= date_quantiles[1])),
        "Q3": predictions_df.filter((pl.col("date") > date_quantiles[1]) & (pl.col("date") <= date_quantiles[2])),
        "Q4_newest": predictions_df.filter(pl.col("date") > date_quantiles[2]),
    }

    # Evaluate each period
    results = {}
    for period_name, period_df in periods.items():
        results[period_name] = evaluate_single_regime(period_df, period_name)

        # Log results
        if results[period_name]:
            r = results[period_name]
            logger.info(
                f"  {period_name:12s}: n={r['n_samples']:7,} | "
                f"Brier={r['model_brier']:.4f} | "
                f"Improvement={r['improvement_pct']:+.2f}%"
                if r["improvement_pct"] is not None
                else ""
            )

    return results


def evaluate_by_moneyness(predictions_df: pl.DataFrame) -> dict[str, dict[str, Any]]:
    """
    Stratify evaluation by moneyness (OTM/ATM/ITM).

    Args:
        predictions_df: DataFrame with predictions, outcomes, and moneyness column

    Returns:
        Dictionary of regime metrics
    """
    logger.info("\n" + "=" * 80)
    logger.info("MONEYNESS REGIME ANALYSIS")
    logger.info("=" * 80)

    if "moneyness" not in predictions_df.columns:
        logger.warning("moneyness column not found - skipping moneyness analysis")
        return {}

    # Define moneyness buckets
    # OTM: moneyness < -0.05, ATM: -0.05 to 0.05, ITM: > 0.05
    regimes = {
        "OTM": predictions_df.filter(pl.col("moneyness") < -0.05),
        "ATM": predictions_df.filter((pl.col("moneyness") >= -0.05) & (pl.col("moneyness") <= 0.05)),
        "ITM": predictions_df.filter(pl.col("moneyness") > 0.05),
    }

    # Evaluate each regime
    results = {}
    for regime_name, regime_df in regimes.items():
        results[regime_name] = evaluate_single_regime(regime_df, regime_name)

        # Log results
        if results[regime_name]:
            r = results[regime_name]
            logger.info(
                f"  {regime_name:12s}: n={r['n_samples']:7,} | "
                f"Brier={r['model_brier']:.4f} | "
                f"Improvement={r['improvement_pct']:+.2f}%"
                if r["improvement_pct"] is not None
                else ""
            )

    return results


def evaluate_all_regimes(predictions_file: str) -> dict[str, Any]:
    """
    Comprehensive regime-stratified evaluation.

    Args:
        predictions_file: Path to predictions parquet file with columns:
            - final_prob: Model predictions
            - outcome: Actual outcomes
            - prob_mid: Baseline predictions (optional)
            - rv_300s: Realized volatility (optional)
            - time_remaining: Seconds to expiration (optional)
            - date: Date column (optional)
            - moneyness: Distance from strike (optional)

    Returns:
        Dictionary with all regime metrics
    """
    logger.info("=" * 80)
    logger.info("REGIME-STRATIFIED MODEL EVALUATION")
    logger.info("=" * 80)
    logger.info(f"Predictions file: {predictions_file}")

    # Load predictions
    predictions_df = pl.read_parquet(predictions_file)
    logger.info(f"Total samples: {len(predictions_df):,}")

    # Overall performance (baseline)
    overall = evaluate_single_regime(predictions_df, "Overall")
    logger.info("\n" + "=" * 80)
    logger.info("OVERALL PERFORMANCE")
    logger.info("=" * 80)
    if overall:
        logger.info(f"  Model Brier:    {overall['model_brier']:.4f}")
        if overall["baseline_brier"]:
            logger.info(f"  Baseline Brier: {overall['baseline_brier']:.4f}")
            logger.info(f"  Improvement:    {overall['improvement_pct']:+.2f}%")

    # Collect all regime results
    results = {
        "overall": overall,
        "volatility_regimes": evaluate_by_volatility_regime(predictions_df),
        "time_to_expiry_regimes": evaluate_by_time_to_expiry(predictions_df),
        "temporal_periods": evaluate_by_temporal_period(predictions_df),
        "moneyness_regimes": evaluate_by_moneyness(predictions_df),
    }

    # Summary analysis
    logger.info("\n" + "=" * 80)
    logger.info("REGIME STABILITY ANALYSIS")
    logger.info("=" * 80)

    # Calculate variance across regimes
    all_improvements = []
    for category, regimes in results.items():
        if category != "overall" and isinstance(regimes, dict):
            for metrics in regimes.values():
                if metrics and "improvement_pct" in metrics and metrics["improvement_pct"] is not None:
                    all_improvements.append(metrics["improvement_pct"])

    if all_improvements:
        regime_variance = float(np.var(all_improvements))
        regime_std = float(np.std(all_improvements))
        regime_range = float(np.max(all_improvements) - np.min(all_improvements))

        logger.info(f"Improvement variance across regimes: {regime_variance:.2f}")
        logger.info(f"Improvement std dev across regimes:  {regime_std:.2f}")
        logger.info(f"Improvement range (max - min):       {regime_range:.2f}%")

        # Identify worst regime
        worst_regime_idx = int(np.argmin(all_improvements))
        best_regime_idx = int(np.argmax(all_improvements))

        logger.info(f"\nWorst regime improvement: {all_improvements[worst_regime_idx]:.2f}%")
        logger.info(f"Best regime improvement:  {all_improvements[best_regime_idx]:.2f}%")

        results["stability_metrics"] = {
            "regime_variance": regime_variance,
            "regime_std": regime_std,
            "regime_range": regime_range,
            "worst_improvement": all_improvements[worst_regime_idx],
            "best_improvement": all_improvements[best_regime_idx],
        }

        # Warning if high variance
        if regime_std > 3.0:
            logger.warning("\n⚠️  HIGH REGIME VARIANCE DETECTED")
            logger.warning(
                "Model performance varies significantly across regimes (std > 3%)."
                "\nThis suggests overfitting to specific market conditions."
                "\nConsider regime-specific models or adaptive weighting."
            )
        elif all_improvements[worst_regime_idx] < 1.0:
            logger.warning("\n⚠️  POOR PERFORMANCE IN SOME REGIMES")
            logger.warning(
                f"Worst regime shows only {all_improvements[worst_regime_idx]:.2f}% improvement."
                "\nModel may fail in certain market conditions."
                "\nReview feature engineering for weak regimes."
            )
    else:
        logger.warning("Insufficient regime data for stability analysis")

    return results


def main() -> None:
    """Run regime analysis on predictions file."""
    import argparse

    parser = argparse.ArgumentParser(description="Regime-stratified model evaluation")
    parser.add_argument(
        "--predictions",
        type=str,
        required=True,
        help="Path to predictions parquet file",
    )
    parser.add_argument(
        "--output",
        type=str,
        help="Optional path to save results JSON",
    )

    args = parser.parse_args()

    # Run analysis
    results = evaluate_all_regimes(args.predictions)

    # Save results if requested
    if args.output:
        import json

        with open(args.output, "w") as f:
            json.dump(results, f, indent=2)
        logger.info(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
