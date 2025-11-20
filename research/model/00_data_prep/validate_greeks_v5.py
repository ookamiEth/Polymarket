#!/usr/bin/env python3
"""
Greek Features Validation Script V5
====================================

Comprehensive validation of Black-Scholes binary option Greeks implementation.

This script performs 3 critical validation steps:
1. Delta-Moneyness correlation diagnostic (investigate negative correlation)
2. Greek bounds and edge case testing (verify theoretical behavior)
3. Implementation comparison against archive reference code

Author: BT Research Team
Date: 2025-11-20
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import polars as pl
from scipy.stats import norm

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


# =============================================================================
# MANUAL GREEK CALCULATION (Reference Implementation)
# =============================================================================


def manual_delta_calculation(
    S: float, K: float, r: float, sigma: float, T: float
) -> float:
    """
    Manually calculate Delta for binary call option.

    Formula: Delta = e^(-rT) × φ(d₂) / (S × σ√T)

    Args:
        S: Spot price
        K: Strike price
        r: Risk-free rate (annual)
        sigma: Volatility (annual)
        T: Time to expiry (years)

    Returns:
        Delta value
    """
    # Handle None values
    if S is None or K is None or r is None or sigma is None or T is None:
        return np.nan
    if T <= 0 or sigma <= 0 or S <= 0:
        return np.nan

    epsilon = 1e-8
    sqrt_T = np.sqrt(T)
    d1_numerator = np.log(S / K) + (r + 0.5 * sigma**2) * T
    d1_denominator = sigma * sqrt_T + epsilon
    d2 = (d1_numerator / d1_denominator) - d1_denominator

    # Standard normal PDF at d2
    phi_d2 = norm.pdf(d2)

    # Binary option delta
    discount = np.exp(-r * T)
    delta = discount * phi_d2 / (S * sigma * sqrt_T + epsilon)

    return delta


def manual_gamma_calculation(
    S: float, K: float, r: float, sigma: float, T: float
) -> float:
    """Calculate Gamma for binary call option."""
    if S is None or K is None or r is None or sigma is None or T is None:
        return np.nan
    if T <= 0 or sigma <= 0 or S <= 0:
        return np.nan

    epsilon = 1e-8
    sqrt_T = np.sqrt(T)
    d1_numerator = np.log(S / K) + (r + 0.5 * sigma**2) * T
    d1_denominator = sigma * sqrt_T + epsilon
    d1 = d1_numerator / d1_denominator
    d2 = d1 - d1_denominator

    phi_d2 = norm.pdf(d2)
    discount = np.exp(-r * T)

    gamma = -discount * phi_d2 * d1 / (S**2 * sigma**2 * T + epsilon)

    return gamma


def manual_vega_calculation(
    S: float, K: float, r: float, sigma: float, T: float
) -> float:
    """Calculate Vega for binary call option."""
    # Check for None FIRST before any comparison
    if S is None or K is None or r is None or sigma is None or T is None:
        return np.nan
    # Now safe to compare to 0
    if T <= 0 or sigma <= 0 or S <= 0:
        return np.nan

    epsilon = 1e-8
    sqrt_T = np.sqrt(T)
    d1_numerator = np.log(S / K) + (r + 0.5 * sigma**2) * T
    d1_denominator = sigma * sqrt_T + epsilon
    d2 = (d1_numerator / d1_denominator) - d1_denominator

    phi_d2 = norm.pdf(d2)
    discount = np.exp(-r * T)

    vega = -discount * phi_d2 * d2 * sqrt_T / (sigma + epsilon)

    return vega


# =============================================================================
# STEP 1: DELTA-MONEYNESS DIAGNOSTIC
# =============================================================================


def step1_delta_moneyness_diagnostic(
    df: pl.DataFrame, output_dir: Path
) -> dict[str, float]:
    """
    Diagnose Delta-Moneyness correlation issue.

    Compares manual Delta calculation with file values and analyzes correlation.

    Args:
        df: Sample DataFrame with Greeks
        output_dir: Directory to save outputs

    Returns:
        Dict of diagnostic metrics
    """
    logger.info("=" * 80)
    logger.info("STEP 1: DELTA-MONEYNESS DIAGNOSTIC")
    logger.info("=" * 80)

    # Calculate manual delta for each row
    logger.info("Computing manual Delta calculations...")

    manual_deltas = []
    for row in df.iter_rows(named=True):
        manual_delta = manual_delta_calculation(
            S=row["S"],
            K=row["K"],
            r=row.get("r", 0.05),  # Default to 5% if missing
            sigma=row["sigma_mid"],
            T=row["T_years"],
        )
        manual_deltas.append(manual_delta)

    df = df.with_columns([pl.Series("delta_manual", manual_deltas)])

    # Compute error metrics
    df = df.with_columns(
        [
            (pl.col("delta") - pl.col("delta_manual")).alias("delta_error"),
            ((pl.col("delta") - pl.col("delta_manual")).abs()).alias(
                "delta_abs_error"
            ),
        ]
    )

    # Statistics
    max_error = df["delta_abs_error"].max()
    mean_error = df["delta_abs_error"].mean()
    median_error = df["delta_abs_error"].median()

    logger.info(f"Delta Error Statistics:")
    logger.info(f"  Max absolute error: {max_error:.2e}")
    logger.info(f"  Mean absolute error: {mean_error:.2e}")
    logger.info(f"  Median absolute error: {median_error:.2e}")

    # Correlation analysis
    corr_file_delta = df.select(
        pl.corr("delta", "moneyness").alias("correlation")
    ).item()
    corr_manual_delta = df.select(
        pl.corr("delta_manual", "moneyness").alias("correlation")
    ).item()

    logger.info(f"\nCorrelation with Moneyness:")
    logger.info(f"  File Delta: {corr_file_delta:.4f}")
    logger.info(f"  Manual Delta: {corr_manual_delta:.4f}")

    # Correlation by time bucket
    logger.info(f"\nCorrelation by Time Bucket:")
    for bucket in ["near", "mid", "far"]:
        bucket_df = df.filter(pl.col("temporal_regime") == bucket)
        if len(bucket_df) > 10:
            corr_bucket = bucket_df.select(
                pl.corr("delta", "moneyness").alias("correlation")
            ).item()
            logger.info(f"  {bucket}: {corr_bucket:.4f}")

    # Save diagnostic sample
    output_file = output_dir / "delta_validation_sample.csv"
    df.select(
        [
            "S",
            "K",
            "sigma_mid",
            "T_years",
            "moneyness",
            "delta",
            "delta_manual",
            "delta_error",
            "temporal_regime",
        ]
    ).write_csv(output_file)
    logger.info(f"✓ Saved diagnostic sample: {output_file}")

    return {
        "max_error": max_error,
        "mean_error": mean_error,
        "corr_file": corr_file_delta,
        "corr_manual": corr_manual_delta,
    }


# =============================================================================
# STEP 2: GREEK BOUNDS & EDGE CASE TESTING
# =============================================================================


def step2_greek_bounds_testing(df: pl.DataFrame, output_dir: Path) -> dict:
    """
    Test Greek values at boundary conditions.

    Args:
        df: Full DataFrame
        output_dir: Directory to save outputs

    Returns:
        Dict of test results
    """
    logger.info("=" * 80)
    logger.info("STEP 2: GREEK BOUNDS & EDGE CASE TESTING")
    logger.info("=" * 80)

    results = {}

    # Test A: ATM options (moneyness ≈ 0)
    logger.info("\nTest A: ATM Options (|moneyness| < 0.01)")
    atm_df = df.filter(pl.col("moneyness").abs() < 0.01)
    if len(atm_df) > 0:
        delta_mean = atm_df["delta"].mean()
        gamma_mean = atm_df["gamma"].mean()
        gamma_negative_pct = (
            atm_df.filter(pl.col("gamma") < 0).height / atm_df.height * 100
        )

        logger.info(f"  Rows: {len(atm_df):,}")
        logger.info(f"  Delta mean: {delta_mean:.4f} (expect ≈ 0.5)")
        logger.info(f"  Gamma mean: {gamma_mean:.6f}")
        logger.info(f"  Gamma negative: {gamma_negative_pct:.1f}% (expect > 70%)")

        results["atm_delta_mean"] = delta_mean
        results["atm_gamma_negative_pct"] = gamma_negative_pct
    else:
        logger.warning("  No ATM options found!")

    # Test B: Deep ITM options (moneyness > 0.05)
    logger.info("\nTest B: Deep ITM Options (moneyness > 0.05)")
    itm_df = df.filter(pl.col("moneyness") > 0.05)
    if len(itm_df) > 0:
        delta_mean = itm_df["delta"].mean()
        gamma_abs_mean = itm_df["gamma"].abs().mean()

        logger.info(f"  Rows: {len(itm_df):,}")
        logger.info(f"  Delta mean: {delta_mean:.4f} (expect > 0.5)")
        logger.info(f"  |Gamma| mean: {gamma_abs_mean:.6f} (expect → 0)")

        results["itm_delta_mean"] = delta_mean
    else:
        logger.warning("  No deep ITM options found!")

    # Test C: Deep OTM options (moneyness < -0.05)
    logger.info("\nTest C: Deep OTM Options (moneyness < -0.05)")
    otm_df = df.filter(pl.col("moneyness") < -0.05)
    if len(otm_df) > 0:
        delta_mean = otm_df["delta"].mean()
        gamma_abs_mean = otm_df["gamma"].abs().mean()

        logger.info(f"  Rows: {len(otm_df):,}")
        logger.info(f"  Delta mean: {delta_mean:.4f} (expect < 0.5)")
        logger.info(f"  |Gamma| mean: {gamma_abs_mean:.6f} (expect → 0)")

        results["otm_delta_mean"] = delta_mean
    else:
        logger.warning("  No deep OTM options found!")

    # Test D: Near-expiry options (T < 60s)
    logger.info("\nTest D: Near-Expiry Options (time_remaining < 60s)")
    near_expiry_df = df.filter(pl.col("time_remaining") < 60)
    if len(near_expiry_df) > 0:
        delta_std = near_expiry_df["delta"].std()
        gamma_abs_max = near_expiry_df["gamma"].abs().max()

        logger.info(f"  Rows: {len(near_expiry_df):,}")
        logger.info(f"  Delta std: {delta_std:.4f} (expect high variance)")
        logger.info(f"  |Gamma| max: {gamma_abs_max:.6f} (expect large)")

        results["near_expiry_delta_std"] = delta_std
    else:
        logger.warning("  No near-expiry options found!")

    # Test E: Long-dated options (T > 600s)
    logger.info("\nTest E: Long-Dated Options (time_remaining > 600s)")
    long_dated_df = df.filter(pl.col("time_remaining") > 600)
    if len(long_dated_df) > 0:
        delta_std = long_dated_df["delta"].std()
        vega_abs_mean = long_dated_df["vega"].abs().mean()

        logger.info(f"  Rows: {len(long_dated_df):,}")
        logger.info(f"  Delta std: {delta_std:.4f}")
        logger.info(f"  |Vega| mean: {vega_abs_mean:.6f}")

        results["long_dated_vega_mean"] = vega_abs_mean
    else:
        logger.warning("  No long-dated options found!")

    # Save bounds validation table
    output_file = output_dir / "greek_bounds_validation.csv"
    bounds_data = pl.DataFrame(
        {
            "test_case": list(results.keys()),
            "value": list(results.values()),
        }
    )
    bounds_data.write_csv(output_file)
    logger.info(f"\n✓ Saved bounds validation: {output_file}")

    return results


# =============================================================================
# STEP 3: ARCHIVE IMPLEMENTATION COMPARISON
# =============================================================================


def step3_archive_comparison(df: pl.DataFrame, output_dir: Path) -> dict:
    """
    Compare current implementation with archive reference.

    Args:
        df: Sample DataFrame
        output_dir: Directory to save outputs

    Returns:
        Dict of comparison metrics
    """
    logger.info("=" * 80)
    logger.info("STEP 3: ARCHIVE IMPLEMENTATION COMPARISON")
    logger.info("=" * 80)

    # Compare file delta vs manual (reference) delta
    logger.info("Computing element-wise differences...")

    manual_deltas = []
    manual_gammas = []
    manual_vegas = []

    for row in df.iter_rows(named=True):
        # Extract values with None checks
        S = row.get("S")
        K = row.get("K")
        r = row.get("r", 0.05)
        sigma = row.get("sigma_mid")
        T = row.get("T_years")

        # Append manual calculations (handle None gracefully)
        manual_deltas.append(manual_delta_calculation(S, K, r, sigma, T))
        manual_gammas.append(manual_gamma_calculation(S, K, r, sigma, T))
        manual_vegas.append(manual_vega_calculation(S, K, r, sigma, T))

    df = df.with_columns(
        [
            pl.Series("delta_ref", manual_deltas),
            pl.Series("gamma_ref", manual_gammas),
            pl.Series("vega_ref", manual_vegas),
        ]
    )

    # Compute errors
    df = df.with_columns(
        [
            (pl.col("delta") - pl.col("delta_ref")).abs().alias("delta_err"),
            (pl.col("gamma") - pl.col("gamma_ref")).abs().alias("gamma_err"),
            (pl.col("vega") - pl.col("vega_ref")).abs().alias("vega_err"),
        ]
    )

    # Statistics
    results = {}
    for greek in ["delta", "gamma", "vega"]:
        max_err = df[f"{greek}_err"].max()
        mean_err = df[f"{greek}_err"].mean()
        median_err = df[f"{greek}_err"].median()

        logger.info(f"\n{greek.upper()} Error Metrics:")
        logger.info(f"  Max: {max_err:.2e}")
        logger.info(f"  Mean: {mean_err:.2e}")
        logger.info(f"  Median: {median_err:.2e}")

        results[f"{greek}_max_error"] = max_err
        results[f"{greek}_mean_error"] = mean_err

        # Flag problematic rows
        if max_err > 1e-3:
            logger.warning(f"  ⚠️ Large errors detected (> 1e-3)!")
            problematic = df.filter(pl.col(f"{greek}_err") > 1e-3)
            logger.warning(f"  Problematic rows: {len(problematic)}")

    # Save comparison
    output_file = output_dir / "implementation_comparison.csv"
    df.select(
        [
            "S",
            "K",
            "sigma_mid",
            "T_years",
            "delta",
            "delta_ref",
            "delta_err",
            "gamma",
            "gamma_ref",
            "gamma_err",
            "vega",
            "vega_ref",
            "vega_err",
        ]
    ).write_csv(output_file)
    logger.info(f"\n✓ Saved comparison: {output_file}")

    return results


# =============================================================================
# MAIN VALIDATION WORKFLOW
# =============================================================================


def main():
    """Main validation workflow."""
    logger.info("=" * 80)
    logger.info("GREEK FEATURES VALIDATION V5")
    logger.info("=" * 80)

    # Setup paths
    script_dir = Path(__file__).parent.resolve()
    model_dir = script_dir.parent
    data_file = model_dir / "data" / "consolidated_features_v5_selected.parquet"
    output_dir = model_dir / "results" / "greek_validation_v5"
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"\nData file: {data_file}")
    logger.info(f"Output dir: {output_dir}")

    # Load sample data
    logger.info(f"\nLoading sample data (10,000 rows)...")
    df = pl.read_parquet(data_file).sample(n=10_000, seed=42)

    logger.info(f"Sample size: {len(df):,} rows")
    logger.info(f"Columns: {len(df.columns)}")

    # Required columns check
    required_cols = [
        "S",
        "K",
        "sigma_mid",
        "T_years",
        "moneyness",
        "delta",
        "gamma",
        "vega",
        "temporal_regime",
        "time_remaining",
    ]
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        logger.error(f"Missing required columns: {missing}")
        return

    # Run validation steps
    all_results = {}

    # Step 1: Delta-Moneyness diagnostic
    step1_results = step1_delta_moneyness_diagnostic(df, output_dir)
    all_results.update({"step1_" + k: v for k, v in step1_results.items()})

    # Step 2: Bounds testing
    step2_results = step2_greek_bounds_testing(df, output_dir)
    all_results.update({"step2_" + k: v for k, v in step2_results.items()})

    # Step 3: Archive comparison
    step3_results = step3_archive_comparison(df, output_dir)
    all_results.update({"step3_" + k: v for k, v in step3_results.items()})

    # Summary
    logger.info("\n" + "=" * 80)
    logger.info("VALIDATION SUMMARY")
    logger.info("=" * 80)

    # Assessment
    assessment = []

    # Check 1: Implementation accuracy
    if all_results.get("step1_max_error", 1.0) < 1e-6:
        assessment.append("✅ Implementation matches manual calculation")
    else:
        assessment.append(
            f"⚠️ Implementation error: {all_results.get('step1_max_error', 0):.2e}"
        )

    # Check 2: Delta correlation
    corr = all_results.get("step1_corr_file", 0)
    if corr > 0.3:
        assessment.append("✅ Delta-Moneyness correlation positive and strong")
    elif corr > 0:
        assessment.append(
            f"⚠️ Delta-Moneyness correlation weak: {corr:.3f} (expect > 0.3)"
        )
    else:
        assessment.append(
            f"❌ Delta-Moneyness correlation NEGATIVE: {corr:.3f} (INVESTIGATE!)"
        )

    # Check 3: Gamma negativity
    gamma_neg = all_results.get("step2_atm_gamma_negative_pct", 0)
    if gamma_neg > 70:
        assessment.append(f"✅ Gamma negative for ATM: {gamma_neg:.1f}%")
    else:
        assessment.append(
            f"⚠️ Gamma negative rate low: {gamma_neg:.1f}% (expect > 70%)"
        )

    # Check 4: ATM delta
    atm_delta = all_results.get("step2_atm_delta_mean", 0)
    if 0.3 < atm_delta < 0.7:
        assessment.append(f"✅ ATM Delta reasonable: {atm_delta:.3f}")
    else:
        assessment.append(
            f"⚠️ ATM Delta unusual: {atm_delta:.3f} (expect ≈ 0.5)"
        )

    for line in assessment:
        logger.info(line)

    # Save summary
    summary_file = output_dir / "validation_summary.txt"
    with open(summary_file, "w") as f:
        f.write("GREEK VALIDATION SUMMARY V5\n")
        f.write("=" * 80 + "\n\n")
        for line in assessment:
            f.write(line + "\n")
        f.write("\n\nDetailed Results:\n")
        for key, value in all_results.items():
            f.write(f"  {key}: {value}\n")

    logger.info(f"\n✓ Validation complete!")
    logger.info(f"✓ Results saved to: {output_dir}")


if __name__ == "__main__":
    main()
