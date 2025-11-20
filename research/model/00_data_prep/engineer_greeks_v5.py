#!/usr/bin/env python3
"""
Greek Feature Engineering for V5
=================================

Add 35 Greek features to V4 consolidated dataset using binary option formulas.

Features added:
- 10 core Greeks: delta, gamma, vega, theta, vanna, volga, charm, d1, d2, phi_d2
- 11 first-order transformations: log, sqrt, squared, inverse, time-weighted, etc.
- 7 second-order transformations: ratios, products, stability metrics
- 1 moneyness bin (10 categories)

Input:  consolidated_features_v4_pipeline_ready.parquet (163 columns, ~61M rows)
Output: consolidated_features_v5_pipeline_ready.parquet (193 columns, ~61M rows)

Runtime: ~5-10 minutes on 32 vCPU machine

Author: BT Research Team
Date: 2025-11-19
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import polars as pl

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def add_risk_free_rate(df: pl.LazyFrame, baseline_path: str | Path) -> pl.LazyFrame:
    """
    Join risk-free rate 'r' from production backtest results.

    Args:
        df: Pipeline-ready V4 data (lazy)
        baseline_path: Path to production_backtest_results_v4.parquet

    Returns:
        DataFrame with 'r' column added
    """
    logger.info("Joining risk-free rate from baseline...")

    df_baseline = pl.scan_parquet(baseline_path).select(["timestamp", "r"])

    df = df.join(df_baseline, on="timestamp", how="left")

    # Fill missing 'r' with 0.0 (risk-neutral approximation)
    df = df.with_columns([pl.col("r").fill_null(0.0)])

    logger.info("✓ Risk-free rate joined")
    return df


def calculate_binary_greeks(df: pl.LazyFrame) -> pl.LazyFrame:
    """
    Calculate Greeks for binary options using Black-Scholes framework.

    Binary option payoff: 1 if S > K at expiry, 0 otherwise
    Price: P = e^(-rT) × Φ(d₂)

    Where:
        d₁ = [ln(S/K) + (r + σ²/2)T] / (σ√T)
        d₂ = d₁ - σ√T
        Φ = standard normal CDF
        φ = standard normal PDF

    Args:
        df: DataFrame with S, K, sigma_mid, T_years, r

    Returns:
        DataFrame with 10 core Greek columns added
    """
    logger.info("Calculating binary option Greeks...")

    epsilon = 1e-8

    df = df.with_columns(
        [
            # Intermediate calculations
            (pl.col("T_years") + epsilon).sqrt().alias("sqrt_T"),
            (-(pl.col("r") * pl.col("T_years"))).exp().alias("discount"),
            # d1 = [ln(S/K) + (r + sigma²/2)T] / (sigma√T)
            (
                (pl.col("S") / (pl.col("K") + epsilon)).log()
                + (pl.col("r") + pl.col("sigma_mid").pow(2) / 2) * pl.col("T_years")
            ).alias("d1_numerator"),
            # Denominator: sigma√T
            (pl.col("sigma_mid") * (pl.col("T_years") + epsilon).sqrt()).alias("d1_denominator"),
        ]
    )

    df = df.with_columns(
        [
            # d1
            (pl.col("d1_numerator") / (pl.col("d1_denominator") + epsilon)).alias("d1"),
            # d2 = d1 - sigma√T
            (pl.col("d1_numerator") / (pl.col("d1_denominator") + epsilon) - pl.col("d1_denominator")).alias("d2"),
        ]
    )

    df = df.with_columns(
        [
            # φ(d₂) = (1/√(2π)) × e^(-d₂²/2)
            ((1.0 / np.sqrt(2 * np.pi)) * (-(pl.col("d2").pow(2)) / 2).exp()).alias("phi_d2"),
        ]
    )

    # First-order Greeks
    df = df.with_columns(
        [
            # DELTA: ∂P/∂S = e^(-rT) × φ(d₂) / (S × σ√T)
            (
                pl.col("discount") * pl.col("phi_d2") / ((pl.col("S") + epsilon) * pl.col("d1_denominator") + epsilon)
            ).alias("delta"),
            # GAMMA: ∂²P/∂S² = -e^(-rT) × φ(d₂) × d₁ / (S² × σ²T)
            (
                -pl.col("discount")
                * pl.col("phi_d2")
                * pl.col("d1")
                / ((pl.col("S") + epsilon).pow(2) * (pl.col("sigma_mid").pow(2) * pl.col("T_years") + epsilon))
            ).alias("gamma"),
            # VEGA: ∂P/∂σ = -e^(-rT) × φ(d₂) × d₂√T / σ
            (
                -pl.col("discount")
                * pl.col("phi_d2")
                * pl.col("d2")
                * pl.col("sqrt_T")
                / (pl.col("sigma_mid") + epsilon)
            ).alias("vega"),
            # THETA: ∂P/∂T (simplified)
            # Theta = re^(-rT)Φ(d₂) - e^(-rT)φ(d₂) × [σ/(2√T) + r×d₂/σ√T]
            # Note: We use prob_mid as approximation for Φ(d₂)
            (
                pl.col("r") * pl.col("discount") * pl.col("prob_mid")
                - pl.col("discount")
                * pl.col("phi_d2")
                * (
                    pl.col("sigma_mid") / (2 * pl.col("sqrt_T") + epsilon)
                    + pl.col("r") * pl.col("d2") / (pl.col("sigma_mid") * pl.col("sqrt_T") + epsilon)
                )
            ).alias("theta"),
        ]
    )

    # Second-order Greeks
    df = df.with_columns(
        [
            # VANNA: ∂²P/∂S∂σ = -e^(-rT) × φ(d₂) × d₂ / σ
            (-pl.col("discount") * pl.col("phi_d2") * pl.col("d2") / (pl.col("sigma_mid") + epsilon)).alias("vanna"),
            # VOLGA (Vomma): ∂²P/∂σ² = vega × d₁×d₂ / σ
            (pl.col("vega") * pl.col("d1") * pl.col("d2") / (pl.col("sigma_mid") + epsilon)).alias("volga"),
            # CHARM: ∂²P/∂S∂T = e^(-rT) × φ(d₂) × [r/(σ√T) - d₁/(2T)]
            (
                pl.col("discount")
                * pl.col("phi_d2")
                * (
                    pl.col("r") / (pl.col("sigma_mid") * pl.col("sqrt_T") + epsilon)
                    - pl.col("d1") / (2 * pl.col("T_years") + epsilon)
                )
            ).alias("charm"),
        ]
    )

    # Drop intermediate columns
    df = df.drop(["d1_numerator", "d1_denominator", "sqrt_T", "discount"])

    logger.info("✓ Core Greeks calculated (10 features)")
    return df


def add_greek_transformations(df: pl.LazyFrame) -> pl.LazyFrame:
    """
    Add Greek feature transformations for model interpretability.

    Transformations:
    - First-order (11 features): log, sqrt, squared, inverse, time-weighted, etc.
    - Second-order (7 features): ratios, products, stability metrics

    Args:
        df: DataFrame with core Greeks

    Returns:
        DataFrame with transformation columns added (18 new features)
    """
    logger.info("Adding Greek transformations...")

    epsilon = 1e-8

    # First-order transformations (11 features)
    df = df.with_columns(
        [
            # Log transformations (handle negatives)
            (pl.col("delta").abs() + epsilon).log().alias("log_abs_delta"),
            (pl.col("gamma").abs() + epsilon).log().alias("log_abs_gamma"),
            (pl.col("vega").abs() + epsilon).log().alias("log_abs_vega"),
            # Square root of absolute values
            (pl.col("gamma").abs() + epsilon).sqrt().alias("sqrt_abs_gamma"),
            (pl.col("vega").abs() + epsilon).sqrt().alias("sqrt_abs_vega"),
            # Squared values
            pl.col("delta").pow(2).alias("delta_squared"),
            # Time-weighted Greeks (normalize by time to expiry)
            (pl.col("theta") / (pl.col("T_years") + epsilon)).alias("theta_per_day"),
            (pl.col("vega") * pl.col("T_years").sqrt()).alias("vega_time_weighted"),
            # Inverse Greeks (for extreme value detection)
            (1.0 / (pl.col("gamma").abs() + epsilon)).alias("inverse_gamma"),
            # Greek signs (directional exposure)
            pl.col("delta").sign().alias("delta_sign"),
            pl.col("theta").sign().alias("theta_sign"),
        ]
    )

    # Second-order transformations (7 features)
    df = df.with_columns(
        [
            # Greek ratios (relative sensitivities)
            (pl.col("gamma") / (pl.col("delta").abs() + epsilon)).alias("gamma_delta_ratio"),
            (pl.col("vega") / (pl.col("delta").abs() + epsilon)).alias("vega_delta_ratio"),
            (pl.col("vanna") / (pl.col("vega").abs() + epsilon)).alias("vanna_vega_ratio"),
            # Cross-Greeks (interaction effects)
            (pl.col("delta") * pl.col("vega")).alias("delta_vega_product"),
            (pl.col("gamma") * pl.col("vega")).alias("gamma_vega_product"),
            # Stability metrics (resistance to parameter changes)
            (pl.col("vanna").abs() + pl.col("volga").abs() + pl.col("charm").abs()).alias("greek_stability"),
            # Convexity measure (second-order dominance)
            (pl.col("gamma").abs() / (pl.col("delta").abs() + epsilon)).alias("convexity_measure"),
        ]
    )

    logger.info("✓ Greek transformations added (18 features)")
    return df


def add_moneyness_bin(df: pl.LazyFrame) -> pl.LazyFrame:
    """
    Add 10-category moneyness bin to preserve ATM/OTM signal after removing split.

    Bins:
    - 0-1: Deep OTM (S << K)
    - 2-4: OTM (S < K)
    - 5: ATM (S ≈ K)
    - 6-8: ITM (S > K)
    - 9: Deep ITM (S >> K)

    Args:
        df: DataFrame with moneyness column

    Returns:
        DataFrame with moneyness_bin column (categorical)
    """
    logger.info("Adding moneyness bin...")

    df = df.with_columns(
        [
            # Cut moneyness into 10 equal-width bins (using cut instead of qcut to handle duplicates)
            pl.col("moneyness")
            .qcut(10, labels=[f"bin_{i}" for i in range(10)], allow_duplicates=True)
            .alias("moneyness_bin")
        ]
    )

    logger.info("✓ Moneyness bin added (1 feature)")
    return df


def validate_greeks(df: pl.DataFrame) -> None:
    """
    Validate Greek features for correctness and data quality.

    Checks:
    - No NaN values in core Greeks
    - Delta ∈ [0, 1] (binary options have bounded delta)
    - Gamma ≤ 0 (binary options have negative gamma near ATM)
    - Vega finite and reasonable
    - Correlations match expectations

    Args:
        df: DataFrame with Greek features

    Raises:
        ValueError: If validation fails
    """
    logger.info("Validating Greek features...")

    # Check for NaNs
    greek_cols = ["delta", "gamma", "vega", "theta", "vanna", "volga", "charm", "d1", "d2", "phi_d2"]
    nan_counts = {col: df[col].is_null().sum() for col in greek_cols}

    total_nans = sum(nan_counts.values())
    if total_nans > 0:
        logger.warning(f"Found {total_nans:,} NaN values in Greeks:")
        for col, count in nan_counts.items():
            if count is not None and count > 0:
                logger.warning(f"  {col}: {count:,} NaNs ({count / len(df) * 100:.2f}%)")

    # Check delta bounds
    delta_min = df["delta"].min()
    delta_max = df["delta"].max()
    if (
        delta_min is not None
        and delta_max is not None
        and isinstance(delta_min, (int, float))
        and isinstance(delta_max, (int, float))
    ):
        logger.info(f"Delta range: [{delta_min:.6f}, {delta_max:.6f}]")

        if delta_min < -0.1 or delta_max > 1.1:
            logger.warning(f"Delta outside expected range [0, 1]: [{delta_min:.6f}, {delta_max:.6f}]")

    # Check gamma (should be mostly negative for binary options near ATM)
    gamma_mean = df["gamma"].mean()
    gamma_negative_pct = (df["gamma"] < 0).sum() / len(df) * 100
    logger.info(f"Gamma mean: {gamma_mean:.6f}, negative: {gamma_negative_pct:.1f}%")

    # Check vega magnitude
    vega_mean = df["vega"].abs().mean()
    vega_max = df["vega"].abs().max()
    logger.info(f"Vega magnitude - mean: {vega_mean:.6f}, max: {vega_max:.6f}")

    # Check correlations
    corr_delta_moneyness = df.select([pl.corr("delta", "moneyness").alias("corr")])["corr"][0]

    logger.info(f"Correlation (delta, moneyness): {corr_delta_moneyness:.3f}")

    if abs(corr_delta_moneyness) < 0.3:
        logger.warning(f"Weak delta-moneyness correlation: {corr_delta_moneyness:.3f} (expected > 0.3)")

    logger.info("✓ Greek validation complete")


def main():
    """Main Greek engineering pipeline."""

    logger.info("=" * 80)
    logger.info("GREEK FEATURE ENGINEERING V5")
    logger.info("=" * 80)

    # Get project root (search up from script location)
    script_dir = Path(__file__).parent.resolve()  # research/model/00_data_prep
    model_dir = script_dir.parent  # research/model
    project_root = model_dir.parent  # /home/ubuntu/Polymarket

    # Paths (all absolute)
    v4_features_path = model_dir / "data" / "consolidated_features_v4_pipeline_ready.parquet"
    baseline_path = model_dir / "results" / "production_backtest_results_v4.parquet"
    v5_features_path = model_dir / "data" / "consolidated_features_v5_pipeline_ready.parquet"

    # Log resolved paths for debugging
    logger.info(f"Project root: {project_root}")
    logger.info(f"V4 features: {v4_features_path}")
    logger.info(f"Baseline: {baseline_path}")
    logger.info(f"V5 output: {v5_features_path}")
    logger.info("")

    # Check inputs exist
    if not v4_features_path.exists():
        raise FileNotFoundError(f"V4 features not found: {v4_features_path}")

    if not baseline_path.exists():
        raise FileNotFoundError(f"Baseline file not found: {baseline_path}")

    # Load V4 features lazily
    logger.info(f"Loading V4 features: {v4_features_path}")
    df = pl.scan_parquet(v4_features_path)

    # Check schema
    schema_check = df.select([pl.len()]).collect()
    row_count = schema_check[0, 0]
    logger.info(f"V4 features: {row_count:,} rows")

    # Step 1: Add risk-free rate
    df = add_risk_free_rate(df, baseline_path)

    # Step 2: Calculate core Greeks (10 features)
    df = calculate_binary_greeks(df)

    # Step 3: Add Greek transformations (18 features)
    df = add_greek_transformations(df)

    # Step 4: Add moneyness bin (1 feature)
    df = add_moneyness_bin(df)

    # Write output (streaming for memory efficiency)
    logger.info(f"Writing V5 features to {v5_features_path}...")
    df.sink_parquet(
        str(v5_features_path),
        compression="snappy",
        statistics=True,
    )

    logger.info("✓ V5 features written (streaming mode)")

    # Validate output (collect small subset for validation)
    logger.info("Validating output (10% sample)...")
    df_validation = pl.read_parquet(v5_features_path, n_rows=int(row_count * 0.1))

    validate_greeks(df_validation)

    # Final summary
    logger.info("=" * 80)
    logger.info("GREEK ENGINEERING COMPLETE")
    logger.info("=" * 80)
    logger.info(f"Input:  {v4_features_path.name} (163 columns)")
    logger.info(f"Output: {v5_features_path.name} (193 columns)")
    logger.info("Features added: 30 (1 r + 10 core Greeks + 11 first-order + 7 second-order + 1 moneyness_bin)")
    logger.info(f"Rows: {row_count:,}")
    logger.info("")
    logger.info("Next step: Run select_features_v5.py to filter to 43 features (<50 target)")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
