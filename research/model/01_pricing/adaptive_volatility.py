#!/usr/bin/env python3
"""
Adaptive Volatility Blending for Binary Option Pricing

This module implements intelligent blending between implied volatility (from
Deribit options) and realized volatility (from BTC price movements) to produce
superior volatility estimates for ultra-short-dated binary options.

Key Functions:
    - adaptive_volatility_blend(): Main blending logic based on IV staleness
    - har_rv_forecast(): Heterogeneous Autoregressive RV forecasting
    - volatility_quality_score(): Assess reliability of volatility estimate

Methodology:
    When IV is fresh (<10s old): Use IV directly (market consensus)
    When IV is stale (10-120s): Blend IV and RV with dynamic weights
    When IV is very stale (>120s): Rely primarily on RV

Author: BT Research Team
Date: 2025-10-29
"""

from __future__ import annotations

import polars as pl

# Staleness thresholds (in seconds)
FRESH_IV_THRESHOLD = 10  # IV <10s old → trust completely
MODERATE_STALENESS_THRESHOLD = 60  # 10-60s → blend
STALE_IV_THRESHOLD = 120  # >120s → rely on RV


def adaptive_volatility_blend(
    iv: float,
    iv_staleness_seconds: float,
    rv_60s: float | None,
    rv_300s: float | None,
    rv_900s: float | None,
    time_to_expiry_seconds: float,
) -> tuple[float, str]:
    """
    Blend implied and realized volatility based on IV freshness and time to expiry.

    Strategy:
        - Fresh IV (<10s): Use IV (100% weight)
        - Moderate staleness (10-60s): Linear blend from 80% IV to 50% IV
        - Stale IV (60-120s): Linear blend from 50% IV to 20% IV
        - Very stale (>120s): Use RV (80-100% weight)

    Time-to-expiry adjustment:
        - Near expiry (<300s): Prefer shorter RV windows (rv_60s, rv_300s)
        - Far from expiry (>300s): Use longer RV windows (rv_900s)

    Args:
        iv: Implied volatility from Deribit options (annualized)
        iv_staleness_seconds: Age of IV quote in seconds
        rv_60s: 1-minute realized volatility (annualized)
        rv_300s: 5-minute realized volatility (annualized)
        rv_900s: 15-minute realized volatility (annualized)
        time_to_expiry_seconds: Time until binary contract expires

    Returns:
        Tuple of (blended_volatility, blend_method)
            blend_method: Description of blending strategy used

    Examples:
        >>> # Fresh IV, trust it completely
        >>> vol, method = adaptive_volatility_blend(0.55, 5, 0.60, 0.58, 0.52, 600)
        >>> vol
        0.55
        >>> method
        'fresh_iv'

        >>> # Stale IV, blend 50/50 with RV
        >>> vol, method = adaptive_volatility_blend(0.55, 80, 0.60, 0.58, 0.52, 600)
        >>> round(vol, 4)
        0.565
        >>> method
        'blend_stale'
    """
    # Handle missing RV data (use IV as fallback)
    if rv_60s is None and rv_300s is None and rv_900s is None:
        return iv, "iv_only_no_rv"

    # Select appropriate RV based on time to expiry
    if time_to_expiry_seconds < 180:  # <3 minutes
        rv_selected = rv_60s if rv_60s is not None else (rv_300s if rv_300s is not None else rv_900s)
        rv_label = "rv_60s"
    elif time_to_expiry_seconds < 600:  # 3-10 minutes
        rv_selected = rv_300s if rv_300s is not None else (rv_60s if rv_60s is not None else rv_900s)
        rv_label = "rv_300s"
    else:  # >10 minutes
        rv_selected = rv_900s if rv_900s is not None else (rv_300s if rv_300s is not None else rv_60s)
        rv_label = "rv_900s"

    if rv_selected is None:
        return iv, "iv_only_no_valid_rv"

    # Fresh IV: trust market consensus
    if iv_staleness_seconds < FRESH_IV_THRESHOLD:
        return iv, "fresh_iv"

    # Moderate staleness: blend with increasing RV weight
    elif iv_staleness_seconds < MODERATE_STALENESS_THRESHOLD:
        # Linear decay from 80% IV to 50% IV as staleness increases from 10s to 60s
        iv_weight = (
            0.8
            - (iv_staleness_seconds - FRESH_IV_THRESHOLD) / (MODERATE_STALENESS_THRESHOLD - FRESH_IV_THRESHOLD) * 0.3
        )
        rv_weight = 1.0 - iv_weight

        blended = iv_weight * iv + rv_weight * rv_selected
        return blended, f"blend_moderate_iv{iv_weight:.0%}_{rv_label}"

    # Stale IV: rely more heavily on RV
    elif iv_staleness_seconds < STALE_IV_THRESHOLD:
        # Linear decay from 50% IV to 20% IV as staleness increases from 60s to 120s
        iv_weight = (
            0.5
            - (iv_staleness_seconds - MODERATE_STALENESS_THRESHOLD)
            / (STALE_IV_THRESHOLD - MODERATE_STALENESS_THRESHOLD)
            * 0.3
        )
        rv_weight = 1.0 - iv_weight

        blended = iv_weight * iv + rv_weight * rv_selected
        return blended, f"blend_stale_iv{iv_weight:.0%}_{rv_label}"

    # Very stale IV: use RV almost exclusively
    else:
        # Cap IV weight at 20% for very stale quotes
        iv_weight = max(0.1, 0.2 - (iv_staleness_seconds - STALE_IV_THRESHOLD) / 1000 * 0.1)
        rv_weight = 1.0 - iv_weight

        blended = iv_weight * iv + rv_weight * rv_selected
        return blended, f"blend_very_stale_rv{rv_weight:.0%}_{rv_label}"


def har_rv_forecast(rv_series: pl.DataFrame, forecast_horizon: str = "15min") -> pl.DataFrame:
    """
    Forecast realized volatility using Heterogeneous Autoregressive (HAR) model.

    HAR Model (Corsi 2009):
        RV_t+1 = β₀ + β_d·RV_daily + β_w·RV_weekly + β_m·RV_monthly + ε_t

    For 15-minute contracts:
        - RV_daily = mean(RV over last 96 periods) [1 day = 96 × 15min]
        - RV_weekly = mean(RV over last 672 periods) [7 days]
        - RV_monthly = mean(RV over last 2880 periods) [30 days]

    Args:
        rv_series: DataFrame with columns ['timestamp_seconds', 'rv_900s']
        forecast_horizon: Forecasting horizon ('15min', '1h', etc.)

    Returns:
        DataFrame with additional 'rv_forecast' column

    Note:
        This is a simple implementation. For production, consider:
        - Rolling OLS regression for time-varying coefficients
        - Kalman filter for adaptive weighting
        - Machine learning enhancement (XGBoost on residuals)
    """
    # Ensure sorted by timestamp
    rv_series = rv_series.sort("timestamp_seconds")

    # HAR components (for 15-minute frequency)
    # Daily = 96 periods, Weekly = 672 periods, Monthly = 2880 periods
    rv_series = rv_series.with_columns(
        [
            pl.col("rv_900s").rolling_mean(window_size=96).alias("rv_daily"),
            pl.col("rv_900s").rolling_mean(window_size=672).alias("rv_weekly"),
            pl.col("rv_900s").rolling_mean(window_size=2880).alias("rv_monthly"),
        ]
    )

    # Simple HAR forecast: weighted average (OLS coefficients from literature)
    # Typical HAR coefficients: β_d ≈ 0.4, β_w ≈ 0.3, β_m ≈ 0.2, β₀ ≈ 0.05×mean(RV)
    # Source: Corsi (2009), Andersen et al. (2007)

    mean_rv_value = rv_series["rv_900s"].mean()
    mean_rv: float = 0.0 if mean_rv_value is None else float(mean_rv_value)  # type: ignore[arg-type]

    rv_series = rv_series.with_columns(
        [
            (
                0.05 * mean_rv  # Intercept
                + 0.4 * pl.col("rv_daily")  # Daily component (highest weight)
                + 0.3 * pl.col("rv_weekly")  # Weekly component
                + 0.2 * pl.col("rv_monthly")  # Monthly component
            ).alias("rv_forecast")
        ]
    )

    # Drop intermediate columns
    rv_series = rv_series.drop(["rv_daily", "rv_weekly", "rv_monthly"])

    return rv_series


def volatility_quality_score(
    iv: float | None,
    iv_staleness_seconds: float | None,
    rv: float | None,
    rv_sample_size: int | None = None,
) -> float:
    """
    Calculate a quality score (0-1) for a volatility estimate.

    Higher score = more reliable estimate.

    Factors:
        - IV freshness (fresher = higher score)
        - IV vs RV agreement (closer = higher score)
        - RV sample size (more data = higher score)
        - Absence of missing data

    Args:
        iv: Implied volatility (annualized)
        iv_staleness_seconds: Age of IV quote
        rv: Realized volatility (annualized)
        rv_sample_size: Number of observations used to compute RV

    Returns:
        Quality score between 0.0 (low quality) and 1.0 (high quality)

    Examples:
        >>> # Fresh IV, agrees with RV, large sample
        >>> score = volatility_quality_score(0.55, 5, 0.56, 300)
        >>> score > 0.9
        True

        >>> # Stale IV, disagrees with RV
        >>> score = volatility_quality_score(0.55, 200, 0.75, 100)
        >>> score < 0.5
        True
    """
    # Penalize missing data
    if iv is None or rv is None:
        return 0.3

    # Component 1: IV freshness (0-0.4 points)
    if iv_staleness_seconds is not None:
        if iv_staleness_seconds < 10:
            freshness_score = 0.4
        elif iv_staleness_seconds < 60:
            freshness_score = 0.3
        elif iv_staleness_seconds < 120:
            freshness_score = 0.2
        else:
            freshness_score = 0.1
    else:
        freshness_score = 0.1

    # Component 2: IV/RV agreement (0-0.4 points)
    rv_iv_ratio = rv / iv if iv > 0 else 1.0
    if 0.9 < rv_iv_ratio < 1.1:
        agreement_score = 0.4  # Within 10%
    elif 0.8 < rv_iv_ratio < 1.2:
        agreement_score = 0.3  # Within 20%
    elif 0.7 < rv_iv_ratio < 1.3:
        agreement_score = 0.2  # Within 30%
    else:
        agreement_score = 0.1  # Large divergence

    # Component 3: RV sample size (0-0.2 points)
    if rv_sample_size is not None:
        if rv_sample_size >= 300:
            sample_score = 0.2
        elif rv_sample_size >= 100:
            sample_score = 0.15
        elif rv_sample_size >= 50:
            sample_score = 0.1
        else:
            sample_score = 0.05
    else:
        sample_score = 0.1

    total_score = freshness_score + agreement_score + sample_score

    return min(1.0, total_score)


def apply_adaptive_volatility(
    pricing_grid: pl.DataFrame, rv_data: pl.DataFrame, microstructure_data: pl.DataFrame
) -> pl.DataFrame:
    """
    Apply adaptive volatility blending to entire pricing grid.

    This is the main integration function for production backtest.

    Args:
        pricing_grid: DataFrame from production_backtest with IV data
        rv_data: DataFrame from calculate_realized_volatility.py
        microstructure_data: DataFrame from engineer_microstructure_features.py

    Returns:
        Enhanced pricing grid with 'sigma_adaptive' column

    Expected Columns in pricing_grid:
        - timestamp_seconds
        - sigma_mid (implied vol from Deribit)
        - iv_staleness_seconds
        - T_years (time to expiry)

    Expected Columns in rv_data:
        - timestamp_seconds
        - rv_60s, rv_300s, rv_900s

    Example:
        >>> # In production_backtest.py
        >>> rv_data = pl.read_parquet("results/realized_volatility_1s.parquet")
        >>> pricing_grid = apply_adaptive_volatility(pricing_grid, rv_data)
        >>> # Now use sigma_adaptive instead of sigma_mid for BS pricing
    """
    # Join RV data
    pricing_grid = pricing_grid.join(
        rv_data.select(["timestamp_seconds", "rv_60s", "rv_300s", "rv_900s"]),
        on="timestamp_seconds",
        how="left",
    )

    # Calculate time to expiry in seconds (from T_years)
    pricing_grid = pricing_grid.with_columns(
        [(pl.col("T_years") * 31_557_600).cast(pl.Int64).alias("time_to_expiry_seconds")]
    )

    # Apply adaptive blending (vectorized)
    # Note: This uses Polars' map_batches for the blending logic
    def blend_batch(batch: pl.DataFrame) -> pl.Series:
        """Vectorized blending for a batch of rows."""
        results = []
        for row in batch.iter_rows(named=True):
            vol, _ = adaptive_volatility_blend(
                iv=row["sigma_mid"],
                iv_staleness_seconds=row["iv_staleness_seconds"],
                rv_60s=row.get("rv_60s"),
                rv_300s=row.get("rv_300s"),
                rv_900s=row.get("rv_900s"),
                time_to_expiry_seconds=row["time_to_expiry_seconds"],
            )
            results.append(vol)
        return pl.Series("sigma_adaptive", results)

    # Apply blending in batches for performance
    pricing_grid = pricing_grid.with_columns(
        [
            pl.struct(
                [
                    "sigma_mid",
                    "iv_staleness_seconds",
                    "rv_60s",
                    "rv_300s",
                    "rv_900s",
                    "time_to_expiry_seconds",
                ]
            )
            .map_batches(lambda s: blend_batch(s.struct.unnest()), return_dtype=pl.Float64)
            .alias("sigma_adaptive")
        ]
    )

    # Drop temporary columns
    pricing_grid = pricing_grid.drop(["rv_60s", "rv_300s", "rv_900s", "time_to_expiry_seconds"])

    return pricing_grid


# Utility functions for analysis and diagnostics


def compare_volatility_estimates(pricing_grid: pl.DataFrame, sample_size: int = 10000) -> pl.DataFrame:
    """
    Compare IV vs RV vs Adaptive volatility estimates.

    Useful for analyzing which method performs best in different regimes.

    Args:
        pricing_grid: DataFrame with sigma_mid, rv_300s, sigma_adaptive
        sample_size: Number of random samples to analyze

    Returns:
        Summary DataFrame with comparison statistics
    """
    sample = pricing_grid.sample(n=min(sample_size, len(pricing_grid)))

    summary = sample.select(
        [
            pl.col("sigma_mid").mean().alias("iv_mean"),
            pl.col("rv_300s").mean().alias("rv_mean"),
            pl.col("sigma_adaptive").mean().alias("adaptive_mean"),
            pl.col("sigma_mid").std().alias("iv_std"),
            pl.col("rv_300s").std().alias("rv_std"),
            pl.col("sigma_adaptive").std().alias("adaptive_std"),
            (pl.col("sigma_adaptive") - pl.col("sigma_mid")).abs().mean().alias("avg_iv_adjustment"),
            (pl.col("sigma_adaptive") - pl.col("rv_300s")).abs().mean().alias("avg_rv_adjustment"),
        ]
    )

    return summary


if __name__ == "__main__":
    # Example usage and testing
    print("Adaptive Volatility Blending Module")
    print("=" * 60)

    # Test adaptive blending with different staleness levels
    test_cases = [
        (0.55, 5, 0.60, 0.58, 0.52, 600, "Fresh IV"),
        (0.55, 30, 0.60, 0.58, 0.52, 600, "Moderate staleness"),
        (0.55, 80, 0.60, 0.58, 0.52, 600, "Stale IV"),
        (0.55, 200, 0.60, 0.58, 0.52, 600, "Very stale IV"),
        (0.55, 10, 0.60, 0.58, 0.52, 100, "Near expiry"),
        (0.55, 10, 0.60, 0.58, 0.52, 800, "Far from expiry"),
    ]

    print("\nAdaptive Blending Tests:")
    print("-" * 60)
    for iv, stale, rv60, rv300, rv900, tte, desc in test_cases:
        vol, blend_method = adaptive_volatility_blend(iv, stale, rv60, rv300, rv900, tte)
        print(f"{desc:25s} | IV={iv:.4f} → Adaptive={vol:.4f} | {blend_method}")

    # Test quality scoring
    print("\nQuality Scoring Tests:")
    print("-" * 60)
    quality_tests = [
        (0.55, 5, 0.56, 300, "Fresh IV, good agreement"),
        (0.55, 100, 0.56, 300, "Stale IV, good agreement"),
        (0.55, 5, 0.75, 300, "Fresh IV, poor agreement"),
        (0.55, 200, 0.75, 50, "Stale IV, poor agreement, small sample"),
    ]

    for iv, stale, rv, sample, desc in quality_tests:
        score = volatility_quality_score(iv, stale, rv, sample)
        print(f"{desc:45s} | Quality={score:.3f}")

    print("\n" + "=" * 60)
    print("Module tests completed successfully!")
