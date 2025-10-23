#!/usr/bin/env python3
"""
Black-Scholes Binary Option Pricing.

Implements the core pricing formulas for binary (digital) options that pay $1 if
the underlying finishes above the strike, $0 otherwise.

Key formulas:
    Price = e^(-rT) × N(d₂)
    d₂ = [ln(S/K) + (r - σ²/2)T] / (σ√T)

Where:
    S = Spot price
    K = Strike price
    r = Risk-free rate (annual, as decimal)
    σ = Implied volatility (annual, as decimal)
    T = Time to expiry (in years)
    N(d₂) = Cumulative normal distribution
"""

# ruff: noqa: N803
# Allow uppercase variable names (S, K, T) as they are standard mathematical notation in finance

from typing import Union

import numpy as np
import polars as pl
from scipy.stats import norm

# Constants
SECONDS_PER_YEAR = 31_557_600  # 365.25 × 24 × 60 × 60


def normal_cdf(x: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    """
    Cumulative normal distribution N(x).

    Args:
        x: Input value(s)

    Returns:
        Probability that standard normal variable ≤ x
    """
    return norm.cdf(x)


def calculate_d2(
    S: Union[float, np.ndarray],
    K: Union[float, np.ndarray],
    r: Union[float, np.ndarray],
    sigma: Union[float, np.ndarray],
    T: Union[float, np.ndarray],
) -> Union[float, np.ndarray]:
    """
    Calculate d₂ parameter for Black-Scholes binary option.

    Formula: d₂ = [ln(S/K) + (r - σ²/2)T] / (σ√T)

    Args:
        S: Spot price
        K: Strike price
        r: Risk-free rate (annual, decimal)
        sigma: Implied volatility (annual, decimal)
        T: Time to expiry (years)

    Returns:
        d₂ value(s)

    Note:
        All inputs can be scalars or numpy arrays.
        When T → 0, function handles edge cases gracefully.
    """
    # Numerator: ln(S/K) + (r - σ²/2)T
    log_moneyness = np.log(S / K)
    drift_adjustment = (r - 0.5 * sigma**2) * T
    numerator = log_moneyness + drift_adjustment

    # Denominator: σ√T
    denominator = sigma * np.sqrt(T)

    # Handle near-zero T (avoid division by zero)
    # When T → 0, d₂ → ±∞ depending on moneyness
    d2 = np.where(
        denominator > 1e-10,  # Normal case
        numerator / denominator,
        np.where(
            S > K,  # If ITM and T ≈ 0
            1e10,  # Very large positive (N(d₂) ≈ 1)
            -1e10,  # Very large negative (N(d₂) ≈ 0)
        ),
    )

    return d2


def binary_option_price(
    S: Union[float, np.ndarray],
    K: Union[float, np.ndarray],
    r: Union[float, np.ndarray],
    sigma: Union[float, np.ndarray],
    T: Union[float, np.ndarray],
) -> Union[float, np.ndarray]:
    """
    Price a binary option using Black-Scholes formula.

    Formula: Price = e^(-rT) × N(d₂)

    Args:
        S: Spot price
        K: Strike price
        r: Risk-free rate (annual, decimal)
        sigma: Implied volatility (annual, decimal)
        T: Time to expiry (years)

    Returns:
        Binary option price (0 to 1)

    Example:
        >>> # ATM binary with 10 minutes left, 40% vol, 5% rate
        >>> S, K = 50000, 50000
        >>> r, sigma = 0.05, 0.40
        >>> T = 600 / 31_557_600  # 10 minutes in years
        >>> price = binary_option_price(S, K, r, sigma, T)
        >>> # price ≈ 0.50 (50% probability when ATM)
    """
    # Calculate d₂
    d2 = calculate_d2(S, K, r, sigma, T)

    # Calculate N(d₂)
    prob = normal_cdf(d2)

    # Apply discount factor e^(-rT)
    discount = np.exp(-r * T)

    # Binary price
    price = discount * prob

    return price


def seconds_to_years(seconds: Union[int, float, np.ndarray]) -> Union[float, np.ndarray]:
    """
    Convert seconds to years.

    Args:
        seconds: Time in seconds

    Returns:
        Time in years (365.25 day year)

    Example:
        >>> seconds_to_years(900)  # 15 minutes
        2.85e-05
    """
    return seconds / SECONDS_PER_YEAR


def add_binary_pricing_columns(
    df: pl.DataFrame,
    spot_col: str = "S",
    strike_col: str = "K",
    rate_col: str = "r",
    sigma_col: str = "sigma",
    time_seconds_col: str = "T_seconds",
) -> pl.DataFrame:
    """
    Add Black-Scholes binary pricing columns to Polars DataFrame.

    This function adds the following columns:
    - T_years: Time to expiry in years
    - d2: The d₂ parameter
    - prob: Risk-neutral probability N(d₂)
    - discount: Discount factor e^(-rT)
    - price: Binary option price

    Args:
        df: Input DataFrame with required columns
        spot_col: Column name for spot price (S)
        strike_col: Column name for strike price (K)
        rate_col: Column name for risk-free rate (r)
        sigma_col: Column name for volatility (sigma)
        time_seconds_col: Column name for time to expiry in seconds

    Returns:
        DataFrame with additional pricing columns

    Example:
        >>> df = pl.DataFrame({
        ...     "S": [50000, 50100],
        ...     "K": [50000, 50000],
        ...     "r": [0.05, 0.05],
        ...     "sigma": [0.40, 0.40],
        ...     "T_seconds": [900, 600],
        ... })
        >>> df_priced = add_binary_pricing_columns(df)
        >>> # df_priced now has columns: T_years, d2, prob, discount, price
    """
    # Convert time to years
    df = df.with_columns([(pl.col(time_seconds_col) / SECONDS_PER_YEAR).alias("T_years")])

    # Calculate d₂ components (vectorized)
    df = df.with_columns(
        [
            # ln(S/K)
            (pl.col(spot_col) / pl.col(strike_col)).log().alias("log_moneyness"),
            # (r - σ²/2)T
            ((pl.col(rate_col) - 0.5 * pl.col(sigma_col) ** 2) * pl.col("T_years")).alias("drift_adjustment"),
            # σ√T
            (pl.col(sigma_col) * pl.col("T_years").sqrt()).alias("vol_sqrt_t"),
        ]
    )

    # Calculate d₂
    # Handle edge case where T → 0 (vol_sqrt_t → 0)
    df = df.with_columns(
        [
            pl.when(pl.col("vol_sqrt_t") > 1e-10)
            .then((pl.col("log_moneyness") + pl.col("drift_adjustment")) / pl.col("vol_sqrt_t"))
            .when(pl.col(spot_col) > pl.col(strike_col))
            .then(pl.lit(1e10))  # ITM with T≈0
            .otherwise(pl.lit(-1e10))  # OTM with T≈0
            .alias("d2")
        ]
    )

    # Apply normal CDF to get probability
    # Note: Polars doesn't have built-in normal CDF, so we use map_batches
    df = df.with_columns([pl.col("d2").map_batches(lambda s: norm.cdf(s.to_numpy())).alias("prob")])

    # Calculate discount factor
    df = df.with_columns([((-pl.col(rate_col) * pl.col("T_years")).exp()).alias("discount")])

    # Final price
    df = df.with_columns([(pl.col("discount") * pl.col("prob")).alias("price")])

    return df


def add_binary_pricing_bid_ask_mid(
    df: pl.DataFrame,
    spot_col: str = "S",
    strike_col: str = "K",
    rate_col: str = "r",
    sigma_bid_col: str = "sigma_bid",
    sigma_ask_col: str = "sigma_ask",
    time_seconds_col: str = "T_seconds",
) -> pl.DataFrame:
    """
    Add binary pricing for bid, ask, and mid volatilities.

    Creates three sets of pricing columns:
    - *_bid: Using bid volatility
    - *_ask: Using ask volatility
    - *_mid: Using (bid + ask) / 2

    Args:
        df: Input DataFrame
        spot_col: Spot price column
        strike_col: Strike price column
        rate_col: Risk-free rate column
        sigma_bid_col: Bid volatility column
        sigma_ask_col: Ask volatility column
        time_seconds_col: Time to expiry (seconds)

    Returns:
        DataFrame with pricing columns: d2_bid, d2_ask, d2_mid, prob_bid, prob_ask,
        prob_mid, price_bid, price_ask, price_mid

    Example:
        >>> df = pl.DataFrame({
        ...     "S": [50000],
        ...     "K": [50000],
        ...     "r": [0.05],
        ...     "sigma_bid": [0.38],
        ...     "sigma_ask": [0.42],
        ...     "T_seconds": [900],
        ... })
        >>> df_priced = add_binary_pricing_bid_ask_mid(df)
        >>> # Now has price_bid, price_ask, price_mid columns
    """
    # Calculate mid volatility
    df = df.with_columns([((pl.col(sigma_bid_col) + pl.col(sigma_ask_col)) / 2).alias("sigma_mid")])

    # Convert time to years
    df = df.with_columns([(pl.col(time_seconds_col) / SECONDS_PER_YEAR).alias("T_years")])

    # Calculate pricing for each volatility
    for suffix, sigma_col in [("bid", sigma_bid_col), ("ask", sigma_ask_col), ("mid", "sigma_mid")]:
        # Calculate d₂ components
        df = df.with_columns(
            [
                # ln(S/K)
                (pl.col(spot_col) / pl.col(strike_col)).log().alias("log_moneyness"),
                # (r - σ²/2)T
                ((pl.col(rate_col) - 0.5 * pl.col(sigma_col) ** 2) * pl.col("T_years")).alias("drift_adjustment"),
                # σ√T
                (pl.col(sigma_col) * pl.col("T_years").sqrt()).alias("vol_sqrt_t"),
            ]
        )

        # Calculate d₂
        df = df.with_columns(
            [
                pl.when(pl.col("vol_sqrt_t") > 1e-10)
                .then((pl.col("log_moneyness") + pl.col("drift_adjustment")) / pl.col("vol_sqrt_t"))
                .when(pl.col(spot_col) > pl.col(strike_col))
                .then(pl.lit(1e10))
                .otherwise(pl.lit(-1e10))
                .alias(f"d2_{suffix}")
            ]
        )

        # Apply normal CDF
        df = df.with_columns(
            [pl.col(f"d2_{suffix}").map_batches(lambda s: norm.cdf(s.to_numpy())).alias(f"prob_{suffix}")]
        )

    # Calculate discount factor (same for all)
    df = df.with_columns([((-pl.col(rate_col) * pl.col("T_years")).exp()).alias("discount")])

    # Final prices
    df = df.with_columns(
        [
            (pl.col("discount") * pl.col("prob_bid")).alias("price_bid"),
            (pl.col("discount") * pl.col("prob_ask")).alias("price_ask"),
            (pl.col("discount") * pl.col("prob_mid")).alias("price_mid"),
        ]
    )

    # Clean up temporary columns
    df = df.drop(["log_moneyness", "drift_adjustment", "vol_sqrt_t"])

    return df


# Example usage and tests
if __name__ == "__main__":
    print("Black-Scholes Binary Option Pricing Module")
    print("=" * 60)

    # Test 1: Scalar pricing
    print("\nTest 1: Scalar Pricing")
    S, K = 50000.0, 50000.0  # ATM
    r, sigma = 0.05, 0.40  # 5% rate, 40% vol
    T_seconds = 900  # 15 minutes
    T_years = seconds_to_years(T_seconds)

    price = binary_option_price(S, K, r, sigma, T_years)
    print(f"Spot: ${S:,.0f}, Strike: ${K:,.0f}")
    print(f"Time: {T_seconds}s ({T_years:.6f} years)")
    print(f"Vol: {sigma * 100:.0f}%, Rate: {r * 100:.1f}%")
    print(f"Price: ${price:.4f} ({price * 100:.2f}%)")
    print("Expected: ~$0.50 (ATM should be ~50%)")

    # Test 2: ITM pricing
    print("\nTest 2: ITM Pricing")
    S_itm = 50500.0  # 1% ITM
    price_itm = binary_option_price(S_itm, K, r, sigma, T_years)
    print(f"Spot: ${S_itm:,.0f}, Strike: ${K:,.0f} ({(S_itm / K - 1) * 100:.1f}% ITM)")
    print(f"Price: ${price_itm:.4f} ({price_itm * 100:.2f}%)")
    print("Expected: >50% (ITM should have higher win prob)")

    # Test 3: OTM pricing
    print("\nTest 3: OTM Pricing")
    S_otm = 49500.0  # 1% OTM
    price_otm = binary_option_price(S_otm, K, r, sigma, T_years)
    print(f"Spot: ${S_otm:,.0f}, Strike: ${K:,.0f} ({(S_otm / K - 1) * 100:.1f}% OTM)")
    print(f"Price: ${price_otm:.4f} ({price_otm * 100:.2f}%)")
    print("Expected: <50% (OTM should have lower win prob)")

    # Test 4: Vectorized DataFrame pricing
    print("\nTest 4: Vectorized DataFrame Pricing")
    df_test = pl.DataFrame(
        {
            "S": [50000.0, 50500.0, 49500.0],
            "K": [50000.0, 50000.0, 50000.0],
            "r": [0.05, 0.05, 0.05],
            "sigma_bid": [0.38, 0.38, 0.38],
            "sigma_ask": [0.42, 0.42, 0.42],
            "T_seconds": [900, 900, 900],
        }
    )

    df_priced = add_binary_pricing_bid_ask_mid(df_test)
    print(df_priced.select(["S", "K", "T_seconds", "price_bid", "price_mid", "price_ask"]))

    print("\n✅ All tests complete!")
