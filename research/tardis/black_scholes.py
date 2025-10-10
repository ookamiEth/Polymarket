"""
Vectorized Black-Scholes Option Pricing and Greeks Calculator

All calculations are vectorized using NumPy and Polars for maximum performance.
NO for loops - operates on entire DataFrames at once.

Formulas based on Black-Scholes-Merton model (1973).
"""

import numpy as np
import polars as pl
from scipy.stats import norm
from scipy.optimize import brentq
from typing import Literal


# ============================================================================
# CORE BLACK-SCHOLES FORMULAS (Vectorized with NumPy)
# ============================================================================


def _d1(S, K, T, r, sigma):
    """Calculate d1 parameter (vectorized)."""
    return (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))


def _d2(S, K, T, r, sigma):
    """Calculate d2 parameter (vectorized)."""
    return _d1(S, K, T, r, sigma) - sigma * np.sqrt(T)


def black_scholes_call(S, K, T, r, sigma):
    """
    Black-Scholes call option price (vectorized).

    Args:
        S: Underlying price (array or scalar)
        K: Strike price (array or scalar)
        T: Time to expiry in years (array or scalar)
        r: Risk-free rate (array or scalar)
        sigma: Volatility as decimal (array or scalar), e.g., 0.5 for 50%

    Returns:
        Call option price (same shape as inputs)
    """
    d1 = _d1(S, K, T, r, sigma)
    d2 = _d2(S, K, T, r, sigma)
    return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)


def black_scholes_put(S, K, T, r, sigma):
    """
    Black-Scholes put option price (vectorized).

    Args:
        S: Underlying price (array or scalar)
        K: Strike price (array or scalar)
        T: Time to expiry in years (array or scalar)
        r: Risk-free rate (array or scalar)
        sigma: Volatility as decimal (array or scalar)

    Returns:
        Put option price (same shape as inputs)
    """
    d1 = _d1(S, K, T, r, sigma)
    d2 = _d2(S, K, T, r, sigma)
    return K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)


# ============================================================================
# IMPLIED VOLATILITY SOLVER (Element-wise with vectorization wrapper)
# ============================================================================


def _solve_iv_single(price, S, K, T, r, option_type, min_iv=0.001, max_iv=5.0):
    """
    Solve for implied volatility for a single option.

    Uses Brent's method for root finding.
    """
    if np.isnan(price) or price <= 0 or T <= 0:
        return np.nan

    # Select pricing function
    if option_type == 'call':
        bs_func = black_scholes_call
    else:
        bs_func = black_scholes_put

    # Objective function: difference between market and model price
    def objective(sigma):
        return bs_func(S, K, T, r, sigma) - price

    try:
        # Brent's method: searches for IV between min_iv and max_iv
        iv = brentq(objective, min_iv, max_iv, maxiter=100)
        return iv
    except (ValueError, RuntimeError):
        # Failed to converge or no solution in range
        return np.nan


def calculate_implied_volatility_vectorized(
    prices: np.ndarray,
    S: np.ndarray,
    K: np.ndarray,
    T: np.ndarray,
    r: np.ndarray,
    option_types: np.ndarray,
    min_iv: float = 0.001,
    max_iv: float = 5.0
) -> np.ndarray:
    """
    Calculate implied volatility for multiple options (vectorized).

    Args:
        prices: Option prices (NumPy array)
        S: Underlying prices (NumPy array)
        K: Strike prices (NumPy array)
        T: Times to expiry in years (NumPy array)
        r: Risk-free rates (NumPy array)
        option_types: Option types ('call' or 'put') (NumPy array of strings)
        min_iv: Minimum IV to search (default 0.1%)
        max_iv: Maximum IV to search (default 500%)

    Returns:
        NumPy array of implied volatilities (as decimals)
    """
    n = len(prices)
    ivs = np.empty(n)

    # Vectorize over array (scipy.optimize.brentq is not vectorized)
    for i in range(n):
        ivs[i] = _solve_iv_single(
            prices[i], S[i], K[i], T[i], r[i], option_types[i], min_iv, max_iv
        )

    return ivs


# ============================================================================
# GREEKS CALCULATOR (Vectorized with NumPy)
# ============================================================================


def calculate_delta_vectorized(S, K, T, r, sigma, option_type):
    """
    Calculate Delta (vectorized).

    Delta = ∂V/∂S (rate of change of option price with respect to underlying)
    """
    d1 = _d1(S, K, T, r, sigma)

    if option_type == 'call':
        return norm.cdf(d1)
    else:  # put
        return norm.cdf(d1) - 1


def calculate_gamma_vectorized(S, K, T, r, sigma):
    """
    Calculate Gamma (vectorized).

    Gamma = ∂²V/∂S² (rate of change of Delta)
    Same for calls and puts.
    """
    d1 = _d1(S, K, T, r, sigma)
    return norm.pdf(d1) / (S * sigma * np.sqrt(T))


def calculate_vega_vectorized(S, K, T, r, sigma):
    """
    Calculate Vega (vectorized).

    Vega = ∂V/∂σ (sensitivity to volatility)
    Same for calls and puts.
    """
    d1 = _d1(S, K, T, r, sigma)
    return S * norm.pdf(d1) * np.sqrt(T)


def calculate_theta_vectorized(S, K, T, r, sigma, option_type):
    """
    Calculate Theta (vectorized).

    Theta = ∂V/∂t (time decay)
    """
    d1 = _d1(S, K, T, r, sigma)
    d2 = _d2(S, K, T, r, sigma)

    if option_type == 'call':
        theta = (
            -(S * norm.pdf(d1) * sigma) / (2 * np.sqrt(T))
            - r * K * np.exp(-r * T) * norm.cdf(d2)
        )
    else:  # put
        theta = (
            -(S * norm.pdf(d1) * sigma) / (2 * np.sqrt(T))
            + r * K * np.exp(-r * T) * norm.cdf(-d2)
        )

    return theta


def calculate_rho_vectorized(S, K, T, r, sigma, option_type):
    """
    Calculate Rho (vectorized).

    Rho = ∂V/∂r (sensitivity to interest rate)
    """
    d2 = _d2(S, K, T, r, sigma)

    if option_type == 'call':
        return K * T * np.exp(-r * T) * norm.cdf(d2)
    else:  # put
        return -K * T * np.exp(-r * T) * norm.cdf(-d2)


# ============================================================================
# POLARS DATAFRAME INTERFACE (Vectorized)
# ============================================================================


def add_implied_volatility_to_dataframe(
    df: pl.DataFrame,
    price_column: str,
    output_column: str,
    underlying_price_column: str = 'underlying_price',
    strike_column: str = 'strike_price',
    time_to_expiry_column: str = 'time_to_expiry',  # in years
    risk_free_rate_column: str = 'risk_free_rate',
    option_type_column: str = 'type',
    min_iv: float = 0.001,
    max_iv: float = 5.0,
) -> pl.DataFrame:
    """
    Add implied volatility column to Polars DataFrame (vectorized).

    Args:
        df: Input DataFrame with option data
        price_column: Column name containing option prices (e.g., 'bid_price')
        output_column: Column name for output IV (e.g., 'bid_iv')
        underlying_price_column: Column with underlying prices
        strike_column: Column with strike prices
        time_to_expiry_column: Column with time to expiry in years
        risk_free_rate_column: Column with risk-free rates
        option_type_column: Column with option types ('call' or 'put')
        min_iv: Minimum IV to search
        max_iv: Maximum IV to search

    Returns:
        DataFrame with new IV column added
    """
    # Extract arrays from DataFrame
    prices = df[price_column].to_numpy()
    S = df[underlying_price_column].to_numpy()
    K = df[strike_column].to_numpy()
    T = df[time_to_expiry_column].to_numpy()
    r = df[risk_free_rate_column].to_numpy()
    option_types = df[option_type_column].to_numpy()

    # Calculate IV vectorized
    ivs = calculate_implied_volatility_vectorized(
        prices, S, K, T, r, option_types, min_iv, max_iv
    )

    # Convert to percentage and add to DataFrame
    df = df.with_columns([
        pl.lit(ivs * 100).alias(output_column)  # Convert to percentage
    ])

    return df


def add_greeks_to_dataframe(
    df: pl.DataFrame,
    iv_column: str = 'mark_iv',  # IV to use for Greeks calculation
    underlying_price_column: str = 'underlying_price',
    strike_column: str = 'strike_price',
    time_to_expiry_column: str = 'time_to_expiry',
    risk_free_rate_column: str = 'risk_free_rate',
    option_type_column: str = 'type',
) -> pl.DataFrame:
    """
    Add Greeks columns to Polars DataFrame (vectorized).

    Args:
        df: Input DataFrame
        iv_column: Column containing IV (as percentage)
        ... (other params same as above)

    Returns:
        DataFrame with delta, gamma, vega, theta, rho columns added
    """
    # Extract arrays
    S = df[underlying_price_column].to_numpy()
    K = df[strike_column].to_numpy()
    T = df[time_to_expiry_column].to_numpy()
    r = df[risk_free_rate_column].to_numpy()
    sigma = df[iv_column].to_numpy() / 100  # Convert percentage to decimal
    option_types = df[option_type_column].to_numpy()

    # Calculate Greeks vectorized
    # Need to handle calls and puts separately
    is_call = option_types == 'call'

    delta = np.where(
        is_call,
        calculate_delta_vectorized(S, K, T, r, sigma, 'call'),
        calculate_delta_vectorized(S, K, T, r, sigma, 'put')
    )

    gamma = calculate_gamma_vectorized(S, K, T, r, sigma)
    vega = calculate_vega_vectorized(S, K, T, r, sigma)

    theta = np.where(
        is_call,
        calculate_theta_vectorized(S, K, T, r, sigma, 'call'),
        calculate_theta_vectorized(S, K, T, r, sigma, 'put')
    )

    rho = np.where(
        is_call,
        calculate_rho_vectorized(S, K, T, r, sigma, 'call'),
        calculate_rho_vectorized(S, K, T, r, sigma, 'put')
    )

    # Add to DataFrame
    df = df.with_columns([
        pl.lit(delta).alias('delta'),
        pl.lit(gamma).alias('gamma'),
        pl.lit(vega).alias('vega'),
        pl.lit(theta).alias('theta'),
        pl.lit(rho).alias('rho'),
    ])

    return df
