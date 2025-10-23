#!/usr/bin/env python3
"""
Demonstration of Binary Option Pricing Calculation

This script shows exactly how we calculate the price of a 15-minute
binary option on Bitcoin using Black-Scholes formula.

Run this to see the calculation step-by-step!
"""

import numpy as np
from scipy.stats import norm
from datetime import datetime, timedelta


def calculate_binary_option_price_detailed(S, K, r, sigma, T_seconds):
    """
    Calculate binary option price with detailed output.

    Args:
        S: Current BTC spot price
        K: Strike price (BTC price at contract open)
        r: Risk-free rate (annual, as decimal)
        sigma: Implied volatility (annual, as decimal)
        T_seconds: Time to expiry in seconds

    Returns:
        Price (probability) between 0 and 1
    """
    print("\n" + "="*60)
    print("BLACK-SCHOLES BINARY OPTION PRICING")
    print("="*60)

    # Input parameters
    print("\n📊 INPUT PARAMETERS:")
    print(f"  Current BTC Price (S): ${S:,.2f}")
    print(f"  Strike Price (K):      ${K:,.2f}")
    print(f"  Moneyness (S/K):       {S/K:.5f}")
    if S > K:
        print(f"  Status:                IN THE MONEY by ${S-K:.2f}")
    elif S < K:
        print(f"  Status:                OUT OF MONEY by ${K-S:.2f}")
    else:
        print(f"  Status:                AT THE MONEY")
    print(f"  Risk-Free Rate (r):    {r:.4f} ({r*100:.2f}% annual)")
    print(f"  Implied Vol (σ):       {sigma:.4f} ({sigma*100:.1f}% annual)")
    print(f"  Time Remaining:        {T_seconds} seconds ({T_seconds/60:.1f} minutes)")

    # Step 1: Convert time to years
    T_years = T_seconds / 31_557_600  # seconds in a year
    print(f"\n⏰ TIME CONVERSION:")
    print(f"  T = {T_seconds} seconds / 31,557,600 seconds/year")
    print(f"  T = {T_years:.8f} years")

    # Step 2: Calculate components of d2
    print(f"\n🧮 CALCULATING d₂:")

    # Component 1: Log moneyness
    log_moneyness = np.log(S / K)
    print(f"  ln(S/K) = ln({S}/{K}) = ln({S/K:.5f})")
    print(f"  ln(S/K) = {log_moneyness:.6f}")

    # Component 2: Drift adjustment
    drift = (r - 0.5 * sigma**2) * T_years
    print(f"\n  Drift = (r - σ²/2) × T")
    print(f"  Drift = ({r:.4f} - {sigma:.4f}²/2) × {T_years:.8f}")
    print(f"  Drift = ({r:.4f} - {0.5*sigma**2:.4f}) × {T_years:.8f}")
    print(f"  Drift = {r - 0.5*sigma**2:.4f} × {T_years:.8f}")
    print(f"  Drift = {drift:.8f}")

    # Component 3: Volatility scaling
    vol_scaled = sigma * np.sqrt(T_years)
    print(f"\n  Vol Scale = σ × √T")
    print(f"  Vol Scale = {sigma:.4f} × √{T_years:.8f}")
    print(f"  Vol Scale = {sigma:.4f} × {np.sqrt(T_years):.6f}")
    print(f"  Vol Scale = {vol_scaled:.6f}")

    # Calculate d2
    numerator = log_moneyness + drift
    d2 = numerator / vol_scaled

    print(f"\n  d₂ = (ln(S/K) + drift) / vol_scale")
    print(f"  d₂ = ({log_moneyness:.6f} + {drift:.8f}) / {vol_scaled:.6f}")
    print(f"  d₂ = {numerator:.6f} / {vol_scaled:.6f}")
    print(f"  d₂ = {d2:.4f}")

    # Step 3: Calculate probability using normal CDF
    prob = norm.cdf(d2)
    print(f"\n📈 PROBABILITY CALCULATION:")
    print(f"  N(d₂) = N({d2:.4f})")
    print(f"  N(d₂) = {prob:.6f}")
    print(f"  Probability = {prob*100:.2f}%")

    # Step 4: Apply discount (minimal for short periods)
    discount = np.exp(-r * T_years)
    print(f"\n💰 DISCOUNT FACTOR:")
    print(f"  Discount = e^(-r×T)")
    print(f"  Discount = e^(-{r:.4f} × {T_years:.8f})")
    print(f"  Discount = e^({-r*T_years:.8f})")
    print(f"  Discount = {discount:.8f}")
    print(f"  Impact: {(1-discount)*100:.4f}% reduction")

    # Final price
    price = discount * prob
    print(f"\n✅ FINAL BINARY OPTION PRICE:")
    print(f"  Price = Discount × Probability")
    print(f"  Price = {discount:.8f} × {prob:.6f}")
    print(f"  Price = {price:.6f}")
    print(f"  Price = ${price:.4f} (or {price*100:.2f}% chance of profit)")

    print("\n" + "="*60)

    return price


def run_example_scenarios():
    """Run several example scenarios to show how parameters affect price."""

    print("\n" + "🚀 "*20)
    print("BINARY OPTION PRICING EXAMPLES")
    print("🚀 "*20)

    # Common parameters
    r = 0.0427  # 4.27% risk-free rate

    # Scenario 1: At the money, 15 minutes left
    print("\n\n📍 SCENARIO 1: At The Money, Full 15 Minutes")
    print("-" * 60)
    S1, K1 = 27500, 27500
    sigma1 = 0.45
    T1 = 900  # 15 minutes
    price1 = calculate_binary_option_price_detailed(S1, K1, r, sigma1, T1)

    # Scenario 2: Slightly in the money, 10 minutes left
    print("\n\n📍 SCENARIO 2: In The Money, 10 Minutes Left")
    print("-" * 60)
    S2, K2 = 27520, 27500
    sigma2 = 0.45
    T2 = 600  # 10 minutes
    price2 = calculate_binary_option_price_detailed(S2, K2, r, sigma2, T2)

    # Scenario 3: Out of the money, 5 minutes left
    print("\n\n📍 SCENARIO 3: Out of The Money, 5 Minutes Left")
    print("-" * 60)
    S3, K3 = 27480, 27500
    sigma3 = 0.45
    T3 = 300  # 5 minutes
    price3 = calculate_binary_option_price_detailed(S3, K3, r, sigma3, T3)

    # Scenario 4: Deep in the money, 1 minute left
    print("\n\n📍 SCENARIO 4: Deep In The Money, 1 Minute Left")
    print("-" * 60)
    S4, K4 = 27550, 27500
    sigma4 = 0.45
    T4 = 60  # 1 minute
    price4 = calculate_binary_option_price_detailed(S4, K4, r, sigma4, T4)

    # Scenario 5: High volatility effect
    print("\n\n📍 SCENARIO 5: High Volatility (80% annual), ATM, 15 Minutes")
    print("-" * 60)
    S5, K5 = 27500, 27500
    sigma5 = 0.80  # 80% volatility!
    T5 = 900  # 15 minutes
    price5 = calculate_binary_option_price_detailed(S5, K5, r, sigma5, T5)

    # Summary
    print("\n\n" + "📊 "*20)
    print("SUMMARY OF ALL SCENARIOS")
    print("📊 "*20)
    print("\n| Scenario | Spot | Strike | Status | Time Left | Vol | Price |")
    print("|----------|------|--------|--------|-----------|-----|-------|")
    print(f"| 1 | ${S1} | ${K1} | ATM | 15 min | {sigma1*100:.0f}% | {price1*100:.1f}% |")
    print(f"| 2 | ${S2} | ${K2} | +${S2-K2} ITM | 10 min | {sigma2*100:.0f}% | {price2*100:.1f}% |")
    print(f"| 3 | ${S3} | ${K3} | -${K3-S3} OTM | 5 min | {sigma3*100:.0f}% | {price3*100:.1f}% |")
    print(f"| 4 | ${S4} | ${K4} | +${S4-K4} ITM | 1 min | {sigma4*100:.0f}% | {price4*100:.1f}% |")
    print(f"| 5 | ${S5} | ${K5} | ATM | 15 min | {sigma5*100:.0f}% | {price5*100:.1f}% |")

    print("\n" + "💡 "*20)
    print("KEY INSIGHTS:")
    print("💡 "*20)
    print("""
    1. AT THE MONEY: Price near 50% when S ≈ K
    2. TIME DECAY: Less time → prices converge to 0 or 100%
    3. MONEYNESS: ITM → higher price, OTM → lower price
    4. VOLATILITY: Higher vol → prices stay closer to 50%
    5. DISCOUNT: Minimal impact for 15-minute options
    """)


def interactive_calculator():
    """Interactive calculator for custom inputs."""
    print("\n" + "🖩 "*20)
    print("INTERACTIVE BINARY OPTION CALCULATOR")
    print("🖩 "*20)

    print("\nEnter your parameters (or press Enter for defaults):")

    try:
        S = float(input("Current BTC Price [$27,500]: ") or 27500)
        K = float(input("Strike Price [$27,500]: ") or 27500)
        sigma = float(input("Implied Volatility % [45]: ") or 45) / 100
        T_seconds = float(input("Seconds Remaining [900]: ") or 900)
        r = float(input("Risk-Free Rate % [4.27]: ") or 4.27) / 100

        price = calculate_binary_option_price_detailed(S, K, r, sigma, T_seconds)

        print(f"\n🎯 TRADING DECISION:")
        if price > 0.55:
            print(f"  Signal: BUY (Price {price*100:.1f}% > 55%)")
        elif price < 0.45:
            print(f"  Signal: SELL (Price {price*100:.1f}% < 45%)")
        else:
            print(f"  Signal: NO TRADE (Price {price*100:.1f}% too close to 50%)")

    except ValueError:
        print("Invalid input! Please enter numbers only.")
    except KeyboardInterrupt:
        print("\nCalculator terminated.")


def main():
    """Main entry point."""
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "--interactive":
        interactive_calculator()
    else:
        run_example_scenarios()
        print("\n" + "="*60)
        print("Run with --interactive for custom calculations:")
        print("  python single_calculation_demo.py --interactive")
        print("="*60)


if __name__ == "__main__":
    main()