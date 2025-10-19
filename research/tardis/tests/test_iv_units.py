#!/usr/bin/env python3

import numpy as np
from py_vollib_vectorized import vectorized_implied_volatility

# Take example from the output:
# Call: Strike=109000, Spot=108226.5, bid_price=0.0105 BTC, T=0.009126 years (~3.3 days)

S = 108226.5  # USD
K = 109000.0  # USD
T = 0.009126  # years
r = 0.0412

# WRONG approach (current code): Convert to USD
price_btc = 0.0105
price_usd_wrong = price_btc * S  # = 1136.38 USD

print("CURRENT (WRONG) APPROACH:")
print(f"Option price (BTC): {price_btc:.6f} BTC")
print(f"Option price (USD): {price_usd_wrong:.2f} USD  <- WRONG!")
print(f"Spot: ${S:.2f}")
print(f"Strike: ${K:.2f}")
print()

try:
    iv_wrong = vectorized_implied_volatility(
        price=np.array([price_usd_wrong]),
        S=np.array([S]),
        K=np.array([K]),
        t=np.array([T]),
        r=np.array([r]),
        flag=np.array(["c"]),
        model="black_scholes",
        return_as="numpy",
    )[0]
    print(f"IV (wrong): {iv_wrong:.6f} ({iv_wrong*100:.2f}%)  <- TOO LOW!")
except Exception as e:
    print(f"Error: {e}")

print()
print("=" * 80)
print()

# CORRECT approach: Keep everything in BTC
S_btc = 1.0  # BTC (normalized)
K_btc = K / S  # Strike in BTC terms = 109000 / 108226.5 = 1.00715

print("CORRECT APPROACH (BTC units):")
print(f"Option price (BTC): {price_btc:.6f} BTC")
print(f"Spot (BTC): {S_btc:.6f} BTC  <- Normalized to 1")
print(f"Strike (BTC): {K_btc:.6f} BTC  <- Strike/Spot ratio")
print()

try:
    iv_correct = vectorized_implied_volatility(
        price=np.array([price_btc]),
        S=np.array([S_btc]),
        K=np.array([K_btc]),
        t=np.array([T]),
        r=np.array([r]),
        flag=np.array(["c"]),
        model="black_scholes",
        return_as="numpy",
    )[0]
    print(f"IV (correct): {iv_correct:.6f} ({iv_correct*100:.2f}%)  <- CORRECT!")
except Exception as e:
    print(f"Error: {e}")

print()
print("=" * 80)
print()
print("KEY INSIGHT:")
print("Black-Scholes requires ALL inputs in the SAME units.")
print("Deribit quotes options in BTC, so we must use:")
print("  - Option prices: BTC")
print("  - Spot: BTC (normalized to 1.0)")
print("  - Strike: BTC (K_usd / S_usd)")
print()
print("The current code incorrectly converts option prices to USD")
print("while keeping strikes in USD, creating unit mismatch.")
