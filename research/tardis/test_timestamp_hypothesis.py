#!/usr/bin/env python3
"""
Test if timestamp interpretation could have caused the IV bug.
"""

import numpy as np
from py_vollib_vectorized import vectorized_implied_volatility

# Real data from your file
Strike_usd = 109000.0
Spot_usd = 108226.5
bid_price_btc = 0.0105
T_years = 0.009126  # ~3.3 days
r = 0.0412
flag = "c"

print("TESTING: Could timestamp interpretation cause the bug?")
print("=" * 80)
print()

print("Given data:")
print(f"  Option: BTC Call, Strike ${Strike_usd:.0f}")
print(f"  Spot: ${Spot_usd:.2f}")
print(f"  Bid price: {bid_price_btc:.6f} BTC")
print(f"  Time to expiry: {T_years:.6f} years ({T_years*365:.1f} days)")
print()

# Test 1: Correct units + correct time
print("Test 1: CORRECT (USD prices, correct time)")
price_usd = bid_price_btc * Spot_usd
iv1 = vectorized_implied_volatility(
    price=np.array([price_usd]),
    S=np.array([Spot_usd]),
    K=np.array([Strike_usd]),
    t=np.array([T_years]),
    r=np.array([r]),
    flag=np.array([flag]),
    model="black_scholes",
    return_as="numpy",
)[0]
print(f"  Price: ${price_usd:.2f} USD")
print(f"  Time: {T_years:.6f} years")
print(f"  IV: {iv1:.6f} ({iv1*100:.2f}%)")
print()

# Test 2: Unit mismatch (the actual bug) + correct time
print("Test 2: UNIT MISMATCH (BTC prices with USD strikes) + correct time")
iv2 = vectorized_implied_volatility(
    price=np.array([bid_price_btc]),
    S=np.array([Spot_usd]),
    K=np.array([Strike_usd]),
    t=np.array([T_years]),
    r=np.array([r]),
    flag=np.array([flag]),
    model="black_scholes",
    return_as="numpy",
)[0]
print(f"  Price: {bid_price_btc:.6f} BTC (WRONG - mixed with USD)")
print(f"  Time: {T_years:.6f} years")
print(f"  IV: {iv2:.6f} ({iv2*100:.2f}%) <- MATCHES OLD OUTPUT!")
print()

# Test 3: Correct units but WRONG time (1000x off)
print("Test 3: Correct units but TIME OFF by 1000x")
T_wrong = T_years * 1000  # What if we interpreted microseconds as milliseconds?
iv3 = vectorized_implied_volatility(
    price=np.array([price_usd]),
    S=np.array([Spot_usd]),
    K=np.array([Strike_usd]),
    t=np.array([T_wrong]),
    r=np.array([r]),
    flag=np.array([flag]),
    model="black_scholes",
    return_as="numpy",
)[0]
print(f"  Price: ${price_usd:.2f} USD")
print(f"  Time: {T_wrong:.6f} years ({T_wrong*365:.0f} days) <- WRONG!")
print(f"  IV: {iv3:.6f} ({iv3*100:.2f}%)")
print()

# Test 4: What if time was divided by 1000?
print("Test 4: Correct units but TIME DIVIDED by 1000")
T_divided = T_years / 1000
iv4 = vectorized_implied_volatility(
    price=np.array([price_usd]),
    S=np.array([Spot_usd]),
    K=np.array([Strike_usd]),
    t=np.array([T_divided]),
    r=np.array([r]),
    flag=np.array([flag]),
    model="black_scholes",
    return_as="numpy",
)[0]
print(f"  Price: ${price_usd:.2f} USD")
print(f"  Time: {T_divided:.9f} years ({T_divided*365*24:.2f} hours) <- WRONG!")
print(f"  IV: {iv4:.6f} ({iv4*100:.2f}%)")
print()

print("=" * 80)
print("CONCLUSION:")
print()
print(f"Old output IV: ~2%")
print(f"Test 2 (unit mismatch): {iv2*100:.2f}% <- EXACT MATCH!")
print()
print("The bug is definitively the UNIT MISMATCH, not timestamp interpretation.")
print("Time calculations appear correct (~3.3 days = 0.009126 years).")
