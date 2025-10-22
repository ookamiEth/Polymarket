#!/usr/bin/env python3

"""
Check what the actual risk-free rates were in October 2023
"""

import polars as pl

rates_file = "/Users/lgierhake/Documents/ETH/BT/research/risk_free_rate/data/blended_lending_rates_2023_2025.parquet"

print("=" * 80)
print("RISK-FREE RATES IN OCTOBER 2023")
print("=" * 80)

# Load rates
df_rates = pl.read_parquet(rates_file)

# Filter for October 2023
from datetime import date

oct_rates = df_rates.filter(
    (pl.col("date") >= date(2023, 10, 1)) &
    (pl.col("date") <= date(2023, 10, 31))
).select([
    "date",
    "blended_supply_apr_ma7"
])

print("\nDaily risk-free rates (7-day MA of blended AAVE USDT/USDC rates):")
print(oct_rates)

print(f"\nStatistics:")
print(f"  Mean rate: {oct_rates['blended_supply_apr_ma7'].mean():.4f}%")
print(f"  Min rate: {oct_rates['blended_supply_apr_ma7'].min():.4f}%")
print(f"  Max rate: {oct_rates['blended_supply_apr_ma7'].max():.4f}%")
print(f"  Std dev: {oct_rates['blended_supply_apr_ma7'].std():.4f}%")

print(f"\nAssuming constant rate was 4.12%:")
print(f"  Average difference from constant: {oct_rates['blended_supply_apr_ma7'].mean() - 4.12:.4f}%")

# Calculate impact on 1.5 day option
avg_rate = oct_rates['blended_supply_apr_ma7'].mean()
time_fraction = 1.5 / 365
print(f"\nFor a 1.5 day option:")
print(f"  Time as fraction of year: {time_fraction:.6f}")
print(f"  Impact of 4.12% rate: {4.12 * time_fraction:.6f}%")
print(f"  Impact of {avg_rate:.2f}% rate: {avg_rate * time_fraction:.6f}%")
print(f"  Difference in impact: {(avg_rate - 4.12) * time_fraction:.6f}%")
print(f"  This translates to ~{abs((avg_rate - 4.12) * time_fraction * 0.5 / 100):.6f} IV points on a 50% IV option")