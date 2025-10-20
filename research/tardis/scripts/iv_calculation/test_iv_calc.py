from py_vollib_vectorized import vectorized_implied_volatility
import polars as pl
import numpy as np

# Load a small sample for Oct 1, 2023
df = pl.scan_parquet('data/consolidated/quotes_1s_atm_short_dated_optimized.parquet')

# Get data from Oct 1, 2023, 00:00:00 UTC (to match existing IV data)
oct1_start = 1696118400
sample = df.filter(
    pl.col('timestamp_seconds') == oct1_start
).collect()

print('Testing IV calculation on Oct 1, 2023 data')
print('='*80)
print(f'Rows found at timestamp {oct1_start}: {len(sample)}')

if len(sample) > 0:
    # Filter for valid prices
    sample_valid = sample.filter(
        (pl.col('bid_price') > 0) & 
        (pl.col('ask_price') > 0) &
        (pl.col('time_to_expiry_days') > 0.001)  # At least ~1.5 hours
    )
    
    print(f'Valid rows for IV calc: {len(sample_valid)}')
    
    # Prepare data for IV calculation
    prices_bid = sample_valid['bid_price'].to_numpy()
    prices_ask = sample_valid['ask_price'].to_numpy()
    S = sample_valid['spot_price'].to_numpy()
    K = sample_valid['strike_price'].to_numpy()
    t = sample_valid['time_to_expiry_days'].to_numpy() / 365.25  # Convert to years
    r = np.full(len(sample_valid), 0.0412)  # Risk-free rate
    
    # Convert call/put to c/p for py_vollib
    flag = sample_valid['type'].str.replace('call', 'c').str.replace('put', 'p').to_numpy()
    
    # CRITICAL: Options are quoted in BTC, need to convert to USD
    prices_bid_usd = prices_bid * S
    prices_ask_usd = prices_ask * S
    
    print(f'\nSample data (first row):')
    print(f'  Symbol: {sample_valid["symbol"][0]}')
    print(f'  Type: {sample_valid["type"][0]}')
    print(f'  Bid: {prices_bid[0]:.6f} BTC (${prices_bid_usd[0]:.2f})')
    print(f'  Ask: {prices_ask[0]:.6f} BTC (${prices_ask_usd[0]:.2f})')
    print(f'  Spot: ${S[0]:.0f}')
    print(f'  Strike: ${K[0]:.0f}')
    print(f'  Time to expiry: {t[0]*365.25:.1f} days')
    print(f'  Moneyness: {sample_valid["moneyness"][0]:.4f}')
    
    # Calculate IVs for first few options
    print('\nCalculating IVs for first 5 options:')
    for i in range(min(5, len(sample_valid))):
        try:
            iv_bid = vectorized_implied_volatility(
                price=prices_bid_usd[i:i+1], 
                S=S[i:i+1], 
                K=K[i:i+1], 
                t=t[i:i+1], 
                r=r[i:i+1], 
                flag=flag[i:i+1], 
                model='black_scholes',
                return_as='numpy'
            )[0]
            
            iv_ask = vectorized_implied_volatility(
                price=prices_ask_usd[i:i+1], 
                S=S[i:i+1], 
                K=K[i:i+1], 
                t=t[i:i+1], 
                r=r[i:i+1], 
                flag=flag[i:i+1], 
                model='black_scholes',
                return_as='numpy'
            )[0]
            
            print(f'  {sample_valid["symbol"][i]:20s} IV bid: {iv_bid:.4f}, IV ask: {iv_ask:.4f}')
        except Exception as e:
            print(f'  {sample_valid["symbol"][i]:20s} Failed: {e}')
