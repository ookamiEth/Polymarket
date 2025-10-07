#!/usr/bin/env python3
"""
Extract top 10 traders by PnL and by volume from trader statistics
"""

import polars as pl

# Load trader statistics
df = pl.read_parquet('research/historical_trading/trader_statistics.parquet')

# Top 10 by PnL
top_pnl = df.sort('total_pnl', descending=True).head(10)
top_pnl.write_parquet('research/historical_trading/top_10_traders_by_pnl.parquet')
print(f'✓ Created top_10_traders_by_pnl.parquet ({len(top_pnl)} traders)')

# Top 10 by Volume
top_volume = df.sort('total_volume_traded', descending=True).head(10)
top_volume.write_parquet('research/historical_trading/top_10_traders_by_volume.parquet')
print(f'✓ Created top_10_traders_by_volume.parquet ({len(top_volume)} traders)')

print('\nTop 10 by PnL:')
for row in top_pnl.select(['trader_address', 'total_pnl', 'total_volume_traded', 'markets_participated', 'win_rate']).iter_rows(named=True):
    addr = row['trader_address'][:10] + '...'
    pnl = row['total_pnl']
    vol = row['total_volume_traded']
    mkts = row['markets_participated']
    wr = row['win_rate'] * 100
    print(f'  {addr}: ${pnl:>10,.2f} | Vol: ${vol:>10,.2f} | Markets: {mkts:>4} | WR: {wr:>5.1f}%')

print('\nTop 10 by Volume:')
for row in top_volume.select(['trader_address', 'total_volume_traded', 'total_pnl', 'markets_participated', 'win_rate']).iter_rows(named=True):
    addr = row['trader_address'][:10] + '...'
    vol = row['total_volume_traded']
    pnl = row['total_pnl']
    mkts = row['markets_participated']
    wr = row['win_rate'] * 100
    print(f'  {addr}: ${vol:>10,.2f} | PnL: ${pnl:>10,.2f} | Markets: {mkts:>4} | WR: {wr:>5.1f}%')
