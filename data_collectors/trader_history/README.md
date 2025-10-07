# Trader Historical Data Collector

Collect comprehensive trading data for individual Polymarket traders.

## Features

- **Trades**: Up to 10,000 most recent trades
- **Activity**: Complete activity timeline (trades, redeems, yields, etc.)
- **Current Positions**: All open positions with P&L
- **Closed Positions**: Historical closed positions (sample)

## Usage

```bash
uv run python trader_historical_data_collector.py
```

## Data Collected

### Trades (10K Limit)
- Market, outcome, side, size, price
- Timestamp, transaction hash
- Date range: ~67 days of recent history

### Activity Timeline
- TRADE, REDEEM, YIELD, CONVERSION, REWARD, MERGE, SPLIT events
- Complete transaction history
- On-chain verification data

### Positions
- Current open positions with unrealized P&L
- Closed positions with realized P&L
- Portfolio value and ROI metrics

## Output

Data saved to `../../data/traders/` in Parquet format:
- `trader_trades.parquet`
- `trader_activity.parquet`
- `trader_current_positions.parquet`
- `trader_closed_positions.parquet`

## API Limitations

**Data API Offset Limit (August 2025):**
- Trades endpoint: 10,000 maximum offset
- Activity endpoint: 10,000 maximum offset

For unlimited history, use Polymarket subgraphs (see main docs).

## Configuration

Edit trader address in script:
```python
TRADER_ADDRESS = "0x24c8cf69a0e0a17eee21f69d29752bfa32e823e1"
```

## Dependencies

```bash
uv pip install polars requests
```
