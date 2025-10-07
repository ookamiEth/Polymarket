# Historical CLOB Order Book Data Collection

This module provides comprehensive tools for collecting and analyzing historical order book data from Polymarket's Central Limit Order Book (CLOB).

## Overview

Polymarket does not provide a historical order book API endpoint. This module solves that problem through two approaches:

1. **Real-time WebSocket Collection**: Capture order book snapshots and updates as they happen
2. **Historical Trade Reconstruction**: Approximate book states from historical trade data

## Features

- ✅ Real-time order book snapshot collection via WebSocket
- ✅ Incremental price change tracking
- ✅ Trade execution monitoring
- ✅ Historical trade data fetching (REST API)
- ✅ Order book state reconstruction
- ✅ Time-series analysis and statistics
- ✅ Data persistence (Parquet format)

## Installation

```bash
# Install required dependencies
uv pip install websockets pandas pyarrow requests
```

## Quick Start

### 1. Collect Real-Time Data

```python
from orderbook_collector import OrderBookCollector
import asyncio

async def main():
    collector = OrderBookCollector(output_dir="data/orderbooks")

    # Example: Trump Yes token
    market_id = "0xdd31ce870876bb410d71f29e3dfea77910bc3effb81a0fb8dc8c282d5968aa5a"
    token_id = "21742633143463906290569050155826241533067272736897614950488156847949938836455"

    # Collect for 1 hour, saving every 60 seconds
    await collector.collect_realtime(
        markets=[market_id],
        token_ids=[token_id],
        duration_minutes=60,
        save_interval_seconds=60
    )

asyncio.run(main())
```

### 2. Fetch Historical Trades

```python
from orderbook_collector import OrderBookCollector
import time

collector = OrderBookCollector()

# Get trades from last 7 days
end_time = int(time.time())
start_time = end_time - (7 * 24 * 60 * 60)

trades_df = collector.collect_historical_trades(
    market="0xdd31ce870876bb410d71f29e3dfea77910bc3effb81a0fb8dc8c282d5968aa5a",
    start_timestamp=start_time,
    end_timestamp=end_time,
    limit=10000
)

print(f"Collected {len(trades_df)} trades")
```

### 3. Reconstruct Order Book States

```python
from orderbook_reconstructor import OrderBookReconstructor

reconstructor = OrderBookReconstructor()

# Load collected data
reconstructor.load_snapshots("data/orderbooks/book_snapshots_*.parquet")
reconstructor.load_price_changes("data/orderbooks/price_changes_*.parquet")

# Build time series (1 second intervals)
asset_id = "21742633143463906290569050155826241533067272736897614950488156847949938836455"
states = reconstructor.build_timeseries(asset_id=asset_id, interval_seconds=1)

# Analyze statistics
stats_df = reconstructor.analyze_book_statistics(states)
print(stats_df.describe())

# Export
reconstructor.export_timeseries(states, "orderbook_timeseries.parquet")
```

## Data Structure

### Book Snapshots
Captured from WebSocket `book` events:
- `timestamp`: Collection time (ISO format)
- `ws_timestamp`: Server timestamp (milliseconds)
- `asset_id`: Token ID
- `market`: Condition ID
- `hash`: Order book hash
- `bids`: List of `{price, size}` (sorted high to low)
- `asks`: List of `{price, size}` (sorted low to high)

### Price Changes
Captured from WebSocket `price_change` events:
- `timestamp`: Collection time
- `ws_timestamp`: Server timestamp
- `market`: Condition ID
- `asset_id`: Token ID
- `side`: "BUY" or "SELL"
- `price`: Price level
- `size`: New size (0 = removed)
- `best_bid`: Current best bid
- `best_ask`: Current best ask
- `hash`: Order hash

### Trades
Captured from WebSocket `last_trade_price` events or REST API:
- `timestamp`: Trade time
- `asset_id`: Token ID
- `market`: Condition ID
- `side`: "BUY" or "SELL"
- `price`: Execution price
- `size`: Trade size
- `fee_rate_bps`: Fee in basis points

## Data Collection Strategies

### Strategy 1: Continuous Real-Time Collection (Recommended)
Start collecting now and build historical database over time.

**Pros:**
- Complete order book snapshots
- All price changes captured
- Accurate timestamps
- Real-time analysis possible

**Cons:**
- Need to run continuously
- No historical data before start

**Use case:** Market making, high-frequency analysis, microstructure research

### Strategy 2: Historical Trade Reconstruction
Fetch historical trades and approximate book states.

**Pros:**
- Can get data from the past
- No continuous collection needed

**Cons:**
- Only approximation of book state
- Missing order placements/cancellations
- Less accurate

**Use case:** Backtesting, historical analysis, trend research

### Strategy 3: Hybrid Approach
Combine historical trades with real-time collection.

**Best practice:**
1. Fetch historical trades for context
2. Start real-time collection
3. Use reconstructor to build complete time series

## WebSocket Events

The collector captures these WebSocket events:

### 1. `book` - Full Order Book Snapshot
Emitted when:
- First subscribing to a market
- After a trade affects the book

### 2. `price_change` - Incremental Update
Emitted when:
- New order placed
- Order cancelled

### 3. `last_trade_price` - Trade Execution
Emitted when:
- Maker and taker orders matched

### 4. `tick_size_change` - Minimum Tick Adjustment
Emitted when:
- Book price reaches limits (>0.96 or <0.04)

## Analysis Examples

### Calculate Spread Over Time
```python
stats_df = reconstructor.analyze_book_statistics(states)
print(stats_df[['timestamp', 'spread', 'mid_price']])
```

### Get Book Depth
```python
state = states[0]
depth = state.get_depth(levels=5)
print(f"Top 5 bids: {depth['bids']}")
print(f"Top 5 asks: {depth['asks']}")
```

### Monitor Best Bid/Ask
```python
for state in states:
    print(f"{state.timestamp}: "
          f"Bid ${state.get_best_bid()[0]} / "
          f"Ask ${state.get_best_ask()[0]}")
```

## Storage Format

Data is saved in Apache Parquet format for efficient storage and querying:

```
data/orderbooks/
├── book_snapshots_20250103_120000.parquet
├── price_changes_20250103_120000.parquet
├── trades_20250103_120000.parquet
└── reconstructed_timeseries.parquet
```

## Limitations

### No Direct Historical API
Polymarket does not provide:
- Historical order book snapshots
- Time-travel queries
- Archived book states

### WebSocket Only for Real-Time
- Must collect going forward
- Cannot retrieve past snapshots
- Need continuous connection

### Trade Reconstruction Approximation
- Missing non-trade events
- Can't capture all order activity
- Less accurate than actual snapshots

## API Endpoints Used

### WebSocket
- **Market Channel**: `wss://ws-subscriptions-clob.polymarket.com/ws/market`
  - Public (no authentication)
  - Subscribe by asset_ids or markets

### REST
- **Trades**: `GET https://data-api.polymarket.com/trades`
  - Query params: market, start, end, limit, offset
  - Historical trade data

- **Order Book**: `GET https://clob.polymarket.com/book`
  - Query params: token_id
  - Current snapshot only

## Best Practices

1. **Start Collecting Early**: Begin real-time collection as soon as possible to build historical database

2. **Save Frequently**: Use `save_interval_seconds=60` to avoid data loss

3. **Monitor Connection**: WebSocket can disconnect - implement reconnection logic for production

4. **Rate Limiting**: REST API calls should be rate-limited (0.5s between requests)

5. **Storage Management**: Parquet files compress well but accumulate - archive periodically

6. **Validation**: Check order book hash to detect missed updates

## Production Considerations

For production use:

1. **Reconnection Logic**: Auto-reconnect WebSocket on disconnect
2. **Error Handling**: Robust error handling and logging
3. **Database Storage**: Use TimescaleDB or InfluxDB for time-series
4. **Monitoring**: Alert on missed snapshots or gaps
5. **Backup**: Regular backup of collected data
6. **Scalability**: Multiple collectors for different markets

## Future Enhancements

Potential improvements:
- [ ] Database integration (PostgreSQL/TimescaleDB)
- [ ] Real-time visualization dashboard
- [ ] Order book imbalance indicators
- [ ] Volume profile analysis
- [ ] Liquidity heatmaps
- [ ] Automated reconnection
- [ ] Multi-market collection
- [ ] Data quality metrics

## Troubleshooting

### WebSocket Won't Connect
```python
# Check connection manually
import websockets
import asyncio

async def test():
    async with websockets.connect("wss://ws-subscriptions-clob.polymarket.com/ws/market") as ws:
        print("Connected!")

asyncio.run(test())
```

### No Data Collected
- Check market_id and token_id are valid
- Verify market is active (has trading activity)
- Check output directory permissions

### Large File Sizes
- Reduce `save_interval_seconds` for smaller files
- Use compression in Parquet writer
- Archive old data

## References

- [Polymarket CLOB Docs](https://docs.polymarket.com/developers/CLOB/introduction)
- [WebSocket Market Channel](https://docs.polymarket.com/developers/CLOB/websocket/market-channel)
- [REST API Endpoints](https://docs.polymarket.com/developers/CLOB/endpoints)

## License

This code is provided for research and educational purposes.
