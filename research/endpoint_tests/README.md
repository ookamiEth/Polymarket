# Polymarket API Endpoint Tests

Comprehensive test scripts for all Polymarket historical data endpoints.

## 📋 Overview

This directory contains Python test scripts that demonstrate how to use each Polymarket API endpoint. Each script makes actual API calls and displays the response structure.

## 🚀 Quick Start

### Install Dependencies

```bash
pip install -r requirements.txt
```

Or using uv (if you're in a uv environment):

```bash
uv pip install -r requirements.txt
```

### Run Individual Tests

```bash
# Price history
python test_price_history.py

# Trades
python test_trades.py
python test_clob_trades.py

# User positions
python test_positions.py
python test_closed_positions.py

# Activity timeline
python test_activity.py

# Top holders
python test_holders.py

# Order book
python test_orderbook.py

# Markets and events
python test_markets.py
python test_events.py

# WebSocket channels
python test_websocket_market.py
python test_websocket_user.py
```

### Run with uv

```bash
uv run python test_price_history.py
```

## 📁 Test Scripts

| Script | Endpoint | Description |
|--------|----------|-------------|
| `test_price_history.py` | `/prices-history` | Historical price data with custom time ranges |
| `test_trades.py` | `/trades` | Trade history from Data API |
| `test_clob_trades.py` | `/data/trades` | Detailed trades from CLOB API (requires auth) |
| `test_positions.py` | `/positions` | Current positions with P&L |
| `test_closed_positions.py` | `/closed-positions` | Historical closed positions |
| `test_activity.py` | `/activity` | Complete activity timeline |
| `test_holders.py` | `/holders` | Top holders for markets |
| `test_orderbook.py` | `/book` | Current order book snapshot |
| `test_markets.py` | `/markets` | Market metadata with volume |
| `test_events.py` | `/events` | Event metadata with aggregated data |
| `test_websocket_market.py` | WebSocket Market | Real-time market updates |
| `test_websocket_user.py` | WebSocket User | User-specific updates (demo) |

## 🔑 Authentication

### Public Endpoints (No Auth Required)

Most test scripts work without authentication:
- Price history
- Trades (Data API)
- Positions
- Activity
- Holders
- Order book
- Markets
- Events
- WebSocket Market channel

### Authenticated Endpoints

These endpoints require authentication:
- `test_clob_trades.py` - CLOB trades (L2 headers)
- `test_websocket_user.py` - User WebSocket channel (API credentials)

## 📊 What Each Test Shows

### Price History (`test_price_history.py`)

- ✅ Custom time ranges
- ✅ Multiple resolutions (fidelity parameter)
- ✅ Preset intervals (1d, 1w, 1m, etc.)
- ✅ Price statistics calculation

**Key Output:**
- Timestamp/price pairs
- Price min/max/average
- Price change percentages

### Trades (`test_trades.py`)

- ✅ Trade history for markets
- ✅ Filter by side (BUY/SELL)
- ✅ Multiple market queries
- ✅ Pagination

**Key Output:**
- Trade size, price, timestamp
- Trader address
- Transaction hash
- Outcome

### Positions (`test_positions.py`)

- ✅ Current positions
- ✅ Unrealized P&L tracking
- ✅ Filter by size
- ✅ Sort by various metrics

**Key Output:**
- Position size
- Entry/current price
- P&L (cash and percentage)
- Realized P&L

### Activity (`test_activity.py`)

- ✅ Complete activity timeline
- ✅ Multiple activity types
- ✅ Filter by type/time
- ✅ Transaction hashes

**Key Output:**
- TRADE, SPLIT, MERGE, REDEEM events
- Timestamps
- On-chain verification data

### Order Book (`test_orderbook.py`)

- ✅ Current order book snapshot
- ✅ Bid/ask levels
- ✅ Liquidity calculation
- ✅ Spread analysis

**Key Output:**
- Bid/ask price levels
- Sizes at each level
- Best bid/ask
- Total liquidity

### Markets & Events

- ✅ Market metadata
- ✅ Volume breakdowns (24hr, 1wk, 1mo, 1yr)
- ✅ Volume by system (AMM vs CLOB)
- ✅ Price changes
- ✅ Current prices

**Key Output:**
- Volume metrics
- Liquidity data
- Price change history
- Timestamps

### WebSocket Channels

- ✅ Real-time market updates
- ✅ Order book changes
- ✅ Trade executions
- ✅ Price updates

**Key Output:**
- Live order book
- Trade events
- Price changes
- Tick size adjustments

## 💡 Usage Examples

### Get 1-Minute Price History

```python
import requests
from datetime import datetime, timedelta

token_id = "21742633143463906290569050155826241533067272736897614950488156847949938836455"
now = int(datetime.now().timestamp())
day_ago = now - (24 * 60 * 60)

response = requests.get(
    "https://clob.polymarket.com/prices-history",
    params={
        "market": token_id,
        "startTs": day_ago,
        "endTs": now,
        "fidelity": 1  # 1-minute resolution
    }
)

history = response.json()["history"]
print(f"Data points: {len(history)}")
```

### Get User's Trading History

```python
import requests

user = "0x56687bf447db6ffa42ffe2204a05edaa20f55839"

response = requests.get(
    "https://data-api.polymarket.com/trades",
    params={
        "user": user,
        "limit": 100,
        "sortBy": "TIMESTAMP",
        "sortDirection": "DESC"
    }
)

trades = response.json()
for trade in trades[:10]:
    print(f"{trade['side']} {trade['size']} @ ${trade['price']}")
```

### Stream Real-Time Market Data

```python
import asyncio
import websockets
import json

async def stream_market():
    uri = "wss://ws-subscriptions-clob.polymarket.com/ws/market"
    market_id = "0xdd31ce870876bb410d71f29e3dfea77910bc3effb81a0fb8dc8c282d5968aa5a"
    token_id = "21742633143463906290569050155826241533067272736897614950488156847949938836455"

    async with websockets.connect(uri) as ws:
        await ws.send(json.dumps({
            "auth": {},
            "markets": [market_id],
            "assets_ids": [token_id],
            "type": "market"
        }))

        while True:
            msg = await ws.recv()
            data = json.loads(msg)
            print(f"Event: {data['event_type']}")

asyncio.run(stream_market())
```

## ⚡ Rate Limits

| Endpoint | Limit | Notes |
|----------|-------|-------|
| Price History | 100 req / 10s | CLOB API |
| Trades (Data API) | 100 req / 10s | Public data |
| Positions | 100 req / 10s | Per user |
| Activity | 100 req / 10s | Per user |
| Order Book | 200 req / 10s | Real-time data |
| Markets | 100 req / 10s | Gamma API |
| Events | 100 req / 10s | Gamma API |

### Rate Limit Best Practices

1. Add delays between requests:
   ```python
   import time
   time.sleep(0.11)  # 100ms between requests
   ```

2. Batch requests when possible
3. Use WebSocket for real-time data instead of polling
4. Cache responses when appropriate

## 🐛 Troubleshooting

### Import Errors

```bash
# Install missing dependencies
pip install requests websockets python-dotenv
```

### Connection Timeouts

- Check your internet connection
- Polymarket API might be experiencing issues
- Try increasing timeout: `timeout=30`

### 401 Unauthorized

- Endpoint requires authentication
- Check if API credentials are needed
- See CLOB trades and User WebSocket documentation

### Empty Responses

- User address might not have data
- Market might not exist
- Try with example addresses/markets from tests

### WebSocket Disconnections

- Implement reconnection logic
- Add heartbeat/ping mechanism
- Handle connection errors gracefully

## 📚 Additional Resources

- **Main Documentation**: `../polymarket_historical_data_endpoints.md`
- **Official Docs**: https://docs.polymarket.com
- **Python Client**: `py_clob_client`
- **TypeScript Client**: `@polymarket/clob-client`

## 🔍 Response Structure Examples

All test scripts print detailed response structures. Look for sections titled:
- "Response Structure Analysis"
- "Field Descriptions"
- "Full JSON Sample"

These sections show exactly what data is returned and how it's structured.

## 🎯 Use Cases

### Deep Data Analysis

1. **Price Analysis**: Use `test_price_history.py` for tick-level data
2. **Trade Flow**: Use `test_trades.py` and `test_activity.py`
3. **Portfolio Tracking**: Use `test_positions.py` and `test_closed_positions.py`
4. **Market Microstructure**: Use `test_orderbook.py` and WebSocket tests
5. **Volume Analysis**: Use `test_markets.py` and `test_events.py`

### Real-Time Applications

1. **Live Prices**: `test_websocket_market.py`
2. **Order Monitoring**: `test_websocket_user.py`
3. **Market Making**: Combine order book + WebSocket
4. **Risk Management**: Activity + positions + WebSocket

## 📝 Notes

- All timestamps are Unix timestamps (seconds since epoch)
- Some timestamps are in milliseconds (noted in comments)
- Prices are decimals between 0 and 1
- Addresses are 0x-prefixed, 40 hex characters
- Condition IDs are 0x-prefixed, 64 hex characters
- Token IDs are large integers (256-bit)

## 🤝 Contributing

Feel free to add more test scripts or improve existing ones:
1. Follow the existing naming convention
2. Include clear documentation
3. Print response structures
4. Add error handling

---

**Last Updated**: October 1, 2025
