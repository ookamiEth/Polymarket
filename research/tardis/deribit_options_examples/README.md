# Deribit Options Data - Tardis.dev Examples

This directory contains example scripts demonstrating how to fetch and analyze Deribit options data using Tardis.dev's APIs.

## Overview

Tardis.dev provides two main approaches for accessing historical cryptocurrency market data:

1. **CSV Download Approach** (`tardis-dev` package) - Normalized, ready-to-use CSV files
2. **Raw API Replay** (`tardis-client` package) - Exchange-native WebSocket message replay

## Free Access

Historical data for the **first day of each month** is available **without an API key**. This makes it perfect for testing and exploring the data format.

Example free dates: 2025-10-01, 2025-11-01, 2025-12-01, etc.

## Scripts

### 1. CSV Download Approach (`01_csv_download_approach.py`)

**Package:** `tardis-dev`
**Data Format:** Normalized CSV files

```python
from tardis_dev import datasets

datasets.download(
    exchange="deribit",
    data_types=["options_chain", "trades", "quotes"],
    from_date="2025-10-01",
    to_date="2025-10-01",
    symbols=["OPTIONS"],  # Special grouped symbol
    api_key=None  # Free access for first day of month
)
```

**Available Data Types for Options:**
- `options_chain` - Comprehensive options data (strikes, greeks, IV, prices)
- `trades` - Individual trade executions
- `quotes` - Best bid/ask quotes
- `book_snapshot_25` - Top 25 order book levels
- `book_snapshot_5` - Top 5 order book levels

**Pros:**
- ✅ Normalized format (same structure across all exchanges)
- ✅ Ready-to-use CSV files
- ✅ Easy to load with pandas
- ✅ Pre-aggregated options_chain data
- ✅ Fast download (no client-side processing)

**Cons:**
- ❌ Less flexibility (fixed data types)
- ❌ Cannot access raw exchange messages
- ❌ Limited to predefined CSV schemas

**When to Use:**
- Quick analysis and backtesting
- Working with aggregated data
- Standard data science workflows
- When you need normalized data across exchanges

---

### 2. Raw API Replay Approach (`02_raw_api_replay_approach.py`)

**Package:** `tardis-client`
**Data Format:** Exchange-native JSON (Deribit WebSocket messages)

```python
from tardis_client import TardisClient, Channel

tardis_client = TardisClient(api_key=None)

messages = tardis_client.replay(
    exchange="deribit",
    from_date="2025-10-01",
    to_date="2025-10-02",
    filters=[Channel(name="ticker", symbols=["BTC-PERPETUAL"])]
)

async for local_timestamp, message in messages:
    print(message)  # Raw Deribit WebSocket message
```

**Available Channels for Options:**
- `ticker` - Comprehensive ticker (greeks, IV, bid/ask, volume, OI)
- `trades` - Trade executions with IV
- `quote` - Best bid/ask quotes
- `book` - Full order book updates
- `markprice.options` - Mark prices
- `estimated_expiration_price` - Settlement estimates
- `deribit_volatility_index` - Volatility index

**Pros:**
- ✅ Full access to raw exchange data
- ✅ Highest granularity (tick-by-tick)
- ✅ All channels available
- ✅ Exact replication of live conditions
- ✅ Custom processing possible

**Cons:**
- ❌ Requires parsing exchange-specific formats
- ❌ More complex code
- ❌ Slower (client-side normalization if needed)
- ❌ Different format for each exchange

**When to Use:**
- Microstructure research
- Custom data processing
- Need tick-by-tick precision
- Accessing non-standard channels
- Replicating live trading conditions exactly

---

## Data Formats

### CSV Format (options_chain)

```csv
exchange,symbol,timestamp,local_timestamp,underlying_price,underlying_index,strike,expiration,option_type,bid,ask,last_price,delta,gamma,theta,vega,rho,implied_volatility,open_interest,volume
deribit,BTC-29NOV25-50000-C,1633046400000000,1633046400050000,50000.0,BTC-USD,50000,2025-11-29,call,0.065,0.067,0.066,0.55,0.00002,-12.5,45.2,3.2,65.5,1000,123.5
```

**Key Fields:**
- `symbol` - Option identifier (format: `{BASE}-{EXPIRY}-{STRIKE}-{TYPE}`)
- `timestamp` - Exchange timestamp (microseconds since epoch)
- `local_timestamp` - Arrival timestamp (microseconds since epoch)
- `underlying_price` - Current price of underlying asset
- Greeks: `delta`, `gamma`, `theta`, `vega`, `rho`
- `implied_volatility` - Implied volatility (%)
- `open_interest` - Total open interest
- `volume` - Trading volume (24h)

### Raw API Format (ticker channel)

```json
{
  "jsonrpc": "2.0",
  "method": "subscription",
  "params": {
    "channel": "ticker.BTC-29NOV25-50000-C.raw",
    "data": {
      "instrument_name": "BTC-29NOV25-50000-C",
      "underlying_price": 50000.0,
      "underlying_index": "BTC-USD",
      "timestamp": 1633046400000,
      "mark_price": 0.067,
      "mark_iv": 65.5,
      "last_price": 0.066,
      "open_interest": 1000,
      "greeks": {
        "delta": 0.55,
        "gamma": 0.00002,
        "theta": -12.5,
        "vega": 45.2,
        "rho": 3.2
      },
      "best_bid_price": 0.065,
      "best_bid_amount": 10.0,
      "best_ask_price": 0.067,
      "best_ask_amount": 15.0,
      "bid_iv": 64.0,
      "ask_iv": 67.0
    }
  }
}
```

---

## Deribit Option Symbol Format

Deribit uses the following format for option symbols:

```
{BASE}-{EXPIRY}-{STRIKE}-{TYPE}
```

**Examples:**
- `BTC-29NOV25-50000-C` - BTC Call option, expiring Nov 29 2025, strike $50,000
- `ETH-29NOV25-2000-P` - ETH Put option, expiring Nov 29 2025, strike $2,000
- `BTC-27DEC25-60000-C` - BTC Call option, expiring Dec 27 2025, strike $60,000

**Components:**
- `BASE` - Underlying asset (BTC, ETH, SOL, etc.)
- `EXPIRY` - Expiration date (DDMMMYY format)
- `STRIKE` - Strike price
- `TYPE` - Option type (C = Call, P = Put)

---

## Installation

### CSV Approach
```bash
pip install tardis-dev pandas
```

### Raw API Approach
The `tardis-client` package is already available locally at:
```
/Users/lgierhake/Documents/ETH/BT/research/tardis-python
```

Or install via pip:
```bash
pip install tardis-client
```

---

## Usage

### Run CSV Download Script
```bash
cd /Users/lgierhake/Documents/ETH/BT/research/deribit_options_examples
uv run python 01_csv_download_approach.py
```

### Run Raw API Replay Script
```bash
cd /Users/lgierhake/Documents/ETH/BT/research/deribit_options_examples
uv run python 02_raw_api_replay_approach.py
```

---

## Filtering for Short-Dated Options

Short-dated options are typically defined as options expiring within 7-30 days.

### From CSV:
```python
import pandas as pd
from datetime import datetime, timedelta

# Load CSV
df = pd.read_csv("deribit_options_chain_2025-10-01_OPTIONS.csv.gz")

# Parse expiry from symbol (BTC-29NOV25-50000-C)
def parse_expiry(symbol):
    parts = symbol.split('-')
    if len(parts) >= 4:
        expiry_str = parts[1]  # 29NOV25
        # Parse: DDMMMYY
        return datetime.strptime(expiry_str, '%d%b%y')
    return None

df['expiry_date'] = df['symbol'].apply(parse_expiry)
df['days_to_expiry'] = (df['expiry_date'] - datetime.now()).dt.days

# Filter for options expiring in 7-30 days
short_dated = df[(df['days_to_expiry'] >= 7) & (df['days_to_expiry'] <= 30)]
```

### From Raw API:
Filter messages by parsing the `instrument_name` field in the same way.

---

## API Documentation

- **Tardis.dev Docs:** https://docs.tardis.dev
- **Deribit Historical Data Details:** https://docs.tardis.dev/historical-data-details/deribit
- **Deribit WebSocket API:** https://docs.deribit.com/v2/#subscriptions
- **CSV Data Types:** https://docs.tardis.dev/downloadable-csv-files#data-types

---

## Notes

1. **Free Access Dates:** First day of each month is free without API key
2. **Date Format:** Use ISO 8601 format (YYYY-MM-DD)
3. **Time Zone:** All timestamps are in UTC
4. **Historical Data:** Deribit data available since 2019-03-30
5. **Options Data:** Available since 2019-03-30 for all Deribit options

---

## Common Use Cases

### 1. Backtesting Options Strategies
→ Use **CSV approach** with `options_chain` data type

### 2. Analyzing Option Order Flow
→ Use **Raw API** with `trades` and `book` channels

### 3. Greeks Analysis Over Time
→ Use **CSV approach** with `options_chain` or **Raw API** with `ticker` channel

### 4. Volatility Surface Construction
→ Use **CSV approach** with `options_chain` for all strikes/expiries

### 5. Market Microstructure Research
→ Use **Raw API** with `book` channel for full order book reconstruction

---

## Questions?

Refer to the Tardis.dev FAQ or contact support via email.

For free trials with full API access, contact Tardis.dev directly.
