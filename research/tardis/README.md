# Tardis.dev - Deribit Options Data

Complete guide and examples for downloading and analyzing Deribit BTC options data using Tardis.dev.

## Quick Start

### 1. Understand the Packages
**Read:** [`TARDIS_PACKAGES_EXPLAINED.md`](TARDIS_PACKAGES_EXPLAINED.md)

**Two packages available:**
- **`tardis-dev`** - Download CSV files (recommended for most use cases)
- **`tardis-client`** - Replay raw API (advanced use cases)

**⭐ NEW: Advanced Topics:**
- **[`DATA_SOURCES_COMPARED.md`](DATA_SOURCES_COMPARED.md)** - Deep dive: Same data, different formats
- **[`CUSTOM_PARQUET_PIPELINE.py`](CUSTOM_PARQUET_PIPELINE.py)** - Build custom pipeline with msgspec + parquet
- **[`WHEN_TO_USE_WHAT.md`](WHEN_TO_USE_WHAT.md)** - Decision guide: CSV vs Parquet

### 2. Understand What You Get
**Read:** [`WHAT_YOU_GET_BTC_OPTIONS.md`](WHAT_YOU_GET_BTC_OPTIONS.md)

**When you download OPTIONS data for October 1, 2025:**
- ✅ Entire day's tick-by-tick timeseries (00:00:00 to 23:59:59 UTC)
- ✅ Every ticker update as it happens (not aggregated)
- ✅ ALL options (BTC, ETH, SOL, etc.) - filter for BTC afterward
- ✅ ~500K - 5M rows for BTC options on one day

### 3. See Example Data
**File:** [`example_deribit_options_chain_2025-10-01_BTC_SAMPLE.csv`](example_deribit_options_chain_2025-10-01_BTC_SAMPLE.csv)

Sample CSV showing realistic BTC options data with:
- Multiple strikes and expiries
- Time progression throughout the day
- All columns: greeks, IV, prices, OI, etc.

### 4. Run Filtering Script
**Script:** [`filter_btc_options.py`](filter_btc_options.py)

Complete workflow demonstrating:
1. Download OPTIONS data (or use example)
2. Filter for BTC options only
3. Parse option symbols
4. Filter for short-dated options
5. Analyze greeks, IV, pricing
6. Export results

```bash
cd /Users/lgierhake/Documents/ETH/BT/research/tardis
uv run python filter_btc_options.py
```

---

## Files in This Directory

```
/research/tardis/
├── README.md                                          # This file
│
├── # BASIC GUIDES
├── TARDIS_PACKAGES_EXPLAINED.md                       # Package comparison
├── WHAT_YOU_GET_BTC_OPTIONS.md                        # Data structure explained
├── filter_btc_options.py                              # Filtering & analysis script
├── example_deribit_options_chain_2025-10-01_BTC_SAMPLE.csv  # Example data
│
├── # ADVANCED GUIDES (NEW!)
├── DATA_SOURCES_COMPARED.md                           # ⭐ Deep dive: CSV vs Raw API
├── CUSTOM_PARQUET_PIPELINE.py                         # ⭐ msgspec + parquet demo
├── WHEN_TO_USE_WHAT.md                                # ⭐ Decision guide
│
├── # REFERENCE
├── docs_tardis_dev_enhanced_20251008_144245.txt      # Full Tardis.dev documentation
│
├── # SOURCE CODE
├── tardis-python/                                     # Local tardis-client package
│   └── tardis_client/
│       ├── __init__.py
│       ├── tardis_client.py
│       └── ...
│
└── # ADDITIONAL EXAMPLES
└── deribit_options_examples/
    ├── 01_csv_download_approach.py
    ├── 02_raw_api_replay_approach.py
    ├── 03_demonstrate_formats.py
    ├── README.md
    └── SUMMARY.md
```

---

## Installation

```bash
# Install tardis-dev for CSV downloads
pip install tardis-dev pandas

# Or use uv (recommended)
uv pip install tardis-dev pandas
```

---

## Example: Download BTC Options

```python
from tardis_dev import datasets

# Download ALL options for Oct 1, 2025 (free access)
datasets.download(
    exchange="deribit",
    data_types=["options_chain"],
    from_date="2025-10-01",
    to_date="2025-10-01",
    symbols=["OPTIONS"],  # ⚠️ Gets ALL options (BTC, ETH, SOL, etc.)
    api_key=None  # Free for first day of month
)

# Filter for BTC only
import pandas as pd
df = pd.read_csv("deribit_options_chain_2025-10-01_OPTIONS.csv.gz")
btc_options = df[df['symbol'].str.startswith('BTC-')]
```

---

## Key Concepts

### 1. OPTIONS Symbol Gets Everything
When you download with `symbols=["OPTIONS"]`, you get **ALL options from Deribit**:
- BTC options (e.g., `BTC-29NOV25-50000-C`)
- ETH options (e.g., `ETH-29NOV25-2000-P`)
- SOL options (e.g., `SOL-29NOV25-100-C`)
- And any other assets

**You must filter for BTC afterward** using pandas.

### 2. Tick-by-Tick Timeseries
The data is **NOT aggregated**. You get:
- Every single ticker update throughout the day
- Updates whenever price/greeks/IV changes
- Can be 1,000-10,000 updates per option per day
- Total: ~500K - 5M rows for BTC options on one day

### 3. Free Access
First day of each month is **free without API key**:
- 2025-10-01 ✅
- 2025-11-01 ✅
- 2025-12-01 ✅

Perfect for testing!

### 4. CSV Schema
The `options_chain` CSV contains 23 columns:
```
exchange, symbol, timestamp, local_timestamp, type, strike_price, expiration,
open_interest, last_price, bid_price, bid_amount, bid_iv, ask_price,
ask_amount, ask_iv, mark_price, mark_iv, underlying_index, underlying_price,
delta, gamma, vega, theta, rho
```

### 5. Timestamps in Microseconds
```python
# Timestamps are in microseconds since epoch
timestamp = 1727740800000000  # microseconds
= 1727740800.000000 seconds
= October 1, 2025, 00:00:00.000000 UTC

# Convert to datetime
df['datetime'] = pd.to_datetime(df['timestamp'], unit='us')
```

---

## Workflow

### Step 1: Download OPTIONS Data
```python
from tardis_dev import datasets

datasets.download(
    exchange="deribit",
    data_types=["options_chain"],
    from_date="2025-10-01",
    to_date="2025-10-01",
    symbols=["OPTIONS"],
    api_key=None
)
```

**Result:** `deribit_options_chain_2025-10-01_OPTIONS.csv.gz` (~200MB-1GB)

### Step 2: Load and Filter for BTC
```python
import pandas as pd

# Load CSV (pandas handles gzip automatically)
df = pd.read_csv("deribit_options_chain_2025-10-01_OPTIONS.csv.gz")

# Filter for BTC options only
btc = df[df['symbol'].str.startswith('BTC-')]

# Convert timestamps
btc['datetime'] = pd.to_datetime(btc['timestamp'], unit='us')
```

### Step 3: Parse Option Symbols
```python
from datetime import datetime

def parse_option(symbol):
    """Parse BTC-29NOV25-50000-C into components."""
    parts = symbol.split('-')
    return {
        'underlying': parts[0],  # BTC
        'expiry_str': parts[1],  # 29NOV25
        'strike': float(parts[2]),  # 50000
        'option_type': 'call' if parts[3] == 'C' else 'put'
    }

# Parse all symbols
parsed = btc['symbol'].apply(parse_option)
parsed_df = pd.DataFrame(parsed.tolist())
btc = pd.concat([btc, parsed_df], axis=1)

# Parse expiry date
btc['expiry_date'] = pd.to_datetime(btc['expiry_str'], format='%d%b%y')
```

### Step 4: Filter for Short-Dated Options
```python
# Calculate days to expiry
btc['days_to_expiry'] = (btc['expiry_date'] - datetime(2025, 10, 1)).dt.days

# Filter: 7-30 days
short_dated = btc[
    (btc['days_to_expiry'] >= 7) &
    (btc['days_to_expiry'] <= 30)
]
```

### Step 5: Analyze
```python
# Basic stats
print(f"Total updates: {len(short_dated):,}")
print(f"Unique options: {short_dated['symbol'].nunique()}")

# BTC price range
print(f"BTC price: ${short_dated['underlying_price'].min():.0f} - ${short_dated['underlying_price'].max():.0f}")

# IV statistics
print(f"IV range: {short_dated['mark_iv'].min():.1f}% - {short_dated['mark_iv'].max():.1f}%")

# Find ATM options
btc_price = short_dated['underlying_price'].mean()
short_dated['strike_diff'] = abs(short_dated['strike'] - btc_price)
atm_options = short_dated.sort_values('strike_diff').head(10)
```

---

## Common Questions

### Q: Why does OPTIONS include all assets?
A: Deribit's grouped symbols work this way. OPTIONS is a special symbol that includes all options contracts. You filter for BTC afterward.

### Q: How many rows will I get?
A: For BTC options on one day: ~500K - 5 million rows depending on volatility and trading activity.

### Q: Is the data aggregated?
A: No! You get every single ticker update as it happens. This is tick-by-tick data, not OHLC bars.

### Q: Can I download just BTC options?
A: No, you must download OPTIONS (all assets) and filter afterward. Tardis.dev doesn't offer BTC-only downloads.

### Q: What's the file size?
A: OPTIONS compressed: ~200MB-1GB. BTC only after filtering: ~100MB-500MB.

### Q: How do I get data for other dates?
A: For dates other than the first of the month, you need an API key. Contact Tardis.dev for free trials.

---

## Use Cases

### 1. Options Greeks Analysis
Track how delta, gamma, theta, vega change throughout the day.

### 2. Volatility Surface
Construct IV surface across strikes and expiries.

### 3. ATM Options Strategy
Focus on at-the-money options for delta-neutral strategies.

### 4. Time Series Analysis
Study how option prices evolve as underlying moves.

### 5. Backtesting
Test options trading strategies using historical data.

---

## Next Steps

1. **Read the docs:** [`TARDIS_PACKAGES_EXPLAINED.md`](TARDIS_PACKAGES_EXPLAINED.md)
2. **See example data:** [`example_deribit_options_chain_2025-10-01_BTC_SAMPLE.csv`](example_deribit_options_chain_2025-10-01_BTC_SAMPLE.csv)
3. **Run the script:** `uv run python filter_btc_options.py`
4. **Download real data:** Set `USE_EXAMPLE=False` in script
5. **Analyze:** Build your own analysis on top of filtered data

---

## Support

- **Tardis.dev Docs:** https://docs.tardis.dev
- **Deribit Data Details:** https://docs.tardis.dev/historical-data-details/deribit
- **CSV Data Types:** https://docs.tardis.dev/downloadable-csv-files
- **Free Trial:** Contact Tardis.dev for full API access

---

**Created:** October 8, 2025
**Location:** `/Users/lgierhake/Documents/ETH/BT/research/tardis/`
