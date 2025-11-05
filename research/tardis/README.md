# Tardis - Deribit Options Market Data Pipeline

A comprehensive data collection and analysis pipeline for Deribit options market data, using Tardis.dev API for historical data retrieval.

## 📊 Overview

This project collects, processes, and analyzes options and perpetual futures data from Deribit exchange, focusing on:
- ATM (at-the-money) options filtering
- Implied volatility calculation
- Quote consolidation and resampling
- Market microstructure analysis

## 🗂 Directory Structure

```
tardis/
├── scripts/                    # Production pipeline scripts
│   ├── 01_download/           # Data collection from Tardis API
│   ├── 02_process/            # Data processing and filtering
│   ├── 03_analyze/            # Analysis and IV calculation
│   ├── utils/                 # Utility functions and models
│   └── examples/              # Example scripts
│
├── data/                      # All data files (60GB+)
│   ├── raw/                   # Original downloads
│   │   ├── options_initial/   # Initial options collection (6.5GB)
│   │   └── perpetual/         # Perpetual futures data (800MB)
│   ├── processed/             # Filtered and enriched data
│   │   ├── options_atm3pct/   # ATM ±3% filtered (22GB)
│   │   └── quotes_atm3pct/    # Quote data with spot prices (31GB)
│   ├── consolidated/          # Final outputs
│   │   ├── quotes_1s_merged.parquet            # Main dataset (8GB)
│   │   └── quotes_1s_atm_short_dated_optimized.parquet # Filtered (4.6GB)
│   └── test/                  # Test datasets
│
├── tests/                     # Test scripts and benchmarks
├── docs/                      # Documentation
│   ├── api/                   # API reference documentation
│   └── reports/               # Analysis reports and findings
├── logs/                      # Execution logs
├── checkpoints/               # Collection progress tracking
└── requirements.txt           # Python dependencies
```

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- uv package manager
- 100GB+ free disk space

### Installation
```bash
# Install dependencies
uv pip install -r requirements.txt
```

### Data Pipeline

The pipeline consists of three main stages:

#### 1. Download Data
```bash
# Download options data
uv run python scripts/01_download/download_options.py \
    --start-date 2024-01-01 \
    --end-date 2024-01-31 \
    --output-dir data/raw/options_initial

# Download perpetual futures for reference prices
uv run python scripts/01_download/download_perpetual.py
```

#### 2. Process & Filter
```bash
# Consolidate quotes and resample to 1-second intervals
uv run python scripts/02_process/consolidate_quotes.py \
    --input-dir data/processed/quotes_atm3pct \
    --output data/consolidated/quotes_1s_merged.parquet

# Filter for ATM short-dated options
uv run python scripts/02_process/filter_atm_options.py \
    --input data/consolidated/quotes_1s_merged.parquet \
    --output data/consolidated/quotes_1s_atm_short_dated_optimized.parquet
```

#### 3. Analyze
```bash
# Calculate implied volatility
uv run python scripts/03_analyze/calculate_implied_volatility.py

# Generate trading insights
uv run python scripts/03_analyze/analyze_trading_insights.py
```

## 📈 Data Description

### Raw Data
- **options_initial/**: Raw options chain data from Tardis API
- **perpetual/**: BTC perpetual futures data for spot price reference

### Processed Data
- **options_atm3pct/**: Options filtered to ±3% moneyness from ATM
- **quotes_atm3pct/**: Bid/ask quotes enriched with spot prices

### Consolidated Data
- **quotes_1s_merged.parquet**: All quotes resampled to 1-second intervals
- **quotes_1s_atm_short_dated_optimized.parquet**: Final dataset with:
  - Moneyness: 0.97-1.03 (ATM ±3%)
  - Time to expiry: ≤7 days
  - Optimized for analysis with spot prices and Greeks

## 🔧 Key Scripts

### Download Scripts
- `download_options.py`: Main options data downloader with parallel workers
- `download_perpetual.py`: Perpetual futures downloader for spot prices
- `download_options_atm.py`: Direct ATM-filtered download
- `download_options_quotes.py`: Quote-specific downloader with spot enrichment

### Processing Scripts
- `consolidate_quotes.py`: Streaming consolidation of quote data
- `filter_atm_options.py`: Optimized ATM filtering with chunked processing
- `resample_to_1s.py`: Resample data to 1-second intervals

### Analysis Scripts
- `calculate_implied_volatility.py`: IV calculation using Black-Scholes
- `analyze_trading_insights.py`: Market microstructure analysis
- `compare_exchange_ivs.py`: Cross-exchange IV comparison
- `analyze_gaps.py`: Gap analysis in quote data

## 💰 Binance Funding Rate Data Collection

**NEW:** Download historical funding rate data for Binance futures markets (USDT and COIN-margined).

### Quick Start

```bash
# 1. Test the setup (downloads 1-2 days for validation)
cd research/tardis
uv run python scripts/01_download/test_funding_rate_download.py

# 2. Download historical funding rates
uv run python scripts/01_download/download_binance_funding_rates.py \
    --from-date 2024-01-01 \
    --to-date 2024-01-31 \
    --symbols BTCUSDT,ETHUSDT \
    --exchanges binance-futures \
    --workers 5
```

### Data Details

**Available Exchanges:**
- `binance-futures`: USDT-margined perpetual swaps (e.g., BTCUSDT, ETHUSDT)
- `binance-delivery`: COIN-margined quarterly futures (e.g., BTCUSD_PERP)

**Data Fields:**
- `funding_rate`: Current funding rate
- `predicted_funding_rate`: Predicted rate for next period
- `funding_timestamp`: Next funding event time
- `mark_price`: Mark price (used for liquidations)
- `index_price`: Underlying index price
- `open_interest`: Total open interest
- `last_price`: Last traded price

**Granularity:** 1-second updates (from Tardis `derivative_ticker` data)

**Output Structure:**
```
data/raw/binance_funding_rates/
├── binance-futures/
│   ├── BTCUSDT/
│   │   ├── 2024-01-01.parquet
│   │   └── 2024-01-02.parquet
│   └── ETHUSDT/
└── binance-delivery/
    └── BTCUSD_PERP/
```

### Usage Options

```bash
# Download both USDT and COIN futures
uv run python scripts/01_download/download_binance_funding_rates.py \
    --from-date 2024-01-01 \
    --to-date 2024-01-31 \
    --symbols BTCUSDT,BTCUSD_PERP \
    --exchanges binance-futures,binance-delivery \
    --workers 5

# Sequential processing (slower but uses less memory)
uv run python scripts/01_download/download_binance_funding_rates.py \
    --from-date 2024-01-01 \
    --to-date 2024-01-31 \
    --symbols BTCUSDT \
    --exchanges binance-futures \
    --workers 1

# Resume interrupted download
uv run python scripts/01_download/download_binance_funding_rates.py \
    --from-date 2024-01-01 \
    --to-date 2024-01-31 \
    --symbols BTCUSDT \
    --exchanges binance-futures \
    --resume
```

### Data Processing: Resampling to 1-Second Intervals

The downloaded funding rate data is event-driven (~1.47 updates/second), so we resample to fixed 1-second intervals for consistent analysis.

**Single File Resampling:**
```bash
# Basic resampling (last value per second)
uv run python scripts/02_process/resample_funding_rates_to_1s.py \
    --input-file data/raw/binance_funding_rates/binance-futures/BTCUSDT/2024-01-01.parquet \
    --output-file data/processed/binance_funding_rates_1s/binance-futures/BTCUSDT/2024-01-01_1s.parquet

# With forward-fill (creates continuous time series)
uv run python scripts/02_process/resample_funding_rates_to_1s.py \
    --input-file data/raw/binance_funding_rates/binance-futures/BTCUSDT/2024-01-01.parquet \
    --output-file data/processed/binance_funding_rates_1s/binance-futures/BTCUSDT/2024-01-01_1s.parquet \
    --method forward_fill \
    --max-fill-gap 60
```

**Batch Processing (All Files):**
```bash
# Process all downloaded files with parallel workers
uv run python scripts/02_process/batch_resample_funding_rates.py \
    --input-dir data/raw/binance_funding_rates \
    --output-dir data/processed/binance_funding_rates_1s \
    --method forward_fill \
    --max-fill-gap 60 \
    --workers 5

# Resume interrupted batch processing
uv run python scripts/02_process/batch_resample_funding_rates.py \
    --input-dir data/raw/binance_funding_rates \
    --output-dir data/processed/binance_funding_rates_1s \
    --method forward_fill \
    --max-fill-gap 60 \
    --workers 5 \
    --resume
```

**Test Resampling Scripts:**
```bash
# Validate resampling functionality
uv run python scripts/02_process/test_resample_funding_rates.py
```

**Resampling Methods:**
- `last`: Take last value per second (sparse output, ~79K-86K rows/day)
- `forward_fill`: Fill gaps with last known value (continuous output, ~86K rows/day)
  - `--max-fill-gap`: Maximum gap to fill in seconds (default: 60s)

**Output Structure:**
```
data/processed/binance_funding_rates_1s/
├── binance-futures/
│   ├── BTCUSDT/
│   │   ├── 2024-01-01_1s.parquet
│   │   └── 2024-01-02_1s.parquet
│   └── ETHUSDT/
└── binance-delivery/
    └── BTCUSD_PERP/
```

### Documentation

See detailed Binance data documentation: `docs/reports/binance_data_available_via_tardis.md`

---

## 📖 Binance Orderbook Snapshot Data Collection

**NEW:** Download historical orderbook snapshot data for Binance markets at two depth levels.

### Available Depth Levels

- **book_snapshot_5**: Top 5 price levels (22 columns total)
  - 5 ask levels + 5 bid levels
  - Lightweight for spread and top-of-book analysis
  - ~2-3× data volume vs funding rates

- **book_snapshot_25**: Top 25 price levels (102 columns total)
  - 25 ask levels + 25 bid levels
  - Deep orderbook for liquidity analysis
  - ~10× data volume vs funding rates

### Quick Start

```bash
# 1. Download book_snapshot_5 data (recommended for most use cases)
cd research/tardis
uv run python scripts/01_download/download_binance_orderbook_5.py \
    --from-date 2024-10-01 \
    --to-date 2024-10-01 \
    --symbols BTCUSDT \
    --exchanges binance-futures

# 2. Resample to 1-second intervals
uv run python scripts/02_process/batch_resample_orderbook_5.py \
    --input-dir data/raw/binance_orderbook_5 \
    --output-dir data/processed/binance_orderbook_5_1s \
    --workers 5

# 3. (Optional) Download book_snapshot_25 for deep orderbook analysis
uv run python scripts/01_download/download_binance_orderbook_25.py \
    --from-date 2024-10-01 \
    --to-date 2024-10-01 \
    --symbols BTCUSDT \
    --exchanges binance-futures
```

### Data Details

**Fields (book_snapshot_5):**
- `exchange`, `symbol`, `timestamp`, `local_timestamp`
- `ask_price_0` to `ask_price_4`: Best ask down to 5th level
- `ask_amount_0` to `ask_amount_4`: Amounts at each ask level
- `bid_price_0` to `bid_price_4`: Best bid down to 5th level
- `bid_amount_0` to `bid_amount_4`: Amounts at each bid level

**Fields (book_snapshot_25):**
- Same structure but with levels 0-24 (50 price columns + 50 amount columns)

**Granularity:** Real-time event-driven updates (multiple per second during active trading)

**Output Structure:**
```
data/
├── raw/
│   ├── binance_orderbook_5/           # Top 5 levels
│   │   └── binance-futures/
│   │       └── BTCUSDT/
│   │           └── 2024-10-01.parquet
│   └── binance_orderbook_25/          # Top 25 levels
│       └── binance-futures/
│           └── BTCUSDT/
│               └── 2024-10-01.parquet
└── processed/
    ├── binance_orderbook_5_1s/        # Resampled to 1s
    │   └── binance-futures/
    │       └── BTCUSDT/
    │           └── 2024-10-01_1s.parquet
    └── binance_orderbook_25_1s/
        └── binance-futures/
            └── BTCUSDT/
                └── 2024-10-01_1s.parquet
```

### Resampling to 1-Second Intervals

Orderbook snapshots are event-driven (multiple updates per second). Resample to fixed 1-second intervals for:
- Consistent time-series analysis
- Alignment with other 1s datasets (funding rates, trades)
- Data reduction (50-90% fewer rows)
- Machine learning features

**Single File:**
```bash
# book_snapshot_5
uv run python scripts/02_process/resample_orderbook_5_to_1s.py \
    --input-file data/raw/binance_orderbook_5/binance-futures/BTCUSDT/2024-10-01.parquet \
    --output-file data/processed/binance_orderbook_5_1s/binance-futures/BTCUSDT/2024-10-01_1s.parquet

# book_snapshot_25
uv run python scripts/02_process/resample_orderbook_25_to_1s.py \
    --input-file data/raw/binance_orderbook_25/binance-futures/BTCUSDT/2024-10-01.parquet \
    --output-file data/processed/binance_orderbook_25_1s/binance-futures/BTCUSDT/2024-10-01_1s.parquet
```

**Batch Processing:**
```bash
# book_snapshot_5 (all files)
uv run python scripts/02_process/batch_resample_orderbook_5.py \
    --input-dir data/raw/binance_orderbook_5 \
    --output-dir data/processed/binance_orderbook_5_1s \
    --workers 5 \
    --resume

# book_snapshot_25 (all files)
uv run python scripts/02_process/batch_resample_orderbook_25.py \
    --input-dir data/raw/binance_orderbook_25 \
    --output-dir data/processed/binance_orderbook_25_1s \
    --workers 5 \
    --resume
```

**Test Scripts:**
```bash
# Validate resampling works correctly
uv run python scripts/02_process/test_resample_orderbook_5.py
uv run python scripts/02_process/test_resample_orderbook_25.py
```

### When to Use Each Depth Level

| Use Case | Recommended Depth |
|----------|-------------------|
| **Spread analysis** | book_snapshot_5 |
| **Top-of-book dynamics** | book_snapshot_5 |
| **Order flow imbalance** | book_snapshot_5 |
| **Market impact modeling** | book_snapshot_5 or 25 |
| **Deep liquidity analysis** | book_snapshot_25 |
| **Large order execution** | book_snapshot_25 |
| **Order book shape research** | book_snapshot_25 |

**General Recommendation:** Start with `book_snapshot_5`. It's sufficient for most microstructure research and requires significantly less storage.

## 📊 Performance Considerations

- Uses Polars for efficient data processing (5-10x faster than Pandas)
- Streaming processing for large datasets (>100M rows)
- Chunked processing to handle billion-row datasets
- Optimized vectorized operations throughout

## 🐛 Known Issues

- Tardis API data can have duplicates - always deduplicate by timestamp
- Compression of parquet files yields minimal savings (already compressed)
- Memory usage can spike with large datasets - use streaming mode

## 📚 Documentation

- `docs/api/`: Deribit and Tardis API documentation
- `docs/reports/`: Analysis reports and optimization findings
- See individual script docstrings for detailed usage

## 🔗 Resources

- [Tardis.dev API](https://docs.tardis.dev/)
- [Deribit API](https://docs.deribit.com/)
- [Polars Documentation](https://pola-rs.github.io/polars/py-polars/html/)

## 📝 License

[Your License Here]

---

Last updated: October 2024
Total data size: ~88GB (after cleanup: ~76GB)