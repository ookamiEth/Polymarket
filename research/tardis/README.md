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