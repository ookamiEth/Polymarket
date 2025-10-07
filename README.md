# Polymarket Data Collection & Analysis

Complete system for collecting, analyzing, and tracking Polymarket trading data across multiple dimensions: leaderboards, markets, CLOB tick data, and real-time order book streaming.

**Last Updated:** October 6, 2025

## Project Overview

This repository contains production-ready tools for collecting comprehensive Polymarket data:

- **CLOB Tick Data**: Historical trade-by-trade data for 14,874+ crypto up/down markets
- **Real-Time Order Book**: WebSocket streaming for live order book snapshots
- **Leaderboard Tracking**: Top 1,000 traders across 4 time periods (day/week/month/all-time)
- **Market Database**: Complete historical market data (113K+ markets, 2021-2025)
- **Trader Analysis**: Individual trader activity, positions, and performance metrics

## Quick Navigation

### Data Collection Tools

| Tool | Location | Purpose |
|------|----------|---------|
| **CLOB Historical Data** | [`data_collectors/clob_historical/`](data_collectors/clob_historical/) | Trade-by-trade tick data for closed markets |
| **Leaderboard Collector** | [`data_collectors/leaderboard/`](data_collectors/leaderboard/) | Top traders by volume/P&L across time periods |
| **Trader History** | [`data_collectors/trader_history/`](data_collectors/trader_history/) | Individual trader trades, positions, activity |

### Research & Testing

| Resource | Location | Purpose |
|----------|----------|---------|
| **API Endpoint Tests** | [`research/endpoint_tests/`](research/endpoint_tests/) | 14 test scripts for all Polymarket APIs |
| **API Discovery** | [`research/api_discovery/`](research/api_discovery/) | Leaderboard API discovery process |
| **Analysis Notebooks** | [`research/notebooks/`](research/notebooks/) | Jupyter notebooks for data analysis |

### Documentation

| Document | Location | Description |
|----------|----------|-------------|
| **API Endpoints Guide** | [`docs/API_ENDPOINTS.md`](docs/API_ENDPOINTS.md) | Complete Polymarket API documentation |
| **CLOB Data Collection PRD** | [`docs/CLOB_DATA_COLLECTION_PRD.md`](docs/CLOB_DATA_COLLECTION_PRD.md) | Historical tick data collection specs |
| **Order Book Streaming** | [`docs/ORDERBOOK_STREAMING_README.md`](docs/ORDERBOOK_STREAMING_README.md) | Real-time WebSocket streaming guide |
| **Up/Down Market Naming** | [`docs/up_down_market_naming.md`](docs/up_down_market_naming.md) | Market title/slug structure docs |
| **Pagination Guide** | [`docs/pagination_guide.md`](docs/pagination_guide.md) | API pagination explained |
| **Leaderboard Docs** | [`docs/leaderboard/`](docs/leaderboard/) | Leaderboard implementation details |

### Data

All collected data is stored in Parquet format for efficient storage and fast queries:

| Data Type | Location | Contents |
|-----------|----------|----------|
| **CLOB Tick Data** | [`data/clob_ticks/`](data/clob_ticks/) | Trade-by-trade data for 14,874+ markets |
| **Leaderboard** | [`data/leaderboard/`](data/leaderboard/) | 4 time periods × 1,000 traders each |
| **Markets** | [`data/markets/`](data/markets/) | 113K+ markets (2021-2025) |
| **Traders** | [`data/traders/`](data/traders/) | Individual trader trades & positions |

## Quick Start

### Installation

```bash
# Clone repository
cd /path/to/BT

# Install dependencies with uv
uv pip install polars requests websockets py-clob-client python-dotenv
```

### Collect CLOB Tick Data (Historical Trades)

```bash
# Run batch collection for all closed crypto up/down markets
./run_collection.sh --resume
```

**Output:** Trade-by-trade parquet files for 14,874+ markets in `data/clob_ticks/`

### Stream Real-Time Order Book

```bash
uv run python scripts/stream_orderbook_realtime.py \
  --markets "0xabc123..." \
  --output-dir data/orderbook_snapshots \
  --interval-ms 1000
```

**Output:** Real-time order book snapshots via WebSocket

### Collect Leaderboard Data

```bash
cd data_collectors/leaderboard
uv run python collect_leaderboard.py
```

**Output:** 4 Parquet files (day/week/month/all), ~650 KB total, 1,000 traders each

### Collect Trader History

```bash
cd data_collectors/trader_history
uv run python trader_historical_data_collector.py
```

**Output:** Trades, positions, activity for specific trader address

## Key Features

### CLOB Historical Tick Data
- ✅ **14,874+ Markets**: All closed crypto up/down markets
- ✅ **Trade-by-Trade**: Individual trade executions with timestamps
- ✅ **Parallel Collection**: Multi-threaded with rate limiting (130 req/10s)
- ✅ **Resume Capability**: Checkpoint-based progress tracking
- ✅ **Duplicate Detection**: Handles Polymarket API pagination bugs
- ✅ **Validation**: No duplicates, sorted timestamps, valid price ranges

### Real-Time Order Book Streaming
- ✅ **WebSocket Streaming**: Live order book snapshots & updates
- ✅ **Incremental Updates**: Price changes, trades, tick size adjustments
- ✅ **Configurable Sampling**: 1-second to 10-second intervals
- ✅ **Parquet Storage**: Compressed time-series data
- ✅ **Production Ready**: 24/7 EC2 deployment guide included

### Leaderboard Collection
- ✅ **4 Time Periods**: day, week, month, all-time
- ✅ **1,000 Traders** per period (API maximum)
- ✅ **15 Data Fields**: volume, P&L, ROI, rankings, profiles
- ✅ **Production Ready**: Rate limiting, validation, error handling
- ✅ **Fast Queries**: Polars DataFrames, ~10ms load time

### Trader Analysis
- ✅ **Complete Activity**: Trades, positions, P&L tracking
- ✅ **10K History**: Recent 10K trades via Data API
- ✅ **Position Tracking**: Current and closed positions
- ✅ **Activity Timeline**: All transaction types

## Data Schema Overview

### Leaderboard Schema (15 Fields)

```
snapshot_timestamp, snapshot_date, period_type
rank_api, rank_by_pnl, rank_by_volume
user_address, username, has_custom_username
volume_usd, pnl_usd, pnl_roi_percentage
profile_image_url, profile_url, is_complete_record
```

### Market Schema (Key Fields)

```
condition_id, question, description
active, closed, archived
volume, volume_24hr, liquidity
outcome_prices, outcomes
created_at, end_date_iso
```

### Trader Schema (Key Fields)

```
Trades: market, outcome, side, size, price, timestamp
Positions: size, pnl_percentage, realized_pnl
Activity: type (TRADE/REDEEM/YIELD/etc), timestamp
```

## Example Queries

### Leaderboard Analysis

```python
import polars as pl

# Load weekly leaderboard
df = pl.read_parquet("data/leaderboard/week/leaderboard_week_20251001.parquet")

# Top 10 by volume
top_10 = df.sort("rank_by_volume").head(10)
print(top_10.select(["username", "volume_usd", "pnl_usd"]))

# Filter high ROI traders
high_roi = df.filter(
    (pl.col("volume_usd") > 100_000) &
    (pl.col("pnl_roi_percentage") > 10)
)
```

### Market Analysis

```python
import polars as pl

# Load all markets
markets = pl.read_parquet("data/markets/all_markets.parquet")

# Active high-volume markets
active = markets.filter(
    (pl.col("active") == True) &
    (pl.col("volume") > 1_000_000)
).sort("volume", descending=True)

# Market statistics
stats = markets.select([
    pl.col("volume").sum().alias("total_volume"),
    pl.col("liquidity").sum().alias("total_liquidity"),
    pl.count().alias("market_count")
])
```

## Production Setup

### Daily Leaderboard Collection

```bash
# Add to crontab
0 0 * * * cd /path/to/BT/data_collectors/leaderboard && uv run python collect_leaderboard.py
```

### Weekly Market Snapshot

```bash
# Add to crontab
0 0 * * 1 cd /path/to/BT/data_collectors/market_snapshot && uv run python market_snapshot_collector.py
```

## API Rate Limits

All collectors implement proper rate limiting:

- **Data API**: 100 requests / 10 seconds
- **CLOB API**: 100 requests / 10 seconds
- **Gamma API**: 100 requests / 10 seconds

Collectors use 0.11s delays between requests to stay within limits.

## File Structure

```
/BT/
├── README.md                          # This file
├── pyproject.toml                     # Python dependencies
├── uv.lock                            # Lock file
├── run_collection.sh                  # CLOB collection script
├── scripts/                           # Data collection scripts
│   ├── batch_fetch_clob_data.py       # Batch CLOB tick data
│   ├── fetch_clob_tick_data.py        # Single market tick data
│   ├── stream_orderbook_realtime.py   # WebSocket streaming
│   ├── setup_polymarket_api.py        # API key setup
│   └── polymarket_data_access.py      # API demo script
├── data_collectors/                   # Specialized collectors
│   ├── clob_historical/               # Historical tick data
│   ├── leaderboard/                   # Top traders
│   └── trader_history/                # Individual traders
├── data/                              # Collected data (Parquet)
│   ├── clob_ticks/                    # Trade-by-trade data
│   ├── leaderboard/                   # 4 periods × 1K traders
│   ├── markets/                       # 113K+ markets
│   └── traders/                       # Trader-specific data
├── docs/                              # Documentation
│   ├── API_ENDPOINTS.md               # API guide
│   ├── CLOB_DATA_COLLECTION_PRD.md    # Tick data specs
│   ├── ORDERBOOK_STREAMING_README.md  # WebSocket guide
│   ├── up_down_market_naming.md       # Market naming
│   ├── pagination_guide.md            # Pagination explained
│   └── leaderboard/                   # Leaderboard docs
├── research/                          # Testing & analysis
│   ├── endpoint_tests/                # API test scripts
│   ├── api_discovery/                 # API research
│   └── notebooks/                     # Analysis notebooks
└── checkpoints/                       # Collection progress
```

## Key Discoveries

### Leaderboard API
- **Endpoint**: `https://data-api.polymarket.com/leaderboard`
- **Critical Parameter**: `timePeriod` (NOT `period`)
- **Valid Periods**: `day`, `week`, `month`, `all`
- **Maximum**: 1,000 traders per period via pagination

### Market API
- **Endpoint**: `https://gamma-api.polymarket.com/markets`
- **No Offset Limit**: Unlike trades (10K limit), markets have unlimited pagination
- **Rate Limit**: 100 req/10s, ~2.5 min for complete collection

### Trader API Limitations
- **Trades Endpoint**: 10K maximum offset (Aug 2025 update)
- **Activity Endpoint**: 10K maximum offset
- **Workaround**: Use Polymarket subgraphs for unlimited history

## Technology Stack

- **Python 3.x** with uv environment
- **Polars** for fast DataFrame operations
- **Requests** for REST API calls
- **Parquet** with Snappy compression for storage
- **WebSockets** for real-time data (optional)

## Performance Benchmarks

### Leaderboard Collection
- **Runtime**: ~90 seconds (4 periods)
- **API Calls**: ~80 requests
- **Output Size**: 650 KB (4 files)
- **Query Speed**: <10ms per file load

### Market Collection
- **Runtime**: ~2.5 minutes (113K markets)
- **API Calls**: 228 requests
- **Output Size**: 38 MB compressed
- **Query Speed**: ~50ms for full dataset

## Dependencies

```toml
polars >= 0.19.0
requests >= 2.31.0
websockets >= 11.0.0  # Optional, for real-time
```

Install with:
```bash
uv pip install polars requests websockets
```

## Future Enhancements

### Planned Features
- [ ] Subgraph integration for unlimited trade history
- [ ] Real-time leaderboard tracking via WebSocket
- [ ] Multi-trader comparison dashboards
- [ ] Historical trend analysis
- [ ] Automated alerting for top trader changes

### Data Enrichment
- [ ] Add trade count per trader
- [ ] Join market metadata with trades
- [ ] Calculate advanced performance metrics
- [ ] Track specific trader cohorts over time

## Contributing

This is a research project. Each data collector is self-contained with its own README and documentation.

To add new collectors:
1. Create directory in `data_collectors/`
2. Follow existing script structure
3. Include README with usage instructions
4. Add validation script
5. Document schema in `docs/`

## Troubleshooting

### Collection Fails
```bash
# Test API connectivity
curl "https://data-api.polymarket.com/leaderboard?timePeriod=week&limit=10"

# Check rate limiting
# Ensure 0.11s delay between requests
```

### Data Validation
```bash
# Validate leaderboard data
cd data_collectors/leaderboard
uv run python validate_parquet.py

# Check file integrity
uv run python -c "import polars as pl; print(pl.read_parquet('../../data/leaderboard/week/*.parquet'))"
```

### Query Issues
```bash
# Test Polars installation
uv run python -c "import polars as pl; print(pl.__version__)"

# Verify file paths
ls -lh data/leaderboard/week/
```

## Resources

- **Polymarket Docs**: https://docs.polymarket.com
- **Subgraph Repo**: https://github.com/Polymarket/polymarket-subgraph
- **Python CLOB Client**: `py_clob_client`
- **API Changelog**: https://docs.polymarket.com/changelog

## License

MIT - Data collection tools for publicly available API data.

---

**Status**: ✅ Production Ready
**Last Collection**: October 1, 2025
**Data Coverage**: 2021-2025 (4.7 years)
