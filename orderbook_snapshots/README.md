# Polymarket Orderbook Streaming Service

**Version:** 1.0.0
**Status:** Development
**Platform:** Polymarket BTC 15-Minute Up/Down Markets

---

## Overview

This service continuously captures orderbook snapshots from Polymarket's BTC 15-minute Up/Down prediction markets. It operates 24/7, automatically switching between market periods and storing high-frequency data for quantitative analysis.

### Key Features

- ✅ **1-second snapshot interval** (3,600 snapshots per hour)
- ✅ **Automatic market switching** every 15 minutes
- ✅ **1-year schedule** pre-generated (35,040 market periods)
- ✅ **Hierarchical storage** organized by date
- ✅ **Parquet compression** (~480KB per market period)
- ✅ **systemd integration** for auto-restart and monitoring

---

## Quick Start

### 1. Generate Schedule (One-time)

```bash
cd /Users/lgierhake/Documents/ETH/BT/orderbook_snapshots
uv run python scripts/generate_schedule.py
```

**Output:** `config/btc_updown_schedule_1year.parquet` (35,040 periods, ~480KB)

### 2. Test Snapshot Collection

```bash
uv run python scripts/test_snapshot.py
```

**What it does:**
- Finds current 15-minute market from schedule
- Queries Gamma API for token IDs
- Polls orderbook every 1 second for 60 seconds
- Saves to `data/test/orderbook_test_YYYYMMDD_HHMMSS.parquet`

### 3. Run Production Service (Future)

```bash
uv run python scripts/stream_continuous.py
```

*Note: Production streaming service is not yet implemented. See [docs/prd.md](docs/prd.md) for specifications.*

---

## Directory Structure

```
orderbook_snapshots/
├── README.md                   # This file
├── config/
│   ├── btc_updown_schedule_1year.parquet  # 1-year market schedule
│   └── streamer_config.yaml              # Service configuration
├── scripts/
│   ├── generate_schedule.py    # Schedule generator
│   ├── test_snapshot.py        # 60-second test script
│   └── stream_continuous.py    # Production streamer (TODO)
├── data/
│   ├── raw/                    # Production data (organized by date)
│   │   └── 2025/10/06/
│   │       ├── orderbook_20251006_1130.parquet
│   │       └── orderbook_20251006_1145.parquet
│   └── test/                   # Test outputs
│       └── orderbook_test_*.parquet
├── logs/
│   └── streamer.log            # Service logs
├── systemd/
│   └── orderbook-streamer.service  # systemd service file
└── docs/
    ├── prd.md                  # Product Requirements Document
    └── api_references.md       # Polymarket API documentation (TODO)
```

---

## Data Schema

### Parquet File Schema

Each market period generates one Parquet file with ~900 snapshots (15 min × 60 sec).

| Column | Type | Description |
|--------|------|-------------|
| `timestamp_ms` | Int64 | Capture timestamp (milliseconds) |
| `market_timestamp_ms` | Int64 | API-reported orderbook timestamp |
| `condition_id` | String | Market condition ID |
| `asset_id` | String | Token ID (UP token) |
| `bid_price_3` | Float64 | 3rd best bid (furthest from mid) |
| `bid_size_3` | Float64 | Size at bid level 3 |
| `bid_price_2` | Float64 | 2nd best bid |
| `bid_size_2` | Float64 | Size at bid level 2 |
| `bid_price_1` | Float64 | **BEST BID** (closest to mid) |
| `bid_size_1` | Float64 | Size at best bid |
| `spread` | Float64 | `ask_price_1 - bid_price_1` |
| `mid_price` | Float64 | `(bid_price_1 + ask_price_1) / 2` |
| `ask_price_1` | Float64 | **BEST ASK** (closest to mid) |
| `ask_size_1` | Float64 | Size at best ask |
| `ask_price_2` | Float64 | 2nd best ask |
| `ask_size_2` | Float64 | Size at ask level 2 |
| `ask_price_3` | Float64 | 3rd best ask (furthest from mid) |
| `ask_size_3` | Float64 | Size at ask level 3 |

### File Naming Convention

```
orderbook_YYYYMMDD_HHMM.parquet

Examples:
- orderbook_20251006_1130.parquet  # Oct 6, 2025, 11:30-11:45
- orderbook_20251006_1145.parquet  # Oct 6, 2025, 11:45-12:00
```

---

## Usage Examples

### Query Single Market Period

```python
import polars as pl

# Load data
df = pl.read_parquet('data/raw/2025/10/06/orderbook_20251006_1130.parquet')

# Calculate average spread
avg_spread = df['spread'].mean()
print(f"Average spread: ${avg_spread:.4f}")

# Find maximum bid size
max_bid = df['bid_size_1'].max()
print(f"Max bid size: {max_bid}")
```

### Load Entire Day (96 Periods)

```python
import polars as pl
from pathlib import Path

# Get all files for Oct 6, 2025
files = Path('data/raw/2025/10/06').glob('*.parquet')
df = pl.concat([pl.read_parquet(f) for f in files])

print(f"Total snapshots: {len(df):,}")
print(f"Time coverage: {len(df) / 60:.1f} minutes")
```

### Time-Series Analysis

```python
import polars as pl

df = pl.read_parquet('data/raw/2025/10/06/orderbook_20251006_1130.parquet')

# Convert timestamp to datetime
df = df.with_columns([
    pl.from_epoch(pl.col('timestamp_ms'), time_unit='ms').alias('datetime')
])

# 10-second rolling average of mid price
rolling_mid = df.with_columns([
    pl.col('mid_price').rolling_mean(window_size=10).alias('mid_10s_avg')
])

print(rolling_mid.select(['datetime', 'mid_price', 'mid_10s_avg']))
```

---

## Configuration

Edit `config/streamer_config.yaml` to customize:

- **Polling interval:** Default 1.0 second
- **Buffer size:** Snapshots to accumulate before writing
- **Log level:** DEBUG | INFO | WARNING | ERROR
- **API endpoints:** Gamma and CLOB URLs
- **Storage paths:** Data and log directories

---

## Monitoring

### Check Service Status (Linux with systemd)

```bash
# View service status
sudo systemctl status orderbook-streamer

# View logs (real-time)
journalctl -u orderbook-streamer -f

# View logs (last 100 lines)
journalctl -u orderbook-streamer -n 100

# Restart service
sudo systemctl restart orderbook-streamer
```

### Metrics

View current metrics:

```bash
# Disk usage
du -sh data/raw

# Number of files collected today
find data/raw/2025/10/06 -name "*.parquet" | wc -l

# Latest file
ls -lth data/raw/2025/10/06 | head -5
```

---

## Troubleshooting

### Issue: "Schedule file not found"

**Solution:**
```bash
cd /Users/lgierhake/Documents/ETH/BT/orderbook_snapshots
uv run python scripts/generate_schedule.py
```

### Issue: "No current market period found"

**Causes:**
1. Schedule outdated (>365 days old)
2. System clock incorrect
3. Outside market hours

**Solution:**
- Regenerate schedule
- Check system time: `date`
- Verify schedule covers current timestamp

### Issue: "Market is marked as CLOSED"

**Cause:** Market period has ended but script still trying to access it

**Solution:**
- Script auto-switches markets every 15 minutes
- If persistent, restart script to force market refresh

### Issue: API Rate Limit (429 Error)

**Cause:** Exceeding 200 requests / 10 seconds

**Solution:**
- Current rate: 1 req/s = 10 req/10s (5% of limit) ✅
- If hitting limit, increase `polling.interval_seconds` in config

---

## API Limits

From Polymarket CLOB documentation:

| Endpoint | Limit | Current Usage |
|----------|-------|---------------|
| `/book` (CLOB) | 200 req / 10s | 10 req / 10s (5%) ✅ |
| `/markets` (Gamma) | Unknown | ~1 req / 15 min |

---

## Development Roadmap

### Phase 1: Core Service ✅ (Current)
- [x] 1-year schedule generation
- [x] Test snapshot collection
- [x] Parquet storage
- [x] Config file structure
- [ ] Production streaming service

### Phase 2: Production Deployment (Week 1-2)
- [ ] Implement `stream_continuous.py`
- [ ] Market auto-switching logic
- [ ] Graceful shutdown handling
- [ ] systemd integration testing
- [ ] 24-hour stress test

### Phase 3: Monitoring & Alerts (Week 3)
- [ ] Metrics dashboard (Grafana)
- [ ] Email/Slack alerts
- [ ] Disk usage monitoring
- [ ] API latency tracking
- [ ] Data quality validation

### Phase 4: Enhancements (Future)
- [ ] WebSocket migration (sub-second latency)
- [ ] Trade data integration
- [ ] Multi-market support (hourly, daily BTC)
- [ ] Participant tracking (maker/taker addresses)
- [ ] TimescaleDB backend

---

## Support & Documentation

- **PRD:** [docs/prd.md](docs/prd.md) - Detailed product requirements
- **Polymarket Docs:** https://docs.polymarket.com/developers/CLOB/introduction
- **Issues:** Contact development team

---

## License

Internal research tool - Not for redistribution

**Last Updated:** 2025-10-06
**Maintainer:** Quant Research Team
