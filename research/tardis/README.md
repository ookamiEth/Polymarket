# Tardis Download - Deribit Options Data Downloader

A Python script for downloading historical Deribit options market data using the tardis-machine server.

## Overview

`tardis_download.py` downloads resampled quote and orderbook data for Deribit BTC and ETH options. It:

- Generates option symbols based on asset, expiry date range, and strike prices
- Fetches sampled quote data (bid/ask prices and amounts) at configurable intervals
- Optionally includes orderbook snapshots at specified depth
- Outputs data in Parquet (recommended) or CSV format
- Handles batching for large symbol sets to avoid timeouts

## Prerequisites

### 1. Tardis Machine Server

The script requires a running [tardis-machine](https://github.com/tardis-dev/tardis-machine) server:

```bash
# Install globally with npm
npm install -g tardis-machine

# Start the server (default port 8000)
npx tardis-machine --port=8000
```

The server must be running before you execute the download script. By default, it expects the server at `http://localhost:8000`.

### 2. Python Dependencies

This project uses **uv** for environment management. Install dependencies:

```bash
# Install required packages
uv pip install polars httpx
```

**Required packages:**
- `polars` - Fast DataFrame library for data processing
- `httpx` - Async HTTP client for API requests
- Python 3.8+ (for asyncio support)

## Installation on a New Device

```bash
# 1. Clone the repository
git clone <your-repo-url>
cd BT/research/tardis

# 2. Install Python dependencies
uv pip install polars httpx

# 3. Install tardis-machine (one-time setup)
npm install -g tardis-machine

# 4. Start tardis-machine server (in a separate terminal)
npx tardis-machine --port=8000

# 5. Verify setup - run a minimal test
uv run python tardis_download.py \
  --from-date 2024-01-01 \
  --to-date 2024-01-01 \
  --assets BTC \
  --min-days 0 \
  --max-days 0 \
  --output-dir ./test_output
```

## Quick Start

Download BTC call and put options for January 1, 2024 with 0-7 days to expiry:

```bash
uv run python tardis_download.py \
  --from-date 2024-01-01 \
  --to-date 2024-01-01 \
  --assets BTC \
  --min-days 0 \
  --max-days 7 \
  --resample-interval 5s \
  --output-dir ./datasets_deribit_options
```

## Command-Line Arguments

### Required Arguments

| Argument | Description | Example |
|----------|-------------|---------|
| `--from-date` | Start date for data download (YYYY-MM-DD) | `2024-01-01` |
| `--to-date` | End date for data download (YYYY-MM-DD) | `2024-01-31` |
| `--assets` | Comma-separated list of assets (BTC, ETH) | `BTC,ETH` |

### Optional Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--min-days` | None | Minimum days to expiry from reference date |
| `--max-days` | None | Maximum days to expiry from reference date |
| `--option-type` | `both` | Option type: `call`, `put`, or `both` |
| `--resample-interval` | `5s` | Sampling interval (e.g., `1s`, `5s`, `1m`, `1h`) |
| `--include-book` | False | Include orderbook snapshots |
| `--book-levels` | 25 | Number of orderbook levels (1-100) |
| `--output-dir` | `./datasets_deribit_options` | Directory for output files |
| `--output-format` | `parquet` | Output format: `parquet` or `csv` |
| `--tardis-machine-url` | `http://localhost:8000` | Tardis-machine server URL |
| `--verbose` `-v` | False | Enable debug logging |

## Usage Examples

### Example 1: Single Day, BTC Only, Short-Dated Options

Download BTC options expiring within 3 days for a single date:

```bash
uv run python tardis_download.py \
  --from-date 2024-03-15 \
  --to-date 2024-03-15 \
  --assets BTC \
  --min-days 0 \
  --max-days 3 \
  --resample-interval 10s
```

### Example 2: Multi-Day Range, BTC + ETH, All Expiries

Download both assets for a week, all available expiries:

```bash
uv run python tardis_download.py \
  --from-date 2024-03-01 \
  --to-date 2024-03-07 \
  --assets BTC,ETH \
  --resample-interval 1m
```

**Note:** Omitting `--min-days` and `--max-days` fetches all available expiries.

### Example 3: Call Options Only with Orderbook Data

Download only call options with 25-level orderbook snapshots:

```bash
uv run python tardis_download.py \
  --from-date 2024-06-01 \
  --to-date 2024-06-01 \
  --assets ETH \
  --min-days 7 \
  --max-days 30 \
  --option-type call \
  --include-book \
  --book-levels 25 \
  --resample-interval 30s
```

### Example 4: High-Frequency Data with Verbose Logging

Download 1-second interval data with debug output:

```bash
uv run python tardis_download.py \
  --from-date 2024-12-01 \
  --to-date 2024-12-01 \
  --assets BTC \
  --min-days 0 \
  --max-days 1 \
  --resample-interval 1s \
  --verbose
```

### Example 5: Remote Tardis-Machine Server

Use a tardis-machine server running on another machine:

```bash
uv run python tardis_download.py \
  --from-date 2024-01-01 \
  --to-date 2024-01-01 \
  --assets BTC \
  --min-days 0 \
  --max-days 7 \
  --tardis-machine-url http://192.168.1.100:8000
```

## Symbol Generation Logic

The script automatically generates Deribit option symbols based on your parameters.

### Strike Price Ranges

- **BTC:** 2,000 to 200,000 in 1,000 increments (199 strikes)
- **ETH:** 200 to 20,000 in 100 increments (199 strikes)

### Symbol Format

Deribit options follow the format: `{ASSET}-{EXPIRY}-{STRIKE}-{TYPE}`

**Example:** `BTC-1MAR24-50000-C`
- Asset: BTC
- Expiry: March 1, 2024
- Strike: $50,000
- Type: Call

### Expiry Date Calculation

Expiry dates are calculated relative to `--from-date`:

- `--min-days 0 --max-days 7` → Options expiring 0-7 days after from-date
- No min/max specified → All strikes for from-date itself

**Total Symbols Example:**
- BTC, 1 asset, 7 expiry dates, 199 strikes, both calls/puts = **2,786 symbols**
- BTC + ETH, 2 assets, 30 days, both = **23,880 symbols**

## Output Format

### File Naming

Output files use the format: `deribit_options_{from-date}_{assets}_{interval}.{ext}`

**Examples:**
- `deribit_options_2024-01-01_BTC_5s.parquet`
- `deribit_options_2024-03-15_BTC_ETH_1m.csv`

### Parquet Schema

| Column | Type | Description |
|--------|------|-------------|
| `exchange` | string | Always "deribit" |
| `symbol` | string | Option symbol (e.g., BTC-1MAR24-50000-C) |
| `timestamp` | int64 | Exchange timestamp (microseconds since epoch) |
| `local_timestamp` | int64 | Local timestamp (microseconds since epoch) |
| `type` | string | "call" or "put" |
| `strike_price` | float64 | Strike price in USD |
| `underlying` | string | "BTC" or "ETH" |
| `expiry_str` | string | Expiry date string (e.g., "1MAR24") |
| `bid_price` | float64 | Best bid price (null if no bid) |
| `bid_amount` | float64 | Best bid quantity (null if no bid) |
| `ask_price` | float64 | Best ask price (null if no ask) |
| `ask_amount` | float64 | Best ask quantity (null if no ask) |

### Reading Output Data

```python
import polars as pl

# Read the Parquet file
df = pl.read_parquet("deribit_options_2024-01-01_BTC_5s.parquet")

# Quick exploration
print(df.head())
print(f"Rows: {df.shape[0]:,}")
print(f"Unique symbols: {df['symbol'].n_unique()}")
print(f"Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")

# Filter for specific strikes
btc_50k = df.filter(pl.col('strike_price') == 50000)

# Calculate bid-ask spread
df_with_spread = df.with_columns(
    (pl.col('ask_price') - pl.col('bid_price')).alias('spread')
)
```

## Performance Considerations

### Batching

The script automatically batches symbol requests to avoid HTTP timeouts:

- **Default batch size:** 50 symbols per request
- Large symbol sets (1000+) are split into multiple batches
- Each batch is processed sequentially with progress logging

### Timeouts

- **HTTP timeout:** 300 seconds (5 minutes) per batch
- **Server check timeout:** 10 seconds
- Increase `HTTP_TIMEOUT_SECONDS` in code for very large batches

### Data Volume Estimates

Approximate output sizes (Parquet format, 5s interval, 24h):

| Configuration | Symbols | Rows | File Size |
|---------------|---------|------|-----------|
| BTC, 1 day expiry, both | ~400 | ~700K | ~15 MB |
| BTC, 7 days expiry, both | ~2,800 | ~5M | ~100 MB |
| BTC+ETH, 30 days, both | ~24K | ~40M | ~800 MB |

**Tip:** Use Parquet format (default) for 5-10x better compression than CSV.

## Troubleshooting

### Error: "tardis-machine server not accessible"

**Cause:** Tardis-machine server is not running or wrong URL

**Solution:**
```bash
# Start server in a separate terminal
npx tardis-machine --port=8000

# Or specify custom URL
--tardis-machine-url http://localhost:8080
```

### Error: "No data received - check symbols exist"

**Cause:** Symbols don't exist for the specified date/expiry range

**Solutions:**
1. Check that options existed on Deribit for that date (markets launched ~2016)
2. Verify expiry dates align with actual Deribit expiries (daily, weekly, monthly)
3. Try a different date range or reduce `--max-days`
4. Use `--verbose` to see detailed API responses

### Warning: "Batch X: No data received"

**Cause:** Some symbol batches have no trading activity

**Solution:** This is normal for illiquid options (far out-of-the-money strikes). The script continues and collects data from other batches.

### Error: "HTTP timeout"

**Cause:** Batch is too large or network is slow

**Solutions:**
1. Reduce date range (`--from-date` to `--to-date`)
2. Reduce expiry range (`--min-days` to `--max-days`)
3. Use faster network connection
4. Modify `HTTP_TIMEOUT_SECONDS` in code

### Parse Errors or Validation Failures

**Cause:** Malformed data from tardis-machine

**Solution:**
- Check tardis-machine version (update with `npm install -g tardis-machine`)
- Review error counts in logs (small numbers are acceptable)
- Use `--verbose` to inspect raw API responses

## Advanced Usage

### Custom Strike Ranges

Modify `BTC_STRIKES` and `ETH_STRIKES` in code (lines 24-25):

```python
# Example: Only round strikes for BTC
BTC_STRIKES = range(10000, 100001, 5000)  # 10K, 15K, 20K, ...
```

### Custom Batch Size

Adjust `DEFAULT_BATCH_SIZE` (line 16) for your network:

```python
DEFAULT_BATCH_SIZE = 100  # Larger batches (may timeout)
DEFAULT_BATCH_SIZE = 20   # Smaller batches (slower but safer)
```

### Resample Intervals

Valid interval formats:
- Seconds: `1s`, `5s`, `10s`, `30s`
- Minutes: `1m`, `5m`, `15m`, `30m`
- Hours: `1h`, `4h`, `12h`
- Days: `1d`

**Note:** Smaller intervals = more data = larger files and longer processing time.

### Running on a Remote Server

If running on a headless server:

```bash
# 1. Start tardis-machine in background
nohup npx tardis-machine --port=8000 > tardis.log 2>&1 &

# 2. Run download script
uv run python tardis_download.py \
  --from-date 2024-01-01 \
  --to-date 2024-12-31 \
  --assets BTC,ETH \
  --min-days 0 \
  --max-days 90 \
  --output-dir /mnt/data/deribit \
  > download.log 2>&1

# 3. Monitor progress
tail -f download.log
```

## License

This script is part of the BT research project. Tardis-machine has its own licensing terms.

## Support

For tardis-machine issues, see: https://github.com/tardis-dev/tardis-machine

For script issues, check the main project repository.
