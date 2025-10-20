# Risk-Free Rate Research

This directory contains tools and data for collecting and analyzing risk-free rate proxies in cryptocurrency markets.

## 📊 Data Collection (Oct 1, 2023 - Oct 1, 2025)

### Available Data

| Data Source | File | Records | Description |
|-------------|------|---------|-------------|
| **Binance BTC Funding Rates** | `data/btc_funding_rates_2023_2025.parquet` | 2,194 | 8-hour funding rates for BTC perpetual futures |
| **Aave USDT Lending Rates** | `data/usdt_lending_rates_2023_2025.parquet` | 731 | Daily USDT lending/borrowing rates on Aave V3 Arbitrum |

### Summary Statistics

**BTC Funding Rates:**
- Mean annualized rate: 9.47%
- Median: 8.62%
- 90.6% of periods had positive rates
- Source: Binance Futures API

**USDT Lending Rates:**
- Mean lending APR: 5.49%
- Mean borrowing APR: 7.37%
- Mean utilization: 80.9%
- Source: Aave V3 Arbitrum (via The Graph)

## 🛠️ Collection Scripts

### Binance Funding Rates
```bash
uv run python collect_binance_funding_rates.py
```
Collects BTC perpetual funding rates from Binance Futures API. Features:
- Automatic pagination (max 1000 records per request)
- Rate limiting (0.65s delay, respects 500 req/5min limit)
- Outputs: Parquet + JSON statistics

### Aave USDT Historical Rates
```bash
uv run python collect_usdt_historical_rates.py
```
Collects daily USDT lending rates from Aave V3 Arbitrum subgraph. Features:
- GraphQL batch queries via The Graph
- Daily snapshots with TVL and utilization metrics
- Outputs: Parquet + JSON summary

### Aave Current Rates (All Markets)
```bash
uv run python collect_aave_lending_rates.py
```
Collects current lending/borrowing rates for all active Aave V3 markets. Features:
- Real-time market data
- Risk-free rate calculation (weighted by TVL)
- Filters for stablecoins and major assets

## 🔍 Helper Tools

### Aave Subgraph Query Tool
```bash
uv run python query_aave_subgraph.py reserves --limit 100
uv run python query_aave_subgraph.py reserve --symbol USDC
uv run python query_aave_subgraph.py stats
```
Generic CLI for querying Aave V3 Arbitrum subgraph. Supports:
- Reserve (market) queries
- Protocol statistics
- Custom GraphQL queries

### Data Validation
```bash
uv run python validate_usdt_data.py
```
Comprehensive validation of USDT lending rates data:
- Date range completeness check
- Duplicate detection
- Rate range validation
- Statistical summaries

## 📚 Documentation

- **`aave_v3_arbitrum_data_guide.md`** - Complete guide to Aave V3 subgraph data
- **`THE_GRAPH_PYTHON_CLI_GUIDE.md`** - Guide to using The Graph Protocol
- **`crypto_risk_free_rate_analysis.md`** - Research notes on crypto risk-free rates
- **`ssrn-5231776.pdf`** - Academic research paper on crypto rates

## 🔑 Environment Setup

Required environment variables in `/eth/bt/.env`:
```bash
GRAPH_API_KEY=your_graph_api_key_here
```

The Graph API key is used for querying Aave V3 Arbitrum subgraph:
- Subgraph ID: `JCNWRypm7FYwV8fx5HhzZPSFaMxgkPuw4TnR3Gpi81zk`
- Endpoint: `https://gateway.thegraph.com/api/{api_key}/subgraphs/id/{subgraph_id}`

## 📈 Data Quality

All data has been validated:
- ✅ Complete date coverage (no missing days)
- ✅ No duplicate records
- ✅ Rates within expected ranges
- ✅ Proper chronological ordering
- ⚠️ 6 days with USDT lending rates >20% (high demand periods)
- ⚠️ 411 gaps in BTC funding data (likely exchange maintenance)

## 🎯 Use Cases

1. **Risk-Free Rate Proxy**: Use USDT lending rates as crypto market risk-free rate
2. **Funding Rate Analysis**: Study perpetual futures funding dynamics
3. **DeFi Yield Comparison**: Compare lending rates across protocols
4. **Market Stress Detection**: High rates indicate capital constraints
5. **Arbitrage Opportunities**: Compare funding vs lending spreads

## 📝 Notes

- All timestamps in UTC
- Parquet format for efficient storage/querying
- Use Polars for data analysis (5-10x faster than Pandas)
- Run with `uv run python` to ensure proper environment
