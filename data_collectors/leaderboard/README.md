# Top Polymarket Traders Analysis

This project identifies and analyzes the top 10 highest volume traders on Polymarket using their official API.

## Quick Start

```bash
# Run the analysis
uv run python find_top_traders.py

# View results
cat results/analysis_summary.md
```

## What This Does

1. Fetches leaderboard data from Polymarket's Data API
2. Sorts traders by total trading volume (USD)
3. Extracts top 10 highest volume traders
4. Generates comprehensive reports in multiple formats

## Output Files

All results are saved to `results/`:

- **`top_10_traders.json`** - Complete data with addresses, volumes, PnL
- **`top_10_traders.csv`** - Spreadsheet format for Excel/analysis
- **`analysis_summary.md`** - Human-readable markdown report

## Key Findings

### Top 10 Traders by Volume (as of 2025-10-01)

| Rank | Username | Volume (USD) | P&L (USD) |
|------|----------|--------------|-----------|
| 1 | 0x5375...aeea | $311,761.80 | $15,317.00 |
| 2 | SpartanWar | $276,902.35 | $5,979.00 |
| 3 | Punchbowl | $266,963.84 | $5,593.00 |
| 4 | piastri | $244,675.61 | $3,921.00 |
| 5 | TheMangler | $158,974.46 | $3,658.00 |
| 6 | 0x02e6...e25 | $154,332.09 | $13,548.00 |
| 7 | aadvark | $153,735.81 | $10,456.00 |
| 8 | fatbojangles | $140,726.38 | $4,652.00 |
| 9 | MariaLo | $140,490.05 | $15,667.00 |
| 10 | Dropper | $136,734.05 | $5,156.00 |

**Combined Statistics**:
- Total Volume: $1,985,296.43
- Total P&L: $83,947.00
- Average Volume per Trader: $198,529.64
- Average P&L per Trader: $8,394.70

## API Discovery

The key discovery was finding Polymarket's undocumented leaderboard endpoint:

```
GET https://data-api.polymarket.com/leaderboard
```

**Parameters**:
- `limit` - Number of results (default 25, max 50)
- `offset` - Pagination offset

**Response Format**:
```json
[
  {
    "rank": "1",
    "user_id": "0x...",
    "user_name": "username",
    "vol": 311761.8,
    "pnl": 15317.0,
    "profile_image": "https://..."
  }
]
```

## Project Structure

```
/top_traders/
├── find_top_traders.py           # Main script
├── requirements.txt              # Python dependencies
├── implementation_log.md         # Detailed implementation log
├── api_research/                 # API discovery scripts
│   ├── test_leaderboard_api.py
│   ├── test_leaderboard_params.py
│   └── discovery_results.json
└── results/                      # Generated output
    ├── top_10_traders.json
    ├── top_10_traders.csv
    └── analysis_summary.md
```

## Implementation Approach

### Initial Plan
Originally planned to:
1. Query top markets by volume
2. Extract top holders from each market
3. Calculate volume from trade history
4. Aggregate and rank

### Actual Implementation (Much Better!)
1. User discovered Polymarket has a public leaderboard
2. Tested potential API endpoints systematically
3. Found working endpoint: `https://data-api.polymarket.com/leaderboard`
4. Single API call retrieves all data
5. Client-side sorting by volume field

**Result**: ~5 second execution instead of estimated 2-5 minutes!

## Technical Details

### API Endpoint
- **URL**: `https://data-api.polymarket.com/leaderboard`
- **Method**: GET
- **Authentication**: Not required
- **Rate Limit**: 100 requests / 10 seconds (Data API)

### Data Fields
- `user_id` - Ethereum wallet address
- `user_name` - Profile username (or address if not set)
- `vol` - Total trading volume in USD
- `pnl` - Profit/loss in USD
- `rank` - Position in leaderboard (by PnL, not volume)
- `profile_image` - Profile picture URL

### Important Notes
1. **Default sorting is by PnL, not volume** - Had to re-sort client-side
2. **Max limit is 50** - Larger requests still return only 50 results
3. **No time period filters** - Appears to be all-time statistics
4. **Volume calculation** - Already computed by API, not raw trade data

## Related Documentation

- **Polymarket Docs**: https://docs.polymarket.com
- **Public Leaderboard**: https://polymarket.com/leaderboard
- **Implementation Log**: [implementation_log.md](implementation_log.md)

## Example Usage

### Get Top 10
```python
import requests

response = requests.get(
    "https://data-api.polymarket.com/leaderboard",
    params={"limit": 50}
)

leaderboard = response.json()
top_10_by_volume = sorted(leaderboard, key=lambda x: x['vol'], reverse=True)[:10]
```

### Access Results
```python
import json

with open('results/top_10_traders.json', 'r') as f:
    data = json.load(f)

print(f"Top trader: {data['top_10_traders'][0]['username']}")
print(f"Volume: ${data['top_10_traders'][0]['total_volume_usd']:,.2f}")
```

## Future Enhancements

Potential additions:
- [ ] Historical tracking (run daily, track changes)
- [ ] Volume distribution analysis
- [ ] Most active markets per trader
- [ ] Win rate calculations
- [ ] Comparison to overall market volume
- [ ] Time series analysis of top traders

## License

MIT License - This is a research/analysis tool for publicly available data.
