# Binary Option Pricing Calibration Research

**Research Question:** How well do implied volatilities from Deribit BTC options predict outcomes of 15-minute binary contracts?

**Period:** October 2023 - September 2025

**Approach:** Price hypothetical Polymarket-style 15-minute binary options using Deribit options IV, then validate against actual BTC price outcomes.

---

## Table of Contents

1. [Research Objective](#research-objective)
2. [Data Sources](#data-sources)
3. [Mathematical Framework](#mathematical-framework)
4. [Methodology](#methodology)
5. [Implementation Structure](#implementation-structure)
6. [Analysis Plan](#analysis-plan)
7. [Performance Standards](#performance-standards)
8. [Running the Backtest](#running-the-backtest)

---

## Research Objective

### Primary Goal
**Validate the accuracy of Black-Scholes binary option pricing for ultra-short-dated (15-minute) BTC contracts.**

### Core Hypothesis
Implied volatilities extracted from Deribit BTC options markets contain sufficient predictive information to accurately price binary options on 15-minute BTC price movements.

### Success Criteria
1. **Calibration**: Model probabilities match realized frequencies (calibration plot shows 45° line)
2. **Brier Score**: < 0.25 (lower is better, 0 = perfect)
3. **No systematic bias**: Regression analysis shows no significant moneyness or time-dependent biases

---

## Data Sources

### 1. BTC Options Implied Volatility
**File:** `/Users/lgierhake/Documents/ETH/BT/research/tardis/data/consolidated/btc_options_atm_shortdated_with_iv_2023_2025.parquet`

**Rows:** 204,722,673
**Period:** Oct 2023 - Sep 2025

**Key Columns:**
- `timestamp_seconds`: Unix timestamp (seconds)
- `symbol`: Option symbol (e.g., "BTC-2OCT23-27250-C")
- `type`: "call" or "put"
- `strike_price`: Option strike
- `expiry_timestamp`: Option expiration (Unix seconds)
- `time_to_expiry_seconds`: Time to expiry
- `spot_price`: BTC spot price at quote time
- `moneyness`: S/K ratio
- `implied_vol_bid`: IV calculated from bid price
- `implied_vol_ask`: IV calculated from ask price
- `has_bid`, `has_ask`: Boolean flags for quote availability
- `iv_calc_status`: "success" if IV calculation succeeded

**Usage:** Extract σ (implied volatility) for pricing binary options.

---

### 2. BTC Spot Price (Perpetual Futures)
**File:** `/Users/lgierhake/Documents/ETH/BT/research/tardis/data/consolidated/btc_perpetual_1s_2023_2025.parquet`

**Rows:** 13,505,552 trades
**Period:** Oct 2023 - Oct 2025

**Key Columns:**
- `timestamp`: Unix timestamp (microseconds)
- `price`: Trade price
- `amount`: Trade size
- `side`: "buy" or "sell"

**Preprocessing Required:** Resample trade data to 1-second bars (VWAP or last price).

**Usage:** Provides S (spot price) for binary option pricing and contract outcomes.

---

### 3. Risk-Free Rate
**File:** `/Users/lgierhake/Documents/ETH/BT/research/risk_free_rate/data/blended_lending_rates_2023_2025.parquet`

**Rows:** 731 days
**Period:** Oct 2023 - Sep 2025

**Key Columns:**
- `date`: Calendar date
- `blended_supply_apr`: Blended USDC/USDT lending rate (annual %)

**Usage:** Provides r (risk-free rate) for discounting. Convert APR to decimal (e.g., 4.27% → 0.0427).

---

## Mathematical Framework

### Binary Option Definition

A **binary (digital) option** pays a fixed amount ($1) if the underlying asset finishes above the strike at expiration, otherwise $0.

**Payoff:**
```
Payoff = { $1  if S(T) > K
         { $0  if S(T) ≤ K
```

Where:
- `S(T)` = Spot price at expiration
- `K` = Strike price (set at contract open)

---

### Black-Scholes Binary Option Pricing

**Fair Value Formula:**
```
Price = e^(-rT) × N(d₂)
```

Where:
- `Price` = Fair value of binary option (probability-adjusted and discounted)
- `e^(-rT)` = Discount factor (present value of $1 received at time T)
- `N(d₂)` = Risk-neutral probability of finishing in-the-money
- `r` = Risk-free interest rate (annual, as decimal)
- `T` = Time to expiration (in years)

---

### Calculating d₂

**Formula:**
```
d₂ = [ln(S/K) + (r - σ²/2)T] / (σ√T)
```

Where:
- `S` = Current spot price (BTC price at time of pricing)
- `K` = Strike price (BTC price at contract open, fixed)
- `ln(S/K)` = Log-moneyness (positive = ITM, negative = OTM)
- `σ` = Implied volatility (annual, as decimal, e.g., 0.45 = 45 vol)
- `T` = Time to expiration (in years)
- `r` = Risk-free rate (annual, as decimal)

**Numerator breakdown:**
- `ln(S/K)`: How far in/out of the money (log scale)
- `(r - σ²/2)T`: Drift adjustment accounting for interest and volatility drag

**Denominator:**
- `σ√T`: Volatility scaled to time horizon

---

### Cumulative Normal Distribution N(d)

**Definition:** `N(d)` returns the probability that a standard normal variable is ≤ d.

**Properties:**
- `N(0) = 0.50` (50% probability)
- `N(1) = 0.841` (84% probability)
- `N(2) = 0.977` (98% probability)
- `N(-1) = 0.159` (16% probability)

**Implementation:**
- Python: `from scipy.stats import norm; norm.cdf(d2)`
- Polars: `pl.col("d2").map_batches(lambda s: norm.cdf(s.to_numpy()))`

---

### Unit Conversions

#### Time to Years
```
T = seconds_remaining / (365.25 × 24 × 60 × 60)
T = seconds_remaining / 31,557,600

For 15 minutes: T = 900 / 31,557,600 = 0.0000285 years
For 1 minute: T = 60 / 31,557,600 = 0.0000019 years
```

#### Implied Volatility
```
σ_decimal = implied_vol (already in decimal form from dataset)

Example: implied_vol_bid = 0.45 means 45% annual volatility
```

#### Risk-Free Rate
```
r_decimal = blended_supply_apr / 100

Example: blended_supply_apr = 4.27 → r = 0.0427
```

---

### Simplified Formula for Short Periods

For ultra-short expirations (T < 1 hour), the discount factor is negligible:

```
e^(-rT) ≈ 1  when T is very small

Therefore: Price ≈ N(d₂)
```

**Example:**
- r = 0.05 (5%)
- T = 900 seconds = 0.0000285 years
- e^(-0.05 × 0.0000285) = e^(-0.000001425) ≈ 0.999998 ≈ 1

**Implication:** For 15-minute options, the price is essentially the risk-neutral probability N(d₂).

---

## Methodology

### 1. Contract Structure

**Schedule:** Fixed hourly schedule matching Polymarket BTC 15-minute markets.

**Contracts per hour:** 4
- Contract 1: Opens :00:00, closes :15:00
- Contract 2: Opens :15:00, closes :30:00
- Contract 3: Opens :30:00, closes :45:00
- Contract 4: Opens :45:00, closes :60:00

**Example:**
```
Contract opens:  2023-10-01 10:00:00 UTC
Contract closes: 2023-10-01 10:15:00 UTC
Strike K:        BTC price at 10:00:00 = $27,500
Outcome:         BTC price at 10:15:00 = $27,650 > $27,500 → Payoff = $1
```

**Total contracts:** ~4 contracts/hour × 24 hours/day × 730 days ≈ **70,000 contracts**

---

### 2. Pricing Workflow (Per Contract)

For each contract, price the binary option **every second** from open to close (900 pricing calculations per contract).

**Step 1: Set Strike (at contract open)**
```
K = BTC_spot_price(contract_open_time)
```

**Step 2: For each second t during contract lifetime:**

**A. Get Spot Price S(t)**
```
S = BTC_perpetual_price_1s(t)
```

**B. Calculate Time Remaining T**
```
T = (contract_close_time - t) / 31,557,600
```

**C. Get Implied Volatility σ(t)**

**Critical constraint:** Only use options that expire **AFTER** the binary contract closes.

```python
# Filter options data at time t:
valid_options = options_df.filter(
    (pl.col("timestamp_seconds") == t) &
    (pl.col("expiry_timestamp") > contract_close_time)
)

# Select closest expiry:
closest_option = valid_options.sort("time_to_expiry_seconds").first()

# Extract IV (three versions):
σ_bid = closest_option["implied_vol_bid"]
σ_ask = closest_option["implied_vol_ask"]
σ_mid = (σ_bid + σ_ask) / 2
```

**D. Get Risk-Free Rate r**
```
date = datetime.fromtimestamp(t).date()
r = risk_free_rates_df.filter(pl.col("date") == date)["blended_supply_apr"][0] / 100
```

**E. Calculate Binary Price**

For each of σ_bid, σ_ask, σ_mid:

```python
# Calculate d2
d2 = (np.log(S / K) + (r - 0.5 * σ**2) * T) / (σ * np.sqrt(T))

# Calculate N(d2)
prob = norm.cdf(d2)

# Discount (approximately 1 for short periods)
discount = np.exp(-r * T)

# Binary price
price = discount * prob
```

**F. Store Result**
```
result = {
    "contract_id": contract_id,
    "timestamp": t,
    "S": S,
    "K": K,
    "T_seconds": contract_close_time - t,
    "T_years": T,
    "r": r,
    "sigma_bid": σ_bid,
    "sigma_ask": σ_ask,
    "sigma_mid": σ_mid,
    "d2_bid": d2_bid,
    "d2_ask": d2_ask,
    "d2_mid": d2_mid,
    "price_bid": price_bid,
    "price_ask": price_ask,
    "price_mid": price_mid,
    "outcome": None  # Set after contract closes
}
```

**Step 3: Observe Outcome (at contract close)**
```
outcome = 1 if BTC_spot_price(contract_close_time) > K else 0

# Update all records for this contract:
results.filter(pl.col("contract_id") == contract_id).update(outcome=outcome)
```

---

### 3. Vectorization Strategy (Critical)

**DO NOT** loop over seconds or contracts in Python. Use Polars vectorization.

**Approach:**

**A. Pre-generate contract schedule:**
```python
# Generate all contract open/close times for 2023-2025
contracts = generate_contract_schedule(start_date="2023-10-01", end_date="2025-09-30")
# columns: contract_id, open_time, close_time
```

**B. Generate all second timestamps:**
```python
# For each contract, create 900 rows (one per second)
pricing_grid = contracts.explode(
    pl.int_range(0, 900, eager=True).alias("seconds_offset")
).with_columns([
    (pl.col("open_time") + pl.col("seconds_offset")).alias("timestamp")
])
# columns: contract_id, open_time, close_time, timestamp
```

**C. Join spot prices:**
```python
pricing_grid = pricing_grid.join(
    perpetual_1s,
    on="timestamp",
    how="left"
).rename({"price": "S"})
```

**D. Join risk-free rates:**
```python
pricing_grid = pricing_grid.with_columns(
    pl.col("timestamp").cast(pl.Date).alias("date")
).join(
    risk_free_rates,
    on="date",
    how="left"
).with_columns([
    (pl.col("blended_supply_apr") / 100).alias("r")
])
```

**E. Join implied volatilities (complex - requires closest expiry logic):**

This is the most complex join. See `01_pricing/match_options_iv.py` for implementation.

**F. Vectorized pricing calculation:**
```python
pricing_grid = pricing_grid.with_columns([
    # Set strike at contract open
    pl.col("S").first().over("contract_id").alias("K"),

    # Calculate time remaining
    ((pl.col("close_time") - pl.col("timestamp"))).alias("T_seconds"),
    ((pl.col("close_time") - pl.col("timestamp")) / 31_557_600).alias("T_years"),

    # Calculate d2 for bid/ask/mid
    (
        (pl.col("S") / pl.col("K")).log() +
        (pl.col("r") - 0.5 * pl.col("sigma_bid")**2) * pl.col("T_years")
    ) / (pl.col("sigma_bid") * pl.col("T_years").sqrt()).alias("d2_bid"),

    # ... repeat for ask and mid
])

# Apply normal CDF (requires mapping function)
from scipy.stats import norm

pricing_grid = pricing_grid.with_columns([
    pl.col("d2_bid").map_batches(lambda s: norm.cdf(s.to_numpy())).alias("prob_bid"),
    pl.col("d2_ask").map_batches(lambda s: norm.cdf(s.to_numpy())).alias("prob_ask"),
    pl.col("d2_mid").map_batches(lambda s: norm.cdf(s.to_numpy())).alias("prob_mid"),
])

# Apply discount
pricing_grid = pricing_grid.with_columns([
    ((-pl.col("r") * pl.col("T_years")).exp()).alias("discount"),
    (pl.col("discount") * pl.col("prob_bid")).alias("price_bid"),
    (pl.col("discount") * pl.col("prob_ask")).alias("price_ask"),
    (pl.col("discount") * pl.col("prob_mid")).alias("price_mid"),
])
```

**G. Join outcomes:**
```python
# Get spot price at close for each contract
outcomes = perpetual_1s.rename({"timestamp": "close_time", "price": "S_close"})

pricing_grid = pricing_grid.join(
    outcomes,
    on="close_time",
    how="left"
).with_columns([
    (pl.col("S_close") > pl.col("K")).cast(pl.Int8).alias("outcome")
])
```

**H. Write results (streaming):**
```python
pricing_grid.sink_parquet(
    "results/pricing_results.parquet",
    compression="snappy",
    streaming=True
)
```

**Expected output size:** ~70,000 contracts × 900 seconds = **63 million rows**

---

## Implementation Structure

```
research/model/
├── README.md                           # This file
│
├── 00_data_prep/                       # Data preparation scripts
│   ├── resample_perpetual_1s.py       # Resample trades → 1s VWAP/last
│   ├── generate_contract_schedule.py  # Generate 15-min contract calendar
│   └── validate_data_alignment.py     # Validate timestamp coverage
│
├── 01_pricing/                         # Pricing engine
│   ├── black_scholes_binary.py        # BS binary pricing functions
│   ├── match_options_iv.py            # Find closest expiry option
│   └── backtest_engine.py             # Main vectorized backtest
│
├── 02_analysis/                        # Analysis scripts
│   ├── calibration_analysis.py        # Calibration plots & metrics
│   ├── time_to_expiry_analysis.py     # Accuracy vs time remaining
│   ├── moneyness_analysis.py          # Accuracy vs S/K
│   └── regression_analysis.py         # Feature regression (optional)
│
├── 03_visualization/                   # Plotting scripts
│   ├── plot_calibration.py            # Calibration plots
│   ├── plot_time_series.py            # Example price evolution
│   └── plot_error_distributions.py    # Error histograms by regime
│
├── notebooks/                          # Jupyter notebooks
│   ├── 01_data_exploration.ipynb      # EDA on datasets
│   ├── 02_single_contract_test.ipynb  # Deep dive on one contract
│   └── 03_full_backtest_results.ipynb # Full results analysis
│
└── results/                            # Output files
    ├── pricing_results.parquet         # Full second-by-second data
    ├── contract_summary.parquet        # One row per contract
    ├── calibration_metrics.csv         # Summary statistics
    └── figures/                        # All plots
```

---

## Analysis Plan

### 1. Calibration Analysis (Primary)

**Goal:** Validate that predicted probabilities match realized frequencies.

**Method:**
1. Bucket all price predictions into deciles: [0-0.1), [0.1-0.2), ..., [0.9-1.0]
2. For each bucket, calculate actual win rate (fraction of outcomes = 1)
3. Plot: x = predicted probability (bucket midpoint), y = actual frequency
4. Perfect calibration = 45° line

**Metrics:**
- **Brier Score:** `mean((price - outcome)²)` (lower is better, range [0, 1])
- **Log Loss:** `mean(-outcome*log(price) - (1-outcome)*log(1-price))`
- **Calibration Error:** `mean(|predicted_prob - actual_freq|)` across buckets

**Code:** `02_analysis/calibration_analysis.py`

**Output:**
- `results/figures/calibration_plot_bid.png`
- `results/figures/calibration_plot_ask.png`
- `results/figures/calibration_plot_mid.png`
- `results/calibration_metrics.csv`

---

### 2. Time-to-Expiry Analysis

**Goal:** Understand how pricing accuracy changes as contract approaches expiration.

**Method:**
1. Create time buckets: [900-600s], [600-300s], [300-60s], [<60s]
2. For each bucket, calculate RMSE and calibration metrics
3. Plot: x = time bucket, y = RMSE

**Hypothesis:** Model should be more accurate with more time remaining (T → 0 may have edge effects).

**Code:** `02_analysis/time_to_expiry_analysis.py`

**Output:**
- `results/figures/accuracy_vs_time.png`
- `results/time_to_expiry_metrics.csv`

---

### 3. Moneyness Analysis

**Goal:** Validate pricing accuracy across different moneyness levels.

**Method:**
1. Create moneyness buckets:
   - Deep OTM: S/K < 0.98
   - OTM: 0.98 ≤ S/K < 0.995
   - ATM: 0.995 ≤ S/K < 1.005
   - ITM: 1.005 ≤ S/K < 1.02
   - Deep ITM: S/K ≥ 1.02
2. Calculate calibration metrics per bucket
3. Plot calibration curves for each bucket

**Hypothesis:** Model should be accurate across all moneyness levels, but may degrade for deep OTM/ITM where σ estimates are less reliable.

**Code:** `02_analysis/moneyness_analysis.py`

**Output:**
- `results/figures/calibration_by_moneyness.png`
- `results/moneyness_metrics.csv`

---

### 4. Regression Analysis (Optional)

**Goal:** Identify systematic biases and feature importance.

**Method:**
1. Create pricing error: `error = outcome - price_mid`
2. Generate features:
   - `moneyness = S/K - 1`
   - `log_moneyness = ln(S/K)`
   - `sigma_mid`
   - `T_seconds`
   - `bid_ask_spread = sigma_ask - sigma_bid`
   - `vol_regime = sigma_mid > rolling_mean(sigma_mid, 30 days)`
   - Interactions: `moneyness × T_seconds`, `sigma × T_seconds`
3. Linear regression: `error ~ β₀ + β₁×features`
4. Analyze coefficients and significance

**Interpretation:**
- `β_moneyness > 0`: Model underprices ITM, overprices OTM
- `β_sigma > 0`: Model underprices in high vol regimes
- `β_T < 0`: Model overprices contracts near expiration

**Code:** `02_analysis/regression_analysis.py`

**Output:**
- `results/regression_coefficients.csv`
- `results/figures/feature_importance.png`

---

## Performance Standards

### Vectorization Requirements

**FORBIDDEN:**
```python
# ❌ NO Python loops over DataFrame rows
for i in range(len(df)):
    price = calculate_price(df[i, "S"], df[i, "K"], ...)

# ❌ NO apply() with row-wise lambdas
df["price"] = df.apply(lambda row: calculate_price(row["S"], row["K"], ...), axis=1)

# ❌ NO iterrows()
for idx, row in df.iterrows():
    ...
```

**REQUIRED:**
```python
# ✅ Vectorized Polars operations
df = df.with_columns([
    ((pl.col("S") / pl.col("K")).log()).alias("log_moneyness")
])

# ✅ Vectorized joins instead of dictionary lookups
df = df.join(spot_prices, on="timestamp", how="left")

# ✅ Batch mapping for functions without native Polars support
df = df.with_columns([
    pl.col("d2").map_batches(lambda s: norm.cdf(s.to_numpy())).alias("N_d2")
])
```

### Memory Management

**For outputs >100M rows:**
```python
# ✅ Use streaming writes
df.sink_parquet("results/pricing_results.parquet", streaming=True)

# ❌ DO NOT collect then write
df.collect().write_parquet("results/pricing_results.parquet")  # OOM risk!
```

**For intermediate aggregations:**
```python
# ✅ Use lazy evaluation
lazy_df = pl.scan_parquet("results/pricing_results.parquet")
stats = lazy_df.select([
    pl.len().alias("count"),
    pl.col("price_mid").mean().alias("mean_price"),
]).collect()

# ❌ DO NOT load full dataset
df = pl.read_parquet("results/pricing_results.parquet")  # 63M rows in memory!
stats = df.select([pl.len(), pl.mean("price_mid")])
```

---

## Running the Backtest

### 1. Data Preparation

```bash
# Resample perpetual trades to 1-second bars
uv run python research/model/00_data_prep/resample_perpetual_1s.py

# Generate contract schedule
uv run python research/model/00_data_prep/generate_contract_schedule.py

# Validate data alignment
uv run python research/model/00_data_prep/validate_data_alignment.py
```

**Outputs:**
- `research/model/results/btc_perpetual_1s_resampled.parquet`
- `research/model/results/contract_schedule.parquet`

---

### 2. Run Backtest

```bash
# Main backtest engine (this will take time - progress bars included)
uv run python research/model/01_pricing/backtest_engine.py
```

**Expected runtime:** 30-60 minutes (depending on machine)

**Output:**
- `research/model/results/pricing_results.parquet` (~63M rows, ~5GB)

---

### 3. Analysis

```bash
# Calibration analysis
uv run python research/model/02_analysis/calibration_analysis.py

# Time-to-expiry analysis
uv run python research/model/02_analysis/time_to_expiry_analysis.py

# Moneyness analysis
uv run python research/model/02_analysis/moneyness_analysis.py

# (Optional) Regression analysis
uv run python research/model/02_analysis/regression_analysis.py
```

**Outputs:** Figures in `results/figures/`, metrics in `results/`

---

### 4. Explore Results

```bash
# Open Jupyter
jupyter notebook research/model/notebooks/03_full_backtest_results.ipynb
```

Or query directly with Polars:
```python
import polars as pl

# Lazy scan results
df = pl.scan_parquet("research/model/results/pricing_results.parquet")

# Example: Get all contracts where model gave >80% probability but lost
bad_predictions = df.filter(
    (pl.col("price_mid") > 0.8) & (pl.col("outcome") == 0)
).collect()

print(bad_predictions.head())
```

---

## Future Extensions

**Phase 2 Research (noted in code for later):**

1. **Jump-Diffusion Pricing**
   - Add discrete jump component to diffusion model
   - Estimate jump frequency (λ) and jump size distribution
   - Compare calibration vs pure diffusion

2. **Event Volatility Decomposition**
   - Identify scheduled events (FOMC, CPI, etc.)
   - Decompose IV into base vol + event vol
   - Price contracts expiring before/after events differently

3. **Real Polymarket Comparison**
   - When Polymarket historical data is available
   - Compare model prices to actual market prices
   - Identify mispricing opportunities

4. **Trading Strategy**
   - Define entry/exit rules based on mispricing
   - Backtest PnL with realistic transaction costs
   - Optimize position sizing

5. **Model Improvements**
   - Incorporate realized volatility forecasts
   - Test GARCH models for volatility
   - Machine learning for volatility prediction

---

## References

**Black-Scholes Binary Option Pricing:**
- Hull, J. C. (2018). *Options, Futures, and Other Derivatives*. Chapter 26: Exotic Options.

**Volatility Modeling:**
- Gatheral, J. (2006). *The Volatility Surface: A Practitioner's Guide*.

**Calibration Testing:**
- Niculescu-Mizil, A., & Caruana, R. (2005). *Predicting Good Probabilities With Supervised Learning*.

---

**Document Version:** 1.0
**Last Updated:** 2025-10-23
**Author:** Claude Code Research Assistant
