# Baseline Regeneration Pipeline

**Purpose**: Regenerate missing baseline data files required for V4 feature engineering

**Last Updated**: 2025-11-12

---

## Overview

This directory contains the complete pipeline for regenerating baseline data files used by the V4 options pricing model. The baseline files include:

1. **BTC Options Data** (`btc_options_atm_shortdated_with_iv_2023_2025.parquet`) - 52GB
2. **BTC Perpetual Data** (`btc_perpetual_1s_2023_2025.parquet`) - 24GB
3. **Lending Rates** (`blended_lending_rates_1s.parquet`) - 2.4GB
4. **Contract Schedule** (`contract_schedule.parquet`) - 1.1MB

**Total Regeneration Time**: 12-18 hours (parallelized)

---

## 📁 Directory Structure

```
baseline_regeneration/
├── README.md                              # This file
├── BASELINE_REGENERATION_GUIDE.md         # Detailed workflow guide
├── regenerate_baseline_files.sh           # Main regeneration script
├── validate_baseline_files.py             # Validation script
└── analysis/                              # IV failure analysis
    ├── README.md
    ├── analyze_iv_failures.py
    └── analyze_iv_failure_clustering.py
```

---

## 🚀 Quick Start

### 1. Regenerate All Baseline Files

**Full regeneration** (12-18 hours):
```bash
cd /home/ubuntu/Polymarket/research/baseline_regeneration
./regenerate_baseline_files.sh
```

**Monitor progress**:
```bash
tail -f /home/ubuntu/Polymarket/research/tardis/logs/regeneration_*.log
```

### 2. Validate Baseline Files

After regeneration completes:
```bash
uv run python validate_baseline_files.py
```

**Expected output**:
```
✅ btc_options_atm_shortdated_with_iv_2023_2025.parquet: 73.4M rows, 52GB
✅ btc_perpetual_1s_2023_2025.parquet: 63.1M rows, 24GB
✅ blended_lending_rates_1s.parquet: 63.1M rows, 2.4GB
✅ contract_schedule.parquet: 2,737 rows, 1.1MB
```

---

## 📋 Baseline Files

### BTC Options (52GB)
**File**: `research/tardis/data/consolidated/btc_options_atm_shortdated_with_iv_2023_2025.parquet`

**Contains**:
- Deribit BTC options quotes (2023-01-01 to 2025-10-31)
- Filtered to ATM (|moneyness| < 3%) and short-dated (≤30 days)
- Implied volatility calculated via Black-Scholes
- ~73.4M rows (from 1.1B raw quotes)

**Generation Time**: 8-12 hours (streaming processing)

**Known Issues**:
- 2.4% IV calculation failures (see `analysis/` for investigation)
- Failures clustered around extreme moneyness and high volatility periods

### BTC Perpetual (24GB)
**File**: `research/tardis/data/consolidated/btc_perpetual_1s_2023_2025.parquet`

**Contains**:
- Deribit BTC perpetual quotes resampled to 1-second intervals
- Spot price reference for options pricing
- ~63.1M rows (1-second resolution over 2.5 years)

**Generation Time**: 2-3 hours

### Lending Rates (2.4GB)
**File**: `research/tardis/data/consolidated/blended_lending_rates_1s.parquet`

**Contains**:
- Blended USD and USDC lending rates from Aave V3
- Resampled to 1-second intervals to match perpetual data
- Risk-free rate proxy for Black-Scholes model
- ~63.1M rows

**Generation Time**: 1-2 hours

### Contract Schedule (1.1MB)
**File**: `research/tardis/data/consolidated/contract_schedule.parquet`

**Contains**:
- Mapping of Deribit contract IDs to expiration dates
- Used for time-to-expiry calculations
- 2,737 unique contracts

**Generation Time**: <1 minute

---

## 🔄 Regeneration Workflow

The `regenerate_baseline_files.sh` script runs 4 pipelines in parallel:

### Pipeline 1: BTC Options with IV (8-12 hours)
```bash
# Step 1: Filter to ATM + short-dated (3-4 hours)
filter_atm_short_dated.py
# Output: 73.4M rows from 1.1B

# Step 2: Calculate implied volatility (5-8 hours, streaming)
calculate_iv_streaming.py
# Uses Black-Scholes with streaming processing
# Memory usage: <5GB (via .sink_parquet)
```

### Pipeline 2: BTC Perpetual (2-3 hours)
```bash
# Resample to 1-second intervals
resample_perpetual_to_1s.py
# Output: 63.1M rows, 24GB
```

### Pipeline 3: Lending Rates (1-2 hours)
```bash
# Step 1: Download Aave data
download_aave_lending_rates.py

# Step 2: Blend USD + USDC rates
blend_lending_rates.py

# Step 3: Resample to 1-second intervals
resample_lending_rates_to_1s.py
# Output: 63.1M rows, 2.4GB
```

### Pipeline 4: Contract Schedule (<1 minute)
```bash
# Generate expiration mapping
generate_contract_schedule.py
# Output: 2,737 rows, 1.1MB
```

---

## ✅ Validation

The `validate_baseline_files.py` script checks:

1. **File existence**: All 4 files present
2. **File sizes**: Within expected ranges (±10%)
3. **Row counts**: Within expected ranges (±5%)
4. **Schema validation**: All required columns present
5. **Data quality**:
   - No null values in critical columns
   - Timestamps sorted and within expected range
   - Price values positive and reasonable
6. **IV quality** (for options file):
   - IV success rate >95%
   - IV values in reasonable range (0.1 to 5.0)

**Exit codes**:
- `0` - All validations passed
- `1` - Validation failures (see output for details)

---

## 🔬 Analysis

The `analysis/` subdirectory contains scripts investigating the 2.4% IV calculation failures in the BTC options baseline file.

**Key Findings**:
- Failures cluster around extreme moneyness (|moneyness| > 2%)
- Higher failure rate during high volatility periods
- Most failures occur near expiration (<1 day remaining)
- Numerical instability in Black-Scholes solver for edge cases

**See**: `analysis/README.md` for detailed investigation

---

## 🛠️ Troubleshooting

### "File not found" error
**Cause**: Baseline files don't exist yet
**Solution**: Run `./regenerate_baseline_files.sh` to generate them

### "IV calculation hanging"
**Cause**: Streaming write may appear stuck but is still processing
**Solution**: Check log file for progress indicators, wait 5-8 hours for completion

### "OOM (out of memory) error"
**Cause**: Non-streaming operation on large dataset
**Solution**: Ensure scripts use `.sink_parquet(streaming=True)` for outputs >100M rows

### "Validation failures"
**Cause**: Regeneration incomplete or data quality issues
**Solution**: Check log files for errors during regeneration, re-run failed pipeline

---

## 📊 Disk Space Requirements

| Stage | Intermediate | Final | Total |
|-------|-------------|-------|-------|
| BTC Options | 60GB | 52GB | 112GB |
| BTC Perpetual | 30GB | 24GB | 54GB |
| Lending Rates | 5GB | 2.4GB | 7.4GB |
| **Total** | **95GB** | **78GB** | **173GB** |

**Recommendation**: Ensure 200GB free space before starting regeneration

---

## ⏱️ Estimated Timeline

| Task | Time | Can Run in Parallel? |
|------|------|---------------------|
| BTC Options (filter) | 3-4 hours | ✅ Yes |
| BTC Options (IV calc) | 5-8 hours | ✅ Yes |
| BTC Perpetual | 2-3 hours | ✅ Yes |
| Lending Rates | 1-2 hours | ✅ Yes |
| Contract Schedule | <1 minute | ✅ Yes |
| **Total (parallel)** | **8-12 hours** | - |
| **Total (sequential)** | **11-18 hours** | - |

**The regeneration script runs all pipelines in parallel automatically.**

---

## 🔗 Related Documentation

- **Detailed Guide**: `BASELINE_REGENERATION_GUIDE.md` - Step-by-step instructions
- **Analysis**: `analysis/README.md` - IV failure investigation
- **V4 Feature Engineering**: `/home/ubuntu/Polymarket/research/model/00_data_prep/engineer_all_features_v4.py`
- **Tardis Data Processing**: `/home/ubuntu/Polymarket/research/tardis/`

---

## 📞 Need Help?

1. Check `BASELINE_REGENERATION_GUIDE.md` for detailed workflow
2. Check `analysis/README.md` for IV failure analysis
3. Check log files in `research/tardis/logs/`
4. Review Tardis processing scripts in `research/tardis/scripts/`

---

**Maintained By**: V4 Development Team
**Last Cleanup**: 2025-11-12 (Moved from /BT root to organized directory)
