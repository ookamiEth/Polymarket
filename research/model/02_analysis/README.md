# Residual EDA: Quick Reference

## Overview

Comprehensive exploratory data analysis of Black-Scholes residuals for 15-minute binary options.

**Key Question Answered:** Do binary options appreciate linearly or non-linearly?

**Answer:** **STRONGLY NON-LINEAR** (CV = 137.6%, 138x delta variation)

---

## Files

### Analysis Script
- **`residual_eda_comprehensive.py`** - Main analysis script
  - Loads 39M test predictions (10% sample by default)
  - Generates 6 publication-quality visualizations
  - Computes comprehensive statistics

### Documentation
- **`RESIDUAL_EDA_SUMMARY.md`** - Full analysis report with interpretations

### Figures (300 DPI, Dark Theme)
Located in `figures/`:

1. **`01_residual_distribution.png`**
   - Histogram of residuals (actual - predicted)
   - Q-Q plot for normality test
   - **Finding:** Near-zero mean, symmetric, slightly heavy-tailed

2. **`02_residuals_vs_predicted.png`**
   - Heteroskedasticity check
   - **Finding:** Funnel shape, variance highest at p=0.5 (ATM)

3. **`03_residuals_vs_moneyness.png`**
   - Systematic bias detection
   - **Finding:** No major bias, well-calibrated across moneyness

4. **`04_probability_vs_moneyness.png`** ⭐
   - **Demonstrates non-linear appreciation**
   - Sigmoid curve (S-shape)
   - **Finding:** Steep at ATM, flat at tails

5. **`05_delta_vs_moneyness.png`** ⭐⭐
   - **Quantifies non-linearity**
   - Delta = ∂probability / ∂moneyness
   - **Finding:** Peak Δ=183.8 at ATM, min Δ≈0 in tails, CV=137.6%

6. **`06_calibration.png`**
   - Predicted vs actual outcomes
   - **Finding:** Brier Score = 0.1615, well-calibrated

---

## Quick Run

```bash
# Run with 10% sample (fast, ~30 seconds)
uv run python research/model/02_analysis/residual_eda_comprehensive.py

# Edit script to use full dataset (slower, ~5 minutes)
# Change: load_data(sample_frac=0.1) → load_data(sample_frac=1.0)
```

---

## Key Statistics

| Metric | Value | Interpretation |
|--------|-------|----------------|
| **Brier Score** | 0.1615 | BS baseline performance |
| **Mean Residual** | 0.0012 | Nearly unbiased |
| **MAE** | 0.349 | Average error magnitude |
| **Delta CV** | 137.6% | **Strong non-linearity** |
| **Max Delta** | 183.8 | Peak sensitivity at ATM |
| **Min Delta** | -2.4 | Low sensitivity in tails |

---

## Key Insights

### 1. Non-Linearity is Definitive

**Evidence:**
- Coefficient of Variation = 137.6% (>>25% threshold)
- Delta varies 138x from tails to ATM
- Clear sigmoid shape (not linear)

**Implication:**
- ✓ Current approach (BS baseline + ML corrections) is correct
- ✗ Linear regression on moneyness would fail

### 2. Heteroskedastic Errors

**Pattern:**
- Highest variance at ATM (p≈0.5)
- Lower variance in tails (p→0, p→1)

**Implication:**
- Consider weighted loss functions
- Separate models for ATM vs OTM might help

### 3. Well-Calibrated Baseline

**Finding:**
- BS model predictions closely match outcomes
- No systematic over/under-prediction

**Implication:**
- Good foundation for ML corrections
- XGBoost improvements are incremental (5.6% Brier reduction)

---

## Modeling Implications

### What NOT to Do ❌

```python
# Linear model assumes constant sensitivity
prob = β₀ + β₁ × moneyness  # WRONG!
```

### Correct Approach ✓

```python
# Use non-linear baseline
prob_bs = e^(-rT) × N(d₂)  # Black-Scholes

# Add ML corrections for residuals
prob_final = prob_bs + XGBoost(features)
```

---

## Risk Management

### Delta by Moneyness

| Moneyness | Delta | Interpretation |
|-----------|-------|----------------|
| **ATM (0%)** | 183.8 | 0.1% move → 18% prob change |
| **0.5% OTM** | ~20-30 | 0.1% move → 2-3% prob change |
| **1% OTM** | ~5-10 | 0.1% move → 0.5-1% prob change |
| **2% OTM** | ~0-5 | Very low sensitivity |

### Position Sizing

**Same $ exposure, vastly different risk:**
- ATM: High P&L volatility, frequent rebalancing needed
- OTM: Low P&L volatility, stable positions

---

## Next Steps

### Immediate
- [x] Confirm non-linearity ✓
- [x] Plot residuals ✓
- [x] Analyze calibration ✓

### Follow-Up
- [ ] Feature importance analysis (XGBoost SHAP values)
- [ ] Regime-specific analysis (high vol vs low vol)
- [ ] Temporal stability (performance over time)
- [ ] Compare to alternative models (jump models, discrete tick models)

### Risk Management
- [ ] Build delta ladder by moneyness
- [ ] Portfolio gamma exposure analysis
- [ ] Simulate delta-hedging strategies

---

## Contact

Questions or issues? Check:
- `RESIDUAL_EDA_SUMMARY.md` - Full analysis report
- `research/model/README.md` - Model documentation
- `research/model/HONEST_VALIDATION_ANALYSIS.md` - Validation methodology
