# Visualization Module - Complete Reference

**Purpose:** Comprehensive evaluation and communication toolkit for the LightGBM residual prediction model for BTC 15-minute binary options.

**Use Case:** After training a model, generate 20+ publication-quality plots to evaluate performance, diagnose issues, compare trials, and communicate results to stakeholders (quant researchers, risk team, traders).

---

## Directory Structure

```
visualization/
├── README.md (this file)
├── __init__.py - Module exports
├── plot_config.py - Centralized styling configuration
├── time_series_plots.py - Temporal analysis (Brier evolution, calibration by regime)
├── grid_search_plots.py - Hyperparameter tuning comparison (training curves, dashboards)
├── wandb_integration.py - W&B upload helpers and run fetching
├── advanced_diagnostics.py - Statistical evaluation (ROC, PR, QQ, Lift, Win Rate, SHAP) [IN PROGRESS]
├── trading_plots.py - Trading simulation (equity curves, P&L) [PLANNED]
├── simulation_plots.py - Uncertainty quantification (Monte Carlo bootstrap) [PLANNED]
└── generate_all_plots.py - Master orchestration script
```

---

## File-by-File Guide

### 1. `plot_config.py` (113 lines)

**Purpose:** Centralized styling configuration for all plots.

**What it does:**
- Defines professional color palette (cyan, magenta, green, red for dark backgrounds)
- Sets font sizes (14pt titles, 12pt labels, 10pt ticks) following CLAUDE.md standards
- Specifies dot sizing range (30-200) for scatter plots
- Provides helper functions for output paths and color selection

**Key Elements:**
```python
COLORS = {
    "primary": "#00D4FF",    # Cyan - main data
    "secondary": "#FF00FF",  # Magenta - secondary data
    "success": "#00FF88",    # Green - improvements
    "danger": "#FF3366",     # Red - baseline/warnings
}

FONT_SIZES = {
    "title": 14,
    "label": 12,
    "tick": 10,
}
```

**When to use:** Import colors/fonts when creating new plots to maintain consistency.

---

### 2. `time_series_plots.py` (533 lines) ✅

**Purpose:** Detect model degradation and regime-specific failures over time.

**Why important:** Models trained on 2 years of BTC data may perform well on average but fail during high volatility or specific time periods. These plots reveal temporal patterns.

#### Functions:

**a) `plot_brier_over_time()`**
- **What:** Dual plot showing Brier score evolution (daily/weekly aggregation)
- **Why:** Detect if model degrades over time or fails in specific periods
- **Output:**
  - Top panel: Baseline vs Model Brier scores (line plot)
  - Bottom panel: Improvement percentage over time
- **Alerts:** Warns if improvement <0% in any period
- **Example:** If Brier is 0.14 in Q1 but 0.18 in Q4, model is drifting

**b) `plot_calibration_by_regime()`**
- **What:** 3-panel calibration curves stratified by volatility (low/mid/high)
- **Why:** Model may be well-calibrated overall but over-confident in high volatility
- **Output:** Scatter plots of predicted vs actual probabilities per regime
- **Interpretation:** Points should lie on diagonal line (perfect calibration)
- **Example:** If high-vol regime shows predictions at 0.6 but actuals at 0.5, model overestimates

**c) `plot_prediction_distribution()`**
- **What:** Histogram + KDE showing prediction distribution
- **Why:** Detect "mode collapse" (model only predicts 0.5, not confident)
- **Output:** Comparison of model vs baseline prediction spread
- **Alerts:** Warns if >80% predictions in [0.45, 0.55] range
- **Example:** Healthy model should use full [0.1, 0.9] range

**d) `generate_time_series_report()`**
- **What:** Master function running all 3 plots above
- **Returns:** Dictionary with plot paths and summary statistics

**Usage:**
```bash
uv run python visualization/time_series_plots.py \
    --test-file ../results/test_predictions.parquet
```

---

### 3. `grid_search_plots.py` (576 lines) ✅

**Purpose:** Compare 16 hyperparameter trials to find optimal configuration and validate grid search.

**Why important:** Grid search tested 16 combinations of learning_rate, num_leaves, max_depth, min_child_samples. These plots show if we explored the right space and found true optimum.

#### Functions:

**a) `plot_training_curves_from_wandb()`**
- **What:** Overlaid validation Brier curves from all 16 trials
- **Why:** Compare convergence speed, stability, final performance
- **Data Source:** Fetches from W&B run histories
- **Output:** Single plot with 16 colored lines, one per trial
- **Interpretation:** Lines should converge; if one diverges, that config is bad
- **Example:** If Trial 12 (31 leaves) converges fastest to lowest Brier, it wins

**b) `plot_grid_search_summary()`**
- **What:** 2x2 dashboard with 4 visualizations
- **Output:**
  1. **Bar chart:** Brier improvement by trial (sorted)
  2. **Scatter:** Runtime vs performance (find sweet spot)
  3. **Sensitivity:** Learning rate vs Brier (colored by num_leaves)
  4. **Table:** Top 5 trials with hyperparameters
- **Why:** Comprehensive view of grid search results in one figure
- **Example:** If trials 10-13 cluster at 12.3% improvement, region is optimal

**c) `generate_grid_search_report()`**
- **What:** Master function running both plots above
- **Returns:** Summary statistics (best Brier, mean improvement, total runtime)

**Usage:**
```bash
uv run python visualization/grid_search_plots.py \
    --results ../results/grid_search_results.parquet
```

---

### 4. `wandb_integration.py` (393 lines) ✅

**Purpose:** Helper functions for W&B (Weights & Biases) experiment tracking.

**Why important:** W&B provides centralized dashboard for all runs, plots, and metrics. These helpers automate upload and fetching.

#### Functions:

**a) `upload_plot()` / `upload_plots_batch()`**
- **What:** Upload single/multiple PNG files to W&B
- **Usage:** `upload_plot("path/to/plot.png", "diagnostics/roc_curve")`
- **Why:** Centralizes plots in W&B dashboard for team visibility

**b) `upload_table()`**
- **What:** Upload data table (e.g., top trials) to W&B
- **Usage:** Show tabular results alongside plots

**c) `fetch_run_history()` / `fetch_multiple_runs()`**
- **What:** Download run histories from W&B API
- **Why:** Used by `plot_training_curves_from_wandb()` to get validation loss curves

**d) `compare_runs()`**
- **What:** Compare specific runs by a metric (e.g., Brier score)
- **Returns:** Sorted list of runs with best highlighted

**e) `log_model_summary()`**
- **What:** Log model config, metrics, and artifacts to W&B in one call
- **Usage:** Called at end of training to record model card

**Usage:**
```python
from visualization.wandb_integration import upload_plots_batch
upload_plots_batch({
    "roc_curve": "plots/roc.png",
    "calibration": "plots/calibration.png",
}, prefix="diagnostics/")
```

---

### 5. `advanced_diagnostics.py` (465 lines) 🚧 IN PROGRESS

**Purpose:** Statistical evaluation plots for binary probabilistic predictions.

**Why important:** Brier score alone doesn't tell full story. These plots diagnose discrimination, calibration, targeting efficiency, and distributional assumptions.

#### Functions (Implemented):

**a) `plot_roc_curve()`** ✅
- **What:** ROC (Receiver Operating Characteristic) curve with AUC (Area Under Curve)
- **Why:** Shows model's ability to separate UP from DOWN outcomes across all thresholds
- **Output:** 2x2 subplot grid:
  1. Overall ROC (model vs baseline)
  2. Low volatility regime
  3. Mid volatility regime
  4. High volatility regime
- **Interpretation:**
  - AUC = 0.5: Random (no skill)
  - AUC = 0.7-0.8: Good (usable trading edge)
  - AUC > 0.8: Excellent
- **Example:** If overall AUC = 0.75 but high-vol AUC = 0.62, model struggles in volatility

**b) `plot_precision_recall()`** ✅
- **What:** Precision-Recall curve with Average Precision (AP)
- **Why:** Better than ROC for imbalanced data (e.g., more UP than DOWN in bull markets)
- **Output:** Precision (PPV) vs Recall (Sensitivity) curve
- **Interpretation:**
  - High precision: Few false positives (good for conservative trading)
  - High recall: Catches most true positives (good for aggressive strategies)
- **Example:** AP = 0.65 means model ranks positives well

#### Functions (Planned):

**c) `plot_qq_residuals()`** 🔜
- **What:** QQ (Quantile-Quantile) plot comparing residuals to normal distribution
- **Why:** Validate assumption of Gaussian errors
- **Interpretation:** Straight line = normal; deviations in tails = fat-tailed risks
- **Example:** If tails deviate upward, BTC jumps create extreme residuals

**d) `plot_lift_chart()`** 🔜
- **What:** Lift chart showing targeting efficiency
- **Why:** Quantify "alpha" - if top 10% predictions capture 30% of outcomes, 3x lift
- **Interpretation:** Steeper curve = better targeting
- **Example:** Used to optimize position sizing (bet more on high-confidence predictions)

**e) `plot_win_rate_heatmap()`** 🔜
- **What:** 2D heatmap of win rate by (moneyness × time-to-expiry)
- **Why:** Identify pockets of strong/weak predictions
- **Output:** Color-coded grid (green = high win rate, red = low)
- **Example:** If OTM + <5min = 40% win rate, avoid that regime

**f) `plot_shap_dependence_analysis()`** 🔜
- **What:** SHAP dependence plots for top 10 features
- **Why:** Show how each feature affects predictions (non-linear effects)
- **Requires:** shap library, trained model, 100K row downsample
- **Example:** If iv_rv_ratio_900s > 1.5 increases predictions, captures vol regime

**g) `generate_diagnostics_report()`** 🚧
- **What:** Master function running all diagnostic plots
- **Current:** Runs ROC + PR curves
- **Final:** Will run all 6 diagnostic plots

**Usage:**
```bash
uv run python visualization/advanced_diagnostics.py \
    --test-file ../results/test_predictions.parquet
```

---

### 6. `trading_plots.py` (PLANNED) 🔜

**Purpose:** Simulate trading on model predictions to evaluate real-world profitability.

**Why important:** Brier score doesn't directly translate to P&L. These plots show cumulative returns, drawdowns, and Sharpe ratios.

#### Planned Functions:

**a) `plot_equity_curve()`**
- **What:** Cumulative P&L over time if trading on model predictions
- **Strategy:**
  - Long (bet UP) if final_prob > 0.55
  - Short (bet DOWN) if final_prob < 0.45
  - Flat (no position) if 0.45 ≤ prob ≤ 0.55
- **Output:** Line plot of cumulative returns with drawdown shading
- **Metrics:** Sharpe ratio, max drawdown, win rate
- **Comparison:** Model vs baseline (prob_mid)
- **Example:** If model Sharpe = 1.2 vs baseline 0.8, model has 50% better risk-adjusted returns

**b) `generate_trading_report()`**
- **What:** Master function for trading simulation
- **Returns:** Sharpe, drawdown, total return, trade count

---

### 7. `simulation_plots.py` (PLANNED) 🔜

**Purpose:** Quantify uncertainty in model performance via Monte Carlo bootstrap.

**Why important:** Single test set may be lucky/unlucky. Bootstrap resampling gives confidence intervals.

#### Planned Functions:

**a) `plot_bootstrap_ci()`**
- **What:** Bootstrap confidence intervals for Brier improvement
- **Method:** Resample test set 1,000 times, recompute Brier each time
- **Output:** Histogram of improvement metric with 95% CI
- **Interpretation:** If CI = [10%, 14%], improvement is robust; if [0%, 24%], uncertain
- **Stratification:** Separate plots per volatility regime
- **Example:** If high-vol CI = [-5%, +15%], model is unreliable in volatility

**b) `generate_simulation_report()`**
- **What:** Master function for Monte Carlo analysis
- **Returns:** Mean, 95% CI, per-regime CIs

---

### 8. `generate_all_plots.py` (382 lines) ✅

**Purpose:** Master orchestration script to generate all plots in one command.

**Why important:** Automates the entire visualization pipeline, coordinates W&B uploads, and generates HTML report.

**What it does:**
1. Loads test predictions (12.5M rows)
2. Generates time-series plots (Brier evolution, calibration, distribution)
3. Generates grid search plots (training curves, summary dashboard)
4. [FUTURE] Generates diagnostics (ROC, PR, QQ, Lift, Win Rate, SHAP)
5. [FUTURE] Generates trading plots (equity curve)
6. [FUTURE] Generates simulation plots (bootstrap CI)
7. Creates HTML report embedding all plots
8. Uploads everything to W&B (if enabled)

**Output:**
```
results/plots/
├── time_series/
│   ├── brier_over_time_daily.png
│   ├── brier_over_time_weekly.png
│   ├── calibration_by_volatility_regime.png
│   └── prediction_distribution.png
├── grid_search/
│   ├── training_curves.png
│   └── summary_dashboard.png
├── diagnostics/ [IN PROGRESS]
│   ├── roc_curve.png
│   ├── precision_recall.png
│   ├── qq_plot_residuals.png
│   ├── lift_chart.png
│   ├── win_rate_heatmap.png
│   └── shap_dependence_*.png (10 plots)
├── trading/ [PLANNED]
│   └── equity_curve.png
├── simulation/ [PLANNED]
│   └── bootstrap_brier_ci.png
└── report.html (comprehensive HTML report)
```

**Usage:**
```bash
# Generate all plots at once
uv run python visualization/generate_all_plots.py \
    --test-file ../results/test_predictions.parquet \
    --grid-search-file ../results/grid_search_results.parquet

# Skip W&B upload (for testing)
uv run python visualization/generate_all_plots.py \
    --test-file ../results/test_predictions.parquet \
    --no-wandb
```

**HTML Report:** Opens in browser, embeds all plots with explanatory text, metrics tables, and warnings.

---

## Quick Start Guide

### Step 1: After Training (Re-run with shuffle=False)

```bash
cd /home/ubuntu/Polymarket/research/model/01_pricing

# Re-train model with temporal ordering fix
uv run python lightgbm_memory_optimized.py \
    --config config/lightgbm_grid_search_config_optimized.yaml
```

This produces:
- `results/test_predictions.parquet` (12.5M rows with final_prob, outcome, residuals)
- `results/grid_search_results.parquet` (16 trials with hyperparameters and metrics)
- `results/model.pkl` (trained model for SHAP analysis)

### Step 2: Generate All Plots

```bash
# Full visualization suite
uv run python visualization/generate_all_plots.py \
    --test-file ../results/test_predictions.parquet \
    --grid-search-file ../results/grid_search_results.parquet \
    --wandb-project lightgbm-residual-tuning
```

### Step 3: View Results

- **Local:** Open `results/plots/report.html` in browser
- **W&B:** Visit W&B dashboard for interactive plots

### Step 4: Diagnose Issues

**If model performance is poor:**
1. Check **Brier time series** → Degradation over time? Add temporal features
2. Check **Calibration by regime** → Fails in high volatility? Tune regularization
3. Check **Win rate heatmap** → Weak in specific moneyness/expiry? Add interactions
4. Check **SHAP plots** → Irrelevant features dominating? Feature selection

**If training is unstable:**
1. Check **Training curves** → Oscillating? Lower learning_rate
2. Check **Complexity graph** → Overfitting? Reduce num_leaves
3. Check **Bootstrap CI** → Wide confidence intervals? Collect more data

---

## Plot Decision Tree

**Use this to choose the right plot for your question:**

```
QUESTION: Is my model better than baseline?
├─ Overall discrimination → ROC Curve (AUC)
├─ Imbalanced data → Precision-Recall Curve (AP)
└─ Targeting efficiency → Lift Chart

QUESTION: Is my model well-calibrated?
├─ Overall calibration → Calibration plot (existing)
├─ By regime → Calibration by regime (time_series_plots)
└─ Confidence spread → Prediction distribution

QUESTION: Does my model degrade over time?
└─ Brier time series (daily/weekly)

QUESTION: Where does my model fail?
├─ By volatility → ROC by regime, Calibration by regime
├─ By moneyness × expiry → Win rate heatmap
└─ By time period → Brier time series

QUESTION: Why does my model make these predictions?
├─ Feature importance → LightGBM gain (existing)
├─ Feature interactions → SHAP dependence plots
└─ Non-linear effects → SHAP dependence (scatter)

QUESTION: Is my model profitable?
├─ Simulated P&L → Equity curve
└─ Risk-adjusted returns → Sharpe ratio (from equity curve)

QUESTION: How confident am I in results?
├─ Bootstrap uncertainty → Monte Carlo CI
└─ Regime-specific uncertainty → Bootstrap CI per regime

QUESTION: Did grid search work?
├─ Convergence → Training curves from W&B
├─ Optimal region → Complexity graph
└─ Hyperparameter sensitivity → Grid search summary dashboard
```

---

## Implementation Status

### ✅ Phase 0: Core Infrastructure (Complete)
- [x] plot_config.py - Centralized styling configuration
- [x] time_series_plots.py - 3 functions (Brier evolution, calibration, distribution)
- [x] grid_search_plots.py - 4 functions (training curves, summary, complexity graph)
- [x] wandb_integration.py - 9 W&B helpers
- [x] generate_all_plots.py - Master orchestration script

### ✅ Phase 1: Statistical Diagnostics (Complete)
- [x] advanced_diagnostics.py: plot_roc_curve() - ROC/AUC with regime stratification
- [x] advanced_diagnostics.py: plot_precision_recall() - PR curves with Average Precision
- [x] advanced_diagnostics.py: plot_qq_residuals() - QQ plots for normality validation
- [x] advanced_diagnostics.py: plot_lift_chart() - Lift & cumulative gains
- [x] advanced_diagnostics.py: plot_win_rate_heatmap() - 3x3 regime performance grid

### ✅ Phase 2: Model Validation (Complete)
- [x] grid_search_plots.py: plot_model_complexity() - Bias-variance tradeoff analysis
- [x] simulation_plots.py: plot_bootstrap_ci() - Monte Carlo bootstrap confidence intervals
- [x] simulation_plots.py: generate_simulation_report() - Full bootstrap analysis

### ✅ Phase 3: Advanced Analysis (Complete)
- [x] shap library installed (v0.49.1 with dependencies: llvmlite, numba, cloudpickle, slicer, tqdm)
- [x] advanced_diagnostics.py: plot_shap_dependence_analysis() - SHAP feature interactions (top 10)
- [x] trading_plots.py: plot_equity_curve() - Simple threshold strategy backtest
- [x] trading_plots.py: generate_trading_report() - Full P&L analysis

### ✅ Final Integration (Complete)
- [x] visualization/__init__.py - Updated with all new exports (23 functions)
- [ ] generate_all_plots.py - Integration with new modules (TODO: add diagnostics/simulation/trading orchestration)
- [x] README.md - Complete documentation updated

---

## Technical Notes

### Memory Constraints
- **Test set:** 12.5M rows, ~2GB in memory
- **SHAP:** MUST downsample to 100K rows (full dataset → OOM)
- **Plotting:** Downsample to 100K-500K for scatter plots (performance)

### Dependencies
- **Installed:** matplotlib, numpy, polars, scikit-learn, scipy, shap (v0.49.1), lightgbm
- **Optional (not installed):** plotly (for interactive dashboards)

### Data Requirements
Test predictions file must have:
- `final_prob` (model predictions)
- `prob_mid` (baseline predictions)
- `outcome` (binary 0/1)
- `residual` (actual - prediction)
- `moneyness` (strike distance)
- `time_remaining` (seconds to expiry)
- `rv_300s` (realized volatility)
- `date` (for time-series)

### Performance Tips
- **Parallel generation:** Run plot modules independently, then combine
- **Downsampling:** For 12.5M rows, downsample to 100K for scatter plots
- **Caching:** Save intermediate aggregations (e.g., daily Brier) to avoid recompute
- **W&B:** Batch upload plots (use `upload_plots_batch()`) for efficiency

---

## Future Enhancements (Phase 3+)

1. **Interactive Dashboard:** Plotly Dash app for real-time exploration
2. **Feature Ablation:** Show Brier improvement when removing each feature
3. **Comparison Plots:** Compare multiple model versions side-by-side
4. **Deployment Monitoring:** Real-time plots updating with live predictions
5. **Risk Decomposition:** P&L attribution by feature contribution

---

## Questions?

**For usage questions:** Check function docstrings in each module
**For bugs:** Check logs for error messages
**For new plots:** Follow patterns in existing modules (time_series_plots.py is best example)

**Example Pattern:**
1. Create function with signature `plot_X(test_df, output_path=None, wandb_log=True)`
2. Apply `apply_plot_style()` at start
3. Use `get_plot_output_path()` for auto-paths
4. Log summary stats with `logger.info()`
5. Upload with `wandb.log()` if enabled
6. Return dict with metrics

---

**Last Updated:** 2025-11-01 (Phase 1 in progress)
