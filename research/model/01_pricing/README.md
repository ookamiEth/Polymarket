# 01_pricing - Options Pricing Models

**Last Updated**: 2025-11-12
**Version**: V4 Architecture

---

## 📁 Directory Structure

```
01_pricing/
├── 📄 V4 Active Scripts (Root)
├── 📂 v3_legacy/          # V3 historical scripts (archived)
├── 📂 docs/               # Documentation organized by topic
├── 📂 logs/               # Execution logs (v4/ and v3_archive/)
├── 📂 config/             # Model configurations
├── 📂 results/            # Model outputs and plots
├── 📂 tests/              # Test suite
├── 📂 visualization/      # Plotting utilities
├── 📂 wandb/              # Weights & Biases run history
└── 📂 archive/            # Historical shell scripts and xgboost runs
```

---

## 🚀 V4 Active Scripts (Root Level)

These are the **current production scripts** for V4 hierarchical modeling:

### Core V4 Pipeline

| Script | Purpose | Status |
|--------|---------|--------|
| `production_backtest_v4.py` | Generate V4 baseline predictions | ✅ Ready |
| `temporal_model_split.py` | 3-way temporal split logic (near/mid/far) | ✅ Ready |
| `train_temporal_models.py` | Train 12 hierarchical models | ✅ Ready |

### Execution Scripts

| Script | Purpose |
|--------|---------|
| `run_multi_horizon_pipeline_v4.sh` | V4 pipeline orchestration |
| `monitor_backtest.sh` | Monitor backtest progress |
| `run_tests.sh` | Run test suite |

---

## 📚 Documentation (docs/)

Documentation is organized by topic for easy navigation:

### Architecture (docs/architecture/)
- `MULTI_HORIZON_IMPLEMENTATION.md` - Multi-horizon architecture
- `DATA_SPLIT_STRATEGY.md` - Temporal and cross-validation strategy

### Optimization (docs/optimization/)
- `GRID_SEARCH_README.md` - Grid search methodology
- `OPTUNA_OPTIMIZATION_EXPLAINED.md` - Bayesian optimization
- `OPTIMIZATION_ROADMAP.md` - Hyperparameter tuning roadmap

### Methodology (docs/methodology/)
- `DATA_LEAKAGE_FIXES.md` - Preventing look-ahead bias
- `WALK_FORWARD_VALIDATION.md` - Walk-forward CV implementation
- `IMPLEMENTATION_SUMMARY.md` - Overall methodology summary

### Usage (docs/usage/)
- `PIPELINE_USAGE.md` - How to run the pipeline
- `CPU_OPTIMIZATION_QUICK_REFERENCE.md` - CPU optimization guide
- `FUNCTIONALITY_COMPARISON.md` - Feature comparison

### Visualization (docs/visualization/)
- `PLOT_GUIDE.md` - Plotting standards and examples
- `VISUALIZATION_PLAN.md` - Visualization strategy
- `VISUALIZATION_QUICKSTART.md` - Quick plotting guide

### Miscellaneous (docs/misc/)
- `CPU_OPTIMIZATION_32vCPU.md` - 32 vCPU optimization details
- `V3_MIGRATION_SUMMARY.md` - V3 to V4 migration notes
- `WANDB_COMPARISON.md` - Weights & Biases tracking

---

## 🗄️ V3 Legacy (v3_legacy/)

Historical V3 scripts organized by function. **Archived November 12, 2025**.

### Core Training (v3_legacy/core/)
- `lightgbm_memory_optimized.py` - Memory-efficient LightGBM training
- `train_multi_horizon.py` - Multi-horizon training
- `evaluate_multi_horizon.py` - Model evaluation
- `predict_multi_horizon.py` - Prediction generation
- `walk_forward_validation.py` - Walk-forward CV

### Optimization (v3_legacy/optimization/)
- `optuna_multi_horizon.py` - Optuna hyperparameter search
- `lightgbm_grid_search.py` - Grid search optimization
- `lightgbm_optuna_search.py` - LightGBM-specific Optuna search

### Analysis (v3_legacy/analysis/)
- `debug_nan_predictions.py` - NaN debugging
- `feature_ablation_study.py` - Feature importance analysis
- `evaluate_by_regime.py` - Regime-specific evaluation
- `model_analysis.py` - Model performance analysis
- `feature_correlation_analysis.py` - Feature correlation

### Utilities (v3_legacy/utilities/)
- `fix_test_predictions.py` - Prediction post-processing
- `generate_html_only.py` - HTML report generation
- `generate_test_predictions.py` - Test set predictions
- `validate_v3_features.py` - V3 feature validation

### Scripts (v3_legacy/scripts/)
- Shell scripts for V3 pipeline automation

**See `v3_legacy/README.md` for restoration instructions.**

---

## 📝 Logs (logs/)

Organized execution logs:

- **logs/v4/** - V4 execution logs (active)
  - `production_backtest_v4.log` - Latest V4 backtest log

- **logs/v3_archive/** - V3 historical logs (archived)
  - Pipeline logs, session logs, training logs

**See `logs/README.md` for log file conventions.**

---

## 🎯 V4 Architecture

### 12 Hierarchical Models

**3 Temporal Windows × 4 Volatility Regimes = 12 Models**

#### Temporal Split
- **near**: time_remaining < 300s
- **mid**: 300s ≤ time_remaining ≤ 900s
- **far**: time_remaining > 900s

#### Volatility Regimes (per temporal window)
- **low_vol_atm**: Low volatility, at-the-money
- **low_vol_otm**: Low volatility, out-of-the-money
- **high_vol_atm**: High volatility, at-the-money
- **high_vol_otm**: High volatility, out-of-the-money

**Example Model**: `near_low_vol_atm.txt`

### Key V4 Improvements

| Metric | V3 | V4 Target |
|--------|----|-----------|
| Brier Score | 0.134 | ≤0.126 |
| Features | 196 | ~55 |
| Moneyness Importance | 64% (near) | <25% |
| Models | 1 | 12 |

---

## 🚦 Quick Start

### Run V4 Pipeline

**Prerequisites**: V4 baseline file must exist (see `/home/ubuntu/Polymarket/research/model/NEXT_STEPS.md`)

```bash
# 1. Generate V4 features (in 00_data_prep/)
cd /home/ubuntu/Polymarket/research/model/00_data_prep
nohup uv run python engineer_all_features_v4.py > v4_feature_generation.log 2>&1 &

# 2. Train 12 V4 models (in 01_pricing/)
cd /home/ubuntu/Polymarket/research/model/01_pricing
nohup uv run python train_temporal_models.py > train_temporal_models.log 2>&1 &

# 3. Monitor progress
tail -f train_temporal_models.log
```

---

## 📊 Outputs

### Model Files
- **results/models_v4/** - 12 trained LightGBM models
  - `near_low_vol_atm.txt`
  - `near_low_vol_otm.txt`
  - `near_high_vol_atm.txt`
  - `near_high_vol_otm.txt`
  - (+ 8 more for mid and far temporal windows)

### Performance Reports
- **results/models_v4/performance_summary.txt** - Overall performance metrics
- **results/plots/v4/** - Visualizations (calibration, feature importance, etc.)

---

## 🧪 Testing

```bash
# Run all tests
./run_tests.sh

# Run specific test suite
uv run pytest tests/unit/
uv run pytest tests/integration/
uv run pytest tests/regression/
```

---

## 📞 Need Help?

1. Check `/home/ubuntu/Polymarket/research/model/QUICK_REFERENCE.md` for one-page command reference
2. Check `/home/ubuntu/Polymarket/research/model/NEXT_STEPS.md` for step-by-step V4 execution guide
3. Check `/home/ubuntu/Polymarket/research/model/STATUS.md` for current progress
4. Check this directory's `docs/usage/PIPELINE_USAGE.md` for detailed usage instructions

---

**Maintained By**: V4 Development Team
**Last Reorganization**: 2025-11-12 (V3 scripts archived, docs organized)
