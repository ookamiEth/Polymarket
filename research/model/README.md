# Binary Option Pricing Calibration Research

**Research Question:** How well do implied volatilities from Deribit BTC options predict outcomes of 15-minute binary contracts?

**Period:** October 2023 - September 2025

**Approach:** Price hypothetical Polymarket-style 15-minute binary options using Deribit options IV, then validate against actual BTC price outcomes.

**Current Version:** V4 - Hierarchical Temporal Model (12 specialized models)

---

## 🚀 Quick Start

### Run Full V4 Pipeline
```bash
# From research/model directory

# 1. Run test suite (~15 min)
bin/run_all_tests_v4.sh

# 2. Generate V4 features (2-3 hours)
cd 00_data_prep
uv run python engineer_all_features_v4.py

# 3. Prepare pipeline data
uv run python prepare_pipeline_data_v4.py

# 4. Train models (24-48 hours)
bin/run_multi_horizon_pipeline_v4.sh

# 5. Run production backtest
cd 01_pricing
uv run python production_backtest_v4.py
```

### Monitor Running Jobs
```bash
# Monitor feature generation
bin/monitor_v4_generation.sh

# Monitor backtest progress
bin/monitor_backtest.sh
```

---

## 📂 Directory Structure

```
research/model/
│
├── README.md                      # This file
├── QUICK_REFERENCE.md             # Common commands reference
│
├── bin/                           # All shell scripts (centralized)
│   ├── run_all_tests_v4.sh        # Run comprehensive test suite
│   ├── run_multi_horizon_pipeline_v4.sh  # Full V4 training pipeline
│   ├── run_tests.sh               # Pricing module unit tests
│   ├── monitor_v4_generation.sh   # Monitor feature generation
│   └── monitor_backtest.sh        # Monitor backtest progress
│
├── docs/                          # All documentation
│   ├── IMPLEMENTATION_PLAN_V4.md  # V4 technical specification
│   ├── QUICK_START_V4.md          # Getting started guide
│   ├── STATUS.md                  # Current project status
│   ├── NEXT_STEPS.md              # Action items
│   ├── TESTING_V4_README.md       # Testing guide
│   ├── DIRECTORY_MAP.md           # Legacy directory map
│   ├── V4_COMPLETION_SUMMARY.md   # V4 completion report
│   └── V4_IMPLEMENTATION_STATUS.md # V4 implementation status
│
├── 00_data_prep/                  # Feature engineering & data preparation
│   ├── engineer_all_features_v4.py         # Main V4 feature generation
│   ├── prepare_pipeline_data_v4.py         # Pipeline data preparation
│   ├── regime_detection_v4.py              # Regime classification
│   ├── impute_missing_features_v4.py       # Handle missing data
│   ├── test_01_preflight_v4.py             # Pre-flight validation
│   ├── test_02_modules_v4.py               # Module unit tests
│   ├── validate_checkpoint_*.py            # Checkpoint validators
│   └── stratified_v4/                      # Stratified data splits
│
├── 01_pricing/                    # Model training & evaluation
│   ├── train_multi_horizon_v4.py           # Train 12 hybrid models
│   ├── optimize_hybrid_model_v4.py         # Optuna hyperparameter tuning
│   ├── evaluate_multi_horizon_v4.py        # Model evaluation
│   ├── production_backtest_v4.py           # Production backtest
│   ├── temporal_model_split.py             # Time-based model routing
│   ├── models_v4/                          # Trained models
│   ├── models_optuna/                      # Optuna-optimized models
│   ├── evaluations/                        # Evaluation outputs
│   ├── tests/                              # Unit & integration tests
│   ├── config/                             # Configuration files
│   └── visualization/                      # Plotting utilities
│
├── 02_analysis/                   # Analysis & diagnostics
│   ├── README.md                           # Analysis guide
│   ├── FEATURE_COMPARISON_V3_V4.md         # V3 vs V4 feature changes
│   ├── calibration_analysis.py             # Calibration diagnostics
│   ├── feature_importance_analysis_v4.csv  # Feature importance results
│   ├── residual_diagnostics.py             # Residual analysis
│   ├── pricing/                            # Pricing-specific analysis
│   │   ├── analyze_calibration_over_time.py
│   │   ├── compute_feature_importance_v4.py
│   │   ├── compute_shap_values_v4.py
│   │   ├── feature_importance/             # Feature importance outputs
│   │   └── shap_values_v4/                 # SHAP analysis outputs
│   └── [other analysis scripts]
│
├── 03_deep_eda/                   # Deep exploratory analysis
│   ├── 01_payoff_nonlinearity.py
│   ├── 02_microstructure.py
│   ├── 03_error_decomposition.py
│   ├── 04_outcome_distributions.py
│   ├── 05_feature_attribution.py
│   ├── technical_paper.pdf                 # Research paper
│   └── technical_paper.tex
│
├── 04_visualization/              # Professional plotting
│   └── professional_plots.py
│
├── data/                          # All data files
│   ├── features_v4/                        # V4 engineered features
│   │   └── intermediate/                   # Intermediate processing
│   └── consolidated_features_v4_pipeline_ready.parquet  # Ready for training
│
├── logs/                          # Unified logs (all scripts)
│   ├── v4/                                 # V4 pipeline logs
│   ├── 01_pricing/                         # Pricing module logs
│   │   ├── v3_archive/                     # Archived V3 logs
│   │   └── wandb_archive/                  # WandB experiment logs (historical)
│   ├── wandb_recent/                       # Recent WandB runs
│   └── production_backtest_v4.log
│
├── wandb/                         # Symlink to logs/wandb_recent/ (for compatibility)
│
├── results/                       # Unified results (all outputs)
│   ├── plots/                              # All visualizations
│   │   └── v4/                             # V4 plots
│   ├── models_v4/                          # Saved models
│   ├── multi_horizon/                      # Multi-horizon predictions
│   ├── multi_horizon_hybrid_v4/            # V4 hybrid outputs
│   ├── 01_pricing_legacy/                  # Legacy pricing results
│   └── [evaluation parquet files]
│
├── archive/                       # Historical code & data (197GB)
│   ├── README.md                           # Archive navigation
│   ├── v3_code/                            # V3 code (archived)
│   │   ├── README.md                       # V3 to V4 transition notes
│   │   ├── v3_legacy/                      # Complete V3 pipeline
│   │   └── shell_scripts/                  # Archived V3 scripts
│   ├── v3_data/                            # V3 features (173GB)
│   ├── v3_backup/                          # V3 backup (24GB)
│   └── docs_historical/                    # Historical documentation
│
├── config/                        # Global configuration
├── checkpoints/                   # Training checkpoints
├── test_logs/                     # Test execution logs
├── wandb/                         # WandB experiment tracking
└── xgb_cache/                     # XGBoost cache

```

---

## 🧠 V4 Architecture Overview

### Hierarchical Temporal Model (12 Models)

**Design Philosophy**: Time-to-expiry dominates option pricing, regimes add context

**Structure:**
```
3 Time Buckets (primary hierarchy):
├── Near (≤30 min)
│   ├── Low volatility, ATM
│   ├── Low volatility, OTM
│   ├── High volatility, ATM
│   └── High volatility, OTM
│
├── Medium (30-60 min)
│   ├── [same 4 regime models]
│   └── ...
│
└── Far (>60 min)
    ├── [same 4 regime models]
    └── ...
```

**Key Improvements vs V3:**
- Time-first hierarchy (better stability)
- 152 features (pruned from 237)
- Regime-aware within time buckets
- Walk-forward validation (10 windows)
- Optuna hyperparameter tuning

---

## 📊 Data Sources

### Primary Data (Tardis.dev)
- **Deribit Options**: BTC-PERPETUAL options data (Oct 2023 - Sep 2025)
  - Trade-by-trade executions
  - Orderbook snapshots (1-second resolution)
  - Greeks (delta, gamma, vega, theta)
  - Implied volatility surface

- **BTC Spot Price**: Coinbase BTC-USD 1-second bars
  - OHLCV data
  - Trade volume
  - Realized volatility

### Derived Features (152 total)
- **Options Microstructure** (26): Bid-ask spreads, quote depths, trade flow
- **Volatility Surface** (24): ATM IV, skew, term structure, curvature
- **Greeks** (18): Delta, gamma, vega weighted by OI and volume
- **Market Regime** (12): Volatility state, trend, momentum
- **Time Features** (8): Time to expiry, time of day, day of week
- **Realized Volatility** (18): Multiple horizons (5min to 4h)
- **Price Action** (22): Returns, momentum, micro movements
- **Volume** (12): Spot and options volume patterns
- **Risk-Free Rate** (2): Blended DeFi lending rates
- **Polynomial Features** (10): Key interaction terms

---

## 🔬 Methodology

### 1. Feature Engineering (`00_data_prep/`)
```bash
# Generate V4 features
cd 00_data_prep
uv run python engineer_all_features_v4.py

# Validate output
uv run python validate_checkpoint_51.py
```

**Process:**
- Load Deribit options data (1.1B rows → 10M ATM options)
- Calculate IV surface features
- Engineer microstructure features
- Detect market regimes
- Compute realized volatility
- Impute missing values
- Output: `data/consolidated_features_v4_pipeline_ready.parquet`

### 2. Model Training (`01_pricing/`)
```bash
# Train 12 hybrid models
cd 01_pricing
uv run python train_multi_horizon_v4.py --model all

# Or use full pipeline
bin/run_multi_horizon_pipeline_v4.sh
```

**Training Strategy:**
- **Walk-Forward Validation**: 10 windows (9-month train, 3-month validation)
- **LightGBM**: Gradient boosted trees (28 threads on 32-vCPU machine)
- **Optuna**: 100 trials for priority models, 50 for others
- **Metrics**: MSE (primary), Brier score, calibration error

### 3. Evaluation & Analysis (`02_analysis/`)
```bash
# Evaluate models
cd 01_pricing
uv run python evaluate_multi_horizon_v4.py

# Analyze calibration
cd ../02_analysis
uv run python calibration_analysis.py

# Feature importance
cd pricing
uv run python compute_feature_importance_v4.py
```

### 4. Production Backtest (`01_pricing/`)
```bash
cd 01_pricing
uv run python production_backtest_v4.py
```

**Backtest Metrics:**
- Calibration plots (predicted vs actual)
- Brier score decomposition
- Time-series performance
- Regime-specific accuracy

---

## 🧪 Testing

### Pre-Flight Checks
```bash
# Comprehensive test suite (~15 min)
bin/run_all_tests_v4.sh

# Skip unit tests (faster)
bin/run_all_tests_v4.sh --skip-unit

# Run specific checkpoint
bin/run_all_tests_v4.sh --checkpoint 51
```

**Test Phases:**
1. **Pre-Flight Validation** (~5 min): Input files, schemas, disk/memory
2. **Module Unit Tests** (~10 min): Feature engineering, regime detection
3. **Checkpoint Validators** (on-demand): Data quality at each pipeline stage

### Unit Tests (Pricing Module)
```bash
# All tests
bin/run_tests.sh

# Unit tests only
bin/run_tests.sh --unit

# With coverage
bin/run_tests.sh --coverage
```

---

## 📈 Performance Standards

### Model Quality Targets (V4)
| Metric | Target | V3 Baseline |
|--------|--------|-------------|
| **Brier Score** | <0.15 | 0.18 |
| **Calibration Error** | <0.03 | 0.05 |
| **R² (predicted vs actual)** | >0.45 | 0.38 |

### Computational Performance (32 vCPU, 256GB RAM)
| Task | Runtime | Parallelism |
|------|---------|-------------|
| **Feature Engineering** | 2-3 hours | 20 workers |
| **Baseline Training (12 models)** | 24 hours | 4 parallel trials |
| **Optuna Optimization** | 60 hours | 28 threads/trial |
| **Production Backtest** | 4-6 hours | Single-threaded |

---

## 🛠️ Common Tasks

### Feature Engineering
```bash
# Generate V4 features
cd 00_data_prep
uv run python engineer_all_features_v4.py

# Monitor progress
bin/monitor_v4_generation.sh

# Validate output
uv run python validate_checkpoint_51.py
```

### Training Models
```bash
# Train single model
cd 01_pricing
uv run python train_multi_horizon_v4.py --model near_low_vol_atm

# Train all models
uv run python train_multi_horizon_v4.py --model all

# Full pipeline with Optuna
bin/run_multi_horizon_pipeline_v4.sh
```

### Analysis
```bash
# Calibration analysis
cd 02_analysis
uv run python calibration_analysis.py

# Feature importance
cd pricing
uv run python compute_feature_importance_v4.py

# SHAP values
uv run python compute_shap_values_v4.py
```

### Backtesting
```bash
cd 01_pricing
uv run python production_backtest_v4.py

# Monitor progress
bin/monitor_backtest.sh
```

---

## 📚 Documentation Index

### Core Documentation
- **README.md** (this file) - Project overview and directory structure
- **QUICK_REFERENCE.md** - Common commands cheat sheet

### Detailed Guides (`docs/`)
- **IMPLEMENTATION_PLAN_V4.md** - Complete V4 technical specification
- **QUICK_START_V4.md** - Step-by-step getting started guide
- **TESTING_V4_README.md** - Comprehensive testing guide
- **STATUS.md** - Current project status and progress
- **NEXT_STEPS.md** - Upcoming action items
- **V4_COMPLETION_SUMMARY.md** - V4 implementation completion report

### Component Documentation
- **00_cpu_optimization/README.md** - 32 vCPU parallelism optimization
- **02_analysis/README.md** - Analysis module guide
- **02_analysis/FEATURE_COMPARISON_V3_V4.md** - V3 to V4 feature changes
- **01_pricing/docs/** - Pricing module detailed documentation

### Research Papers
- **03_deep_eda/technical_paper.pdf** - Full research paper and methodology

---

## 🔄 Pipeline Workflows

### Full V4 Pipeline (End-to-End)
```bash
# 1. Test infrastructure
bin/run_all_tests_v4.sh

# 2. Generate features (2-3 hours)
cd 00_data_prep
nohup uv run python engineer_all_features_v4.py > ../logs/v4_feature_generation.log 2>&1 &

# Monitor: bin/monitor_v4_generation.sh

# 3. Validate features
uv run python validate_checkpoint_51.py

# 4. Prepare pipeline data
uv run python prepare_pipeline_data_v4.py
uv run python validate_checkpoint_52.py

# 5. Train models (24-48 hours)
cd ../01_pricing
nohup bin/run_multi_horizon_pipeline_v4.sh > ../logs/pipeline_v4.log 2>&1 &

# 6. Validate training
uv run python validate_checkpoint_53.py

# 7. Production backtest
nohup uv run python production_backtest_v4.py > ../logs/production_backtest_v4.log 2>&1 &

# Monitor: bin/monitor_backtest.sh
```

### Quick Development Iteration
```bash
# Modify features
cd 00_data_prep
./check_code.sh engineer_all_features_v4.py --fix
uv run python engineer_all_features_v4.py

# Train single model for testing
cd ../01_pricing
uv run python train_multi_horizon_v4.py --model near_low_vol_atm

# Quick evaluation
uv run python evaluate_multi_horizon_v4.py
```

---

## 🐛 Troubleshooting

### Memory Issues
- V4 features use ~50GB RAM during generation
- Training uses ~100GB RAM with 28 threads
- If OOM: reduce `num_threads` in config or use smaller batches

### Missing Data
- Run `cd 00_data_prep && uv run python check_data_availability.py`
- Ensure Tardis data downloaded: `research/tardis/data/processed/`

### Failed Tests
- Check logs in `test_logs/`
- Run specific test: `cd 00_data_prep && uv run python test_01_preflight_v4.py`
- Validate checkpoints after each pipeline stage

### Slow Training
- Check CPU utilization: `htop`
- Verify parallel settings in config: `config/multi_horizon_regime_config.yaml`
- See `00_cpu_optimization/README.md` for tuning guide

---

## 📧 Contact & Support

For questions or issues, see:
- **Documentation**: All guides in `docs/`
- **Code Quality**: `./check_code.sh path/to/file.py --fix` (Ruff + Pyright)
- **Architecture**: `docs/IMPLEMENTATION_PLAN_V4.md`

---

## 📝 Version History

### V4 (Current) - Hierarchical Temporal Model
- 12 specialized models (3 time buckets × 4 regime models)
- 152 pruned features
- Walk-forward validation
- Optuna hyperparameter tuning
- **Status**: Production-ready

### V3 (Archived) - Regime-First Model
- 4 regime-based models
- 237 features (over-engineered)
- Single validation split
- Manual hyperparameter tuning
- **Status**: Archived in `archive/v3_code/`

---

## 🎯 Current Status

**Version**: V4 (Production)
**Last Updated**: November 2025
**Models Trained**: 12 hybrid temporal models
**Features**: 152 (pruned from 237)
**Data Period**: October 2023 - September 2025

See `docs/STATUS.md` for detailed current progress and `docs/NEXT_STEPS.md` for upcoming work.
