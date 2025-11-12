# V3 Legacy Scripts

**Archived**: 2025-11-12
**Status**: Historical reference only

---

## Overview

This directory contains V3 scripts that were used before the V4 hierarchical architecture. These scripts are **archived for reference** and should not be used for active development.

**V4 Active Scripts**: See parent directory (`01_pricing/`)

---

## Directory Structure

```
v3_legacy/
├── core/              # Training and prediction
├── optimization/      # Hyperparameter tuning
├── analysis/          # Model analysis and debugging
├── utilities/         # Support utilities
└── scripts/           # Shell scripts
```

---

## Core Training (core/)

| Script | Purpose | Last Used |
|--------|---------|-----------|
| `lightgbm_memory_optimized.py` | Memory-efficient LightGBM training | Nov 2025 |
| `train_multi_horizon.py` | Multi-horizon model training | Nov 2025 |
| `evaluate_multi_horizon.py` | Model evaluation | Nov 2025 |
| `predict_multi_horizon.py` | Generate predictions | Nov 2025 |
| `walk_forward_validation.py` | Walk-forward cross-validation | Nov 2025 |

---

## Optimization (optimization/)

| Script | Purpose | Last Used |
|--------|---------|-----------|
| `optuna_multi_horizon.py` | Optuna hyperparameter search | Nov 2025 |
| `lightgbm_grid_search.py` | Grid search optimization | Nov 2025 |
| `lightgbm_optuna_search.py` | LightGBM-specific Optuna search | Nov 2025 |

---

## Analysis (analysis/)

| Script | Purpose | Last Used |
|--------|---------|-----------|
| `debug_nan_predictions.py` | Debug NaN predictions | Nov 2025 |
| `feature_ablation_study.py` | Feature importance analysis | Nov 2025 |
| `evaluate_by_regime.py` | Regime-specific evaluation | Nov 2025 |
| `model_analysis.py` | Model performance analysis | Nov 2025 |
| `feature_correlation_analysis.py` | Feature correlation analysis | Nov 2025 |

---

## Utilities (utilities/)

| Script | Purpose | Last Used |
|--------|---------|-----------|
| `fix_test_predictions.py` | Fix prediction issues | Nov 2025 |
| `generate_html_only.py` | Generate HTML reports | Nov 2025 |
| `generate_test_predictions.py` | Generate test predictions | Nov 2025 |
| `validate_v3_features.py` | Validate V3 features | Nov 2025 |

---

## Why V3 Scripts Were Archived

### Key V3 Limitations

1. **Single Model Architecture**: One model for all time windows and market conditions
2. **196 Features**: 87% of features had <1% importance (171/196)
3. **Moneyness Dominance**: Single feature (moneyness) had 64% importance
4. **No Temporal Specialization**: Same model for near (5 min) and far (15 min) horizons
5. **No Regime Adaptation**: Same hyperparameters for all market conditions

### V4 Improvements

| Aspect | V3 | V4 |
|--------|----|----|
| Architecture | 1 model | 12 models (3 temporal × 4 volatility) |
| Features | 196 | ~55 (aggressive pruning) |
| Moneyness Importance | 64% | <25% (8 transformations) |
| Brier Score | 0.134 | ≤0.126 (target) |
| Temporal Specialization | None | 3 windows (near/mid/far) |
| Regime Adaptation | None | 4 volatility regimes |

---

## Restoration (If Needed)

If you need to reference or restore V3 functionality:

### View V3 Script
```bash
less v3_legacy/core/train_multi_horizon.py
```

### Copy to Active Directory
```bash
cp v3_legacy/analysis/feature_ablation_study.py ../feature_ablation_study_v3.py
```

### Compare V3 vs V4
```bash
diff v3_legacy/core/train_multi_horizon.py ../train_temporal_models.py
```

---

## V3 Results Reference

V3 results are preserved in:
- `results/multi_horizon/` (103GB) - Multi-horizon model outputs
- `logs/v3_archive/` - Historical execution logs
- `../archive/` - XGBoost runs and shell scripts

---

## Migration Notes

**See**: `/home/ubuntu/Polymarket/research/model/01_pricing/docs/misc/V3_MIGRATION_SUMMARY.md` for detailed migration notes from V3 to V4.

**Key Changes**:
- Temporal split logic moved to `temporal_model_split.py`
- Regime detection enhanced in `/home/ubuntu/Polymarket/research/model/00_data_prep/regime_detection_v4.py`
- 12-model training in `train_temporal_models.py`
- Feature engineering redesigned in `/home/ubuntu/Polymarket/research/model/00_data_prep/engineer_all_features_v4.py`

---

## Questions?

For V4 implementation questions, see:
- `/home/ubuntu/Polymarket/research/model/IMPLEMENTATION_PLAN_V4.md`
- `/home/ubuntu/Polymarket/research/model/NEXT_STEPS.md`
- `/home/ubuntu/Polymarket/research/model/01_pricing/README.md`

---

**Archived**: 2025-11-12
**Active Development**: V4 scripts in parent directory
