# Multi-Horizon ML Pipeline Test Suite

Comprehensive test suite for the multi-horizon LightGBM training pipeline with stratified test fixtures (<500K rows).

## Overview

This test suite validates all 4 phases of the multi-horizon pipeline:
1. **Phase 1 Training** - Walk-forward validation with 3 bucket models
2. **Phase 1 Evaluation** - ≥15% improvement target validation
3. **Phase 2 Optimization** - Per-bucket hyperparameter tuning
4. **Phase 2 Re-evaluation** - Final performance metrics

## Quick Start

```bash
# Generate test fixtures (run once)
cd tests/fixtures
uv run python generate_test_data.py

# Run all tests
cd ../..
./run_tests.sh

# Run with coverage
./run_tests.sh --coverage

# Run only unit tests
./run_tests.sh --unit
```

## Test Suite Structure

```
tests/
├── unit/                       # Unit tests for individual functions
│   ├── test_metrics.py         # ✅ Metrics calculation tests (18 tests)
│   ├── test_stratification.py  # Bucket stratification tests
│   ├── test_walk_forward.py    # Walk-forward validation tests
│   └── test_smoothing.py       # Boundary smoothing tests
├── integration/                # End-to-end pipeline tests
│   ├── test_pipeline_pilot.py  # Pilot mode integration test
│   ├── test_phase1_training.py # Phase 1 training tests
│   └── test_evaluation.py      # Evaluation tests
├── fixtures/                   # Test data (<500K rows)
│   ├── test_train.parquet      # 60K rows (stratified)
│   ├── test_val.parquet        # 20K rows (stratified)
│   ├── test_test.parquet       # 20K rows (stratified)
│   └── generate_test_data.py   # Test data generator
├── conftest.py                 # ✅ Pytest configuration + fixtures
└── README.md                   # This file
```

## Test Fixtures

### Generation

Test fixtures are created by stratified sampling from the full production dataset:

```bash
cd tests/fixtures
uv run python generate_test_data.py
```

**Output:**
- 100,000 total rows (33% near, 33% mid, 34% far)
- 60/20/20 temporal train/val/test split
- 23 months temporal coverage (Oct 2023 - Sep 2025)
- 41.59 MB total size (well under <500K row requirement)

**Stratification:**
- **Near bucket** (0-300s): 33,000 samples
- **Mid bucket** (300-600s): 33,000 samples
- **Far bucket** (600-900s): 34,000 samples

### Data Quality

✅ All 59 production features present
✅ Temporal coverage sufficient for walk-forward testing
✅ Bucket distribution matches production (33%/33%/34%)
✅ Required columns: `outcome`, `prob_mid`, `time_remaining`, `date`

## Running Tests

### All Tests

```bash
./run_tests.sh
```

### Unit Tests Only

```bash
./run_tests.sh --unit
```

### Integration Tests Only

```bash
./run_tests.sh --integration
```

### With Coverage Report

```bash
./run_tests.sh --coverage
# View coverage: open htmlcov/index.html
```

### Fast Mode (Skip Slow Tests)

```bash
./run_tests.sh --fast
```

## Test Categories

### Unit Tests

#### `test_metrics.py` (✅ 18 tests, all passing)

Tests core metrics calculation functions:

**Brier Score Tests:**
- ✅ Baseline Brier from prob_mid
- ✅ Model Brier from residual predictions
- ✅ Positive improvement when model beats baseline
- ✅ Negative improvement when model worse than baseline

**Residual Metrics Tests:**
- ✅ MSE/RMSE/MAE calculation correctness
- ✅ RMSE = sqrt(MSE) validation
- ✅ Perfect predictions give zero error

**Probability Clipping Tests:**
- ✅ Probabilities < 0 clipped to 0
- ✅ Probabilities > 1 clipped to 1

**Edge Cases:**
- ✅ Perfect baseline (Brier = 0)
- ✅ Random 50/50 baseline
- ✅ Known improvement percentage (75% test)

**Calibration Tests:**
- ✅ Perfect calibration gives low error
- ✅ Miscalibrated predictions give high error

**Metrics Dictionary:**
- ✅ All required fields present
- ✅ Sample count correct
- ✅ Label preserved

#### `test_stratification.py` (TODO)

Tests bucket stratification logic:
- Correct filtering by time_remaining ranges
- No data leakage across buckets
- Edge cases (boundary values)
- Row counts match expected distribution

#### `test_walk_forward.py` (TODO)

Tests walk-forward validation:
- Correct number of windows generated
- No train/val overlap
- Chronological ordering enforced
- Windows stop before holdout period

#### `test_smoothing.py` (TODO)

Tests boundary smoothing interpolation:
- Hard routing outside smoothing zones
- Linear interpolation within zones (270-330s, 570-630s)
- Weights sum to 1.0
- No discontinuities at boundaries

### Integration Tests

#### `test_pipeline_pilot.py` (TODO)

End-to-end pilot mode test:
- Train 3 buckets without walk-forward
- Quick evaluation without plots
- Verify all outputs created
- Check metrics validity

#### `test_phase1_training.py` (TODO)

Phase 1 training tests:
- Stratification creates correct bucket files
- Walk-forward generates expected windows
- Models trained successfully
- Config files contain all metrics

#### `test_evaluation.py` (TODO)

Evaluation tests:
- Load 3 bucket models
- Generate predictions on test set
- Per-bucket performance breakdown
- Multi-horizon vs single model comparison

## Pytest Configuration

### Fixtures

**Directory Paths:**
- `project_root` - Project root directory
- `fixtures_dir` - Test fixtures directory
- `test_output_dir` - Temporary test outputs

**Test Data:**
- `test_train_file` / `test_train_df` - 60K training data
- `test_val_file` / `test_val_df` - 20K validation data
- `test_test_file` / `test_test_df` - 20K test data
- `synthetic_predictions_df` - Synthetic data for quick tests

**Configs:**
- `config_dir` - Config directory
- `multi_horizon_config_file` - multi_horizon_config.yaml path

### Helper Assertions

**`assert_valid_parquet(file_path, min_rows=1)`**
- Validates Parquet file exists and is readable
- Checks row count meets minimum

**`assert_metrics_valid(metrics, check_improvement=True)`**
- Validates metrics dictionary structure
- Checks all required fields present
- Validates value ranges (Brier in [0,1], positive improvement)

**`assert_no_data_leakage(train_dates, val_dates)`**
- Ensures val dates > train dates (chronological)
- Catches temporal data leakage

**`assert_bucket_distribution(df, expected_near_pct=33.0, ...)`**
- Validates bucket distribution within tolerance
- Checks near/mid/far percentages

### Markers

```python
@pytest.mark.unit           # Unit test
@pytest.mark.integration    # Integration test
@pytest.mark.slow           # Slow test (skip with --fast)
```

## Success Criteria

### Current Status

✅ Test infrastructure complete
✅ Test fixtures generated (100K rows)
✅ Pytest configuration + helpers
✅ Metrics calculation tests (18/18 passing)
⏳ Remaining unit tests (TODO)
⏳ Integration tests (TODO)

### Target Coverage

- **Unit tests:** >80% coverage for core functions
- **Integration tests:** All 4 pipeline phases validated
- **Test runtime:** <10 minutes for full suite
- **Test data:** <500K rows (✅ 100K rows)

## Test Data Specification

### Required Columns

| Column | Type | Description |
|--------|------|-------------|
| `outcome` | Float64 | Binary outcome (0 or 1) |
| `prob_mid` | Float64 | Baseline probability [0, 1] |
| `time_remaining` | Float64 | Seconds to expiry [0, 900] |
| `date` | Date | Date for temporal splits |
| 59 features | Float64 | Model input features |

### Bucket Distribution

| Bucket | Time Range | Samples | Percentage |
|--------|------------|---------|------------|
| Near | 0-300s (0-5min) | 33,000 | 33% |
| Mid | 300-600s (5-10min) | 33,000 | 33% |
| Far | 600-900s (10-15min) | 34,000 | 34% |

### Temporal Coverage

- **Full range:** Oct 2023 - Sep 2025 (23 months)
- **Train:** Oct 2023 - Dec 2024 (60K rows)
- **Val:** Dec 2024 - May 2025 (20K rows)
- **Test:** May 2025 - Sep 2025 (20K rows)

**Walk-forward windows:** Expected 3-4 windows (reduced from 14 in production due to smaller test dataset)

## Continuous Integration

### Pre-Pipeline Validation

Add test validation to production pipeline:

```bash
# In run_multi_horizon_pipeline.sh
if [[ "$RUN_TESTS" == "true" ]]; then
    ./run_tests.sh --fast || exit 1
fi
```

### Test Before Deploy

```bash
# Validate before starting production training
./run_tests.sh --coverage

# If tests pass, run production pipeline
./run_multi_horizon_pipeline.sh full
```

## Troubleshooting

### Test Fixtures Missing

```bash
cd tests/fixtures
uv run python generate_test_data.py
```

### Import Errors

Ensure you're running from the project root:

```bash
cd /home/ubuntu/Polymarket/research/model/01_pricing
./run_tests.sh
```

### Pytest Not Found

Install testing dependencies:

```bash
uv pip install pytest pytest-cov pytest-mock
```

### Tests Failing

Run with verbose output:

```bash
uv run pytest tests/ -vv -s
```

## Next Steps

1. **Complete unit tests:**
   - `test_stratification.py`
   - `test_walk_forward.py`
   - `test_smoothing.py`

2. **Add integration tests:**
   - `test_pipeline_pilot.py` (critical)
   - `test_phase1_training.py`
   - `test_evaluation.py`

3. **Add regression tests:**
   - Performance tracking
   - Model size monitoring
   - Data leakage detection

4. **CI/CD integration:**
   - Add `--validate` flag to pipeline script
   - Pre-commit hooks for test validation

## References

- **Pipeline Documentation:** `MULTI_HORIZON_IMPLEMENTATION.md`
- **Walk-Forward Validation:** `WALK_FORWARD_VALIDATION.md`
- **Pipeline Usage:** `PIPELINE_USAGE.md`
- **Main Config:** `config/multi_horizon_config.yaml`

## Contact

For issues or questions about the test suite, refer to the main project documentation or check existing test implementations for examples.
