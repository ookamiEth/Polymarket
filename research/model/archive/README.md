# Archive Directory

Historical content from V3 development and earlier sessions.

## Structure

### v3_data/ (173GB)
V3 feature files archived on 2025-11-12
- `consolidated_features_v3.parquet` (86GB)
- `features_v3/` directory (87GB)

**Can be restored if needed for comparison with V4 results.**

### v3_backup/ (24GB)
Contaminated backup from November 3, 2025
- `CONTAMINATED_BACKUP_20251103/`

**Kept for reference only. Do not use for active work.**

### session_logs/
Development session logs from October-November 2025
- `session_2025-10-29_04-39-UTC.md`
- `session_2025-10-30_xgboost_external_memory.md`
- `session_2025-11-01_lightgbm_optimization.md`

**Historical development notes and debugging sessions.**

### docs_historical/
V3-era documentation (October-November 2025)
- Analysis documents
- Strategy documents
- Technical guides
- Downloaded reference material

**Superseded by V4 documentation in root directory.**

---

## Total Archive Size

Approximately **197GB** of historical data preserved for reference.

## Restoration

If you need to restore any archived content:

```bash
# Example: Restore V3 features for comparison
cp archive/v3_data/consolidated_features_v3.parquet data/

# Example: Reference old documentation
less archive/docs_historical/IMPROVEMENT_STRATEGY.md
```

---

**Archived**: 2025-11-12
**Active V4 Development**: See parent directory