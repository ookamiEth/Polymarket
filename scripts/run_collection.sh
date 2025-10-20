#!/bin/bash
# CONSOLIDATED collection script - streams data directly into 19 category files
# No individual files created - data consolidated as collected
uv run python scripts/batch_fetch_clob_data_consolidated.py \
  --workers 5 \
  --output-dir data/clob_ticks_consolidated \
  --checkpoint-file checkpoints/consolidated_collection.jsonl \
  --resume \
  --verbose
