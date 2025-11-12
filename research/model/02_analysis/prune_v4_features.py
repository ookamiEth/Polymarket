#!/usr/bin/env python3
"""
Script to systematically remove SMA and 1800s features from V4 implementation
"""

import re

# Read the file
file_path = "/home/ubuntu/Polymarket/research/model/00_data_prep/engineer_all_features_v4.py"
with open(file_path, "r") as f:
    content = f.read()

# Count initial occurrences
sma_count_before = content.lower().count("sma")
count_1800_before = content.count("1800")
print(f"Before: {sma_count_before} SMA references, {count_1800_before} 1800s references")

# Pattern replacements
replacements = [
    # Remove SMA calculations
    (r'    # Spread SMAs.*?\n(?:.*?\n)*?    \)\n', '    # REMOVED: Spread SMAs (pruned)\n'),
    (r'    # Imbalance SMAs.*?\n(?:.*?\n)*?    \)\n', '    # REMOVED: Imbalance SMAs (pruned)\n'),
    (r'    logger\.info\("Computing .*? SMAs\.\.\."\).*?\n', ''),
    (r'            pl\.col\(".*?"\)\.rolling_mean\(window_size=\d+\)\.alias\(".*?_sma_.*?"\),?\n', ''),

    # Remove 1800s calculations
    (r'            pl\.col\(".*?"\)\.ewm_mean\(span=1800\)\.alias\(".*?_ema_1800s"\),?\n', ''),
    (r'            pl\.col\(".*?"\)\.rolling_mean\(window_size=1800\)\.alias\(".*?_sma_1800s"\),?\n', ''),
    (r'            pl\.col\(".*?"\)\.rolling_std\(window_size=1800\)\.alias\(".*?_vol_1800s"\),?\n', ''),
    (r'            .*?_1800s.*?,?\n', ''),

    # Remove from column selections
    (r'            ".*?_sma_.*?",?\n', ''),
    (r'            ".*?_1800s",?\n', ''),

    # Fix trailing commas
    (r',(\s*\])', r'\1'),
    (r',(\s*\))', r'\1'),
]

# Apply replacements
for pattern, replacement in replacements:
    content = re.sub(pattern, replacement, content, flags=re.MULTILINE)

# Update feature counts in log messages
content = re.sub(r'✓ Wrote 11 funding rate features', '✓ Wrote 5 funding rate features (pruned from 11)', content)
content = re.sub(r'✓ Wrote 32 orderbook L0 features', '✓ Wrote 18 orderbook L0 features (pruned from 32)', content)
content = re.sub(r'✓ Wrote 11 price basis features', '✓ Wrote 5 basis features (pruned from 11)', content)
content = re.sub(r'✓ Wrote 6 open interest features', '✓ Wrote 4 OI features (pruned from 6)', content)
content = re.sub(r'✓ Wrote 79 RV/momentum/range/EMA features', '✓ Wrote ~45 RV/momentum/range features (pruned)', content)

# Count final occurrences
sma_count_after = content.lower().count("sma")
count_1800_after = content.count("1800")
print(f"After: {sma_count_after} SMA references, {count_1800_after} 1800s references")
print(f"Removed: {sma_count_before - sma_count_after} SMA references, {count_1800_before - count_1800_after} 1800s references")

# Write back
with open(file_path, "w") as f:
    f.write(content)

print("✅ Pruning complete!")