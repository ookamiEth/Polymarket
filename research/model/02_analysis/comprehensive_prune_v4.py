#!/usr/bin/env python3
"""
Comprehensive script to remove all SMA and 1800s features from V4
"""

import re

def prune_v4_features():
    file_path = "/home/ubuntu/Polymarket/research/model/00_data_prep/engineer_all_features_v4.py"

    with open(file_path, "r") as f:
        lines = f.readlines()

    new_lines = []
    skip_next = False
    in_sma_block = False

    for i, line in enumerate(lines):
        # Skip lines that are marked for removal
        if skip_next:
            skip_next = False
            continue

        # Skip SMA calculation lines
        if "rolling_mean" in line and ("sma" in line.lower() or "1800" in line):
            continue

        # Skip 1800s EMA lines
        if "span=1800" in line or "window_size=1800" in line:
            continue

        # Skip column references with SMA or 1800s
        if ('"' in line or "'" in line) and ("_sma_" in line or "_1800s" in line):
            # Check if this is in a list and handle comma
            if line.strip().endswith(","):
                # Check if next line also needs removal
                if i + 1 < len(lines):
                    next_line = lines[i + 1]
                    if "_sma_" not in next_line and "_1800s" not in next_line:
                        # Keep the comma for the next line
                        pass
            continue

        # Update log messages
        if "✓ Wrote 11 funding rate features" in line:
            line = line.replace("11 funding rate features", "5 funding rate features (pruned from 11)")
        elif "✓ Wrote 32 orderbook L0 features" in line:
            line = line.replace("32 orderbook L0 features", "18 orderbook L0 features (pruned from 32)")
        elif "✓ Wrote 11 price basis features" in line:
            line = line.replace("11 price basis features", "5 basis features (pruned from 11)")
        elif "✓ Wrote 6 open interest features" in line:
            line = line.replace("6 open interest features", "4 OI features (pruned from 6)")
        elif "✓ Wrote 79 RV/momentum/range/EMA features" in line:
            line = line.replace("79 RV/momentum/range/EMA features", "~45 RV/momentum/range features (pruned)")

        new_lines.append(line)

    # Write back
    with open(file_path, "w") as f:
        f.writelines(new_lines)

    print("✅ Comprehensive pruning complete!")

    # Verify by counting
    content = "".join(new_lines)
    sma_count = len(re.findall(r'_sma_|rolling_mean.*sma', content, re.IGNORECASE))
    count_1800 = len(re.findall(r'1800', content))

    print(f"Remaining SMA references: {sma_count}")
    print(f"Remaining 1800s references: {count_1800}")

    # Show what's left
    if sma_count > 0:
        print("\nRemaining SMA lines:")
        for i, line in enumerate(new_lines):
            if "_sma_" in line.lower() or ("rolling_mean" in line and "sma" in line.lower()):
                print(f"  Line {i+1}: {line.strip()}")

    if count_1800 > 10:  # Allow some in comments
        print("\nRemaining 1800s lines (excluding comments):")
        for i, line in enumerate(new_lines):
            if "1800" in line and not line.strip().startswith("#") and "removed" not in line.lower():
                print(f"  Line {i+1}: {line.strip()}")

if __name__ == "__main__":
    prune_v4_features()