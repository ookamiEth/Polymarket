#!/usr/bin/env python3
"""
Quick check of test progress
"""
import time
from pathlib import Path
from datetime import datetime

LOG_FILE = Path("logs/streamer.log")
DATA_DIR = Path("data/raw")

print("=" * 80)
print("TEST T007 PROGRESS CHECK")
print("=" * 80)
print(f"Time: {datetime.now().strftime('%H:%M:%S')}")
print()

# Check if service is running
try:
    with open(LOG_FILE, 'r') as f:
        lines = f.readlines()

    print(f"📊 Log lines: {len(lines)}")

    # Find important events
    events = {
        'Pre-fetch': 0,
        'transition': 0,
        'Flushing': 0,
        'ERROR': 0,
        'WARNING': 0
    }

    for line in lines:
        for keyword in events:
            if keyword.lower() in line.lower():
                events[keyword] += 1

    print(f"🔍 Key events:")
    for event, count in events.items():
        if count > 0:
            print(f"   {event}: {count}")

    # Check for parquet files
    parquet_files = sorted(DATA_DIR.glob("**/*.parquet"))
    print(f"\n📁 Parquet files: {len(parquet_files)}")
    for f in parquet_files:
        size_kb = f.stat().st_size / 1024
        mtime = datetime.fromtimestamp(f.stat().st_mtime).strftime('%H:%M:%S')
        print(f"   {f.name} ({size_kb:.1f} KB) - created at {mtime}")

    # Show last 5 log lines
    print(f"\n📝 Last 5 log entries:")
    for line in lines[-5:]:
        print(f"   {line.strip()}")

    # Calculate time until transition
    transition_time = 1759772700  # 19:45:00
    current_time = int(time.time())
    remaining = transition_time - current_time

    if remaining > 0:
        print(f"\n⏰ Time until first transition: {remaining//60}m {remaining%60}s")
    else:
        print(f"\n✅ First transition should have occurred {-remaining}s ago")

except Exception as e:
    print(f"❌ Error: {e}")

print("=" * 80)
