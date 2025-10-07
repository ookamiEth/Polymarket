#!/usr/bin/env python3
"""
Monitor the streaming service for 30 minutes
Shows real-time updates and validates market transitions
"""

import time
import sys
from pathlib import Path
from datetime import datetime

LOG_FILE = Path("logs/streamer.log")
DATA_DIR = Path("data/raw")

def main():
    print("=" * 80)
    print("STREAMING SERVICE MONITOR")
    print("=" * 80)
    print("Monitoring for 30 minutes (2 market periods)")
    print(f"Start time: {datetime.now().strftime('%H:%M:%S')}")
    print()

    start_time = time.time()
    last_line_count = 0
    last_file_count = 0

    while time.time() - start_time < 1800:  # 30 minutes
        try:
            # Count log lines
            with open(LOG_FILE, 'r') as f:
                lines = f.readlines()
                current_line_count = len(lines)

            # Count parquet files
            parquet_files = list(DATA_DIR.glob("**/*.parquet"))
            current_file_count = len(parquet_files)

            # Check for new activity
            if current_line_count > last_line_count:
                new_lines = lines[last_line_count:]
                for line in new_lines:
                    # Print important events
                    if any(keyword in line for keyword in [
                        "Pre-fetching",
                        "transition",
                        "FLUSH",
                        "Flushing",
                        "Next market",
                        "ERROR",
                        "WARNING"
                    ]):
                        timestamp = datetime.now().strftime('%H:%M:%S')
                        print(f"[{timestamp}] {line.strip()}")

                last_line_count = current_line_count

            # Check for new files
            if current_file_count > last_file_count:
                timestamp = datetime.now().strftime('%H:%M:%S')
                print(f"[{timestamp}] 🗄️  New Parquet file created! Total files: {current_file_count}")
                for f in parquet_files:
                    if f.stat().st_mtime > time.time() - 10:  # Created in last 10s
                        size_kb = f.stat().st_size / 1024
                        print(f"[{timestamp}]    → {f} ({size_kb:.1f} KB)")

                last_file_count = current_file_count

            # Status update every minute
            elapsed = time.time() - start_time
            if int(elapsed) % 60 == 0 and elapsed > 0:
                timestamp = datetime.now().strftime('%H:%M:%S')
                print(f"\n[{timestamp}] ⏱️  Status: {elapsed/60:.0f} min elapsed, {current_line_count} log lines, {current_file_count} files")
                print()

            time.sleep(1)

        except KeyboardInterrupt:
            print("\n\nMonitoring stopped by user")
            break
        except Exception as e:
            print(f"Error: {e}")
            time.sleep(1)

    # Final summary
    print()
    print("=" * 80)
    print("MONITORING COMPLETE")
    print("=" * 80)
    print(f"Duration: {(time.time() - start_time)/60:.1f} minutes")
    print(f"Log lines: {current_line_count}")
    print(f"Parquet files: {current_file_count}")
    print()

    # List all parquet files
    if parquet_files:
        print("Files created:")
        for f in parquet_files:
            size_kb = f.stat().st_size / 1024
            print(f"  {f} ({size_kb:.1f} KB)")
    else:
        print("No parquet files created")

if __name__ == "__main__":
    main()
