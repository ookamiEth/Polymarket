#!/usr/bin/env python3
"""Quick validation of parquet file"""
import polars as pl
import sys

file_path = sys.argv[1] if len(sys.argv) > 1 else "data/raw/2025/10/06/orderbook_20251006_1915.parquet"

print(f"Validating: {file_path}")
print()

df = pl.read_parquet(file_path)

print(f"Rows: {len(df)}")
print(f"Columns: {len(df.columns)}")
print(f"File size: {df.estimated_size('kb'):.1f} KB")
print()
print("Schema:")
for col in df.columns:
    print(f"  {col}: {df[col].dtype}")
print()
print("Sample data (first 3 rows):")
print(df.head(3))
print()
print("Sample data (last 3 rows):")
print(df.tail(3))
