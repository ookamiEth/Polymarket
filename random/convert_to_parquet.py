#!/usr/bin/env python3
"""
Convert MegaETH deposit CSV to Parquet format.

Handles quoted columns and comma-separated numbers in TokenValue field.
"""

import polars as pl


def convert_csv_to_parquet(
    csv_path: str,
    parquet_path: str,
) -> None:
    """
    Convert CSV file to Parquet format with proper type handling.

    Args:
        csv_path: Path to input CSV file
        parquet_path: Path to output Parquet file
    """
    print(f"Reading CSV from: {csv_path}")

    # Read CSV with Polars
    df = pl.read_csv(
        csv_path,
        infer_schema_length=1000,
        quote_char='"',
    )

    print(f"Loaded {len(df):,} rows")

    # Clean TokenValue column: remove commas and convert to float
    df = df.with_columns(pl.col("TokenValue").str.replace_all(",", "").cast(pl.Float64).alias("TokenValue"))

    # Convert UnixTimestamp to proper integer type
    df = df.with_columns(pl.col("UnixTimestamp").cast(pl.Int64))

    # Parse DateTime column as proper datetime
    df = df.with_columns(pl.col("DateTime (UTC)").str.strptime(pl.Datetime, "%Y-%m-%d %H:%M:%S").alias("DateTime"))

    # Show summary
    print("\nData summary:")
    print(f"  Unique depositors (From): {df['From'].n_unique():,}")
    print(f"  Total TokenValue (USDT): {df['TokenValue'].sum():,.2f}")
    print(f"  Date range: {df['DateTime'].min()} to {df['DateTime'].max()}")

    # Write to Parquet with snappy compression
    print(f"\nWriting Parquet to: {parquet_path}")
    df.write_parquet(
        parquet_path,
        compression="snappy",
        statistics=True,
    )

    print("✅ Conversion complete!")
    print(f"   Output size: {len(df):,} rows")


def main() -> None:
    """Entry point for the script."""
    csv_path = (
        "/Users/lgierhake/Downloads/export-address-token-0xab02bf85a7a851b6a379ea3d5bd3b9b4f5dd8461_withnotes.csv"
    )
    parquet_path = "/Users/lgierhake/Downloads/megaeth_deposits.parquet"

    convert_csv_to_parquet(csv_path, parquet_path)


if __name__ == "__main__":
    main()
