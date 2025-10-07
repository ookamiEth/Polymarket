#!/usr/bin/env python3
"""
Example Usage: Historical CLOB Data Collection

This script demonstrates all major features of the CLOB historical data collector:
1. Real-time WebSocket collection
2. Historical trade fetching
3. Order book reconstruction
4. Analysis and visualization

Run this to get started with collecting Polymarket order book data.
"""

import asyncio
import time
from datetime import datetime, timedelta
from pathlib import Path

from orderbook_collector import OrderBookCollector
from orderbook_reconstructor import OrderBookReconstructor


def print_section(title: str):
    """Print formatted section header"""
    print(f"\n{'='*80}")
    print(f" {title}")
    print(f"{'='*80}\n")


async def example_1_realtime_collection():
    """Example 1: Collect real-time order book data"""
    print_section("EXAMPLE 1: Real-Time Order Book Collection")

    collector = OrderBookCollector(output_dir="data/orderbooks/example1")

    # Popular markets - Trump 2024 Election
    market_id = "0xdd31ce870876bb410d71f29e3dfea77910bc3effb81a0fb8dc8c282d5968aa5a"
    token_id_yes = "21742633143463906290569050155826241533067272736897614950488156847949938836455"

    print("This will collect order book data for 2 minutes...")
    print("Press Ctrl+C to stop early\n")

    await collector.collect_realtime(
        markets=[market_id],
        token_ids=[token_id_yes],
        duration_minutes=2,
        save_interval_seconds=30
    )

    print("\n✓ Real-time collection complete!")
    print(f"  Data saved to: {collector.output_dir}")


def example_2_historical_trades():
    """Example 2: Fetch historical trades"""
    print_section("EXAMPLE 2: Historical Trade Data")

    collector = OrderBookCollector(output_dir="data/orderbooks/example2")

    # Fetch last 24 hours of trades
    end_time = int(time.time())
    start_time = end_time - (24 * 60 * 60)

    market_id = "0xdd31ce870876bb410d71f29e3dfea77910bc3effb81a0fb8dc8c282d5968aa5a"

    print(f"Fetching trades from last 24 hours...")
    print(f"Start: {datetime.fromtimestamp(start_time)}")
    print(f"End: {datetime.fromtimestamp(end_time)}")

    trades_df = collector.collect_historical_trades(
        market=market_id,
        start_timestamp=start_time,
        end_timestamp=end_time,
        limit=1000
    )

    if not trades_df.empty:
        print(f"\n✓ Collected {len(trades_df)} trades")
        print(f"\nTrade Summary:")
        print(f"  Columns: {list(trades_df.columns)}")
        print(f"\nFirst 5 trades:")
        print(trades_df.head())

        # Analyze trades
        print(f"\nTrade Statistics:")
        print(f"  Total volume: {trades_df['size'].sum():.2f}")
        print(f"  Avg price: ${trades_df['price'].mean():.4f}")
        print(f"  Price range: ${trades_df['price'].min():.4f} - ${trades_df['price'].max():.4f}")

        # Side distribution
        side_counts = trades_df['side'].value_counts()
        print(f"\nSide Distribution:")
        for side, count in side_counts.items():
            print(f"  {side}: {count} trades ({count/len(trades_df)*100:.1f}%)")
    else:
        print("\nNo trades found in this period")


def example_3_reconstruction():
    """Example 3: Reconstruct order book from collected data"""
    print_section("EXAMPLE 3: Order Book Reconstruction")

    reconstructor = OrderBookReconstructor()
    data_dir = Path("data/orderbooks/example1")

    if not data_dir.exists():
        print(f"No data found in {data_dir}")
        print("Run Example 1 first to collect data")
        return

    # Load collected data
    print("Loading collected data...")
    reconstructor.load_snapshots(str(data_dir / "book_snapshots_*.parquet"))
    reconstructor.load_price_changes(str(data_dir / "price_changes_*.parquet"))
    reconstructor.load_trades(str(data_dir / "trades_*.parquet"))

    if not reconstructor.snapshots and not reconstructor.price_changes:
        print("No data loaded - run Example 1 first")
        return

    # Get asset ID
    asset_id = reconstructor.snapshots[0]['asset_id'] if reconstructor.snapshots else \
               reconstructor.price_changes[0]['asset_id']

    print(f"\nAsset ID: {asset_id[:30]}...")

    # Build time series
    print("\nReconstructing order book states...")
    states = reconstructor.build_timeseries(
        asset_id=asset_id,
        interval_seconds=1
    )

    if not states:
        print("No states reconstructed")
        return

    print(f"\n✓ Reconstructed {len(states)} book states")
    print(f"  Time range: {states[0].timestamp} to {states[-1].timestamp}")

    # Show sample states
    print(f"\nSample Book States:")
    for i in range(min(3, len(states))):
        state = states[i]
        print(f"\n  [{i+1}] {state.timestamp}")
        print(f"      Best Bid: ${state.get_best_bid()[0]:.4f} x {state.get_best_bid()[1]:.2f}")
        print(f"      Best Ask: ${state.get_best_ask()[0]:.4f} x {state.get_best_ask()[1]:.2f}")
        print(f"      Spread: ${state.get_spread():.6f}")
        print(f"      Mid Price: ${state.get_mid_price():.4f}")

    # Compute statistics
    print("\nComputing statistics...")
    stats_df = reconstructor.analyze_book_statistics(states)

    print(f"\nBook Statistics Summary:")
    print(stats_df[['spread', 'mid_price', 'bid_levels', 'ask_levels']].describe())

    # Export
    output_file = data_dir / "reconstructed_timeseries.parquet"
    reconstructor.export_timeseries(states, str(output_file))
    print(f"\n✓ Exported to {output_file}")


def example_4_current_snapshot():
    """Example 4: Get current order book snapshot"""
    print_section("EXAMPLE 4: Current Order Book Snapshot")

    collector = OrderBookCollector()

    token_id = "21742633143463906290569050155826241533067272736897614950488156847949938836455"

    print(f"Fetching current order book...")
    print(f"Token ID: {token_id[:30]}...")

    book = collector.get_orderbook_summary(token_id)

    if book:
        print(f"\n✓ Current Order Book:")
        print(f"  Timestamp: {book.get('timestamp')}")
        print(f"  Hash: {book.get('hash', '')[:30]}...")

        bids = book.get('bids', [])
        asks = book.get('asks', [])

        print(f"\n  Top 5 Bids:")
        for i, bid in enumerate(bids[:5], 1):
            print(f"    {i}. ${bid['price']:8} x {bid['size']:10}")

        print(f"\n  Top 5 Asks:")
        for i, ask in enumerate(asks[:5], 1):
            print(f"    {i}. ${ask['price']:8} x {ask['size']:10}")

        if bids and asks:
            spread = float(asks[0]['price']) - float(bids[0]['price'])
            mid = (float(asks[0]['price']) + float(bids[0]['price'])) / 2
            print(f"\n  Spread: ${spread:.6f}")
            print(f"  Mid Price: ${mid:.4f}")
    else:
        print("\nFailed to fetch order book")


async def example_5_multi_market():
    """Example 5: Collect multiple markets simultaneously"""
    print_section("EXAMPLE 5: Multi-Market Collection")

    collector = OrderBookCollector(output_dir="data/orderbooks/example5")

    # Multiple markets
    markets = [
        {
            "name": "Trump Election 2024",
            "market_id": "0xdd31ce870876bb410d71f29e3dfea77910bc3effb81a0fb8dc8c282d5968aa5a",
            "token_id": "21742633143463906290569050155826241533067272736897614950488156847949938836455"
        }
        # Add more markets here
    ]

    market_ids = [m["market_id"] for m in markets]
    token_ids = [m["token_id"] for m in markets]

    print(f"Collecting {len(markets)} markets:")
    for m in markets:
        print(f"  - {m['name']}")

    print("\nCollection duration: 1 minute")

    await collector.collect_realtime(
        markets=market_ids,
        token_ids=token_ids,
        duration_minutes=1,
        save_interval_seconds=20
    )

    print("\n✓ Multi-market collection complete!")


def show_menu():
    """Display interactive menu"""
    print("\n" + "="*80)
    print(" POLYMARKET HISTORICAL CLOB DATA COLLECTOR - EXAMPLES")
    print("="*80)
    print("\nSelect an example to run:\n")
    print("  1. Real-time order book collection (2 minutes)")
    print("  2. Fetch historical trades (last 24h)")
    print("  3. Reconstruct order book from collected data")
    print("  4. Get current order book snapshot")
    print("  5. Multi-market collection (1 minute)")
    print("  6. Run all examples sequentially")
    print("  0. Exit")
    print("\n" + "="*80)


async def run_all_examples():
    """Run all examples in sequence"""
    print_section("RUNNING ALL EXAMPLES")

    print("This will run all examples sequentially...")
    print("Total time: ~5 minutes\n")

    input("Press Enter to continue or Ctrl+C to cancel...")

    # Example 1: Real-time collection
    await example_1_realtime_collection()

    # Example 2: Historical trades
    example_2_historical_trades()

    # Example 3: Reconstruction
    example_3_reconstruction()

    # Example 4: Current snapshot
    example_4_current_snapshot()

    # Example 5: Multi-market
    await example_5_multi_market()

    print_section("ALL EXAMPLES COMPLETE")
    print("Check the data/orderbooks/ directory for collected data")


async def main():
    """Main interactive menu"""
    while True:
        show_menu()

        try:
            choice = input("\nEnter choice (0-6): ").strip()

            if choice == "0":
                print("\nExiting...")
                break

            elif choice == "1":
                await example_1_realtime_collection()

            elif choice == "2":
                example_2_historical_trades()

            elif choice == "3":
                example_3_reconstruction()

            elif choice == "4":
                example_4_current_snapshot()

            elif choice == "5":
                await example_5_multi_market()

            elif choice == "6":
                await run_all_examples()

            else:
                print("\nInvalid choice. Please enter 0-6.")

            input("\n\nPress Enter to continue...")

        except KeyboardInterrupt:
            print("\n\nInterrupted by user")
            break
        except Exception as e:
            print(f"\nError: {e}")
            input("\nPress Enter to continue...")


if __name__ == "__main__":
    print("\n" + "="*80)
    print(" POLYMARKET HISTORICAL CLOB DATA COLLECTOR")
    print("="*80)
    print("\nDependencies: websockets, pandas, pyarrow, requests")
    print("Install: uv pip install websockets pandas pyarrow requests")
    print()

    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n\nExiting...")
