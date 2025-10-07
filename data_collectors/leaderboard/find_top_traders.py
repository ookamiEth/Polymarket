#!/usr/bin/env python3
"""
Find Top 10 Highest Volume Traders on Polymarket

This script uses the discovered leaderboard API to identify
the top traders by volume on Polymarket.

Usage:
    uv run python find_top_traders.py
"""

import requests
import json
import csv
from datetime import datetime
from pathlib import Path

LEADERBOARD_URL = "https://data-api.polymarket.com/leaderboard"
DATA_API_URL = "https://data-api.polymarket.com"

def get_leaderboard(limit=50):
    """Fetch leaderboard data from Polymarket API"""
    print(f"Fetching top {limit} traders from leaderboard...")

    params = {"limit": limit}
    response = requests.get(LEADERBOARD_URL, params=params, timeout=10)
    response.raise_for_status()

    data = response.json()
    print(f"✓ Retrieved {len(data)} traders")

    return data

def get_user_additional_stats(address, max_attempts=1):
    """Get additional statistics for a user"""
    stats = {
        "total_value": None,
        "current_positions_count": None,
        "trade_count_sample": None
    }

    try:
        # Get total value of positions
        value_url = f"{DATA_API_URL}/value"
        value_resp = requests.get(value_url, params={"user": address}, timeout=10)
        if value_resp.status_code == 200:
            value_data = value_resp.json()
            if value_data:
                stats["total_value"] = value_data[0].get("value")

        # Get current positions count
        positions_url = f"{DATA_API_URL}/positions"
        positions_resp = requests.get(
            positions_url,
            params={"user": address, "limit": 1},
            timeout=10
        )
        if positions_resp.status_code == 200:
            # We'd need to paginate to get exact count, for now just check if they have positions
            positions_data = positions_resp.json()
            stats["current_positions_count"] = len(positions_data) if positions_data else 0

        # Get a sample trade to verify activity
        trades_url = f"{DATA_API_URL}/trades"
        trades_resp = requests.get(
            trades_url,
            params={"user": address, "limit": 1},
            timeout=10
        )
        if trades_resp.status_code == 200:
            trades_data = trades_resp.json()
            stats["trade_count_sample"] = len(trades_data)

    except Exception as e:
        print(f"  Warning: Could not fetch additional stats for {address[:10]}...: {e}")

    return stats

def analyze_top_traders(data, top_n=10, fetch_additional=False):
    """Analyze and rank traders by volume"""
    print(f"\nAnalyzing top {top_n} traders by volume...")

    # Sort by volume (descending)
    sorted_by_volume = sorted(
        data,
        key=lambda x: float(x.get("vol", 0)),
        reverse=True
    )

    top_traders = []
    for i, trader in enumerate(sorted_by_volume[:top_n], 1):
        address = trader.get("user_id", "")
        username = trader.get("user_name", address)
        volume = float(trader.get("vol", 0))
        pnl = float(trader.get("pnl", 0))

        trader_info = {
            "rank": i,
            "address": address,
            "username": username,
            "total_volume_usd": volume,
            "pnl_usd": pnl,
            "profile_image": trader.get("profile_image", ""),
            "profile_url": f"https://polymarket.com/@{username}" if username != address else ""
        }

        # Optionally fetch additional stats (slower)
        if fetch_additional:
            print(f"  Fetching additional stats for rank #{i} ({username[:20]}...)")
            additional = get_user_additional_stats(address)
            trader_info.update(additional)

        top_traders.append(trader_info)
        print(f"  #{i}: {username[:30]:<30} | Volume: ${volume:>15,.2f} | PnL: ${pnl:>12,.2f}")

    return top_traders

def save_results(top_traders):
    """Save results in multiple formats"""
    timestamp = datetime.now().isoformat()
    results_dir = Path("/Users/lgierhake/Documents/ETH/BT/top_traders/results")
    results_dir.mkdir(exist_ok=True)

    # JSON output
    json_data = {
        "generated_at": timestamp,
        "data_source": "polymarket_data_api_leaderboard",
        "endpoint": LEADERBOARD_URL,
        "top_10_traders": top_traders,
        "summary": {
            "total_volume": sum(t["total_volume_usd"] for t in top_traders),
            "total_pnl": sum(t["pnl_usd"] for t in top_traders),
            "avg_volume_per_trader": sum(t["total_volume_usd"] for t in top_traders) / len(top_traders),
            "avg_pnl_per_trader": sum(t["pnl_usd"] for t in top_traders) / len(top_traders)
        }
    }

    json_file = results_dir / "top_10_traders.json"
    with open(json_file, 'w') as f:
        json.dump(json_data, f, indent=2)
    print(f"\n✓ JSON saved to: {json_file}")

    # CSV output
    csv_file = results_dir / "top_10_traders.csv"
    with open(csv_file, 'w', newline='') as f:
        if top_traders:
            writer = csv.DictWriter(f, fieldnames=top_traders[0].keys())
            writer.writeheader()
            writer.writerows(top_traders)
    print(f"✓ CSV saved to: {csv_file}")

    # Markdown report
    md_file = results_dir / "analysis_summary.md"
    with open(md_file, 'w') as f:
        f.write("# Top 10 Polymarket Traders by Volume\n\n")
        f.write(f"**Generated**: {timestamp}\n")
        f.write(f"**Data Source**: Polymarket Data API Leaderboard\n\n")

        f.write("## Summary Statistics\n\n")
        f.write(f"- **Total Combined Volume**: ${json_data['summary']['total_volume']:,.2f}\n")
        f.write(f"- **Total Combined P&L**: ${json_data['summary']['total_pnl']:,.2f}\n")
        f.write(f"- **Average Volume per Trader**: ${json_data['summary']['avg_volume_per_trader']:,.2f}\n")
        f.write(f"- **Average P&L per Trader**: ${json_data['summary']['avg_pnl_per_trader']:,.2f}\n\n")

        f.write("## Top 10 Traders\n\n")
        f.write("| Rank | Username | Volume (USD) | P&L (USD) | Address |\n")
        f.write("|------|----------|--------------|-----------|----------|\n")

        for trader in top_traders:
            username = trader['username'][:25]
            address_short = trader['address'][:12] + "..."
            volume = f"${trader['total_volume_usd']:,.2f}"
            pnl = f"${trader['pnl_usd']:,.2f}"
            f.write(f"| {trader['rank']} | {username} | {volume} | {pnl} | `{address_short}` |\n")

        f.write("\n## Details\n\n")
        for trader in top_traders:
            f.write(f"### {trader['rank']}. {trader['username']}\n\n")
            f.write(f"- **Address**: `{trader['address']}`\n")
            f.write(f"- **Total Volume**: ${trader['total_volume_usd']:,.2f}\n")
            f.write(f"- **P&L**: ${trader['pnl_usd']:,.2f}\n")
            if trader.get('profile_url'):
                f.write(f"- **Profile**: {trader['profile_url']}\n")
            f.write("\n")

    print(f"✓ Markdown report saved to: {md_file}")

def main():
    print("="*80)
    print(" POLYMARKET TOP TRADERS ANALYSIS")
    print("="*80)

    try:
        # Fetch leaderboard data
        leaderboard_data = get_leaderboard(limit=100)  # Get top 100 to analyze

        # Analyze and get top 10 by volume
        top_10 = analyze_top_traders(leaderboard_data, top_n=10, fetch_additional=False)

        # Save results
        save_results(top_10)

        print("\n" + "="*80)
        print(" ANALYSIS COMPLETE")
        print("="*80)

        print(f"\nTop trader by volume: {top_10[0]['username']}")
        print(f"Volume: ${top_10[0]['total_volume_usd']:,.2f}")
        print(f"\nAll results saved to: /Users/lgierhake/Documents/ETH/BT/top_traders/results/")

    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0

if __name__ == "__main__":
    exit(main())
