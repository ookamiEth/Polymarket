#!/usr/bin/env python3
"""
Realistic Liquidity Rewards Analysis

IMPORTANT: The 'competitive' field is the TOTAL reward pool for ALL market makers,
not what YOU earn. Your actual earnings depend on:
1. Your share of total liquidity depth (Q_final)
2. Number of competing market makers
3. How tight your spreads are vs. competitors

This script provides realistic earnings estimates.
"""

import polars as pl
from pathlib import Path

def load_crypto_markets():
    """Load the latest crypto rewards data."""

    data_path = Path('data/rewards')
    snapshots = list(data_path.glob('crypto_rewards_*.parquet'))

    if not snapshots:
        print("Error: No crypto rewards data found!")
        print("Run: uv run python research/endpoint_tests/extract_crypto_rewards.py")
        return None

    latest = max(snapshots, key=lambda p: p.stat().st_mtime)
    print(f"Loading: {latest.name}\n")

    return pl.read_parquet(latest)

def estimate_competition_level(liquidity, volume):
    """
    Estimate number of active market makers based on liquidity and volume.

    Rules of thumb:
    - Low liquidity (<$50k) = 1-3 MMs
    - Medium liquidity ($50k-$200k) = 3-8 MMs
    - High liquidity (>$200k) = 8-20 MMs
    - Higher volume/liquidity ratio = more competition
    """

    if liquidity < 50000:
        base_mms = 2
    elif liquidity < 200000:
        base_mms = 5
    else:
        base_mms = 12

    # Adjust for volume (high volume attracts more MMs)
    if liquidity > 0:
        vol_ratio = volume / liquidity
        if vol_ratio > 1.0:  # Very active market
            base_mms = int(base_mms * 1.5)
        elif vol_ratio < 0.1:  # Inactive market
            base_mms = max(1, int(base_mms * 0.5))

    return base_mms

def calculate_realistic_earnings(df):
    """
    Calculate realistic daily earnings estimates.

    Assumptions:
    - You're a good MM but not dominant
    - You capture 10-30% of reward pool depending on competition
    - Spread profit depends on many factors we can't easily estimate
    """

    return df.with_columns([
        # Estimate number of competing MMs
        pl.struct(['liquidity_clob', 'volume_24hr'])
            .map_elements(
                lambda x: estimate_competition_level(x['liquidity_clob'], x['volume_24hr']),
                return_dtype=pl.Int64
            )
            .alias('estimated_mms'),

        # Your likely share of reward pool (1/N with some variance)
        # Conservative: assume you get 60% of "fair share" due to competition
        pl.when(pl.col('competitive').is_not_null())
            .then(
                (pl.col('competitive') /
                 pl.struct(['liquidity_clob', 'volume_24hr'])
                    .map_elements(
                        lambda x: max(1, estimate_competition_level(x['liquidity_clob'], x['volume_24hr'])),
                        return_dtype=pl.Int64
                    )
                ) * 0.6  # Conservative multiplier
            )
            .otherwise(0)
            .alias('realistic_reward_share'),
    ]).with_columns([
        # Return on capital from rewards alone (assuming you need 2x avg order size in capital)
        (pl.col('realistic_reward_share') * 365 /
         (pl.col('rewards_min_size') * 2).replace(0, 1000))  # Default $1000 capital
            .alias('annualized_return_pct_rewards_only'),
    ])

def print_realistic_analysis(df):
    """Print corrected analysis with realistic numbers."""

    print("=" * 80)
    print(" REALISTIC CRYPTO MARKET MAKING PROFIT ANALYSIS")
    print("=" * 80)

    print("\n⚠️  IMPORTANT:")
    print("  - 'competitive' = TOTAL pool for ALL market makers (not just you!)")
    print("  - Your share depends on your liquidity depth vs. competitors")
    print("  - These show REWARDS ONLY (spread profit requires order book analysis)")
    print("  - Real MM profit mostly comes from spread capture, not rewards")

    print("\n" + "=" * 80)
    print(" TOP 20 MARKETS BY REALISTIC DAILY PROFIT")
    print("=" * 80)

    # Filter for active markets with data
    active = df.filter(
        (pl.col('volume_24hr') > 100) &
        (pl.col('competitive').is_not_null())
    ).sort('realistic_reward_share', descending=True)

    if active.is_empty():
        print("No active markets with sufficient data")
        return

    print(f"\nFound {len(active)} active markets with rewards\n")

    for i, row in enumerate(active.head(20).iter_rows(named=True), 1):
        print(f"{i}. {row['question'][:65]}")
        print(f"   Total Pool: ${row['competitive']:.2f}/day | Est. MMs: {row['estimated_mms']}")
        print(f"   ★ Your Est. Reward Share: ${row['realistic_reward_share']:.2f}/day")
        print(f"   Volume: ${row['volume_24hr']:,.0f} | Liquidity: ${row['liquidity_clob']:,.0f}")
        print(f"   Capital Needed: ~${row['rewards_min_size'] * 2:.0f} | Rewards ROI: {row['annualized_return_pct_rewards_only']:.1f}%/year")
        print()

    print("=" * 80)
    print(" PROFITABILITY BREAKDOWN")
    print("=" * 80)

    # Categorize by reward level
    high_reward = active.filter(pl.col('realistic_reward_share') > 0.50)
    medium_reward = active.filter(
        (pl.col('realistic_reward_share') >= 0.10) &
        (pl.col('realistic_reward_share') <= 0.50)
    )
    low_reward = active.filter(pl.col('realistic_reward_share') < 0.10)

    print(f"\nHigh Rewards (>$0.50/day):       {len(high_reward)} markets")
    print(f"Medium Rewards ($0.10-$0.50/day): {len(medium_reward)} markets")
    print(f"Low Rewards (<$0.10/day):         {len(low_reward)} markets")

    total_rewards = active['realistic_reward_share'].sum()
    print(f"\nTotal Est. Daily Rewards (if you MM all markets): ${total_rewards:.2f}")
    print(f"Monthly: ${total_rewards * 30:.2f} | Yearly: ${total_rewards * 365:.2f}")
    print(f"\n⚠️  NOTE: This is REWARDS ONLY. Spread capture profit is typically 10-100x higher!")

    print("\n" + "=" * 80)
    print(" THE REALITY OF MM PROFITS")
    print("=" * 80)

    print("\n💡 Where the REAL money comes from:")
    print("   1. SPREAD CAPTURE (90-99% of profit)")
    print("      - Buy at bid, sell at ask")
    print("      - Profit = (ask - bid) × shares filled")
    print("      - Depends on: volume, spread width, fill rate, competition")
    print("      - Cannot be estimated without order book data")
    print()
    print("   2. LIQUIDITY REWARDS (1-10% of profit)")
    print("      - These ~$0.10-$0.50/day numbers you see")
    print("      - Shared pool among all MMs")
    print("      - Your share ≈ (your Q_final) × (competitive pool)")
    print("      - Paid for keeping orders in the book")
    print()
    print("   3. RISKS")
    print("      - Inventory risk: market moves against you")
    print("      - Need capital for two-sided quoting")
    print("      - Requires constant rebalancing")

    print("\n" + "=" * 80)

def compare_updown_vs_longterm():
    """Compare Up/Down markets vs long-term prediction markets."""

    print("\n" + "=" * 80)
    print(" UP/DOWN vs LONG-TERM MARKETS COMPARISON")
    print("=" * 80)

    # Load Up/Down markets
    updown_path = Path('data/rewards')
    updown_files = list(updown_path.glob('updown_markets_*.parquet'))

    if updown_files:
        updown_df = pl.read_parquet(max(updown_files, key=lambda p: p.stat().st_mtime))

        # Calculate for one active Up/Down market
        active_updown = updown_df.filter(pl.col('volume_24hr') > 50).head(1)

        if not active_updown.is_empty():
            row = active_updown.row(0, named=True)

            print("\n📊 EXAMPLE: Hourly Up/Down Market")
            print(f"   Market: {row['question']}")

            # Handle None values
            comp = row.get('competitive') if row.get('competitive') is not None else 0.8
            print(f"   Total Pool: ${comp:.2f}/day")
            print(f"   Market Lifespan: 1 hour")
            print(f"   Volume: ${row['volume_24hr']:,.2f}")

            # Realistic calculation
            est_mms = 3  # Usually 2-4 MMs on these
            hourly_pool = comp / 24  # Per hour
            your_share = (hourly_pool / est_mms) * 0.6

            print(f"\n   Est. competing MMs: {est_mms}")
            print(f"   Hourly pool: ${hourly_pool:.4f}")
            print(f"   Your hourly share: ${your_share:.4f}")
            print(f"   ★ REALISTIC HOURLY PROFIT: ${your_share:.3f}")
            print(f"   If you MM 24/7: ${your_share * 24:.2f}/day")

    print("\n📊 EXAMPLE: Long-term Prediction Market")
    print("   Market: Bitcoin reaches $200k by Dec 31")
    print("   Total Pool: $0.83/day")
    print("   Volume: $47,149/day")
    print("   Market Lifespan: ~90 days")

    est_mms = 5
    your_reward_share = (0.83 / est_mms) * 0.6

    print(f"\n   Est. competing MMs: {est_mms}")
    print(f"   Your reward share: ${your_reward_share:.2f}/day")
    print(f"   Over 90 days: ${your_reward_share * 90:.2f} from rewards")
    print(f"   Plus: Unknown spread profit (depends on fill rate & spread width)")

    print("\n💡 Long-term markets are better:")
    print("   - Higher volume = more spread capture opportunity")
    print("   - Stable positions over time")
    print("   - Can optimize strategy over weeks/months")

def main():
    print("=" * 80)
    print(" CORRECTED CRYPTO REWARDS ANALYSIS")
    print("=" * 80)
    print()

    df = load_crypto_markets()
    if df is None:
        return

    # Calculate realistic numbers
    df = calculate_realistic_earnings(df)

    # Print analysis
    print_realistic_analysis(df)

    # Compare market types
    compare_updown_vs_longterm()

    print("\n" + "=" * 80)
    print(" BOTTOM LINE")
    print("=" * 80)
    print("\n✅ Liquidity rewards exist:")
    print("   - ~$0.10-$0.50/day per market (your share)")
    print("   - Shared pool among all competing MMs")
    print("   - Paid for keeping tight orders in the book")
    print()
    print("⚠️  But spread capture is the real profit:")
    print("   - Cannot estimate without order book data")
    print("   - Depends on: fill rate, spread width, volume, competition")
    print("   - Typically 10-100x larger than reward amounts")
    print()
    print("💰 Capital & Risk:")
    print("   - Need $5k-$20k per market for meaningful size")
    print("   - Requires 24/7 monitoring and rebalancing")
    print("   - Inventory risk can wipe out profits")

    print("\n🎯 BEST MARKETS:")
    print("   1. High volume (>$10k/day)")
    print("   2. Medium competition (5-10 MMs, not 20+)")
    print("   3. Long-term duration (weeks/months, not hours)")

    print("\n❌ AVOID:")
    print("   1. Hourly Up/Down markets (<$0.01/hour rewards)")
    print("   2. Low volume (<$500/day)")
    print("   3. Extremely competitive (20+ MMs)")

if __name__ == "__main__":
    main()
