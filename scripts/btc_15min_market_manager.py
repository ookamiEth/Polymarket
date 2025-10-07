#!/usr/bin/env python3
"""
Bitcoin 15-Minute Market Discovery and Management System

Discovers, tracks, and rotates through active 15-minute BTC "Up or Down" markets
for continuous orderbook streaming.

Features:
- Auto-discovery of new 15-min BTC markets
- Market rotation every 15 minutes
- Graceful transitions between markets
- Buffer period handling to catch late-opening markets
- Persistent state management

Usage:
    # Discover current and upcoming markets
    uv run python scripts/btc_15min_market_manager.py discover

    # Monitor and output next market info
    uv run python scripts/btc_15min_market_manager.py next

    # Run continuous discovery (daemon mode)
    uv run python scripts/btc_15min_market_manager.py daemon
"""

import json
import time
import requests
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import List, Dict, Optional
import re

# ==============================================================================
# Configuration
# ==============================================================================

GAMMA_API_URL = "https://gamma-api.polymarket.com/markets"
CLOB_API_URL = "https://clob.polymarket.com/markets"

# Market discovery settings
CHECK_INTERVAL_SECONDS = 60  # Check for new markets every minute
MARKET_LOOKAHEAD_MINUTES = 30  # Look for markets opening in next 30 mins
MARKET_BUFFER_MINUTES = 5  # Start streaming 5 mins before official end time

# Storage
STATE_FILE = Path("data/market_state/btc_15min_current.json")
DISCOVERED_MARKETS_FILE = Path("data/market_state/btc_15min_discovered.json")

# ==============================================================================
# Market Discovery
# ==============================================================================

class BTCMarketDiscovery:
    """
    Discovers active and upcoming 15-minute BTC "Up or Down" markets.
    """

    def __init__(self):
        self.discovered_markets = self._load_discovered_markets()

    def _load_discovered_markets(self) -> Dict:
        """Load previously discovered markets from file."""
        if DISCOVERED_MARKETS_FILE.exists():
            with open(DISCOVERED_MARKETS_FILE, 'r') as f:
                return json.load(f)
        return {"markets": [], "last_update": None}

    def _save_discovered_markets(self):
        """Save discovered markets to file."""
        DISCOVERED_MARKETS_FILE.parent.mkdir(parents=True, exist_ok=True)
        with open(DISCOVERED_MARKETS_FILE, 'w') as f:
            json.dump(self.discovered_markets, f, indent=2)

    def is_btc_15min_market(self, market: Dict) -> bool:
        """
        Check if market is a Bitcoin 15-minute Up or Down market.

        Patterns:
        - Title: "Bitcoin Up or Down - October 6, 2:15PM-2:30PM ET"
        - Slug: "btc-up-or-down-15m-{timestamp}"
        """
        slug = market.get('slug', '')
        question = market.get('question', '')

        # Check slug pattern
        if 'btc-up-or-down-15m-' in slug:
            return True

        # Check question pattern
        if ('bitcoin' in question.lower() and
            'up or down' in question.lower() and
            '-' in question and  # Has time range like "2:15PM-2:30PM"
            (':' in question or 'am' in question.lower() or 'pm' in question.lower())):

            # Verify it's a 15-minute interval (has both start and end time)
            # Format: "HH:MM{AM|PM}-HH:MM{AM|PM}"
            time_pattern = r'\d{1,2}:\d{2}[AP]M-\d{1,2}:\d{2}[AP]M'
            if re.search(time_pattern, question):
                return True

        return False

    def discover_active_markets(self) -> List[Dict]:
        """
        Query Polymarket API for currently active 15-min BTC markets.
        """
        print("🔍 Discovering active 15-min BTC markets...")

        active_markets = []

        try:
            # Query Gamma API
            params = {
                "closed": "false",
                "active": "true",
                "limit": 500
            }

            response = requests.get(GAMMA_API_URL, params=params, timeout=10)
            response.raise_for_status()
            markets = response.json()

            print(f"   Retrieved {len(markets)} total active markets")

            # Filter for 15-min BTC markets
            for market in markets:
                if self.is_btc_15min_market(market):
                    active_markets.append(self._extract_market_info(market))

            print(f"   ✅ Found {len(active_markets)} active 15-min BTC markets")

        except Exception as e:
            print(f"   ❌ Error querying API: {e}")

        # Update discovered markets cache
        self.discovered_markets["markets"] = active_markets
        self.discovered_markets["last_update"] = datetime.now(timezone.utc).isoformat()
        self._save_discovered_markets()

        return active_markets

    def _extract_market_info(self, market: Dict) -> Dict:
        """Extract relevant information from market data."""

        # Parse end time from question or endDate
        end_time = None
        if market.get('endDate'):
            try:
                end_time = datetime.fromisoformat(market['endDate'].replace('Z', '+00:00'))
            except:
                pass

        # Extract token IDs
        token_ids = market.get('clobTokenIds', '').split(',') if market.get('clobTokenIds') else []

        return {
            'condition_id': market.get('conditionId'),
            'slug': market.get('slug'),
            'question': market.get('question'),
            'token_id_up': token_ids[0] if len(token_ids) > 0 else None,
            'token_id_down': token_ids[1] if len(token_ids) > 1 else None,
            'end_date_iso': market.get('endDate'),
            'end_time_utc': end_time.isoformat() if end_time else None,
            'active': market.get('active', False),
            'closed': market.get('closed', False),
            'discovered_at': datetime.now(timezone.utc).isoformat(),
        }

# ==============================================================================
# Market Rotation Manager
# ==============================================================================

class MarketRotationManager:
    """
    Manages rotation between 15-minute markets for continuous streaming.
    """

    def __init__(self):
        self.discovery = BTCMarketDiscovery()
        self.current_state = self._load_state()

    def _load_state(self) -> Dict:
        """Load current market state."""
        if STATE_FILE.exists():
            with open(STATE_FILE, 'r') as f:
                return json.load(f)
        return {
            "current_market": None,
            "next_market": None,
            "last_rotation": None,
            "rotation_count": 0
        }

    def _save_state(self):
        """Save current state to file."""
        STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
        with open(STATE_FILE, 'w') as f:
            json.dump(self.current_state, f, indent=2)

    def get_current_market(self) -> Optional[Dict]:
        """Get the market that should be streamed right now."""

        # Discover active markets
        active_markets = self.discovery.discover_active_markets()

        if not active_markets:
            print("⚠️  No active 15-min BTC markets found")
            return None

        # Find market that's currently active (not yet expired)
        now = datetime.now(timezone.utc)

        current_market = None
        for market in active_markets:
            if market.get('end_time_utc'):
                end_time = datetime.fromisoformat(market['end_time_utc'])

                # Market is current if it hasn't ended yet
                if end_time > now:
                    current_market = market
                    break

        if current_market:
            print(f"✅ Current market: {current_market['question']}")
            print(f"   Ends: {current_market['end_time_utc']}")
            self.current_state["current_market"] = current_market
            self._save_state()
        else:
            print("⚠️  No current market found (all may have expired)")

        return current_market

    def get_next_market(self) -> Optional[Dict]:
        """Get the next market that will open."""

        active_markets = self.discovery.discover_active_markets()

        if not active_markets:
            return None

        # Sort by end time and get the next one
        now = datetime.now(timezone.utc)

        future_markets = []
        for market in active_markets:
            if market.get('end_time_utc'):
                end_time = datetime.fromisoformat(market['end_time_utc'])
                if end_time > now:
                    future_markets.append(market)

        if len(future_markets) >= 2:
            # Sort by end time
            future_markets.sort(key=lambda m: m['end_time_utc'])

            # Next market is the second one (first is current)
            next_market = future_markets[1]
            print(f"📅 Next market: {next_market['question']}")
            print(f"   Starts: {next_market['end_time_utc']}")

            self.current_state["next_market"] = next_market
            self._save_state()

            return next_market

        return None

    def should_rotate(self) -> bool:
        """Check if we should rotate to a new market."""

        current = self.current_state.get("current_market")

        if not current or not current.get('end_time_utc'):
            return True

        now = datetime.now(timezone.utc)
        end_time = datetime.fromisoformat(current['end_time_utc'])

        # Rotate if market has ended
        if now >= end_time:
            print("🔄 Current market has ended - rotating")
            return True

        # Rotate if we're within buffer period of end
        buffer_time = end_time - timedelta(minutes=MARKET_BUFFER_MINUTES)
        if now >= buffer_time:
            print(f"🔄 Within {MARKET_BUFFER_MINUTES}-min buffer - preparing rotation")
            return True

        return False

    def rotate_to_next(self) -> Optional[Dict]:
        """Rotate to the next market."""

        print("\n" + "="*80)
        print(" MARKET ROTATION")
        print("="*80)

        # Get next market
        next_market = self.get_next_market()

        if not next_market:
            print("❌ No next market available - will retry discovery")
            return None

        # Update state
        self.current_state["current_market"] = next_market
        self.current_state["last_rotation"] = datetime.now(timezone.utc).isoformat()
        self.current_state["rotation_count"] = self.current_state.get("rotation_count", 0) + 1
        self._save_state()

        print(f"\n✅ Rotated to market #{self.current_state['rotation_count']}")
        print(f"   {next_market['question']}")
        print("="*80 + "\n")

        return next_market

# ==============================================================================
# CLI
# ==============================================================================

def cmd_discover():
    """Discover and display active markets."""
    print("="*80)
    print(" BTC 15-MIN MARKET DISCOVERY")
    print("="*80)
    print()

    discovery = BTCMarketDiscovery()
    markets = discovery.discover_active_markets()

    if markets:
        print(f"\n📊 Found {len(markets)} markets:\n")
        for i, m in enumerate(markets, 1):
            print(f"{i}. {m['question']}")
            print(f"   Slug: {m['slug']}")
            print(f"   Condition ID: {m['condition_id']}")
            print(f"   End time: {m['end_time_utc']}")
            print()
    else:
        print("\n❌ No active 15-min BTC markets found")
        print("\nPossible reasons:")
        print("  - Markets may be closed/inactive at this time")
        print("  - Markets may only be available during specific hours")
        print("  - API may have changed")

def cmd_next():
    """Get next market to stream."""
    manager = MarketRotationManager()
    current = manager.get_current_market()
    next_market = manager.get_next_market()

    print("\n" + "="*80)
    print(" MARKET STATUS")
    print("="*80)

    if current:
        print(f"\n✅ CURRENT MARKET:")
        print(f"   {current['question']}")
        print(f"   Condition ID: {current['condition_id']}")
        print(f"   Ends: {current['end_time_utc']}")

    if next_market:
        print(f"\n📅 NEXT MARKET:")
        print(f"   {next_market['question']}")
        print(f"   Condition ID: {next_market['condition_id']}")
        print(f"   Starts: {next_market['end_time_utc']}")

def cmd_daemon():
    """Run continuous market discovery and rotation monitoring."""
    print("="*80)
    print(" BTC 15-MIN MARKET MANAGER (DAEMON MODE)")
    print("="*80)
    print(f"\nCheck interval: {CHECK_INTERVAL_SECONDS}s")
    print(f"Market buffer: {MARKET_BUFFER_MINUTES} min")
    print("\nPress Ctrl+C to stop\n")

    manager = MarketRotationManager()

    try:
        while True:
            print(f"\n[{datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S')} UTC]")

            # Check if rotation needed
            if manager.should_rotate():
                manager.rotate_to_next()
            else:
                current = manager.current_state.get("current_market")
                if current:
                    end_time = datetime.fromisoformat(current['end_time_utc'])
                    remaining = end_time - datetime.now(timezone.utc)
                    print(f"✅ Current market: {current['question'][:60]}...")
                    print(f"   Time remaining: {int(remaining.total_seconds() / 60)} minutes")
                else:
                    print("⚠️  No current market")

            # Sleep
            time.sleep(CHECK_INTERVAL_SECONDS)

    except KeyboardInterrupt:
        print("\n\n🛑 Stopped by user")

def main():
    import sys

    if len(sys.argv) < 2:
        print("Usage:")
        print("  uv run python scripts/btc_15min_market_manager.py discover  # Discover active markets")
        print("  uv run python scripts/btc_15min_market_manager.py next      # Get next market info")
        print("  uv run python scripts/btc_15min_market_manager.py daemon    # Run continuous monitoring")
        sys.exit(1)

    command = sys.argv[1]

    if command == "discover":
        cmd_discover()
    elif command == "next":
        cmd_next()
    elif command == "daemon":
        cmd_daemon()
    else:
        print(f"Unknown command: {command}")
        sys.exit(1)

if __name__ == '__main__':
    main()
