#!/usr/bin/env python3
"""
Production Orderbook Streaming Service

Continuously captures orderbook snapshots from Polymarket BTC 15-minute markets.
Automatically switches markets every 15 minutes and stores data in Parquet format.

Features:
- 24/7 continuous operation
- Automatic market discovery and switching
- Graceful shutdown handling (SIGTERM/SIGINT)
- Error recovery with exponential backoff
- Hierarchical data storage by date
- Comprehensive logging and metrics

Usage:
    cd /orderbook_snapshots
    uv run python scripts/stream_continuous.py

    # Or as a service:
    launchctl load ~/Library/LaunchAgents/com.orderbook.streamer.plist
"""

import signal
import sys
import time
import json
import yaml
import logging
import requests
import polars as pl
from pathlib import Path
from datetime import datetime, timezone
from logging.handlers import RotatingFileHandler
from collections import Counter, deque
from typing import Optional, Dict, Any

# ============================================================================
# Configuration Loader
# ============================================================================

class ConfigLoader:
    """Load and validate configuration from YAML file"""

    def __init__(self, config_path: str = "config/streamer_config.yaml"):
        self.config_path = Path(config_path)
        self.config = self._load_config()

    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from YAML file"""
        if not self.config_path.exists():
            raise FileNotFoundError(f"Config file not found: {self.config_path}")

        with open(self.config_path) as f:
            config = yaml.safe_load(f)

        # Validate required sections
        required = ['service', 'polling', 'schedule', 'storage', 'logging', 'apis']
        for section in required:
            if section not in config:
                raise ValueError(f"Missing required config section: {section}")

        return config

    def get(self, *keys, default=None):
        """Get nested config value with dot notation"""
        value = self.config
        for key in keys:
            if isinstance(value, dict) and key in value:
                value = value[key]
            else:
                return default
        return value


# ============================================================================
# Logger Setup
# ============================================================================

class StreamerLogger:
    """Setup rotating file logger with structured format"""

    def __init__(self, config: ConfigLoader):
        self.config = config
        self.logger = self._setup_logger()

    def _setup_logger(self) -> logging.Logger:
        """Configure logger with file and console handlers"""
        logger = logging.getLogger('orderbook_streamer')

        # Set level from config
        level_str = self.config.get('logging', 'level', default='INFO')
        level = getattr(logging, level_str)
        logger.setLevel(level)

        # Create logs directory
        log_file = Path(self.config.get('logging', 'file', default='logs/streamer.log'))
        log_file.parent.mkdir(parents=True, exist_ok=True)

        # Rotating file handler
        max_bytes = self.config.get('logging', 'max_bytes', default=104857600)  # 100MB
        backup_count = self.config.get('logging', 'backup_count', default=5)

        file_handler = RotatingFileHandler(
            log_file,
            maxBytes=max_bytes,
            backupCount=backup_count
        )

        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)

        # Format
        fmt = self.config.get('logging', 'format',
                             default='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        formatter = logging.Formatter(fmt)

        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)

        logger.addHandler(file_handler)
        logger.addHandler(console_handler)

        return logger

    def info(self, msg: str):
        self.logger.info(msg)

    def warning(self, msg: str):
        self.logger.warning(msg)

    def error(self, msg: str):
        self.logger.error(msg)

    def critical(self, msg: str):
        self.logger.critical(msg)

    def debug(self, msg: str):
        self.logger.debug(msg)


# ============================================================================
# Market Scheduler
# ============================================================================

class MarketScheduler:
    """Manage market schedule and track current period"""

    def __init__(self, config: ConfigLoader, logger: StreamerLogger):
        self.config = config
        self.logger = logger
        self.schedule = None
        self.current_market = None
        self.next_market = None
        self._load_schedule()

    def _load_schedule(self):
        """Load market schedule from Parquet file"""
        schedule_file = Path(self.config.get('schedule', 'file'))

        if not schedule_file.exists():
            raise FileNotFoundError(f"Schedule file not found: {schedule_file}")

        self.logger.info(f"Loading schedule from {schedule_file}")
        self.schedule = pl.read_parquet(schedule_file)
        self.logger.info(f"Loaded {len(self.schedule):,} market periods")

        # Check if schedule needs refresh
        self._check_schedule_expiration()

    def _check_schedule_expiration(self):
        """Check if schedule is approaching expiration"""
        if self.schedule is None:
            return

        max_ts = self.schedule['end_timestamp'].max()
        current_ts = int(time.time())
        days_remaining = (max_ts - current_ts) / 86400

        threshold = self.config.get('schedule', 'refresh_threshold_days', default=30)

        if days_remaining < threshold:
            self.logger.warning(
                f"Schedule expires in {days_remaining:.1f} days! "
                f"Please regenerate schedule (threshold: {threshold} days)"
            )

    def find_current_market(self) -> Optional[Dict[str, Any]]:
        """Find the current active market period"""
        current_ts = int(time.time())

        # Find current period from schedule
        current_period = self.schedule.filter(
            (pl.col("start_timestamp") <= current_ts) &
            (pl.col("end_timestamp") > current_ts)
        )

        if len(current_period) == 0:
            self.logger.error("No current market period found in schedule")
            return None

        # Extract market info
        slug = current_period['slug'][0]
        start_ts = current_period['start_timestamp'][0]
        end_ts = current_period['end_timestamp'][0]

        self.logger.info(f"Found current market: {slug}")

        # Query Gamma API for token IDs
        gamma_url = self.config.get('apis', 'gamma_url')
        try:
            response = requests.get(
                f"{gamma_url}/markets",
                params={"slug": slug},
                timeout=10
            )
            response.raise_for_status()

            market_data = response.json()
            if not market_data or len(market_data) == 0:
                self.logger.error(f"Market not found in API: {slug}")
                return None

            market = market_data[0]

            # Parse token IDs
            token_ids_str = market.get('clobTokenIds', '')
            token_ids = json.loads(token_ids_str) if token_ids_str else []

            if len(token_ids) < 2:
                self.logger.error(f"Invalid token count: {len(token_ids)}")
                return None

            return {
                'slug': slug,
                'condition_id': market.get('conditionId'),
                'start_timestamp': start_ts,
                'end_timestamp': end_ts,
                'token_ids': token_ids,
                'active': market.get('active', False),
                'closed': market.get('closed', False)
            }

        except Exception as e:
            self.logger.error(f"Error querying Gamma API: {e}")
            return None

    def get_next_market(self) -> Optional[Dict[str, Any]]:
        """Get the next market period"""
        if not self.current_market:
            return None

        next_start_ts = self.current_market['end_timestamp']

        # Find next period
        next_period = self.schedule.filter(
            pl.col("start_timestamp") == next_start_ts
        )

        if len(next_period) == 0:
            self.logger.error("No next market period found")
            return None

        slug = next_period['slug'][0]
        self.logger.info(f"Pre-fetching next market: {slug}")

        # Query API for token IDs
        gamma_url = self.config.get('apis', 'gamma_url')
        try:
            response = requests.get(
                f"{gamma_url}/markets",
                params={"slug": slug},
                timeout=10
            )
            response.raise_for_status()

            market_data = response.json()
            if not market_data:
                return None

            market = market_data[0]
            token_ids = json.loads(market.get('clobTokenIds', '[]'))

            return {
                'slug': slug,
                'condition_id': market.get('conditionId'),
                'start_timestamp': next_period['start_timestamp'][0],
                'end_timestamp': next_period['end_timestamp'][0],
                'token_ids': token_ids,
                'active': market.get('active', False),
                'closed': market.get('closed', False)
            }

        except Exception as e:
            self.logger.error(f"Error pre-fetching next market: {e}")
            return None


# ============================================================================
# Orderbook Poller
# ============================================================================

class OrderbookPoller:
    """Poll orderbook API and handle retries"""

    def __init__(self, config: ConfigLoader, logger: StreamerLogger):
        self.config = config
        self.logger = logger
        self.api_url = config.get('apis', 'clob_url')
        self.timeout = config.get('polling', 'timeout_seconds', default=5)
        self.retry_attempts = config.get('polling', 'retry_attempts', default=3)
        self.backoff_multiplier = config.get('polling', 'backoff_multiplier', default=2)

        # Metrics
        self.latencies = deque(maxlen=100)
        self.error_count = Counter()

    def poll(self, token_id: str) -> Optional[Dict[str, Any]]:
        """Poll orderbook with retry logic"""
        for attempt in range(self.retry_attempts):
            try:
                start_time = time.time()
                capture_time_ms = int(start_time * 1000)

                response = requests.get(
                    f"{self.api_url}/book",
                    params={"token_id": token_id},
                    timeout=self.timeout
                )

                latency_ms = (time.time() - start_time) * 1000
                self.latencies.append(latency_ms)

                if response.status_code == 429:
                    self.logger.warning("Rate limit hit (429), backing off...")
                    time.sleep(10)
                    continue

                response.raise_for_status()

                book = response.json()

                # Parse and flatten
                return self._parse_orderbook(book, capture_time_ms)

            except requests.Timeout:
                self.error_count['timeout'] += 1
                if attempt < self.retry_attempts - 1:
                    sleep_time = self.backoff_multiplier ** attempt
                    self.logger.warning(f"Timeout (attempt {attempt+1}), retrying in {sleep_time}s")
                    time.sleep(sleep_time)
                else:
                    self.logger.error("Max retries exceeded (timeout)")
                    return None

            except requests.RequestException as e:
                self.error_count['request_error'] += 1
                if attempt < self.retry_attempts - 1:
                    sleep_time = self.backoff_multiplier ** attempt
                    self.logger.warning(f"Request error: {e}, retrying in {sleep_time}s")
                    time.sleep(sleep_time)
                else:
                    self.logger.error(f"Max retries exceeded: {e}")
                    return None

            except Exception as e:
                self.error_count['parse_error'] += 1
                self.logger.error(f"Unexpected error: {e}")
                return None

        return None

    def _parse_orderbook(self, book: Dict, capture_time_ms: int) -> Dict[str, Any]:
        """Parse and flatten orderbook data"""
        market = book.get('market', '')
        asset_id = book.get('asset_id', '')
        market_timestamp = int(book.get('timestamp', capture_time_ms))
        bids = book.get('bids', [])
        asks = book.get('asks', [])

        # Extract top 3 levels
        best_3_bids = bids[-3:] if len(bids) >= 3 else bids
        best_3_asks = asks[-3:] if len(asks) >= 3 else asks

        result = {
            'timestamp_ms': capture_time_ms,
            'market_timestamp_ms': market_timestamp,
            'condition_id': market,
            'asset_id': asset_id,
        }

        # Bids (3 → 2 → 1)
        for i in range(3):
            if i < len(best_3_bids):
                bid = best_3_bids[i]
                result[f'bid_price_{3-i}'] = float(bid['price'])
                result[f'bid_size_{3-i}'] = float(bid['size'])
            else:
                result[f'bid_price_{3-i}'] = None
                result[f'bid_size_{3-i}'] = None

        # Calculate spread and mid
        best_bid = float(best_3_bids[-1]['price']) if best_3_bids else 0.0
        best_ask = float(best_3_asks[-1]['price']) if best_3_asks else 0.0

        result['spread'] = round((best_ask - best_bid), 3) if (best_bid > 0 and best_ask > 0) else None
        result['mid_price'] = round(((best_bid + best_ask) / 2), 3) if (best_bid > 0 and best_ask > 0) else None

        # Asks (1 → 2 → 3)
        for i in range(3):
            if i < len(best_3_asks):
                ask = best_3_asks[-(i+1)]
                result[f'ask_price_{i+1}'] = float(ask['price'])
                result[f'ask_size_{i+1}'] = float(ask['size'])
            else:
                result[f'ask_price_{i+1}'] = None
                result[f'ask_size_{i+1}'] = None

        return result

    def get_metrics(self) -> Dict[str, Any]:
        """Get poller metrics"""
        if not self.latencies:
            return {'avg_latency_ms': 0, 'p95_latency_ms': 0}

        sorted_latencies = sorted(self.latencies)
        p95_index = int(len(sorted_latencies) * 0.95)

        return {
            'avg_latency_ms': sum(self.latencies) / len(self.latencies),
            'p95_latency_ms': sorted_latencies[p95_index] if sorted_latencies else 0,
            'error_counts': dict(self.error_count)
        }


# ============================================================================
# Data Writer
# ============================================================================

class DataWriter:
    """Write snapshots to hierarchical Parquet files"""

    def __init__(self, config: ConfigLoader, logger: StreamerLogger):
        self.config = config
        self.logger = logger
        self.base_dir = Path(config.get('storage', 'raw_data_dir', default='data/raw'))
        self.buffer_size = config.get('storage', 'buffer_size', default=1000)
        self.buffer = []

    def add_snapshot(self, snapshot: Dict[str, Any]):
        """Add snapshot to buffer"""
        self.buffer.append(snapshot)

    def should_flush(self) -> bool:
        """Check if buffer should be flushed"""
        return len(self.buffer) >= self.buffer_size

    def flush(self, market: Dict[str, Any]):
        """Write buffer to Parquet file"""
        if not self.buffer:
            self.logger.debug("Buffer empty, nothing to flush")
            return

        try:
            # Create DataFrame
            df = pl.DataFrame(self.buffer)

            # Generate output path (using UTC)
            dt = datetime.fromtimestamp(market['start_timestamp'], tz=timezone.utc)
            year = dt.strftime("%Y")
            month = dt.strftime("%m")
            day = dt.strftime("%d")
            filename = f"orderbook_{dt.strftime('%Y%m%d_%H%M')}.parquet"

            output_dir = self.base_dir / year / month / day
            output_dir.mkdir(parents=True, exist_ok=True)

            output_file = output_dir / filename

            # Atomic write (write to temp, then rename)
            temp_file = output_file.with_suffix('.parquet.tmp')
            df.write_parquet(temp_file, compression='snappy')
            temp_file.rename(output_file)

            file_size_kb = output_file.stat().st_size / 1024

            self.logger.info(
                f"Wrote {len(df)} snapshots to {output_file} ({file_size_kb:.1f}KB)"
            )

            # Clear buffer
            self.buffer = []

        except Exception as e:
            self.logger.error(f"Error writing Parquet: {e}")
            raise


# ============================================================================
# Main Orderbook Streamer
# ============================================================================

class OrderbookStreamer:
    """Main streaming service orchestrator"""

    def __init__(self):
        # Load config
        self.config = ConfigLoader()

        # Setup logger
        self.logger = StreamerLogger(self.config)
        self.logger.info("=" * 80)
        self.logger.info("ORDERBOOK STREAMING SERVICE STARTING")
        self.logger.info("=" * 80)

        # Initialize components
        self.scheduler = MarketScheduler(self.config, self.logger)
        self.poller = OrderbookPoller(self.config, self.logger)
        self.writer = DataWriter(self.config, self.logger)

        # State
        self.shutdown_flag = False
        self.current_market = None
        self.next_market = None
        self.next_market_prefetched = False

        # Metrics
        self.snapshots_collected = 0
        self.start_time = time.time()

        # Polling config
        self.poll_interval = self.config.get('polling', 'interval_seconds', default=1.0)
        self.preload_seconds = self.config.get('schedule', 'preload_next_market_seconds', default=60)

    def setup_signal_handlers(self):
        """Setup graceful shutdown handlers"""
        def signal_handler(signum, frame):
            sig_name = signal.Signals(signum).name
            self.logger.info(f"Received signal {sig_name}, initiating graceful shutdown...")
            self.shutdown_flag = True

        signal.signal(signal.SIGTERM, signal_handler)
        signal.signal(signal.SIGINT, signal_handler)

        self.logger.info("Signal handlers registered")

    def initialize_market(self):
        """Find and initialize current market"""
        self.logger.info("Initializing current market...")
        self.current_market = self.scheduler.find_current_market()

        if not self.current_market:
            raise RuntimeError("Failed to find current market")

        time_remaining = self.current_market['end_timestamp'] - int(time.time())
        self.logger.info(
            f"Initialized market: {self.current_market['slug']} "
            f"({time_remaining//60}m {time_remaining%60}s remaining)"
        )

    def check_market_transition(self):
        """Check if market transition is needed"""
        current_time = int(time.time())
        time_until_end = self.current_market['end_timestamp'] - current_time

        # Pre-fetch next market
        if time_until_end <= self.preload_seconds and not self.next_market_prefetched:
            self.logger.info(f"Pre-fetching next market ({time_until_end}s remaining)...")
            self.next_market = self.scheduler.get_next_market()

            if self.next_market:
                self.next_market_prefetched = True
                self.logger.info(f"Next market ready: {self.next_market['slug']}")
            else:
                self.logger.warning("Failed to pre-fetch next market, will retry")

        # Transition at boundary
        if current_time >= self.current_market['end_timestamp']:
            self.logger.info("Market transition triggered!")

            # Flush current market data
            self.logger.info("Flushing buffer for current market...")
            self.writer.flush(self.current_market)

            # Switch to next market
            if self.next_market:
                self.current_market = self.next_market
                self.next_market = None
                self.next_market_prefetched = False

                self.logger.info(f"Switched to market: {self.current_market['slug']}")
            else:
                # Fallback: find current market again
                self.logger.warning("Next market not available, finding current...")
                self.initialize_market()

    def run(self):
        """Main event loop"""
        try:
            # Setup
            self.setup_signal_handlers()
            self.initialize_market()

            # Get token selection
            token_selection = self.config.get('market', 'token_selection', default='UP')
            token_index = 0 if token_selection == 'UP' else 1

            self.logger.info(f"Starting polling (tracking {token_selection} token)")
            self.logger.info("Press Ctrl+C to stop gracefully")
            self.logger.info("=" * 80)

            last_status_log = time.time()
            last_transition_check = time.time()

            while not self.shutdown_flag:
                loop_start = time.time()

                # Check for market transition every 10 seconds
                if time.time() - last_transition_check >= 10:
                    self.check_market_transition()
                    last_transition_check = time.time()

                # Poll orderbook
                token_id = self.current_market['token_ids'][token_index]

                # DEBUG: Log first few polls
                if self.snapshots_collected < 3:
                    self.logger.info(f"Polling token_id: {token_id}")

                snapshot = self.poller.poll(token_id)

                if snapshot:
                    self.writer.add_snapshot(snapshot)
                    self.snapshots_collected += 1

                    # DEBUG: Log first few snapshots
                    if self.snapshots_collected <= 3:
                        self.logger.info(f"Snapshot #{self.snapshots_collected} collected")

                    # Periodic flush
                    if self.writer.should_flush():
                        self.writer.flush(self.current_market)
                else:
                    # DEBUG: Log if poll failed
                    if self.snapshots_collected < 3:
                        self.logger.warning("Poll returned None")

                # Status log every 5 minutes
                if time.time() - last_status_log >= 300:
                    metrics = self.poller.get_metrics()
                    uptime = time.time() - self.start_time

                    self.logger.info(
                        f"Status: {self.snapshots_collected} snapshots collected, "
                        f"uptime: {uptime/3600:.1f}h, "
                        f"avg latency: {metrics.get('avg_latency_ms', 0):.0f}ms"
                    )
                    last_status_log = time.time()

                # Sleep to maintain interval
                loop_elapsed = time.time() - loop_start
                sleep_time = max(0, self.poll_interval - loop_elapsed)
                if sleep_time > 0:
                    time.sleep(sleep_time)

            # Graceful shutdown
            self.logger.info("=" * 80)
            self.logger.info("GRACEFUL SHUTDOWN INITIATED")
            self.logger.info("=" * 80)

            self.logger.info("Flushing final buffer...")
            self.writer.flush(self.current_market)

            uptime = time.time() - self.start_time
            self.logger.info(f"Total snapshots collected: {self.snapshots_collected}")
            self.logger.info(f"Total uptime: {uptime/3600:.2f} hours")
            self.logger.info("Shutdown complete")

        except Exception as e:
            self.logger.critical(f"Fatal error: {e}")
            import traceback
            self.logger.critical(traceback.format_exc())
            sys.exit(1)


# ============================================================================
# Main Entry Point
# ============================================================================

def main():
    """Main entry point"""
    streamer = OrderbookStreamer()
    streamer.run()

if __name__ == "__main__":
    main()
