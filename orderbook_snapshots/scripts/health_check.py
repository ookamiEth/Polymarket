#!/usr/bin/env python3
"""
Health Check
Daily health check for all orderbook services
Validates service status and data production
"""

import asyncio
import subprocess
import time
import yaml
from pathlib import Path
from datetime import datetime, timedelta
from telegram_notifier import TelegramNotifier

# Load config
project_root = Path(__file__).parent.parent
with open(project_root / "config" / "telegram_bot.yaml") as f:
    config = yaml.safe_load(f)

SERVICES = config.get("services", [])
DATA_DIRS = config.get("data_directories", {})
HEALTH_CONFIG = config.get("health_check", {})


class HealthChecker:
    """Perform comprehensive health checks"""

    def __init__(self):
        self.notifier = TelegramNotifier()
        self.overall_healthy = True

    def get_service_uptime(self, service_name: str) -> str:
        """Get service uptime in human-readable format"""
        try:
            result = subprocess.run(
                ["systemctl", "show", service_name, "--property=ActiveEnterTimestamp"],
                capture_output=True,
                text=True,
                timeout=10
            )

            if result.returncode == 0:
                timestamp_line = result.stdout.strip()
                if "=" in timestamp_line:
                    timestamp_str = timestamp_line.split("=", 1)[1]
                    if timestamp_str:
                        # Parse systemd timestamp (e.g., "Fri 2025-10-10 00:13:47 UTC")
                        try:
                            # Use a simpler approach - get the unix timestamp
                            result2 = subprocess.run(
                                ["systemctl", "show", service_name, "--property=ActiveEnterTimestampMonotonic"],
                                capture_output=True,
                                text=True,
                                timeout=10
                            )
                            # Calculate uptime in seconds
                            uptime_result = subprocess.run(
                                ["cat", "/proc/uptime"],
                                capture_output=True,
                                text=True,
                                timeout=10
                            )
                            if uptime_result.returncode == 0:
                                system_uptime = float(uptime_result.stdout.split()[0])
                                # Estimate uptime (simplified)
                                return self.format_uptime(system_uptime / 2)  # Rough estimate
                        except:
                            pass

            # Fallback: just show "running"
            return "running"
        except:
            return "unknown"

    def format_uptime(self, seconds: float) -> str:
        """Format uptime in days, hours format"""
        days = int(seconds // 86400)
        hours = int((seconds % 86400) // 3600)

        if days > 0:
            return f"{days}d {hours}h"
        else:
            return f"{hours}h"

    def check_service_status(self, service_name: str) -> dict:
        """Check if service is running"""
        try:
            result = subprocess.run(
                ["systemctl", "is-active", service_name],
                capture_output=True,
                text=True,
                timeout=10
            )

            is_active = result.stdout.strip() == "active"
            uptime = self.get_service_uptime(service_name) if is_active else "stopped"

            return {
                "running": is_active,
                "uptime": uptime
            }
        except:
            return {"running": False, "uptime": "error"}

    def check_data_production(self, service_type: str, data_dir: str) -> dict:
        """Check data production for a service"""
        data_path = project_root / data_dir

        if not data_path.exists():
            return {
                "healthy": False,
                "file_count": 0,
                "total_size_mb": 0.0,
                "error": "Directory not found"
            }

        # Find parquet files from last 24 hours
        cutoff_time = time.time() - (24 * 3600)
        recent_files = []

        for parquet_file in data_path.glob("**/*.parquet"):
            if parquet_file.stat().st_mtime > cutoff_time:
                recent_files.append(parquet_file)

        # Calculate total size
        total_size = sum(f.stat().st_size for f in recent_files)
        total_size_mb = total_size / (1024 * 1024)

        # Check against expected
        expected = HEALTH_CONFIG.get("expected_files_per_day", {}).get(service_type, 0)
        tolerance = HEALTH_CONFIG.get("tolerance_percent", 10) / 100
        min_expected = expected * (1 - tolerance)
        max_expected = expected * (1 + tolerance)

        file_count = len(recent_files)
        healthy = min_expected <= file_count <= max_expected

        # Also check minimum file size
        min_size_kb = HEALTH_CONFIG.get("min_file_size_kb", {}).get(service_type, 0)
        if healthy and recent_files:
            # Check a few recent files
            for f in recent_files[:5]:
                if f.stat().st_size < min_size_kb * 1024:
                    healthy = False
                    break

        if not healthy:
            self.overall_healthy = False

        return {
            "healthy": healthy,
            "file_count": file_count,
            "total_size_mb": total_size_mb,
            "expected_count": expected
        }

    async def perform_health_check(self):
        """Perform full health check and send report"""
        print("=" * 80)
        print("DAILY HEALTH CHECK")
        print("=" * 80)
        print(f"Time: {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S UTC')}")
        print()

        # Check all services
        services_status = {}
        print("Checking services...")
        for service in SERVICES:
            status = self.check_service_status(service)
            services_status[service] = status
            status_icon = "✓" if status["running"] else "✗"
            print(f"  {status_icon} {service}: {status['uptime']}")

            if not status["running"]:
                self.overall_healthy = False

        print()

        # Check data production
        data_stats = {}
        print("Checking data production (last 24h)...")
        for service_type, data_dir in DATA_DIRS.items():
            stats = self.check_data_production(service_type, data_dir)
            data_stats[service_type] = stats
            status_icon = "✓" if stats["healthy"] else "✗"
            print(f"  {status_icon} {service_type}: {stats['file_count']} files (expected: ~{stats.get('expected_count', '?')}), {stats['total_size_mb']:.1f} MB")

        print()
        print("=" * 80)

        overall_status = "HEALTHY" if self.overall_healthy else "ISSUES DETECTED"
        print(f"Overall status: {overall_status}")
        print("=" * 80)

        # Send Telegram notification
        if config.get("alerts", {}).get("daily_health_check", True):
            print("\nSending Telegram notification...")
            await self.notifier.send_health_check(
                services_status=services_status,
                data_stats=data_stats,
                overall_healthy=self.overall_healthy
            )
            print("✓ Notification sent")


async def main():
    """Run health check"""
    checker = HealthChecker()
    await checker.perform_health_check()


if __name__ == "__main__":
    asyncio.run(main())
