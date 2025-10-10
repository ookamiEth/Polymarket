#!/usr/bin/env python3
"""
Service Monitor
Monitors systemd services and sends alerts when they crash or restart
"""

import asyncio
import subprocess
import time
import yaml
from pathlib import Path
from datetime import datetime
from collections import defaultdict
from telegram_notifier import TelegramNotifier

# Load config
project_root = Path(__file__).parent.parent
with open(project_root / "config" / "telegram_bot.yaml") as f:
    config = yaml.safe_load(f)

SERVICES = config.get("services", [])
CHECK_INTERVAL = config.get("monitor", {}).get("check_interval", 30)
DEBOUNCE_TIME = config.get("monitor", {}).get("debounce_time", 60)


class ServiceMonitor:
    """Monitor systemd services for crashes and restarts"""

    def __init__(self):
        self.notifier = TelegramNotifier()
        self.service_states = {}
        self.restart_counts = defaultdict(int)
        self.last_alert_time = defaultdict(float)

    def get_service_status(self, service_name: str) -> dict:
        """Get current status of a systemd service"""
        try:
            # Get service active state
            result = subprocess.run(
                ["systemctl", "show", service_name, "--property=ActiveState,SubState,NRestarts"],
                capture_output=True,
                text=True,
                timeout=10
            )

            if result.returncode != 0:
                return {"error": True, "message": result.stderr}

            # Parse output
            props = {}
            for line in result.stdout.strip().split("\n"):
                if "=" in line:
                    key, value = line.split("=", 1)
                    props[key] = value

            # Get last log entry for additional context
            log_result = subprocess.run(
                ["journalctl", "-u", service_name, "-n", "1", "--no-pager", "-o", "cat"],
                capture_output=True,
                text=True,
                timeout=10
            )

            return {
                "active_state": props.get("ActiveState", "unknown"),
                "sub_state": props.get("SubState", "unknown"),
                "restart_count": int(props.get("NRestarts", "0")),
                "last_log": log_result.stdout.strip() if log_result.returncode == 0 else "",
                "error": False
            }
        except Exception as e:
            return {"error": True, "message": str(e)}

    def should_send_alert(self, service_name: str) -> bool:
        """Check if enough time has passed since last alert (debounce)"""
        now = time.time()
        last_alert = self.last_alert_time.get(service_name, 0)
        if now - last_alert > DEBOUNCE_TIME:
            self.last_alert_time[service_name] = now
            return True
        return False

    async def check_service(self, service_name: str):
        """Check a single service and send alerts if needed"""
        status = self.get_service_status(service_name)

        if status.get("error"):
            print(f"Error checking {service_name}: {status.get('message')}")
            return

        active_state = status["active_state"]
        sub_state = status["sub_state"]
        restart_count = status["restart_count"]

        # Initialize service state if first time
        if service_name not in self.service_states:
            self.service_states[service_name] = {
                "active_state": active_state,
                "sub_state": sub_state,
                "restart_count": restart_count
            }
            print(f"Initialized monitoring for {service_name}: {active_state}/{sub_state}")
            return

        prev_state = self.service_states[service_name]

        # Detect crash (active -> failed)
        if active_state == "failed" and prev_state["active_state"] != "failed":
            if self.should_send_alert(service_name) and config.get("alerts", {}).get("crash_alerts", True):
                timestamp = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")
                status_info = f"\n<b>Last log:</b> {status['last_log'][:200]}" if status["last_log"] else ""
                await self.notifier.send_crash_alert(service_name, timestamp, status_info)
                print(f"[{timestamp}] CRASH ALERT sent for {service_name}")

        # Detect restart (restart count increased)
        if restart_count > prev_state["restart_count"]:
            if self.should_send_alert(service_name) and config.get("alerts", {}).get("restart_alerts", True):
                timestamp = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")
                await self.notifier.send_restart_alert(service_name, timestamp, restart_count)
                print(f"[{timestamp}] RESTART ALERT sent for {service_name} (count: {restart_count})")

        # Update state
        self.service_states[service_name] = {
            "active_state": active_state,
            "sub_state": sub_state,
            "restart_count": restart_count
        }

    async def monitor_loop(self):
        """Main monitoring loop"""
        print("=" * 80)
        print("ORDERBOOK SERVICE MONITOR STARTED")
        print("=" * 80)
        print(f"Monitoring {len(SERVICES)} services:")
        for service in SERVICES:
            print(f"  - {service}")
        print(f"\nCheck interval: {CHECK_INTERVAL}s")
        print(f"Debounce time: {DEBOUNCE_TIME}s")
        print(f"Time: {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S UTC')}")
        print("=" * 80)
        print()

        while True:
            try:
                # Check all services
                for service in SERVICES:
                    await self.check_service(service)

                # Wait before next check
                await asyncio.sleep(CHECK_INTERVAL)

            except KeyboardInterrupt:
                print("\n\nMonitor stopped by user")
                break
            except Exception as e:
                print(f"Error in monitor loop: {e}")
                await asyncio.sleep(CHECK_INTERVAL)


async def main():
    """Start the service monitor"""
    monitor = ServiceMonitor()
    await monitor.monitor_loop()


if __name__ == "__main__":
    asyncio.run(main())
