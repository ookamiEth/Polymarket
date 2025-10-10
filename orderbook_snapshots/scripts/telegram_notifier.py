#!/usr/bin/env python3
"""
Telegram Notifier
Sends notifications to Telegram for service monitoring
"""

import os
import sys
import asyncio
from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv
from telegram import Bot
from telegram.error import TelegramError
import yaml

# Load environment variables
project_root = Path(__file__).parent.parent
load_dotenv(project_root / ".env")

# Load config
with open(project_root / "config" / "telegram_bot.yaml") as f:
    config = yaml.safe_load(f)

BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")

if not BOT_TOKEN:
    raise ValueError("TELEGRAM_BOT_TOKEN not found in .env file")


class TelegramNotifier:
    """Send formatted notifications via Telegram"""

    def __init__(self):
        self.bot = Bot(token=BOT_TOKEN)
        self.chat_id = CHAT_ID
        self.emoji = config.get("alerts", {}).get("emoji", {})

    async def send_message(self, message: str, parse_mode="HTML"):
        """Send a message to Telegram"""
        if not self.chat_id:
            print("ERROR: TELEGRAM_CHAT_ID not set in .env file")
            print("Run this script with --get-chat-id to get your chat ID")
            return False

        try:
            await self.bot.send_message(
                chat_id=self.chat_id,
                text=message,
                parse_mode=parse_mode
            )
            return True
        except TelegramError as e:
            print(f"Failed to send Telegram message: {e}")
            return False

    async def send_crash_alert(self, service_name: str, timestamp: str, status_info: str = ""):
        """Send service crash alert"""
        emoji_error = self.emoji.get("error", "🚨")
        message = f"""{emoji_error} <b>SERVICE CRASHED</b>

<b>Service:</b> {service_name}
<b>Time:</b> {timestamp}
<b>Status:</b> Failed
{status_info}
<b>Action:</b> Will auto-restart in 10s"""

        await self.send_message(message)

    async def send_restart_alert(self, service_name: str, timestamp: str, restart_count: int = 0):
        """Send service restart alert"""
        emoji_restart = self.emoji.get("restart", "🔄")
        message = f"""{emoji_restart} <b>SERVICE RESTARTED</b>

<b>Service:</b> {service_name}
<b>Time:</b> {timestamp}
<b>Restart count:</b> {restart_count}
<b>Status:</b> Running"""

        await self.send_message(message)

    async def send_health_check(self, services_status: dict, data_stats: dict, overall_healthy: bool = True):
        """Send daily health check report"""
        emoji_success = self.emoji.get("success", "✅")
        emoji_warning = self.emoji.get("warning", "⚠️")
        emoji_data = self.emoji.get("data", "📊")

        now = datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")

        # Build services section
        services_lines = []
        for service, info in services_status.items():
            status_icon = emoji_success if info["running"] else emoji_warning
            services_lines.append(f"{status_icon} {service} ({info['uptime']})")

        # Build data section
        data_lines = []
        for service_type, stats in data_stats.items():
            status_icon = emoji_success if stats["healthy"] else emoji_warning
            data_lines.append(
                f"{status_icon} {service_type}: {stats['file_count']} files, {stats['total_size_mb']:.1f} MB"
            )

        overall_icon = emoji_success if overall_healthy else emoji_warning
        status_text = "All systems operational! 🎉" if overall_healthy else "Some issues detected"

        message = f"""{overall_icon} <b>DAILY HEALTH CHECK</b> - {now}

<b>Services Status:</b>
{chr(10).join(services_lines)}

<b>{emoji_data} Data Production (Last 24h):</b>
{chr(10).join(data_lines)}

{status_text}"""

        await self.send_message(message)

    async def send_test_message(self):
        """Send a test message"""
        emoji_info = self.emoji.get("info", "ℹ️")
        message = f"""{emoji_info} <b>TEST MESSAGE</b>

Telegram bot monitoring is configured and working!

<b>Time:</b> {datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")}
<b>Bot:</b> @ookami_polymarket_devops_bot
<b>Status:</b> Connected ✅"""

        await self.send_message(message)

    async def get_chat_id_from_updates(self):
        """Get chat ID from recent updates"""
        try:
            updates = await self.bot.get_updates()
            if updates:
                for update in updates:
                    if update.message:
                        chat_id = update.message.chat_id
                        username = update.message.from_user.username or "Unknown"
                        print(f"Found chat ID: {chat_id}")
                        print(f"Username: @{username}")
                        print(f"\nAdd this to your .env file:")
                        print(f"TELEGRAM_CHAT_ID={chat_id}")
                        return chat_id
                print("No messages found. Send /start to the bot first.")
            else:
                print("No updates available. Send /start to the bot first.")
        except TelegramError as e:
            print(f"Error getting updates: {e}")
        return None


async def main():
    """CLI interface for telegram notifier"""
    import argparse

    parser = argparse.ArgumentParser(description="Telegram Notifier CLI")
    parser.add_argument("--get-chat-id", action="store_true", help="Get your Telegram chat ID")
    parser.add_argument("--test", action="store_true", help="Send a test message")
    args = parser.parse_args()

    notifier = TelegramNotifier()

    if args.get_chat_id:
        print("Getting chat ID from recent messages...")
        print("Make sure you've sent /start to @ookami_polymarket_devops_bot first")
        await notifier.get_chat_id_from_updates()
    elif args.test:
        print("Sending test message...")
        success = await notifier.send_test_message()
        if success:
            print("✓ Test message sent successfully!")
        else:
            print("✗ Failed to send test message")
            sys.exit(1)
    else:
        parser.print_help()


if __name__ == "__main__":
    asyncio.run(main())
