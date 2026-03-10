"""
notifications.py - Discord webhook notifications for KARL.

Sends analysis reports, training results, and alerts to Discord.
"""

import json
import os
import urllib.request
from pathlib import Path
from typing import Optional

from karl.config import DISCORD_WEBHOOK


def _get_webhook_url() -> str:
    """Resolve Discord webhook URL from config or environment."""
    if DISCORD_WEBHOOK:
        return DISCORD_WEBHOOK

    # Fallback: check webhooks.env file
    webhook_file = Path.home() / "flows" / "feed-hub" / "webhooks.env"
    if webhook_file.exists():
        with open(webhook_file) as f:
            for line in f:
                if line.startswith("DISCORD_WEBHOOK_SERVICE_HEALTH="):
                    return line.split("=", 1)[1].strip().strip('"')
    return ""


def post_discord(message: str) -> bool:
    """Post a message to the configured Discord webhook.

    Args:
        message: Message content (Markdown supported)

    Returns:
        True if message was sent successfully
    """
    webhook = _get_webhook_url()
    if not webhook:
        return False

    try:
        data = json.dumps({"content": message}).encode()
        req = urllib.request.Request(
            webhook,
            data=data,
            headers={"Content-Type": "application/json"},
        )
        urllib.request.urlopen(req, timeout=5)
        return True
    except Exception:
        return False
