#!/usr/bin/env python3
"""
karl_training_flow.py — Prefect flow for weekly KARL LoRA training.

Runs weekly (Sunday 3am):
  1. Export trajectories to advantage-weighted SFT
  2. Upload to Mac5
  3. Trigger MLX LoRA fine-tune
  4. Report results to Discord

Deploy: prefect deployment build karl_training_flow.py:karl_training -n karl-weekly -q default
"""

import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

# Prefect imports (optional — runs standalone too)
try:
    from prefect import flow, task, get_run_logger
    from prefect.deployments import Deployment
    HAS_PREFECT = True
except ImportError:
    HAS_PREFECT = False
    # Stub decorators for standalone execution
    def flow(fn=None, **kw):
        return fn if fn else lambda f: f
    def task(fn=None, **kw):
        return fn if fn else lambda f: f

KARL_DIR = Path(__file__).parent
sys.path.insert(0, str(KARL_DIR))

DISCORD_WEBHOOK = os.environ.get(
    "DISCORD_WEBHOOK_KARL",
    os.environ.get("DISCORD_WEBHOOK_SERVICE_HEALTH", ""),
)

WEBHOOK_FILE = Path.home() / "flows" / "feed-hub" / "webhooks.env"


def _get_discord_webhook() -> str:
    """Load Discord webhook from env or file."""
    if DISCORD_WEBHOOK:
        return DISCORD_WEBHOOK
    if WEBHOOK_FILE.exists():
        with open(WEBHOOK_FILE) as f:
            for line in f:
                if line.startswith("DISCORD_WEBHOOK_SERVICE_HEALTH="):
                    return line.split("=", 1)[1].strip().strip('"')
    return ""


def _post_discord(message: str) -> None:
    """Post message to Discord webhook."""
    webhook = _get_discord_webhook()
    if not webhook:
        return
    try:
        import urllib.request
        data = json.dumps({"content": message}).encode()
        req = urllib.request.Request(
            webhook,
            data=data,
            headers={"Content-Type": "application/json"},
        )
        urllib.request.urlopen(req, timeout=5)
    except Exception:
        pass


@task(name="export-karl-sft")
def export_sft_task() -> dict:
    """Export trajectories to SFT format."""
    from sft_exporter import export_sft
    return export_sft()


@task(name="upload-to-mac5")
def upload_task() -> dict:
    """Upload training data to Mac5."""
    from karl_trainer import upload_to_mac5
    return upload_to_mac5()


@task(name="trigger-training")
def trigger_task() -> dict:
    """Trigger training on Mac5."""
    from karl_trainer import trigger_training
    return trigger_training()


@task(name="report-results")
def report_task(results: dict) -> None:
    """Report training results to Discord."""
    export = results.get("steps", {}).get("export", {})
    status = results.get("status", "unknown")

    msg = (
        f"**KARL Training Report** ({datetime.now(timezone.utc).strftime('%Y-%m-%d')})\n"
        f"Status: `{status}`\n"
        f"Trajectories: {export.get('total_records', 0)} "
        f"({export.get('examples', 0)} SFT examples, "
        f"{export.get('oversampled', 0)} oversampled)\n"
    )

    trigger = results.get("steps", {}).get("trigger", {})
    if trigger:
        msg += f"Training: {trigger.get('method', '?')}, exit={trigger.get('exit_code', '?')}\n"

    monitor = results.get("steps", {}).get("monitor", {})
    if monitor and monitor.get("complete"):
        msg += f"Loss: {monitor.get('loss')}, Adapter: v{monitor.get('adapter_version')}"

    _post_discord(msg)


@flow(name="karl-weekly-training", log_prints=True)
def karl_training() -> dict:
    """Weekly KARL training pipeline."""
    results = {"steps": {}, "started_at": datetime.now(timezone.utc).isoformat()}

    # Step 1: Export
    print("[karl] Exporting trajectories...")
    export_stats = export_sft_task()
    results["steps"]["export"] = export_stats

    if export_stats.get("examples", 0) < 10:
        results["status"] = "insufficient_data"
        print(f"[karl] Only {export_stats.get('examples', 0)} examples — skipping training (need 10+)")
        report_task(results)
        return results

    # Step 2: Upload
    print("[karl] Uploading to Mac5...")
    upload_result = upload_task()
    results["steps"]["upload"] = upload_result

    if not upload_result.get("success"):
        results["status"] = "upload_failed"
        print(f"[karl] Upload failed: {upload_result.get('error')}")
        report_task(results)
        return results

    # Step 3: Trigger
    print("[karl] Triggering training...")
    trigger_result = trigger_task()
    results["steps"]["trigger"] = trigger_result

    if trigger_result.get("exit_code", 1) != 0:
        results["status"] = "training_failed"
        print(f"[karl] Training failed: {trigger_result.get('output', '')[:200]}")
    else:
        results["status"] = "complete"

    # Step 4: Report
    report_task(results)
    results["completed_at"] = datetime.now(timezone.utc).isoformat()

    return results


if __name__ == "__main__":
    result = karl_training()
    print(json.dumps(result, indent=2, default=str))
