#!/usr/bin/env python3
"""
karl_analysis_flow.py — Prefect flow for daily KARL analysis.

Runs daily at 6:30am:
  1. Update skill weights from trajectory rewards
  2. Analyze shadow routing data
  3. Check vector routing promotion readiness
  4. Report to Discord

Deploy: prefect deployment build karl_analysis_flow.py:karl_analysis -n karl-daily -q default
"""

import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

try:
    from prefect import flow, task
    HAS_PREFECT = True
except ImportError:
    HAS_PREFECT = False
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
    if DISCORD_WEBHOOK:
        return DISCORD_WEBHOOK
    if WEBHOOK_FILE.exists():
        with open(WEBHOOK_FILE) as f:
            for line in f:
                if line.startswith("DISCORD_WEBHOOK_SERVICE_HEALTH="):
                    return line.split("=", 1)[1].strip().strip('"')
    return ""


def _post_discord(message: str) -> None:
    webhook = _get_discord_webhook()
    if not webhook:
        return
    try:
        import urllib.request
        data = json.dumps({"content": message}).encode()
        req = urllib.request.Request(
            webhook, data=data,
            headers={"Content-Type": "application/json"},
        )
        urllib.request.urlopen(req, timeout=5)
    except Exception:
        pass


@task(name="update-skill-weights")
def update_weights_task() -> dict:
    from weight_updater import update_weights
    return update_weights()


@task(name="analyze-shadow-routing")
def shadow_analysis_task() -> dict:
    from trajectory_bridge import analyze_shadow_routing
    return analyze_shadow_routing()


@task(name="check-promotion")
def promotion_task() -> dict:
    from trajectory_bridge import check_promotion_readiness
    return check_promotion_readiness()


@task(name="skill-health")
def health_task() -> dict:
    from trajectory_bridge import analyze_skill_health
    return analyze_skill_health()


@task(name="persist-karl-status")
def persist_status_task(weights: dict, shadow: dict, promotion: dict, health: dict) -> None:
    """Write KARL status to JSON for Nexus dashboard consumption."""
    status = {
        "weights": weights,
        "shadow": shadow,
        "promotion": promotion,
        "health": health,
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }
    status_path = KARL_DIR / "karl_status.json"
    try:
        with open(status_path, "w") as f:
            json.dump(status, f, indent=2, default=str)
    except Exception:
        pass


@task(name="report-karl-analysis")
def report_task(weights: dict, shadow: dict, promotion: dict, health: dict) -> None:
    date = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    lines = [f"**KARL Daily Analysis** ({date})"]

    # Weights
    updated = weights.get("updated", 0)
    if updated:
        lines.append(f"Weights: {updated} skills updated")
        for name, info in weights.get("updates", {}).items():
            delta = info.get("delta", 0)
            arrow = "+" if delta >= 0 else ""
            lines.append(f"  `{name}`: {arrow}{delta:.4f} (reward={info.get('mean_reward', 0):.3f})")
    else:
        lines.append(f"Weights: no updates ({weights.get('message', '')})")

    # Shadow routing
    records = shadow.get("records", 0)
    hit_rate = shadow.get("hit_rate", 0)
    agreement = shadow.get("agreement_rate", 0)
    lines.append(f"Shadow: {records} records, {hit_rate:.0%} cache hit, {agreement:.0%} agreement")

    # Health
    total = health.get("total_trajectories", 0)
    scored = health.get("scored_trajectories", 0)
    lines.append(f"Trajectories: {total} total, {scored} scored")

    # Promotion
    ready = promotion.get("ready", False)
    lines.append(f"Vector promotion: {'READY' if ready else 'Not ready'}")
    if ready:
        lines.append(">> ACTION: Vector routing ready for promotion to active mode")

    _post_discord("\n".join(lines))


@flow(name="karl-daily-analysis", log_prints=True)
def karl_analysis() -> dict:
    """Daily KARL trajectory analysis."""
    results = {"started_at": datetime.now(timezone.utc).isoformat()}

    print("[karl] Running weight updates...")
    weights = update_weights_task()
    results["weights"] = weights

    print("[karl] Analyzing shadow routing...")
    shadow = shadow_analysis_task()
    results["shadow"] = shadow

    print("[karl] Checking promotion readiness...")
    promotion = promotion_task()
    results["promotion"] = promotion

    print("[karl] Computing skill health...")
    health = health_task()
    results["health"] = health

    print("[karl] Persisting status...")
    persist_status_task(weights, shadow, promotion, health)

    print("[karl] Reporting...")
    report_task(weights, shadow, promotion, health)

    results["completed_at"] = datetime.now(timezone.utc).isoformat()
    return results


if __name__ == "__main__":
    result = karl_analysis()
    print(json.dumps(result, indent=2, default=str))
