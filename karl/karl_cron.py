#!/usr/bin/env python3
"""
karl_cron.py — Scheduled KARL automation.

Runs periodic maintenance tasks:
  1. Auto-refresh centroids (if due)
  2. Log skill evolution snapshot
  3. Run integrity check
  4. Generate and optionally post health digest

Usage:
    python3 karl_cron.py              # Run all tasks, print digest
    python3 karl_cron.py --post       # Run all tasks + post digest to Discord
    python3 karl_cron.py --dry-run    # Show what would run without executing
"""

import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

KARL_DIR = Path(__file__).parent
CRON_LOG = KARL_DIR / "karl_cron.log"


def _log(msg: str) -> None:
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    line = f"[{ts}] {msg}"
    print(line)
    with open(CRON_LOG, "a") as f:
        f.write(line + "\n")


def run_cron(post_digest: bool = False, dry_run: bool = False):
    """Execute all scheduled KARL tasks."""
    results = {}
    start = time.time()

    _log("=== KARL Cron Start ===")

    # 1. Auto-refresh centroids
    _log("Step 1: Centroid auto-refresh")
    if not dry_run:
        try:
            from embedding_cache import auto_refresh_centroids
            r = auto_refresh_centroids()
            results["centroid_refresh"] = r.get("status", "unknown")
            _log(f"  Centroid refresh: {r.get('status', 'unknown')}")
            if r.get("high_drift_skills"):
                _log(f"  WARNING: High drift detected in {len(r['high_drift_skills'])} skills")
        except Exception as e:
            results["centroid_refresh"] = f"error: {e}"
            _log(f"  Centroid refresh error: {e}")
    else:
        _log("  [dry-run] Would refresh centroids if due")
        results["centroid_refresh"] = "dry_run"

    # 2. Log skill evolution snapshot
    _log("Step 2: Skill evolution snapshot")
    if not dry_run:
        try:
            from trajectory_bridge import log_skill_evolution
            r = log_skill_evolution()
            skills_tracked = r.get("skills_tracked", 0)
            results["skill_evolution"] = f"{skills_tracked} skills"
            _log(f"  Skill evolution: {skills_tracked} skills tracked")
        except Exception as e:
            results["skill_evolution"] = f"error: {e}"
            _log(f"  Skill evolution error: {e}")
    else:
        _log("  [dry-run] Would log skill evolution snapshot")
        results["skill_evolution"] = "dry_run"

    # 3. Integrity check
    _log("Step 3: Integrity check")
    if not dry_run:
        try:
            from trajectory_bridge import check_integrity
            r = check_integrity()
            issues = r.get("issues", [])
            results["integrity"] = f"{len(issues)} issues"
            _log(f"  Integrity: {len(issues)} issues found")
            for issue in issues[:3]:
                _log(f"    - {issue}")
        except Exception as e:
            results["integrity"] = f"error: {e}"
            _log(f"  Integrity error: {e}")
    else:
        _log("  [dry-run] Would check integrity")
        results["integrity"] = "dry_run"

    # 4. Health digest
    _log("Step 4: Health digest")
    if not dry_run:
        try:
            from trajectory_bridge import generate_health_digest
            digest = generate_health_digest()
            results["digest"] = "generated"
            _log(f"  Digest generated ({len(digest.get('digest', ''))} chars)")

            if post_digest:
                try:
                    from trajectory_bridge import post_health_digest
                    pr = post_health_digest()
                    results["digest_posted"] = pr.get("status", "unknown")
                    _log(f"  Digest posted: {pr.get('status', 'unknown')}")
                except Exception as e:
                    results["digest_posted"] = f"error: {e}"
                    _log(f"  Digest post error: {e}")
        except Exception as e:
            results["digest"] = f"error: {e}"
            _log(f"  Digest error: {e}")
    else:
        _log("  [dry-run] Would generate health digest")
        results["digest"] = "dry_run"

    elapsed = round(time.time() - start, 2)
    results["elapsed_seconds"] = elapsed
    _log(f"=== KARL Cron Complete ({elapsed}s) ===")

    return results


def main():
    post = "--post" in sys.argv
    dry_run = "--dry-run" in sys.argv

    results = run_cron(post_digest=post, dry_run=dry_run)
    print(json.dumps(results, indent=2, default=str))


if __name__ == "__main__":
    main()
