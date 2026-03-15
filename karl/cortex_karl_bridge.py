"""
cortex_karl_bridge.py — Bridges Cortex behavioral intelligence with KARL trajectories.

Joins Cortex entries (corrections, routing decisions, skill invocations) with
KARL trajectory records by session_id. Provides:
  - bridge_domain(): enrich trajectory with Cortex domain/skill data
  - bridge_corrections(): match corrections to trajectory windows
  - get_session_cortex(): all Cortex entries for a session
  - backfill_bridge(): annotate all trajectories with Cortex signals
"""

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

CORTEX_DIR = Path.home() / ".claude" / "cortex"
ENTRIES_FILE = CORTEX_DIR / "entries.jsonl"
KARL_DIR = Path(__file__).parent
STORE_PATH = KARL_DIR / "trajectories.jsonl"


def _load_cortex_entries() -> List[Dict]:
    """Load all Cortex entries."""
    if not ENTRIES_FILE.exists():
        return []
    entries = []
    with open(ENTRIES_FILE) as f:
        for line in f:
            try:
                entries.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return entries


def _load_trajectories() -> List[str]:
    """Load trajectory lines (preserving raw JSON for rewrite)."""
    if not STORE_PATH.exists():
        return []
    with open(STORE_PATH) as f:
        return f.readlines()


def get_session_cortex(session_id: str) -> Dict[str, List[Dict]]:
    """Get all Cortex entries for a session, grouped by type."""
    entries = _load_cortex_entries()
    grouped: Dict[str, List[Dict]] = {}
    for e in entries:
        if e.get("session_id") == session_id:
            entry_type = e.get("type", "unknown")
            grouped.setdefault(entry_type, []).append(e)
    return grouped


def bridge_domain(trajectory: Dict) -> Tuple[Optional[str], Optional[str]]:
    """
    Enrich a trajectory with domain info from Cortex routing decisions.

    Returns (skill_name, domain) from the most relevant Cortex entry.
    Falls back to trajectory's existing skill if no Cortex match.
    """
    session_id = trajectory.get("session_id")
    if not session_id:
        return None, None

    cortex = get_session_cortex(session_id)

    # Priority 1: routing_decision entries (explicit skill routing)
    routing = cortex.get("routing_decision", [])
    if routing:
        latest = routing[-1]
        return latest.get("skill"), latest.get("domain")

    # Priority 2: invocation_record entries (skill was injected)
    invocations = cortex.get("invocation_record", [])
    if invocations:
        latest = invocations[-1]
        return latest.get("skill"), latest.get("domain")

    return None, None


def bridge_corrections(trajectory: Dict) -> List[Dict]:
    """
    Find Cortex correction entries that fall within a trajectory's time window.

    Returns list of correction entries that occurred during or shortly after
    this trajectory's execution.
    """
    session_id = trajectory.get("session_id")
    timing = trajectory.get("timing", {})
    started = timing.get("started_at")
    ended = timing.get("ended_at")

    if not session_id or not started:
        return []

    try:
        start_dt = datetime.fromisoformat(started)
        # Allow 5 minutes after end for correction detection
        if ended:
            end_dt = datetime.fromisoformat(ended)
        else:
            end_dt = start_dt
    except (ValueError, TypeError):
        return []

    corrections = []
    entries = _load_cortex_entries()
    for e in entries:
        if e.get("type") != "correction":
            continue
        try:
            e_ts = datetime.fromisoformat(e.get("ts", ""))
            # Correction within trajectory window + 5 min buffer
            if start_dt <= e_ts <= end_dt.replace(
                minute=end_dt.minute + 5 if end_dt.minute < 55 else 59
            ):
                corrections.append(e)
        except (ValueError, TypeError):
            continue

    return corrections


def infer_success(trajectory: Dict) -> Optional[bool]:
    """
    Infer task success from trajectory signals.

    Priority:
      1. Explicit build_success outcome signal
      2. Cortex correction absence (no correction = likely success)
      3. Process signal: >80% tool success rate
      4. Session continuation (user kept working = not a failure)

    Returns True/False/None.
    """
    outcome = trajectory.get("outcome", {})

    # 1. Explicit build success
    build = outcome.get("build_success")
    if build is not None:
        return build

    # 2. Correction detection (correction = failure)
    correction = outcome.get("correction_detected")
    if correction is True:
        return False

    # 3. Tool success rate
    traj = trajectory.get("trajectory", {})
    total = traj.get("total_tools", 0)
    successes = traj.get("successes", 0)
    if total >= 3:
        rate = successes / total
        if rate >= 0.85:
            return True
        if rate < 0.5:
            return False

    # 4. Session continuation
    continued = outcome.get("session_continued")
    if continued is True:
        return True

    return None


def backfill_from_cortex(force: bool = False) -> Dict[str, int]:
    """
    Backfill trajectory outcome signals from Cortex correction entries.

    Scans Cortex entries for corrections, and annotates matching
    trajectories with correction_detected=True and inferred success.
    """
    lines = _load_trajectories()
    if not lines:
        return {"total": 0, "updated": 0}

    entries = _load_cortex_entries()
    correction_sessions = set()
    for e in entries:
        if e.get("type") == "correction":
            sid = e.get("session_id")
            if sid:
                correction_sessions.add(sid)

    updated = 0
    updated_lines = []
    for line in lines:
        try:
            record = json.loads(line)
            session_id = record.get("session_id", "")
            outcome = record.get("outcome", {})

            changed = False
            if session_id in correction_sessions and outcome.get("correction_detected") is None:
                outcome["correction_detected"] = True
                changed = True

            # Infer success if not set
            if outcome.get("success_inferred") is None or force:
                success = infer_success(record)
                if success is not None:
                    outcome["success_inferred"] = success
                    changed = True

            if changed:
                record["outcome"] = outcome
                updated_lines.append(json.dumps(record, default=str) + "\n")
                updated += 1
            else:
                updated_lines.append(line)
        except json.JSONDecodeError:
            updated_lines.append(line)

    if updated > 0:
        import fcntl
        with open(STORE_PATH, "w") as f:
            fcntl.flock(f.fileno(), fcntl.LOCK_EX)
            try:
                f.writelines(updated_lines)
            finally:
                fcntl.flock(f.fileno(), fcntl.LOCK_UN)

    return {"total": len(lines), "updated": updated}


def backfill_bridge(force: bool = False) -> Dict[str, int]:
    """
    Annotate all trajectories with Cortex bridge data.

    Adds cortex_bridge field to each record with:
      - cortex_skill: skill from routing decision
      - cortex_domain: domain from routing decision
      - corrections_in_window: count of corrections during trajectory
      - cortex_entries: count of all Cortex entries for session
    """
    lines = _load_trajectories()
    if not lines:
        return {"total": 0, "enriched": 0, "skipped": 0}

    entries = _load_cortex_entries()

    # Index entries by session_id (direct match)
    by_session: Dict[str, List[Dict]] = {}
    for e in entries:
        sid = e.get("session_id")
        if sid:
            by_session.setdefault(sid, []).append(e)

    # Index invocation_records by skill name (for skill-based matching)
    invocations_by_skill: Dict[str, List[Dict]] = {}
    for e in entries:
        if e.get("type") == "invocation_record":
            skill = e.get("skill")
            if skill:
                invocations_by_skill.setdefault(skill, []).append(e)

    enriched = 0
    skipped = 0
    updated_lines = []

    for line in lines:
        try:
            record = json.loads(line)
            if record.get("cortex_bridge") and not force:
                skipped += 1
                updated_lines.append(line)
                continue

            session_id = record.get("session_id", "")
            session_entries = by_session.get(session_id, [])

            # Bridge domain
            skill_name = None
            domain = None
            for e in reversed(session_entries):
                if e.get("type") in ("routing_decision", "invocation_record"):
                    skill_name = e.get("skill")
                    domain = e.get("domain")
                    break

            # Count corrections in window
            corrections = [e for e in session_entries if e.get("type") == "correction"]

            # If Cortex has a better skill label, upgrade
            existing_skill = record.get("skill", {})
            if skill_name and not existing_skill.get("name"):
                record["skill"] = {"name": skill_name, "domain": domain}

            record["cortex_bridge"] = {
                "cortex_skill": skill_name,
                "cortex_domain": domain,
                "corrections_in_window": len(corrections),
                "cortex_entries": len(session_entries),
            }

            updated_lines.append(json.dumps(record, default=str) + "\n")
            enriched += 1

        except json.JSONDecodeError:
            updated_lines.append(line)

    if enriched > 0:
        import fcntl
        with open(STORE_PATH, "w") as f:
            fcntl.flock(f.fileno(), fcntl.LOCK_EX)
            try:
                f.writelines(updated_lines)
            finally:
                fcntl.flock(f.fileno(), fcntl.LOCK_UN)

    return {"total": len(lines), "enriched": enriched, "skipped": skipped}


if __name__ == "__main__":
    import sys
    if "--backfill" in sys.argv:
        force = "--force" in sys.argv
        print("[bridge] Backfilling Cortex bridge data...")
        stats = backfill_bridge(force=force)
        print(f"[bridge] Done: {stats}")
    elif "--session" in sys.argv:
        sid = sys.argv[sys.argv.index("--session") + 1]
        cortex = get_session_cortex(sid)
        print(f"Cortex entries for {sid}:")
        for entry_type, entries in cortex.items():
            print(f"  {entry_type}: {len(entries)}")
    else:
        print("Usage: cortex_karl_bridge.py --backfill [--force] | --session <id>")
