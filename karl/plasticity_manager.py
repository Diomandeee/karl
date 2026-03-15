"""
plasticity_manager.py — Adapter health monitoring and training trigger logic.

Tracks:
  - Reward drift: is the adapter degrading over time?
  - Domain coverage gaps: are new domains going untrained?
  - Training freshness: how stale is the current adapter?
  - Trigger conditions: when should retraining happen?

The manager reads trajectory data and compares recent performance
against the adapter's training baseline to detect degradation.
"""

import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

KARL_DIR = Path(__file__).parent
STORE_PATH = KARL_DIR / "trajectories.jsonl"
BASELINES_PATH = KARL_DIR / "domain_baselines.json"
STATE_PATH = KARL_DIR / "plasticity_state.json"

# Thresholds
DRIFT_THRESHOLD = 0.08  # Mean reward drop > 8% = drifting
STALE_HOURS = 72  # Adapter older than 72h = stale
NEW_TRAJECTORIES_TRIGGER = 50  # N new trajectories since last train = retrain
COVERAGE_GAP_THRESHOLD = 5  # Domain with N+ trajectories but 0 in training set = gap


def load_state() -> Dict:
    """Load plasticity state (last training info, baselines)."""
    if STATE_PATH.exists():
        try:
            return json.loads(STATE_PATH.read_text())
        except Exception:
            pass
    return {
        "last_train_ts": 0,
        "last_train_version": "unknown",
        "last_train_count": 0,
        "trained_domains": [],
    }


def save_state(state: Dict) -> None:
    """Persist plasticity state."""
    STATE_PATH.write_text(json.dumps(state, indent=2, default=str))


def record_training(version: str, train_count: int, domains: List[str]) -> None:
    """Record that a training run completed."""
    state = load_state()
    state.update({
        "last_train_ts": time.time(),
        "last_train_version": version,
        "last_train_count": train_count,
        "trained_domains": domains,
    })
    save_state(state)


def get_recent_rewards(window_hours: float = 24.0) -> List[Dict]:
    """Get trajectory rewards from the last N hours."""
    if not STORE_PATH.exists():
        return []

    cutoff = time.time() - (window_hours * 3600)
    recent = []

    with open(STORE_PATH, "r") as f:
        for line in f:
            try:
                record = json.loads(line)
                ts = record.get("timing", {}).get("start_ts") or record.get("timing", {}).get("end_ts", 0)
                if isinstance(ts, str):
                    continue  # Skip non-numeric timestamps
                if ts > cutoff:
                    reward = record.get("outcome", {}).get("reward_score")
                    domain = record.get("skill", {}).get("domain") or "_global"
                    if reward is not None:
                        recent.append({"reward": reward, "domain": domain, "ts": ts})
            except json.JSONDecodeError:
                continue

    return recent


def compute_drift(window_hours: float = 24.0) -> Dict[str, Any]:
    """
    Compute reward drift: compare recent mean vs training baseline.

    Returns drift metrics per domain and global.
    """
    baselines = {}
    if BASELINES_PATH.exists():
        try:
            baselines = json.loads(BASELINES_PATH.read_text())
        except Exception:
            pass

    recent = get_recent_rewards(window_hours)
    if not recent:
        return {"status": "no_data", "drifts": {}}

    # Group by domain
    by_domain: Dict[str, List[float]] = {}
    for r in recent:
        by_domain.setdefault(r["domain"], []).append(r["reward"])

    drifts = {}
    for domain, rewards in by_domain.items():
        mean = sum(rewards) / len(rewards)
        baseline = baselines.get(domain, {})
        baseline_mean = baseline.get("mean", 0.5) if isinstance(baseline, dict) else (baseline if isinstance(baseline, (int, float)) else 0.5)
        drift = baseline_mean - mean  # Positive = degradation
        drifts[domain] = {
            "recent_mean": round(mean, 4),
            "baseline": round(baseline_mean, 4),
            "drift": round(drift, 4),
            "count": len(rewards),
            "drifting": drift > DRIFT_THRESHOLD,
        }

    # Global drift
    all_rewards = [r["reward"] for r in recent]
    global_mean = sum(all_rewards) / len(all_rewards)
    global_baseline = sum(
        d["baseline"] * d["count"] for d in drifts.values()
    ) / max(sum(d["count"] for d in drifts.values()), 1)
    global_drift = global_baseline - global_mean

    return {
        "status": "drifting" if global_drift > DRIFT_THRESHOLD else "stable",
        "global_drift": round(global_drift, 4),
        "global_mean": round(global_mean, 4),
        "global_baseline": round(global_baseline, 4),
        "domains": drifts,
        "total_recent": len(recent),
    }


def check_coverage_gaps() -> List[str]:
    """Find domains with trajectories but no representation in training."""
    state = load_state()
    trained_domains = set(state.get("trained_domains", []))

    if not STORE_PATH.exists():
        return []

    domain_counts: Dict[str, int] = {}
    with open(STORE_PATH, "r") as f:
        for line in f:
            try:
                record = json.loads(line)
                domain = record.get("skill", {}).get("domain")
                if domain:
                    domain_counts[domain] = domain_counts.get(domain, 0) + 1
            except json.JSONDecodeError:
                continue

    gaps = []
    for domain, count in domain_counts.items():
        if count >= COVERAGE_GAP_THRESHOLD and domain not in trained_domains:
            gaps.append(domain)

    return gaps


def should_retrain() -> Tuple[bool, List[str]]:
    """
    Determine if retraining is warranted.

    Returns (should_retrain, reasons).
    """
    state = load_state()
    reasons = []

    # Check staleness
    hours_since = (time.time() - state["last_train_ts"]) / 3600 if state["last_train_ts"] > 0 else float("inf")
    if hours_since > STALE_HOURS:
        reasons.append(f"Adapter stale: {hours_since:.0f}h since last training (threshold: {STALE_HOURS}h)")

    # Check new trajectory count
    total = 0
    if STORE_PATH.exists():
        with open(STORE_PATH, "r") as f:
            total = sum(1 for _ in f)
    new_since_train = total - state.get("last_train_count", 0)
    if new_since_train >= NEW_TRAJECTORIES_TRIGGER:
        reasons.append(f"{new_since_train} new trajectories since last training (threshold: {NEW_TRAJECTORIES_TRIGGER})")

    # Check drift
    drift = compute_drift(window_hours=24.0)
    if drift.get("status") == "drifting":
        reasons.append(f"Reward drift detected: {drift['global_drift']:.3f}")

    # Check coverage gaps
    gaps = check_coverage_gaps()
    if gaps:
        reasons.append(f"Coverage gaps: {', '.join(gaps)}")

    return len(reasons) > 0, reasons


def health_report() -> Dict[str, Any]:
    """Full plasticity health report."""
    state = load_state()
    drift = compute_drift()
    gaps = check_coverage_gaps()
    retrain, retrain_reasons = should_retrain()

    hours_since = (time.time() - state["last_train_ts"]) / 3600 if state["last_train_ts"] > 0 else None

    return {
        "adapter_version": state.get("last_train_version", "unknown"),
        "hours_since_training": round(hours_since, 1) if hours_since else None,
        "training_examples": state.get("last_train_count", 0),
        "trained_domains": state.get("trained_domains", []),
        "drift": drift,
        "coverage_gaps": gaps,
        "should_retrain": retrain,
        "retrain_reasons": retrain_reasons,
    }


if __name__ == "__main__":
    if "--record" in sys.argv:
        # Record a training event: --record <version> <count> <domain1,domain2,...>
        import sys as _sys
        args = _sys.argv[_sys.argv.index("--record") + 1:]
        if len(args) >= 3:
            record_training(args[0], int(args[1]), args[2].split(","))
            print(f"Recorded training: {args[0]} ({args[1]} examples)")
        else:
            print("Usage: --record <version> <count> <domain1,domain2,...>")
    elif "--drift" in sys.argv:
        drift = compute_drift()
        print(json.dumps(drift, indent=2))
    elif "--retrain" in sys.argv:
        should, reasons = should_retrain()
        print(f"Should retrain: {should}")
        for r in reasons:
            print(f"  - {r}")
    else:
        report = health_report()
        print(json.dumps(report, indent=2))
