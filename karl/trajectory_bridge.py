"""
trajectory_bridge.py - KARL Trajectory Intelligence Bridge.

Bridges trajectory data into routing decisions and analysis:

  1. Shadow Router Analysis  - agreement rate, vector lift, promotion readiness
  2. Skill Health Report     - per-skill success rates and trends
  3. Technique Recommendations - trajectory-weighted suggestions
  4. Promotion Gate          - data-driven decision on activating vector routing

Usage:
    from karl.trajectory_bridge import full_report, check_promotion_readiness
    report = full_report(as_json=True)
    gate = check_promotion_readiness()
"""

import json
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from karl.config import (
    STORE_PATH,
    SHADOW_PATH,
    SKILL_EMBEDDINGS_PATH,
    PROMOTION_MIN_RECORDS,
    PROMOTION_MIN_AGREEMENT,
    PROMOTION_MIN_VECTOR_LIFT,
    PROMOTION_MIN_CACHE_HIT_RATE,
)


def load_shadow_records() -> List[dict]:
    """Load routing shadow comparison logs."""
    if not SHADOW_PATH.exists():
        return []
    records = []
    with open(SHADOW_PATH) as f:
        for line in f:
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return records


def load_trajectories() -> List[dict]:
    """Load trajectory records with reward data."""
    if not STORE_PATH.exists():
        return []
    records = []
    with open(STORE_PATH) as f:
        for line in f:
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return records


def analyze_shadow_routing() -> Dict[str, Any]:
    """Analyze shadow routing data: agreement rate, coverage, latency.

    Compares regex vs vector skill selections logged during shadow mode.
    """
    records = load_shadow_records()
    if not records:
        return {"status": "no_data", "records": 0}

    total = len(records)
    hits = sum(1 for r in records if r.get("vector_status") == "hit")
    misses = sum(1 for r in records if r.get("vector_status") == "miss")
    agrees = sum(1 for r in records if r.get("agree") is True)
    disagrees = sum(1 for r in records if r.get("agree") is False)
    comparable = agrees + disagrees

    elapsed_values = [r.get("elapsed_ms", 0) for r in records if r.get("elapsed_ms")]
    avg_elapsed = sum(elapsed_values) / len(elapsed_values) if elapsed_values else 0
    max_elapsed = max(elapsed_values) if elapsed_values else 0

    regex_matches = sum(1 for r in records if r.get("regex"))
    regex_coverage = regex_matches / total if total else 0
    vector_matches = sum(1 for r in records if r.get("vector"))
    vector_coverage = vector_matches / total if total else 0
    agreement_rate = agrees / comparable if comparable else 0

    regex_skills = Counter(r.get("regex") for r in records if r.get("regex"))
    vector_skills = Counter(r.get("vector") for r in records if r.get("vector"))

    timestamps = [r.get("ts", "") for r in records if r.get("ts")]
    time_range = None
    if timestamps:
        time_range = {"first": min(timestamps), "last": max(timestamps)}

    return {
        "status": "ok",
        "records": total,
        "cache_hits": hits,
        "cache_misses": misses,
        "hit_rate": round(hits / total, 4) if total else 0,
        "comparable": comparable,
        "agrees": agrees,
        "disagrees": disagrees,
        "agreement_rate": round(agreement_rate, 4),
        "regex_coverage": round(regex_coverage, 4),
        "vector_coverage": round(vector_coverage, 4),
        "avg_elapsed_ms": round(avg_elapsed, 2),
        "max_elapsed_ms": round(max_elapsed, 2),
        "regex_skills": dict(regex_skills.most_common()),
        "vector_skills": dict(vector_skills.most_common()),
        "time_range": time_range,
    }


def analyze_skill_health() -> Dict[str, Any]:
    """Per-skill health metrics from trajectory data.

    Computes trajectory counts, mean rewards, process scores, and trend
    detection (comparing first vs second half of trajectory history).
    """
    trajectories = load_trajectories()
    if not trajectories:
        return {"status": "no_data", "skills": {}}

    skill_data: Dict[str, Dict] = defaultdict(lambda: {
        "trajectories": 0,
        "rewards": [],
        "success_rates": [],
        "tool_counts": [],
        "sessions": set(),
    })

    for record in trajectories:
        skill_name = record.get("skill", {}).get("name")
        if not skill_name:
            continue

        data = skill_data[skill_name]
        data["trajectories"] += 1

        outcome = record.get("outcome", {})
        reward = outcome.get("reward_score")
        if reward is not None:
            data["rewards"].append(reward)

        process = outcome.get("process_score")
        if process is not None:
            data["success_rates"].append(process)

        events = record.get("trajectory", {}).get("events", [])
        data["tool_counts"].append(len(events))

        session = record.get("session_id", "")
        if session:
            data["sessions"].add(session)

    health = {}
    for skill_name, data in skill_data.items():
        rewards = data["rewards"]
        mean_reward = sum(rewards) / len(rewards) if rewards else None
        success_rates = data["success_rates"]
        mean_success = sum(success_rates) / len(success_rates) if success_rates else None
        tool_counts = data["tool_counts"]
        mean_tools = sum(tool_counts) / len(tool_counts) if tool_counts else 0

        # Trend: compare first half vs second half
        trend = "stable"
        if len(rewards) >= 6:
            mid = len(rewards) // 2
            first_half = sum(rewards[:mid]) / mid
            second_half = sum(rewards[mid:]) / (len(rewards) - mid)
            delta = second_half - first_half
            if delta > 0.05:
                trend = "improving"
            elif delta < -0.05:
                trend = "declining"

        health[skill_name] = {
            "trajectories": data["trajectories"],
            "unique_sessions": len(data["sessions"]),
            "mean_reward": round(mean_reward, 4) if mean_reward is not None else None,
            "mean_process_score": round(mean_success, 4) if mean_success is not None else None,
            "mean_tools_per_session": round(mean_tools, 1),
            "trend": trend,
        }

    sorted_health = dict(
        sorted(health.items(), key=lambda x: x[1]["trajectories"], reverse=True)
    )

    return {
        "status": "ok",
        "total_trajectories": len(trajectories),
        "scored_trajectories": sum(
            1 for t in trajectories
            if t.get("outcome", {}).get("reward_score") is not None
        ),
        "skills": sorted_health,
    }


def technique_recommendations() -> Dict[str, Any]:
    """Generate technique weight adjustments based on trajectory tool patterns.

    Maps successful trajectory tool patterns to technique IDs:
    - High Read/Grep -> research-heavy -> boost G01 (Brainstorm), R01 (Refine)
    - High Write/Edit -> output-heavy -> boost D01 (Distribute), G06 (Rapid)
    - High Bash -> ops-heavy -> boost G14 (Systematic)
    """
    trajectories = load_trajectories()
    if not trajectories:
        return {"status": "no_data"}

    scored = [t for t in trajectories if t.get("outcome", {}).get("reward_score") is not None]
    if not scored:
        return {"status": "no_scored_data", "total": len(trajectories)}

    high_reward = [t for t in scored if t["outcome"]["reward_score"] >= 0.65]
    low_reward = [t for t in scored if t["outcome"]["reward_score"] < 0.45]

    def tool_distribution(records: List[dict]) -> Dict[str, float]:
        counts: Counter = Counter()
        total = 0
        for r in records:
            for event in r.get("trajectory", {}).get("events", []):
                tool_name = event.get("tool_name", "")
                counts[tool_name] += 1
                total += 1
        return {k: round(v / total, 3) for k, v in counts.most_common()} if total else {}

    high_tools = tool_distribution(high_reward)
    low_tools = tool_distribution(low_reward)

    adjustments = {}

    read_ratio = high_tools.get("Read", 0) + high_tools.get("Grep", 0) + high_tools.get("Glob", 0)
    if read_ratio > 0.4:
        adjustments["G01"] = {"direction": "boost", "reason": "High read/research in successful trajectories"}
        adjustments["R01"] = {"direction": "boost", "reason": "Research-refine pattern succeeds"}

    write_ratio = high_tools.get("Write", 0) + high_tools.get("Edit", 0)
    if write_ratio > 0.3:
        adjustments["D01"] = {"direction": "boost", "reason": "Output-heavy trajectories succeed"}
        adjustments["G06"] = {"direction": "boost", "reason": "Rapid generation pattern works"}

    bash_ratio = high_tools.get("Bash", 0)
    if bash_ratio > 0.25:
        adjustments["G14"] = {"direction": "boost", "reason": "Systematic ops patterns succeed"}

    return {
        "status": "ok",
        "high_reward_count": len(high_reward),
        "low_reward_count": len(low_reward),
        "high_reward_tools": high_tools,
        "low_reward_tools": low_tools,
        "technique_adjustments": adjustments,
    }


def check_promotion_readiness() -> Dict[str, Any]:
    """Check if vector routing is ready to be promoted from shadow to active.

    Requirements:
      1. 100+ shadow records
      2. 50%+ cache hit rate
      3. 80%+ agreement rate when both produce results
      4. 5%+ reward lift on disagreements where vector was correct
    """
    shadow = analyze_shadow_routing()
    if shadow.get("status") != "ok":
        return {"ready": False, "reason": "No shadow data", "details": shadow}

    records_check = shadow["records"] >= PROMOTION_MIN_RECORDS
    hit_rate_check = shadow["hit_rate"] >= PROMOTION_MIN_CACHE_HIT_RATE
    agreement_check = shadow["agreement_rate"] >= PROMOTION_MIN_AGREEMENT

    # Lift check: correlate shadow with trajectory rewards
    shadow_records = load_shadow_records()
    trajectories = load_trajectories()

    session_rewards: Dict[str, float] = {}
    for t in trajectories:
        sid = t.get("session_id", "")
        reward = t.get("outcome", {}).get("reward_score")
        if sid and reward is not None:
            session_rewards[sid] = reward

    agree_rewards = []
    disagree_rewards = []
    for sr in shadow_records:
        sid = sr.get("session_id", "")
        reward = session_rewards.get(sid)
        if reward is None:
            continue
        if sr.get("agree") is True:
            agree_rewards.append(reward)
        elif sr.get("agree") is False:
            disagree_rewards.append(reward)

    mean_agree = sum(agree_rewards) / len(agree_rewards) if agree_rewards else None
    mean_disagree = sum(disagree_rewards) / len(disagree_rewards) if disagree_rewards else None

    lift = None
    lift_check = False
    if mean_agree is not None and mean_disagree is not None:
        lift = round(mean_agree - mean_disagree, 4)
        lift_check = lift >= PROMOTION_MIN_VECTOR_LIFT

    checks = {
        "min_records": {
            "required": PROMOTION_MIN_RECORDS,
            "actual": shadow["records"],
            "pass": records_check,
        },
        "cache_hit_rate": {
            "required": PROMOTION_MIN_CACHE_HIT_RATE,
            "actual": shadow["hit_rate"],
            "pass": hit_rate_check,
        },
        "agreement_rate": {
            "required": PROMOTION_MIN_AGREEMENT,
            "actual": shadow["agreement_rate"],
            "pass": agreement_check,
        },
        "vector_lift": {
            "required": PROMOTION_MIN_VECTOR_LIFT,
            "actual": lift,
            "pass": lift_check,
        },
    }

    all_pass = all(c["pass"] for c in checks.values())

    return {
        "ready": all_pass,
        "checks": checks,
        "recommendation": (
            "PROMOTE: Vector routing meets all criteria. Switch from shadow to active."
            if all_pass else
            "HOLD: Accumulate more data. "
            + ", ".join(k for k, v in checks.items() if not v["pass"])
            + " not yet met."
        ),
    }


def full_report(as_json: bool = False) -> str:
    """Generate comprehensive KARL intelligence report."""
    shadow = analyze_shadow_routing()
    health = analyze_skill_health()
    techniques = technique_recommendations()
    promotion = check_promotion_readiness()

    if as_json:
        return json.dumps({
            "shadow_routing": shadow,
            "skill_health": health,
            "technique_recommendations": techniques,
            "promotion_readiness": promotion,
            "generated_at": datetime.now(timezone.utc).isoformat(),
        }, indent=2, default=str)

    lines = ["=" * 60, "KARL Trajectory Intelligence Report", "=" * 60, ""]

    lines.append("## Shadow Routing Analysis")
    lines.append(f"  Records: {shadow.get('records', 0)}")
    lines.append(f"  Cache hit rate: {shadow.get('hit_rate', 0):.1%}")
    lines.append(f"  Agreement rate: {shadow.get('agreement_rate', 0):.1%}")
    lines.append(f"  Avg latency: {shadow.get('avg_elapsed_ms', 0):.1f}ms")
    lines.append("")

    lines.append("## Skill Health")
    lines.append(f"  Total trajectories: {health.get('total_trajectories', 0)}")
    if health.get("skills"):
        lines.append(f"  {'Skill':<20} {'N':>4} {'Reward':>8} {'Trend':>10}")
        lines.append(f"  {'-' * 44}")
        for name, data in health["skills"].items():
            reward_str = f"{data['mean_reward']:.3f}" if data["mean_reward"] is not None else "N/A"
            lines.append(f"  {name:<20} {data['trajectories']:>4} {reward_str:>8} {data['trend']:>10}")
    lines.append("")

    lines.append("## Vector Routing Promotion")
    lines.append(f"  Ready: {'YES' if promotion.get('ready') else 'NO'}")
    for check_name, check_data in promotion.get("checks", {}).items():
        status = "PASS" if check_data["pass"] else "FAIL"
        lines.append(f"  [{status}] {check_name}: {check_data.get('actual')} (need {check_data.get('required')})")
    lines.append(f"  >> {promotion.get('recommendation', '')}")

    return "\n".join(lines)
