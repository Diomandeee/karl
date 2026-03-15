#!/usr/bin/env python3
"""generate_status.py — Generates karl_status.json for the nexus-portal dashboard.

Run periodically (cron, feed-hub flow, or manual) to update the dashboard data.
Outputs to ~/.claude/karl/karl_status.json which the dashboard API serves.

Usage:
    python3 generate_status.py           # Generate + write
    python3 generate_status.py --print   # Generate + print to stdout
"""

import json
import sys
from datetime import datetime, timezone, timedelta
from pathlib import Path

KARL_DIR = Path(__file__).parent
STATUS_FILE = KARL_DIR / "karl_status.json"
REWARD_TREND_FILE = KARL_DIR / "reward_trend.jsonl"

sys.path.insert(0, str(KARL_DIR))


def _load_reward_trend(days=7):
    """Load recent reward trend data."""
    if not REWARD_TREND_FILE.exists():
        return []
    cutoff = datetime.now(timezone.utc) - timedelta(days=days)
    entries = []
    with open(REWARD_TREND_FILE) as f:
        for line in f:
            try:
                r = json.loads(line)
                ts = datetime.fromisoformat(r["ts"])
                if ts >= cutoff:
                    entries.append(r)
            except (json.JSONDecodeError, KeyError, ValueError):
                continue
    return entries


def _compute_quality_distribution():
    """Count trajectory quality grades."""
    from collections import Counter
    from trajectory_bridge import load_trajectories
    trajectories = load_trajectories()
    grades = Counter(t.get("quality", {}).get("grade", "unknown") for t in trajectories)
    return dict(grades)


def _compute_promotion_forecast(shadow_records, target=100):
    """Estimate days until promotion based on shadow record accumulation rate."""
    from trajectory_bridge import load_shadow_records, _load_thresholds
    thresholds = _load_thresholds()
    target = thresholds.get("min_shadow_records", target)

    records = load_shadow_records()
    if len(records) < 2:
        return {"status": "insufficient_data", "records": len(records), "target": target}

    timestamps = []
    for r in records:
        ts = r.get("ts")
        if ts:
            try:
                timestamps.append(datetime.fromisoformat(ts))
            except (ValueError, TypeError):
                continue

    if len(timestamps) < 2:
        return {"status": "no_timestamps", "records": len(records), "target": target}

    timestamps.sort()
    span_seconds = (timestamps[-1] - timestamps[0]).total_seconds()
    span_days = span_seconds / 86400

    # If span is too short (<1 hour), we can't reliably estimate a daily rate
    if span_seconds < 3600:
        remaining = max(target - len(records), 0)
        return {
            "status": "ok",
            "records": len(records),
            "target": target,
            "daily_rate": None,
            "days_remaining": None,
            "estimated_date": None,
            "span_days": round(span_days, 3),
            "note": "Insufficient time span for rate estimation (records span < 1 hour)",
        }

    daily_rate = len(records) / span_days
    # Cap at reasonable maximum (100/day) to avoid misleading projections
    daily_rate = min(daily_rate, 100.0)
    remaining = max(target - len(records), 0)
    days_remaining = round(remaining / daily_rate, 1) if daily_rate > 0 and remaining > 0 else 0.0

    return {
        "status": "ok",
        "records": len(records),
        "target": target,
        "daily_rate": round(daily_rate, 2),
        "days_remaining": days_remaining,
        "estimated_date": (datetime.now(timezone.utc) + timedelta(days=days_remaining)).strftime("%Y-%m-%d") if days_remaining and days_remaining > 0 and days_remaining < 365 else None,
        "span_days": round(span_days, 1),
    }


def _compute_skill_coverage(skill_embs, skill_health, skill_descriptions):
    """Compute coverage: which skills have centroids, data, or both."""
    all_skills = set(skill_descriptions.keys())
    has_centroid = set(skill_embs.keys())
    has_data = set(skill_health.keys())

    both = has_centroid & has_data
    centroid_only = has_centroid - has_data
    data_only = has_data - has_centroid
    neither = all_skills - has_centroid - has_data

    return {
        "total_defined": len(all_skills),
        "with_centroid_and_data": len(both),
        "centroid_only": sorted(centroid_only),
        "data_only": sorted(data_only),
        "no_coverage": sorted(neither),
        "coverage_pct": round(len(both) / len(all_skills) * 100, 1) if all_skills else 0,
    }


def _compute_sft_readiness():
    """Check SFT data readiness for the status dashboard."""
    try:
        from sft_exporter import check_sft_readiness
        return check_sft_readiness()
    except Exception:
        return {"status": "error", "ready": False}


def generate() -> dict:
    """Generate the full KARL status object for the dashboard."""
    from trajectory_bridge import (
        analyze_shadow_routing,
        analyze_skill_health,
        check_promotion_readiness,
        backfill_shadow_agreement,
        per_skill_readiness,
        get_hybrid_routing_table,
    )
    from reward_engine import get_reward_stats
    from embedding_cache import load_skill_embeddings, compute_adaptive_timeout, SKILL_DESCRIPTIONS

    shadow = analyze_shadow_routing()
    health = analyze_skill_health()
    promotion = check_promotion_readiness()
    reward_stats = get_reward_stats()
    agreement = backfill_shadow_agreement()
    timeout_stats = compute_adaptive_timeout()
    skill_ready = per_skill_readiness(min_samples=3)

    # Compute weight updates from skill embeddings
    skill_embs = load_skill_embeddings()
    weight_updates = {}
    for name, (_, weight) in skill_embs.items():
        if name in health.get("skills", {}):
            skill_data = health["skills"][name]
            reward = skill_data.get("mean_reward")
            if reward is not None:
                weight_updates[name] = {
                    "old_weight": round(weight, 4),
                    "new_weight": round(weight, 4),
                    "delta": 0.0,
                    "mean_reward": round(reward, 4),
                }

    # Reward trend (7-day rolling)
    trend_entries = _load_reward_trend(days=7)
    trend_data = {
        "entries": len(trend_entries),
        "points": [{"ts": e["ts"], "mean": e["mean"]} for e in trend_entries[-50:]],
    }
    if trend_entries:
        means = [e["mean"] for e in trend_entries]
        trend_data["rolling_mean"] = round(sum(means) / len(means), 4)

    # Config info
    config = {}
    config_path = KARL_DIR / "config.json"
    if config_path.exists():
        try:
            config = json.loads(config_path.read_text())
        except (json.JSONDecodeError, OSError):
            pass

    # Embedding cache stats
    from embedding_cache import (
        CACHE_PATH, SKILL_CACHE_PATH, EMBEDDING_DIM, _load_cache, _cache,
        centroid_diversity, list_centroid_versions,
    )
    _load_cache()
    prompt_cache_entries = len(_cache)
    skill_centroids = len(skill_embs)
    centroid_div = centroid_diversity() if skill_centroids >= 2 else {}
    centroid_versions_count = len(list_centroid_versions())

    status = {
        "shadow": shadow if shadow.get("status") == "ok" else {
            "records": 0, "cache_hits": 0, "cache_misses": 0,
            "hit_rate": 0, "agreement_rate": 0, "comparable": 0,
            "agrees": 0, "disagrees": 0, "regex_coverage": 0,
            "vector_coverage": 0, "avg_elapsed_ms": 0,
            "regex_skills": {}, "vector_skills": {},
        },
        "accuracy": {
            "annotated": agreement.get("annotated", 0),
            "regex_accuracy": agreement.get("regex_accuracy"),
            "vector_accuracy": agreement.get("vector_accuracy"),
            "top3_accuracy": agreement.get("top3_accuracy"),
            "mrr": agreement.get("mrr"),
            "reward_weighted_accuracy": agreement.get("reward_weighted_accuracy"),
        },
        "latency": {
            "status": timeout_stats.get("status"),
            "p50_ms": timeout_stats.get("p50_ms"),
            "p95_ms": timeout_stats.get("p95_ms"),
            "recommended_timeout": timeout_stats.get("recommended_timeout"),
            "current_budget_ms": timeout_stats.get("current_budget_ms"),
            "samples": timeout_stats.get("samples", 0),
        },
        "health": {
            "total_trajectories": health.get("total_trajectories", 0),
            "scored_trajectories": health.get("scored_trajectories", 0),
            "skills": health.get("skills", {}),
            "quality_distribution": health.get("quality_distribution", _compute_quality_distribution()),
        },
        "promotion": {
            "ready": promotion.get("ready", False),
            "checks": promotion.get("checks", {}),
            "recommendation": promotion.get("recommendation", "No data yet"),
        },
        "weights": {
            "updated": len(weight_updates),
            "updates": weight_updates,
        },
        "reward_stats": {
            "mean": reward_stats.get("mean"),
            "range": reward_stats.get("range"),
            "scored": reward_stats.get("scored"),
            "by_domain": reward_stats.get("by_domain", {}),
        },
        "reward_trend": trend_data,
        "embedding": {
            "model": config.get("embedding", {}).get("model", "gemini-embedding-001"),
            "dimension": config.get("embedding", {}).get("dimension", EMBEDDING_DIM),
            "cache_entries": prompt_cache_entries,
            "skill_centroids": skill_centroids,
            "centroid_diversity": {
                "avg_similarity": centroid_div.get("avg_pairwise_similarity"),
                "max_similarity": centroid_div.get("max_similarity", {}).get("similarity"),
                "max_similarity_pair": centroid_div.get("max_similarity", {}).get("pair"),
                "health": centroid_div.get("health"),
                "clustered_skills": centroid_div.get("clustered_skills", []),
            } if centroid_div.get("status") == "ok" else {},
            "centroid_versions": centroid_versions_count,
        },
        "config": {
            "routing_mode": config.get("routing_mode", "shadow"),
            "auto_promote": config.get("auto_promote", False),
            "embed_timeout": config.get("embed_timeout"),
            "last_centroid_refresh": config.get("last_centroid_refresh"),
        },
        "sft": {
            "base_model": config.get("sft", {}).get("base_model", ""),
            "train_host": config.get("sft", {}).get("train_host", "mac4"),
            "readiness": _compute_sft_readiness(),
        },
        "skill_readiness": skill_ready,
        "hybrid_routing": get_hybrid_routing_table(),
        "skill_coverage": _compute_skill_coverage(skill_embs, health.get("skills", {}), SKILL_DESCRIPTIONS),
        "promotion_forecast": _compute_promotion_forecast(shadow.get("records", 0)),
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }

    return status


def main():
    status = generate()

    if "--print" in sys.argv:
        print(json.dumps(status, indent=2, default=str))
    else:
        STATUS_FILE.write_text(json.dumps(status, indent=2, default=str))
        print(f"Written to {STATUS_FILE} ({len(json.dumps(status))} bytes)")


if __name__ == "__main__":
    main()
