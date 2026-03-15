#!/usr/bin/env python3
"""
trajectory_bridge.py — KARL Trajectory Intelligence Bridge.

Bridges trajectory data into Evolution World and routing decisions:
  1. Shadow Router Analysis — agreement rate, vector lift, promotion readiness
  2. Technique Recommendations — trajectory-weighted technique suggestions for EW
  3. Skill Health Report — per-skill success rates and trends
  4. Promotion Gate — data-driven decision on activating vector routing

Usage:
    python3 trajectory_bridge.py                # Full health report
    python3 trajectory_bridge.py --shadow       # Shadow routing analysis
    python3 trajectory_bridge.py --techniques   # EW technique recommendations
    python3 trajectory_bridge.py --promote      # Check if vector routing is ready
    python3 trajectory_bridge.py --json         # Machine-readable output
"""

import json
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

KARL_DIR = Path(__file__).parent
SHADOW_PATH = KARL_DIR / "routing_shadow.jsonl"
TRAJECTORY_PATH = KARL_DIR / "trajectories.jsonl"
SKILL_EMBEDDINGS_PATH = KARL_DIR / "skill_embeddings.pkl"
CONFIG_PATH = KARL_DIR / "config.json"


def wilson_lower_bound(successes: int, total: int, z: float = 1.96) -> float:
    """Compute Wilson score lower bound for a proportion.

    Returns the lower bound of the confidence interval for a binomial proportion.
    Used to determine the worst-case accuracy estimate for a skill, accounting
    for small sample sizes. z=1.96 gives a 95% confidence interval.
    """
    if total == 0:
        return 0.0
    p = successes / total
    denominator = 1 + z * z / total
    centre = p + z * z / (2 * total)
    spread = z * ((p * (1 - p) + z * z / (4 * total)) / total) ** 0.5
    return max(0.0, (centre - spread) / denominator)


# Promotion thresholds — loaded from config.json at runtime
def _load_thresholds():
    """Read promotion thresholds from config.json, with fallbacks."""
    defaults = {"min_shadow_records": 100, "min_agreement_rate": 0.50, "min_vector_lift": 0.03}
    if CONFIG_PATH.exists():
        try:
            cfg = json.loads(CONFIG_PATH.read_text())
            thresholds = cfg.get("promotion_thresholds", {})
            return {k: thresholds.get(k, v) for k, v in defaults.items()}
        except (json.JSONDecodeError, OSError):
            pass
    return defaults

MIN_SHADOW_RECORDS = 100
MIN_AGREEMENT_RATE = 0.50
MIN_VECTOR_LIFT = 0.03


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
    if not TRAJECTORY_PATH.exists():
        return []
    records = []
    with open(TRAJECTORY_PATH) as f:
        for line in f:
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return records


def analyze_shadow_routing() -> Dict[str, Any]:
    """Analyze shadow routing data: agreement rate, coverage, vector quality."""
    records = load_shadow_records()
    if not records:
        return {"status": "no_data", "records": 0}

    total = len(records)
    hits = sum(1 for r in records if r.get("vector_status") == "hit")
    misses = sum(1 for r in records if r.get("vector_status") == "miss")
    agrees = sum(1 for r in records if r.get("agree") is True)
    disagrees = sum(1 for r in records if r.get("agree") is False)
    comparable = agrees + disagrees  # Records where both produced a result

    # Timing stats
    elapsed_values = [r.get("elapsed_ms", 0) for r in records if r.get("elapsed_ms")]
    avg_elapsed = sum(elapsed_values) / len(elapsed_values) if elapsed_values else 0
    max_elapsed = max(elapsed_values) if elapsed_values else 0

    # Regex coverage (how often regex matches)
    regex_matches = sum(1 for r in records if r.get("regex"))
    regex_coverage = regex_matches / total if total else 0

    # Vector coverage (how often vector has a cache hit + result)
    vector_matches = sum(1 for r in records if r.get("vector"))
    vector_coverage = vector_matches / total if total else 0

    # Agreement rate (only when both have results)
    agreement_rate = agrees / comparable if comparable else 0

    # Collect unique skills seen
    regex_skills = Counter(r.get("regex") for r in records if r.get("regex"))
    vector_skills = Counter(r.get("vector") for r in records if r.get("vector"))

    # Time range
    timestamps = [r.get("ts", "") for r in records if r.get("ts")]
    time_range = None
    if timestamps:
        first = min(timestamps)
        last = max(timestamps)
        time_range = {"first": first, "last": last}

    # Accuracy against actual trajectory skill (if annotated)
    annotated = [r for r in records if r.get("actual_skill")]
    regex_correct = sum(1 for r in annotated if r.get("regex_correct"))
    vector_correct = sum(1 for r in annotated if r.get("vector_correct"))

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
        "annotated": len(annotated),
        "regex_accuracy": round(regex_correct / len(annotated), 4) if annotated else None,
        "vector_accuracy": round(vector_correct / len(annotated), 4) if annotated else None,
    }


def per_skill_readiness(min_samples: int = 10) -> Dict[str, Any]:
    """Compute per-skill promotion readiness from annotated shadow records.

    Returns which skills have enough data and accuracy to route actively.
    """
    records = load_shadow_records()
    annotated = [r for r in records if r.get("actual_skill")]
    if not annotated:
        return {"status": "no_data", "skills": {}}

    # Group by actual skill
    skill_stats = defaultdict(lambda: {"total": 0, "vector_correct": 0, "regex_correct": 0})
    for r in annotated:
        actual = r["actual_skill"]
        skill_stats[actual]["total"] += 1
        if r.get("vector_correct"):
            skill_stats[actual]["vector_correct"] += 1
        if r.get("regex_correct"):
            skill_stats[actual]["regex_correct"] += 1

    skills = {}
    for skill_name, stats in sorted(skill_stats.items()):
        total = stats["total"]
        vec_acc = stats["vector_correct"] / total if total else 0
        reg_acc = stats["regex_correct"] / total if total else 0

        # Wilson lower bound accounts for sample size uncertainty
        vec_lower = wilson_lower_bound(stats["vector_correct"], total)
        # A skill is ready if its worst-case accuracy (Wilson lower) >= 0.3
        # AND it has at least min_samples
        ready = total >= min_samples and vec_lower >= 0.3

        skills[skill_name] = {
            "samples": total,
            "vector_accuracy": round(vec_acc, 3),
            "vector_lower_bound": round(vec_lower, 3),
            "regex_accuracy": round(reg_acc, 3),
            "ready": ready,
            "enough_data": total >= min_samples,
        }

    ready_count = sum(1 for s in skills.values() if s["ready"])
    return {
        "status": "ok",
        "total_skills": len(skills),
        "ready_skills": ready_count,
        "skills": skills,
    }


def backfill_shadow_agreement() -> Dict[str, Any]:
    """Cross-reference shadow records with trajectories to compute real agreement.

    For each shadow record, finds the trajectory with the matching session_id,
    gets the actual skill used, and annotates whether regex/vector predicted
    the correct skill. Updates routing_shadow.jsonl in place.
    """
    records = load_shadow_records()
    trajectories = load_trajectories()
    if not records or not trajectories:
        return {"status": "no_data", "updated": 0}

    # Build session_id -> actual_skill map from trajectories
    session_skills = {}
    for t in trajectories:
        sid = t.get("session_id", "")[:16]  # Shadow records use truncated session_id
        skill = t.get("skill", {}).get("name")
        if sid and skill:
            session_skills[sid] = skill

    updated = 0
    for record in records:
        sid = record.get("session_id", "")
        actual = session_skills.get(sid)
        if actual and record.get("actual_skill") != actual:
            record["actual_skill"] = actual
            record["regex_correct"] = record.get("regex") == actual
            record["vector_correct"] = record.get("vector") == actual
            updated += 1

    if updated > 0:
        # Rewrite shadow log with annotations
        SHADOW_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(SHADOW_PATH, "w") as f:
            for record in records:
                f.write(json.dumps(record, separators=(",", ":"), default=str) + "\n")

    # Compute annotated agreement stats
    annotated = [r for r in records if r.get("actual_skill")]
    regex_correct = sum(1 for r in annotated if r.get("regex_correct"))
    vector_correct = sum(1 for r in annotated if r.get("vector_correct"))

    # Compute top-3 and MRR from records with top_k data
    top3_hits = sum(1 for r in annotated if r.get("in_top3"))
    rr_sum = sum(r.get("reciprocal_rank", 0) for r in annotated)
    has_topk = sum(1 for r in annotated if r.get("top_k"))

    # Reward-weighted accuracy: weight each prediction by its trajectory reward
    session_rewards = {}
    for t in trajectories:
        sid = t.get("session_id", "")[:16]
        reward = t.get("outcome", {}).get("reward_score")
        if sid and reward is not None:
            session_rewards[sid] = reward

    weighted_correct = 0.0
    weighted_total = 0.0
    for r in annotated:
        reward = session_rewards.get(r.get("session_id", ""), 0.5)
        weighted_total += reward
        if r.get("vector_correct"):
            weighted_correct += reward

    reward_weighted_accuracy = round(weighted_correct / weighted_total, 4) if weighted_total > 0 else None

    return {
        "status": "ok",
        "updated": updated,
        "annotated": len(annotated),
        "regex_accuracy": round(regex_correct / len(annotated), 4) if annotated else 0,
        "vector_accuracy": round(vector_correct / len(annotated), 4) if annotated else 0,
        "top3_accuracy": round(top3_hits / has_topk, 4) if has_topk else None,
        "mrr": round(rr_sum / has_topk, 4) if has_topk else None,
        "reward_weighted_accuracy": reward_weighted_accuracy,
    }


def confusion_matrix() -> Dict[str, Any]:
    """Build confusion matrix from annotated shadow records.

    Shows actual_skill vs vector prediction to identify systematic misclassifications.
    Returns matrix dict, per-skill precision/recall, and ASCII table.
    """
    records = load_shadow_records()
    annotated = [r for r in records if r.get("actual_skill") and r.get("vector")]

    if len(annotated) < 5:
        return {"status": "insufficient_data", "annotated": len(annotated)}

    # Build confusion: {actual: {predicted: count}}
    matrix: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for r in annotated:
        actual = r["actual_skill"]
        predicted = r["vector"]
        matrix[actual][predicted] += 1

    # Compute per-skill precision and recall
    all_skills = sorted(set(list(matrix.keys()) + [p for row in matrix.values() for p in row]))
    per_skill = {}
    for skill in all_skills:
        tp = matrix.get(skill, {}).get(skill, 0)
        fn = sum(v for k, v in matrix.get(skill, {}).items() if k != skill)
        fp = sum(matrix.get(a, {}).get(skill, 0) for a in all_skills if a != skill)
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        per_skill[skill] = {
            "tp": tp, "fp": fp, "fn": fn,
            "precision": round(precision, 3),
            "recall": round(recall, 3),
            "f1": round(f1, 3),
        }

    # Build ASCII table (top confusions)
    confusions = []
    for actual, predictions in matrix.items():
        for predicted, count in predictions.items():
            if actual != predicted:
                confusions.append((actual, predicted, count))
    confusions.sort(key=lambda x: -x[2])

    ascii_lines = ["Top confusions (actual -> predicted : count):"]
    for actual, predicted, count in confusions[:15]:
        ascii_lines.append(f"  {actual:25s} -> {predicted:25s} : {count}")

    return {
        "status": "ok",
        "annotated": len(annotated),
        "skills": len(all_skills),
        "matrix": {a: dict(p) for a, p in matrix.items()},
        "per_skill": per_skill,
        "top_confusions": [{"actual": a, "predicted": p, "count": c} for a, p, c in confusions[:10]],
        "ascii": "\n".join(ascii_lines),
    }


def resolve_confusion_pairs(top_n: int = 5) -> Dict[str, Any]:
    """Analyze top confused skill pairs and suggest resolution strategies.

    For each confused pair:
    1. Identifies distinguishing features (cwd paths, tool patterns)
    2. Computes centroid similarity and separation gap
    3. Suggests fix: refine centroid, merge skills, or increase penalty
    """
    from embedding_cache import load_skill_embeddings, cosine_similarity, GENERIC_SKILLS

    records = load_shadow_records()
    annotated = [r for r in records if r.get("actual_skill") and r.get("vector")]

    if len(annotated) < 5:
        return {"status": "insufficient_data", "annotated": len(annotated)}

    # Count confusions
    confusion_counts: Dict[Tuple[str, str], int] = {}
    for r in annotated:
        actual = r["actual_skill"]
        predicted = r["vector"]
        if actual != predicted:
            pair = (actual, predicted)
            confusion_counts[pair] = confusion_counts.get(pair, 0) + 1

    # Sort by error count
    sorted_pairs = sorted(confusion_counts.items(), key=lambda x: -x[1])[:top_n]

    embs = load_skill_embeddings()
    resolutions = []

    for (actual, predicted), count in sorted_pairs:
        sim = 0.0
        if actual in embs and predicted in embs:
            sim = cosine_similarity(embs[actual][0], embs[predicted][0])

        # Determine fix strategy
        if sim > 0.95:
            strategy = "merge_or_separate"
            explanation = f"Centroids nearly identical ({sim:.3f}). Consider merging skills or adding very distinctive exemplar prompts."
        elif sim > 0.85:
            strategy = "refine_centroids"
            explanation = f"Centroids too close ({sim:.3f}). Run 'karl refine --apply' to push apart, or add exemplar prompts."
        elif actual in GENERIC_SKILLS or predicted in GENERIC_SKILLS:
            strategy = "increase_penalty"
            explanation = f"Generic skill involved. Consider increasing generic_penalty from 0.01 to 0.02."
        else:
            strategy = "add_exemplars"
            explanation = f"Moderate similarity ({sim:.3f}). Add more exemplar prompts to strengthen centroids."

        # Check bidirectional confusion
        reverse_count = confusion_counts.get((predicted, actual), 0)
        bidirectional = reverse_count > 0

        # Weighted impact: errors * (1 / total_for_skill) gives fraction of skill's data that's wrong
        total_for_actual = sum(1 for r in annotated if r.get("actual_skill") == actual)
        error_rate = count / total_for_actual if total_for_actual > 0 else 0

        resolutions.append({
            "actual": actual,
            "predicted": predicted,
            "errors": count,
            "reverse_errors": reverse_count,
            "bidirectional": bidirectional,
            "centroid_similarity": round(sim, 4),
            "error_rate": round(error_rate, 3),
            "strategy": strategy,
            "explanation": explanation,
        })

    return {
        "status": "ok",
        "total_errors": sum(confusion_counts.values()),
        "total_annotated": len(annotated),
        "accuracy": round(1 - sum(confusion_counts.values()) / len(annotated), 3),
        "resolutions": resolutions,
    }


def auto_resolve_confusion(top_n: int = 5, dry_run: bool = True) -> Dict[str, Any]:
    """Auto-resolve confused skill pairs by generating targeted contrastive seeds.

    For each top confused pair, extracts distinguishing features from trajectories
    (cwd patterns, tool usage, prompt keywords) and generates synthetic shadow
    records with contrastive prompts that help separate the confused skills.

    Steps:
    1. Identify top-N confused pairs (from resolve_confusion_pairs)
    2. For each pair, extract trajectory features for both skills
    3. Generate contrastive prompts emphasizing distinguishing features
    4. Embed contrastive prompts and add as shadow records
    5. Return seed count and expected accuracy improvement

    Args:
        top_n: Number of confused pairs to resolve
        dry_run: If True, show plan without writing
    """
    from embedding_cache import (
        load_skill_embeddings, embed_sync, rank_skills, cosine_similarity,
    )

    # Get confusion analysis
    confusion = resolve_confusion_pairs(top_n=top_n)
    if confusion.get("status") != "ok":
        return {"status": "no_confusion_data", "detail": confusion}

    resolutions = confusion.get("resolutions", [])
    if not resolutions:
        return {"status": "no_confusions", "accuracy": confusion.get("accuracy")}

    # Load trajectories for feature extraction
    trajectories = []
    if TRAJECTORY_PATH.exists():
        with open(TRAJECTORY_PATH) as f:
            for line in f:
                try:
                    trajectories.append(json.loads(line))
                except json.JSONDecodeError:
                    continue

    # Extract per-skill features from trajectories
    skill_features: Dict[str, Dict] = {}
    for t in trajectories:
        skill_name = t.get("skill", {}).get("name", "")
        if not skill_name:
            continue
        if skill_name not in skill_features:
            skill_features[skill_name] = {
                "cwds": [], "tools": [], "keywords": [], "prompts": [],
            }
        ctx = t.get("context", {})
        cwd = ctx.get("cwd", "")
        if cwd:
            skill_features[skill_name]["cwds"].append(cwd)
        prompt = ctx.get("prompt_text", ctx.get("user_prompt", ""))
        if prompt and len(prompt) > 10:
            skill_features[skill_name]["prompts"].append(prompt[:500])
            # Extract keywords (words > 5 chars, lowered)
            words = set(w.lower() for w in prompt.split() if len(w) > 5 and w.isalpha())
            skill_features[skill_name]["keywords"].extend(words)
        events = t.get("trajectory", {}).get("events", [])
        tools = [e.get("tool_name", "") for e in events[:10] if e.get("tool_name")]
        skill_features[skill_name]["tools"].extend(tools)

    def _build_contrastive_prompts(skill: str, confused_with: str) -> List[str]:
        """Generate prompts that strongly distinguish 'skill' from 'confused_with'."""
        feats = skill_features.get(skill, {})
        confused_feats = skill_features.get(confused_with, {})

        prompts = []

        # Use distinctive cwd paths
        skill_cwds = set(feats.get("cwds", []))
        confused_cwds = set(confused_feats.get("cwds", []))
        unique_cwds = skill_cwds - confused_cwds
        for cwd in list(unique_cwds)[:2]:
            project = cwd.rstrip("/").split("/")[-1]
            prompts.append(f"Working in {project} directory, need to fix a bug in the codebase")
            prompts.append(f"I'm in {cwd} and need to implement a new feature for {project}")

        # Use distinctive keywords
        skill_kws = set(feats.get("keywords", []))
        confused_kws = set(confused_feats.get("keywords", []))
        unique_kws = skill_kws - confused_kws
        if unique_kws:
            kw_sample = list(unique_kws)[:5]
            prompts.append(f"Help me with {' '.join(kw_sample[:3])} in the project")

        # Use distinctive tool patterns
        from collections import Counter as _Counter
        skill_tools = _Counter(feats.get("tools", []))
        confused_tools = _Counter(confused_feats.get("tools", []))
        # Find tools used much more in target skill
        for tool, count in skill_tools.most_common(3):
            if count > confused_tools.get(tool, 0) * 2:
                prompts.append(f"Need to use {tool} to update the {skill} configuration")

        # Use actual prompts as seeds (slightly modified)
        for p in feats.get("prompts", [])[:2]:
            if len(p) > 20:
                prompts.append(p[:300])

        # Fallback: generate from skill name
        if not prompts:
            prompts.append(f"Working on {skill} related task, need help with implementation")
            prompts.append(f"Debug an issue in the {skill} module")

        return prompts[:6]  # Max 6 per skill per pair

    # Generate seeds for each confused pair
    seeds = []
    for res in resolutions:
        actual = res["actual"]
        predicted = res["predicted"]
        errors = res["errors"]

        # Generate contrastive prompts for the ACTUAL skill (the one being misclassified)
        contrastive = _build_contrastive_prompts(actual, predicted)

        for prompt in contrastive:
            seeds.append({
                "prompt": prompt,
                "actual_skill": actual,
                "confused_with": predicted,
                "errors_targeted": errors,
            })

    result = {
        "status": "ok",
        "pairs_targeted": len(resolutions),
        "seeds_generated": len(seeds),
        "dry_run": dry_run,
        "seeds": [{"skill": s["actual_skill"], "vs": s["confused_with"],
                    "prompt": s["prompt"][:100]} for s in seeds],
    }

    if dry_run:
        return result

    # Embed and write shadow records
    embs = load_skill_embeddings()
    written = 0
    with open(SHADOW_PATH, "a") as f:
        for seed in seeds:
            # Embed the contrastive prompt
            emb = embed_sync(seed["prompt"])
            if not emb:
                continue

            # Run vector routing to get predicted skill
            ranking = rank_skills(emb, embs)
            if not ranking:
                continue

            vector_selection = ranking[0][0]
            similarity = ranking[0][1]
            top_k = [{"skill": s, "sim": round(sim, 4)} for s, sim in ranking[:5]]

            record = {
                "session_id": f"auto_resolve_{written}_{int(time.time())}",
                "regex": seed["actual_skill"],
                "vector": vector_selection,
                "vector_status": "hit",
                "similarity": round(similarity, 4),
                "top_k": top_k,
                "in_top3": seed["actual_skill"] in [r[0] for r in ranking[:3]],
                "reciprocal_rank": 1.0 / (next(
                    (i + 1 for i, (s, _) in enumerate(ranking) if s == seed["actual_skill"]),
                    len(ranking)
                )),
                "agree": vector_selection == seed["actual_skill"],
                "elapsed_ms": 0,
                "source": "auto_resolve",
                "actual_skill": seed["actual_skill"],
                "regex_correct": True,
                "vector_correct": vector_selection == seed["actual_skill"],
                "ts": datetime.now(timezone.utc).isoformat(),
            }
            f.write(json.dumps(record, separators=(",", ":"), default=str) + "\n")
            written += 1

    result["written"] = written
    result["dry_run"] = False
    return result


def skill_accuracy_breakdown() -> Dict[str, Any]:
    """Per-skill accuracy breakdown with weighted accuracy loss.

    Returns each skill's accuracy, sample count, and contribution to
    overall accuracy loss. Skills with many samples but low accuracy
    drag the aggregate most.
    """
    records = load_shadow_records()
    annotated = [r for r in records if r.get("actual_skill") and r.get("vector")]

    if len(annotated) < 5:
        return {"status": "insufficient_data"}

    total = len(annotated)
    by_skill: Dict[str, Dict[str, int]] = {}

    for r in annotated:
        actual = r["actual_skill"]
        if actual not in by_skill:
            by_skill[actual] = {"total": 0, "correct": 0, "in_top3": 0}
        by_skill[actual]["total"] += 1
        if r.get("vector_correct"):
            by_skill[actual]["correct"] += 1
        if r.get("in_top3"):
            by_skill[actual]["in_top3"] += 1

    total_correct = sum(s["correct"] for s in by_skill.values())
    overall_accuracy = total_correct / total if total > 0 else 0

    skills = []
    for skill, stats in sorted(by_skill.items(), key=lambda x: -x[1]["total"]):
        acc = stats["correct"] / stats["total"] if stats["total"] > 0 else 0
        top3_acc = stats["in_top3"] / stats["total"] if stats["total"] > 0 else 0
        # Weighted accuracy loss: how much this skill drags down overall accuracy
        # If this skill had 100% accuracy, overall would improve by (errors / total)
        errors = stats["total"] - stats["correct"]
        accuracy_drag = errors / total  # Fraction of total errors from this skill

        skills.append({
            "skill": skill,
            "samples": stats["total"],
            "correct": stats["correct"],
            "accuracy": round(acc, 3),
            "top3_accuracy": round(top3_acc, 3),
            "errors": errors,
            "accuracy_drag": round(accuracy_drag, 3),
        })

    return {
        "status": "ok",
        "total": total,
        "overall_accuracy": round(overall_accuracy, 3),
        "skills": skills,
    }


def accuracy_by_source() -> Dict[str, Any]:
    """Break down vector accuracy by record source (organic vs synthetic).

    Shadow records from live routing have source='live' (or missing).
    Records from shadow_seeder have source='shadow_seeder'.
    This lets us measure if organic routing performance differs from seeded.
    """
    records = load_shadow_records()
    annotated = [r for r in records if r.get("actual_skill") and r.get("vector")]

    if len(annotated) < 5:
        return {"status": "insufficient_data", "annotated": len(annotated)}

    by_source: Dict[str, Dict[str, int]] = {}
    for r in annotated:
        src = r.get("source", "live")
        if src not in by_source:
            by_source[src] = {"total": 0, "correct": 0, "in_top3": 0, "rr_sum": 0.0}
        by_source[src]["total"] += 1
        if r.get("vector_correct"):
            by_source[src]["correct"] += 1
        if r.get("in_top3"):
            by_source[src]["in_top3"] += 1
        by_source[src]["rr_sum"] += r.get("reciprocal_rank", 0)

    sources = []
    for src, stats in sorted(by_source.items(), key=lambda x: -x[1]["total"]):
        n = stats["total"]
        acc = stats["correct"] / n
        top3 = stats["in_top3"] / n
        mrr = stats["rr_sum"] / n
        sources.append({
            "source": src,
            "count": n,
            "correct": stats["correct"],
            "accuracy": round(acc, 4),
            "top3_accuracy": round(top3, 4),
            "mrr": round(mrr, 4),
        })

    return {
        "status": "ok",
        "total": len(annotated),
        "sources": sources,
    }


def replay_trajectories(since: Optional[str] = None) -> Dict[str, Any]:
    """Replay vector matching on historical prompts with current centroids.

    Re-runs embedding + ranking on each trajectory's prompt to see what
    the CURRENT centroids would route it to. Compares against the original
    routing and actual skill to detect regressions from centroid changes.

    Args:
        since: Optional YYYY-MM-DD date filter. Only replay trajectories after this date.

    Returns accuracy comparison: original routing vs replayed routing.
    """
    import time as _time
    from embedding_cache import load_skill_embeddings, embed_sync, rank_skills

    trajectories = load_trajectories()
    if not trajectories:
        return {"status": "no_trajectories"}

    # Filter by date if specified
    if since:
        filtered = []
        for t in trajectories:
            ts = t.get("ts", t.get("timestamp", ""))
            if ts and ts[:10] >= since:
                filtered.append(t)
        trajectories = filtered

    embs = load_skill_embeddings()
    if not embs:
        return {"status": "no_centroids"}

    results = []
    original_correct = 0
    replay_correct = 0
    changed = 0
    skipped = 0

    for t in trajectories:
        actual_skill = t.get("skill", {}).get("name", "")
        if not actual_skill:
            skipped += 1
            continue

        # Get prompt text
        ctx = t.get("context", {})
        prompt = ctx.get("prompt_text", ctx.get("user_prompt", ""))
        if not prompt or len(prompt) < 10:
            skipped += 1
            continue

        # Embed and rank with current centroids
        emb = embed_sync(prompt[:4000])
        if not emb:
            skipped += 1
            continue

        ranking = rank_skills(emb, embs)
        if not ranking:
            skipped += 1
            continue

        replay_pick = ranking[0][0]
        replay_sim = ranking[0][1]

        # Get original shadow record routing for this trajectory
        session_id = t.get("session_id", "")[:16]
        original_pick = None
        shadow_records = load_shadow_records()
        for sr in shadow_records:
            if sr.get("session_id", "")[:16] == session_id:
                original_pick = sr.get("vector", "")
                break

        original_was_correct = original_pick == actual_skill if original_pick else None
        replay_is_correct = replay_pick == actual_skill

        if original_was_correct is True:
            original_correct += 1
        if replay_is_correct:
            replay_correct += 1
        if original_pick and original_pick != replay_pick:
            changed += 1

        results.append({
            "skill": actual_skill,
            "original": original_pick,
            "replay": replay_pick,
            "replay_sim": round(replay_sim, 4),
            "original_correct": original_was_correct,
            "replay_correct": replay_is_correct,
            "changed": original_pick != replay_pick if original_pick else None,
        })

    total = len(results)
    matched_original = sum(1 for r in results if r["original"] is not None)

    return {
        "status": "ok",
        "total_replayed": total,
        "skipped": skipped,
        "original_accuracy": round(original_correct / matched_original, 4) if matched_original else None,
        "replay_accuracy": round(replay_correct / total, 4) if total else None,
        "accuracy_delta": round(
            (replay_correct / total) - (original_correct / matched_original), 4
        ) if total and matched_original else None,
        "routing_changes": changed,
        "routing_change_rate": round(changed / matched_original, 4) if matched_original else None,
        "improvements": sum(1 for r in results if r["replay_correct"] and not r.get("original_correct")),
        "regressions": sum(1 for r in results if not r["replay_correct"] and r.get("original_correct")),
        "since": since,
    }


def reward_calibration(normalize: bool = False) -> Dict[str, Any]:
    """Analyze per-skill reward distributions and detect anomalies.

    Computes mean, std, min, max per skill. Identifies anomalous patterns:
    - all_high: all rewards > global_mean + 1 std
    - all_low: all rewards < global_mean - 1 std
    - high_variance: std > 2x global std
    - single_sample: only 1 data point

    Optionally returns z-score normalized rewards for fairer cross-skill comparison.
    """
    trajectories = load_trajectories()
    by_skill: Dict[str, List[float]] = {}
    all_rewards = []

    for t in trajectories:
        skill = t.get("skill", {}).get("name", "")
        reward = t.get("outcome", {}).get("reward_score")
        if not skill or reward is None:
            continue
        by_skill.setdefault(skill, []).append(reward)
        all_rewards.append(reward)

    if not all_rewards:
        return {"status": "no_reward_data"}

    global_mean = sum(all_rewards) / len(all_rewards)
    global_std = (sum((r - global_mean) ** 2 for r in all_rewards) / len(all_rewards)) ** 0.5

    skills = []
    anomalies = []
    for skill in sorted(by_skill.keys(), key=lambda s: -len(by_skill[s])):
        rs = by_skill[skill]
        n = len(rs)
        mean = sum(rs) / n
        std = (sum((r - mean) ** 2 for r in rs) / n) ** 0.5 if n > 1 else 0
        mn, mx = min(rs), max(rs)

        entry = {
            "skill": skill,
            "n": n,
            "mean": round(mean, 4),
            "std": round(std, 4),
            "min": round(mn, 4),
            "max": round(mx, 4),
            "z_mean": round((mean - global_mean) / global_std, 3) if global_std > 0 else 0,
        }

        # Anomaly detection
        flags = []
        if n == 1:
            flags.append("single_sample")
        elif n >= 3:
            if all(r > global_mean + global_std for r in rs):
                flags.append("all_high")
            elif all(r < global_mean - global_std for r in rs):
                flags.append("all_low")
            if global_std > 0 and std > 2 * global_std:
                flags.append("high_variance")

        if flags:
            entry["anomalies"] = flags
            anomalies.append(entry)

        if normalize and global_std > 0:
            entry["normalized_rewards"] = [round((r - global_mean) / global_std, 3) for r in rs]

        skills.append(entry)

    return {
        "status": "ok",
        "total_trajectories": len(all_rewards),
        "global_mean": round(global_mean, 4),
        "global_std": round(global_std, 4),
        "skills": skills,
        "anomalies": anomalies,
        "anomaly_count": len(anomalies),
    }


def confidence_analysis() -> Dict[str, Any]:
    """Analyze routing confidence to find optimal rejection thresholds.

    Tests various similarity thresholds and margin cutoffs to find where
    rejecting low-confidence matches and falling back to regex would
    improve overall accuracy.

    For each threshold, shows: how many records would be rejected,
    accuracy on accepted records, accuracy on rejected records.
    """
    records = load_shadow_records()
    annotated = [r for r in records if r.get("actual_skill") and r.get("vector")]

    if len(annotated) < 10:
        return {"status": "insufficient_data", "annotated": len(annotated)}

    # Test similarity thresholds
    sim_thresholds = [0.5, 0.55, 0.6, 0.65, 0.7, 0.75]
    sim_results = []
    for thresh in sim_thresholds:
        accepted = [r for r in annotated if r.get("similarity", 0) >= thresh]
        rejected = [r for r in annotated if r.get("similarity", 0) < thresh]
        acc_accepted = (sum(1 for r in accepted if r.get("vector_correct")) /
                        len(accepted)) if accepted else 0
        acc_rejected = (sum(1 for r in rejected if r.get("vector_correct")) /
                        len(rejected)) if rejected else 0
        sim_results.append({
            "threshold": thresh,
            "accepted": len(accepted),
            "rejected": len(rejected),
            "rejection_rate": round(len(rejected) / len(annotated), 3),
            "accuracy_accepted": round(acc_accepted, 4),
            "accuracy_rejected": round(acc_rejected, 4),
        })

    # Test margin thresholds (gap between top-1 and top-2)
    margin_thresholds = [0.01, 0.02, 0.03, 0.05, 0.1]
    margin_results = []
    for thresh in margin_thresholds:
        accepted = []
        rejected = []
        for r in annotated:
            top_k = r.get("top_k", [])
            if len(top_k) >= 2:
                t1 = top_k[0].get("sim", 0) if isinstance(top_k[0], dict) else 0
                t2 = top_k[1].get("sim", 0) if isinstance(top_k[1], dict) else 0
                margin = t1 - t2
            else:
                margin = 1.0
            if margin >= thresh:
                accepted.append(r)
            else:
                rejected.append(r)

        acc_accepted = (sum(1 for r in accepted if r.get("vector_correct")) /
                        len(accepted)) if accepted else 0
        acc_rejected = (sum(1 for r in rejected if r.get("vector_correct")) /
                        len(rejected)) if rejected else 0
        margin_results.append({
            "min_margin": thresh,
            "accepted": len(accepted),
            "rejected": len(rejected),
            "rejection_rate": round(len(rejected) / len(annotated), 3),
            "accuracy_accepted": round(acc_accepted, 4),
            "accuracy_rejected": round(acc_rejected, 4),
        })

    # Find optimal threshold (highest accuracy with <20% rejection)
    best_sim = max(
        [r for r in sim_results if r["rejection_rate"] < 0.2],
        key=lambda x: x["accuracy_accepted"],
        default=None,
    )

    return {
        "status": "ok",
        "total": len(annotated),
        "overall_accuracy": round(
            sum(1 for r in annotated if r.get("vector_correct")) / len(annotated), 4
        ),
        "similarity_thresholds": sim_results,
        "margin_thresholds": margin_results,
        "recommended_sim_threshold": best_sim["threshold"] if best_sim else None,
    }


def accuracy_forecast(target_n: int = 500) -> Dict[str, Any]:
    """Predict accuracy at N trajectories using Wilson confidence intervals.

    Uses current per-skill accuracy and sample sizes to project accuracy
    at larger N values. Skills with few samples have wide confidence
    intervals that narrow with more data.
    """
    import math

    records = load_shadow_records()
    annotated = [r for r in records if r.get("actual_skill") and r.get("vector")]
    n_current = len(annotated)

    if n_current < 10:
        return {"status": "insufficient_data", "current_n": n_current}

    # Per-skill stats
    by_skill: Dict[str, Dict] = {}
    for r in annotated:
        actual = r["actual_skill"]
        by_skill.setdefault(actual, {"total": 0, "correct": 0})
        by_skill[actual]["total"] += 1
        if r.get("vector_correct"):
            by_skill[actual]["correct"] += 1

    # Wilson CI for current accuracy
    total_correct = sum(s["correct"] for s in by_skill.values())
    current_acc = total_correct / n_current

    z = 1.96  # 95% CI
    # Wilson score interval
    p_hat = current_acc
    denom = 1 + z**2 / n_current
    center = (p_hat + z**2 / (2 * n_current)) / denom
    margin = z * math.sqrt(p_hat * (1 - p_hat) / n_current + z**2 / (4 * n_current**2)) / denom
    wilson_lower = max(0, center - margin)
    wilson_upper = min(1, center + margin)

    # Forecast at different N values
    forecasts = []
    for target in [200, 500, 1000, target_n] if target_n not in [200, 500, 1000] else [200, 500, 1000]:
        # Project CI width at target N (proportional to 1/sqrt(N))
        scale = math.sqrt(n_current / target)
        projected_margin = margin * scale
        projected_lower = max(0, current_acc - projected_margin * z / 1.96)
        projected_upper = min(1, current_acc + projected_margin * z / 1.96)

        forecasts.append({
            "target_n": target,
            "projected_accuracy": round(current_acc, 4),
            "ci_lower": round(projected_lower, 4),
            "ci_upper": round(projected_upper, 4),
            "ci_width": round(projected_upper - projected_lower, 4),
        })

    # Per-skill forecast — which skills will improve with more data
    skill_forecasts = []
    for skill, stats in sorted(by_skill.items(), key=lambda x: x[1]["total"]):
        n = stats["total"]
        correct = stats["correct"]
        acc = correct / n if n > 0 else 0
        # Wilson lower bound at current N
        p = acc
        d = 1 + z**2 / n
        c = (p + z**2 / (2 * n)) / d
        m = z * math.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / d
        wlb = max(0, c - m)

        skill_forecasts.append({
            "skill": skill,
            "samples": n,
            "accuracy": round(acc, 3),
            "wilson_lower": round(wlb, 3),
            "needs_data": n < 10,
            "confidence": "low" if n < 5 else "medium" if n < 15 else "high",
        })

    # Skills most likely to improve with more data (low samples, moderate acc)
    growth_potential = [
        s for s in skill_forecasts
        if s["needs_data"] and s["accuracy"] >= 0.3
    ]

    return {
        "status": "ok",
        "current_n": n_current,
        "current_accuracy": round(current_acc, 4),
        "wilson_ci": {
            "lower": round(wilson_lower, 4),
            "upper": round(wilson_upper, 4),
            "width": round(wilson_upper - wilson_lower, 4),
        },
        "forecasts": forecasts,
        "skill_forecasts": skill_forecasts,
        "growth_potential_skills": [s["skill"] for s in growth_potential],
    }


def difficulty_analysis() -> Dict[str, Any]:
    """Score routing difficulty of shadow records.

    Difficulty factors:
    1. margin: gap between top-1 and top-2 similarity (small gap = harder)
    2. regex_ambiguity: number of regex skill matches for the prompt's context
    3. centroid_distance: absolute distance to nearest centroid (far = harder)
    4. multi_skill_indicators: keywords matching multiple skill domains

    Returns per-record difficulty scores + overall distribution stats.
    """
    records = load_shadow_records()
    annotated = [r for r in records if r.get("actual_skill") and r.get("top_k")]

    if len(annotated) < 5:
        return {"status": "insufficient_data", "annotated": len(annotated)}

    scored = []
    for r in annotated:
        top_k = r.get("top_k", [])
        similarity = r.get("similarity", 0)

        # Factor 1: margin between top-1 and top-2
        if len(top_k) >= 2:
            top1_sim = top_k[0].get("sim", 0) if isinstance(top_k[0], dict) else 0
            top2_sim = top_k[1].get("sim", 0) if isinstance(top_k[1], dict) else 0
            margin = top1_sim - top2_sim
        else:
            margin = 0.5  # Default generous margin for single-result cases

        # Factor 2: centroid distance (inverted — closer is easier)
        dist_score = 1 - similarity if similarity else 0.5

        # Factor 3: how many skills appear in top-k with sim > 0.7
        competitive_count = sum(
            1 for k in top_k
            if (isinstance(k, dict) and k.get("sim", 0) > 0.7)
        )
        competition_score = min(1.0, competitive_count / 5)

        # Composite difficulty: 0 = easy, 1 = hard
        difficulty = round(
            0.4 * (1 - min(1.0, margin * 5))  # small margin = hard
            + 0.3 * dist_score                  # far from centroid = hard
            + 0.3 * competition_score,           # many competitors = hard
            3
        )

        scored.append({
            "session_id": r.get("session_id", "")[:12],
            "actual": r.get("actual_skill", ""),
            "predicted": r.get("vector", ""),
            "correct": r.get("vector_correct", False),
            "similarity": round(similarity, 4),
            "margin": round(margin, 4),
            "competitive_skills": competitive_count,
            "difficulty": difficulty,
        })

    # Sort by difficulty descending
    scored.sort(key=lambda x: -x["difficulty"])

    # Stats
    diffs = [s["difficulty"] for s in scored]
    avg_diff = sum(diffs) / len(diffs)
    easy = sum(1 for d in diffs if d < 0.3)
    medium = sum(1 for d in diffs if 0.3 <= d < 0.6)
    hard = sum(1 for d in diffs if d >= 0.6)

    # Accuracy by difficulty band
    for band_name, band_filter in [("easy", lambda d: d < 0.3),
                                    ("medium", lambda d: 0.3 <= d < 0.6),
                                    ("hard", lambda d: d >= 0.6)]:
        band_records = [s for s in scored if band_filter(s["difficulty"])]
        band_correct = sum(1 for s in band_records if s["correct"])
        band_acc = band_correct / len(band_records) if band_records else 0

    return {
        "status": "ok",
        "total": len(scored),
        "avg_difficulty": round(avg_diff, 3),
        "distribution": {"easy": easy, "medium": medium, "hard": hard},
        "accuracy_by_difficulty": {
            "easy": round(sum(1 for s in scored if s["difficulty"] < 0.3 and s["correct"]) /
                         max(1, easy), 3),
            "medium": round(sum(1 for s in scored if 0.3 <= s["difficulty"] < 0.6 and s["correct"]) /
                           max(1, medium), 3),
            "hard": round(sum(1 for s in scored if s["difficulty"] >= 0.6 and s["correct"]) /
                         max(1, hard), 3),
        },
        "hardest_records": scored[:10],
        "easiest_records": scored[-5:],
    }


def log_skill_evolution() -> Dict[str, Any]:
    """Snapshot per-skill accuracy metrics to skill_evolution.jsonl.

    Called periodically (e.g., every 15 min by timer). Tracks how each
    skill's accuracy changes over time for Grafana visualization.
    """
    records = load_shadow_records()
    annotated = [r for r in records if r.get("actual_skill") and r.get("vector")]

    if len(annotated) < 5:
        return {"status": "insufficient_data"}

    by_skill: Dict[str, Dict[str, int]] = {}
    for r in annotated:
        actual = r["actual_skill"]
        if actual not in by_skill:
            by_skill[actual] = {"total": 0, "correct": 0, "in_top3": 0}
        by_skill[actual]["total"] += 1
        if r.get("vector_correct"):
            by_skill[actual]["correct"] += 1
        if r.get("in_top3"):
            by_skill[actual]["in_top3"] += 1

    ts = datetime.now(timezone.utc).isoformat()
    snapshot = {
        "ts": ts,
        "total_annotated": len(annotated),
        "skills": {},
    }

    for skill, stats in by_skill.items():
        n = stats["total"]
        snapshot["skills"][skill] = {
            "samples": n,
            "accuracy": round(stats["correct"] / n, 4) if n > 0 else 0,
            "top3": round(stats["in_top3"] / n, 4) if n > 0 else 0,
        }

    # Append to evolution log
    evo_path = KARL_DIR / "skill_evolution.jsonl"
    try:
        with open(evo_path, "a") as f:
            f.write(json.dumps(snapshot, separators=(",", ":"), default=str) + "\n")
    except OSError:
        pass

    return {"status": "ok", "skills_tracked": len(by_skill), "timestamp": ts}


def show_skill_evolution(skill_name: Optional[str] = None, limit: int = 10) -> Dict[str, Any]:
    """Show how a skill's accuracy has changed over evolution snapshots.

    If no skill_name, shows aggregate trends across all skills.
    """
    evo_path = KARL_DIR / "skill_evolution.jsonl"
    if not evo_path.exists():
        return {"status": "no_data", "message": "Run 'karl log-skill-evolution' first"}

    snapshots = []
    with open(evo_path) as f:
        for line in f:
            try:
                snapshots.append(json.loads(line))
            except json.JSONDecodeError:
                continue

    if not snapshots:
        return {"status": "no_data"}

    # Take last N snapshots
    recent = snapshots[-limit:]

    if skill_name:
        # Track specific skill
        points = []
        for s in recent:
            skill_data = s.get("skills", {}).get(skill_name)
            if skill_data:
                points.append({
                    "ts": s["ts"],
                    "accuracy": skill_data["accuracy"],
                    "top3": skill_data["top3"],
                    "samples": skill_data["samples"],
                })
        return {
            "status": "ok",
            "skill": skill_name,
            "snapshots": len(points),
            "evolution": points,
            "improving": points[-1]["accuracy"] > points[0]["accuracy"] if len(points) >= 2 else None,
        }
    else:
        # Aggregate: for each skill, compare first vs last snapshot
        first = recent[0].get("skills", {})
        last = recent[-1].get("skills", {})
        all_skills = set(first.keys()) | set(last.keys())

        evolution = []
        for sk in sorted(all_skills):
            f_data = first.get(sk, {})
            l_data = last.get(sk, {})
            f_acc = f_data.get("accuracy", 0)
            l_acc = l_data.get("accuracy", 0)
            delta = l_acc - f_acc
            evolution.append({
                "skill": sk,
                "first_accuracy": round(f_acc, 4),
                "last_accuracy": round(l_acc, 4),
                "delta": round(delta, 4),
                "trend": "improving" if delta > 0.02 else ("degrading" if delta < -0.02 else "stable"),
            })
        evolution.sort(key=lambda x: x["delta"])

        improving = [e for e in evolution if e["trend"] == "improving"]
        degrading = [e for e in evolution if e["trend"] == "degrading"]

        return {
            "status": "ok",
            "snapshots_analyzed": len(recent),
            "skills_tracked": len(all_skills),
            "improving": len(improving),
            "degrading": len(degrading),
            "stable": len(evolution) - len(improving) - len(degrading),
            "evolution": evolution,
        }


def analyze_skill_health() -> Dict[str, Any]:
    """Per-skill health metrics from trajectory data."""
    trajectories = load_trajectories()
    if not trajectories:
        return {"status": "no_data", "skills": {}}

    skill_data: Dict[str, Dict] = defaultdict(lambda: {
        "trajectories": 0,
        "rewards": [],
        "tool_counts": [],
        "success_rates": [],
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

        total_tools = record.get("trajectory", {}).get("total_tools", 0)
        data["tool_counts"].append(total_tools)

        session = record.get("session_id", "")
        if session:
            data["sessions"].add(session)

    # Compute health metrics
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

    # Sort by trajectory count descending
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
    """Generate technique weight adjustments for EW L2 based on trajectory data.

    Maps trajectory tool patterns to EW technique effectiveness:
    - High Read/Grep → research-heavy → favor G01 (Brainstorm), R01 (Refine)
    - High Write/Edit → output-heavy → favor D01 (Distribute), G06 (Rapid)
    - High Bash → ops-heavy → favor G14 (Systematic), G15 (Automate)
    - Corrections detected → technique wasn't effective → penalize
    """
    trajectories = load_trajectories()
    if not trajectories:
        return {"status": "no_data"}

    scored = [t for t in trajectories if t.get("outcome", {}).get("reward_score") is not None]
    if not scored:
        return {"status": "no_scored_data", "total": len(trajectories)}

    # Aggregate tool patterns by reward tier
    high_reward = [t for t in scored if t["outcome"]["reward_score"] >= 0.65]
    low_reward = [t for t in scored if t["outcome"]["reward_score"] < 0.45]

    def tool_distribution(records: List[dict]) -> Dict[str, float]:
        counts: Counter = Counter()
        total = 0
        for r in records:
            tool_seq = r.get("trajectory", {}).get("tool_sequence", [])
            for tool_name in tool_seq:
                counts[tool_name] += 1
                total += 1
        return {k: round(v / total, 3) for k, v in counts.most_common()} if total else {}

    high_tools = tool_distribution(high_reward)
    low_tools = tool_distribution(low_reward)

    # Compute technique weight suggestions based on high-reward patterns
    adjustments = {}

    # Read-heavy success → favor research techniques
    read_ratio = high_tools.get("Read", 0) + high_tools.get("Grep", 0) + high_tools.get("Glob", 0)
    if read_ratio > 0.4:
        adjustments["G01"] = {"direction": "boost", "reason": "High read/research in successful trajectories"}
        adjustments["R01"] = {"direction": "boost", "reason": "Research-refine pattern succeeds"}

    # Write/Edit heavy success → favor output techniques
    write_ratio = high_tools.get("Write", 0) + high_tools.get("Edit", 0)
    if write_ratio > 0.3:
        adjustments["D01"] = {"direction": "boost", "reason": "Output-heavy trajectories succeed"}
        adjustments["G06"] = {"direction": "boost", "reason": "Rapid generation pattern works"}

    # Bash-heavy success → favor systematic approaches
    bash_ratio = high_tools.get("Bash", 0)
    if bash_ratio > 0.25:
        adjustments["G14"] = {"direction": "boost", "reason": "Systematic ops patterns succeed"}

    # Check correction rate in low-reward trajectories
    corrections = sum(
        1 for t in low_reward
        if t.get("outcome", {}).get("reward_components", {}).get("correction_detected")
    )
    if corrections > len(low_reward) * 0.3 and low_reward:
        adjustments["R07"] = {"direction": "boost", "reason": "High correction rate, need more refinement"}

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

    Requirements (config-driven, defaults below):
      1. 200+ annotated shadow records
      2. 50%+ vector_accuracy (vector prediction matches actual trajectory skill)
      3. 70%+ vector coverage (vector produces a result, not "miss")
      4. Positive reward lift: vector-correct sessions have higher reward than vector-incorrect
    """
    # Load thresholds from config
    config = {}
    if CONFIG_PATH.exists():
        try:
            config = json.loads(CONFIG_PATH.read_text())
        except (json.JSONDecodeError, OSError):
            pass
    thresholds = config.get("promotion_thresholds", {})
    min_records = thresholds.get("min_shadow_records", MIN_SHADOW_RECORDS)
    min_accuracy = thresholds.get("min_hit_rate", 0.50)  # repurposed: vector accuracy
    min_coverage = thresholds.get("min_agreement_rate", 0.70)  # repurposed: vector coverage
    min_lift = thresholds.get("min_vector_lift", MIN_VECTOR_LIFT)

    shadow = analyze_shadow_routing()
    if shadow.get("status") != "ok":
        return {"ready": False, "reason": "No shadow data", "details": shadow}

    # Check 1: enough annotated records
    annotated = shadow.get("annotated", 0)
    records_check = annotated >= min_records

    # Check 2: vector accuracy (vector vs actual trajectory skill)
    vector_accuracy = shadow.get("vector_accuracy")
    accuracy_check = vector_accuracy is not None and vector_accuracy >= min_accuracy

    # Check 3: vector coverage (how often vector produces a result)
    vector_coverage = shadow.get("vector_coverage", 0)
    coverage_check = vector_coverage >= min_coverage

    # Check 4: reward lift — vector-correct sessions vs vector-incorrect
    shadow_records = load_shadow_records()
    trajectories = load_trajectories()
    session_rewards = {}
    for t in trajectories:
        sid = t.get("session_id", "")[:16]
        reward = t.get("outcome", {}).get("reward_score")
        if sid and reward is not None:
            session_rewards[sid] = reward

    correct_rewards = []
    incorrect_rewards = []
    for sr in shadow_records:
        sid = sr.get("session_id", "")
        reward = session_rewards.get(sid)
        if reward is None:
            continue
        if sr.get("vector_correct") is True:
            correct_rewards.append(reward)
        elif sr.get("vector_correct") is False:
            incorrect_rewards.append(reward)

    mean_correct = sum(correct_rewards) / len(correct_rewards) if correct_rewards else None
    mean_incorrect = sum(incorrect_rewards) / len(incorrect_rewards) if incorrect_rewards else None

    lift = None
    lift_check = False
    if mean_correct is not None and mean_incorrect is not None:
        lift = round(mean_correct - mean_incorrect, 4)
        lift_check = lift >= min_lift

    checks = {
        "annotated_records": {
            "required": min_records, "actual": annotated, "pass": records_check,
        },
        "vector_accuracy": {
            "required": min_accuracy,
            "actual": round(vector_accuracy, 4) if vector_accuracy is not None else None,
            "pass": accuracy_check,
        },
        "vector_coverage": {
            "required": min_coverage, "actual": round(vector_coverage, 4), "pass": coverage_check,
        },
        "vector_lift": {
            "required": min_lift,
            "actual": lift,
            "pass": lift_check,
            "correct_reward": round(mean_correct, 4) if mean_correct else None,
            "incorrect_reward": round(mean_incorrect, 4) if mean_incorrect else None,
        },
    }

    all_pass = all(c["pass"] for c in checks.values())

    return {
        "ready": all_pass,
        "checks": checks,
        "recommendation": (
            "PROMOTE: Vector routing meets all criteria. Switch from shadow to active."
            if all_pass else
            "HOLD: Accumulate more data. " + ", ".join(
                k for k, v in checks.items() if not v["pass"]
            ) + " not yet met."
        ),
    }


def simulate_lift(fix_pairs: Optional[List[Tuple[str, str]]] = None) -> Dict[str, Any]:
    """Simulate vector lift improvement by fixing confused pairs.

    If fix_pairs is None, simulates fixing ALL confused pairs (upper bound).
    Otherwise, simulates fixing only the specified (actual, predicted) pairs.

    Returns current vs simulated accuracy, lift, and promotion readiness.
    """
    records = load_shadow_records()
    annotated = [r for r in records if r.get("actual_skill") and r.get("vector")]
    if not annotated:
        return {"status": "no_data"}

    trajectories = load_trajectories()
    session_rewards = {}
    for t in trajectories:
        sid = t.get("session_id", "")[:16]
        reward = t.get("outcome", {}).get("reward_score")
        if sid and reward is not None:
            session_rewards[sid] = reward

    # Current state
    current_correct = sum(1 for r in annotated if r.get("vector_correct"))
    current_accuracy = current_correct / len(annotated)

    # Simulate fixes
    simulated_correct = 0
    fixed_count = 0
    for r in annotated:
        actual = r.get("actual_skill", "")
        predicted = r.get("vector", "")
        is_correct = r.get("vector_correct", False)

        if not is_correct and fix_pairs is not None:
            # Check if this error would be fixed
            if (actual, predicted) in fix_pairs:
                simulated_correct += 1
                fixed_count += 1
            else:
                pass  # Still wrong
        elif not is_correct and fix_pairs is None:
            # Fix all errors
            simulated_correct += 1
            fixed_count += 1
        else:
            simulated_correct += 1 if is_correct else 0

    simulated_accuracy = (current_correct + fixed_count) / len(annotated)

    # Compute lift change
    correct_rewards = []
    incorrect_rewards = []
    sim_correct_rewards = []
    sim_incorrect_rewards = []

    for r in annotated:
        sid = r.get("session_id", "")
        reward = session_rewards.get(sid)
        if reward is None:
            continue

        actual = r.get("actual_skill", "")
        predicted = r.get("vector", "")
        is_correct = r.get("vector_correct", False)

        if is_correct:
            correct_rewards.append(reward)
            sim_correct_rewards.append(reward)
        else:
            incorrect_rewards.append(reward)
            # Would this be fixed?
            fixed = (fix_pairs is None) or ((actual, predicted) in (fix_pairs or []))
            if fixed:
                sim_correct_rewards.append(reward)
            else:
                sim_incorrect_rewards.append(reward)

    current_lift = None
    if correct_rewards and incorrect_rewards:
        current_lift = sum(correct_rewards) / len(correct_rewards) - sum(incorrect_rewards) / len(incorrect_rewards)

    simulated_lift = None
    if sim_correct_rewards and sim_incorrect_rewards:
        simulated_lift = sum(sim_correct_rewards) / len(sim_correct_rewards) - sum(sim_incorrect_rewards) / len(sim_incorrect_rewards)

    thresholds = _load_thresholds()
    min_lift = thresholds.get("min_vector_lift", 0.03)

    return {
        "status": "ok",
        "annotated": len(annotated),
        "fix_pairs": [list(p) for p in fix_pairs] if fix_pairs else "all",
        "fixes_applied": fixed_count,
        "current": {
            "accuracy": round(current_accuracy, 4),
            "lift": round(current_lift, 4) if current_lift is not None else None,
            "would_promote": current_lift is not None and current_lift >= min_lift,
        },
        "simulated": {
            "accuracy": round(simulated_accuracy, 4),
            "lift": round(simulated_lift, 4) if simulated_lift is not None else None,
            "would_promote": simulated_lift is not None and simulated_lift >= min_lift,
        },
        "improvement": {
            "accuracy_gain": round(simulated_accuracy - current_accuracy, 4),
            "lift_gain": round((simulated_lift or 0) - (current_lift or 0), 4) if simulated_lift else None,
        },
    }


ACCURACY_TREND_PATH = KARL_DIR / "accuracy_trend.jsonl"


def log_accuracy_trend() -> Dict[str, Any]:
    """Log current accuracy metrics to accuracy_trend.jsonl.

    Called by the status refresh timer alongside log_reward_trend.
    Tracks vector_accuracy, top3_accuracy, mrr, reward_weighted over time.
    """
    agreement = backfill_shadow_agreement()
    if agreement.get("status") != "ok":
        return {"status": "no_data"}

    # Count organic vs synthetic shadow records
    organic_count = 0
    organic_evaluable = 0
    synthetic_count = 0
    organic_correct = 0
    shadow_path = KARL_DIR / "routing_shadow.jsonl"
    if shadow_path.exists():
        for line in shadow_path.read_text().splitlines():
            if not line.strip():
                continue
            try:
                r = json.loads(line)
                src = r.get("source", "")
                if src in ("shadow_seeder", "reshadow"):
                    synthetic_count += 1
                else:
                    organic_count += 1
                    if r.get("vector_correct") is not None:
                        organic_evaluable += 1
                        if r.get("vector_correct") is True:
                            organic_correct += 1
            except (json.JSONDecodeError, KeyError):
                pass

    organic_accuracy = organic_correct / organic_evaluable if organic_evaluable else None

    entry = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "annotated": agreement.get("annotated", 0),
        "vector_accuracy": agreement.get("vector_accuracy"),
        "top3_accuracy": agreement.get("top3_accuracy"),
        "mrr": agreement.get("mrr"),
        "reward_weighted_accuracy": agreement.get("reward_weighted_accuracy"),
        "organic_count": organic_count,
        "synthetic_count": synthetic_count,
        "organic_accuracy": round(organic_accuracy, 4) if organic_accuracy is not None else None,
    }

    # Checkpoint: log milestone when organic records hit 200
    if organic_count >= 200:
        entry["milestone"] = "organic_200_reached"

    try:
        with open(ACCURACY_TREND_PATH, "a") as f:
            f.write(json.dumps(entry, default=str) + "\n")
    except OSError:
        return {"status": "write_error"}

    return {"status": "ok", "entry": entry}


def show_accuracy_trend(days: int = 7) -> Dict[str, Any]:
    """Show accuracy trend over the last N days."""
    if not ACCURACY_TREND_PATH.exists():
        return {"status": "no_data", "entries": 0}

    cutoff = datetime.now(timezone.utc) - timedelta(days=days)
    entries = []
    with open(ACCURACY_TREND_PATH) as f:
        for line in f:
            try:
                r = json.loads(line)
                ts = datetime.fromisoformat(r["ts"])
                if ts >= cutoff:
                    entries.append(r)
            except (json.JSONDecodeError, KeyError, ValueError):
                continue

    if not entries:
        return {"status": "no_data", "entries": 0}

    latest = entries[-1]
    first = entries[0]
    delta_acc = None
    if latest.get("vector_accuracy") is not None and first.get("vector_accuracy") is not None:
        delta_acc = round(latest["vector_accuracy"] - first["vector_accuracy"], 4)

    return {
        "status": "ok",
        "entries": len(entries),
        "days": days,
        "latest": latest,
        "first": first,
        "delta_vector_accuracy": delta_acc,
        "trend": "improving" if delta_acc and delta_acc > 0.01 else (
            "declining" if delta_acc and delta_acc < -0.01 else "stable"
        ),
    }


def get_hybrid_routing_table(min_samples: int = 3, min_accuracy: float = 0.5) -> Dict[str, Any]:
    """Build a hybrid routing table: vector for ready skills, regex for others.

    Instead of all-or-nothing promotion, this lets us benefit from skills
    where vector routing is already accurate while falling back to regex
    for problematic skills like shell-session/comp-core.

    Returns:
        {
            "mode": "hybrid",
            "vector_skills": ["spore", "hef-evolution", ...],
            "regex_skills": ["shell-session", "comp-core", ...],
            "table": {skill: {"route": "vector"|"regex", "accuracy": float, "samples": int}}
        }
    """
    readiness = per_skill_readiness(min_samples=min_samples)
    skills = readiness.get("skills", {})

    # Load routing overrides from config
    overrides = {}
    try:
        config = json.loads(CONFIG_PATH.read_text())
        overrides = config.get("routing_overrides", {})
    except (OSError, json.JSONDecodeError):
        pass

    vector_skills = []
    regex_skills = []
    overridden = []
    table = {}

    for name, data in skills.items():
        entry = {
            "accuracy": data["vector_accuracy"],
            "lower_bound": data.get("vector_lower_bound", 0),
            "samples": data["samples"],
        }
        # Check for manual override
        if name in overrides:
            forced = overrides[name]
            entry["route"] = forced
            entry["override"] = True
            overridden.append({"skill": name, "forced": forced})
            if forced == "vector":
                vector_skills.append(name)
            else:
                regex_skills.append(name)
        elif data.get("ready"):
            vector_skills.append(name)
            entry["route"] = "vector"
        else:
            regex_skills.append(name)
            entry["route"] = "regex"
        table[name] = entry

    return {
        "mode": "hybrid",
        "vector_skills": sorted(vector_skills),
        "regex_skills": sorted(regex_skills),
        "vector_count": len(vector_skills),
        "regex_count": len(regex_skills),
        "overrides": overridden,
        "table": table,
    }


def get_ew_technique_weights() -> Dict[str, float]:
    """Get trajectory-informed technique weights for Evolution World L2.

    Returns a dict of technique_id -> weight_adjustment that can be applied
    to L2's MethodGenome.technique_weights.
    """
    recs = technique_recommendations()
    if recs.get("status") != "ok":
        return {}

    # Map direction to weight multiplier
    direction_map = {"boost": 1.15, "penalize": 0.85, "neutral": 1.0}
    weights = {}
    for technique_id, info in recs.get("technique_adjustments", {}).items():
        direction = info.get("direction", "neutral")
        weights[technique_id] = direction_map.get(direction, 1.0)

    return weights


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

    # Shadow routing
    lines.append("## Shadow Routing Analysis")
    lines.append(f"  Records: {shadow.get('records', 0)}")
    lines.append(f"  Cache hit rate: {shadow.get('hit_rate', 0):.1%}")
    lines.append(f"  Agreement rate: {shadow.get('agreement_rate', 0):.1%} ({shadow.get('comparable', 0)} comparable)")
    lines.append(f"  Avg latency: {shadow.get('avg_elapsed_ms', 0):.1f}ms")
    lines.append(f"  Regex coverage: {shadow.get('regex_coverage', 0):.1%}")
    lines.append(f"  Vector coverage: {shadow.get('vector_coverage', 0):.1%}")
    lines.append("")

    # Skill health
    lines.append("## Skill Health")
    lines.append(f"  Total trajectories: {health.get('total_trajectories', 0)}")
    lines.append(f"  Scored: {health.get('scored_trajectories', 0)}")
    if health.get("skills"):
        lines.append(f"  {'Skill':<20} {'N':>4} {'Reward':>8} {'Trend':>10}")
        lines.append(f"  {'-'*44}")
        for name, data in health["skills"].items():
            reward_str = f"{data['mean_reward']:.3f}" if data["mean_reward"] is not None else "N/A"
            lines.append(f"  {name:<20} {data['trajectories']:>4} {reward_str:>8} {data['trend']:>10}")
    lines.append("")

    # Technique recommendations
    lines.append("## EW Technique Recommendations")
    if techniques.get("technique_adjustments"):
        for tid, info in techniques["technique_adjustments"].items():
            lines.append(f"  {tid}: {info['direction']} — {info['reason']}")
    else:
        lines.append("  No adjustments recommended yet (need more data)")
    lines.append("")

    # Promotion readiness
    lines.append("## Vector Routing Promotion")
    lines.append(f"  Ready: {'YES' if promotion.get('ready') else 'NO'}")
    for check_name, check_data in promotion.get("checks", {}).items():
        status = "PASS" if check_data["pass"] else "FAIL"
        lines.append(f"  [{status}] {check_name}: {check_data.get('actual')} (need {check_data.get('required')})")
    lines.append(f"  >> {promotion.get('recommendation', '')}")
    lines.append("")

    return "\n".join(lines)


def auto_promote() -> Dict[str, Any]:
    """Check promotion readiness and auto-promote if config allows.

    Reads config.json for routing_mode and auto_promote flag.
    If all promotion checks pass and auto_promote is True, flips
    routing_mode from "shadow" to "active". Includes safety rollback:
    if active routing causes reward regression (>5% drop in 48h),
    auto-reverts to shadow mode.

    Returns action taken and promotion details.
    """
    # Load config
    config = {}
    if CONFIG_PATH.exists():
        try:
            config = json.loads(CONFIG_PATH.read_text())
        except (json.JSONDecodeError, OSError):
            config = {}

    current_mode = config.get("routing_mode", "shadow")
    auto = config.get("auto_promote", False)

    # Check readiness
    readiness = check_promotion_readiness()
    is_ready = readiness.get("ready", False)

    action = "none"
    reason = ""

    if current_mode == "shadow" and is_ready and auto:
        # Snapshot centroids before promotion
        try:
            from embedding_cache import save_centroid_snapshot
            save_centroid_snapshot(label="pre_promotion")
        except Exception:
            pass

        # Promote
        config["routing_mode"] = "active"
        config["promoted_at"] = datetime.now(timezone.utc).isoformat()
        config["pre_promotion_reward_mean"] = _get_current_reward_mean()
        config["promotion_checks"] = {k: v for k, v in readiness.get("checks", {}).items()}
        CONFIG_PATH.write_text(json.dumps(config, indent=2, default=str))
        action = "promoted"
        reason = "All promotion checks passed. Vector routing is now active."

        # Log promotion event
        _log_promotion_event(readiness)

        # Notify Discord
        _notify_discord(
            f"**KARL Promoted to Active Routing**\n"
            f"Vector accuracy: {readiness.get('checks', {}).get('accuracy', {}).get('value', '?')}\n"
            f"Shadow records: {readiness.get('checks', {}).get('annotated', {}).get('value', '?')}\n"
            f"Lift: {readiness.get('checks', {}).get('vector_lift', {}).get('value', '?')}"
        )

    elif current_mode == "active":
        # Safety check: has reward regressed since promotion?
        pre_mean = config.get("pre_promotion_reward_mean")
        current_mean = _get_current_reward_mean()
        if pre_mean is not None and current_mean is not None:
            drop = pre_mean - current_mean
            if drop > 0.05:  # >5% regression
                config["routing_mode"] = "shadow"
                config["rollback_at"] = datetime.now(timezone.utc).isoformat()
                config["rollback_reason"] = f"Reward regression: {pre_mean:.3f} -> {current_mean:.3f}"
                CONFIG_PATH.write_text(json.dumps(config, indent=2, default=str))
                action = "rollback"
                reason = f"Reward dropped {drop:.3f} since promotion. Reverted to shadow mode."
                _notify_discord(
                    f"**KARL Rollback: Active -> Shadow**\n"
                    f"Reward regression: {pre_mean:.3f} -> {current_mean:.3f} (drop: {drop:.3f})"
                )

    elif current_mode == "shadow" and not is_ready:
        reason = readiness.get("recommendation", "Not ready")

    return {
        "action": action,
        "current_mode": config.get("routing_mode", "shadow"),
        "readiness": readiness,
        "reason": reason,
    }


def post_promotion_monitor() -> Dict[str, Any]:
    """Monitor A/B performance after promotion to active mode.

    Compares reward distributions between vector-routed and regex-routed
    tasks in trajectories logged since promotion. Returns whether vector
    routing is outperforming, matching, or underperforming regex.

    Auto-recommends rollback if vector underperforms by >5%.
    """
    config = {}
    if CONFIG_PATH.exists():
        try:
            config = json.loads(CONFIG_PATH.read_text())
        except (json.JSONDecodeError, OSError):
            pass

    promoted_at = config.get("promoted_at")
    if not promoted_at:
        return {"status": "not_promoted", "message": "No promotion timestamp found"}

    # Parse promotion timestamp
    try:
        promo_dt = datetime.fromisoformat(promoted_at)
    except (ValueError, TypeError):
        return {"status": "invalid_timestamp"}

    # Load post-promotion trajectories
    trajectories = load_trajectories()
    post_promo = []
    for t in trajectories:
        ts = t.get("context", {}).get("timestamp") or t.get("ts", "")
        if not ts:
            continue
        try:
            t_dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
            if t_dt > promo_dt:
                post_promo.append(t)
        except (ValueError, TypeError):
            continue

    if len(post_promo) < 5:
        return {
            "status": "insufficient_data",
            "post_promotion_tasks": len(post_promo),
            "promoted_at": promoted_at,
            "message": "Need at least 5 post-promotion tasks for comparison",
        }

    # Split by routing method
    vector_tasks = []
    regex_tasks = []
    for t in post_promo:
        matched_by = t.get("skill", {}).get("matched_by", "regex")
        reward = t.get("outcome", {}).get("reward_score")
        if reward is None:
            continue
        if matched_by == "vector":
            vector_tasks.append(reward)
        else:
            regex_tasks.append(reward)

    result = {
        "status": "ok",
        "promoted_at": promoted_at,
        "hours_since_promotion": round((datetime.now(timezone.utc) - promo_dt).total_seconds() / 3600, 1),
        "post_promotion_tasks": len(post_promo),
        "vector_tasks": len(vector_tasks),
        "regex_tasks": len(regex_tasks),
    }

    if vector_tasks:
        result["vector_mean_reward"] = round(sum(vector_tasks) / len(vector_tasks), 4)
    if regex_tasks:
        result["regex_mean_reward"] = round(sum(regex_tasks) / len(regex_tasks), 4)

    # Compare
    if vector_tasks and regex_tasks:
        v_mean = sum(vector_tasks) / len(vector_tasks)
        r_mean = sum(regex_tasks) / len(regex_tasks)
        delta = v_mean - r_mean
        result["reward_delta"] = round(delta, 4)

        if delta > 0.02:
            result["verdict"] = "vector_outperforming"
            result["recommendation"] = "Keep active mode. Vector routing is outperforming regex."
        elif delta < -0.05:
            result["verdict"] = "vector_underperforming"
            result["recommendation"] = "ROLLBACK recommended. Vector routing is underperforming by >5%."
        else:
            result["verdict"] = "comparable"
            result["recommendation"] = "Performance is comparable. Continue monitoring."
    elif not vector_tasks:
        result["verdict"] = "no_vector_tasks"
        result["recommendation"] = "No vector-routed tasks yet. Vector routing may not be triggering."

    # Overall reward check vs pre-promotion baseline
    pre_mean = config.get("pre_promotion_reward_mean")
    if pre_mean is not None and post_promo:
        all_rewards = [t.get("outcome", {}).get("reward_score") for t in post_promo
                       if t.get("outcome", {}).get("reward_score") is not None]
        if all_rewards:
            post_mean = sum(all_rewards) / len(all_rewards)
            result["pre_promotion_mean"] = round(pre_mean, 4)
            result["post_promotion_mean"] = round(post_mean, 4)
            result["overall_delta"] = round(post_mean - pre_mean, 4)

    return result


def backfill_quality_grades(dry_run=False) -> Dict[str, Any]:
    """Retroactively assign quality grades to trajectories missing them."""
    trajectories = load_trajectories()
    if not trajectories:
        return {"status": "empty", "total": 0}

    updated = 0
    already_graded = 0
    for t in trajectories:
        if t.get("quality", {}).get("grade"):
            already_graded += 1
            continue

        events = t.get("trajectory", {}).get("events", [])
        total_tools = t.get("trajectory", {}).get("total_tools", len(events))
        successes = t.get("trajectory", {}).get("successes", 0)
        reward = t.get("outcome", {}).get("reward_score")

        if total_tools == 0:
            success_rate = 0.0
            error_rate = 1.0
        else:
            success_rate = successes / total_tools
            error_rate = 1.0 - success_rate

        # Count corrections: sequential same-tool calls on same file
        correction_count = 0
        for i in range(1, len(events)):
            if (events[i].get("tool_name") == events[i - 1].get("tool_name")
                    and events[i].get("key_params", {}).get("file_path")
                    == events[i - 1].get("key_params", {}).get("file_path")
                    and events[i].get("tool_name") in ("Edit", "Write")):
                correction_count += 1

        tool_names = set(e.get("tool_name", "") for e in events)

        # Grade
        grade = "high"
        if error_rate > 0.3 or correction_count > 2:
            grade = "low"
        elif error_rate > 0.15 or correction_count > 0:
            grade = "medium"
        # Also check reward — very low reward = low quality
        if reward is not None and reward < 0.3:
            grade = "low"

        t["quality"] = {
            "grade": grade,
            "success_rate": round(success_rate, 3),
            "error_rate": round(error_rate, 3),
            "correction_count": correction_count,
            "tool_diversity": len(tool_names),
            "backfilled": True,
        }
        updated += 1

    result = {
        "total": len(trajectories),
        "updated": updated,
        "already_graded": already_graded,
        "dry_run": dry_run,
    }

    # Count grade distribution
    grades = Counter(t.get("quality", {}).get("grade", "unknown") for t in trajectories)
    result["distribution"] = dict(grades)

    if not dry_run and updated > 0:
        with open(TRAJECTORY_PATH, "w") as f:
            for t in trajectories:
                f.write(json.dumps(t, default=str) + "\n")
        result["written"] = True

    return result


def archive_old_records(days=30, dry_run=False) -> Dict[str, Any]:
    """Move trajectory and shadow records older than N days to archive files."""
    from datetime import datetime, timezone, timedelta
    cutoff = datetime.now(timezone.utc) - timedelta(days=days)
    cutoff_str = cutoff.isoformat()
    result = {"trajectories_archived": 0, "shadow_archived": 0, "dry_run": dry_run}

    def _archive_file(source_path, archive_prefix):
        if not source_path.exists():
            return 0, 0
        keep = []
        archive = []
        with open(source_path) as f:
            for line in f:
                try:
                    r = json.loads(line)
                    ts = r.get("ts") or r.get("timestamp") or r.get("created_at", "")
                    if ts and ts < cutoff_str:
                        archive.append(line)
                    else:
                        keep.append(line)
                except json.JSONDecodeError:
                    keep.append(line)

        if not archive:
            return 0, len(keep)

        if not dry_run:
            # Write archive file
            month = cutoff.strftime("%Y%m")
            archive_path = source_path.parent / f"{archive_prefix}_archive_{month}.jsonl"
            with open(archive_path, "a") as f:
                for line in archive:
                    f.write(line)
            # Rewrite active file with only recent records
            with open(source_path, "w") as f:
                for line in keep:
                    f.write(line)

        return len(archive), len(keep)

    t_archived, t_kept = _archive_file(TRAJECTORY_PATH, "trajectories")
    result["trajectories_archived"] = t_archived
    result["trajectories_kept"] = t_kept

    s_archived, s_kept = _archive_file(SHADOW_PATH, "routing_shadow")
    result["shadow_archived"] = s_archived
    result["shadow_kept"] = s_kept

    return result


def analyze_lift_threshold() -> Dict[str, Any]:
    """Analyze the vector lift metric and recommend a threshold.

    Computes the actual lift distribution and statistical significance.
    Lift = mean_reward(correct) - mean_reward(incorrect).

    In shadow mode, lift is weakly informative because routing accuracy
    doesn't affect actual task execution (regex router handles it).
    """
    import math

    records = load_shadow_records()
    trajectories = load_trajectories()

    session_rewards = {}
    for t in trajectories:
        sid = t.get("session_id", "")[:16]
        reward = t.get("outcome", {}).get("reward_score")
        if sid and reward is not None:
            session_rewards[sid] = reward

    correct_rewards = []
    incorrect_rewards = []
    for r in records:
        sid = r.get("session_id", "")
        reward = session_rewards.get(sid)
        if reward is None:
            continue
        if r.get("vector_correct"):
            correct_rewards.append(reward)
        else:
            incorrect_rewards.append(reward)

    if not correct_rewards or not incorrect_rewards:
        return {"status": "insufficient_data"}

    mean_c = sum(correct_rewards) / len(correct_rewards)
    mean_i = sum(incorrect_rewards) / len(incorrect_rewards)
    lift = mean_c - mean_i

    # Compute standard error of the difference (two-sample t-test)
    var_c = sum((x - mean_c) ** 2 for x in correct_rewards) / max(1, len(correct_rewards) - 1)
    var_i = sum((x - mean_i) ** 2 for x in incorrect_rewards) / max(1, len(incorrect_rewards) - 1)
    se = math.sqrt(var_c / len(correct_rewards) + var_i / len(incorrect_rewards))

    # 95% CI for the lift
    ci_lower = lift - 1.96 * se
    ci_upper = lift + 1.96 * se

    # Is the lift statistically significant?
    t_stat = lift / se if se > 0 else 0
    significant = abs(t_stat) > 1.96  # p < 0.05

    # Recommend threshold
    if significant and lift > 0:
        recommended = round(max(0.01, lift * 0.5), 3)  # Half the observed lift
    else:
        recommended = 0.01  # Minimal threshold if lift is not significant

    thresholds = _load_thresholds()
    current = thresholds.get("min_vector_lift", 0.03)

    return {
        "status": "ok",
        "correct_count": len(correct_rewards),
        "incorrect_count": len(incorrect_rewards),
        "mean_correct_reward": round(mean_c, 4),
        "mean_incorrect_reward": round(mean_i, 4),
        "lift": round(lift, 4),
        "standard_error": round(se, 4),
        "confidence_interval_95": [round(ci_lower, 4), round(ci_upper, 4)],
        "t_statistic": round(t_stat, 4),
        "statistically_significant": significant,
        "current_threshold": current,
        "recommended_threshold": recommended,
        "would_pass_current": lift >= current,
        "would_pass_recommended": lift >= recommended,
        "recommendation": (
            f"Lift is {'not ' if not significant else ''}statistically significant (t={t_stat:.2f}). "
            f"Current threshold {current} is {'achievable' if lift >= current else 'too aggressive'}. "
            f"Recommended: {recommended} (based on observed distribution)."
        ),
    }


def generate_dashboard_json() -> Dict[str, Any]:
    """Generate comprehensive KARL dashboard data for Nexus Portal.

    Aggregates all key metrics into a single JSON structure for the portal:
    - Routing mode and promotion status
    - Overall and per-skill accuracy
    - Hybrid routing table
    - Centroid health and similarity matrix highlights
    - Reward calibration summary
    - SFT pipeline status
    - Recent trends
    """
    from embedding_cache import centroid_diversity, skill_similarity_matrix

    try:
        config = json.loads(CONFIG_PATH.read_text())
    except (json.JSONDecodeError, OSError):
        config = {}
    mode = config.get("routing_mode", "shadow")

    # Promotion
    readiness = check_promotion_readiness()

    # Accuracy
    breakdown = skill_accuracy_breakdown()
    by_source = accuracy_by_source()

    # Hybrid routing
    hybrid = get_hybrid_routing_table()
    table = hybrid.get("table", {})
    vector_skills = [s for s, v in table.items() if v.get("route") == "vector"]
    regex_skills = [s for s, v in table.items() if v.get("route") == "regex"]

    # Centroid health
    cdiv = centroid_diversity()

    # Similarity highlights
    sim_matrix = skill_similarity_matrix()
    merge_candidates = sim_matrix.get("merge_candidates", [])[:5]

    # Reward
    reward_cal = reward_calibration()

    # SFT
    try:
        from sft_exporter import check_sft_readiness
        sft = check_sft_readiness()
    except Exception:
        sft = {"status": "unavailable"}

    # Confusion top 5
    confusion = resolve_confusion_pairs(top_n=5)
    top_confusions = [
        {"actual": r["actual"], "predicted": r["predicted"], "errors": r["errors"],
         "similarity": r["centroid_similarity"]}
        for r in confusion.get("resolutions", [])
    ]

    dashboard = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "routing_mode": mode,
        "promotion": {
            "ready": readiness.get("ready", False),
            "checks": readiness.get("checks", {}),
        },
        "accuracy": {
            "overall": breakdown.get("overall_accuracy"),
            "total_records": breakdown.get("total"),
            "by_source": by_source.get("sources", []),
            "per_skill": [
                {"skill": s["skill"], "accuracy": s["accuracy"], "samples": s["samples"]}
                for s in breakdown.get("skills", [])[:15]
            ],
        },
        "hybrid_routing": {
            "vector_count": len(vector_skills),
            "regex_count": len(regex_skills),
            "vector_skills": vector_skills,
        },
        "centroids": {
            "health": cdiv.get("health"),
            "avg_similarity": cdiv.get("avg_pairwise_similarity"),
            "max_similarity": cdiv.get("max_similarity", {}).get("similarity"),
            "max_pair": cdiv.get("max_similarity", {}).get("pair"),
            "clustered": cdiv.get("clustered_skills", []),
        },
        "merge_candidates": merge_candidates,
        "top_confusions": top_confusions,
        "rewards": {
            "global_mean": reward_cal.get("global_mean"),
            "global_std": reward_cal.get("global_std"),
            "anomaly_count": reward_cal.get("anomaly_count", 0),
        },
        "sft": {
            "ready": sft.get("go_nogo", {}).get("ready") if isinstance(sft.get("go_nogo"), dict) else False,
            "exportable": sft.get("dataset", {}).get("total_records") if isinstance(sft.get("dataset"), dict) else None,
        },
    }

    # Write to file
    dashboard_path = KARL_DIR / "karl_dashboard.json"
    dashboard_path.write_text(json.dumps(dashboard, indent=2, default=str))

    return dashboard


def generate_health_digest() -> Dict[str, Any]:
    """Generate a compact KARL health digest for Discord/reporting.

    Covers: promotion status, accuracy, centroid health, SFT readiness,
    top confused pairs, and recommended next actions.
    """
    from embedding_cache import centroid_diversity, list_centroid_versions
    from sft_exporter import check_sft_readiness

    # Promotion
    readiness = check_promotion_readiness()

    # Accuracy
    records = load_shadow_records()
    annotated = [r for r in records if r.get("actual_skill") and r.get("vector")]
    v_correct = sum(1 for r in annotated if r.get("vector_correct"))
    v_acc = round(v_correct / len(annotated), 3) if annotated else 0
    top3 = sum(1 for r in annotated if r.get("in_top3"))
    t3_acc = round(top3 / len(annotated), 3) if annotated else 0

    # Centroid health
    cdiv = centroid_diversity()
    c_health = cdiv.get("health", "unknown")
    c_avg = cdiv.get("avg_pairwise_similarity")
    c_max = cdiv.get("max_similarity", {}).get("similarity")

    # Hybrid routing
    hybrid = get_hybrid_routing_table()

    # SFT
    sft = check_sft_readiness()

    # Config
    config = {}
    if CONFIG_PATH.exists():
        try:
            config = json.loads(CONFIG_PATH.read_text())
        except Exception:
            pass

    # Top confused pairs (for actionable insight)
    confusion = {}
    for r in annotated:
        if not r.get("vector_correct") and r.get("vector"):
            pair = f"{r['actual_skill']}→{r['vector']}"
            confusion[pair] = confusion.get(pair, 0) + 1
    top_confused = sorted(confusion.items(), key=lambda x: -x[1])[:3]

    # Build digest
    mode = config.get("routing_mode", "shadow")
    lines = [
        f"**KARL Health Digest** ({datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')})",
        f"Mode: `{mode}` | Promotion: {'READY' if readiness.get('ready') else 'NOT READY'}",
        f"Accuracy: {v_acc:.1%} | Top-3: {t3_acc:.1%} | Records: {len(annotated)}",
        f"Centroids: {c_health} (avg sim: {c_avg}, max: {c_max})",
        f"Hybrid: {hybrid.get('vector_count', 0)} vector, {hybrid.get('regex_count', 0)} regex",
        f"SFT: {'READY' if sft.get('ready') else 'NOT READY'} ({sft.get('exportable', 0)} exportable)",
        f"Versions: {len(list_centroid_versions())} snapshots",
    ]

    if top_confused:
        lines.append("Top confused: " + ", ".join(f"{p} ({c})" for p, c in top_confused))

    # Recommended actions
    actions = []
    if not readiness.get("ready"):
        checks = readiness.get("checks", {})
        for name, check in checks.items():
            if not check.get("pass"):
                actions.append(f"Fix {name}: {check.get('actual', '?')} < {check.get('required', '?')}")
    if c_health == "poor":
        actions.append("Run `karl refine --apply` to improve centroid separation")
    if mode == "shadow" and readiness.get("ready"):
        actions.append("Ready to promote: `karl promote`")
    if not sft.get("ready"):
        actions.append("SFT not ready: need more high-quality trajectories")

    if actions:
        lines.append("Actions: " + " | ".join(actions))

    digest_text = "\n".join(lines)

    return {
        "status": "ok",
        "digest": digest_text,
        "mode": mode,
        "promotion_ready": readiness.get("ready", False),
        "accuracy": v_acc,
        "top3": t3_acc,
        "centroid_health": c_health,
        "sft_ready": sft.get("ready", False),
    }


def post_health_digest() -> Dict[str, Any]:
    """Generate and post KARL health digest to Discord."""
    digest = generate_health_digest()
    if digest.get("status") == "ok":
        _notify_discord(digest["digest"])
    return digest


def _notify_discord(message: str) -> None:
    """Post a notification to Discord via webhook (best-effort)."""
    import urllib.request
    webhook_file = Path.home() / "flows" / "feed-hub" / "webhooks.env"
    webhook = ""
    if webhook_file.exists():
        try:
            with open(webhook_file) as f:
                for line in f:
                    if line.startswith("DISCORD_WEBHOOK_SERVICE_HEALTH="):
                        webhook = line.split("=", 1)[1].strip().strip('"')
                        break
        except Exception:
            pass
    if not webhook:
        return
    try:
        data = json.dumps({"content": message}).encode()
        req = urllib.request.Request(
            webhook, data=data,
            headers={"Content-Type": "application/json"},
        )
        urllib.request.urlopen(req, timeout=5)
    except Exception:
        pass


def _log_promotion_event(readiness: Dict) -> None:
    """Log a promotion event to promotion_log.jsonl."""
    log_path = KARL_DIR / "promotion_log.jsonl"
    event = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "action": "promoted",
        "checks": readiness.get("checks", {}),
    }
    try:
        with open(log_path, "a") as f:
            f.write(json.dumps(event, default=str) + "\n")
    except Exception:
        pass


def _get_current_reward_mean():
    """Get current mean reward from trajectories."""
    trajectories = load_trajectories()
    rewards = [
        t.get("outcome", {}).get("reward_score")
        for t in trajectories
        if t.get("outcome", {}).get("reward_score") is not None
    ]
    return sum(rewards) / len(rewards) if rewards else None


def dedup_trajectories(dry_run=False) -> Dict[str, Any]:
    """Remove duplicate trajectories by session_id. Keeps the latest version."""
    trajectories = load_trajectories()
    if not trajectories:
        return {"status": "empty", "total": 0, "duplicates": 0}

    seen = {}
    duplicates = 0
    for i, t in enumerate(trajectories):
        sid = t.get("session_id", "")
        if not sid:
            sid = f"__nosid_{i}"
        if sid in seen:
            duplicates += 1
            # Keep the one with higher reward or later timestamp
            existing = seen[sid]
            existing_reward = existing.get("outcome", {}).get("reward_score", 0) or 0
            new_reward = t.get("outcome", {}).get("reward_score", 0) or 0
            if new_reward >= existing_reward:
                seen[sid] = t
        else:
            seen[sid] = t

    deduped = list(seen.values())
    result = {
        "status": "ok",
        "total_before": len(trajectories),
        "total_after": len(deduped),
        "duplicates": duplicates,
        "dry_run": dry_run,
    }

    if not dry_run and duplicates > 0:
        with open(TRAJECTORY_PATH, "w") as f:
            for t in deduped:
                f.write(json.dumps(t, default=str) + "\n")
        result["written"] = True

    return result


def check_integrity() -> Dict[str, Any]:
    """Check trajectory data integrity and report issues."""
    trajectories = load_trajectories()
    if not trajectories:
        return {"status": "empty", "total": 0}

    issues = []
    total = len(trajectories)
    missing_session_id = 0
    missing_skill = 0
    invalid_reward = 0
    missing_timestamp = 0
    empty_context = 0

    for i, t in enumerate(trajectories):
        # Session ID
        if not t.get("session_id"):
            missing_session_id += 1
            if missing_session_id <= 3:
                issues.append(f"Record {i}: missing session_id")

        # Skill name
        skill_name = t.get("skill", {}).get("name")
        if not skill_name:
            missing_skill += 1
            if missing_skill <= 3:
                issues.append(f"Record {i}: missing skill.name")

        # Reward score
        reward = t.get("outcome", {}).get("reward_score")
        if reward is not None:
            if not (0 <= reward <= 1):
                invalid_reward += 1
                if invalid_reward <= 3:
                    issues.append(f"Record {i}: reward {reward} out of [0, 1]")

        # Timestamp
        if not t.get("timestamp") and not t.get("ts"):
            missing_timestamp += 1

        # Context
        ctx = t.get("context", {}).get("user_prompt", "")
        if not ctx:
            empty_context += 1

    return {
        "status": "ok" if not issues else "issues_found",
        "total": total,
        "missing_session_id": missing_session_id,
        "missing_skill": missing_skill,
        "invalid_reward": invalid_reward,
        "missing_timestamp": missing_timestamp,
        "empty_context": empty_context,
        "issue_count": missing_session_id + missing_skill + invalid_reward,
        "sample_issues": issues[:10],
    }


REWARD_TREND_PATH = KARL_DIR / "reward_trend.jsonl"


def log_reward_trend() -> Dict[str, Any]:
    """Append a daily reward summary to reward_trend.jsonl.

    Only appends if today's entry doesn't already exist.
    Computes mean, min, max, count from current trajectories.
    """
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")

    # Check if today already logged
    existing_dates = set()
    if REWARD_TREND_PATH.exists():
        with open(REWARD_TREND_PATH) as f:
            for line in f:
                try:
                    r = json.loads(line)
                    ts = r.get("ts", "")[:10]
                    existing_dates.add(ts)
                except json.JSONDecodeError:
                    continue
    if today in existing_dates:
        return {"status": "already_logged", "date": today}

    trajectories = load_trajectories()
    rewards = [
        t.get("outcome", {}).get("reward_score")
        for t in trajectories
        if t.get("outcome", {}).get("reward_score") is not None
    ]
    if not rewards:
        return {"status": "no_rewards", "date": today}

    entry = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "date": today,
        "mean": round(sum(rewards) / len(rewards), 4),
        "min": round(min(rewards), 4),
        "max": round(max(rewards), 4),
        "count": len(rewards),
        "total_trajectories": len(trajectories),
    }

    # Compute per-quality-grade means
    grade_rewards = defaultdict(list)
    for t in trajectories:
        grade = t.get("quality", {}).get("grade", "unknown")
        score = t.get("outcome", {}).get("reward_score")
        if score is not None:
            grade_rewards[grade].append(score)
    entry["by_grade"] = {
        g: round(sum(rs) / len(rs), 4) for g, rs in grade_rewards.items() if rs
    }

    with open(REWARD_TREND_PATH, "a") as f:
        f.write(json.dumps(entry, default=str) + "\n")

    return {"status": "logged", "entry": entry}


def show_trend(days: int = 14) -> Dict[str, Any]:
    """Show recent reward trend data."""
    if not REWARD_TREND_PATH.exists():
        return {"status": "no_data", "entries": []}

    cutoff = datetime.now(timezone.utc) - timedelta(days=days)
    entries = []
    with open(REWARD_TREND_PATH) as f:
        for line in f:
            try:
                r = json.loads(line)
                ts = datetime.fromisoformat(r["ts"])
                if ts >= cutoff:
                    entries.append(r)
            except (json.JSONDecodeError, KeyError, ValueError):
                continue

    means = [e["mean"] for e in entries if "mean" in e]
    return {
        "status": "ok",
        "entries": entries,
        "count": len(entries),
        "rolling_mean": round(sum(means) / len(means), 4) if means else None,
    }


if __name__ == "__main__":
    as_json = "--json" in sys.argv

    if "--backfill-quality" in sys.argv:
        dry = "--dry-run" in sys.argv
        result = backfill_quality_grades(dry_run=dry)
        print(json.dumps(result, indent=2, default=str))
    elif "--archive" in sys.argv:
        dry = "--dry-run" in sys.argv
        days = 30
        for i, arg in enumerate(sys.argv):
            if arg == "--days" and i + 1 < len(sys.argv):
                days = int(sys.argv[i + 1])
        result = archive_old_records(days=days, dry_run=dry)
        print(json.dumps(result, indent=2, default=str))
    elif "--dedup" in sys.argv:
        dry = "--dry-run" in sys.argv
        result = dedup_trajectories(dry_run=dry)
        print(json.dumps(result, indent=2, default=str))
    elif "--integrity" in sys.argv:
        result = check_integrity()
        print(json.dumps(result, indent=2, default=str))
    elif "--backfill-agreement" in sys.argv:
        result = backfill_shadow_agreement()
        print(json.dumps(result, indent=2, default=str))
    elif "--shadow" in sys.argv:
        result = analyze_shadow_routing()
        print(json.dumps(result, indent=2, default=str))
    elif "--techniques" in sys.argv:
        result = technique_recommendations()
        print(json.dumps(result, indent=2, default=str))
    elif "--promote" in sys.argv:
        result = check_promotion_readiness()
        print(json.dumps(result, indent=2, default=str))
    elif "--auto-promote" in sys.argv:
        result = auto_promote()
        print(json.dumps(result, indent=2, default=str))
    elif "--health" in sys.argv:
        result = analyze_skill_health()
        print(json.dumps(result, indent=2, default=str))
    elif "--skill-readiness" in sys.argv:
        result = per_skill_readiness()
        print(json.dumps(result, indent=2, default=str))
    elif "--log-accuracy" in sys.argv:
        result = log_accuracy_trend()
        print(json.dumps(result, indent=2, default=str))
    elif "--log-trend" in sys.argv:
        result = log_reward_trend()
        print(json.dumps(result, indent=2, default=str))
    elif "--trend" in sys.argv:
        days = 14
        for i, arg in enumerate(sys.argv):
            if arg == "--days" and i + 1 < len(sys.argv):
                days = int(sys.argv[i + 1])
        result = show_trend(days=days)
        print(json.dumps(result, indent=2, default=str))
    else:
        print(full_report(as_json=as_json))
