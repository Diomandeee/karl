#!/usr/bin/env python3
"""
shadow_seeder.py — DEPRECATED: Synthetic shadow seeding disabled as of 2026-03-15.

Reason: 93% of shadow records were synthetic, inflating accuracy metrics.
Vector routing showed 67.9% on seeded data but 0% on organic data.
KARL now accumulates only organic shadow records from real routing events.

The seed_shadow_records() and reshadow_records() functions still exist
for historical reference but will refuse to run unless KARL_ALLOW_SEEDING=1
is set in environment (for emergency use only).

Original description:
Reads trajectories.jsonl, extracts prompt text, runs vector routing on each to
generate shadow records. Only processes trajectories that don't already have
corresponding shadow entries (by session_id).
"""

import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

KARL_DIR = Path(__file__).parent
TRAJECTORY_PATH = KARL_DIR / "trajectories.jsonl"
SHADOW_PATH = KARL_DIR / "routing_shadow.jsonl"

sys.path.insert(0, str(KARL_DIR))


def load_existing_shadow_session_ids():
    """Get set of session_ids already in shadow log."""
    ids = set()
    if SHADOW_PATH.exists():
        with open(SHADOW_PATH) as f:
            for line in f:
                try:
                    r = json.loads(line)
                    sid = r.get("session_id", "")
                    if sid:
                        ids.add(sid)
                except json.JSONDecodeError:
                    continue
    return ids


def _build_synthetic_prompt(trajectory):
    """Build a prompt from trajectory metadata when prompt_text is missing."""
    parts = []

    # Use cwd to identify the project
    cwd = trajectory.get("context", {}).get("cwd", "")
    if cwd:
        # Extract project name from path
        path_parts = cwd.rstrip("/").split("/")
        project = path_parts[-1] if path_parts else ""
        if project and project not in (".", "~"):
            parts.append(f"Working in {project}")

    # Use skill name
    skill = trajectory.get("skill", {}).get("name", "")
    if skill:
        parts.append(f"task type: {skill}")

    # Use tool events to describe the activity
    events = trajectory.get("trajectory", {}).get("events", [])
    if events:
        tools_used = [e.get("tool_name", "") for e in events[:5] if e.get("tool_name")]
        if tools_used:
            parts.append(f"using {', '.join(tools_used)}")

        # Extract file paths for context
        files = []
        for e in events[:5]:
            fp = e.get("key_params", {}).get("file_path", "")
            if fp:
                fname = fp.split("/")[-1]
                if fname:
                    files.append(fname)
        if files:
            parts.append(f"on {', '.join(files[:3])}")

    return " ".join(parts)


def seed_shadow_records(dry_run=False):
    """Generate shadow records from trajectory prompts. DEPRECATED — synthetic seeding disabled."""
    import os
    if os.environ.get("KARL_ALLOW_SEEDING") != "1":
        return {"status": "blocked", "reason": "Synthetic seeding disabled (2026-03-15). Set KARL_ALLOW_SEEDING=1 to override."}
    from embedding_cache import load_skill_embeddings, cache_get, rank_skills, embed_sync

    trajectories = []
    with open(TRAJECTORY_PATH) as f:
        for line in f:
            try:
                trajectories.append(json.loads(line))
            except json.JSONDecodeError:
                continue

    existing_sids = load_existing_shadow_session_ids()
    skill_embs = load_skill_embeddings()

    if not skill_embs:
        return {"error": "No skill embeddings loaded. Run: python3 embedding_cache.py --refresh"}

    candidates = []
    for t in trajectories:
        sid = t.get("session_id", "")[:16]
        if not sid or sid in existing_sids:
            continue

        # Extract prompt text from various locations
        prompt = (
            t.get("context", {}).get("prompt_text", "")
            or t.get("context", {}).get("user_prompt", "")
            or ""
        )
        # Construct synthetic prompt from metadata when text is missing
        if len(prompt) < 10:
            prompt = _build_synthetic_prompt(t)
        if len(prompt) < 10:
            continue

        skill_name = t.get("skill", {}).get("name", "")
        candidates.append({
            "session_id": sid,
            "prompt": prompt[:4000],
            "actual_skill": skill_name,
        })

    result = {
        "total_trajectories": len(trajectories),
        "existing_shadow": len(existing_sids),
        "candidates": len(candidates),
        "dry_run": dry_run,
    }

    if dry_run or not candidates:
        return result

    # Generate shadow records
    seeded = 0
    errors = 0
    new_records = []

    for c in candidates:
        prompt = c["prompt"]

        # Try to get embedding (sync with longer timeout for seeding)
        embedding = embed_sync(prompt, timeout=5.0)
        if not embedding:
            errors += 1
            continue

        # Rank skills (get top-3 for richer analysis)
        rankings = rank_skills(embedding, skill_embs, threshold=0.0)
        vector_skill = rankings[0][0] if rankings else None
        vector_sim = rankings[0][1] if rankings else 0
        top_k = [{"skill": name, "score": round(score, 4)} for name, score in rankings[:3]]

        # Check if actual skill is in top-3
        top3_names = [r[0] for r in rankings[:3]]
        in_top3 = c["actual_skill"] in top3_names if c["actual_skill"] else None
        reciprocal_rank = 0
        if c["actual_skill"]:
            for rank_idx, (rname, _) in enumerate(rankings):
                if rname == c["actual_skill"]:
                    reciprocal_rank = 1.0 / (rank_idx + 1)
                    break

        shadow_entry = {
            "session_id": c["session_id"],
            "regex": c["actual_skill"],
            "vector": vector_skill,
            "vector_status": "hit" if vector_skill else "miss",
            "similarity": round(vector_sim, 4),
            "top_k": top_k,
            "in_top3": in_top3,
            "reciprocal_rank": round(reciprocal_rank, 4),
            "agree": c["actual_skill"] == vector_skill if c["actual_skill"] and vector_skill else None,
            "elapsed_ms": 0,
            "source": "shadow_seeder",
            "actual_skill": c["actual_skill"],
            "regex_correct": True,
            "vector_correct": vector_skill == c["actual_skill"] if c["actual_skill"] else None,
            "ts": datetime.now(timezone.utc).isoformat(),
        }
        new_records.append(shadow_entry)
        seeded += 1

        # Rate limit to avoid hammering embed API
        time.sleep(0.1)

    # Append to shadow file
    if new_records:
        with open(SHADOW_PATH, "a") as f:
            for r in new_records:
                f.write(json.dumps(r, separators=(",", ":"), default=str) + "\n")

    # Compute accuracy from seeded records
    annotated = [r for r in new_records if r.get("actual_skill")]
    vector_correct = sum(1 for r in annotated if r.get("vector_correct"))
    top3_hits = sum(1 for r in annotated if r.get("in_top3"))
    rr_sum = sum(r.get("reciprocal_rank", 0) for r in annotated)

    result["seeded"] = seeded
    result["errors"] = errors
    result["vector_accuracy"] = round(vector_correct / len(annotated), 4) if annotated else None
    result["top3_accuracy"] = round(top3_hits / len(annotated), 4) if annotated else None
    result["mrr"] = round(rr_sum / len(annotated), 4) if annotated else None
    result["total_shadow_after"] = len(existing_sids) + seeded

    return result


def reshadow_records(dry_run=True):
    """Re-evaluate all shadow records using current centroids. DEPRECATED — synthetic seeding disabled.

    Re-embeds prompts from trajectories and re-runs ranking with the
    current skill embeddings. Updates shadow records in-place with new
    predictions while preserving actual_skill annotations.
    """
    import os
    if os.environ.get("KARL_ALLOW_SEEDING") != "1":
        return {"status": "blocked", "reason": "Reshadow disabled (2026-03-15). Set KARL_ALLOW_SEEDING=1 to override."}
    from embedding_cache import load_skill_embeddings, embed_sync, rank_skills, build_prompt_embedding_text

    skill_embs = load_skill_embeddings()
    if not skill_embs:
        return {"error": "No skill embeddings loaded"}

    # Load trajectories by session_id for prompt lookup
    trajectory_prompts = {}
    with open(TRAJECTORY_PATH) as f:
        for line in f:
            try:
                t = json.loads(line)
                sid = t.get("session_id", "")[:16]
                prompt = (
                    t.get("context", {}).get("prompt_text", "")
                    or t.get("context", {}).get("user_prompt", "")
                    or ""
                )
                if len(prompt) < 10:
                    prompt = _build_synthetic_prompt(t)
                cwd = t.get("context", {}).get("cwd", "")
                if sid and len(prompt) >= 10:
                    trajectory_prompts[sid] = (prompt[:4000], cwd)
            except json.JSONDecodeError:
                continue

    # Load existing shadow records
    shadow_records = []
    if SHADOW_PATH.exists():
        with open(SHADOW_PATH) as f:
            for line in f:
                try:
                    shadow_records.append(json.loads(line))
                except json.JSONDecodeError:
                    continue

    updated = 0
    unchanged = 0
    skipped = 0
    improvements = 0
    regressions = 0

    for r in shadow_records:
        sid = r.get("session_id", "")
        actual = r.get("actual_skill", "")
        if not actual or sid not in trajectory_prompts:
            skipped += 1
            continue

        prompt, cwd = trajectory_prompts[sid]
        embed_text = build_prompt_embedding_text(prompt, cwd)
        embedding = embed_sync(embed_text, timeout=5.0)
        if not embedding:
            skipped += 1
            continue

        rankings = rank_skills(embedding, skill_embs, threshold=0.0)
        new_vector = rankings[0][0] if rankings else None
        new_sim = rankings[0][1] if rankings else 0
        new_top_k = [{"skill": name, "score": round(score, 4)} for name, score in rankings[:3]]
        new_correct = new_vector == actual

        old_correct = r.get("vector_correct", False)
        if new_correct != old_correct:
            if new_correct:
                improvements += 1
            else:
                regressions += 1

        # Update record
        r["vector"] = new_vector
        r["similarity"] = round(new_sim, 4)
        r["top_k"] = new_top_k
        r["vector_correct"] = new_correct
        r["agree"] = new_vector == r.get("regex")
        top3_names = [rk[0] for rk in rankings[:3]]
        r["in_top3"] = actual in top3_names
        rr = 0
        for idx, (rname, _) in enumerate(rankings):
            if rname == actual:
                rr = 1.0 / (idx + 1)
                break
        r["reciprocal_rank"] = round(rr, 4)
        r["reshadowed_at"] = datetime.now(timezone.utc).isoformat()
        updated += 1

        time.sleep(0.05)

    # Compute new accuracy
    annotated = [r for r in shadow_records if r.get("actual_skill")]
    new_correct_count = sum(1 for r in annotated if r.get("vector_correct"))
    new_top3 = sum(1 for r in annotated if r.get("in_top3"))
    new_mrr = sum(r.get("reciprocal_rank", 0) for r in annotated)

    result = {
        "total_shadow": len(shadow_records),
        "updated": updated,
        "skipped": skipped,
        "improvements": improvements,
        "regressions": regressions,
        "net_improvement": improvements - regressions,
        "new_accuracy": round(new_correct_count / len(annotated), 4) if annotated else None,
        "new_top3": round(new_top3 / len(annotated), 4) if annotated else None,
        "new_mrr": round(new_mrr / len(annotated), 4) if annotated else None,
        "dry_run": dry_run,
    }

    if not dry_run and updated > 0:
        with open(SHADOW_PATH, "w") as f:
            for r in shadow_records:
                f.write(json.dumps(r, separators=(",", ":"), default=str) + "\n")

    return result


def main():
    if "--dry-run" in sys.argv:
        result = seed_shadow_records(dry_run=True)
        print(json.dumps(result, indent=2, default=str))
    elif "--seed" in sys.argv or "--annotate" in sys.argv:
        result = seed_shadow_records(dry_run=False)
        print(json.dumps(result, indent=2, default=str))

        if "--annotate" in sys.argv:
            from trajectory_bridge import backfill_shadow_agreement
            ann = backfill_shadow_agreement()
            print("Annotation:", json.dumps(ann, indent=2, default=str))
    else:
        print(__doc__)


if __name__ == "__main__":
    main()
