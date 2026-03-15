"""
trajectory_tap.py — Lightweight trajectory recording for KARL.

Records tool-use sequences with outcome signals during Claude Code sessions.
Four tap points wire into existing hooks:
  Tap A: ops_trigger.py — init session buffer on skill injection
  Tap B: post_tool_hook.py — append tool event after each tool use
  Tap C: response_hook.py/session_end — flush buffer to trajectories.jsonl
  Tap D: ops_trigger.py — annotate previous record with cross-turn signals

Data flows: hooks → session buffer (JSON) → trajectories.jsonl (append-only)
"""

import fcntl
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

KARL_DIR = Path(__file__).parent
BUFFER_DIR = KARL_DIR / "buffers"
STORE_PATH = KARL_DIR / "trajectories.jsonl"

# Canonical writer: only this hostname writes to shared JSONL files.
# Other nodes write to per-session buffers only (synced via Syncthing).
# This prevents Syncthing conflict files from concurrent appends.
import socket as _socket
_CANONICAL_WRITER = os.environ.get("KARL_CANONICAL_WRITER", "claude-code-vm")
_HOSTNAME = _socket.gethostname()
IS_CANONICAL_WRITER = _HOSTNAME == _CANONICAL_WRITER or os.environ.get("KARL_FORCE_WRITE") == "1"

# Ensure directories exist
BUFFER_DIR.mkdir(parents=True, exist_ok=True)


# Skill inference patterns: (path_regex, skill_name, domain)
SKILL_PATTERNS = [
    # Creative / Evo
    (r"Desktop/(?:evo-cube-output|Comp-Core).*(?:evo|evolution)", "evo-cubed", "creative"),
    (r"Desktop/evo-cube-output", "evo-cubed", "creative"),
    (r"hef-evolutions/|hef/", "hef-evolution", "creative"),
    (r"Desktop/Frameworks", "frameworks", "creative"),
    # Infra / Ops
    (r"\.claude/cortex/", "cortex-ops", "infra"),
    (r"\.claude/karl/|Desktop/karl", "karl-trajectory", "infra"),
    (r"\.claude/hooks/", "hook-maintenance", "infra"),
    (r"monitoring/nexus-portal/", "nexus-portal", "web"),
    (r"monitoring/", "monitoring-ops", "infra"),
    (r"Desktop/mesh-dashboard", "mesh-dashboard", "infra"),
    (r"mesh-viz|mesh-boot|mesh-anchor", "mesh-ops", "infra"),
    (r"docker-compose|Dockerfile|\.service$", "deploy-ops", "infra"),
    (r"\.claude/skills/|skill-forge", "skill-forge", "infra"),
    (r"\.claude/orchestrator/", "pane-orchestrator", "systems"),
    (r"projects/account-pool/", "account-pool", "infra"),
    (r"projects/mesh-node-agent/", "mesh-node-agent", "infra"),
    (r"projects/self-healing-code/", "self-healing-code", "systems"),
    (r"projects/obsidian_vault_writer/", "vault-writer", "knowledge"),
    (r"projects/symphony/", "symphony", "systems"),
    (r"projects/ocp/", "ocp", "systems"),
    (r"projects/screen-capture/", "screen-capture", "infra"),
    # Automation
    (r"flows/feed-hub/", "feed-hub-flow", "automation"),
    # ML
    (r"projects/creator-shield/", "creator-shield", "ml"),
    (r"projects/agent-intelligence/", "agent-intelligence", "ml"),
    (r"Desktop/nko-brain-scanner", "nko-brain-scanner", "ml"),
    # Systems
    (r"projects/evolution_world/", "evolution-world", "systems"),
    (r"Desktop/Comp-Core", "comp-core", "systems"),
    # iOS
    (r"Desktop/OpenClawHub/", "openclaw-hub", "ios"),
    (r"Desktop/Spore/", "spore", "ios"),
    (r"Desktop/CreativeDirector/", "creative-director", "ios"),
    (r"Desktop/SecuriClaw", "securiclaw", "ios"),
    (r"Desktop/SpeakFlow/", "speakflow", "ios"),
    (r"Desktop/Serenity.Soother|Serenity Soother", "serenity-soother", "ios"),
    (r"Desktop/FirstDate", "firstdate", "ios"),
    (r"Desktop/BWB", "bwb", "ios"),
    (r"\.xcodeproj|\.swift$", "ios-build", "ios"),
    # Knowledge
    (r"obsidian-vault/", "vault-ops", "knowledge"),
    # Web
    (r"Desktop/Comp-Core/apps/web/learnnko|Desktop/learnnko|Desktop/NKo", "learnnko", "web"),
    (r"Desktop/Comp-Core/apps/web/cc-dashboard", "cc-dashboard", "web"),
    (r"Desktop/milkmendelivery|Meaning Full Power", "milkmen", "web"),
    # Desktop
    (r"Desktop/Comp-Core/apps/tauri", "tauri-desktop", "desktop"),
    # Data
    (r"supabase|migration", "supabase-ops", "data"),
    (r"projects/discrawl/", "discrawl", "data"),
]

# Tool-pattern inference: (tool_name_set, skill_name, domain)
TOOL_SKILL_PATTERNS = [
    ({"Bash"}, "shell-ops", "infra"),
]


def _infer_skill(buffer: dict) -> tuple:
    """Infer skill name and domain from file paths and tool patterns.

    Returns (skill_name, skill_domain) or (None, None).
    """
    import re

    # Already has skill? Return it.
    if buffer.get("skill_name"):
        return buffer["skill_name"], buffer.get("skill_domain")

    # Collect all file paths from tool events
    file_paths = []
    tool_names = set()
    for event in buffer.get("tool_events", []):
        tool_names.add(event.get("tool_name", ""))
        for key in ("file_path", "command", "pattern"):
            val = event.get("key_params", {}).get(key, "")
            if val:
                file_paths.append(val)

    # Also check cwd and prompt
    if buffer.get("cwd"):
        file_paths.append(buffer["cwd"])
    if buffer.get("prompt_text"):
        file_paths.append(buffer["prompt_text"])

    # Score each pattern by frequency of matches
    scores = {}
    for path in file_paths:
        for pattern, skill, domain in SKILL_PATTERNS:
            if re.search(pattern, path, re.IGNORECASE):
                key = (skill, domain)
                scores[key] = scores.get(key, 0) + 1

    if scores:
        best = max(scores, key=scores.get)
        return best

    return None, None


def _session_buffer_path(session_id: str) -> Path:
    """Get the buffer file path for a session."""
    safe_id = session_id.replace("/", "_").replace("..", "_")[:64]
    return BUFFER_DIR / f"{safe_id}.json"


def init_session_buffer(
    session_id: str,
    skill_name: Optional[str] = None,
    skill_domain: Optional[str] = None,
    prompt_text: Optional[str] = None,
    cwd: Optional[str] = None,
    git_repo: Optional[str] = None,
) -> bool:
    """
    Tap A: Initialize a session buffer when a skill is injected or a session starts.
    Called from ops_trigger.py after skill injection.
    Returns True if buffer was created successfully.
    """
    buf_path = _session_buffer_path(session_id)

    buffer = {
        "session_id": session_id,
        "started_at": datetime.now(timezone.utc).isoformat(),
        "skill_name": skill_name,
        "skill_domain": skill_domain,
        "prompt_text": (prompt_text or "")[:500],
        "cwd": cwd,
        "git_repo": git_repo,
        "tool_events": [],
        "tap_a_ts": datetime.now(timezone.utc).isoformat(),
    }

    try:
        with open(buf_path, "w") as f:
            json.dump(buffer, f, default=str)
        return True
    except Exception:
        return False


def append_tool_event(
    session_id: str,
    tool_name: str,
    tool_input: Optional[Dict[str, Any]] = None,
    success: Optional[bool] = None,
    exit_code: Optional[int] = None,
    duration_ms: Optional[float] = None,
) -> bool:
    """
    Tap B: Append a tool event to the session buffer.
    Called from post_tool_hook.py after each tool use.
    Lightweight: just appends to the JSON buffer file.
    """
    buf_path = _session_buffer_path(session_id)
    if not buf_path.exists():
        # No buffer initialized — this session wasn't triggered by a skill
        # Auto-create a minimal buffer so we still capture trajectories
        init_session_buffer(session_id)

    try:
        with open(buf_path, "r") as f:
            buffer = json.load(f)
    except (json.JSONDecodeError, FileNotFoundError):
        return False

    # Extract key parameters (not the full content, to keep buffer small)
    key_params = {}
    if tool_input:
        for key in ("file_path", "command", "pattern", "query", "url", "description"):
            if key in tool_input:
                val = str(tool_input[key])
                key_params[key] = val[:200]

    event = {
        "tool_name": tool_name,
        "key_params": key_params,
        "success": success,
        "exit_code": exit_code,
        "duration_ms": duration_ms,
        "ts": datetime.now(timezone.utc).isoformat(),
    }

    buffer["tool_events"].append(event)

    try:
        with open(buf_path, "w") as f:
            json.dump(buffer, f, default=str)
        return True
    except Exception:
        return False


def flush_session(
    session_id: str,
    outcome_signals: Optional[Dict[str, Any]] = None,
) -> Optional[Dict]:
    """
    Tap C: Flush session buffer to a trajectory record and append to store.
    Called from response_hook.py (Stop event) or session_end_hook.py.

    Returns the trajectory record if successful, None otherwise.
    """
    buf_path = _session_buffer_path(session_id)
    if not buf_path.exists():
        return None

    try:
        with open(buf_path, "r") as f:
            buffer = json.load(f)
    except (json.JSONDecodeError, FileNotFoundError):
        return None

    tool_events = buffer.get("tool_events", [])
    if not tool_events:
        # No tools used — skip recording
        _cleanup_buffer(buf_path)
        return None

    # Build tool sequence summary
    tool_names = [e.get("tool_name", "?") for e in tool_events]
    tool_counts = {}
    for name in tool_names:
        tool_counts[name] = tool_counts.get(name, 0) + 1

    # Count successes/failures
    successes = sum(1 for e in tool_events if e.get("success") is True)
    failures = sum(1 for e in tool_events if e.get("success") is False)
    bash_errors = sum(
        1 for e in tool_events
        if e.get("tool_name") == "Bash" and e.get("exit_code", 0) != 0
    )

    # Compute duration
    started_at = buffer.get("started_at", "")
    ended_at = datetime.now(timezone.utc).isoformat()
    try:
        start_dt = datetime.fromisoformat(started_at)
        end_dt = datetime.fromisoformat(ended_at)
        duration_s = (end_dt - start_dt).total_seconds()
    except (ValueError, TypeError):
        duration_s = None

    # Default outcome signals
    if outcome_signals is None:
        outcome_signals = {}

    # Derive build_success from buffer build_signals (set by post_tool_hook)
    build_signals = buffer.get("build_signals", [])
    if build_signals and "build_success" not in outcome_signals:
        # Overall build success = all build commands succeeded
        outcome_signals["build_success"] = all(s.get("success") for s in build_signals)

    # Infer skill if not explicitly set
    skill_name, skill_domain = _infer_skill(buffer)

    record = {
        "id": f"traj_{session_id[:8]}_{int(time.time())}",
        "session_id": session_id,
        "channel": "live",
        "recorded_at": ended_at,
        "skill": {
            "name": skill_name,
            "domain": skill_domain,
        },
        "context": {
            "prompt_text": buffer.get("prompt_text", ""),
            "cwd": buffer.get("cwd"),
            "git_repo": buffer.get("git_repo"),
        },
        "trajectory": {
            "tool_sequence": tool_names,
            "tool_counts": tool_counts,
            "total_tools": len(tool_events),
            "successes": successes,
            "failures": failures,
            "bash_errors": bash_errors,
            "events": tool_events[:50],  # Cap at 50 events for storage
        },
        "outcome": {
            "annotation_status": "pending" if not outcome_signals else "annotated",
            "correction_detected": outcome_signals.get("correction_detected"),
            "build_success": outcome_signals.get("build_success"),
            "redo_detected": outcome_signals.get("redo_detected"),
            "session_continued": outcome_signals.get("session_continued"),
            "reward_score": outcome_signals.get("reward_score"),
        },
        "timing": {
            "started_at": started_at,
            "ended_at": ended_at,
            "duration_s": duration_s,
        },
    }

    # Session quality scoring
    total_tools = len(tool_events)
    success_rate = successes / total_tools if total_tools else 0
    error_rate = (failures + bash_errors) / total_tools if total_tools else 0
    correction_count = sum(1 for e in tool_events if e.get("is_correction"))
    quality = "high"
    if error_rate > 0.3 or correction_count > 2:
        quality = "low"
    elif error_rate > 0.15 or correction_count > 0:
        quality = "medium"

    record["quality"] = {
        "grade": quality,
        "success_rate": round(success_rate, 3),
        "error_rate": round(error_rate, 3),
        "correction_count": correction_count,
        "tool_diversity": len(tool_counts),
    }

    # Enrich with Cortex bridge data (domain, corrections, success inference)
    try:
        from cortex_karl_bridge import bridge_domain, bridge_corrections, infer_success
        cortex_skill, cortex_domain = bridge_domain(record)
        if cortex_skill and not record["skill"].get("name"):
            record["skill"]["name"] = cortex_skill
        if cortex_domain and not record["skill"].get("domain"):
            record["skill"]["domain"] = cortex_domain

        corrections = bridge_corrections(record)
        if corrections:
            record["outcome"]["correction_detected"] = True
            record["outcome"]["corrections_in_window"] = len(corrections)

        inferred = infer_success(record)
        if inferred is not None:
            record["outcome"]["success_inferred"] = inferred
    except Exception:
        pass  # Bridge enrichment is optional

    # Compute reward score before storing
    try:
        from reward_engine import compute_reward, compute_advantage
        reward_data = compute_reward(record)
        record["outcome"]["reward_score"] = reward_data["reward_score"]
        record["outcome"]["reward_components"] = reward_data["components"]
        record["outcome"]["outcome_score"] = reward_data["outcome_score"]
        record["outcome"]["process_score"] = reward_data["process_score"]
        record["outcome"]["efficiency_score"] = reward_data["efficiency_score"]
        record["outcome"]["advantage"] = round(
            compute_advantage(record, reward_data["reward_score"]), 4
        )
        record["outcome"]["annotation_status"] = "scored"
    except Exception:
        pass  # Store without reward if engine fails

    # Shadow routing: record what vector routing would predict for this prompt
    # This supplements the UserPromptSubmit hook (which only fires interactively)
    try:
        prompt_text = record.get("context", {}).get("prompt_text", "")
        if prompt_text and len(prompt_text) > 10:
            from embedding_cache import (
                load_skill_embeddings, rank_skills, _fetch_embedding,
                build_prompt_embedding_text,
            )
            skill_embs = load_skill_embeddings()
            if skill_embs:
                cwd = record.get("context", {}).get("cwd", "")
                embed_text = build_prompt_embedding_text(prompt_text, cwd)
                embedding = _fetch_embedding(embed_text, timeout=8)
                if embedding:
                    rankings = rank_skills(embedding, skill_embs, threshold=0.0)
                    vector_skill = rankings[0][0] if rankings else None
                    vector_sim = rankings[0][1] if rankings else 0
                    top_k = [{"skill": n, "score": round(s, 4)} for n, s in rankings[:3]]
                    top3_names = [r[0] for r in rankings[:3]]
                    # actual_skill: use regex-inferred skill_name as ground truth
                    actual = skill_name or record.get("skill", {}).get("name")
                    reciprocal_rank = 0.0
                    if actual:
                        for ri, (rn, _) in enumerate(rankings):
                            if rn == actual:
                                reciprocal_rank = 1.0 / (ri + 1)
                                break
                    shadow_entry = {
                        "session_id": session_id[:16],
                        "regex": skill_name,
                        "vector": vector_skill,
                        "actual_skill": actual,
                        "vector_status": "hit" if vector_sim > 0 else "miss",
                        "similarity": round(vector_sim, 4),
                        "top_k": top_k,
                        "in_top3": actual in top3_names if actual else None,
                        "reciprocal_rank": round(reciprocal_rank, 4),
                        "agree": actual == vector_skill if actual and vector_skill else None,
                        "vector_correct": actual == vector_skill if actual else None,
                        "regex_correct": True if skill_name and actual == skill_name else None,
                        "elapsed_ms": 0,
                        "source": "trajectory_tap",
                        "ts": datetime.now(timezone.utc).isoformat(),
                    }
                    if IS_CANONICAL_WRITER:
                        shadow_path = KARL_DIR / "routing_shadow.jsonl"
                        with open(shadow_path, "a") as sf:
                            sf.write(json.dumps(shadow_entry, default=str) + "\n")
    except Exception:
        pass  # Shadow routing is optional, never block trajectory recording

    # Append to store
    if append_to_store(record):
        _cleanup_buffer(buf_path)
        return record
    return None


def append_to_store(record: Dict) -> bool:
    """Append a trajectory record to trajectories.jsonl with file locking.

    Only writes on the canonical writer node (cloud-vm) to prevent
    Syncthing conflicts. Non-canonical nodes keep data in session buffers
    which sync to the canonical node for eventual flush.
    """
    if not IS_CANONICAL_WRITER:
        return True  # Silently skip — buffer will sync and flush on canonical node
    try:
        with open(STORE_PATH, "a") as f:
            fcntl.flock(f.fileno(), fcntl.LOCK_EX)
            try:
                f.write(json.dumps(record, default=str) + "\n")
            finally:
                fcntl.flock(f.fileno(), fcntl.LOCK_UN)
        return True
    except Exception:
        return False


def annotate_previous(
    session_id: str,
    correction_detected: Optional[bool] = None,
    redo_detected: Optional[bool] = None,
) -> bool:
    """
    Tap D: Annotate the previous trajectory record with cross-turn signals.
    Called at the start of the next UserPromptSubmit in ops_trigger.py.
    Updates the most recent record for this session in trajectories.jsonl.
    """
    if not STORE_PATH.exists():
        return False

    try:
        with open(STORE_PATH, "r") as f:
            lines = f.readlines()
    except Exception:
        return False

    # Find the last record for this session
    updated = False
    for i in range(len(lines) - 1, -1, -1):
        try:
            record = json.loads(lines[i])
            if record.get("session_id") == session_id:
                outcome = record.get("outcome", {})
                if correction_detected is not None:
                    outcome["correction_detected"] = correction_detected
                if redo_detected is not None:
                    outcome["redo_detected"] = redo_detected
                # Update annotation status
                if any(v is not None for v in [correction_detected, redo_detected]):
                    outcome["annotation_status"] = "annotated"
                record["outcome"] = outcome

                # Re-score reward with new correction/redo signals
                try:
                    from reward_engine import compute_reward, compute_advantage
                    reward_data = compute_reward(record)
                    outcome["reward_score"] = reward_data["reward_score"]
                    outcome["reward_components"] = reward_data["components"]
                    outcome["outcome_score"] = reward_data["outcome_score"]
                    outcome["process_score"] = reward_data["process_score"]
                    outcome["advantage"] = round(
                        compute_advantage(record, reward_data["reward_score"]), 4
                    )
                    outcome["annotation_status"] = "re-scored"
                    record["outcome"] = outcome
                except Exception:
                    pass

                lines[i] = json.dumps(record, default=str) + "\n"
                updated = True
                break
        except json.JSONDecodeError:
            continue

    if updated:
        try:
            with open(STORE_PATH, "w") as f:
                fcntl.flock(f.fileno(), fcntl.LOCK_EX)
                try:
                    f.writelines(lines)
                finally:
                    fcntl.flock(f.fileno(), fcntl.LOCK_UN)
            return True
        except Exception:
            return False
    return False


def _cleanup_buffer(buf_path: Path):
    """Remove a session buffer file after flushing."""
    try:
        buf_path.unlink(missing_ok=True)
    except Exception:
        pass


def get_store_stats() -> Dict[str, Any]:
    """Get statistics about the trajectory store."""
    if not STORE_PATH.exists():
        return {"total": 0, "size_bytes": 0}

    total = 0
    channels = {}
    skills = {}
    with_reward = 0

    try:
        with open(STORE_PATH, "r") as f:
            for line in f:
                try:
                    record = json.loads(line)
                    total += 1
                    ch = record.get("channel", "unknown")
                    channels[ch] = channels.get(ch, 0) + 1
                    skill = record.get("skill", {}).get("name")
                    if skill:
                        skills[skill] = skills.get(skill, 0) + 1
                    if record.get("outcome", {}).get("reward_score") is not None:
                        with_reward += 1
                except json.JSONDecodeError:
                    continue
    except Exception:
        pass

    return {
        "total": total,
        "size_bytes": STORE_PATH.stat().st_size if STORE_PATH.exists() else 0,
        "channels": channels,
        "skills": skills,
        "with_reward": with_reward,
    }
