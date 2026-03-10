"""
trajectory_tap.py - Lightweight trajectory recording for KARL.

Records tool-use sequences with outcome signals during AI coding sessions.
Four tap points wire into the agent's hook system:

  Tap A: init_session_buffer    - Initialize on session/skill start
  Tap B: append_tool_event      - Record each tool use
  Tap C: flush_session          - Flush buffer + compute reward on completion
  Tap D: annotate_previous      - Cross-turn correction/redo detection

Data flows: hooks -> session buffer (JSON) -> trajectories.jsonl (append-only)
"""

import fcntl
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from karl.config import BUFFER_DIR, STORE_PATH, DATA_DIR

# Ensure directories exist on import
DATA_DIR.mkdir(parents=True, exist_ok=True)
BUFFER_DIR.mkdir(parents=True, exist_ok=True)


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
    Tap A: Initialize a session buffer when a skill is injected or session starts.

    Args:
        session_id: Unique session identifier
        skill_name: Name of the activated skill (e.g., "ops:deploy")
        skill_domain: Domain category of the skill
        prompt_text: The user's prompt (truncated to 500 chars)
        cwd: Current working directory
        git_repo: Git repository name

    Returns:
        True if buffer was created successfully
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

    Called after each tool use (Read, Edit, Write, Bash, Grep, Glob, Task, etc.).
    Extracts key parameters to keep the buffer compact.

    Args:
        session_id: Session to append to
        tool_name: Name of the tool (e.g., "Read", "Bash", "Edit")
        tool_input: Tool call parameters (key fields extracted)
        success: Whether the tool call succeeded
        exit_code: Exit code for Bash commands
        duration_ms: How long the tool call took

    Returns:
        True if event was appended successfully
    """
    buf_path = _session_buffer_path(session_id)
    if not buf_path.exists():
        init_session_buffer(session_id)

    try:
        with open(buf_path, "r") as f:
            buffer = json.load(f)
    except (json.JSONDecodeError, FileNotFoundError):
        return False

    # Extract key parameters (not full content, to keep buffer small)
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

    Computes reward score before writing. Cleans up the buffer file.

    Args:
        session_id: Session to flush
        outcome_signals: Optional pre-computed outcome signals

    Returns:
        The trajectory record if successful, None otherwise
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
        _cleanup_buffer(buf_path)
        return None

    # Build tool sequence summary
    tool_names = [e.get("tool_name", "?") for e in tool_events]
    tool_counts: Dict[str, int] = {}
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

    if outcome_signals is None:
        outcome_signals = {}

    record = {
        "id": f"traj_{session_id[:8]}_{int(time.time())}",
        "session_id": session_id,
        "channel": "live",
        "recorded_at": ended_at,
        "skill": {
            "name": buffer.get("skill_name"),
            "domain": buffer.get("skill_domain"),
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

    # Compute reward score before storing
    try:
        from karl.reward_engine import compute_reward, compute_advantage
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

    if append_to_store(record):
        _cleanup_buffer(buf_path)
        return record
    return None


def append_to_store(record: Dict) -> bool:
    """Append a trajectory record to trajectories.jsonl with file locking."""
    try:
        STORE_PATH.parent.mkdir(parents=True, exist_ok=True)
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

    Called at the start of the next prompt to detect corrections ("no, I meant...",
    "try again", "that is wrong") which retroactively signal the previous
    trajectory produced a poor result.

    Args:
        session_id: Session whose last trajectory to annotate
        correction_detected: Whether user corrected the previous response
        redo_detected: Whether user asked for a redo

    Returns:
        True if annotation was applied successfully
    """
    if not STORE_PATH.exists():
        return False

    try:
        with open(STORE_PATH, "r") as f:
            lines = f.readlines()
    except Exception:
        return False

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
                if any(v is not None for v in [correction_detected, redo_detected]):
                    outcome["annotation_status"] = "annotated"
                record["outcome"] = outcome
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


def _cleanup_buffer(buf_path: Path) -> None:
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
    channels: Dict[str, int] = {}
    skills: Dict[str, int] = {}
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
