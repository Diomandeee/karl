"""
extractor.py - Historical trajectory backfill from verbose prompt logs.

Extracts tool-use trajectories from existing session logs, normalizing
tool names across different agent formats (Codex, Claude Code).

Usage:
    from karl.extractor import extract_trajectories
    trajectories = extract_trajectories()
    trajectories = extract_trajectories(dry_run=True)
"""

import fcntl
import hashlib
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from karl.config import VERBOSE_LOG, STORE_PATH
from karl.schema import normalize_record, normalize_tool_name

def _tool_params(tool_call: Dict[str, Any]) -> Dict[str, Any]:
    params = (
        tool_call.get("parameters")
        or tool_call.get("tool_input")
        or tool_call.get("input")
        or {}
    )
    return params if isinstance(params, dict) else {}


def _tool_success(tool_call: Dict[str, Any]) -> Optional[bool]:
    success = tool_call.get("success")
    if success is not None:
        return bool(success)
    if tool_call.get("is_error") is not None:
        return not bool(tool_call.get("is_error"))
    exit_code = tool_call.get("exit_code")
    if exit_code is not None:
        return exit_code == 0
    return None


def _legacy_tool_calls(tool_calls: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Build the old mega_extract tool-call shape for dedupe compatibility."""
    legacy = []
    for tc in tool_calls:
        legacy.append({
            "tool": normalize_tool_name(tc.get("tool_name", "")),
            # mega_extract used tool_input. Newer prompt logs usually store
            # parameters instead, so missing tool_input must stay {} to match
            # the already-harvested May corpus hashes.
            "input_preview": str(tc.get("tool_input", {}))[:200],
            "success": not tc.get("is_error", False),
            "duration_ms": tc.get("duration_ms", 0),
        })
    return legacy


def _legacy_content_hash(prompt: str, legacy_calls: List[Dict[str, Any]]) -> str:
    payload = f"{prompt[:500]}{str(legacy_calls)[:500]}"
    return hashlib.md5(payload.encode()).hexdigest()


def _record_signature(record: Dict[str, Any]) -> Optional[str]:
    context = record.get("context", {})
    if isinstance(context, dict) and context.get("source_hash"):
        return str(context["source_hash"])

    trajectory = record.get("trajectory", {})
    if not isinstance(trajectory, dict):
        return None

    if "prompt" in trajectory or "tool_calls" in trajectory:
        return _legacy_content_hash(
            str(trajectory.get("prompt", "")),
            trajectory.get("tool_calls", []) if isinstance(trajectory.get("tool_calls"), list) else [],
        )

    prompt = ""
    if isinstance(context, dict):
        prompt = str(context.get("prompt_text", ""))
    events = trajectory.get("events", [])
    if not prompt and not events:
        return None
    payload = json.dumps(
        {
            "prompt": prompt[:500],
            "events": events[:50],
            "session_id": record.get("session_id", ""),
        },
        sort_keys=True,
        default=str,
    )
    return hashlib.sha256(payload.encode()).hexdigest()


def _load_existing_signatures() -> set[str]:
    signatures: set[str] = set()
    if not STORE_PATH.exists():
        return signatures
    with open(STORE_PATH, "r", encoding="utf-8") as f:
        for line in f:
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            signature = _record_signature(record)
            if signature:
                signatures.add(signature)
    return signatures


def _append_direct(record: Dict[str, Any]) -> bool:
    """Append directly to the repo store.

    trajectory_tap.append_to_store is intentionally gated to one canonical
    host. Batch harvesting is an explicit local maintenance operation, so it
    needs its own locked append path instead of silently returning success on
    non-canonical hosts.
    """
    STORE_PATH.parent.mkdir(parents=True, exist_ok=True)
    try:
        with open(STORE_PATH, "a", encoding="utf-8") as f:
            fcntl.flock(f.fileno(), fcntl.LOCK_EX)
            try:
                f.write(json.dumps(record, default=str) + "\n")
                f.flush()
            finally:
                fcntl.flock(f.fileno(), fcntl.LOCK_UN)
        return True
    except OSError:
        return False


def extract_trajectories(
    verbose_path: Optional[Path] = None,
    dry_run: bool = False,
) -> List[Dict]:
    """Extract trajectories from verbose prompt logs.

    Args:
        verbose_path: Path to verbose-all.jsonl (default from config)
        dry_run: Preview without writing to store

    Returns:
        List of extracted trajectory records
    """
    path = verbose_path or VERBOSE_LOG
    if not path.exists():
        return []

    existing_signatures = _load_existing_signatures()

    trajectories = []
    total_entries = 0
    skipped_no_tools = 0
    skipped_duplicate = 0

    with open(path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            total_entries += 1
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                continue

            session_id = entry.get("session_id", "")
            prompt_text_full = entry.get("prompt_text", "")
            turns = entry.get("assistant_turns", [])

            all_tool_calls = []
            for turn in turns:
                for tc in turn.get("tool_calls", []):
                    all_tool_calls.append(tc)

            if not all_tool_calls:
                skipped_no_tools += 1
                continue

            legacy_calls = _legacy_tool_calls(all_tool_calls)
            source_hash = _legacy_content_hash(prompt_text_full, legacy_calls)
            if source_hash in existing_signatures:
                skipped_duplicate += 1
                continue

            tool_names = [normalize_tool_name(tc.get("tool_name", "?")) for tc in all_tool_calls]
            tool_counts: Dict[str, int] = {}
            for name in tool_names:
                tool_counts[name] = tool_counts.get(name, 0) + 1

            events = []
            for tc in all_tool_calls[:50]:
                raw_name = tc.get("tool_name", "?")
                norm_name = normalize_tool_name(raw_name)
                params = _tool_params(tc)
                key_params = {}
                for key in ("file_path", "command", "pattern", "query", "path", "description"):
                    if key in params:
                        key_params[key] = str(params[key])[:200]

                success = _tool_success(tc)
                exit_code = tc.get("exit_code")

                events.append({
                    "tool_name": norm_name,
                    "original_name": raw_name if raw_name != norm_name else None,
                    "key_params": key_params,
                    "success": success,
                    "exit_code": exit_code,
                    "ts": tc.get("timestamp", ""),
                })

            successes = sum(1 for e in events if e.get("success") is True)
            failures = sum(1 for e in events if e.get("success") is False)
            bash_errors = sum(
                1 for e in events
                if e.get("tool_name") == "Bash"
                and (e.get("success") is False or e.get("exit_code") not in (None, 0))
            )

            prompt_text = prompt_text_full[:500]
            env = entry.get("environment", {})
            timing = entry.get("timing", {})
            domain = entry.get("orbit_project_name") or entry.get("git_repo") or "unknown"
            started_at = (
                entry.get("prompt_timestamp")
                or entry.get("captured_at")
                or entry.get("timestamp")
                or ""
            )
            ended_at = (
                entry.get("response_timestamp")
                or entry.get("captured_at")
                or entry.get("timestamp")
                or ""
            )

            record = {
                "schema_version": 2,
                "id": f"traj_bf_{source_hash[:16]}",
                "session_id": session_id,
                "source": "verbose-all",
                "channel": "backfill",
                "recorded_at": datetime.now(timezone.utc).isoformat(),
                "skill": {"name": None, "domain": None},
                "domain": domain,
                "context": {
                    "source": "verbose-all",
                    "source_hash": source_hash,
                    "prompt_log_line": total_entries,
                    "prompt_text": prompt_text,
                    "cwd": env.get("cwd") if isinstance(env, dict) else None,
                    "git_repo": entry.get("git_repo"),
                },
                "trajectory": {
                    "tool_sequence": tool_names,
                    "tool_counts": tool_counts,
                    "total_tools": len(tool_names),
                    "successes": successes,
                    "failures": failures,
                    "bash_errors": bash_errors,
                    "events": events,
                },
                "outcome": {
                    "annotation_status": "pending",
                    "correction_detected": None,
                    "build_success": None,
                    "redo_detected": None,
                    "session_continued": None,
                    "reward_score": None,
                },
                "timing": {
                    "started_at": started_at,
                    "ended_at": ended_at,
                    "duration_s": (
                        timing.get("total_duration_ms", 0) / 1000
                        if isinstance(timing, dict) and timing.get("total_duration_ms")
                        else None
                    ),
                },
            }

            trajectories.append(normalize_record(record))
            existing_signatures.add(source_hash)

    if not dry_run:
        written = 0
        for record in trajectories:
            if _append_direct(record):
                written += 1
        print(
            f"[extractor] scanned={total_entries} no_tools={skipped_no_tools} "
            f"duplicates={skipped_duplicate} extracted={len(trajectories)} "
            f"written={written}"
        )
    else:
        print(
            f"[extractor] scanned={total_entries} no_tools={skipped_no_tools} "
            f"duplicates={skipped_duplicate} extracted={len(trajectories)}"
        )

    return trajectories
