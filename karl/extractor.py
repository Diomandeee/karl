"""
extractor.py - Historical trajectory backfill from verbose prompt logs.

Extracts tool-use trajectories from existing session logs, normalizing
tool names across different agent formats (Codex, Claude Code).

Usage:
    from karl.extractor import extract_trajectories
    trajectories = extract_trajectories()
    trajectories = extract_trajectories(dry_run=True)
"""

import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from karl.config import VERBOSE_LOG, STORE_PATH
from karl.trajectory_tap import append_to_store

# Tool name normalization: external format -> Claude Code format
TOOL_NAME_MAP = {
    "exec_command": "Bash",
    "shell_command": "Bash",
    "apply_patch": "Edit",
    "apply_diff": "Edit",
    "read_file": "Read",
    "list_files": "Glob",
    "search_files": "Grep",
    "write_file": "Write",
    "create_file": "Write",
}


def normalize_tool_name(name: str) -> str:
    """Normalize a tool name to Claude Code vocabulary."""
    return TOOL_NAME_MAP.get(name, name)


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

    # Load existing session IDs to avoid duplicates
    existing_sessions: set = set()
    if STORE_PATH.exists():
        with open(STORE_PATH, "r") as f:
            for line in f:
                try:
                    record = json.loads(line)
                    existing_sessions.add(record.get("session_id", ""))
                except json.JSONDecodeError:
                    continue

    trajectories = []
    total_entries = 0
    skipped_no_tools = 0
    skipped_duplicate = 0

    with open(path, "r") as f:
        for line in f:
            total_entries += 1
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                continue

            session_id = entry.get("session_id", "")
            turns = entry.get("assistant_turns", [])

            all_tool_calls = []
            for turn in turns:
                for tc in turn.get("tool_calls", []):
                    all_tool_calls.append(tc)

            if not all_tool_calls:
                skipped_no_tools += 1
                continue

            if session_id in existing_sessions:
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
                params = tc.get("parameters", {})
                key_params = {}
                for key in ("file_path", "command", "pattern", "query", "path"):
                    if key in params:
                        key_params[key] = str(params[key])[:200]

                success = tc.get("success")
                exit_code = tc.get("exit_code")
                if success is None and exit_code is not None:
                    success = exit_code == 0

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
                if e.get("tool_name") == "Bash" and e.get("exit_code", 0) != 0
            )

            prompt_text = entry.get("prompt_text", "")[:500]
            env = entry.get("environment", {})
            timing = entry.get("timing", {})

            record = {
                "id": f"traj_bf_{session_id[:8]}_{total_entries}",
                "session_id": session_id,
                "channel": "backfill",
                "recorded_at": datetime.now(timezone.utc).isoformat(),
                "skill": {"name": None, "domain": None},
                "context": {
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
                    "started_at": entry.get("prompt_timestamp", ""),
                    "ended_at": entry.get("response_timestamp", ""),
                    "duration_s": (
                        timing.get("total_duration_ms", 0) / 1000
                        if isinstance(timing, dict) and timing.get("total_duration_ms")
                        else None
                    ),
                },
            }

            trajectories.append(record)
            existing_sessions.add(session_id)

    if not dry_run:
        written = 0
        for record in trajectories:
            if append_to_store(record):
                written += 1

    return trajectories
