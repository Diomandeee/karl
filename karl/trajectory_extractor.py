#!/usr/bin/env python3
"""
trajectory_extractor.py — Historical backfill from verbose-all.jsonl.

Extracts tool-use trajectories from existing Codex and Claude Code sessions
in verbose-all.jsonl. Normalizes tool names to Claude Code vocabulary.

Usage:
    python3 trajectory_extractor.py              # Extract all
    python3 trajectory_extractor.py --dry-run    # Preview only
    python3 trajectory_extractor.py --stats      # Show store stats
"""

import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

VERBOSE_PATH = Path.home() / ".claude" / "prompt-logs" / "verbose-all.jsonl"
KARL_DIR = Path(__file__).parent
STORE_PATH = KARL_DIR / "trajectories.jsonl"

# Codex tool name → Claude Code tool name mapping
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


def extract_trajectories(dry_run: bool = False) -> List[Dict]:
    """Extract trajectories from verbose-all.jsonl."""
    if not VERBOSE_PATH.exists():
        print(f"[extractor] verbose-all.jsonl not found at {VERBOSE_PATH}")
        return []

    # Load existing trajectory IDs to avoid duplicates
    existing_sessions = set()
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

    with open(VERBOSE_PATH, "r") as f:
        for line in f:
            total_entries += 1
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                continue

            session_id = entry.get("session_id", "")
            turns = entry.get("assistant_turns", [])

            # Collect all tool calls across turns
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

            # Normalize tool names
            tool_names = [normalize_tool_name(tc.get("tool_name", "?")) for tc in all_tool_calls]
            tool_counts = {}
            for name in tool_names:
                tool_counts[name] = tool_counts.get(name, 0) + 1

            # Extract key parameters
            events = []
            for tc in all_tool_calls[:50]:
                raw_name = tc.get("tool_name", "?")
                norm_name = normalize_tool_name(raw_name)
                params = tc.get("parameters", {})
                key_params = {}
                for key in ("file_path", "command", "pattern", "query", "path"):
                    if key in params:
                        key_params[key] = str(params[key])[:200]

                # Determine success
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

            # Count successes/failures
            successes = sum(1 for e in events if e.get("success") is True)
            failures = sum(1 for e in events if e.get("success") is False)
            bash_errors = sum(
                1 for e in events
                if e.get("tool_name") == "Bash" and e.get("exit_code", 0) != 0
            )

            # Build record
            prompt_text = entry.get("prompt_text", "")[:500]
            env = entry.get("environment", {})
            timing = entry.get("timing", {})

            record = {
                "id": f"traj_bf_{session_id[:8]}_{total_entries}",
                "session_id": session_id,
                "channel": "backfill",
                "recorded_at": datetime.now(timezone.utc).isoformat(),
                "skill": {
                    "name": None,
                    "domain": None,
                },
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
                    "duration_s": timing.get("total_duration_ms", 0) / 1000 if isinstance(timing, dict) and timing.get("total_duration_ms") else None,
                },
            }

            trajectories.append(record)
            existing_sessions.add(session_id)  # Prevent intra-run duplicates

    print(f"[extractor] Scanned {total_entries} verbose entries")
    print(f"[extractor] Skipped {skipped_no_tools} (no tools), {skipped_duplicate} (duplicate)")
    print(f"[extractor] Extracted {len(trajectories)} new trajectories")

    if dry_run:
        print("\n[dry-run] Sample records:")
        for r in trajectories[:3]:
            tools = r["trajectory"]["tool_sequence"][:5]
            print(f"  {r['id']}: {r['trajectory']['total_tools']} tools {tools}")
        return trajectories

    # Write to store
    written = 0
    from trajectory_tap import append_to_store
    for record in trajectories:
        if append_to_store(record):
            written += 1

    print(f"[extractor] Wrote {written} records to {STORE_PATH}")
    return trajectories


def show_stats():
    """Show trajectory store statistics."""
    from trajectory_tap import get_store_stats
    stats = get_store_stats()
    print(f"\nTrajectory Store: {STORE_PATH}")
    print(f"  Total records: {stats['total']}")
    print(f"  Size: {stats['size_bytes'] / 1024:.1f} KB")
    print(f"  Channels: {stats.get('channels', {})}")
    print(f"  Skills: {stats.get('skills', {})}")
    print(f"  With reward: {stats.get('with_reward', 0)}")


if __name__ == "__main__":
    if "--stats" in sys.argv:
        show_stats()
    elif "--dry-run" in sys.argv:
        extract_trajectories(dry_run=True)
    else:
        extract_trajectories(dry_run=False)
