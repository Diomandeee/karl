#!/usr/bin/env python3
"""
mega_extract.py — Extract trajectories from ALL available data sources.

Sources:
  1. verbose-all.jsonl (7,375 entries with tool calls)
  2. prompts-all.jsonl (6,954 prompt/response pairs)
  3. archived verbose logs (146 entries across 11 sessions)
  4. project prompt logs (10,225 entries)

Outputs to trajectories.jsonl with reward scores computed inline.
"""

import json
import hashlib
import sys
import time
from pathlib import Path
from typing import Dict, List, Any

KARL_DIR = Path(__file__).parent
STORE_PATH = KARL_DIR / "trajectories.jsonl"
PROMPT_LOGS = Path.home() / ".claude/prompt-logs"

# Deduplicate by content hash
seen_hashes = set()
all_records = []


def content_hash(record: Dict) -> str:
    """Hash the trajectory content for deduplication."""
    traj = record.get("trajectory", {})
    prompt = traj.get("prompt", "")[:500]
    tools = str(traj.get("tool_calls", []))[:500]
    return hashlib.md5(f"{prompt}{tools}".encode()).hexdigest()


def extract_from_verbose(path: Path, source: str = "verbose") -> List[Dict]:
    """Extract trajectories from prompt-logger enriched JSONL format.

    Each entry has: prompt_text, assistant_turns[{tool_calls, text_response}],
    files_modified, errors, complexity_score, etc.
    """
    if not path.exists():
        return []

    records = []
    with open(path) as f:
        for line in f:
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                continue

            prompt = entry.get("prompt_text", "")
            if not prompt or len(prompt) < 10:
                continue

            session_id = entry.get("session_id", "unknown")
            domain = entry.get("orbit_project_name") or entry.get("git_repo") or "unknown"

            # Aggregate all assistant turns
            turns = entry.get("assistant_turns", [])
            tool_calls = []
            response_parts = []

            for turn in turns:
                # Tool calls
                for tc in turn.get("tool_calls", []):
                    tool_calls.append({
                        "tool": normalize_tool(tc.get("tool_name", "")),
                        "input_preview": str(tc.get("tool_input", {}))[:200],
                        "success": not tc.get("is_error", False),
                        "duration_ms": tc.get("duration_ms", 0),
                    })
                # Response text
                text = turn.get("text_response", "")
                if text:
                    response_parts.append(text)

            response = "\n".join(response_parts)

            # Skip entries with no substance
            if not response and not tool_calls:
                continue

            record = build_record(
                session_id=session_id,
                prompt=prompt[:3000],
                response=response[:3000],
                tool_calls=tool_calls,
                source=source,
                domain=domain,
                extra={
                    "files_modified": entry.get("files_modified", []),
                    "files_created": entry.get("files_created", []),
                    "errors": entry.get("errors", []),
                    "complexity": entry.get("complexity_score"),
                    "total_tokens": entry.get("total_token_usage", {}),
                    "git_dirty": entry.get("git_is_dirty", False),
                    "file_diffs": bool(entry.get("file_diffs")),
                },
            )

            h = content_hash(record)
            if h not in seen_hashes:
                seen_hashes.add(h)
                records.append(record)

    return records


def extract_from_prompts(path: Path, source: str = "prompts") -> List[Dict]:
    """Extract from prompts-all.jsonl (prompt/response pairs)."""
    if not path.exists():
        return []

    records = []
    with open(path) as f:
        for line in f:
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                continue

            prompt = entry.get("prompt", entry.get("user_message", ""))
            response = entry.get("response", entry.get("assistant_message", ""))
            tools = entry.get("tools_used", entry.get("tool_calls", []))
            session_id = entry.get("session_id", entry.get("sessionId", "unknown"))

            if not prompt or not response:
                continue

            # Build tool list
            tool_calls = []
            if isinstance(tools, list):
                for t in tools:
                    if isinstance(t, dict):
                        tool_calls.append({
                            "tool": normalize_tool(t.get("name", t.get("tool", ""))),
                            "input_preview": str(t.get("input", t.get("args", {})))[:200],
                        })
                    elif isinstance(t, str):
                        tool_calls.append({"tool": normalize_tool(t), "input_preview": ""})

            record = build_record(
                session_id=session_id,
                prompt=prompt[:2000],
                response=response[:2000],
                tool_calls=tool_calls,
                source=source,
            )
            h = content_hash(record)
            if h not in seen_hashes:
                seen_hashes.add(h)
                records.append(record)

    return records


def build_record(session_id: str, prompt: str, response: str,
                 tool_calls: List[Dict], source: str, domain: str = "unknown",
                 extra: Dict = None) -> Dict:
    """Build a trajectory record in KARL format."""
    record_id = hashlib.md5(f"{session_id}-{prompt[:100]}-{time.time()}".encode()).hexdigest()[:16]

    # Compute basic outcome signals
    has_error = any("error" in str(t.get("input_preview", "")).lower() for t in tool_calls)
    tool_count = len(tool_calls)
    tool_types = list(set(t.get("tool", "") for t in tool_calls))

    record = {
        "id": record_id,
        "session_id": session_id,
        "skill": source,
        "domain": domain,
        "trajectory": {
            "prompt": prompt[:3000],
            "response": response[:3000],
            "tool_calls": tool_calls[:30],
            "tool_count": tool_count,
            "tool_types": tool_types,
        },
        "context": {
            "source": source,
            "domain": domain,
            "extracted_at": time.time(),
        },
        "timing": {
            "tool_count": tool_count,
        },
        "outcome": {
            "has_error": has_error,
            "completed": not has_error,
            "tool_diversity": len(tool_types),
        },
    }
    if extra:
        record["outcome"]["files_modified"] = len(extra.get("files_modified", []))
        record["outcome"]["files_created"] = len(extra.get("files_created", []))
        record["outcome"]["has_diffs"] = extra.get("file_diffs", False)
        record["outcome"]["errors"] = extra.get("errors", [])
        record["context"]["complexity"] = extra.get("complexity")
        record["context"]["total_tokens"] = extra.get("total_tokens", {})
        # Better success signal: tools that reported success
        success_count = sum(1 for tc in tool_calls if tc.get("success", True))
        record["outcome"]["tool_success_rate"] = success_count / max(tool_count, 1)
    return record


def normalize_tool(name: str) -> str:
    """Normalize tool names across Codex/Claude/OpenCode."""
    TOOL_MAP = {
        "exec_command": "Bash", "shell_command": "Bash", "shell": "Bash",
        "apply_patch": "Edit", "apply_diff": "Edit",
        "read_file": "Read", "cat_file": "Read",
        "write_file": "Write", "create_file": "Write",
        "search_files": "Grep", "ripgrep": "Grep",
        "list_files": "Glob", "find_files": "Glob",
    }
    return TOOL_MAP.get(name, name)


def compute_reward_inline(record: Dict) -> Dict:
    """Lightweight reward computation without importing reward_engine."""
    traj = record.get("trajectory", {})
    outcome = record.get("outcome", {})

    # Process score: tool success rate
    tool_count = traj.get("tool_count", 0)
    has_error = outcome.get("has_error", False)
    process = 0.8 if not has_error else 0.3
    if tool_count > 0:
        process += min(0.2, tool_count * 0.02)

    # Outcome score
    completed = outcome.get("completed", True)
    outcome_score = 0.7 if completed else 0.3

    # Efficiency
    diversity = outcome.get("tool_diversity", 1)
    efficiency = min(1.0, 0.3 + diversity * 0.1 + (0.1 if tool_count < 10 else 0))

    # Prompt quality bonus: longer, more specific prompts
    prompt_len = len(traj.get("prompt", ""))
    prompt_bonus = min(0.15, prompt_len / 2000 * 0.15)

    # Response quality: has actual content
    response_len = len(traj.get("response", ""))
    response_bonus = min(0.1, response_len / 1000 * 0.1)

    reward = (0.30 * outcome_score + 0.25 * process + 0.20 * efficiency
              + 0.15 * prompt_bonus + 0.10 * response_bonus)

    record["reward_score"] = round(reward, 4)
    record["outcome_score"] = round(outcome_score, 4)
    record["process_score"] = round(process, 4)
    record["efficiency_score"] = round(efficiency, 4)

    return record


def main():
    global all_records, seen_hashes

    print("=== KARL Mega Extraction ===")
    print()

    # Source 1: verbose-all.jsonl
    verbose_all = PROMPT_LOGS / "verbose-all.jsonl"
    print(f"1. verbose-all.jsonl...")
    r1 = extract_from_verbose(verbose_all, "verbose-all")
    print(f"   → {len(r1)} trajectories")
    all_records.extend(r1)

    # Source 2: prompts-all.jsonl
    prompts_all = PROMPT_LOGS / "prompts-all.jsonl"
    print(f"2. prompts-all.jsonl...")
    r2 = extract_from_prompts(prompts_all, "prompts-all")
    print(f"   → {len(r2)} trajectories")
    all_records.extend(r2)

    # Source 3: archived verbose logs
    archive_dir = PROMPT_LOGS / "archive"
    archived_count = 0
    if archive_dir.exists():
        for verbose_file in sorted(archive_dir.glob("*/verbose.jsonl")):
            session_name = verbose_file.parent.name[:12]
            r = extract_from_verbose(verbose_file, f"archive-{session_name}")
            archived_count += len(r)
            all_records.extend(r)
    print(f"3. archived sessions → {archived_count} trajectories")

    # Source 4: project prompt logs
    projects_dir = PROMPT_LOGS / "projects"
    project_count = 0
    if projects_dir.exists():
        for prompts_file in sorted(projects_dir.glob("*/prompts.jsonl")):
            project_name = prompts_file.parent.name[:12]
            r = extract_from_prompts(prompts_file, f"project-{project_name}")
            project_count += len(r)
            all_records.extend(r)
    print(f"4. project logs → {project_count} trajectories")

    print()
    print(f"Total unique trajectories: {len(all_records)}")

    # Score all
    print("Computing rewards...")
    for r in all_records:
        if "reward_score" not in r:
            compute_reward_inline(r)

    avg_reward = sum(r.get("reward_score", 0) for r in all_records) / max(len(all_records), 1)
    high = sum(1 for r in all_records if r.get("reward_score", 0) > 0.5)
    print(f"Average reward: {avg_reward:.3f}")
    print(f"Above 0.5: {high}")

    # Write
    with open(STORE_PATH, "w") as f:
        for r in all_records:
            f.write(json.dumps(r) + "\n")
    print(f"\nWrote {len(all_records)} trajectories to {STORE_PATH}")

    # Export SFT
    print("\nExporting SFT training data...")
    export_sft(all_records)


def export_sft(records: List[Dict]):
    """Export advantage-weighted SFT data for MLX LoRA training."""
    # Compute baseline per source
    from collections import defaultdict
    source_scores = defaultdict(list)
    for r in records:
        src = r.get("context", {}).get("source", "unknown")
        source_scores[src].append(r.get("reward_score", 0))

    baselines = {src: sum(scores)/len(scores) for src, scores in source_scores.items()}

    # Select positive-advantage examples
    sft_examples = []
    for r in records:
        src = r.get("context", {}).get("source", "unknown")
        baseline = baselines.get(src, 0.5)
        advantage = r.get("reward_score", 0) - baseline

        if advantage < -0.05:  # Skip clearly below average
            continue

        traj = r.get("trajectory", {})
        prompt = traj.get("prompt", "").strip()
        response = traj.get("response", "").strip()

        if not prompt or not response:
            continue
        if len(prompt) < 20 or len(response) < 20:
            continue

        # Build ChatML format
        system_msg = (
            "You are Mohamed's cognitive twin, an AI assistant that has learned from his working patterns, "
            "decisions, and problem-solving approaches across software engineering, iOS development, "
            "multi-machine mesh operations, and creative production. Respond with his communication style: "
            "direct, technical, opinionated. Give concrete recommendations, not vague suggestions."
        )

        example = {
            "messages": [
                {"role": "system", "content": system_msg},
                {"role": "user", "content": prompt[:2000]},
                {"role": "assistant", "content": response[:2000]},
            ]
        }

        # Oversample high-advantage examples
        copies = 1
        if advantage > 0.05:
            copies = 2
        if advantage > 0.10:
            copies = 3

        for _ in range(copies):
            sft_examples.append(example)

    # Shuffle
    import random
    random.seed(42)
    random.shuffle(sft_examples)

    # Split 90/10
    split = max(1, int(len(sft_examples) * 0.9))
    train = sft_examples[:split]
    valid = sft_examples[split:]

    train_path = KARL_DIR / "train.jsonl"
    valid_path = KARL_DIR / "valid.jsonl"

    with open(train_path, "w") as f:
        for ex in train:
            f.write(json.dumps(ex) + "\n")

    with open(valid_path, "w") as f:
        for ex in valid:
            f.write(json.dumps(ex) + "\n")

    print(f"  Train: {len(train)} examples → {train_path}")
    print(f"  Valid: {len(valid)} examples → {valid_path}")
    print(f"  Total: {len(sft_examples)} (with oversampling)")


if __name__ == "__main__":
    main()
