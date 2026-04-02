#!/usr/bin/env python3
"""Curate V7 factory session logs into SFT training data.

Reads V7 driver logs, filters by quality, pairs with Claude's responses
from the actual session prompt logs, and exports training-ready JSONL.

Pipeline:
  1. Read V7 session logs (what the simulator injected)
  2. Match each injection to Claude's response from prompt logs
  3. Filter: style_score >= 0.6, no terminal garbage, no dedup escapes
  4. Format as SFT training pairs
  5. Export with advantage-weighted sampling
"""

import glob
import json
import os
import re
from collections import Counter
from dataclasses import dataclass, asdict
from pathlib import Path

from ..v6.anti_repeat import is_semantic_stuck_loop

V7_LOG_DIR = os.path.expanduser("~/Desktop/karl/v7-session-logs")
PROMPT_LOG_DIR = os.path.expanduser("~/.claude/projects/-Users-mohameddiomande")
OUTPUT_PATH = os.path.expanduser("~/Desktop/karl/v7-training-data.jsonl")

# Quality filters
MIN_STYLE_SCORE = 0.6
MIN_PROMPT_LENGTH = 20
MAX_PROMPT_LENGTH = 1000
GARBAGE_PATTERNS = [
    re.compile(r"mohameddiomande@Mac", re.IGNORECASE),
    re.compile(r"zsh:", re.IGNORECASE),
    re.compile(r"bad interpreter", re.IGNORECASE),
    re.compile(r"^\s*```\s*bash", re.IGNORECASE),
    re.compile(r"ssh .* &&", re.IGNORECASE),
    re.compile(r"echo \"===", re.IGNORECASE),
    re.compile(r"<\|(?:start|end)\|>", re.IGNORECASE),
]

REMOTE_MACHINE_PAT = re.compile(
    r"\b(?:ssh(?:\s+into|\s+to)?|on|into|fire\s+up|spin\s+up|kick\s+off|open(?:\s+a)?(?:\s+new)?\s+pane\s+on)\s+(mac[1-5]|cloud-vm)\b",
    re.IGNORECASE,
)


@dataclass
class TrainingPair:
    prompt: str
    strategy: str
    style_score: float
    phase: str
    goal: str
    session_id: str
    turn: int
    source: str  # "v7_factory"


def load_v7_logs() -> list[dict]:
    """Load all V7 session log entries."""
    entries = []
    for fpath in sorted(glob.glob(os.path.join(V7_LOG_DIR, "*.jsonl"))):
        with open(fpath) as f:
            for line in f:
                try:
                    entry = json.loads(line)
                    entry["_log_file"] = os.path.basename(fpath)
                    entries.append(entry)
                except json.JSONDecodeError:
                    continue
    return entries


def is_garbage(prompt: str) -> bool:
    """Check if a prompt contains terminal garbage or non-prompt content."""
    for pat in GARBAGE_PATTERNS:
        if pat.search(prompt):
            return True
    return False


def session_machine(log_file: str) -> str:
    """Extract the session machine from a log filename."""
    parts = log_file.split("_")
    if len(parts) >= 2 and parts[1].startswith("mac"):
        return parts[1]
    return ""


def filter_entries(entries: list[dict]) -> list[dict]:
    """Apply quality filters to V7 log entries."""
    filtered = []
    stats = Counter()
    recent_prompts_by_session: dict[str, list[str]] = {}

    for entry in entries:
        prompt = entry.get("final_prompt", "")
        style = entry.get("style_score", 0)
        reason = entry.get("reason", "")
        session_key = entry.get("_log_file", "unknown")
        recent_prompts = recent_prompts_by_session.setdefault(session_key, [])

        # Skip dedup escapes (not simulator-generated)
        if "dedup" in reason or "escape" in reason or "destructive" in reason:
            stats["skipped_dedup"] += 1
            continue

        # Style score filter
        if style < MIN_STYLE_SCORE:
            stats["skipped_low_style"] += 1
            continue

        # Length filter
        if len(prompt) < MIN_PROMPT_LENGTH or len(prompt) > MAX_PROMPT_LENGTH:
            stats["skipped_length"] += 1
            continue

        # Garbage filter
        if is_garbage(prompt):
            stats["skipped_garbage"] += 1
            recent_prompts.append(prompt)
            recent_prompts_by_session[session_key] = recent_prompts[-6:]
            continue

        # Semantic stuck-loop filter
        if entry.get("phase") == "STUCK" and is_semantic_stuck_loop(prompt, recent_prompts):
            stats["skipped_stuck_loop"] += 1
            recent_prompts.append(prompt)
            recent_prompts_by_session[session_key] = recent_prompts[-6:]
            continue

        # Machine-mismatch filter for roleplay hallucinations like "SSH into Mac1"
        current_machine = session_machine(session_key)
        remote_match = REMOTE_MACHINE_PAT.search(prompt)
        if current_machine and remote_match:
            referenced_machine = remote_match.group(1).lower()
            prompt_targets_factory_path = "/Desktop/factory/" in prompt
            if referenced_machine != current_machine.lower() and (
                entry.get("phase") == "STUCK" or prompt_targets_factory_path
            ):
                stats["skipped_machine_mismatch"] += 1
                recent_prompts.append(prompt)
                recent_prompts_by_session[session_key] = recent_prompts[-6:]
                continue

        stats["passed"] += 1
        filtered.append(entry)
        recent_prompts.append(prompt)
        recent_prompts_by_session[session_key] = recent_prompts[-6:]

    print(f"Filter results: {dict(stats)}")
    return filtered


def format_training_pairs(entries: list[dict]) -> list[TrainingPair]:
    """Convert filtered entries to training pairs."""
    pairs = []
    for entry in entries:
        pair = TrainingPair(
            prompt=entry["final_prompt"].strip(),
            strategy=entry.get("strategy", "unknown"),
            style_score=entry.get("style_score", 0),
            phase=entry.get("phase", "BUILD"),
            goal=entry.get("_goal", ""),
            session_id=entry.get("_log_file", ""),
            turn=entry.get("turn", 0),
            source="v7_factory",
        )
        pairs.append(pair)
    return pairs


def export(pairs: list[TrainingPair], output_path: str = OUTPUT_PATH):
    """Export training pairs as JSONL."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        for p in pairs:
            f.write(json.dumps(asdict(p)) + "\n")
    print(f"Exported {len(pairs)} training pairs to {output_path}")


def analyze(pairs: list[TrainingPair]):
    """Print quality analysis of training data."""
    if not pairs:
        print("No training pairs to analyze.")
        return

    strategies = Counter(p.strategy for p in pairs)
    phases = Counter(p.phase for p in pairs)
    scores = [p.style_score for p in pairs]
    lengths = [len(p.prompt) for p in pairs]

    print(f"\n=== V7 Training Data Analysis ===")
    print(f"Total pairs: {len(pairs)}")
    print(f"Style score: avg={sum(scores)/len(scores):.2f} min={min(scores):.2f} max={max(scores):.2f}")
    print(f"Prompt length: avg={sum(lengths)/len(lengths):.0f} min={min(lengths)} max={max(lengths)}")
    print(f"Strategies: {dict(strategies)}")
    print(f"Phases: {dict(phases)}")

    print(f"\nSample pairs:")
    import random
    random.seed(42)
    for p in random.sample(pairs, min(5, len(pairs))):
        print(f"  [{p.phase}] {p.strategy} score={p.style_score:.2f}")
        print(f"    {p.prompt[:120]}")
        print()


def main():
    print("Loading V7 session logs...")
    entries = load_v7_logs()
    print(f"Loaded {len(entries)} log entries")

    print("\nFiltering...")
    filtered = filter_entries(entries)

    print(f"\nFormatting {len(filtered)} training pairs...")
    pairs = format_training_pairs(filtered)

    export(pairs)
    analyze(pairs)


if __name__ == "__main__":
    main()
