"""
sft_exporter.py - Convert KARL trajectories to advantage-weighted SFT data.

Transforms trajectory records into ChatML JSONL format for LoRA fine-tuning.
High-reward trajectories are oversampled proportional to their advantage
following the OAPL-Lite approach.

Pipeline:
  1. Filter trajectories with reward scores
  2. Compute advantage = reward - domain_baseline
  3. Positive-advantage -> include (up to 3x oversample)
  4. Negative-advantage -> exclude
  5. Merge with synthetic QA examples (from synthetic_qa.py)
  6. Output ChatML JSONL for MLX LoRA training

Usage:
    from karl.sft_exporter import export_sft
    stats = export_sft()                     # Export to train/valid JSONL
    stats = export_sft(dry_run=True)         # Preview
    stats = export_sft(min_reward=0.6)       # Filter by minimum reward
"""

import fcntl
import hashlib
import json
import random
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

import karl.config as config

# S6: Patterns that may indicate leaked credentials in prompt text
_CREDENTIAL_PATTERNS = [
    re.compile(r"(?:api[_-]?key|token|secret|password|passwd|credential)[\s=:]+\S+", re.I),
    re.compile(r"(?:sk|pk|rk|ak)-[a-zA-Z0-9]{20,}"),  # API key formats
    re.compile(r"Bearer\s+[a-zA-Z0-9._\-]+"),
    re.compile(r"ghp_[a-zA-Z0-9]{36}"),  # GitHub PATs
    re.compile(r"xox[bprs]-[a-zA-Z0-9\-]+"),  # Slack tokens
    re.compile(r"eyJ[a-zA-Z0-9_\-]{20,}\.[a-zA-Z0-9_\-]{20,}"),  # JWTs
]


def _strip_credentials(text: str) -> str:
    """Remove potential credential patterns from text before including in training data."""
    for pattern in _CREDENTIAL_PATTERNS:
        text = pattern.sub("[REDACTED]", text)
    return text


def _trajectory_to_plan(record: Dict) -> str:
    """Convert a trajectory record into a tool plan summary.

    This is what the model learns to generate: given a prompt,
    predict an effective tool-use sequence.
    """
    events = record.get("trajectory", {}).get("events", [])
    if not events:
        return ""

    parts = []
    for i, event in enumerate(events[:20], 1):
        tool = event.get("tool_name", "?")
        params = event.get("key_params", {})
        success = event.get("success")
        status = "ok" if success else ("fail" if success is False else "?")

        if tool == "Read" and "file_path" in params:
            desc = f"Read {_short_path(params['file_path'])}"
        elif tool == "Edit" and "file_path" in params:
            desc = f"Edit {_short_path(params['file_path'])}"
        elif tool == "Write" and "file_path" in params:
            desc = f"Write {_short_path(params['file_path'])}"
        elif tool == "Bash" and "command" in params:
            desc = f"Bash: {params['command'][:80]}"
        elif tool == "Grep" and "pattern" in params:
            desc = f"Grep '{params['pattern'][:40]}'"
        elif tool == "Glob" and "pattern" in params:
            desc = f"Glob '{params['pattern'][:40]}'"
        elif tool == "Task" and "description" in params:
            desc = f"Task: {params['description'][:60]}"
        else:
            param_str = ", ".join(f"{k}={v[:30]}" for k, v in list(params.items())[:2])
            desc = f"{tool}({param_str})" if param_str else tool

        parts.append(f"{i}. [{status}] {desc}")

    outcome = record.get("outcome", {})
    reward = outcome.get("reward_score")
    total = record.get("trajectory", {}).get("total_tools", 0)
    successes = record.get("trajectory", {}).get("successes", 0)

    summary = f"\n\nResult: {successes}/{total} tools succeeded"
    if reward is not None:
        summary += f", reward={reward:.2f}"

    return "\n".join(parts) + summary


def _short_path(path: str) -> str:
    """Shorten a file path for training data."""
    parts = path.split("/")
    if len(parts) > 3:
        return "/".join([".."] + parts[-2:])
    return path


def _compute_oversample_count(advantage: float) -> int:
    """Determine how many times to include based on advantage.

    advantage <= 0:   0 (filter out)
    advantage 0-0.1:  1
    advantage 0.1-0.3: 2
    advantage > 0.3:  3 (max)
    """
    if advantage <= config.SFT_ADVANTAGE_THRESHOLD:
        return 0
    if advantage <= 0.1:
        return 1
    if advantage <= 0.3:
        return 2
    return config.SFT_MAX_OVERSAMPLE


def export_sft(
    min_reward: float = 0.0,
    dry_run: bool = False,
    train_split: Optional[float] = None,
) -> Dict[str, Any]:
    """Export trajectories to advantage-weighted SFT format.

    Args:
        min_reward: Minimum reward score to include
        dry_run: Preview without writing files
        train_split: Train/valid split ratio (default from config)

    Returns:
        Stats dict with counts, baselines, and file paths
    """
    split = train_split if train_split is not None else config.SFT_TRAIN_SPLIT

    if not config.STORE_PATH.exists():
        return {"error": "No trajectory store found"}

    # C10: Lock store for consistent read during export
    records = []
    with open(config.STORE_PATH) as f:
        fcntl.flock(f.fileno(), fcntl.LOCK_SH)
        try:
            for line in f:
                try:
                    record = json.loads(line)
                    reward = record.get("outcome", {}).get("reward_score")
                    if reward is not None and reward >= min_reward:
                        records.append(record)
                except json.JSONDecodeError:
                    continue
        finally:
            fcntl.flock(f.fileno(), fcntl.LOCK_UN)

    if not records:
        return {"total": 0, "message": "No scored trajectories found"}

    # Compute domain baselines
    domain_rewards: Dict[str, List[float]] = {}
    for r in records:
        domain = r.get("skill", {}).get("domain") or "_global"
        domain_rewards.setdefault(domain, []).append(r["outcome"]["reward_score"])
    baselines = {k: sum(v) / len(v) for k, v in domain_rewards.items()}

    # Generate SFT examples with advantage weighting
    examples = []
    seen_hashes: set = set()
    filtered_low_advantage = 0
    filtered_too_short = 0
    oversampled = 0

    for record in records:
        events = record.get("trajectory", {}).get("events", [])
        if len(events) < config.SFT_MIN_TOOL_EVENTS:
            filtered_too_short += 1
            continue

        reward = record["outcome"]["reward_score"]
        domain = record.get("skill", {}).get("domain") or "_global"
        baseline = baselines.get(domain, 0.5)
        advantage = reward - baseline

        count = _compute_oversample_count(advantage)
        if count == 0:
            filtered_low_advantage += 1
            continue

        prompt = _strip_credentials(record.get("context", {}).get("prompt_text", ""))
        if not prompt or len(prompt) < 10:
            prompt = f"[Task in {record.get('context', {}).get('cwd', 'unknown project')}]"

        plan = _trajectory_to_plan(record)
        if not plan or len(plan) < 20:
            continue

        content_hash = hashlib.sha256((prompt + plan).encode()).hexdigest()[:16]
        if content_hash in seen_hashes:
            continue
        seen_hashes.add(content_hash)

        example = {
            "messages": [
                {"role": "system", "content": config.SFT_SYSTEM_PROMPT},
                {"role": "user", "content": prompt[:4000]},
                {"role": "assistant", "content": plan[:4000]},
            ]
        }

        for _ in range(count):
            examples.append(example)
        if count > 1:
            oversampled += count - 1

    # Merge synthetic QA data
    synthetic_count = 0
    if config.SYNTHETIC_PATH.exists():
        with open(config.SYNTHETIC_PATH) as f:
            for line in f:
                try:
                    syn_example = json.loads(line)
                    msgs = syn_example.get("messages", [])
                    if len(msgs) >= 3:
                        syn_hash = hashlib.sha256(
                            (msgs[1]["content"] + msgs[2]["content"]).encode()
                        ).hexdigest()[:16]
                        if syn_hash not in seen_hashes:
                            seen_hashes.add(syn_hash)
                            examples.append(syn_example)
                            synthetic_count += 1
                except (json.JSONDecodeError, KeyError):
                    continue

    if not examples:
        return {
            "total_records": len(records),
            "examples": 0,
            "filtered_low_advantage": filtered_low_advantage,
            "filtered_too_short": filtered_too_short,
        }

    if dry_run:
        return {
            "total_records": len(records),
            "examples": len(examples),
            "unique": len(seen_hashes),
            "oversampled": oversampled,
            "synthetic": synthetic_count,
            "filtered_low_advantage": filtered_low_advantage,
            "filtered_too_short": filtered_too_short,
            "baselines": {k: round(v, 4) for k, v in baselines.items()},
        }

    # Write combined output
    config.SFT_OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(config.SFT_OUTPUT_PATH, "w") as f:
        for ex in examples:
            f.write(json.dumps(ex) + "\n")

    # Train/valid split
    random.seed(42)
    shuffled = list(examples)
    random.shuffle(shuffled)
    split_idx = int(len(shuffled) * split)
    train_examples = shuffled[:split_idx]
    valid_examples = shuffled[split_idx:]

    with open(config.TRAIN_PATH, "w") as f:
        for ex in train_examples:
            f.write(json.dumps(ex) + "\n")

    with open(config.VALID_PATH, "w") as f:
        for ex in valid_examples:
            f.write(json.dumps(ex) + "\n")

    return {
        "total_records": len(records),
        "examples": len(examples),
        "unique": len(seen_hashes),
        "oversampled": oversampled,
        "synthetic": synthetic_count,
        "train": len(train_examples),
        "valid": len(valid_examples),
        "filtered_low_advantage": filtered_low_advantage,
        "filtered_too_short": filtered_too_short,
        "baselines": {k: round(v, 4) for k, v in baselines.items()},
        "output": str(config.SFT_OUTPUT_PATH),
        "train_file": str(config.TRAIN_PATH),
        "valid_file": str(config.VALID_PATH),
    }
