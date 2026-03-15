#!/usr/bin/env python3
"""
sft_exporter.py — Convert KARL trajectories to advantage-weighted SFT data.

Transforms trajectory records into ChatML JSONL format for MLX LoRA training.
High-reward trajectories are oversampled proportional to their advantage.

OAPL-Lite approach:
  1. Filter trajectories with reward scores
  2. Compute advantage = reward - baseline
  3. Positive-advantage trajectories → include (up to 3x oversample)
  4. Negative-advantage trajectories → exclude (or include 1x as negative signal)
  5. Output ChatML JSONL compatible with Mac5's finetune-daemon

Output format (per line):
{
  "messages": [
    {"role": "system", "content": "..."},
    {"role": "user", "content": "[prompt]"},
    {"role": "assistant", "content": "[tool plan summary]"}
  ]
}

Usage:
    python3 sft_exporter.py                    # Export to karl-sft.jsonl
    python3 sft_exporter.py --stats            # Show export stats
    python3 sft_exporter.py --dry-run          # Preview without writing
    python3 sft_exporter.py --min-reward 0.6   # Filter by minimum reward
    python3 sft_exporter.py --quality high     # Only high-quality sessions
    python3 sft_exporter.py --quality medium+  # Medium and high quality
"""

import json
import hashlib
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

KARL_DIR = Path(__file__).parent
STORE_PATH = KARL_DIR / "trajectories.jsonl"
OUTPUT_PATH = KARL_DIR / "karl-sft.jsonl"
TRAIN_PATH = KARL_DIR / "train.jsonl"
VALID_PATH = KARL_DIR / "valid.jsonl"

SYSTEM_PROMPT = (
    "You are an expert software engineering assistant. Given a task, "
    "plan the optimal sequence of tool uses to accomplish it efficiently. "
    "Consider which tools to use, in what order, and what parameters. "
    "Prefer reading before editing, testing after changes, and using "
    "the most specific tool available."
)

# Advantage-weighted oversampling config
MAX_OVERSAMPLE = 3       # Max times a high-reward trajectory appears
ADVANTAGE_THRESHOLD = 0.0  # Include if advantage >= this
MIN_TOOL_EVENTS = 2      # Skip trivial trajectories


def _trajectory_to_plan(record: Dict) -> str:
    """Convert a trajectory record into a tool plan summary.

    This is what the model learns to generate: given a prompt,
    predict an effective tool-use sequence.
    """
    events = record.get("trajectory", {}).get("events", [])
    if not events:
        return ""

    parts = []
    for i, event in enumerate(events[:20], 1):  # Cap at 20 steps
        tool = event.get("tool_name", "?")
        params = event.get("key_params", {})
        success = event.get("success")
        status = "ok" if success else ("fail" if success is False else "?")

        # Build concise step description
        if tool == "Read" and "file_path" in params:
            desc = f"Read {_short_path(params['file_path'])}"
        elif tool == "Edit" and "file_path" in params:
            desc = f"Edit {_short_path(params['file_path'])}"
        elif tool == "Write" and "file_path" in params:
            desc = f"Write {_short_path(params['file_path'])}"
        elif tool == "Bash" and "command" in params:
            cmd = params["command"][:80]
            desc = f"Bash: {cmd}"
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

    # Add outcome summary
    outcome = record.get("outcome", {})
    reward = outcome.get("reward_score")
    total = record.get("trajectory", {}).get("total_tools", 0)
    successes = record.get("trajectory", {}).get("successes", 0)

    summary = f"\n\nResult: {successes}/{total} tools succeeded"
    if reward is not None:
        summary += f", reward={reward:.2f}"

    return "\n".join(parts) + summary


def _short_path(path: str) -> str:
    """Shorten a file path for the training data."""
    parts = path.split("/")
    if len(parts) > 3:
        return "/".join([".."] + parts[-2:])
    return path


def _compute_oversample_count(advantage: float) -> int:
    """Determine how many times to include this trajectory based on advantage.

    advantage <= 0: include 0 times (filter out)
    advantage 0-0.1: include 1 time
    advantage 0.1-0.3: include 2 times
    advantage > 0.3: include 3 times (max)
    """
    if advantage <= ADVANTAGE_THRESHOLD:
        return 0
    if advantage <= 0.1:
        return 1
    if advantage <= 0.3:
        return 2
    return MAX_OVERSAMPLE


QUALITY_LEVELS = {"high": 3, "medium": 2, "low": 1}


def _passes_quality_filter(record, quality_filter):
    """Check if a trajectory passes the quality filter.
    quality_filter: None (no filter), "high", "medium+", "low+"
    """
    if quality_filter is None:
        return True
    grade = record.get("quality", {}).get("grade", "medium")
    level = QUALITY_LEVELS.get(grade, 2)
    if quality_filter == "high":
        return level >= 3
    elif quality_filter == "medium+" or quality_filter == "medium":
        return level >= 2
    elif quality_filter == "low+":
        return level >= 1
    return True


def export_sft(
    min_reward: float = 0.0,
    dry_run: bool = False,
    train_split: float = 0.9,
    quality_filter: str = None,
) -> Dict[str, Any]:
    """Export trajectories to advantage-weighted SFT format.

    Args:
        quality_filter: "high", "medium+", or None for no filter.

    Returns stats dict.
    """
    if not STORE_PATH.exists():
        return {"error": "No trajectory store found"}

    # Load all scored trajectories
    records = []
    filtered_quality = 0
    with open(STORE_PATH) as f:
        for line in f:
            try:
                record = json.loads(line)
                reward = record.get("outcome", {}).get("reward_score")
                if reward is not None and reward >= min_reward:
                    if not _passes_quality_filter(record, quality_filter):
                        filtered_quality += 1
                        continue
                    records.append(record)
            except json.JSONDecodeError:
                continue

    if not records:
        return {"total": 0, "message": "No scored trajectories found"}

    # Use FlowRL balanced sampling if available
    use_flowrl = "--flowrl" in sys.argv if not dry_run else False
    if use_flowrl:
        try:
            from flow_sampler import FlowRLSampler
            sampler = FlowRLSampler()
            records = sampler.sample(strategy="balanced", size=min(len(records), 80))
        except ImportError:
            pass

    # Compute domain baselines for advantage
    domain_rewards: Dict[str, List[float]] = {}
    for r in records:
        domain = r.get("skill", {}).get("domain") or "_global"
        domain_rewards.setdefault(domain, []).append(
            r["outcome"]["reward_score"]
        )
    baselines = {k: sum(v) / len(v) for k, v in domain_rewards.items()}

    # Generate SFT examples with advantage weighting
    examples = []
    seen_hashes = set()
    filtered_low_advantage = 0
    filtered_too_short = 0
    oversampled = 0

    for record in records:
        events = record.get("trajectory", {}).get("events", [])
        if len(events) < MIN_TOOL_EVENTS:
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

        # Build the training example
        prompt = record.get("context", {}).get("prompt_text", "")
        if not prompt or len(prompt) < 10:
            prompt = f"[Task in {record.get('context', {}).get('cwd', 'unknown project')}]"

        plan = _trajectory_to_plan(record)
        if not plan or len(plan) < 20:
            continue

        # Deduplicate
        content_hash = hashlib.sha256(
            (prompt + plan).encode()
        ).hexdigest()[:16]
        if content_hash in seen_hashes:
            continue
        seen_hashes.add(content_hash)

        example = {
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt[:4000]},
                {"role": "assistant", "content": plan[:4000]},
            ]
        }

        # Add multiple copies for oversampling
        for _ in range(count):
            examples.append(example)
        if count > 1:
            oversampled += count - 1

    # Include synthetic QA data (Phase 6)
    synthetic_count = 0
    synthetic_path = KARL_DIR / "synthetic_qa.jsonl"
    if synthetic_path.exists():
        with open(synthetic_path) as f:
            for line in f:
                try:
                    syn_example = json.loads(line)
                    # Deduplicate against trajectory examples
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
            "filtered_quality": filtered_quality,
            "quality_filter": quality_filter,
        }

    if dry_run:
        print(f"\n[dry-run] Would write {len(examples)} examples")
        print(f"Sample:\n{json.dumps(examples[0], indent=2)[:500]}")
        return {
            "total_records": len(records),
            "examples": len(examples),
            "unique": len(seen_hashes),
            "oversampled": oversampled,
            "filtered_low_advantage": filtered_low_advantage,
            "filtered_too_short": filtered_too_short,
            "filtered_quality": filtered_quality,
            "quality_filter": quality_filter,
            "baselines": {k: round(v, 4) for k, v in baselines.items()},
        }

    # Write combined output
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w") as f:
        for ex in examples:
            f.write(json.dumps(ex) + "\n")

    # Train/valid split
    import random
    random.seed(42)
    random.shuffle(examples)
    split_idx = int(len(examples) * train_split)
    train_examples = examples[:split_idx]
    valid_examples = examples[split_idx:]

    with open(TRAIN_PATH, "w") as f:
        for ex in train_examples:
            f.write(json.dumps(ex) + "\n")

    with open(VALID_PATH, "w") as f:
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
        "filtered_quality": filtered_quality,
        "quality_filter": quality_filter,
        "baselines": {k: round(v, 4) for k, v in baselines.items()},
        "output": str(OUTPUT_PATH),
        "train_file": str(TRAIN_PATH),
        "valid_file": str(VALID_PATH),
    }


def check_sft_readiness(min_high_quality: int = 30, min_skills: int = 5) -> Dict[str, Any]:
    """Check if we have enough data for meaningful SFT training.

    Criteria:
      - min_high_quality high/medium quality trajectories with tool events
      - min_skills distinct skills represented
      - Positive advantage data available
    """
    from collections import Counter

    if not STORE_PATH.exists():
        return {"status": "no_data", "ready": False}

    records = []
    with open(STORE_PATH) as f:
        for line in f:
            try:
                r = json.loads(line)
                if r.get("outcome", {}).get("reward_score") is not None:
                    records.append(r)
            except json.JSONDecodeError:
                continue

    if not records:
        return {"status": "no_scored_data", "ready": False}

    # Quality breakdown
    quality_counts = Counter()
    skill_counts = Counter()
    exportable = 0
    for r in records:
        grade = r.get("quality", {}).get("grade", "unknown")
        quality_counts[grade] += 1
        skill = r.get("skill", {}).get("name", "unknown")
        skill_counts[skill] += 1
        # Check if exportable (has tool events in trajectory or context)
        tools = r.get("trajectory", {}).get("events", []) or r.get("context", {}).get("tool_events", [])
        if len(tools) >= MIN_TOOL_EVENTS:
            exportable += 1

    high_quality = quality_counts.get("high", 0) + quality_counts.get("medium", 0)
    distinct_skills = len(skill_counts)

    # Compute mean advantage
    rewards = [r["outcome"]["reward_score"] for r in records]
    mean_reward = sum(rewards) / len(rewards)
    positive_advantage = sum(1 for r in rewards if r > mean_reward)

    ready = high_quality >= min_high_quality and distinct_skills >= min_skills

    return {
        "status": "ok",
        "ready": ready,
        "total_scored": len(records),
        "exportable": exportable,
        "quality": dict(quality_counts),
        "high_quality_count": high_quality,
        "min_high_quality_required": min_high_quality,
        "distinct_skills": distinct_skills,
        "min_skills_required": min_skills,
        "top_skills": dict(skill_counts.most_common(10)),
        "mean_reward": round(mean_reward, 4),
        "positive_advantage_count": positive_advantage,
        "recommendation": (
            "READY: Sufficient data for SFT training. Run 'karl export' to generate."
            if ready else
            f"NOT READY: Need {max(0, min_high_quality - high_quality)} more high-quality + "
            f"{max(0, min_skills - distinct_skills)} more skills."
        ),
    }


def _parse_args():
    min_r = 0.0
    quality = None
    for i, arg in enumerate(sys.argv):
        if arg == "--min-reward" and i + 1 < len(sys.argv):
            min_r = float(sys.argv[i + 1])
        if arg == "--quality" and i + 1 < len(sys.argv):
            quality = sys.argv[i + 1]
    return min_r, quality


if __name__ == "__main__":
    min_r, quality = _parse_args()
    if "--stats" in sys.argv:
        stats = export_sft(dry_run=True, quality_filter=quality)
        print(json.dumps(stats, indent=2))
    elif "--dry-run" in sys.argv:
        stats = export_sft(min_reward=min_r, dry_run=True, quality_filter=quality)
        print(json.dumps(stats, indent=2))
    else:
        stats = export_sft(min_reward=min_r, quality_filter=quality)
        print(f"[sft_exporter] {json.dumps(stats, indent=2)}")
