#!/usr/bin/env python3
"""Prepare V7 training data: SFT + DPO for Vast.ai.

Merges three sources:
  1. V4 trajectories (2,540 SFT examples via prepare_karl_data.py)
  2. V7.2 factory pairs (75 multi-chain, style 0.98)
  3. DPO preference pairs (173 from trajectory + factory)

Output:
  data/v7-sft-train.jsonl   — SFT training split (90%)
  data/v7-sft-eval.jsonl    — SFT eval split (10%)
  data/v7-dpo-train.jsonl   — DPO training pairs
"""

import json
import os
import random
import re
from pathlib import Path

KARL_DIR = Path(__file__).parent.parent / "karl"
TRAJ_PATH = KARL_DIR / "trajectories.jsonl"
FACTORY_PATH = Path(__file__).parent.parent / "v7-training-data.jsonl"
DPO_PATH = Path(__file__).parent.parent / "v7-dpo-pairs.jsonl"
OUT_DIR = Path(__file__).parent / "data"

SYSTEM_PROMPT = (
    "You are Mohamed's cognitive twin. You've learned from 4,587 of his real "
    "prompts across Rust daemons, Python ML, SwiftUI apps, mesh networking, "
    "and creative production. You think like him: direct, conversational, "
    "opinionated. You reference specific files, ports, services, and machines. "
    "Short sentences. No corporate speak. Get to the point."
)

INSCRIPTIONS = [
    "stabilization", "transition", "oscillation", "correction", "exploration",
    "convergence", "expansion", "regression", "stagnation", "completion",
]

TAG_RE = re.compile(r'<[^>]{1,100}>')


def clean_text(text):
    text = TAG_RE.sub('', text)
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text.strip()


def classify_inscription(record):
    """Classify trajectory into inscription category from reward signals."""
    reward = float(record.get("reward_score", 0.5))
    outcome = float(record.get("outcome_score", 0.5))
    process = float(record.get("process_score", 0.5))

    if reward >= 0.65:
        return "completion" if outcome >= 0.7 else "convergence"
    elif reward >= 0.55:
        if process >= 0.6:
            return "stabilization"
        else:
            return "transition"
    elif reward >= 0.50:
        return "exploration" if outcome >= 0.5 else "oscillation"
    elif reward >= 0.45:
        return "correction" if outcome < 0.4 else "stagnation"
    else:
        return "regression"


def load_trajectories() -> list[dict]:
    """Load and format V4 trajectories as SFT examples."""
    if not TRAJ_PATH.exists():
        print(f"WARNING: {TRAJ_PATH} not found")
        return []

    examples = []
    seen = set()

    for line in open(TRAJ_PATH):
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue

        traj = record.get("trajectory", {})
        prompt = clean_text(traj.get("prompt", ""))
        response = clean_text(traj.get("response", ""))

        if not prompt or not response or len(prompt) < 10 or len(response) < 10:
            continue

        # Dedup on prompt
        key = prompt[:100].lower()
        if key in seen:
            continue
        seen.add(key)

        inscription = classify_inscription(record)
        reward = record.get("reward_score", 0.5)

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt[:2000]},
            {"role": "assistant", "content": response[:2000]},
        ]

        examples.append({
            "messages": messages,
            "inscription": inscription,
            "reward": round(reward, 4),
            "source": "trajectory",
        })

    return examples


def load_factory_pairs() -> list[dict]:
    """Load V7.2 factory pairs as SFT examples."""
    if not FACTORY_PATH.exists():
        print(f"WARNING: {FACTORY_PATH} not found")
        return []

    examples = []
    for line in open(FACTORY_PATH):
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue

        prompt = record.get("prompt", "")
        if not prompt or len(prompt) < 15:
            continue

        # Factory pairs are prompts Mohamed would type — pair with a synthetic response
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"[Session: {record.get('phase', 'BUILD')}] What should I do next?"},
            {"role": "assistant", "content": prompt},
        ]

        examples.append({
            "messages": messages,
            "inscription": "exploration" if record.get("phase") == "EXPLORE" else "stabilization",
            "reward": record.get("style_score", 0.9),
            "source": "v7_factory",
        })

    return examples


def load_dpo_pairs() -> list[dict]:
    """Load DPO preference pairs."""
    if not DPO_PATH.exists():
        print(f"WARNING: {DPO_PATH} not found")
        return []

    pairs = []
    for line in open(DPO_PATH):
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue

        if not record.get("chosen") or not record.get("rejected"):
            continue

        pairs.append({
            "prompt": record.get("prompt", ""),
            "chosen": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": record["prompt"]},
                {"role": "assistant", "content": record["chosen"][:2000]},
            ],
            "rejected": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": record["prompt"]},
                {"role": "assistant", "content": record["rejected"][:2000]},
            ],
            "reward_delta": record.get("reward_delta", 0),
            "source": record.get("source", "trajectory_store"),
        })

    return pairs


def main():
    random.seed(42)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=== KARL V7 Data Preparation ===\n")

    # Load all SFT sources
    traj_examples = load_trajectories()
    print(f"Trajectory SFT examples: {len(traj_examples)}")

    factory_examples = load_factory_pairs()
    print(f"Factory SFT examples: {len(factory_examples)}")

    all_sft = traj_examples + factory_examples
    random.shuffle(all_sft)

    # 90/10 split
    split_idx = int(len(all_sft) * 0.9)
    train_sft = all_sft[:split_idx]
    eval_sft = all_sft[split_idx:]

    # Load DPO
    dpo_pairs = load_dpo_pairs()
    print(f"DPO preference pairs: {len(dpo_pairs)}")

    # Write
    sft_train_path = OUT_DIR / "v7-sft-train.jsonl"
    sft_eval_path = OUT_DIR / "v7-sft-eval.jsonl"
    dpo_train_path = OUT_DIR / "v7-dpo-train.jsonl"

    with open(sft_train_path, "w") as f:
        for ex in train_sft:
            f.write(json.dumps(ex) + "\n")

    with open(sft_eval_path, "w") as f:
        for ex in eval_sft:
            f.write(json.dumps(ex) + "\n")

    with open(dpo_train_path, "w") as f:
        for p in dpo_pairs:
            f.write(json.dumps(p) + "\n")

    print(f"\nSFT train: {len(train_sft)} -> {sft_train_path}")
    print(f"SFT eval:  {len(eval_sft)} -> {sft_eval_path}")
    print(f"DPO train: {len(dpo_pairs)} -> {dpo_train_path}")

    # Stats
    from collections import Counter
    sources = Counter(ex["source"] for ex in train_sft)
    inscriptions = Counter(ex["inscription"] for ex in train_sft)
    print(f"\nSFT sources: {dict(sources)}")
    print(f"Inscriptions: {dict(inscriptions)}")
    print(f"\nTotal training examples: {len(train_sft)} SFT + {len(dpo_pairs)} DPO")


if __name__ == "__main__":
    main()
