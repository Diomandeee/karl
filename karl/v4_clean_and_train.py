#!/usr/bin/env python3
"""v4_clean_and_train.py - Clean data + export for KARL v4 training."""

import json, hashlib, random, re, os
from pathlib import Path
from collections import defaultdict

KARL_DIR = Path(__file__).parent
STORE_PATH = KARL_DIR / "trajectories.jsonl"
OUTPUT_DIR = Path.home() / "Desktop" / "karl-v4-data"

SYSTEM = (
    "You are Mohamed's cognitive twin. You have learned from months of his real "
    "software engineering sessions across iOS development, Rust daemons, mesh networking, "
    "machine learning pipelines, and creative production. You think like him: direct, "
    "technical, opinionated. You know the specific tools (meshd, bridged, NATS, OPA, "
    "MLX, Thunder-Train, MotionMix, KARL) and make concrete recommendations. "
    "Short sentences. No corporate speak. Get to the point."
)

TAG_RE = re.compile(r'<[^>]{1,100}>')
REPEAT_RE = re.compile(r'(.{30,}?)\1{2,}')


def clean_response(text):
    """Strip tool XML artifacts and repeated blocks."""
    text = TAG_RE.sub('', text)
    text = REPEAT_RE.sub(r'\1', text)
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text.strip()


def content_hash(prompt, response):
    return hashlib.md5((prompt + response).encode()).hexdigest()


def main():
    with open(STORE_PATH) as f:
        records = [json.loads(l) for l in f if l.strip()]
    print(f"Raw trajectories: {len(records)}")

    examples = []
    seen = set()
    stats = defaultdict(int)

    for r in records:
        traj = r.get("trajectory", {})
        prompt = traj.get("prompt", "").strip()[:4000]   # FIX: 4000 char limit (was 2000)
        response = traj.get("response", "").strip()[:4000]

        if not prompt or not response:
            stats["empty"] += 1
            continue

        if len(prompt) < 50:
            stats["short_prompt"] += 1
            continue

        if len(response) < 50:
            stats["short_response"] += 1
            continue

        response = clean_response(response)
        if len(response) < 50:
            stats["artifacts_only"] += 1
            continue

        h = content_hash(prompt, response)
        if h in seen:
            stats["duplicate"] += 1
            continue
        seen.add(h)

        reward = r.get("reward_score", 0.5)

        examples.append({
            "messages": [
                {"role": "system", "content": SYSTEM},
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": response},
            ],
            "reward": reward,
            "domain": r.get("domain", r.get("context", {}).get("domain", "unknown")),
        })

    print(f"Clean examples: {len(examples)}")
    print(f"Filtered: {json.dumps(dict(stats), indent=2)}")

    # Split 90/10 BEFORE oversampling (FIX: prevents leakage)
    random.seed(42)
    random.shuffle(examples)
    split = max(1, int(len(examples) * 0.9))
    train_raw = examples[:split]
    valid_raw = examples[split:]

    # Oversample high-reward in train only (max 2x, FIX: was 3x)
    mean_reward = sum(e["reward"] for e in train_raw) / max(len(train_raw), 1)
    train = []
    for e in train_raw:
        train.append(e)
        if e["reward"] > mean_reward + 0.05:
            train.append(e)  # 2x max

    print(f"\nTrain: {len(train)} (from {len(train_raw)} raw, {len(train)-len(train_raw)} oversampled)")
    print(f"Valid: {len(valid_raw)} (zero overlap guaranteed)")
    print(f"Mean reward: {mean_reward:.3f}")

    # Write
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_DIR / "train.jsonl", "w") as f:
        for e in train:
            f.write(json.dumps({"messages": e["messages"]}) + "\n")
    with open(OUTPUT_DIR / "valid.jsonl", "w") as f:
        for e in valid_raw:
            f.write(json.dumps({"messages": e["messages"]}) + "\n")

    print(f"\nWritten to {OUTPUT_DIR}")
    print(f"  train.jsonl: {len(train)} examples")
    print(f"  valid.jsonl: {len(valid_raw)} examples")

    # Domain breakdown
    domains = defaultdict(int)
    for e in train:
        domains[e["domain"]] += 1
    print(f"\nTop domains in train:")
    for d, c in sorted(domains.items(), key=lambda x: -x[1])[:10]:
        print(f"  {d}: {c}")


if __name__ == "__main__":
    main()
