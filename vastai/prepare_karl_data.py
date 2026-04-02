#!/usr/bin/env python3
"""Prepare KARL trajectory data for Vast.ai Anticipatory Transformer training.

Converts KARL trajectories (3,190 records) into anticipation-format JSONL with:
- 7 trajectory scalars derived from KARL reward signals
- Inscription classification from scalar geometry
- 3-source split: real conversations, high-reward, synthetic structured

Output format matches anticipation-geometry/vastai/src/data.py expectations:
  {
    "messages": [...],
    "scalars": {"commitment": float, "uncertainty": float, ...},
    "inscription": str,
    "position": float
  }
"""

import hashlib
import json
import random
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

KARL_DIR = Path(__file__).parent.parent / "karl"
TRAJ_PATH = KARL_DIR / "trajectories.jsonl"
OUT_DIR = Path(__file__).parent / "data"

# Twin system prompt (cognitive twin personality)
SYSTEM_PROMPT = (
    "You are Mohamed's cognitive twin. You have learned from months of his real "
    "software engineering sessions across iOS development, Rust daemons, mesh networking, "
    "machine learning pipelines, and creative production. You think like him: direct, "
    "technical, opinionated. You know the specific tools (meshd, bridged, NATS, OPA, "
    "MLX, Thunder-Train, MotionMix, KARL) and make concrete recommendations. "
    "Short sentences. No corporate speak. Get to the point."
)

INSCRIPTIONS = [
    "stabilization", "transition", "oscillation", "correction", "exploration",
    "convergence", "expansion", "regression", "stagnation", "completion",
]

TAG_RE = re.compile(r'<[^>]{1,100}>')
REPEAT_RE = re.compile(r'(.{30,}?)\1{2,}')


def clean_text(text):
    """Strip tool XML artifacts and repeated blocks."""
    text = TAG_RE.sub('', text)
    text = REPEAT_RE.sub(r'\1', text)
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text.strip()


def derive_scalars(record):
    """Derive 4 anticipation scalars from KARL trajectory metadata.

    Mapping:
    - commitment = f(outcome_score, process_score) - how fully committed the response is
    - uncertainty = f(1 - tool_success_rate, has_error) - response uncertainty
    - transition_pressure = f(tool_diversity, domain_complexity) - pressure to change approach
    - recovery_margin = f(efficiency_score, completed) - capacity to recover from errors
    """
    outcome = record.get("outcome", {})
    ctx = record.get("context", {})
    traj = record.get("trajectory", {})

    outcome_score = float(record.get("outcome_score", 0.5))
    process_score = float(record.get("process_score", 0.5))
    efficiency_score = float(record.get("efficiency_score", 0.5))
    reward = float(record.get("reward_score", 0.5))
    tool_success = float(outcome.get("tool_success_rate", 1.0))
    has_error = bool(outcome.get("has_error", False))
    completed = bool(outcome.get("completed", True))
    tool_count = int(traj.get("tool_count", 0))
    tool_types = traj.get("tool_types", [])
    n_unique_tools = len(set(tool_types)) if tool_types else 0
    tool_diversity = n_unique_tools / max(len(tool_types), 1) if tool_types else 0.5

    # Commitment: spread across [0,1] using full signal range
    # Low tools + low outcome = low commitment; many tools + high outcome = high
    commitment = (
        outcome_score * 0.3 +
        process_score * 0.2 +
        min(tool_count / 10.0, 1.0) * 0.2 +
        (1.0 if completed else 0.0) * 0.15 +
        reward * 0.15
    )

    # Uncertainty: spread using multiple failure signals + noise from tool diversity
    uncertainty = (
        (1.0 - tool_success) * 0.3 +
        (0.5 if has_error else 0.0) +
        (0.3 if not completed else 0.0) +
        (1.0 - outcome_score) * 0.2 +
        tool_diversity * 0.15  # More diverse tools = more uncertainty/exploration
    )
    uncertainty = max(0.0, min(1.0, uncertainty))

    # Transition pressure: center around 0, wide spread
    # Many tools, diverse types, incomplete = high pressure
    tp_raw = (
        tool_diversity * 0.4 +
        min(tool_count / 15.0, 1.0) * 0.3 +
        (0.3 if not completed else 0.0) -
        process_score * 0.3  # High process = low pressure
    )
    transition_pressure = max(-1.0, min(1.0, tp_raw))

    # Recovery margin: spread using efficiency + completion + inverse uncertainty
    recovery_margin = (
        efficiency_score * 0.4 +
        (1.0 if completed else 0.0) * 0.2 +
        tool_success * 0.2 +
        (1.0 - uncertainty) * 0.2
    )

    return {
        "commitment": round(max(0, min(1, commitment)), 6),
        "uncertainty": round(max(0, min(1, uncertainty)), 6),
        "transition_pressure": round(max(-1, min(1, transition_pressure)), 6),
        "recovery_margin": round(max(0, min(1, recovery_margin)), 6),
    }


def classify_inscription(scalars):
    """Classify behavioral motif from scalar geometry (same as anticipation-geometry)."""
    c = scalars["commitment"]
    u = scalars["uncertainty"]
    tp = scalars["transition_pressure"]
    rm = scalars["recovery_margin"]

    if c > 0.8 and u < 0.3:
        return "convergence" if tp < 0.1 else "completion"
    if u > 0.7:
        return "oscillation" if tp > 0.3 else "exploration"
    if tp > 0.5:
        return "transition"
    if rm < 0.3:
        return "regression"
    if c > 0.6 and u < 0.4:
        return "stabilization"
    if tp < -0.2:
        return "stagnation"
    if c < 0.3:
        return "expansion" if u > 0.4 else "correction"
    return "stabilization"


def convert_trajectory(record, position=0.5):
    """Convert a KARL trajectory to anticipation format."""
    traj = record.get("trajectory", {})
    prompt = traj.get("prompt", "").strip()
    response = traj.get("response", "").strip()

    if not prompt or not response:
        return None

    # Clean
    prompt = clean_text(prompt)[:4000]
    response = clean_text(response)[:4000]

    if len(prompt) < 50 or len(response) < 50:
        return None

    scalars = derive_scalars(record)
    inscription = classify_inscription(scalars)

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": prompt},
        {"role": "assistant", "content": response},
    ]

    return {
        "messages": messages,
        "scalars": scalars,
        "inscription": inscription,
        "position": round(position, 4),
        "domain": record.get("domain", "unknown"),
        "reward": float(record.get("reward_score", 0.5)),
    }


def write_jsonl(records, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")
    print(f"  {path.name}: {len(records)} records")


def main():
    random.seed(42)

    print("Loading KARL trajectories...")
    with open(TRAJ_PATH) as f:
        raw = [json.loads(l) for l in f if l.strip()]
    print(f"  Raw: {len(raw)} trajectories")

    # Convert all with position derived from order within session
    session_groups = defaultdict(list)
    for r in raw:
        session_groups[r.get("session_id", "unknown")].append(r)

    all_records = []
    seen_hashes = set()
    stats = Counter()

    # First pass: derive raw scalars for all records
    raw_with_scalars = []
    for session_id, session_records in session_groups.items():
        for i, r in enumerate(session_records):
            position = (i + 1) / max(len(session_records), 1)
            traj = r.get("trajectory", {})
            prompt = clean_text(traj.get("prompt", "").strip())[:4000]
            response = clean_text(traj.get("response", "").strip())[:4000]
            if not prompt or not response or len(prompt) < 50 or len(response) < 50:
                stats["filtered"] += 1
                continue
            h = hashlib.md5((prompt + response).encode()).hexdigest()
            if h in seen_hashes:
                stats["duplicate"] += 1
                continue
            seen_hashes.add(h)
            scalars = derive_scalars(r)
            raw_with_scalars.append((r, scalars, position, prompt, response))

    # Percentile normalization: spread each scalar to [0,1] using rank
    if raw_with_scalars:
        for key in ["commitment", "uncertainty", "transition_pressure", "recovery_margin"]:
            vals = sorted(set(s[key] for _, s, _, _, _ in raw_with_scalars))
            rank_map = {v: i / max(len(vals) - 1, 1) for i, v in enumerate(vals)}
            for item in raw_with_scalars:
                item[1][key] = round(rank_map[item[1][key]], 6)

    # Second pass: classify inscriptions on normalized scalars
    for r, scalars, position, prompt, response in raw_with_scalars:
        inscription = classify_inscription(scalars)
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": response},
        ]
        all_records.append({
            "messages": messages,
            "scalars": scalars,
            "inscription": inscription,
            "position": round(position, 4),
            "domain": r.get("domain", "unknown"),
            "reward": float(r.get("reward_score", 0.5)),
        })

    print(f"  Clean: {len(all_records)} (filtered {stats['filtered']}, deduped {stats['duplicate']})")

    # Sort by reward for stratified splitting
    all_records.sort(key=lambda x: x["reward"], reverse=True)

    # Split into 3 sources based on reward tiers
    n = len(all_records)
    high_reward = [r for r in all_records if r["reward"] >= 0.6]     # Top tier
    mid_reward = [r for r in all_records if 0.45 <= r["reward"] < 0.6]  # Middle
    low_reward = [r for r in all_records if r["reward"] < 0.45]       # Lower

    print(f"\n  Reward tiers: high={len(high_reward)}, mid={len(mid_reward)}, low={len(low_reward)}")

    # Shuffle within tiers
    random.shuffle(high_reward)
    random.shuffle(mid_reward)
    random.shuffle(low_reward)

    # Source 1: real_conv — all tiers mixed (main training data)
    # Source 2: high_reward — oversampled high-quality examples (analogous to sft_106k)
    # Source 3: synthetic — structured examples for regularization

    # Split eval from real_conv (10% of total)
    all_shuffled = high_reward + mid_reward + low_reward
    random.shuffle(all_shuffled)
    eval_size = max(50, int(len(all_shuffled) * 0.1))
    eval_set = all_shuffled[:eval_size]
    train_pool = all_shuffled[eval_size:]

    # Source 1: real_conv_train — everything
    real_conv = train_pool

    # Source 2: high_reward_train — just the high-reward examples (will be oversampled via mix weight)
    high_reward_train = [r for r in train_pool if r["reward"] >= 0.6]

    # Source 3: Generate synthetic structured with known scalar patterns
    synth = generate_synthetic_karl(500)

    print(f"\n  Train sources:")
    write_jsonl(real_conv, OUT_DIR / "real_conv_train.jsonl")
    write_jsonl(eval_set, OUT_DIR / "real_conv_eval.jsonl")
    write_jsonl(high_reward_train, OUT_DIR / "high_reward_train.jsonl")
    write_jsonl(synth, OUT_DIR / "synth_struct.jsonl")

    # Inscription distribution
    insc_counts = Counter(r["inscription"] for r in real_conv)
    print(f"\n  Inscription distribution:")
    for insc, cnt in insc_counts.most_common():
        print(f"    {insc}: {cnt} ({100*cnt/len(real_conv):.1f}%)")

    # Domain distribution
    domain_counts = Counter(r["domain"] for r in real_conv)
    print(f"\n  Domain distribution (top 10):")
    for dom, cnt in domain_counts.most_common(10):
        print(f"    {dom}: {cnt}")

    # Scalar stats
    print(f"\n  Scalar statistics:")
    for key in ["commitment", "uncertainty", "transition_pressure", "recovery_margin"]:
        vals = [r["scalars"][key] for r in real_conv]
        print(f"    {key}: mean={sum(vals)/len(vals):.3f}, min={min(vals):.3f}, max={max(vals):.3f}")


def generate_synthetic_karl(n, seed=42):
    """Generate synthetic examples with known behavioral patterns for regularization."""
    random.seed(seed)
    records = []

    templates = [
        # Mesh/infra
        ("The meshd daemon on {machine} is not responding on :9451. Debug.", "Check the LaunchAgent first."),
        ("Deploy {service} to {machine} and verify health.", "SSH in, pull latest, restart the systemd unit."),
        ("NATS JetStream on :4222 is lagging. {n} pending messages.", "Check consumer ack policy and max_deliver."),
        # iOS
        ("Build and deploy {app} to TestFlight.", "Archive with Xcode, upload via altool, wait for processing."),
        ("The {view} in {app} crashes on appear. SwiftUI.", "Check for force-unwrapped optionals in the view model."),
        # ML
        ("Train a LoRA adapter on {model} with {n} examples.", "Use mlx_lm lora with batch_size=1 on Mac5."),
        ("The validation loss plateaued at {loss}. Next steps?", "Try lower learning rate, more layers, or DPO."),
        # Creative
        ("Generate a {style} video for {platform}.", "Use the 7-element cinematic formula."),
    ]

    machines = ["Mac1", "Mac2", "Mac4", "Mac5", "cloud-vm"]
    services = ["bridged", "meshd", "NATS", "OPA", "pane-orchestrator"]
    apps = ["Spore", "MeshControl", "MotionMix", "FirstDate", "CreatorShield"]
    views = ["DashboardView", "SettingsView", "RecordingView", "TopologyView"]
    models = ["Qwen3-4B", "Qwen2.5-7B", "Llama-3.1-8B"]

    for i in range(n):
        tmpl_user, tmpl_asst = random.choice(templates)
        user = tmpl_user.format(
            machine=random.choice(machines),
            service=random.choice(services),
            app=random.choice(apps),
            view=random.choice(views),
            model=random.choice(models),
            n=random.randint(100, 50000),
            loss=round(random.uniform(1.5, 3.5), 3),
            style=random.choice(["cinematic", "documentary", "motion-graphics"]),
            platform=random.choice(["Instagram", "TikTok", "YouTube"]),
        )
        asst = tmpl_asst + f" This is domain-specific knowledge example #{i}."

        # Known scalar patterns
        pattern = random.choice(INSCRIPTIONS)
        if pattern == "convergence":
            scalars = {"commitment": 0.85, "uncertainty": 0.1, "transition_pressure": 0.05, "recovery_margin": 0.9}
        elif pattern == "exploration":
            scalars = {"commitment": 0.4, "uncertainty": 0.6, "transition_pressure": 0.2, "recovery_margin": 0.7}
        elif pattern == "oscillation":
            scalars = {"commitment": 0.5, "uncertainty": 0.75, "transition_pressure": 0.4, "recovery_margin": 0.5}
        elif pattern == "regression":
            scalars = {"commitment": 0.3, "uncertainty": 0.6, "transition_pressure": 0.1, "recovery_margin": 0.2}
        else:
            scalars = {
                "commitment": random.uniform(0.3, 0.8),
                "uncertainty": random.uniform(0.1, 0.7),
                "transition_pressure": random.gauss(0, 0.2),
                "recovery_margin": random.uniform(0.3, 1.0),
            }

        scalars = {k: round(max(0, min(1, v)), 6) for k, v in scalars.items()}

        records.append({
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user},
                {"role": "assistant", "content": asst},
            ],
            "scalars": scalars,
            "inscription": pattern,
            "position": round(random.uniform(0, 1), 4),
            "domain": "synthetic",
            "reward": 0.7,
        })

    return records


if __name__ == "__main__":
    main()
