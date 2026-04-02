#!/usr/bin/env python3
"""KARL V5 Cognitive Twin — Continuous Opinion Generator.

Queries the twin on real project topics every 60s.
Logs to ~/Desktop/karl/opinions.jsonl
"""

import json, time, random, datetime, sys
import urllib.request

MLX_URL = "http://localhost:8100/v1/chat/completions"
MODEL = "mlx-community/Qwen3-4B-Instruct-2507-4bit"
LOG = "/Users/mohameddiomande/Desktop/karl/opinions.jsonl"

SYSTEM = (
    "You are Mohamed's cognitive twin. You have learned from months of his real "
    "software engineering sessions across iOS development, Rust daemons, mesh networking, "
    "machine learning pipelines, and creative production. You think like him: direct, "
    "technical, opinionated. You know the specific tools (meshd, bridged, NATS, OPA, "
    "MLX, Thunder-Train, MotionMix, KARL) and make concrete recommendations. "
    "Short sentences. No corporate speak. Get to the point."
)

PROMPTS = [
    # Architecture
    "We have 5 Macs, NATS, OPA, meshd, bridged, and Supabase. What's the single weakest link?",
    "Should meshd stay as a PTY daemon or become a proper container orchestrator?",
    "NATS JetStream vs Redis Streams for the mesh event bus. Pick one and defend it.",
    "OPA policies are under 5ms eval time. Is that fast enough for real-time mesh decisions?",
    "bridged replaces 15 Python/Node bridges with one Rust binary. What could still go wrong?",

    # KARL / ML
    "KARL trained on 2,540 examples with inscription conditioning. Next move to improve it?",
    "Should we do DPO or more SFT for cognitive twin style transfer?",
    "The anticipation head predicts behavioral scalars at 91% accuracy. What do we do with that signal?",
    "MLX on Mac5 vs Vast.ai RTX 4090 for training. When to use which?",
    "We have 329K conversation turns in Supabase. Best strategy to curate 10K high-quality training examples?",

    # iOS / Product
    "46 iOS apps. Most are dead weight. Give me the ruthless 3-step plan to get to 5 that matter.",
    "MotionMix records 4K video but the RecordingService has had 3 crash bugs in a month. Root cause?",
    "Should FirstDate be a standalone app or a feature inside another app?",
    "CreatorShield for content creators. What's the one feature that makes or breaks it?",
    "TestFlight beta testing is a mess with 46 apps. Fix the pipeline.",

    # Infrastructure
    "Mac1 is the orchestrator but keeps running out of disk. Structural fix, not a band-aid.",
    "We run LaunchAgents for 10 services. Half of them conflict on startup. Architecture fix?",
    "Syncthing between Mac1 and cloud-vm keeps having conflicts. Should we switch to something else?",
    "The pane orchestrator spawns Claude sessions across 5 machines. 30% miss rate on injection. Why?",
    "Rate limiting hits 5/6 machines simultaneously. Is the mesh architecture fundamentally broken?",

    # Strategy
    "Grand Diomande consulting: $2.5K audits, $5-15K MCP dev, $8-20K buildouts. Is the pricing right?",
    "Koatji plant-based milk vs consulting vs app revenue. Where should Mohamed focus next 90 days?",
    "CognitiveHire replaces interviews with cognitive twins. Is this a real product or a science project?",
    "The Worm Cascade runs 834 sessions with 4 LLM providers. Is this useful or just expensive?",
    "Revenue convergence crucible says $640/mo by June. Is that realistic given current state?",

    # Creative
    "MotionMix as a fitness app with motion detection. Competitor landscape?",
    "Cinematic AI video with Veo 3.1 and the 7-element formula. Is the quality good enough to sell?",
    "Content hierarchy: IG face, TikTok show, LinkedIn close, Substack depth. Missing anything?",
    "Calvin Klein model + AI builder. How do you brand that without it being cringe?",
    "SpeakFlow as a Wispr competitor at $49 lifetime. Sustainable?",
]

def query_twin(prompt):
    payload = json.dumps({
        "model": MODEL,
        "messages": [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": prompt},
        ],
        "max_tokens": 300,
        "temperature": 0.8,
    }).encode()

    req = urllib.request.Request(MLX_URL, data=payload, headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            data = json.loads(resp.read())
            return data["choices"][0]["message"]["content"].replace("<|im_end|>", "").strip()
    except Exception as e:
        return f"ERROR: {e}"

def main():
    print(f"KARL V5 Twin Opinion Loop — {len(PROMPTS)} prompts, logging to {LOG}")
    random.seed(int(time.time()))
    shuffled = list(PROMPTS)
    random.shuffle(shuffled)

    for i, prompt in enumerate(shuffled):
        ts = datetime.datetime.now().isoformat()
        print(f"\n[{i+1}/{len(shuffled)}] {ts}")
        print(f"Q: {prompt[:80]}...")

        t0 = time.time()
        response = query_twin(prompt)
        elapsed = round(time.time() - t0, 2)

        print(f"A: {response[:200]}...")
        print(f"({elapsed}s)")

        entry = {
            "timestamp": ts,
            "prompt": prompt,
            "response": response,
            "elapsed": elapsed,
        }
        with open(LOG, "a") as f:
            f.write(json.dumps(entry) + "\n")

        if i < len(shuffled) - 1:
            time.sleep(5)

    print(f"\nDone. {len(shuffled)} opinions logged to {LOG}")

if __name__ == "__main__":
    main()
