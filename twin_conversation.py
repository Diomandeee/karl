#!/usr/bin/env python3
"""KARL V5 — Autonomous Conversation Engine.

Two-agent loop:
  INTERVIEWER (base Qwen3 Instruct, no adapter) asks probing follow-ups
  TWIN (V5 adapter) responds with opinions and decisions

The conversation runs autonomously. Each exchange builds on the previous.
Logs full transcripts to ~/Desktop/karl/conversations/
"""

import json, time, datetime, os, random, sys
import urllib.request

MLX_URL = "http://localhost:8100/v1/chat/completions"
MODEL = "mlx-community/Qwen3-4B-Instruct-2507-4bit"
CONV_DIR = os.path.expanduser("~/Desktop/karl/conversations")
os.makedirs(CONV_DIR, exist_ok=True)

TWIN_SYSTEM = (
    "You are Mohamed's cognitive twin. You have deep knowledge of his actual projects: "
    "meshd (Rust PTY daemon on 5 Macs), NATS+OPA policy mesh, KARL trajectory intelligence, "
    "MotionMix (fitness + motion detection iOS app), 46 iOS apps on TestFlight, "
    "bridged (Rust unified bridge), Thunder-Train (distributed MLX training), "
    "Supabase (141 tables, 329K turns), Grand Diomande consulting, Koatji plant-based milk. "
    "You think like Mohamed: direct, technical, opinionated. Short sentences. "
    "No corporate speak. Make concrete decisions. Name specific files, tools, ports. "
    "If you don't know, say so. Never hedge when you have a real opinion."
)

INTERVIEWER_SYSTEM = (
    "You are a sharp technical interviewer. You're talking to Mohamed's cognitive twin, "
    "an AI trained on his real engineering sessions. Your job is to probe deeper. "
    "Ask follow-up questions that challenge the twin's reasoning. "
    "Push for specifics: numbers, tradeoffs, what could go wrong. "
    "One question at a time. Keep it under 2 sentences. "
    "If the twin gives a weak answer, call it out."
)

SCENARIOS = [
    {
        "title": "Mesh Architecture Review",
        "seed": "You're running 5 Macs with NATS, OPA, meshd, bridged, and Supabase as your mesh backbone. Walk me through the biggest architectural risk and how you'd fix it today.",
    },
    {
        "title": "KARL Training Strategy",
        "seed": "KARL V5 just finished training on a 4090 with inscription conditioning. You got 2.051 NLL. The model has opinions but it's on 2,540 examples. What's the path to a real cognitive twin?",
    },
    {
        "title": "Revenue Reality Check",
        "seed": "You have 46 iOS apps, a consulting brand, and a plant-based milk company. $640/mo target by June. Give me the honest breakdown of what's actually going to generate revenue.",
    },
    {
        "title": "iOS App Fleet Triage",
        "seed": "46 apps on TestFlight. 5 generating revenue. You're one person. Walk me through which apps survive and which get killed, and why.",
    },
    {
        "title": "Distributed Training Architecture",
        "seed": "You have Thunder-Train on Mac4+Mac5 over Thunderbolt 5, MLX server on Mac5, and Vast.ai RTX 4090s on demand. Design the optimal training pipeline for the next 6 months.",
    },
    {
        "title": "Cognitive Twin Product Viability",
        "seed": "CognitiveHire replaces interviews with cognitive twins. You just built one for yourself. Is this a real product or a demo? What would it take to sell this?",
    },
    {
        "title": "Infrastructure Debt",
        "seed": "Mac1 orchestrates everything but runs out of disk. 10 LaunchAgents conflict on startup. Pane injection has a 30% miss rate. Rate limiting hits 5/6 machines. Is the whole mesh a house of cards?",
    },
    {
        "title": "Creative Production Pipeline",
        "seed": "MotionMix records 4K on phones, Mac1 handles audio, Mac4 runs Premiere, Mac2 runs TouchDesigner. The RecordingService has crashed 3 times this month. How do you stabilize this for actual content production?",
    },
]


def query(system, messages, max_tokens=300, temp=0.8):
    """Query the MLX server."""
    full_messages = [{"role": "system", "content": system}] + messages
    payload = json.dumps({
        "model": MODEL,
        "messages": full_messages,
        "max_tokens": max_tokens,
        "temperature": temp,
    }).encode()

    req = urllib.request.Request(MLX_URL, data=payload, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=120) as resp:
        data = json.loads(resp.read())
        return data["choices"][0]["message"]["content"].replace("<|im_end|>", "").strip()


def run_conversation(scenario, turns=8):
    """Run a multi-turn autonomous conversation."""
    title = scenario["title"]
    ts = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    log_path = os.path.join(CONV_DIR, f"{ts}_{title.lower().replace(' ', '-')}.json")

    print(f"\n{'='*70}")
    print(f"  SCENARIO: {title}")
    print(f"  {turns} turns, logging to {os.path.basename(log_path)}")
    print(f"{'='*70}\n")

    history = []
    transcript = []

    # Seed question from interviewer
    seed = scenario["seed"]
    print(f"[INTERVIEWER] {seed}\n")
    history.append({"role": "user", "content": seed})
    transcript.append({"role": "interviewer", "content": seed, "turn": 0})

    for turn in range(turns):
        # TWIN responds
        t0 = time.time()
        twin_response = query(TWIN_SYSTEM, history, max_tokens=350, temp=0.8)
        twin_time = round(time.time() - t0, 2)

        print(f"[TWIN] ({twin_time}s)")
        print(f"{twin_response}\n")

        history.append({"role": "assistant", "content": twin_response})
        transcript.append({
            "role": "twin", "content": twin_response,
            "turn": turn + 1, "elapsed": twin_time
        })

        if turn >= turns - 1:
            break

        # INTERVIEWER follows up
        # Give interviewer the full conversation so far to generate a follow-up
        interviewer_messages = []
        for h in history:
            if h["role"] == "user":
                interviewer_messages.append({"role": "assistant", "content": h["content"]})
            else:
                interviewer_messages.append({"role": "user", "content": h["content"]})

        interviewer_messages.append({
            "role": "user",
            "content": "Based on what the twin just said, ask ONE sharp follow-up question. Push deeper. Challenge weak points. Be specific."
        })

        t0 = time.time()
        follow_up = query(INTERVIEWER_SYSTEM, interviewer_messages, max_tokens=100, temp=0.9)
        int_time = round(time.time() - t0, 2)

        print(f"[INTERVIEWER] ({int_time}s)")
        print(f"{follow_up}\n")

        history.append({"role": "user", "content": follow_up})
        transcript.append({
            "role": "interviewer", "content": follow_up,
            "turn": turn + 1, "elapsed": int_time
        })

        time.sleep(2)

    # Save transcript
    record = {
        "scenario": title,
        "timestamp": ts,
        "turns": len(transcript),
        "transcript": transcript,
    }
    with open(log_path, "w") as f:
        json.dump(record, f, indent=2)

    print(f"\n  Saved: {log_path}")
    return record


def main():
    print("KARL V5 — Autonomous Conversation Engine")
    print(f"  MLX server: {MLX_URL}")
    print(f"  Scenarios: {len(SCENARIOS)}")
    print(f"  Output: {CONV_DIR}/")

    # Run all scenarios
    random.seed(int(time.time()))
    scenarios = list(SCENARIOS)
    random.shuffle(scenarios)

    for i, scenario in enumerate(scenarios):
        print(f"\n[{i+1}/{len(scenarios)}]", flush=True)
        try:
            run_conversation(scenario, turns=6)
        except Exception as e:
            print(f"  ERROR: {e}")
        time.sleep(5)

    print(f"\n{'='*70}")
    print(f"  All {len(scenarios)} conversations complete.")
    print(f"  Transcripts in {CONV_DIR}/")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
