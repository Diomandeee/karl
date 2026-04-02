#!/usr/bin/env python3
"""Benchmark open-source models for Mohamed roleplay quality.

Tests each model on 5 session contexts, scores with style_validator,
ranks by quality + cost efficiency.

Usage:
    python3 -m karl.v7.model_benchmark
    python3 -m karl.v7.model_benchmark --api-key YOUR_KEY
    python3 -m karl.v7.model_benchmark --provider openai  # uses OPENAI_API_KEY
"""

import argparse
import json
import os
import time
import urllib.request
from dataclasses import dataclass

from .style_validator import validate
from .session_context_reader import SessionContext
from .agent_roleplay import _get_exemplars, _get_domain_exemplars, _get_correction_patterns, _build_multi_chain_messages

# Models to benchmark
TOGETHER_MODELS = [
    {"id": "zai-org/GLM-5", "name": "GLM-5", "input_cost": 1.00, "output_cost": 3.20},
    {"id": "Qwen/Qwen3.5-397B-A17B", "name": "Qwen3.5 397B", "input_cost": 0.60, "output_cost": 3.60},
    {"id": "MiniMaxAI/MiniMax-M2.5", "name": "MiniMax M2.5", "input_cost": 0.30, "output_cost": 1.20},
    {"id": "moonshotai/Kimi-K2.5", "name": "Kimi K2.5", "input_cost": 0.50, "output_cost": 2.80},
    {"id": "zai-org/GLM-4.7", "name": "GLM 4.7", "input_cost": 0.45, "output_cost": 2.00},
    {"id": "openai/gpt-oss-120b", "name": "GPT-OSS 120B", "input_cost": 0.15, "output_cost": 0.60},
    {"id": "openai/gpt-oss-20b", "name": "GPT-OSS 20B", "input_cost": 0.05, "output_cost": 0.20},
    {"id": "deepseek-ai/DeepSeek-R1-0528", "name": "DeepSeek R1", "input_cost": 3.00, "output_cost": 7.00},
    {"id": "deepseek-ai/DeepSeek-V3.1", "name": "DeepSeek V3.1", "input_cost": 0.60, "output_cost": 1.70},
]

OPENAI_MODELS = [
    {"id": "gpt-5.4-mini", "name": "GPT-5.4-mini", "input_cost": 0.15, "output_cost": 0.60},
]

# Test contexts
TEST_CONTEXTS = [
    (SessionContext(
        files_created=["/Users/m/Desktop/mesh-event-viewer/src/main.rs"],
        project_dir="/Users/m/Desktop/mesh-event-viewer",
        last_claude_action="wrote main.rs",
        lines_of_code_written=200,
    ), "BUILD", "Build a Rust CLI that reads NATS JetStream events",
     ["Add clap CLI args", "Handle NATS reconnection"]),

    (SessionContext(
        files_modified=["/Users/m/Desktop/twin-api/main.py"],
        errors_seen=["ImportError: cannot import name 'TwinClient'"],
        tools_used={"Edit": 3, "Bash": 2},
        project_dir="/Users/m/Desktop/twin-api",
        last_claude_action="edited main.py",
    ), "BUILD", "Build a FastAPI service wrapping MLX twin at :8100",
     ["Fix imports", "Add /chat endpoint"]),

    (SessionContext(
        project_dir="/Users/m/Desktop/mesh-pulse",
        last_claude_action="scaffolded Next.js app",
        files_created=["/Users/m/Desktop/mesh-pulse/app/page.tsx"],
    ), "EXPLORE", "Build a Next.js dashboard polling meshd :9451 on 5 Macs",
     ["Add status cards", "Wire up server-side polling"]),

    (SessionContext(
        errors_seen=["ConnectionRefusedError: [Errno 111] Connection refused"],
        tools_used={"Bash": 8},
        project_dir="/Users/m/Desktop/nats-bridge",
        last_claude_action="ran: curl localhost:4222",
    ), "STUCK", "Set up NATS bridge for mesh events",
     ["Fix connection", "Test event publishing"]),

    (SessionContext(
        files_created=["/Users/m/Desktop/inscription-router/src/main.rs"],
        tools_used={"Write": 2, "Bash": 4},
        project_dir="/Users/m/Desktop/inscription-router",
        last_claude_action="ran: cargo test",
        lines_of_code_written=300,
    ), "CLOSE", "Build a Rust HTTP service classifying prompts into inscriptions",
     ["Final integration test"]),
]


def query_api(messages: list[dict], model_id: str, api_key: str,
              base_url: str = "https://api.together.xyz/v1") -> tuple[str | None, float]:
    """Query a chat API via curl (bypasses Cloudflare)."""
    import subprocess
    payload = json.dumps({
        "model": model_id,
        "messages": messages,
        "temperature": 0.85,
    })

    url = f"{base_url}/chat/completions"
    t0 = time.time()
    try:
        result = subprocess.run(
            ["curl", "-s", "--max-time", "30", url,
             "-H", f"Authorization: Bearer {api_key}",
             "-H", "Content-Type: application/json",
             "-d", payload],
            capture_output=True, text=True, timeout=35,
        )
        latency = time.time() - t0
        if result.returncode == 0 and result.stdout.strip():
            data = json.loads(result.stdout)
            if "choices" in data:
                return data["choices"][0]["message"]["content"].strip(), latency
        return None, latency
    except Exception:
        return None, time.time() - t0


def clean_response(raw: str) -> str:
    """Strip meta-commentary from roleplay output."""
    raw = raw.strip().strip('"\'').strip()
    lines = [l.strip() for l in raw.split("\n") if l.strip()]
    skip = ["here", "i would", "as mohamed", "the next", "this prompt",
            "note:", "explanation:", "output:", "prompt:", "sure", "based on", "given the"]
    for line in lines:
        if not any(line.lower().startswith(x) for x in skip):
            return line[:500]
    return lines[0][:500] if lines else ""


@dataclass
class ModelResult:
    model_name: str
    model_id: str
    avg_score: float
    pass_rate: float
    avg_latency: float
    input_cost: float
    output_cost: float
    prompts: list[str]
    scores: list[float]


def benchmark_model(model: dict, api_key: str, base_url: str,
                    exemplars: list[str], corrections: list[dict]) -> ModelResult:
    """Benchmark a single model across all test contexts."""
    prompts = []
    scores = []
    latencies = []

    for ctx, phase, goal, remaining in TEST_CONTEXTS:
        domain_ex = _get_domain_exemplars(goal)
        messages = _build_multi_chain_messages(
            ctx, goal, remaining, phase,
            exemplars, domain_ex, corrections,
        )

        raw, latency = query_api(messages, model["id"], api_key, base_url)
        latencies.append(latency)

        if raw:
            cleaned = clean_response(raw)
            if cleaned and len(cleaned) >= 15:
                score = validate(cleaned)
                prompts.append(cleaned)
                scores.append(score.overall)
            else:
                prompts.append(f"[EMPTY: {raw[:50]}]")
                scores.append(0.0)
        else:
            prompts.append("[FAILED]")
            scores.append(0.0)

    avg_score = sum(scores) / len(scores) if scores else 0
    pass_rate = sum(1 for s in scores if s >= 0.4) / len(scores) if scores else 0

    return ModelResult(
        model_name=model["name"],
        model_id=model["id"],
        avg_score=avg_score,
        pass_rate=pass_rate,
        avg_latency=sum(latencies) / len(latencies) if latencies else 0,
        input_cost=model["input_cost"],
        output_cost=model["output_cost"],
        prompts=prompts,
        scores=scores,
    )


def main():
    parser = argparse.ArgumentParser(description="Model benchmark for Mohamed roleplay")
    parser.add_argument("--api-key", default="", help="Together AI API key")
    parser.add_argument("--provider", default="together", choices=["together", "openai", "both"])
    args = parser.parse_args()

    # Resolve API keys
    together_key = args.api_key or os.environ.get("TOGETHER_API_KEY", "")
    openai_key = os.environ.get("OPENAI_API_KEY", "")

    # Load exemplars + corrections
    exemplars = _get_exemplars()
    corrections = _get_correction_patterns(20)
    print(f"Loaded {len(exemplars)} exemplars, {len(corrections)} corrections\n")

    models_to_test = []
    if args.provider in ("together", "both") and together_key:
        models_to_test.extend([(m, together_key, "https://api.together.xyz/v1") for m in TOGETHER_MODELS])
    if args.provider in ("openai", "both") and openai_key:
        models_to_test.extend([(m, openai_key, "https://api.openai.com/v1") for m in OPENAI_MODELS])

    if not models_to_test:
        print("ERROR: No API keys available. Set TOGETHER_API_KEY or OPENAI_API_KEY")
        return

    print(f"Testing {len(models_to_test)} models x {len(TEST_CONTEXTS)} contexts\n")

    results = []
    for model, key, url in models_to_test:
        print(f"Testing: {model['name']}...", end=" ", flush=True)
        result = benchmark_model(model, key, url, exemplars, corrections)
        results.append(result)
        print(f"score={result.avg_score:.2f} pass={result.pass_rate:.0%} latency={result.avg_latency:.1f}s")

    # Rank by score
    results.sort(key=lambda r: -r.avg_score)

    print(f"\n{'='*80}")
    print(f"{'Model':25s} {'Score':>6s} {'Pass':>5s} {'Latency':>8s} {'$/1M in':>8s} {'$/1M out':>9s}")
    print(f"{'='*80}")
    for r in results:
        print(f"{r.model_name:25s} {r.avg_score:6.2f} {r.pass_rate:5.0%} {r.avg_latency:7.1f}s ${r.input_cost:7.2f} ${r.output_cost:8.2f}")

    print(f"\n{'='*80}")
    print(f"BEST: {results[0].model_name} (score={results[0].avg_score:.2f})")
    print(f"\nSample prompts from winner:")
    for i, p in enumerate(results[0].prompts):
        print(f"  [{i+1}] {p[:150]}")

    # Save results
    out = os.path.expanduser("~/Desktop/karl/v7-model-benchmark.json")
    with open(out, "w") as f:
        json.dump([{
            "model": r.model_name, "id": r.model_id,
            "avg_score": r.avg_score, "pass_rate": r.pass_rate,
            "avg_latency": r.avg_latency, "prompts": r.prompts, "scores": r.scores,
        } for r in results], f, indent=2)
    print(f"\nResults saved to {out}")


if __name__ == "__main__":
    main()
