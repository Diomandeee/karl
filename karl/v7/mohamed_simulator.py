"""Mohamed Simulator V7.2: Agent roleplay + context-dense fallback.

V7.2: Primary strategy is agent_roleplay — Claude roleplays as Mohamed
using 20 real exemplars + session context + style rules. Falls back to
context_compose (V7.1) when the agent is unavailable or rate-limited.

Hierarchy:
  1. agent_roleplay (Claude --print as Mohamed) — highest quality
  2. context_compose (goal-aware templates) — fast local fallback
  3. rag_nearest (TF-IDF from 4,587 real prompts) — exploration
  4. goal_compose (plan-step framing) — close/stuck phases
"""

import os
import random
import re
from dataclasses import dataclass

from .prompt_corpus import search_similar, get_corpus, PromptEntry
from .style_validator import validate, StyleScore
from .session_context_reader import SessionContext
from .agent_roleplay import generate_as_mohamed

# Phase -> strategy mapping (agent_roleplay is always tried first)
PHASE_STRATEGY = {
    "EXPLORE": "agent_roleplay",
    "BUILD": "agent_roleplay",
    "CLOSE": "agent_roleplay",
    "STUCK": "agent_roleplay",
    "IDLE": "rag_nearest",
}

# Mohamed's natural connectors
CONNECTORS = [
    "and then", "so", "now", "also", "but first",
    "after that", "make sure", "and", "then",
    "once that's done", "before we move on",
]

# Infrastructure vocabulary per project archetype
INFRA_CONTEXT = {
    "rust": {
        "build_cmd": "cargo build --release",
        "test_cmd": "cargo test",
        "lint_cmd": "cargo clippy",
        "verify": ["make sure cargo build passes with no warnings",
                    "run cargo clippy and fix any lints",
                    "make sure all error types implement std::error::Error"],
        "vocab": ["crate", "module", "impl", "trait", "struct", "enum",
                  "tokio", "async-nats", "clap", "serde", "axum"],
    },
    "python": {
        "build_cmd": "pip install -e .",
        "test_cmd": "pytest -v",
        "lint_cmd": "ruff check .",
        "verify": ["run pytest and make sure all tests pass",
                    "check the types with mypy",
                    "make sure the Pydantic models validate correctly"],
        "vocab": ["FastAPI", "endpoint", "Pydantic", "httpx", "async",
                  "decorator", "dependency injection", "middleware"],
    },
    "nextjs": {
        "build_cmd": "npm run build",
        "test_cmd": "npm run dev",
        "lint_cmd": "npm run lint",
        "verify": ["run npm run build and fix any TypeScript errors",
                    "check that the page renders on localhost:3000",
                    "make sure Tailwind classes are working"],
        "vocab": ["component", "server component", "client component", "shadcn",
                  "Tailwind", "App Router", "server action", "fetch"],
    },
    "ios": {
        "build_cmd": "xcodebuild build",
        "test_cmd": "xcodebuild test",
        "lint_cmd": "swiftlint",
        "verify": ["make sure it builds in Xcode with no errors",
                    "check the SwiftUI preview renders correctly",
                    "make sure optionals are properly unwrapped"],
        "vocab": ["SwiftUI", "@State", "@ObservedObject", "NavigationStack",
                  "async/await", "URLSession", "Codable"],
    },
}

# Mesh infrastructure Mohamed references
MESH_CONTEXT = {
    "machines": ["Mac1", "Mac2", "Mac4", "Mac5", "cloud-vm"],
    "services": ["meshd :9451", "NATS :4222", "RAG++ :8000", "MLX :8100",
                 "Grafana :3000", "Prefect :4200", "OPA :8181"],
    "tailscale": {"mac1": "localhost", "mac2": "100.119.214.88",
                  "mac4": "100.91.231.93", "mac5": "100.109.94.124",
                  "cloud-vm": "100.114.92.88"},
}


@dataclass
class SimulatorResult:
    prompt: str
    strategy: str
    style_score: StyleScore
    source_prompt: str
    attempts: int


def _detect_archetype(goal: str, project_dir: str) -> str:
    """Detect project archetype from goal text and directory."""
    goal_l = goal.lower()
    if any(w in goal_l for w in ["rust", "cargo", "axum", "tokio"]):
        return "rust"
    if any(w in goal_l for w in ["python", "fastapi", "flask", "prefect", "pytest"]):
        return "python"
    if any(w in goal_l for w in ["next", "react", "dashboard", "shadcn", "tailwind"]):
        return "nextjs"
    if any(w in goal_l for w in ["swift", "ios", "swiftui", "xcode"]):
        return "ios"
    return "python"  # default


def _file_ref(ctx: SessionContext) -> str | None:
    """Get a specific file reference from context."""
    files = ctx.files_created + ctx.files_modified
    if files:
        return os.path.basename(random.choice(files))
    return None


def _error_ref(ctx: SessionContext) -> str | None:
    """Get a specific error reference from context."""
    if ctx.errors_seen:
        err = ctx.errors_seen[-1]
        # Extract the error type/message, not the full trace
        parts = err.split(":")
        if len(parts) >= 2:
            return parts[0].strip() + ": " + parts[1].strip()[:50]
        return err[:60]
    return None


# === STRATEGY: context_compose (replaces template_compose) ===

def context_compose(ctx: SessionContext, goal: str = "",
                    plan_remaining: list[str] | None = None) -> tuple[str, str]:
    """Build a dense, context-specific prompt grounded in what Claude just did.

    Unlike template_compose, this references actual files, errors, plan steps,
    and infrastructure vocabulary.
    """
    arch = _detect_archetype(goal, ctx.project_dir)
    infra = INFRA_CONTEXT.get(arch, INFRA_CONTEXT["python"])
    parts = []

    # Layer 1: What to do next (from plan or context)
    file = _file_ref(ctx)
    error = _error_ref(ctx)

    if error and random.random() < 0.7:
        # Error-driven prompt — Mohamed's most common correction pattern
        templates = [
            f"the {error} needs to be fixed before we move on",
            f"I think the issue is {error}, can you trace it?",
            f"figure out why we're getting {error}",
            f"that {error} is blocking us, fix it first",
            f"don't just patch around {error}, find the root cause",
        ]
        parts.append(random.choice(templates))

    elif file and random.random() < 0.6:
        # File-driven prompt — reference what was just touched
        templates = [
            f"now that {file} is done, let's verify it works",
            f"I think {file} needs error handling for edge cases",
            f"check {file} compiles cleanly and run {infra['test_cmd']}",
            f"let's add tests for {file}",
            f"read through {file} one more time, I want to make sure the types are right",
            f"{file} looks good but make sure it handles the unhappy path too",
        ]
        parts.append(random.choice(templates))

    elif plan_remaining:
        # Plan-driven prompt — reference the next step with specifics
        next_step = plan_remaining[0].strip()
        # Add archetype vocabulary to make it specific
        vocab_word = random.choice(infra["vocab"])
        templates = [
            f"let's move on to: {next_step}",
            f"now {next_step.lower()}, keep in mind we're using {vocab_word}",
            f"can we {next_step.lower()} next?",
            f"go ahead and {next_step.lower()}",
            f"I think {next_step.lower()} is the priority right now",
        ]
        parts.append(random.choice(templates))

    else:
        # Goal-driven fallback — still more specific than "build the next feature"
        proj = os.path.basename(ctx.project_dir) if ctx.project_dir else "this"
        templates = [
            f"what's left to do on {proj}? let's keep moving",
            f"run {infra['test_cmd']} and show me where we stand",
            f"I think we should focus on the core logic next",
            f"can we wire up the main entry point for {proj}?",
            f"let's make sure {proj} actually runs end to end",
        ]
        parts.append(random.choice(templates))

    # Layer 2: Verification clause (40% chance)
    if random.random() < 0.4:
        verify = random.choice(infra["verify"])
        connector = random.choice(["and then", "also", "after that", "then"])
        parts.append(f"{connector} {verify.lower()}")

    # Layer 3: Infrastructure context (20% chance — Mohamed references mesh stuff)
    if random.random() < 0.2 and any(w in goal.lower() for w in ["mesh", "nats", "health", "deploy"]):
        service = random.choice(MESH_CONTEXT["services"])
        mesh_refs = [
            f"keep in mind {service} needs to be reachable",
            f"make sure it works when {service} is down",
            f"test against {service} on localhost",
        ]
        parts.append(random.choice(mesh_refs))

    prompt = ". ".join(p.rstrip(".") for p in parts if p)
    return prompt, ""


# === STRATEGY: goal_compose (replaces full_generate) ===

def goal_compose(ctx: SessionContext, goal: str = "",
                 plan_remaining: list[str] | None = None) -> tuple[str, str]:
    """Build a prompt from goal progress + what remains + session state."""
    arch = _detect_archetype(goal, ctx.project_dir)
    infra = INFRA_CONTEXT.get(arch, INFRA_CONTEXT["python"])
    parts = []

    # Opening: reference what was accomplished or what's next
    file = _file_ref(ctx)
    error = _error_ref(ctx)

    if plan_remaining and len(plan_remaining) <= 2:
        # Near the end — wrap-up mode
        closers = [
            f"we're almost done. {plan_remaining[0].lower()} and then let's commit",
            f"last thing: {plan_remaining[0].lower()}. then run {infra['test_cmd']} one more time",
            f"finish up {plan_remaining[0].lower()} and we can ship this",
        ]
        parts.append(random.choice(closers))

    elif plan_remaining:
        next_step = plan_remaining[0].strip()
        # Frame the step with what came before
        if ctx.last_claude_action:
            parts.append(f"ok, after {ctx.last_claude_action}, now {next_step.lower()}")
        else:
            frames = [
                f"let's {next_step.lower()} next",
                f"can we {next_step.lower()} now?",
                f"go ahead and {next_step.lower()}",
            ]
            parts.append(random.choice(frames))

    elif goal:
        # No plan remaining — reference the goal
        parts.append(f"let's keep working on {goal.lower()[:60]}")

    else:
        proj = os.path.basename(ctx.project_dir) if ctx.project_dir else "this"
        parts.append(f"what's the current state of {proj}? run {infra['test_cmd']}")

    # Add error awareness
    if error and random.random() < 0.5:
        parts.append(f"but first handle that {error}")

    # Add file-specific check
    if file and random.random() < 0.3 and not error:
        parts.append(f"make sure {file} is solid")

    # Signature phrase injection (30%)
    if random.random() < 0.3:
        sigs = [
            "just get it working first, we can clean up later",
            "don't over-engineer this",
            "make sure to test the edge cases",
            "keep it simple",
            f"I think {random.choice(infra['vocab'])} is the right call here",
        ]
        parts.append(random.choice(sigs))

    prompt = ". ".join(p.rstrip(".") for p in parts if p)
    return prompt, ""


# === STRATEGY: rag_nearest (improved domain filtering) ===

def rag_nearest(ctx: SessionContext, goal: str = "") -> tuple[str, str]:
    """Find similar real prompts with domain-aware filtering.

    V7.1: Filters by category, length, domain relevance.
    Falls back to context_compose if no good match.
    """
    keywords = ctx.keywords()
    if goal:
        goal_words = [w.lower() for w in goal.split() if len(w) > 3]
        keywords = goal_words[:5] + keywords[:5]
    if not keywords:
        keywords = ["build", "implement", "test"]
    query = " ".join(keywords[:12])

    results = search_similar(query, n=10)
    if not results:
        return context_compose(ctx, goal)

    # Strict filtering
    filtered = []
    for entry, score in results:
        text = entry.text
        # Length: 30-400 chars (no one-liners, no pastes)
        if len(text) > 400 or len(text) < 30:
            continue
        # Skip approvals and pastes
        if entry.category in ("paste", "approval"):
            continue
        # Skip terminal garbage
        if any(x in text.lower() for x in [
            "traceback", "```", "error:", "mohameddiomande@",
            "zsh:", "ssh ", "echo \"==", "bad interpreter",
            "/usr/bin/", "pip install", "npm install",
        ]):
            continue
        # Skip prompts with file paths (they won't match current project)
        if re.search(r"/Users/\w+/Desktop/\w+", text):
            continue
        # Prefer conversational and contextual categories
        if entry.category in ("conversational", "contextual", "question"):
            filtered.append((entry, score + 0.1))  # boost relevance
        else:
            filtered.append((entry, score))

    if not filtered:
        return context_compose(ctx, goal)

    # Sort by adjusted score and pick from top 3
    filtered.sort(key=lambda x: -x[1])
    idx = random.randint(0, min(2, len(filtered) - 1))
    entry, _score = filtered[idx]

    # Light adaptation: swap project names only if safe
    text = entry.text
    if ctx.project_dir and ctx.project_dir not in ("/tmp/test", "/tmp", ""):
        proj_name = os.path.basename(ctx.project_dir)
        if proj_name and len(proj_name) > 2:
            for generic in ["the project", "the app", "this app", "the repo"]:
                if generic in text.lower():
                    text = re.sub(re.escape(generic), proj_name, text,
                                  count=1, flags=re.IGNORECASE)
                    break

    return text, entry.text


# === Main generate function ===

def generate(
    ctx: SessionContext,
    phase: str = "BUILD",
    goal: str = "",
    plan_remaining: list[str] | None = None,
    max_attempts: int = 3,
    machine: str = "",
    pane_id: str = "",
) -> SimulatorResult:
    """Generate a Mohamed-style prompt for the current session state.

    Strategy hierarchy:
      1. agent_roleplay (Claude as Mohamed) — best quality, ~5-10s
      2. context_compose / rag_nearest — fast local fallback
      3. goal_compose — plan-step framing
    """
    # Try agent roleplay first (primary strategy for all phases)
    agent_prompt, agent_ok = generate_as_mohamed(
        ctx,
        goal,
        plan_remaining,
        phase,
        machine=machine,
        pane_id=pane_id,
    )
    if agent_ok and agent_prompt:
        score = validate(agent_prompt)
        if score.passed:
            return SimulatorResult(
                prompt=agent_prompt,
                strategy="agent_roleplay",
                style_score=score,
                source_prompt="",
                attempts=1,
            )

    # Fallback chain: context_compose -> rag_nearest -> goal_compose
    fallback_strategies = ["context_compose", "rag_nearest", "goal_compose"]
    # Reorder based on phase
    if phase == "EXPLORE":
        fallback_strategies = ["rag_nearest", "context_compose", "goal_compose"]
    elif phase in ("CLOSE", "STUCK"):
        fallback_strategies = ["goal_compose", "context_compose", "rag_nearest"]

    for attempt, strat in enumerate(fallback_strategies):
        if strat == "rag_nearest":
            prompt, source = rag_nearest(ctx, goal=goal)
        elif strat == "context_compose":
            prompt, source = context_compose(ctx, goal, plan_remaining)
        else:
            prompt, source = goal_compose(ctx, goal, plan_remaining)

        prompt = prompt.strip()
        if not prompt:
            continue

        score = validate(prompt)
        if score.passed or attempt == len(fallback_strategies) - 1:
            return SimulatorResult(
                prompt=prompt,
                strategy=strat,
                style_score=score,
                source_prompt=source,
                attempts=attempt + 2,  # +2 because agent was attempt 1
            )

    fallback = "status" if phase == "EXPLORE" else "continue"
    return SimulatorResult(
        prompt=fallback,
        strategy="fallback",
        style_score=validate(fallback),
        source_prompt="",
        attempts=max_attempts + 1,
    )


# === Test harness ===

def _make_synthetic_contexts():
    return [
        (SessionContext(
            files_created=["/Users/m/Desktop/app/src/main.rs"],
            project_dir="/Users/m/Desktop/mesh-event-viewer",
            last_claude_action="wrote main.rs",
        ), "EXPLORE", "Build a Rust CLI that reads NATS JetStream events", ["Set up Cargo.toml", "Implement NATS client"]),
        (SessionContext(
            files_modified=["/Users/m/Desktop/twin-api/main.py"],
            errors_seen=["ImportError: cannot import name 'TwinClient'"],
            tools_used={"Edit": 3, "Bash": 2},
            project_dir="/Users/m/Desktop/twin-api",
            last_claude_action="edited main.py",
            lines_of_code_written=45,
        ), "BUILD", "Build a Python FastAPI service wrapping the MLX twin", ["Fix the import", "Add /chat endpoint"]),
        (SessionContext(
            files_created=["/Users/m/Desktop/mesh-pulse/app/page.tsx"],
            tools_used={"Write": 2, "Bash": 1},
            project_dir="/Users/m/Desktop/mesh-pulse",
            last_claude_action="wrote page.tsx",
        ), "BUILD", "Build a Next.js dashboard polling meshd health endpoints", ["Add status cards", "Wire up polling"]),
        (SessionContext(
            project_dir="/Users/m/Desktop/speakd",
            last_claude_action="read README.md",
        ), "EXPLORE", "Investigate voice latency issue", []),
        (SessionContext(
            files_created=["/Users/m/Desktop/app/deploy.sh"],
            files_modified=["/Users/m/Desktop/app/config.json"],
            tools_used={"Write": 1, "Edit": 1, "Bash": 3},
            project_dir="/Users/m/Desktop/inscription-router",
            last_claude_action="committed changes",
            lines_of_code_written=80,
        ), "CLOSE", "Build a Rust HTTP service that classifies prompts into inscriptions", ["Run final tests"]),
        (SessionContext(
            errors_seen=["ConnectionRefusedError: [Errno 111] Connection refused"],
            tools_used={"Bash": 8},
            project_dir="/Users/m/Desktop/nats-bridge",
            last_claude_action="ran: curl localhost:4222",
        ), "STUCK", "Set up NATS bridge for mesh events", ["Fix connection", "Test bridge"]),
        (SessionContext(
            files_modified=["/Users/m/Desktop/ios/App/ContentView.swift"],
            tools_used={"Edit": 6, "Bash": 1},
            project_dir="/Users/m/Desktop/MeshMonitor",
            last_claude_action="edited ContentView.swift",
            lines_of_code_written=120,
        ), "BUILD", "Build a SwiftUI iOS app showing mesh node health", ["Implement theme toggle"]),
        (SessionContext(
            project_dir="/Users/m/Desktop/karl-data-scaler",
        ), "EXPLORE", "Build a Python script that pulls 329K rows from memory_turns", []),
        (SessionContext(
            files_created=["/Users/m/Desktop/api/src/routes.py", "/Users/m/Desktop/api/tests/test_routes.py"],
            tools_used={"Write": 2, "Bash": 4, "Read": 3},
            project_dir="/Users/m/Desktop/twin-api",
            last_claude_action="ran: pytest -v",
            lines_of_code_written=200,
        ), "CLOSE", "Build REST API for the cognitive twin", []),
        (SessionContext(
            files_modified=["/Users/m/Desktop/daemon/src/main.rs"],
            errors_seen=["error[E0382]: use of moved value: `stream`"],
            tools_used={"Edit": 4, "Bash": 6},
            project_dir="/Users/m/Desktop/karl-scorer",
            last_claude_action="ran: cargo build",
            lines_of_code_written=60,
        ), "BUILD", "Build a Rust binary that watches trajectories.jsonl and scores in real-time", ["Clone the stream handle", "Run cargo test"]),
    ]


def main():
    print("Mohamed Simulator V7.1 — Test Run")
    print("=" * 60)

    try:
        corpus = get_corpus()
        print(f"Corpus loaded: {len(corpus)} prompts")
    except Exception as e:
        print(f"Corpus not available ({e})")

    print()
    contexts = _make_synthetic_contexts()
    total_score = 0.0
    passed = 0

    for i, (ctx, phase, goal, remaining) in enumerate(contexts, 1):
        result = generate(ctx, phase=phase, goal=goal, plan_remaining=remaining)
        tag = "PASS" if result.style_score.passed else "FAIL"
        total_score += result.style_score.overall
        if result.style_score.passed:
            passed += 1

        print(f"[{i:2d}] Phase={phase:<8s} Strategy={result.strategy:<18s} "
              f"Score={result.style_score.overall:.2f} [{tag}] "
              f"Attempts={result.attempts}")
        print(f"     Prompt: {result.prompt[:120]}")
        if result.source_prompt:
            print(f"     Source: {result.source_prompt[:80]}")
        print()

    avg = total_score / len(contexts)
    print(f"Results: {passed}/{len(contexts)} passed, avg score {avg:.2f}")


if __name__ == "__main__":
    main()
