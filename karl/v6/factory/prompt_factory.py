#!/usr/bin/env python3
"""Generate rich, pre-corrected prompts for session spawning.

Takes correction pairs + project goals and produces prompts that
pre-face the corrections Mohamed would have made. Each prompt
contains constraints, context, and directness baked in.

The output prompts are designed to be injected into real Claude Code
sessions, where the full interaction lands in prompt logs as training data.
"""

import json
import os
import random
from dataclasses import dataclass, asdict
from pathlib import Path


@dataclass
class RichPrompt:
    project: str
    project_dir: str
    goal: str
    prompt: str
    preface_constraints: list[str]
    failure_modes_addressed: list[str]
    expected_tools: list[str]
    tags: list[str]


# Constraint templates derived from correction patterns
CONSTRAINT_TEMPLATES = {
    "over_explain": [
        "Execute directly. Don't explain your approach, just write the code.",
        "Skip the explanation. Write the implementation and run it.",
        "Just do it. No preamble, no plan summary, just code.",
    ],
    "wrong_direction": [
        "Don't use {anti_pattern}. Use {correct_approach} instead.",
        "Don't create new files unless absolutely necessary. Edit existing ones.",
        "Don't refactor surrounding code. Only change what's needed for this task.",
    ],
    "stalled": [
        "Start immediately. Read the relevant file and begin editing.",
        "Don't ask which approach to take. Pick the simplest one and go.",
        "If you're unsure, try the most direct approach first. Don't ask.",
    ],
    "corner_cut": [
        "Make sure you test the changes before saying they're done.",
        "Make sure error handling covers the edge cases, not just the happy path.",
        "Make sure the types are correct. Don't use any or force-unwraps.",
    ],
    "missing_context": [
        "Keep in mind this runs on {machine} with {service} on port {port}.",
        "Keep in mind the existing code uses {pattern}. Match that style.",
        "Keep in mind there are {count} other files that depend on this module.",
    ],
    "shallow": [
        "Figure out the root cause before proposing a fix. Read the full error trace.",
        "Don't fix the symptom. Trace the bug to its origin.",
        "Investigate why this happens, not just what happens.",
    ],
}

# Project archetypes with relevant constraints
PROJECT_ARCHETYPES = {
    "rust_daemon": {
        "tools": ["Read", "Write", "Edit", "Bash"],
        "constraints": [
            "Keep in mind this is a Rust project. cargo build must pass with no warnings.",
            "Don't add new dependencies to Cargo.toml unless absolutely necessary.",
            "Make sure all error types implement std::error::Error.",
        ],
    },
    "nextjs_app": {
        "tools": ["Read", "Write", "Bash"],
        "constraints": [
            "Keep in mind this uses Next.js App Router, not Pages Router.",
            "Don't use client components unless you need interactivity. Default to server.",
            "Make sure Tailwind classes are used, not inline styles.",
        ],
    },
    "python_api": {
        "tools": ["Read", "Write", "Edit", "Bash"],
        "constraints": [
            "Keep in mind this uses FastAPI with async endpoints.",
            "Don't add global state. Use dependency injection.",
            "Make sure all endpoints have Pydantic models for request and response.",
        ],
    },
    "ios_app": {
        "tools": ["Read", "Write", "Edit", "Bash"],
        "constraints": [
            "Keep in mind this is SwiftUI, not UIKit.",
            "Don't force-unwrap optionals. Use guard let or if let.",
            "Make sure views are broken into small, reusable components.",
        ],
    },
    "generic": {
        "tools": ["Read", "Write", "Edit", "Bash", "Grep", "Glob"],
        "constraints": [
            "Execute directly. Don't explain unless asked.",
            "Make sure you test before saying it's done.",
        ],
    },
}


def detect_archetype(project_dir: str) -> str:
    d = Path(project_dir)
    if (d / "Cargo.toml").exists():
        return "rust_daemon"
    if (d / "next.config.mjs").exists() or (d / "next.config.js").exists() or (d / "next.config.ts").exists():
        return "nextjs_app"
    if (d / "pyproject.toml").exists() or (d / "requirements.txt").exists():
        return "python_api"
    if any(d.glob("*.xcodeproj")) or any(d.glob("*.xcworkspace")):
        return "ios_app"
    return "generic"


def build_rich_prompt(
    goal: str,
    project_dir: str,
    project_name: str = "",
    correction_pairs: list[dict] = None,
    archetype: str = None,
    extra_constraints: list[str] = None,
) -> RichPrompt:
    """Build a rich prompt with pre-baked corrections."""

    if not archetype:
        archetype = detect_archetype(project_dir)

    arch = PROJECT_ARCHETYPES.get(archetype, PROJECT_ARCHETYPES["generic"])

    # Start with the goal
    parts = [goal.strip()]

    # Add archetype constraints
    constraints = list(arch["constraints"])

    # Add failure-mode constraints from correction data
    failure_modes = set()
    if correction_pairs:
        # Pick relevant constraints from mined pairs
        modes = set(p.get("failure_mode", "") for p in correction_pairs[:20])
        for mode in modes:
            if mode in CONSTRAINT_TEMPLATES:
                tmpl = random.choice(CONSTRAINT_TEMPLATES[mode])
                # Fill in template vars with generic values
                tmpl = tmpl.replace("{anti_pattern}", "unnecessary abstractions")
                tmpl = tmpl.replace("{correct_approach}", "direct implementation")
                tmpl = tmpl.replace("{machine}", "Mac1")
                tmpl = tmpl.replace("{service}", "meshd")
                tmpl = tmpl.replace("{port}", "9451")
                tmpl = tmpl.replace("{pattern}", "the existing conventions")
                tmpl = tmpl.replace("{count}", "several")
                constraints.append(tmpl)
                failure_modes.add(mode)

    if extra_constraints:
        constraints.extend(extra_constraints)

    # Assemble: goal + constraints as a natural paragraph
    constraint_block = " ".join(constraints)
    prompt = f"{parts[0]}\n\n{constraint_block}"

    if not project_name:
        project_name = Path(project_dir).name

    return RichPrompt(
        project=project_name,
        project_dir=str(project_dir),
        goal=goal,
        prompt=prompt,
        preface_constraints=constraints,
        failure_modes_addressed=list(failure_modes),
        expected_tools=arch["tools"],
        tags=[archetype],
    )


def generate_batch(
    goals: list[dict],
    correction_pairs_path: str = None,
    output_path: str = None,
) -> list[RichPrompt]:
    """Generate a batch of rich prompts for session spawning.

    goals: list of {"goal": str, "project_dir": str, "project_name": str}
    """

    # Load correction pairs if available
    pairs = []
    if correction_pairs_path and os.path.exists(correction_pairs_path):
        with open(correction_pairs_path) as f:
            pairs = [json.loads(l) for l in f if l.strip()]
        print(f"Loaded {len(pairs)} correction pairs")

    prompts = []
    for g in goals:
        rp = build_rich_prompt(
            goal=g["goal"],
            project_dir=g.get("project_dir", "~/Desktop/scratch"),
            project_name=g.get("project_name", ""),
            correction_pairs=pairs,
            extra_constraints=g.get("constraints", []),
        )
        prompts.append(rp)

    if output_path:
        with open(output_path, "w") as f:
            for p in prompts:
                f.write(json.dumps(asdict(p)) + "\n")
        print(f"Wrote {len(prompts)} rich prompts to {output_path}")

    return prompts


# Pre-built goal library — real, relevant projects for the mesh
GOAL_LIBRARY = [
    {
        "goal": "Build a Rust CLI that reads NATS JetStream events from MESH_EVENTS and renders them in a colored terminal table with filtering by machine name and event type. Use clap for args, colored for output, and async-nats for the NATS client.",
        "project_dir": "~/Desktop/mesh-event-viewer",
        "project_name": "mesh-event-viewer",
        "constraints": ["Keep in mind NATS runs on localhost:4222.", "Make sure it handles NATS being offline gracefully."],
    },
    {
        "goal": "Build a Python FastAPI service that wraps the MLX cognitive twin at http://100.109.94.124:8100 as a REST API. Endpoints: POST /chat (conversation), POST /opinion (topic -> take), POST /drive (pane output -> next prompt), GET /health. Use httpx async, Pydantic models, proper error handling.",
        "project_dir": "~/Desktop/twin-api",
        "project_name": "twin-api",
        "constraints": ["Keep in mind MLX cold starts take 30s.", "Don't block on MLX calls, use async."],
    },
    {
        "goal": "Build a Next.js dashboard that polls meshd health endpoints across 5 Macs every 10 seconds and displays real-time status cards with colored dots per service. Dark theme, shadcn/ui components. Server-side polling to avoid CORS.",
        "project_dir": "~/Desktop/mesh-pulse",
        "project_name": "mesh-pulse",
        "constraints": ["Keep in mind machines are on Tailscale IPs.", "Make sure it works when some machines are offline."],
    },
    {
        "goal": "Build a Rust HTTP service using axum that classifies prompts into 10 inscription categories (stabilization, transition, oscillation, correction, exploration, convergence, expansion, regression, stagnation, completion) based on keyword analysis and prompt structure. Include /classify, /route, /stats endpoints.",
        "project_dir": "~/Desktop/inscription-router",
        "project_name": "inscription-router",
    },
    {
        "goal": "Build a Python script that connects to Supabase, pulls the 329K rows from memory_turns, filters for high-quality conversation pairs, computes 4 anticipation scalars, classifies inscriptions, and exports 10K curated training examples as JSONL. Include dedup, 90/10 split, domain logging.",
        "project_dir": "~/Desktop/karl-data-scaler",
        "project_name": "karl-data-scaler",
        "constraints": ["Keep in mind Supabase URL and key come from env vars.", "Don't pull all 329K at once, paginate with limit/offset."],
    },
    {
        "goal": "Build a SwiftUI iOS app called MeshMonitor that displays the health of 5 Mac mesh nodes. Each node shows name, IP, uptime, and service status. Pull data from meshd :9451/health on each machine. Add pull-to-refresh and a Settings screen for configuring IPs.",
        "project_dir": "~/Desktop/MeshMonitor",
        "project_name": "MeshMonitor",
    },
    {
        "goal": "Build a Rust binary that watches ~/Desktop/karl/karl/trajectories.jsonl for new lines, computes reward scores in real-time, and publishes scored trajectories to NATS subject karl.scored. Include a --backfill flag to score all existing trajectories.",
        "project_dir": "~/Desktop/karl-scorer",
        "project_name": "karl-scorer",
        "constraints": ["Keep in mind the reward engine has 5 signals.", "Don't recompute already-scored trajectories."],
    },
    {
        "goal": "Build a Python Prefect flow that runs every 6 hours: pulls new prompt logs from all Claude Code sessions, extracts correction pairs using signature phrase detection, generates rich prompts for the next training batch, and writes them to ~/Desktop/karl/factory-output/. Include a dashboard block for Prefect UI.",
        "project_dir": "~/Desktop/karl-factory-flow",
        "project_name": "karl-factory-flow",
    },
]


if __name__ == "__main__":
    pairs_path = os.path.expanduser("~/Desktop/karl/v6-correction-pairs.jsonl")
    out = os.path.expanduser("~/Desktop/karl/v6-rich-prompts.jsonl")
    prompts = generate_batch(GOAL_LIBRARY, pairs_path, out)
    for p in prompts:
        print(f"[{p.project}] {p.prompt[:100]}...")
        print(f"  Constraints: {len(p.preface_constraints)}, Modes: {p.failure_modes_addressed}")
        print()
