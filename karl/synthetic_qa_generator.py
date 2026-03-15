#!/usr/bin/env python3
"""
synthetic_qa_generator.py — Generate synthetic SFT training data for underrepresented skills.

Creates ChatML-format Q&A pairs using skill descriptions and tool-use templates
for skills that have too few real trajectory examples.

Usage:
    python3 synthetic_qa_generator.py               # Dry-run
    python3 synthetic_qa_generator.py --apply        # Write to synthetic_qa.jsonl
    python3 synthetic_qa_generator.py --min-count 5  # Only for skills with <5 examples
"""

import json
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, List, Any

KARL_DIR = Path(__file__).parent
TRAJECTORY_PATH = KARL_DIR / "trajectories.jsonl"
OUTPUT_PATH = KARL_DIR / "synthetic_qa.jsonl"

sys.path.insert(0, str(KARL_DIR))

SYSTEM_PROMPT = (
    "You are an expert software engineering assistant. Given a task, "
    "plan the optimal sequence of tool uses to accomplish it efficiently."
)

# Templates per skill domain
DOMAIN_TEMPLATES = {
    "ios": [
        "Build and test the {project} iOS app",
        "Fix the build error in {project} — Xcode says missing module",
        "Add a new SwiftUI view to {project} for {feature}",
        "Deploy {project} to TestFlight for beta testing",
        "Update the {project} Info.plist and project settings",
    ],
    "web": [
        "Deploy the {project} web application to Vercel",
        "Fix the CSS layout issue in {project} dashboard",
        "Add a new API endpoint to {project} backend",
        "Update dependencies in {project} and fix breaking changes",
    ],
    "infra": [
        "Restart the {project} service on cloud-vm",
        "Check the {project} Docker container logs for errors",
        "Update the {project} systemd service configuration",
        "Debug the {project} connection failure to Supabase",
        "Deploy updated {project} configuration to all mesh nodes",
    ],
    "creative": [
        "Generate a creative {feature} using the {project} framework",
        "Evolve the {feature} idea through {project} pipeline",
        "Run a {project} creative session for content generation",
    ],
    "ml": [
        "Train the {project} model with updated data",
        "Debug the {project} inference pipeline error",
        "Update the {project} scoring configuration",
    ],
    "systems": [
        "Debug the {project} compilation failure",
        "Update the {project} Rust dependencies and fix breaking changes",
        "Add a new module to {project} for {feature}",
    ],
    "data": [
        "Run the {project} data pipeline and verify output",
        "Fix the {project} indexing failure",
    ],
    "automation": [
        "Deploy new {project} flows to Prefect",
        "Debug the failing {project} scheduled task",
        "Add a new {project} automation flow for {feature}",
    ],
    "knowledge": [
        "Sync the {project} vault with latest content",
        "Fix the {project} API writer connection error",
    ],
    "general": [
        "Set up the {project} development environment",
        "Review and clean up the {project} codebase",
        "Run tests for {project} and fix failures",
    ],
}

# Skill-to-domain mapping + feature hints
SKILL_FEATURES = {
    "creator-shield": ("ml", ["toxicity detection", "content moderation", "NLP scoring"]),
    "skill-forge": ("infra", ["skill detection", "trigger configuration", "registry update"]),
    "tauri-desktop": ("systems", ["Tauri app", "cross-platform UI", "Rust backend"]),
    "authoring-session": ("general", ["new feature", "code writing", "file creation"]),
    "monitoring-ops": ("infra", ["Grafana dashboard", "Prometheus alert", "Docker monitoring"]),
    "evolution-world": ("systems", ["framework template", "graduation check", "daemon status"]),
    "test": ("general", ["test suite", "coverage report", "regression check"]),
    "self-healing-code": ("systems", ["auto-repair pipeline", "error detection", "fix validation"]),
    "vault-ops": ("knowledge", ["daily notes", "vault structure", "search indexing"]),
    "supabase-ops": ("infra", ["database migration", "RLS policy", "query optimization"]),
    "nko-brain-scanner": ("ml", ["OCR model", "handwriting recognition", "training pipeline"]),
    "serenity-soother": ("ios", ["ASMR audio", "ambient scenes", "therapeutic content"]),
    "cortex-ops": ("infra", ["behavioral rules", "hook configuration", "correction patterns"]),
    "karl-trajectory": ("infra", ["trajectory capture", "reward computation", "embedding cache"]),
}

# Tool sequence templates by domain
TOOL_SEQUENCES = {
    "ios": [
        ("Read", "project.yml"), ("Bash", "xcodegen generate"),
        ("Edit", "ContentView.swift"), ("Bash", "xcodebuild -scheme {project} build"),
    ],
    "web": [
        ("Read", "package.json"), ("Bash", "npm install"),
        ("Edit", "src/index.ts"), ("Bash", "npm run build"),
    ],
    "infra": [
        ("Read", "docker-compose.yml"), ("Bash", "systemctl --user status {service}"),
        ("Edit", "config.json"), ("Bash", "systemctl --user restart {service}"),
    ],
    "creative": [
        ("Read", "framework.yaml"), ("Bash", "python3 evolve.py"),
        ("Write", "output.md"),
    ],
    "ml": [
        ("Read", "config.py"), ("Bash", "python3 train.py"),
        ("Read", "metrics.json"), ("Edit", "config.py"),
    ],
    "systems": [
        ("Read", "Cargo.toml"), ("Bash", "cargo build"),
        ("Edit", "src/main.rs"), ("Bash", "cargo test"),
    ],
    "data": [
        ("Read", "pipeline.py"), ("Bash", "python3 pipeline.py"),
        ("Bash", "wc -l output.jsonl"),
    ],
    "automation": [
        ("Read", "flow.py"), ("Bash", "prefect deploy flow.py"),
        ("Bash", "prefect flow-run list"),
    ],
    "knowledge": [
        ("Read", "api.py"), ("Bash", "python3 api.py --sync"),
        ("Write", "daily.md"),
    ],
    "general": [
        ("Read", "README.md"), ("Bash", "git status"),
        ("Edit", "main.py"), ("Bash", "python3 -m pytest"),
    ],
}


def _build_tool_plan(domain: str, project: str, success_rate: float = 0.9) -> str:
    """Build a tool-use plan string from templates."""
    import random
    tools = TOOL_SEQUENCES.get(domain, TOOL_SEQUENCES["general"])
    parts = []
    total = len(tools)
    successes = 0
    for i, (tool, target) in enumerate(tools, 1):
        target = target.replace("{project}", project).replace("{service}", project)
        success = random.random() < success_rate
        status = "ok" if success else "fail"
        if success:
            successes += 1
        if tool == "Bash":
            parts.append(f"{i}. [{status}] Bash: {target}")
        else:
            parts.append(f"{i}. [{status}] {tool} ../{target}")
    reward = 0.55 + (successes / total) * 0.35
    parts.append(f"\nResult: {successes}/{total} tools succeeded, reward={reward:.2f}")
    return "\n".join(parts)


def generate_synthetic_qa(min_count: int = 3, dry_run: bool = True) -> Dict[str, Any]:
    """Generate synthetic QA for underrepresented skills."""
    import random
    random.seed(42)

    # Count existing trajectories per skill
    skill_counts = Counter()
    with open(TRAJECTORY_PATH) as f:
        for line in f:
            try:
                r = json.loads(line)
                skill = r.get("skill", {}).get("name", "unknown")
                skill_counts[skill] += 1
            except json.JSONDecodeError:
                continue

    # Find underrepresented skills
    from embedding_cache import SKILL_DESCRIPTIONS
    underrep = {}
    for skill, count in skill_counts.items():
        if count < min_count and skill in SKILL_DESCRIPTIONS:
            underrep[skill] = count

    if not underrep:
        return {"status": "no_underrepresented", "threshold": min_count}

    # Generate synthetic examples
    examples = []
    for skill, count in sorted(underrep.items()):
        domain, features = SKILL_FEATURES.get(skill, ("general", ["feature update"]))
        templates = DOMAIN_TEMPLATES.get(domain, DOMAIN_TEMPLATES["general"])
        target_count = min_count - count  # Generate enough to reach min_count

        for i in range(target_count):
            template = templates[i % len(templates)]
            feature = features[i % len(features)]
            prompt = template.replace("{project}", skill).replace("{feature}", feature)
            plan = _build_tool_plan(domain, skill)

            example = {
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                    {"role": "assistant", "content": plan},
                ],
            }
            examples.append(example)

    result = {
        "status": "ok",
        "underrepresented_skills": len(underrep),
        "skill_gaps": {k: v for k, v in sorted(underrep.items(), key=lambda x: x[1])},
        "generated": len(examples),
        "threshold": min_count,
        "dry_run": dry_run,
    }

    if dry_run:
        return result

    # Write to file (overwrite — sft_exporter deduplicates anyway)
    with open(OUTPUT_PATH, "w") as f:
        for ex in examples:
            f.write(json.dumps(ex) + "\n")

    result["output"] = str(OUTPUT_PATH)
    return result


if __name__ == "__main__":
    min_count = 3
    for i, arg in enumerate(sys.argv):
        if arg == "--min-count" and i + 1 < len(sys.argv):
            min_count = int(sys.argv[i + 1])

    dry = "--apply" not in sys.argv
    result = generate_synthetic_qa(min_count=min_count, dry_run=dry)
    print(json.dumps(result, indent=2))
