"""
agentic_synth.py — Synthetic training data generation for KARL.

Generates high-quality synthetic trajectories by:
  1. Extracting patterns from existing high-reward trajectories
  2. Recombining tool sequences with domain-appropriate variations
  3. Creating counterfactual examples (what should have happened instead)
  4. Augmenting with noise for robustness

Output: synthetic_qa.jsonl (consumed by sft_exporter.py)
"""

import json
import random
import hashlib
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional

KARL_DIR = Path(__file__).parent
STORE_PATH = KARL_DIR / "trajectories.jsonl"
OUTPUT_PATH = KARL_DIR / "synthetic_qa.jsonl"

SYSTEM_PROMPT = (
    "You are an expert software engineering assistant. Given a task, "
    "plan the optimal sequence of tool uses to accomplish it efficiently."
)

# Template patterns extracted from high-reward trajectories
CANONICAL_PATTERNS = {
    "read_edit_verify": {
        "sequence": ["Read", "Read", "Edit", "Bash"],
        "description": "Read context → Edit target → Verify change",
        "domains": ["ios", "infra", "systems", "web"],
    },
    "search_read_write": {
        "sequence": ["Grep", "Read", "Write", "Bash"],
        "description": "Search for pattern → Read match → Write new file → Test",
        "domains": ["infra", "systems", "data"],
    },
    "explore_plan_implement": {
        "sequence": ["Glob", "Read", "Read", "Edit", "Edit", "Bash"],
        "description": "Explore structure → Read files → Implement changes → Test",
        "domains": ["ios", "web", "systems"],
    },
    "debug_diagnose_fix": {
        "sequence": ["Bash", "Grep", "Read", "Edit", "Bash"],
        "description": "Run failing command → Search for cause → Read context → Fix → Verify",
        "domains": ["infra", "ios", "web"],
    },
    "create_module": {
        "sequence": ["Glob", "Read", "Write", "Write", "Bash"],
        "description": "Check existing → Read patterns → Write implementation → Write test → Run test",
        "domains": ["systems", "ml", "infra"],
    },
    "deploy_verify": {
        "sequence": ["Read", "Bash", "Bash", "Bash"],
        "description": "Read config → Build → Deploy → Verify health",
        "domains": ["infra", "ios"],
    },
}

# Domain-specific task prompts
DOMAIN_PROMPTS = {
    "ios": [
        "Fix the build error in {app} — archive is failing with missing module",
        "Add a new view for {feature} in {app} using SwiftUI",
        "Update the Info.plist for {app} with the new privacy descriptions",
        "Wire {service} into the TCA reducer for {app}",
    ],
    "infra": [
        "Deploy the updated {service} to cloud-vm via Docker",
        "Fix the systemd service for {daemon} — it's crash-looping",
        "Set up the SSH tunnel for {service} from Mac1 to cloud-vm",
        "Update the prometheus.yml to scrape {service} on port {port}",
    ],
    "systems": [
        "Add a new invariant check to Evolution World for {concept}",
        "Create the {module} module in the evolution engine",
        "Wire the {signal} signal into the orchestrator loop",
        "Implement {pattern} pattern in the topology system",
    ],
    "web": [
        "Add the {page} page to Nexus Portal",
        "Fix the API endpoint for {feature} in the dashboard",
        "Update the Tailwind styles for the {component} component",
    ],
    "data": [
        "Run the Supabase migration for {table}",
        "Fix the RLS policy on {table} — anon access is broken",
        "Add a new column {column} to the {table} table",
    ],
    "ml": [
        "Train a LoRA adapter on the latest trajectory data",
        "Fix the data pipeline for {dataset} — deduplication is off",
        "Evaluate model performance on the {domain} holdout set",
    ],
    "automation": [
        "Deploy the {flow} Prefect flow to cloud-vm",
        "Fix the cron schedule for {flow} — it's not triggering",
        "Add error handling to {flow} for transient failures",
    ],
}

# Placeholder values for template substitution
PLACEHOLDERS = {
    "app": ["Spore", "OpenClawHub", "SecuriClaw", "CreativeDirector", "SpeakFlow"],
    "feature": ["settings", "onboarding", "analytics", "notifications", "profile"],
    "service": ["rag-plus-plus", "graph-kernel", "agent-intelligence", "creator-shield"],
    "daemon": ["numu-daemon", "finetune-daemon", "pane-orchestrator", "feed-hub"],
    "module": ["topology", "ecology", "forcing", "feedback", "immune"],
    "concept": ["population entropy", "convergence detection", "monoculture prevention"],
    "signal": ["drift", "staleness", "interrupt", "correction"],
    "pattern": ["archipelago", "hub-spoke", "gradient", "fractal"],
    "page": ["compute", "guild", "flows", "sweep", "alerts"],
    "component": ["StatCard", "StatusBadge", "RefreshTimer"],
    "table": ["trajectories", "pane_tokens", "ecology_snapshots", "guild_members"],
    "column": ["last_active_at", "drift_score", "domain_affinity"],
    "dataset": ["karl-sft", "trajectory-filtered", "cortex-corrections"],
    "flow": ["morning_brief", "garden_tender", "vault_sync", "creator_shield_monitor"],
    "domain": ["ios", "infra", "systems", "ml"],
    "port": ["8001", "8010", "9090", "3001", "8093"],
}


def _fill_template(template: str) -> str:
    """Fill a prompt template with random placeholder values."""
    result = template
    for key, values in PLACEHOLDERS.items():
        placeholder = "{" + key + "}"
        if placeholder in result:
            result = result.replace(placeholder, random.choice(values), 1)
    return result


def _tool_step(tool: str, domain: str, step_num: int) -> str:
    """Generate a realistic tool step description."""
    if tool == "Read":
        files = {
            "ios": ["AppDelegate.swift", "ContentView.swift", "project.yml"],
            "infra": ["docker-compose.yml", "prometheus.yml", "daemon.py"],
            "systems": ["engine.py", "invariants.py", "state.py"],
            "web": ["page.tsx", "api.ts", "layout.tsx"],
        }
        f = random.choice(files.get(domain, ["file.py"]))
        return f"{step_num}. [ok] Read ../{f}"
    elif tool == "Edit":
        return f"{step_num}. [ok] Edit ../{random.choice(['main.py', 'handler.swift', 'index.ts'])}"
    elif tool == "Write":
        return f"{step_num}. [ok] Write ../{random.choice(['new_module.py', 'NewView.swift', 'component.tsx'])}"
    elif tool == "Bash":
        cmds = {
            "ios": ["xcodebuild archive", "xcrun altool", "swift build"],
            "infra": ["docker compose up -d", "systemctl restart", "ssh cloud-vm"],
            "systems": ["python3 -m pytest", "python3 engine.py", "python3 -c"],
            "web": ["npm run build", "npm test", "npx next build"],
        }
        cmd = random.choice(cmds.get(domain, ["python3 test.py"]))
        return f"{step_num}. [ok] Bash: {cmd}"
    elif tool == "Grep":
        return f"{step_num}. [ok] Grep '{random.choice(['TODO', 'error', 'import', 'def '])}'"
    elif tool == "Glob":
        return f"{step_num}. [ok] Glob '**/*.{random.choice(['py', 'swift', 'ts'])}'"
    return f"{step_num}. [ok] {tool}"


def generate_from_patterns(count: int = 30) -> List[Dict]:
    """Generate synthetic examples from canonical patterns."""
    examples = []
    for _ in range(count):
        pattern_name = random.choice(list(CANONICAL_PATTERNS.keys()))
        pattern = CANONICAL_PATTERNS[pattern_name]
        domain = random.choice(pattern["domains"])

        # Generate prompt
        domain_prompts = DOMAIN_PROMPTS.get(domain, DOMAIN_PROMPTS["systems"])
        prompt = _fill_template(random.choice(domain_prompts))

        # Generate tool steps
        steps = []
        for i, tool in enumerate(pattern["sequence"], 1):
            steps.append(_tool_step(tool, domain, i))

        plan = "\n".join(steps)
        plan += f"\n\nResult: {len(pattern['sequence'])}/{len(pattern['sequence'])} tools succeeded, reward=0.65"

        examples.append({
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": plan},
            ],
            "synthetic": True,
            "pattern": pattern_name,
            "domain": domain,
        })

    return examples


def generate_from_trajectories(
    min_reward: float = 0.55,
    count: int = 20,
) -> List[Dict]:
    """Generate augmented examples from high-reward trajectories."""
    if not STORE_PATH.exists():
        return []

    # Load high-reward trajectories
    good_records = []
    with open(STORE_PATH, "r") as f:
        for line in f:
            try:
                record = json.loads(line)
                reward = record.get("outcome", {}).get("reward_score")
                if reward and reward >= min_reward:
                    good_records.append(record)
            except json.JSONDecodeError:
                continue

    if not good_records:
        return []

    examples = []
    for _ in range(min(count, len(good_records))):
        record = random.choice(good_records)
        events = record.get("trajectory", {}).get("events", [])
        if len(events) < 3:
            continue

        prompt = record.get("context", {}).get("prompt_text", "")
        if not prompt or len(prompt) < 10:
            continue

        # Augment: slightly rephrase the prompt
        augmented_prompt = prompt
        swaps = [
            ("Fix", "Resolve"), ("Add", "Implement"), ("Update", "Modify"),
            ("Create", "Build"), ("Deploy", "Push"), ("the", "this"),
        ]
        for old, new in swaps:
            if old in augmented_prompt and random.random() > 0.7:
                augmented_prompt = augmented_prompt.replace(old, new, 1)
                break

        # Build plan from actual events
        steps = []
        for i, e in enumerate(events[:15], 1):
            tool = e.get("tool_name", "?")
            params = e.get("key_params", {})
            success = e.get("success")
            status = "ok" if success else ("fail" if success is False else "?")
            if tool == "Read" and "file_path" in params:
                fp = params["file_path"].split("/")[-1]
                steps.append(f"{i}. [{status}] Read {fp}")
            elif tool == "Bash" and "command" in params:
                steps.append(f"{i}. [{status}] Bash: {params['command'][:60]}")
            else:
                steps.append(f"{i}. [{status}] {tool}")

        plan = "\n".join(steps)
        total = record.get("trajectory", {}).get("total_tools", 0)
        successes = record.get("trajectory", {}).get("successes", 0)
        plan += f"\n\nResult: {successes}/{total} tools succeeded"

        examples.append({
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": augmented_prompt[:2000]},
                {"role": "assistant", "content": plan},
            ],
            "synthetic": True,
            "augmented_from": record.get("id", "")[:16],
            "domain": record.get("skill", {}).get("domain", "_global"),
        })

    return examples


def generate_counterfactuals(count: int = 10) -> List[Dict]:
    """Generate counterfactual 'what not to do' examples from low-reward trajectories."""
    if not STORE_PATH.exists():
        return []

    low_records = []
    with open(STORE_PATH, "r") as f:
        for line in f:
            try:
                record = json.loads(line)
                reward = record.get("outcome", {}).get("reward_score")
                if reward and reward < 0.4:
                    low_records.append(record)
            except json.JSONDecodeError:
                continue

    if not low_records:
        return []

    examples = []
    for record in low_records[:count]:
        prompt = record.get("context", {}).get("prompt_text", "")
        if not prompt or len(prompt) < 10:
            continue

        events = record.get("trajectory", {}).get("events", [])
        # Identify the failure pattern
        failure_steps = [e for e in events if e.get("success") is False]
        if not failure_steps:
            continue

        # Build a corrected plan
        domain = record.get("skill", {}).get("domain", "systems")
        pattern = random.choice(list(CANONICAL_PATTERNS.values()))
        steps = []
        for i, tool in enumerate(pattern["sequence"], 1):
            steps.append(_tool_step(tool, domain, i))

        plan = "Corrected approach:\n" + "\n".join(steps)
        plan += f"\n\nResult: {len(pattern['sequence'])}/{len(pattern['sequence'])} tools succeeded"

        examples.append({
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt[:2000]},
                {"role": "assistant", "content": plan},
            ],
            "synthetic": True,
            "counterfactual_from": record.get("id", "")[:16],
            "domain": domain,
        })

    return examples


def export_synthetic(
    pattern_count: int = 30,
    augment_count: int = 20,
    counterfactual_count: int = 10,
) -> Dict[str, Any]:
    """Generate and export all synthetic training data."""
    pattern_examples = generate_from_patterns(pattern_count)
    augmented_examples = generate_from_trajectories(count=augment_count)
    counterfactual_examples = generate_counterfactuals(counterfactual_count)

    all_examples = pattern_examples + augmented_examples + counterfactual_examples

    # Deduplicate
    seen = set()
    unique = []
    for ex in all_examples:
        msgs = ex.get("messages", [])
        if len(msgs) >= 3:
            h = hashlib.sha256(
                (msgs[1]["content"] + msgs[2]["content"]).encode()
            ).hexdigest()[:16]
            if h not in seen:
                seen.add(h)
                unique.append(ex)

    # Write output
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w") as f:
        for ex in unique:
            f.write(json.dumps(ex, default=str) + "\n")

    # Domain distribution
    domain_dist = defaultdict(int)
    for ex in unique:
        domain_dist[ex.get("domain", "_global")] += 1

    return {
        "pattern_examples": len(pattern_examples),
        "augmented_examples": len(augmented_examples),
        "counterfactual_examples": len(counterfactual_examples),
        "total_unique": len(unique),
        "output": str(OUTPUT_PATH),
        "domain_distribution": dict(domain_dist),
    }


if __name__ == "__main__":
    import sys
    if "--export" in sys.argv:
        random.seed(42)
        stats = export_synthetic()
        print(f"\nSynthetic Data Export")
        print(f"{'=' * 40}")
        for k, v in stats.items():
            print(f"  {k}: {v}")
    elif "--stats" in sys.argv:
        random.seed(42)
        p = generate_from_patterns(5)
        a = generate_from_trajectories(count=5)
        c = generate_counterfactuals(5)
        print(f"Pattern: {len(p)}, Augmented: {len(a)}, Counterfactual: {len(c)}")
    else:
        print("Usage: agentic_synth.py --export | --stats")
