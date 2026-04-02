#!/usr/bin/env python3
"""Spawn real Claude Code sessions with rich prompts across the mesh.

Each session:
  1. Opens a tmux pane
  2. Launches Claude Code with --dangerously-skip-permissions
  3. Injects a rich, pre-corrected prompt
  4. Optionally wires V6 driver to continue the session
  5. All turns land in prompt logs automatically = training data

The sessions are the training data factory. No API calls needed.
"""

import json
import os
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path

RICH_PROMPTS_PATH = Path.home() / "Desktop" / "karl" / "v6-rich-prompts.jsonl"
SPAWN_LOG = Path.home() / "Desktop" / "karl" / "v6-spawn-log.jsonl"


@dataclass
class SpawnTarget:
    machine: str        # mac1, mac2, mac4, mac5
    tmux_session: str   # agent-codex, agent-claude2, etc.
    pane_id: str        # 1.1
    account: str        # account1, account2


# Available spawn targets
TARGETS = [
    SpawnTarget("mac1", "agent-codex", "1.1", "account1"),
    SpawnTarget("mac1", "agent-claude2", "1.1", "account1"),
    SpawnTarget("mac1", "agent-opencode", "1.1", "account1"),
    # Mac2-5 via SSH + tmux (uses second account)
    # SpawnTarget("mac2", "claude", "0.0", "account2"),
    # SpawnTarget("mac4", "claude", "0.0", "account2"),
    # SpawnTarget("mac5", "claude", "0.0", "account2"),
]


def kill_pane(target: SpawnTarget):
    """Kill whatever's running in the pane."""
    pane = f"{target.tmux_session}:{target.pane_id}"
    if target.machine in ("mac1", "localhost"):
        subprocess.run(["tmux", "send-keys", "-t", pane, "C-c", "C-c"], timeout=3, capture_output=True)
        time.sleep(0.5)
        subprocess.run(["tmux", "send-keys", "-t", pane, "/exit", "Enter"], timeout=3, capture_output=True)
        time.sleep(1)
        subprocess.run(["tmux", "send-keys", "-t", pane, "exit", "Enter"], timeout=3, capture_output=True)
        time.sleep(2)
        subprocess.run(["tmux", "respawn-pane", "-t", pane, "-k"], timeout=3, capture_output=True)
        time.sleep(2)
    else:
        ssh = ["ssh", "-o", "ConnectTimeout=5", target.machine]
        subprocess.run(ssh + ["tmux", "send-keys", "-t", pane, "C-c", "C-c"], timeout=10, capture_output=True)
        time.sleep(1)


def launch_claude(target: SpawnTarget, project_dir: str):
    """Launch Claude Code in the pane."""
    pane = f"{target.tmux_session}:{target.pane_id}"
    cmd = f"cd {project_dir} && claude --dangerously-skip-permissions"

    if target.machine in ("mac1", "localhost"):
        subprocess.run(["tmux", "send-keys", "-t", pane, cmd, "Enter"], timeout=5, capture_output=True)
    else:
        ssh = ["ssh", "-o", "ConnectTimeout=5", target.machine]
        subprocess.run(ssh + [f"tmux send-keys -t {pane} '{cmd}' Enter"], timeout=10, capture_output=True)

    time.sleep(12)  # Wait for Claude to start


def inject_prompt(target: SpawnTarget, prompt: str):
    """Inject a rich prompt into the Claude session."""
    pane = f"{target.tmux_session}:{target.pane_id}"

    if target.machine in ("mac1", "localhost"):
        subprocess.run(["tmux", "load-buffer", "-"],
                       input=prompt.encode(), timeout=5, capture_output=True)
        subprocess.run(["tmux", "paste-buffer", "-t", pane], timeout=5, capture_output=True)
        time.sleep(1)
        subprocess.run(["tmux", "send-keys", "-t", pane, "Enter"], timeout=5, capture_output=True)
    else:
        escaped = prompt.replace("'", "'\\''")
        ssh = ["ssh", "-o", "ConnectTimeout=5", target.machine]
        cmd = f"printf '%s' '{escaped}' | tmux load-buffer - && tmux paste-buffer -t {pane} && tmux send-keys -t {pane} Enter"
        subprocess.run(ssh + [cmd], timeout=15, capture_output=True)


def start_v6_driver(target: SpawnTarget, goal: str, project_dir: str, turns: int = 25):
    """Start V6 driver in background (legacy — uses 4B MLX twin)."""
    pane = f"{target.tmux_session}:{target.pane_id}"
    safe_goal = goal[:200].replace('"', '\\"').replace("'", "")
    driver_cmd = (
        f"sleep 90 && cd ~/Desktop/karl && python3 -u -m karl.v6.driver "
        f"{target.machine} {pane} "
        f'--goal "{safe_goal}" '
        f"--project-dir {project_dir} "
        f"--turns {turns} --interval 60 "
        f"> /tmp/v6-factory-{target.tmux_session}.log 2>&1"
    )
    subprocess.Popen(["bash", "-c", driver_cmd],
                     start_new_session=True,
                     stdout=subprocess.DEVNULL,
                     stderr=subprocess.DEVNULL)


def start_v7_driver(target: SpawnTarget, goal: str, project_dir: str, turns: int = 25):
    """Start V7 Mohamed-simulator driver in background.

    Uses RAG-grounded prompt generation (4,587 real prompts) instead of the 4B MLX twin.
    No MLX dependency — runs locally with zero network calls.
    """
    pane = f"{target.tmux_session}:{target.pane_id}"
    safe_goal = goal[:200].replace('"', '\\"').replace("'", "")
    driver_cmd = (
        f"sleep 90 && cd ~/Desktop/karl && python3 -u -m karl.v7.driver "
        f"{target.machine} {pane} "
        f'--goal "{safe_goal}" '
        f"--project-dir {project_dir} "
        f"--turns {turns} --interval 60 "
        f"> /tmp/v7-factory-{target.tmux_session}.log 2>&1"
    )
    subprocess.Popen(["bash", "-c", driver_cmd],
                     start_new_session=True,
                     stdout=subprocess.DEVNULL,
                     stderr=subprocess.DEVNULL)


def spawn_batch(
    prompts_path: str = None,
    targets: list[SpawnTarget] = None,
    with_driver: bool = True,
    turns: int = 25,
    dry_run: bool = False,
    driver_version: str = "v7",
):
    """Spawn a batch of sessions across targets."""
    if prompts_path is None:
        prompts_path = str(RICH_PROMPTS_PATH)
    if targets is None:
        targets = TARGETS

    # Load prompts
    with open(prompts_path) as f:
        prompts = [json.loads(l) for l in f if l.strip()]

    print(f"Loaded {len(prompts)} rich prompts, {len(targets)} targets")

    # Assign prompts to targets (round-robin if more prompts than targets)
    assignments = []
    for i, prompt in enumerate(prompts):
        target = targets[i % len(targets)]
        assignments.append((target, prompt))

    for i, (target, prompt_data) in enumerate(assignments):
        pane = f"{target.tmux_session}:{target.pane_id}"
        project = prompt_data["project"]
        project_dir = prompt_data["project_dir"]
        rich_prompt = prompt_data["prompt"]
        goal = prompt_data["goal"]

        print(f"\n[{i+1}/{len(assignments)}] {project} -> {target.machine}:{pane}")
        print(f"  Goal: {goal[:80]}...")
        print(f"  Constraints: {len(prompt_data.get('preface_constraints', []))}")

        if dry_run:
            print(f"  [DRY RUN] Would spawn")
            continue

        # 1. Kill existing session
        print(f"  Killing pane...", flush=True)
        kill_pane(target)

        # 2. Create project dir
        os.makedirs(os.path.expanduser(project_dir), exist_ok=True)

        # 3. Launch Claude
        print(f"  Launching Claude...", flush=True)
        launch_claude(target, project_dir)

        # 4. Inject rich prompt
        print(f"  Injecting prompt ({len(rich_prompt)} chars)...", flush=True)
        inject_prompt(target, rich_prompt)

        # 5. Optionally start driver
        if with_driver:
            if driver_version == "v7":
                print(f"  V7 simulator driver in 90s ({turns} turns)...", flush=True)
                start_v7_driver(target, goal, project_dir, turns)
            else:
                print(f"  V6 twin driver in 90s ({turns} turns)...", flush=True)
                start_v6_driver(target, goal, project_dir, turns)

        # 6. Log
        entry = {
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "target": f"{target.machine}:{pane}",
            "project": project,
            "goal": goal[:200],
            "prompt_length": len(rich_prompt),
            "with_driver": with_driver,
            "driver_version": driver_version if with_driver else None,
        }
        with open(SPAWN_LOG, "a") as f:
            f.write(json.dumps(entry) + "\n")

        # Wait between spawns to avoid overwhelming
        if i < len(assignments) - 1:
            print(f"  Waiting 5s before next spawn...")
            time.sleep(5)

    print(f"\nSpawned {len(assignments)} sessions.")
    if with_driver:
        prefix = "v7" if driver_version == "v7" else "v6"
        print(f"{prefix.upper()} drivers activate in 90s. Monitor: /tmp/{prefix}-factory-*.log")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="KARL Session Factory")
    parser.add_argument("--prompts", default=str(RICH_PROMPTS_PATH))
    parser.add_argument("--turns", type=int, default=25)
    parser.add_argument("--no-driver", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--driver", choices=["v6", "v7"], default="v7",
                        help="Driver version: v6 (4B MLX twin) or v7 (Mohamed simulator)")
    args = parser.parse_args()

    spawn_batch(args.prompts, TARGETS, not args.no_driver, args.turns,
                args.dry_run, args.driver)
