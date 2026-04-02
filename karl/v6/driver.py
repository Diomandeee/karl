#!/usr/bin/env python3
"""KARL V6 — Cognitive Twin Session Driver.

Drop-in replacement for twin_session_driver.py with persistent context,
anti-repeat, phase awareness, and task tracking.

Usage:
    python -m karl.v6.driver localhost agent-codex:1.1 \
        --goal "Build a health dashboard for the mesh" \
        --project-dir ~/Desktop/mesh-health-dashboard \
        --turns 30 --interval 45

    python -m karl.v6.driver localhost agent-codex:1.1 --dry-run --turns 5
"""

import argparse
import datetime
import json
import os
import subprocess
import sys
import time
import urllib.request

from .session_state import SessionState
from .project_context import detect_project
from .terminal_parser import parse_terminal
from .anti_repeat import is_duplicate, is_destructive, get_escape_prompt
from .context_stack import build_context_stack

MLX_URL = os.environ.get("MLX_URL", "http://100.109.94.124:8100/v1/chat/completions")
MLX_MODEL = "mlx-community/Qwen3-4B-Instruct-2507-4bit"
LOG_DIR = os.path.expanduser("~/Desktop/karl/v6-session-logs")
os.makedirs(LOG_DIR, exist_ok=True)


def read_pane(machine: str, pane_id: str) -> str | None:
    """Read tmux pane content."""
    if machine in ("localhost", "mac1"):
        try:
            r = subprocess.run(
                ["tmux", "capture-pane", "-t", pane_id, "-p"],
                capture_output=True, text=True, timeout=5,
            )
            if r.returncode == 0 and r.stdout.strip():
                return r.stdout
        except Exception:
            pass
    # Remote SSH fallback
    try:
        r = subprocess.run(
            ["ssh", "-o", "ConnectTimeout=5", machine,
             f"tmux capture-pane -t {pane_id} -p"],
            capture_output=True, text=True, timeout=15,
        )
        if r.returncode == 0 and r.stdout.strip():
            return r.stdout
    except Exception:
        pass
    return None


def inject_prompt(machine: str, pane_id: str, prompt: str) -> bool:
    """Inject a prompt into a tmux pane."""
    if machine in ("localhost", "mac1"):
        try:
            subprocess.run(["tmux", "load-buffer", "-"],
                           input=prompt.encode(), timeout=5)
            subprocess.run(["tmux", "paste-buffer", "-t", pane_id], timeout=5)
            subprocess.run(["tmux", "send-keys", "-t", pane_id, "Enter"], timeout=5)
            return True
        except Exception:
            return False
    try:
        escaped = prompt.replace("'", "'\\''")
        cmd = (f"printf '%s' '{escaped}' | tmux load-buffer - && "
               f"tmux paste-buffer -t {pane_id} && "
               f"tmux send-keys -t {pane_id} Enter")
        r = subprocess.run(
            ["ssh", "-o", "ConnectTimeout=5", machine, cmd],
            capture_output=True, text=True, timeout=15,
        )
        return r.returncode == 0
    except Exception:
        return False


def query_twin(messages: list[dict], max_tokens: int = 150, temp: float = 0.7) -> str:
    """Query the MLX cognitive twin."""
    payload = json.dumps({
        "model": MLX_MODEL,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temp,
    }).encode()
    req = urllib.request.Request(MLX_URL, data=payload,
                                headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=60) as resp:
        data = json.loads(resp.read())
        return data["choices"][0]["message"]["content"].replace("<|im_end|>", "").strip()


def decompose_goal(goal: str) -> list[str]:
    """Use the twin to decompose a goal into 5-8 steps."""
    messages = [
        {"role": "system", "content": "Decompose this goal into 5-8 concrete steps. "
         "Each step is one sentence, imperative mood. Output ONLY the numbered list."},
        {"role": "user", "content": goal},
    ]
    try:
        raw = query_twin(messages, max_tokens=200, temp=0.5)
        steps = []
        for line in raw.split("\n"):
            line = line.strip().lstrip("0123456789.-) ")
            if line and len(line) > 5:
                steps.append(line[:80])
        return steps[:8] if steps else [goal]
    except Exception:
        return [goal]


def drive_session(
    machine: str,
    pane_id: str,
    goal: str = "",
    project_dir: str = "",
    max_turns: int = 30,
    interval: int = 45,
    dry_run: bool = False,
    claude_cmd: str = "claude --dangerously-skip-permissions",
):
    session_id = SessionState.make_id(machine, pane_id)
    state = SessionState.load(session_id)

    if state is None:
        # New session
        state = SessionState(
            session_id=session_id,
            machine=machine,
            pane_id=pane_id,
            max_turns=max_turns,
        )

        # Detect project
        if project_dir:
            ctx = detect_project(project_dir)
            state.project_name = ctx.name
            state.project_dir = project_dir
        else:
            state.project_dir = os.getcwd()
            ctx = detect_project(state.project_dir)
            state.project_name = ctx.name

        # Set goal and decompose
        if goal:
            state.goal = goal
            print(f"Decomposing goal into steps...", flush=True)
            state.plan_steps = decompose_goal(goal)
            print(f"  {len(state.plan_steps)} steps: {state.plan_steps}", flush=True)

        state.save()

    # Log setup
    ts = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    log_path = os.path.join(LOG_DIR, f"{ts}_{session_id}.jsonl")

    print(f"KARL V6 Session Driver", flush=True)
    print(f"  Session: {session_id}", flush=True)
    print(f"  Project: {state.project_name} ({state.project_dir})", flush=True)
    print(f"  Goal: {state.goal}", flush=True)
    print(f"  Steps: {len(state.plan_steps)}", flush=True)
    print(f"  Turns: {max_turns}, Interval: {interval}s", flush=True)
    print(f"  Dry run: {dry_run}", flush=True)
    print(f"  Log: {log_path}", flush=True)
    print(flush=True)

    for turn in range(state.turn, max_turns):
        print(f"--- Turn {turn + 1}/{max_turns} [{state.phase}] ---", flush=True)

        # 1. Read pane
        raw = read_pane(machine, pane_id)
        if not raw:
            print(f"  Could not read pane. Waiting...", flush=True)
            time.sleep(interval)
            continue

        # 2. Parse terminal
        term = parse_terminal(raw)

        # 2a. Session dead — restart Claude
        if not term.is_alive:
            print(f"  SESSION DEAD. Restarting Claude...", flush=True)
            if not dry_run:
                inject_prompt(machine, pane_id, claude_cmd)
            time.sleep(15)
            continue

        # 2b. Claude still working — wait
        if term.is_working:
            print(f"  Claude working. Waiting...", flush=True)
            time.sleep(min(interval, 20))
            continue

        # 3. Update state
        state.update_pane_hash(term.content_hash)
        turn_pct = (turn + 1) / max_turns
        state.update_phase(turn_pct)

        # 4. Build context stack
        messages = build_context_stack(state, term.content)

        # 5. Query twin
        t0 = time.time()
        try:
            # Use higher temp when stuck
            temp = 0.95 if state.phase == "STUCK" else 0.7
            raw_prompt = query_twin(messages, temp=temp)
        except Exception as e:
            print(f"  MLX error: {e}. Using plan fallback.", flush=True)
            # Plan-only fallback
            remaining = [s for s in state.plan_steps if s not in state.goals_completed]
            raw_prompt = remaining[0] if remaining else "status"
        elapsed = round(time.time() - t0, 2)

        # 6. Validate
        final_prompt = raw_prompt
        reason = "twin"

        if is_destructive(raw_prompt):
            print(f"  BLOCKED destructive: {raw_prompt[:60]}", flush=True)
            final_prompt = get_escape_prompt(state.phase)
            reason = "destructive_block"
        elif is_duplicate(raw_prompt, state.last_prompts):
            print(f"  BLOCKED duplicate: {raw_prompt[:60]}", flush=True)
            final_prompt = get_escape_prompt(state.phase)
            reason = "dedup"
        elif raw_prompt.strip().lower() == "status" and state.consecutive_status >= 1:
            print(f"  BLOCKED status spam", flush=True)
            remaining = [s for s in state.plan_steps if s not in state.goals_completed]
            if remaining:
                final_prompt = remaining[0]
                reason = "goal_redirect"
            else:
                final_prompt = get_escape_prompt(state.phase)
                reason = "escape"

        print(f"  [{reason}] ({elapsed}s): {final_prompt[:100]}", flush=True)

        # 7. Inject
        if dry_run:
            print(f"  [DRY RUN] Would inject", flush=True)
        else:
            ok = inject_prompt(machine, pane_id, final_prompt)
            print(f"  Injected: {ok}", flush=True)

        # 8. Update state
        state.record_prompt(final_prompt)

        # Check for goal completion signals in terminal
        for step in state.plan_steps:
            if step not in state.goals_completed:
                keywords = step.lower().split()[:3]
                if any(kw in term.content.lower() for kw in keywords if len(kw) > 3):
                    if any(sig in term.content.lower() for sig in
                           ["done", "success", "passed", "created", "written", "committed"]):
                        state.goals_completed.append(step)
                        print(f"  GOAL COMPLETE: {step[:60]}", flush=True)

        # Digest entry
        outcome = "error" if term.error_detected else "progress"
        state.record_digest(
            summary=final_prompt[:100],
            outcome=outcome,
            tools=[term.last_tool] if term.last_tool else [],
        )

        state.save()

        # Log
        entry = {
            "timestamp": datetime.datetime.now().isoformat(),
            "turn": turn + 1, "phase": state.phase,
            "twin_prompt": raw_prompt, "final_prompt": final_prompt,
            "reason": reason, "elapsed": elapsed,
            "pane_hash": term.content_hash,
            "is_stuck": state.is_stuck(),
            "goals_done": len(state.goals_completed),
        }
        with open(log_path, "a") as f:
            f.write(json.dumps(entry) + "\n")

        time.sleep(interval)

    print(f"\nDone. {state.turn} turns. Goals: {len(state.goals_completed)}/{len(state.plan_steps)}")
    print(f"Log: {log_path}")


def main():
    parser = argparse.ArgumentParser(description="KARL V6 Cognitive Twin Session Driver")
    parser.add_argument("machine", help="Target machine (mac1, mac2, localhost)")
    parser.add_argument("pane", help="Tmux pane ID")
    parser.add_argument("--goal", default="", help="Session goal")
    parser.add_argument("--project-dir", default="", help="Project directory")
    parser.add_argument("--turns", type=int, default=30)
    parser.add_argument("--interval", type=int, default=45)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--claude-cmd", default="claude --dangerously-skip-permissions")
    args = parser.parse_args()

    drive_session(
        args.machine, args.pane, args.goal, args.project_dir,
        args.turns, args.interval, args.dry_run, args.claude_cmd,
    )


if __name__ == "__main__":
    main()
