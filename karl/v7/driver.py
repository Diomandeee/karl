#!/usr/bin/env python3
"""KARL V7 — Mohamed Simulator Session Driver.

Replaces the 4B MLX twin with a RAG-grounded Mohamed simulator that
generates prompts using real prompt DNA (4,587 prompts) instead of
a fine-tuned model.

All sessions are training data: every turn that gets injected lands
in Claude's prompt logs automatically for V8 training.

Usage:
    python -m karl.v7.driver localhost agent-codex:1.1 \
        --goal "Build a Rust HTTP health checker for the mesh" \
        --project-dir ~/Desktop/mesh-health \
        --turns 30 --interval 45

    python -m karl.v7.driver localhost agent-codex:1.1 --dry-run --turns 5
"""

import argparse
import datetime
import json
import os
import subprocess
import sys
import time

from ..v6.session_state import SessionState
from ..v6.project_context import detect_project
from ..v6.terminal_parser import parse_terminal
from ..v6.anti_repeat import is_duplicate, is_destructive, get_escape_prompt

from .prompt_corpus import get_corpus
from .session_context_reader import parse_session_context
from .mohamed_simulator import generate as simulate
from .style_validator import validate as style_validate

LOG_DIR = os.path.expanduser("~/Desktop/karl/v7-session-logs")
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
    """Inject a prompt into a tmux pane via load-buffer+paste-buffer."""
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


def decompose_goal_local(goal: str) -> list[str]:
    """Rule-based goal decomposer — no MLX needed.

    Extracts action phrases from the goal and generates 4-6 concrete steps
    by detecting goal type (rust/nextjs/python/ios/generic).
    """
    goal_l = goal.lower()

    if "rust" in goal_l or "cargo" in goal_l:
        steps = [
            f"cargo new {_slug(goal)} --name {_slug(goal)} and set up Cargo.toml",
            "implement the core data structures and types",
            "implement the main logic and error handling",
            "add CLI argument parsing with clap",
            "test with cargo test and fix any errors",
        ]
    elif "next" in goal_l or "react" in goal_l or "dashboard" in goal_l:
        steps = [
            "scaffold the Next.js app with npx create-next-app",
            "implement the main page component and layout",
            "add data fetching with server-side rendering",
            "style with Tailwind and verify responsiveness",
            "run npm run build and fix any TypeScript errors",
        ]
    elif "python" in goal_l or "fastapi" in goal_l or "flask" in goal_l:
        steps = [
            "set up the project with pyproject.toml and install deps",
            "implement the core models and schemas",
            "implement the API endpoints with error handling",
            "add tests with pytest",
            "run pytest and fix any failures",
        ]
    elif "swift" in goal_l or "ios" in goal_l or "swiftui" in goal_l:
        steps = [
            "create the Xcode project and set up the main view",
            "implement the data models and services",
            "implement the main UI components",
            "wire up the data to the UI with @State and @ObservedObject",
            "build and fix any Swift compilation errors",
        ]
    elif "prefect" in goal_l or "flow" in goal_l or "pipeline" in goal_l:
        steps = [
            "set up the Prefect flow with @flow decorator",
            "implement the core tasks with @task decorator",
            "add error handling and retries",
            "add logging and notifications",
            "run the flow locally and verify output",
        ]
    else:
        # Generic
        words = goal.split()
        verb = words[0] if words else "build"
        obj = " ".join(words[1:4]) if len(words) > 1 else "the project"
        steps = [
            f"set up the project structure for {obj}",
            f"implement the core {obj}",
            f"add error handling and edge cases",
            f"write tests and run them",
            f"clean up and commit",
        ]

    return [s[:80] for s in steps[:6]]


def _slug(text: str) -> str:
    """Convert goal text to a filesystem-safe slug."""
    import re
    words = re.findall(r"[a-zA-Z0-9]+", text.lower())[:3]
    return "-".join(words) if words else "project"


def drive_session(
    machine: str,
    pane_id: str,
    goal: str = "",
    project_dir: str = "",
    max_turns: int = 30,
    interval: int = 45,
    dry_run: bool = False,
    claude_cmd: str = "claude --dangerously-skip-permissions",
    fresh: bool = True,
):
    """Main driver loop — injects Mohamed-style prompts into a Claude Code session."""
    session_id = SessionState.make_id(machine, pane_id)

    if fresh:
        # Always start with clean state for factory runs
        state = None
    else:
        state = SessionState.load(session_id)

    if state is None:
        state = SessionState(
            session_id=session_id,
            machine=machine,
            pane_id=pane_id,
            max_turns=max_turns,
        )
        if project_dir:
            ctx = detect_project(project_dir)
            state.project_name = ctx.name
            state.project_dir = project_dir
        else:
            state.project_dir = os.getcwd()
            ctx = detect_project(state.project_dir)
            state.project_name = ctx.name

        if goal:
            state.goal = goal
            state.plan_steps = decompose_goal_local(goal)

        state.save()

    # Pre-warm corpus (loads once, cached globally)
    try:
        corpus = get_corpus()
        corpus_size = len(corpus)
    except Exception:
        corpus_size = 0

    ts = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    log_path = os.path.join(LOG_DIR, f"{ts}_{session_id}.jsonl")

    print(f"KARL V7 Session Driver", flush=True)
    print(f"  Session: {session_id}", flush=True)
    print(f"  Project: {state.project_name} ({state.project_dir})", flush=True)
    print(f"  Goal: {state.goal}", flush=True)
    print(f"  Steps: {state.plan_steps}", flush=True)
    print(f"  Corpus: {corpus_size} prompts", flush=True)
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

        # 2. Parse terminal state
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

        # 3. Update session phase
        previous_hash = state.pane_hash
        was_stuck = state.phase == "STUCK"
        state.update_pane_hash(term.content_hash)
        if was_stuck and previous_hash and term.content_hash != previous_hash:
            state.consecutive_dupes = 0
            state.consecutive_status = 0
        turn_pct = (turn + 1) / max_turns
        state.update_phase(turn_pct)

        # 4. Build session context from pane output (feeds the simulator)
        session_ctx = parse_session_context(raw, project_dir=state.project_dir)

        # 5. Identify plan_remaining
        remaining = [s for s in state.plan_steps if s not in state.goals_completed]

        # 6. Generate Mohamed-style prompt
        t0 = time.time()
        result = simulate(
            ctx=session_ctx,
            phase=state.phase,
            goal=state.goal,
            plan_remaining=remaining if remaining else None,
            machine=machine,
            pane_id=pane_id,
        )
        elapsed = round(time.time() - t0, 4)
        raw_prompt = result.prompt

        # 7. Dedup + destructive guards
        final_prompt = raw_prompt
        reason = f"v7:{result.strategy}"

        if is_destructive(raw_prompt):
            print(f"  BLOCKED destructive: {raw_prompt[:60]}", flush=True)
            final_prompt = get_escape_prompt(state.phase)
            reason = "destructive_block"
        elif is_duplicate(raw_prompt, state.last_prompts, phase=state.phase):
            print(f"  BLOCKED duplicate: {raw_prompt[:60]}", flush=True)
            final_prompt = get_escape_prompt(state.phase)
            reason = "dedup"
        elif raw_prompt.strip().lower() == "status" and state.consecutive_status >= 1:
            print(f"  BLOCKED status spam", flush=True)
            if remaining:
                final_prompt = remaining[0]
                reason = "goal_redirect"
            else:
                final_prompt = get_escape_prompt(state.phase)
                reason = "escape"

        # 8. Style score logging
        style = result.style_score
        print(f"  [{reason}] score={style.overall:.2f} ({elapsed}s): {final_prompt[:100]}", flush=True)
        if not style.passed:
            print(f"    WARNING: style check failed ({style.overall:.2f})", flush=True)

        # 9. Inject
        if dry_run:
            print(f"  [DRY RUN] Would inject", flush=True)
        else:
            ok = inject_prompt(machine, pane_id, final_prompt)
            print(f"  Injected: {ok}", flush=True)

        # 10. Update state
        state.record_prompt(final_prompt)

        # Check goal completion
        for step in state.plan_steps:
            if step not in state.goals_completed:
                keywords = step.lower().split()[:3]
                if any(kw in term.content.lower() for kw in keywords if len(kw) > 3):
                    if any(sig in term.content.lower() for sig in
                           ["done", "success", "passed", "created", "written", "committed"]):
                        state.goals_completed.append(step)
                        print(f"  GOAL COMPLETE: {step[:60]}", flush=True)

        # Digest
        outcome = "error" if term.error_detected else "progress"
        state.record_digest(
            summary=final_prompt[:100],
            outcome=outcome,
            tools=[term.last_tool] if term.last_tool else [],
        )
        state.save()

        # Log entry
        entry = {
            "timestamp": datetime.datetime.now().isoformat(),
            "turn": turn + 1,
            "phase": state.phase,
            "strategy": result.strategy,
            "raw_prompt": raw_prompt,
            "final_prompt": final_prompt,
            "reason": reason,
            "style_score": style.overall,
            "style_passed": style.passed,
            "elapsed_ms": round(elapsed * 1000),
            "source_prompt": result.source_prompt[:80] if result.source_prompt else "",
            "errors_in_pane": session_ctx.errors_seen[:2],
            "files_created": session_ctx.files_created[:3],
            "goals_done": len(state.goals_completed),
            "goals_total": len(state.plan_steps),
        }
        with open(log_path, "a") as f:
            f.write(json.dumps(entry) + "\n")

        time.sleep(interval)

    print(f"\nDone. {state.turn} turns completed.")
    print(f"Goals: {len(state.goals_completed)}/{len(state.plan_steps)}")
    print(f"Log: {log_path}")


def main():
    parser = argparse.ArgumentParser(description="KARL V7 Mohamed Simulator Session Driver")
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
