#!/usr/bin/env python3
"""KARL V5 — Cognitive Twin Session Driver.

Reads real Claude session state from panes, generates the next prompt
Mohamed would type, and injects it to continue the session autonomously.

Flow:
  1. Read pane output (last N lines of a real Claude session)
  2. Feed to twin: "Given this session state, what would Mohamed say next?"
  3. Twin generates the next user prompt
  4. Inject into the pane via meshd or tmux
  5. Wait for Claude to respond
  6. Repeat
"""

import json, time, datetime, os, sys, subprocess, re
import urllib.request

MLX_URL = os.environ.get("MLX_URL", "http://100.109.94.124:8100/v1/chat/completions")
MODEL = "mlx-community/Qwen3-4B-Instruct-2507-4bit"
LOG_DIR = os.path.expanduser("~/Desktop/karl/session-logs")
os.makedirs(LOG_DIR, exist_ok=True)

TWIN_SYSTEM = """You are Mohamed's cognitive twin, acting as the USER in a Claude Code terminal session.

Your job: look at the current session output (what Claude just did or said) and generate THE NEXT PROMPT that Mohamed would type.

Rules:
- You are the human side of the conversation. You type prompts, Claude responds.
- Be direct. Short prompts. Mohamed doesn't write paragraphs as prompts.
- If Claude just finished a task, tell it what to do next.
- If Claude hit an error, tell it how to fix it or what to try instead.
- If Claude is asking a question, answer it decisively.
- If the work looks done, say "status" or move to the next logical task.
- Never generate Claude's response. Only generate what Mohamed would type.
- Keep it under 2 sentences. Mohamed types short prompts.
- If you genuinely don't know what to do next, say "status" to get bearings.

Output ONLY the prompt text. No quotes, no "I would say", no explanation. Just the raw prompt."""


def read_pane(machine, pane_id):
    """Read last 80 lines from a pane via meshd or tmux."""
    if machine == "localhost" or machine == "mac1":
        # Local tmux pane — use -p without -S for visible content
        try:
            result = subprocess.run(
                ["tmux", "capture-pane", "-t", pane_id, "-p"],
                capture_output=True, text=True, timeout=5
            )
            if result.returncode == 0 and result.stdout.strip():
                return result.stdout
        except Exception:
            pass

    # Try meshd HTTP API
    try:
        url = f"http://{machine}:9451/read/{pane_id}"
        req = urllib.request.Request(url, headers={"Accept": "text/plain"})
        with urllib.request.urlopen(req, timeout=10) as resp:
            return resp.read().decode()
    except Exception:
        pass

    # Try tmux over SSH
    try:
        result = subprocess.run(
            ["ssh", "-o", "ConnectTimeout=5", machine,
             f"tmux capture-pane -t {pane_id} -p -S -80"],
            capture_output=True, text=True, timeout=15
        )
        if result.returncode == 0:
            return result.stdout
    except Exception:
        pass

    return None


def inject_prompt(machine, pane_id, prompt):
    """Inject a prompt into a pane via tmux send-keys."""
    escaped = prompt.replace("'", "'\\''")

    if machine == "localhost" or machine == "mac1":
        # Local tmux
        try:
            # Use load-buffer to avoid escaping issues
            subprocess.run(
                ["tmux", "load-buffer", "-"],
                input=prompt.encode(), timeout=5
            )
            subprocess.run(
                ["tmux", "paste-buffer", "-t", pane_id],
                timeout=5
            )
            subprocess.run(
                ["tmux", "send-keys", "-t", pane_id, "Enter"],
                timeout=5
            )
            return True
        except Exception:
            pass

    # Remote via SSH + tmux
    try:
        # load-buffer + paste-buffer avoids escaping hell
        cmd = f"printf '%s' '{escaped}' | tmux load-buffer - && tmux paste-buffer -t {pane_id} && tmux send-keys -t {pane_id} Enter"
        result = subprocess.run(
            ["ssh", "-o", "ConnectTimeout=5", machine, cmd],
            capture_output=True, text=True, timeout=15
        )
        return result.returncode == 0
    except Exception:
        return False


def query_twin(session_output):
    """Ask the twin what Mohamed would type next."""
    messages = [
        {"role": "user", "content": f"Here is the current Claude Code session output (last 80 lines):\n\n```\n{session_output}\n```\n\nWhat does Mohamed type next?"}
    ]
    payload = json.dumps({
        "model": MODEL,
        "messages": [{"role": "system", "content": TWIN_SYSTEM}] + messages,
        "max_tokens": 100,
        "temperature": 0.7,
    }).encode()

    req = urllib.request.Request(MLX_URL, data=payload, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=60) as resp:
        data = json.loads(resp.read())
        return data["choices"][0]["message"]["content"].replace("<|im_end|>", "").strip()


def drive_session(machine, pane_id, max_turns=10, interval=30, dry_run=False,
                   claude_cmd="claude --dangerously-skip-permissions", seed_prompt=None):
    """Drive a session by reading state and injecting twin-generated prompts."""
    ts = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    safe_pane = pane_id.replace(":", "_").replace(".", "_")
    log_path = os.path.join(LOG_DIR, f"{ts}_{machine}_{safe_pane}.jsonl")

    print(f"Twin Session Driver")
    print(f"  Target: {machine}:{pane_id}")
    print(f"  Claude cmd: {claude_cmd}")
    print(f"  Max turns: {max_turns}")
    print(f"  Interval: {interval}s")
    print(f"  Dry run: {dry_run}")
    print(f"  Seed: {seed_prompt[:80] if seed_prompt else 'none'}...")
    print(f"  Log: {log_path}")
    print()

    # If seed prompt provided, inject it as the first thing
    if seed_prompt and not dry_run:
        print(f"  Injecting seed prompt...")
        inject_prompt(machine, pane_id, seed_prompt)
        time.sleep(interval)  # Wait for Claude to process the seed

    for turn in range(max_turns):
        print(f"--- Turn {turn + 1}/{max_turns} ---")

        # 1. Read pane
        output = read_pane(machine, pane_id)
        if not output:
            print(f"  Could not read pane {machine}:{pane_id}")
            time.sleep(interval)
            continue

        # Trim to last 80 meaningful lines
        lines = [l for l in output.strip().split("\n") if l.strip()]
        context = "\n".join(lines[-80:])
        print(f"  Read {len(lines)} lines from pane")

        # 2a. Detect if session dropped to shell (not in Claude anymore)
        last_lines_raw = "\n".join(lines[-5:])
        if re.search(r'(zsh|bash)\s*$', last_lines_raw) or re.search(r'\$\s*$', last_lines_raw):
            # Check if it looks like a shell prompt, not Claude
            if not any(c in last_lines_raw for c in ["❯", "Claude", "Synthesizing", "Crunched"]):
                print(f"  SESSION DEAD — shell prompt detected. Restarting Claude...")
                inject_prompt(machine, pane_id, claude_cmd)
                time.sleep(15)
                continue

        # 2b. Check if Claude is still working (look for spinner or tool activity)
        last_lines = last_lines_raw.lower()
        if any(w in last_lines for w in ["running", "searching", "reading", "writing", "building", "synthesizing", "crunched"]):
            print(f"  Claude appears to be working. Waiting...")
            time.sleep(interval)
            continue

        # 3. Generate next prompt
        t0 = time.time()
        next_prompt = query_twin(context)
        elapsed = round(time.time() - t0, 2)

        print(f"  Twin says ({elapsed}s): {next_prompt[:120]}")

        # 4. Log
        entry = {
            "timestamp": datetime.datetime.now().isoformat(),
            "turn": turn + 1,
            "machine": machine,
            "pane": pane_id,
            "context_lines": len(lines),
            "twin_prompt": next_prompt,
            "elapsed": elapsed,
            "injected": not dry_run,
        }
        with open(log_path, "a") as f:
            f.write(json.dumps(entry) + "\n")

        # 5. Inject (or just log in dry run)
        if dry_run:
            print(f"  [DRY RUN] Would inject: {next_prompt[:80]}")
        else:
            ok = inject_prompt(machine, pane_id, next_prompt)
            print(f"  Injected: {ok}")

        # 6. Wait for Claude to process
        print(f"  Waiting {interval}s for response...")
        time.sleep(interval)

    print(f"\nDone. {max_turns} turns. Log: {log_path}")


def main():
    import argparse
    parser = argparse.ArgumentParser(description="KARL V5 Twin Session Driver")
    parser.add_argument("machine", help="Target machine (mac1, mac2, mac5, localhost)")
    parser.add_argument("pane", help="Tmux pane ID (e.g., 0:0.1, claude-session)")
    parser.add_argument("--turns", type=int, default=10, help="Max turns")
    parser.add_argument("--interval", type=int, default=30, help="Seconds between turns")
    parser.add_argument("--dry-run", action="store_true", help="Don't inject, just log what twin would say")
    parser.add_argument("--claude-cmd", default="claude --dangerously-skip-permissions", help="Command to launch Claude")
    parser.add_argument("--seed", default=None, help="Initial prompt to inject before twin takes over")
    args = parser.parse_args()

    drive_session(args.machine, args.pane, args.turns, args.interval, args.dry_run,
                  args.claude_cmd, args.seed)


if __name__ == "__main__":
    main()
