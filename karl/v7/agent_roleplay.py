#!/usr/bin/env python3
"""Agent Roleplay: Use a coding agent to generate Mohamed-style prompts.

Instead of templates or TF-IDF search, this gives Claude (via --print)
the full context of:
  1. Mohamed's prompt DNA (20 exemplars from 4,587 real prompts)
  2. Current session state (files, errors, plan progress)
  3. The project goal and remaining steps
  4. Style rules (signature phrases, anti-patterns, voice)

The agent generates what Mohamed would actually type next.
All output runs through style_validator before delivery.

Usage:
    from karl.v7.agent_roleplay import generate_as_mohamed
    prompt = generate_as_mohamed(ctx, goal, plan_remaining)
"""

import json
import os
import random
import subprocess
import urllib.request
from pathlib import Path

from .prompt_corpus import get_corpus, search_similar, PromptEntry
from .session_context_reader import SessionContext

# Backend config — supports OpenAI and Together AI
TOGETHER_API_KEY = os.environ.get("TOGETHER_API_KEY", "8a755cb3637f39c6a397caffa33c71b8ac4d98cbe697914fcf7de0a4f413ca84")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")

# Default: Together AI GPT-OSS 120B (score 1.00, $0.15/$0.60 per 1M)
BACKEND = os.environ.get("KARL_ROLEPLAY_BACKEND", "together")
TOGETHER_MODEL = os.environ.get("KARL_TOGETHER_MODEL", "openai/gpt-oss-120b")
OPENAI_MODEL = "gpt-5.4-mini"

EXEMPLAR_CACHE_PATH = os.path.expanduser("~/Desktop/karl/v7-mohamed-exemplars.json")
CORRECTION_PAIRS_PATH = os.path.expanduser("~/Desktop/karl/v6-correction-pairs.jsonl")
KNOWLEDGE_PATH = os.path.expanduser("~/Desktop/karl/v7-knowledge-injection.md")


def _build_exemplars(n: int = 50) -> list[str]:
    """Select n diverse, high-quality exemplar prompts from the corpus.

    With 400K context, we can afford 50+ exemplars for richer style grounding.
    """
    corpus = get_corpus()
    good = [e for e in corpus
            if e.category in ("conversational", "contextual", "question")
            and 60 < e.length < 500
            and not any(x in e.text.lower() for x in [
                "traceback", "```", "mohameddiomande@", "zsh:", "/usr/bin/"
            ])]

    # Group by domain for diversity
    domains: dict[str, list[PromptEntry]] = {}
    for e in good:
        d = e.project_domain
        if d not in domains:
            domains[d] = []
        domains[d].append(e)

    # Sample 2-3 per domain, sorted by domain size
    samples = []
    for _d, es in sorted(domains.items(), key=lambda x: -len(x[1])):
        if len(samples) >= n:
            break
        pick = random.sample(es, min(3, len(es)))
        samples.extend(pick)

    return [e.text for e in samples[:n]]


def _get_exemplars() -> list[str]:
    """Load or build exemplar prompts (cached on disk)."""
    if os.path.exists(EXEMPLAR_CACHE_PATH):
        try:
            with open(EXEMPLAR_CACHE_PATH) as f:
                exemplars = json.load(f)
            if isinstance(exemplars, list) and len(exemplars) >= 30:
                return exemplars
        except (json.JSONDecodeError, OSError):
            pass

    exemplars = _build_exemplars(50)
    os.makedirs(os.path.dirname(EXEMPLAR_CACHE_PATH), exist_ok=True)
    with open(EXEMPLAR_CACHE_PATH, "w") as f:
        json.dump(exemplars, f)
    return exemplars


def _get_domain_exemplars(goal: str, n: int = 10) -> list[str]:
    """Get exemplars specifically relevant to the current goal domain.

    With 400K context we can include 10+ domain-specific examples.
    """
    results = search_similar(goal, n=n * 3)
    filtered = []
    for entry, score in results:
        if entry.category in ("paste", "approval"):
            continue
        if len(entry.text) > 500 or len(entry.text) < 30:
            continue
        if any(x in entry.text.lower() for x in ["```", "traceback"]):
            continue
        filtered.append(entry.text)
        if len(filtered) >= n:
            break
    return filtered


_knowledge_cache: str | None = None


def _get_knowledge() -> str:
    """Load the full knowledge injection document (infrastructure, decisions, rules)."""
    global _knowledge_cache
    if _knowledge_cache is not None:
        return _knowledge_cache
    if os.path.exists(KNOWLEDGE_PATH):
        with open(KNOWLEDGE_PATH) as f:
            _knowledge_cache = f.read()
        return _knowledge_cache
    return ""


def _get_correction_patterns(n: int = 30) -> list[dict]:
    """Load correction pairs — what Mohamed corrects and how.

    These teach the model Mohamed's failure-mode awareness.
    """
    if not os.path.exists(CORRECTION_PAIRS_PATH):
        return []
    pairs = []
    with open(CORRECTION_PAIRS_PATH) as f:
        for line in f:
            try:
                pairs.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    # Sample diverse failure modes
    by_mode: dict[str, list] = {}
    for p in pairs:
        mode = p.get("failure_mode", "unknown")
        if mode not in by_mode:
            by_mode[mode] = []
        by_mode[mode].append(p)

    sampled = []
    for mode, ps in sorted(by_mode.items(), key=lambda x: -len(x[1])):
        pick = random.sample(ps, min(4, len(ps)))
        sampled.extend(pick)
        if len(sampled) >= n:
            break
    return sampled[:n]


def _build_multi_chain_messages(
    ctx: SessionContext,
    goal: str,
    plan_remaining: list[str] | None,
    phase: str,
    exemplars: list[str],
    domain_exemplars: list[str],
    correction_patterns: list[dict],
    previous_prompts: list[str] | None = None,
    knowledge: str = "",
    machine: str = "",
    pane_id: str = "",
) -> list[dict]:
    """Build multi-chain messages exploiting 400K context.

    Chain 1 (system): Full identity + style DNA + infrastructure knowledge
    Chain 2 (user turn 1): 50 real exemplar prompts as style grounding
    Chain 3 (assistant turn 1): Acknowledgment of style patterns
    Chain 4 (user turn 2): Correction patterns — what Mohamed corrects and why
    Chain 5 (assistant turn 2): Internalization of correction awareness
    Chain 6 (user turn 3): Current session state + generate prompt
    """

    # --- Session context ---
    ctx_parts = []
    if ctx.files_created:
        ctx_parts.append(f"Files created: {', '.join(ctx.files_created[:8])}")
    if ctx.files_modified:
        ctx_parts.append(f"Files modified: {', '.join(ctx.files_modified[:8])}")
    if ctx.errors_seen:
        ctx_parts.append(f"Errors:\n" + "\n".join(f"  - {e}" for e in ctx.errors_seen[:5]))
    if ctx.tools_used:
        ctx_parts.append(f"Tools used: {', '.join(f'{k}({v})' for k, v in ctx.tools_used.items())}")
    if ctx.last_claude_action:
        ctx_parts.append(f"Claude's last action: {ctx.last_claude_action}")
    if ctx.lines_of_code_written > 0:
        ctx_parts.append(f"~{ctx.lines_of_code_written} lines of code written this session")
    session_summary = "\n".join(ctx_parts) if ctx_parts else "Session just started. Nothing built yet."

    plan_text = ""
    if plan_remaining:
        plan_text = "Remaining plan steps:\n" + "\n".join(f"  {i+1}. {s}" for i, s in enumerate(plan_remaining[:8]))
    else:
        plan_text = "All planned steps are complete. Time to verify, test, and commit."

    prev_text = ""
    if previous_prompts:
        prev_text = "\nPrompts you already sent this session (DO NOT repeat these):\n" + "\n".join(
            f'  - "{p[:150]}"' for p in previous_prompts[-8:]
        )

    # --- Chain 1: System (identity + infrastructure + style rules) ---
    current_machine = machine or "current machine"
    current_pane = pane_id or "current pane"

    system = f"""You are Mohamed Diomande. You are driving a Claude Code session by typing prompts into the terminal.

WHO YOU ARE:
- Builder and architect running a 5-Mac mesh network (Mac1-5 + cloud-vm) with 60+ autonomous AI agents
- You build in Rust, Python, Swift, TypeScript. You ship fast.
- You run NATS on :4222, meshd on :9451, RAG++ on :8000, MLX on Mac5:8100, Grafana on :3000, Prefect on :4200
- Machines: Mac1 (build host, Xcode), Mac2 (iOS, TD), Mac3 (creative), Mac4 (Adobe, compute), Mac5 (ML, MLX, fine-tuning)
- Cloud-vm at 100.114.92.88 runs Docker Compose infra (Grafana, Prefect, Nexus)
- You have Supabase with 141 tables, 329K+ memory_turns rows
- Projects: KARL (trajectory intelligence), meshd, bridged, speakd, MotionMix, Spore, MeshControl, NKo ASR

CURRENT EXECUTION CONTEXT:
- This exact Claude session is already running on {current_machine}, pane {current_pane}
- Stay on this machine by default
- Do NOT suggest SSHing to another machine just to continue work in the same project directory
- Only bring up another machine if the goal explicitly depends on a remote service or host-specific capability

YOUR VOICE:
- 59% conversational ("I think we should...", "so what if...", "the thing is...")
- 20% questions ("what's the status on...", "can we...", "does it make sense to...")
- 14% contextual ("so after that...", "now that we have...", "since the error...")
- 5% corrections ("don't do that", "make sure", "keep in mind")
- 2% imperative (only for simple: "yes", "status", "continue")
- Signature phrases: "just" (928x), "let's" (561x), "don't" (538x), "I think" (272x), "can we" (187x), "make sure" (184x)
- Short sentences. Like texting. Never flowery. Never AI-sounding.
- NEVER use: "certainly", "I'd be happy to", "let me", "here's", "delve", "leverage", "holistic"
- You reference specific files, errors, port numbers, machine names
- You delegate: "you decide what's best", "figure it out", "just do it"
- You sometimes give multi-clause prompts connected with "and then", "so", "but first", "after that"

OUTPUT RULES:
- Output ONLY the prompt text Mohamed would type. Nothing else.
- No quotes around it. No explanation. No "Here's what I'd say". No meta-commentary.
- Do NOT start with an imperative verb alone. Start conversationally.
- Reference specific things from the session state: file names, error messages, plan steps.
- Each prompt must be DIFFERENT from previous ones — never repeat yourself."""

    # --- Chain 2: Exemplar prompts (50 real prompts for style grounding) ---
    random.shuffle(exemplars)
    exemplar_block = "\n".join(f"{i+1}. {e}" for i, e in enumerate(exemplars[:50]))

    domain_block = ""
    if domain_exemplars:
        domain_block = "\n\nPrompts from projects similar to the current one:\n" + "\n".join(
            f"- {e}" for e in domain_exemplars[:10]
        )

    user_1 = f"""Here are 50 of my real prompts from past sessions. Study my voice, my patterns, my rhythm:

{exemplar_block}
{domain_block}

Internalize my style. Note how I think, what I focus on, how I phrase things."""

    assistant_1 = """I've studied your prompts. Key patterns I see:
- You lead with "I think" or "let's" to frame direction conversationally
- You reference specific files, services, and ports by name
- You chain clauses with "and then", "so", "but first"
- You preface constraints: "make sure", "keep in mind", "don't"
- You ask questions that guide rather than command: "can we", "what if"
- Short, direct. You don't explain yourself. You state what you want and move."""

    # --- Chain 3: Correction patterns (what Mohamed corrects) ---
    correction_block = ""
    if correction_patterns:
        samples = random.sample(correction_patterns, min(20, len(correction_patterns)))
        lines = []
        for p in samples:
            lines.append(f"- [{p.get('failure_mode', '?')}] trigger: \"{p.get('trigger_phrase', '')}\" → \"{p.get('user_prompt', '')[:120]}\"")
        correction_block = "\n".join(lines)

    user_2 = f"""Here are patterns of what I correct in Claude Code sessions. Each is a failure mode I react to:

{correction_block if correction_block else "No correction data available."}

These show what frustrates me: over-explaining, going the wrong direction, stalling, cutting corners, missing context, shallow fixes. When you generate my prompt, preemptively address these — I'd rather prevent the failure than correct it after."""

    assistant_2 = """I understand your correction patterns:
- "just" → you want execution, not explanation
- "don't" → agent went wrong direction, you redirect
- "let's" → agent stalled or asked permission when it should have acted
- "make sure" → you're preempting a quality miss
- "keep in mind" → surfacing context the agent should already know
- "figure out" → agent gave a surface answer, you want depth
I'll generate prompts that preempt these failure modes naturally."""

    # --- Chain 4: Knowledge injection (infrastructure, decisions, gotchas) ---
    knowledge_chain = []
    if knowledge:
        user_k = f"""Here is my complete system knowledge — infrastructure, architecture decisions, rules, and gotchas. Use this to make prompts that reference REAL services, ports, and machines:

{knowledge[:20000]}"""

        assistant_k = """I've internalized your system knowledge. I know:
- Your 5-Mac mesh (Mac1 orchestrator, Mac2 TD/motion, Mac3 storage, Mac4 Adobe/compute, Mac5 ML/MLX)
- Key services: meshd :9451, NATS :4222, RAG++ :8000, MLX :8100, bridged :9460
- Your decision patterns: no-pause execution, mesh over monolith, externalized context for 4B models
- Your gotchas: no heavy ML on Mac1, yt-dlp brew only, SSH aliases not IPs, unset CLAUDECODE for child panes
- Your code rules: no wrappers, no deferred work, no premature abstractions, verify after edit
I'll reference these naturally in the prompts I generate."""

        knowledge_chain = [
            {"role": "user", "content": user_k},
            {"role": "assistant", "content": assistant_k},
        ]

    # --- Chain 5: Current session + generate ---
    user_3 = f"""CURRENT SESSION:
Goal: {goal}
Phase: {phase} (EXPLORE=orient, BUILD=implement, CLOSE=verify+commit, STUCK=unblock)
Project: {os.path.basename(ctx.project_dir) if ctx.project_dir else 'unknown'}
Directory: {ctx.project_dir}
Machine: {current_machine} (THIS session already runs here)
Pane: {current_pane}

Session state:
{session_summary}

{plan_text}
{prev_text}

Generate my next prompt. Remember: sound like ME, reference the specific files/errors/steps from above, use your knowledge of my infrastructure naturally, and don't repeat anything I already said. Keep the work on this machine and this pane unless the goal explicitly requires a remote machine."""

    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user_1},
        {"role": "assistant", "content": assistant_1},
        {"role": "user", "content": user_2},
        {"role": "assistant", "content": assistant_2},
        *knowledge_chain,
        {"role": "user", "content": user_3},
    ]


def _query_api(messages: list[dict], timeout: int = 30) -> str | None:
    """Query the configured LLM backend (Together AI or OpenAI).

    Uses subprocess curl to bypass Cloudflare blocks on urllib.
    """
    if BACKEND == "together" and TOGETHER_API_KEY:
        model = TOGETHER_MODEL
        url = "https://api.together.xyz/v1/chat/completions"
        key = TOGETHER_API_KEY
    elif OPENAI_API_KEY:
        model = OPENAI_MODEL
        url = "https://api.openai.com/v1/chat/completions"
        key = OPENAI_API_KEY
    else:
        return None

    payload = json.dumps({
        "model": model,
        "messages": messages,
        "temperature": 0.85,
    })

    try:
        result = subprocess.run(
            ["curl", "-s", "--max-time", str(timeout), url,
             "-H", f"Authorization: Bearer {key}",
             "-H", "Content-Type: application/json",
             "-d", payload],
            capture_output=True, text=True, timeout=timeout + 5,
        )
        if result.returncode == 0 and result.stdout.strip():
            data = json.loads(result.stdout)
            if "choices" in data:
                return data["choices"][0]["message"]["content"].strip()
            if "error" in data:
                return None
    except Exception:
        pass
    return None


def _clean_response(raw: str) -> str:
    """Strip meta-commentary and quotes from roleplay output."""
    raw = raw.strip().strip('"\'').strip()
    lines = [l.strip() for l in raw.split("\n") if l.strip()]
    if not lines:
        return ""

    # Skip lines that are meta-commentary
    skip_prefixes = [
        "here", "i would", "as mohamed", "the next", "this prompt",
        "note:", "explanation:", "output:", "prompt:", "sure",
        "based on", "given the",
    ]
    for line in lines:
        if not any(line.lower().startswith(x) for x in skip_prefixes):
            return line[:500]

    return lines[0][:500]


def generate_as_mohamed(
    ctx: SessionContext,
    goal: str = "",
    plan_remaining: list[str] | None = None,
    phase: str = "BUILD",
    timeout: int = 20,
    previous_prompts: list[str] | None = None,
    machine: str = "",
    pane_id: str = "",
) -> tuple[str, bool]:
    """Generate a prompt via multi-chain roleplay on GPT-5.4-mini (400K context).

    Chain: identity → 50 exemplars → correction patterns → session state → generate.
    Returns (prompt_text, success).
    """
    exemplars = _get_exemplars()
    domain_exemplars = _get_domain_exemplars(goal) if goal else []
    corrections = _get_correction_patterns(20)
    knowledge = _get_knowledge()

    messages = _build_multi_chain_messages(
        ctx, goal, plan_remaining, phase,
        exemplars, domain_exemplars, corrections,
        previous_prompts, knowledge, machine, pane_id,
    )

    raw = _query_api(messages, timeout=timeout)
    if raw:
        cleaned = _clean_response(raw)
        if cleaned and len(cleaned) >= 15:
            return cleaned, True

    return "", False


if __name__ == "__main__":
    print("Agent Roleplay — Mohamed Simulator Test")
    print("=" * 60)

    # Build exemplars
    exemplars = _get_exemplars()
    print(f"Loaded {len(exemplars)} exemplars")

    # Test contexts
    contexts = [
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
         ["Fix imports", "Add /chat endpoint", "Add /health endpoint"]),

        (SessionContext(
            project_dir="/Users/m/Desktop/mesh-pulse",
            last_claude_action="scaffolded Next.js app",
            files_created=["/Users/m/Desktop/mesh-pulse/app/page.tsx"],
        ), "EXPLORE", "Build a Next.js dashboard polling meshd :9451 on 5 Macs",
         ["Add status cards", "Wire up server-side polling", "Dark theme with shadcn"]),

        (SessionContext(
            errors_seen=["ConnectionRefusedError: [Errno 111] Connection refused"],
            tools_used={"Bash": 8},
            project_dir="/Users/m/Desktop/nats-bridge",
            last_claude_action="ran: curl localhost:4222",
        ), "STUCK", "Set up NATS bridge for mesh events",
         ["Fix connection", "Test event publishing"]),

        (SessionContext(
            files_created=["/Users/m/Desktop/api/src/routes.py", "/Users/m/Desktop/api/tests/test_routes.py"],
            tools_used={"Write": 2, "Bash": 4},
            project_dir="/Users/m/Desktop/inscription-router",
            last_claude_action="ran: cargo test",
            lines_of_code_written=300,
        ), "CLOSE", "Build a Rust HTTP service classifying prompts into inscriptions",
         ["Final integration test"]),
    ]

    from .style_validator import validate

    for i, (ctx, phase, goal, remaining) in enumerate(contexts, 1):
        print(f"\n[{i}] Phase={phase} Goal={goal[:50]}...")
        prompt, ok = generate_as_mohamed(ctx, goal, remaining, phase)
        if ok:
            score = validate(prompt)
            tag = "PASS" if score.passed else "FAIL"
            print(f"  [{tag}] score={score.overall:.2f}: {prompt[:150]}")
        else:
            print(f"  FAILED to generate")
