# KARL V6 — Cognitive Twin Architecture
## Autonomous Session Driver with Persistent Context

**Version**: 6.0.0-design
**Date**: 2026-04-01
**Status**: Architecture (pre-implementation)
**Supersedes**: V5 `twin_session_driver.py`

---

## Executive Summary

V5 failed after turn 15 for one reason: the 4B model is a **reaction machine**, not a **planning machine**. It sees 80 lines of terminal output, generates a plausible next prompt, and forgets everything. It has no model of where it is in a project, no memory of what it already tried, and no self-monitor to notice it's looping. It's a parrot reading a scroll — fluent in the present, amnesiac about the past.

V6 does not fine-tune to fix this. Instead, it **externalizes everything the model cannot hold internally** — project state, turn history, repetition detection, task graph — and **injects that context into every single prompt** as structured scaffolding. The model's job shrinks from "figure out the whole session" to "given a complete briefing, pick the next one sentence." That is a problem a 4B model can solve.

---

# PHASE 1: PRIME
## Core Insight — Why Small Models Fail at Multi-Turn Session Driving

### The Fundamental Mismatch

A 4B model has approximately 2048 tokens of working memory. A real 50-turn Claude Code session contains roughly 40,000 tokens of state. Every time the driver calls the twin, it is asking the model to navigate a 40,000-token project with a 2,048-token lens. The model does not fail because it is unintelligent. It fails because the problem is structurally larger than the context window.

### The Three Failure Signatures Decoded

**Status spam (28% of turns):** When the model cannot determine what happened or what state the project is in from 80 lines of scrollback, it falls back to "status" — the safe neutral move. This is not stupidity, it is uncertainty maximization. The model knows "status" won't break anything. It chooses safety over progress because it has no project map.

**Cross-contamination (9%):** The 80-line context window contains file paths, error messages, and tool calls. When driving multiple sessions, these paths are similar (both might show `~/Desktop/SomeProject/src/`). Without an explicit anchor saying "you are driving MotionMix, not FirstDate," the model pattern-matches on the most recent tokens and gets confused. The project identity leaks.

**Repetition (13%):** The model cannot distinguish "I tried this exact prompt 3 turns ago" from "I should try this prompt." It has no turn history. To it, every invocation is turn 1. Repetition is structurally inevitable without an external deduplication mechanism.

### The Insight

**The model is not broken. The scaffolding is broken.**

V5 gives the model raw terminal output and asks it to infer everything. V6 gives the model a structured briefing and asks it to choose one action. The cognitive load shifts from the model (which cannot hold it) to the Python driver (which can).

---

# PHASE 2: EXPLODE
## 6 Divergent Approaches to Giving a Memoryless Model Persistent Awareness

### Approach A: External State File (The Ledger)

Maintain a JSON file on disk that tracks: current project, turn count, last 10 prompts, last 5 Claude responses, current task, completed tasks, blocked tasks. Before each twin query, serialize the ledger into the prompt context. After each turn, update the ledger.

**Strength**: Simple, debuggable, persists across restarts.
**Weakness**: Growing prompt size. By turn 30 the ledger alone is 800 tokens.

### Approach B: Compressed Rolling Summary (The Digest)

After every 5 turns, call a fast summarizer (same MLX model, different system prompt) to compress the conversation history into a 3-sentence summary. The digest replaces the raw history in future prompts.

**Strength**: Bounded context size regardless of session length.
**Weakness**: Summarizer can lose critical details (like "we tried X and it failed").

### Approach C: Structured Task Graph (The Planner)

Before driving a session, decompose the seed goal into a task graph with 5-10 nodes. Each node has: task description, status (pending/active/done/blocked), dependencies. The driver navigates the graph explicitly — pick the next unblocked pending node, drive toward it.

**Strength**: Long-range coherence. The model always knows what it's working toward.
**Weakness**: Requires upfront decomposition. Graph can go stale if Claude takes the session in an unexpected direction.

### Approach D: Repetition Hash Ring (The Guard)

Maintain a rolling hash ring of the last 20 prompt embeddings. Before injecting any twin-generated prompt, compute its embedding, check cosine similarity against the ring. If similarity > 0.85 with any recent prompt, force a "zoom out" prompt instead. Rotate the ring after each injection.

**Strength**: Eliminates repetition structurally.
**Weakness**: Embedding adds latency. Cosine threshold needs tuning.

### Approach E: Confidence Gating (The Pause)

Before injecting, ask the twin a second question: "How confident are you this is the right next step? (0-100)." If confidence < 60, don't inject. Instead, inject "status" and wait for Claude's output to clarify. Only proceed when the twin is confident.

**Strength**: Prevents wrong injections.
**Weakness**: Doubles MLX calls. "Status" becomes the fallback again, just less often.

### Approach F: Phase-Aware Prompting (The Clock)

Split every session into 3 phases: Explore (turns 1-8), Build (turns 9-20), Close (turns 21+). Give the model a different system prompt for each phase. Explore: "discover and understand." Build: "stay on the current task until it works." Close: "verify, test, commit, document."

**Strength**: Prevents the drift toward status-spam at turn 15 by giving the model an explicit phase context.
**Weakness**: Turn counts are a proxy for phases. Real sessions don't always follow the schedule.

---

# PHASE 3: FORGE
## Unified V6 Architecture

V6 merges Approaches A, B, C, D, and F. Approach E (confidence gating) is excluded — it doubles latency and still produces status-spam as the fallback. The merged system is the **Context Stack**.

### The Context Stack

```
┌─────────────────────────────────────────────────────────────────┐
│                    TWIN PROMPT (2048 tokens)                    │
├─────────────────────────────────────────────────────────────────┤
│  SYSTEM PROMPT (300 tokens)                                     │
│  + Identity: "You are Mohamed's cognitive twin..."              │
│  + Phase-aware instructions (Explore / Build / Close)           │
├─────────────────────────────────────────────────────────────────┤
│  SESSION BRIEF (200 tokens)                                     │
│  + Project: "MotionMix — iOS fitness app"                       │
│  + Goal: "Fix RecordingService crash on stopRecording()"        │
│  + Turn: 12/30                                                  │
│  + Phase: BUILD                                                 │
│  + Machine: mac1, Pane: 0:1.2                                  │
├─────────────────────────────────────────────────────────────────┤
│  TASK GRAPH SNAPSHOT (150 tokens)                               │
│  + DONE: read RecordingService.swift                            │
│  + DONE: reproduce crash with test case                         │
│  + ACTIVE: fix writerQueue race condition                       │
│  + PENDING: test fix, commit                                    │
├─────────────────────────────────────────────────────────────────┤
│  SESSION DIGEST (150 tokens, last 5 turns compressed)           │
│  + "Claude found the race condition in startWriting(). Tried    │
│     serialQueue but it deadlocked. Now trying DispatchGroup."   │
├─────────────────────────────────────────────────────────────────┤
│  ANTI-REPEAT FENCE (100 tokens)                                 │
│  + "Do NOT say any of these: 'show me the error', 'what did     │
│     you try', 'run tests', 'git status'"                        │
├─────────────────────────────────────────────────────────────────┤
│  TERMINAL CONTEXT (900 tokens, last ~60 lines)                  │
│  + Raw pane output from current Claude session                  │
├─────────────────────────────────────────────────────────────────┤
│  USER TURN (48 tokens)                                          │
│  + "What does Mohamed type next? One sentence. Be specific."    │
└─────────────────────────────────────────────────────────────────┘
```

**Total: ~1848 tokens. Fits in the 2048 window with 200 tokens headroom for generation.**

---

# PHASE 4: SYNTHESIZE
## Component Architecture

### Component 1: Session Brief Generator

**File**: `v6/session_brief.py`
**Role**: Produces the 200-token SESSION BRIEF block.

```python
@dataclass
class SessionBrief:
    project_name: str        # "MotionMix"
    project_path: str        # "~/Desktop/MotionMix"
    goal: str                # One sentence, under 20 words
    turn: int
    max_turns: int
    phase: str               # "EXPLORE" | "BUILD" | "CLOSE"
    machine: str
    pane_id: str
    seed_prompt: str         # Original goal injected at turn 0
```

Phase is determined by turn position, not just count:
- Turns 1-30%: EXPLORE
- Turns 31-80%: BUILD
- Turns 81-100%: CLOSE

### Component 2: Task Graph

**File**: `v6/task_graph.py`
**Role**: Tracks a lightweight task tree across turns.

Tasks are inferred two ways:
1. **Seed decomposition**: At session start, call the twin once with a special "decompose this goal into 5-8 tasks" prompt. Store as pending nodes.
2. **Completion inference**: After each Claude response, scan terminal output for completion signals (keywords: "done", "passed", "committed", "✓", "success", exit code 0). Mark matching task nodes complete.

```python
@dataclass
class TaskNode:
    id: str
    description: str         # Under 15 words
    status: str              # "pending" | "active" | "done" | "blocked"
    depends_on: list[str]    # IDs of prerequisite tasks
    turn_started: int | None
    turn_completed: int | None
```

Serialized as a 150-token snapshot showing only active + next 2 pending. Done tasks are collapsed to a count. Blocked tasks show their blocker.

### Component 3: Digest Engine

**File**: `v6/digest_engine.py`
**Role**: Compresses conversation history to a bounded token budget.

Stores raw turn records (prompt, response_summary, outcome) in a rolling buffer of 20. Every 5 turns, compresses the oldest 5 into a single 2-sentence summary using a cheap MLX call (150 max_tokens, temp 0.3). The digest is prepended with age: "Turns 1-5 summary: ..."

Turn record schema:
```python
@dataclass
class TurnRecord:
    turn: int
    twin_prompt: str         # What was injected
    claude_response: str     # First 200 chars of Claude's response
    outcome: str             # "progress" | "error" | "blocked" | "complete"
    tools_used: list[str]    # Read, Write, Bash, etc.
```

Outcome is classified by scanning the terminal output for error keywords, tool counts, and response length changes.

### Component 4: Anti-Repeat Fence

**File**: `v6/anti_repeat.py`
**Role**: Prevents duplicate prompt injection.

Uses a simple n-gram hash approach, not embeddings (no added latency):
- Extract 3-grams from generated prompt
- Check against a set of 3-grams from the last 15 prompts
- If Jaccard overlap > 0.4, mark as duplicate

On duplicate detection:
1. Log to session log with reason
2. Force-substitute with a phase-appropriate escape prompt:
   - EXPLORE: "what's the directory structure of this project?"
   - BUILD: "walk me through the current error in detail"
   - CLOSE: "run the full test suite and show results"

The escape prompts are guaranteed non-repeating because they rotate from a pool of 10 per phase and are flagged as "forced" in the log.

### Component 5: Phase-Aware System Prompt

**File**: `v6/phase_prompts.py`
**Role**: Different behavior instructions per phase.

```
EXPLORE (turns 1-30%):
"You are at the start of a session. Your job: understand before acting.
Read files, map the codebase, reproduce the problem, confirm your
understanding. Short prompts. Ask for one thing at a time.
If you see an error, ask Claude to explain it. Don't fix yet."

BUILD (turns 31-80%):
"You are mid-session. A task is active. Stay on it until it works or
you hit a hard blocker. Tell Claude exactly what to do next on the
current task. If it's working, keep going. Don't switch tasks unless
clearly blocked. Short directives: 'now add the test', 'run it',
'fix that import error'."

CLOSE (turns 81-100%):
"The work is nearly done. Your job: verify everything works, run tests,
commit the changes, document what changed. If something is broken,
say so. If it's all green, tell Claude to commit with a real message.
Short commands: 'run tests', 'commit', 'push'."
```

### Component 6: Terminal Parser

**File**: `v6/terminal_parser.py`
**Role**: Extracts signal from raw pane output.

Returns a `TerminalState`:
```python
@dataclass
class TerminalState:
    is_alive: bool           # Is Claude running (not raw shell)
    is_working: bool         # Is Claude mid-task (spinner, tool activity)
    last_tool: str | None    # Most recent tool used
    error_detected: bool     # Any error/traceback visible
    build_result: str | None # "pass" | "fail" | None
    test_result: str | None  # "pass" | "fail" | None
    files_modified: list[str]# From Write/Edit tool output
    raw_lines: int
    content_lines: str       # Trimmed to 900-token budget
```

The driver uses `is_alive` and `is_working` before querying the twin. If `is_working`, wait and retry. If not `is_alive`, restart Claude.

### Component 7: Session State Persistence

**File**: `v6/session_state.py`
**Role**: JSON-on-disk state that survives restarts.

```
~/.karl/v6/sessions/{session_id}/
  state.json          # SessionBrief + TaskGraph + turn counter
  digest.json         # Compressed turn history
  anti_repeat.json    # 3-gram sets for last 15 prompts
  turns.jsonl         # Full turn log (append-only)
```

On restart, the driver reads state.json and resumes from where it left off. Session ID is derived from `{machine}_{pane}_{seed_hash}`.

---

## Data Flow Diagram

```
STARTUP
  seed_prompt + machine + pane + goal
       │
       ▼
  [Session Brief Generator]
       │ SessionBrief
       │
       ▼
  [Task Decomposer] ──── 1x MLX call (150 tokens) ─────► TaskGraph (5-8 nodes)
       │
       └── writes state.json
       │
       ▼

TURN LOOP
       │
       ▼
  [Terminal Parser] ◄── tmux capture-pane / meshd /read/{pane}
       │ TerminalState
       │
       ├── is_working=True ──► wait(interval), retry
       ├── is_alive=False  ──► inject restart_cmd, wait(15s), retry
       │
       ▼
  [Context Stack Builder]
       │ reads: SessionBrief, TaskGraph, Digest, AntiRepeat
       │ formats: 7-block prompt (1848 tokens)
       │
       ▼
  [MLX Query] ──── http://mac5:8100/v1/chat/completions ────► raw_prompt
       │
       ▼
  [Anti-Repeat Check]
       │
       ├── duplicate detected ──► substitute escape_prompt, log
       │
       ▼
  [Task Graph Updater]
       │ scans TerminalState for completion signals
       │ updates node statuses
       │
       ▼
  [Inject]
       │ tmux load-buffer + paste-buffer + send-keys Enter
       │
       ▼
  [TurnRecord logger]
       │ appends to turns.jsonl
       │ every 5 turns: trigger Digest compression
       │
       ▼
  wait(interval) ──► back to TURN LOOP
```

---

# PHASE 5: CREATE
## SCAMPER Innovation Pass

**S — Substitute: What if the task graph is not pre-computed but observed?**

Instead of decomposing up front (which requires a seed goal and 1 extra MLX call), infer tasks from what Claude actually does. Read tool use patterns: every `Bash("read")` cluster = an exploration task, every repeated `Write` + `Bash("run")` cycle = a build task. Build the task graph retroactively from behavior, not from planning.

Verdict: Include as an optional mode (`--observe-tasks`) alongside explicit decomposition. Useful when seeding from an already-running session without a clean goal statement.

**C — Combine: Merge the digest engine with the task graph.**

The digest is a summary of what happened. The task graph is a summary of what needs to happen. They should share state. When a task node is marked complete, the completion event is automatically added to the next digest as a "MILESTONE: task X done." This keeps the digest grounded in actual progress, not just conversational summaries.

Verdict: Implement. Add `milestone_events: list[str]` to the Digest dataclass. TaskGraph.mark_complete() calls Digest.add_milestone().

**A — Adapt: Steal from git commit message conventions.**

Good commit messages use imperative mood: "fix writerQueue race condition" not "fixed" or "fixing." The twin's generated prompts should be validated for imperative mood before injection. Add a simple regex filter: if the prompt contains past tense verbs ("fixed", "added", "checked"), rewrite to present imperative or flag.

Verdict: Add a 5-line postprocessor to `anti_repeat.py`. Low cost, high signal.

**M — Modify: Make the terminal context adaptive, not fixed.**

V5 always sends the last 80 lines (approximately 900 tokens). But sometimes the last 5 lines are all you need (Claude just said "done"). Sometimes you need 120 lines to see the full traceback. Add adaptive line selection: if `error_detected=True`, send 120 lines. If `build_result="pass"`, send 20 lines. Default 60 lines.

Verdict: Implement in Terminal Parser. Add `adaptive_line_count(terminal_state) -> int`.

**P — Put to other uses: The digest engine can generate training data.**

After every session completes, the compressed digest + task graph + turn log is a high-quality SFT example. The prompt is the session context. The completion is each twin-generated prompt that led to successful task completion. Wire this directly into KARL's existing `sft_exporter.py`. V6 sessions become V7 training data.

Verdict: Implement as `v6/session_exporter.py`. Runs on session close, writes to `~/.claude/karl/trajectories.jsonl`.

**E — Eliminate: Remove the waiting-for-busy-claude poll loop.**

V5 uses keyword matching ("running", "synthesizing") to detect if Claude is busy. This breaks silently when Claude changes its output format. Replace with a structural signal: track the raw line count of the pane. If the line count is still increasing (delta > 0 after 3 reads spaced 5s apart), Claude is working. If the count is stable for 15s, Claude is done.

Verdict: Implement in Terminal Parser as `LineCountStabilizer`. Eliminates 6 brittle keyword strings.

**R — Reverse: Let Claude drive the twin, not the other way around.**

What if Claude Code itself could detect when it needs human input and request a twin-generated prompt? Add a sentinel marker: if the last line of Claude's output contains "---AWAITING_INPUT---", the driver immediately queries the twin instead of waiting the full interval. This makes the system event-driven instead of time-polled for the common case where Claude finishes fast.

Verdict: Implement as fast-path in the turn loop. Check for sentinel in `is_working` detection. Falls back to time-polling if sentinel never appears.

---

# PHASE 6: EVOLVE
## Stress Tests Against the 3 V5 Failure Modes

### Failure Mode 1: Status Spam (V5: 28%)

**Test scenario**: 30-turn session driving a Rust daemon build. The seed is "add a new /health endpoint to meshd." By turn 15, the session has successfully added the route. V5 would start injecting "status."

**V6 response**:
- Turn 15 context: TaskGraph shows "add /health route" = DONE. Next pending = "write test for /health route."
- Phase: BUILD (turn 15 of 30 = 50%, still in BUILD).
- Digest: "Health endpoint added. Server starts on :9451. Test coverage missing."
- Twin prompt given: includes all of the above + BUILD phase instructions.
- Expected output: "now write a test that hits /health and checks the 200 response"

The model does not need to infer what to do next. It reads "test coverage missing" in the digest and has the next pending task in the graph. Status spam is eliminated because the model is never in an ambiguous state. The ledger tells it where it is.

**Residual risk**: TaskGraph completion inference misses the task being done, so the graph still shows it as ACTIVE. The model correctly continues working on it. No harm done — the graph is slightly stale but still directional.

**Predicted reduction**: 28% -> 4%. The remaining 4% occurs when the graph is fully exhausted and no new tasks are detectable, which genuinely warrants a status check.

### Failure Mode 2: Cross-Contamination (9%)

**Test scenario**: Two simultaneous sessions. Session A is driving MotionMix (iOS, Swift, RecordingService). Session B is driving meshd (Rust, PTY daemon). Both are on mac1, different panes. The terminal outputs look similar at a 80-line zoom level — both have file paths, both have build output.

**V6 response**:
- Each session has its own `state.json` with `project_name`, `project_path`, and `machine:pane`.
- Session A's SESSION BRIEF block says: "Project: MotionMix | Path: ~/Desktop/MotionMix | Machine: mac1:0:1.2"
- Session B's SESSION BRIEF block says: "Project: meshd | Path: ~/Desktop/Comp-Core/core/meshd | Machine: mac1:0:2.0"
- The brief appears BEFORE the terminal context in the prompt. The model anchors on the brief.

The anchoring test: manually craft a prompt where session A's brief is prepended to session B's terminal output. With the V5 system prompt alone, models generate Swift suggestions for Rust problems ~9% of the time. With the brief anchoring, the project identity is explicit in the first 200 tokens — the model's attention is locked to the correct project before it reads a single line of terminal output.

**Predicted reduction**: 9% -> 1%. The remaining 1% is genuine ambiguity where two projects happen to have identical error patterns.

### Failure Mode 3: Repetition (13%)

**Test scenario**: The session is stuck. Claude has tried 3 different approaches to fix a deadlock in an async queue. Each approach failed. The twin keeps generating "try using a DispatchQueue.sync" variant.

**V6 response**:
- Anti-Repeat Fence lists the last 15 prompts as forbidden n-grams.
- "try using a DispatchQueue.sync" shares 3-grams with "use DispatchQueue.sync" from turn 12.
- Jaccard overlap > 0.4 triggers substitution.
- Phase = BUILD. Escape prompt rotated from BUILD pool: "walk me through the current error in detail"

But more importantly, the Digest says: "DispatchQueue approaches failed. DispatchGroup deadlocked. Actor isolation tried and reverted." The task graph shows the active node as BLOCKED with a blocker annotation.

The twin, reading the digest, now knows "all queue-based approaches failed" and is more likely to suggest a fundamentally different strategy (actor model, AsyncStream, semaphore) rather than rehashing the same approach.

**Predicted reduction**: 13% -> 3%. The remaining 3% is cases where the BLOCKED state is not correctly inferred, leaving the model to guess — and occasionally guess the same wrong answer twice before the fence catches it.

---

## V6 Implementation Plan

### File Structure

```
karl/
  v6/
    __init__.py
    driver.py             # Main entrypoint, replaces twin_session_driver.py
    session_brief.py      # SessionBrief dataclass + phase logic
    task_graph.py         # TaskNode dataclass + decomposer + updater
    digest_engine.py      # TurnRecord + rolling buffer + compression
    anti_repeat.py        # 3-gram hash ring + escape prompt rotation
    phase_prompts.py      # 3 phase-aware system prompt strings
    terminal_parser.py    # TerminalState + LineCountStabilizer + adaptive lines
    session_state.py      # JSON persistence layer
    context_stack.py      # Assembles all blocks into the final prompt
    session_exporter.py   # Exports session as KARL SFT training data
```

### Key Differences from V5

| Aspect | V5 | V6 |
|--------|----|-----|
| System prompt | Static, generic | Phase-aware (3 variants) |
| Context given | Raw 80 lines | 7-block context stack (1848 tokens) |
| Project identity | None | SESSION BRIEF anchors every prompt |
| Turn history | None | Digest (compressed rolling, bounded 150 tokens) |
| Task awareness | None | TaskGraph with status tracking |
| Repetition guard | None | 3-gram hash ring, escape rotation |
| State persistence | None (in-memory only) | JSON-on-disk, survives restarts |
| Terminal detection | Keyword matching | LineCountStabilizer (structural) |
| Training output | None | Auto-exports successful sessions to KARL |
| Session ID | Ephemeral | Derived hash, consistent |

### CLI Interface

```bash
# Basic usage (mirrors V5 API)
python -m karl.v6.driver mac1 0:1.2 \
  --goal "fix RecordingService crash on stopRecording" \
  --project MotionMix \
  --turns 30 \
  --interval 30

# Observe mode (no decomposition, infer tasks from behavior)
python -m karl.v6.driver mac5 0:0.1 \
  --goal "continue current session" \
  --observe-tasks \
  --turns 20

# Resume an existing session
python -m karl.v6.driver mac1 0:1.2 \
  --resume session_id_hash

# Dry run (print context stack without injecting)
python -m karl.v6.driver mac1 0:1.2 \
  --goal "..." --dry-run --turns 5
```

### Token Budget Verification

```
Block               | Max tokens | Reason
--------------------|-----------|---------------------------
SYSTEM prompt       | 300       | Phase-aware instructions
SESSION BRIEF       | 200       | Project + goal + turn + phase
TASK GRAPH          | 150       | Active + 2 pending nodes
DIGEST              | 150       | Last 5 turns compressed
ANTI-REPEAT FENCE   | 100       | 15 prompts as forbidden phrases
TERMINAL CONTEXT    | 900       | Adaptive: 20-120 lines
USER TURN           | 48        | "What does Mohamed type next?"
--------------------|-----------|
TOTAL INPUT         | 1848      |
GENERATION BUDGET   | 200       | Max tokens for output
--------------------|-----------|
WINDOW TOTAL        | 2048      | Fits the constraint exactly
```

---

## What V6 Does Not Solve

**Multi-session parallelism coordination**: Two V6 drivers on the same project (e.g., one fixing bugs, one writing tests) will not coordinate. They each have their own state.json and will not see each other's injections. This is a V7 problem (cross-session NATS pub/sub).

**Goal drift**: If Claude interprets the goal differently than the seed and starts building something adjacent, the TaskGraph will not update correctly because the tasks were decomposed from the original seed. The driver will keep pushing toward the original tasks while Claude is doing something different. Detection: add a goal-drift check in Terminal Parser (compare file paths being modified against project_path expectation).

**Long-running blocked tasks**: If a task stays ACTIVE for more than 10 turns with `error_detected=True` consistently, the driver should escalate. V6 logs this but does not auto-escalate. Add a `--escalate-on-stuck N` flag that injects a hard-coded "you've been stuck on this for N turns, summarize blockers and suggest a different approach" after N turns of detected stuckness.

---

## Training Data Flywheel

The V6 session exporter closes the loop with KARL:

```
V6 Session
  ├── Turn log (turns.jsonl) → success/failure labels
  ├── Digest (digest.json)   → compressed context
  └── Task graph (state.json) → structured supervision

session_exporter.py
  ├── Selects turns where: prompt led to task completion
  ├── Formats as: system + context_stack → twin_prompt
  └── Writes to: ~/.claude/karl/trajectories.jsonl

KARL reward engine
  ├── Scores the trajectory: task_completion=1.0 is a strong outcome signal
  └── Adds to next SFT export for V7 training

V7 training data
  ├── Input: context stacks (structured, bounded)
  └── Output: correct single-sentence directives
  → A model trained on this no longer needs the scaffolding as a crutch
```

By turn 15, V5 degrades because it has no scaffolding. V6 never degrades because the scaffolding is always present. The V6 sessions generate clean, context-grounded training data. V7 is trained on those sessions and internalizes the scaffolding — it no longer needs the external injection because the structure became part of the model itself.

That is the long-term goal: the scaffolding trains itself away.

---

*Document generated: 2026-04-01*
*Next step: implement `karl/v6/` starting with `session_brief.py`, `task_graph.py`, `context_stack.py` in that order.*
