---
name: KARL V6 Architecture
description: V6 cognitive twin session driver design — context stack approach fixing V5's status spam, cross-contamination, and repetition failures
type: project
---

V6 architecture designed 2026-04-01. Full document at `/Users/mohameddiomande/Desktop/karl/KARL_V6_ARCHITECTURE.md`.

The core insight: V5 fails because it gives a memoryless 4B model raw terminal output and asks it to infer everything. V6 externalizes all state into a 7-block context stack injected into every prompt, shrinking the model's job to "given a complete briefing, pick the next one sentence."

**Why:** V5 failure modes: 28% status spam (no project map), 9% cross-contamination (no identity anchor), 13% repetition (no turn history). All three are scaffolding failures, not model failures.

**How to apply:** V6 file structure is `karl/v6/` with 9 files. Build order: `session_brief.py` -> `task_graph.py` -> `context_stack.py` -> `driver.py`. Token budget fits exactly in 2048: 1848 input + 200 generation.

## Key Design Decisions

- 7-block context stack (system 300t + brief 200t + task graph 150t + digest 150t + anti-repeat 100t + terminal 900t + user 48t = 1848t)
- 3 phases: EXPLORE (0-30%), BUILD (31-80%), CLOSE (81-100%) with different system prompts per phase
- Anti-repeat uses 3-gram Jaccard overlap (>0.4 threshold), not embeddings — avoids latency
- Terminal busy detection via LineCountStabilizer (structural) not keyword matching (brittle)
- Session state JSON on disk survives restarts: `~/.karl/v6/sessions/{session_id}/`
- V6 sessions auto-export to KARL trajectories.jsonl as training data for V7

## Predicted Failure Mode Reductions
- Status spam: 28% -> 4%
- Cross-contamination: 9% -> 1%
- Repetition: 13% -> 3%
