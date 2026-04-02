# KARL V7 Cognitive Twin — Full Handoff Document

**Author**: Claude Opus 4.6 (session f2129eae)
**Date**: 2026-04-02
**For**: Codex (continuation agent)
**Project**: ~/Desktop/karl/

---

## 1. WHAT WAS BUILT

KARL V7 is a recursive training data factory that generates Mohamed-authentic prompts, injects them into live Claude Code sessions, captures all interactions as training data, and trains a 4B cognitive twin adapter on Vast.ai.

### Architecture Overview

```
GPT-OSS 120B (Together AI)     →  Generates Mohamed-style prompts
  ↓                                  (multi-chain: 50 exemplars + 20 corrections + 24K knowledge doc)
V7.2 Driver                    →  Injects prompts into tmux panes via load-buffer/paste-buffer
  ↓
Claude Code Sessions           →  Build real projects (Rust, Python, Next.js, SwiftUI)
  ↓
Session Logs (JSONL)           →  Every turn captured with strategy, style score, phase
  ↓
Curation Pipeline              →  Filters: style >= 0.4, no dedup, no terminal garbage
  ↓
Training Data                  →  SFT pairs + DPO preference pairs
  ↓
Vast.ai RTX 4090               →  QLoRA fine-tuning on Qwen3-4B-Instruct-2507
  ↓
Mac5 MLX Server (:8100)       →  Local inference (free, ~50ms)
```

### The Recursive Loop (Goal)

```
Phase 1: GPT-OSS 120B generates prompts → sessions → training data
Phase 2: Train 4B adapter on Vast.ai
Phase 3: Deploy adapter to Mac5
Phase 4: 4B adapter replaces GPT-OSS 120B as prompt generator
Phase 5: New sessions → new training data → retrain → V7.1, V7.2...
Exit: When 4B model matches GPT-OSS 120B quality (style >= 0.95 + correct infra references)
```

---

## 2. CURRENT STATE

### What's Running RIGHT NOW

- **Factory PID**: Running in background (started ~2026-04-02T10:50 UTC)
- **Progress**: 31 session logs, 1,114 clean training pairs (avg style 0.99)
- **Batch**: Mid-run, 55 remaining projects of 60 total, 11 batches of 5 panes each
- **Panes**: agent-claude2, agent-codex, agent-gemini (Mac1) + claude (Mac2) + claude (Mac4)
- **Together AI**: GPT-OSS 120B, key `8a755cb3637f39c6a397caffa33c71b8ac4d98cbe697914fcf7de0a4f413ca84`
- **Monitor**: `tail -f ~/Desktop/karl/v7-factory-logs/master.log`
- **Cost so far**: ~$4 of $86 budget

### V7 SFT Adapter (TRAINED, NOT YET EVALUATED)

- **Location**: `~/Desktop/karl/vastai/karl-v7-sft-adapter/` (PEFT format, 150MB)
- **MLX converted**: `~/Desktop/karl/vastai/karl-v7-mlx-adapter/` (126MB)
- **Deployed to Mac5**: `mac5:~/models/karl-v7-adapter/`
- **Training stats**: 1500 steps, 20.6 min on RTX 4090, eval NLL 2.255, commitment acc 99.5%
- **NOT YET SERVING**: MLX server on Mac5 has not been restarted with the V7 adapter

### Data Artifacts

| File | Location | Size | Description |
|------|----------|------|-------------|
| `v7-training-data.jsonl` | `~/Desktop/karl/` | 1,114 lines | Curated factory pairs (auto-updated) |
| `v7-dpo-pairs.jsonl` | `~/Desktop/karl/` | 173 lines | DPO preference pairs from trajectory + factory |
| `v7-prompt-corpus.jsonl` | `~/Desktop/karl/` | 4,587 lines | All Mohamed prompts indexed with TF-IDF |
| `v7-knowledge-injection.md` | `~/Desktop/karl/` | 24K chars | Full system knowledge for roleplay injection |
| `v7-mohamed-exemplars.json` | `~/Desktop/karl/` | 50 entries | Cached diverse exemplar prompts |
| `v7-model-benchmark.json` | `~/Desktop/karl/` | 10 models | Benchmark results (GPT-OSS 120B won) |
| `trajectories.jsonl` | `~/Desktop/karl/karl/` | 3,190 lines | V4 trajectory store (6-signal reward scored) |
| `v6-correction-pairs.jsonl` | `~/Desktop/karl/` | 1,766 lines | Mined correction pairs with failure modes |
| `v6-rich-prompts.jsonl` | `~/Desktop/karl/` | 8 lines | Rich prompts from correction miner |
| Session logs | `~/Desktop/karl/v7-session-logs/` | 31 files | Raw per-session JSONL with turn-by-turn data |
| Driver logs | `~/Desktop/karl/v7-factory-logs/` | ~20 files | stdout/stderr from each V7.2 driver instance |
| V7 SFT adapter | `~/Desktop/karl/vastai/karl-v7-sft-adapter/` | 150MB | PEFT LoRA weights + anticipation modules |
| V7 MLX adapter | `~/Desktop/karl/vastai/karl-v7-mlx-adapter/` | 126MB | Converted for Mac5 MLX serving |

---

## 3. CODE INVENTORY

All V7 code lives in `~/Desktop/karl/karl/v7/` (3,299 lines across 11 files):

### Core Pipeline

| File | Lines | Purpose |
|------|-------|---------|
| `prompt_corpus.py` | 170 | Extracts 4,587 prompts from 534 Claude sessions, TF-IDF search index |
| `style_validator.py` | 174 | Scores prompts against Mohamed's DNA (signature phrases, AI slop detection, imperative ratio) |
| `session_context_reader.py` | 198 | Parses raw tmux pane output into structured SessionContext (files, errors, tools, last action) |
| `mohamed_simulator.py` | 531 | 4-strategy prompt generator: agent_roleplay → context_compose → rag_nearest → goal_compose |
| `agent_roleplay.py` | 491 | Multi-chain LLM roleplay (8-turn chain: system + exemplars + corrections + knowledge + session) |
| `driver.py` | 378 | Session driver: reads pane, generates prompt, validates, injects, logs. Replaces V6's 4B MLX twin |
| `full_factory.py` | 678 | Batch orchestrator: 60 projects, pane discovery, session spawning, driver management, auto-curation |
| `curate_training_data.py` | 183 | Quality filter: style score >= 0.6, no terminal garbage, no dedup escapes. Exports clean JSONL |
| `dpo_generator.py` | 247 | Generates DPO preference pairs from trajectory reward deltas (top vs bottom quartile per skill) |
| `model_benchmark.py` | 248 | Benchmarks LLM backends for roleplay quality (tested 10 models, GPT-OSS 120B won) |
| `__init__.py` | 1 | Package marker |

### V6 Dependencies (still used by V7)

| File | Location | Purpose |
|------|----------|---------|
| `session_state.py` | `karl/v6/` | JSON-on-disk session persistence (~/.karl-sessions/) |
| `project_context.py` | `karl/v6/` | Filesystem-based project detection (language, framework, name) |
| `terminal_parser.py` | `karl/v6/` | Parses tmux capture-pane output for is_alive, is_working, content_hash |
| `anti_repeat.py` | `karl/v6/` | 3-gram Jaccard dedup + escape prompt rotation (10 per phase + STUCK) |
| `context_stack.py` | `karl/v6/` | 7-block context assembly for the old MLX twin (still used for state tracking) |
| `correction_miner.py` | `karl/v6/factory/` | Mines 1,766 correction pairs from 534 sessions using signature phrase detection |
| `session_spawner.py` | `karl/v6/factory/` | Session spawning via tmux respawn-pane + load-buffer injection |

### Reward Engine (modified)

| File | Location | Change |
|------|----------|--------|
| `reward_engine.py` | `karl/` | Added 6th signal: `_compute_wasted_motion()` (tool retries, error loops, read waste, undo patterns). Weights rebalanced to 6-signal: outcome 0.25, process 0.22, efficiency 0.13, verification 0.13, consistency 0.13, motion 0.14. |

### Training Pipeline

| File | Location | Purpose |
|------|----------|---------|
| `prepare_v7_data.py` | `vastai/` | Merges trajectories + factory pairs + DPO into train/eval splits |
| `karl_v7_qlora.yaml` | `vastai/configs/` | Training config: Qwen3-4B-Instruct, QLoRA r=16, inscription + gate, 1500 steps |
| `train.py` | `vastai/` | Custom training loop with anticipation modules (inscription, gate, scalar projection) |
| `run_v7.sh` | `vastai/` | Vast.ai launch script (deps install + model download + training) |

---

## 4. BUGS TO FIX

### BUG 1: STUCK Loop — Roleplay Doesn't Know Which Machine It's On

**Severity**: HIGH — wastes 30+ turns per affected session
**File**: `karl/v7/agent_roleplay.py`, `_build_multi_chain_messages()`
**Problem**: The consulting-site session ran on Mac4 but the roleplay kept generating "SSH into Mac1 and scaffold." The multi-chain prompt says "Mac1 is the orchestrator" but doesn't tell the model which machine THIS session is on.
**Evidence**: Session `20260402-114703_mac4_claude_0_0.jsonl` — 43 consecutive STUCK turns all saying "SSH into Mac1."
**Fix**: Add the current machine and pane ID to the session context block in `_build_multi_chain_messages()`:
```python
# In the user_3 message, add:
Machine: {ctx.project_dir.split('/')[0] if '/' in ctx.project_dir else 'mac1'} (THIS session runs here)
Pane: {pane_id}
DO NOT suggest SSHing to another machine. You ARE on this machine.
```
**Also**: The driver needs to pass `machine` and `pane_id` through to `generate_as_mohamed()`, which currently doesn't receive them.

### BUG 2: STUCK Phase Never Escapes

**Severity**: MEDIUM — sessions stay STUCK indefinitely
**File**: `karl/v6/session_state.py`, `update_phase()`
**Problem**: Phase transitions from STUCK back to BUILD never happen. The phase detector checks `is_stuck()` which looks at consecutive same-hash pane outputs, but once STUCK, there's no mechanism to exit STUCK even if the pane content changes.
**Fix**: In `driver.py`, after injecting a prompt, if the pane hash changes on the NEXT turn, transition back to BUILD:
```python
if state.phase == "STUCK" and term.content_hash != state.last_pane_hash:
    state.phase = "BUILD"  # Claude is responding again
```

### BUG 3: Dedup Doesn't Catch Semantic Repetition in STUCK

**Severity**: MEDIUM — 43 variations of "scaffold the Next.js app" all pass dedup
**File**: `karl/v6/anti_repeat.py`, `is_duplicate()`
**Problem**: 3-gram Jaccard threshold of 0.4 doesn't catch "I think we should SSH into Mac1 and run npx create-next-app" vs "I think we should fire up Mac1 and run npx create-next-app" — different enough n-grams to pass.
**Fix**: Add a secondary check: if STUCK phase and 5+ consecutive prompts share the same top-3 keywords (e.g., "scaffold", "create-next-app", "Mac1"), force an escape:
```python
def is_stuck_loop(prompt: str, recent_prompts: list[str], threshold: int = 5) -> bool:
    """Detect semantic repetition in STUCK phase."""
    if len(recent_prompts) < threshold:
        return False
    # Extract top keywords from each recent prompt
    from collections import Counter
    all_words = Counter()
    for p in recent_prompts[-threshold:]:
        words = set(p.lower().split())
        all_words.update(words)
    # If any word appears in ALL recent prompts, it's a loop
    return any(count >= threshold for word, count in all_words.most_common(5) if len(word) > 4)
```

### BUG 4: Knowledge Injection File May Be Empty After Crash

**Severity**: LOW — factory still works but without infrastructure context
**File**: `~/Desktop/karl/v7-knowledge-injection.md`
**Current state**: 0 bytes (lost in crash)
**Fix**: Re-run the knowledge extraction:
```bash
# The knowledge was extracted from the research agent output
# Re-extract from the agent output file if it still exists:
python3 -c "
content = open('/private/tmp/claude-501/-Users-mohameddiomande/8d1a60e8-6be6-4949-a1dd-7de7ad096487/tasks/a9e8d735714e7f13c.output').read()
start = content.find('# MOHAMED\'S SYSTEM KNOWLEDGE')
end = content.find('**Injection ready for GPT-OSS 120B')
if start > 0 and end > 0:
    open('v7-knowledge-injection.md', 'w').write(content[start:end].strip())
"
```
If that tmp file is gone, re-run the Explore agent that gathered SOUL.md, AGENTS.md, CLAUDE.md, and all 42 memory topic files. The knowledge doc covers 8 sections: Infrastructure, Architecture Decisions, Rules & Preferences, Personality & Voice, Project Knowledge, Critical Gotchas, Skill Routing, and Training Data Summary.

### BUG 5: Exemplar Cache May Be Empty After Crash

**Severity**: LOW — auto-rebuilds on next run but takes 10s
**File**: `~/Desktop/karl/v7-mohamed-exemplars.json`
**Current state**: 0 bytes
**Fix**: Delete the file and let `_get_exemplars()` rebuild it:
```bash
rm ~/Desktop/karl/v7-mohamed-exemplars.json
# Next factory run auto-rebuilds from the corpus
```

---

## 5. WHAT NEEDS TO HAPPEN NEXT

### Immediate (while factory is still running)

1. **Fix Bug 1** (machine context in roleplay) — prevents future STUCK loops
2. **Fix Bug 3** (semantic dedup) — catches repetitive STUCK prompts
3. **Rebuild knowledge injection file** (Bug 4) — restores infrastructure context
4. **Rebuild exemplar cache** (Bug 5)

### After Factory Completes (~8 hours from start)

5. **Run curation on all session logs**: `cd ~/Desktop/karl && python3 -m karl.v7.curate_training_data`
   - Expected output: ~2,500+ clean training pairs in `v7-factory-full.jsonl`
   - Filter the STUCK loops (sessions where 10+ consecutive prompts share keywords)

6. **Merge with existing training data**:
   ```bash
   cd ~/Desktop/karl && python3 vastai/prepare_v7_data.py
   ```
   This merges: V4 trajectories (2,463 SFT) + factory pairs (2,500+) + DPO pairs (173)

7. **Train V7.1 on Vast.ai** (with the larger dataset):
   ```bash
   # Create RTX 4090 instance ($0.28/hr)
   vastai create instance <OFFER_ID> --image pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime --disk 100 --label karl-v7.1-sft

   # Upload data + code, launch training
   # Use the same pipeline as V7 but with updated data files
   ```

8. **Deploy V7.1 adapter to Mac5**:
   ```bash
   # Convert PEFT → MLX (same process as V7)
   # Upload to mac5:~/models/karl-v7.1-adapter/
   # Restart MLX server with new adapter
   ssh mac5 "pkill -f mlx_lm || true; sleep 2; \
     nohup python3 -m mlx_lm.server \
       --model mlx-community/Qwen3-4B-Instruct-2507-4bit \
       --adapter-path ~/models/karl-v7.1-adapter \
       --host 0.0.0.0 --port 8100 \
       > ~/Desktop/mlx-v7.1-server.log 2>&1 &"
   ```

9. **Evaluate V7.1**: Run the model_benchmark against the Mac5-served adapter to see if it matches GPT-OSS 120B quality. If style score >= 0.95, proceed to recursive loop.

10. **Recursive loop** (if V7.1 quality is sufficient):
    - Switch `agent_roleplay.py` backend from Together AI to Mac5 MLX
    - Run another factory batch with the LOCAL model generating prompts
    - Train V7.2 on the new data
    - Repeat until quality saturates

### Cost Budget

| Item | Cost | Status |
|------|------|--------|
| Together AI (factory) | ~$11 of $86 | Running |
| Vast.ai V7 training | $0.09 | Complete |
| Vast.ai V7.1 training | ~$0.15 (est) | Pending |
| Mac5 MLX serving | Free | Pending |
| **Remaining Together budget** | **~$75** | Available for future factory runs |

---

## 6. MULTI-CHAIN PROMPT ARCHITECTURE

The core innovation is the 8-turn multi-chain message array sent to GPT-OSS 120B via Together AI. This exploits the 131K context window:

```
Message 1 (system):     Mohamed identity + infrastructure map + style rules (~2K tokens)
Message 2 (user):       50 real Mohamed prompts as style exemplars (~8K tokens)
Message 3 (assistant):  Style pattern acknowledgment
Message 4 (user):       20 correction pairs with failure modes (~3K tokens)
Message 5 (assistant):  Correction awareness internalization
Message 6 (user):       24K knowledge doc (infrastructure, decisions, gotchas) (~6K tokens)
Message 7 (assistant):  Knowledge internalization acknowledgment
Message 8 (user):       Current session state + plan + previous prompts + "generate" (~1K tokens)
```

Total: ~20K tokens per call. At GPT-OSS 120B rates ($0.15/$0.60 per 1M), that's $0.004 per prompt.

The chain teaches the model:
1. WHO Mohamed is (identity, voice, signature phrases)
2. HOW Mohamed prompts (50 real examples across domains)
3. WHAT Mohamed corrects (failure mode awareness)
4. WHAT Mohamed knows (infrastructure, ports, machines, services)
5. WHERE Mohamed is right now (session context, files, errors, plan)

---

## 7. MODEL BENCHMARK RESULTS

Tested 10 models across 5 session contexts. All models received the same multi-chain prompt.

| Model | Score | Pass | Latency | $/1M in | $/1M out | Provider |
|-------|-------|------|---------|---------|----------|----------|
| MiniMax M2.5 | 1.00 | 100% | 7.0s | $0.30 | $1.20 | Together |
| GPT-OSS 120B | 1.00 | 100% | 5.5s | $0.15 | $0.60 | Together |
| DeepSeek R1 | 1.00 | 100% | 10.4s | $3.00 | $7.00 | Together |
| GPT-5.4-mini | 1.00 | 100% | 1.6s | $0.15 | $0.60 | OpenAI |
| GLM 4.7 | 0.98 | 100% | 5.6s | $0.45 | $2.00 | Together |
| DeepSeek V3.1 | 0.98 | 100% | 2.8s | $0.60 | $1.70 | Together |
| GPT-OSS 20B | 0.98 | 100% | 4.4s | $0.05 | $0.20 | Together |
| GLM-5 | 0.80 | 80% | 16.6s | $1.00 | $3.20 | Together |
| Kimi K2.5 | 0.39 | 40% | 21.2s | $0.50 | $2.80 | Together |
| Qwen3.5 397B | 0.00 | 0% | 25.4s | $0.60 | $3.60 | Together |

**GPT-OSS 120B selected** for factory: best balance of quality (1.00), speed (5.5s), and cost ($0.15/$0.60).

---

## 8. V7 SFT TRAINING RESULTS

Trained on Vast.ai RTX 4090 (instance 34014560, Texas, $0.28/hr). 20.6 minutes total.

| Step | Train Loss | Eval NLL | Scalar MSE | Commitment Acc |
|------|-----------|----------|------------|---------------|
| 300 | 4.63 | 3.604 | 0.463 | 36.0% |
| 600 | 4.15 | 3.259 | 0.074 | 66.5% |
| 900 | 3.56 | 2.834 | 0.046 | 96.5% |
| 1200 | 2.87 | 2.372 | 0.034 | 99.5% |
| 1500 | 2.65 | 2.255 | 0.022 | 99.5% |

- **Model**: Qwen/Qwen3-4B-Instruct-2507
- **Method**: QLoRA (r=16, alpha=32, dropout=0.05, 7 target modules)
- **Data**: 2,284 SFT examples (2,222 trajectories + 62 factory)
- **Trainable params**: 33M / 4B total (0.8%)
- **Anticipation**: inscription=True, gate=True, lse=False
- **Checkpoints**: 300, 600, 900, 1200, 1500, final

V5 comparison: V5 eval loss was 1.843 (35 examples, 500 steps). V7 eval NLL 2.255 is higher but on 65x more diverse data. V7 commitment accuracy (99.5%) beats V5 (91%).

---

## 9. REWARD ENGINE — 6-SIGNAL COMPOSITE

The reward engine (`karl/reward_engine.py`) was upgraded from 5 to 6 signals:

```
reward = 0.25 * outcome + 0.22 * process + 0.13 * efficiency
       + 0.13 * verification + 0.13 * consistency + 0.14 * motion
```

### Signal 6: Wasted Motion (`_compute_wasted_motion`)

Inspired by TPO's linearity score. Penalizes wasted actions in linear agent sessions:

- **Tool retry loops**: Same tool called 3+ times in a row (weight: 2x per loop)
- **Read waste**: Files read but never written (weight: 0.3 * total * ratio)
- **Error retry loops**: Consecutive Bash failures (weight: 1.5x per loop)
- **Undo patterns**: Write then Edit same file immediately (weight: 1x per undo)

Formula: `score = exp(-0.15 * waste_count)`

Test results:
- Clean trajectory (read → edit → test → read → edit → build): motion = 1.00
- Waste trajectory (4 reads, 3 bash failures, 1 undo): motion = 0.185

---

## 10. TPO/IRCP/RCP ANALYSIS (from deep dive)

Three trajectory intelligence libraries were analyzed for extractable patterns:

- **TPO** (Topology Preference Optimization): Complete algorithm, never deployed. Extracted: linearity score (now wasted motion signal), preference pair generation (now DPO generator).
- **IRCP** (Inverse Ring Contextual Propagation): Complete Rust + Python, never deployed. Extractable: dual-signal scoring (reward × embedding similarity) for training example selection.
- **RCP** (Ring Contextual Propagation): Vaporware — README exists, code never written.

**Recommendation**: Do NOT resurrect the full stack. Cherry-pick specific algorithms as needed. The wasted motion signal and DPO generator already extract the most valuable patterns.

Source files:
- TPO: `~/Desktop/Comp-Core/backend/cc-trajectory/legacy/cc-tpo-original/cc-tpo/packages/tpo/`
- IRCP: `~/Desktop/Comp-Core/core/retrieval/cc-rag-plus-plus/crates/core/src/trajectory/ircp.rs`
- RCP: `~/Desktop/Comp-Core/core/_recovered/retrieval/cc-rag-plus-plus/packages/rcp/README.md`

---

## 11. ENVIRONMENT & KEYS

| Variable | Value | Purpose |
|----------|-------|---------|
| `TOGETHER_API_KEY` | `8a755cb3637f39c6a397caffa33c71b8ac4d98cbe697914fcf7de0a4f413ca84` | Together AI ($86 credits, GPT-OSS 120B) |
| `OPENAI_API_KEY` | In env | GPT-5.4-mini (backup backend) |
| `KARL_ROLEPLAY_BACKEND` | `together` | Which LLM backend for roleplay |
| `KARL_TOGETHER_MODEL` | `openai/gpt-oss-120b` | Model ID on Together |

Vast.ai SSH key: `~/.ssh/id_vastai` (must use `-i` flag explicitly)

**Cloudflare gotcha**: Together AI blocks Python `urllib` requests (no User-Agent). The `agent_roleplay.py` uses `subprocess.run(["curl", ...])` to bypass this. Do NOT switch back to urllib.

---

## 12. QUICK COMMANDS

```bash
# Check factory status
ps aux | grep "full_factory\|karl.v7.driver" | grep -v grep | wc -l
tail -f ~/Desktop/karl/v7-factory-logs/master.log

# Run curation
cd ~/Desktop/karl && python3 -m karl.v7.curate_training_data

# Test roleplay (single prompt)
cd ~/Desktop/karl && TOGETHER_API_KEY=8a755cb3637f39c6a397caffa33c71b8ac4d98cbe697914fcf7de0a4f413ca84 \
  python3 -m karl.v7.agent_roleplay

# Run model benchmark
cd ~/Desktop/karl && python3 -m karl.v7.model_benchmark --provider both \
  --api-key 8a755cb3637f39c6a397caffa33c71b8ac4d98cbe697914fcf7de0a4f413ca84

# Prepare training data
cd ~/Desktop/karl && python3 vastai/prepare_v7_data.py

# Generate DPO pairs
cd ~/Desktop/karl && python3 -m karl.v7.dpo_generator

# Full factory (dry run)
cd ~/Desktop/karl && TOGETHER_API_KEY=8a755cb3637f39c6a397caffa33c71b8ac4d98cbe697914fcf7de0a4f413ca84 \
  python3 -m karl.v7.full_factory --dry-run

# Full factory (live)
cd ~/Desktop/karl && TOGETHER_API_KEY=8a755cb3637f39c6a397caffa33c71b8ac4d98cbe697914fcf7de0a4f413ca84 \
  python3 -m karl.v7.full_factory --turns 50 --batches 12 --interval 45

# Vast.ai: create instance
vastai search offers 'gpu_name=RTX_4090 num_gpus=1 disk_space>=100 dph<=0.40 reliability>0.98' --order 'dph' --limit 5
vastai create instance <ID> --image pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime --disk 100 --label karl-v7.1-sft
```

---

## 13. SESSION EVOLUTION (what happened in this session)

### V7.0 → V7.1 → V7.2 evolution

1. **V7.0** (templates): `template_compose` generated hollow prompts ("build the next feature"). Batch 1: 79% pass, avg 0.91, 18% dedup.
2. **V7.1** (context-dense templates): Added infrastructure context, file/error references, archetype vocabulary. 10/10 pass, avg 0.97.
3. **V7.2** (agent roleplay): GPT-5.4-mini roleplays as Mohamed with 50 exemplars + corrections. 10/10 pass, avg 0.99. Then switched to GPT-OSS 120B on Together AI (same quality, $86 budget).
4. **V7.2+knowledge**: Added 24K knowledge injection document (SOUL, AGENTS, CLAUDE.md, 42 memory files). Model now references specific ports, machines, services naturally.

### Key decisions made

- **4B MLX twin killed**: V6 used Qwen3-4B on Mac5 to generate prompts. V7 replaced it with GPT-OSS 120B (or local template fallback). The 4B model couldn't hold context or generate diverse prompts.
- **DPO added**: V7 generates DPO preference pairs from trajectory reward deltas (top vs bottom quartile per skill domain). 173 pairs from 3,190 trajectories.
- **6th reward signal**: Wasted motion score (tool retries, error loops, read waste) added to reward engine. Inspired by TPO linearity but adapted for linear sessions.
- **Together AI over OpenAI**: $86 credits, GPT-OSS 120B at $0.15/$0.60 beats GPT-5.4-mini on cost. Both score 1.00.
- **curl over urllib**: Cloudflare blocks Python urllib on Together AI. All API calls use subprocess curl.
