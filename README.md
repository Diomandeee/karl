# KARL - Knowledge Agents via Reinforcement Learning

Trajectory-based intelligence for AI coding agents. KARL records what an agent does during real work sessions, scores those recordings based on outcome quality, and uses the highest-scoring trajectories to improve future performance through LoRA fine-tuning and learned skill routing.

Based on the KARL paper ([arXiv 2603.05218](https://arxiv.org/abs/2603.05218)).

## How It Works

```
User Prompt
    |
    v
[Tap A] Init session buffer + shadow route (vector similarity)
    |
    v
Agent works: Read, Edit, Bash, Grep, ...
    |
    v
[Tap B] Record each tool call (name, params, success/fail)
    |
    v
Agent finishes
    |
    v
[Tap C] Flush buffer -> compute 3-signal reward -> trajectories.jsonl
    |
    v
Next prompt arrives
    |
    v
[Tap D] Detect corrections ("no, I meant...", "try again") -> annotate previous
```

The trajectory store grows over time. Weekly, the best trajectories are exported as advantage-weighted SFT data and used to train a LoRA adapter via MLX.

## Reward Engine

Three signals, composite weighted:

| Signal | Weight | Measures |
|--------|--------|----------|
| **Outcome** | 40% | Was the user satisfied? No correction, no redo, build passed, session continued |
| **Process** | 35% | Did tools work? Success rate, bash exit codes, error density |
| **Efficiency** | 25% | Was it efficient? Tool diversity (Shannon entropy), tools/minute, file touch rate |

Composite reward is `[0, 1]`. Advantage = reward - domain baseline, used for OAPL-Lite oversampling.

## Skill Routing

KARL adds a vector similarity layer alongside existing regex-based routing:

1. Pre-compute embeddings for all active skills (intent + workflow + historical prompts)
2. On each prompt, embed asynchronously and cache for the next prompt
3. Shadow mode: compare regex vs vector selections, log agreement
4. Promotion gate: activate vector routing when data thresholds are met

**Promotion requirements:**
- 100+ shadow routing records
- 50%+ embedding cache hit rate
- 80%+ agreement between regex and vector
- 5%+ reward lift on vector-only matches

## Installation

```bash
pip install -e .

# With Prefect support (for automated flows):
pip install -e ".[prefect]"

# With dev tools:
pip install -e ".[dev]"
```

## CLI

```bash
# Show trajectory store stats
karl status

# Backfill rewards for unscored trajectories
karl backfill
karl backfill --force  # Recompute all

# Export SFT training data
karl export
karl export --dry-run
karl export --min-reward 0.6

# Run full training pipeline (export -> upload -> train)
karl train
karl train --dry-run

# Shadow routing analysis + promotion check
karl analyze

# Full intelligence report
karl report
karl report --json

# Generate skill embeddings
karl bootstrap
karl bootstrap --dry-run

# Generate synthetic QA from git diffs
karl synthetic --days 14
karl synthetic --dry-run

# Show/update skill weights
karl weights
karl weights --update
karl weights --update --dry-run

# Extract trajectories from verbose logs
karl extract
karl extract --dry-run
```

## Architecture

```
karl/
  config.py              # Centralized configuration (env vars + defaults)
  trajectory_tap.py      # 4 tap points: init, append, flush, annotate
  reward_engine.py       # 3-signal composite reward scorer
  embedding_cache.py     # LRU prompt cache + skill embeddings + vector math
  weight_updater.py      # EMA weight updates from reward data
  sft_exporter.py        # Advantage-weighted SFT export (OAPL-Lite)
  synthetic_qa.py        # Synthetic Q&A from git diffs
  trajectory_bridge.py   # Shadow analysis, promotion gate, skill health
  trainer.py             # Remote LoRA training pipeline (SSH + SCP + MLX)
  bootstrap.py           # Skill embedding generation
  extractor.py           # Historical trajectory backfill
  notifications.py       # Discord webhook notifications
  cli.py                 # Command-line interface
```

### Data Files (generated at runtime)

| File | Purpose |
|------|---------|
| `trajectories.jsonl` | Append-only trajectory store |
| `routing_shadow.jsonl` | Shadow routing comparison log |
| `synthetic_qa.jsonl` | Synthetic training examples from git diffs |
| `skill_embeddings.pkl` | Pre-computed skill vectors |
| `prompt_embedding_cache.pkl` | LRU prompt embedding cache |
| `train.jsonl` / `valid.jsonl` | SFT training splits |
| `karl_status.json` | Dashboard status (for Nexus Portal) |

## Configuration

All parameters are configurable via environment variables:

```bash
# Paths
export KARL_DATA_DIR=~/.karl/data
export KARL_SKILLS_DIR=~/.claude/skills

# Embedding
export KARL_EMBED_URL=http://localhost:8000/api/rag/embed
export KARL_EMBEDDING_DIM=3072
export KARL_CACHE_MAX_ENTRIES=500

# Reward weights
export KARL_REWARD_W_OUTCOME=0.40
export KARL_REWARD_W_PROCESS=0.35
export KARL_REWARD_W_EFFICIENCY=0.25

# Training
export KARL_TRAIN_SSH_ALIAS=mac5
export KARL_MLX_MODEL=mlx-community/gemma-3-1b-it-4bit
export KARL_MLX_ITERS=500

# Notifications
export KARL_DISCORD_WEBHOOK=https://discord.com/api/webhooks/...
```

See `karl/config.py` for the complete list.

## Training Pipeline

KARL uses MLX LoRA fine-tuning on Apple Silicon:

1. **Export**: Filter trajectories by reward, compute advantage, oversample high-reward examples
2. **Synthetic augmentation**: Generate additional training data from git commit diffs
3. **Upload**: SCP training data to remote compute node
4. **Train**: MLX LoRA with configurable model, iterations, layers, learning rate
5. **Monitor**: Poll training daemon for completion and metrics

```bash
# Preview what would be exported
karl export --dry-run

# Full training cycle
karl train

# Check remote daemon health
karl train --dry-run
```

## Integration

### Hook System

Wire KARL into your agent's hook system:

```python
from karl import init_session_buffer, append_tool_event, flush_session, annotate_previous

# On session start (UserPromptSubmit)
init_session_buffer(session_id, skill_name="ops:deploy", prompt_text=prompt)

# After each tool use (PostToolUse)
append_tool_event(session_id, "Read", tool_input={"file_path": path}, success=True)

# On completion (Stop)
record = flush_session(session_id)

# On next prompt (detect corrections)
annotate_previous(session_id, correction_detected=True)
```

### Shadow Routing

```python
from karl.embedding_cache import cache_get, embed_async, rank_skills, load_skill_embeddings

# Embed prompt asynchronously (result available on next call)
embed_async(prompt_text)

# Check cache on subsequent calls
embedding = cache_get(prompt_text)
if embedding:
    skills = load_skill_embeddings()
    ranked = rank_skills(embedding, skills)
    # Compare with regex result, log to shadow file
```

## Tests

```bash
pip install -e ".[dev]"
pytest
pytest --cov=karl --cov-report=term-missing
```

## License

MIT
