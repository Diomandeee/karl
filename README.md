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

The trajectory store grows over time. The best trajectories are exported as advantage-weighted SFT data and used to train a LoRA adapter via MLX.

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
2. On each prompt, embed and compare against skill centroids
3. Shadow mode: compare regex vs vector selections, log agreement and accuracy
4. Hybrid routing table: per-skill decision on vector vs regex based on measured accuracy
5. Confidence analysis: similarity thresholds, margin analysis, rejection curves
6. Per-skill routing overrides for manual control

**Promotion requirements:**
- 200+ organic shadow routing records
- Vector accuracy exceeds regex on real (non-synthetic) data
- Per-skill accuracy measured independently

## Installation

```bash
pip install -e .

# With Prefect support (for automated flows):
pip install -e ".[prefect]"

# With dev tools:
pip install -e ".[dev]"
```

## CLI

KARL ships with 67 commands. Core commands:

```bash
# Status and health
karl status                  # Trajectory store stats
karl health-digest           # Full system health report
karl dashboard               # Generate JSON dashboard
karl organic-status          # Organic data accumulation progress

# Trajectory management
karl backfill                # Backfill rewards for unscored trajectories
karl backfill --force        # Recompute all rewards
karl replay                  # Replay trajectory sequence

# Skill routing
karl analyze                 # Shadow routing analysis
karl accuracy                # Vector vs regex accuracy breakdown
karl accuracy-trend          # Accuracy over time
karl accuracy-source         # Accuracy by data source (organic vs synthetic)
karl confidence-analysis     # Similarity/margin threshold analysis
karl hybrid                  # Per-skill routing table (vector vs regex)
karl routing-override        # Manual per-skill route overrides
karl promote                 # Check promotion readiness
karl auto-promote            # Promote if thresholds met

# Embeddings and centroids
karl bootstrap               # Generate skill embeddings
karl diversity               # Centroid separation analysis
karl exemplars               # List exemplar prompts per skill
karl add-exemplar            # Add exemplar prompt
karl remove-exemplar         # Remove exemplar prompt
karl rebuild-centroid        # Rebuild centroid from exemplars
karl centroid-snapshot       # Save centroid version
karl centroid-rollback       # Restore previous centroid version

# Training pipeline
karl export                  # Export SFT training data
karl sft-preflight           # Pre-flight checks before training
karl sft-launch              # Launch remote LoRA training
karl sft-status              # Check training status
karl sft-eval                # Evaluate trained model
karl train-dryrun            # Preview training data selection

# Analysis
karl report                  # Full intelligence report
karl skill-breakdown         # Per-skill accuracy and confusion
karl confusion-resolve       # Resolve confused skill pairs
karl skill-matrix            # Skill similarity matrix
karl merge-candidates        # Find mergeable skills
karl reward-calibration      # Reward distribution analysis
karl difficulty              # Per-skill difficulty scoring
karl accuracy-forecast       # Wilson CI accuracy projections
karl lift-analysis           # Vector lift over regex by skill

# Automation
karl cron                    # Run periodic maintenance
karl auto-refresh            # Refresh centroids and trends
```

Run `karl help` for the complete list.

## Architecture

```
karl/
  # Core pipeline (4 taps)
  trajectory_tap.py          # Session buffer init, tool event append, flush, annotate
  reward_engine.py           # 3-signal composite reward scorer
  embedding_cache.py         # Prompt cache + skill embeddings + centroid management
  trajectory_bridge.py       # Shadow analysis, promotion, hybrid routing, analytics

  # CLI and automation
  karl_cli.py                # 67-command CLI
  karl_cron.py               # Periodic maintenance jobs
  config.json                # Runtime configuration

  # Training pipeline
  sft_exporter.py            # Advantage-weighted SFT export (OAPL-Lite)
  sft_dispatch.py            # Training job dispatch
  sft_launcher.py            # Remote LoRA training via SSH + MLX
  karl_trainer.py            # Training orchestration
  synthetic_qa.py            # Synthetic Q&A from git diffs
  synthetic_qa_generator.py  # Extended synthetic generation
  agentic_synth.py           # Agentic synthesis patterns
  rl2f_data_builder.py       # RL-to-fine-tune data preparation

  # Analysis and evaluation
  evaluator.py               # Model evaluation
  trajectory_diversity.py    # Trajectory diversity metrics
  trajectory_filter.py       # Quality filtering
  trajectory_extractor.py    # Historical trajectory backfill
  process_fingerprint.py     # Session fingerprinting
  flow_sampler.py            # Sampling strategies
  plasticity_manager.py      # Centroid plasticity control

  # Integration
  cortex_karl_bridge.py      # Bridge to Cortex rule system
  shadow_seeder.py           # Shadow record seeding (deprecated)
  bootstrap_skill_embeddings.py  # Initial embedding generation
  weight_updater.py          # EMA weight updates
  metrics_exporter.py        # Prometheus metrics export
  generate_status.py         # Dashboard status generation
  notifications.py           # Discord webhook notifications

  # Supporting
  config.py                  # Environment-based configuration
  types.py                   # Type definitions
  bootstrap.py               # Skill embedding bootstrap
  extractor.py               # Trajectory extraction
  entity_bridge.py           # Entity resolution
```

### Data Files (generated at runtime)

| File | Purpose |
|------|---------|
| `trajectories.jsonl` | Append-only trajectory store |
| `routing_shadow.jsonl` | Shadow routing comparison log |
| `accuracy_trend.jsonl` | Accuracy metrics over time |
| `skill_evolution.jsonl` | Skill accuracy changes |
| `promotion_log.jsonl` | Promotion decision history |
| `exemplar_registry.jsonl` | Curated exemplar prompts |
| `train.jsonl` / `valid.jsonl` | SFT training splits |
| `eval-holdout.jsonl` | Held-out evaluation set |
| `karl-sft.jsonl` | Formatted SFT training data |
| `config.json` | Runtime routing configuration |
| `karl_dashboard.json` | Dashboard status |

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

# Canonical writer (multi-node setups)
export KARL_CANONICAL_WRITER=claude-code-vm
```

See `karl/config.py` for the complete list.

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
pytest                                          # 164 tests
pytest --cov=karl --cov-report=term-missing
```

## License

MIT
