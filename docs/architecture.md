# KARL Architecture

## System Overview

KARL is a trajectory-based learning system for AI coding agents. It implements a closed-loop pipeline:

```
Recording -> Scoring -> Analysis -> Training -> Improved Routing
    ^                                                |
    |                                                |
    +------------- Better trajectories <-------------+
```

## Components

### 1. Trajectory Recording (trajectory_tap.py)

Four tap points capture agent behavior:

- **Tap A** (`init_session_buffer`): Opens a JSON buffer when a session starts or a skill is activated. Captures the prompt, working directory, and skill name.

- **Tap B** (`append_tool_event`): After every tool call (Read, Edit, Write, Bash, Grep, Glob, Task), appends the tool name, key parameters, success/failure, and timing to the buffer.

- **Tap C** (`flush_session`): When the agent finishes responding, flushes the buffer to `trajectories.jsonl`. Runs the reward engine to compute a score before writing.

- **Tap D** (`annotate_previous`): On the next prompt, examines the text for correction signals ("no, I meant...", "try again"). Retroactively annotates the previous trajectory with a failure signal.

Session buffers are individual JSON files in `data/buffers/`. The trajectory store is an append-only JSONL file with file locking (fcntl) for concurrent safety.

### 2. Reward Engine (reward_engine.py)

Three-signal composite scorer:

```
reward = 0.40 * outcome + 0.35 * process + 0.25 * efficiency
```

**Outcome score** (cross-turn signals):
- No correction detected: +0.35
- No redo requested: +0.25
- Build succeeded: +0.20
- Session continued: +0.20
- Base: 0.5 when no signals available

**Process score** (within-turn quality):
- Tool success rate (45%): successes / total
- Bash cleanliness (30%): 1 - errors / bash_count
- Error density (25%): penalizes 3+ consecutive failures

**Efficiency score** (trajectory shape):
- Tool diversity (35%): Shannon entropy of tool distribution
- Duration efficiency (35%): 2-8 tools/min is ideal
- File touch rate (30%): proportion of Write/Edit operations

The **advantage** is computed as `reward - domain_baseline`, enabling OAPL-Lite oversampling: high-advantage trajectories appear up to 3x in training data.

### 3. Embedding & Routing (embedding_cache.py)

**Design constraint**: Agent hooks have a 500ms budget. Embedding API calls take 300-500ms. Solution: embed asynchronously and cache for the next prompt.

- LRU cache with 500 entries and 24-hour TTL
- Pickle persistence across sessions
- Async embed via background thread (daemon=True)
- Critical: must join the daemon thread with a timeout before process exit, or the cache never gets populated

Skill embeddings are pre-computed from SKILL.md files:
- Intent description + workflow steps + gotchas + historical trigger prompts
- Stored as {name: (vector, weight)} in `skill_embeddings.pkl`

Vector routing ranks skills by `cosine_similarity(prompt, skill) * trajectory_weight`.

### 4. Weight Updater (weight_updater.py)

Exponential moving average (EMA) updates skill routing weights:

```python
target = 0.5 + reward  # Maps [0,1] reward to [0.5, 1.5] target
new_weight = current * (1 - alpha) + target * alpha  # alpha=0.1
```

Bounds: [0.5, 1.5]. No skill is fully suppressed or dominant.

### 5. SFT Export (sft_exporter.py)

Converts trajectories to ChatML JSONL for LoRA training:

```json
{
  "messages": [
    {"role": "system", "content": "You are an expert software engineering assistant..."},
    {"role": "user", "content": "[the task prompt]"},
    {"role": "assistant", "content": "1. [ok] Read src/main.py\n2. [ok] Edit src/main.py\n..."}
  ]
}
```

Advantage-weighted sampling (OAPL-Lite):
- advantage > 0.3: 3x oversample
- advantage 0.1-0.3: 2x
- advantage 0-0.1: 1x
- advantage <= 0: exclude

Synthetic QA examples are merged after trajectory examples with deduplication.

### 6. Shadow Routing & Promotion Gate (trajectory_bridge.py)

The shadow router runs alongside regex routing, logging both selections without injecting anything. The promotion gate checks four criteria before activating vector routing:

| Check | Threshold | Rationale |
|-------|-----------|-----------|
| Min records | 100 | Statistical significance |
| Cache hit rate | 50% | Cache must be warm |
| Agreement rate | 80% | Vector shouldn't diverge wildly |
| Vector lift | 5% reward | Vector must demonstrably improve outcomes |

### 7. Training Pipeline (trainer.py)

```
Export SFT -> SCP to compute node -> MLX LoRA train -> Monitor
```

- SSH via alias (multiplexed connections, avoids auth failures)
- MLX LoRA with gemma-3-1b-it-4bit base model
- Configurable iterations, batch size, layers, learning rate
- Supports both daemon-triggered and direct training

## Data Flow

```
                                    +-----------------+
                                    |  Shadow Router  |
                                    | (vector compare)|
                                    +--------+--------+
                                             |
User Prompt -----> Tap A (init) -----> Tap B (record tools) -----> Tap C (flush + score)
                     |                       |                            |
                     v                       v                            v
               session buffer          tool events              trajectories.jsonl
                (JSON file)            (appended)                  (JSONL store)
                                                                       |
                                                              +--------+--------+
                                                              |                 |
                                                         Weight Updates    SFT Export
                                                              |                 |
                                                        skill_embeddings   train/valid.jsonl
                                                            .pkl                |
                                                                          SCP -> Mac5
                                                                                |
                                                                          MLX LoRA Train
                                                                                |
                                                                          LoRA Adapter
```
