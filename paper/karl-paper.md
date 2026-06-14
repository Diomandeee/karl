# Trajectory Memory Ledger

## Schema-Normalized Experience Replay for Self-Improving Coding Agents

**Mohamed Diomande**


June 2026

---

## Abstract

We present the Trajectory Memory Ledger, implemented in KARL, a schema-normalized experience replay system for improving AI coding agent performance through closed-loop feedback. The ledger records complete tool-use sequences during real coding sessions, normalizes them into an append-only schema, scores them using a six-signal composite reward function (outcome, process, efficiency, verification, consistency, and wasted motion), and uses the highest-scoring trajectories to generate advantage-weighted supervised fine-tuning data. Unlike approaches that rely on static benchmarks or human preference labels, the Trajectory Memory Ledger derives training signal from observable agent behavior and implicit user feedback. The current normalized deployment corpus contains 7,468 scored trajectories, 67,409 observed tool events, and 73,470 recovered tool steps across 50+ active projects. From this store, KARL exports 3,678 ChatML training examples (3,310 train / 368 validation). We describe the system architecture, schema normalization, reward design, OAPL-Lite export, and entity bridge for performance-based skill decay.

---

## 1. Introduction

AI coding agents powered by large language models have reached the point where they routinely perform complex multi-step software engineering tasks: reading files, editing code, running tests, deploying services, managing infrastructure. The quality of these interactions varies widely. Some sessions produce clean, efficient outcomes. Others spiral through repeated failures, unnecessary tool calls, and incorrect approaches that require user correction.

This variance is not random. It correlates with the nature of the task, the skill being applied, the agent's routing decision, and patterns in the tool-use sequence itself. Yet most agent frameworks treat each session as independent, learning nothing from the trajectory of past interactions.

The Trajectory Memory Ledger addresses this gap with three contributions:

1. **Trajectory Recording**: A four-tap instrumentation system that captures complete tool-use sequences with timing, parameters, success/failure signals, and cross-turn outcome annotations, all within a 500ms hook budget.

2. **Six-Signal Reward Engine**: A composite scoring function that evaluates trajectories on outcome quality, process quality, efficiency, verification, consistency, and wasted motion, without requiring explicit human labels.

3. **Advantage-Weighted Training Pipeline**: An OAPL-Lite approach that oversamples high-advantage trajectories up to 3x for LoRA fine-tuning, combined with a shadow routing system that learns when vector-based skill selection outperforms regex matching.

KARL is the lightweight Python implementation of the ledger. It integrates with any agent framework supporting hook events and has been deployed across a five-machine mesh orchestrating 80+ operational skills, with batch backfill and live flow capture feeding one normalized trajectory store.

## 2. Related Work

**Agent Benchmarks.** SWE-Bench (Jimenez et al., 2024) and similar benchmarks evaluate agent coding ability on curated tasks with known solutions. These provide point-in-time measurements but not continuous improvement signals. The Trajectory Memory Ledger complements benchmarks by learning from the distribution of real tasks the agent actually encounters.

**RLHF and Preference Learning.** Reinforcement Learning from Human Feedback (Ouyang et al., 2022) has become standard for language model alignment. Constitutional AI (Bai et al., 2022) reduces annotation burden through model self-critique. The ledger avoids both by deriving reward from observable behavior: tool success rates, user corrections, and session outcomes require no explicit human labeling.

**Process Reward Models.** Lightman et al. (2023) showed that rewarding correct reasoning steps, not just final answers, improves mathematical problem-solving. The ledger's process score operates on the same principle: a trajectory where 10/10 tools succeed is scored higher than one where 7/10 succeed even if both reach the same outcome.

**Agent Training.** Databricks' Agent Training work (2025) introduced the concept of trajectory-based learning for coding agents, capturing tool-use sequences and using them for fine-tuning. The Trajectory Memory Ledger extends this with explicit advantage weighting, entity-level performance tracking, and an automated pipeline from recording through training.

**Skill Routing.** Mixture of Experts (Shazeer et al., 2017) and more recent router-based architectures select among specialized models. The ledger's routing is simpler but analogous: given a prompt, select which operational skill (a structured SKILL.md document with workflow steps, gotchas, and trigger patterns) best applies. The innovation is learning routing from trajectory outcomes rather than static heuristics.

## 3. System Architecture

### 3.1 Overview

KARL implements the Trajectory Memory Ledger as a closed-loop pipeline:

```
Recording -> Scoring -> Analysis -> Training -> Improved Routing
    ^                                                |
    |                                                |
    +------------- Better trajectories <-------------+
```

The system consists of eight components, each independently deployable:

1. **Trajectory Tap** (recording)
2. **Rust Ledger Daemon** (durable ingestion, schema normalization, cursor state, locked append, metrics)
3. **Reward Engine** (scoring)
4. **Embedding Cache** (routing infrastructure)
5. **Weight Updater** (routing optimization)
6. **SFT Exporter** (training data generation)
7. **Entity Bridge** (skill entity intelligence)
8. **Trainer** (remote LoRA fine-tuning)

### 3.2 Trajectory Recording

KARL instruments the agent's hook system at four points:

**Tap A (init_session_buffer):** Fires on `UserPromptSubmit`. Creates a JSON buffer file with the session ID, skill name, prompt text (truncated to 500 characters), working directory, and git repository context. Buffer files live in `data/buffers/` with sanitized session IDs as filenames.

**Tap B (append_tool_event):** Fires on `PostToolUse`. Appends a compact event record to the session buffer: tool name, key parameters (file paths, commands, patterns -- truncated to 200 characters), success/failure flag, exit code for Bash commands, and timestamp. A session buffer can accumulate up to 50 events before capping.

**Tap C (flush_session):** Fires on `Stop`. Reads the buffer, computes summary statistics (tool counts, success rate, bash error count, duration), runs the reward engine, and appends the complete trajectory record to `trajectories.jsonl` with file-level locking (fcntl). The buffer file is then deleted.

**Tap D (annotate_previous):** Fires on the next `UserPromptSubmit`. Examines the new prompt for correction signals. If detected, walks the trajectory store backwards to find the previous record for this session and annotates it with `correction_detected: true`. This retroactive annotation is the key signal that turns implicit user dissatisfaction into explicit training data.

The entire tap pipeline operates within a 500ms hook budget enforced by SIGALRM. No tap blocks the agent's response.

Live mesh-orchestration flow events are handled by `trajectory-ledgerd`, a Rust daemon that tails dated gateway event files, tracks cursor state by `(date, seq)`, reduces completed flows into schema-v2 trajectory cards, scores them at emit time, appends them to the ledger with file locking, and writes Prometheus-format metrics. This moves the failure-prone live collection path into a typed single binary while preserving Python for research, export, and training workflows.

### 3.3 Trajectory Record Format

Each schema-v2 trajectory record contains:

```json
{
  "schema_version": 2,
  "id": "traj_{session_prefix}_{unix_timestamp}",
  "session_id": "uuid",
  "source": "verbose-all",
  "channel": "live",
  "domain": "ops",
  "recorded_at": "ISO-8601",
  "skill": {"name": "ops:deploy", "domain": "ops"},
  "context": {
    "prompt_text": "deploy the discord bot...",
    "cwd": "/Users/dev/projects/bot",
    "git_repo": "discord-bot"
  },
  "trajectory": {
    "tool_sequence": ["Read", "Read", "Edit", "Bash", "Bash"],
    "tool_counts": {"Read": 2, "Edit": 1, "Bash": 2},
    "total_tools": 5,
    "successes": 5,
    "failures": 0,
    "bash_errors": 0,
    "observed_event_count": 5,
    "placeholder_event_count": 0,
    "events": [...]
  },
  "outcome": {
    "annotation_status": "scored",
    "correction_detected": false,
    "build_success": true,
    "reward_score": 0.7825,
    "advantage": 0.2825,
    "reward_components": {...}
  },
  "timing": {
    "started_at": "ISO-8601",
    "ended_at": "ISO-8601",
    "duration_s": 65.0
  }
}
```

The store is append-only JSONL with fcntl file locking for concurrent safety. Historical rows from earlier writers are normalized into the same schema. When an old log was capped, KARL records explicit placeholder events so `total_tools`, `tool_sequence`, and `events` remain length-consistent while preserving the distinction between observed events and recovered tool-step slots.

## 4. Reward Engine

### 4.1 Design Principles

The reward function must satisfy three constraints:

1. **Zero human annotation.** All signals derive from observable agent behavior and implicit user feedback.
2. **Multi-dimensional.** A trajectory can succeed at the task but be inefficient, or fail at the task but demonstrate good process. The reward must capture both dimensions.
3. **Bounded and interpretable.** Scores in [0, 1] with clear component decomposition for debugging.

### 4.2 Six-Signal Composite

The deployed reward engine is a weighted combination of six signals:

$$R = 0.25 R_{outcome} + 0.22 R_{process} + 0.13 R_{efficiency} + 0.13 R_{verification} + 0.13 R_{consistency} + 0.14 R_{motion}$$

The original outcome/process/efficiency formulation is retained as the conceptual base. Deployment added verification, consistency, and wasted-motion penalties after early ablations showed that process shape carried most of the ranking signal.

#### 4.2.1 Outcome Score (R_outcome)

Cross-turn signals that indicate whether the user was satisfied:

| Signal | Weight | Interpretation |
|--------|--------|----------------|
| No correction detected | 0.35 | User did not say "no, I meant..." |
| No redo requested | 0.25 | User did not ask to try again |
| Build succeeded | 0.20 | Bash commands exited 0 |
| Session continued | 0.20 | User sent another prompt (vs. abandoning) |

When no signals are available (first turn, no builds), the outcome score defaults to 0.5. Each available signal is weighted proportionally to the number of available signals:

$$R_{outcome} = \frac{\sum_{i \in available} w_i \cdot s_i}{\sum_{i \in available} w_i}$$

where $s_i \in \{0, 1\}$ and $w_i$ is the signal weight. This prevents sessions with fewer signals from being systematically penalized.

#### 4.2.2 Process Score (R_process)

Within-turn quality metrics:

- **Tool success rate** (45% weight): $\frac{successes}{total\_tools}$
- **Bash cleanliness** (30% weight): $1 - \frac{bash\_errors}{bash\_count}$
- **Error density** (25% weight): Penalizes trajectories with 3+ consecutive tool failures, which indicate the agent is guessing rather than reasoning.

The consecutive failure penalty is computed as:

$$penalty = \frac{max(0, max\_consecutive\_failures - 2)}{total\_tools} \cdot 0.5$$

This targets the specific failure mode where an agent tries the same approach repeatedly without adapting.

#### 4.2.3 Efficiency Score (R_efficiency)

Trajectory shape metrics:

- **Tool diversity** (35% weight): Shannon entropy of the tool distribution, normalized by log(number of distinct tools). A trajectory that only calls Read is less diverse than one using Read, Edit, and Bash.

$$H = -\sum_{t \in tools} p(t) \log_2 p(t)$$

- **Duration efficiency** (35% weight): Tools per minute, with a [2, 8] ideal range. Below 2 tools/minute suggests the agent is stalling. Above 8 suggests it is not reading tool outputs.

- **File touch rate** (30% weight): Proportion of Write and Edit operations in the trajectory. A session that only reads files without modifying anything is typically less productive.

#### 4.2.4 Verification Score (R_verification)

Verification rewards trajectories that check their own work. It looks for tests, build commands, and read-after-write behavior. This signal separates sessions that merely edit files from sessions that close the loop with evidence.

#### 4.2.5 Consistency Score (R_consistency)

Consistency rewards coherent tool ordering and penalizes contradictory or thrashing patterns. Examples include repeated failed Bash calls without adaptation, write/write loops on the same target, and mutation before enough context has been gathered.

#### 4.2.6 Wasted Motion Score (R_motion)

Wasted motion penalizes tool retries, error loops, excessive reads after the answer is already known, and other low-linearity behavior. The goal is not to make trajectories shorter at all costs, but to prefer direct, adaptive progress over circular motion.

### 4.3 Advantage Computation

The advantage is computed relative to a domain-specific baseline:

$$A = R - \bar{R}_{domain}$$

where $\bar{R}_{domain}$ is the mean reward across all trajectories in the same skill domain. When insufficient domain data exists, a global baseline of 0.5 is used.

Positive advantage indicates the trajectory outperformed the typical trajectory for its domain. This is the key signal for training data selection.

## 5. Training Pipeline

### 5.1 OAPL-Lite: Advantage-Weighted SFT

We use a simplified version of Online Advantage-weighted Policy Learning for supervised fine-tuning:

1. **Filter**: Remove trajectories with fewer than 2 tool events (too short to be meaningful).
2. **Score**: Compute advantage for each trajectory.
3. **Oversample**: Include trajectories proportional to their advantage:

| Advantage | Copies in Training Data |
|-----------|------------------------|
| > 0.3 | 3x (maximum) |
| 0.1 - 0.3 | 2x |
| 0.0 - 0.1 | 1x |
| <= 0.0 | Excluded |

4. **Format**: Convert to ChatML JSONL with a system prompt, the user's task prompt, and the tool-use plan as the assistant response.
5. **Split**: 90/10 train/validation split with fixed random seed for reproducibility.
6. **Merge**: Synthetic QA examples generated from git commit diffs are appended after deduplication.

The training format teaches the model to generate effective tool-use plans given a task prompt:

```json
{
  "messages": [
    {"role": "system", "content": "You are an expert software engineering assistant..."},
    {"role": "user", "content": "deploy the discord bot to cloud-vm"},
    {"role": "assistant", "content": "1. [ok] Read ../src/config.py\n2. [ok] Edit ../src/config.py\n3. [ok] Bash: docker-compose build bot\n4. [ok] Bash: docker-compose up -d bot\n\nResult: 4/4 tools succeeded, reward=0.78"}
  ]
}
```

On the normalized schema-v2 corpus, the exporter produced 3,678 ChatML examples: 3,310 training rows and 368 validation rows. The exported plans average 9.26 tool steps, with a range of 2 to 20 steps after placeholder events are excluded from plan text.

### 5.2 Synthetic QA Augmentation

KARL generates additional training data from git commit diffs:

1. Scan recent commits (configurable lookback window, default 7 days).
2. Filter diffs by size: minimum 5 lines, maximum 200 lines.
3. For each qualifying diff, generate a question-answer pair where the question is "what changed?" and the answer describes the modification with file context.

This provides training signal from actual codebase changes even during periods with low interactive usage.

### 5.3 LoRA Fine-Tuning

Training runs on Apple Silicon via MLX:

```
Export SFT data -> SCP to compute node -> MLX LoRA train -> Monitor
```

Default configuration:
- Base model: `mlx-community/gemma-3-1b-it-4bit`
- LoRA rank: 8, 4 layers
- Learning rate: 1e-5
- Batch size: 1
- Max sequence length: 256
- 500 iterations

SSH connections use multiplexed ControlMaster for reliability. The trainer monitors the remote process and can interface with a fine-tune daemon for automated scheduling.

## 6. Skill Routing

### 6.1 Shadow Router

KARL operates a shadow vector router alongside the existing regex-based skill router:

1. **Pre-compute** embeddings for all active skills from their SKILL.md content (intent descriptions, workflow steps, gotchas, historical trigger prompts).
2. **On each prompt**, check the embedding cache for a cached prompt vector.
3. **If cache hit**: Compute weighted cosine similarity against all skill embeddings and select the top match.
4. **If cache miss**: Fire an asynchronous embedding request (background thread, daemon=True) so the vector is available on the next prompt.
5. **Log both selections** (regex and vector) to `routing_shadow.jsonl` without injecting the vector selection.

The shadow router has a hard performance budget: cache lookups take <1ms, and the async embedding request is non-blocking.

### 6.2 Promotion Gate

The shadow router graduates to active routing when four conditions are met simultaneously:

| Check | Threshold | Rationale |
|-------|-----------|-----------|
| Minimum records | 100 | Statistical significance |
| Cache hit rate | 50% | Embedding cache must be warm |
| Agreement rate | 80% | Vector shouldn't diverge wildly from regex |
| Reward lift | 5% | Vector must produce better outcomes on disagreements |

Until all four conditions pass, regex routing remains authoritative and vector routing is purely observational.

### 6.3 Weight Updates

Skill routing weights are updated via exponential moving average from reward data:

$$w_{new} = w_{current} \cdot (1 - \alpha) + w_{target} \cdot \alpha$$

where $\alpha = 0.1$ and $w_{target} = 0.5 + R$ maps the [0, 1] reward range to a [0.5, 1.5] target weight. Weights are bounded to [0.5, 1.5] to prevent any skill from being fully suppressed or dominant.

The final routing score is:

$$score(skill, prompt) = cos(embed(prompt), embed(skill)) \times weight(skill)$$

## 7. Entity Bridge

### 7.1 Motivation

Traditional skill management uses time-based decay: a skill not invoked in 30 days gets flagged, 60 days gets disabled, 90 days gets archived. This creates a perverse incentive where a frequently-used but consistently failing skill never decays, while a rarely-used but highly effective skill gets archived.

KARL's entity bridge replaces time-based decay with performance-based intelligence by feeding trajectory rewards back into per-skill entity state.

### 7.2 Entity State

Each skill maintains a persistent entity record:

```json
{
  "skill": "ops:deploy",
  "total_activations": 47,
  "useful_activations": 38,
  "suppressed_count": 3,
  "hot_topics": ["docker", "containers", "systemctl", "restart"],
  "cold_topics": ["terraform", "lambda"],
  "confidence_calibration": 0.72,
  "last_activated": "2026-03-10T14:30:00Z"
}
```

### 7.3 Update Rules

On each trajectory flush (Tap C), the entity bridge:

1. **Increments total_activations** unconditionally.
2. **Increments useful_activations** if reward >= 0.6.
3. **Updates confidence** via EMA: $conf_{new} = conf \cdot 0.9 + reward \cdot 0.1$
4. **Evolves hot_topics** from prompts of successful trajectories (reward >= 0.6).
5. **Evolves cold_topics** from prompts of corrected trajectories.
6. **Increments suppressed_count** on corrections.

### 7.4 Performance-Based Decay

The decay detector uses entity data instead of (or in addition to) time:

| Condition | Action |
|-----------|--------|
| confidence < 0.3 AND activations > 10 | Disable (actively harmful) |
| confidence < 0.5 AND inactive 30+ days | Warn (declining) |
| inactive 60+ days AND no reward data | Archive (never used) |
| confidence > 0.7 AND activations > 20 | Candidate for vector routing promotion |

This ensures that heavily-used but poorly-performing skills get attention, while rarely-used but effective skills are preserved.

## 8. Deployment

### 8.1 Production Configuration

KARL is deployed across a five-machine mesh:

| Machine | Role | KARL Components |
|---------|------|----------------|
| Mac1 | Orchestrator | All 4 taps, shadow router, entity bridge |
| Mac2 | iOS domain | Tap A/B/C (trajectory recording) |
| Mac3 | Creative domain | Tap A/B/C (trajectory recording) |
| Mac4 | Compute | exo cluster worker |
| Mac5 | Compute | LoRA training, MLX server |

Trajectory data from all machines consolidates to a central store via Syncthing. Live gateway flow data is ingested by `trajectory-ledgerd`, which preserves date-scoped cursor semantics so daily event files can restart at `seq=1` without dropping data. Training runs weekly on Mac5 (M4, 16GB).

### 8.2 Scale Characteristics

The current normalized deployment store contains:

- **Trajectory volume**: 7,468 scored records
- **Observed tool events**: 67,409 directly observed events
- **Recovered tool steps**: 73,470 total steps, including 6,061 explicit placeholders from capped historical logs
- **Source mix**: 7,300 `verbose-all` records, 43 live aura-gateway flow records, and 125 archive records
- **Training export**: 3,678 ChatML examples, split into 3,310 train and 368 validation rows
- **Reward distribution**: mean 0.6632, median 0.6675, standard deviation 0.0498, range [0.4666, 0.8165]
- **Hook latency target**: <5ms for cache hits, <500ms for cache misses (async)
- **Rust ledger daemon**: date-scoped cursor, locked JSONL append, score-at-emit, Prometheus text metrics
- **Entity updates**: <1ms per flush (JSON read/write)

### 8.3 Configuration

All 40+ parameters are configurable via environment variables with sensible defaults:

```bash
# Core paths
export KARL_DATA_DIR=~/.karl/data
export KARL_SKILLS_DIR=~/.claude/skills

# Reward weights in the deployed six-signal engine
export KARL_REWARD_W_OUTCOME=0.25
export KARL_REWARD_W_PROCESS=0.22
export KARL_REWARD_W_EFFICIENCY=0.13
export KARL_REWARD_W_VERIFICATION=0.13
export KARL_REWARD_W_CONSISTENCY=0.13
export KARL_REWARD_W_MOTION=0.14

# Training
export KARL_TRAIN_SSH_ALIAS=mac5
export KARL_MLX_MODEL=mlx-community/gemma-3-1b-it-4bit
export KARL_MLX_ITERS=500
```

## 9. Discussion

### 9.1 Reward Signal Quality

The six-signal reward captures different failure modes:

- **High outcome, low process**: The agent got lucky despite tool failures. The process score prevents this from being oversampled in training.
- **High process, low outcome**: The agent worked correctly but on the wrong task. The outcome score (via corrections) catches this.
- **High outcome, low efficiency**: The agent succeeded but used too many tools or lacked diversity. The efficiency score provides a parsimony incentive.
- **High mutation, low verification**: The agent changed files but did not test or inspect the result. The verification score catches this.
- **High process, high wasted motion**: The tools succeeded, but the path looped through avoidable retries. The motion score catches this.

The composite nature means no single axis can dominate, reducing reward hacking risk.

### 9.2 Cold Start

KARL requires ~100 trajectories before the shadow router has enough data for promotion analysis. During cold start, regex routing remains authoritative and trajectories accumulate passively. The backfill command can bootstrap from existing verbose logs if available.

### 9.3 Normalized Signal Ablation

After schema normalization, we reran a leave-one-out ranking ablation on all 7,468 scored trajectories. The corpus contains 67,409 observed tool events and 73,470 recovered tool steps. For each signal, we removed that signal, renormalized the remaining weights, and measured Spearman rank correlation against the full six-signal reward ranking.

| Rank | Removed Signal | Rank Correlation | Top-20 Overlap | Rank Impact |
|------|----------------|------------------|----------------|-------------|
| 1 | Verification | 0.5666 | 2/20 | 0.4334 |
| 2 | Process | 0.8909 | 19/20 | 0.1091 |
| 3 | Efficiency | 0.8938 | 15/20 | 0.1062 |
| 4 | Wasted motion | 0.8949 | 15/20 | 0.1051 |
| 5 | Consistency | 0.9346 | 12/20 | 0.0654 |
| 6 | Outcome | 1.0000 | 20/20 | 0.0000 |

**Signal means:** outcome = 0.5000, process = 0.9502, efficiency = 0.5571, verification = 0.3799, consistency = 0.6430, motion = 0.8836.

The strongest current differentiator is verification. Removing it drops rank correlation to 0.5666 and leaves only 2 of the top 20 trajectories in place. Process, efficiency, and motion each affect ranking at roughly the same order of magnitude. Outcome has zero rank impact in this normalized backfill because most historical batch records do not contain cross-turn correction or redo annotations, so the outcome channel defaults to 0.5. This should be read as a data-availability limitation, not as evidence that outcome feedback is intrinsically useless.

### 9.4 Advantage Selection Check

On the normalized exportable subset (5,805 records with at least two observed events), the top 35 domain-advantage trajectories have mean reward 0.7734. A deterministic random-35 control with seed 42 has mean reward 0.6771. The reward effect size is Cohen's d = 2.7159; the domain-advantage effect size is Cohen's d = 2.7917. This reproduces the earlier qualitative conclusion that high-advantage curation selects substantially stronger training examples than random selection. It is a selection-quality check, not a fresh LoRA training run.

### 9.5 Limitations

**Outcome attribution**: The correction detector uses regex patterns and heuristics. Subtle dissatisfaction (user switches tasks without correcting) is not captured. Future work could incorporate session-level engagement metrics.

**Single-turn trajectories**: KARL records within a single agent response. Multi-turn collaborative sessions where the user and agent iterate together are recorded as separate trajectories, losing some conversation-level signal.

**Historical placeholder events**: Earlier logs capped detailed event payloads. Schema v2 preserves recovered `total_tools` by inserting explicit placeholders, but SFT export excludes those placeholders from plan text. Paper metrics therefore distinguish 67,409 observed events from 73,470 recovered tool steps.

**Outcome sparsity in backfilled data**: Most historical records lack cross-turn correction/redo annotations, so outcome scores are often neutral. Live taps and future data should make this channel more informative.

**Model capacity**: The current LoRA training uses a 1B parameter base model (gemma-3-1b-it-4bit). The fine-tuned model learns tool-use planning patterns but cannot replace the frontier model for actual code generation. It serves as a routing and planning advisor, not a replacement.

## 10. Conclusion

The Trajectory Memory Ledger demonstrates that trajectory-based learning can turn ordinary coding-agent work into a reusable improvement signal. By recording what agents do, normalizing heterogeneous logs into one schema, scoring process quality, and exporting the best trajectories, the system creates a practical feedback loop for skill routing and tool-use planning.

The normalized corpus now contains 7,468 scored trajectories and 67,409 observed tool events. The schema-v2 ablation sharpens the deployed reward design: verification is the most load-bearing ranking signal in the current corpus, while process, efficiency, and wasted motion provide secondary but meaningful ranking structure. Outcome remains under-instrumented in historical backfill data and should be interpreted cautiously until more live cross-turn annotations accumulate.

The entity bridge extends this from session-level learning to skill-level intelligence, replacing time-based decay with performance-based adaptation. Skills that consistently produce poor trajectories lose confidence and routing weight, while skills that consistently succeed gain both.

KARL, the reference implementation of the Trajectory Memory Ledger, is open-source at [github.com/Diomandeee/karl](https://github.com/Diomandeee/karl) and designed for easy integration with any agent framework that supports hook events.

## References

1. Bai, Y., et al. "Constitutional AI: Harmlessness from AI Feedback." arXiv:2212.08073 (2022).

2. Jimenez, C. E., et al. "SWE-bench: Can Language Models Resolve Real-World GitHub Issues?" ICLR 2024.

3. Lightman, H., et al. "Let's Verify Step by Step." arXiv:2305.20050 (2023).

4. Ouyang, L., et al. "Training language models to follow instructions with human feedback." NeurIPS 2022.

5. Shazeer, N., et al. "Outrageously Large Neural Networks: The Sparsely-Gated Mixture-of-Experts Layer." ICLR 2017.

6. Databricks. "Agent Training: Trajectory-Based Learning for Coding Agents." Technical Report (2025).

---

## Appendix A: Reward Component Details

### Outcome Signal Availability

The outcome score adapts to available signals. In a fresh session with no prior context, only the "no correction" signal may be available. The score normalizes by the sum of available signal weights:

```python
available_weight = sum(w for signal, w in signals if signal is not None)
if available_weight > 0:
    score = sum(w * v for (signal, w), v in zip(signals, values) if signal is not None)
    score /= available_weight
else:
    score = 0.5  # No signals: neutral
```

### Shannon Entropy Normalization

Tool diversity uses normalized Shannon entropy:

```python
H = -sum(p * log2(p) for p in tool_distribution if p > 0)
H_max = log2(num_distinct_tools) if num_distinct_tools > 1 else 1.0
diversity = H / H_max
```

A single-tool trajectory scores 0.0. A perfectly uniform distribution across N tools scores 1.0.

## Appendix B: OAPL-Lite vs Full OAPL

Full Online Advantage-weighted Policy Learning uses on-policy rollouts with a value baseline and continuous policy updates. OAPL-Lite simplifies this to:

- **Offline** trajectories (from production hooks, not generated rollouts)
- **Domain-mean** baseline (instead of a learned value function)
- **Discrete** oversampling tiers (instead of continuous importance weights)
- **Periodic** batch training (weekly, instead of continuous updates)

This trades sample efficiency for implementation simplicity. With thousands of normalized trajectories available, the offline approach provides enough signal to generate curated SFT splits while keeping training runs reproducible.

## Appendix C: Entity Bridge Integration

The correction detector in the Cortex system fires on the `Stop` hook event, scoring the user's most recent prompt for behavioral correction patterns ("don't do X", "always Y", "never Z"). When confidence exceeds 0.6, it now calls KARL's Tap D:

```python
# In correction_detector.py Stop hook:
if confidence >= CONFIDENCE_THRESHOLD:
    _write_correction(prompt, confidence, matched)
    from karl.trajectory_tap import annotate_previous
    annotate_previous(session_id, correction_detected=True)
```

This creates a bidirectional flow: Cortex detects corrections and feeds KARL, KARL scores trajectories and feeds entity state, entity state informs routing and decay decisions that Cortex manages.

```
User Correction -> Cortex Detector -> KARL Tap D -> Trajectory Annotation
                                                           |
                                              Reward Scoring (lower outcome)
                                                           |
                                              Entity Bridge (confidence drops)
                                                           |
                                              Decay Detector (may flag/disable)
```
