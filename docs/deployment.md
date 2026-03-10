# Deployment Guide

## Quick Start

```bash
# Clone and install
git clone https://github.com/Diomandeee/karl.git
cd karl
pip install -e ".[dev]"

# Verify installation
karl status
pytest
```

## Prerequisites

- Python 3.10+
- An embedding API endpoint (e.g., RAG++ with Gemini embedding, or any OpenAI-compatible endpoint)
- For training: Apple Silicon Mac with MLX installed, accessible via SSH

## Setting Up Trajectory Recording

### 1. Configure data directory

```bash
export KARL_DATA_DIR=~/.karl/data
```

KARL will create `data/`, `data/buffers/`, and all JSONL files automatically.

### 2. Wire into your agent's hooks

KARL needs four integration points with your agent framework:

```python
# In your UserPromptSubmit hook:
from karl import init_session_buffer
init_session_buffer(session_id, skill_name=skill, prompt_text=prompt)

# In your PostToolUse hook:
from karl import append_tool_event
append_tool_event(session_id, tool_name, tool_input=params, success=ok)

# In your Stop/SessionEnd hook:
from karl import flush_session
flush_session(session_id)

# In your next UserPromptSubmit (correction detection):
from karl import annotate_previous
if is_correction(prompt):
    annotate_previous(session_id, correction_detected=True)
```

### 3. Historical backfill (optional)

If you have existing verbose logs in JSONL format:

```bash
export KARL_VERBOSE_LOG=~/.claude/prompt-logs/verbose-all.jsonl
karl extract
karl backfill
```

## Setting Up Skill Routing

### 1. Configure embedding endpoint

```bash
export KARL_EMBED_URL=http://localhost:8000/api/rag/embed
export KARL_EMBEDDING_DIM=3072  # Match your model's output dimension
```

The endpoint should accept `POST {"text": "..."}` and return `{"embedding": [...]}`.

### 2. Bootstrap skill embeddings

```bash
# Point to your skills directory
export KARL_SKILLS_DIR=~/.claude/skills

# Generate embeddings
karl bootstrap
```

### 3. Enable shadow routing

In your routing hook, add vector comparison:

```python
from karl.embedding_cache import cache_get, embed_async, rank_skills, load_skill_embeddings

# Embed current prompt (result cached for next call)
embed_async(prompt_text)

# Check cache
embedding = cache_get(prompt_text)
if embedding:
    skills = load_skill_embeddings()
    ranked = rank_skills(embedding, skills)
    vector_pick = ranked[0][0] if ranked else None
    # Log comparison to routing_shadow.jsonl
```

### 4. Monitor promotion readiness

```bash
karl analyze
```

When all four promotion checks pass, switch from shadow to active vector routing.

## Setting Up Training

### 1. Configure SSH access

```bash
export KARL_TRAIN_SSH_ALIAS=mac5  # Your SSH config alias
export KARL_TRAIN_HOST=100.109.94.124
```

Ensure your `~/.ssh/config` has multiplexed connections:

```
Host mac5
    HostName 100.109.94.124
    User your_user
    IdentitiesOnly yes
    IdentityFile ~/.ssh/id_ed25519
    ControlMaster auto
    ControlPath ~/.ssh/sockets/%r@%h-%p
    ControlPersist 600
```

### 2. Install MLX on the compute node

```bash
ssh mac5
pip install mlx-lm
```

### 3. Run training

```bash
# Preview export
karl export --dry-run

# Full pipeline
karl train
```

## Automated Scheduling (Prefect)

```bash
pip install "karl[prefect]"

# Deploy daily analysis flow
prefect deployment build karl/flows.py:karl_analysis -n karl-daily -q default

# Deploy weekly training flow
prefect deployment build karl/flows.py:karl_training -n karl-weekly -q default
```

## Discord Notifications

```bash
export KARL_DISCORD_WEBHOOK=https://discord.com/api/webhooks/YOUR_WEBHOOK
```

KARL will post daily analysis summaries and weekly training reports.
