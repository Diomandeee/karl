# Configuration Reference

All KARL parameters are configurable via environment variables. Defaults are designed for immediate use.

## Paths

| Variable | Default | Description |
|----------|---------|-------------|
| `KARL_DIR` | Package root | Base KARL directory |
| `KARL_DATA_DIR` | `$KARL_DIR/data` | Data storage directory |
| `KARL_SKILLS_DIR` | `~/.claude/skills` | Directory containing SKILL.md files |
| `KARL_CORTEX_ENTRIES` | `~/.claude/cortex/entries.jsonl` | Cortex invocation log |
| `KARL_VERBOSE_LOG` | `~/.claude/prompt-logs/verbose-all.jsonl` | Verbose session log for backfill |

## Embedding

| Variable | Default | Description |
|----------|---------|-------------|
| `KARL_EMBED_URL` | `http://localhost:8000/api/rag/embed` | Embedding API endpoint |
| `KARL_EMBEDDING_DIM` | `3072` | Expected embedding dimension |
| `KARL_EMBED_TIMEOUT` | `8` | API call timeout in seconds |
| `KARL_CACHE_MAX_ENTRIES` | `500` | Max entries in prompt embedding cache |
| `KARL_CACHE_TTL` | `86400` | Cache entry TTL in seconds (24h) |

## Reward Engine

| Variable | Default | Description |
|----------|---------|-------------|
| `KARL_REWARD_W_OUTCOME` | `0.40` | Weight for outcome score |
| `KARL_REWARD_W_PROCESS` | `0.35` | Weight for process score |
| `KARL_REWARD_W_EFFICIENCY` | `0.25` | Weight for efficiency score |

Weights must sum to 1.0.

## Weight Updater

| Variable | Default | Description |
|----------|---------|-------------|
| `KARL_WEIGHT_ALPHA` | `0.1` | EMA learning rate |
| `KARL_WEIGHT_MIN` | `0.5` | Minimum skill weight |
| `KARL_WEIGHT_MAX` | `1.5` | Maximum skill weight |

## SFT Export

| Variable | Default | Description |
|----------|---------|-------------|
| `KARL_SFT_MAX_OVERSAMPLE` | `3` | Max oversampling for high-advantage trajectories |
| `KARL_SFT_ADVANTAGE_THRESHOLD` | `0.0` | Minimum advantage to include |
| `KARL_SFT_MIN_TOOL_EVENTS` | `2` | Skip trajectories with fewer tools |
| `KARL_SFT_TRAIN_SPLIT` | `0.9` | Train/valid split ratio |

## Promotion Gate

| Variable | Default | Description |
|----------|---------|-------------|
| `KARL_PROMOTION_MIN_RECORDS` | `100` | Minimum shadow routing records |
| `KARL_PROMOTION_MIN_CACHE_HIT` | `0.50` | Minimum cache hit rate |
| `KARL_PROMOTION_MIN_AGREEMENT` | `0.80` | Minimum regex/vector agreement |
| `KARL_PROMOTION_MIN_LIFT` | `0.05` | Minimum reward lift from vector |

## Training

| Variable | Default | Description |
|----------|---------|-------------|
| `KARL_TRAIN_SSH_ALIAS` | `mac5` | SSH config alias for compute node |
| `KARL_TRAIN_HOST` | `100.109.94.124` | Compute node IP (for daemon health check) |
| `KARL_TRAIN_DAEMON_PORT` | `9200` | Finetune daemon port |
| `KARL_TRAIN_REMOTE_DIR` | `~/Desktop/homelab/compute-pair/karl-training` | Remote training data directory |
| `KARL_TRAIN_MERGED_DIR` | `~/Desktop/homelab/compute-pair/merged-training` | Remote merged data directory |
| `KARL_MLX_MODEL` | `mlx-community/gemma-3-1b-it-4bit` | Base model for LoRA |
| `KARL_MLX_ITERS` | `500` | Training iterations |
| `KARL_MLX_BATCH_SIZE` | `1` | Training batch size |
| `KARL_MLX_NUM_LAYERS` | `4` | Number of LoRA layers |
| `KARL_MLX_MAX_SEQ_LEN` | `256` | Maximum sequence length |
| `KARL_MLX_LR` | `1e-5` | Learning rate |

## Notifications

| Variable | Default | Description |
|----------|---------|-------------|
| `KARL_DISCORD_WEBHOOK` | (none) | Discord webhook URL for reports |

Falls back to `DISCORD_WEBHOOK_SERVICE_HEALTH` environment variable.

## Synthetic QA

| Variable | Default | Description |
|----------|---------|-------------|
| `KARL_SYNTHETIC_MIN_DIFF` | `5` | Minimum diff lines to generate QA |
| `KARL_SYNTHETIC_MAX_DIFF` | `200` | Maximum diff lines to process |
| `KARL_SYNTHETIC_DAYS` | `7` | Default lookback days for git diffs |
