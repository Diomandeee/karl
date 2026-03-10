"""
config.py - Centralized configuration for KARL.

All paths, thresholds, and tuning parameters in one place.
Override via environment variables or config file.
"""

import os
from pathlib import Path
from typing import Optional


def _env_path(key: str, default: str) -> Path:
    """Resolve a path from environment variable or default."""
    return Path(os.environ.get(key, default)).expanduser()


def _env_float(key: str, default: float) -> float:
    return float(os.environ.get(key, str(default)))


def _env_int(key: str, default: int) -> int:
    return int(os.environ.get(key, str(default)))


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

KARL_DIR = _env_path("KARL_DIR", str(Path(__file__).parent.parent))
DATA_DIR = _env_path("KARL_DATA_DIR", str(KARL_DIR / "data"))
BUFFER_DIR = DATA_DIR / "buffers"
STORE_PATH = DATA_DIR / "trajectories.jsonl"
SHADOW_PATH = DATA_DIR / "routing_shadow.jsonl"
SYNTHETIC_PATH = DATA_DIR / "synthetic_qa.jsonl"
SKILL_EMBEDDINGS_PATH = DATA_DIR / "skill_embeddings.pkl"
PROMPT_CACHE_PATH = DATA_DIR / "prompt_embedding_cache.pkl"
STATUS_PATH = DATA_DIR / "karl_status.json"
TRAIN_PATH = DATA_DIR / "train.jsonl"
VALID_PATH = DATA_DIR / "valid.jsonl"
SFT_OUTPUT_PATH = DATA_DIR / "karl-sft.jsonl"

# External paths (configurable)
SKILLS_DIR = _env_path("KARL_SKILLS_DIR", "~/.claude/skills")
SKILLS_REGISTRY = SKILLS_DIR / "registry.json"
CORTEX_ENTRIES = _env_path("KARL_CORTEX_ENTRIES", "~/.claude/cortex/entries.jsonl")
VERBOSE_LOG = _env_path("KARL_VERBOSE_LOG", "~/.claude/prompt-logs/verbose-all.jsonl")

# ---------------------------------------------------------------------------
# Embedding
# ---------------------------------------------------------------------------

EMBED_URL = os.environ.get("KARL_EMBED_URL", "http://localhost:8000/api/rag/embed")
EMBEDDING_DIM = _env_int("KARL_EMBEDDING_DIM", 3072)  # gemini-embedding-001
EMBED_TIMEOUT = _env_int("KARL_EMBED_TIMEOUT", 8)

# Prompt embedding cache
CACHE_MAX_ENTRIES = _env_int("KARL_CACHE_MAX_ENTRIES", 500)
CACHE_TTL_SECONDS = _env_int("KARL_CACHE_TTL", 86400)  # 24 hours

# ---------------------------------------------------------------------------
# Reward engine
# ---------------------------------------------------------------------------

REWARD_W_OUTCOME = _env_float("KARL_REWARD_W_OUTCOME", 0.40)
REWARD_W_PROCESS = _env_float("KARL_REWARD_W_PROCESS", 0.35)
REWARD_W_EFFICIENCY = _env_float("KARL_REWARD_W_EFFICIENCY", 0.25)

# ---------------------------------------------------------------------------
# Weight updater (EMA)
# ---------------------------------------------------------------------------

WEIGHT_ALPHA = _env_float("KARL_WEIGHT_ALPHA", 0.1)
WEIGHT_MIN = _env_float("KARL_WEIGHT_MIN", 0.5)
WEIGHT_MAX = _env_float("KARL_WEIGHT_MAX", 1.5)

# ---------------------------------------------------------------------------
# SFT export
# ---------------------------------------------------------------------------

SFT_MAX_OVERSAMPLE = _env_int("KARL_SFT_MAX_OVERSAMPLE", 3)
SFT_ADVANTAGE_THRESHOLD = _env_float("KARL_SFT_ADVANTAGE_THRESHOLD", 0.0)
SFT_MIN_TOOL_EVENTS = _env_int("KARL_SFT_MIN_TOOL_EVENTS", 2)
SFT_TRAIN_SPLIT = _env_float("KARL_SFT_TRAIN_SPLIT", 0.9)

SFT_SYSTEM_PROMPT = (
    "You are an expert software engineering assistant. Given a task, "
    "plan the optimal sequence of tool uses to accomplish it efficiently. "
    "Consider which tools to use, in what order, and what parameters. "
    "Prefer reading before editing, testing after changes, and using "
    "the most specific tool available."
)

# ---------------------------------------------------------------------------
# Promotion gate
# ---------------------------------------------------------------------------

PROMOTION_MIN_RECORDS = _env_int("KARL_PROMOTION_MIN_RECORDS", 100)
PROMOTION_MIN_CACHE_HIT_RATE = _env_float("KARL_PROMOTION_MIN_CACHE_HIT", 0.50)
PROMOTION_MIN_AGREEMENT = _env_float("KARL_PROMOTION_MIN_AGREEMENT", 0.80)
PROMOTION_MIN_VECTOR_LIFT = _env_float("KARL_PROMOTION_MIN_LIFT", 0.05)

# ---------------------------------------------------------------------------
# Training (Mac5 / remote)
# ---------------------------------------------------------------------------

TRAIN_SSH_ALIAS = os.environ.get("KARL_TRAIN_SSH_ALIAS", "mac5")
TRAIN_HOST = os.environ.get("KARL_TRAIN_HOST", "100.109.94.124")
TRAIN_DAEMON_PORT = _env_int("KARL_TRAIN_DAEMON_PORT", 9200)
TRAIN_REMOTE_DIR = os.environ.get(
    "KARL_TRAIN_REMOTE_DIR",
    "~/Desktop/homelab/compute-pair/karl-training",
)
TRAIN_MERGED_DIR = os.environ.get(
    "KARL_TRAIN_MERGED_DIR",
    "~/Desktop/homelab/compute-pair/merged-training",
)

# MLX LoRA config
MLX_MODEL = os.environ.get("KARL_MLX_MODEL", "mlx-community/gemma-3-1b-it-4bit")
MLX_ITERS = _env_int("KARL_MLX_ITERS", 500)
MLX_BATCH_SIZE = _env_int("KARL_MLX_BATCH_SIZE", 1)
MLX_NUM_LAYERS = _env_int("KARL_MLX_NUM_LAYERS", 4)
MLX_MAX_SEQ_LEN = _env_int("KARL_MLX_MAX_SEQ_LEN", 256)
MLX_LR = _env_float("KARL_MLX_LR", 1e-5)

# ---------------------------------------------------------------------------
# Notifications
# ---------------------------------------------------------------------------

DISCORD_WEBHOOK = os.environ.get(
    "KARL_DISCORD_WEBHOOK",
    os.environ.get("DISCORD_WEBHOOK_SERVICE_HEALTH", ""),
)

# ---------------------------------------------------------------------------
# Synthetic QA
# ---------------------------------------------------------------------------

SYNTHETIC_MIN_DIFF_LINES = _env_int("KARL_SYNTHETIC_MIN_DIFF", 5)
SYNTHETIC_MAX_DIFF_LINES = _env_int("KARL_SYNTHETIC_MAX_DIFF", 200)
SYNTHETIC_DEFAULT_DAYS = _env_int("KARL_SYNTHETIC_DAYS", 7)


def ensure_dirs() -> None:
    """Create required directories."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    BUFFER_DIR.mkdir(parents=True, exist_ok=True)
