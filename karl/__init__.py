"""
KARL - Knowledge Agents via Reinforcement Learning

Trajectory-based intelligence layer for AI coding agents.
Captures tool-use trajectories with outcome signals,
computes multi-signal reward scores, and enables
learned skill routing via LoRA fine-tuning.

Based on the KARL paper (arXiv 2603.05218).
"""

__version__ = "1.0.0"
__author__ = "Mohamed Diomande"

from karl.trajectory_tap import (
    init_session_buffer,
    append_tool_event,
    flush_session,
    annotate_previous,
    get_store_stats,
)
from karl.reward_engine import compute_reward, compute_advantage, backfill_rewards
from karl.embedding_cache import (
    cache_get,
    cache_store,
    embed_async,
    embed_sync,
    flush_cache,
    load_skill_embeddings,
    save_skill_embeddings,
    cosine_similarity,
    rank_skills,
)
from karl.entity_bridge import (
    update_entity_from_trajectory,
    get_entity_health,
    get_all_entity_health,
)

__all__ = [
    "init_session_buffer",
    "append_tool_event",
    "flush_session",
    "annotate_previous",
    "get_store_stats",
    "compute_reward",
    "compute_advantage",
    "backfill_rewards",
    "cache_get",
    "cache_store",
    "embed_async",
    "embed_sync",
    "flush_cache",
    "load_skill_embeddings",
    "save_skill_embeddings",
    "cosine_similarity",
    "rank_skills",
    "update_entity_from_trajectory",
    "get_entity_health",
    "get_all_entity_health",
]
