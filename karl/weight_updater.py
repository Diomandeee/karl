"""
weight_updater.py - EMA weight updates for skill embeddings from reward data.

Updates trajectory_weight for each skill based on accumulated reward signals.
Weight bounds: [0.5, 1.5] -- no skill fully suppressed or dominant.

A skill with consistent corrections (low reward) -> weight trends toward 0.5.
A skill with consistent success (high reward) -> weight trends toward 1.5.

Usage:
    from karl.weight_updater import update_weights
    result = update_weights()           # Apply weight updates
    result = update_weights(dry_run=True)  # Preview changes
"""

import fcntl
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

from karl.config import STORE_PATH, WEIGHT_ALPHA, WEIGHT_MIN, WEIGHT_MAX
from karl.embedding_cache import load_skill_embeddings, save_skill_embeddings


def _reward_to_target(reward: float) -> float:
    """Map reward [0, 1] to weight target [0.5, 1.5].

    reward=0   -> target=0.5 (penalize)
    reward=0.5 -> target=1.0 (neutral)
    reward=1.0 -> target=1.5 (boost)
    """
    return WEIGHT_MIN + reward * (WEIGHT_MAX - WEIGHT_MIN)


def _ema_update(current: float, target: float, alpha: float = WEIGHT_ALPHA) -> float:
    """Exponential moving average weight update with bounds."""
    new = current * (1 - alpha) + target * alpha
    return max(WEIGHT_MIN, min(WEIGHT_MAX, new))


def collect_skill_rewards() -> Dict[str, List[float]]:
    """Collect reward scores grouped by skill from trajectory store."""
    if not STORE_PATH.exists():
        return {}

    rewards_by_skill: Dict[str, List[float]] = {}
    # C11: Shared lock for consistent read
    with open(STORE_PATH) as f:
        fcntl.flock(f.fileno(), fcntl.LOCK_SH)
        try:
            for line in f:
                try:
                    record = json.loads(line)
                    skill_name = record.get("skill", {}).get("name")
                    reward = record.get("outcome", {}).get("reward_score")
                    if skill_name and reward is not None:
                        rewards_by_skill.setdefault(skill_name, []).append(reward)
                except json.JSONDecodeError:
                    continue
        finally:
            fcntl.flock(f.fileno(), fcntl.LOCK_UN)

    return rewards_by_skill


def update_weights(dry_run: bool = False) -> Dict[str, Any]:
    """Update skill embedding weights based on trajectory rewards.

    For each skill with reward data:
      1. Compute mean reward from all trajectories
      2. Map to weight target via _reward_to_target
      3. Apply EMA update to current weight

    Args:
        dry_run: If True, preview changes without saving

    Returns:
        Stats dict with old/new weights and deltas
    """
    skill_embeddings = load_skill_embeddings()
    if not skill_embeddings:
        return {"error": "No skill embeddings loaded"}

    rewards_by_skill = collect_skill_rewards()
    if not rewards_by_skill:
        return {"message": "No reward data yet", "skills": len(skill_embeddings)}

    updates = {}
    for skill_name, rewards in rewards_by_skill.items():
        if skill_name not in skill_embeddings:
            continue

        embedding, current_weight = skill_embeddings[skill_name]
        mean_reward = sum(rewards) / len(rewards)
        target = _reward_to_target(mean_reward)
        new_weight = _ema_update(current_weight, target)

        updates[skill_name] = {
            "old_weight": round(current_weight, 4),
            "new_weight": round(new_weight, 4),
            "mean_reward": round(mean_reward, 4),
            "trajectory_count": len(rewards),
            "delta": round(new_weight - current_weight, 4),
        }

        if not dry_run:
            skill_embeddings[skill_name] = (embedding, new_weight)

    if not dry_run and updates:
        save_skill_embeddings(skill_embeddings)

    return {
        "updated": len(updates),
        "dry_run": dry_run,
        "updates": updates,
    }
