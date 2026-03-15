"""
weight_updater.py — EMA weight updates for skill embeddings from reward data.

Updates trajectory_weight for each skill based on accumulated reward signals.
Weight bounds: [0.5, 1.5] — no skill fully suppressed or dominant.

A skill with consistent corrections (low reward) → weight trends toward 0.5.
A skill with consistent success (high reward) → weight trends toward 1.5.

Usage:
    python3 weight_updater.py              # Update weights from trajectory data
    python3 weight_updater.py --stats      # Show current weights
    python3 weight_updater.py --dry-run    # Preview weight changes
"""

import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

KARL_DIR = Path(__file__).parent
STORE_PATH = KARL_DIR / "trajectories.jsonl"

sys.path.insert(0, str(KARL_DIR))

from embedding_cache import load_skill_embeddings, save_skill_embeddings

# EMA config
ALPHA = 0.1      # Learning rate
WEIGHT_MIN = 0.5
WEIGHT_MAX = 1.5


def _reward_to_target(reward: float) -> float:
    """Map reward [0, 1] to weight target [0.5, 1.5].

    reward=0 → target=0.5 (penalize)
    reward=0.5 → target=1.0 (neutral)
    reward=1.0 → target=1.5 (boost)
    """
    return 0.5 + reward


def _ema_update(current: float, target: float, alpha: float = ALPHA) -> float:
    """Exponential moving average weight update with bounds."""
    new = current * (1 - alpha) + target * alpha
    return max(WEIGHT_MIN, min(WEIGHT_MAX, new))


def collect_skill_rewards() -> Dict[str, List[float]]:
    """Collect reward scores grouped by skill from trajectory store."""
    if not STORE_PATH.exists():
        return {}

    rewards_by_skill: Dict[str, List[float]] = {}

    with open(STORE_PATH) as f:
        for line in f:
            try:
                record = json.loads(line)
                skill_name = record.get("skill", {}).get("name")
                reward = record.get("outcome", {}).get("reward_score")

                if skill_name and reward is not None:
                    rewards_by_skill.setdefault(skill_name, []).append(reward)
            except json.JSONDecodeError:
                continue

    return rewards_by_skill


def update_weights(dry_run: bool = False) -> Dict[str, Any]:
    """Update skill embedding weights based on trajectory rewards.

    For each skill with reward data:
      1. Compute mean reward from all trajectories
      2. Map to weight target
      3. Apply EMA update to current weight

    Returns stats dict with old/new weights.
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


def show_weights() -> None:
    """Display current skill weights."""
    embeddings = load_skill_embeddings()
    rewards = collect_skill_rewards()

    print(f"\nSkill Embedding Weights ({len(embeddings)} skills)")
    print(f"{'Skill':<20} {'Weight':>8} {'Trajectories':>13} {'Mean Reward':>12}")
    print("-" * 55)

    for name in sorted(embeddings.keys()):
        _, weight = embeddings[name]
        skill_rewards = rewards.get(name, [])
        n = len(skill_rewards)
        mean_r = sum(skill_rewards) / n if n > 0 else None
        mean_str = f"{mean_r:.4f}" if mean_r is not None else "N/A"
        print(f"{name:<20} {weight:>8.4f} {n:>13} {mean_str:>12}")


if __name__ == "__main__":
    if "--stats" in sys.argv:
        show_weights()
    elif "--dry-run" in sys.argv:
        result = update_weights(dry_run=True)
        print(json.dumps(result, indent=2))
    else:
        result = update_weights()
        print(f"[weight_updater] {json.dumps(result, indent=2)}")
