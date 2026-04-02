#!/usr/bin/env python3
"""Generate DPO preference pairs from KARL's trajectory store.

Pairs trajectories within the same skill domain by reward delta.
Top quartile = chosen, bottom quartile = rejected.
No topology needed — linear sessions, cross-session comparison.

Output: JSONL with {prompt, chosen, rejected, reward_delta, skill, source}
"""

import json
import os
import random
from collections import defaultdict
from dataclasses import dataclass, asdict
from pathlib import Path

TRAJECTORIES_PATH = Path(__file__).parent.parent / "trajectories.jsonl"
OUTPUT_PATH = os.path.expanduser("~/Desktop/karl/v7-dpo-pairs.jsonl")

MIN_REWARD_DELTA = 0.03  # Minimum reward difference for a valid pair
MIN_PROMPT_LENGTH = 20
MAX_PAIRS_PER_DOMAIN = 50


@dataclass
class DPOPair:
    prompt: str
    chosen: str
    rejected: str
    reward_delta: float
    chosen_reward: float
    rejected_reward: float
    skill: str
    source: str  # "trajectory_store"


def load_trajectories(path: str = None) -> list[dict]:
    """Load scored trajectories."""
    p = Path(path) if path else TRAJECTORIES_PATH
    if not p.exists():
        return []
    trajs = []
    with open(p) as f:
        for line in f:
            try:
                t = json.loads(line)
                if t.get("reward_score") is not None:
                    trajs.append(t)
            except json.JSONDecodeError:
                continue
    return trajs


def _extract_prompt_response(traj: dict) -> tuple[str, str]:
    """Extract the prompt and response from a trajectory record."""
    t = traj.get("trajectory", {})
    prompt = t.get("prompt", "")
    response = t.get("response", "")
    return prompt.strip(), response.strip()


def generate_pairs(
    trajectories: list[dict],
    min_delta: float = MIN_REWARD_DELTA,
    max_per_domain: int = MAX_PAIRS_PER_DOMAIN,
) -> list[DPOPair]:
    """Generate DPO preference pairs from trajectory reward deltas.

    Strategy: within each skill domain, pair top quartile (chosen)
    with bottom quartile (rejected) trajectories.
    """
    # Group by skill domain
    by_skill: dict[str, list[dict]] = defaultdict(list)
    for t in trajectories:
        skill = t.get("skill", "unknown")
        if isinstance(skill, dict):
            skill = skill.get("label", skill.get("domain", "unknown"))
        prompt, response = _extract_prompt_response(t)
        if len(prompt) < MIN_PROMPT_LENGTH or not response:
            continue
        by_skill[skill].append(t)

    pairs = []
    stats = {"domains": 0, "pairs": 0, "skipped_small": 0, "skipped_delta": 0}

    for skill, trajs in sorted(by_skill.items(), key=lambda x: -len(x[1])):
        if len(trajs) < 4:
            stats["skipped_small"] += 1
            continue

        stats["domains"] += 1

        # Sort by reward
        trajs.sort(key=lambda t: t.get("reward_score", 0))

        # Top quartile = chosen, bottom quartile = rejected
        q_size = max(1, len(trajs) // 4)
        bottom = trajs[:q_size]
        top = trajs[-q_size:]

        # Generate pairs: each top paired with each bottom (capped)
        domain_pairs = []
        for chosen in top:
            for rejected in bottom:
                c_reward = chosen.get("reward_score", 0)
                r_reward = rejected.get("reward_score", 0)
                delta = c_reward - r_reward

                if delta < min_delta:
                    stats["skipped_delta"] += 1
                    continue

                c_prompt, c_response = _extract_prompt_response(chosen)
                r_prompt, r_response = _extract_prompt_response(rejected)

                # Use the chosen prompt as the shared prompt context
                domain_pairs.append(DPOPair(
                    prompt=c_prompt,
                    chosen=c_response,
                    rejected=r_response,
                    reward_delta=round(delta, 4),
                    chosen_reward=round(c_reward, 4),
                    rejected_reward=round(r_reward, 4),
                    skill=skill,
                    source="trajectory_store",
                ))

        # Cap per domain to prevent one big domain from dominating
        if len(domain_pairs) > max_per_domain:
            domain_pairs = random.sample(domain_pairs, max_per_domain)

        pairs.extend(domain_pairs)
        stats["pairs"] += len(domain_pairs)

    return pairs, stats


def generate_factory_pairs(
    factory_log_dir: str = None,
    min_delta: float = 0.05,
) -> list[DPOPair]:
    """Generate DPO pairs from V7 factory session logs.

    When multiple factory sessions addressed similar goals,
    compare turns with different style scores.
    """
    log_dir = factory_log_dir or os.path.expanduser("~/Desktop/karl/v7-session-logs")
    if not os.path.isdir(log_dir):
        return []

    import glob
    all_turns = []
    for fpath in glob.glob(os.path.join(log_dir, "*.jsonl")):
        with open(fpath) as f:
            for line in f:
                try:
                    t = json.loads(line)
                    all_turns.append(t)
                except json.JSONDecodeError:
                    continue

    if len(all_turns) < 10:
        return []

    # Sort by style score — top half chosen, bottom half rejected
    all_turns.sort(key=lambda t: t.get("style_score", 0))
    mid = len(all_turns) // 2
    bottom = all_turns[:mid]
    top = all_turns[mid:]

    pairs = []
    for chosen in random.sample(top, min(20, len(top))):
        for rejected in random.sample(bottom, min(3, len(bottom))):
            c_score = chosen.get("style_score", 0)
            r_score = rejected.get("style_score", 0)
            if c_score - r_score < min_delta:
                continue

            pairs.append(DPOPair(
                prompt=f"Session phase: {chosen.get('phase', 'BUILD')}",
                chosen=chosen.get("final_prompt", ""),
                rejected=rejected.get("final_prompt", ""),
                reward_delta=round(c_score - r_score, 4),
                chosen_reward=round(c_score, 4),
                rejected_reward=round(r_score, 4),
                skill="factory_prompt_quality",
                source="v7_factory",
            ))

    return pairs[:50]


def export(pairs: list[DPOPair], output_path: str = OUTPUT_PATH):
    """Export DPO pairs as JSONL."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        for p in pairs:
            f.write(json.dumps(asdict(p)) + "\n")
    print(f"Exported {len(pairs)} DPO pairs to {output_path}")


def main():
    print("KARL V7 DPO Pair Generator")
    print("=" * 60)

    # Trajectory-based pairs
    trajs = load_trajectories()
    print(f"Loaded {len(trajs)} scored trajectories")

    pairs, stats = generate_pairs(trajs)
    print(f"\nTrajectory pairs: {stats}")

    # Factory-based pairs
    factory_pairs = generate_factory_pairs()
    print(f"Factory pairs: {len(factory_pairs)}")

    all_pairs = pairs + factory_pairs
    print(f"\nTotal DPO pairs: {len(all_pairs)}")

    if all_pairs:
        export(all_pairs)

        # Analysis
        deltas = [p.reward_delta for p in all_pairs]
        skills = defaultdict(int)
        for p in all_pairs:
            skills[p.skill] += 1
        sources = defaultdict(int)
        for p in all_pairs:
            sources[p.source] += 1

        print(f"\nReward delta: avg={sum(deltas)/len(deltas):.3f} min={min(deltas):.3f} max={max(deltas):.3f}")
        print(f"By skill: {dict(sorted(skills.items(), key=lambda x: -x[1])[:10])}")
        print(f"By source: {dict(sources)}")

        # Sample
        print(f"\nSample pairs:")
        for p in random.sample(all_pairs, min(3, len(all_pairs))):
            print(f"  [{p.skill}] delta={p.reward_delta:.3f}")
            print(f"    Chosen ({p.chosen_reward:.3f}):  {p.chosen[:100]}")
            print(f"    Rejected ({p.rejected_reward:.3f}): {p.rejected[:100]}")
            print()


if __name__ == "__main__":
    main()
