"""
reward_engine.py - Multi-signal reward computation for KARL trajectories.

Computes a composite reward score from three signal levels:

  1. Outcome reward  (40%) - cross-turn signals (correction, redo, continuation)
  2. Process reward  (35%) - within-turn signals (exit codes, error rate, tool success)
  3. Efficiency score (25%) - tool diversity, duration, file modification patterns

All scores normalized to [0, 1]. Higher = better trajectory.

The composite reward enables advantage-weighted SFT (OAPL-Lite):
  advantage = reward - domain_baseline
  positive advantage -> oversample in training
  negative advantage -> exclude from training

Usage:
    from karl.reward_engine import compute_reward, backfill_rewards
    score = compute_reward(trajectory_record)
    backfill_rewards()  # annotate all pending records
"""

import fcntl
import json
import logging
import math
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

from karl.config import (
    STORE_PATH,
    REWARD_W_OUTCOME,
    REWARD_W_PROCESS,
    REWARD_W_EFFICIENCY,
)


def compute_reward(record: Dict) -> Dict[str, Any]:
    """
    Compute multi-signal reward for a trajectory record.

    Args:
        record: A trajectory record with 'outcome', 'trajectory', and 'timing' keys

    Returns:
        Dict with reward_score [0,1], individual scores, and signal components
    """
    outcome = record.get("outcome", {})
    trajectory = record.get("trajectory", {})
    timing = record.get("timing", {})

    outcome_score, outcome_components = _compute_outcome(outcome)
    process_score, process_components = _compute_process(trajectory)
    efficiency_score, efficiency_components = _compute_efficiency(trajectory, timing)

    composite = (
        REWARD_W_OUTCOME * outcome_score
        + REWARD_W_PROCESS * process_score
        + REWARD_W_EFFICIENCY * efficiency_score
    )

    return {
        "reward_score": round(composite, 4),
        "outcome_score": round(outcome_score, 4),
        "process_score": round(process_score, 4),
        "efficiency_score": round(efficiency_score, 4),
        "components": {
            **outcome_components,
            **process_components,
            **efficiency_components,
        },
    }


def _compute_outcome(outcome: Dict) -> Tuple[float, Dict]:
    """
    Outcome reward from cross-turn signals.

    Signals (when available):
      - correction_detected: False=+0.35, True=-0.35
      - redo_detected: False=+0.25, True=-0.25
      - build_success: True=+0.20, False=-0.10
      - session_continued: True=+0.20, False=0

    Base score 0.5 (neutral when no signals available).
    """
    score = 0.5
    available = 0

    correction = outcome.get("correction_detected")
    redo = outcome.get("redo_detected")
    build = outcome.get("build_success")
    continued = outcome.get("session_continued")

    components: Dict[str, Any] = {}

    if correction is not None:
        available += 1
        delta = -0.35 if correction else 0.35
        score += delta
        components["r_no_correction"] = 0.0 if correction else 1.0

    if redo is not None:
        available += 1
        delta = -0.25 if redo else 0.25
        score += delta
        components["r_no_redo"] = 0.0 if redo else 1.0

    if build is not None:
        available += 1
        delta = 0.20 if build else -0.10
        score += delta
        components["r_build_pass"] = 1.0 if build else 0.0

    if continued is not None:
        available += 1
        delta = 0.20 if continued else 0.0
        score += delta
        components["r_session_continued"] = 1.0 if continued else 0.0

    components["outcome_signals_available"] = available
    return max(0.0, min(1.0, score)), components


def _compute_process(trajectory: Dict) -> Tuple[float, Dict]:
    """
    Process reward from within-turn signals.

    Components:
      - Tool success rate (45%): successes / total_tools
      - Bash cleanliness (30%): 1 - (bash_errors / bash_count)
      - Error density (25%): penalizes consecutive failures
    """
    total = trajectory.get("total_tools", 0)
    successes = trajectory.get("successes", 0)
    failures = trajectory.get("failures", 0)
    bash_errors = trajectory.get("bash_errors", 0)

    if total == 0:
        return 0.5, {"r_success_rate": 0.5, "r_bash_clean": 1.0, "r_error_density": 1.0}

    success_rate = successes / total if total > 0 else 0.5

    events = trajectory.get("events", [])
    bash_count = sum(1 for e in events if e.get("tool_name") == "Bash")
    bash_clean = 1.0
    if bash_count > 0:
        bash_clean = 1.0 - (bash_errors / bash_count)

    error_density = 1.0
    if failures > 0:
        max_consecutive = _max_consecutive_failures(events)
        if max_consecutive >= 3:
            error_density = max(0.0, 1.0 - (max_consecutive - 2) * 0.15)

    process = success_rate * 0.45 + bash_clean * 0.30 + error_density * 0.25

    return max(0.0, min(1.0, process)), {
        "r_success_rate": round(success_rate, 4),
        "r_bash_clean": round(bash_clean, 4),
        "r_error_density": round(error_density, 4),
    }


def _compute_efficiency(trajectory: Dict, timing: Dict) -> Tuple[float, Dict]:
    """
    Efficiency score from trajectory shape.

    Components:
      - Tool diversity (35%): Shannon entropy normalized by log(unique tools)
      - Duration efficiency (35%): tools per minute (ideal: 2-8/min)
      - File touch rate (30%): proportion of Write/Edit vs Read-only
    """
    total = trajectory.get("total_tools", 0)
    tool_counts = trajectory.get("tool_counts", {})
    duration = timing.get("duration_s")

    if total == 0:
        return 0.5, {"r_diversity": 0.5, "r_duration_eff": 0.5, "r_file_touch": 0.5}

    # Tool diversity via Shannon entropy
    unique_tools = len(tool_counts)
    if unique_tools <= 1:
        diversity = 0.3  # Monoculture penalty
    else:
        entropy = 0.0
        for count in tool_counts.values():
            p = count / total
            if p > 0:
                entropy -= p * math.log2(p)
        max_entropy = math.log2(unique_tools)
        diversity = entropy / max_entropy if max_entropy > 0 else 0.5

    # Duration efficiency
    duration_eff = 0.5
    if duration and duration > 0:
        tools_per_min = (total / duration) * 60
        if 2 <= tools_per_min <= 8:
            duration_eff = 1.0
        elif tools_per_min > 8:
            duration_eff = 0.8
        elif tools_per_min >= 1:
            duration_eff = 0.6
        else:
            duration_eff = max(0.2, 0.5 - (1.0 - tools_per_min) * 0.3)

    # File touch rate
    mutation_tools = {"Write", "Edit", "NotebookEdit"}
    mutation_count = sum(c for t, c in tool_counts.items() if t in mutation_tools)
    file_touch = min(1.0, mutation_count / max(total * 0.3, 1))

    efficiency = diversity * 0.35 + duration_eff * 0.35 + file_touch * 0.30

    return max(0.0, min(1.0, efficiency)), {
        "r_diversity": round(diversity, 4),
        "r_duration_eff": round(duration_eff, 4),
        "r_file_touch": round(file_touch, 4),
    }


def _max_consecutive_failures(events: List[Dict]) -> int:
    """Count maximum consecutive tool failures."""
    max_run = 0
    current_run = 0
    for e in events:
        if e.get("success") is False:
            current_run += 1
            max_run = max(max_run, current_run)
        else:
            current_run = 0
    return max_run


def compute_advantage(
    record: Dict,
    reward_score: float,
    domain_baseline: Optional[float] = None,
    beta: float = 1.0,
) -> float:
    """
    Compute advantage: A = (reward - baseline) / beta.

    Used by OAPL-Lite for advantage-weighted SFT sampling.
    If no domain baseline provided, uses 0.5 (neutral).
    """
    baseline = domain_baseline if domain_baseline is not None else 0.5
    return (reward_score - baseline) / beta


def backfill_rewards(force: bool = False) -> Dict[str, int]:
    """
    Backfill reward scores for all records in the trajectory store.

    Two-pass algorithm:
      1. Compute domain baselines from existing scored records
      2. Score unscored records and compute advantage vs baseline

    Args:
        force: If True, recompute even for already-scored records

    Returns:
        Stats dict with total, scored, skipped, errors counts
    """
    if not STORE_PATH.exists():
        return {"total": 0, "scored": 0, "skipped": 0, "errors": 0}

    scored = 0
    skipped = 0
    errors = 0

    try:
        # C3: Lock entire read-modify-write cycle to prevent lost writes
        with open(STORE_PATH, "r+") as f:
            fcntl.flock(f.fileno(), fcntl.LOCK_EX)
            try:
                lines = f.readlines()
                updated_lines = []

                # Pass 1: compute domain baselines
                domain_rewards: Dict[str, List[float]] = {}
                for line in lines:
                    try:
                        record = json.loads(line)
                        domain = record.get("skill", {}).get("domain")
                        existing_reward = record.get("outcome", {}).get("reward_score")
                        if existing_reward is not None:
                            bucket = domain or "_global"
                            domain_rewards.setdefault(bucket, []).append(existing_reward)
                    except json.JSONDecodeError:
                        pass

                domain_baselines = {
                    k: sum(v) / len(v) for k, v in domain_rewards.items() if v
                }

                # Pass 2: score and annotate
                for line in lines:
                    try:
                        record = json.loads(line)
                        existing = record.get("outcome", {}).get("reward_score")

                        if existing is not None and not force:
                            skipped += 1
                            updated_lines.append(line)
                            continue

                        reward_data = compute_reward(record)

                        domain = record.get("skill", {}).get("domain") or "_global"
                        baseline = domain_baselines.get(domain, 0.5)
                        advantage = compute_advantage(
                            record, reward_data["reward_score"], baseline
                        )

                        outcome = record.get("outcome", {})
                        outcome["reward_score"] = reward_data["reward_score"]
                        outcome["reward_components"] = reward_data["components"]
                        outcome["outcome_score"] = reward_data["outcome_score"]
                        outcome["process_score"] = reward_data["process_score"]
                        outcome["efficiency_score"] = reward_data["efficiency_score"]
                        outcome["advantage"] = round(advantage, 4)
                        outcome["annotation_status"] = "scored"
                        record["outcome"] = outcome

                        updated_lines.append(json.dumps(record, default=str) + "\n")
                        scored += 1

                    except (json.JSONDecodeError, Exception) as exc:
                        logger.debug("backfill error: %s", exc)
                        errors += 1
                        updated_lines.append(line)

                if scored > 0:
                    f.seek(0)
                    f.writelines(updated_lines)
                    f.truncate()
            finally:
                fcntl.flock(f.fileno(), fcntl.LOCK_UN)
    except Exception as exc:
        logger.warning("backfill_rewards file error: %s", exc)
        errors += 1
        return {"total": 0, "scored": scored, "skipped": skipped, "errors": errors}

    return {
        "total": len(lines),
        "scored": scored,
        "skipped": skipped,
        "errors": errors,
        "domain_baselines": domain_baselines,
    }


def get_reward_stats() -> Dict[str, Any]:
    """Get reward distribution stats across the store."""
    if not STORE_PATH.exists():
        return {}

    rewards: List[float] = []
    by_channel: Dict[str, List[float]] = {}
    by_domain: Dict[str, List[float]] = {}

    try:
        with open(STORE_PATH, "r") as f:
            for line in f:
                try:
                    record = json.loads(line)
                    r = record.get("outcome", {}).get("reward_score")
                    if r is not None:
                        rewards.append(r)
                        ch = record.get("channel", "unknown")
                        by_channel.setdefault(ch, []).append(r)
                        domain = record.get("skill", {}).get("domain") or "_global"
                        by_domain.setdefault(domain, []).append(r)
                except json.JSONDecodeError:
                    continue
    except Exception:
        return {}

    if not rewards:
        return {"scored": 0}

    return {
        "scored": len(rewards),
        "mean": round(sum(rewards) / len(rewards), 4),
        "min": round(min(rewards), 4),
        "max": round(max(rewards), 4),
        "by_channel": {
            k: {"count": len(v), "mean": round(sum(v) / len(v), 4)}
            for k, v in by_channel.items()
        },
        "by_domain": {
            k: {"count": len(v), "mean": round(sum(v) / len(v), 4)}
            for k, v in by_domain.items()
        },
    }
