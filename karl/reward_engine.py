"""
reward_engine.py — Multi-signal reward computation for KARL trajectories.

Computes a composite reward score from three signal levels:
  1. Outcome reward  — cross-turn signals (correction, redo, session continuation)
  2. Process reward  — within-turn signals (exit codes, error rate, tool success)
  3. Efficiency score — duration, tool count, file modification patterns

The composite reward is a weighted blend:
  reward = 0.40 * outcome + 0.35 * process + 0.25 * efficiency

All scores normalized to [0, 1]. Higher = better trajectory.

Usage:
    from reward_engine import compute_reward, backfill_rewards
    score = compute_reward(trajectory_record)
    backfill_rewards()  # annotate all pending records
"""

import fcntl
import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

KARL_DIR = Path(__file__).parent
STORE_PATH = KARL_DIR / "trajectories.jsonl"

# Weight coefficients for composite reward (6-signal)
W_OUTCOME = 0.25
W_PROCESS = 0.22
W_EFFICIENCY = 0.13
W_VERIFICATION = 0.13
W_CONSISTENCY = 0.13
W_MOTION = 0.14  # Wasted motion penalty (inspired by TPO linearity)


def compute_reward(record: Dict) -> Dict[str, Any]:
    """
    Compute multi-signal reward for a trajectory record.

    Returns dict with:
      - reward_score: float [0, 1] composite
      - outcome_score: float [0, 1]
      - process_score: float [0, 1]
      - efficiency_score: float [0, 1]
      - components: dict with individual signal values
    """
    outcome = record.get("outcome", {})
    trajectory = record.get("trajectory", {})
    timing = record.get("timing", {})

    # --- Outcome Score (cross-turn signals) ---
    outcome_score, outcome_components = _compute_outcome(outcome)

    # --- Process Score (within-turn signals) ---
    process_score, process_components = _compute_process(trajectory)

    # --- Efficiency Score ---
    efficiency_score, efficiency_components = _compute_efficiency(
        trajectory, timing
    )

    # --- Verification Score ---
    verification_score, verification_components = _compute_verification(trajectory)

    # --- Consistency Score ---
    consistency_score, consistency_components = _compute_consistency(trajectory)

    # --- Wasted Motion Score (TPO-inspired linearity) ---
    motion_score, motion_components = _compute_wasted_motion(trajectory)

    # --- Composite (6-signal) ---
    composite = (
        W_OUTCOME * outcome_score
        + W_PROCESS * process_score
        + W_EFFICIENCY * efficiency_score
        + W_VERIFICATION * verification_score
        + W_CONSISTENCY * consistency_score
        + W_MOTION * motion_score
    )

    return {
        "reward_score": round(composite, 4),
        "outcome_score": round(outcome_score, 4),
        "process_score": round(process_score, 4),
        "efficiency_score": round(efficiency_score, 4),
        "verification_score": round(verification_score, 4),
        "consistency_score": round(consistency_score, 4),
        "motion_score": round(motion_score, 4),
        "components": {
            **outcome_components,
            **process_components,
            **efficiency_components,
            **verification_components,
            **consistency_components,
        },
    }


def _compute_outcome(outcome: Dict) -> Tuple[float, Dict]:
    """
    Outcome reward from cross-turn signals.

    Signals (when available):
      - correction_detected: False=+0.35, True=-0.35, None=0
      - redo_detected: False=+0.25, True=-0.25, None=0
      - build_success: True=+0.20, False=-0.10, None=0
      - session_continued: True=+0.20, False=0, None=0

    Score is sum of available signals, normalized to [0, 1].
    """
    score = 0.5  # Base: neutral when no signals available
    available = 0

    correction = outcome.get("correction_detected")
    redo = outcome.get("redo_detected")
    build = outcome.get("build_success")
    continued = outcome.get("session_continued")

    components = {}

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
    Process reward from within-turn signals with temporal weighting.

    Components:
      - Tool success rate (temporally weighted): later steps count more
      - Bash error rate: 1 - (bash_errors / bash_count)
      - Error density: penalize high failure concentration
    """
    total = trajectory.get("total_tools", 0)
    successes = trajectory.get("successes", 0)
    failures = trajectory.get("failures", 0)
    bash_errors = trajectory.get("bash_errors", 0)

    if total == 0:
        return 0.5, {"r_success_rate": 0.5, "r_bash_clean": 1.0, "r_error_density": 1.0, "r_temporal_weight": 0.5}

    events = trajectory.get("events", [])

    # Temporally weighted success rate: later steps matter more
    # Weight grows linearly from 0.5 (first step) to 1.5 (last step)
    weighted_success = 0.0
    weight_sum = 0.0
    for i, e in enumerate(events):
        w = 0.5 + (i / max(len(events) - 1, 1))  # [0.5, 1.5]
        weight_sum += w
        if e.get("success") is not False:
            weighted_success += w
    success_rate = weighted_success / weight_sum if weight_sum > 0 else 0.5

    # Bash cleanliness (only count bash tools)
    bash_count = sum(1 for e in events if e.get("tool_name") == "Bash")
    bash_clean = 1.0
    if bash_count > 0:
        bash_clean = 1.0 - (bash_errors / bash_count)

    # Error density: consecutive failures are worse than scattered ones
    error_density = 1.0
    if failures > 0:
        max_consecutive = _max_consecutive_failures(events)
        # Penalize: 3+ consecutive failures is bad
        if max_consecutive >= 3:
            error_density = max(0.0, 1.0 - (max_consecutive - 2) * 0.15)

    # Late-stage failure penalty: failures in the last 25% of events are extra costly
    late_start = max(0, len(events) - max(1, len(events) // 4))
    late_events = events[late_start:]
    late_failures = sum(1 for e in late_events if e.get("success") is False)
    late_penalty = max(0.0, 1.0 - late_failures * 0.15) if late_events else 1.0

    # Weighted blend (added temporal weight component)
    process = success_rate * 0.40 + bash_clean * 0.25 + error_density * 0.20 + late_penalty * 0.15

    return max(0.0, min(1.0, process)), {
        "r_success_rate": round(success_rate, 4),
        "r_bash_clean": round(bash_clean, 4),
        "r_error_density": round(error_density, 4),
        "r_temporal_weight": round(late_penalty, 4),
    }


def _compute_efficiency(trajectory: Dict, timing: Dict) -> Tuple[float, Dict]:
    """
    Efficiency score from trajectory shape.

    Components:
      - Tool diversity: more diverse = usually better (read→edit→test vs 50x Bash)
      - Duration efficiency: very long sessions with few tools = inefficient
      - File touch rate: proportion of Write/Edit vs Read-only
    """
    total = trajectory.get("total_tools", 0)
    tool_counts = trajectory.get("tool_counts", {})
    events = trajectory.get("events", [])
    duration = timing.get("duration_s")

    if total == 0:
        return 0.5, {"r_diversity": 0.5, "r_duration_eff": 0.5, "r_file_touch": 0.5}

    # Tool diversity: Shannon entropy normalized by log(total unique)
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

    # Duration efficiency: tools per minute (ideal: 2-8 tools/min)
    duration_eff = 0.5
    if duration and duration > 0:
        tools_per_min = (total / duration) * 60
        if tools_per_min >= 2 and tools_per_min <= 8:
            duration_eff = 1.0
        elif tools_per_min > 8:
            duration_eff = 0.8  # Fast is fine
        elif tools_per_min >= 1:
            duration_eff = 0.6
        else:
            duration_eff = max(0.2, 0.5 - (1.0 - tools_per_min) * 0.3)

    # File touch rate: proportion of mutation tools
    mutation_tools = {"Write", "Edit", "NotebookEdit"}
    mutation_count = sum(
        c for t, c in tool_counts.items() if t in mutation_tools
    )
    file_touch = min(1.0, mutation_count / max(total * 0.3, 1))

    # Weighted blend
    efficiency = diversity * 0.35 + duration_eff * 0.35 + file_touch * 0.30

    return max(0.0, min(1.0, efficiency)), {
        "r_diversity": round(diversity, 4),
        "r_duration_eff": round(duration_eff, 4),
        "r_file_touch": round(file_touch, 4),
    }


def _compute_verification(trajectory: Dict) -> Tuple[float, Dict]:
    """
    Verification reward: did the agent verify its work?

    Checks for:
      - Test execution (pytest, npm test, cargo test, etc.)
      - Build verification (xcodebuild, cargo build, make)
      - Read-after-write (read a file that was just edited)
      - Git diff/status after changes
    """
    events = trajectory.get("events", [])
    tool_counts = trajectory.get("tool_counts", {})
    total = trajectory.get("total_tools", 0)

    if total == 0:
        return 0.5, {"r_has_test": 0.0, "r_has_build": 0.0, "r_read_after_write": 0.0}

    has_mutation = any(t in tool_counts for t in ("Write", "Edit", "NotebookEdit"))

    # Test detection
    has_test = False
    has_build = False
    wrote_files = set()
    read_after_write = False

    for e in events:
        tool = e.get("tool_name", "")
        params = e.get("key_params", {})
        cmd = (params.get("command", "") or "").lower()

        if tool == "Bash":
            if any(kw in cmd for kw in ("pytest", "npm test", "bun test", "cargo test", "go test", "make test")):
                has_test = True
            if any(kw in cmd for kw in ("xcodebuild", "cargo build", "make ", "npm run build", "bun build")):
                has_build = True

        if tool in ("Write", "Edit"):
            fp = params.get("file_path", "")
            if fp:
                wrote_files.add(fp)

        if tool == "Read" and wrote_files:
            fp = params.get("file_path", "")
            if fp in wrote_files:
                read_after_write = True

    # Score: verification only matters if there were mutations
    if not has_mutation:
        return 0.6, {"r_has_test": 0.0, "r_has_build": 0.0, "r_read_after_write": 0.0, "r_no_mutation": 1.0}

    test_score = 1.0 if has_test else 0.0
    build_score = 1.0 if has_build else 0.0
    raw_score = 0.3 if read_after_write else 0.0

    verification = test_score * 0.4 + build_score * 0.3 + raw_score * 0.3

    return max(0.0, min(1.0, verification)), {
        "r_has_test": test_score,
        "r_has_build": build_score,
        "r_read_after_write": 1.0 if read_after_write else 0.0,
    }


def _compute_consistency(trajectory: Dict) -> Tuple[float, Dict]:
    """
    Internal consistency score: does the trajectory follow a coherent plan?

    Checks for:
      - Read-before-write: did the agent read a file before editing it?
      - Tool transition coherence: logical tool ordering (Read→Edit, not Edit→Read→Edit)
      - Avoid thrashing: same file edited multiple times in quick succession
    """
    events = trajectory.get("events", [])
    total = trajectory.get("total_tools", 0)

    if total < 3:
        return 0.6, {"r_read_before_write": 0.5, "r_no_thrashing": 1.0}

    # Read-before-write score
    read_files = set()
    writes_with_read = 0
    writes_total = 0
    for e in events:
        tool = e.get("tool_name", "")
        fp = e.get("key_params", {}).get("file_path", "")
        if tool == "Read" and fp:
            read_files.add(fp)
        elif tool in ("Edit", "Write") and fp:
            writes_total += 1
            if fp in read_files:
                writes_with_read += 1

    rbw_score = writes_with_read / writes_total if writes_total > 0 else 0.5

    # Thrashing: same file edited 3+ times
    edit_counts: Dict[str, int] = {}
    for e in events:
        if e.get("tool_name") in ("Edit", "Write"):
            fp = e.get("key_params", {}).get("file_path", "")
            if fp:
                edit_counts[fp] = edit_counts.get(fp, 0) + 1

    thrash_files = sum(1 for c in edit_counts.values() if c >= 3)
    no_thrash = 1.0 if thrash_files == 0 else max(0.0, 1.0 - thrash_files * 0.2)

    consistency = rbw_score * 0.6 + no_thrash * 0.4

    return max(0.0, min(1.0, consistency)), {
        "r_read_before_write": round(rbw_score, 4),
        "r_no_thrashing": round(no_thrash, 4),
    }


def _compute_wasted_motion(trajectory: Dict) -> Tuple[float, Dict]:
    """
    Wasted motion penalty inspired by TPO's linearity score.

    In linear agent sessions, "branching" manifests as:
      - Tool retries: same tool called 3+ times in a row (agent stuck in a loop)
      - Read-without-act: reading files without editing (exploration that goes nowhere)
      - Error retry loops: Bash fail → Bash fail → Bash fail (not learning from errors)
      - Undo patterns: Write then Edit same file immediately (didn't think first)

    Score: 1.0 = clean forward motion, 0.0 = pure thrashing.
    Formula: exp(-lambda * waste_count) where waste_count is total wasted actions.
    """
    events = trajectory.get("events", [])
    total = trajectory.get("total_tools", 0)

    if total < 3:
        return 0.8, {"r_retry_loops": 0, "r_read_waste": 0.0, "r_error_loops": 0, "r_undo": 0}

    # 1. Tool retry loops: same tool 3+ times consecutively
    retry_loops = 0
    streak = 1
    for i in range(1, len(events)):
        curr = events[i].get("tool_name", "")
        prev = events[i - 1].get("tool_name", "")
        if curr == prev and curr:
            streak += 1
            if streak >= 3:
                retry_loops += 1
        else:
            streak = 1

    # 2. Read waste: reads that never lead to writes on the same file
    read_files = []
    written_files = set()
    for e in events:
        tool = e.get("tool_name", "")
        fp = e.get("key_params", {}).get("file_path", "")
        if tool == "Read" and fp:
            read_files.append(fp)
        elif tool in ("Write", "Edit") and fp:
            written_files.add(fp)
    reads_without_write = sum(1 for f in read_files if f not in written_files)
    read_waste = reads_without_write / max(len(read_files), 1) if read_files else 0.0

    # 3. Error retry loops: consecutive Bash failures
    error_loops = 0
    bash_fail_streak = 0
    for e in events:
        if e.get("tool_name") == "Bash" and e.get("success") is False:
            bash_fail_streak += 1
            if bash_fail_streak >= 2:
                error_loops += 1
        else:
            bash_fail_streak = 0

    # 4. Undo patterns: Write/Edit same file within 2 steps
    undo_count = 0
    for i in range(1, min(len(events), len(events))):
        curr = events[i]
        prev = events[i - 1]
        if (curr.get("tool_name") in ("Edit", "Write") and
                prev.get("tool_name") in ("Write", "Edit")):
            curr_fp = curr.get("key_params", {}).get("file_path", "")
            prev_fp = prev.get("key_params", {}).get("file_path", "")
            if curr_fp and curr_fp == prev_fp:
                undo_count += 1

    # Composite: exp(-lambda * waste)
    waste = retry_loops * 2 + error_loops * 1.5 + undo_count * 1 + read_waste * total * 0.3
    lam = 0.15
    score = math.exp(-lam * waste)

    return max(0.0, min(1.0, score)), {
        "r_retry_loops": retry_loops,
        "r_read_waste": round(read_waste, 4),
        "r_error_loops": error_loops,
        "r_undo": undo_count,
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
    domain_std: Optional[float] = None,
) -> float:
    """
    Compute z-score advantage: A = (reward - baseline) / max(std, beta).

    Uses domain standard deviation when available (z-score normalization),
    falls back to beta scaling. This normalizes advantage across domains
    with different reward variances.
    """
    baseline = domain_baseline if domain_baseline is not None else 0.5
    divisor = domain_std if domain_std and domain_std > 0.01 else beta
    return (reward_score - baseline) / divisor


def backfill_rewards(force: bool = False) -> Dict[str, int]:
    """
    Backfill reward scores for all records in trajectories.jsonl.

    Args:
        force: If True, recompute even for already-scored records.

    Returns stats dict.
    """
    if not STORE_PATH.exists():
        return {"total": 0, "scored": 0, "skipped": 0, "errors": 0}

    lines = []
    try:
        with open(STORE_PATH, "r") as f:
            lines = f.readlines()
    except Exception:
        return {"total": 0, "scored": 0, "skipped": 0, "errors": 0}

    scored = 0
    skipped = 0
    errors = 0
    updated_lines = []

    # First pass: compute domain baselines
    domain_rewards = {}
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

    # Compute global mean for Bayesian smoothing
    all_rewards = [r for v in domain_rewards.values() for r in v]
    global_mean = sum(all_rewards) / len(all_rewards) if all_rewards else 0.5

    # Bayesian smoothed baselines: blend domain mean toward global mean
    # for sparse domains. strength=10 means 10 "virtual" observations at
    # global mean. A domain with 100 trajs gets ~91% domain, ~9% global.
    # A domain with 1 traj gets ~9% domain, ~91% global.
    SMOOTH_STRENGTH = 10
    domain_baselines = {}
    for k, v in domain_rewards.items():
        if v:
            raw_mean = sum(v) / len(v)
            n = len(v)
            domain_baselines[k] = (n * raw_mean + SMOOTH_STRENGTH * global_mean) / (n + SMOOTH_STRENGTH)
        else:
            domain_baselines[k] = global_mean

    # Compute domain standard deviations for z-score advantage
    domain_stds = {}
    for k, v in domain_rewards.items():
        if len(v) >= 5:
            mean = sum(v) / len(v)
            variance = sum((x - mean) ** 2 for x in v) / len(v)
            domain_stds[k] = max(variance ** 0.5, 0.02)
        else:
            # Sparse domains: use global std as fallback to prevent wild z-scores
            global_var = sum((x - global_mean) ** 2 for x in all_rewards) / len(all_rewards) if all_rewards else 0.04
            domain_stds[k] = max(global_var ** 0.5, 0.02)

    # Second pass: compute and annotate
    for line in lines:
        try:
            record = json.loads(line)
            existing = record.get("outcome", {}).get("reward_score")

            if existing is not None and not force:
                skipped += 1
                updated_lines.append(line)
                continue

            reward_data = compute_reward(record)

            # Compute z-score advantage
            domain = record.get("skill", {}).get("domain") or "_global"
            baseline = domain_baselines.get(domain, 0.5)
            std = domain_stds.get(domain, 1.0)
            advantage = compute_advantage(
                record, reward_data["reward_score"], baseline, domain_std=std
            )

            # Update record
            outcome = record.get("outcome", {})
            outcome["reward_score"] = reward_data["reward_score"]
            outcome["reward_components"] = reward_data["components"]
            outcome["outcome_score"] = reward_data["outcome_score"]
            outcome["process_score"] = reward_data["process_score"]
            outcome["efficiency_score"] = reward_data["efficiency_score"]
            outcome["verification_score"] = reward_data.get("verification_score")
            outcome["consistency_score"] = reward_data.get("consistency_score")
            outcome["advantage"] = round(advantage, 4)
            outcome["annotation_status"] = "scored"
            record["outcome"] = outcome

            updated_lines.append(json.dumps(record, default=str) + "\n")
            scored += 1

        except (json.JSONDecodeError, Exception) as e:
            errors += 1
            updated_lines.append(line)

    # Write back
    if scored > 0:
        try:
            with open(STORE_PATH, "w") as f:
                fcntl.flock(f.fileno(), fcntl.LOCK_EX)
                try:
                    f.writelines(updated_lines)
                finally:
                    fcntl.flock(f.fileno(), fcntl.LOCK_UN)
        except Exception as e:
            print(f"[reward] Write error: {e}", file=sys.stderr)
            errors += 1

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

    rewards = []
    by_channel = {}
    by_domain = {}

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


if __name__ == "__main__":
    if "--backfill" in sys.argv:
        force = "--force" in sys.argv
        print("[reward] Backfilling rewards...")
        stats = backfill_rewards(force=force)
        print(f"[reward] Done: {stats}")
    elif "--stats" in sys.argv:
        stats = get_reward_stats()
        print(f"\nReward Distribution:")
        print(f"  Scored: {stats.get('scored', 0)}")
        print(f"  Mean:   {stats.get('mean', 'N/A')}")
        print(f"  Range:  [{stats.get('min', 'N/A')}, {stats.get('max', 'N/A')}]")
        print(f"  By channel: {stats.get('by_channel', {})}")
        print(f"  By domain:  {stats.get('by_domain', {})}")
    else:
        print("Usage: reward_engine.py --backfill [--force] | --stats")
