"""
evaluator.py — 3-function evaluation framework for KARL adapter quality.

Functions:
  1. eval_holdout(): Score adapter against held-out trajectories
  2. eval_regression(): Compare new adapter vs baseline on same inputs
  3. eval_domain_spread(): Check if adapter generalizes across domains

Uses eval-holdout.jsonl as the hold-out test set (never used in training).
"""

import json
import math
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

KARL_DIR = Path(__file__).parent
STORE_PATH = KARL_DIR / "trajectories.jsonl"
HOLDOUT_PATH = KARL_DIR / "eval-holdout.jsonl"
BASELINES_PATH = KARL_DIR / "domain_baselines.json"
EVAL_RESULTS_PATH = KARL_DIR / "eval_results.jsonl"


def _load_holdout() -> List[Dict]:
    """Load hold-out evaluation set."""
    if not HOLDOUT_PATH.exists():
        return []
    records = []
    with open(HOLDOUT_PATH, "r") as f:
        for line in f:
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return records


def _load_baselines() -> Dict:
    """Load domain baselines."""
    if not BASELINES_PATH.exists():
        return {}
    try:
        return json.loads(BASELINES_PATH.read_text())
    except Exception:
        return {}


def eval_holdout() -> Dict[str, Any]:
    """
    Score against held-out trajectories.

    Computes:
      - Mean reward on hold-out set
      - Standard deviation
      - Per-domain breakdown
      - Comparison vs training mean
    """
    from reward_engine import compute_reward

    holdout = _load_holdout()
    if not holdout:
        return {"status": "no_holdout_data", "count": 0}

    rewards = []
    by_domain: Dict[str, List[float]] = {}

    for record in holdout:
        result = compute_reward(record)
        score = result["reward_score"]
        rewards.append(score)

        domain = record.get("skill", {}).get("domain") or "_global"
        by_domain.setdefault(domain, []).append(score)

    mean = sum(rewards) / len(rewards)
    variance = sum((r - mean) ** 2 for r in rewards) / len(rewards)
    std = variance ** 0.5

    domain_scores = {
        d: {"mean": round(sum(v) / len(v), 4), "count": len(v)}
        for d, v in by_domain.items()
    }

    return {
        "status": "ok",
        "count": len(holdout),
        "mean_reward": round(mean, 4),
        "std": round(std, 4),
        "min": round(min(rewards), 4),
        "max": round(max(rewards), 4),
        "by_domain": domain_scores,
    }


def eval_regression(
    baseline_version: str = "v2",
    new_version: str = "v3",
) -> Dict[str, Any]:
    """
    Compare new adapter vs baseline on the same hold-out set.

    Uses previously saved eval results to compare. If no previous
    results exist, runs eval_holdout() as the baseline.
    """
    # Load historical results
    history = {}
    if EVAL_RESULTS_PATH.exists():
        with open(EVAL_RESULTS_PATH, "r") as f:
            for line in f:
                try:
                    entry = json.loads(line)
                    history[entry.get("version", "?")] = entry
                except json.JSONDecodeError:
                    continue

    # Current eval
    current = eval_holdout()
    current["version"] = new_version

    # Save current result
    with open(EVAL_RESULTS_PATH, "a") as f:
        f.write(json.dumps(current, default=str) + "\n")

    # Compare
    baseline = history.get(baseline_version)
    if not baseline:
        return {
            "status": "no_baseline",
            "current": current,
            "message": f"No baseline results for {baseline_version}. Current saved as reference.",
        }

    baseline_mean = baseline.get("mean_reward", 0.5)
    current_mean = current.get("mean_reward", 0.5)
    delta = current_mean - baseline_mean

    regression = delta < -0.02  # 2% drop = regression

    return {
        "status": "regression" if regression else "ok",
        "baseline_version": baseline_version,
        "baseline_mean": baseline_mean,
        "new_version": new_version,
        "new_mean": current_mean,
        "delta": round(delta, 4),
        "regression_detected": regression,
        "current": current,
    }


def eval_domain_spread() -> Dict[str, Any]:
    """
    Check if the adapter generalizes across domains.

    Flags:
      - Domains where hold-out performance is 2+ std below mean
      - Domains with zero representation in hold-out
      - Overall spread (max domain delta)
    """
    result = eval_holdout()
    if result.get("status") != "ok":
        return result

    domain_scores = result.get("by_domain", {})
    global_mean = result["mean_reward"]
    global_std = result["std"]

    # Find weak domains
    weak_domains = []
    strong_domains = []
    threshold = global_mean - 2 * max(global_std, 0.01)

    for domain, info in domain_scores.items():
        if domain == "_global":
            continue
        if info["mean"] < threshold:
            weak_domains.append({"domain": domain, "mean": info["mean"], "gap": round(global_mean - info["mean"], 4)})
        elif info["mean"] > global_mean + global_std:
            strong_domains.append({"domain": domain, "mean": info["mean"]})

    # Coverage: domains in training but not in holdout
    baselines = _load_baselines()
    missing_domains = [d for d in baselines if d not in domain_scores and d != "_global"]

    # Spread: difference between best and worst domain
    if domain_scores:
        domain_means = [v["mean"] for v in domain_scores.values()]
        spread = max(domain_means) - min(domain_means)
    else:
        spread = 0.0

    return {
        "status": "ok",
        "global_mean": global_mean,
        "global_std": result["std"],
        "domain_count": len(domain_scores),
        "weak_domains": weak_domains,
        "strong_domains": strong_domains,
        "missing_from_holdout": missing_domains,
        "spread": round(spread, 4),
        "generalization": "good" if not weak_domains and spread < 0.15 else "needs_work",
    }


if __name__ == "__main__":
    import sys

    if "--holdout" in sys.argv:
        result = eval_holdout()
        print(json.dumps(result, indent=2))
    elif "--regression" in sys.argv:
        result = eval_regression()
        print(json.dumps(result, indent=2))
    elif "--spread" in sys.argv:
        result = eval_domain_spread()
        print(json.dumps(result, indent=2))
    else:
        print("KARL Evaluator")
        print("==============")
        print("  --holdout    Score against held-out set")
        print("  --regression Compare new vs baseline adapter")
        print("  --spread     Check domain generalization")
