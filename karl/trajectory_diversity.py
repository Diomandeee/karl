"""
trajectory_diversity.py — Measures distribution health of KARL trajectories.

Provides:
  - Gini coefficient: inequality of domain/skill distribution
  - Shannon entropy: information-theoretic diversity
  - Coverage score: fraction of known domains represented
  - Diversity report: combined analysis for training data curation

Low diversity = training collapse risk.
High Gini = one domain dominates, need more exploration.
"""

import json
import math
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Tuple

KARL_DIR = Path(__file__).parent
STORE_PATH = KARL_DIR / "trajectories.jsonl"

# Known domains from skill patterns
KNOWN_DOMAINS = {
    "infra", "ios", "web", "systems", "creative", "ml",
    "data", "knowledge", "automation", "desktop", "frontend",
}


def gini_coefficient(values: List[float]) -> float:
    """
    Compute Gini coefficient for a distribution.
    0 = perfect equality, 1 = maximum inequality.
    """
    if not values or all(v == 0 for v in values):
        return 0.0
    sorted_vals = sorted(values)
    n = len(sorted_vals)
    total = sum(sorted_vals)
    if total == 0:
        return 0.0
    cumulative = 0.0
    gini_sum = 0.0
    for i, v in enumerate(sorted_vals):
        cumulative += v
        gini_sum += (2 * (i + 1) - n - 1) * v
    return gini_sum / (n * total)


def shannon_entropy(counts: Dict[str, int]) -> Tuple[float, float]:
    """
    Compute Shannon entropy and normalized entropy (evenness).

    Returns (entropy, evenness) where evenness is in [0, 1].
    """
    total = sum(counts.values())
    if total == 0 or len(counts) <= 1:
        return 0.0, 0.0

    entropy = 0.0
    for count in counts.values():
        if count > 0:
            p = count / total
            entropy -= p * math.log2(p)

    max_entropy = math.log2(len(counts))
    evenness = entropy / max_entropy if max_entropy > 0 else 0.0

    return round(entropy, 4), round(evenness, 4)


def domain_coverage(counts: Dict[str, int]) -> float:
    """Fraction of known domains that have at least 1 trajectory."""
    present = sum(1 for d in KNOWN_DOMAINS if counts.get(d, 0) > 0)
    return present / len(KNOWN_DOMAINS) if KNOWN_DOMAINS else 0.0


def load_distributions() -> Tuple[Counter, Counter, Counter]:
    """
    Load domain, skill, and reward tier distributions from the store.

    Returns (domain_counts, skill_counts, reward_tiers).
    """
    domain_counts: Counter = Counter()
    skill_counts: Counter = Counter()
    reward_tiers: Counter = Counter()

    if not STORE_PATH.exists():
        return domain_counts, skill_counts, reward_tiers

    with open(STORE_PATH, "r") as f:
        for line in f:
            try:
                record = json.loads(line)
                skill = record.get("skill", {})
                domain = skill.get("domain") or "_global"
                name = skill.get("name") or "_unknown"
                domain_counts[domain] += 1
                skill_counts[name] += 1

                reward = record.get("outcome", {}).get("reward_score")
                if reward is not None:
                    if reward >= 0.6:
                        reward_tiers["high"] += 1
                    elif reward >= 0.4:
                        reward_tiers["mid"] += 1
                    else:
                        reward_tiers["low"] += 1
            except json.JSONDecodeError:
                continue

    return domain_counts, skill_counts, reward_tiers


def diversity_report() -> Dict[str, Any]:
    """
    Generate a comprehensive diversity report.

    Returns dict with all diversity metrics for training data health.
    """
    domain_counts, skill_counts, reward_tiers = load_distributions()

    domain_gini = gini_coefficient(list(domain_counts.values()))
    skill_gini = gini_coefficient(list(skill_counts.values()))

    domain_entropy, domain_evenness = shannon_entropy(dict(domain_counts))
    skill_entropy, skill_evenness = shannon_entropy(dict(skill_counts))

    coverage = domain_coverage(dict(domain_counts))

    total = sum(domain_counts.values())

    # Health assessment
    health = "healthy"
    warnings = []

    if domain_gini > 0.5:
        health = "imbalanced"
        warnings.append(f"Domain Gini {domain_gini:.2f} > 0.5: one domain dominates")

    if coverage < 0.6:
        health = "sparse"
        warnings.append(f"Coverage {coverage:.0%}: too few domains represented")

    if domain_evenness < 0.6:
        warnings.append(f"Domain evenness {domain_evenness:.2f}: distribution is skewed")

    # Find over/under-represented domains
    mean_count = total / max(len(domain_counts), 1)
    over = {d: c for d, c in domain_counts.items() if c > mean_count * 2}
    under = {d: c for d, c in domain_counts.items() if c < mean_count * 0.5}

    return {
        "total_trajectories": total,
        "unique_domains": len(domain_counts),
        "unique_skills": len(skill_counts),
        "domain_gini": round(domain_gini, 4),
        "skill_gini": round(skill_gini, 4),
        "domain_entropy": domain_entropy,
        "domain_evenness": domain_evenness,
        "skill_entropy": skill_entropy,
        "skill_evenness": skill_evenness,
        "domain_coverage": round(coverage, 4),
        "reward_tiers": dict(reward_tiers),
        "health": health,
        "warnings": warnings,
        "over_represented": dict(over),
        "under_represented": dict(under),
        "domain_counts": dict(domain_counts),
    }


if __name__ == "__main__":
    report = diversity_report()
    print(f"\nTrajectory Diversity Report")
    print(f"{'=' * 40}")
    print(f"  Total:     {report['total_trajectories']}")
    print(f"  Domains:   {report['unique_domains']} ({report['domain_coverage']:.0%} coverage)")
    print(f"  Skills:    {report['unique_skills']}")
    print(f"  Domain Gini:    {report['domain_gini']} (0=equal, 1=max inequality)")
    print(f"  Domain Entropy: {report['domain_entropy']} bits (evenness: {report['domain_evenness']})")
    print(f"  Skill Gini:     {report['skill_gini']}")
    print(f"  Reward tiers:   {report['reward_tiers']}")
    print(f"  Health: {report['health']}")
    if report['warnings']:
        for w in report['warnings']:
            print(f"  WARNING: {w}")
    if report['over_represented']:
        print(f"  Over-represented:  {report['over_represented']}")
    if report['under_represented']:
        print(f"  Under-represented: {report['under_represented']}")
