"""
flow_sampler.py — Distribution-proportional trajectory sampling for KARL.

Implements FlowRL-style sampling: instead of uniform random, samples proportional
to domain distribution to prevent training collapse on overrepresented domains.

Usage:
    from flow_sampler import sample_batch, get_domain_distribution
    batch = sample_batch(batch_size=32, strategy="balanced")
"""

import json
import math
import random
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional

KARL_DIR = Path(__file__).parent
STORE_PATH = KARL_DIR / "trajectories.jsonl"
HOLDOUT_PATH = KARL_DIR / "eval-holdout.jsonl"


def _load_records(exclude_holdout: bool = True) -> List[Dict]:
    """Load all trajectory records, optionally excluding hold-out set."""
    if not STORE_PATH.exists():
        return []

    holdout_ids = set()
    if exclude_holdout and HOLDOUT_PATH.exists():
        with open(HOLDOUT_PATH) as f:
            for line in f:
                try:
                    holdout_ids.add(json.loads(line).get("id"))
                except json.JSONDecodeError:
                    continue

    records = []
    with open(STORE_PATH) as f:
        for line in f:
            try:
                r = json.loads(line)
                if r.get("id") not in holdout_ids:
                    records.append(r)
            except json.JSONDecodeError:
                continue
    return records


def get_domain_distribution(records: Optional[List[Dict]] = None) -> Dict[str, int]:
    """Get trajectory count per domain."""
    if records is None:
        records = _load_records()
    counts = Counter()
    for r in records:
        domain = r.get("skill", {}).get("domain") or "_global"
        counts[domain] += 1
    return dict(counts)


def sample_batch(
    batch_size: int = 32,
    strategy: str = "balanced",
    min_reward: Optional[float] = None,
    exclude_holdout: bool = True,
) -> List[Dict]:
    """
    Sample a batch of trajectories.

    Strategies:
      - "uniform": plain random sampling
      - "balanced": equal samples per domain (FlowRL distribution matching)
      - "advantage": weighted by advantage score (positive advantage oversampled)
      - "top_k": highest reward trajectories only
    """
    records = _load_records(exclude_holdout=exclude_holdout)

    if min_reward is not None:
        records = [
            r for r in records
            if (r.get("outcome", {}).get("reward_score") or 0) >= min_reward
        ]

    if not records:
        return []

    batch_size = min(batch_size, len(records))

    if strategy == "uniform":
        return random.sample(records, batch_size)

    elif strategy == "balanced":
        return _sample_balanced(records, batch_size)

    elif strategy == "advantage":
        return _sample_advantage_weighted(records, batch_size)

    elif strategy == "top_k":
        sorted_recs = sorted(
            records,
            key=lambda r: r.get("outcome", {}).get("reward_score", 0),
            reverse=True,
        )
        return sorted_recs[:batch_size]

    else:
        return random.sample(records, batch_size)


def _sample_balanced(records: List[Dict], batch_size: int) -> List[Dict]:
    """
    FlowRL balanced sampling: equal representation per domain.

    Domains with fewer samples get oversampled (with replacement),
    domains with more get undersampled.
    """
    by_domain: Dict[str, List[Dict]] = {}
    for r in records:
        domain = r.get("skill", {}).get("domain") or "_global"
        by_domain.setdefault(domain, []).append(r)

    n_domains = len(by_domain)
    if n_domains == 0:
        return []

    per_domain = max(1, batch_size // n_domains)
    remainder = batch_size - (per_domain * n_domains)

    batch = []
    for domain, recs in by_domain.items():
        n = per_domain
        # Distribute remainder to smallest domains first
        if remainder > 0 and len(recs) <= per_domain:
            n += 1
            remainder -= 1
        if len(recs) >= n:
            batch.extend(random.sample(recs, n))
        else:
            # Oversample with replacement
            batch.extend(random.choices(recs, k=n))

    random.shuffle(batch)
    return batch[:batch_size]


def _sample_advantage_weighted(records: List[Dict], batch_size: int) -> List[Dict]:
    """
    Sample weighted by advantage score.

    Positive advantage trajectories are oversampled, negative undersampled.
    Uses softmax-like temperature scaling.
    """
    advantages = []
    for r in records:
        adv = r.get("outcome", {}).get("advantage", 0.0)
        advantages.append(adv)

    if not advantages:
        return random.sample(records, batch_size)

    # Temperature-scaled softmax weights
    temperature = 2.0
    max_adv = max(advantages) if advantages else 0
    weights = []
    for a in advantages:
        w = math.exp((a - max_adv) / temperature)
        weights.append(max(w, 0.01))  # Floor to avoid zero probability

    return random.choices(records, weights=weights, k=batch_size)


def export_sft_batch(
    batch: List[Dict],
    output_path: Optional[Path] = None,
) -> Path:
    """
    Export a trajectory batch as SFT training data.

    Each record becomes: {"prompt": ..., "completion": ..., "weight": ...}
    """
    if output_path is None:
        output_path = KARL_DIR / "sft_batch.jsonl"

    with open(output_path, "w") as f:
        for r in batch:
            prompt = r.get("context", {}).get("prompt_text", "")
            tool_seq = r.get("trajectory", {}).get("tool_sequence", [])
            reward = r.get("outcome", {}).get("reward_score", 0.5)
            advantage = r.get("outcome", {}).get("advantage", 0.0)

            sft_record = {
                "prompt": prompt[:1000],
                "completion": " → ".join(tool_seq[:20]),
                "weight": max(0.1, 1.0 + advantage),
                "reward": reward,
                "domain": r.get("skill", {}).get("domain", "_global"),
                "skill": r.get("skill", {}).get("name"),
            }
            f.write(json.dumps(sft_record) + "\n")

    return output_path


class FlowRLSampler:
    """Wrapper class for sft_exporter.py integration."""

    def __init__(self, store_path: Path = None):
        self.store_path = store_path or STORE_PATH

    def sample(self, strategy: str = "balanced", size: int = 80) -> List[Dict]:
        records = _load_records()
        return sample_batch(batch_size=size, strategy=strategy)

    def stats(self) -> Dict[str, int]:
        return get_domain_distribution()


if __name__ == "__main__":
    import sys

    if "--dist" in sys.argv:
        dist = get_domain_distribution()
        total = sum(dist.values())
        print(f"Domain distribution ({total} trajectories):")
        for domain, count in sorted(dist.items(), key=lambda x: -x[1]):
            pct = count / total * 100
            print(f"  {domain:20s}: {count:3d} ({pct:5.1f}%)")

    elif "--sample" in sys.argv:
        strategy = "balanced"
        batch_size = 32
        for i, arg in enumerate(sys.argv):
            if arg == "--strategy" and i + 1 < len(sys.argv):
                strategy = sys.argv[i + 1]
            if arg == "--size" and i + 1 < len(sys.argv):
                batch_size = int(sys.argv[i + 1])

        batch = sample_batch(batch_size=batch_size, strategy=strategy)
        print(f"Sampled {len(batch)} trajectories (strategy={strategy}):")
        dist = Counter(r.get("skill", {}).get("domain", "_global") for r in batch)
        for domain, count in sorted(dist.items(), key=lambda x: -x[1]):
            print(f"  {domain}: {count}")

    elif "--export" in sys.argv:
        strategy = "balanced"
        batch = sample_batch(batch_size=32, strategy=strategy)
        path = export_sft_batch(batch)
        print(f"Exported {len(batch)} records to {path}")

    else:
        print("Usage: flow_sampler.py --dist | --sample [--strategy balanced|uniform|advantage|top_k] [--size N] | --export")
