"""
trajectory_filter.py — 6-stage data quality filter for KARL training data.

Filter pipeline:
  1. Length gate: reject trajectories with < 3 or > 200 tool events
  2. Reward floor: reject trajectories with reward < 0.25
  3. Skill check: reject trajectories with no skill label
  4. Diversity check: reject if too many same-file edits (thrashing)
  5. Duplication check: reject near-duplicate trajectories (prompt similarity)
  6. Recency bias: boost recent trajectories in sampling weight

Each stage returns (pass: bool, reason: str). The pipeline short-circuits
on first failure. Trajectories that pass all stages get a quality_score.
"""

import json
import math
import hashlib
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

KARL_DIR = Path(__file__).parent
STORE_PATH = KARL_DIR / "trajectories.jsonl"

# Filter thresholds
MIN_TOOLS = 3
MAX_TOOLS = 200
REWARD_FLOOR = 0.25
MAX_SAME_FILE_EDITS = 5
PROMPT_SIMILARITY_THRESHOLD = 0.85
RECENCY_HALF_LIFE_DAYS = 14.0


class TrajectoryFilter:
    """6-stage trajectory quality filter."""

    def __init__(
        self,
        min_tools: int = MIN_TOOLS,
        max_tools: int = MAX_TOOLS,
        reward_floor: float = REWARD_FLOOR,
        max_same_file: int = MAX_SAME_FILE_EDITS,
        similarity_threshold: float = PROMPT_SIMILARITY_THRESHOLD,
        recency_half_life: float = RECENCY_HALF_LIFE_DAYS,
    ):
        self.min_tools = min_tools
        self.max_tools = max_tools
        self.reward_floor = reward_floor
        self.max_same_file = max_same_file
        self.similarity_threshold = similarity_threshold
        self.recency_half_life = recency_half_life
        self._seen_hashes: set = set()

    def _stage_length(self, record: Dict) -> Tuple[bool, str]:
        """Stage 1: Tool count bounds."""
        total = record.get("trajectory", {}).get("total_tools", 0)
        if total < self.min_tools:
            return False, f"too_short ({total} < {self.min_tools})"
        if total > self.max_tools:
            return False, f"too_long ({total} > {self.max_tools})"
        return True, "ok"

    def _stage_reward(self, record: Dict) -> Tuple[bool, str]:
        """Stage 2: Minimum reward threshold."""
        reward = record.get("outcome", {}).get("reward_score")
        if reward is None:
            return False, "no_reward"
        if reward < self.reward_floor:
            return False, f"low_reward ({reward:.3f} < {self.reward_floor})"
        return True, "ok"

    def _stage_skill(self, record: Dict) -> Tuple[bool, str]:
        """Stage 3: Must have a skill label."""
        skill_name = record.get("skill", {}).get("name")
        if not skill_name:
            return False, "no_skill_label"
        return True, "ok"

    def _stage_diversity(self, record: Dict) -> Tuple[bool, str]:
        """Stage 4: No excessive file thrashing."""
        events = record.get("trajectory", {}).get("events", [])
        edit_counts: Counter = Counter()
        for e in events:
            if e.get("tool_name") in ("Edit", "Write"):
                fp = e.get("key_params", {}).get("file_path", "")
                if fp:
                    edit_counts[fp] += 1

        worst = edit_counts.most_common(1)
        if worst and worst[0][1] > self.max_same_file:
            return False, f"thrashing ({worst[0][0]}: {worst[0][1]}x)"
        return True, "ok"

    def _stage_dedup(self, record: Dict) -> Tuple[bool, str]:
        """Stage 5: Near-duplicate detection via prompt hash."""
        prompt = record.get("prompt", "")
        if not prompt:
            prompt = record.get("context", {}).get("prompt_text", "")

        # Normalize: lowercase, strip whitespace, take first 200 chars
        normalized = " ".join(prompt.lower().split())[:200]
        h = hashlib.sha256(normalized.encode()).hexdigest()[:16]

        if h in self._seen_hashes:
            return False, "near_duplicate"
        self._seen_hashes.add(h)
        return True, "ok"

    def _stage_recency(self, record: Dict) -> Tuple[bool, float]:
        """Stage 6: Recency weight (always passes, returns weight)."""
        import time

        ts = record.get("timing", {}).get("start_ts", 0)
        if isinstance(ts, str):
            return True, 0.5

        age_days = (time.time() - ts) / 86400 if ts > 0 else 30.0
        weight = math.exp(-0.693 * age_days / self.recency_half_life)
        return True, round(max(0.1, weight), 4)

    def filter_one(self, record: Dict) -> Dict[str, Any]:
        """
        Run all 6 stages on a single trajectory.

        Returns dict with:
          - passed: bool
          - quality_score: float (0-1, only if passed)
          - stage_results: list of (stage, passed, detail)
          - recency_weight: float
        """
        stages = [
            ("length", self._stage_length),
            ("reward", self._stage_reward),
            ("skill", self._stage_skill),
            ("diversity", self._stage_diversity),
            ("dedup", self._stage_dedup),
        ]

        stage_results = []
        for name, fn in stages:
            passed, detail = fn(record)
            stage_results.append((name, passed, detail))
            if not passed:
                return {
                    "passed": False,
                    "failed_stage": name,
                    "reason": detail,
                    "stage_results": stage_results,
                    "quality_score": 0.0,
                    "recency_weight": 0.0,
                }

        # Recency (always passes)
        _, recency_weight = self._stage_recency(record)
        stage_results.append(("recency", True, f"weight={recency_weight}"))

        # Quality score: reward * recency * diversity bonus
        reward = record.get("outcome", {}).get("reward_score", 0.5)
        quality = reward * recency_weight

        return {
            "passed": True,
            "quality_score": round(quality, 4),
            "recency_weight": recency_weight,
            "stage_results": stage_results,
        }

    def filter_all(self, records: Optional[List[Dict]] = None) -> Dict[str, Any]:
        """
        Filter all trajectories and return stats.

        If records is None, loads from STORE_PATH.
        """
        if records is None:
            records = []
            if STORE_PATH.exists():
                with open(STORE_PATH, "r") as f:
                    for line in f:
                        try:
                            records.append(json.loads(line))
                        except json.JSONDecodeError:
                            continue

        self._seen_hashes.clear()
        passed = []
        rejected = []
        rejection_reasons: Counter = Counter()

        for record in records:
            result = self.filter_one(record)
            if result["passed"]:
                record["_quality_score"] = result["quality_score"]
                record["_recency_weight"] = result["recency_weight"]
                passed.append(record)
            else:
                rejection_reasons[result.get("failed_stage", "unknown")] += 1
                rejected.append(result)

        return {
            "total": len(records),
            "passed": len(passed),
            "rejected": len(rejected),
            "pass_rate": round(len(passed) / max(len(records), 1), 4),
            "rejection_breakdown": dict(rejection_reasons),
            "records": passed,
        }


if __name__ == "__main__":
    f = TrajectoryFilter()
    result = f.filter_all()
    print(f"\nTrajectory Filter Report")
    print(f"{'=' * 40}")
    print(f"  Total:    {result['total']}")
    print(f"  Passed:   {result['passed']}")
    print(f"  Rejected: {result['rejected']}")
    print(f"  Pass rate: {result['pass_rate']:.1%}")
    print(f"  Rejections: {result['rejection_breakdown']}")
    if result["records"]:
        qualities = [r["_quality_score"] for r in result["records"]]
        print(f"  Quality range: [{min(qualities):.3f}, {max(qualities):.3f}]")
        print(f"  Quality mean:  {sum(qualities)/len(qualities):.3f}")
