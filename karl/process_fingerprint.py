"""
process_fingerprint.py — Rich process signal extraction from tool trajectories.

Extracts 6 signals from a trajectory:
  1. Tool flow signature: ordered tool-type sequence (e.g., Read→Edit→Bash)
  2. Mutation depth: how many unique files were modified
  3. Research ratio: proportion of Read/Grep/Glob vs Write/Edit
  4. Verification presence: test/build after mutation
  5. Error recovery: did the agent recover from failures?
  6. Scope coherence: did tools operate in a consistent directory?

These signals create a "fingerprint" that characterizes the agent's
problem-solving approach, useful for reward shaping and trajectory clustering.
"""

import json
import math
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Tuple

KARL_DIR = Path(__file__).parent
STORE_PATH = KARL_DIR / "trajectories.jsonl"

# Tool categories
RESEARCH_TOOLS = {"Read", "Grep", "Glob", "WebSearch", "WebFetch"}
MUTATION_TOOLS = {"Write", "Edit", "NotebookEdit"}
EXECUTION_TOOLS = {"Bash"}
PLANNING_TOOLS = {"EnterPlanMode", "ExitPlanMode", "TaskCreate", "TaskUpdate"}


def extract_fingerprint(record: Dict) -> Dict[str, Any]:
    """
    Extract a 6-signal process fingerprint from a trajectory record.

    Returns dict with all 6 signals plus a compact signature string.
    """
    events = record.get("trajectory", {}).get("events", [])
    tool_counts = record.get("trajectory", {}).get("tool_counts", {})
    total = record.get("trajectory", {}).get("total_tools", 0)

    if total == 0:
        return _empty_fingerprint()

    # 1. Tool flow signature: bigram sequence of tool categories
    flow = _tool_flow_signature(events)

    # 2. Mutation depth: unique files modified
    modified_files = set()
    for e in events:
        if e.get("tool_name") in MUTATION_TOOLS:
            fp = e.get("key_params", {}).get("file_path", "")
            if fp:
                modified_files.add(fp)
    mutation_depth = len(modified_files)

    # 3. Research ratio: research tools / total
    research_count = sum(tool_counts.get(t, 0) for t in RESEARCH_TOOLS)
    mutation_count = sum(tool_counts.get(t, 0) for t in MUTATION_TOOLS)
    research_ratio = research_count / total if total > 0 else 0.0

    # 4. Verification presence: test/build commands after any mutation
    has_mutation = mutation_count > 0
    verification = _check_verification(events) if has_mutation else None

    # 5. Error recovery: failures followed by successful retries
    recovery_score = _compute_recovery(events)

    # 6. Scope coherence: directory consistency
    scope_coherence = _compute_scope_coherence(events)

    # Compact signature: R{research}M{mutation}V{verification}E{recovery}
    sig_parts = [
        f"R{research_count}",
        f"M{mutation_depth}",
        f"V{'Y' if verification else 'N'}",
        f"E{recovery_score:.1f}",
    ]
    signature = "".join(sig_parts)

    return {
        "tool_flow": flow,
        "mutation_depth": mutation_depth,
        "modified_files": list(modified_files)[:10],
        "research_ratio": round(research_ratio, 4),
        "research_count": research_count,
        "mutation_count": mutation_count,
        "verification": verification,
        "recovery_score": round(recovery_score, 4),
        "scope_coherence": round(scope_coherence, 4),
        "signature": signature,
        "total_events": total,
    }


def _empty_fingerprint() -> Dict[str, Any]:
    return {
        "tool_flow": [],
        "mutation_depth": 0,
        "modified_files": [],
        "research_ratio": 0.0,
        "research_count": 0,
        "mutation_count": 0,
        "verification": None,
        "recovery_score": 0.0,
        "scope_coherence": 0.0,
        "signature": "R0M0VNE0.0",
        "total_events": 0,
    }


def _categorize_tool(tool_name: str) -> str:
    """Map tool name to category."""
    if tool_name in RESEARCH_TOOLS:
        return "R"
    if tool_name in MUTATION_TOOLS:
        return "M"
    if tool_name in EXECUTION_TOOLS:
        return "X"
    if tool_name in PLANNING_TOOLS:
        return "P"
    return "O"


def _tool_flow_signature(events: List[Dict]) -> List[str]:
    """Extract tool category bigrams (e.g., ['RR', 'RM', 'MX'])."""
    categories = [_categorize_tool(e.get("tool_name", "")) for e in events]
    bigrams = []
    for i in range(len(categories) - 1):
        bigram = categories[i] + categories[i + 1]
        bigrams.append(bigram)
    # Return unique bigrams preserving order
    seen = set()
    unique = []
    for b in bigrams:
        if b not in seen:
            seen.add(b)
            unique.append(b)
    return unique


def _check_verification(events: List[Dict]) -> bool:
    """Check if any execution follows a mutation (build/test after edit)."""
    last_mutation_idx = -1
    for i, e in enumerate(events):
        if e.get("tool_name") in MUTATION_TOOLS:
            last_mutation_idx = i
        elif e.get("tool_name") == "Bash" and last_mutation_idx >= 0:
            cmd = (e.get("key_params", {}).get("command", "") or "").lower()
            if any(kw in cmd for kw in ("test", "build", "check", "lint", "compile")):
                return True
    return False


def _compute_recovery(events: List[Dict]) -> float:
    """
    Score error recovery: failures followed by successful retries.
    Returns 0-1 where 1 = perfect recovery from all errors.
    """
    failures = 0
    recoveries = 0
    in_failure = False

    for e in events:
        if e.get("success") is False:
            if not in_failure:
                failures += 1
                in_failure = True
        elif in_failure and e.get("success") is not False:
            recoveries += 1
            in_failure = False

    if failures == 0:
        return 1.0  # No failures = no recovery needed
    return recoveries / failures


def _compute_scope_coherence(events: List[Dict]) -> float:
    """
    Measure directory coherence: are tools operating in related paths?
    Returns 0-1 where 1 = all files in same directory tree.
    """
    paths = []
    for e in events:
        fp = e.get("key_params", {}).get("file_path", "")
        if fp:
            paths.append(fp)

    if len(paths) < 2:
        return 1.0

    # Find common prefix length
    parts_list = [p.split("/") for p in paths]
    min_depth = min(len(p) for p in parts_list)

    common_depth = 0
    for i in range(min_depth):
        if all(p[i] == parts_list[0][i] for p in parts_list):
            common_depth += 1
        else:
            break

    avg_depth = sum(len(p) for p in parts_list) / len(parts_list)
    return common_depth / avg_depth if avg_depth > 0 else 0.0


def fingerprint_all() -> Dict[str, Any]:
    """Fingerprint all trajectories and compute aggregate stats."""
    if not STORE_PATH.exists():
        return {"total": 0, "fingerprints": []}

    fingerprints = []
    flow_counter: Counter = Counter()

    with open(STORE_PATH, "r") as f:
        for line in f:
            try:
                record = json.loads(line)
                fp = extract_fingerprint(record)
                fp["session_id"] = record.get("session_id", "?")[:16]
                fingerprints.append(fp)
                for bigram in fp["tool_flow"]:
                    flow_counter[bigram] += 1
            except json.JSONDecodeError:
                continue

    verified = sum(1 for f in fingerprints if f["verification"])
    mean_recovery = sum(f["recovery_score"] for f in fingerprints) / max(len(fingerprints), 1)
    mean_coherence = sum(f["scope_coherence"] for f in fingerprints) / max(len(fingerprints), 1)

    return {
        "total": len(fingerprints),
        "verification_rate": round(verified / max(len(fingerprints), 1), 4),
        "mean_recovery": round(mean_recovery, 4),
        "mean_scope_coherence": round(mean_coherence, 4),
        "top_flow_bigrams": flow_counter.most_common(10),
        "signatures": [f["signature"] for f in fingerprints[-10:]],
    }


if __name__ == "__main__":
    stats = fingerprint_all()
    print(f"\nProcess Fingerprint Report")
    print(f"{'=' * 40}")
    print(f"  Total:            {stats['total']}")
    print(f"  Verification rate: {stats['verification_rate']:.1%}")
    print(f"  Mean recovery:     {stats['mean_recovery']:.2f}")
    print(f"  Mean coherence:    {stats['mean_scope_coherence']:.2f}")
    print(f"  Top flow bigrams:  {stats['top_flow_bigrams'][:5]}")
    print(f"  Recent signatures: {stats['signatures'][-5:]}")
