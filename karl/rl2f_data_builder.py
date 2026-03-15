"""
rl2f_data_builder.py — Multi-turn training data pipeline for KARL RL2F.

Builds SFT/DPO training data from KARL trajectories:
  1. SFT pairs: (prompt, trajectory_summary) for supervised fine-tuning
  2. DPO pairs: (prompt, chosen_trajectory, rejected_trajectory) for preference learning
  3. Correction pairs: (original_bad, corrected_good) from Cortex corrections

Pipeline:
  load_trajectories → filter → pair → format → export

Output formats:
  - JSONL for MLX LoRA (chat template)
  - Parquet for HuggingFace datasets
"""

import json
import hashlib
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

KARL_DIR = Path(__file__).parent
STORE_PATH = KARL_DIR / "trajectories.jsonl"
CORTEX_PATH = Path.home() / ".claude" / "cortex" / "entries.jsonl"
OUTPUT_DIR = KARL_DIR / "training_data"


def _summarize_trajectory(record: Dict) -> str:
    """Create a concise summary of a trajectory for training."""
    events = record.get("trajectory", {}).get("events", [])
    skill = record.get("skill", {})
    total = record.get("trajectory", {}).get("total_tools", 0)

    parts = []
    if skill.get("name"):
        parts.append(f"Task: {skill['name']} ({skill.get('domain', '?')})")

    # Summarize tool sequence (compact)
    tool_sequence = []
    for e in events[:20]:  # Cap at 20 events
        tool = e.get("tool_name", "?")
        params = e.get("key_params", {})
        if tool == "Read":
            fp = params.get("file_path", "")
            tool_sequence.append(f"Read {Path(fp).name}" if fp else "Read")
        elif tool == "Edit":
            fp = params.get("file_path", "")
            tool_sequence.append(f"Edit {Path(fp).name}" if fp else "Edit")
        elif tool == "Write":
            fp = params.get("file_path", "")
            tool_sequence.append(f"Write {Path(fp).name}" if fp else "Write")
        elif tool == "Bash":
            cmd = (params.get("command", "") or "")[:60]
            tool_sequence.append(f"Bash: {cmd}")
        elif tool in ("Grep", "Glob"):
            tool_sequence.append(f"{tool}: {params.get('pattern', '')[:40]}")
        else:
            tool_sequence.append(tool)

    parts.append("Steps:\n" + "\n".join(f"  {i+1}. {s}" for i, s in enumerate(tool_sequence)))

    if total > 20:
        parts.append(f"... ({total - 20} more steps)")

    return "\n".join(parts)


def build_sft_pairs(
    min_reward: float = 0.45,
    max_pairs: int = 200,
) -> List[Dict]:
    """
    Build SFT training pairs: (system, user_prompt, assistant_response).

    Selects high-reward trajectories and formats them as chat turns.
    """
    if not STORE_PATH.exists():
        return []

    pairs = []
    with open(STORE_PATH, "r") as f:
        for line in f:
            try:
                record = json.loads(line)
                reward = record.get("outcome", {}).get("reward_score")
                if reward is None or reward < min_reward:
                    continue

                prompt = record.get("context", {}).get("prompt_text", "")
                if not prompt or len(prompt) < 10:
                    continue

                summary = _summarize_trajectory(record)
                if not summary:
                    continue

                pairs.append({
                    "messages": [
                        {"role": "system", "content": "You are a skilled software engineer agent. Execute tasks efficiently using available tools."},
                        {"role": "user", "content": prompt[:2000]},
                        {"role": "assistant", "content": summary},
                    ],
                    "reward": reward,
                    "domain": record.get("skill", {}).get("domain", "_global"),
                    "session_id": record.get("session_id", "")[:16],
                })
            except json.JSONDecodeError:
                continue

    # Sort by reward descending, cap at max
    pairs.sort(key=lambda x: x["reward"], reverse=True)
    return pairs[:max_pairs]


def build_dpo_pairs(
    reward_gap: float = 0.15,
    max_pairs: int = 100,
) -> List[Dict]:
    """
    Build DPO preference pairs: (prompt, chosen, rejected).

    Finds trajectory pairs in the same domain where reward gap > threshold.
    """
    if not STORE_PATH.exists():
        return []

    by_domain: Dict[str, List[Dict]] = defaultdict(list)
    with open(STORE_PATH, "r") as f:
        for line in f:
            try:
                record = json.loads(line)
                reward = record.get("outcome", {}).get("reward_score")
                if reward is None:
                    continue
                domain = record.get("skill", {}).get("domain") or "_global"
                prompt = record.get("context", {}).get("prompt_text", "")
                if not prompt:
                    continue
                by_domain[domain].append({
                    "record": record,
                    "reward": reward,
                    "prompt": prompt[:2000],
                })
            except json.JSONDecodeError:
                continue

    pairs = []
    for domain, records in by_domain.items():
        if len(records) < 2:
            continue

        # Sort by reward
        sorted_recs = sorted(records, key=lambda x: x["reward"], reverse=True)

        # Pair top with bottom
        for i in range(min(len(sorted_recs) // 2, max_pairs)):
            high = sorted_recs[i]
            low = sorted_recs[-(i + 1)]

            if high["reward"] - low["reward"] < reward_gap:
                continue

            chosen_summary = _summarize_trajectory(high["record"])
            rejected_summary = _summarize_trajectory(low["record"])

            pairs.append({
                "prompt": high["prompt"],
                "chosen": chosen_summary,
                "rejected": rejected_summary,
                "chosen_reward": high["reward"],
                "rejected_reward": low["reward"],
                "domain": domain,
            })

    return pairs[:max_pairs]


def build_correction_pairs(max_pairs: int = 50) -> List[Dict]:
    """
    Build correction pairs from KARL trajectories + Cortex entries.

    Two sources:
      1. Trajectories flagged with correction_detected=True
      2. Cortex correction entries matched by session_id
    """
    if not STORE_PATH.exists():
        return []

    # Source 1: Corrected trajectories
    corrections = []
    with open(STORE_PATH, "r") as f:
        for line in f:
            try:
                record = json.loads(line)
                outcome = record.get("outcome", {})
                if outcome.get("correction_detected"):
                    corrections.append(record)
            except json.JSONDecodeError:
                continue

    # Source 2: Cortex correction entries (enrich with correction text)
    cortex_corrections: Dict[str, str] = {}
    if CORTEX_PATH.exists():
        with open(CORTEX_PATH, "r") as f:
            for line in f:
                try:
                    entry = json.loads(line)
                    if entry.get("type") == "correction":
                        sid = entry.get("session_id", "")
                        text = entry.get("correction_text", "") or entry.get("details", "")
                        if sid and text:
                            cortex_corrections[sid] = text
                except json.JSONDecodeError:
                    continue

    pairs = []
    for record in corrections[:max_pairs]:
        prompt = record.get("context", {}).get("prompt_text", "")
        if not prompt:
            continue

        summary = _summarize_trajectory(record)
        session_id = record.get("session_id", "")

        # Prefer Cortex correction text over generic context
        correction_ctx = cortex_corrections.get(session_id, "")
        if not correction_ctx:
            correction_ctx = record.get("outcome", {}).get("correction_context", "")

        pairs.append({
            "messages": [
                {"role": "system", "content": "You are learning from a correction. The previous approach was wrong."},
                {"role": "user", "content": f"Original task: {prompt[:1000]}\nCorrection: {correction_ctx[:500]}"},
                {"role": "assistant", "content": f"I understand the correction. Here is the improved approach:\n{summary}"},
            ],
            "type": "correction",
            "domain": record.get("skill", {}).get("domain", "_global"),
            "cortex_enriched": bool(cortex_corrections.get(session_id)),
        })

    return pairs


def export_training_data(
    output_dir: Optional[Path] = None,
    sft_min_reward: float = 0.45,
    dpo_gap: float = 0.15,
) -> Dict[str, Any]:
    """
    Export all training data to JSONL files.

    Creates:
      - sft_train.jsonl: SFT training pairs
      - dpo_train.jsonl: DPO preference pairs
      - correction_train.jsonl: Correction pairs
    """
    out = output_dir or OUTPUT_DIR
    out.mkdir(parents=True, exist_ok=True)

    sft = build_sft_pairs(min_reward=sft_min_reward)
    dpo = build_dpo_pairs(reward_gap=dpo_gap)
    corrections = build_correction_pairs()

    def _write_jsonl(path: Path, records: List[Dict]) -> int:
        with open(path, "w") as f:
            for r in records:
                f.write(json.dumps(r, default=str) + "\n")
        return len(records)

    sft_count = _write_jsonl(out / "sft_train.jsonl", sft)
    dpo_count = _write_jsonl(out / "dpo_train.jsonl", dpo)
    corr_count = _write_jsonl(out / "correction_train.jsonl", corrections)

    return {
        "output_dir": str(out),
        "sft_pairs": sft_count,
        "dpo_pairs": dpo_count,
        "correction_pairs": corr_count,
        "total": sft_count + dpo_count + corr_count,
    }


if __name__ == "__main__":
    import sys

    if "--export" in sys.argv:
        result = export_training_data()
        print(f"\nTraining Data Export")
        print(f"{'=' * 40}")
        print(f"  SFT pairs:        {result['sft_pairs']}")
        print(f"  DPO pairs:        {result['dpo_pairs']}")
        print(f"  Correction pairs: {result['correction_pairs']}")
        print(f"  Total:            {result['total']}")
        print(f"  Output: {result['output_dir']}")
    elif "--stats" in sys.argv:
        sft = build_sft_pairs()
        dpo = build_dpo_pairs()
        print(f"Available: {len(sft)} SFT, {len(dpo)} DPO pairs")
        if sft:
            domains = defaultdict(int)
            for p in sft:
                domains[p["domain"]] += 1
            print(f"SFT domains: {dict(domains)}")
    else:
        print("Usage: rl2f_data_builder.py --export | --stats")
