#!/usr/bin/env python3
"""Mine correction signals from 5,448+ real prompts.

Extracts DPO-style pairs: (context, correction_trigger, what_should_have_happened).
Each signature phrase maps to a failure mode that training data should pre-empt.
"""

import json
import os
import glob
import re
from dataclasses import dataclass, asdict
from pathlib import Path

SESSION_DIR = Path.home() / ".claude" / "projects" / "-Users-mohameddiomande"

CORRECTION_PATTERNS = {
    "just": {
        "regex": r"\bjust\b",
        "failure": "over_explain",
        "meaning": "Model talked when it should have acted. Pre-face: execute directly.",
        "preface": "Execute directly. Don't explain unless asked.",
    },
    "don't": {
        "regex": r"\bdon'?t\b",
        "failure": "wrong_direction",
        "meaning": "Model was doing the wrong thing. Pre-face: state what NOT to do.",
        "preface": None,  # Extracted from context — the thing after "don't"
    },
    "let's": {
        "regex": r"\blet'?s\b",
        "failure": "stalled",
        "meaning": "Model stalled or asked permission. Pre-face: propose and execute.",
        "preface": "Propose a plan and start executing. Don't ask for permission.",
    },
    "i want you to": {
        "regex": r"\bi want you to\b",
        "failure": "off_track",
        "meaning": "Model went off-track, needed redirect. Pre-face: state goal explicitly.",
        "preface": None,  # Extracted — the task after "I want you to"
    },
    "make sure": {
        "regex": r"\bmake sure\b",
        "failure": "corner_cut",
        "meaning": "Model was about to skip a quality gate. Pre-face: state the constraint.",
        "preface": None,  # Extracted — the constraint after "make sure"
    },
    "keep in mind": {
        "regex": r"\bkeep in mind\b",
        "failure": "missing_context",
        "meaning": "Model lacked context it should have known. Pre-face: surface constraint.",
        "preface": None,  # Extracted — the context after "keep in mind"
    },
    "figure out": {
        "regex": r"\bfigure out\b",
        "failure": "shallow",
        "meaning": "Model gave surface answer. Pre-face: investigate deeply before acting.",
        "preface": "Investigate the root cause before proposing a fix.",
    },
    "why is": {
        "regex": r"\bwhy is\b",
        "failure": "no_root_cause",
        "meaning": "Model fixed symptom not cause. Pre-face: explain root cause first.",
        "preface": "Explain the root cause before fixing.",
    },
    "it's not working": {
        "regex": r"\bit'?s not working\b",
        "failure": "incomplete_fix",
        "meaning": "Model's previous fix didn't work. Pre-face: verify before claiming done.",
        "preface": "Test your changes before saying they're done.",
    },
    "can we": {
        "regex": r"\bcan we\b",
        "failure": "no_proposal",
        "meaning": "User had to suggest what model should have proposed.",
        "preface": "Suggest next steps proactively.",
    },
}


@dataclass
class CorrectionPair:
    session_id: str
    turn_index: int
    trigger_phrase: str
    failure_mode: str
    user_prompt: str           # The correction prompt Mohamed typed
    preface_constraint: str    # What should have been pre-faced
    context_before: str        # Previous assistant output (what triggered correction)
    extracted_directive: str   # The actual task extracted from the correction


def extract_directive(prompt: str, trigger: str) -> str:
    """Extract the actual directive from a correction prompt."""
    lower = prompt.lower()
    idx = lower.find(trigger)
    if idx >= 0:
        after = prompt[idx + len(trigger):].strip()
        # Take up to the next sentence boundary
        end = min(
            (after.find(".") if after.find(".") > 0 else 999),
            (after.find(",") if after.find(",") > 20 else 999),
            200,
        )
        return after[:end].strip()
    return prompt[:200]


def mine_sessions() -> list[CorrectionPair]:
    """Mine all Claude Code sessions for correction pairs."""
    session_files = sorted(glob.glob(str(SESSION_DIR / "*.jsonl")))
    print(f"Scanning {len(session_files)} session files...")

    pairs = []
    total_prompts = 0

    for sf in session_files:
        session_id = Path(sf).stem
        turns = []

        try:
            with open(sf) as f:
                for line in f:
                    if not line.strip():
                        continue
                    d = json.loads(line)
                    msg_type = d.get("type", "")

                    if msg_type == "user":
                        content = d.get("message", {}).get("content", [])
                        text = ""
                        if isinstance(content, list):
                            for part in content:
                                if isinstance(part, dict) and part.get("type") == "text":
                                    text = part.get("text", "")
                                    break
                        elif isinstance(content, str):
                            text = content
                        if text and len(text) > 10:
                            turns.append({"role": "user", "text": text})
                            total_prompts += 1

                    elif msg_type == "assistant":
                        content = d.get("message", {}).get("content", [])
                        text = ""
                        if isinstance(content, list):
                            for part in content:
                                if isinstance(part, dict) and part.get("type") == "text":
                                    text += part.get("text", "") + " "
                        elif isinstance(content, str):
                            text = content
                        if text:
                            turns.append({"role": "assistant", "text": text[:500]})

        except (json.JSONDecodeError, KeyError):
            continue

        # Scan user turns for correction triggers
        for i, turn in enumerate(turns):
            if turn["role"] != "user":
                continue

            prompt = turn["text"]

            # Skip system-injected prompts
            if any(prompt.startswith(x) for x in [
                "<task-notification", "<command-", "Base directory",
                "This session is being continued", "You are now executing",
            ]):
                continue

            for phrase, config in CORRECTION_PATTERNS.items():
                if re.search(config["regex"], prompt.lower()[:300]):
                    # Find the assistant turn before this
                    context = ""
                    for j in range(i - 1, -1, -1):
                        if turns[j]["role"] == "assistant":
                            context = turns[j]["text"][:300]
                            break

                    directive = extract_directive(prompt, phrase)
                    preface = config["preface"] or directive

                    pairs.append(CorrectionPair(
                        session_id=session_id,
                        turn_index=i,
                        trigger_phrase=phrase,
                        failure_mode=config["failure"],
                        user_prompt=prompt[:500],
                        preface_constraint=preface[:200],
                        context_before=context,
                        extracted_directive=directive[:200],
                    ))

    print(f"Scanned {total_prompts} prompts across {len(session_files)} sessions")
    print(f"Extracted {len(pairs)} correction pairs")
    return pairs


def save_pairs(pairs: list[CorrectionPair], output_path: str):
    """Save correction pairs as JSONL."""
    with open(output_path, "w") as f:
        for p in pairs:
            f.write(json.dumps(asdict(p)) + "\n")
    print(f"Saved to {output_path}")


def summarize(pairs: list[CorrectionPair]):
    """Print distribution summary."""
    from collections import Counter
    triggers = Counter(p.trigger_phrase for p in pairs)
    failures = Counter(p.failure_mode for p in pairs)

    print(f"\nCorrection triggers:")
    for phrase, count in triggers.most_common():
        print(f"  {phrase:20s}: {count}")

    print(f"\nFailure modes:")
    for mode, count in failures.most_common():
        print(f"  {mode:20s}: {count}")

    # Sample pairs
    print(f"\nSample pairs:")
    import random
    random.seed(42)
    for p in random.sample(pairs, min(5, len(pairs))):
        print(f"  [{p.failure_mode}] trigger='{p.trigger_phrase}'")
        print(f"    Prompt: {p.user_prompt[:120]}")
        print(f"    Preface: {p.preface_constraint[:120]}")
        print()


if __name__ == "__main__":
    pairs = mine_sessions()
    out = os.path.expanduser("~/Desktop/karl/v6-correction-pairs.jsonl")
    save_pairs(pairs, out)
    summarize(pairs)
