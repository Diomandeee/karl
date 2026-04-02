"""Score generated prompts against Mohamed's real prompting distribution.

Returns a StyleScore with individual checks and an overall 0-1 score.
Hard-fails on obvious AI slop or extreme lengths.
"""

import re
from dataclasses import dataclass

# Mohamed's signature phrases (correction signals, not personality)
SIGNATURE_PHRASES = [
    r"\bjust\b", r"\blet'?s\b", r"\bdon'?t\b", r"\bi think\b",
    r"\bcan we\b", r"\bmake sure\b", r"\bkeep in mind\b",
    r"\bfigure out\b", r"\bgo ahead\b", r"\btry\b",
]
_SIG_PATTERNS = [re.compile(p, re.IGNORECASE) for p in SIGNATURE_PHRASES]

# AI slop phrases that Mohamed would never type
AI_FILLERS = [
    r"\bi'?d be happy to\b", r"\blet me\b", r"\bhere'?s\b",
    r"\bcertainly\b", r"\babsolutely\b", r"\bof course\b",
    r"\bi'?ll go ahead and\b", r"\bgreat question\b",
    r"\bdelve\b", r"\bleverage\b", r"\bseamless\b",
    r"\bexcited to\b", r"\bthrilled\b", r"\bgame.?changer\b",
    r"\bcutting.?edge\b", r"\bholistic\b", r"\bsynergy\b",
    r"\bi hope this\b", r"\bas an ai\b", r"\bas a language model\b",
]
_AI_PATTERNS = [re.compile(p, re.IGNORECASE) for p in AI_FILLERS]

# Imperative-starting words
IMPERATIVE_STARTERS = {
    "run", "build", "deploy", "fix", "add", "create", "delete", "update",
    "install", "push", "commit", "check", "test", "start", "stop", "read",
    "write", "set", "get", "list", "show", "open", "close", "move", "copy",
    "remove", "rename", "merge", "rebase", "pull", "fetch", "reset",
    "configure", "enable", "disable", "execute", "implement", "refactor",
}

# Conversational markers Mohamed uses
CONVERSATIONAL_MARKERS = [
    r"\bi\b", r"\bwe\b", r"\bour\b", r"\bmy\b", r"\bso\b",
    r"\bnow\b", r"\bthis\b", r"\bthat\b", r"\bthe\b",
    r"\bwhat\b", r"\bhow\b", r"\bwhy\b", r"\bwhere\b",
    r"\bactually\b", r"\bbasically\b", r"\balso\b",
]
_CONV_PATTERNS = [re.compile(p, re.IGNORECASE) for p in CONVERSATIONAL_MARKERS]


@dataclass
class StyleScore:
    length_ok: bool             # 80-500 preferred, hard fail <20 or >2000
    signature_density: float    # 0-3 per prompt ideal
    signature_count: int
    imperative_ratio: float     # should be <0.20
    conversational_count: int   # at least 1 for prompts >100 chars
    generic_filler_detected: bool
    filler_phrases_found: list[str]
    overall: float              # 0.0 to 1.0

    @property
    def passed(self) -> bool:
        return self.overall >= 0.4


def validate(prompt: str) -> StyleScore:
    """Score a prompt against Mohamed's real distribution."""
    text = prompt.strip()
    length = len(text)
    words = text.split()
    word_count = len(words)

    # --- Length check ---
    if length < 20 or length > 2000:
        length_ok = False
        length_score = 0.0
    elif 80 <= length <= 500:
        length_ok = True
        length_score = 1.0
    elif length < 80:
        length_ok = True
        length_score = 0.6  # short but acceptable
    else:
        length_ok = True
        length_score = 0.5  # long but acceptable

    # --- Signature phrase density ---
    sig_count = sum(1 for p in _SIG_PATTERNS if p.search(text))
    if 0 <= sig_count <= 3:
        sig_score = 1.0
    elif sig_count <= 5:
        sig_score = 0.5
    else:
        sig_score = 0.1  # way too many, sounds forced

    # --- Imperative ratio ---
    imperative_words = 0
    for w in words:
        if w.lower().rstrip(",.!?:;") in IMPERATIVE_STARTERS:
            imperative_words += 1
    imp_ratio = imperative_words / max(word_count, 1)
    if imp_ratio < 0.20:
        imp_score = 1.0
    elif imp_ratio < 0.35:
        imp_score = 0.6
    else:
        imp_score = 0.2  # too robotic

    # --- Conversational markers ---
    conv_count = sum(1 for p in _CONV_PATTERNS if p.search(text))
    if length > 100 and conv_count == 0:
        conv_score = 0.3  # suspicious for a long prompt
    elif conv_count >= 1:
        conv_score = 1.0
    else:
        conv_score = 0.7  # short prompt, ok without markers

    # --- AI filler detection ---
    filler_found = []
    for p in _AI_PATTERNS:
        m = p.search(text)
        if m:
            filler_found.append(m.group())
    filler_detected = len(filler_found) > 0
    filler_score = 0.0 if filler_detected else 1.0

    # --- Overall weighted score ---
    overall = (
        length_score * 0.20
        + sig_score * 0.15
        + imp_score * 0.20
        + conv_score * 0.15
        + filler_score * 0.30  # AI filler is the strongest signal
    )

    # Hard penalties
    if not length_ok:
        overall *= 0.3
    if filler_detected:
        overall *= 0.5

    return StyleScore(
        length_ok=length_ok,
        signature_density=sig_count / max(word_count / 20, 1),
        signature_count=sig_count,
        imperative_ratio=round(imp_ratio, 3),
        conversational_count=conv_count,
        generic_filler_detected=filler_detected,
        filler_phrases_found=filler_found,
        overall=round(min(overall, 1.0), 3),
    )


if __name__ == "__main__":
    test_prompts = [
        # Good Mohamed-style
        "Let's fix that import error in the config file, I think it broke after the refactor",
        "just run the tests and commit if they pass",
        "Can we deploy this to the VM and make sure the NATS connection works?",
        "status",
        "yes",
        "The crash is happening because the recording service starts before the writer queue. "
        "Don't try to fix the queue, just delay the start by 2 seconds.",
        # AI slop (should fail)
        "I'd be happy to help you with that! Let me analyze the codebase and provide a comprehensive solution.",
        "Certainly! Here's a detailed breakdown of the architecture.",
        "I'll go ahead and implement a holistic solution that leverages the existing infrastructure.",
    ]

    for p in test_prompts:
        score = validate(p)
        tag = "PASS" if score.passed else "FAIL"
        print(f"[{tag}] {score.overall:.2f} | {p[:80]}")
        if score.filler_phrases_found:
            print(f"  AI filler: {score.filler_phrases_found}")
