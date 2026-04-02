"""3-gram hash ring for repetition detection + escape prompt rotation."""

import re

ESCAPE_PROMPTS = {
    "EXPLORE": [
        "what files are in this project? list the directory structure",
        "read the main entry point and explain what this project does",
        "what dependencies does this project use?",
        "show me the most recently modified files",
        "what tests exist? run them",
        "can we look at the config files and see how this is set up?",
        "I think we should start by reading the README or any docs",
        "let's check git log and see what changed recently",
        "what's the project structure? I want to understand the architecture",
        "figure out which files are most important and read those first",
    ],
    "BUILD": [
        "walk me through the current error in detail",
        "try a completely different approach to this problem",
        "what's the simplest possible fix? do that first",
        "add a test for what you just built",
        "check if the build passes. if not, fix the first error only",
        "now implement the next piece we haven't touched yet",
        "can we refactor what you just wrote to be cleaner?",
        "I think the types are wrong here. check and fix them",
        "make sure the error handling covers edge cases too",
        "let's add some logging so we can debug this later",
    ],
    "CLOSE": [
        "run the full test suite and show results",
        "commit everything with a descriptive message",
        "list what was accomplished this session",
        "check for any obvious bugs or missing error handling",
        "clean up any temporary files or debug output",
        "push it and let's see the final state",
        "make sure nothing is broken before we wrap up",
        "let's do one final review of the changes",
        "status",
        "run a quick smoke test to verify the main flow works",
    ],
    "STUCK": [
        "ok let's try something completely different",
        "what exactly is the error? show me the full trace",
        "can we simplify the approach? I think we're overcomplicating this",
        "let's step back and think about what we're actually trying to do",
        "figure out why this isn't working. read the relevant source code",
        "I think the issue is upstream. check the dependencies",
        "just skip this part for now and move to the next task",
        "try the fix without the fancy parts. get the basics working first",
    ],
}

# Track rotation index per phase
_rotation = {"EXPLORE": 0, "BUILD": 0, "CLOSE": 0}

STOPWORDS = {
    "about", "actually", "after", "again", "already", "also", "and", "another",
    "any", "around", "back", "basics", "before", "being", "but", "can", "check",
    "claude", "code", "completely", "current", "different", "dont", "figure",
    "first", "for", "from", "get", "going", "have", "here", "into", "issue",
    "just", "keep", "lets", "look", "make", "move", "need", "next", "now",
    "open", "over", "overcomplicating", "part", "right", "run", "same", "session",
    "should", "show", "simplify", "something", "step", "that", "the", "then",
    "there", "they", "thing", "this", "trace", "trying", "what", "why", "with",
    "work", "working", "would",
}


def extract_ngrams(text: str, n: int = 3) -> set[str]:
    """Extract character n-grams from normalized text."""
    text = text.lower().strip()
    words = text.split()
    if len(words) < n:
        return {text}
    return {' '.join(words[i:i+n]) for i in range(len(words) - n + 1)}


def jaccard_similarity(a: set, b: set) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def extract_keywords(text: str) -> set[str]:
    """Extract stable lexical anchors for semantic repetition checks."""
    keywords = set()
    for token in re.findall(r"[a-z0-9][a-z0-9.-]*", text.lower()):
        if len(token) <= 3 or token in STOPWORDS:
            continue
        keywords.add(token)
    return keywords


def is_semantic_stuck_loop(
    prompt: str,
    recent_prompts: list[str],
    threshold: int = 3,
    min_shared_keywords: int = 2,
) -> bool:
    """Detect stuck loops where wording changes but intent does not."""
    if len(recent_prompts) < threshold:
        return False

    prompt_keywords = extract_keywords(prompt)
    if not prompt_keywords:
        return False

    common = set(prompt_keywords)
    for prev in recent_prompts[-threshold:]:
        prev_keywords = extract_keywords(prev)
        if not prev_keywords:
            return False
        common &= prev_keywords
        if len(common) < min_shared_keywords:
            return False

    return True


def is_duplicate(
    prompt: str,
    recent_prompts: list[str],
    threshold: float = 0.4,
    phase: str = "",
) -> bool:
    """Check if prompt is too similar to any recent prompt."""
    if not recent_prompts:
        return False

    prompt_grams = extract_ngrams(prompt)

    # Exact match
    if prompt.strip().lower() in [p.strip().lower() for p in recent_prompts]:
        return True

    # Fuzzy match via 3-gram Jaccard
    for prev in recent_prompts[-15:]:
        prev_grams = extract_ngrams(prev)
        if jaccard_similarity(prompt_grams, prev_grams) > threshold:
            return True

    if phase == "STUCK" and is_semantic_stuck_loop(prompt, recent_prompts):
        return True

    return False


def get_escape_prompt(phase: str) -> str:
    """Get the next escape prompt for this phase, rotating through the pool."""
    phase = phase if phase in ESCAPE_PROMPTS else "BUILD"
    pool = ESCAPE_PROMPTS[phase]
    idx = _rotation.get(phase, 0)
    prompt = pool[idx % len(pool)]
    _rotation[phase] = idx + 1
    return prompt


DESTRUCTIVE_PATTERNS = [
    "rm -rf", "git push --force", "git push -f", "drop table",
    "git reset --hard", "DELETE FROM", "kill -9",
    "git checkout .", "git clean -f",
]


def is_destructive(prompt: str) -> bool:
    """Block prompts that could cause data loss."""
    lower = prompt.lower()
    return any(pat.lower() in lower for pat in DESTRUCTIVE_PATTERNS)
