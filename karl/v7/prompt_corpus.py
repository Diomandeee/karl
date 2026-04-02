"""Extract and index user prompts from Claude session JSONL files.

Builds a searchable corpus with TF-IDF nearest-neighbor retrieval. No external deps.
"""
import glob, json, math, os, re
from collections import Counter
from dataclasses import dataclass, asdict
from pathlib import Path

SESSION_DIR = os.path.expanduser("~/.claude/projects/-Users-mohameddiomande")
CORPUS_PATH = os.path.expanduser("~/Desktop/karl/v7-prompt-corpus.jsonl")

SKIP_PATTERNS = [re.compile(p, re.I) for p in [
    r"^You are working on \*\*", r"^\[Request interrupted", r"^<local-command-",
    r"^<command-(name|message|args)>", r"^<local-command-stdout>",
    r"^<local-command-caveat>", r"task-notification", r"^/\w+", r"^<skill-",
]]
SIGNATURE_PHRASES = {k: re.compile(p, re.I) for k, p in {
    "just": r"\bjust\b", "let's": r"\blet'?s\b", "don't": r"\bdon'?t\b",
    "i think": r"\bi think\b", "can we": r"\bcan we\b", "make sure": r"\bmake sure\b",
    "keep in mind": r"\bkeep in mind\b", "figure out": r"\bfigure out\b",
    "go ahead": r"\bgo ahead\b", "try": r"\btry\b",
}.items()}
CATEGORIES = {k: re.compile(p, re.I) for k, p in {
    "approval": r"^(yes|yeah|yep|ok|sure|do it|go|continue|proceed|correct|right)\b",
    "question": r"\?\s*$",
    "imperative": r"^(run|build|deploy|fix|add|create|delete|update|install|push|commit)\b",
    "paste": r"(```|https?://|error:|traceback|file://)",
    "contextual": r"^(so |ok so |now |after |before |since |because )",
    "conversational": r"^(i |we |my |let|what|how|why|where|when|the |this |that )",
}.items()}


@dataclass
class PromptEntry:
    text: str
    length: int
    category: str
    correction_triggers: list[str]
    project_domain: str
    session_id: str


def _extract_text(content) -> str:
    if isinstance(content, str): return content.strip()
    if isinstance(content, list):
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text":
                return item.get("text", "").strip()
    return ""


def _classify(text):
    for cat, pat in CATEGORIES.items():
        if pat.search(text): return cat
    return "conversational"


def _domain_from_cwd(cwd):
    if not cwd: return "unknown"
    parts = Path(cwd).parts
    for i, p in enumerate(parts):
        if p == "Desktop" and i + 1 < len(parts):
            return parts[i + 1].lower().replace(" ", "-")
    return parts[-1].lower() if parts else "unknown"


def build_corpus(session_dir=SESSION_DIR, output_path=CORPUS_PATH) -> list[PromptEntry]:
    """Extract all user prompts from session JSONLs into a corpus."""
    files = sorted(glob.glob(os.path.join(session_dir, "*.jsonl")))
    entries, seen = [], set()
    for fpath in files:
        sid = Path(fpath).stem
        try:
            with open(fpath) as f:
                for line in f:
                    try: obj = json.loads(line)
                    except json.JSONDecodeError: continue
                    if obj.get("type") != "user": continue
                    text = _extract_text(obj.get("message", {}).get("content", ""))
                    text = re.sub(r"^[>$%]\s*", "", text).strip()
                    if not text or len(text) < 2 or any(p.search(text) for p in SKIP_PATTERNS):
                        continue
                    key = text.lower()[:200]
                    if key in seen: continue
                    seen.add(key)
                    triggers = [n for n, p in SIGNATURE_PHRASES.items() if p.search(text)]
                    entries.append(PromptEntry(text, len(text), _classify(text),
                                              triggers, _domain_from_cwd(obj.get("cwd", "")), sid))
        except (OSError, UnicodeDecodeError): continue
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        for e in entries: f.write(json.dumps(asdict(e)) + "\n")
    return entries


_WORD_RE = re.compile(r"[a-z0-9]+")


class CorpusIndex:
    """Simple TF-IDF index over prompt texts."""
    def __init__(self, entries: list[PromptEntry]):
        self.entries, self.doc_count = entries, len(entries)
        self._tf, self._df = [], Counter()
        for e in entries:
            tf = Counter(_WORD_RE.findall(e.text.lower()))
            self._tf.append(tf)
            for term in set(tf): self._df[term] += 1

    def _vec(self, tf):
        return {t: c * math.log((self.doc_count + 1) / (self._df.get(t, 0) + 1))
                for t, c in tf.items()}

    def search(self, query: str, n: int = 3) -> list[tuple[PromptEntry, float]]:
        qv = self._vec(Counter(_WORD_RE.findall(query.lower())))
        scored = []
        for i, e in enumerate(self.entries):
            dv = self._vec(self._tf[i])
            shared = set(qv) & set(dv)
            if not shared: continue
            dot = sum(qv[t] * dv[t] for t in shared)
            ma = math.sqrt(sum(v * v for v in qv.values()))
            mb = math.sqrt(sum(v * v for v in dv.values()))
            if ma and mb: scored.append((e, dot / (ma * mb)))
        scored.sort(key=lambda x: -x[1])
        return scored[:n]


_corpus, _index = None, None


def _load_corpus():
    global _corpus, _index
    if _corpus is not None: return _corpus, _index
    if os.path.exists(CORPUS_PATH):
        entries = []
        with open(CORPUS_PATH) as f:
            for line in f:
                try: entries.append(PromptEntry(**json.loads(line)))
                except (json.JSONDecodeError, TypeError): continue
        if entries:
            _corpus, _index = entries, CorpusIndex(entries)
            return _corpus, _index
    entries = build_corpus()
    _corpus, _index = entries, CorpusIndex(entries)
    return _corpus, _index


def search_similar(query: str, n: int = 3) -> list[tuple[PromptEntry, float]]:
    """Find n most similar real prompts to query text."""
    return _load_corpus()[1].search(query, n)


def get_corpus() -> list[PromptEntry]:
    """Return the full corpus, loading/building if needed."""
    return _load_corpus()[0]


if __name__ == "__main__":
    print("Building prompt corpus...")
    entries = build_corpus()
    cats = Counter(e.category for e in entries)
    lengths = [e.length for e in entries]
    triggers = Counter(t for e in entries for t in e.correction_triggers)
    print(f"Extracted {len(entries)} prompts | Categories: {dict(cats)}")
    print(f"Length: tiny={sum(1 for l in lengths if l<50)}, "
          f"med={sum(1 for l in lengths if 50<=l<200)}, long={sum(1 for l in lengths if l>=200)}")
    print(f"Triggers: {triggers.most_common(7)}")
    for entry, score in search_similar("fix the import error in the config file"):
        print(f"  [{score:.3f}] {entry.text[:100]}")
