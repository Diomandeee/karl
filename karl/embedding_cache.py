"""
embedding_cache.py — Async embedding cache for KARL skill routing.

Provides:
  - LRU cache for prompt embeddings (max 500, 24h TTL)
  - Pickle persistence at ~/.claude/karl/prompt_embedding_cache.pkl
  - Async embed via RAG++ /api/rag/embed (non-blocking, fires and forgets)
  - Skill embedding loader from local pickle + Supabase fallback
  - Cosine similarity computation (no numpy dependency)

Design constraint: hook budget is 500ms. Embedding API takes 300-500ms.
So we embed ASYNC and use the cache for the NEXT prompt, not the current one.
"""

import hashlib
import math
import os
import pickle
import subprocess
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

KARL_DIR = Path(__file__).parent
CACHE_PATH = KARL_DIR / "prompt_embedding_cache.pkl"
SKILL_CACHE_PATH = KARL_DIR / "skill_embeddings.pkl"
RAG_EMBED_URL = os.environ.get("RAG_EMBED_URL", "http://localhost:8000/api/rag/embed")

# Cache config
MAX_ENTRIES = 500
TTL_SECONDS = 86400  # 24 hours
EMBEDDING_DIM = 3072  # gemini-embedding-001

# In-memory cache: {hash: (embedding, timestamp)}
_cache: Dict[str, Tuple[List[float], float]] = {}
_cache_loaded = False
_skill_cache: Dict[str, Tuple[List[float], float]] = {}  # {name: (vector, weight)}
_skill_cache_loaded = False
_skill_cache_mtime: float = 0.0


def cache_key(text: str) -> str:
    """Generate cache key from text."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def _load_cache() -> None:
    """Load prompt embedding cache from disk."""
    global _cache, _cache_loaded
    if _cache_loaded:
        return
    if CACHE_PATH.exists():
        try:
            with open(CACHE_PATH, "rb") as f:
                _cache = pickle.load(f)
            # Evict expired entries
            now = time.time()
            expired = [k for k, (_, ts) in _cache.items() if now - ts > TTL_SECONDS]
            for k in expired:
                del _cache[k]
        except Exception:
            _cache = {}
    _cache_loaded = True


def _save_cache() -> None:
    """Persist prompt embedding cache to disk."""
    try:
        CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(CACHE_PATH, "wb") as f:
            pickle.dump(_cache, f)
    except Exception:
        pass


def cache_get(text: str) -> Optional[List[float]]:
    """Get cached embedding for text. Returns None on miss."""
    _load_cache()
    key = cache_key(text)
    entry = _cache.get(key)
    if entry is None:
        return None
    embedding, ts = entry
    if time.time() - ts > TTL_SECONDS:
        del _cache[key]
        return None
    return embedding


def cache_store(text: str, embedding: List[float]) -> None:
    """Store embedding in cache."""
    _load_cache()
    key = cache_key(text)
    _cache[key] = (embedding, time.time())
    # Evict oldest if over limit
    if len(_cache) > MAX_ENTRIES:
        oldest = min(_cache.items(), key=lambda x: x[1][1])
        del _cache[oldest[0]]
    _save_cache()


def _fetch_embedding(text: str, timeout: float = 8) -> Optional[List[float]]:
    """Fetch embedding from RAG++ endpoint. Returns embedding or None."""
    try:
        import json
        import urllib.request
        req = urllib.request.Request(
            RAG_EMBED_URL,
            data=json.dumps({"text": text[:4000]}).encode("utf-8"),
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = json.loads(resp.read())
            embedding = data.get("embedding", [])
            if embedding and len(embedding) == EMBEDDING_DIM:
                return embedding
    except Exception:
        pass
    return None


def embed_async(text: str) -> None:
    """Fire-and-forget: compute embedding in background thread, store in cache.

    This is called from the hook after regex routing completes.
    The embedding will be available for the NEXT prompt via cache_get().
    """
    def _do_embed():
        embedding = _fetch_embedding(text, timeout=8)
        if embedding:
            cache_store(text, embedding)

    t = threading.Thread(target=_do_embed, daemon=True)
    t.start()


def embed_sync(text: str, timeout: float = 0.4) -> Optional[List[float]]:
    """Synchronous embed: fetch embedding within budget, cache it, return it.

    Used for real-time vector routing where we need the result NOW.
    Falls back to cache if the API call would exceed the timeout.
    """
    # Check cache first
    cached = cache_get(text)
    if cached is not None:
        return cached

    # Synchronous fetch with tight timeout
    embedding = _fetch_embedding(text, timeout=timeout)
    if embedding:
        cache_store(text, embedding)
        return embedding

    return None


def compute_adaptive_timeout() -> Dict[str, Any]:
    """Compute optimal embed timeout from shadow routing latency data.

    Reads elapsed_ms from routing_shadow.jsonl, computes p50/p95,
    and suggests a timeout = p95 + 50ms headroom, clamped to [0.2, 1.0].
    """
    import json as _json
    shadow_path = KARL_DIR / "routing_shadow.jsonl"
    if not shadow_path.exists():
        return {"status": "no_data", "recommended_timeout": 0.4}

    latencies = []
    with open(shadow_path) as f:
        for line in f:
            try:
                r = _json.loads(line)
                ms = r.get("elapsed_ms")
                if ms and ms > 0:
                    latencies.append(ms)
            except _json.JSONDecodeError:
                continue

    if len(latencies) < 10:
        return {"status": "insufficient_data", "samples": len(latencies), "recommended_timeout": 0.4}

    latencies.sort()
    p50 = latencies[len(latencies) // 2]
    p95_idx = int(len(latencies) * 0.95)
    p95 = latencies[min(p95_idx, len(latencies) - 1)]

    # Recommended timeout = p95 + 50ms headroom, clamped [200ms, 1000ms]
    recommended_ms = min(max(p95 + 50, 200), 1000)
    recommended_s = round(recommended_ms / 1000, 3)

    # Read current budget from config
    current_budget_ms = 350
    config_path = KARL_DIR / "config.json"
    try:
        if config_path.exists():
            import json as _j
            cfg = _j.loads(config_path.read_text())
            et = cfg.get("embed_timeout")
            if et:
                current_budget_ms = int(et * 1000)
    except Exception:
        pass

    return {
        "status": "ok",
        "samples": len(latencies),
        "p50_ms": round(p50, 1),
        "p95_ms": round(p95, 1),
        "recommended_timeout": recommended_s,
        "current_budget_ms": current_budget_ms,
    }


def build_prompt_embedding_text(prompt: str, cwd: str = "") -> str:
    """Build the text to embed for a prompt, with project context."""
    project = os.path.basename(cwd) if cwd else ""
    if project:
        return f"[project:{project}] {prompt[:3900]}"
    return prompt[:4000]


# ---------------------------------------------------------------------------
# Skill Embeddings
# ---------------------------------------------------------------------------


def load_skill_embeddings() -> Dict[str, Tuple[List[float], float]]:
    """Load skill embeddings from local pickle cache.

    Returns: {skill_name: (embedding_vector, trajectory_weight)}
    Mtime-based caching — reloads when file changes.
    """
    global _skill_cache, _skill_cache_loaded, _skill_cache_mtime

    if not SKILL_CACHE_PATH.exists():
        return {}

    mtime = SKILL_CACHE_PATH.stat().st_mtime
    if _skill_cache_loaded and mtime == _skill_cache_mtime:
        return _skill_cache

    try:
        with open(SKILL_CACHE_PATH, "rb") as f:
            _skill_cache = pickle.load(f)
        _skill_cache_mtime = mtime
        _skill_cache_loaded = True
    except Exception:
        _skill_cache = {}
        _skill_cache_loaded = True

    return _skill_cache


def save_skill_embeddings(embeddings: Dict[str, Tuple[List[float], float]]) -> None:
    """Persist skill embeddings to local pickle."""
    try:
        SKILL_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(SKILL_CACHE_PATH, "wb") as f:
            pickle.dump(embeddings, f)
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Centroid Refresh
# ---------------------------------------------------------------------------

# Skill descriptions for embedding centroids — maps skill name to text
SKILL_DESCRIPTIONS = {
    "spore": "iOS SwiftUI app Spore plant growth journaling CloudKit haptics",
    "openclaw-hub": "iOS SwiftUI OpenClawHub mesh control dashboard glasses gateway feed aggregator",
    "creative-director": "iOS SwiftUI CreativeDirector video recording teleprompter content creation",
    "securiclaw": "iOS SwiftUI SecuriClaw home security camera app face recognition motion detection bilateral fusion stealth mode wake word",
    "speakflow": "iOS SwiftUI SpeakFlow speech fluency reading mode teleprompter",
    "serenity-soother": "iOS SwiftUI Serenity Soother therapeutic ASMR ambient sound",
    "firstdate": "iOS SwiftUI FirstDate dating conversation practice",
    "bwb": "iOS SwiftUI BWB wine bar coffee shop point of sale kiosk customer",
    "ios-build": "iOS Xcode Swift project build compilation deployment",
    "learnnko": "Web app LearnNKo language learning NKo script writing system",
    "milkmen": "Web Milk Men delivery route optimization Koatji plant-based milk e-commerce",
    "cc-dashboard": "Web dashboard Comp-Core system monitoring administration",
    "nexus-portal": "Next.js web Nexus Portal mesh monitoring service health dashboard",
    "evo-cubed": "Creative evolution Evo Cubed recursive ideation generation refinement",
    "hef-evolution": "Creative HEF evolution technique generation distribution",
    "frameworks": "Creative frameworks methodology design idea generation",
    "cortex-ops": "Infrastructure Cortex behavioral intelligence operations configuration",
    "karl-trajectory": "Infrastructure KARL reinforcement learning trajectory capture reward engine",
    "hook-maintenance": "Infrastructure Claude Code hooks PostToolUse PreToolUse maintenance debugging",
    "monitoring-ops": "Infrastructure monitoring Prometheus Grafana Docker alerting dashboards",
    "mesh-dashboard": "Infrastructure mesh network dashboard node health task queue visualization",
    "mesh-ops": "Infrastructure mesh node operations Syncthing Tailscale connectivity sync",
    "deploy-ops": "Infrastructure deployment Docker systemd service rollout container management",
    "skill-forge": "Infrastructure Claude Code skill forging auto-trigger registry management",
    "pane-orchestrator": "Infrastructure pane orchestrator tmux terminal session management",
    "account-pool": "Infrastructure multi-tenant account pool rotation rate limit management",
    "mesh-node-agent": "Infrastructure mesh node agent distributed task execution reconciliation",
    "screen-capture": "Infrastructure screen capture pane recording session documentation",
    "evolution-world": "Systems Evolution World framework template generation graduation pipeline",
    "comp-core": "Systems Comp-Core monorepo Rust TypeScript Python multi-project cc-graph-kernel cc-discord-gateway RAG rag-plusplus agent orchestration",
    "self-healing-code": "Systems self-healing code automatic error detection repair pipeline",
    "creator-shield": "Machine learning Creator Shield toxicity scoring content moderation NLP",
    "agent-intelligence": "Machine learning Agent Intelligence behavioral analysis registry",
    "nko-brain-scanner": "Machine learning NKo Brain Scanner handwriting OCR recognition training",
    "feed-hub-flow": "Automation Prefect feed hub flows scheduled tasks content pipeline sync",
    "supabase-ops": "Database Supabase schema migrations RLS policies query optimization",
    "discrawl": "Data Discrawl Discord message archival search indexing",
    "vault-ops": "Knowledge Obsidian vault writing notes organization daily journal",
    "vault-writer": "Knowledge Obsidian vault API writer programmatic note creation",
    "tauri-desktop": "Desktop Tauri cross-platform application Rust web frontend",
    "authoring-session": "General code authoring writing new files creating features implementing",
    "mixed-session": "Mixed multi-topic session switching between multiple projects research browsing general questions no single project focus",
    "shell-session": "Generic shell terminal bash git status diff log system admin scripting infrastructure debugging no specific project",
    "test": "Testing session running tests debugging failures verifying correctness",
}


def _gather_exemplar_prompts(
    store_path: Path, max_per_skill: int = 5
) -> Dict[str, List[str]]:
    """Gather real trajectory prompts per skill for exemplar-based centroids.

    Returns {skill_name: [prompt1, prompt2, ...]} with up to max_per_skill prompts.
    Selects the longest prompts (most distinctive) per skill.
    """
    import json as _json

    # Boilerplate prefixes to strip or skip — these are shared across all skills
    # and pull centroids toward a common point
    BOILERPLATE_PREFIXES = [
        "# ENRICHED SUB-AGENT TASK",
        "## INHERITED CONTEXT",
        "You are a sub-agent working on a Pulse",
    ]

    skill_prompts: Dict[str, List[Tuple[int, str]]] = {}
    if not store_path.exists():
        return {}

    with open(store_path) as f:
        for line in f:
            try:
                r = _json.loads(line)
                name = r.get("skill", {}).get("name")
                if not name:
                    continue
                prompt = (
                    r.get("context", {}).get("prompt_text", "")
                    or r.get("context", {}).get("user_prompt", "")
                    or ""
                )
                if len(prompt) < 20:
                    continue

                # Strip enriched sub-agent boilerplate — extract the actual task
                clean_prompt = prompt
                if any(prompt.strip().startswith(bp) for bp in BOILERPLATE_PREFIXES):
                    # Try to extract the actual task after "TASK:" or "## Task"
                    for marker in ["TASK:", "## Task", "### Task", "Your task:"]:
                        idx = prompt.find(marker)
                        if idx >= 0:
                            clean_prompt = prompt[idx:].strip()
                            break
                    else:
                        # No task marker found — skip this entirely
                        continue

                # Also incorporate cwd context for distinctiveness
                cwd = r.get("context", {}).get("cwd", "")
                if cwd:
                    project = cwd.rstrip("/").split("/")[-1]
                    if project and project not in (".", "~", "mohameddiomande"):
                        clean_prompt = f"[{project}] {clean_prompt}"

                if len(clean_prompt) < 20:
                    continue

                if name not in skill_prompts:
                    skill_prompts[name] = []
                skill_prompts[name].append((len(clean_prompt), clean_prompt[:4000]))
            except _json.JSONDecodeError:
                continue

    result = {}
    for name, entries in skill_prompts.items():
        # Deduplicate by first 100 chars to avoid near-identical exemplars
        seen_prefixes = set()
        unique_entries = []
        for length, prompt in entries:
            prefix = prompt[:100].lower()
            if prefix not in seen_prefixes:
                seen_prefixes.add(prefix)
                unique_entries.append((length, prompt))

        # Sort by length descending (longer = more distinctive context)
        unique_entries.sort(key=lambda x: -x[0])
        result[name] = [p for _, p in unique_entries[:max_per_skill]]
    return result


def _average_vectors(vectors: List[List[float]]) -> List[float]:
    """Average a list of embedding vectors element-wise."""
    if not vectors:
        return []
    dim = len(vectors[0])
    avg = [0.0] * dim
    for vec in vectors:
        for i in range(dim):
            avg[i] += vec[i]
    n = len(vectors)
    return [x / n for x in avg]


def refresh_skill_centroids(
    store_path: Optional[Path] = None,
    timeout_per_skill: float = 10,
    use_exemplars: bool = True,
    max_exemplars: int = 5,
) -> Dict[str, Any]:
    """Rebuild skill_embeddings.pkl using exemplar-based centroids.

    For each skill:
    1. Embed the skill description (always)
    2. If use_exemplars=True, also embed up to max_exemplars real prompts
    3. Average all embeddings to form the centroid

    This produces more distinctive centroids than description-only.
    Returns summary dict with success count, errors, etc.
    """
    import json as _json
    from collections import Counter

    if store_path is None:
        store_path = KARL_DIR / "trajectories.jsonl"

    # Count trajectories per skill
    skill_counts: Counter = Counter()
    if store_path.exists():
        with open(store_path) as f:
            for line in f:
                try:
                    r = _json.loads(line)
                    name = r.get("skill", {}).get("name")
                    if name:
                        skill_counts[name] += 1
                except _json.JSONDecodeError:
                    continue

    # Gather exemplar prompts if enabled
    exemplar_prompts = {}
    if use_exemplars:
        exemplar_prompts = _gather_exemplar_prompts(store_path, max_exemplars)

    embeddings = {}
    errors = []
    exemplar_stats = {}
    for skill_name, description in SKILL_DESCRIPTIONS.items():
        count = skill_counts.get(skill_name, 0)
        weight = max(0.5, math.log2(count + 1)) if count > 0 else 0.5

        desc_vec = _fetch_embedding(description, timeout=timeout_per_skill)
        if not desc_vec:
            errors.append(skill_name)
            continue

        exemplar_vecs = []
        skill_exemplars = exemplar_prompts.get(skill_name, [])
        for prompt in skill_exemplars:
            vec = _fetch_embedding(prompt, timeout=timeout_per_skill)
            if vec:
                exemplar_vecs.append(vec)
            time.sleep(0.05)

        all_vecs = [desc_vec] + exemplar_vecs
        centroid = _average_vectors(all_vecs)
        embeddings[skill_name] = (centroid, round(weight, 2))
        exemplar_stats[skill_name] = len(exemplar_vecs)

    # Drift detection: compare old vs new centroids
    drift_report = {}
    drifted_skills = []
    old_embeddings = load_skill_embeddings()
    if old_embeddings and embeddings:
        for skill_name, (new_vec, new_weight) in embeddings.items():
            if skill_name in old_embeddings:
                old_vec, old_weight = old_embeddings[skill_name]
                sim = cosine_similarity(old_vec, new_vec)
                drift = 1.0 - sim
                drift_report[skill_name] = round(drift, 4)
                if drift > 0.15:
                    drifted_skills.append({"skill": skill_name, "drift": round(drift, 4)})

    if embeddings:
        save_skill_embeddings(embeddings)
        global _skill_cache_loaded
        _skill_cache_loaded = False

    return {
        "success": len(embeddings),
        "errors": errors,
        "total_trajectories": sum(skill_counts.values()),
        "skill_count": len(skill_counts),
        "exemplar_stats": exemplar_stats,
        "total_exemplars_embedded": sum(exemplar_stats.values()),
        "drift": drift_report,
        "drifted_skills": drifted_skills,
    }


# ---------------------------------------------------------------------------
# Vector Math (no numpy needed — embeddings are small enough)
# ---------------------------------------------------------------------------


def cosine_similarity(a: List[float], b: List[float]) -> float:
    """Compute cosine similarity between two vectors."""
    if len(a) != len(b) or len(a) == 0:
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


GENERIC_SKILLS = {"shell-session", "mixed-session", "authoring-session", "test"}


def _load_skill_penalties() -> Dict[str, float]:
    """Load per-skill penalties from config.json.

    Config format: {"skill_penalties": {"shell-session": 0.02, ...}}
    Falls back to GENERIC_SKILLS default (0.01) for unconfigured generic skills.
    """
    import json as _json
    config_path = KARL_DIR / "config.json"
    try:
        config = _json.loads(config_path.read_text())
        return config.get("skill_penalties", {})
    except (OSError, _json.JSONDecodeError):
        return {}


def rank_skills(
    prompt_embedding: List[float],
    skill_embeddings: Dict[str, Tuple[List[float], float]],
    threshold: float = 0.35,
    weight_bonus: float = 0.02,
    generic_penalty: float = 0.01,
) -> List[Tuple[str, float]]:
    """Rank skills by cosine similarity with weight bonus and generic penalty.

    Score = similarity + (weight_bonus * log_weight) - penalty
    Generic session-type skills (shell-session, mixed-session, etc.) get
    a default penalty. Per-skill overrides can be set in config.json
    via "skill_penalties": {"skill-name": 0.02}.
    Returns list of (skill_name, score) above threshold, sorted desc.
    """
    custom_penalties = _load_skill_penalties()

    scores = []
    for name, (skill_vec, weight) in skill_embeddings.items():
        sim = cosine_similarity(prompt_embedding, skill_vec)
        score = sim + weight_bonus * weight
        # Per-skill penalty from config, or default for generic skills
        if name in custom_penalties:
            score -= custom_penalties[name]
        elif name in GENERIC_SKILLS:
            score -= generic_penalty
        if score > threshold:
            scores.append((name, score))
    scores.sort(key=lambda x: -x[1])
    return scores


EXEMPLAR_REGISTRY = KARL_DIR / "exemplar_registry.jsonl"


def list_exemplars(skill_name: Optional[str] = None) -> Dict[str, Any]:
    """List registered exemplars, optionally filtered by skill.

    Returns exemplars from two sources:
    1. Auto-gathered from trajectories (via _gather_exemplar_prompts)
    2. Manually added via exemplar_registry.jsonl
    """
    import json as _json
    TRAJ_PATH = KARL_DIR / "trajectories.jsonl"

    # Auto-gathered exemplars
    auto = _gather_exemplar_prompts(TRAJ_PATH, max_per_skill=5)

    # Manual exemplars from registry
    manual: Dict[str, List[Dict]] = {}
    if EXEMPLAR_REGISTRY.exists():
        with open(EXEMPLAR_REGISTRY) as f:
            for line in f:
                try:
                    entry = _json.loads(line)
                    if entry.get("action") == "remove":
                        continue
                    sk = entry.get("skill", "")
                    if sk not in manual:
                        manual[sk] = []
                    manual[sk].append(entry)
                except _json.JSONDecodeError:
                    continue

    # Merge
    all_skills = sorted(set(list(auto.keys()) + list(manual.keys())))
    if skill_name:
        all_skills = [s for s in all_skills if s == skill_name]

    result = {}
    for sk in all_skills:
        auto_prompts = auto.get(sk, [])
        manual_entries = manual.get(sk, [])
        result[sk] = {
            "auto": [p[:200] for p in auto_prompts],
            "manual": [{"prompt": e["prompt"][:200], "added": e.get("ts", "")}
                       for e in manual_entries],
            "total": len(auto_prompts) + len(manual_entries),
        }

    return {"status": "ok", "skills": result, "total_skills": len(result)}


def add_exemplar(skill_name: str, prompt: str) -> Dict[str, Any]:
    """Add a manual exemplar for a skill."""
    import json as _json
    from datetime import datetime, timezone

    if not skill_name or not prompt or len(prompt) < 10:
        return {"status": "error", "message": "Skill and prompt (min 10 chars) required"}

    entry = {
        "action": "add",
        "skill": skill_name,
        "prompt": prompt[:4000],
        "ts": datetime.now(timezone.utc).isoformat(),
    }
    with open(EXEMPLAR_REGISTRY, "a") as f:
        f.write(_json.dumps(entry, separators=(",", ":")) + "\n")

    return {"status": "ok", "skill": skill_name, "prompt_preview": prompt[:100]}


def remove_exemplar(skill_name: str, index: int) -> Dict[str, Any]:
    """Mark a manual exemplar as removed (by index in the registry)."""
    import json as _json
    from datetime import datetime, timezone

    if not EXEMPLAR_REGISTRY.exists():
        return {"status": "error", "message": "No exemplar registry"}

    entries = []
    with open(EXEMPLAR_REGISTRY) as f:
        for line in f:
            try:
                entries.append(_json.loads(line))
            except _json.JSONDecodeError:
                continue

    # Find the Nth active entry for this skill
    active = [(i, e) for i, e in enumerate(entries)
              if e.get("skill") == skill_name and e.get("action") == "add"]

    if index >= len(active):
        return {"status": "error", "message": f"Index {index} out of range (have {len(active)})"}

    orig_idx, entry = active[index]
    remove_entry = {
        "action": "remove",
        "skill": skill_name,
        "prompt": entry["prompt"][:100],
        "ts": datetime.now(timezone.utc).isoformat(),
    }
    with open(EXEMPLAR_REGISTRY, "a") as f:
        f.write(_json.dumps(remove_entry, separators=(",", ":")) + "\n")

    return {"status": "ok", "removed": entry["prompt"][:100]}


def rebuild_centroid(skill_name: str) -> Dict[str, Any]:
    """Rebuild a single skill's centroid from its description + exemplars.

    Uses both auto-gathered trajectory exemplars and manual registry entries.
    """
    import json as _json
    import pickle as _pickle

    TRAJ_PATH = KARL_DIR / "trajectories.jsonl"
    auto = _gather_exemplar_prompts(TRAJ_PATH, max_per_skill=5)

    # Manual exemplars
    manual_prompts = []
    if EXEMPLAR_REGISTRY.exists():
        active: Dict[str, List[str]] = {}
        removed: Dict[str, set] = {}
        with open(EXEMPLAR_REGISTRY) as f:
            for line in f:
                try:
                    e = _json.loads(line)
                    sk = e.get("skill", "")
                    if e.get("action") == "add":
                        active.setdefault(sk, []).append(e["prompt"])
                    elif e.get("action") == "remove":
                        removed.setdefault(sk, set()).add(e["prompt"][:100])
                except _json.JSONDecodeError:
                    continue
        for p in active.get(skill_name, []):
            if p[:100] not in removed.get(skill_name, set()):
                manual_prompts.append(p)

    # All exemplar prompts for this skill
    all_prompts = auto.get(skill_name, []) + manual_prompts

    # Get skill description embedding
    embs = load_skill_embeddings()
    if skill_name not in embs:
        return {"status": "error", "message": f"Skill {skill_name} not in embeddings"}

    # Embed description (from skills directory)
    skills_dir = KARL_DIR.parent / "cortex" / "skills"
    desc_text = ""
    skill_file = skills_dir / skill_name / "SKILL.md"
    if skill_file.exists():
        content = skill_file.read_text()[:2000]
        desc_text = f"Skill: {skill_name}\n{content}"
    else:
        desc_text = f"Skill: {skill_name}"

    desc_vec = embed_sync(desc_text)
    if not desc_vec:
        return {"status": "error", "message": "Failed to embed description"}

    # Embed exemplars
    exemplar_vecs = []
    for p in all_prompts:
        vec = embed_sync(p[:4000])
        if vec:
            exemplar_vecs.append(vec)

    # Average all vectors
    all_vecs = [desc_vec] + exemplar_vecs
    centroid = _average_vectors(all_vecs)

    # Update embeddings
    embs[skill_name] = (centroid, {"exemplar_count": len(exemplar_vecs)})
    _pickle.dump(embs, open(SKILL_CACHE_PATH, "wb"))

    return {
        "status": "ok",
        "skill": skill_name,
        "description_embedded": True,
        "exemplars_embedded": len(exemplar_vecs),
        "total_vectors_averaged": len(all_vecs),
    }


def centroid_diversity() -> Dict[str, Any]:
    """Measure inter-centroid separation to detect clustering issues.

    Returns average pairwise distance, min-separation pair, per-skill stats.
    """
    embs = load_skill_embeddings()
    if len(embs) < 2:
        return {"status": "insufficient_centroids", "count": len(embs)}

    names = list(embs.keys())
    sims = []
    min_pair = ("", "", 1.0)
    max_pair = ("", "", 0.0)
    per_skill = {}

    for i in range(len(names)):
        skill_sims = []
        for j in range(len(names)):
            if i == j:
                continue
            sim = cosine_similarity(embs[names[i]][0], embs[names[j]][0])
            sims.append(sim)
            skill_sims.append(sim)
            if sim < min_pair[2]:
                min_pair = (names[i], names[j], sim)
            if sim > max_pair[2]:
                max_pair = (names[i], names[j], sim)
        per_skill[names[i]] = round(sum(skill_sims) / len(skill_sims), 4) if skill_sims else 0

    avg_sim = sum(sims) / len(sims) if sims else 0
    # Cluster detection: skills with avg similarity > 0.85 to all others are poorly separated
    clustered = [name for name, avg in per_skill.items() if avg > 0.85]

    return {
        "status": "ok",
        "centroids": len(embs),
        "avg_pairwise_similarity": round(avg_sim, 4),
        "min_separation": {"pair": [min_pair[0], min_pair[1]], "similarity": round(min_pair[2], 4)},
        "max_similarity": {"pair": [max_pair[0], max_pair[1]], "similarity": round(max_pair[2], 4)},
        "clustered_skills": clustered,
        "health": "good" if avg_sim < 0.75 else ("warning" if avg_sim < 0.85 else "poor"),
    }


def skill_similarity_matrix() -> Dict[str, Any]:
    """Full NxN skill similarity matrix with merge recommendations.

    Returns pairwise similarity for all skill centroids plus merge candidates
    (pairs with similarity > 0.85).
    """
    embs = load_skill_embeddings()
    if len(embs) < 2:
        return {"status": "insufficient_centroids", "count": len(embs)}

    names = sorted(embs.keys())
    matrix: Dict[str, Dict[str, float]] = {}
    merge_candidates = []

    for i, a in enumerate(names):
        matrix[a] = {}
        for j, b in enumerate(names):
            if i == j:
                matrix[a][b] = 1.0
                continue
            sim = cosine_similarity(embs[a][0], embs[b][0])
            matrix[a][b] = round(sim, 4)
            if j > i and sim > 0.85:
                merge_candidates.append({
                    "skill_a": a,
                    "skill_b": b,
                    "similarity": round(sim, 4),
                    "confidence": round(min(1.0, (sim - 0.85) / 0.1), 2),
                    "recommendation": (
                        "strong_merge" if sim > 0.95
                        else "likely_merge" if sim > 0.90
                        else "investigate"
                    ),
                })

    merge_candidates.sort(key=lambda x: -x["similarity"])

    return {
        "status": "ok",
        "skills": names,
        "matrix": matrix,
        "merge_candidates": merge_candidates,
        "total_pairs": len(names) * (len(names) - 1) // 2,
        "high_similarity_count": len(merge_candidates),
    }


CENTROID_VERSIONS_DIR = KARL_DIR / "centroid_versions"


def save_centroid_snapshot(label: Optional[str] = None) -> Dict[str, Any]:
    """Save current centroids as a versioned snapshot.

    Stores centroids + metadata (timestamp, accuracy, centroid count).
    Returns snapshot info.
    """
    import json as _json
    from datetime import datetime, timezone

    embs = load_skill_embeddings()
    if not embs:
        return {"status": "no_centroids"}

    CENTROID_VERSIONS_DIR.mkdir(parents=True, exist_ok=True)

    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    snapshot_name = f"centroids_{ts}"
    if label:
        snapshot_name += f"_{label}"

    snapshot_path = CENTROID_VERSIONS_DIR / f"{snapshot_name}.pkl"
    meta_path = CENTROID_VERSIONS_DIR / f"{snapshot_name}.json"

    # Save centroids
    with open(snapshot_path, "wb") as f:
        pickle.dump(embs, f)

    # Save metadata
    div = centroid_diversity()
    meta = {
        "timestamp": ts,
        "label": label,
        "centroids": len(embs),
        "avg_similarity": div.get("avg_pairwise_similarity"),
        "max_similarity": div.get("max_similarity", {}).get("similarity"),
        "health": div.get("health"),
    }
    with open(meta_path, "w") as f:
        _json.dump(meta, f, indent=2)

    return {"status": "ok", "snapshot": snapshot_name, "path": str(snapshot_path), "meta": meta}


def list_centroid_versions() -> List[Dict[str, Any]]:
    """List all centroid snapshots with metadata."""
    import json as _json

    if not CENTROID_VERSIONS_DIR.exists():
        return []

    versions = []
    for meta_file in sorted(CENTROID_VERSIONS_DIR.glob("centroids_*.json")):
        try:
            meta = _json.loads(meta_file.read_text())
            pkl_path = meta_file.with_suffix(".pkl")
            meta["has_data"] = pkl_path.exists()
            meta["name"] = meta_file.stem
            versions.append(meta)
        except (Exception,):
            continue

    return versions


def rollback_centroids(version_name: str) -> Dict[str, Any]:
    """Restore centroids from a previous snapshot.

    Saves current centroids as a pre-rollback snapshot first.
    """
    pkl_path = CENTROID_VERSIONS_DIR / f"{version_name}.pkl"
    if not pkl_path.exists():
        return {"status": "not_found", "version": version_name}

    # Auto-snapshot current state before rollback
    save_centroid_snapshot(label="pre_rollback")

    try:
        with open(pkl_path, "rb") as f:
            embs = pickle.load(f)
        save_skill_embeddings(embs)
        global _skill_cache_loaded
        _skill_cache_loaded = False
        return {"status": "ok", "restored": version_name, "centroids": len(embs)}
    except Exception as e:
        return {"status": "error", "error": str(e)}


def _normalize(vec: List[float]) -> List[float]:
    """L2-normalize a vector."""
    norm = math.sqrt(sum(x * x for x in vec))
    if norm == 0:
        return vec
    return [x / norm for x in vec]


def identify_hard_negatives() -> Dict[str, Any]:
    """Identify confused skill pairs from shadow routing data.

    Returns confusion pairs ranked by error count, with the centroid
    similarity between the confused pair (higher = harder to separate).
    """
    import json as _json
    shadow_path = KARL_DIR / "routing_shadow.jsonl"
    if not shadow_path.exists():
        return {"status": "no_shadow_data", "pairs": []}

    confusion_pairs: Dict[Tuple[str, str], int] = {}
    with open(shadow_path) as f:
        for line in f:
            try:
                r = _json.loads(line)
                if r.get("vector_correct"):
                    continue
                actual = r.get("actual_skill", "")
                predicted = r.get("vector", "")
                if actual and predicted and actual != predicted:
                    pair = (actual, predicted)
                    confusion_pairs[pair] = confusion_pairs.get(pair, 0) + 1
            except _json.JSONDecodeError:
                continue

    embs = load_skill_embeddings()
    pairs = []
    for (actual, predicted), count in sorted(confusion_pairs.items(), key=lambda x: -x[1]):
        sim = 0.0
        if actual in embs and predicted in embs:
            sim = cosine_similarity(embs[actual][0], embs[predicted][0])
        pairs.append({
            "actual": actual,
            "predicted": predicted,
            "errors": count,
            "centroid_similarity": round(sim, 4),
        })

    return {"status": "ok", "total_errors": sum(confusion_pairs.values()), "pairs": pairs}


def refine_centroids(
    alpha: float = 0.15,
    min_errors: int = 2,
    dry_run: bool = True,
) -> Dict[str, Any]:
    """Contrastive centroid refinement using hard-negative mining.

    For each confused pair (actual, predicted_wrong) with >= min_errors:
    - Push actual's centroid AWAY from predicted_wrong's centroid
    - Push predicted_wrong's centroid AWAY from actual's centroid (smaller step)

    The update: centroid_new = normalize(centroid - alpha * direction_to_confused)

    Args:
        alpha: Step size for centroid adjustment (0.0-0.5). Higher = more aggressive.
        min_errors: Minimum confusion count to trigger adjustment.
        dry_run: If True, compute but don't save adjustments.

    Returns summary with before/after similarities for each adjusted pair.
    """
    hard_neg = identify_hard_negatives()
    if hard_neg.get("status") != "ok":
        return {"status": "no_data", "adjustments": []}

    embs = load_skill_embeddings()
    if not embs:
        return {"status": "no_centroids", "adjustments": []}

    # Work on copies
    adjusted = {name: (list(vec), weight) for name, (vec, weight) in embs.items()}
    adjustments = []

    for pair_info in hard_neg["pairs"]:
        if pair_info["errors"] < min_errors:
            continue

        actual = pair_info["actual"]
        predicted = pair_info["predicted"]
        if actual not in adjusted or predicted not in adjusted:
            continue

        actual_vec = adjusted[actual][0]
        predicted_vec = adjusted[predicted][0]

        sim_before = cosine_similarity(actual_vec, predicted_vec)

        # Direction from actual to predicted
        dim = len(actual_vec)
        direction = [predicted_vec[i] - actual_vec[i] for i in range(dim)]

        # Push actual AWAY from predicted (full alpha)
        new_actual = [actual_vec[i] - alpha * direction[i] for i in range(dim)]
        new_actual = _normalize(new_actual)

        # Push predicted AWAY from actual (half alpha — less aggressive)
        new_predicted = [predicted_vec[i] + (alpha * 0.5) * direction[i] for i in range(dim)]
        new_predicted = _normalize(new_predicted)

        sim_after = cosine_similarity(new_actual, new_predicted)

        adjusted[actual] = (new_actual, adjusted[actual][1])
        adjusted[predicted] = (new_predicted, adjusted[predicted][1])

        adjustments.append({
            "actual": actual,
            "predicted": predicted,
            "errors": pair_info["errors"],
            "sim_before": round(sim_before, 4),
            "sim_after": round(sim_after, 4),
            "improvement": round(sim_before - sim_after, 4),
        })

    if not dry_run and adjustments:
        # Convert back to the expected format
        save_data = {name: (vec, weight) for name, (vec, weight) in adjusted.items()}
        save_skill_embeddings(save_data)
        global _skill_cache_loaded
        _skill_cache_loaded = False

    return {
        "status": "ok",
        "dry_run": dry_run,
        "pairs_adjusted": len(adjustments),
        "adjustments": adjustments,
    }


def auto_refresh_centroids(force: bool = False) -> Dict[str, Any]:
    """Auto-refresh centroids if interval has elapsed, with safety snapshot.

    Checks config.centroid_refresh_interval_hours and last_centroid_refresh.
    If due (or force=True):
    1. Snapshots current centroids
    2. Refreshes from trajectories (exemplar-based)
    3. Detects drift, alerts on high drift
    4. Updates config with refresh timestamp

    Returns refresh result or skip reason.
    """
    import json as _json
    from datetime import datetime, timezone

    config_path = KARL_DIR / "config.json"
    config = {}
    if config_path.exists():
        try:
            config = _json.loads(config_path.read_text())
        except (_json.JSONDecodeError, OSError):
            pass

    interval_hours = config.get("centroid_refresh_interval_hours", 24)
    last_refresh = config.get("last_centroid_refresh")

    if not force and last_refresh:
        try:
            last_dt = datetime.fromisoformat(last_refresh)
            elapsed = (datetime.now(timezone.utc) - last_dt).total_seconds() / 3600
            if elapsed < interval_hours:
                return {
                    "status": "skipped",
                    "reason": f"Only {elapsed:.1f}h since last refresh (interval: {interval_hours}h)",
                    "next_refresh_hours": round(interval_hours - elapsed, 1),
                }
        except (ValueError, TypeError):
            pass

    # Safety snapshot
    snapshot = save_centroid_snapshot(label="pre-auto-refresh")

    # Refresh
    result = refresh_skill_centroids()
    result["snapshot"] = snapshot.get("name", "")

    # Check for high drift
    high_drift = [s for s in result.get("drifted_skills", []) if s.get("drift", 0) > 0.2]
    if high_drift:
        result["drift_alert"] = True
        result["high_drift_skills"] = high_drift

    # Update config
    config["last_centroid_refresh"] = datetime.now(timezone.utc).isoformat()
    try:
        config_path.write_text(_json.dumps(config, indent=2, default=str))
    except OSError:
        pass

    result["status"] = "refreshed"
    return result


def iterative_refine(
    max_rounds: int = 5,
    alpha: float = 0.10,
    min_errors: int = 2,
    min_improvement: float = 0.005,
    dry_run: bool = True,
) -> Dict[str, Any]:
    """Iterative centroid refinement with convergence check.

    Runs refine_centroids in a loop, measuring accuracy after each round
    via shadow re-evaluation. Stops when:
    - Accuracy stops improving (delta < min_improvement)
    - Accuracy regresses
    - max_rounds reached
    - No more pairs to refine

    Args:
        max_rounds: Maximum refinement iterations.
        alpha: Step size per round (lower than single-shot for stability).
        min_errors: Minimum confusion count to trigger pair adjustment.
        min_improvement: Stop if accuracy gain < this threshold.
        dry_run: If True, simulate without saving any changes.

    Returns per-round metrics and convergence info.
    """
    from shadow_seeder import reshadow_records

    # Snapshot before iterating
    if not dry_run:
        save_centroid_snapshot(label="pre-iterative-refine")

    # Get baseline accuracy from current shadow records
    baseline = reshadow_records(dry_run=True)
    baseline_acc = baseline.get("new_accuracy", 0) or 0
    baseline_top3 = baseline.get("new_top3", 0) or 0
    baseline_mrr = baseline.get("new_mrr", 0) or 0

    rounds = []
    best_acc = baseline_acc
    converged = False
    converge_reason = ""

    for i in range(max_rounds):
        # Run one refinement round
        refine_result = refine_centroids(alpha=alpha, min_errors=min_errors, dry_run=dry_run)

        if refine_result.get("pairs_adjusted", 0) == 0:
            converge_reason = "no_pairs_to_refine"
            converged = True
            break

        # Measure accuracy after this round
        eval_result = reshadow_records(dry_run=True)
        new_acc = eval_result.get("new_accuracy", 0) or 0
        new_top3 = eval_result.get("new_top3", 0) or 0
        new_mrr = eval_result.get("new_mrr", 0) or 0
        delta = new_acc - best_acc

        round_info = {
            "round": i + 1,
            "pairs_adjusted": refine_result.get("pairs_adjusted", 0),
            "accuracy": round(new_acc, 4),
            "top3": round(new_top3, 4),
            "mrr": round(new_mrr, 4),
            "delta": round(delta, 4),
        }
        rounds.append(round_info)

        if delta < -min_improvement:
            # Regression — rollback this round
            converge_reason = "regression"
            converged = True
            if not dry_run:
                # Rollback by re-running with negative would be complex,
                # so we snapshot-rollback
                versions = list_centroid_versions()
                if versions:
                    rollback_centroids(versions[-1]["name"])
            break

        if delta < min_improvement:
            converge_reason = "plateau"
            converged = True
            break

        best_acc = new_acc

    # Apply the reshadow if we improved and not dry-running
    final_acc = rounds[-1]["accuracy"] if rounds else baseline_acc
    if not dry_run and final_acc > baseline_acc:
        reshadow_records(dry_run=False)

    return {
        "status": "ok",
        "dry_run": dry_run,
        "baseline_accuracy": round(baseline_acc, 4),
        "baseline_top3": round(baseline_top3, 4),
        "baseline_mrr": round(baseline_mrr, 4),
        "final_accuracy": round(final_acc, 4),
        "total_improvement": round(final_acc - baseline_acc, 4),
        "rounds": rounds,
        "converged": converged,
        "converge_reason": converge_reason,
        "max_rounds": max_rounds,
        "alpha": alpha,
    }


if __name__ == "__main__":
    import sys as _sys
    import json as _j
    if "--refresh" in _sys.argv:
        result = refresh_skill_centroids()
        print(_j.dumps(result, indent=2))
    elif "--diversity" in _sys.argv:
        result = centroid_diversity()
        print(_j.dumps(result, indent=2))
    elif "--hard-negatives" in _sys.argv:
        result = identify_hard_negatives()
        print(_j.dumps(result, indent=2))
    elif "--refine" in _sys.argv:
        apply = "--apply" in _sys.argv
        alpha = 0.15
        for i, a in enumerate(_sys.argv):
            if a == "--alpha" and i + 1 < len(_sys.argv):
                alpha = float(_sys.argv[i + 1])
        result = refine_centroids(alpha=alpha, dry_run=not apply)
        print(_j.dumps(result, indent=2))
    else:
        print("Usage: embedding_cache.py --refresh | --diversity | --hard-negatives | --refine [--apply] [--alpha N]")
