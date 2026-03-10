"""
embedding_cache.py - Async embedding cache for KARL skill routing.

Provides:
  - LRU cache for prompt embeddings (configurable max entries + TTL)
  - Pickle persistence across sessions
  - Async embed via external API (non-blocking, fire-and-forget)
  - Skill embedding loader with mtime-based refresh
  - Cosine similarity computation (no numpy dependency)

Design constraint: agent hook budget is 500ms. Embedding API takes 300-500ms.
We embed ASYNC and use the cache for the NEXT prompt, not the current one.
"""

import hashlib
import json
import math
import os
import pickle
import threading
import time
import urllib.request
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from karl.config import (
    PROMPT_CACHE_PATH,
    SKILL_EMBEDDINGS_PATH,
    EMBED_URL,
    EMBEDDING_DIM,
    EMBED_TIMEOUT,
    CACHE_MAX_ENTRIES,
    CACHE_TTL_SECONDS,
)

# In-memory cache: {hash: (embedding, timestamp)}
_cache: Dict[str, Tuple[List[float], float]] = {}
_cache_loaded = False
_skill_cache: Dict[str, Tuple[List[float], float]] = {}
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
    if PROMPT_CACHE_PATH.exists():
        try:
            with open(PROMPT_CACHE_PATH, "rb") as f:
                _cache = pickle.load(f)
            now = time.time()
            expired = [k for k, (_, ts) in _cache.items() if now - ts > CACHE_TTL_SECONDS]
            for k in expired:
                del _cache[k]
        except Exception:
            _cache = {}
    _cache_loaded = True


def _save_cache() -> None:
    """Persist prompt embedding cache to disk."""
    try:
        PROMPT_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(PROMPT_CACHE_PATH, "wb") as f:
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
    if time.time() - ts > CACHE_TTL_SECONDS:
        del _cache[key]
        return None
    return embedding


def cache_store(text: str, embedding: List[float]) -> None:
    """Store embedding in cache with LRU eviction."""
    _load_cache()
    key = cache_key(text)
    _cache[key] = (embedding, time.time())
    if len(_cache) > CACHE_MAX_ENTRIES:
        oldest = min(_cache.items(), key=lambda x: x[1][1])
        del _cache[oldest[0]]
    _save_cache()


def embed_async(text: str, embed_url: Optional[str] = None) -> None:
    """Fire-and-forget: compute embedding in background thread, store in cache.

    Called from the hook after routing. The embedding will be available
    for the NEXT prompt via cache_get(). Uses daemon=True so we don't
    block process exit, but callers should join the thread with a
    timeout before exiting to ensure the cache gets populated.
    """
    url = embed_url or EMBED_URL

    def _do_embed():
        try:
            req = urllib.request.Request(
                url,
                data=json.dumps({"text": text[:4000]}).encode("utf-8"),
                headers={"Content-Type": "application/json"},
            )
            with urllib.request.urlopen(req, timeout=EMBED_TIMEOUT) as resp:
                data = json.loads(resp.read())
                embedding = data.get("embedding", [])
                if embedding and len(embedding) == EMBEDDING_DIM:
                    cache_store(text, embedding)
        except Exception:
            pass

    t = threading.Thread(target=_do_embed, daemon=True)
    t.start()
    return t  # Caller can join if needed


def embed_sync(text: str, embed_url: Optional[str] = None) -> Optional[List[float]]:
    """Synchronous embedding call. Returns embedding or None."""
    url = embed_url or EMBED_URL
    try:
        req = urllib.request.Request(
            url,
            data=json.dumps({"text": text[:4000]}).encode("utf-8"),
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=EMBED_TIMEOUT) as resp:
            data = json.loads(resp.read())
            embedding = data.get("embedding", [])
            if embedding and len(embedding) == EMBEDDING_DIM:
                cache_store(text, embedding)
                return embedding
    except Exception:
        pass
    return None


def build_prompt_embedding_text(prompt: str, cwd: str = "") -> str:
    """Build the text to embed for a prompt, with optional project context."""
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
    Mtime-based caching: reloads when file changes on disk.
    """
    global _skill_cache, _skill_cache_loaded, _skill_cache_mtime

    if not SKILL_EMBEDDINGS_PATH.exists():
        return {}

    mtime = SKILL_EMBEDDINGS_PATH.stat().st_mtime
    if _skill_cache_loaded and mtime == _skill_cache_mtime:
        return _skill_cache

    try:
        with open(SKILL_EMBEDDINGS_PATH, "rb") as f:
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
        SKILL_EMBEDDINGS_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(SKILL_EMBEDDINGS_PATH, "wb") as f:
            pickle.dump(embeddings, f)
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Vector Math (no numpy needed)
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


def rank_skills(
    prompt_embedding: List[float],
    skill_embeddings: Dict[str, Tuple[List[float], float]],
    threshold: float = 0.35,
) -> List[Tuple[str, float]]:
    """Rank skills by weighted cosine similarity to prompt embedding.

    Args:
        prompt_embedding: The prompt's embedding vector
        skill_embeddings: {name: (vector, weight)} from load_skill_embeddings()
        threshold: Minimum weighted score to include

    Returns:
        List of (skill_name, weighted_score) above threshold, sorted desc
    """
    scores = []
    for name, (skill_vec, weight) in skill_embeddings.items():
        sim = cosine_similarity(prompt_embedding, skill_vec)
        weighted = sim * weight
        if weighted > threshold:
            scores.append((name, weighted))
    scores.sort(key=lambda x: -x[1])
    return scores
