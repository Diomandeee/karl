"""
embedding_cache.py - Async embedding cache for KARL skill routing.

Provides:
  - LRU cache for prompt embeddings (configurable max entries + TTL)
  - JSON persistence across sessions
  - Async embed via bounded thread pool (non-blocking, fire-and-forget)
  - Skill embedding loader with mtime-based refresh
  - Cosine similarity computation (numpy-accelerated)

Design constraint: agent hook budget is 500ms. Embedding API takes 300-500ms.
We embed ASYNC and use the cache for the NEXT prompt, not the current one.
"""

import hashlib
import json
import logging
import math
import os
import threading
import time
import urllib.request
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

from karl.config import (
    PROMPT_CACHE_PATH,
    SKILL_EMBEDDINGS_PATH,
    EMBED_URL,
    EMBEDDING_DIM,
    EMBED_TIMEOUT,
    CACHE_MAX_ENTRIES,
    CACHE_TTL_SECONDS,
)

# In-memory cache: OrderedDict for O(1) LRU eviction
_cache: OrderedDict[str, Tuple[List[float], float]] = OrderedDict()
_cache_lock = threading.Lock()
_cache_loaded = False
_cache_stores_since_save = 0
_SAVE_INTERVAL = 10  # Save to disk every N stores

_skill_cache: Dict[str, Tuple[List[float], float]] = {}
_skill_cache_lock = threading.Lock()
_skill_cache_loaded = False
_skill_cache_mtime: float = 0.0

# Bounded thread pool for async embedding (C4)
_embed_pool = ThreadPoolExecutor(max_workers=4)


def cache_key(text: str) -> str:
    """Generate cache key from text."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def _load_cache() -> None:
    """Load prompt embedding cache from disk (JSON format)."""
    global _cache, _cache_loaded
    if _cache_loaded:
        return
    cache_path = PROMPT_CACHE_PATH.with_suffix(".json")
    if cache_path.exists():
        try:
            with open(cache_path, "r") as f:
                raw = json.load(f)
            _cache = {k: (v[0], v[1]) for k, v in raw.items()}
            now = time.time()
            expired = [k for k, (_, ts) in _cache.items() if now - ts > CACHE_TTL_SECONDS]
            for k in expired:
                del _cache[k]
        except Exception:
            _cache = {}
    _cache_loaded = True


def _save_cache() -> None:
    """Persist prompt embedding cache to disk (JSON format)."""
    try:
        cache_path = PROMPT_CACHE_PATH.with_suffix(".json")
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        serializable = {k: [v[0], v[1]] for k, v in _cache.items()}
        with open(cache_path, "w") as f:
            json.dump(serializable, f)
    except Exception:
        pass


def cache_get(text: str) -> Optional[List[float]]:
    """Get cached embedding for text. Returns None on miss."""
    with _cache_lock:
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
    """Store embedding in cache with O(1) LRU eviction. Batched disk saves."""
    global _cache_stores_since_save
    with _cache_lock:
        _load_cache()
        key = cache_key(text)
        _cache[key] = (embedding, time.time())
        _cache.move_to_end(key)
        if len(_cache) > CACHE_MAX_ENTRIES:
            _cache.popitem(last=False)  # O(1) eviction of oldest
        _cache_stores_since_save += 1
        if _cache_stores_since_save >= _SAVE_INTERVAL:
            _save_cache()
            _cache_stores_since_save = 0


def embed_async(text: str, embed_url: Optional[str] = None) -> "concurrent.futures.Future[None]":
    """Fire-and-forget: compute embedding in bounded thread pool, store in cache.

    Called from the hook after routing. The embedding will be available
    for the NEXT prompt via cache_get(). Uses ThreadPoolExecutor(max_workers=4)
    to bound concurrent embed requests (C4).

    Returns a Future that callers can .result(timeout=N) if needed.
    """
    url = embed_url or EMBED_URL

    def _do_embed() -> None:
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
            logger.debug("embed_async failed for text len=%d", len(text))

    return _embed_pool.submit(_do_embed)


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


def flush_cache() -> None:
    """Force-save the prompt cache to disk. Call on process exit."""
    with _cache_lock:
        _save_cache()


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
    """Load skill embeddings from local JSON cache.

    Returns: {skill_name: (embedding_vector, trajectory_weight)}
    Mtime-based caching: reloads when file changes on disk.
    """
    global _skill_cache, _skill_cache_loaded, _skill_cache_mtime

    skill_path = SKILL_EMBEDDINGS_PATH.with_suffix(".json")
    if not skill_path.exists():
        return {}

    mtime = skill_path.stat().st_mtime
    if _skill_cache_loaded and mtime == _skill_cache_mtime:
        return _skill_cache

    try:
        with _skill_cache_lock:
            with open(skill_path, "r") as f:
                raw = json.load(f)
            _skill_cache = {k: (v[0], v[1]) for k, v in raw.items()}
            _skill_cache_mtime = mtime
            _skill_cache_loaded = True
    except Exception:
        _skill_cache = {}
        _skill_cache_loaded = True

    return _skill_cache


def save_skill_embeddings(embeddings: Dict[str, Tuple[List[float], float]]) -> None:
    """Persist skill embeddings to local JSON."""
    try:
        skill_path = SKILL_EMBEDDINGS_PATH.with_suffix(".json")
        skill_path.parent.mkdir(parents=True, exist_ok=True)
        serializable = {k: [v[0], v[1]] for k, v in embeddings.items()}
        with _skill_cache_lock:
            with open(skill_path, "w") as f:
                json.dump(serializable, f)
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Vector Math (numpy-accelerated)
# ---------------------------------------------------------------------------


def cosine_similarity(a: List[float], b: List[float]) -> float:
    """Compute cosine similarity between two vectors using numpy."""
    if len(a) != len(b) or len(a) == 0:
        return 0.0
    va = np.asarray(a, dtype=np.float32)
    vb = np.asarray(b, dtype=np.float32)
    norm_a = np.linalg.norm(va)
    norm_b = np.linalg.norm(vb)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.dot(va, vb) / (norm_a * norm_b))


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
