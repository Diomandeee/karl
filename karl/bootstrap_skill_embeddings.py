#!/usr/bin/env python3
"""
bootstrap_skill_embeddings.py — Generate embeddings for all active skills.

Reads each active forged SKILL.md, builds embedding text (intent + workflow +
historical prompts), calls RAG++ /api/rag/embed, and stores in:
  - Local pickle: ~/.claude/karl/skill_embeddings.pkl
  - Supabase: skill_embeddings table (for cross-machine sync)

Usage:
    python3 bootstrap_skill_embeddings.py
    python3 bootstrap_skill_embeddings.py --dry-run
"""

import json
import os
import sys
import time
import urllib.request
from pathlib import Path
from typing import Dict, List, Optional, Tuple

KARL_DIR = Path(__file__).parent
sys.path.insert(0, str(KARL_DIR))

from embedding_cache import save_skill_embeddings, EMBEDDING_DIM

SKILLS_DIR = Path.home() / ".claude" / "skills"
REGISTRY_PATH = SKILLS_DIR / "registry.json"
RAG_EMBED_URL = "http://localhost:8000/api/rag/embed"
CORTEX_ENTRIES = Path.home() / ".claude" / "cortex" / "entries.jsonl"


def _load_active_skills() -> Dict[str, dict]:
    """Load active forged skills from registry."""
    if not REGISTRY_PATH.exists():
        return {}
    with open(REGISTRY_PATH) as f:
        reg = json.load(f)
    return {
        name: info
        for name, info in reg.get("skills", {}).items()
        if info.get("status") == "active" and info.get("forged")
    }


def _load_skill_md(skill_name: str) -> Optional[str]:
    """Load SKILL.md content (without frontmatter)."""
    skill_path = SKILLS_DIR / skill_name / "SKILL.md"
    if not skill_path.exists():
        return None
    content = skill_path.read_text()
    # Strip frontmatter
    if content.startswith("---"):
        end = content.find("---", 3)
        if end > 0:
            content = content[end + 3:].strip()
    return content


def _extract_sections(content: str) -> Dict[str, str]:
    """Extract Intent, Workflow, and Gotchas sections from SKILL.md."""
    sections = {"intent": "", "workflow": "", "gotchas": ""}
    current = None
    lines = content.split("\n")

    for line in lines:
        lower = line.lower().strip()
        if "intent" in lower and line.startswith("#"):
            current = "intent"
            continue
        elif "workflow" in lower and line.startswith("#"):
            current = "workflow"
            continue
        elif "gotcha" in lower and line.startswith("#"):
            current = "gotchas"
            continue
        elif line.startswith("# ") or line.startswith("## "):
            current = None

        if current:
            sections[current] += line + "\n"

    return sections


def _get_historical_prompts(skill_name: str, limit: int = 5) -> List[str]:
    """Get top historical trigger prompts for this skill from entries.jsonl."""
    prompts = []
    if not CORTEX_ENTRIES.exists():
        return prompts

    try:
        with open(CORTEX_ENTRIES) as f:
            for line in f:
                try:
                    entry = json.loads(line)
                    if (
                        entry.get("type") == "invocation_record"
                        and entry.get("skill") == skill_name
                    ):
                        prompt = entry.get("trigger_prompt", "")
                        if prompt and len(prompt) > 5:
                            prompts.append(prompt)
                except json.JSONDecodeError:
                    continue
    except Exception:
        pass

    # Deduplicate and take top N
    seen = set()
    unique = []
    for p in prompts:
        key = p.lower().strip()
        if key not in seen:
            seen.add(key)
            unique.append(p)
    return unique[:limit]


def build_skill_embedding_text(skill_name: str, content: str) -> str:
    """Build the embedding text for a skill."""
    sections = _extract_sections(content)
    historical = _get_historical_prompts(skill_name)

    parts = [f"Skill: {skill_name}"]

    if sections["intent"]:
        parts.append(f"Intent: {sections['intent'][:200]}")
    if sections["workflow"]:
        parts.append(f"Workflow: {sections['workflow'][:300]}")
    if sections["gotchas"]:
        parts.append(f"Gotchas: {sections['gotchas'][:200]}")
    if historical:
        parts.append(f"Used for prompts like: {'; '.join(historical)}")

    return "\n".join(parts)[:4000]


def _embed_text(text: str) -> Optional[List[float]]:
    """Call RAG++ embed endpoint synchronously."""
    try:
        req = urllib.request.Request(
            RAG_EMBED_URL,
            data=json.dumps({"text": text}).encode("utf-8"),
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read())
            embedding = data.get("embedding", [])
            if embedding and len(embedding) == EMBEDDING_DIM:
                return embedding
    except Exception as e:
        print(f"  [error] Embedding failed: {e}")
    return None


def bootstrap(dry_run: bool = False) -> Dict[str, int]:
    """Generate and store embeddings for all active skills."""
    skills = _load_active_skills()
    print(f"[bootstrap] Found {len(skills)} active forged skills")

    results: Dict[str, Tuple[List[float], float]] = {}
    embedded = 0
    failed = 0

    for name in sorted(skills.keys()):
        content = _load_skill_md(name)
        if not content:
            print(f"  [{name}] SKIP — no SKILL.md")
            failed += 1
            continue

        embed_text = build_skill_embedding_text(name, content)
        print(f"  [{name}] Embedding ({len(embed_text)} chars)...", end=" ")

        if dry_run:
            print("DRY-RUN")
            continue

        embedding = _embed_text(embed_text)
        if embedding:
            results[name] = (embedding, 1.0)  # Default weight 1.0
            embedded += 1
            print(f"OK ({len(embedding)} dims)")
        else:
            failed += 1
            print("FAILED")

        # Rate limit courtesy
        time.sleep(0.3)

    if not dry_run and results:
        save_skill_embeddings(results)
        print(f"\n[bootstrap] Saved {len(results)} embeddings to pickle")

    return {"total": len(skills), "embedded": embedded, "failed": failed}


if __name__ == "__main__":
    dry_run = "--dry-run" in sys.argv
    stats = bootstrap(dry_run=dry_run)
    print(f"\n[bootstrap] Result: {stats}")
