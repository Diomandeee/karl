"""
bootstrap.py - Generate embeddings for all active skills.

Reads each active skill definition, builds embedding text from intent,
workflow, gotchas, and historical prompts, then calls the embedding API
and stores results to the local pickle cache.

Usage:
    from karl.bootstrap import bootstrap_skill_embeddings
    stats = bootstrap_skill_embeddings()
"""

import json
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from karl.config import SKILLS_DIR, SKILLS_REGISTRY, CORTEX_ENTRIES, EMBEDDING_DIM
from karl.embedding_cache import embed_sync, save_skill_embeddings


def _load_active_skills() -> Dict[str, dict]:
    """Load active forged skills from registry."""
    if not SKILLS_REGISTRY.exists():
        return {}
    with open(SKILLS_REGISTRY) as f:
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
    if content.startswith("---"):
        end = content.find("---", 3)
        if end > 0:
            content = content[end + 3:].strip()
    return content


def _extract_sections(content: str) -> Dict[str, str]:
    """Extract Intent, Workflow, and Gotchas sections from SKILL.md."""
    sections = {"intent": "", "workflow": "", "gotchas": ""}
    current = None
    for line in content.split("\n"):
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
    """Get top historical trigger prompts for a skill."""
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

    seen = set()
    unique = []
    for p in prompts:
        key = p.lower().strip()
        if key not in seen:
            seen.add(key)
            unique.append(p)
    return unique[:limit]


def build_skill_embedding_text(skill_name: str, content: str) -> str:
    """Build the embedding text for a skill from its SKILL.md content."""
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


def bootstrap_skill_embeddings(dry_run: bool = False) -> Dict[str, int]:
    """Generate and store embeddings for all active skills.

    Args:
        dry_run: Preview without saving embeddings

    Returns:
        Stats dict with total, embedded, and failed counts
    """
    skills = _load_active_skills()

    results: Dict[str, Tuple[List[float], float]] = {}
    embedded = 0
    failed = 0

    for name in sorted(skills.keys()):
        content = _load_skill_md(name)
        if not content:
            failed += 1
            continue

        embed_text = build_skill_embedding_text(name, content)

        if dry_run:
            continue

        embedding = embed_sync(embed_text)
        if embedding:
            results[name] = (embedding, 1.0)  # Default weight 1.0
            embedded += 1
        else:
            failed += 1

        time.sleep(0.3)  # Rate limit courtesy

    if not dry_run and results:
        save_skill_embeddings(results)

    return {"total": len(skills), "embedded": embedded, "failed": failed}
