"""
entity_bridge.py - Bridge between KARL trajectory rewards and SEA skill entities.

Updates SEA entity state (confidence, hot/cold topics, activation counts)
based on trajectory reward signals. Replaces time-based decay with
performance-based entity intelligence.

SEA entities live at $KARL_SEA_DIR/{skill_name}/state.json.
Each entity tracks: total_activations, useful_activations, hot_topics,
cold_topics, confidence_calibration, last_activated.

Usage:
    from karl.entity_bridge import update_entity_from_trajectory
    update_entity_from_trajectory(record)  # Called from flush_session
"""

import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import karl.config as config

# SEA entity directory (configurable)
SEA_DIR = Path(config._env_path(
    "KARL_SEA_DIR",
    str(Path.home() / ".clawdbot" / "skill-memory"),
))

# Thresholds
USEFUL_REWARD_THRESHOLD = 0.6
CONFIDENCE_ALPHA = 0.1
MAX_TOPICS = 15
MIN_TOPIC_LEN = 3

# Stop words for topic extraction
_STOP_WORDS = frozenset({
    "the", "a", "an", "is", "it", "to", "in", "for", "of", "and", "or",
    "but", "on", "at", "by", "with", "from", "as", "this", "that", "be",
    "are", "was", "have", "has", "do", "does", "did", "will", "would",
    "can", "could", "should", "not", "so", "if", "all", "also", "just",
    "please", "need", "want", "make", "get", "let", "use", "try", "i",
    "me", "my", "we", "you", "your", "its", "what", "how", "deploy",
})


def _load_entity(skill_name: str) -> Optional[Dict[str, Any]]:
    """Load SEA entity state.json for a skill."""
    entity_dir = SEA_DIR / skill_name
    state_path = entity_dir / "state.json"
    if not state_path.exists():
        return None
    try:
        with open(state_path) as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return None


def _save_entity(skill_name: str, entity: Dict[str, Any]) -> bool:
    """Save SEA entity state.json."""
    entity_dir = SEA_DIR / skill_name
    entity_dir.mkdir(parents=True, exist_ok=True)
    state_path = entity_dir / "state.json"
    try:
        with open(state_path, "w") as f:
            json.dump(entity, f, indent=2)
        return True
    except OSError:
        return False


def _extract_topics(text: str, limit: int = 10) -> List[str]:
    """Extract meaningful topic words from a prompt."""
    words = re.findall(r'[a-z][a-z0-9_-]+', text.lower())
    seen = set()
    topics = []
    for w in words:
        if w not in _STOP_WORDS and len(w) >= MIN_TOPIC_LEN and w not in seen:
            seen.add(w)
            topics.append(w)
            if len(topics) >= limit:
                break
    return topics


def _update_topic_list(
    existing: List[str],
    new_topics: List[str],
    max_items: int = MAX_TOPICS,
) -> List[str]:
    """Merge new topics into existing list, keeping most recent unique."""
    combined = list(dict.fromkeys(new_topics + existing))
    return combined[:max_items]


def _default_entity(skill_name: str) -> Dict[str, Any]:
    """Create a default SEA entity state."""
    return {
        "skill": skill_name,
        "total_activations": 0,
        "useful_activations": 0,
        "suppressed_count": 0,
        "hot_topics": [],
        "cold_topics": [],
        "context_window": 25,
        "confidence_calibration": 0.5,
        "last_activated": None,
    }


def update_entity_from_trajectory(record: Dict[str, Any]) -> Optional[Dict]:
    """Update an SEA entity based on a scored trajectory record.

    Called from flush_session after reward computation.

    Args:
        record: A trajectory record with outcome.reward_score

    Returns:
        Updated entity dict, or None if no entity to update
    """
    skill_name = record.get("skill", {}).get("name")
    if not skill_name:
        return None

    reward = record.get("outcome", {}).get("reward_score")
    if reward is None:
        return None

    # Load or create entity
    entity = _load_entity(skill_name) or _default_entity(skill_name)

    # Update activation counts
    entity["total_activations"] = entity.get("total_activations", 0) + 1
    if reward >= USEFUL_REWARD_THRESHOLD:
        entity["useful_activations"] = entity.get("useful_activations", 0) + 1

    # EMA confidence update
    old_conf = entity.get("confidence_calibration", 0.5)
    entity["confidence_calibration"] = round(
        old_conf * (1 - CONFIDENCE_ALPHA) + reward * CONFIDENCE_ALPHA, 4
    )

    # Update timestamp
    entity["last_activated"] = datetime.now(timezone.utc).isoformat()

    # Extract topics from prompt
    prompt_text = record.get("context", {}).get("prompt_text", "")
    if prompt_text:
        topics = _extract_topics(prompt_text)
        if topics:
            correction = record.get("outcome", {}).get("correction_detected")
            if correction:
                # Failed trajectory: topics become cold
                entity["cold_topics"] = _update_topic_list(
                    entity.get("cold_topics", []), topics
                )
                entity["suppressed_count"] = entity.get("suppressed_count", 0) + 1
            elif reward >= USEFUL_REWARD_THRESHOLD:
                # Successful trajectory: topics become hot
                entity["hot_topics"] = _update_topic_list(
                    entity.get("hot_topics", []), topics
                )

    _save_entity(skill_name, entity)
    return entity


def get_entity_health(skill_name: str) -> Optional[Dict[str, Any]]:
    """Get health metrics for an SEA entity.

    Returns:
        Dict with success_rate, confidence, activation_count, or None
    """
    entity = _load_entity(skill_name)
    if not entity:
        return None

    total = entity.get("total_activations", 0)
    useful = entity.get("useful_activations", 0)

    return {
        "skill": skill_name,
        "total_activations": total,
        "useful_activations": useful,
        "success_rate": round(useful / total, 4) if total > 0 else None,
        "confidence": entity.get("confidence_calibration", 0.5),
        "hot_topics": entity.get("hot_topics", [])[:5],
        "cold_topics": entity.get("cold_topics", [])[:5],
        "last_activated": entity.get("last_activated"),
    }


def get_all_entity_health() -> List[Dict[str, Any]]:
    """Get health metrics for all SEA entities."""
    if not SEA_DIR.exists():
        return []

    results = []
    for entry in sorted(SEA_DIR.iterdir()):
        if entry.is_dir() and (entry / "state.json").exists():
            health = get_entity_health(entry.name)
            if health:
                results.append(health)

    return sorted(results, key=lambda x: x.get("confidence", 0), reverse=True)
