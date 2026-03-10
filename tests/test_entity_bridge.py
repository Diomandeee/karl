"""Tests for the entity bridge."""

import json
import pytest
from pathlib import Path

import karl.entity_bridge as bridge


@pytest.fixture(autouse=True)
def temp_sea_dir(tmp_path):
    """Override SEA directory to temp."""
    bridge.SEA_DIR = tmp_path
    yield tmp_path


def _make_record(skill="ops:deploy", reward=0.7, correction=False, prompt="deploy to cloud-vm"):
    return {
        "skill": {"name": skill, "domain": skill.split(":")[0] if ":" in skill else None},
        "context": {"prompt_text": prompt},
        "outcome": {
            "reward_score": reward,
            "correction_detected": correction,
        },
    }


class TestUpdateEntity:
    def test_creates_entity(self, tmp_path):
        record = _make_record()
        entity = bridge.update_entity_from_trajectory(record)
        assert entity is not None
        assert entity["total_activations"] == 1
        assert entity["useful_activations"] == 1  # 0.7 >= 0.6

    def test_increments_activations(self, tmp_path):
        bridge.update_entity_from_trajectory(_make_record(reward=0.8))
        bridge.update_entity_from_trajectory(_make_record(reward=0.3))
        entity = bridge.update_entity_from_trajectory(_make_record(reward=0.9))
        assert entity["total_activations"] == 3
        assert entity["useful_activations"] == 2  # 0.8 and 0.9

    def test_confidence_ema(self, tmp_path):
        # Start at 0.5, update with 1.0 reward
        entity = bridge.update_entity_from_trajectory(_make_record(reward=1.0))
        # EMA: 0.5 * 0.9 + 1.0 * 0.1 = 0.55
        assert entity["confidence_calibration"] == 0.55

    def test_hot_topics_from_success(self, tmp_path):
        entity = bridge.update_entity_from_trajectory(
            _make_record(reward=0.8, prompt="deploy docker containers to production")
        )
        assert len(entity["hot_topics"]) > 0
        assert "docker" in entity["hot_topics"]

    def test_cold_topics_from_correction(self, tmp_path):
        entity = bridge.update_entity_from_trajectory(
            _make_record(reward=0.3, correction=True, prompt="fix the broken auth system")
        )
        assert len(entity["cold_topics"]) > 0
        assert "broken" in entity["cold_topics"] or "auth" in entity["cold_topics"]

    def test_no_skill_returns_none(self, tmp_path):
        record = {"skill": {}, "outcome": {"reward_score": 0.5}}
        assert bridge.update_entity_from_trajectory(record) is None

    def test_no_reward_returns_none(self, tmp_path):
        record = {"skill": {"name": "ops:deploy"}, "outcome": {}}
        assert bridge.update_entity_from_trajectory(record) is None

    def test_persists_to_disk(self, tmp_path):
        bridge.update_entity_from_trajectory(_make_record())
        state_path = tmp_path / "ops:deploy" / "state.json"
        assert state_path.exists()
        data = json.loads(state_path.read_text())
        assert data["skill"] == "ops:deploy"


class TestEntityHealth:
    def test_no_entity(self, tmp_path):
        assert bridge.get_entity_health("nonexistent") is None

    def test_health_after_updates(self, tmp_path):
        bridge.update_entity_from_trajectory(_make_record(reward=0.8))
        bridge.update_entity_from_trajectory(_make_record(reward=0.4))
        health = bridge.get_entity_health("ops:deploy")
        assert health["total_activations"] == 2
        assert health["useful_activations"] == 1
        assert health["success_rate"] == 0.5

    def test_all_entity_health(self, tmp_path):
        bridge.update_entity_from_trajectory(_make_record(skill="ops:deploy", reward=0.8))
        bridge.update_entity_from_trajectory(_make_record(skill="ops:git", reward=0.6))
        results = bridge.get_all_entity_health()
        assert len(results) == 2
        # Sorted by confidence descending
        assert results[0]["skill"] == "ops:deploy"


class TestExtractTopics:
    def test_basic(self):
        topics = bridge._extract_topics("deploy the docker containers to production server")
        assert "docker" in topics
        assert "containers" in topics
        assert "production" in topics
        # Stop words excluded
        assert "the" not in topics

    def test_short_words_excluded(self):
        topics = bridge._extract_topics("go to vm")
        assert "go" not in topics
        assert "to" not in topics
        assert "vm" not in topics  # only 2 chars
