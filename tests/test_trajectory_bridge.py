"""Tests for trajectory_bridge.py — shadow routing analysis + promotion gate."""

import json
import tempfile
from pathlib import Path
from unittest import mock

import pytest

from karl import trajectory_bridge


@pytest.fixture
def temp_data_dir(tmp_path):
    """Set up temporary data directory with shadow and trajectory files."""
    with mock.patch.object(trajectory_bridge, "SHADOW_PATH", tmp_path / "shadow.jsonl"), \
         mock.patch.object(trajectory_bridge, "STORE_PATH", tmp_path / "trajectories.jsonl"), \
         mock.patch.object(trajectory_bridge, "SKILL_EMBEDDINGS_PATH", tmp_path / "skill_embed.json"):
        yield tmp_path


def _write_shadow(path: Path, records: list):
    with open(path, "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")


def _write_trajectories(path: Path, records: list):
    with open(path, "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")


class TestAnalyzeShadowRouting:
    def test_no_data(self, temp_data_dir):
        result = trajectory_bridge.analyze_shadow_routing()
        assert result["status"] == "no_data"

    def test_basic_analysis(self, temp_data_dir):
        shadow_path = temp_data_dir / "shadow.jsonl"
        records = [
            {"vector_status": "hit", "agree": True, "regex": "ops:deploy", "vector": "ops:deploy", "elapsed_ms": 10, "ts": "2026-01-01T00:00:00Z"},
            {"vector_status": "hit", "agree": True, "regex": "ops:ios", "vector": "ops:ios", "elapsed_ms": 15, "ts": "2026-01-01T00:01:00Z"},
            {"vector_status": "miss", "agree": False, "regex": "ops:git", "vector": None, "elapsed_ms": 5, "ts": "2026-01-01T00:02:00Z"},
        ]
        _write_shadow(shadow_path, records)
        result = trajectory_bridge.analyze_shadow_routing()
        assert result["status"] == "ok"
        assert result["records"] == 3
        assert result["cache_hits"] == 2
        assert result["agrees"] == 2
        assert result["agreement_rate"] > 0


class TestAnalyzeSkillHealth:
    def test_no_data(self, temp_data_dir):
        result = trajectory_bridge.analyze_skill_health()
        assert result["status"] == "no_data"

    def test_basic_health(self, temp_data_dir):
        store_path = temp_data_dir / "trajectories.jsonl"
        records = [
            {"skill": {"name": "ops:deploy"}, "outcome": {"reward_score": 0.8}, "trajectory": {"events": [{"tool_name": "Bash"}]}, "session_id": "s1"},
            {"skill": {"name": "ops:deploy"}, "outcome": {"reward_score": 0.6}, "trajectory": {"events": [{"tool_name": "Bash"}]}, "session_id": "s2"},
            {"skill": {"name": "ops:ios"}, "outcome": {"reward_score": 0.9}, "trajectory": {"events": [{"tool_name": "Read"}]}, "session_id": "s3"},
        ]
        _write_trajectories(store_path, records)
        result = trajectory_bridge.analyze_skill_health()
        assert result["status"] == "ok"
        assert "ops:deploy" in result["skills"]
        assert result["skills"]["ops:deploy"]["trajectories"] == 2


class TestPromotionReadiness:
    def test_no_data(self, temp_data_dir):
        result = trajectory_bridge.check_promotion_readiness()
        assert result["ready"] is False

    def test_lift_formula_direction(self, temp_data_dir):
        """Verify X2 fix: lift = mean_agree - mean_disagree (not inverted)."""
        shadow_path = temp_data_dir / "shadow.jsonl"
        store_path = temp_data_dir / "trajectories.jsonl"

        shadow_records = [
            {"agree": True, "vector_status": "hit", "session_id": f"agree_{i}", "regex": "x", "vector": "x"}
            for i in range(60)
        ] + [
            {"agree": False, "vector_status": "hit", "session_id": f"disagree_{i}", "regex": "x", "vector": "y"}
            for i in range(60)
        ]
        _write_shadow(shadow_path, shadow_records)

        traj_records = [
            {"session_id": f"agree_{i}", "skill": {"name": "x"}, "outcome": {"reward_score": 0.9}}
            for i in range(60)
        ] + [
            {"session_id": f"disagree_{i}", "skill": {"name": "x"}, "outcome": {"reward_score": 0.3}}
            for i in range(60)
        ]
        _write_trajectories(store_path, traj_records)

        result = trajectory_bridge.check_promotion_readiness()
        lift = result["checks"]["vector_lift"]["actual"]
        # Agree sessions have higher reward (0.9) than disagree (0.3)
        # lift = mean_agree - mean_disagree = 0.9 - 0.3 = 0.6 (positive)
        assert lift > 0, f"Lift should be positive (agree > disagree), got {lift}"


class TestTechniqueRecommendations:
    def test_no_data(self, temp_data_dir):
        result = trajectory_bridge.technique_recommendations()
        assert result["status"] == "no_data"


class TestFullReport:
    def test_text_report(self, temp_data_dir):
        report = trajectory_bridge.full_report()
        assert "KARL Trajectory Intelligence Report" in report

    def test_json_report(self, temp_data_dir):
        report_json = trajectory_bridge.full_report(as_json=True)
        data = json.loads(report_json)
        assert "shadow_routing" in data
        assert "skill_health" in data
