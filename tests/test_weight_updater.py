"""Tests for weight_updater.py — EMA skill weight updates."""

import json
import tempfile
from pathlib import Path
from unittest import mock

import pytest

from karl import weight_updater
from karl.config import WEIGHT_ALPHA, WEIGHT_MIN, WEIGHT_MAX


class TestRewardToTarget:
    def test_zero_reward(self):
        assert weight_updater._reward_to_target(0.0) == WEIGHT_MIN

    def test_half_reward(self):
        assert weight_updater._reward_to_target(0.5) == pytest.approx(1.0, abs=0.01)

    def test_full_reward(self):
        assert weight_updater._reward_to_target(1.0) == WEIGHT_MAX


class TestEmaUpdate:
    def test_same_target(self):
        result = weight_updater._ema_update(1.0, 1.0)
        assert result == pytest.approx(1.0)

    def test_bounds(self):
        result = weight_updater._ema_update(0.1, 0.0, alpha=1.0)
        assert result >= WEIGHT_MIN

        result = weight_updater._ema_update(2.0, 2.0, alpha=1.0)
        assert result <= WEIGHT_MAX


class TestCollectSkillRewards:
    def test_empty_store(self, tmp_path):
        with mock.patch("karl.weight_updater.STORE_PATH", tmp_path / "missing.jsonl"):
            result = weight_updater.collect_skill_rewards()
            assert result == {}

    def test_collects_by_skill(self, tmp_path):
        store = tmp_path / "traj.jsonl"
        records = [
            {"skill": {"name": "ops:deploy"}, "outcome": {"reward_score": 0.8}},
            {"skill": {"name": "ops:deploy"}, "outcome": {"reward_score": 0.6}},
            {"skill": {"name": "ops:ios"}, "outcome": {"reward_score": 0.9}},
        ]
        with open(store, "w") as f:
            for r in records:
                f.write(json.dumps(r) + "\n")

        with mock.patch("karl.weight_updater.STORE_PATH", store):
            result = weight_updater.collect_skill_rewards()
            assert len(result["ops:deploy"]) == 2
            assert len(result["ops:ios"]) == 1


class TestUpdateWeights:
    def test_no_embeddings(self, tmp_path):
        with mock.patch("karl.weight_updater.load_skill_embeddings", return_value={}):
            result = weight_updater.update_weights()
            assert "error" in result

    def test_no_rewards(self, tmp_path):
        embeddings = {"ops:deploy": ([0.1] * 10, 1.0)}
        with mock.patch("karl.weight_updater.load_skill_embeddings", return_value=embeddings), \
             mock.patch("karl.weight_updater.collect_skill_rewards", return_value={}):
            result = weight_updater.update_weights()
            assert result["message"] == "No reward data yet"

    def test_dry_run(self, tmp_path):
        embeddings = {"ops:deploy": ([0.1] * 10, 1.0)}
        rewards = {"ops:deploy": [0.8, 0.9]}
        with mock.patch("karl.weight_updater.load_skill_embeddings", return_value=embeddings), \
             mock.patch("karl.weight_updater.collect_skill_rewards", return_value=rewards), \
             mock.patch("karl.weight_updater.save_skill_embeddings") as save_mock:
            result = weight_updater.update_weights(dry_run=True)
            assert result["updated"] == 1
            assert result["dry_run"] is True
            save_mock.assert_not_called()
