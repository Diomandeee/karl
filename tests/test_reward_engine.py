"""Tests for the KARL reward engine."""

import pytest
from karl.reward_engine import (
    compute_reward,
    compute_advantage,
    _compute_outcome,
    _compute_process,
    _compute_efficiency,
    _max_consecutive_failures,
)


class TestComputeReward:
    """Test composite reward computation."""

    def test_empty_record(self):
        record = {"outcome": {}, "trajectory": {}, "timing": {}}
        result = compute_reward(record)
        assert 0 <= result["reward_score"] <= 1
        assert "outcome_score" in result
        assert "process_score" in result
        assert "efficiency_score" in result
        assert "components" in result

    def test_perfect_trajectory(self):
        record = {
            "outcome": {
                "correction_detected": False,
                "redo_detected": False,
                "build_success": True,
                "session_continued": True,
            },
            "trajectory": {
                "total_tools": 5,
                "successes": 5,
                "failures": 0,
                "bash_errors": 0,
                "tool_counts": {"Read": 2, "Edit": 2, "Bash": 1},
                "events": [
                    {"tool_name": "Read", "success": True},
                    {"tool_name": "Read", "success": True},
                    {"tool_name": "Edit", "success": True},
                    {"tool_name": "Edit", "success": True},
                    {"tool_name": "Bash", "success": True},
                ],
            },
            "timing": {"duration_s": 120},
        }
        result = compute_reward(record)
        assert result["reward_score"] > 0.7
        assert result["outcome_score"] > 0.8

    def test_bad_trajectory(self):
        record = {
            "outcome": {
                "correction_detected": True,
                "redo_detected": True,
            },
            "trajectory": {
                "total_tools": 3,
                "successes": 0,
                "failures": 3,
                "bash_errors": 2,
                "tool_counts": {"Bash": 3},
                "events": [
                    {"tool_name": "Bash", "success": False, "exit_code": 1},
                    {"tool_name": "Bash", "success": False, "exit_code": 1},
                    {"tool_name": "Bash", "success": False, "exit_code": 1},
                ],
            },
            "timing": {"duration_s": 60},
        }
        result = compute_reward(record)
        assert result["reward_score"] < 0.3


class TestOutcomeScore:
    """Test outcome signal scoring."""

    def test_no_signals(self):
        score, components = _compute_outcome({})
        assert score == 0.5  # Neutral baseline

    def test_no_correction(self):
        score, _ = _compute_outcome({"correction_detected": False})
        assert score > 0.5

    def test_correction_detected(self):
        score, _ = _compute_outcome({"correction_detected": True})
        assert score < 0.5

    def test_all_positive(self):
        score, _ = _compute_outcome({
            "correction_detected": False,
            "redo_detected": False,
            "build_success": True,
            "session_continued": True,
        })
        assert score >= 0.9

    def test_score_bounded(self):
        score, _ = _compute_outcome({
            "correction_detected": True,
            "redo_detected": True,
            "build_success": False,
        })
        assert 0.0 <= score <= 1.0


class TestProcessScore:
    """Test process quality scoring."""

    def test_empty_trajectory(self):
        score, _ = _compute_process({"total_tools": 0})
        assert score == 0.5

    def test_all_success(self):
        score, components = _compute_process({
            "total_tools": 5,
            "successes": 5,
            "failures": 0,
            "bash_errors": 0,
            "events": [{"tool_name": "Read", "success": True}] * 5,
        })
        assert score > 0.8
        assert components["r_success_rate"] == 1.0

    def test_bash_errors_penalized(self):
        events = [
            {"tool_name": "Bash", "success": False, "exit_code": 1},
            {"tool_name": "Bash", "success": False, "exit_code": 1},
        ]
        score, components = _compute_process({
            "total_tools": 2,
            "successes": 0,
            "failures": 2,
            "bash_errors": 2,
            "events": events,
        })
        assert components["r_bash_clean"] == 0.0


class TestEfficiencyScore:
    """Test efficiency scoring."""

    def test_monoculture_penalty(self):
        score, components = _compute_efficiency(
            {"total_tools": 10, "tool_counts": {"Bash": 10}},
            {"duration_s": 60},
        )
        assert components["r_diversity"] == 0.3

    def test_diverse_tools(self):
        score, components = _compute_efficiency(
            {"total_tools": 4, "tool_counts": {"Read": 1, "Edit": 1, "Bash": 1, "Grep": 1}},
            {"duration_s": 120},
        )
        assert components["r_diversity"] > 0.9  # High entropy


class TestConsecutiveFailures:
    """Test consecutive failure detection."""

    def test_no_failures(self):
        events = [{"success": True}, {"success": True}]
        assert _max_consecutive_failures(events) == 0

    def test_single_failure(self):
        events = [{"success": True}, {"success": False}, {"success": True}]
        assert _max_consecutive_failures(events) == 1

    def test_consecutive_failures(self):
        events = [
            {"success": False}, {"success": False}, {"success": False},
            {"success": True}, {"success": False},
        ]
        assert _max_consecutive_failures(events) == 3


class TestAdvantage:
    """Test advantage computation."""

    def test_above_baseline(self):
        adv = compute_advantage({}, 0.7, domain_baseline=0.5)
        assert adv > 0

    def test_below_baseline(self):
        adv = compute_advantage({}, 0.3, domain_baseline=0.5)
        assert adv < 0

    def test_default_baseline(self):
        adv = compute_advantage({}, 0.5)
        assert adv == 0.0
