"""Tests for the SFT exporter."""

import json
import pytest
from pathlib import Path
from unittest.mock import patch

import karl.config as config


@pytest.fixture(autouse=True)
def temp_data_dir(tmp_path):
    """Override paths to use temp directory."""
    config.DATA_DIR = tmp_path
    config.STORE_PATH = tmp_path / "trajectories.jsonl"
    config.SYNTHETIC_PATH = tmp_path / "synthetic_qa.jsonl"
    config.TRAIN_PATH = tmp_path / "train.jsonl"
    config.VALID_PATH = tmp_path / "valid.jsonl"
    config.SFT_OUTPUT_PATH = tmp_path / "karl-sft.jsonl"
    yield tmp_path


def _write_trajectory(path: Path, record: dict):
    """Helper to write a trajectory record."""
    with open(path, "a") as f:
        f.write(json.dumps(record) + "\n")


def _make_trajectory(reward: float = 0.6, tools: int = 5):
    """Create a sample trajectory record."""
    return {
        "session_id": f"test-{reward}",
        "skill": {"name": "ops:deploy", "domain": "ops"},
        "context": {"prompt_text": "deploy the service to cloud-vm"},
        "trajectory": {
            "total_tools": tools,
            "successes": tools,
            "failures": 0,
            "bash_errors": 0,
            "tool_counts": {"Read": 2, "Edit": 2, "Bash": 1},
            "events": [
                {"tool_name": "Read", "key_params": {"file_path": "/src/main.py"}, "success": True},
                {"tool_name": "Read", "key_params": {"file_path": "/config.yml"}, "success": True},
                {"tool_name": "Edit", "key_params": {"file_path": "/src/main.py"}, "success": True},
                {"tool_name": "Edit", "key_params": {"file_path": "/config.yml"}, "success": True},
                {"tool_name": "Bash", "key_params": {"command": "pytest"}, "success": True},
            ][:tools],
        },
        "outcome": {"reward_score": reward},
    }


class TestSFTExporter:
    def test_no_store(self, temp_data_dir):
        from karl.sft_exporter import export_sft
        result = export_sft(dry_run=True)
        assert "error" in result

    def test_export_dry_run(self, temp_data_dir):
        from karl.sft_exporter import export_sft
        _write_trajectory(config.STORE_PATH, _make_trajectory(0.9))
        _write_trajectory(config.STORE_PATH, _make_trajectory(0.95))

        result = export_sft(dry_run=True)
        # Both records loaded; at least one should have positive advantage
        assert result["total_records"] == 2
        assert result["examples"] >= 1

    def test_export_creates_files(self, temp_data_dir):
        from karl.sft_exporter import export_sft
        for r in [0.5, 0.6, 0.7, 0.8]:
            _write_trajectory(config.STORE_PATH, _make_trajectory(r))

        result = export_sft()
        assert config.TRAIN_PATH.exists()
        assert config.VALID_PATH.exists()
        assert result["train"] + result["valid"] == result["examples"]

    def test_min_reward_filter(self, temp_data_dir):
        from karl.sft_exporter import export_sft
        _write_trajectory(config.STORE_PATH, _make_trajectory(0.3))
        _write_trajectory(config.STORE_PATH, _make_trajectory(0.9))

        result = export_sft(min_reward=0.5, dry_run=True)
        assert result["total_records"] == 1

    def test_short_trajectory_filtered(self, temp_data_dir):
        from karl.sft_exporter import export_sft
        _write_trajectory(config.STORE_PATH, _make_trajectory(0.8, tools=1))

        result = export_sft(dry_run=True)
        assert result.get("filtered_too_short", 0) >= 1

    def test_synthetic_merge(self, temp_data_dir):
        from karl.sft_exporter import export_sft
        _write_trajectory(config.STORE_PATH, _make_trajectory(0.7))

        synthetic = {
            "messages": [
                {"role": "system", "content": "system"},
                {"role": "user", "content": "fix the login bug"},
                {"role": "assistant", "content": "1. [ok] Read auth.py\n2. [ok] Edit auth.py"},
            ]
        }
        with open(config.SYNTHETIC_PATH, "w") as f:
            f.write(json.dumps(synthetic) + "\n")

        result = export_sft()
        assert result.get("synthetic", 0) >= 1
