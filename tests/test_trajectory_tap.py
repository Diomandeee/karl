"""Tests for trajectory tap (recording) system."""

import json
import os
import tempfile
import pytest
from pathlib import Path
from unittest.mock import patch

import karl.config as config


@pytest.fixture(autouse=True)
def temp_data_dir(tmp_path):
    """Override KARL data paths to use temp directory."""
    config.DATA_DIR = tmp_path
    config.BUFFER_DIR = tmp_path / "buffers"
    config.STORE_PATH = tmp_path / "trajectories.jsonl"
    config.BUFFER_DIR.mkdir(parents=True, exist_ok=True)

    # Re-import to pick up new paths
    import importlib
    import karl.trajectory_tap as tap
    tap.BUFFER_DIR = config.BUFFER_DIR
    tap.STORE_PATH = config.STORE_PATH

    # Patch the module-level references
    with patch.object(tap, 'BUFFER_DIR', config.BUFFER_DIR), \
         patch.object(tap, 'STORE_PATH', config.STORE_PATH):
        yield tmp_path


class TestInitSessionBuffer:
    def test_creates_buffer(self, temp_data_dir):
        from karl.trajectory_tap import init_session_buffer
        result = init_session_buffer("test-session-123", skill_name="ops:deploy")
        assert result is True
        buf_path = config.BUFFER_DIR / "test-session-123.json"
        assert buf_path.exists()

    def test_buffer_content(self, temp_data_dir):
        from karl.trajectory_tap import init_session_buffer
        init_session_buffer(
            "session-abc",
            skill_name="ops:ios",
            prompt_text="build the iOS app",
            cwd="/Users/test/project",
        )
        buf_path = config.BUFFER_DIR / "session-abc.json"
        with open(buf_path) as f:
            data = json.load(f)
        assert data["skill_name"] == "ops:ios"
        assert data["prompt_text"] == "build the iOS app"
        assert data["tool_events"] == []

    def test_sanitizes_session_id(self, temp_data_dir):
        from karl.trajectory_tap import init_session_buffer
        result = init_session_buffer("../../../etc/passwd")
        assert result is True
        # Should not create file outside buffer dir
        assert not Path("/etc/passwd.json").exists()


class TestAppendToolEvent:
    def test_appends_event(self, temp_data_dir):
        from karl.trajectory_tap import init_session_buffer, append_tool_event
        init_session_buffer("s1")
        result = append_tool_event(
            "s1",
            tool_name="Read",
            tool_input={"file_path": "/test/file.py"},
            success=True,
        )
        assert result is True

        buf_path = config.BUFFER_DIR / "s1.json"
        with open(buf_path) as f:
            data = json.load(f)
        assert len(data["tool_events"]) == 1
        assert data["tool_events"][0]["tool_name"] == "Read"

    def test_auto_creates_buffer(self, temp_data_dir):
        from karl.trajectory_tap import append_tool_event
        result = append_tool_event("new-session", tool_name="Bash")
        assert result is True

    def test_truncates_params(self, temp_data_dir):
        from karl.trajectory_tap import init_session_buffer, append_tool_event
        init_session_buffer("s2")
        long_cmd = "x" * 500
        append_tool_event("s2", "Bash", tool_input={"command": long_cmd})

        buf_path = config.BUFFER_DIR / "s2.json"
        with open(buf_path) as f:
            data = json.load(f)
        assert len(data["tool_events"][0]["key_params"]["command"]) == 200


class TestFlushSession:
    def test_flush_writes_to_store(self, temp_data_dir):
        from karl.trajectory_tap import init_session_buffer, append_tool_event, flush_session
        init_session_buffer("flush-test")
        append_tool_event("flush-test", "Read", success=True)
        append_tool_event("flush-test", "Edit", success=True)

        record = flush_session("flush-test")
        assert record is not None
        assert record["trajectory"]["total_tools"] == 2
        assert record["trajectory"]["successes"] == 2
        assert config.STORE_PATH.exists()

    def test_flush_empty_skips(self, temp_data_dir):
        from karl.trajectory_tap import init_session_buffer, flush_session
        init_session_buffer("empty-session")
        record = flush_session("empty-session")
        assert record is None

    def test_flush_cleans_buffer(self, temp_data_dir):
        from karl.trajectory_tap import init_session_buffer, append_tool_event, flush_session
        init_session_buffer("cleanup-test")
        append_tool_event("cleanup-test", "Bash", success=True)
        flush_session("cleanup-test")
        assert not (config.BUFFER_DIR / "cleanup-test.json").exists()


class TestGetStoreStats:
    def test_empty_store(self, temp_data_dir):
        from karl.trajectory_tap import get_store_stats
        stats = get_store_stats()
        assert stats["total"] == 0

    def test_with_records(self, temp_data_dir):
        from karl.trajectory_tap import init_session_buffer, append_tool_event, flush_session, get_store_stats
        init_session_buffer("s1", skill_name="ops:deploy")
        append_tool_event("s1", "Bash", success=True)
        flush_session("s1")

        stats = get_store_stats()
        assert stats["total"] == 1
        assert stats["skills"].get("ops:deploy", 0) >= 1
