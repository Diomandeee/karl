"""Tests for cli.py — command-line interface."""

import sys
from unittest import mock

import pytest

from karl.cli import main


class TestCLICommands:
    def test_no_args_shows_help(self, capsys):
        with mock.patch("sys.argv", ["karl"]):
            with pytest.raises(SystemExit) as exc_info:
                main()
            assert exc_info.value.code == 1

    def test_status_runs(self, tmp_path):
        with mock.patch("sys.argv", ["karl", "status"]), \
             mock.patch("karl.config.DATA_DIR", tmp_path), \
             mock.patch("karl.config.BUFFER_DIR", tmp_path / "buffers"), \
             mock.patch("karl.config.STORE_PATH", tmp_path / "traj.jsonl"):
            main()

    def test_backfill_runs(self, tmp_path):
        store = tmp_path / "traj.jsonl"
        store.touch()
        with mock.patch("sys.argv", ["karl", "backfill"]), \
             mock.patch("karl.config.DATA_DIR", tmp_path), \
             mock.patch("karl.config.BUFFER_DIR", tmp_path / "buffers"), \
             mock.patch("karl.config.STORE_PATH", store):
            main()

    def test_export_dry_run(self, tmp_path):
        with mock.patch("sys.argv", ["karl", "export", "--dry-run"]), \
             mock.patch("karl.config.DATA_DIR", tmp_path), \
             mock.patch("karl.config.BUFFER_DIR", tmp_path / "buffers"), \
             mock.patch("karl.config.STORE_PATH", tmp_path / "traj.jsonl"):
            main()
