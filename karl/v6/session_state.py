"""Per-session state tracking. Survives restarts via JSON-on-disk."""

import hashlib
import json
import os
import tempfile
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path

STATE_DIR = Path.home() / ".karl-sessions"
STATE_DIR.mkdir(parents=True, exist_ok=True)


@dataclass
class SessionState:
    session_id: str
    machine: str
    pane_id: str
    project_name: str = ""
    project_dir: str = ""
    goal: str = ""
    plan_steps: list[str] = field(default_factory=list)
    phase: str = "EXPLORE"  # EXPLORE | BUILD | CLOSE | STUCK | IDLE
    turn: int = 0
    max_turns: int = 30
    last_prompts: list[str] = field(default_factory=list)  # ring buffer of 5
    consecutive_dupes: int = 0
    consecutive_status: int = 0
    pane_hash: str = ""
    pane_hash_streak: int = 0
    goals_completed: list[str] = field(default_factory=list)
    digest_entries: list[dict] = field(default_factory=list)  # compressed turns
    created_at: float = 0.0
    updated_at: float = 0.0

    def __post_init__(self):
        if not self.created_at:
            self.created_at = time.time()
        self.updated_at = time.time()

    @staticmethod
    def make_id(machine: str, pane_id: str) -> str:
        safe = pane_id.replace(":", "_").replace(".", "_")
        return f"{machine}_{safe}"

    def record_prompt(self, prompt: str):
        self.last_prompts.append(prompt)
        if len(self.last_prompts) > 5:
            self.last_prompts = self.last_prompts[-5:]

        # Track status spam
        if prompt.strip().lower() == "status":
            self.consecutive_status += 1
        else:
            self.consecutive_status = 0

        # Track exact dupes
        if len(self.last_prompts) >= 2 and self.last_prompts[-1] == self.last_prompts[-2]:
            self.consecutive_dupes += 1
        else:
            self.consecutive_dupes = 0

        self.turn += 1
        self.updated_at = time.time()

    def record_digest(self, summary: str, outcome: str, tools: list[str]):
        self.digest_entries.append({
            "turn": self.turn, "summary": summary[:150],
            "outcome": outcome, "tools": tools,
        })
        if len(self.digest_entries) > 8:
            self.digest_entries = self.digest_entries[-8:]

    def update_phase(self, turn_pct: float):
        if self.is_stuck():
            self.phase = "STUCK"
        elif turn_pct <= 0.30:
            self.phase = "EXPLORE"
        elif turn_pct <= 0.80:
            self.phase = "BUILD"
        else:
            self.phase = "CLOSE"

    def is_stuck(self) -> bool:
        return (self.consecutive_dupes >= 2 or
                self.consecutive_status >= 2 or
                self.pane_hash_streak >= 3)

    def update_pane_hash(self, content_hash: str):
        if content_hash == self.pane_hash:
            self.pane_hash_streak += 1
        else:
            self.pane_hash = content_hash
            self.pane_hash_streak = 0

    def to_brief(self) -> str:
        pct = f"{self.turn}/{self.max_turns}"
        goals_done = len(self.goals_completed)
        goals_total = len(self.plan_steps)
        return (
            f"Project: {self.project_name} ({self.project_dir})\n"
            f"Goal: {self.goal}\n"
            f"Turn: {pct} | Phase: {self.phase} | "
            f"Goals: {goals_done}/{goals_total} done"
        )

    def to_digest(self) -> str:
        if not self.digest_entries:
            return "No turns recorded yet."
        lines = []
        for d in self.digest_entries[-5:]:
            lines.append(f"Turn {d['turn']}: {d['summary']} [{d['outcome']}]")
        return "\n".join(lines)

    def save(self):
        path = STATE_DIR / f"{self.session_id}.json"
        fd, tmp_name = tempfile.mkstemp(
            dir=STATE_DIR,
            prefix=f"{self.session_id}.",
            suffix=".tmp",
        )
        try:
            with os.fdopen(fd, "w") as f:
                json.dump(asdict(self), f, indent=2)
            os.replace(tmp_name, path)  # atomic, even under overlapping writers
        finally:
            if os.path.exists(tmp_name):
                os.remove(tmp_name)

    @classmethod
    def load(cls, session_id: str) -> "SessionState | None":
        path = STATE_DIR / f"{session_id}.json"
        if not path.exists():
            return None
        try:
            with open(path) as f:
                data = json.load(f)
            return cls(**data)
        except (json.JSONDecodeError, TypeError):
            return None
