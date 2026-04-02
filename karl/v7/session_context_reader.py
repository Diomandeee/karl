"""Extract structured session context from raw tmux pane output.

Parses Claude Code tool output to determine what has been built,
what errors occurred, and what the agent is currently doing.
Feeds into the simulator for grounded prompt generation.
"""

import re
from dataclasses import dataclass, field

# Tool output patterns (Claude Code format)
WRITE_PAT = re.compile(r"(?:Write|Created|Wrote)\s+(?:to\s+)?['\"]?(/[^\s'\"]+)")
EDIT_PAT = re.compile(r"Edit(?:ed)?\s+(?:file\s+)?['\"]?(/[^\s'\"]+)")
READ_PAT = re.compile(r"Read\s+['\"]?(/[^\s'\"]+)")
BASH_PAT = re.compile(r"(?:Bash|Running|ran)\s*[:\(]?\s*(.{10,80})")
GREP_PAT = re.compile(r"Grep\s+(?:for\s+)?['\"]?([^\s'\"]+)")
GLOB_PAT = re.compile(r"Glob\s+['\"]?([^\s'\"]+)")

ERROR_PATTERNS = [
    re.compile(r"(?:Error|ERROR|error):\s*(.{10,200})"),
    re.compile(r"Traceback \(most recent call last\)"),
    re.compile(r"(?:FAILED|FAIL|failed)\s*[-:]?\s*(.{0,100})"),
    re.compile(r"panic:\s*(.{10,200})"),
    re.compile(r"(?:ModuleNotFoundError|ImportError|SyntaxError|TypeError|ValueError):\s*(.+)"),
    re.compile(r"(?:fatal|FATAL):\s*(.{10,200})"),
    re.compile(r"command not found:\s*(\S+)"),
]

# Lines of code heuristic
CODE_LINE_PAT = re.compile(r"^\s*[\w\d\.\(\)\[\]{}<>=+\-*/,;:!@#$%^&|~`]+", re.MULTILINE)

TOOL_NAMES = {"Read", "Write", "Edit", "Bash", "Grep", "Glob", "Agent", "WebFetch", "WebSearch"}


@dataclass
class SessionContext:
    files_created: list[str] = field(default_factory=list)
    files_modified: list[str] = field(default_factory=list)
    errors_seen: list[str] = field(default_factory=list)
    tools_used: dict[str, int] = field(default_factory=dict)
    last_claude_action: str = ""
    project_dir: str = ""
    lines_of_code_written: int = 0

    def summary(self) -> str:
        """One-line summary for prompt generation context."""
        parts = []
        if self.files_created:
            parts.append(f"created {len(self.files_created)} files")
        if self.files_modified:
            parts.append(f"modified {len(self.files_modified)} files")
        if self.errors_seen:
            parts.append(f"{len(self.errors_seen)} errors")
        if self.lines_of_code_written > 0:
            parts.append(f"~{self.lines_of_code_written} lines written")
        if self.last_claude_action:
            parts.append(f"last: {self.last_claude_action}")
        return "; ".join(parts) if parts else "session just started"

    def keywords(self) -> list[str]:
        """Extract keywords for TF-IDF search against corpus."""
        words = []
        for f in self.files_created + self.files_modified:
            # Extract meaningful parts of file paths
            parts = f.replace("/", " ").replace("_", " ").replace("-", " ").replace(".", " ").split()
            words.extend(p.lower() for p in parts if len(p) > 2)
        for err in self.errors_seen[:3]:
            words.extend(w.lower() for w in err.split()[:6] if len(w) > 2)
        if self.last_claude_action:
            words.extend(w.lower() for w in self.last_claude_action.split() if len(w) > 2)
        return words


def parse_session_context(raw_output: str, project_dir: str = "") -> SessionContext:
    """Parse raw tmux pane output into structured SessionContext."""
    ctx = SessionContext(project_dir=project_dir)
    lines = raw_output.split("\n")

    seen_files_created: set[str] = set()
    seen_files_modified: set[str] = set()
    tool_counts: dict[str, int] = {t: 0 for t in TOOL_NAMES}

    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue

        # Detect tool usage
        for tool in TOOL_NAMES:
            if tool in stripped:
                tool_counts[tool] += 1

        # File creation (Write tool)
        m = WRITE_PAT.search(stripped)
        if m:
            path = m.group(1)
            if path not in seen_files_created:
                seen_files_created.add(path)
                ctx.files_created.append(path)

        # File modification (Edit tool)
        m = EDIT_PAT.search(stripped)
        if m:
            path = m.group(1)
            if path not in seen_files_modified and path not in seen_files_created:
                seen_files_modified.add(path)
                ctx.files_modified.append(path)

        # Error detection
        for pat in ERROR_PATTERNS:
            m = pat.search(stripped)
            if m:
                err_text = m.group(1) if m.lastindex else stripped[:150]
                err_text = err_text.strip()
                if err_text and err_text not in ctx.errors_seen:
                    ctx.errors_seen.append(err_text[:200])
                break

    ctx.tools_used = {k: v for k, v in tool_counts.items() if v > 0}

    # Estimate lines of code written from Write/Edit output density
    write_count = tool_counts.get("Write", 0)
    edit_count = tool_counts.get("Edit", 0)
    # Rough: each Write ~30 lines, each Edit ~5 lines
    ctx.lines_of_code_written = write_count * 30 + edit_count * 5

    # Last Claude action: scan from bottom for the most recent tool call
    last_action = _extract_last_action(lines)
    ctx.last_claude_action = last_action

    # Cap errors to 5 most recent
    ctx.errors_seen = ctx.errors_seen[-5:]

    return ctx


def _extract_last_action(lines: list[str]) -> str:
    """Find the most recent meaningful action from terminal output."""
    # Scan from bottom
    for line in reversed(lines[-50:]):
        stripped = line.strip()
        if not stripped:
            continue

        # Write/create
        m = WRITE_PAT.search(stripped)
        if m:
            fname = m.group(1).split("/")[-1]
            return f"wrote {fname}"

        # Edit
        m = EDIT_PAT.search(stripped)
        if m:
            fname = m.group(1).split("/")[-1]
            return f"edited {fname}"

        # Bash command
        m = BASH_PAT.search(stripped)
        if m:
            cmd = m.group(1).strip()[:60]
            return f"ran: {cmd}"

        # Read
        m = READ_PAT.search(stripped)
        if m:
            fname = m.group(1).split("/")[-1]
            return f"read {fname}"

        # Test results
        if re.search(r"\d+ passed", stripped, re.IGNORECASE):
            return "tests ran"
        if "commit" in stripped.lower() and ("success" in stripped.lower() or "[" in stripped):
            return "committed changes"

    return ""


if __name__ == "__main__":
    # Synthetic test
    sample_output = """
    Read /Users/mohameddiomande/Desktop/karl/karl/v6/driver.py
    Wrote to /Users/mohameddiomande/Desktop/karl/karl/v7/prompt_corpus.py
    Edit /Users/mohameddiomande/Desktop/karl/karl/v7/__init__.py
    Bash: python3 -m pytest tests/ -v
    Error: ModuleNotFoundError: No module named 'karl.v7'
    Wrote to /Users/mohameddiomande/Desktop/karl/karl/v7/style_validator.py
    Bash: ruff check karl/v7/
    5 passed, 2 failed
    """
    ctx = parse_session_context(sample_output, "/Users/mohameddiomande/Desktop/karl")
    print(f"Files created: {ctx.files_created}")
    print(f"Files modified: {ctx.files_modified}")
    print(f"Errors: {ctx.errors_seen}")
    print(f"Tools: {ctx.tools_used}")
    print(f"Last action: {ctx.last_claude_action}")
    print(f"Lines written: {ctx.lines_of_code_written}")
    print(f"Summary: {ctx.summary()}")
    print(f"Keywords: {ctx.keywords()}")
