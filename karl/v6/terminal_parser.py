"""Parse raw tmux pane output into structured signals."""

import hashlib
import re
from dataclasses import dataclass


ANSI_RE = re.compile(r'\x1b\[[0-9;]*[a-zA-Z]')
TIMESTAMP_RE = re.compile(r'\d{2}:\d{2}:\d{2}')
SPINNER_WORDS = {"synthesizing", "crunched", "roosting", "doodling", "boogieing",
                 "perambulating", "elucidating", "razzle-dazzling"}


@dataclass
class TerminalState:
    is_alive: bool          # Claude running (not raw shell)
    is_working: bool        # Claude mid-task
    error_detected: bool
    last_tool: str | None
    raw_lines: int
    content_hash: str       # MD5 of normalized content
    content: str            # Trimmed for prompt assembly


def strip_ansi(text: str) -> str:
    return ANSI_RE.sub('', text)


def normalize_for_hash(text: str) -> str:
    """Strip volatile content (timestamps, spinners) for stable hashing."""
    text = strip_ansi(text)
    text = TIMESTAMP_RE.sub('', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


def parse_terminal(raw_output: str, error_mode: bool = False) -> TerminalState:
    """Parse raw tmux output into structured state."""
    clean = strip_ansi(raw_output)
    lines = [l for l in clean.split('\n') if l.strip()]

    # Session alive check
    last_5 = '\n'.join(lines[-5:]).lower() if lines else ''
    is_alive = True
    if re.search(r'(zsh|bash)\s*$', last_5):
        if not any(c in last_5 for c in ['❯', 'claude', 'synthesizing', 'crunched']):
            is_alive = False

    # Working check — structural: look for tool activity indicators
    is_working = False
    for word in SPINNER_WORDS:
        if word in last_5:
            is_working = True
            break
    if 'running' in last_5 and ('bash' in last_5 or 'tool' in last_5):
        is_working = True

    # Error detection
    error_detected = any(kw in last_5 for kw in ['error', 'traceback', 'failed', 'exception', 'panic'])

    # Last tool
    last_tool = None
    tool_re = re.compile(r'(Read|Write|Edit|Bash|Grep|Glob|Agent)\(', re.IGNORECASE)
    for line in reversed(lines[-20:]):
        m = tool_re.search(line)
        if m:
            last_tool = m.group(1)
            break

    # Content hash (stable, ignores timestamps and spinners)
    hash_input = normalize_for_hash('\n'.join(lines[-40:]))
    content_hash = hashlib.md5(hash_input.encode()).hexdigest()[:12]

    # Adaptive line count
    if error_detected:
        max_lines = 120
    elif is_working:
        max_lines = 30
    else:
        max_lines = 60

    content = '\n'.join(lines[-max_lines:])

    return TerminalState(
        is_alive=is_alive,
        is_working=is_working,
        error_detected=error_detected,
        last_tool=last_tool,
        raw_lines=len(lines),
        content_hash=content_hash,
        content=content,
    )
