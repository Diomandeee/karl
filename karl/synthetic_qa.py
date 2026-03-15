#!/usr/bin/env python3
"""
synthetic_qa.py — Synthetic Q&A Generation from codebase changes.

Generates SFT training examples by:
  1. Scanning recent git diffs for meaningful changes
  2. Creating "how would you accomplish X?" questions from the diffs
  3. Generating step-by-step tool-use plans as answers
  4. Writing to train.jsonl format for LoRA fine-tuning

This is KARL Phase 6: self-play data augmentation.
Runs weekly alongside trajectory-based SFT export.

Usage:
    python3 synthetic_qa.py                    # Generate from last 7 days of diffs
    python3 synthetic_qa.py --days 14          # Generate from last 14 days
    python3 synthetic_qa.py --dry-run          # Preview without writing
    python3 synthetic_qa.py --diff-only        # Just show eligible diffs
"""

import json
import os
import re
import subprocess
import sys
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

KARL_DIR = Path(__file__).parent
REPO_ROOT = Path.home()  # Main karl repo
SYNTHETIC_PATH = KARL_DIR / "synthetic_qa.jsonl"

SYSTEM_PROMPT = (
    "You are an expert software engineering assistant. Given a task, plan the "
    "optimal sequence of tool uses to accomplish it efficiently. Consider which "
    "tools to use, in what order, and what parameters. Prefer reading before "
    "editing, testing after changes, and using the most specific tool available."
)

# File patterns to include in synthetic QA
INCLUDE_PATTERNS = [
    r"\.py$", r"\.ts$", r"\.tsx$", r"\.swift$", r"\.js$",
    r"\.yml$", r"\.yaml$", r"\.json$", r"\.md$", r"\.sh$",
]

# Paths to exclude (too noisy or auto-generated)
EXCLUDE_PATHS = [
    "node_modules", ".git", "__pycache__", "dist/", "build/",
    ".next/", "package-lock.json", "yarn.lock", ".DS_Store",
    "*.pyc", "tsconfig.tsbuildinfo",
]

# Minimum diff size to be worth generating QA from
MIN_DIFF_LINES = 5
MAX_DIFF_LINES = 200


def _run_git(args: List[str], cwd: str = None) -> Tuple[int, str]:
    """Run a git command and return (returncode, output)."""
    try:
        result = subprocess.run(
            ["git"] + args,
            cwd=cwd or str(REPO_ROOT),
            capture_output=True, text=True, timeout=30,
        )
        return result.returncode, result.stdout
    except Exception as e:
        return 1, str(e)


def get_recent_commits(days: int = 7, repo: str = None) -> List[Dict]:
    """Get commits from the last N days with their diffs."""
    since = (datetime.now(timezone.utc) - timedelta(days=days)).strftime("%Y-%m-%d")
    rc, log_output = _run_git(
        ["log", f"--since={since}", "--format=%H|%s|%aI", "--no-merges"],
        cwd=repo,
    )
    if rc != 0:
        return []

    commits = []
    for line in log_output.strip().split("\n"):
        if not line or "|" not in line:
            continue
        parts = line.split("|", 2)
        if len(parts) < 3:
            continue
        sha, subject, date = parts
        commits.append({"sha": sha.strip(), "subject": subject.strip(), "date": date.strip()})

    return commits


def get_commit_diff(sha: str, repo: str = None) -> Dict:
    """Get the diff for a specific commit."""
    rc, diff_output = _run_git(
        ["diff", f"{sha}^..{sha}", "--stat", "--no-color"],
        cwd=repo,
    )
    stat = diff_output.strip() if rc == 0 else ""

    rc, diff_output = _run_git(
        ["diff", f"{sha}^..{sha}", "--no-color", "-U3"],
        cwd=repo,
    )
    full_diff = diff_output if rc == 0 else ""

    # Parse changed files
    rc, files_output = _run_git(
        ["diff", f"{sha}^..{sha}", "--name-only"],
        cwd=repo,
    )
    files = [f.strip() for f in files_output.strip().split("\n") if f.strip()] if rc == 0 else []

    return {
        "stat": stat,
        "diff": full_diff,
        "files": files,
        "file_count": len(files),
    }


def _should_include_file(filepath: str) -> bool:
    """Check if a file should be included in synthetic QA."""
    for exclude in EXCLUDE_PATHS:
        if exclude in filepath:
            return False
    return any(re.search(pat, filepath) for pat in INCLUDE_PATTERNS)


def _classify_change(subject: str, files: List[str], diff: str) -> Optional[str]:
    """Classify the type of change for question generation.

    Returns: category string or None if not worth generating QA for.
    """
    subject_lower = subject.lower()

    if any(w in subject_lower for w in ["fix", "bug", "patch", "hotfix"]):
        return "bugfix"
    if any(w in subject_lower for w in ["feat", "add", "implement", "create", "new"]):
        return "feature"
    if any(w in subject_lower for w in ["refactor", "reorganize", "clean", "move"]):
        return "refactor"
    if any(w in subject_lower for w in ["deploy", "infra", "docker", "ci"]):
        return "ops"
    if any(w in subject_lower for w in ["update", "upgrade", "bump"]):
        return "update"
    if any(w in subject_lower for w in ["test", "spec"]):
        return "testing"
    if any(w in subject_lower for w in ["doc", "readme", "comment"]):
        return None  # Skip docs-only changes

    # Fallback: look at file types
    py_files = [f for f in files if f.endswith(".py")]
    swift_files = [f for f in files if f.endswith(".swift")]
    ts_files = [f for f in files if f.endswith((".ts", ".tsx"))]

    if swift_files:
        return "ios"
    if py_files:
        return "python"
    if ts_files:
        return "typescript"

    return "general"


def _generate_question(subject: str, category: str, files: List[str]) -> str:
    """Generate a natural-sounding task question from a commit."""
    # Extract key context from files
    project = ""
    for f in files:
        parts = f.split("/")
        if len(parts) >= 2:
            project = parts[0] if parts[0] not in (".", "src", "lib") else parts[1] if len(parts) > 2 else ""
            break

    # Strip conventional commit prefixes
    clean_subject = re.sub(r"^(feat|fix|refactor|chore|docs|test|ci|build|perf)\([^)]*\):\s*", "", subject)
    clean_subject = re.sub(r"^(feat|fix|refactor|chore|docs|test|ci|build|perf):\s*", "", clean_subject)

    templates = {
        "bugfix": [
            f"{clean_subject}",
            f"Fix the issue: {clean_subject}",
        ],
        "feature": [
            f"{clean_subject}",
            f"Implement: {clean_subject}",
        ],
        "refactor": [
            f"{clean_subject}",
            f"Refactor: {clean_subject}",
        ],
        "ops": [
            f"{clean_subject}",
            f"Set up: {clean_subject}",
        ],
        "ios": [
            f"{clean_subject}" + (f" in {project}" if project else ""),
        ],
        "python": [
            f"{clean_subject}" + (f" in {project}" if project else ""),
        ],
        "typescript": [
            f"{clean_subject}" + (f" in {project}" if project else ""),
        ],
    }

    options = templates.get(category, [clean_subject])
    return options[0]


def _generate_plan_from_diff(files: List[str], diff: str) -> str:
    """Generate a step-by-step tool-use plan from the diff.

    Maps actual file operations to Claude Code tool sequences.
    """
    steps = []
    step_num = 0

    # Group files by operation type
    read_first = []
    edited = []
    created = []
    bash_ops = []

    for f in files:
        if not _should_include_file(f):
            continue

        if f.endswith((".sh", ".service", ".yml", ".yaml")):
            bash_ops.append(f)
        elif f in diff and "+++ b/" + f in diff and "--- a/" + f in diff:
            edited.append(f)
            read_first.append(f)
        elif f in diff and "--- /dev/null" in diff:
            created.append(f)
        else:
            edited.append(f)
            read_first.append(f)

    # Generate steps
    for f in read_first[:5]:
        step_num += 1
        steps.append(f"{step_num}. [ok] Read {f}")

    for f in edited[:5]:
        step_num += 1
        steps.append(f"{step_num}. [ok] Edit {f}")

    for f in created[:3]:
        step_num += 1
        steps.append(f"{step_num}. [ok] Write {f}")

    for f in bash_ops[:2]:
        step_num += 1
        steps.append(f"{step_num}. [ok] Bash: deploy/configure {f}")

    if not steps:
        return ""

    total = len(steps)
    plan = "\n".join(steps)
    plan += f"\n\nResult: {total}/{total} tools succeeded, reward=0.75"
    return plan


def generate_synthetic_qa(
    days: int = 7,
    repo: str = None,
    dry_run: bool = False,
) -> Dict[str, Any]:
    """Generate synthetic Q&A examples from recent commits."""
    commits = get_recent_commits(days=days, repo=repo)
    if not commits:
        return {"status": "no_commits", "days": days}

    examples = []
    skipped = 0

    for commit in commits:
        diff_data = get_commit_diff(commit["sha"], repo=repo)

        # Filter files
        relevant_files = [f for f in diff_data["files"] if _should_include_file(f)]
        if not relevant_files:
            skipped += 1
            continue

        # Check diff size
        diff_lines = diff_data["diff"].count("\n")
        if diff_lines < MIN_DIFF_LINES or diff_lines > MAX_DIFF_LINES:
            skipped += 1
            continue

        # Classify
        category = _classify_change(commit["subject"], relevant_files, diff_data["diff"])
        if not category:
            skipped += 1
            continue

        # Generate Q&A
        question = _generate_question(commit["subject"], category, relevant_files)
        plan = _generate_plan_from_diff(relevant_files, diff_data["diff"])
        if not plan:
            skipped += 1
            continue

        example = {
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": question},
                {"role": "assistant", "content": plan},
            ]
        }
        examples.append(example)

    # Write to file
    if not dry_run and examples:
        SYNTHETIC_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(SYNTHETIC_PATH, "w") as f:
            for ex in examples:
                f.write(json.dumps(ex) + "\n")

    return {
        "status": "ok",
        "commits_scanned": len(commits),
        "examples_generated": len(examples),
        "skipped": skipped,
        "dry_run": dry_run,
        "output_path": str(SYNTHETIC_PATH) if not dry_run else None,
        "sample": examples[0] if examples else None,
    }


if __name__ == "__main__":
    days = 7
    dry_run = "--dry-run" in sys.argv
    diff_only = "--diff-only" in sys.argv

    for arg in sys.argv:
        if arg.startswith("--days"):
            try:
                days = int(sys.argv[sys.argv.index(arg) + 1])
            except (IndexError, ValueError):
                pass

    if diff_only:
        commits = get_recent_commits(days=days)
        for c in commits[:20]:
            diff = get_commit_diff(c["sha"])
            relevant = [f for f in diff["files"] if _should_include_file(f)]
            if relevant:
                category = _classify_change(c["subject"], relevant, diff["diff"])
                print(f"[{category or 'skip'}] {c['subject'][:60]} ({len(relevant)} files)")
    else:
        result = generate_synthetic_qa(days=days, dry_run=dry_run)
        print(json.dumps(result, indent=2, default=str))
