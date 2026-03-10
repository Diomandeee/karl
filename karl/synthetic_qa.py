"""
synthetic_qa.py - Synthetic Q&A generation from codebase changes.

Generates SFT training examples by:
  1. Scanning recent git diffs for meaningful changes
  2. Creating task questions from commit messages
  3. Generating step-by-step tool-use plans from actual diffs
  4. Writing ChatML JSONL for LoRA fine-tuning

This is KARL's self-play data augmentation pipeline.
Bridges the gap between live trajectory count and training data needs.

Usage:
    from karl.synthetic_qa import generate_synthetic_qa
    stats = generate_synthetic_qa(days=14)
    stats = generate_synthetic_qa(days=7, dry_run=True)
"""

import json
import re
import subprocess
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from karl.config import (
    SYNTHETIC_PATH,
    SFT_SYSTEM_PROMPT,
    SYNTHETIC_MIN_DIFF_LINES,
    SYNTHETIC_MAX_DIFF_LINES,
    SYNTHETIC_DEFAULT_DAYS,
)

# File patterns to include
INCLUDE_PATTERNS = [
    r"\.py$", r"\.ts$", r"\.tsx$", r"\.swift$", r"\.js$",
    r"\.yml$", r"\.yaml$", r"\.json$", r"\.md$", r"\.sh$",
]

# Paths to exclude (noisy or auto-generated)
EXCLUDE_PATHS = [
    "node_modules", ".git", "__pycache__", "dist/", "build/",
    ".next/", "package-lock.json", "yarn.lock", ".DS_Store",
    "*.pyc", "tsconfig.tsbuildinfo",
]


def _run_git(args: List[str], cwd: Optional[str] = None) -> Tuple[int, str]:
    """Run a git command and return (returncode, output)."""
    try:
        result = subprocess.run(
            ["git"] + args,
            cwd=cwd or str(Path.home()),
            capture_output=True, text=True, timeout=30,
        )
        return result.returncode, result.stdout
    except Exception as e:
        return 1, str(e)


def get_recent_commits(days: int = SYNTHETIC_DEFAULT_DAYS, repo: Optional[str] = None) -> List[Dict]:
    """Get commits from the last N days with metadata."""
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


def get_commit_diff(sha: str, repo: Optional[str] = None) -> Dict:
    """Get the diff and changed files for a specific commit."""
    rc, stat_output = _run_git(["diff", f"{sha}^..{sha}", "--stat", "--no-color"], cwd=repo)
    stat = stat_output.strip() if rc == 0 else ""

    rc, diff_output = _run_git(["diff", f"{sha}^..{sha}", "--no-color", "-U3"], cwd=repo)
    full_diff = diff_output if rc == 0 else ""

    rc, files_output = _run_git(["diff", f"{sha}^..{sha}", "--name-only"], cwd=repo)
    files = [f.strip() for f in files_output.strip().split("\n") if f.strip()] if rc == 0 else []

    return {"stat": stat, "diff": full_diff, "files": files, "file_count": len(files)}


def _should_include_file(filepath: str) -> bool:
    """Check if a file should be included in synthetic QA."""
    for exclude in EXCLUDE_PATHS:
        if exclude in filepath:
            return False
    return any(re.search(pat, filepath) for pat in INCLUDE_PATTERNS)


def _classify_change(subject: str, files: List[str], diff: str) -> Optional[str]:
    """Classify the type of change. Returns None for skip-worthy changes."""
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
        return None  # Skip docs-only

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
    project = ""
    for f in files:
        parts = f.split("/")
        if len(parts) >= 2:
            project = parts[0] if parts[0] not in (".", "src", "lib") else parts[1] if len(parts) > 2 else ""
            break

    # Strip conventional commit prefixes
    clean = re.sub(r"^(feat|fix|refactor|chore|docs|test|ci|build|perf)\([^)]*\):\s*", "", subject)
    clean = re.sub(r"^(feat|fix|refactor|chore|docs|test|ci|build|perf):\s*", "", clean)

    templates = {
        "bugfix": [f"{clean}", f"Fix the issue: {clean}"],
        "feature": [f"{clean}", f"Implement: {clean}"],
        "refactor": [f"{clean}", f"Refactor: {clean}"],
        "ops": [f"{clean}", f"Set up: {clean}"],
    }

    suffix = f" in {project}" if project else ""
    default = [f"{clean}{suffix}"]
    options = templates.get(category, default)
    return options[0]


def _generate_plan_from_diff(files: List[str], diff: str) -> str:
    """Generate a step-by-step tool-use plan from the diff."""
    steps = []
    step_num = 0

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
    days: int = SYNTHETIC_DEFAULT_DAYS,
    repo: Optional[str] = None,
    dry_run: bool = False,
) -> Dict[str, Any]:
    """Generate synthetic Q&A examples from recent commits.

    Args:
        days: Look back this many days for commits
        repo: Git repo path (defaults to home directory)
        dry_run: Preview without writing

    Returns:
        Stats dict with commit count, examples generated, and sample
    """
    commits = get_recent_commits(days=days, repo=repo)
    if not commits:
        return {"status": "no_commits", "days": days}

    examples = []
    skipped = 0

    for commit in commits:
        diff_data = get_commit_diff(commit["sha"], repo=repo)

        relevant_files = [f for f in diff_data["files"] if _should_include_file(f)]
        if not relevant_files:
            skipped += 1
            continue

        diff_lines = diff_data["diff"].count("\n")
        if diff_lines < SYNTHETIC_MIN_DIFF_LINES or diff_lines > SYNTHETIC_MAX_DIFF_LINES:
            skipped += 1
            continue

        category = _classify_change(commit["subject"], relevant_files, diff_data["diff"])
        if not category:
            skipped += 1
            continue

        question = _generate_question(commit["subject"], category, relevant_files)
        plan = _generate_plan_from_diff(relevant_files, diff_data["diff"])
        if not plan:
            skipped += 1
            continue

        examples.append({
            "messages": [
                {"role": "system", "content": SFT_SYSTEM_PROMPT},
                {"role": "user", "content": question},
                {"role": "assistant", "content": plan},
            ]
        })

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
