"""Reads project identity from filesystem. Runs once at session start."""

import json
import os
from dataclasses import dataclass
from pathlib import Path


@dataclass
class ProjectContext:
    name: str
    language: str
    summary: str
    key_files: list[str]
    build_cmd: str


def detect_project(project_dir: str) -> ProjectContext:
    """Auto-detect project identity from filesystem markers."""
    d = Path(project_dir).expanduser()
    name = d.name
    language = "unknown"
    summary = ""
    key_files = []
    build_cmd = ""

    # CLAUDE.md — richest source
    claude_md = d / "CLAUDE.md"
    if claude_md.exists():
        text = claude_md.read_text()[:800]
        summary = text.split("\n\n")[0][:200] if text else ""

    # package.json (Node/JS/TS)
    pkg = d / "package.json"
    if pkg.exists():
        try:
            data = json.loads(pkg.read_text())
            name = data.get("name", name)
            language = "typescript" if (d / "tsconfig.json").exists() else "javascript"
            summary = summary or data.get("description", "")[:200]
            scripts = data.get("scripts", {})
            build_cmd = f"npm run {list(scripts.keys())[0]}" if scripts else "npm run build"
            key_files.append("package.json")
        except (json.JSONDecodeError, IndexError):
            pass

    # Cargo.toml (Rust)
    cargo = d / "Cargo.toml"
    if cargo.exists():
        language = "rust"
        text = cargo.read_text()[:500]
        for line in text.split("\n"):
            if line.startswith("name"):
                name = line.split("=")[-1].strip().strip('"')
                break
        build_cmd = "cargo build"
        key_files.append("Cargo.toml")

    # pyproject.toml or setup.py (Python)
    pyproject = d / "pyproject.toml"
    setup = d / "setup.py"
    if pyproject.exists() or setup.exists():
        language = "python"
        build_cmd = "pip install -e ."
        key_files.append("pyproject.toml" if pyproject.exists() else "setup.py")

    # requirements.txt (Python fallback)
    if (d / "requirements.txt").exists() and language == "unknown":
        language = "python"
        key_files.append("requirements.txt")

    # Detect key source files
    for pattern in ["src/main.*", "app/page.*", "main.*", "index.*", "lib.*"]:
        for f in d.glob(pattern):
            if f.is_file():
                key_files.append(str(f.relative_to(d)))

    if not summary:
        readme = d / "README.md"
        if readme.exists():
            summary = readme.read_text()[:300].split("\n\n")[0]

    if not summary:
        summary = f"{name} ({language} project)"

    return ProjectContext(
        name=name,
        language=language,
        summary=summary[:200],
        key_files=key_files[:8],
        build_cmd=build_cmd or "make",
    )
