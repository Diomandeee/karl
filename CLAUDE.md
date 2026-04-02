# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What is KARL

KARL (Knowledge Agents via Reinforcement Learning) records what AI coding agents do during real work sessions as trajectories, scores them with a multi-signal reward engine, and uses the best trajectories for LoRA fine-tuning via MLX. It also provides vector-based skill routing that shadows and can replace regex routing.

Paper: [arXiv 2603.05218](https://arxiv.org/abs/2603.05218)

## Build and Development

```bash
pip install -e "."              # Core install
pip install -e ".[dev]"         # Dev tools (pytest, ruff, mypy)
pip install -e ".[prefect]"     # With Prefect flow support
```

## Commands

```bash
# Tests
pytest                                      # Full suite (164 tests)
pytest tests/test_trajectory_tap.py -v      # Single module
pytest --cov=karl --cov-report=term-missing # With coverage

# Lint
ruff check karl/ tests/                     # Check
ruff format karl/ tests/                    # Format

# Type check
mypy karl/ --ignore-missing-imports

# CLI (installed as `karl` entry point)
karl status         # Trajectory store stats
karl backfill       # Score unscored trajectories
karl export         # Export SFT training data
karl train          # Run LoRA training pipeline
karl analyze        # Shadow routing analysis
karl report         # Full intelligence report
```

The extended CLI lives in `karl/karl_cli.py` (67 commands). The simpler `karl/cli.py` is the pip-installed entry point (`karl` command) with 10 core subcommands.

## Architecture

### Core Pipeline (4 Taps)

The system records agent sessions through 4 tap points in `trajectory_tap.py`:
1. **Tap A** - `init_session_buffer()`: Start recording, run shadow skill routing
2. **Tap B** - `append_tool_event()`: Log each tool call (name, params, success)
3. **Tap C** - `flush_session()`: End recording, compute reward, write to `trajectories.jsonl`
4. **Tap D** - `annotate_previous()`: Detect corrections on next prompt, update prior trajectory

### Reward Engine (5-signal)

`reward_engine.py` computes composite scores. Note: the README documents 3 signals with weights 0.40/0.35/0.25, but the actual code uses 5 signals:
- Outcome (0.30), Process (0.25), Efficiency (0.15), Verification (0.15), Consistency (0.15)

### Skill Routing

`embedding_cache.py` + `trajectory_bridge.py` handle vector-based skill routing that shadows regex routing. Promotion from regex to vector requires 200+ organic shadow records and measured accuracy gains.

### Training Pipeline

`sft_exporter.py` -> `sft_dispatch.py` -> `sft_launcher.py`: Export advantage-weighted SFT data (OAPL-Lite oversampling), dispatch to remote machine (default: mac5 via SSH), run LoRA training with MLX.

### Key Data Files (all in `karl/` at runtime)

- `trajectories.jsonl` - Append-only trajectory store (the core dataset)
- `train.jsonl` / `valid.jsonl` - SFT training splits
- `karl-sft.jsonl` - Formatted SFT output

## Configuration

All config via environment variables in `karl/config.py`. Key ones:
- `KARL_DATA_DIR` - Data directory (default: `./data`)
- `KARL_SKILLS_DIR` - Skills directory (default: `~/.claude/skills`)
- `KARL_EMBED_URL` - Embedding API endpoint
- `KARL_TRAIN_SSH_ALIAS` - Remote training machine (default: `mac5`)
- `KARL_MLX_MODEL` - Base model for LoRA (default: `mlx-community/gemma-3-1b-it-4bit`)

## CI

GitHub Actions runs on push/PR to main: ruff check, mypy, pytest across Python 3.10-3.12.

## Conventions

- Conventional commits: `feat(module):`, `fix(module):`, `refactor(module):`
- Branch naming: `feat/<name>`, `fix/<name>`
- Ruff config: line-length 100, E501 ignored (formatter handles it), isort with `karl` as first-party
- Python 3.10+ required
