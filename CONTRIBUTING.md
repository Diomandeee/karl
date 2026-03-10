# Contributing to KARL

## Development Setup

```bash
# Clone
git clone https://github.com/Diomandeee/karl.git
cd karl

# Install with dev dependencies
pip install -e ".[dev]"

# Verify
pytest -v
```

## Running Tests

```bash
# Full suite
pytest -v

# With coverage
pytest --cov=karl --cov-report=term-missing

# Single module
pytest tests/test_trajectory_tap.py -v
```

## Linting

```bash
# Check
ruff check karl/ tests/

# Format
ruff format karl/ tests/

# Type check
mypy karl/
```

## Branch Conventions

- `main` - stable, tested
- `feat/<name>` - new features
- `fix/<name>` - bug fixes

## Commit Messages

Use conventional commits:

```
feat(module): add new capability
fix(module): resolve specific issue
refactor(module): restructure without behavior change
```

## Module Overview

| Module | Purpose |
|--------|---------|
| `trajectory_tap.py` | Live tool-use recording (4 tap points) |
| `reward_engine.py` | Multi-signal reward computation |
| `embedding_cache.py` | Async embedding cache + skill routing |
| `trajectory_bridge.py` | Shadow routing analysis + promotion gate |
| `weight_updater.py` | EMA weight updates from rewards |
| `sft_exporter.py` | Advantage-weighted SFT data export |
| `trainer.py` | Remote LoRA training pipeline |
| `entity_bridge.py` | SEA skill entity updates |
| `bootstrap.py` | Skill embedding generation |
| `synthetic_qa.py` | Self-play data from git diffs |
| `extractor.py` | Historical trajectory backfill |
| `config.py` | Centralized configuration |
| `cli.py` | Command-line interface |
| `notifications.py` | Discord webhook alerts |
