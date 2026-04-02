---
name: MLX Trainer MCP Research
description: Deep investigation of mlx-lm LoRA pipeline for designing production MLX Trainer MCP server on Mac5 (16GB M4). Covers parameter reference, memory budgets, gotchas, data quality, MCP tool specs.
type: reference
---

Research completed 2026-03-31. mlx-lm 0.29.1 on Mac5 (M4 16GB, Xcode Python 3.9.6).
Key findings: train/valid data leakage (85 examples), 439 exact duplicates in train set, 12% memory free during 4B training.
Finetune daemon at :9200 (FastAPI), mesh-compute (Rust binary) on :9400, both KeepAlive LaunchAgents.
Fused model has no _name_or_path set, causing server to show base model ID.
