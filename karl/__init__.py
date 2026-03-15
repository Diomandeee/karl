"""
KARL — Knowledge Agents via Reinforcement Learning (Adapted)

Trajectory-based intelligence layer for Claude Code.
Captures tool-use trajectories with outcome signals,
enabling learned skill routing and LoRA-based policy improvement.

Modules:
    trajectory_tap: Live session trajectory recording
    trajectory_extractor: Historical backfill from verbose logs
    reward_engine: Outcome signal computation (future)
    embedding_cache: Skill embedding store (future)
"""

__version__ = "0.1.0"
