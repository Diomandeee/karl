"""
types.py - Structured return types for KARL public APIs.

TypedDicts and dataclasses replacing raw Dict[str, Any] returns (A3-A6).
"""

from typing import Any, Dict, List, Optional
from typing import TypedDict


class RewardResult(TypedDict):
    """Return type for compute_reward() (A5)."""
    reward_score: float
    outcome_score: float
    process_score: float
    efficiency_score: float
    components: Dict[str, Any]


class OutcomeSignals(TypedDict, total=False):
    """Input type for flush_session outcome_signals (A4)."""
    correction_detected: Optional[bool]
    build_success: Optional[bool]
    redo_detected: Optional[bool]
    session_continued: Optional[bool]
    reward_score: Optional[float]


class ShadowAnalysis(TypedDict):
    """Return type for analyze_shadow_routing() (A8)."""
    status: str
    records: int
    cache_hits: int
    cache_misses: int
    hit_rate: float
    comparable: int
    agrees: int
    disagrees: int
    agreement_rate: float
    regex_coverage: float
    vector_coverage: float
    avg_elapsed_ms: float
    max_elapsed_ms: float
    regex_skills: Dict[str, int]
    vector_skills: Dict[str, int]
    time_range: Optional[Dict[str, str]]


class SkillHealth(TypedDict):
    """Per-skill health entry."""
    trajectories: int
    unique_sessions: int
    mean_reward: Optional[float]
    mean_process_score: Optional[float]
    mean_tools_per_session: float
    trend: str


class PromotionCheck(TypedDict):
    """Single check in promotion readiness."""
    required: Any
    actual: Any
    # TypedDict doesn't support 'pass' as key name, use string
    pass_: bool


class PromotionResult(TypedDict):
    """Return type for check_promotion_readiness()."""
    ready: bool
    checks: Dict[str, Dict[str, Any]]
    recommendation: str


class EntityState(TypedDict, total=False):
    """SEA entity state schema (A6)."""
    skill: str
    total_activations: int
    useful_activations: int
    suppressed_count: int
    hot_topics: List[str]
    cold_topics: List[str]
    context_window: int
    confidence_calibration: float
    last_activated: Optional[str]


class BackfillResult(TypedDict):
    """Return type for backfill_rewards()."""
    total: int
    scored: int
    skipped: int
    errors: int
    domain_baselines: Dict[str, float]


class WeightUpdateResult(TypedDict, total=False):
    """Return type for update_weights()."""
    updated: int
    dry_run: bool
    updates: Dict[str, Dict[str, Any]]
    error: str
    message: str
    skills: int


class StoreStats(TypedDict):
    """Return type for get_store_stats()."""
    total: int
    size_bytes: int
    channels: Dict[str, int]
    skills: Dict[str, int]
    with_reward: int
