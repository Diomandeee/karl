"""
flows.py - Prefect flow definitions for scheduled KARL operations.

Two flows:
  - karl_analysis: Daily shadow routing analysis + skill health + promotion check
  - karl_training: Weekly SFT export + remote LoRA training pipeline

Usage:
    # Register with Prefect
    prefect deployment build karl/flows.py:karl_analysis -n karl-daily -q default
    prefect deployment build karl/flows.py:karl_training -n karl-weekly -q default

Requires: pip install "karl[prefect]"
"""

import json
import logging

logger = logging.getLogger(__name__)

try:
    from prefect import flow, task
except ImportError:
    raise ImportError(
        "Prefect is required for KARL flows. Install with: pip install 'karl[prefect]'"
    )


@task(name="backfill-rewards")
def task_backfill():
    from karl.reward_engine import backfill_rewards
    return backfill_rewards()


@task(name="shadow-analysis")
def task_shadow_analysis():
    from karl.trajectory_bridge import analyze_shadow_routing
    return analyze_shadow_routing()


@task(name="skill-health")
def task_skill_health():
    from karl.trajectory_bridge import analyze_skill_health
    return analyze_skill_health()


@task(name="promotion-check")
def task_promotion_check():
    from karl.trajectory_bridge import check_promotion_readiness
    return check_promotion_readiness()


@task(name="weight-update")
def task_weight_update():
    from karl.weight_updater import update_weights
    return update_weights()


@task(name="notify-discord")
def task_notify(message: str):
    from karl.notifications import post_discord
    post_discord(message)


@task(name="sft-export")
def task_export(min_reward: float = 0.0):
    from karl.sft_exporter import export_sft
    return export_sft(min_reward=min_reward)


@task(name="training-cycle")
def task_train(dry_run: bool = False):
    from karl.trainer import full_training_cycle
    return full_training_cycle(dry_run=dry_run)


@flow(name="karl-analysis")
def karl_analysis():
    """Daily KARL analysis: backfill rewards, analyze routing, update weights."""
    backfill = task_backfill()
    shadow = task_shadow_analysis()
    health = task_skill_health()
    promotion = task_promotion_check()
    weights = task_weight_update()

    summary = (
        f"**KARL Daily Analysis**\n"
        f"Backfill: {backfill.get('scored', 0)} scored\n"
        f"Shadow: {shadow.get('records', 0)} records, "
        f"{shadow.get('agreement_rate', 0):.0%} agreement\n"
        f"Promotion: {'READY' if promotion.get('ready') else 'HOLD'}\n"
        f"Weights: {weights.get('updated', 0)} updated"
    )
    task_notify(summary)

    return {
        "backfill": backfill,
        "shadow": shadow,
        "health": health,
        "promotion": promotion,
        "weights": weights,
    }


@flow(name="karl-training")
def karl_training(min_reward: float = 0.0, dry_run: bool = False):
    """Weekly KARL training: export SFT data, trigger remote LoRA training."""
    backfill = task_backfill()
    export = task_export(min_reward=min_reward)

    if export.get("examples", 0) == 0:
        task_notify("**KARL Training**: No training data available, skipping.")
        return {"status": "no_data", "export": export}

    if dry_run:
        task_notify(
            f"**KARL Training (dry run)**: {export.get('examples', 0)} examples, "
            f"{export.get('train', 0)} train / {export.get('valid', 0)} valid"
        )
        return {"status": "dry_run", "export": export}

    train_result = task_train()

    summary = (
        f"**KARL Training Complete**\n"
        f"Examples: {export.get('examples', 0)} "
        f"({export.get('train', 0)} train / {export.get('valid', 0)} valid)\n"
        f"Status: {train_result.get('status', 'unknown')}"
    )
    task_notify(summary)

    return {"status": "complete", "export": export, "training": train_result}
