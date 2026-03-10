"""
cli.py - Command-line interface for KARL.

Provides subcommands for all KARL operations:
  karl status       - Show trajectory store stats and reward distribution
  karl backfill     - Backfill rewards for unscored trajectories
  karl export       - Export SFT training data
  karl train        - Run full training pipeline
  karl analyze      - Run shadow routing analysis and promotion check
  karl report       - Generate full intelligence report
  karl bootstrap    - Generate skill embeddings
  karl synthetic    - Generate synthetic QA from git diffs
  karl weights      - Show/update skill weights
"""

import argparse
import json
import sys

from karl.config import ensure_dirs


def cmd_status(args):
    """Show trajectory store stats."""
    from karl.trajectory_tap import get_store_stats
    from karl.reward_engine import get_reward_stats

    ensure_dirs()
    store_stats = get_store_stats()
    reward_stats = get_reward_stats()

    print(f"\nTrajectory Store")
    print(f"  Total records: {store_stats['total']}")
    print(f"  Size: {store_stats['size_bytes'] / 1024:.1f} KB")
    print(f"  Channels: {store_stats.get('channels', {})}")
    print(f"  Skills: {store_stats.get('skills', {})}")
    print(f"  With reward: {store_stats.get('with_reward', 0)}")

    if reward_stats.get("scored"):
        print(f"\nReward Distribution")
        print(f"  Scored: {reward_stats['scored']}")
        print(f"  Mean:   {reward_stats.get('mean', 'N/A')}")
        print(f"  Range:  [{reward_stats.get('min', 'N/A')}, {reward_stats.get('max', 'N/A')}]")
        print(f"  By domain: {reward_stats.get('by_domain', {})}")


def cmd_backfill(args):
    """Backfill rewards for unscored trajectories."""
    from karl.reward_engine import backfill_rewards
    stats = backfill_rewards(force=args.force)
    print(json.dumps(stats, indent=2))


def cmd_export(args):
    """Export SFT training data."""
    from karl.sft_exporter import export_sft
    stats = export_sft(min_reward=args.min_reward, dry_run=args.dry_run)
    print(json.dumps(stats, indent=2))


def cmd_train(args):
    """Run full training pipeline."""
    from karl.trainer import full_training_cycle
    result = full_training_cycle(dry_run=args.dry_run)
    print(json.dumps(result, indent=2, default=str))


def cmd_analyze(args):
    """Run shadow routing analysis and promotion check."""
    from karl.trajectory_bridge import analyze_shadow_routing, check_promotion_readiness
    shadow = analyze_shadow_routing()
    promotion = check_promotion_readiness()
    print(json.dumps({"shadow": shadow, "promotion": promotion}, indent=2, default=str))


def cmd_report(args):
    """Generate full intelligence report."""
    from karl.trajectory_bridge import full_report
    print(full_report(as_json=args.json))


def cmd_bootstrap(args):
    """Generate skill embeddings."""
    from karl.bootstrap import bootstrap_skill_embeddings
    stats = bootstrap_skill_embeddings(dry_run=args.dry_run)
    print(json.dumps(stats, indent=2))


def cmd_synthetic(args):
    """Generate synthetic QA from git diffs."""
    from karl.synthetic_qa import generate_synthetic_qa
    stats = generate_synthetic_qa(days=args.days, dry_run=args.dry_run)
    print(json.dumps(stats, indent=2, default=str))


def cmd_weights(args):
    """Show or update skill weights."""
    from karl.weight_updater import update_weights, collect_skill_rewards
    from karl.embedding_cache import load_skill_embeddings

    if args.update:
        result = update_weights(dry_run=args.dry_run)
        print(json.dumps(result, indent=2))
    else:
        embeddings = load_skill_embeddings()
        rewards = collect_skill_rewards()

        print(f"\nSkill Embedding Weights ({len(embeddings)} skills)")
        print(f"{'Skill':<20} {'Weight':>8} {'Trajectories':>13} {'Mean Reward':>12}")
        print("-" * 55)

        for name in sorted(embeddings.keys()):
            _, weight = embeddings[name]
            skill_rewards = rewards.get(name, [])
            n = len(skill_rewards)
            mean_r = sum(skill_rewards) / n if n > 0 else None
            mean_str = f"{mean_r:.4f}" if mean_r is not None else "N/A"
            print(f"{name:<20} {weight:>8.4f} {n:>13} {mean_str:>12}")


def cmd_extract(args):
    """Extract trajectories from verbose logs."""
    from karl.extractor import extract_trajectories
    trajectories = extract_trajectories(dry_run=args.dry_run)
    print(f"Extracted {len(trajectories)} trajectories")


def main():
    parser = argparse.ArgumentParser(
        prog="karl",
        description="KARL - Knowledge Agents via Reinforcement Learning",
    )
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # status
    subparsers.add_parser("status", help="Show trajectory store stats")

    # backfill
    p = subparsers.add_parser("backfill", help="Backfill rewards")
    p.add_argument("--force", action="store_true", help="Recompute all")

    # export
    p = subparsers.add_parser("export", help="Export SFT training data")
    p.add_argument("--min-reward", type=float, default=0.0)
    p.add_argument("--dry-run", action="store_true")

    # train
    p = subparsers.add_parser("train", help="Run training pipeline")
    p.add_argument("--dry-run", action="store_true")

    # analyze
    subparsers.add_parser("analyze", help="Shadow routing analysis")

    # report
    p = subparsers.add_parser("report", help="Full intelligence report")
    p.add_argument("--json", action="store_true")

    # bootstrap
    p = subparsers.add_parser("bootstrap", help="Generate skill embeddings")
    p.add_argument("--dry-run", action="store_true")

    # synthetic
    p = subparsers.add_parser("synthetic", help="Generate synthetic QA")
    p.add_argument("--days", type=int, default=7)
    p.add_argument("--dry-run", action="store_true")

    # weights
    p = subparsers.add_parser("weights", help="Show/update skill weights")
    p.add_argument("--update", action="store_true")
    p.add_argument("--dry-run", action="store_true")

    # extract
    p = subparsers.add_parser("extract", help="Extract from verbose logs")
    p.add_argument("--dry-run", action="store_true")

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(1)

    ensure_dirs()

    commands = {
        "status": cmd_status,
        "backfill": cmd_backfill,
        "export": cmd_export,
        "train": cmd_train,
        "analyze": cmd_analyze,
        "report": cmd_report,
        "bootstrap": cmd_bootstrap,
        "synthetic": cmd_synthetic,
        "weights": cmd_weights,
        "extract": cmd_extract,
    }

    commands[args.command](args)


if __name__ == "__main__":
    main()
