#!/usr/bin/env python3
"""
karl — Unified CLI for KARL (Knowledge Agents via Reinforcement Learning).

Usage:
    karl status          Full status report (JSON)
    karl shadow          Shadow routing analysis
    karl health          Per-skill health report
    karl promote         Check promotion readiness
    karl auto-promote    Run auto-promotion check
    karl trend           Show reward trend (14d)
    karl log-trend       Log today's reward summary
    karl dedup           Deduplicate trajectories (dry-run)
    karl dedup --apply   Deduplicate trajectories (apply)
    karl integrity       Check trajectory data integrity
    karl annotate        Backfill shadow actual_skill annotations
    karl quality         Backfill quality grades (dry-run)
    karl quality --apply Backfill quality grades (apply)
    karl archive         Archive old records (dry-run, 30d)
    karl export          Export SFT training data
    karl train --dry-run Show training plan
    karl train --status  Check Mac4 training status
    karl centroids       Refresh skill centroids
    karl metrics         Print Prometheus metrics to stdout
    karl config          Show current config
    karl confusion       Show confusion matrix (actual vs predicted)
    karl diversity       Centroid separation analysis
    karl seed            Shadow seeder dry-run
    karl seed --apply    Shadow seeder (generate records)
    karl promote-sim     Simulate promotion (per-skill routing plan)
    karl hybrid          Show hybrid routing table (vector vs regex per skill)
    karl sft-ready       Check SFT training data readiness
    karl synth-qa        Generate synthetic QA for underrepresented skills
    karl hard-negatives  Identify confused skill pairs from shadow data
    karl refine          Contrastive centroid refinement (dry-run)
    karl refine --apply  Contrastive centroid refinement (apply)
    karl confusion-resolve  Analyze top confused pairs with fix suggestions
    karl skill-breakdown    Per-skill accuracy breakdown with accuracy drag
    karl lift-sim           Simulate vector lift if confused pairs were fixed
    karl train-dryrun       Comprehensive SFT training data validation and go/no-go
    karl centroid-snapshot   Save current centroids as versioned snapshot
    karl centroid-versions   List centroid version snapshots
    karl centroid-rollback V Restore centroids from snapshot V
    karl lift-analysis       Statistical analysis of vector lift metric
    karl health-digest       Generate KARL health summary
    karl post-digest         Post health digest to Discord
    karl promote             Check readiness and auto-promote if ready
    karl ab-monitor          Post-promotion A/B monitoring (vector vs regex)
    karl force-promote       Override promotion with justification
    karl reshadow            Re-evaluate shadow records with current centroids (dry-run)
    karl reshadow --apply    Re-evaluate and save updated shadow records
    karl accuracy-source     Accuracy breakdown by record source (organic vs seeded)
    karl log-skill-evolution  Snapshot per-skill accuracy to evolution log
    karl skill-evolution     Show skill accuracy trends over time
    karl skill-evolution --skill X  Show evolution for specific skill
    karl auto-refresh        Auto-refresh centroids if interval elapsed
    karl auto-refresh --force  Force refresh even if interval not elapsed
    karl iterative-refine    Iterative centroid refinement with convergence (dry-run)
    karl iterative-refine --apply  Iterative refinement (apply changes)
    karl sft-preflight       Check Mac4 readiness for SFT training
    karl sft-launch          Transfer data and launch SFT training on Mac4
    karl sft-launch --dry-run  Show what would happen without launching
    karl sft-status          Check SFT training status on Mac4
    karl sft-eval            Evaluate trained adapter on holdout set
    karl sft-fetch           Download trained adapter from Mac4
"""

import json
import sys
from pathlib import Path

KARL_DIR = Path(__file__).parent
sys.path.insert(0, str(KARL_DIR))


def cmd_status():
    from generate_status import generate
    print(json.dumps(generate(), indent=2, default=str))


def cmd_shadow():
    from trajectory_bridge import analyze_shadow_routing
    print(json.dumps(analyze_shadow_routing(), indent=2, default=str))


def cmd_health():
    from trajectory_bridge import analyze_skill_health
    print(json.dumps(analyze_skill_health(), indent=2, default=str))


def cmd_promote_readiness():
    from trajectory_bridge import check_promotion_readiness
    print(json.dumps(check_promotion_readiness(), indent=2, default=str))


def cmd_auto_promote():
    from trajectory_bridge import auto_promote
    print(json.dumps(auto_promote(), indent=2, default=str))


def cmd_trend():
    from trajectory_bridge import show_trend
    days = 14
    for i, a in enumerate(sys.argv):
        if a == "--days" and i + 1 < len(sys.argv):
            days = int(sys.argv[i + 1])
    print(json.dumps(show_trend(days=days), indent=2, default=str))


def cmd_log_trend():
    from trajectory_bridge import log_reward_trend
    print(json.dumps(log_reward_trend(), indent=2, default=str))


def cmd_dedup():
    from trajectory_bridge import dedup_trajectories
    dry = "--apply" not in sys.argv
    print(json.dumps(dedup_trajectories(dry_run=dry), indent=2, default=str))


def cmd_integrity():
    from trajectory_bridge import check_integrity
    print(json.dumps(check_integrity(), indent=2, default=str))


def cmd_annotate():
    from trajectory_bridge import backfill_shadow_agreement
    print(json.dumps(backfill_shadow_agreement(), indent=2, default=str))


def cmd_quality():
    from trajectory_bridge import backfill_quality_grades
    dry = "--apply" not in sys.argv
    print(json.dumps(backfill_quality_grades(dry_run=dry), indent=2, default=str))


def cmd_archive():
    from trajectory_bridge import archive_old_records
    dry = "--apply" not in sys.argv
    days = 30
    for i, a in enumerate(sys.argv):
        if a == "--days" and i + 1 < len(sys.argv):
            days = int(sys.argv[i + 1])
    print(json.dumps(archive_old_records(days=days, dry_run=dry), indent=2, default=str))


def cmd_export():
    from sft_exporter import export_sft
    quality = None
    for i, a in enumerate(sys.argv):
        if a == "--quality" and i + 1 < len(sys.argv):
            quality = sys.argv[i + 1]
    result = export_sft(quality_filter=quality)
    print(json.dumps(result, indent=2, default=str))


def cmd_train():
    if "--status" in sys.argv:
        from sft_launcher import check_training_status
        print(json.dumps(check_training_status(), indent=2, default=str))
    elif "--dry-run" in sys.argv:
        from sft_launcher import find_latest_sft_export, validate_sft_data, prepare_training_data, launch_training
        sft = find_latest_sft_export()
        if not sft:
            print('{"error": "No SFT export found"}')
            return
        v = validate_sft_data(sft)
        p = prepare_training_data(sft)
        l = launch_training(dry_run=True)
        print(json.dumps({"validation": v, "split": p, "training": l}, indent=2, default=str))
    else:
        print("Usage: karl train --dry-run | karl train --status")


def cmd_centroids():
    from embedding_cache import refresh_skill_centroids
    print(json.dumps(refresh_skill_centroids(), indent=2, default=str))


def cmd_metrics():
    from metrics_exporter import generate_metrics
    print(generate_metrics())


def cmd_confusion():
    from trajectory_bridge import confusion_matrix
    result = confusion_matrix()
    if result.get("ascii"):
        print(result["ascii"])
        print()
    # Print per-skill stats sorted by F1
    for name, stats in sorted(result.get("per_skill", {}).items(), key=lambda x: -x[1].get("f1", 0)):
        print(f"  {name:25s} P={stats['precision']:.3f} R={stats['recall']:.3f} F1={stats['f1']:.3f}")
    print(f"\n{result.get('annotated', 0)} annotated, {result.get('skills', 0)} skills")


def cmd_diversity():
    from embedding_cache import centroid_diversity
    print(json.dumps(centroid_diversity(), indent=2, default=str))


def cmd_seed():
    from shadow_seeder import seed_shadow_records
    dry = "--apply" not in sys.argv
    if dry:
        print(json.dumps(seed_shadow_records(dry_run=True), indent=2, default=str))
    else:
        print(json.dumps(seed_shadow_records(dry_run=False), indent=2, default=str))


def cmd_promote_sim():
    """Simulate what promotion would look like — which skills would route via vector."""
    from trajectory_bridge import check_promotion_readiness, per_skill_readiness
    from embedding_cache import load_skill_embeddings, GENERIC_SKILLS

    promo = check_promotion_readiness()
    ready = per_skill_readiness(min_samples=3)
    embs = load_skill_embeddings()

    print("=== Promotion Simulation ===\n")
    print(f"Gate status: {'PASS' if promo.get('ready') else 'HOLD'}")
    for check, data in promo.get("checks", {}).items():
        status = "PASS" if data.get("pass") else "FAIL"
        print(f"  {check:25s} {status:4s}  required={data.get('required')}  actual={data.get('actual')}")

    print(f"\n=== Per-Skill Routing Plan ===\n")
    skills = ready.get("skills", {})
    vector_routed = []
    regex_fallback = []
    for name, s in sorted(skills.items(), key=lambda x: -x[1].get("vector_accuracy", 0)):
        if s.get("ready"):
            vector_routed.append(name)
        else:
            regex_fallback.append(name)

    print(f"Vector-routed ({len(vector_routed)} skills):")
    for name in vector_routed:
        s = skills[name]
        generic = " [generic]" if name in GENERIC_SKILLS else ""
        print(f"  {name:25s} acc={s.get('vector_accuracy', 0):.1%}  samples={s.get('samples', 0)}{generic}")

    print(f"\nRegex fallback ({len(regex_fallback)} skills):")
    for name in regex_fallback:
        s = skills[name]
        reason = "low_accuracy" if s.get("enough_data") else "insufficient_data"
        print(f"  {name:25s} acc={s.get('vector_accuracy', 0):.1%}  samples={s.get('samples', 0)}  reason={reason}")

    # Coverage impact
    centroid_only = set(embs.keys()) - set(skills.keys())
    print(f"\nCentroid-only (no trajectory data, {len(centroid_only)} skills):")
    for name in sorted(centroid_only)[:10]:
        print(f"  {name}")
    if len(centroid_only) > 10:
        print(f"  ... and {len(centroid_only) - 10} more")


def cmd_accuracy_trend():
    from trajectory_bridge import show_accuracy_trend
    days = 7
    for i, a in enumerate(sys.argv):
        if a == "--days" and i + 1 < len(sys.argv):
            days = int(sys.argv[i + 1])
    print(json.dumps(show_accuracy_trend(days=days), indent=2, default=str))


def cmd_log_accuracy():
    from trajectory_bridge import log_accuracy_trend
    print(json.dumps(log_accuracy_trend(), indent=2, default=str))


def cmd_hybrid():
    """Show hybrid routing table: which skills route via vector vs regex."""
    from trajectory_bridge import get_hybrid_routing_table
    table = get_hybrid_routing_table()
    print(f"=== Hybrid Routing Table ===\n")
    print(f"Vector-routed: {table['vector_count']} skills")
    for name in table["vector_skills"]:
        d = table["table"][name]
        print(f"  {name:25s} acc={d['accuracy']:.1%}  samples={d['samples']}")
    print(f"\nRegex fallback: {table['regex_count']} skills")
    for name in table["regex_skills"]:
        d = table["table"][name]
        print(f"  {name:25s} acc={d['accuracy']:.1%}  samples={d['samples']}")
    print(json.dumps({"vector_count": table["vector_count"], "regex_count": table["regex_count"]}, indent=2))


def cmd_sft_ready():
    from sft_exporter import check_sft_readiness
    result = check_sft_readiness()
    print(json.dumps(result, indent=2, default=str))


def cmd_synth_qa():
    from synthetic_qa_generator import generate_synthetic_qa
    dry = "--apply" not in sys.argv
    min_count = 3
    for i, a in enumerate(sys.argv):
        if a == "--min-count" and i + 1 < len(sys.argv):
            min_count = int(sys.argv[i + 1])
    print(json.dumps(generate_synthetic_qa(min_count=min_count, dry_run=dry), indent=2, default=str))


def cmd_confusion_resolve():
    from trajectory_bridge import resolve_confusion_pairs
    result = resolve_confusion_pairs()
    print(json.dumps(result, indent=2, default=str))


def cmd_skill_breakdown():
    from trajectory_bridge import skill_accuracy_breakdown
    result = skill_accuracy_breakdown()
    print(json.dumps(result, indent=2, default=str))


def cmd_accuracy_source():
    from trajectory_bridge import accuracy_by_source
    result = accuracy_by_source()
    print(json.dumps(result, indent=2, default=str))


def cmd_centroid_snapshot():
    from embedding_cache import save_centroid_snapshot
    label = None
    for i, a in enumerate(sys.argv):
        if a == "--label" and i + 1 < len(sys.argv):
            label = sys.argv[i + 1]
    result = save_centroid_snapshot(label=label)
    print(json.dumps(result, indent=2, default=str))


def cmd_centroid_versions():
    from embedding_cache import list_centroid_versions
    versions = list_centroid_versions()
    print(json.dumps(versions, indent=2, default=str))


def cmd_centroid_rollback():
    from embedding_cache import rollback_centroids
    if len(sys.argv) < 3:
        print('{"error": "Usage: karl centroid-rollback <version_name>"}')
        return
    version = sys.argv[2]
    result = rollback_centroids(version)
    print(json.dumps(result, indent=2, default=str))


def cmd_lift_analysis():
    from trajectory_bridge import analyze_lift_threshold
    result = analyze_lift_threshold()
    print(json.dumps(result, indent=2, default=str))


def cmd_promote():
    from trajectory_bridge import check_promotion_readiness
    result = check_promotion_readiness()
    print(json.dumps(result, indent=2, default=str))


def cmd_health_digest():
    from trajectory_bridge import generate_health_digest
    result = generate_health_digest()
    print(result.get("digest", ""))
    print()
    print(json.dumps({k: v for k, v in result.items() if k != "digest"}, indent=2, default=str))


def cmd_post_digest():
    from trajectory_bridge import post_health_digest
    result = post_health_digest()
    print(json.dumps(result, indent=2, default=str))


def cmd_ab_monitor():
    from trajectory_bridge import post_promotion_monitor
    result = post_promotion_monitor()
    print(json.dumps(result, indent=2, default=str))


def cmd_force_promote():
    reason = None
    for i, a in enumerate(sys.argv):
        if a == "--reason" and i + 1 < len(sys.argv):
            reason = sys.argv[i + 1]
    if not reason:
        print('{"error": "Usage: karl force-promote --reason \\"justification text\\""}')
        return

    config_path = KARL_DIR / "config.json"
    config = {}
    if config_path.exists():
        try:
            config = json.loads(config_path.read_text())
        except (json.JSONDecodeError, OSError):
            pass

    from datetime import datetime, timezone
    from trajectory_bridge import check_promotion_readiness, _get_current_reward_mean

    # Snapshot centroids before promotion
    try:
        from embedding_cache import save_centroid_snapshot
        save_centroid_snapshot(label="pre_force_promote")
    except Exception:
        pass

    readiness = check_promotion_readiness()

    config["routing_mode"] = "active"
    config["promoted_at"] = datetime.now(timezone.utc).isoformat()
    config["promotion_type"] = "force"
    config["promotion_reason"] = reason
    config["pre_promotion_reward_mean"] = _get_current_reward_mean()
    config["promotion_checks"] = {k: v for k, v in readiness.get("checks", {}).items()}
    config_path.write_text(json.dumps(config, indent=2, default=str))

    # Log event
    log_path = KARL_DIR / "promotion_log.jsonl"
    event = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "action": "force_promoted",
        "reason": reason,
        "checks": readiness.get("checks", {}),
    }
    with open(log_path, "a") as f:
        f.write(json.dumps(event, default=str) + "\n")

    print(json.dumps({
        "action": "force_promoted",
        "reason": reason,
        "readiness": readiness,
    }, indent=2, default=str))


def cmd_reshadow():
    from shadow_seeder import reshadow_records
    apply = "--apply" in sys.argv
    result = reshadow_records(dry_run=not apply)
    print(json.dumps(result, indent=2, default=str))


def cmd_log_skill_evolution():
    from trajectory_bridge import log_skill_evolution
    result = log_skill_evolution()
    print(json.dumps(result, indent=2, default=str))


def cmd_skill_evolution():
    from trajectory_bridge import show_skill_evolution
    skill = None
    for i, a in enumerate(sys.argv):
        if a == "--skill" and i + 1 < len(sys.argv):
            skill = sys.argv[i + 1]
    result = show_skill_evolution(skill_name=skill)
    print(json.dumps(result, indent=2, default=str))


def cmd_auto_refresh():
    from embedding_cache import auto_refresh_centroids
    force = "--force" in sys.argv
    result = auto_refresh_centroids(force=force)
    print(json.dumps(result, indent=2, default=str))


def cmd_iterative_refine():
    from embedding_cache import iterative_refine
    apply = "--apply" in sys.argv
    # Parse optional flags
    alpha = 0.10
    max_rounds = 5
    for i, a in enumerate(sys.argv):
        if a == "--alpha" and i + 1 < len(sys.argv):
            alpha = float(sys.argv[i + 1])
        if a == "--rounds" and i + 1 < len(sys.argv):
            max_rounds = int(sys.argv[i + 1])
    result = iterative_refine(max_rounds=max_rounds, alpha=alpha, dry_run=not apply)
    print(json.dumps(result, indent=2, default=str))


def cmd_sft_preflight():
    from sft_dispatch import preflight
    result = preflight()
    print(json.dumps(result, indent=2, default=str))


def cmd_sft_launch():
    from sft_dispatch import launch_training
    dry_run = "--dry-run" in sys.argv
    result = launch_training(dry_run=dry_run)
    print(json.dumps(result, indent=2, default=str))


def cmd_sft_status():
    from sft_dispatch import check_status
    result = check_status()
    print(json.dumps(result, indent=2, default=str))


def cmd_sft_eval():
    from sft_dispatch import evaluate_adapter
    result = evaluate_adapter()
    print(json.dumps(result, indent=2, default=str))


def cmd_sft_fetch():
    from sft_dispatch import fetch_adapter
    result = fetch_adapter()
    print(json.dumps(result, indent=2, default=str))


def cmd_train_dryrun():
    """Comprehensive SFT training data validation with go/no-go."""
    train_path = KARL_DIR / "train.jsonl"
    valid_path = KARL_DIR / "valid.jsonl"
    all_path = KARL_DIR / "karl-sft.jsonl"

    result = {"files": {}, "dataset": {}, "go_nogo": {}}

    # Check file existence
    for name, path in [("train", train_path), ("valid", valid_path), ("all", all_path)]:
        if path.exists():
            lines = []
            with open(path) as f:
                for line in f:
                    try:
                        lines.append(json.loads(line))
                    except json.JSONDecodeError:
                        pass
            result["files"][name] = {
                "path": str(path),
                "records": len(lines),
                "size_kb": round(path.stat().st_size / 1024, 1),
            }
        else:
            result["files"][name] = {"path": str(path), "exists": False}

    # Validate ChatML format on train.jsonl
    train_records = []
    if train_path.exists():
        issues = []
        with open(train_path) as f:
            for i, line in enumerate(f):
                try:
                    r = json.loads(line)
                    train_records.append(r)
                    msgs = r.get("messages", [])
                    if not msgs:
                        issues.append(f"Line {i}: no messages")
                    else:
                        roles = [m.get("role") for m in msgs]
                        if "user" not in roles:
                            issues.append(f"Line {i}: missing user role")
                        if "assistant" not in roles:
                            issues.append(f"Line {i}: missing assistant role")
                except json.JSONDecodeError:
                    issues.append(f"Line {i}: invalid JSON")

        result["dataset"]["format_issues"] = issues[:5]
        result["dataset"]["format_valid"] = len(issues) == 0

    # Dataset statistics
    if train_records:
        from collections import Counter
        token_counts = []
        tool_counts = []
        unique_prompts = set()
        for r in train_records:
            msgs = r.get("messages", [])
            total_chars = sum(len(m.get("content", "")) for m in msgs)
            token_counts.append(total_chars // 4)  # rough token estimate
            # Count tool steps in assistant response
            for m in msgs:
                if m.get("role") == "assistant":
                    content = m.get("content", "")
                    steps = [l for l in content.split("\n") if l.strip().startswith(("1.", "2.", "3."))]
                    tool_counts.append(len(steps) if steps else content.count("[ok]") + content.count("[fail]"))
                if m.get("role") == "user":
                    unique_prompts.add(m.get("content", "")[:100])

        result["dataset"]["train_records"] = len(train_records)
        result["dataset"]["unique_prompts"] = len(unique_prompts)
        result["dataset"]["estimated_tokens"] = {
            "total": sum(token_counts),
            "avg_per_record": round(sum(token_counts) / len(token_counts)) if token_counts else 0,
            "min": min(token_counts) if token_counts else 0,
            "max": max(token_counts) if token_counts else 0,
        }
        result["dataset"]["avg_tool_steps"] = round(sum(tool_counts) / len(tool_counts), 1) if tool_counts else 0

    # Estimate training time (Mac4 MLX benchmarks: ~0.5s per record per epoch)
    config_path = KARL_DIR / "config.json"
    epochs = 3
    if config_path.exists():
        try:
            cfg = json.loads(config_path.read_text())
            sft = cfg.get("sft", {})
            result["dataset"]["model"] = sft.get("base_model", "unknown")
        except (json.JSONDecodeError, OSError):
            pass

    n_train = len(train_records) if train_records else 0
    est_time_s = n_train * epochs * 0.5
    result["dataset"]["estimated_training_time"] = {
        "seconds": round(est_time_s),
        "minutes": round(est_time_s / 60, 1),
        "epochs": epochs,
    }

    # Go/No-Go checks
    checks = {}
    checks["has_train_data"] = n_train >= 20
    checks["has_valid_data"] = valid_path.exists() and (valid_path.stat().st_size > 100)
    checks["format_valid"] = result.get("dataset", {}).get("format_valid", False)
    checks["diverse_prompts"] = result.get("dataset", {}).get("unique_prompts", 0) >= 10
    checks["reasonable_tokens"] = result.get("dataset", {}).get("estimated_tokens", {}).get("avg_per_record", 0) >= 50

    all_pass = all(checks.values())
    result["go_nogo"] = {
        "ready": all_pass,
        "checks": checks,
        "recommendation": "GO: Training data meets all quality thresholds." if all_pass
            else "NO-GO: " + ", ".join(k for k, v in checks.items() if not v) + " failed.",
    }

    print(json.dumps(result, indent=2, default=str))


def cmd_lift_sim():
    from trajectory_bridge import simulate_lift
    # Parse --fix-pairs actual:predicted,actual:predicted
    fix_pairs = None
    for i, a in enumerate(sys.argv):
        if a == "--fix-pairs" and i + 1 < len(sys.argv):
            pairs_str = sys.argv[i + 1]
            fix_pairs = []
            for p in pairs_str.split(","):
                parts = p.strip().split(":")
                if len(parts) == 2:
                    fix_pairs.append(tuple(parts))
    result = simulate_lift(fix_pairs=fix_pairs)
    print(json.dumps(result, indent=2, default=str))


def cmd_hard_negatives():
    from embedding_cache import identify_hard_negatives
    result = identify_hard_negatives()
    print(json.dumps(result, indent=2, default=str))


def cmd_refine():
    from embedding_cache import refine_centroids
    apply = "--apply" in sys.argv
    alpha = 0.15
    for i, a in enumerate(sys.argv):
        if a == "--alpha" and i + 1 < len(sys.argv):
            alpha = float(sys.argv[i + 1])
    result = refine_centroids(alpha=alpha, dry_run=not apply)
    print(json.dumps(result, indent=2, default=str))


def cmd_config():
    config_path = KARL_DIR / "config.json"
    if config_path.exists():
        print(config_path.read_text())
    else:
        print("{}")


def cmd_skill_matrix():
    from embedding_cache import skill_similarity_matrix
    result = skill_similarity_matrix()
    # Print compact: just merge candidates + summary, skip full matrix
    if result.get("status") == "ok":
        print(f"=== Skill Similarity Matrix ({len(result['skills'])} skills) ===\n")
        mc = result.get("merge_candidates", [])
        if mc:
            print(f"Merge candidates ({len(mc)} pairs with similarity > 0.85):")
            for c in mc:
                print(f"  {c['skill_a']:25s} <-> {c['skill_b']:25s}  "
                      f"sim={c['similarity']:.4f}  [{c['recommendation']}]")
        else:
            print("No merge candidates (all pairs below 0.85 similarity)")
        print(f"\nTotal skill pairs: {result['total_pairs']}")
    else:
        print(json.dumps(result, indent=2, default=str))


def cmd_merge_candidates():
    from embedding_cache import skill_similarity_matrix
    result = skill_similarity_matrix()
    mc = result.get("merge_candidates", [])
    print(json.dumps(mc, indent=2, default=str))


def cmd_accuracy_forecast():
    from trajectory_bridge import accuracy_forecast
    target = 500
    for i, a in enumerate(sys.argv):
        if a == "--target" and i + 1 < len(sys.argv):
            target = int(sys.argv[i + 1])
    result = accuracy_forecast(target_n=target)
    print(json.dumps(result, indent=2, default=str))


def cmd_exemplars():
    from embedding_cache import list_exemplars
    skill = None
    for i, a in enumerate(sys.argv):
        if a == "--skill" and i + 1 < len(sys.argv):
            skill = sys.argv[i + 1]
    if len(sys.argv) >= 3 and not sys.argv[2].startswith("--"):
        skill = sys.argv[2]
    result = list_exemplars(skill_name=skill)
    if result.get("status") == "ok":
        for sk, data in result["skills"].items():
            print(f"\n=== {sk} ({data['total']} exemplars) ===")
            for i, p in enumerate(data["auto"]):
                print(f"  [auto-{i}] {p[:120]}")
            for i, e in enumerate(data["manual"]):
                print(f"  [manual-{i}] {e['prompt'][:120]}")
    else:
        print(json.dumps(result, indent=2, default=str))


def cmd_add_exemplar():
    from embedding_cache import add_exemplar
    if len(sys.argv) < 4:
        print("Usage: karl add-exemplar SKILL \"prompt text\"")
        return
    skill = sys.argv[2]
    prompt = " ".join(sys.argv[3:])
    result = add_exemplar(skill, prompt)
    print(json.dumps(result, indent=2, default=str))


def cmd_remove_exemplar():
    from embedding_cache import remove_exemplar
    if len(sys.argv) < 4:
        print("Usage: karl remove-exemplar SKILL INDEX")
        return
    skill = sys.argv[2]
    idx = int(sys.argv[3])
    result = remove_exemplar(skill, idx)
    print(json.dumps(result, indent=2, default=str))


def cmd_rebuild_centroid():
    from embedding_cache import rebuild_centroid
    if len(sys.argv) < 3:
        print("Usage: karl rebuild-centroid SKILL")
        return
    skill = sys.argv[2]
    result = rebuild_centroid(skill)
    print(json.dumps(result, indent=2, default=str))


def cmd_routing_override():
    """Manage per-skill routing overrides.

    karl routing-override                    List current overrides
    karl routing-override SKILL vector       Force SKILL to vector routing
    karl routing-override SKILL regex        Force SKILL to regex routing
    karl routing-override SKILL --remove     Remove override for SKILL
    """
    config_path = KARL_DIR / "config.json"
    config = {}
    if config_path.exists():
        try:
            config = json.loads(config_path.read_text())
        except (json.JSONDecodeError, OSError):
            pass

    overrides = config.get("routing_overrides", {})

    # List mode
    if len(sys.argv) < 3:
        if not overrides:
            print("No routing overrides configured.")
        else:
            print("=== Routing Overrides ===\n")
            for skill, route in sorted(overrides.items()):
                print(f"  {skill:25s} -> {route}")
        return

    skill = sys.argv[2]

    # Remove mode
    if "--remove" in sys.argv:
        if skill in overrides:
            del overrides[skill]
            config["routing_overrides"] = overrides
            config_path.write_text(json.dumps(config, indent=2, default=str))
            print(json.dumps({"action": "removed", "skill": skill}, indent=2))
        else:
            print(json.dumps({"action": "noop", "skill": skill, "reason": "no override found"}))
        return

    # Set mode
    if len(sys.argv) < 4:
        current = overrides.get(skill, "auto")
        print(f"{skill}: {current}")
        print("Usage: karl routing-override SKILL vector|regex|--remove")
        return

    route = sys.argv[3]
    if route not in ("vector", "regex"):
        print(f"Invalid route: {route}. Must be 'vector' or 'regex'.")
        return

    overrides[skill] = route
    config["routing_overrides"] = overrides
    config_path.write_text(json.dumps(config, indent=2, default=str))
    print(json.dumps({"action": "set", "skill": skill, "route": route}, indent=2))


def cmd_confidence():
    from trajectory_bridge import confidence_analysis
    result = confidence_analysis()
    if result.get("status") == "ok":
        print(f"=== Routing Confidence Analysis ({result['total']} records) ===\n")
        print(f"Overall accuracy: {result['overall_accuracy']:.1%}\n")
        print("Similarity thresholds:")
        for s in result["similarity_thresholds"]:
            print(f"  sim>={s['threshold']:.2f}  accepted={s['accepted']:3d}  "
                  f"rejected={s['rejected']:3d}  "
                  f"acc_accepted={s['accuracy_accepted']:.1%}  "
                  f"acc_rejected={s['accuracy_rejected']:.1%}")
        print("\nMargin thresholds (top-1 - top-2):")
        for m in result["margin_thresholds"]:
            print(f"  margin>={m['min_margin']:.2f}  accepted={m['accepted']:3d}  "
                  f"rejected={m['rejected']:3d}  "
                  f"acc_accepted={m['accuracy_accepted']:.1%}  "
                  f"acc_rejected={m['accuracy_rejected']:.1%}")
        rec = result.get("recommended_sim_threshold")
        if rec:
            print(f"\nRecommended sim threshold: {rec} (<20% rejection)")
    else:
        print(json.dumps(result, indent=2, default=str))


def cmd_difficulty():
    from trajectory_bridge import difficulty_analysis
    result = difficulty_analysis()
    print(json.dumps(result, indent=2, default=str))


def cmd_dashboard():
    from trajectory_bridge import generate_dashboard_json
    result = generate_dashboard_json()
    print(json.dumps(result, indent=2, default=str))


def cmd_replay():
    from trajectory_bridge import replay_trajectories
    since = None
    for i, a in enumerate(sys.argv):
        if a == "--since" and i + 1 < len(sys.argv):
            since = sys.argv[i + 1]
    result = replay_trajectories(since=since)
    print(json.dumps(result, indent=2, default=str))


def cmd_reward_calibration():
    from trajectory_bridge import reward_calibration
    normalize = "--normalize" in sys.argv
    result = reward_calibration(normalize=normalize)
    print(json.dumps(result, indent=2, default=str))


def cmd_auto_resolve():
    from trajectory_bridge import auto_resolve_confusion
    apply = "--apply" in sys.argv
    top_n = 5
    for i, a in enumerate(sys.argv):
        if a == "--top" and i + 1 < len(sys.argv):
            top_n = int(sys.argv[i + 1])
    result = auto_resolve_confusion(top_n=top_n, dry_run=not apply)
    print(json.dumps(result, indent=2, default=str))


def cmd_cron():
    from karl_cron import run_cron
    post = "--post" in sys.argv
    dry_run = "--dry-run" in sys.argv
    result = run_cron(post_digest=post, dry_run=dry_run)
    print(json.dumps(result, indent=2, default=str))


def cmd_organic_status():
    """Show organic data accumulation progress toward 200-record evaluation threshold."""
    shadow_path = Path(__file__).parent / "routing_shadow.jsonl"
    organic = []
    synthetic = []
    if shadow_path.exists():
        for line in shadow_path.read_text().splitlines():
            if not line.strip():
                continue
            try:
                r = json.loads(line)
                src = r.get("source", "")
                if src in ("shadow_seeder", "reshadow"):
                    synthetic.append(r)
                else:
                    organic.append(r)
            except (json.JSONDecodeError, KeyError):
                pass

    target = 200
    pct = len(organic) / target * 100 if target else 0
    bar_len = 40
    filled = int(bar_len * min(pct, 100) / 100)
    bar = "#" * filled + "-" * (bar_len - filled)

    print(f"=== Organic Data Accumulation ===\n")
    print(f"  Organic:   {len(organic):4d} / {target} [{bar}] {pct:.1f}%")
    print(f"  Synthetic: {len(synthetic):4d} (frozen, no new seeding)")
    print(f"  Total:     {len(organic) + len(synthetic):4d}")

    if organic:
        correct = sum(1 for r in organic if r.get("vector_correct") is True)
        evaluable = sum(1 for r in organic if r.get("vector_correct") is not None)
        if evaluable:
            print(f"\n  Organic vector accuracy: {correct}/{evaluable} = {correct/evaluable*100:.1f}%")
        else:
            print(f"\n  Organic vector accuracy: not yet evaluable (need actual_skill annotations)")

        # Date range
        dates = [r.get("ts", "") for r in organic if r.get("ts")]
        if dates:
            print(f"  Date range: {min(dates)[:10]} to {max(dates)[:10]}")
    else:
        print("\n  No organic records yet. They accumulate from real routing events.")

    if pct < 100:
        remaining = target - len(organic)
        print(f"\n  {remaining} more organic records needed before re-evaluation.")
    else:
        print(f"\n  THRESHOLD REACHED. Run 'karl accuracy' to evaluate on organic data.")


COMMANDS = {
    "status": cmd_status,
    "shadow": cmd_shadow,
    "health": cmd_health,
    "promote": cmd_promote,
    "auto-promote": cmd_auto_promote,
    "trend": cmd_trend,
    "log-trend": cmd_log_trend,
    "dedup": cmd_dedup,
    "integrity": cmd_integrity,
    "annotate": cmd_annotate,
    "quality": cmd_quality,
    "archive": cmd_archive,
    "export": cmd_export,
    "train": cmd_train,
    "centroids": cmd_centroids,
    "metrics": cmd_metrics,
    "config": cmd_config,
    "confusion": cmd_confusion,
    "diversity": cmd_diversity,
    "seed": cmd_seed,
    "promote-sim": cmd_promote_sim,
    "hybrid": cmd_hybrid,
    "accuracy-trend": cmd_accuracy_trend,
    "log-accuracy": cmd_log_accuracy,
    "sft-ready": cmd_sft_ready,
    "synth-qa": cmd_synth_qa,
    "hard-negatives": cmd_hard_negatives,
    "refine": cmd_refine,
    "confusion-resolve": cmd_confusion_resolve,
    "skill-breakdown": cmd_skill_breakdown,
    "lift-sim": cmd_lift_sim,
    "train-dryrun": cmd_train_dryrun,
    "centroid-snapshot": cmd_centroid_snapshot,
    "centroid-versions": cmd_centroid_versions,
    "centroid-rollback": cmd_centroid_rollback,
    "lift-analysis": cmd_lift_analysis,
    "force-promote": cmd_force_promote,
    "reshadow": cmd_reshadow,
    "accuracy-source": cmd_accuracy_source,
    "iterative-refine": cmd_iterative_refine,
    "sft-preflight": cmd_sft_preflight,
    "sft-launch": cmd_sft_launch,
    "sft-status": cmd_sft_status,
    "ab-monitor": cmd_ab_monitor,
    "auto-refresh": cmd_auto_refresh,
    "sft-eval": cmd_sft_eval,
    "sft-fetch": cmd_sft_fetch,
    "log-skill-evolution": cmd_log_skill_evolution,
    "skill-evolution": cmd_skill_evolution,
    "health-digest": cmd_health_digest,
    "post-digest": cmd_post_digest,
    "cron": cmd_cron,
    "auto-resolve": cmd_auto_resolve,
    "skill-matrix": cmd_skill_matrix,
    "merge-candidates": cmd_merge_candidates,
    "reward-calibration": cmd_reward_calibration,
    "replay": cmd_replay,
    "dashboard": cmd_dashboard,
    "difficulty": cmd_difficulty,
    "accuracy-forecast": cmd_accuracy_forecast,
    "exemplars": cmd_exemplars,
    "add-exemplar": cmd_add_exemplar,
    "remove-exemplar": cmd_remove_exemplar,
    "rebuild-centroid": cmd_rebuild_centroid,
    "confidence-analysis": cmd_confidence,
    "routing-override": cmd_routing_override,
    "organic-status": cmd_organic_status,
}


def main():
    if len(sys.argv) < 2 or sys.argv[1] in ("-h", "--help", "help"):
        print(__doc__)
        print("Available commands:")
        for name in COMMANDS:
            print(f"  karl {name}")
        return

    cmd = sys.argv[1]
    if cmd in COMMANDS:
        COMMANDS[cmd]()
    else:
        print(f"Unknown command: {cmd}")
        print(f"Available: {', '.join(COMMANDS.keys())}")
        sys.exit(1)


if __name__ == "__main__":
    main()
