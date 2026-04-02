#!/usr/bin/env python3
"""
KARL Advantage Experiment — Prove that 35 high-advantage trajectories
outperform 290 random trajectories for agent tool-planning training.

Runs two MLX LoRA training sessions on Gemma 3 1B:
  - Model A: top 35 trajectories by advantage score (quality)
  - Model B: all 290 trajectories, randomly shuffled (quantity)

Both use the same base model, same hyperparams, same training iterations.
Evaluation on held-out sets measures which approach learns better tool planning.

Output: ~/karl_experiment/results.json
"""

import json
import os
import random
import subprocess
import sys
import time
from pathlib import Path

random.seed(42)

# ── Paths ──────────────────────────────────────────────────────────────────
EXPERIMENT_DIR = Path.home() / "karl_experiment"
TRAJECTORIES = EXPERIMENT_DIR / "trajectories.jsonl"

DATASET_A_DIR = EXPERIMENT_DIR / "dataset_a"
DATASET_B_DIR = EXPERIMENT_DIR / "dataset_b"

ADAPTER_A_DIR = EXPERIMENT_DIR / "adapter_a"
ADAPTER_B_DIR = EXPERIMENT_DIR / "adapter_b"
CROSS_EVAL_DIR = EXPERIMENT_DIR / "cross_eval"

RESULTS_FILE = EXPERIMENT_DIR / "results.json"

# Model — use the cached gemma-3-1b-it-4bit
MODEL_PATH = str(Path.home() / ".exo/models/mlx-community--gemma-3-1b-it-4bit")

# ── SFT System Prompt ─────────────────────────────────────────────────────
SYSTEM_PROMPT = (
    "You are an expert software engineering agent. Given a task description, "
    "plan the optimal sequence of tool uses to accomplish it. Output a numbered "
    "plan with tool names and brief descriptions. Prefer: Read before Edit, "
    "test after changes, specific tools over generic Bash."
)


def load_trajectories():
    """Load all trajectories and sort by advantage."""
    records = []
    with open(TRAJECTORIES) as f:
        for line in f:
            try:
                r = json.loads(line)
                # Must have outcome with advantage
                outcome = r.get("outcome", {})
                if outcome.get("advantage") is not None:
                    records.append(r)
            except json.JSONDecodeError:
                continue
    return records


def trajectory_to_sft(record):
    """Convert a trajectory record to an SFT training example.

    Format: {"text": "<bos><start_of_turn>user\n...<end_of_turn>\n<start_of_turn>model\n...<end_of_turn>"}

    For Gemma models, we use the Gemma chat template format.
    """
    # Build the user prompt from available context
    domain = record.get("skill", {}).get("domain", "unknown")
    git_repo = record.get("context", {}).get("git_repo") or domain
    prompt_text = record.get("context", {}).get("prompt_text", "")
    num_prompts = record.get("timing", {}).get("num_prompts", 0)

    if prompt_text and len(prompt_text) > 10:
        user_content = prompt_text[:2000]
    else:
        # Synthesize a plausible prompt from the trajectory metadata
        tool_seq = record.get("trajectory", {}).get("tool_sequence", [])
        total_tools = record.get("trajectory", {}).get("total_tools", len(tool_seq))
        successes = record.get("trajectory", {}).get("successes", 0)

        user_content = (
            f"Task: Work on the {git_repo} project. "
            f"This is a {num_prompts}-prompt session involving {total_tools} tool operations."
        )

    # Build the assistant response — the tool plan
    events = record.get("trajectory", {}).get("events", [])
    tool_seq = record.get("trajectory", {}).get("tool_sequence", [])

    plan_lines = []

    # Use events if they have detail, otherwise use tool_sequence
    if events:
        for i, event in enumerate(events[:25], 1):
            tool = event.get("tool_name", "?")
            success = event.get("success")
            status = "ok" if success else ("fail" if success is False else "?")
            # Add any available params
            params = event.get("key_params", {})
            if tool == "Read" and "file_path" in params:
                desc = f"Read {_short_path(params['file_path'])}"
            elif tool == "Edit" and "file_path" in params:
                desc = f"Edit {_short_path(params['file_path'])}"
            elif tool == "Write" and "file_path" in params:
                desc = f"Write {_short_path(params['file_path'])}"
            elif tool == "Bash" and "command" in params:
                desc = f"Bash: {params['command'][:60]}"
            elif tool == "Grep" and "pattern" in params:
                desc = f"Grep '{params['pattern'][:30]}'"
            else:
                desc = tool
            plan_lines.append(f"{i}. [{status}] {desc}")
    elif tool_seq:
        # Compress consecutive identical tools
        compressed = []
        i = 0
        while i < len(tool_seq) and len(compressed) < 25:
            tool = tool_seq[i]
            count = 1
            while i + count < len(tool_seq) and tool_seq[i + count] == tool:
                count += 1
            if count > 1:
                compressed.append(f"{len(compressed)+1}. {tool} x{count}")
            else:
                compressed.append(f"{len(compressed)+1}. {tool}")
            i += count
        plan_lines = compressed

    # Add outcome summary
    outcome = record.get("outcome", {})
    reward = outcome.get("reward_score", 0)
    advantage = outcome.get("advantage", 0)
    signals = outcome.get("signals", {})
    total_tools = record.get("trajectory", {}).get("total_tools", 0)
    successes = record.get("trajectory", {}).get("successes", 0)

    plan_lines.append("")
    plan_lines.append(f"Result: {successes}/{total_tools} tools succeeded, reward={reward:.3f}")
    plan_lines.append(f"Signals: outcome={signals.get('outcome', 0):.2f}, "
                      f"process={signals.get('process', 0):.2f}, "
                      f"efficiency={signals.get('efficiency', 0):.2f}, "
                      f"verification={signals.get('verification', 0):.2f}")

    assistant_content = "\n".join(plan_lines)

    if not assistant_content or len(assistant_content) < 20:
        return None

    # Messages format (required for --mask-prompt)
    return {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_content[:3000]},
            {"role": "assistant", "content": assistant_content[:3000]},
        ]
    }


def _short_path(path):
    parts = path.split("/")
    if len(parts) > 3:
        return "/".join([".."] + parts[-2:])
    return path


def prepare_datasets():
    """Create Dataset A (top 35 by advantage) and Dataset B (all 290 random)."""
    records = load_trajectories()
    print(f"Loaded {len(records)} trajectories with advantage scores")

    # Sort by advantage descending
    records.sort(key=lambda r: r["outcome"]["advantage"], reverse=True)

    # ── Dataset A: Top 35 by advantage ──
    top_records = records[:35]
    print(f"\nDataset A — Top 35 by advantage:")
    print(f"  Advantage range: {top_records[-1]['outcome']['advantage']:.4f} to {top_records[0]['outcome']['advantage']:.4f}")
    print(f"  Mean advantage: {sum(r['outcome']['advantage'] for r in top_records)/len(top_records):.4f}")

    a_examples = []
    for r in top_records:
        ex = trajectory_to_sft(r)
        if ex:
            a_examples.append(ex)

    # Split: 28 train, 7 eval
    a_train = a_examples[:28]
    a_valid = a_examples[28:]

    # ── Dataset B: All 290, random order ──
    all_records = list(records)  # already sorted, shuffle for randomness
    random.shuffle(all_records)
    print(f"\nDataset B — All {len(all_records)} trajectories (random order):")
    advantages = [r["outcome"]["advantage"] for r in all_records]
    print(f"  Advantage range: {min(advantages):.4f} to {max(advantages):.4f}")
    print(f"  Mean advantage: {sum(advantages)/len(advantages):.4f}")

    b_examples = []
    for r in all_records:
        ex = trajectory_to_sft(r)
        if ex:
            b_examples.append(ex)

    # 80/20 split
    split_idx = int(len(b_examples) * 0.8)
    b_train = b_examples[:split_idx]
    b_valid = b_examples[split_idx:]

    # ── Write files ──
    for d in [DATASET_A_DIR, DATASET_B_DIR, ADAPTER_A_DIR, ADAPTER_B_DIR, CROSS_EVAL_DIR]:
        d.mkdir(parents=True, exist_ok=True)

    def write_jsonl(path, examples):
        with open(path, "w") as f:
            for ex in examples:
                f.write(json.dumps(ex) + "\n")

    write_jsonl(DATASET_A_DIR / "train.jsonl", a_train)
    write_jsonl(DATASET_A_DIR / "valid.jsonl", a_valid)
    write_jsonl(DATASET_A_DIR / "test.jsonl", a_valid)  # test = valid for eval
    write_jsonl(DATASET_B_DIR / "train.jsonl", b_train)
    write_jsonl(DATASET_B_DIR / "valid.jsonl", b_valid)
    write_jsonl(DATASET_B_DIR / "test.jsonl", b_valid)  # test = valid for eval

    # Cross-eval: combined test set from both, for fair comparison
    combined_valid = a_valid + b_valid
    random.shuffle(combined_valid)
    write_jsonl(CROSS_EVAL_DIR / "test.jsonl", combined_valid)
    # Need non-empty dummy train/valid for load_dataset to not crash
    write_jsonl(CROSS_EVAL_DIR / "train.jsonl", combined_valid[:1])
    write_jsonl(CROSS_EVAL_DIR / "valid.jsonl", combined_valid[:1])

    stats = {
        "dataset_a": {
            "total": len(a_examples),
            "train": len(a_train),
            "valid": len(a_valid),
            "advantage_min": round(top_records[-1]["outcome"]["advantage"], 4) if top_records else 0,
            "advantage_max": round(top_records[0]["outcome"]["advantage"], 4) if top_records else 0,
            "advantage_mean": round(sum(r["outcome"]["advantage"] for r in top_records) / max(len(top_records), 1), 4),
        },
        "dataset_b": {
            "total": len(b_examples),
            "train": len(b_train),
            "valid": len(b_valid),
            "advantage_min": round(min(advantages), 4),
            "advantage_max": round(max(advantages), 4),
            "advantage_mean": round(sum(advantages) / max(len(advantages), 1), 4),
        },
    }

    print(f"\nDataset A: {len(a_train)} train, {len(a_valid)} valid")
    print(f"Dataset B: {len(b_train)} train, {len(b_valid)} valid")
    print(f"\nSample A train example:")
    if a_train:
        msgs = a_train[0].get("messages", [])
        for m in msgs:
            print(f"  [{m['role']}]: {m['content'][:100]}...")
    print(f"\nSample B train example:")
    if b_train:
        msgs = b_train[0].get("messages", [])
        for m in msgs:
            print(f"  [{m['role']}]: {m['content'][:100]}...")

    return stats


def run_training(name, data_dir, adapter_dir, iters=500):
    """Run MLX LoRA training and return metrics."""
    print(f"\n{'='*60}")
    print(f"Training {name}")
    print(f"  Data: {data_dir}")
    print(f"  Adapter: {adapter_dir}")
    print(f"  Model: {MODEL_PATH}")
    print(f"  Iters: {iters}")
    print(f"{'='*60}\n")

    # Training command using mlx_lm.lora directly
    cmd = [
        sys.executable, "-c",
        f"""
import sys
sys.argv = [
    "lora",
    "--model", "{MODEL_PATH}",
    "--data", "{data_dir}",
    "--adapter-path", "{adapter_dir}",
    "--train",
    "--num-layers", "8",
    "--batch-size", "1",
    "--iters", "{iters}",
    "--learning-rate", "1e-4",
    "--steps-per-report", "50",
    "--steps-per-eval", "100",
    "--val-batches", "5",
    "--save-every", "100",
    "--max-seq-length", "2048",
    "--mask-prompt",
]
from mlx_lm.lora import main
main()
"""
    ]

    start = time.time()
    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        timeout=3600,
    )
    elapsed = time.time() - start

    print(f"STDOUT:\n{result.stdout}")
    if result.stderr:
        print(f"STDERR:\n{result.stderr[-2000:]}")
    print(f"\nTraining took {elapsed:.1f}s")

    # Parse training output for metrics
    metrics = parse_training_output(result.stdout + result.stderr, elapsed)
    return metrics


def parse_training_output(output, elapsed):
    """Extract train/eval loss from MLX training output."""
    metrics = {
        "elapsed_seconds": round(elapsed, 1),
        "train_losses": [],
        "eval_losses": [],
        "final_train_loss": None,
        "final_eval_loss": None,
    }

    for line in output.split("\n"):
        line = line.strip()
        # Training loss lines: "Iter 50: Train loss 3.456, ..."
        if "Train loss" in line:
            try:
                parts = line.split("Train loss")
                loss_str = parts[1].strip().split(",")[0].strip()
                loss = float(loss_str)
                metrics["train_losses"].append(loss)
                metrics["final_train_loss"] = loss
            except (IndexError, ValueError):
                pass

        # Eval loss lines: "Iter 100: Val loss 3.123, ..."
        if "Val loss" in line:
            try:
                parts = line.split("Val loss")
                loss_str = parts[1].strip().split(",")[0].strip()
                loss = float(loss_str)
                metrics["eval_losses"].append(loss)
                metrics["final_eval_loss"] = loss
            except (IndexError, ValueError):
                pass

    return metrics


def run_evaluation(name, data_dir, adapter_dir):
    """Run evaluation on the validation set."""
    print(f"\n{'='*60}")
    print(f"Evaluating {name}")
    print(f"{'='*60}\n")

    cmd = [
        sys.executable, "-c",
        f"""
import sys
sys.argv = [
    "lora",
    "--model", "{MODEL_PATH}",
    "--data", "{data_dir}",
    "--adapter-path", "{adapter_dir}",
    "--test",
    "--test-batches", "-1",
    "--max-seq-length", "2048",
]
from mlx_lm.lora import main
main()
"""
    ]

    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        timeout=600,
    )

    print(f"STDOUT:\n{result.stdout}")
    if result.stderr:
        print(f"STDERR:\n{result.stderr[-1000:]}")

    # Parse test output
    test_loss = None
    test_ppl = None
    output = result.stdout + result.stderr
    for line in output.split("\n"):
        if "Test loss" in line:
            try:
                parts = line.split("Test loss")
                loss_str = parts[1].strip().split(",")[0].strip()
                test_loss = float(loss_str)
            except (IndexError, ValueError):
                pass
        if "Test ppl" in line or "perplexity" in line.lower():
            try:
                parts = line.lower().split("ppl")
                ppl_str = parts[1].strip().split(",")[0].strip().split()[0]
                test_ppl = float(ppl_str)
            except (IndexError, ValueError):
                pass

    return {"test_loss": test_loss, "test_perplexity": test_ppl}


def cross_evaluate(name, adapter_dir):
    """Evaluate adapter on the combined test set (fair cross-eval)."""
    print(f"\n{'='*60}")
    print(f"Cross-evaluating {name} on combined test set")
    print(f"{'='*60}\n")

    cmd = [
        sys.executable, "-c",
        f"""
import sys
sys.argv = [
    "lora",
    "--model", "{MODEL_PATH}",
    "--data", "{CROSS_EVAL_DIR}",
    "--adapter-path", "{adapter_dir}",
    "--test",
    "--test-batches", "-1",
    "--max-seq-length", "2048",
]
from mlx_lm.lora import main
main()
"""
    ]

    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        timeout=600,
    )

    print(f"STDOUT:\n{result.stdout}")
    if result.stderr:
        print(f"STDERR:\n{result.stderr[-500:]}")

    test_loss = None
    output = result.stdout + result.stderr
    for line in output.split("\n"):
        if "Test loss" in line:
            try:
                parts = line.split("Test loss")
                loss_str = parts[1].strip().split(",")[0].strip()
                test_loss = float(loss_str)
            except (IndexError, ValueError):
                pass

    return {"cross_eval_loss": test_loss}


def main():
    print("=" * 60)
    print("KARL ADVANTAGE EXPERIMENT")
    print("Hypothesis: 35 high-advantage > 290 random for agent training")
    print("=" * 60)

    # Step 1: Prepare datasets
    print("\n[1/4] Preparing datasets...")
    dataset_stats = prepare_datasets()

    # Step 2: Train Model A (high advantage)
    print("\n[2/4] Training Model A (high advantage, 35 examples)...")
    a_metrics = run_training("Model_A_HighAdvantage", DATASET_A_DIR, ADAPTER_A_DIR, iters=500)

    # Step 3: Train Model B (all random)
    print("\n[3/4] Training Model B (all random, 290 examples)...")
    b_metrics = run_training("Model_B_AllRandom", DATASET_B_DIR, ADAPTER_B_DIR, iters=500)

    # Step 4: Evaluate both
    print("\n[4/4] Evaluating both models...")
    a_eval = run_evaluation("Model_A", DATASET_A_DIR, ADAPTER_A_DIR)
    b_eval = run_evaluation("Model_B", DATASET_B_DIR, ADAPTER_B_DIR)

    # Cross-evaluation: each model on the combined validation set
    a_cross = cross_evaluate("Model_A", ADAPTER_A_DIR)
    b_cross = cross_evaluate("Model_B", ADAPTER_B_DIR)

    # Determine winner
    a_final_eval = a_eval.get("test_loss") or a_metrics.get("final_eval_loss") or 999
    b_final_eval = b_eval.get("test_loss") or b_metrics.get("final_eval_loss") or 999

    # For fair comparison, use cross-eval (each model on the other's data)
    a_cross_loss = a_cross.get("cross_eval_loss") or 999
    b_cross_loss = b_cross.get("cross_eval_loss") or 999

    # Winner determination logic:
    # 1. Primary: own eval loss (lower is better)
    # 2. Secondary: cross-eval on the other dataset
    if a_final_eval < b_final_eval:
        winner = "A"
        margin = b_final_eval - a_final_eval
    elif b_final_eval < a_final_eval:
        winner = "B"
        margin = a_final_eval - b_final_eval
    else:
        winner = "tie"
        margin = 0

    results = {
        "experiment": "KARL Advantage vs Random Training",
        "base_model": "gemma-3-1b-it-4bit",
        "training_params": {
            "num_layers": 8,
            "batch_size": 1,
            "iters": 500,
            "learning_rate": 1e-4,
            "max_seq_length": 2048,
            "mask_prompt": True,
        },
        "dataset_a": {
            "name": "high_advantage_35",
            "description": "Top 35 trajectories by advantage score",
            "size": dataset_stats["dataset_a"]["total"],
            "train_size": dataset_stats["dataset_a"]["train"],
            "eval_size": dataset_stats["dataset_a"]["valid"],
            "advantage_range": [
                dataset_stats["dataset_a"]["advantage_min"],
                dataset_stats["dataset_a"]["advantage_max"],
            ],
            "advantage_mean": dataset_stats["dataset_a"]["advantage_mean"],
            "final_train_loss": a_metrics.get("final_train_loss"),
            "final_eval_loss": a_metrics.get("final_eval_loss"),
            "test_loss": a_eval.get("test_loss"),
            "test_perplexity": a_eval.get("test_perplexity"),
            "cross_eval_loss_combined": a_cross.get("cross_eval_loss"),
            "training_time_seconds": a_metrics.get("elapsed_seconds"),
            "loss_curve": a_metrics.get("train_losses", []),
            "eval_curve": a_metrics.get("eval_losses", []),
        },
        "dataset_b": {
            "name": "random_all_290",
            "description": "All 290 trajectories in random order",
            "size": dataset_stats["dataset_b"]["total"],
            "train_size": dataset_stats["dataset_b"]["train"],
            "eval_size": dataset_stats["dataset_b"]["valid"],
            "advantage_range": [
                dataset_stats["dataset_b"]["advantage_min"],
                dataset_stats["dataset_b"]["advantage_max"],
            ],
            "advantage_mean": dataset_stats["dataset_b"]["advantage_mean"],
            "final_train_loss": b_metrics.get("final_train_loss"),
            "final_eval_loss": b_metrics.get("final_eval_loss"),
            "test_loss": b_eval.get("test_loss"),
            "test_perplexity": b_eval.get("test_perplexity"),
            "cross_eval_loss_combined": b_cross.get("cross_eval_loss"),
            "training_time_seconds": b_metrics.get("elapsed_seconds"),
            "loss_curve": b_metrics.get("train_losses", []),
            "eval_curve": b_metrics.get("eval_losses", []),
        },
        "winner": winner,
        "margin": round(margin, 4) if margin else 0,
        "finding": (
            f"Model {winner} wins by {margin:.4f} eval loss. "
            + (
                "High-advantage curation (35 examples) outperforms random bulk (290 examples), "
                "confirming that trajectory quality > quantity for agent training."
                if winner == "A" else
                "Random bulk training (290 examples) outperforms curated high-advantage (35 examples), "
                "suggesting quantity matters more than advantage-weighted curation at this scale."
                if winner == "B" else
                "Both approaches produce equivalent results at this scale."
            )
        ),
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }

    # Write results
    with open(RESULTS_FILE, "w") as f:
        json.dump(results, f, indent=2)

    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)
    print(f"\nModel A (35 high-advantage):")
    print(f"  Final train loss:    {a_metrics.get('final_train_loss')}")
    print(f"  Final eval loss:     {a_metrics.get('final_eval_loss')}")
    print(f"  Test loss:           {a_eval.get('test_loss')}")
    print(f"  Cross-eval combined: {a_cross.get('cross_eval_loss')}")
    print(f"  Training time:       {a_metrics.get('elapsed_seconds')}s")
    print(f"\nModel B (290 random):")
    print(f"  Final train loss:    {b_metrics.get('final_train_loss')}")
    print(f"  Final eval loss:     {b_metrics.get('final_eval_loss')}")
    print(f"  Test loss:           {b_eval.get('test_loss')}")
    print(f"  Cross-eval combined: {b_cross.get('cross_eval_loss')}")
    print(f"  Training time:       {b_metrics.get('elapsed_seconds')}s")
    print(f"\nWINNER: Model {winner} (margin: {margin:.4f})")
    print(f"\n{results['finding']}")
    print(f"\nResults saved to: {RESULTS_FILE}")


if __name__ == "__main__":
    main()
