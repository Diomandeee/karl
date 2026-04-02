#!/usr/bin/env python3
"""
KARL Comparison Experiment — Paper 4 Core Result

HYPOTHESIS: 35 high-advantage trajectories produce a better model
than 35 (or more) random trajectories.

Datasets:
  A  = Top 35 by advantage score (curated, high-signal)
  B  = 35 random trajectories (same size, fair comparison)
  B2 = All remaining non-test trajectories (scale comparison)
  Test = 20% hold-out, stratified by reward quartile

Training: MLX LoRA on Qwen/Qwen2.5-3B-Instruct
  - 500 iterations, batch_size=1, lr=1e-4, num-layers=8

Evaluation:
  - Training loss curves
  - Validation loss on hold-out
  - Generation quality on 10 test prompts
"""

import json
import os
import random
import subprocess
import sys
import time
from pathlib import Path

EXPERIMENT_DIR = Path.home() / "karl-experiment"
TRAJECTORIES = EXPERIMENT_DIR / "trajectories.jsonl"
RESULTS_FILE = EXPERIMENT_DIR / "comparison_results.json"

# MLX LoRA config
BASE_MODEL = "Qwen/Qwen2.5-7B-Instruct"
NUM_ITERS = 500
BATCH_SIZE = 1
LEARNING_RATE = 1e-4
NUM_LAYERS = 8
SEED = 42

SYSTEM_PROMPT = (
    "You are an expert software engineering assistant. Given a task, "
    "plan the optimal sequence of tool uses to accomplish it efficiently. "
    "Consider which tools to use, in what order, and what parameters. "
    "Prefer reading before editing, testing after changes, and using "
    "the most specific tool available."
)


def load_trajectories():
    """Load all trajectories from JSONL file."""
    records = []
    with open(TRAJECTORIES) as f:
        for line in f:
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    print(f"Loaded {len(records)} trajectories")
    return records


def trajectory_to_sft(record):
    """Convert a trajectory record to SFT chat format."""
    events = record.get("trajectory", {}).get("events", [])
    if len(events) < 2:
        return None

    # Build the user prompt from context
    context = record.get("context", {})
    prompt = context.get("prompt_text", "")
    if not prompt or len(prompt) < 10:
        git_repo = context.get("git_repo", "unknown project")
        skill = record.get("skill", {}).get("domain", "unknown")
        prompt = f"Complete the following task in the {git_repo} project (domain: {skill}). Use the available tools efficiently."

    # Build the assistant response (tool plan)
    parts = []
    for i, event in enumerate(events[:20], 1):
        tool = event.get("tool_name", "?")
        params = event.get("key_params", {})
        success = event.get("success")
        status = "ok" if success else ("fail" if success is False else "?")

        if tool == "Read" and "file_path" in params:
            p = params["file_path"].split("/")
            short = "/".join([".."] + p[-2:]) if len(p) > 3 else params["file_path"]
            desc = f"Read {short}"
        elif tool == "Edit" and "file_path" in params:
            p = params["file_path"].split("/")
            short = "/".join([".."] + p[-2:]) if len(p) > 3 else params["file_path"]
            desc = f"Edit {short}"
        elif tool == "Write" and "file_path" in params:
            p = params["file_path"].split("/")
            short = "/".join([".."] + p[-2:]) if len(p) > 3 else params["file_path"]
            desc = f"Write {short}"
        elif tool == "Bash" and "command" in params:
            desc = f"Bash: {params['command'][:80]}"
        elif tool == "Grep" and "pattern" in params:
            desc = f"Grep '{params['pattern'][:40]}'"
        elif tool == "Glob" and "pattern" in params:
            desc = f"Glob '{params['pattern'][:40]}'"
        else:
            param_str = ", ".join(
                f"{k}={str(v)[:30]}"
                for k, v in list((params or {}).items())[:2]
            )
            desc = f"{tool}({param_str})" if param_str else tool

        parts.append(f"{i}. [{status}] {desc}")

    # Add outcome summary
    outcome = record.get("outcome", {})
    reward = outcome.get("reward_score", 0)
    total = record.get("trajectory", {}).get("total_tools", 0)
    successes = record.get("trajectory", {}).get("successes", 0)
    plan = "\n".join(parts)
    plan += f"\n\nResult: {successes}/{total} tools succeeded, reward={reward:.2f}"

    if len(plan) < 20:
        return None

    return {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt[:4000]},
            {"role": "assistant", "content": plan[:4000]},
        ]
    }


def split_datasets(records):
    """Split trajectories into experimental datasets.

    Returns: (dataset_a, dataset_b, dataset_b2, test_set, stats)
    """
    random.seed(SEED)

    # Sort by advantage score (descending)
    scored = [r for r in records if r.get("outcome", {}).get("advantage") is not None]
    scored.sort(key=lambda r: r["outcome"]["advantage"], reverse=True)

    print(f"\nScored trajectories: {len(scored)}")
    print(f"Advantage range: {scored[-1]['outcome']['advantage']:.4f} to {scored[0]['outcome']['advantage']:.4f}")

    # Stratified test split: divide into quartiles by reward, take 20% from each
    quartile_size = len(scored) // 4
    test_set = []
    remaining = []

    for q in range(4):
        start = q * quartile_size
        end = (q + 1) * quartile_size if q < 3 else len(scored)
        quartile = scored[start:end]
        random.shuffle(quartile)
        n_test = max(1, len(quartile) // 5)  # 20%
        test_set.extend(quartile[:n_test])
        remaining.extend(quartile[n_test:])

    print(f"Test set: {len(test_set)} (stratified 20%)")
    print(f"Remaining for training: {len(remaining)}")

    # Re-sort remaining by advantage
    remaining.sort(key=lambda r: r["outcome"]["advantage"], reverse=True)

    # Dataset A: Top 35 by advantage from remaining
    dataset_a = remaining[:35]
    non_a_remaining = remaining[35:]

    # Dataset B: 35 random from remaining (excluding top-35)
    random.shuffle(non_a_remaining)
    dataset_b = non_a_remaining[:35]

    # Dataset B2: ALL remaining (including both a-eligible and non-a)
    # For fair comparison, use all non-test data EXCEPT dataset_a
    dataset_b2 = non_a_remaining  # Everything not in A or test

    # Stats
    a_advantages = [r["outcome"]["advantage"] for r in dataset_a]
    b_advantages = [r["outcome"]["advantage"] for r in dataset_b]
    b2_advantages = [r["outcome"]["advantage"] for r in dataset_b2]
    test_advantages = [r["outcome"]["advantage"] for r in test_set]

    a_rewards = [r["outcome"]["reward_score"] for r in dataset_a]
    b_rewards = [r["outcome"]["reward_score"] for r in dataset_b]

    stats = {
        "total_trajectories": len(records),
        "scored": len(scored),
        "test_size": len(test_set),
        "dataset_a_size": len(dataset_a),
        "dataset_b_size": len(dataset_b),
        "dataset_b2_size": len(dataset_b2),
        "dataset_a": {
            "advantage_mean": sum(a_advantages) / len(a_advantages),
            "advantage_min": min(a_advantages),
            "advantage_max": max(a_advantages),
            "reward_mean": sum(a_rewards) / len(a_rewards),
            "reward_min": min(a_rewards),
            "reward_max": max(a_rewards),
        },
        "dataset_b": {
            "advantage_mean": sum(b_advantages) / len(b_advantages),
            "advantage_min": min(b_advantages),
            "advantage_max": max(b_advantages),
            "reward_mean": sum(b_rewards) / len(b_rewards),
            "reward_min": min(b_rewards),
            "reward_max": max(b_rewards),
        },
        "dataset_b2": {
            "size": len(dataset_b2),
            "advantage_mean": sum(b2_advantages) / len(b2_advantages) if b2_advantages else 0,
        },
        "test_set": {
            "advantage_mean": sum(test_advantages) / len(test_advantages),
        },
    }

    print(f"\n--- Dataset A (Top 35 by advantage) ---")
    print(f"  Advantage: {stats['dataset_a']['advantage_min']:.4f} to {stats['dataset_a']['advantage_max']:.4f} (mean {stats['dataset_a']['advantage_mean']:.4f})")
    print(f"  Reward: {stats['dataset_a']['reward_min']:.4f} to {stats['dataset_a']['reward_max']:.4f} (mean {stats['dataset_a']['reward_mean']:.4f})")

    print(f"\n--- Dataset B (35 Random) ---")
    print(f"  Advantage: {stats['dataset_b']['advantage_min']:.4f} to {stats['dataset_b']['advantage_max']:.4f} (mean {stats['dataset_b']['advantage_mean']:.4f})")
    print(f"  Reward: {stats['dataset_b']['reward_min']:.4f} to {stats['dataset_b']['reward_max']:.4f} (mean {stats['dataset_b']['reward_mean']:.4f})")

    print(f"\n--- Dataset B2 (All {len(dataset_b2)} non-A remaining) ---")
    print(f"  Advantage mean: {stats['dataset_b2']['advantage_mean']:.4f}")

    return dataset_a, dataset_b, dataset_b2, test_set, stats


def write_sft_data(records, data_dir, name):
    """Convert records to SFT and write train/valid splits."""
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)

    examples = []
    skipped = 0
    for r in records:
        ex = trajectory_to_sft(r)
        if ex:
            examples.append(ex)
        else:
            skipped += 1

    if not examples:
        print(f"  WARNING: No valid examples for {name}!")
        return 0

    # 90/10 train/valid split
    random.seed(SEED)
    random.shuffle(examples)
    split_idx = max(1, int(len(examples) * 0.9))
    train = examples[:split_idx]
    valid = examples[split_idx:] if split_idx < len(examples) else [examples[-1]]

    # Ensure valid has at least 1 example
    if not valid:
        valid = [train[-1]]

    train_path = data_dir / "train.jsonl"
    valid_path = data_dir / "valid.jsonl"

    with open(train_path, "w") as f:
        for ex in train:
            f.write(json.dumps(ex) + "\n")

    with open(valid_path, "w") as f:
        for ex in valid:
            f.write(json.dumps(ex) + "\n")

    print(f"  {name}: {len(train)} train, {len(valid)} valid (skipped {skipped})")
    return len(examples)


def write_test_data(records, data_dir):
    """Write test set for evaluation."""
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)

    examples = []
    for r in records:
        ex = trajectory_to_sft(r)
        if ex:
            examples.append(ex)

    test_path = data_dir / "test.jsonl"
    # Also write as valid.jsonl for MLX eval
    valid_path = data_dir / "valid.jsonl"

    with open(test_path, "w") as f:
        for ex in examples:
            f.write(json.dumps(ex) + "\n")

    with open(valid_path, "w") as f:
        for ex in examples:
            f.write(json.dumps(ex) + "\n")

    print(f"  Test set: {len(examples)} examples written")
    return len(examples)


def train_model(data_dir, adapter_dir, name):
    """Train a LoRA adapter using MLX.

    Returns: dict with training metrics
    """
    print(f"\n{'='*60}")
    print(f"TRAINING MODEL {name}")
    print(f"{'='*60}")

    data_dir = Path(data_dir)
    adapter_dir = Path(adapter_dir)
    adapter_dir.mkdir(parents=True, exist_ok=True)

    # Verify data exists
    train_file = data_dir / "train.jsonl"
    valid_file = data_dir / "valid.jsonl"
    if not train_file.exists():
        return {"error": f"No training data at {train_file}"}

    train_count = sum(1 for _ in open(train_file))
    valid_count = sum(1 for _ in open(valid_file)) if valid_file.exists() else 0
    print(f"  Train examples: {train_count}")
    print(f"  Valid examples: {valid_count}")

    cmd = [
        sys.executable, "-m", "mlx_lm", "lora",
        "--model", BASE_MODEL,
        "--data", str(data_dir),
        "--train",
        "--adapter-path", str(adapter_dir),
        "--iters", str(NUM_ITERS),
        "--batch-size", str(BATCH_SIZE),
        "--learning-rate", str(LEARNING_RATE),
        "--num-layers", str(NUM_LAYERS),
        "--seed", str(SEED),
    ]

    print(f"  Command: {' '.join(cmd)}")
    print(f"  Starting at {time.strftime('%H:%M:%S')}")

    start_time = time.time()
    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        cwd=str(EXPERIMENT_DIR),
    )

    elapsed = time.time() - start_time
    print(f"  Finished in {elapsed:.1f}s ({elapsed/60:.1f}m)")

    # Parse training output for loss values
    stdout = result.stdout
    stderr = result.stderr
    output = stdout + "\n" + stderr

    # Save full output
    log_file = adapter_dir / "training_log.txt"
    with open(log_file, "w") as f:
        f.write(f"=== STDOUT ===\n{stdout}\n\n=== STDERR ===\n{stderr}\n")

    # Extract loss values from output
    train_losses = []
    val_losses = []
    for line in output.split("\n"):
        # MLX LoRA outputs lines like: "Iter 10: Train loss 3.456, Val loss 2.789"
        # or: "Iter 10: Train loss 3.456, It/sec 1.23"
        if "Train loss" in line:
            try:
                parts = line.split("Train loss")
                loss_str = parts[1].strip().split(",")[0].strip()
                train_losses.append(float(loss_str))
            except (IndexError, ValueError):
                pass
        if "Val loss" in line:
            try:
                parts = line.split("Val loss")
                loss_str = parts[1].strip().split(",")[0].strip().rstrip(".")
                val_losses.append(float(loss_str))
            except (IndexError, ValueError):
                pass

    metrics = {
        "name": name,
        "train_examples": train_count,
        "valid_examples": valid_count,
        "iterations": NUM_ITERS,
        "elapsed_seconds": round(elapsed, 1),
        "return_code": result.returncode,
        "train_losses": train_losses,
        "val_losses": val_losses,
        "final_train_loss": train_losses[-1] if train_losses else None,
        "final_val_loss": val_losses[-1] if val_losses else None,
        "best_val_loss": min(val_losses) if val_losses else None,
        "adapter_path": str(adapter_dir),
    }

    if result.returncode != 0:
        metrics["error"] = stderr[-2000:] if len(stderr) > 2000 else stderr
        print(f"  ERROR (rc={result.returncode})")
        print(f"  stderr: {stderr[-500:]}")
    else:
        print(f"  Final train loss: {metrics['final_train_loss']}")
        print(f"  Final val loss: {metrics['final_val_loss']}")
        print(f"  Best val loss: {metrics['best_val_loss']}")

    return metrics


def evaluate_on_test(adapter_dir, test_dir, name):
    """Evaluate a trained adapter on the test set.

    Uses MLX LoRA eval mode to compute test loss.
    """
    print(f"\n--- Evaluating {name} on test set ---")

    adapter_dir = Path(adapter_dir)
    test_dir = Path(test_dir)

    # Check adapter exists
    adapter_file = adapter_dir / "adapters.safetensors"
    if not adapter_file.exists():
        print(f"  No adapter found at {adapter_file}")
        return {"error": "no adapter file"}

    cmd = [
        sys.executable, "-m", "mlx_lm", "lora",
        "--model", BASE_MODEL,
        "--data", str(test_dir),
        "--adapter-path", str(adapter_dir),
        "--test",
    ]

    print(f"  Command: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True, cwd=str(EXPERIMENT_DIR))

    output = result.stdout + "\n" + result.stderr

    # Parse test loss
    test_loss = None
    for line in output.split("\n"):
        if "Test loss" in line:
            try:
                parts = line.split("Test loss")
                loss_str = parts[1].strip().split(",")[0].strip().rstrip(".")
                test_loss = float(loss_str)
            except (IndexError, ValueError):
                pass
        # Also check for "test_loss" format
        if "test_loss" in line.lower():
            try:
                parts = line.lower().split("test_loss")
                loss_str = parts[1].strip().strip(":= ").split()[0].strip().rstrip(".")
                test_loss = float(loss_str)
            except (IndexError, ValueError):
                pass

    # Also try perplexity
    test_ppl = None
    for line in output.split("\n"):
        if "perplexity" in line.lower() or "ppl" in line.lower():
            try:
                import re
                nums = re.findall(r"[\d.]+", line)
                if nums:
                    test_ppl = float(nums[-1])
            except (ValueError, IndexError):
                pass

    eval_result = {
        "name": name,
        "test_loss": test_loss,
        "test_perplexity": test_ppl,
        "return_code": result.returncode,
        "raw_output": output[-2000:],
    }

    if test_loss is not None:
        print(f"  Test loss: {test_loss:.4f}")
    if test_ppl is not None:
        print(f"  Test perplexity: {test_ppl:.4f}")
    if result.returncode != 0:
        print(f"  Eval error (rc={result.returncode})")
        print(f"  Output: {output[-500:]}")

    return eval_result


def generate_completions(adapter_dir, test_dir, name, n_prompts=10):
    """Generate completions for test prompts to compare quality."""
    print(f"\n--- Generating completions for {name} ---")

    adapter_dir = Path(adapter_dir)
    test_dir = Path(test_dir)

    # Load test prompts
    test_file = test_dir / "test.jsonl"
    if not test_file.exists():
        return {"error": "no test file"}

    prompts = []
    with open(test_file) as f:
        for line in f:
            try:
                ex = json.loads(line)
                msgs = ex.get("messages", [])
                if len(msgs) >= 2:
                    prompts.append(msgs[1]["content"])
            except json.JSONDecodeError:
                continue

    prompts = prompts[:n_prompts]
    completions = []

    for i, prompt in enumerate(prompts):
        print(f"  Prompt {i+1}/{len(prompts)}...")

        # Use mlx_lm generate
        cmd = [
            sys.executable, "-m", "mlx_lm", "generate",
            "--model", BASE_MODEL,
            "--adapter-path", str(adapter_dir),
            "--prompt", prompt[:500],
            "--max-tokens", "200",
        ]

        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=120,
            cwd=str(EXPERIMENT_DIR),
        )

        output = result.stdout.strip()
        completions.append({
            "prompt": prompt[:200],
            "completion": output[-500:] if output else "(empty)",
            "return_code": result.returncode,
        })

    return {
        "name": name,
        "num_prompts": len(prompts),
        "completions": completions,
    }


def compute_baseline_test_loss(test_dir):
    """Compute baseline test loss (no adapter, raw model)."""
    print(f"\n--- Computing baseline test loss (no adapter) ---")

    test_dir = Path(test_dir)

    # Create a dummy adapter dir with no adapter so we can test raw model
    # Actually, just run without --adapter-path
    cmd = [
        sys.executable, "-m", "mlx_lm", "lora",
        "--model", BASE_MODEL,
        "--data", str(test_dir),
        "--test",
    ]

    result = subprocess.run(cmd, capture_output=True, text=True, cwd=str(EXPERIMENT_DIR))
    output = result.stdout + "\n" + result.stderr

    test_loss = None
    for line in output.split("\n"):
        if "Test loss" in line:
            try:
                parts = line.split("Test loss")
                loss_str = parts[1].strip().split(",")[0].strip().rstrip(".")
                test_loss = float(loss_str)
            except (IndexError, ValueError):
                pass
        if "test_loss" in line.lower():
            try:
                parts = line.lower().split("test_loss")
                loss_str = parts[1].strip().strip(":= ").split()[0].strip().rstrip(".")
                test_loss = float(loss_str)
            except (IndexError, ValueError):
                pass

    print(f"  Baseline test loss: {test_loss}")
    return {"baseline_test_loss": test_loss, "raw_output": output[-1000:]}


def main():
    print("=" * 60)
    print("KARL COMPARISON EXPERIMENT")
    print("Hypothesis: 35 high-advantage > 35 random trajectories")
    print("=" * 60)

    # Step 1: Load data
    records = load_trajectories()

    # Step 2: Split into experimental datasets
    dataset_a, dataset_b, dataset_b2, test_set, split_stats = split_datasets(records)

    # Step 3: Convert to SFT format and write
    print("\n--- Writing SFT data ---")
    dir_a = EXPERIMENT_DIR / "data_a"
    dir_b = EXPERIMENT_DIR / "data_b"
    dir_b2 = EXPERIMENT_DIR / "data_b2"
    dir_test = EXPERIMENT_DIR / "data_test"

    n_a = write_sft_data(dataset_a, dir_a, "Dataset A (Top-35 Advantage)")
    n_b = write_sft_data(dataset_b, dir_b, "Dataset B (35 Random)")
    n_b2 = write_sft_data(dataset_b2, dir_b2, "Dataset B2 (All Remaining)")
    n_test = write_test_data(test_set, dir_test)

    # Step 4: Compute baseline
    baseline = compute_baseline_test_loss(dir_test)

    # Step 5: Train models
    metrics_a = train_model(dir_a, EXPERIMENT_DIR / "adapters_a", "Model A (Top-35)")
    metrics_b = train_model(dir_b, EXPERIMENT_DIR / "adapters_b", "Model B (Random-35)")
    metrics_b2 = train_model(dir_b2, EXPERIMENT_DIR / "adapters_b2", "Model B2 (All Remaining)")

    # Step 6: Evaluate on test set
    eval_a = evaluate_on_test(EXPERIMENT_DIR / "adapters_a", dir_test, "Model A")
    eval_b = evaluate_on_test(EXPERIMENT_DIR / "adapters_b", dir_test, "Model B")
    eval_b2 = evaluate_on_test(EXPERIMENT_DIR / "adapters_b2", dir_test, "Model B2")

    # Step 7: Generate completions for comparison
    gen_a = generate_completions(EXPERIMENT_DIR / "adapters_a", dir_test, "Model A", n_prompts=5)
    gen_b = generate_completions(EXPERIMENT_DIR / "adapters_b", dir_test, "Model B", n_prompts=5)

    # Step 8: Compile results
    results = {
        "experiment": "KARL Advantage-Weighted SFT Comparison",
        "hypothesis": "35 high-advantage trajectories produce a better model than 35 random trajectories",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "base_model": BASE_MODEL,
        "config": {
            "iterations": NUM_ITERS,
            "batch_size": BATCH_SIZE,
            "learning_rate": LEARNING_RATE,
            "num_layers": NUM_LAYERS,
            "seed": SEED,
        },
        "data_split": split_stats,
        "sft_counts": {
            "dataset_a": n_a,
            "dataset_b": n_b,
            "dataset_b2": n_b2,
            "test": n_test,
        },
        "baseline": baseline,
        "training": {
            "model_a": metrics_a,
            "model_b": metrics_b,
            "model_b2": metrics_b2,
        },
        "evaluation": {
            "model_a": eval_a,
            "model_b": eval_b,
            "model_b2": eval_b2,
        },
        "generations": {
            "model_a": gen_a,
            "model_b": gen_b,
        },
        "conclusion": {},
    }

    # Compute conclusion
    a_test = eval_a.get("test_loss")
    b_test = eval_b.get("test_loss")
    b2_test = eval_b2.get("test_loss")
    base_test = baseline.get("baseline_test_loss")

    if a_test and b_test:
        delta = b_test - a_test
        pct = (delta / b_test) * 100 if b_test else 0
        results["conclusion"] = {
            "model_a_test_loss": a_test,
            "model_b_test_loss": b_test,
            "model_b2_test_loss": b2_test,
            "baseline_test_loss": base_test,
            "delta_a_vs_b": round(delta, 4),
            "improvement_pct": round(pct, 2),
            "hypothesis_supported": a_test < b_test,
            "a_beats_b": a_test < b_test,
            "a_beats_baseline": a_test < base_test if base_test else None,
            "b2_beats_a": b2_test < a_test if b2_test and a_test else None,
            "summary": (
                f"Model A (top-35 advantage) achieved test loss {a_test:.4f} vs "
                f"Model B (35 random) at {b_test:.4f}. "
                f"{'HYPOTHESIS SUPPORTED' if a_test < b_test else 'HYPOTHESIS NOT SUPPORTED'}: "
                f"advantage-weighted selection {'improves' if a_test < b_test else 'does not improve'} "
                f"over random selection by {abs(pct):.1f}%."
            ),
        }
        if b2_test:
            results["conclusion"]["summary"] += (
                f" Model B2 (all {n_b2} remaining) achieved {b2_test:.4f}. "
                f"Scale {'helps' if b2_test < a_test else 'does not overcome curation'}."
            )

    # Save results
    with open(RESULTS_FILE, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n{'='*60}")
    print("EXPERIMENT COMPLETE")
    print(f"{'='*60}")
    print(f"\nResults saved to: {RESULTS_FILE}")

    if results.get("conclusion"):
        print(f"\n{results['conclusion'].get('summary', 'No conclusion')}")

    return results


if __name__ == "__main__":
    main()
