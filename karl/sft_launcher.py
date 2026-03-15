#!/usr/bin/env python3
"""
KARL SFT Training Launcher — Fine-tunes routing model on Mac4 via MLX LoRA.

Reads the SFT JSONL export, validates format, deploys to Mac4, and launches
MLX LoRA fine-tuning with OAPL-Lite advantage weighting.

Usage:
    python3 sft_launcher.py --dry-run         # Validate data + show plan
    python3 sft_launcher.py --launch          # Deploy + start training on Mac4
    python3 sft_launcher.py --status          # Check running training status
    python3 sft_launcher.py --prepare-only    # Prepare data but don't launch
"""

import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

KARL_DIR = Path(__file__).parent
SFT_DIR = KARL_DIR / "sft_export"
CONFIG_PATH = KARL_DIR / "config.json"
TRAINING_LOG = KARL_DIR / "training_runs.jsonl"

# Mac4 connection
MAC4_HOST = os.environ.get("MAC4_HOST", "mac4")
MAC4_TRAINING_DIR = "/Users/mohameddiomande/.claude/karl/training"
MAC4_VENV_PYTHON = "/Users/mohameddiomande/.claude/karl/venv/bin/python3"

# Model config
BASE_MODEL = "mlx-community/Qwen2.5-Coder-7B-Instruct-4bit"
LORA_RANK = 8
LORA_LAYERS = 16
LEARNING_RATE = 1e-5
BATCH_SIZE = 2
EPOCHS = 3
MAX_SEQ_LEN = 2048


def find_latest_sft_export():
    """Find the most recent SFT export file."""
    candidates = []
    # Check sft_export/ subdirectory
    if SFT_DIR.exists():
        candidates.extend(SFT_DIR.glob("sft_*.jsonl"))
    # Check KARL_DIR root for sft_batch.jsonl etc
    candidates.extend(KARL_DIR.glob("sft_*.jsonl"))
    # Filter out non-data files (like sft_exporter.py, sft_launcher.py)
    candidates = [p for p in candidates if p.suffix == ".jsonl"]
    if not candidates:
        return None
    return sorted(candidates, key=lambda p: p.stat().st_mtime, reverse=True)[0]


def validate_sft_data(path):
    """Validate SFT JSONL format and content. Supports both formats:
    - messages format: {"messages": [{"role": "user", ...}, {"role": "assistant", ...}]}
    - prompt/completion format: {"prompt": "...", "completion": "..."}
    """
    issues = []
    warnings = []
    records = []
    fmt = None
    with open(path) as f:
        for i, line in enumerate(f):
            try:
                r = json.loads(line)
                records.append(r)
            except json.JSONDecodeError:
                issues.append(f"Line {i}: invalid JSON")
                continue

            # Detect format from first record
            if fmt is None:
                if "messages" in r:
                    fmt = "messages"
                elif "prompt" in r:
                    fmt = "prompt_completion"
                else:
                    issues.append(f"Line {i}: unknown format (no 'messages' or 'prompt')")
                    continue

            if fmt == "messages":
                if "messages" not in r:
                    issues.append(f"Line {i}: missing 'messages' field")
                else:
                    msgs = r["messages"]
                    if not isinstance(msgs, list) or len(msgs) < 2:
                        issues.append(f"Line {i}: messages must have at least 2 entries")
                    else:
                        roles = [m.get("role") for m in msgs]
                        if "user" not in roles:
                            issues.append(f"Line {i}: no user message")
                        if "assistant" not in roles:
                            issues.append(f"Line {i}: no assistant message")
            elif fmt == "prompt_completion":
                if not r.get("prompt"):
                    warnings.append(f"Line {i}: empty prompt (will be skipped)")
                if not r.get("completion"):
                    warnings.append(f"Line {i}: empty completion (will be skipped)")

    # Count usable records (non-empty prompt+completion)
    usable = [r for r in records if r.get("prompt") or r.get("messages")]
    if fmt == "prompt_completion":
        usable = [r for r in records if r.get("prompt") and r.get("completion")]

    # Stats
    total = len(records)
    has_weight = sum(1 for r in records if "weight" in r)
    weights = [r.get("weight", 1.0) for r in records]
    avg_weight = sum(weights) / len(weights) if weights else 0

    return {
        "valid": len(issues) == 0,
        "format": fmt or "unknown",
        "total": total,
        "usable": len(usable),
        "skipped": total - len(usable),
        "issues": issues[:10],
        "warnings": warnings[:10],
        "has_weight": has_weight,
        "avg_weight": round(avg_weight, 4),
        "min_weight": round(min(weights), 4) if weights else 0,
        "max_weight": round(max(weights), 4) if weights else 0,
    }


def check_mac4_ready():
    """Check if Mac4 is reachable and has required tools."""
    result = {"reachable": False, "mlx_lm": False, "disk_gb": 0, "errors": []}

    try:
        out = subprocess.run(
            ["ssh", "-o", "ConnectTimeout=5", MAC4_HOST, "echo ok"],
            capture_output=True, text=True, timeout=10,
        )
        result["reachable"] = out.returncode == 0
    except Exception as e:
        result["errors"].append(f"SSH failed: {e}")
        return result

    # Check mlx-lm (in venv)
    out = subprocess.run(
        ["ssh", MAC4_HOST, f"{MAC4_VENV_PYTHON} -c 'import mlx_lm; print(mlx_lm.__version__)' 2>/dev/null || echo 'NOT_INSTALLED'"],
        capture_output=True, text=True, timeout=15,
    )
    version = out.stdout.strip()
    if version and version != "NOT_INSTALLED":
        result["mlx_lm"] = True
        result["mlx_lm_version"] = version
    else:
        result["errors"].append("mlx-lm not installed — run: pip3 install mlx-lm")

    # Disk space
    out = subprocess.run(
        ["ssh", MAC4_HOST, "df -g / | tail -1 | awk '{print $4}'"],
        capture_output=True, text=True, timeout=10,
    )
    try:
        result["disk_gb"] = int(out.stdout.strip())
    except ValueError:
        pass

    return result


def _to_messages_format(record):
    """Convert prompt/completion format to messages format for MLX."""
    if "messages" in record:
        return record
    prompt = record.get("prompt", "")
    completion = record.get("completion", "")
    converted = {
        "messages": [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": completion},
        ]
    }
    if "weight" in record:
        converted["weight"] = record["weight"]
    return converted


def prepare_training_data(sft_path, output_dir=None):
    """Convert SFT JSONL to MLX-compatible train/valid split (messages format)."""
    records = []
    with open(sft_path) as f:
        for line in f:
            try:
                r = json.loads(line)
                # Skip empty prompt/completion records
                if "prompt" in r and (not r.get("prompt") or not r.get("completion")):
                    continue
                records.append(_to_messages_format(r))
            except json.JSONDecodeError:
                continue

    if not records:
        return {"error": "No valid records"}

    # 90/10 train/valid split
    split_idx = int(len(records) * 0.9)
    train = records[:split_idx]
    valid = records[split_idx:]

    if output_dir is None:
        output_dir = KARL_DIR / "training_data"
    else:
        output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(output_dir / "train.jsonl", "w") as f:
        for r in train:
            f.write(json.dumps(r, default=str) + "\n")
    with open(output_dir / "valid.jsonl", "w") as f:
        for r in valid:
            f.write(json.dumps(r, default=str) + "\n")

    return {
        "output_dir": str(output_dir),
        "train_count": len(train),
        "valid_count": len(valid),
    }


def deploy_to_mac4(training_data_dir):
    """SCP training data to Mac4."""
    # Create remote directory
    subprocess.run(
        ["ssh", MAC4_HOST, f"mkdir -p {MAC4_TRAINING_DIR}"],
        check=True, timeout=10,
    )

    # SCP training data
    for fname in ["train.jsonl", "valid.jsonl"]:
        local_path = Path(training_data_dir) / fname
        if local_path.exists():
            subprocess.run(
                ["scp", str(local_path), f"{MAC4_HOST}:{MAC4_TRAINING_DIR}/{fname}"],
                check=True, timeout=30,
            )

    return {"deployed": True, "remote_dir": MAC4_TRAINING_DIR}


def launch_training(dry_run=False):
    """Launch MLX LoRA training on Mac4."""
    cmd = (
        f"cd {MAC4_TRAINING_DIR} && "
        f"nohup {MAC4_VENV_PYTHON} -m mlx_lm.lora "
        f"--model {BASE_MODEL} "
        f"--train "
        f"--data {MAC4_TRAINING_DIR} "
        f"--lora-layers {LORA_LAYERS} "
        f"--lora-rank {LORA_RANK} "
        f"--learning-rate {LEARNING_RATE} "
        f"--batch-size {BATCH_SIZE} "
        f"--iters {EPOCHS * 100} "
        f"--val-batches 5 "
        f"--save-every 50 "
        f"--adapter-path {MAC4_TRAINING_DIR}/adapters "
        f"> {MAC4_TRAINING_DIR}/training.log 2>&1 &"
    )

    result = {
        "command": cmd,
        "model": BASE_MODEL,
        "lora_rank": LORA_RANK,
        "lora_layers": LORA_LAYERS,
        "learning_rate": LEARNING_RATE,
        "dry_run": dry_run,
    }

    if not dry_run:
        out = subprocess.run(
            ["ssh", MAC4_HOST, cmd],
            capture_output=True, text=True, timeout=30,
        )
        result["launched"] = out.returncode == 0
        if out.stderr:
            result["stderr"] = out.stderr[:500]

        # Log training run
        log_entry = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "model": BASE_MODEL,
            "host": "mac4",
            "status": "launched" if out.returncode == 0 else "failed",
        }
        with open(TRAINING_LOG, "a") as f:
            f.write(json.dumps(log_entry, default=str) + "\n")

    return result


def check_training_status():
    """Check if training is running on Mac4."""
    result = {"running": False, "log_tail": ""}

    try:
        # Check process
        out = subprocess.run(
            ["ssh", MAC4_HOST, "pgrep -f 'mlx_lm.lora' > /dev/null && echo RUNNING || echo STOPPED"],
            capture_output=True, text=True, timeout=10,
        )
        result["running"] = "RUNNING" in out.stdout

        # Tail log
        out = subprocess.run(
            ["ssh", MAC4_HOST, f"tail -20 {MAC4_TRAINING_DIR}/training.log 2>/dev/null || echo 'No log file'"],
            capture_output=True, text=True, timeout=10,
        )
        result["log_tail"] = out.stdout.strip()

        # Check adapters
        out = subprocess.run(
            ["ssh", MAC4_HOST, f"ls -la {MAC4_TRAINING_DIR}/adapters/ 2>/dev/null | tail -5 || echo 'No adapters yet'"],
            capture_output=True, text=True, timeout=10,
        )
        result["adapters"] = out.stdout.strip()

    except Exception as e:
        result["error"] = str(e)

    return result


def main():
    if "--status" in sys.argv:
        result = check_training_status()
        print(json.dumps(result, indent=2))
        return

    if "--dry-run" in sys.argv or "--launch" in sys.argv or "--prepare-only" in sys.argv:
        dry_run = "--dry-run" in sys.argv
        prepare_only = "--prepare-only" in sys.argv

        # Find SFT export
        sft_path = find_latest_sft_export()
        if not sft_path:
            print("ERROR: No SFT export found in", SFT_DIR)
            sys.exit(1)
        print(f"SFT export: {sft_path.name}")

        # Validate
        validation = validate_sft_data(sft_path)
        print(f"Validation: {validation['total']} records ({validation['usable']} usable, {validation['skipped']} skipped), format={validation['format']}")
        if not validation["valid"]:
            print(f"Hard errors: {validation['issues']}")
            sys.exit(1)
        if validation["warnings"]:
            print(f"  Warnings: {len(validation['warnings'])} (empty records will be skipped)")
        if validation["usable"] == 0:
            print("ERROR: No usable records after filtering")
            sys.exit(1)
        print(f"  Weights: avg={validation['avg_weight']}, range=[{validation['min_weight']}, {validation['max_weight']}]")

        # Prepare train/valid split
        prep = prepare_training_data(sft_path)
        print(f"Split: {prep['train_count']} train, {prep['valid_count']} valid")

        if prepare_only:
            print(f"Data prepared at: {prep['output_dir']}")
            return

        if dry_run:
            print("\n--- DRY RUN ---")
            print(f"Would deploy to: {MAC4_HOST}:{MAC4_TRAINING_DIR}")
            launch_result = launch_training(dry_run=True)
            print(f"Model: {launch_result['model']}")
            print(f"LoRA: rank={launch_result['lora_rank']}, layers={launch_result['lora_layers']}")
            print(f"LR: {launch_result['learning_rate']}")
            return

        # Check Mac4 readiness
        mac4 = check_mac4_ready()
        print(f"Mac4: reachable={mac4['reachable']}, mlx-lm={mac4['mlx_lm']}, disk={mac4['disk_gb']}GB")
        if not mac4["reachable"]:
            print("ERROR: Mac4 unreachable")
            sys.exit(1)
        if not mac4["mlx_lm"]:
            print("ERROR: mlx-lm not installed on Mac4. Run: ssh mac4 'pip3 install mlx-lm'")
            sys.exit(1)

        # Deploy
        deploy = deploy_to_mac4(prep["output_dir"])
        print(f"Deployed to: {deploy['remote_dir']}")

        # Launch
        result = launch_training(dry_run=False)
        print(f"Training launched: {result.get('launched', False)}")
        if result.get("stderr"):
            print(f"Stderr: {result['stderr']}")

    else:
        print(__doc__)


if __name__ == "__main__":
    main()
