"""
trainer.py - KARL training pipeline: local export -> remote LoRA fine-tune.

Orchestrates the full training cycle:
  1. Export trajectories to advantage-weighted SFT (sft_exporter)
  2. Upload training data to remote compute node via SCP
  3. Trigger MLX LoRA training
  4. Monitor training progress
  5. Report results

Usage:
    from karl.trainer import full_training_cycle
    result = full_training_cycle()
    result = full_training_cycle(dry_run=True)
"""

import json
import subprocess
import time
import urllib.request
from pathlib import Path
from typing import Any, Dict, Optional

from karl.config import (
    TRAIN_PATH,
    VALID_PATH,
    TRAIN_SSH_ALIAS,
    TRAIN_HOST,
    TRAIN_DAEMON_PORT,
    TRAIN_REMOTE_DIR,
    TRAIN_MERGED_DIR,
    MLX_MODEL,
    MLX_ITERS,
    MLX_BATCH_SIZE,
    MLX_NUM_LAYERS,
    MLX_MAX_SEQ_LEN,
    MLX_LR,
)
from karl.sft_exporter import export_sft


def _ssh_cmd(cmd: str, timeout: int = 30) -> tuple:
    """Execute command on remote compute node via SSH alias."""
    full_cmd = f"ssh {TRAIN_SSH_ALIAS} '{cmd}'"
    try:
        result = subprocess.run(
            full_cmd, shell=True, capture_output=True, text=True, timeout=timeout
        )
        return result.returncode, result.stdout + result.stderr
    except subprocess.TimeoutExpired:
        return 1, "SSH timeout"
    except Exception as e:
        return 1, str(e)


def _scp_file(local_path: Path, remote_path: str) -> bool:
    """Copy file to remote node via SCP."""
    full_remote = f"{TRAIN_SSH_ALIAS}:{remote_path}"
    try:
        result = subprocess.run(
            ["scp", str(local_path), full_remote],
            capture_output=True, text=True, timeout=30,
        )
        return result.returncode == 0
    except Exception:
        return False


def check_daemon_health() -> Optional[Dict]:
    """Check remote finetune daemon health."""
    daemon_url = f"http://{TRAIN_HOST}:{TRAIN_DAEMON_PORT}"
    try:
        req = urllib.request.Request(f"{daemon_url}/status")
        with urllib.request.urlopen(req, timeout=5) as resp:
            return json.loads(resp.read())
    except Exception:
        return None


def export_training_data(min_reward: float = 0.0) -> Dict[str, Any]:
    """Step 1: Export trajectories to SFT format."""
    return export_sft(min_reward=min_reward)


def upload_training_data() -> Dict[str, Any]:
    """Step 2: Upload training data to remote node."""
    if not TRAIN_PATH.exists() or not VALID_PATH.exists():
        return {"success": False, "error": "No training data files"}

    rc, out = _ssh_cmd(f"mkdir -p {TRAIN_REMOTE_DIR}")
    if rc != 0:
        return {"success": False, "error": f"mkdir failed: {out}"}

    train_ok = _scp_file(TRAIN_PATH, f"{TRAIN_REMOTE_DIR}/train.jsonl")
    valid_ok = _scp_file(VALID_PATH, f"{TRAIN_REMOTE_DIR}/valid.jsonl")

    if not train_ok or not valid_ok:
        return {"success": False, "error": "SCP upload failed"}

    _ssh_cmd(
        f"cp {TRAIN_REMOTE_DIR}/train.jsonl {TRAIN_MERGED_DIR}/train.jsonl && "
        f"cp {TRAIN_REMOTE_DIR}/valid.jsonl {TRAIN_MERGED_DIR}/valid.jsonl"
    )

    rc, out = _ssh_cmd(f"wc -l {TRAIN_REMOTE_DIR}/train.jsonl {TRAIN_REMOTE_DIR}/valid.jsonl")

    return {
        "success": True,
        "train_uploaded": train_ok,
        "valid_uploaded": valid_ok,
        "remote_info": out.strip(),
    }


def trigger_training() -> Dict[str, Any]:
    """Step 3: Trigger MLX LoRA training on remote node."""
    status = check_daemon_health()

    if status:
        rc, out = _ssh_cmd(
            f"python3 ~/Desktop/homelab/compute-pair/finetune-daemon.py --trigger-now",
            timeout=300,
        )
        return {
            "method": "daemon_trigger",
            "exit_code": rc,
            "output": out[:1000],
            "daemon_status": status,
        }

    # Daemon not running -- train directly
    rc, out = _ssh_cmd(
        f"python3 -m mlx_lm lora "
        f"--model {MLX_MODEL} "
        f"--data {TRAIN_REMOTE_DIR} "
        f"--adapter-path ~/adapters/latest/karl-latest "
        f"--train "
        f"--iters {MLX_ITERS} "
        f"--batch-size {MLX_BATCH_SIZE} "
        f"--num-layers {MLX_NUM_LAYERS} "
        f"--max-seq-length {MLX_MAX_SEQ_LEN} "
        f"--learning-rate {MLX_LR} "
        f"2>&1 | tail -20",
        timeout=600,
    )
    return {
        "method": "direct_mlx",
        "exit_code": rc,
        "output": out[:2000],
    }


def monitor_training(poll_interval: int = 15, max_polls: int = 40) -> Dict[str, Any]:
    """Step 4: Monitor training progress via daemon metrics."""
    for i in range(max_polls):
        status = check_daemon_health()
        if not status:
            time.sleep(poll_interval)
            continue

        metrics = status.get("metrics", {})
        loss = metrics.get("finetune_loss", 0)
        iters = metrics.get("finetune_iterations_total", 0)
        version = metrics.get("finetune_adapter_version", 0)

        if loss > 0 and iters > 0:
            return {
                "complete": True,
                "loss": loss,
                "iterations": iters,
                "adapter_version": version,
                "polls": i + 1,
            }

        time.sleep(poll_interval)

    return {"complete": False, "polls": max_polls, "message": "Timed out waiting"}


def full_training_cycle(
    min_reward: float = 0.0,
    dry_run: bool = False,
) -> Dict[str, Any]:
    """Run the complete KARL training pipeline.

    Args:
        min_reward: Minimum reward score for SFT export
        dry_run: Preview without training

    Returns:
        Results dict with per-step outputs and overall status
    """
    results: Dict[str, Any] = {"steps": {}}

    # Step 1: Export
    export_stats = export_training_data(min_reward=min_reward)
    results["steps"]["export"] = export_stats

    if export_stats.get("examples", 0) == 0:
        results["status"] = "no_data"
        return results

    if dry_run:
        results["status"] = "dry_run"
        return results

    # Step 2: Upload
    upload_result = upload_training_data()
    results["steps"]["upload"] = upload_result
    if not upload_result.get("success"):
        results["status"] = "upload_failed"
        return results

    # Step 3: Trigger
    trigger_result = trigger_training()
    results["steps"]["trigger"] = trigger_result

    if trigger_result.get("exit_code", 1) != 0:
        results["status"] = "trigger_failed"
        return results

    # Step 4: Monitor
    if trigger_result.get("method") == "daemon_trigger":
        monitor_result = monitor_training()
        results["steps"]["monitor"] = monitor_result

    results["status"] = "complete"
    return results
