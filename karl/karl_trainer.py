#!/usr/bin/env python3
"""
karl_trainer.py — KARL training pipeline: Mac1 → Mac5 LoRA fine-tune.

Orchestrates the full training cycle:
  1. Export trajectories to advantage-weighted SFT (sft_exporter)
  2. Upload training data to Mac5 via SCP
  3. Trigger MLX LoRA training via finetune-daemon --trigger-now
  4. Monitor training progress via :9200/status
  5. Report results (adapter version, loss, examples)

Usage:
    python3 karl_trainer.py                # Full training cycle
    python3 karl_trainer.py --dry-run      # Preview without training
    python3 karl_trainer.py --status       # Check Mac5 daemon status
    python3 karl_trainer.py --export-only  # Export SFT only, don't train
"""

import json
import os
import subprocess
import sys
import time
import urllib.request
from pathlib import Path
from typing import Any, Dict, Optional

KARL_DIR = Path(__file__).parent
sys.path.insert(0, str(KARL_DIR))

MAC5_HOST = "100.109.94.124"
MAC5_USER = "mohameddiomande"
MAC5_SSH_ALIAS = "mac5"  # Uses ~/.ssh/config with multiplexing
MAC5_DAEMON_URL = f"http://{MAC5_HOST}:9200"
MAC5_TRAINING_DIR = "~/Desktop/homelab/compute-pair/karl-training"
MAC5_MERGED_DIR = "~/Desktop/homelab/compute-pair/merged-training"
MAC5_DAEMON_SCRIPT = "~/Desktop/homelab/compute-pair/finetune-daemon.py"

# Local paths
TRAIN_PATH = KARL_DIR / "train.jsonl"
VALID_PATH = KARL_DIR / "valid.jsonl"


def _ssh_cmd(cmd: str, timeout: int = 30) -> tuple[int, str]:
    """Execute command on Mac5 via SSH alias (uses multiplexed connection)."""
    full_cmd = f"ssh {MAC5_SSH_ALIAS} '{cmd}'"
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
    """Copy file to Mac5 via SCP (uses SSH alias for multiplexing)."""
    full_remote = f"{MAC5_SSH_ALIAS}:{remote_path}"
    try:
        result = subprocess.run(
            ["scp", str(local_path), full_remote],
            capture_output=True, text=True, timeout=30,
        )
        return result.returncode == 0
    except Exception:
        return False


def _check_daemon_health() -> Optional[Dict]:
    """Check Mac5 finetune daemon health and status."""
    try:
        req = urllib.request.Request(f"{MAC5_DAEMON_URL}/status")
        with urllib.request.urlopen(req, timeout=5) as resp:
            return json.loads(resp.read())
    except Exception:
        return None


def export_training_data(min_reward: float = 0.0) -> Dict[str, Any]:
    """Step 1: Export trajectories to SFT format."""
    from sft_exporter import export_sft
    return export_sft(min_reward=min_reward)


def upload_to_mac5() -> Dict[str, Any]:
    """Step 2: Upload training data to Mac5."""
    if not TRAIN_PATH.exists() or not VALID_PATH.exists():
        return {"success": False, "error": "No training data files"}

    # Ensure remote directory exists
    rc, out = _ssh_cmd(f"mkdir -p {MAC5_TRAINING_DIR}")
    if rc != 0:
        return {"success": False, "error": f"mkdir failed: {out}"}

    # Upload train and valid files
    train_ok = _scp_file(TRAIN_PATH, f"{MAC5_TRAINING_DIR}/train.jsonl")
    valid_ok = _scp_file(VALID_PATH, f"{MAC5_TRAINING_DIR}/valid.jsonl")

    if not train_ok or not valid_ok:
        return {"success": False, "error": "SCP upload failed"}

    # Also copy to merged-training (where daemon looks)
    rc, _ = _ssh_cmd(
        f"cp {MAC5_TRAINING_DIR}/train.jsonl {MAC5_MERGED_DIR}/train.jsonl && "
        f"cp {MAC5_TRAINING_DIR}/valid.jsonl {MAC5_MERGED_DIR}/valid.jsonl"
    )

    # Count lines on remote
    rc, out = _ssh_cmd(f"wc -l {MAC5_TRAINING_DIR}/train.jsonl {MAC5_TRAINING_DIR}/valid.jsonl")

    return {
        "success": True,
        "train_uploaded": train_ok,
        "valid_uploaded": valid_ok,
        "remote_info": out.strip(),
    }


def trigger_training() -> Dict[str, Any]:
    """Step 3: Trigger MLX LoRA training on Mac5."""
    # First check daemon health
    status = _check_daemon_health()

    if status:
        # Daemon is running — trigger via CLI
        rc, out = _ssh_cmd(
            f"python3 {MAC5_DAEMON_SCRIPT} --trigger-now",
            timeout=300,  # Training can take several minutes
        )
        return {
            "method": "daemon_trigger",
            "exit_code": rc,
            "output": out[:1000],
            "daemon_status": status,
        }

    # Daemon not running — trigger training directly
    rc, out = _ssh_cmd(
        f"python3 -m mlx_lm lora "
        f"--model mlx-community/gemma-3-1b-it-4bit "
        f"--data {MAC5_TRAINING_DIR} "
        f"--adapter-path ~/adapters/latest/karl-v1 "
        f"--train "
        f"--iters 500 "
        f"--batch-size 1 "
        f"--num-layers 4 "
        f"--max-seq-length 256 "
        f"--learning-rate 1e-5 "
        f"2>&1 | tail -20",
        timeout=600,  # 10 min timeout for direct training
    )
    return {
        "method": "direct_mlx",
        "exit_code": rc,
        "output": out[:2000],
    }


def monitor_training(poll_interval: int = 15, max_polls: int = 40) -> Dict[str, Any]:
    """Step 4: Monitor training progress via daemon metrics."""
    for i in range(max_polls):
        status = _check_daemon_health()
        if not status:
            time.sleep(poll_interval)
            continue

        # Check if training is complete
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
    """Run the complete KARL training pipeline."""
    results = {"steps": {}}

    # Step 1: Export
    print("[karl] Step 1: Exporting trajectories to SFT...")
    export_stats = export_training_data(min_reward=min_reward)
    results["steps"]["export"] = export_stats
    print(f"  {export_stats.get('examples', 0)} examples "
          f"({export_stats.get('train', 0)} train, {export_stats.get('valid', 0)} valid)")

    if export_stats.get("examples", 0) == 0:
        results["status"] = "no_data"
        return results

    if dry_run:
        results["status"] = "dry_run"
        print("[karl] Dry run — skipping upload and training")
        return results

    # Step 2: Upload
    print("[karl] Step 2: Uploading to Mac5...")
    upload_result = upload_to_mac5()
    results["steps"]["upload"] = upload_result
    if not upload_result.get("success"):
        results["status"] = "upload_failed"
        print(f"  FAILED: {upload_result.get('error')}")
        return results
    print(f"  {upload_result.get('remote_info', 'OK')}")

    # Step 3: Trigger
    print("[karl] Step 3: Triggering MLX LoRA training...")
    trigger_result = trigger_training()
    results["steps"]["trigger"] = trigger_result
    print(f"  Method: {trigger_result.get('method')}, "
          f"exit: {trigger_result.get('exit_code')}")

    if trigger_result.get("exit_code", 1) != 0:
        results["status"] = "trigger_failed"
        print(f"  Output: {trigger_result.get('output', '')[:200]}")
        return results

    # Step 4: Monitor (only if daemon-triggered)
    if trigger_result.get("method") == "daemon_trigger":
        print("[karl] Step 4: Monitoring training progress...")
        monitor_result = monitor_training()
        results["steps"]["monitor"] = monitor_result
        if monitor_result.get("complete"):
            print(f"  Complete! Loss: {monitor_result.get('loss')}, "
                  f"Adapter: v{monitor_result.get('adapter_version')}")
        else:
            print(f"  {monitor_result.get('message', 'Unknown')}")

    results["status"] = "complete"
    return results


if __name__ == "__main__":
    if "--status" in sys.argv:
        status = _check_daemon_health()
        if status:
            print(f"Mac5 daemon: ONLINE")
            print(json.dumps(status, indent=2))
        else:
            print("Mac5 daemon: OFFLINE or unreachable")
            # Try SSH health check
            rc, out = _ssh_cmd("curl -s http://localhost:9200/health 2>/dev/null || echo 'down'")
            print(f"  SSH check: {out.strip()}")
    elif "--export-only" in sys.argv:
        stats = export_training_data()
        print(json.dumps(stats, indent=2))
    elif "--dry-run" in sys.argv:
        result = full_training_cycle(dry_run=True)
        print(f"\n{json.dumps(result, indent=2)}")
    else:
        result = full_training_cycle()
        print(f"\n{json.dumps(result, indent=2)}")
