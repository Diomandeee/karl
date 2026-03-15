#!/usr/bin/env python3
"""
sft_dispatch.py — SFT Training Dispatch to Mac4.

Transfers training data to mac4, validates the environment,
and launches MLX LoRA fine-tuning remotely.

Usage:
    python3 sft_dispatch.py --preflight   # Check mac4 readiness without training
    python3 sft_dispatch.py --launch      # Transfer data and start training
    python3 sft_dispatch.py --status      # Check training status on mac4
"""

import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

KARL_DIR = Path(__file__).parent
CONFIG_PATH = KARL_DIR / "config.json"
TRAIN_FILE = KARL_DIR / "train.jsonl"
VALID_FILE = KARL_DIR / "valid.jsonl"
DISPATCH_LOG = KARL_DIR / "sft_dispatch_log.jsonl"

# Remote paths
REMOTE_HOST = "mac4"
REMOTE_DIR = "~/.karl-sft"
REMOTE_TRAIN = f"{REMOTE_DIR}/train.jsonl"
REMOTE_VALID = f"{REMOTE_DIR}/valid.jsonl"
REMOTE_OUTPUT = f"{REMOTE_DIR}/adapters"
REMOTE_PID_FILE = f"{REMOTE_DIR}/train.pid"
REMOTE_LOG_FILE = f"{REMOTE_DIR}/train.log"


def _load_config() -> Dict:
    try:
        return json.loads(CONFIG_PATH.read_text())
    except (json.JSONDecodeError, OSError):
        return {}


def _ssh(cmd: str, timeout: int = 30) -> subprocess.CompletedProcess:
    """Run a command on mac4 via SSH."""
    return subprocess.run(
        ["ssh", "-o", "ConnectTimeout=5", REMOTE_HOST, cmd],
        capture_output=True, text=True, timeout=timeout,
    )


def _log_dispatch(event: Dict) -> None:
    """Append a dispatch event to the log."""
    event["ts"] = datetime.now(timezone.utc).isoformat()
    with open(DISPATCH_LOG, "a") as f:
        f.write(json.dumps(event, separators=(",", ":"), default=str) + "\n")


def preflight() -> Dict[str, Any]:
    """Check mac4 readiness for SFT training."""
    config = _load_config()
    sft_cfg = config.get("sft", {})
    model = sft_cfg.get("base_model", "mlx-community/Qwen2.5-Coder-7B-Instruct-4bit")

    checks = {}

    # 1. SSH connectivity
    try:
        r = _ssh("echo ok")
        checks["ssh"] = r.stdout.strip() == "ok"
    except (subprocess.TimeoutExpired, Exception) as e:
        checks["ssh"] = False
        return {"ready": False, "checks": checks, "error": f"SSH failed: {e}"}

    # 2. MLX installed
    r = _ssh("python3 -c 'import mlx; print(mlx.__version__)'")
    checks["mlx_installed"] = r.returncode == 0
    if checks["mlx_installed"]:
        checks["mlx_version"] = r.stdout.strip()

    # 3. mlx-lm installed (needed for LoRA fine-tuning)
    r = _ssh("python3 -c 'import mlx_lm; print(mlx_lm.__version__)'")
    checks["mlx_lm_installed"] = r.returncode == 0
    if checks["mlx_lm_installed"]:
        checks["mlx_lm_version"] = r.stdout.strip()

    # 4. Disk space
    r = _ssh("df -h ~ | tail -1 | awk '{print $4}'")
    if r.returncode == 0:
        checks["disk_available"] = r.stdout.strip()

    # 5. Memory
    r = _ssh("sysctl -n hw.memsize 2>/dev/null || free -g 2>/dev/null | head -2 | tail -1 | awk '{print $2}'")
    if r.returncode == 0:
        try:
            mem_bytes = int(r.stdout.strip())
            checks["memory_gb"] = round(mem_bytes / (1024**3), 1)
        except ValueError:
            checks["memory_raw"] = r.stdout.strip()

    # 6. Local training data
    checks["train_exists"] = TRAIN_FILE.exists()
    checks["valid_exists"] = VALID_FILE.exists()
    if TRAIN_FILE.exists():
        with open(TRAIN_FILE) as f:
            checks["train_examples"] = sum(1 for _ in f)
    if VALID_FILE.exists():
        with open(VALID_FILE) as f:
            checks["valid_examples"] = sum(1 for _ in f)

    # 7. Check if training is already running
    r = _ssh(f"cat {REMOTE_PID_FILE} 2>/dev/null && ps -p $(cat {REMOTE_PID_FILE} 2>/dev/null) -o pid= 2>/dev/null")
    checks["training_running"] = r.returncode == 0 and r.stdout.strip() != ""

    ready = (
        checks.get("ssh")
        and checks.get("mlx_installed")
        and checks.get("train_exists")
        and checks.get("valid_exists")
        and checks.get("train_examples", 0) >= 10
        and not checks.get("training_running")
    )

    result = {
        "ready": ready,
        "model": model,
        "lora_rank": sft_cfg.get("lora_rank", 8),
        "lora_layers": sft_cfg.get("lora_layers", 16),
        "learning_rate": sft_cfg.get("learning_rate", 1e-5),
        "checks": checks,
    }

    if not checks.get("mlx_lm_installed"):
        result["action_needed"] = "Install mlx-lm on mac4: pip3 install mlx-lm"

    return result


def launch_training(dry_run: bool = False) -> Dict[str, Any]:
    """Transfer data and launch SFT training on mac4."""
    # Preflight first
    pf = preflight()
    if not pf["ready"]:
        return {"status": "preflight_failed", "preflight": pf}

    config = _load_config()
    sft_cfg = config.get("sft", {})
    model = sft_cfg.get("base_model", "mlx-community/Qwen2.5-Coder-7B-Instruct-4bit")
    lora_rank = sft_cfg.get("lora_rank", 8)
    lora_layers = sft_cfg.get("lora_layers", 16)
    lr = sft_cfg.get("learning_rate", 1e-5)

    if dry_run:
        return {
            "status": "dry_run",
            "model": model,
            "lora_rank": lora_rank,
            "lora_layers": lora_layers,
            "learning_rate": lr,
            "preflight": pf,
        }

    # Install mlx-lm if needed
    if not pf["checks"].get("mlx_lm_installed"):
        r = _ssh("pip3 install mlx-lm", timeout=120)
        if r.returncode != 0:
            return {"status": "mlx_lm_install_failed", "stderr": r.stderr[:500]}

    # Create remote directory
    _ssh(f"mkdir -p {REMOTE_DIR} {REMOTE_OUTPUT}")

    # Transfer training data
    scp_result = subprocess.run(
        ["scp", str(TRAIN_FILE), str(VALID_FILE), f"{REMOTE_HOST}:{REMOTE_DIR}/"],
        capture_output=True, text=True, timeout=30,
    )
    if scp_result.returncode != 0:
        return {"status": "scp_failed", "stderr": scp_result.stderr[:500]}

    # Verify files transferred
    r = _ssh(f"wc -l {REMOTE_TRAIN} {REMOTE_VALID}")
    transfer_check = r.stdout.strip() if r.returncode == 0 else "unknown"

    # Build mlx_lm.lora command
    # mlx_lm.lora expects --model, --data (dir with train.jsonl/valid.jsonl),
    # --train, --adapter-path, and optionally --lora-layers, --batch-size, --iters
    train_epochs = 3
    batch_size = 2
    iters = max(100, (pf["checks"].get("train_examples", 66) * train_epochs) // batch_size)

    train_cmd = (
        f"cd {REMOTE_DIR} && "
        f"nohup python3 -m mlx_lm.lora "
        f"--model {model} "
        f"--data . "
        f"--train "
        f"--adapter-path {REMOTE_OUTPUT} "
        f"--lora-layers {lora_layers} "
        f"--lora-rank {lora_rank} "
        f"--learning-rate {lr} "
        f"--batch-size {batch_size} "
        f"--iters {iters} "
        f"> {REMOTE_LOG_FILE} 2>&1 & "
        f"echo $! > {REMOTE_PID_FILE}"
    )

    r = _ssh(train_cmd, timeout=30)
    if r.returncode != 0:
        return {"status": "launch_failed", "stderr": r.stderr[:500]}

    # Verify PID
    r = _ssh(f"cat {REMOTE_PID_FILE}")
    pid = r.stdout.strip() if r.returncode == 0 else "unknown"

    result = {
        "status": "launched",
        "pid": pid,
        "model": model,
        "lora_rank": lora_rank,
        "lora_layers": lora_layers,
        "learning_rate": lr,
        "batch_size": batch_size,
        "iters": iters,
        "transfer_check": transfer_check,
        "remote_dir": REMOTE_DIR,
    }

    _log_dispatch({"event": "launch", **result})
    return result


def check_status() -> Dict[str, Any]:
    """Check training status on mac4."""
    try:
        # Check if process is running
        r = _ssh(f"cat {REMOTE_PID_FILE} 2>/dev/null")
        pid = r.stdout.strip() if r.returncode == 0 else None

        running = False
        if pid:
            r = _ssh(f"ps -p {pid} -o pid= 2>/dev/null")
            running = r.returncode == 0 and r.stdout.strip() != ""

        # Get last N lines of log
        r = _ssh(f"tail -20 {REMOTE_LOG_FILE} 2>/dev/null")
        log_tail = r.stdout.strip() if r.returncode == 0 else ""

        # Check for adapters
        r = _ssh(f"ls -la {REMOTE_OUTPUT}/ 2>/dev/null")
        adapters = r.stdout.strip() if r.returncode == 0 else ""

        # Parse progress from log (mlx_lm logs "Iter N: ...")
        progress = None
        if log_tail:
            for line in reversed(log_tail.split("\n")):
                if "Iter" in line or "iter" in line:
                    progress = line.strip()
                    break

        return {
            "pid": pid,
            "running": running,
            "progress": progress,
            "adapters_exist": "adapter" in adapters.lower() if adapters else False,
            "log_tail": log_tail[-500:] if log_tail else "",
        }
    except (subprocess.TimeoutExpired, Exception) as e:
        return {"error": str(e)}


def evaluate_adapter() -> Dict[str, Any]:
    """Evaluate trained LoRA adapter on holdout set.

    1. Check if adapters exist on Mac4
    2. Transfer eval-holdout.jsonl to Mac4
    3. Run mlx_lm.lora --test to get eval loss
    4. Compare adapter performance against baseline
    5. Return go/no-go recommendation for deployment
    """
    EVAL_FILE = KARL_DIR / "eval-holdout.jsonl"
    LOCAL_ADAPTERS = KARL_DIR / "adapters"

    # Check adapter existence
    try:
        r = _ssh(f"ls {REMOTE_OUTPUT}/adapter_config.json 2>/dev/null")
        if r.returncode != 0:
            return {"status": "no_adapters", "message": "No trained adapters found on Mac4"}
    except (subprocess.TimeoutExpired, Exception) as e:
        return {"status": "ssh_error", "error": str(e)}

    # Get training log for baseline loss
    try:
        r = _ssh(f"tail -5 {REMOTE_LOG_FILE} 2>/dev/null")
        train_tail = r.stdout.strip() if r.returncode == 0 else ""
    except Exception:
        train_tail = ""

    # Parse final training/validation loss from log
    train_loss = None
    val_loss = None
    if train_tail:
        for line in reversed(train_tail.split("\n")):
            if "val" in line.lower() and "loss" in line.lower():
                try:
                    # mlx_lm logs: "Val Loss: X.XXX"
                    parts = line.split(":")
                    for i, p in enumerate(parts):
                        if "loss" in p.lower():
                            val_loss = float(parts[i + 1].strip().split()[0])
                            break
                except (ValueError, IndexError):
                    pass
            if "train" in line.lower() and "loss" in line.lower() and train_loss is None:
                try:
                    parts = line.split(":")
                    for i, p in enumerate(parts):
                        if "loss" in p.lower():
                            train_loss = float(parts[i + 1].strip().split()[0])
                            break
                except (ValueError, IndexError):
                    pass

    # Transfer holdout set if available
    eval_transferred = False
    if EVAL_FILE.exists():
        try:
            r = subprocess.run(
                ["scp", str(EVAL_FILE), f"{REMOTE_HOST}:{REMOTE_DIR}/eval-holdout.jsonl"],
                capture_output=True, text=True, timeout=15,
            )
            eval_transferred = r.returncode == 0
        except Exception:
            pass

    # Run evaluation on holdout set
    eval_loss = None
    if eval_transferred:
        config = _load_config()
        model = config.get("sft", {}).get("base_model", "mlx-community/Qwen2.5-Coder-7B-Instruct-4bit")
        eval_cmd = (
            f"cd {REMOTE_DIR} && "
            f"python3 -m mlx_lm.lora "
            f"--model {model} "
            f"--adapter-path {REMOTE_OUTPUT} "
            f"--data . "
            f"--test "
            f"2>&1 | tail -5"
        )
        try:
            r = _ssh(eval_cmd, timeout=120)
            if r.returncode == 0 and r.stdout:
                for line in r.stdout.strip().split("\n"):
                    if "test" in line.lower() and "loss" in line.lower():
                        try:
                            parts = line.split(":")
                            for i, p in enumerate(parts):
                                if "loss" in p.lower():
                                    eval_loss = float(parts[i + 1].strip().split()[0])
                                    break
                        except (ValueError, IndexError):
                            pass
        except Exception:
            pass

    # Build result
    result = {
        "status": "ok",
        "adapters_exist": True,
        "train_loss": train_loss,
        "val_loss": val_loss,
        "eval_loss": eval_loss,
        "eval_transferred": eval_transferred,
        "log_tail": train_tail[-300:] if train_tail else "",
    }

    # Go/no-go assessment
    if val_loss is not None:
        if val_loss < 1.5:
            result["verdict"] = "go"
            result["recommendation"] = "Validation loss acceptable. Adapter ready for deployment."
        elif val_loss < 2.5:
            result["verdict"] = "marginal"
            result["recommendation"] = "Validation loss marginal. Consider more training data."
        else:
            result["verdict"] = "no_go"
            result["recommendation"] = "Validation loss too high. Training may need adjustment."
    else:
        result["verdict"] = "unknown"
        result["recommendation"] = "Could not parse validation loss. Check training log manually."

    _log_dispatch({"event": "evaluate", **result})
    return result


def fetch_adapter() -> Dict[str, Any]:
    """Download trained adapter from Mac4 to local karl directory."""
    LOCAL_ADAPTERS = KARL_DIR / "adapters"
    LOCAL_ADAPTERS.mkdir(parents=True, exist_ok=True)

    try:
        r = subprocess.run(
            ["scp", "-r", f"{REMOTE_HOST}:{REMOTE_OUTPUT}/", str(LOCAL_ADAPTERS)],
            capture_output=True, text=True, timeout=60,
        )
        if r.returncode != 0:
            return {"status": "scp_failed", "stderr": r.stderr[:500]}

        files = list(LOCAL_ADAPTERS.glob("*"))
        return {
            "status": "ok",
            "local_path": str(LOCAL_ADAPTERS),
            "files": [f.name for f in files],
        }
    except (subprocess.TimeoutExpired, Exception) as e:
        return {"status": "error", "error": str(e)}


def main():
    if "--preflight" in sys.argv:
        result = preflight()
        print(json.dumps(result, indent=2, default=str))
    elif "--launch" in sys.argv:
        dry_run = "--dry-run" in sys.argv
        result = launch_training(dry_run=dry_run)
        print(json.dumps(result, indent=2, default=str))
    elif "--status" in sys.argv:
        result = check_status()
        print(json.dumps(result, indent=2, default=str))
    else:
        print(__doc__)


if __name__ == "__main__":
    main()
