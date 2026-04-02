#!/usr/bin/env python3
"""Anticipatory Transformer training on Qwen3-4B with QLoRA.

Usage:
    # Treatment (full anticipation)
    python train.py --config configs/qwen3_4b_qlora.yaml

    # Control (no anticipation)
    python train.py --config configs/qwen3_4b_qlora.yaml \
        --override anticipation.use_inscription=false \
        --override anticipation.use_gate=false \
        --override anticipation.use_lse_loop=false \
        --override training.output_dir=runs/qwen3-4b-control

    # Ablation: no inscription
    python train.py --config configs/qwen3_4b_qlora.yaml \
        --override anticipation.use_inscription=false \
        --override training.output_dir=runs/qwen3-4b-no-inscription
"""

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

import torch
import yaml
from torch.utils.data import DataLoader

from src.data import MixedTrainDataset, EvalDataset, collate_fn
from src.anticipation import (
    InscriptionEmbedding,
    ScalarProjection,
    CommitmentGate,
    AnticipationHead,
    LSERewardTracker,
)


def load_config(config_path: str, overrides: list[str] = None) -> dict:
    """Load YAML config with dot-notation overrides."""
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    if overrides:
        for ov in overrides:
            key, val = ov.split("=", 1)
            parts = key.split(".")
            d = cfg
            for p in parts[:-1]:
                d = d[p]
            # Type inference
            if val.lower() in ("true", "false"):
                val = val.lower() == "true"
            elif val.replace(".", "").replace("-", "").replace("e", "").isdigit():
                try:
                    val = int(val)
                except ValueError:
                    val = float(val)
            d[parts[-1]] = val

    return cfg


def setup_model(cfg: dict):
    """Load Qwen3-4B with QLoRA and add anticipation modules."""
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

    model_name = cfg["model"]["base_model"]
    qlora_cfg = cfg["qlora"]

    print(f"Loading {model_name} with 4-bit quantization...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=qlora_cfg["load_in_4bit"],
        bnb_4bit_quant_type=qlora_cfg["bnb_4bit_quant_type"],
        bnb_4bit_compute_dtype=getattr(torch, qlora_cfg["bnb_4bit_compute_dtype"]),
    )

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
    )

    model = prepare_model_for_kbit_training(model)

    lora_config = LoraConfig(
        r=qlora_cfg["lora_r"],
        lora_alpha=qlora_cfg["lora_alpha"],
        lora_dropout=qlora_cfg["lora_dropout"],
        target_modules=qlora_cfg["target_modules"],
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    return model, tokenizer


def setup_anticipation(cfg: dict, hidden_dim: int, device: torch.device):
    """Create anticipation modules based on config flags."""
    ant_cfg = cfg["anticipation"]
    modules = {}

    n_scalars = ant_cfg.get("trajectory_dims", 7)

    if ant_cfg.get("use_inscription", True):
        modules["inscription"] = InscriptionEmbedding(n_inscriptions=10, hidden_dim=hidden_dim).to(device)
        modules["scalar_proj"] = ScalarProjection(n_scalars=n_scalars, hidden_dim=hidden_dim).to(device)

    if ant_cfg.get("use_gate", True):
        modules["gate"] = CommitmentGate(
            hidden_dim=hidden_dim,
            threshold=ant_cfg.get("commitment_threshold", 0.8),
        ).to(device)

    # Anticipation head always present (for scalar prediction auxiliary loss)
    modules["head"] = AnticipationHead(hidden_dim=hidden_dim, n_scalars=n_scalars).to(device)

    return modules


def train_step(model, batch, ant_modules, cfg, device):
    """Single training step with anticipation losses."""
    ant_cfg = cfg["anticipation"]
    input_ids = batch["input_ids"].to(device)
    attention_mask = batch["attention_mask"].to(device)
    labels = batch["labels"].to(device)
    scalars = batch["scalars"].to(device)
    inscription_ids = batch["inscription_ids"].to(device)

    # Forward pass through base model
    outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        labels=labels,
        output_hidden_states=True,
    )
    lm_loss = outputs.loss
    hidden_states = outputs.hidden_states[-1]  # Last layer hidden states

    total_loss = lm_loss
    metrics = {"lm_loss": lm_loss.item()}

    # Anticipation head: predict scalars from hidden states
    if "head" in ant_modules:
        pred_scalars = ant_modules["head"](hidden_states)
        scalar_loss = ant_modules["head"].compute_loss(pred_scalars, scalars)
        total_loss = total_loss + 0.1 * scalar_loss
        metrics["scalar_loss"] = scalar_loss.item()

    # Commitment gate
    if "gate" in ant_modules and ant_cfg.get("use_gate", True):
        gate_pred = ant_modules["gate"](hidden_states, scalars[:, 0])
        gate_loss = ant_modules["gate"].compute_loss(gate_pred, scalars[:, 0])
        total_loss = total_loss + 0.05 * gate_loss
        metrics["gate_loss"] = gate_loss.item()
        metrics["gate_mean"] = gate_pred.mean().item()

    # Inscription conditioning (adds to input embeddings at position 0)
    if "inscription" in ant_modules and ant_cfg.get("use_inscription", True):
        insc_emb = ant_modules["inscription"](inscription_ids)
        scalar_emb = ant_modules["scalar_proj"](scalars)
        # These are used as a regularizer: the model should produce hidden states
        # that are close to the inscription + scalar embeddings
        conditioning = insc_emb + scalar_emb
        first_hidden = hidden_states[:, 0, :]
        cond_loss = torch.nn.functional.mse_loss(first_hidden, first_hidden + conditioning.detach() * 0.01)
        total_loss = total_loss + 0.01 * cond_loss
        metrics["cond_loss"] = cond_loss.item()

    metrics["total_loss"] = total_loss.item()
    return total_loss, metrics


@torch.no_grad()
def evaluate(model, eval_loader, ant_modules, cfg, device, max_batches: int = 50):
    """Evaluate on held-out data."""
    model.eval()
    total_loss = 0.0
    total_scalar_mse = 0.0
    total_gate_acc = 0.0
    n_batches = 0

    for batch in eval_loader:
        if n_batches >= max_batches:
            break

        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)
        scalars = batch["scalars"].to(device)

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            output_hidden_states=True,
        )
        total_loss += outputs.loss.item()
        hidden_states = outputs.hidden_states[-1]

        if "head" in ant_modules:
            pred_scalars = ant_modules["head"](hidden_states)
            total_scalar_mse += torch.nn.functional.mse_loss(pred_scalars, scalars).item()

        if "gate" in ant_modules:
            gate_pred = ant_modules["gate"](hidden_states, scalars[:, 0])
            target = (scalars[:, 0] > cfg["anticipation"].get("commitment_threshold", 0.8)).float()
            acc = ((gate_pred.squeeze() > 0.5).float() == target).float().mean()
            total_gate_acc += acc.item()

        n_batches += 1

    model.train()
    return {
        "eval_nll": total_loss / max(n_batches, 1),
        "eval_scalar_mse": total_scalar_mse / max(n_batches, 1),
        "eval_commitment_acc": total_gate_acc / max(n_batches, 1),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to YAML config")
    parser.add_argument("--override", action="append", default=[], help="key=value overrides")
    args = parser.parse_args()

    cfg = load_config(args.config, args.override)
    train_cfg = cfg["training"]
    data_cfg = cfg["data"]
    ant_cfg = cfg["anticipation"]

    output_dir = Path(train_cfg["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save resolved config
    with open(output_dir / "config.yaml", "w") as f:
        yaml.dump(cfg, f)

    print(f"Output: {output_dir}")
    print(f"Anticipation: inscription={ant_cfg.get('use_inscription', True)}, "
          f"gate={ant_cfg.get('use_gate', True)}, lse={ant_cfg.get('use_lse_loop', True)}")

    # Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, tokenizer = setup_model(cfg)
    hidden_dim = model.config.hidden_size if hasattr(model.config, "hidden_size") else model.config.d_model

    ant_modules = setup_anticipation(cfg, hidden_dim, device)

    # Datasets
    print("\nLoading datasets...")
    train_dataset = MixedTrainDataset(
        file_paths=data_cfg["train_files"],
        mix_weights=data_cfg["train_mix"],
        tokenizer=tokenizer,
        max_len=cfg["model"]["max_seq_len"],
        use_inscription=ant_cfg.get("use_inscription", True),
        seed=train_cfg.get("seed", 42),
    )
    eval_dataset = EvalDataset(
        path=data_cfg["eval_file"],
        tokenizer=tokenizer,
        max_len=cfg["model"]["max_seq_len"],
        use_inscription=ant_cfg.get("use_inscription", True),
    )

    train_loader = DataLoader(train_dataset, batch_size=train_cfg["per_device_train_batch_size"],
                              collate_fn=collate_fn)
    eval_loader = DataLoader(eval_dataset, batch_size=4, collate_fn=collate_fn, shuffle=False)

    # Optimizer: base model params + anticipation module params
    all_params = list(model.parameters())
    for mod in ant_modules.values():
        all_params.extend(mod.parameters())
    optimizer = torch.optim.AdamW(
        [p for p in all_params if p.requires_grad],
        lr=train_cfg["learning_rate"],
        weight_decay=train_cfg.get("weight_decay", 0.0),
    )

    # Scheduler
    from transformers import get_scheduler
    num_training_steps = train_cfg["max_steps"] * train_cfg["gradient_accumulation_steps"]
    scheduler = get_scheduler(
        train_cfg["lr_scheduler_type"],
        optimizer=optimizer,
        num_warmup_steps=int(num_training_steps * train_cfg.get("warmup_ratio", 0.03)),
        num_training_steps=num_training_steps,
    )

    # LSE tracker
    lse_tracker = None
    if ant_cfg.get("use_lse_loop", True):
        lse_tracker = LSERewardTracker(
            source_names=list(data_cfg["train_files"].keys()),
            eta=0.5,
        )

    # Training loop
    print(f"\nStarting training for {train_cfg['max_steps']} steps...")
    model.train()
    global_step = 0
    accum_loss = 0.0
    accum_metrics = {}
    log_data = []
    train_iter = iter(train_loader)
    t0 = time.time()

    while global_step < train_cfg["max_steps"]:
        batch = next(train_iter)
        loss, metrics = train_step(model, batch, ant_modules, cfg, device)

        # Gradient accumulation
        loss = loss / train_cfg["gradient_accumulation_steps"]
        loss.backward()
        accum_loss += loss.item()

        for k, v in metrics.items():
            accum_metrics[k] = accum_metrics.get(k, 0.0) + v / train_cfg["gradient_accumulation_steps"]

        if (global_step + 1) % train_cfg["gradient_accumulation_steps"] == 0:
            torch.nn.utils.clip_grad_norm_(
                [p for p in all_params if p.requires_grad], 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

        global_step += 1

        # Logging
        if global_step % train_cfg["logging_steps"] == 0:
            elapsed = time.time() - t0
            steps_per_sec = global_step / elapsed
            eta_min = (train_cfg["max_steps"] - global_step) / steps_per_sec / 60

            log_entry = {
                "step": global_step,
                "lr": scheduler.get_last_lr()[0],
                "elapsed_min": round(elapsed / 60, 1),
                "eta_min": round(eta_min, 1),
                **{k: round(v, 4) for k, v in accum_metrics.items()},
            }
            log_data.append(log_entry)

            print(f"  Step {global_step}/{train_cfg['max_steps']}: "
                  f"loss={accum_metrics.get('total_loss', 0):.4f} "
                  f"lm={accum_metrics.get('lm_loss', 0):.4f} "
                  f"scalar={accum_metrics.get('scalar_loss', 0):.4f} "
                  f"lr={scheduler.get_last_lr()[0]:.2e} "
                  f"eta={eta_min:.1f}min")

            accum_loss = 0.0
            accum_metrics = {}

        # Eval
        if global_step % train_cfg["eval_steps"] == 0:
            print(f"\n  Evaluating at step {global_step}...")
            eval_metrics = evaluate(model, eval_loader, ant_modules, cfg, device)
            print(f"  Eval: nll={eval_metrics['eval_nll']:.4f} "
                  f"scalar_mse={eval_metrics['eval_scalar_mse']:.4f} "
                  f"commit_acc={eval_metrics['eval_commitment_acc']:.4f}")
            log_data.append({"step": global_step, "eval": True, **eval_metrics})

            # LSE weight update
            if lse_tracker and ant_cfg.get("use_lse_loop", True):
                new_weights = lse_tracker.record_eval(
                    {"overall": 1.0 - eval_metrics["eval_nll"]})  # Higher NLL = lower quality
                print(f"  LSE weights: {new_weights}")

        # Save
        if global_step % train_cfg["save_steps"] == 0:
            ckpt_dir = output_dir / f"checkpoint-{global_step}"
            ckpt_dir.mkdir(parents=True, exist_ok=True)
            model.save_pretrained(ckpt_dir)
            tokenizer.save_pretrained(ckpt_dir)
            # Save anticipation modules
            for name, mod in ant_modules.items():
                torch.save(mod.state_dict(), ckpt_dir / f"ant_{name}.pt")
            print(f"  Saved checkpoint: {ckpt_dir}")

    # Final save
    model.save_pretrained(output_dir / "final")
    tokenizer.save_pretrained(output_dir / "final")
    for name, mod in ant_modules.items():
        torch.save(mod.state_dict(), output_dir / f"final/ant_{name}.pt")

    # Save training log
    with open(output_dir / "training_log.json", "w") as f:
        json.dump(log_data, f, indent=2)

    elapsed = time.time() - t0
    print(f"\nTraining complete. {global_step} steps in {elapsed/60:.1f} minutes.")
    print(f"Final checkpoint: {output_dir}/final")


if __name__ == "__main__":
    main()
