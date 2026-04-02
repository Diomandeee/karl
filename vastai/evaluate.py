#!/usr/bin/env python3
"""Evaluate trained checkpoints with bootstrap confidence intervals.

Usage:
    python evaluate.py --ckpt runs/qwen3-4b-treatment/final \
        --eval_file /workspace/data/real_conv_eval.jsonl \
        --bootstrap 1000 --metrics nll,commitment_acc,scalar_mse,ece
"""

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.data import EvalDataset, collate_fn
from src.anticipation import AnticipationHead, CommitmentGate


def load_model(ckpt_path: str):
    """Load QLoRA model from checkpoint."""
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from peft import PeftModel

    ckpt = Path(ckpt_path)

    # Check if this is a PEFT adapter
    if (ckpt / "adapter_config.json").exists():
        with open(ckpt / "adapter_config.json") as f:
            adapter_cfg = json.load(f)
        base_model_name = adapter_cfg.get("base_model_name_or_path", "Qwen/Qwen3-4B-Base")

        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_name, quantization_config=bnb_config,
            device_map="auto", trust_remote_code=True, torch_dtype=torch.bfloat16)
        model = PeftModel.from_pretrained(base_model, str(ckpt))
    else:
        model = AutoModelForCausalLM.from_pretrained(
            str(ckpt), device_map="auto", trust_remote_code=True, torch_dtype=torch.bfloat16)

    tokenizer = AutoTokenizer.from_pretrained(str(ckpt), trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    return model, tokenizer


def compute_ece(confidences: np.ndarray, accuracies: np.ndarray, n_bins: int = 15) -> float:
    """Expected Calibration Error."""
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        mask = (confidences >= bin_boundaries[i]) & (confidences < bin_boundaries[i + 1])
        if mask.sum() == 0:
            continue
        bin_acc = accuracies[mask].mean()
        bin_conf = confidences[mask].mean()
        ece += mask.sum() / len(confidences) * abs(bin_acc - bin_conf)
    return ece


def evaluate_checkpoint(model, tokenizer, eval_path: str, device: torch.device,
                        ant_modules: dict, config: dict) -> dict:
    """Run full evaluation on a checkpoint, returning per-record scores."""
    use_inscription = config.get("anticipation", {}).get("use_inscription", True)
    eval_dataset = EvalDataset(eval_path, tokenizer, max_len=2048, use_inscription=use_inscription)
    eval_loader = DataLoader(eval_dataset, batch_size=4, collate_fn=collate_fn, shuffle=False)

    model.eval()
    all_nll = []
    all_scalar_mse = []
    all_gate_conf = []
    all_gate_correct = []
    threshold = config.get("anticipation", {}).get("commitment_threshold", 0.8)

    with torch.no_grad():
        for batch in eval_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            scalars = batch["scalars"].to(device)

            outputs = model(
                input_ids=input_ids, attention_mask=attention_mask,
                labels=labels, output_hidden_states=True)

            # Per-example NLL
            all_nll.append(outputs.loss.item())

            hidden_states = outputs.hidden_states[-1]

            # Scalar MSE
            if "head" in ant_modules:
                pred = ant_modules["head"](hidden_states)
                mse = ((pred - scalars) ** 2).mean(dim=-1)
                all_scalar_mse.extend(mse.cpu().numpy().tolist())

            # Gate accuracy + calibration
            if "gate" in ant_modules:
                gate_pred = ant_modules["gate"](hidden_states, scalars[:, 0])
                target = (scalars[:, 0] > threshold).float()
                conf = gate_pred.squeeze().cpu().numpy()
                correct = ((gate_pred.squeeze() > 0.5).float() == target).float().cpu().numpy()
                all_gate_conf.extend(conf.tolist())
                all_gate_correct.extend(correct.tolist())

    return {
        "nll_values": all_nll,
        "scalar_mse_values": all_scalar_mse,
        "gate_conf": all_gate_conf,
        "gate_correct": all_gate_correct,
    }


def bootstrap_ci(values: list[float], n_bootstrap: int = 1000, ci: float = 0.95) -> dict:
    """Compute bootstrap confidence interval for the mean."""
    arr = np.array(values)
    n = len(arr)
    if n == 0:
        return {"mean": 0.0, "ci_low": 0.0, "ci_high": 0.0, "std": 0.0}

    means = []
    rng = np.random.RandomState(42)
    for _ in range(n_bootstrap):
        sample = rng.choice(arr, size=n, replace=True)
        means.append(sample.mean())

    means = np.sort(means)
    alpha = (1 - ci) / 2
    return {
        "mean": float(arr.mean()),
        "ci_low": float(means[int(alpha * n_bootstrap)]),
        "ci_high": float(means[int((1 - alpha) * n_bootstrap)]),
        "std": float(arr.std()),
        "n": n,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", required=True, help="Checkpoint path")
    parser.add_argument("--eval_file", required=True, help="Eval JSONL path")
    parser.add_argument("--bootstrap", type=int, default=1000, help="Bootstrap samples")
    parser.add_argument("--metrics", default="nll,commitment_acc,scalar_mse,ece",
                        help="Comma-separated metrics")
    parser.add_argument("--output", default=None, help="Output JSON path")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt_path = Path(args.ckpt)

    # Load config
    config = {}
    for cfg_path in [ckpt_path / "config.yaml", ckpt_path.parent / "config.yaml"]:
        if cfg_path.exists():
            import yaml
            with open(cfg_path) as f:
                config = yaml.safe_load(f)
            break

    print(f"Loading checkpoint: {args.ckpt}")
    model, tokenizer = load_model(args.ckpt)
    hidden_dim = model.config.hidden_size if hasattr(model.config, "hidden_size") else 2048

    # Load anticipation modules
    ant_modules = {}
    head_path = ckpt_path / "ant_head.pt"
    if head_path.exists():
        ant_modules["head"] = AnticipationHead(hidden_dim=hidden_dim).to(device)
        ant_modules["head"].load_state_dict(torch.load(head_path, map_location=device))
        ant_modules["head"].eval()

    gate_path = ckpt_path / "ant_gate.pt"
    if gate_path.exists():
        ant_modules["gate"] = CommitmentGate(hidden_dim=hidden_dim).to(device)
        ant_modules["gate"].load_state_dict(torch.load(gate_path, map_location=device))
        ant_modules["gate"].eval()

    print(f"Evaluating on {args.eval_file}...")
    raw = evaluate_checkpoint(model, tokenizer, args.eval_file, device, ant_modules, config)

    metrics_requested = args.metrics.split(",")
    results = {"checkpoint": str(args.ckpt), "eval_file": args.eval_file}

    if "nll" in metrics_requested and raw["nll_values"]:
        results["nll"] = bootstrap_ci(raw["nll_values"], args.bootstrap)
        print(f"  NLL: {results['nll']['mean']:.4f} [{results['nll']['ci_low']:.4f}, {results['nll']['ci_high']:.4f}]")

    if "scalar_mse" in metrics_requested and raw["scalar_mse_values"]:
        results["scalar_mse"] = bootstrap_ci(raw["scalar_mse_values"], args.bootstrap)
        print(f"  Scalar MSE: {results['scalar_mse']['mean']:.4f} [{results['scalar_mse']['ci_low']:.4f}, {results['scalar_mse']['ci_high']:.4f}]")

    if "commitment_acc" in metrics_requested and raw["gate_correct"]:
        results["commitment_acc"] = bootstrap_ci(raw["gate_correct"], args.bootstrap)
        print(f"  Commitment Acc: {results['commitment_acc']['mean']:.4f} [{results['commitment_acc']['ci_low']:.4f}, {results['commitment_acc']['ci_high']:.4f}]")

    if "ece" in metrics_requested and raw["gate_conf"] and raw["gate_correct"]:
        ece = compute_ece(np.array(raw["gate_conf"]), np.array(raw["gate_correct"]))
        results["ece"] = {"value": float(ece)}
        print(f"  ECE: {ece:.4f}")

    # Save results
    out_path = args.output or str(ckpt_path / "eval_results.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
