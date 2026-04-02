"""Dataset loading with weighted mix sampling for Anticipatory Transformer."""

import json
import random
from pathlib import Path
from typing import Optional

import torch
from torch.utils.data import Dataset, IterableDataset


INSCRIPTION_VOCAB = [
    "stabilization", "transition", "oscillation", "correction", "exploration",
    "convergence", "expansion", "regression", "stagnation", "completion",
]
INSCRIPTION_TO_ID = {s: i for i, s in enumerate(INSCRIPTION_VOCAB)}
SCALAR_KEYS = ["commitment", "uncertainty", "transition_pressure", "recovery_margin"]


def load_jsonl(path: str | Path) -> list[dict]:
    records = []
    with open(path) as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
    return records


def format_messages(record: dict, tokenizer, max_len: int, use_inscription: bool = True) -> dict:
    """Format a record into tokenized input with anticipation metadata."""
    messages = record["messages"]
    scalars = record.get("scalars", {})
    inscription = record.get("inscription", "stabilization")
    position = record.get("position", 0.5)

    # Build text: optionally prepend inscription token
    parts = []
    if use_inscription:
        parts.append(f"[MOTIF:{inscription}]")

    for msg in messages:
        role = msg["role"]
        content = msg["content"]
        parts.append(f"<|im_start|>{role}\n{content}<|im_end|>")

    text = "\n".join(parts)
    encoded = tokenizer(text, truncation=True, max_length=max_len, return_tensors="pt")

    # Scalar tensor (4 dims from data, pad to trajectory_dims with derived features)
    s = scalars
    scalar_vec = [
        s.get("commitment", 0.5),
        s.get("uncertainty", 0.5),
        s.get("transition_pressure", 0.0),
        s.get("recovery_margin", 0.5),
        position,
        # Derived: focus = commitment - uncertainty
        s.get("commitment", 0.5) - s.get("uncertainty", 0.5),
        # Derived: stability = 1 - abs(transition_pressure)
        1.0 - abs(s.get("transition_pressure", 0.0)),
    ]

    return {
        "input_ids": encoded["input_ids"].squeeze(0),
        "attention_mask": encoded["attention_mask"].squeeze(0),
        "scalars": torch.tensor(scalar_vec, dtype=torch.float32),
        "inscription_id": INSCRIPTION_TO_ID.get(inscription, 0),
    }


class MixedTrainDataset(IterableDataset):
    """Weighted mix of multiple JSONL sources, streamed with replacement."""

    def __init__(self, file_paths: dict[str, str], mix_weights: dict[str, float],
                 tokenizer, max_len: int, use_inscription: bool = True, seed: int = 42):
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.use_inscription = use_inscription
        self.seed = seed

        # Load all sources
        self.sources = {}
        self.weights = []
        self.source_names = []
        for name, path in file_paths.items():
            p = Path(path)
            if p.exists() and p.stat().st_size > 0:
                records = load_jsonl(p)
                if records:
                    self.sources[name] = records
                    self.weights.append(mix_weights.get(name, 1.0 / len(file_paths)))
                    self.source_names.append(name)

        # Normalize weights
        total = sum(self.weights)
        self.weights = [w / total for w in self.weights]
        print(f"  Dataset sources: {', '.join(f'{n}({len(self.sources[n])})' for n in self.source_names)}")
        print(f"  Mix weights: {', '.join(f'{n}={w:.2f}' for n, w in zip(self.source_names, self.weights))}")

    def __iter__(self):
        rng = random.Random(self.seed)
        while True:
            # Weighted source selection
            source_name = rng.choices(self.source_names, weights=self.weights, k=1)[0]
            record = rng.choice(self.sources[source_name])
            try:
                yield format_messages(record, self.tokenizer, self.max_len, self.use_inscription)
            except Exception:
                continue


class EvalDataset(Dataset):
    """Standard eval dataset (not iterable, for deterministic evaluation)."""

    def __init__(self, path: str | Path, tokenizer, max_len: int, use_inscription: bool = True):
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.use_inscription = use_inscription
        self.records = load_jsonl(path)

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        return format_messages(self.records[idx], self.tokenizer, self.max_len, self.use_inscription)


def collate_fn(batch: list[dict]) -> dict:
    """Pad batch to same length."""
    max_len = max(b["input_ids"].size(0) for b in batch)
    input_ids = torch.zeros(len(batch), max_len, dtype=torch.long)
    attention_mask = torch.zeros(len(batch), max_len, dtype=torch.long)
    labels = torch.full((len(batch), max_len), -100, dtype=torch.long)
    scalars = torch.stack([b["scalars"] for b in batch])
    inscription_ids = torch.tensor([b["inscription_id"] for b in batch], dtype=torch.long)

    for i, b in enumerate(batch):
        seq_len = b["input_ids"].size(0)
        input_ids[i, :seq_len] = b["input_ids"]
        attention_mask[i, :seq_len] = b["attention_mask"]
        labels[i, :seq_len] = b["input_ids"]  # Causal LM: labels = input_ids

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
        "scalars": scalars,
        "inscription_ids": inscription_ids,
    }
