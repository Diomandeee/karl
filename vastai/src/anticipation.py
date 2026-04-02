"""Anticipation modules: inscription embedding, commitment gate, LSE loop.

These are lightweight additions to a frozen/LoRA'd base model.
They implement the behavioral motif retrieval mechanism described in the paper.

Components:
  InscriptionEmbedding: Embeds 10 behavioral motifs into model hidden dim
  ScalarProjection:     Projects 7 trajectory scalars to hidden dim
  CommitmentGate:       Binary gate based on commitment scalar
  AnticipationHead:     Predicts scalars from hidden states (auxiliary loss)
  LSERewardTracker:     Tracks per-source rewards for weight updates
"""

import math
from collections import defaultdict

import torch
import torch.nn as nn
import torch.nn.functional as F


class InscriptionEmbedding(nn.Module):
    """Embed behavioral motif labels into model hidden space.

    Maps 10 inscription categories to dense vectors that are added
    to the first token's hidden state, conditioning the model on
    the session's behavioral pattern.
    """

    def __init__(self, n_inscriptions: int = 10, hidden_dim: int = 2048):
        super().__init__()
        self.embedding = nn.Embedding(n_inscriptions, hidden_dim)
        self.scale = nn.Parameter(torch.tensor(0.1))  # Start small

    def forward(self, inscription_ids: torch.Tensor) -> torch.Tensor:
        """Returns (batch, hidden_dim) inscription embeddings."""
        return self.embedding(inscription_ids) * self.scale


class ScalarProjection(nn.Module):
    """Project 7 trajectory scalars to hidden dim for residual addition."""

    def __init__(self, n_scalars: int = 7, hidden_dim: int = 2048):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(n_scalars, hidden_dim // 4),
            nn.GELU(),
            nn.Linear(hidden_dim // 4, hidden_dim),
        )
        self.scale = nn.Parameter(torch.tensor(0.1))

    def forward(self, scalars: torch.Tensor) -> torch.Tensor:
        """Returns (batch, hidden_dim) scalar projection."""
        return self.proj(scalars) * self.scale


class CommitmentGate(nn.Module):
    """Binary gate: emit if commitment > threshold, buffer otherwise.

    In training, we use a soft gate (sigmoid) for gradient flow.
    The gate modulates the loss: high-commitment tokens get full loss weight,
    low-commitment tokens get reduced weight.
    """

    def __init__(self, hidden_dim: int = 2048, threshold: float = 0.8):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_dim, 1)
        self.threshold = threshold

    def forward(self, hidden_states: torch.Tensor, commitment_scalar: torch.Tensor) -> torch.Tensor:
        """Returns (batch, 1) gate values in [0, 1]."""
        # Predicted gate from hidden states
        gate_logit = self.gate_proj(hidden_states[:, -1, :])  # Use last token
        gate_pred = torch.sigmoid(gate_logit)
        return gate_pred

    def compute_loss(self, gate_pred: torch.Tensor, commitment_scalar: torch.Tensor) -> torch.Tensor:
        """BCE loss between predicted gate and thresholded commitment."""
        target = (commitment_scalar > self.threshold).float().unsqueeze(-1)
        return F.binary_cross_entropy(gate_pred, target)


class AnticipationHead(nn.Module):
    """Predicts trajectory scalars from hidden states (auxiliary task).

    This head is trained to predict the 7 anticipation scalars from
    the model's hidden representations. The loss serves as a regularizer
    that encourages the model to encode behavioral patterns.
    """

    def __init__(self, hidden_dim: int = 2048, n_scalars: int = 7):
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 4),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 4, n_scalars),
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Predict scalars from mean-pooled hidden states."""
        pooled = hidden_states.mean(dim=1)  # (batch, hidden_dim)
        return self.head(pooled)  # (batch, n_scalars)

    def compute_loss(self, pred_scalars: torch.Tensor, target_scalars: torch.Tensor) -> torch.Tensor:
        """MSE loss between predicted and target scalars."""
        return F.mse_loss(pred_scalars, target_scalars)


class LSERewardTracker:
    """Track per-source rewards for LSE-style weight updates.

    After each eval, computes reward = mean_quality - baseline per source,
    then updates mix weights via exponential scaling:
        new_weight = old_weight * exp(eta * reward)
    """

    def __init__(self, source_names: list[str], eta: float = 0.5, baseline: float = 0.3):
        self.eta = eta
        self.baseline = baseline
        self.source_names = source_names
        self.rewards_history: list[dict] = []
        self.current_weights = {name: 1.0 / len(source_names) for name in source_names}

    def record_eval(self, per_source_quality: dict[str, float]):
        """Record eval quality per source and update weights."""
        rewards = {}
        for name in self.source_names:
            q = per_source_quality.get(name, self.baseline)
            rewards[name] = q - self.baseline

        # Exponential weight update
        for name in self.source_names:
            r = rewards.get(name, 0.0)
            self.current_weights[name] *= math.exp(self.eta * max(-5, min(5, r)))

        # Normalize
        total = sum(self.current_weights.values())
        self.current_weights = {k: v / total for k, v in self.current_weights.items()}

        self.rewards_history.append({"rewards": rewards, "weights": dict(self.current_weights)})
        return self.current_weights

    def get_weights(self) -> dict[str, float]:
        return dict(self.current_weights)
