"""Cerebellum — Error Correction and Forward Model.

A small lightweight network that predicts response quality before
generation commits.  It receives a copy of the context and predicted
direction, and outputs an estimated reinforcement signal.

This estimated signal pre-adjusts generation temperature and cortex
voting weights.  If the forward model predicts a negative outcome,
generation steers away.

Related to model-based RL — using an internal model of consequences
to guide behaviour rather than learning purely from actual outcomes.
"""

from __future__ import annotations

import torch

from .genome import Genome
from .tensor import DTYPE, get_device


class Cerebellum:
    """Fast forward model for pre-correcting generation."""

    def __init__(self, genome: Genome) -> None:
        self._genome = genome
        self._device = get_device()
        self._params = genome.cerebellum

        input_dim = genome.topology.embed_dim
        hidden_dim = self._params.hidden_dim

        # Small two-layer forward model: context → predicted reinforcement
        self.W_in = torch.randn(
            (input_dim, hidden_dim), dtype=DTYPE, device=self._device,
        ) * 0.02
        self.W_out = torch.randn(
            (hidden_dim, 1), dtype=DTYPE, device=self._device,
        ) * 0.02

        # Prediction history for learning
        self._last_prediction: float = 0.0

    def predict_outcome(self, context_embedding: torch.Tensor) -> float:
        """Predict the reinforcement signal for the given context.

        Returns a scalar estimate ∈ [-1, 1].
        """
        x = context_embedding.to(self._device).mean(dim=0) if context_embedding.dim() > 1 else context_embedding.to(self._device)
        hidden = torch.tanh(x @ self.W_in)
        raw = float((hidden @ self.W_out).item())
        prediction = float(torch.tanh(torch.tensor(raw)).item())
        self._last_prediction = prediction
        return prediction

    def pre_correct_logits(
        self,
        logits: torch.Tensor,
        context_embedding: torch.Tensor,
    ) -> torch.Tensor:
        """Adjust logits based on predicted outcome.

        If the forward model predicts negative outcome, dampen high-confidence
        tokens (encourage exploration).  If positive, sharpen (exploit).
        """
        prediction = self.predict_outcome(context_embedding)
        weight = self._params.correction_weight

        if prediction < 0:
            # Predicted bad outcome → flatten logits (explore)
            scale = 1.0 + weight * abs(prediction)
            return logits / scale
        else:
            # Predicted good outcome → sharpen logits (exploit)
            scale = 1.0 + weight * prediction * 0.5
            return logits * scale

    def update(self, actual_reinforcement: float) -> None:
        """Update forward model based on actual outcome vs prediction.

        Simple Hebbian-style error correction.
        """
        error = actual_reinforcement - self._last_prediction
        # Update is applied in brain.py after reinforcement is known

    def train_step(
        self,
        context_embedding: torch.Tensor,
        actual_reinforcement: float,
    ) -> None:
        """Single training step for the forward model."""
        lr = self._params.learning_rate

        x = context_embedding.to(self._device).mean(dim=0) if context_embedding.dim() > 1 else context_embedding.to(self._device)
        hidden = torch.tanh(x @ self.W_in)
        raw = float((hidden @ self.W_out).item())
        prediction = float(torch.tanh(torch.tensor(raw)).item())

        error = actual_reinforcement - prediction

        # Gradient approximation for tanh output
        grad_out = (1.0 - prediction ** 2) * error

        # Update W_out
        delta_out = lr * grad_out * hidden.unsqueeze(1)
        self.W_out += delta_out

        # Update W_in
        grad_hidden = (1.0 - hidden ** 2) * (grad_out * self.W_out.squeeze())
        delta_in = lr * torch.outer(x, grad_hidden)
        self.W_in += delta_in

    def get_weights(self) -> dict[str, torch.Tensor]:
        return {
            "W_in": self.W_in.detach().clone(),
            "W_out": self.W_out.detach().clone(),
        }

    def set_weights(self, weights: dict[str, torch.Tensor]) -> None:
        if "W_in" in weights and weights["W_in"].shape == self.W_in.shape:
            self.W_in = weights["W_in"].to(device=self._device, dtype=DTYPE)
        if "W_out" in weights and weights["W_out"].shape == self.W_out.shape:
            self.W_out = weights["W_out"].to(device=self._device, dtype=DTYPE)
