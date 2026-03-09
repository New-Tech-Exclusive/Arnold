"""Thalamic Gating — Active routing of inputs to transformer regions.

The router sits between the reinforcement gate and transformer.  Every signal
passes through here, and the router decides which transformer regions
receive each input — and with what weight.

Implements a thalamocortical feedback loop:
  Input → Router → weighted routing to regions
  Transformer output → feedback to router → adjusts next routing

This produces genuine biological attention (selective routing)
rather than mathematical attention (dot-product weighting).
"""

from __future__ import annotations

import torch

from .genome import Genome
from .tensor import DTYPE, get_device
from .types_ import TransformerRegion, NeuromodulatorState


class Router:
    """Active input router with cortical feedback loop."""

    def __init__(self, genome: Genome) -> None:
        self._genome = genome
        self._device = get_device()
        self._params = genome.router

        embed_dim = genome.topology.embed_dim
        n_regions = len(genome.topology.active_regions)
        self._regions = list(genome.topology.active_regions)

        # Routing projection: embed_dim → n_regions
        self.W_route = torch.randn(
            (embed_dim, n_regions), dtype=DTYPE, device=self._device,
        ) * 0.02
        self.W_route.requires_grad_(True)

        # Cortical feedback: transformer_hidden → n_regions
        self.W_feedback = torch.randn(
            (genome.topology.transformer_hidden, n_regions), dtype=DTYPE, device=self._device,
        ) * 0.01
        self.W_feedback.requires_grad_(True)

        # Persistent feedback state from last transformer output
        self._feedback_state = torch.zeros(n_regions, dtype=DTYPE, device=self._device)

    def route(
        self,
        embeddings: torch.Tensor,
        neuromodulators: NeuromodulatorState,
        surprise_score: float = 0.0,
    ) -> dict[TransformerRegion, float]:
        """Compute routing weights for each transformer region.

        High surprise biases toward parietal.
        Neuromodulatory state modulates routing sharpness:
          high norepinephrine → broader (explore), low → focused (exploit).
        """
        mean_emb = embeddings.to(self._device).mean(dim=0)  # (embed_dim,)
        raw_scores = mean_emb @ self.W_route  # (n_regions,)

        # Add cortical feedback
        scores = raw_scores + self._params.feedback_strength * self._feedback_state

        # Norepinephrine modulates routing sharpness
        sharpness = self._params.routing_sharpness / (1.0 + neuromodulators.norepinephrine)

        # Softmax to routing probabilities
        routing_probs = self._softmax(scores * sharpness)

        # Build region weight map
        weights: dict[TransformerRegion, float] = {}
        for i, region in enumerate(self._regions):
            weights[region] = float(routing_probs[i])

        return weights

    def receive_feedback(
        self,
        region_hiddens: dict[TransformerRegion, torch.Tensor],
    ) -> None:
        """Update feedback state from transformer output.

        The winning region's activation pattern modulates next routing.
        """
        n = len(self._regions)
        feedback = torch.zeros(n, dtype=DTYPE, device=self._device)
        for i, region in enumerate(self._regions):
            if region in region_hiddens:
                h = region_hiddens[region].to(self._device)
                # Project hidden to routing score
                feedback[i] = float((h @ self.W_feedback).mean())

        # Blend with previous feedback (momentum)
        self._feedback_state = 0.8 * self._feedback_state + 0.2 * feedback

    def get_weights(self) -> dict[str, torch.Tensor]:
        return {
            "W_route": self.W_route.detach().clone(),
            "W_feedback": self.W_feedback.detach().clone(),
        }

    def parameters(self) -> list[torch.Tensor]:
        """Return trainable tensors for gradient updates."""
        return [self.W_route, self.W_feedback]

    def set_weights(self, weights: dict[str, torch.Tensor]) -> None:
        if "W_route" in weights and weights["W_route"].shape == self.W_route.shape:
            self.W_route = weights["W_route"].to(device=self._device, dtype=DTYPE)
        if "W_feedback" in weights and weights["W_feedback"].shape == self.W_feedback.shape:
            self.W_feedback = weights["W_feedback"].to(device=self._device, dtype=DTYPE)

    @staticmethod
    def _softmax(x: torch.Tensor) -> torch.Tensor:
        shifted = x - torch.max(x)
        exp = torch.exp(shifted)
        return exp / (torch.sum(exp) + 1e-12)
