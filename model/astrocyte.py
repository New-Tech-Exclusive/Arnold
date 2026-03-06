"""Astrocyte Layer — Metabolic Pathway Penalties.

Biological brains are ~50% glial cells.  Astrocytes modulate synaptic
transmission based on metabolic signals and activity history.

Pathways used heavily become metabolically expensive.  This layer
applies a slow-moving penalty to overused pathways, forcing the
network to develop alternative routes — producing cognitive flexibility.

Distinct from EWC: EWC protects important weights from being overwritten.
The astrocyte layer penalises overuse of ANY pathway, forcing diversity.
"""

from __future__ import annotations

import torch

from .genome import Genome
from .tensor import DTYPE, get_device
from .types_ import CortexRegion


class AstrocyteLayer:
    """Slow-timescale metabolic modulation of cortex pathways."""

    def __init__(self, genome: Genome) -> None:
        self._genome = genome
        self._device = get_device()
        self._params = genome.astrocyte

        hidden_dim = genome.topology.cortex_hidden

        # Per-neuron usage counters for each region
        self._usage: dict[CortexRegion, torch.Tensor] = {}
        for region in genome.topology.active_regions:
            self._usage[region] = torch.zeros(
                hidden_dim, dtype=DTYPE, device=self._device,
            )

    def record_activity(
        self,
        region: CortexRegion,
        fired_indices: torch.Tensor,
    ) -> None:
        """Track which neurons fired — increment usage counters."""
        if region not in self._usage:
            return
        if fired_indices.numel() > 0:
            valid = fired_indices[fired_indices < self._usage[region].shape[0]]
            self._usage[region][valid] += 1.0

    def get_penalty(self, region: CortexRegion) -> torch.Tensor:
        """Return metabolic penalty tensor for the given region.

        Overused neurons get penalised more.  Returns a multiplicative
        scale factor ∈ (0, 1] where 1.0 means no penalty.
        """
        if region not in self._usage:
            return torch.ones(
                self._genome.topology.cortex_hidden,
                dtype=DTYPE, device=self._device,
            )

        usage = self._usage[region]
        target = self._params.target_usage
        scale = self._params.penalty_scale

        # Penalty increases with usage above target
        excess = torch.relu(usage - target)
        penalty = 1.0 / (1.0 + scale * excess)
        return penalty

    def apply_penalties(self, cortex) -> None:
        """Apply metabolic penalties to cortex W_in weights.

        Overused input pathways are scaled down, pushing the network
        to find alternative routes.
        """
        for region in self._genome.topology.active_regions:
            if region not in cortex.regions:
                continue
            penalty = self.get_penalty(region)
            # Modulate W_in columns by penalty (overused neurons get dampened)
            cortex.regions[region].W_in *= penalty.unsqueeze(0)

    def decay(self) -> None:
        """Slow decay of usage counters — metabolic recovery."""
        for region in self._usage:
            self._usage[region] *= self._params.metabolic_decay

    def get_usage(self) -> dict[CortexRegion, torch.Tensor]:
        return {r: v.detach().clone() for r, v in self._usage.items()}

    def set_usage(self, usage: dict[CortexRegion, torch.Tensor]) -> None:
        for r, v in usage.items():
            if r in self._usage and v.shape == self._usage[r].shape:
                self._usage[r] = v.to(device=self._device, dtype=DTYPE)
