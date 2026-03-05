"""Tier 7 — The Plasticity System (Learning Rate Governor).

Not a layer — a system that runs across all plastic layers and governs how
fast they can change at any developmental stage.

plasticity(age) = (P_max - P_floor) × e^(-λ × age) + P_floor

Per-region λ values mean regions close at different rates.
Mood can temporarily raise or lower effective plasticity above/below baseline.
"""

from __future__ import annotations

import math

import numpy as np

from .genome import Genome
from .types_ import CortexRegion, DevelopmentalStage, MoodState


class PlasticitySystem:
    """Governs all Hebbian update rates across the brain."""

    def __init__(self, genome: Genome) -> None:
        self._genome = genome
        self._params = genome.plasticity

    def base_rate(self, region: CortexRegion, age: int) -> float:
        """Age-dependent base plasticity for a region.

        plasticity(age) = (P_max - P_floor) × e^(-λ_region × age) + P_floor
        """
        lam = self._params.region_lambdas.get(region, self._params.lambda_decay)
        return (
            (self._params.p_max - self._params.p_floor)
            * math.exp(-lam * age)
            + self._params.p_floor
        )

    def effective_rate(
        self,
        region: CortexRegion,
        age: int,
        mood: MoodState,
        ewc_protection: float = 0.0,
    ) -> float:
        """Effective plasticity after mood modulation and EWC damping.

        effective = base × mood_factor × (1 - ewc_protection)
        """
        base = self.base_rate(region, age)

        # Mood modulation: high openness boosts, low openness dampens
        openness_center = 0.5
        if mood.openness > openness_center:
            mood_factor = 1.0 + self._params.mood_plasticity_boost * (
                mood.openness - openness_center
            ) / (1.0 - openness_center)
        else:
            mood_factor = 1.0 - self._params.mood_plasticity_dampen * (
                openness_center - mood.openness
            ) / openness_center

        # EWC damping
        ewc_factor = 1.0 - ewc_protection

        return float(np.clip(base * mood_factor * ewc_factor, 0.0, self._params.p_max))

    def all_rates(
        self,
        age: int,
        mood: MoodState,
        ewc_protections: dict[CortexRegion, float] | None = None,
    ) -> dict[CortexRegion, float]:
        """Compute effective rates for all active regions."""
        ewc = ewc_protections or {}
        rates = {}
        for region in self._genome.topology.active_regions:
            prot = float(np.mean(ewc[region])) if region in ewc else 0.0
            rates[region] = self.effective_rate(region, age, mood, prot)
        return rates

    def stage(self, age: int) -> DevelopmentalStage:
        """Current developmental stage based on age."""
        return self._genome.development.stage_for_age(age)

    def describe(self, age: int, mood: MoodState) -> dict:
        """Human-readable summary."""
        stage = self.stage(age)
        rates = self.all_rates(age, mood)
        return {
            "developmental_stage": stage.value,
            "age": age,
            "mood_openness": mood.openness,
            "per_region_plasticity": {r.value: round(v, 6) for r, v in rates.items()},
        }
