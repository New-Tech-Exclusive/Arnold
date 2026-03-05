"""Tier 5 — The Novelty Detector (Surprise Scoring System).

Runs inline on every incoming message before the cortex processes it.
Compares incoming input against current cortex weights to calculate how
surprising the input is relative to what the model has already learned.
"""

from __future__ import annotations

import torch

from .genome import Genome
from .types_ import CortexRegion, StructuredRepresentation


class NoveltyDetector:
    """Scores how novel / surprising an input is relative to learned weights."""

    def __init__(self, genome: Genome) -> None:
        self._genome = genome
        self._params = genome.novelty

    def score(
        self,
        structured: StructuredRepresentation,
        cortex_forward_readonly,  # callable: embeddings → dict[region, RegionActivation]
    ) -> float:
        """Compute novelty score ∈ [0, 1] for the given input.

        Activates the cortex in *read-only* mode.  Measures how confidently
        existing weights predicted this input.  Low confidence = high novelty.

        Args:
            structured: Brainstem output.
            cortex_forward_readonly: A function that takes (seq_len, embed_dim)
                embeddings and returns region activations without recording traces.

        Returns:
            Scalar novelty score between 0.0 and 1.0.
        """
        embeddings = structured.embeddings  # (seq_len, embed_dim)
        token_ids = structured.token_ids     # (seq_len,)

        # Read-only forward pass through cortex
        activations = cortex_forward_readonly(embeddings)

        # For each active region, measure prediction confidence
        prediction_confidences: list[float] = []

        for region, act in activations.items():
            if region == CortexRegion.OCCIPITAL:
                continue  # dormant

            logits = act.logits  # (vocab_size,)

            # Convert to probability distribution via softmax
            probs = self._softmax(logits / self._params.temperature)

            # Measure how much probability mass landed on the actual tokens
            if len(token_ids) > 0:
                # Average probability the model assigned to the actual tokens
                valid_ids = token_ids[token_ids < probs.shape[0]]
                if int(valid_ids.numel()) > 0:
                    assigned_probs = probs[valid_ids]
                    confidence = float(assigned_probs.mean())
                else:
                    confidence = 0.0
            else:
                confidence = 0.0

            prediction_confidences.append(confidence)

        if not prediction_confidences:
            return 1.0  # no predictions → maximum novelty

        mean_confidence = float(sum(prediction_confidences) / max(len(prediction_confidences), 1))

        # Novelty = 1 - confidence  (clamped)
        novelty = float(max(min(1.0 - mean_confidence, 1.0), 0.0))
        return novelty

    @staticmethod
    def _softmax(x: torch.Tensor) -> torch.Tensor:
        shifted = x - torch.max(x)
        exp = torch.exp(shifted)
        return exp / (torch.sum(exp) + 1e-12)
