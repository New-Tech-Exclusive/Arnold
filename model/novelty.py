"""Tier 5 — The Novelty Detector (Surprise Scoring System).

Runs inline on every incoming message before the cortex processes it.
Compares incoming input against current cortex weights to calculate how
surprising the input is relative to what the model has already learned.
"""

from __future__ import annotations

import math
import torch

from .genome import Genome
from .types_ import CortexRegion, StructuredRepresentation


class NoveltyDetector:
    """Scores how novel / surprising an input is relative to learned weights."""

    def __init__(self, genome: Genome) -> None:
        self._genome = genome
        self._params = genome.novelty
        # Running EMA of raw prediction errors — prevents novelty saturating at 1.0
        # when untrained weights produce uniformly large errors.
        self._error_ema: float = 2.0

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
        raw_errors: list[float] = []

        for region, act in activations.items():
            if region == CortexRegion.OCCIPITAL:
                continue  # dormant

            # --- Prediction-error confidence (reliable from day 1) ---
            # Normalize by running EMA baseline so untrained models (large uniform
            # errors) don't saturate novelty at 1.0 and pin NE at max forever.
            raw_err = max(act.prediction_error, 0.0)
            raw_errors.append(raw_err)
            normalized_err = raw_err / (self._error_ema + 1e-6)
            err_confidence = float(math.exp(-normalized_err))

            # --- Logit-based confidence (grows meaningful as W_out trains) ---
            logits = act.logits  # (vocab_size,)
            logits = torch.nan_to_num(logits, nan=0.0, posinf=10.0, neginf=-10.0)
            probs = self._softmax(logits / self._params.temperature)
            if len(token_ids) > 0:
                valid_ids = token_ids[token_ids < probs.shape[0]]
                logit_confidence = float(probs[valid_ids].mean()) if int(valid_ids.numel()) > 0 else 0.0
            else:
                logit_confidence = 0.0

            # Blend: prediction error signal is strong from day 1;
            # logit confidence contribution grows as W_out trains up.
            confidence = 0.7 * err_confidence + 0.3 * logit_confidence
            prediction_confidences.append(confidence)

        # Update EMA baseline from this call's prediction errors
        if raw_errors:
            batch_mean = sum(raw_errors) / len(raw_errors)
            self._error_ema = 0.90 * self._error_ema + 0.10 * batch_mean

        if not prediction_confidences:
            return 0.5  # no predictions → moderate novelty (not max, to avoid NE spike)

        # Filter NaN values before averaging
        valid = [c for c in prediction_confidences if c == c]
        if not valid:
            return 0.5
        mean_confidence = float(sum(valid) / len(valid))

        # Novelty = 1 - confidence  (clamped)
        novelty = float(max(min(1.0 - mean_confidence, 1.0), 0.0))
        return novelty

    @staticmethod
    def _softmax(x: torch.Tensor) -> torch.Tensor:
        shifted = x - torch.max(x)
        exp = torch.exp(shifted)
        return exp / (torch.sum(exp) + 1e-12)
