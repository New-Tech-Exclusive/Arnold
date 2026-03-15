"""Tier 5 — The Novelty Detector (Surprise Scoring System).

Runs inline on every incoming message before the transformer processes it.
Compares incoming input against current transformer weights to calculate how
surprising the input is relative to what the model has already learned.
"""

from __future__ import annotations

import math
import torch

from .genome import Genome
from .types_ import TransformerRegion, StructuredRepresentation


class SurpriseDetector:
    """Scores how novel / surprising an input is relative to learned weights."""

    def __init__(self, genome: Genome) -> None:
        self._genome = genome
        self._params = genome.surprise
        # Running EMA of raw prediction errors — prevents surprise saturating at 1.0
        # when untrained weights produce uniformly large errors.
        # EMA baseline for raw prediction error; starts empty and seeds on first pass
        self._error_ema: dict[TransformerRegion, float] = {}

    def score(
        self,
        structured: StructuredRepresentation,
        transformer_forward_readonly,  # callable: embeddings → dict[region, RegionActivation]
    ) -> float:
        """Compute surprise score ∈ [0, 1] for the given input.

        Activates the transformer in *read-only* mode.  Measures how confidently
        existing weights predicted this input.  Low confidence = high surprise.

        Args:
            structured: Encoder output.
            transformer_forward_readonly: A function that takes (seq_len, embed_dim)
                embeddings and returns region activations without recording traces.

        Returns:
            Scalar surprise score between 0.0 and 1.0.
        """
        embeddings = structured.embeddings  # (seq_len, embed_dim)
        token_ids = structured.token_ids     # (seq_len,)

        # Read-only forward pass through transformer
        activations = transformer_forward_readonly(embeddings)

        # For each active region, measure prediction confidence
        prediction_confidences: list[float] = []
        raw_errors: dict[TransformerRegion, float] = {}

        for region, act in activations.items():
            if region == TransformerRegion.OCCIPITAL:
                continue  # dormant

            # --- Prediction-error confidence (reliable from day 1) ---
            # Normalize by running EMA baseline so untrained models (large uniform
            # errors) don't saturate surprise at 1.0 and pin NE at max forever.
            raw_err = max(act.prediction_error, 0.0)
            raw_errors[region] = raw_err

            # If this is the first time we see this region, initialise EMA
            if region not in self._error_ema:
                self._error_ema[region] = max(raw_err, 1.0)
                normalized_err = 0.5
            else:
                normalized_err = raw_err / (self._error_ema[region] + 1e-6)
            err_confidence = float(math.exp(-normalized_err))

            # --- Logit-based confidence (grows meaningful as W_out trains) ---
            logits = act.logits  # (vocab_size,)
            logits = torch.nan_to_num(logits, nan=0.0, posinf=10.0, neginf=-10.0)
            probs = self._softmax(logits / self._params.temperature)
            # detach before taking mean to avoid gradients leaking / warnings
            if len(token_ids) > 0:
                valid_ids = token_ids[token_ids < probs.shape[0]]
                if int(valid_ids.numel()) > 0:
                    logit_confidence = float(probs[valid_ids].detach().mean().item())
                else:
                    logit_confidence = 0.0
            else:
                logit_confidence = 0.0

            # Blend: prediction error signal is strong from day 1;
            # logit confidence contribution grows as W_out trains up.
            confidence = 0.7 * err_confidence + 0.3 * logit_confidence
            prediction_confidences.append(confidence)

        # Update EMA baseline from this call's prediction errors
        for region, batch_mean in raw_errors.items():
            prev = self._error_ema.get(region, 2.0)
            self._error_ema[region] = 0.90 * prev + 0.10 * batch_mean

        if not prediction_confidences:
            return 0.5  # no predictions → moderate surprise (not max, to avoid NE spike)

        # Filter NaN values before averaging
        valid = [c for c in prediction_confidences if c == c]
        if not valid:
            return 0.5
        mean_confidence = float(sum(valid) / len(valid))

        # Novelty = 1 - confidence  (clamped)
        surprise = float(max(min(1.0 - mean_confidence, 1.0), 0.0))
        return surprise

    @staticmethod
    def _softmax(x: torch.Tensor) -> torch.Tensor:
        shifted = x - torch.max(x)
        exp = torch.exp(shifted)
        return exp / (torch.sum(exp) + 1e-12)
