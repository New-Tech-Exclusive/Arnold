"""Tier 9 — The Generation Interface (Runtime Loop).

Orchestrates what actually happens during a conversation: the per-turn
processing pipeline and the between-turn partial updates.
"""

from __future__ import annotations

import torch

from .genome import Genome
from .tensor import DTYPE
from .types_ import (
    CortexRegion,
    HebbianTrace,
    MoodState,
    PersonalityVector,
    ContextualizedInput,
    RegionActivation,
)


class GenerationInterface:
    """The runtime loop that runs during conversation."""

    def __init__(self, genome: Genome, rng: object | None = None) -> None:
        self._genome = genome

    # ------------------------------------------------------------------
    # Token sampling
    # ------------------------------------------------------------------

    def sample_token(
        self,
        logits: torch.Tensor,
        mood: MoodState,
        novelty_score: float,
    ) -> int:
        """Sample a token from combined cortex logits.

        Dynamic temperature from mood and novelty:
          temp = base + valence_boost + novelty_boost
        """
        gen = self._genome.generation
        nov = self._genome.novelty

        # Dynamic temperature
        temp = gen.base_temperature
        temp += mood.valence * 0.1           # positive mood → slightly warmer
        temp += novelty_score * nov.temperature_boost  # novelty → more exploratory
        temp = float(max(min(temp, gen.max_temperature), gen.min_temperature))

        # Softmax with temperature
        probs = self._softmax(logits / temp)

        # Sample
        token_id = int(torch.multinomial(probs, num_samples=1).item())
        return token_id

    def generate_sequence(
        self,
        initial_logits: torch.Tensor,
        mood: MoodState,
        novelty_score: float,
        max_tokens: int = 64,
        eos_token: int | None = None,
        brainstem=None,
    ) -> list[int]:
        """Generate a sequence of tokens.

        Each step the logits are advanced by mixing the current distribution
        with the brainstem's co-occurrence signal for the last sampled token,
        preventing the distribution from collapsing to a single repeated token.
        """
        tokens: list[int] = []
        logits = initial_logits.detach().clone()

        for _ in range(max_tokens):
            token = self.sample_token(logits, mood, novelty_score)
            tokens.append(token)

            if eos_token is not None and token == eos_token:
                break

            # Advance logits using brainstem co-occurrence if available,
            # otherwise fall back to small random perturbation.
            if brainstem is not None:
                emb = brainstem.token_embeddings[token]           # (embed_dim,)
                # Project through co-occurrence → syntax → output_projection
                # to get next-token signal in embed space, then re-score vocab.
                hidden = torch.relu(emb @ brainstem.cooccurrence_weights)  # (bs_hidden,)
                recon = hidden @ brainstem.output_projection               # (embed_dim,)
                # Dot each vocab embedding against the predicted next embedding.
                next_signal = brainstem.token_embeddings @ recon            # (vocab_size,)
                # Mix: retain 60% of current context, add 40% next-token signal.
                logits = 0.60 * logits + 0.40 * next_signal
            else:
                logits = logits + torch.randn_like(logits, dtype=DTYPE) * 0.05

        return tokens

    # ------------------------------------------------------------------
    # Between-turn partial update
    # ------------------------------------------------------------------

    def partial_weight_update(
        self,
        cortex,  # Cortex instance
        traces: list[HebbianTrace],
        reinforcement_strength: float,
        plasticity_rates: dict[CortexRegion, float],
        ewc_scalars: dict[CortexRegion, torch.Tensor],
    ) -> bool:
        """Small, bounded, targeted weight update between turns.

        Only fires when reinforcement signal exceeds the immediate-update threshold.
        Returns True if an update was applied.
        """
        params = self._genome.reinforcement
        if abs(reinforcement_strength) < params.immediate_update_threshold:
            return False

        magnitude = params.immediate_update_magnitude
        sign = 1.0 if reinforcement_strength > 0 else -1.0

        for trace in traces:
            region = trace.region
            if region not in cortex.regions:
                continue

            base_lr = plasticity_rates.get(region, 0.01)

            # Apply EWC attenuation per-unit
            ewc = ewc_scalars.get(region)

            effective_lr = base_lr * magnitude * sign

            # Scale trace strengths by EWC protection
            if ewc is not None and int(ewc.numel()) > 0:
                n_post = int(trace.post_indices.numel())
                for i_idx, pre_i_t in enumerate(trace.pre_indices.tolist()):
                    pre_i = int(pre_i_t)
                    for j_idx, post_j_t in enumerate(trace.post_indices.tolist()):
                        post_j = int(post_j_t)
                        flat = i_idx * n_post + j_idx
                        if flat < int(trace.activation_strengths.numel()):
                            prot = float(ewc[post_j]) if post_j < int(ewc.numel()) else 0.0
                            dampened_lr = effective_lr * (1.0 - prot)
                            s = float(trace.activation_strengths[flat])
                            if pre_i < cortex.regions[region].hidden_dim and \
                               post_j < cortex.regions[region].hidden_dim:
                                cortex.regions[region].W_hidden[pre_i, post_j] += dampened_lr * s
            else:
                cortex.regions[region].hebbian_update(trace, abs(effective_lr))

        return True

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _softmax(x: torch.Tensor) -> torch.Tensor:
        shifted = x - torch.max(x)
        exp = torch.exp(shifted)
        return exp / (torch.sum(exp) + 1e-12)
