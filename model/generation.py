"""Tier 9 — The Generation Interface (Runtime Loop).

Orchestrates what actually happens during a conversation: the per-turn
processing pipeline and the between-turn partial updates.

Implements:
  - Recurrent hidden state (carry-forward across tokens)
  - Context buffer with dot-product attention (sliding window)
  - In-loop repetition penalty
  - Entropy-adaptive sampling
  - Top-k / top-p (nucleus) sampling
  - Soft prompt anchoring for topic coherence
"""

from __future__ import annotations

import math
from collections import deque

import torch

from .genome import Genome
from .tensor import DTYPE, get_device
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
        self._device = get_device()

    # ------------------------------------------------------------------
    # Token sampling  (top-k, top-p, entropy-adaptive, rep penalty)
    # ------------------------------------------------------------------

    def sample_token(
        self,
        logits: torch.Tensor,
        mood: MoodState,
        novelty_score: float,
        generated_so_far: list[int] | None = None,
        norepinephrine: float | None = None,
    ) -> int:
        """Sample a token with full quality pipeline.

        1. Apply repetition penalty on already-generated tokens.
        2. Dynamic temperature from norepinephrine (or mood fallback) + novelty.
        3. Entropy-adaptive narrowing.
        4. Top-k filtering.
        5. Top-p (nucleus) filtering.
        6. Multinomial sample.
        """
        gen = self._genome.generation
        nov = self._genome.novelty

        logits = logits.clone()

        # --- 1. In-loop repetition penalty ---
        if generated_so_far and gen.repetition_penalty != 1.0:
            seen = set(generated_so_far)
            for tok_id in seen:
                if 0 <= tok_id < logits.shape[0]:
                    if logits[tok_id] > 0:
                        logits[tok_id] /= gen.repetition_penalty
                    else:
                        logits[tok_id] *= gen.repetition_penalty

        # --- 2. Dynamic temperature ---
        # Norepinephrine directly controls temperature: high NE → higher temp (more exploration)
        temp = gen.base_temperature
        # Sanitize inputs before using them
        if norepinephrine is not None and norepinephrine == norepinephrine:  # NaN guard
            # Gentler effect: NE range 0..1 → temp delta of -0.1..+0.14
            temp += (float(max(min(norepinephrine, 1.0), 0.0)) - 0.3) * 0.2
        else:
            temp += mood.valence * 0.05
        if novelty_score == novelty_score:  # NaN guard
            temp += float(max(min(novelty_score, 1.0), 0.0)) * nov.temperature_boost
        temp = float(max(min(temp, gen.max_temperature), gen.min_temperature))

        # Guard against NaN/Inf logits before sampling
        if not torch.isfinite(logits).all():
            logits = logits.nan_to_num(0.0).clamp(-50.0, 50.0)

        scaled = logits / max(temp, 1e-8)

        # --- 3. Entropy-adaptive narrowing ---
        probs = self._softmax(scaled)
        entropy = -float(torch.sum(probs * torch.log(probs + 1e-12)).item())

        effective_top_k = gen.top_k
        if entropy > gen.entropy_high_threshold:
            # Model is confused → narrow to top-5 (avoids collapsing to top-3
            # which is effectively random on untrained uniform logits)
            effective_top_k = min(gen.top_k, 5)
        elif entropy < gen.entropy_low_threshold:
            # Model is confident → sample normally
            effective_top_k = gen.top_k

        # --- 4. Top-k filtering ---
        if effective_top_k > 0 and effective_top_k < logits.shape[0]:
            top_vals, _ = torch.topk(scaled, effective_top_k)
            threshold = top_vals[-1]
            scaled[scaled < threshold] = float('-inf')

        # --- 5. Top-p (nucleus) filtering ---
        if gen.top_p < 1.0:
            sorted_logits, sorted_indices = torch.sort(scaled, descending=True)
            cum_probs = torch.cumsum(self._softmax(sorted_logits), dim=0)
            cutoff_mask = cum_probs > gen.top_p
            # Shift mask right so at least one token is always kept
            cutoff_mask[1:] = cutoff_mask[:-1].clone()
            cutoff_mask[0] = False
            sorted_logits[cutoff_mask] = float('-inf')
            # Scatter back
            scaled.scatter_(0, sorted_indices, sorted_logits)

        # --- 6. Final softmax and sample ---
        probs = self._softmax(scaled)
        token_id = int(torch.multinomial(probs, num_samples=1).item())
        return token_id

    # ------------------------------------------------------------------
    # Sequence generation with recurrent state + context attention
    # ------------------------------------------------------------------

    def generate_sequence(
        self,
        initial_logits: torch.Tensor,
        mood: MoodState,
        novelty_score: float,
        max_tokens: int = 64,
        eos_token: int | None = None,
        brainstem=None,
        cortex=None,
        anchor_embeddings: torch.Tensor | None = None,
        norepinephrine: float | None = None,
    ) -> list[int]:
        """Generate a token sequence with recurrent hidden state and context attention.

        Key improvements over the previous version:
        1. Recurrent hidden state carried across every token.
        2. Context buffer (deque) with dot-product attention over recent states.
        3. Repetition penalty applied at every step inside the loop.
        4. Entropy-adaptive + top-k + top-p sampling.
        5. Soft prompt anchor keeps generation on-topic.
        6. Norepinephrine directly controls generation temperature.
        """
        gen = self._genome.generation
        device = self._device

        tokens: list[int] = []
        logits = initial_logits.detach().clone().to(device)

        # Bootstrap initial logits from brainstem's pretrained vocabulary.
        # The cortex W_out is random at birth; it adds only a tiny correction
        # that grows meaningful over thousands of training turns.
        if brainstem is not None and anchor_embeddings is not None:
            mean_emb = anchor_embeddings.mean(dim=0).to(device)          # (embed_dim,)
            co_h = torch.relu(mean_emb @ brainstem.cooccurrence_weights.to(device))  # (hidden_dim,)
            base_recon = co_h @ brainstem.output_projection.to(device)    # (embed_dim,)
            base_logits = brainstem.token_embeddings.to(device) @ base_recon  # (vocab_size,)
            # Cortex logits are a small learned correction on top of brainstem base
            logits = base_logits + 0.05 * logits

        # --- Recurrent hidden state ---
        # Use brainstem_hidden as the recurrent dimension
        if brainstem is not None:
            hidden_dim = brainstem.cooccurrence_weights.shape[1]  # brainstem_hidden
        else:
            hidden_dim = logits.shape[0]

        hidden_state = torch.zeros(hidden_dim, dtype=DTYPE, device=device)

        # --- Context buffer for sliding-window attention ---
        ctx_buf: deque[torch.Tensor] = deque(maxlen=gen.context_buffer_size)

        # --- Soft prompt anchor ---
        anchor: torch.Tensor | None = None
        if anchor_embeddings is not None and brainstem is not None:
            # Project mean of input embeddings to vocab-sized anchor bias
            mean_emb = anchor_embeddings.mean(dim=0)  # (embed_dim,)
            anchor = brainstem.token_embeddings @ mean_emb  # (vocab_size,)

        # Get recurrent weight from cortex (use first active region)
        W_rec: torch.Tensor | None = None
        if cortex is not None:
            for region in cortex._genome.topology.active_regions:
                if region in cortex.regions:
                    W_rec = cortex.regions[region].W_recurrent.to(device)
                    break

        for _ in range(max_tokens):
            # --- Apply soft prompt anchor ---
            step_logits = logits.clone()
            if anchor is not None:
                step_logits = step_logits + gen.anchor_weight * anchor

            # --- Sample token ---
            token = self.sample_token(
                step_logits, mood, novelty_score, generated_so_far=tokens,
                norepinephrine=norepinephrine,
            )
            tokens.append(token)

            if eos_token is not None and token == eos_token:
                break

            # --- Update recurrent hidden state ---
            if brainstem is not None:
                emb = brainstem.token_embeddings[token].to(device)  # (embed_dim,)
                co_hidden = torch.relu(emb @ brainstem.cooccurrence_weights.to(device))  # (hidden_dim,)

                # Recurrent update: h = tanh(mix * W_rec @ h + (1-mix) * co_hidden)
                rec_input = co_hidden
                if W_rec is not None:
                    rec_input = gen.recurrent_mix * (hidden_state @ W_rec) + (1.0 - gen.recurrent_mix) * co_hidden
                hidden_state = torch.tanh(rec_input)

                # --- Context buffer attention ---
                ctx_buf.append(hidden_state.detach().clone())

                if len(ctx_buf) > 1:
                    stack = torch.stack(list(ctx_buf))  # (K, hidden_dim)
                    scores = hidden_state @ stack.T / math.sqrt(hidden_dim)
                    attn = self._softmax(scores)
                    context_vec = attn @ stack
                    hidden_state = hidden_state + gen.context_attention_weight * context_vec

                # --- Produce next logits from hidden state ---
                recon = hidden_state @ brainstem.output_projection.to(device)  # (embed_dim,)
                next_signal = brainstem.token_embeddings.to(device) @ recon  # (vocab_size,)

                # Blend: 50% existing context, 50% recurrent next-token signal
                logits = 0.50 * logits + 0.50 * next_signal
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
