"""Tier 3 - The Transformer (Plastic Hebbian Layers).

Implements:
  - Predictive coding (Karl Friston's Free Energy Principle)
  - Oja's Rule (self-normalising Hebbian updates)
  - BCM sliding threshold per neuron
  - Homeostatic synaptic scaling
  - Neuromodulatory gain control
  - Sparse k-Winners-Take-All activations
  - Recurrent hidden state for generation
"""

from __future__ import annotations

import torch

from .genome import Genome
from .tensor import DTYPE, get_device
from .types_ import (
    ContextualizedInput,
    TransformerRegion,
    HebbianTrace,
    PersonalityVector,
    RegionActivation,
)


class TransformerRegionModule:
    """One region of the transformer - a set of Hebbian weight matrices.

    Implements:
      - Predictive coding: maintains prediction of next input, propagates error
      - Oja's Rule (self-normalising Hebbian updates)
      - BCM sliding threshold per neuron
      - Sparse k-Winners-Take-All (only top 2% of neurons fire)
      - Homeostatic synaptic scaling
      - Neuromodulatory gain control
      - Recurrent hidden state for generation
    """

    # k-WTA sparsity: only top K_PERCENT of hidden neurons fire per step
    K_PERCENT: float = 0.02

    def __init__(
        self,
        region: TransformerRegion,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        genome: Genome,
        rng: object | None = None,
    ) -> None:
        self.region = region
        self._genome = genome
        self._device = get_device()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        self.W_in = torch.randn((input_dim, hidden_dim), dtype=DTYPE, device=self._device) * 0.02
        self.W_in.requires_grad_(True)
        self.W_hidden = torch.randn((hidden_dim, hidden_dim), dtype=DTYPE, device=self._device) * 0.02
        self.W_hidden.requires_grad_(True)
        self.W_out = torch.randn((hidden_dim, output_dim), dtype=DTYPE, device=self._device) * 0.02
        self.W_out.requires_grad_(True)
        # Recurrent weight matrix for generation carry-forward
        self.W_recurrent = torch.randn((hidden_dim, hidden_dim), dtype=DTYPE, device=self._device) * 0.01
        self.W_recurrent.requires_grad_(True)
        # Layer norm on hidden activations stabilises logit magnitudes and
        # prevents vocabulary collapse (one token dominating all outputs).
        self._ln_scale = torch.ones(hidden_dim, dtype=DTYPE, device=self._device)
        self._ln_scale.requires_grad_(True)
        self._ln_bias = torch.zeros(hidden_dim, dtype=DTYPE, device=self._device)
        self._ln_bias.requires_grad_(True)

        self.last_fired: torch.Tensor = torch.zeros(hidden_dim, dtype=torch.int64, device=self._device)
        self._last_pre: torch.Tensor | None = None
        self._last_post: torch.Tensor | None = None

        # BCM sliding threshold — one per neuron (initialised from genome)
        self.theta = torch.full(
            (hidden_dim,), genome.bcm.theta_init, dtype=DTYPE, device=self._device,
        )

        # Running average activation for homeostatic scaling
        self._avg_activation = torch.full(
            (hidden_dim,), genome.homeostatic.target_rate, dtype=DTYPE, device=self._device,
        )
        self._forward_count = 0

        # Neuromodulatory gain (set externally before forward during generation)
        self._gain: float = 1.0

        # Predictive coding: prediction of next input
        self._prediction = torch.zeros(input_dim, dtype=DTYPE, device=self._device)
        self._last_prediction_error: torch.Tensor | None = None

        # k-WTA: number of winners
        self._k = max(1, int(hidden_dim * self.K_PERCENT))

    def set_gain(self, gain: float) -> None:
        """Set neuromodulatory gain (called by reinforcement system)."""
        self._gain = gain

    def forward(self, x: torch.Tensor, routing_weight: float = 1.0, readonly: bool = False) -> RegionActivation:
        """Forward pass with predictive coding and sparse k-WTA.

        1. Compute prediction error (actual input - prediction).
        2. Only error propagates through the processing pipeline.
        3. Apply k-Winners-Take-All for sparse representations.
        4. Update prediction toward actual input (skipped when readonly=True).
        """
        pooled = x.mean(dim=0).nan_to_num(0.0)

        # --- Predictive coding: compute error ---
        pc = self._genome.predictive_coding
        prediction_error = pooled - self._prediction
        self._last_prediction_error = prediction_error.clone()

        # Only the error propagates — not the raw input
        signal = prediction_error * pc.error_gain * routing_weight

        # Update prediction toward actual input (skip in read-only mode)
        if not readonly:
            self._prediction = self._prediction + pc.prediction_lr * prediction_error.detach()

        pre = signal @ self.W_in
        hidden = torch.relu(pre)
        post = torch.relu(hidden @ self.W_hidden)

        # Apply neuromodulatory gain — sharpens or diffuses activations
        post = post * self._gain

        # --- k-Winners-Take-All: only top k% of neurons fire ---
        if self._k < post.shape[0]:
            topk_vals, topk_idx = torch.topk(post, self._k)
            sparse_post = torch.zeros_like(post)
            sparse_post[topk_idx] = topk_vals
            post = sparse_post

        # Layer-norm before vocab projection to keep logit magnitudes balanced.
        post_norm = torch.layer_norm(post, [post.shape[-1]], self._ln_scale, self._ln_bias)

        # BCM: use per-neuron sliding threshold instead of fixed value
        fired_mask = post > self.theta
        fired_indices = torch.where(fired_mask)[0]

        self.last_fired[fired_indices] = 0
        self.last_fired[~fired_mask] += 1

        self._last_pre = hidden
        self._last_post = post

        # Update BCM theta: track running average of post-synaptic activity
        bcm = self._genome.bcm
        self.theta += bcm.theta_lr * (post.detach() - self.theta)

        # Track average activation for homeostatic scaling
        self._avg_activation = 0.99 * self._avg_activation + 0.01 * post.detach()
        self._forward_count += 1

        logits = post_norm @ self.W_out
        # Guard against NaN/Inf in logits — clamp to a safe range
        logits = logits.nan_to_num(0.0).clamp(-50.0, 50.0)
        return RegionActivation(
            region=self.region,
            logits=logits,
            hidden=post,
            fired_indices=fired_indices,
            prediction_error=self.prediction_error_magnitude,
        )

    @property
    def prediction_error_magnitude(self) -> float:
        """Mean absolute prediction error from last forward pass."""
        if self._last_prediction_error is None:
            return 1.0
        return float(self._last_prediction_error.abs().mean().item())

    def homeostatic_scale(self) -> None:
        """Multiplicatively scale W_in to maintain target activity rate."""
        params = self._genome.homeostatic
        target = params.target_rate
        factor = params.scaling_factor

        # Per-neuron: if avg below target, scale up; if above, scale down
        ratio = target / (self._avg_activation + 1e-8)
        # Clamp the adjustment to avoid wild jumps
        adjustment = 1.0 + factor * (ratio - 1.0).clamp(-1.0, 1.0)
        with torch.no_grad():
            self.W_in *= adjustment.unsqueeze(0)  # broadcast over input_dim

    def get_hebbian_trace(self) -> HebbianTrace | None:
        if self._last_pre is None or self._last_post is None:
            return None
        # BCM: use per-neuron theta instead of fixed threshold
        pre_fired = torch.where(self._last_pre > self.theta[:self._last_pre.shape[0]])[0]
        post_fired = torch.where(self._last_post > self.theta)[0]

        if pre_fired.numel() == 0 or post_fired.numel() == 0:
            return None

        pre_strengths = self._last_pre[pre_fired]
        post_strengths = self._last_post[post_fired]
        strengths = torch.outer(pre_strengths, post_strengths).reshape(-1)

        return HebbianTrace(
            region=self.region,
            pre_indices=pre_fired,
            post_indices=post_fired,
            activation_strengths=strengths,
        )

    # ------------------------------------------------------------------
    # Gradient helpers
    # ------------------------------------------------------------------

    def parameters(self) -> list[torch.Tensor]:
        """Return weight tensors that should receive gradient updates."""
        return [
            self.W_in,
            self.W_hidden,
            self.W_out,
            self.W_recurrent,
            self._ln_scale,
            self._ln_bias,
        ]

    def hebbian_update(self, trace: HebbianTrace, learning_rate: float) -> None:
        """Oja's Rule: Δw = η · post · (pre - post · w).

        Self-normalising — extracts principal components without unbounded growth.
        BCM modulation: only strengthen when post > theta (LTP), weaken when
        post < theta (LTD).
        """
        params = self._genome.hebbian
        n_post = int(trace.post_indices.numel())
        if n_post == 0:
            return

        with torch.no_grad():
            for i_idx, pre_i_t in enumerate(trace.pre_indices.tolist()):
                pre_i = int(pre_i_t)
                if pre_i >= self.hidden_dim:
                    continue

                start = i_idx * n_post
                pre_strength_slice = trace.activation_strengths[start:start + n_post]

                for j_idx, post_j_t in enumerate(trace.post_indices.tolist()):
                    post_j = int(post_j_t)
                    if post_j >= self.hidden_dim:
                        continue

                    pre_val = float(pre_strength_slice[j_idx]) if j_idx < pre_strength_slice.numel() else 0.0
                    post_val = float(self._last_post[post_j]) if self._last_post is not None else abs(pre_val)
                    w = float(self.W_hidden[pre_i, post_j])

                    # Oja's Rule: Δw = η · post · (pre - post · w)
                    delta = learning_rate * post_val * (pre_val - post_val * w)

                    # BCM modulation: sign depends on post vs theta
                    theta_j = float(self.theta[post_j])
                    if post_val < theta_j:
                        delta = -abs(delta)  # LTD: weaken

                    self.W_hidden[pre_i, post_j] = max(
                        params.min_weight,
                        min(params.max_weight, w + delta),
                    )

    def hebbian_update_batch(self, traces: list[HebbianTrace], learning_rate: float) -> None:
        """Batch Oja's Rule update across multiple traces."""
        params = self._genome.hebbian

        with torch.no_grad():
            for trace in traces:
                if trace.region != self.region:
                    continue
                n_post = int(trace.post_indices.numel())
                for i_idx, pre_i_t in enumerate(trace.pre_indices.tolist()):
                    pre_i = int(pre_i_t)
                    for j_idx, post_j_t in enumerate(trace.post_indices.tolist()):
                        post_j = int(post_j_t)
                        flat_idx = i_idx * n_post + j_idx
                        if flat_idx < trace.activation_strengths.numel():
                            pre_val = float(trace.activation_strengths[flat_idx])
                            w = float(self.W_hidden[pre_i, post_j])
                            post_val = abs(pre_val)
                            # Oja's Rule
                            delta = learning_rate * post_val * (pre_val - post_val * w)
                            theta_j = float(self.theta[post_j]) if post_j < self.theta.numel() else 0.3
                            if post_val < theta_j:
                                delta = -abs(delta)
                            if pre_i < self.hidden_dim and post_j < self.hidden_dim:
                                self.W_hidden[pre_i, post_j] += delta

        self.W_hidden.clamp_(params.min_weight, params.max_weight)

    def apply_weight_decay(self, decay_rate: float) -> None:
        with torch.no_grad():
            self.W_in *= (1.0 - decay_rate)
            self.W_hidden *= (1.0 - decay_rate)
            self.W_out *= (1.0 - decay_rate)

    def prune(self, age_threshold: int, weight_threshold: float) -> int:
        stale_mask = self.last_fired > age_threshold
        weak_mask = self.W_hidden.abs().max(dim=1).values < weight_threshold
        prune_mask = stale_mask & weak_mask
        pruned_count = int(prune_mask.sum().item())

        if pruned_count > 0:
            with torch.no_grad():
                self.W_hidden[prune_mask, :] = 0.0
                self.W_hidden[:, prune_mask] = 0.0
                self.W_in[:, prune_mask] = 0.0
                self.W_out[prune_mask, :] = 0.0

        return pruned_count


class InterRegionWiring:
    """Hebbian connections between transformer regions (Tier 3E).

    Supports STDP: tracks firing order between regions for causal
    strengthening (pre→post) and anti-causal weakening (post→pre).
    """

    def __init__(self, genome: Genome, rng: object | None = None) -> None:
        self._genome = genome
        self._device = get_device()
        hidden = genome.topology.transformer_hidden
        active = genome.topology.active_regions

        self.connections: dict[tuple[TransformerRegion, TransformerRegion], torch.Tensor] = {}
        self.highway_strengths: dict[tuple[TransformerRegion, TransformerRegion], float] = {}
        # STDP: track firing order (higher = fired earlier in sequence)
        self._firing_order: dict[TransformerRegion, int] = {}

        for i, r1 in enumerate(active):
            for r2 in active[i + 1:]:
                key = (r1, r2)
                self.connections[key] = torch.randn((hidden, hidden), dtype=DTYPE, device=self._device) * 0.01
                self.highway_strengths[key] = 0.0

    def record_firing_order(self, region: TransformerRegion, order: int) -> None:
        """Record when a region fired relative to others (STDP)."""
        self._firing_order[region] = order

    def modulate(self, activations: dict[TransformerRegion, RegionActivation]) -> dict[TransformerRegion, torch.Tensor]:
        modulations: dict[TransformerRegion, torch.Tensor] = {}
        hidden_dim = self._genome.topology.transformer_hidden
        stdp = self._genome.stdp

        for region in self._genome.topology.active_regions:
            modulations[region] = torch.zeros(hidden_dim, dtype=DTYPE, device=self._device)

        for (r1, r2), W in self.connections.items():
            if r1 not in activations or r2 not in activations:
                continue
            h1 = activations[r1].hidden
            h2 = activations[r2].hidden

            # STDP: scale modulation based on firing order
            order1 = self._firing_order.get(r1, 0)
            order2 = self._firing_order.get(r2, 0)
            dt = float(order1 - order2)  # negative = r1 fired first (causal)
            if dt < 0:
                # r1 fired before r2 → strengthen r1→r2 pathway
                stdp_scale = stdp.a_plus * float(torch.exp(torch.tensor(dt / stdp.tau_plus)))
            elif dt > 0:
                # r2 fired before r1 → weaken this direction
                stdp_scale = -stdp.a_minus * float(torch.exp(torch.tensor(-dt / stdp.tau_minus)))
            else:
                stdp_scale = 0.0

            mod_1_to_2 = h1 @ W
            mod_2_to_1 = h2 @ W.T

            base_scale = 0.1
            modulations[r1] = modulations[r1] + mod_2_to_1 * (base_scale + stdp_scale)
            modulations[r2] = modulations[r2] + mod_1_to_2 * (base_scale - stdp_scale)

            coactivation = float(torch.dot(h1.detach(), h2.detach()) / (torch.linalg.norm(h1.detach()) * torch.linalg.norm(h2.detach()) + 1e-12))
            self.highway_strengths[(r1, r2)] = 0.99 * self.highway_strengths[(r1, r2)] + 0.01 * coactivation

        return modulations

    def hebbian_update(self, activations: dict[TransformerRegion, RegionActivation], learning_rate: float) -> None:
        """Oja's Rule for inter-region connections."""
        params = self._genome.hebbian
        with torch.no_grad():
            for (r1, r2), W in self.connections.items():
                if r1 not in activations or r2 not in activations:
                    continue
                h1 = activations[r1].hidden
                h2 = activations[r2].hidden
                # Oja: Δw = η · h2 · (h1 - h2 · W)^T  (column-wise normalisation)
                outer = torch.outer(h1, h2)
                norm_term = torch.outer(h2 @ W.T, h2)
                delta = learning_rate * (outer - norm_term)
                W += delta
                W.clamp_(params.min_weight, params.max_weight)

    def get_highway_map(self) -> dict[tuple[TransformerRegion, TransformerRegion], float]:
        return dict(self.highway_strengths)


class TransformerVoter:
    """Combines region logits via personality-weighted vote."""

    def __init__(self, genome: Genome) -> None:
        self._genome = genome
        self._device = get_device()

    def vote(
        self,
        activations: dict[TransformerRegion, RegionActivation],
        personality: PersonalityVector,
        mood_valence: float = 0.0,
    ) -> torch.Tensor:
        voting = self._genome.voting
        vocab = self._genome.topology.vocab_size

        region_weights: dict[TransformerRegion, float] = {}
        for region in self._genome.topology.active_regions:
            region_weights[region] = voting.base_weight

        for region, trait_name, multiplier in voting.region_trait_map:
            if region in region_weights:
                trait_val = getattr(personality, trait_name, 0.5)
                region_weights[region] += multiplier * trait_val

        total = sum(region_weights.values()) or 1.0
        for r in region_weights:
            region_weights[r] /= total

        combined = torch.zeros(vocab, dtype=DTYPE, device=self._device)
        for region, act in activations.items():
            if region in region_weights:
                combined += region_weights[region] * act.logits

        return combined

    def compute_region_weights(self, personality: PersonalityVector) -> dict[TransformerRegion, float]:
        voting = self._genome.voting
        weights: dict[TransformerRegion, float] = {}
        for region in self._genome.topology.active_regions:
            weights[region] = voting.base_weight
        for region, trait_name, multiplier in voting.region_trait_map:
            if region in weights:
                weights[region] += multiplier * getattr(personality, trait_name, 0.5)
        total = sum(weights.values()) or 1.0
        return {r: w / total for r, w in weights.items()}


class Transformer:
    """The complete Tier 3 transformer with all regions, wiring, and voting."""

    def __init__(self, genome: Genome, rng: object | None = None) -> None:
        self._genome = genome
        self._device = get_device()
        topo = genome.topology

        self.regions: dict[TransformerRegion, TransformerRegionModule] = {}
        for region in topo.regions:
            self.regions[region] = TransformerRegionModule(
                region=region,
                input_dim=topo.embed_dim,
                hidden_dim=topo.transformer_hidden,
                output_dim=topo.vocab_size,
                genome=genome,
                rng=rng,
            )

        self.wiring = InterRegionWiring(genome, rng)
        self.voter = TransformerVoter(genome)

    def forward(
        self,
        ctx_input: ContextualizedInput,
        personality: PersonalityVector,
        routing_weights: dict[TransformerRegion, float] | None = None,
    ) -> tuple[torch.Tensor, list[HebbianTrace], dict[TransformerRegion, RegionActivation]]:
        embeddings = ctx_input.structured.embeddings

        # Record firing order for STDP (based on region activation magnitude)
        order_counter = 0

        activations: dict[TransformerRegion, RegionActivation] = {}
        for region in self._genome.topology.active_regions:
            route_w = routing_weights.get(region, 1.0) if routing_weights else 1.0
            activations[region] = self.regions[region].forward(embeddings, route_w)
            self.wiring.record_firing_order(region, order_counter)
            order_counter += 1

        modulations = self.wiring.modulate(activations)

        for region in self._genome.topology.active_regions:
            mod = modulations.get(region)
            if mod is not None:
                refined_hidden = torch.relu(activations[region].hidden + mod)
                refined_logits = refined_hidden @ self.regions[region].W_out
                activations[region] = RegionActivation(
                    region=region,
                    logits=refined_logits,
                    hidden=refined_hidden,
                    fired_indices=torch.where(
                        refined_hidden > self.regions[region].theta
                    )[0],
                )

        traces: list[HebbianTrace] = []
        for region in self._genome.topology.active_regions:
            trace = self.regions[region].get_hebbian_trace()
            if trace is not None:
                traces.append(trace)

        combined_logits = self.voter.vote(activations, personality, ctx_input.mood.valence)
        return combined_logits, traces, activations

    def forward_readonly(self, embeddings: torch.Tensor) -> dict[TransformerRegion, RegionActivation]:
        """Read-only forward pass — does NOT update prediction state."""
        activations: dict[TransformerRegion, RegionActivation] = {}
        for region in self._genome.topology.active_regions:
            activations[region] = self.regions[region].forward(embeddings, readonly=True)
        return activations

    def get_weights(self) -> dict[TransformerRegion, torch.Tensor]:
        out: dict[TransformerRegion, torch.Tensor] = {}
        for region, mod in self.regions.items():
            out[region] = torch.cat([
                mod.W_in.reshape(-1),
                mod.W_hidden.reshape(-1),
                mod.W_out.reshape(-1),
                mod.W_recurrent.reshape(-1),
            ]).detach().clone()
        return out

    def set_weights(self, weights: dict[TransformerRegion, torch.Tensor]) -> None:
        for region, blob in weights.items():
            if region not in self.regions:
                continue
            mod = self.regions[region]
            b = blob.to(device=self._device, dtype=DTYPE)
            sizes = [
                mod.input_dim * mod.hidden_dim,
                mod.hidden_dim * mod.hidden_dim,
                mod.hidden_dim * mod.output_dim,
                mod.hidden_dim * mod.hidden_dim,  # W_recurrent
            ]
            expected = sum(sizes)
            if b.numel() == expected:
                parts = torch.split(b, sizes)
                mod.W_in = parts[0].reshape(mod.input_dim, mod.hidden_dim)
                mod.W_hidden = parts[1].reshape(mod.hidden_dim, mod.hidden_dim)
                mod.W_out = parts[2].reshape(mod.hidden_dim, mod.output_dim)
                mod.W_recurrent = parts[3].reshape(mod.hidden_dim, mod.hidden_dim)
            else:
                # Backward compat: old checkpoints without W_recurrent
                old_sizes = [
                    mod.input_dim * mod.hidden_dim,
                    mod.hidden_dim * mod.hidden_dim,
                    mod.hidden_dim * mod.output_dim,
                ]
                if b.numel() == sum(old_sizes):
                    parts = torch.split(b, old_sizes)
                    mod.W_in = parts[0].reshape(mod.input_dim, mod.hidden_dim)
                    mod.W_hidden = parts[1].reshape(mod.hidden_dim, mod.hidden_dim)
                    mod.W_out = parts[2].reshape(mod.hidden_dim, mod.output_dim)
                else:
                    raise RuntimeError(
                        f"Transformer weight blob for {region.value} has {b.numel()} elements, "
                        f"expected {expected} (new) or {sum(old_sizes)} (legacy)"
                    )

    def get_inter_region_weights(self) -> dict[tuple[TransformerRegion, TransformerRegion], torch.Tensor]:
        return {k: v.detach().clone() for k, v in self.wiring.connections.items()}

    def set_inter_region_weights(self, weights: dict[tuple[TransformerRegion, TransformerRegion], torch.Tensor]) -> None:
        for key, arr in weights.items():
            if key in self.wiring.connections:
                self.wiring.connections[key] = arr.to(device=self._device, dtype=DTYPE).clone()

    def set_neuromodulatory_gain(self, arousal: float = 0.0, surprise: float = 0.0, norepinephrine: float = 0.3) -> None:
        """Neuromodulatory gain: norepinephrine drives sharpening/diffusion.

        High NE → sharpen activations (focused processing).
        Falls back to arousal+surprise if NE not set.
        """
        # Sanitize inputs
        if norepinephrine != norepinephrine: norepinephrine = 0.3
        if surprise != surprise: surprise = 0.0
        # Gentler modulation: NE=1.0, surprise=1.0 → gain=1.4 (was 1.8)
        gain = 1.0 + 0.3 * norepinephrine + 0.1 * surprise
        gain = max(0.5, min(gain, 1.5))
        for mod in self.regions.values():
            mod.set_gain(gain)

    def homeostatic_scale_all(self) -> None:
        """Run homeostatic synaptic scaling on all regions."""
        for mod in self.regions.values():
            mod.homeostatic_scale()

    def mean_prediction_error(self) -> float:
        """Mean prediction error magnitude across all active regions."""
        errors = []
        for region in self._genome.topology.active_regions:
            if region in self.regions:
                val = self.regions[region].prediction_error_magnitude
                if val == val:  # filter NaN
                    errors.append(val)
        return float(sum(errors) / max(len(errors), 1)) if errors else 0.5

    def parameters(self) -> list[torch.Tensor]:
        """Collect parameters from all sub-regions, wiring and voter."""
        params: list[torch.Tensor] = []
        for mod in self.regions.values():
            params.extend(mod.parameters())
        # router/corrector handled elsewhere; wiring has no gradients
        return params

    def get_predictions(self) -> dict[TransformerRegion, torch.Tensor]:
        """Get prediction tensors for persistence."""
        return {
            region: mod._prediction.detach().clone()
            for region, mod in self.regions.items()
        }

    def set_predictions(self, predictions: dict[TransformerRegion, torch.Tensor]) -> None:
        """Restore prediction tensors from persistence."""
        for region, pred in predictions.items():
            if region in self.regions and pred.shape == self.regions[region]._prediction.shape:
                self.regions[region]._prediction = pred.to(
                    device=self._device, dtype=DTYPE,
                )

    def run_autonomous(self, steps: int = 50) -> list[dict[TransformerRegion, RegionActivation]]:
        """REM sleep: run transformer autonomously without external input.

        Driven purely by internal dynamics (predictions feeding back).
        Returns activation patterns for structural integration.
        """
        patterns = []
        for _ in range(steps):
            # Each region uses its own prediction as input
            activations: dict[TransformerRegion, RegionActivation] = {}
            for region in self._genome.topology.active_regions:
                if region in self.regions:
                    mod = self.regions[region]
                    # Feed prediction back as input (dreaming)
                    fake_input = mod._prediction.unsqueeze(0)  # (1, input_dim)
                    activations[region] = mod.forward(fake_input)

            # Let inter-region wiring modulate
            modulations = self.wiring.modulate(activations)
            for region in self._genome.topology.active_regions:
                mod_vec = modulations.get(region)
                if mod_vec is not None and region in activations:
                    refined = torch.relu(activations[region].hidden + mod_vec)
                    activations[region] = RegionActivation(
                        region=region,
                        logits=refined @ self.regions[region].W_out,
                        hidden=refined,
                        fired_indices=torch.where(refined > self.regions[region].theta)[0],
                    )

            patterns.append(activations)
        return patterns
