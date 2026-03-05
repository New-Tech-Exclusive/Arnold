"""Tier 3 - The Cortex (Plastic Hebbian Layers)."""

from __future__ import annotations

import torch

from .genome import Genome
from .tensor import DTYPE, get_device
from .types_ import (
    ContextualizedInput,
    CortexRegion,
    HebbianTrace,
    PersonalityVector,
    RegionActivation,
)


class CortexRegionModule:
    """One region of the cortex - a set of Hebbian weight matrices."""

    def __init__(
        self,
        region: CortexRegion,
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
        self.W_hidden = torch.randn((hidden_dim, hidden_dim), dtype=DTYPE, device=self._device) * 0.02
        self.W_out = torch.randn((hidden_dim, output_dim), dtype=DTYPE, device=self._device) * 0.02
        # Layer norm on hidden activations stabilises logit magnitudes and
        # prevents vocabulary collapse (one token dominating all outputs).
        self._ln_scale = torch.ones(hidden_dim, dtype=DTYPE, device=self._device)
        self._ln_bias = torch.zeros(hidden_dim, dtype=DTYPE, device=self._device)

        self.last_fired: torch.Tensor = torch.zeros(hidden_dim, dtype=torch.int64, device=self._device)
        self._last_pre: torch.Tensor | None = None
        self._last_post: torch.Tensor | None = None

    def forward(self, x: torch.Tensor) -> RegionActivation:
        pooled = x.mean(dim=0)
        pre = pooled @ self.W_in
        hidden = torch.relu(pre)
        post = torch.relu(hidden @ self.W_hidden)

        # Layer-norm before vocab projection to keep logit magnitudes balanced.
        post_norm = torch.layer_norm(post, [post.shape[-1]], self._ln_scale, self._ln_bias)

        threshold = self._genome.hebbian.coactivation_threshold
        fired_mask = post > threshold
        fired_indices = torch.where(fired_mask)[0]

        self.last_fired[fired_indices] = 0
        self.last_fired[~fired_mask] += 1

        self._last_pre = hidden
        self._last_post = post

        logits = post_norm @ self.W_out
        return RegionActivation(
            region=self.region,
            logits=logits,
            hidden=post,
            fired_indices=fired_indices,
        )

    def get_hebbian_trace(self) -> HebbianTrace | None:
        if self._last_pre is None or self._last_post is None:
            return None
        threshold = self._genome.hebbian.coactivation_threshold
        pre_fired = torch.where(self._last_pre > threshold)[0]
        post_fired = torch.where(self._last_post > threshold)[0]
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

    def hebbian_update(self, trace: HebbianTrace, learning_rate: float) -> None:
        params = self._genome.hebbian
        exp = params.strengthening_fn_exponent
        n_post = int(trace.post_indices.numel())

        for i_idx, pre_i_t in enumerate(trace.pre_indices.tolist()):
            pre_i = int(pre_i_t)
            start = i_idx * n_post
            end = (i_idx + 1) * n_post
            pre_strength = trace.activation_strengths[start:end] if n_post > 0 else torch.empty(0, dtype=DTYPE, device=self._device)

            for j_idx, post_j_t in enumerate(trace.post_indices.tolist()):
                post_j = int(post_j_t)
                strength = float(pre_strength[j_idx]) if j_idx < pre_strength.numel() else 0.0
                delta = learning_rate * (abs(strength) ** exp) * (1.0 if strength >= 0 else -1.0)
                if pre_i < self.hidden_dim and post_j < self.hidden_dim:
                    self.W_hidden[pre_i, post_j] += delta
                    self.W_hidden[pre_i, post_j] = torch.clamp(
                        self.W_hidden[pre_i, post_j], params.min_weight, params.max_weight,
                    )

    def hebbian_update_batch(self, traces: list[HebbianTrace], learning_rate: float) -> None:
        params = self._genome.hebbian
        exp = params.strengthening_fn_exponent

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
                        s = float(trace.activation_strengths[flat_idx])
                        delta = learning_rate * (abs(s) ** exp) * (1.0 if s >= 0 else -1.0)
                        if pre_i < self.hidden_dim and post_j < self.hidden_dim:
                            self.W_hidden[pre_i, post_j] += delta

        self.W_hidden.clamp_(params.min_weight, params.max_weight)

    def apply_weight_decay(self, decay_rate: float) -> None:
        self.W_in *= (1.0 - decay_rate)
        self.W_hidden *= (1.0 - decay_rate)
        self.W_out *= (1.0 - decay_rate)

    def prune(self, age_threshold: int, weight_threshold: float) -> int:
        stale_mask = self.last_fired > age_threshold
        weak_mask = self.W_hidden.abs().max(dim=1).values < weight_threshold
        prune_mask = stale_mask & weak_mask
        pruned_count = int(prune_mask.sum().item())

        if pruned_count > 0:
            self.W_hidden[prune_mask, :] = 0.0
            self.W_hidden[:, prune_mask] = 0.0
            self.W_in[:, prune_mask] = 0.0
            self.W_out[prune_mask, :] = 0.0

        return pruned_count


class InterRegionWiring:
    """Hebbian connections between cortex regions (Tier 3E)."""

    def __init__(self, genome: Genome, rng: object | None = None) -> None:
        self._genome = genome
        self._device = get_device()
        hidden = genome.topology.cortex_hidden
        active = genome.topology.active_regions

        self.connections: dict[tuple[CortexRegion, CortexRegion], torch.Tensor] = {}
        self.highway_strengths: dict[tuple[CortexRegion, CortexRegion], float] = {}

        for i, r1 in enumerate(active):
            for r2 in active[i + 1:]:
                key = (r1, r2)
                self.connections[key] = torch.randn((hidden, hidden), dtype=DTYPE, device=self._device) * 0.01
                self.highway_strengths[key] = 0.0

    def modulate(self, activations: dict[CortexRegion, RegionActivation]) -> dict[CortexRegion, torch.Tensor]:
        modulations: dict[CortexRegion, torch.Tensor] = {}
        hidden_dim = self._genome.topology.cortex_hidden

        for region in self._genome.topology.active_regions:
            modulations[region] = torch.zeros(hidden_dim, dtype=DTYPE, device=self._device)

        for (r1, r2), W in self.connections.items():
            if r1 not in activations or r2 not in activations:
                continue
            h1 = activations[r1].hidden
            h2 = activations[r2].hidden

            mod_1_to_2 = h1 @ W
            mod_2_to_1 = h2 @ W.T

            modulations[r1] = modulations[r1] + mod_2_to_1 * 0.1
            modulations[r2] = modulations[r2] + mod_1_to_2 * 0.1

            coactivation = float(torch.dot(h1, h2) / (torch.linalg.norm(h1) * torch.linalg.norm(h2) + 1e-12))
            self.highway_strengths[(r1, r2)] = 0.99 * self.highway_strengths[(r1, r2)] + 0.01 * coactivation

        return modulations

    def hebbian_update(self, activations: dict[CortexRegion, RegionActivation], learning_rate: float) -> None:
        params = self._genome.hebbian
        for (r1, r2), W in self.connections.items():
            if r1 not in activations or r2 not in activations:
                continue
            h1 = activations[r1].hidden
            h2 = activations[r2].hidden
            delta = learning_rate * torch.outer(h1, h2)
            W += delta
            W.clamp_(params.min_weight, params.max_weight)

    def get_highway_map(self) -> dict[tuple[CortexRegion, CortexRegion], float]:
        return dict(self.highway_strengths)


class CortexVoter:
    """Combines region logits via personality-weighted vote."""

    def __init__(self, genome: Genome) -> None:
        self._genome = genome
        self._device = get_device()

    def vote(
        self,
        activations: dict[CortexRegion, RegionActivation],
        personality: PersonalityVector,
        mood_valence: float = 0.0,
    ) -> torch.Tensor:
        voting = self._genome.voting
        vocab = self._genome.topology.vocab_size

        region_weights: dict[CortexRegion, float] = {}
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

    def compute_region_weights(self, personality: PersonalityVector) -> dict[CortexRegion, float]:
        voting = self._genome.voting
        weights: dict[CortexRegion, float] = {}
        for region in self._genome.topology.active_regions:
            weights[region] = voting.base_weight
        for region, trait_name, multiplier in voting.region_trait_map:
            if region in weights:
                weights[region] += multiplier * getattr(personality, trait_name, 0.5)
        total = sum(weights.values()) or 1.0
        return {r: w / total for r, w in weights.items()}


class Cortex:
    """The complete Tier 3 cortex with all regions, wiring, and voting."""

    def __init__(self, genome: Genome, rng: object | None = None) -> None:
        self._genome = genome
        self._device = get_device()
        topo = genome.topology

        self.regions: dict[CortexRegion, CortexRegionModule] = {}
        for region in topo.regions:
            self.regions[region] = CortexRegionModule(
                region=region,
                input_dim=topo.embed_dim,
                hidden_dim=topo.cortex_hidden,
                output_dim=topo.vocab_size,
                genome=genome,
                rng=rng,
            )

        self.wiring = InterRegionWiring(genome, rng)
        self.voter = CortexVoter(genome)

    def forward(
        self,
        ctx_input: ContextualizedInput,
        personality: PersonalityVector,
    ) -> tuple[torch.Tensor, list[HebbianTrace], dict[CortexRegion, RegionActivation]]:
        embeddings = ctx_input.structured.embeddings

        activations: dict[CortexRegion, RegionActivation] = {}
        for region in self._genome.topology.active_regions:
            activations[region] = self.regions[region].forward(embeddings)

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
                        refined_hidden > self._genome.hebbian.coactivation_threshold
                    )[0],
                )

        traces: list[HebbianTrace] = []
        for region in self._genome.topology.active_regions:
            trace = self.regions[region].get_hebbian_trace()
            if trace is not None:
                traces.append(trace)

        combined_logits = self.voter.vote(activations, personality, ctx_input.mood.valence)
        return combined_logits, traces, activations

    def forward_readonly(self, embeddings: torch.Tensor) -> dict[CortexRegion, RegionActivation]:
        activations: dict[CortexRegion, RegionActivation] = {}
        for region in self._genome.topology.active_regions:
            activations[region] = self.regions[region].forward(embeddings)
        return activations

    def get_weights(self) -> dict[CortexRegion, torch.Tensor]:
        out: dict[CortexRegion, torch.Tensor] = {}
        for region, mod in self.regions.items():
            out[region] = torch.cat([
                mod.W_in.reshape(-1),
                mod.W_hidden.reshape(-1),
                mod.W_out.reshape(-1),
            ]).detach().clone()
        return out

    def set_weights(self, weights: dict[CortexRegion, torch.Tensor]) -> None:
        for region, blob in weights.items():
            if region not in self.regions:
                continue
            mod = self.regions[region]
            b = blob.to(device=self._device, dtype=DTYPE)
            sizes = [
                mod.input_dim * mod.hidden_dim,
                mod.hidden_dim * mod.hidden_dim,
                mod.hidden_dim * mod.output_dim,
            ]
            parts = torch.split(b, sizes)
            mod.W_in = parts[0].reshape(mod.input_dim, mod.hidden_dim)
            mod.W_hidden = parts[1].reshape(mod.hidden_dim, mod.hidden_dim)
            mod.W_out = parts[2].reshape(mod.hidden_dim, mod.output_dim)

    def get_inter_region_weights(self) -> dict[tuple[CortexRegion, CortexRegion], torch.Tensor]:
        return {k: v.detach().clone() for k, v in self.wiring.connections.items()}

    def set_inter_region_weights(self, weights: dict[tuple[CortexRegion, CortexRegion], torch.Tensor]) -> None:
        for key, arr in weights.items():
            if key in self.wiring.connections:
                self.wiring.connections[key] = arr.to(device=self._device, dtype=DTYPE).clone()
