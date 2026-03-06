"""Tier 6 — The Consolidation Engine (Offline Learning System).

Runs between sessions (or when the hippocampus buffer is full).
This is where actual permanent learning happens.

Implements three-stage sleep consolidation (Tononi & Cirelli):

  Stage A — Targeted Replay (NREM 1-2)
    Sharp-wave ripple replay of high-priority records.
    Interleaved with core memory prototypes (CLS theory).

  Stage B — Global Synaptic Downscaling (NREM 3 / Slow-wave sleep)
    ALL weights multiplicatively downscaled by a small factor (0.995).
    Synaptic Homeostasis Hypothesis — prevents weight saturation.

  Stage C — REM / Structural Integration
    Hippocampus disconnects.  Cortex runs autonomously, generating
    activation patterns driven purely by internal dynamics.  New memories
    that activate similar patterns to old memories get their inter-region
    highways strengthened — conceptual integration.

Plus:
  Stage 4 — EWC Protection
  Stage 5 — Personality Vector Update
  Stage 6 — Pruning
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from .tensor import DTYPE, get_device

from .genome import Genome
from .types_ import (
    CortexRegion,
    EpisodicRecord,
    HebbianTrace,
    MoodState,
    PersonalityVector,
)


@dataclass
class ConsolidationReport:
    """Summary of what happened during one consolidation cycle."""
    records_replayed: int = 0
    total_replay_passes: int = 0
    connections_pruned: int = 0
    weights_protected: int = 0
    personality_deltas: dict[str, float] | None = None
    global_downscale_applied: bool = False
    rem_patterns_generated: int = 0
    rem_integrations: int = 0


class ConsolidationEngine:
    """Three-stage sleep consolidation engine.

    Stage A: Targeted replay with interleaved core memories (CLS).
    Stage B: Global synaptic downscaling (Synaptic Homeostasis Hypothesis).
    Stage C: REM — cortex autonomous processing for structural integration.
    Plus: EWC protection, personality update, pruning.
    """

    # Maximum number of core memory prototypes retained
    MAX_CORE_MEMORIES = 100

    def __init__(self, genome: Genome) -> None:
        self._genome = genome
        # Core memories: high-priority activation prototypes for interleaved replay
        self._core_memories: list[EpisodicRecord] = []

    def run(
        self,
        records: list[EpisodicRecord],
        cortex,            # Cortex instance
        ewc_scalars: dict[CortexRegion, torch.Tensor],
        personality: PersonalityVector,
        plasticity_rates: dict[CortexRegion, float],
        developmental_age: int,
    ) -> ConsolidationReport:
        """Execute the full consolidation pipeline.

        Stage A: Replay (NREM stages 1-2)
        Stage B: Global downscaling (NREM stage 3 / slow-wave)
        Stage C: REM structural integration
        Stage 4: Pruning
        Stage 5: EWC Protection
        Stage 6: Personality Update
        """
        report = ConsolidationReport()

        # Stage A: Targeted Replay (NREM 1-2)
        self._stage_replay(records, cortex, plasticity_rates, report)

        # Stage B: Global Synaptic Downscaling (NREM 3 / Slow-wave)
        self._stage_global_downscale(cortex, report)

        # Stage C: REM — Structural Integration
        self._stage_rem(cortex, report)

        # Stage 4: Pruning
        self._stage_pruning(cortex, developmental_age, report)

        # Stage 5: EWC Protection
        self._stage_ewc(cortex, ewc_scalars, report)

        # Stage 6: Personality Update
        self._stage_personality(records, personality, report)

        return report

    # ==================================================================
    # Stage 1 — Replay
    # ==================================================================

    def _stage_replay(
        self,
        records: list[EpisodicRecord],
        cortex,
        plasticity_rates: dict[CortexRegion, float],
        report: ConsolidationReport,
    ) -> None:
        """Re-fire stored activation patterns through cortex weights.

        Implements Complementary Learning Systems (McClelland et al. 1995):
        new records are interleaved with core memory prototypes during replay
        to prevent catastrophic forgetting.
        """
        params = self._genome.consolidation

        if not records:
            return

        # --- Interleaved replay: merge new records with core memories ---
        # Core memories are replayed at lower learning rate to maintain stability
        replay_list: list[tuple[EpisodicRecord, float]] = []  # (record, lr_scale)

        # Add new records at full learning rate
        max_priority = max(r.consolidation_priority for r in records) or 1.0
        for record in records:
            norm_priority = record.consolidation_priority / max_priority
            replay_list.append((record, norm_priority))

        # Interleave core memories at reduced rate (10% of normal)
        for core_rec in self._core_memories:
            replay_list.append((core_rec, 0.1))

        for record, lr_scale in replay_list:
            norm_priority = lr_scale if lr_scale <= 0.1 else lr_scale
            n_replays = int(
                params.replay_passes_min
                + norm_priority * (params.replay_passes_max - params.replay_passes_min)
            )
            n_replays = max(1, n_replays)

            for _ in range(n_replays):
                for trace in record.hebbian_traces:
                    region = trace.region
                    if region not in cortex.regions:
                        continue
                    lr = plasticity_rates.get(region, 0.01) * lr_scale
                    cortex.regions[region].hebbian_update(trace, lr)

                report.total_replay_passes += 1

            report.records_replayed += 1

        # --- Update core memories: keep highest-priority records ---
        for record in records:
            if len(self._core_memories) < self.MAX_CORE_MEMORIES:
                self._core_memories.append(record)
            else:
                # Replace lowest-priority core memory if this one is higher
                min_idx = min(
                    range(len(self._core_memories)),
                    key=lambda i: self._core_memories[i].consolidation_priority,
                )
                if record.consolidation_priority > self._core_memories[min_idx].consolidation_priority:
                    self._core_memories[min_idx] = record

        # Also update inter-region wiring based on co-activated records
        # Gather all traces that co-occurred within the same record
        for record in records:
            region_hiddens: dict[CortexRegion, torch.Tensor] = {}
            for trace in record.hebbian_traces:
                if trace.region in cortex.regions:
                    # Reconstruct approximate hidden from trace
                    hidden = torch.zeros(
                        cortex.regions[trace.region].hidden_dim,
                        dtype=DTYPE,
                        device=get_device(),
                    )
                    hidden[trace.post_indices] = trace.activation_strengths[:len(trace.post_indices)]
                    region_hiddens[trace.region] = hidden

            if len(region_hiddens) >= 2:
                # Build mock activations for inter-region update
                from .types_ import RegionActivation
                mock_acts = {}
                for r, h in region_hiddens.items():
                    mock_acts[r] = RegionActivation(
                        region=r,
                        logits=torch.zeros(self._genome.topology.vocab_size, dtype=DTYPE, device=get_device()),
                        hidden=h,
                        fired_indices=torch.where(h > 0)[0],
                    )
                mean_lr = float(sum(plasticity_rates.values()) / max(len(plasticity_rates), 1))
                cortex.wiring.hebbian_update(mock_acts, mean_lr * 0.1)

    # ==================================================================
    # Stage B — Global Synaptic Downscaling (Slow-wave sleep)
    # ==================================================================

    def _stage_global_downscale(
        self,
        cortex,
        report: ConsolidationReport,
    ) -> None:
        """Synaptic Homeostasis Hypothesis (Tononi & Cirelli).

        ALL synapses weaken slightly — a universal multiplicative downscaling.
        This is NOT targeted pruning.  EVERY weight gets multiplied by a
        factor just below 1.0 (e.g. 0.995).  Prevents runaway potentiation
        and weight saturation over thousands of sessions.
        """
        factor = self._genome.sleep.global_downscale_factor

        for region in self._genome.topology.active_regions:
            if region not in cortex.regions:
                continue
            mod = cortex.regions[region]
            mod.W_in *= factor
            mod.W_hidden *= factor
            mod.W_out *= factor
            mod.W_recurrent *= factor

        # Also downscale inter-region wiring
        for key in cortex.wiring.connections:
            cortex.wiring.connections[key] *= factor

        report.global_downscale_applied = True

    # ==================================================================
    # Stage C — REM (Structural Integration)
    # ==================================================================

    def _stage_rem(
        self,
        cortex,
        report: ConsolidationReport,
    ) -> None:
        """REM sleep: hippocampus disconnects, cortex runs autonomously.

        The cortex generates activation patterns driven purely by internal
        dynamics (predictions feeding back).  Patterns that activate similar
        regions to existing memories get their inter-region highways
        strengthened — this is how conceptual integration happens.
        """
        sleep = self._genome.sleep
        rem_steps = sleep.rem_steps
        rem_lr = sleep.rem_integration_lr

        # Run cortex autonomously
        patterns = cortex.run_autonomous(steps=rem_steps)
        report.rem_patterns_generated = len(patterns)

        # Structural integration: strengthen inter-region connections
        # where co-activation is high during autonomous processing
        integrations = 0
        from .types_ import RegionActivation
        for act_dict in patterns:
            if len(act_dict) >= 2:
                cortex.wiring.hebbian_update(act_dict, rem_lr)
                integrations += 1

        report.rem_integrations = integrations

    # ==================================================================
    # Stage 4 — Pruning
    # ==================================================================

    def _stage_pruning(
        self,
        cortex,
        developmental_age: int,
        report: ConsolidationReport,
    ) -> None:
        """Remove weak, unused connections.  The model gets leaner."""
        params = self._genome.consolidation

        total_pruned = 0
        for region in self._genome.topology.active_regions:
            if region in cortex.regions:
                n = cortex.regions[region].prune(
                    age_threshold=params.pruning_age_threshold,
                    weight_threshold=params.pruning_threshold,
                )
                total_pruned += n

        # Also apply passive weight decay
        decay = self._genome.hebbian.decay_rate
        for region in self._genome.topology.active_regions:
            if region in cortex.regions:
                cortex.regions[region].apply_weight_decay(decay)

        report.connections_pruned = total_pruned

    # ==================================================================
    # Stage 5 — EWC Protection
    # ==================================================================

    def _stage_ewc(
        self,
        cortex,
        ewc_scalars: dict[CortexRegion, torch.Tensor],
        report: ConsolidationReport,
    ) -> None:
        """Compute Fisher-information-based protection for important weights.

        High sensitivity weights get a resistance scalar so future Hebbian
        updates are attenuated:  effective_lr = lr × (1 - protection).
        """
        params = self._genome.ewc
        total_protected = 0

        for region in self._genome.topology.active_regions:
            if region not in cortex.regions:
                continue
            mod = cortex.regions[region]

            # Estimate importance via weight magnitude as a proxy for Fisher info.
            # (True Fisher requires gradient samples; we approximate with |w|.)
            W = mod.W_hidden  # (hidden, hidden)
            importance = W.abs().mean(dim=1)  # per-unit importance

            # Normalise to [0, 1]
            imp_max = float(importance.max().item()) or 1.0
            normalised = importance / imp_max

            # Protection = normalised^exponent, scaled, clamped
            protection = torch.pow(normalised, params.protection_exponent)
            protection *= params.protection_growth_rate
            protection = torch.clamp(protection, 0.0, params.max_protection)

            # Accumulate with existing protection (max-blend)
            if region in ewc_scalars and ewc_scalars[region].shape == protection.shape:
                ewc_scalars[region] = torch.maximum(ewc_scalars[region], protection)
            else:
                ewc_scalars[region] = protection

            total_protected += int((ewc_scalars[region] > 0.1).sum().item())

        report.weights_protected = total_protected

    # ==================================================================
    # Stage 6 — Personality Vector Update
    # ==================================================================

    def _stage_personality(
        self,
        records: list[EpisodicRecord],
        personality: PersonalityVector,
        report: ConsolidationReport,
    ) -> None:
        """Update personality traits based on session signal patterns.

        Each trait responds to specific aggregate patterns in the records.
        Drift is bounded by genome.personality.max_drift_per_cycle.
        """
        if not records:
            report.personality_deltas = {}
            return

        max_drift = self._genome.personality.max_drift_per_cycle
        deltas: dict[str, float] = {}

        # Aggregate statistics from records
        reinforcements = torch.tensor([r.reinforcement for r in records], dtype=DTYPE, device=get_device())
        novelties = torch.tensor([r.novelty_score for r in records], dtype=DTYPE, device=get_device())
        valences = torch.tensor([r.mood_at_event.valence for r in records], dtype=DTYPE, device=get_device())

        mean_reinforcement = float(reinforcements.mean().item()) if int(reinforcements.numel()) > 0 else 0.0
        mean_valence = float(valences.mean().item()) if int(valences.numel()) > 0 else 0.0

        def _clamp01(v: float) -> float:
            return float(max(min(v, 1.0), 0.0))

        # Curiosity: high novelty + positive reinforcement
        novelty_pos = float(torch.mean(novelties * torch.clamp(reinforcements, min=0)).item())
        delta_curiosity = max(min(novelty_pos * 0.1, max_drift), -max_drift)
        personality.curiosity = _clamp01(personality.curiosity + delta_curiosity)
        deltas["curiosity"] = delta_curiosity

        # Warmth: positive valence in conversational contexts
        delta_warmth = max(min(mean_valence * 0.05, max_drift), -max_drift)
        personality.warmth = _clamp01(personality.warmth + delta_warmth)
        deltas["warmth"] = delta_warmth

        # Assertiveness: net direction of corrections vs affirmations
        delta_assert = max(min(mean_reinforcement * 0.04, max_drift), -max_drift)
        personality.assertiveness = _clamp01(personality.assertiveness + delta_assert)
        deltas["assertiveness"] = delta_assert

        # Creativity: novelty-heavy responses that got positive signal
        high_novelty_mask = novelties > 0.5
        if bool(high_novelty_mask.any()):
            creative_signal = float(reinforcements[high_novelty_mask].mean().item())
        else:
            creative_signal = 0.0
        delta_creative = max(min(creative_signal * 0.05, max_drift), -max_drift)
        personality.creativity = _clamp01(personality.creativity + delta_creative)
        deltas["creativity"] = delta_creative

        # Caution: qualified outputs working better than confident ones
        # Proxy: negative reinforcement episodes → caution up
        neg_ratio = float((reinforcements < 0).sum().item()) / max(int(reinforcements.numel()), 1)
        delta_caution = max(min((neg_ratio - 0.3) * 0.05, max_drift), -max_drift)
        personality.caution = _clamp01(personality.caution + delta_caution)
        deltas["caution"] = delta_caution

        # Humor: tonal lightness positively received (proxy: high valence + low arousal)
        arousal = torch.tensor([r.mood_at_event.arousal for r in records], dtype=DTYPE, device=get_device())
        low_arousal_pos = float(torch.mean(valences * (1.0 - arousal)).item())
        delta_humor = max(min(low_arousal_pos * 0.04, max_drift), -max_drift)
        personality.humor = _clamp01(personality.humor + delta_humor)
        deltas["humor"] = delta_humor

        report.personality_deltas = deltas
