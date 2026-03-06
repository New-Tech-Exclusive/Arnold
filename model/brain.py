"""Brain - Top-level orchestrator that wires all tiers together.

Processing flow:
  Input arrives
    → Thalamic router (decides which regions get what)
    → Predictive coding (each region generates prediction, computes error)
    → Prediction error propagates (not raw input)
    → Four neuromodulators update based on error magnitude and sign
    → Cortex regions process with sparse k-WTA activations
    → Basal ganglia: is this a habituated pattern? Shortcut if yes
    → Cerebellum: predict outcome, pre-correct generation
    → Generation with recurrent state + context attention
    → Output

  Between sessions:
    → DMN phase (internal thought generation)
    → Three-stage sleep consolidation
    → Astrocyte layer applies slow metabolic penalties
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path

import torch

from .astrocyte import AstrocyteLayer
from .basal_ganglia import BasalGanglia
from .brainstem import Brainstem
from .cerebellum import Cerebellum
from .consolidation import ConsolidationEngine, ConsolidationReport
from .cortex import Cortex
from .generation import GenerationInterface
from .genome import Genome
from .hippocampus import Hippocampus
from .limbic import LimbicSystem
from .novelty import NoveltyDetector
from .plasticity import PlasticitySystem
from .tensor import DTYPE, get_device, seed_all
from .thalamus import Thalamus
from .types_ import (
    BrainState,
    CortexRegion,
    HebbianTrace,
    MoodState,
    NeuromodulatorState,
    PersonalityVector,
    ReinforcementSignal,
    ReinforcementType,
)
from .weight_store import WeightStore


@dataclass
class TurnResult:
    """Everything produced by a single turn of processing."""

    generated_tokens: list[int]
    novelty_score: float
    mood: MoodState
    reinforcement_strength: float
    partial_update_applied: bool
    region_weights: dict[CortexRegion, float]
    neuromodulators: NeuromodulatorState | None = None
    prediction_error: float = 0.0
    thalamic_routing: dict[CortexRegion, float] | None = None


class Brain:
    """The complete neural architecture - all nine tiers assembled."""

    def __init__(
        self,
        genome: Genome | None = None,
        storage_dir: str | Path = "./brain_data",
        seed: int = 42,
        tokenizer: object | None = None,
        restore: bool = True,
    ) -> None:
        self._genome = genome or Genome()
        seed_all(seed)
        self._storage_dir = Path(storage_dir)

        self.weight_store = WeightStore(self._storage_dir)

        self._personality = PersonalityVector()
        self._mood_baseline = MoodState()
        self._neuromodulator_baseline = NeuromodulatorState()
        self._developmental_age = 0
        self._ewc_scalars: dict[CortexRegion, torch.Tensor] = {}
        self._consolidation_meta: dict = {
            "total_sessions": 0,
            "total_consolidations": 0,
            "avg_reinforcement": 0.0,
        }

        # Core tiers
        self.brainstem = Brainstem(self._genome)
        self.cortex = Cortex(self._genome)
        self.novelty_detector = NoveltyDetector(self._genome)
        self.consolidation_engine = ConsolidationEngine(self._genome)
        self.plasticity_system = PlasticitySystem(self._genome)
        self.generation = GenerationInterface(self._genome)

        # New components
        self.thalamus = Thalamus(self._genome)
        self.basal_ganglia = BasalGanglia(self._genome)
        self.cerebellum = Cerebellum(self._genome)
        self.astrocyte = AstrocyteLayer(self._genome)

        self.limbic: LimbicSystem | None = None
        self.hippocampus: Hippocampus | None = None

        self._tokenizer = tokenizer

        self._previous_model_text: str | None = None
        self._previous_traces: list[HebbianTrace] = []
        self._last_structured = None  # cached for soft-prompt anchoring
        self._session_active = False

        if restore:
            self._try_restore()

    def birth(self, pretraining_corpus: list[torch.Tensor] | None = None) -> None:
        if pretraining_corpus:
            self.brainstem.pretrain(pretraining_corpus)

        # Keep brainstem plastic after birth so user interactions can continue
        # to shape low-level token structure online.
        self.brainstem.unfreeze()

        self._personality = PersonalityVector()
        self._mood_baseline = MoodState()
        self._developmental_age = 0
        self._ewc_scalars = {}
        self._save_state()

    def session_start(self) -> None:
        self.limbic = LimbicSystem(
            self._genome, self._mood_baseline, self._neuromodulator_baseline,
        )
        self.hippocampus = Hippocampus(self._genome)
        self.hippocampus.set_interaction_counter(self._developmental_age)

        self._previous_model_text = None
        self._previous_traces = []
        self._session_active = True

    def session_end(self) -> ConsolidationReport | None:
        if not self._session_active:
            return None

        self._session_active = False

        if self.limbic:
            self._mood_baseline = self.limbic.session_end_baseline()
            self._neuromodulator_baseline = self.limbic.session_end_neuromodulator_baseline()

        if self.hippocampus:
            self._developmental_age = self.hippocampus.interaction_count

        # DMN phase: internal thought generation before consolidation
        try:
            self._run_dmn_phase()
        except Exception:
            pass  # DMN is best-effort; don't block shutdown

        report = self._consolidate()

        # Astrocyte: decay usage counters and apply metabolic penalties
        self.astrocyte.decay()
        if self._developmental_age % self._genome.astrocyte.update_interval == 0:
            self.astrocyte.apply_penalties(self.cortex)

        # Basal ganglia: decay unused habits
        self.basal_ganglia.decay_habits()

        self._consolidation_meta["total_sessions"] += 1
        self._consolidation_meta["total_consolidations"] += 1
        self._save_state()
        return report

    @staticmethod
    def _forced_signals_from_reinforcement(
        external_reinforcement: float | None,
    ) -> list[ReinforcementSignal] | None:
        if external_reinforcement is None:
            return None
        value = float(external_reinforcement)
        if not math.isfinite(value):
            return None
        value = max(min(value, 1.0), -1.0)
        if abs(value) < 1e-8:
            return []
        signal_type = (
            ReinforcementType.DIRECT_AFFIRMATION
            if value > 0.0 else
            ReinforcementType.DIRECT_CORRECTION
        )
        return [ReinforcementSignal(signal_type, value)]

    def process_turn(
        self,
        user_text: str,
        external_reinforcement: float | None = None,
    ) -> TurnResult:
        assert self._session_active, "Call session_start() before processing turns"
        assert self.limbic is not None
        assert self.hippocampus is not None

        token_ids = self._tokenize(user_text)
        structured = self.brainstem.process(token_ids)

        # Online brainstem update: light Hebbian pass on this turn's tokens
        self.brainstem.online_update(token_ids)

        novelty_score = self.novelty_detector.score(structured, self.cortex.forward_readonly)

        # Prediction error from cortex (unified with novelty via predictive coding)
        pred_error = self.cortex.mean_prediction_error()

        forced_signals = self._forced_signals_from_reinforcement(external_reinforcement)

        ctx_input = self.limbic.process(
            structured, user_text, self._previous_model_text,
            novelty_score, pred_error,
            forced_signals=forced_signals,
        )
        reinforcement_strength = sum(s.strength for s in ctx_input.reinforcement_signals)
        neuromod = self.limbic.neuromodulators

        # Neuromodulatory gain: norepinephrine drives cortex sharpening
        self.cortex.set_neuromodulatory_gain(
            norepinephrine=neuromod.norepinephrine, novelty=novelty_score,
        )

        # Thalamic routing: decide which regions get what
        routing_weights = self.thalamus.route(
            structured.embeddings, neuromod, novelty_score,
        )

        combined_logits, traces, activations = self.cortex.forward(
            ctx_input, self._personality, routing_weights,
        )

        # Thalamic feedback from cortex output
        region_hiddens = {r: a.hidden for r, a in activations.items()}
        self.thalamus.receive_feedback(region_hiddens)

        # Astrocyte: record activity
        for region, act in activations.items():
            self.astrocyte.record_activity(region, act.fired_indices)

        # Cerebellum: pre-correct logits
        combined_logits = self.cerebellum.pre_correct_logits(
            combined_logits, structured.embeddings,
        )

        # Basal ganglia: check for habituated shortcut
        context_emb = structured.embeddings.mean(dim=0)
        habit_tokens = self.basal_ganglia.check_habit(context_emb)

        if habit_tokens is not None:
            generated_tokens = habit_tokens
        else:
            # Limit generation length by developmental stage — early-stage models
            # produce lower-quality output; short sequences reduce noise volume.
            _age = self._developmental_age
            if _age < 500:       # infancy
                _max_tokens = 16
            elif _age < 5_000:   # childhood
                _max_tokens = 32
            elif _age < 50_000:  # adolescence
                _max_tokens = 48
            else:                # adulthood
                _max_tokens = 64
            generated_tokens = self.generation.generate_sequence(
                combined_logits,
                self.limbic.mood,
                novelty_score,
                max_tokens=_max_tokens,
                brainstem=self.brainstem,
                cortex=self.cortex,
                anchor_embeddings=structured.embeddings,
                norepinephrine=neuromod.norepinephrine,
            )

        self.hippocampus.record(
            token_ids=token_ids,
            traces=traces,
            novelty_score=novelty_score,
            mood=self.limbic.mood,
            reinforcement=reinforcement_strength,
        )

        # Plasticity rates gated by acetylcholine
        plasticity_rates = self.plasticity_system.all_rates(
            self._developmental_age,
            self.limbic.mood,
            {r: float(s.mean().item()) for r, s in self._ewc_scalars.items()},
            acetylcholine=neuromod.acetylcholine,
        )

        partial_applied = self.generation.partial_weight_update(
            self.cortex,
            self._previous_traces,
            reinforcement_strength,
            plasticity_rates,
            self._ewc_scalars,
        )

        # Cerebellum: update forward model with actual reinforcement
        self.cerebellum.train_step(structured.embeddings, reinforcement_strength)

        # Basal ganglia: record sequence for potential habit formation
        self.basal_ganglia.record_sequence(context_emb, generated_tokens, reinforcement_strength)

        if self.hippocampus.needs_consolidation:
            self._consolidate()

        # Periodic homeostatic synaptic scaling
        homeo = self._genome.homeostatic
        if self._developmental_age > 0 and self._developmental_age % homeo.update_interval == 0:
            self.cortex.homeostatic_scale_all()

        self._previous_model_text = self._detokenize(generated_tokens)
        self._previous_traces = traces
        self._developmental_age += 1
        region_weights = self.cortex.voter.compute_region_weights(self._personality)

        return TurnResult(
            generated_tokens=generated_tokens,
            novelty_score=novelty_score,
            mood=MoodState(
                self.limbic.mood.valence,
                self.limbic.mood.arousal,
                self.limbic.mood.openness,
            ),
            reinforcement_strength=reinforcement_strength,
            partial_update_applied=partial_applied,
            region_weights=region_weights,
            neuromodulators=neuromod,
            prediction_error=pred_error,
            thalamic_routing=routing_weights,
        )

    def prepare_turn(self, user_text: str) -> tuple[torch.Tensor, list, float, float]:
        """Prepare a turn for streaming generation (used by chat_server).

        Returns (logits, traces, novelty, reinforcement).
        Side effects: thalamic routing, neuromodulator update, cortex forward,
        hippocampus record, cerebellum pre-correction, astrocyte tracking.
        """
        assert self._session_active, "Call session_start() before processing turns"
        assert self.limbic is not None
        assert self.hippocampus is not None

        token_ids = self._tokenize(user_text)
        structured = self.brainstem.process(token_ids)

        # Online brainstem update for streaming/chat-server path.
        self.brainstem.online_update(token_ids)

        novelty_score = self.novelty_detector.score(structured, self.cortex.forward_readonly)
        pred_error = self.cortex.mean_prediction_error()

        ctx_input = self.limbic.process(
            structured, user_text, self._previous_model_text,
            novelty_score, pred_error,
        )
        reinforcement_strength = sum(s.strength for s in ctx_input.reinforcement_signals)
        neuromod = self.limbic.neuromodulators

        # Neuromodulatory gain
        self.cortex.set_neuromodulatory_gain(
            norepinephrine=neuromod.norepinephrine, novelty=novelty_score,
        )

        # Thalamic routing
        routing_weights = self.thalamus.route(
            structured.embeddings, neuromod, novelty_score,
        )

        combined_logits, traces, activations = self.cortex.forward(
            ctx_input, self._personality, routing_weights,
        )

        # Thalamic feedback
        region_hiddens = {r: a.hidden for r, a in activations.items()}
        self.thalamus.receive_feedback(region_hiddens)

        # Astrocyte tracking
        for region, act in activations.items():
            self.astrocyte.record_activity(region, act.fired_indices)

        # Cerebellum pre-correction
        combined_logits = self.cerebellum.pre_correct_logits(
            combined_logits, structured.embeddings,
        )

        self.hippocampus.record(
            token_ids=token_ids,
            traces=traces,
            novelty_score=novelty_score,
            mood=self.limbic.mood,
            reinforcement=reinforcement_strength,
        )

        # Store structured embeddings for soft-prompt anchoring in the server
        self._last_structured = structured

        return combined_logits, traces, novelty_score, reinforcement_strength

    def finalize_turn(
        self,
        generated_tokens: list[int],
        traces: list,
        novelty_score: float,
        reinforcement_strength: float,
    ) -> None:
        assert self.limbic is not None
        assert self.hippocampus is not None

        neuromod = self.limbic.neuromodulators

        plasticity_rates = self.plasticity_system.all_rates(
            self._developmental_age,
            self.limbic.mood,
            {r: float(s.mean().item()) for r, s in self._ewc_scalars.items()},
            acetylcholine=neuromod.acetylcholine,
        )

        signal = float(max(min(novelty_score * 0.4 + abs(reinforcement_strength) * 0.6, 1.0), 0.0))

        for trace in traces:
            region = trace.region
            if region not in self.cortex.regions:
                continue
            lr = plasticity_rates.get(region, 0.01) * signal * 0.08
            ewc = self._ewc_scalars.get(region)
            if ewc is not None and int(ewc.numel()) > 0:
                lr *= float(max(min(1.0 - ewc.mean().item(), 1.0), 0.0))
            self.cortex.regions[region].hebbian_update(trace, lr)

        # Cerebellum: update forward model with actual reinforcement
        if self._last_structured is not None:
            self.cerebellum.train_step(self._last_structured.embeddings, reinforcement_strength)

        # Basal ganglia: record sequence for habit formation
        if self._last_structured is not None:
            context_emb = self._last_structured.embeddings.mean(dim=0)
            self.basal_ganglia.record_sequence(context_emb, generated_tokens, reinforcement_strength)

        if self.hippocampus.needs_consolidation:
            self._consolidate()

        self._previous_model_text = self._detokenize(generated_tokens)
        self._previous_traces = traces
        self._developmental_age += 1

        # Periodic homeostatic synaptic scaling
        homeo = self._genome.homeostatic
        if self._developmental_age > 0 and self._developmental_age % homeo.update_interval == 0:
            self.cortex.homeostatic_scale_all()

    def count_parameters(self) -> int:
        total = 0
        for arr in self.brainstem.get_weights().values():
            total += int(arr.numel())
        for region_module in self.cortex.regions.values():
            total += int(region_module.W_in.numel())
            total += int(region_module.W_hidden.numel())
            total += int(region_module.W_out.numel())
            total += int(region_module.W_recurrent.numel())
        # Thalamus
        total += int(self.thalamus.W_route.numel())
        total += int(self.thalamus.W_feedback.numel())
        # Cerebellum
        total += int(self.cerebellum.W_in.numel())
        total += int(self.cerebellum.W_out.numel())
        return total

    # ------------------------------------------------------------------
    # Default Mode Network
    # ------------------------------------------------------------------

    def _run_dmn_phase(self) -> None:
        """DMN: internal thought generation between sessions.

        The model generates internal "thought" sequences not directed at
        any external input.  These draw on recent hippocampus buffer
        contents and explore associative connections.

        The Hebbian traces get added to the consolidation buffer.
        """
        dmn = self._genome.dmn
        if self.hippocampus is None or not self.hippocampus.size:
            return

        recent = self.hippocampus.peek()[:5]  # peek at top-5 recent experiences
        if not recent:
            return

        device = get_device()
        for _ in range(dmn.num_thoughts):
            # Pick a recent experience and use its hidden activations as seed
            record = recent[_ % len(recent)]
            if not record.hebbian_traces:
                continue

            # Build a synthetic input from the record's activation pattern
            trace = record.hebbian_traces[0]
            hidden = torch.zeros(
                self._genome.topology.cortex_hidden,
                dtype=DTYPE, device=device,
            )
            n = min(trace.post_indices.numel(), trace.activation_strengths.numel())
            valid = trace.post_indices[:n]
            valid = valid[valid < hidden.shape[0]]
            hidden[valid] = trace.activation_strengths[:valid.numel()]

            # Project hidden back to embed space for cortex input
            # Use first active region's W_in transpose as projection
            region = self._genome.topology.active_regions[0]
            if region in self.cortex.regions:
                W_in = self.cortex.regions[region].W_in
                # pseudo-inverse projection
                thought_emb = hidden[:W_in.shape[1]] @ W_in.T  # (embed_dim,)
                thought_emb = thought_emb.unsqueeze(0)  # (1, embed_dim)

                # Run cortex read-only to generate DMN traces
                activations = self.cortex.forward_readonly(thought_emb)
                for r, act in activations.items():
                    dmn_trace = self.cortex.regions[r].get_hebbian_trace()
                    if dmn_trace is not None:
                        # Apply DMN traces at reduced strength
                        self.cortex.regions[r].hebbian_update(
                            dmn_trace, dmn.association_strength,
                        )

    def _consolidate(self) -> ConsolidationReport:
        records = []
        if self.hippocampus:
            records = self.hippocampus.drain()

        plasticity_rates = self.plasticity_system.all_rates(
            self._developmental_age,
            self._mood_baseline,
            {r: float(s.mean().item()) for r, s in self._ewc_scalars.items()},
        )

        return self.consolidation_engine.run(
            records=records,
            cortex=self.cortex,
            ewc_scalars=self._ewc_scalars,
            personality=self._personality,
            plasticity_rates=plasticity_rates,
            developmental_age=self._developmental_age,
        )

    def _save_state(self) -> None:
        state = BrainState(
            brainstem_weights=self.brainstem.get_weights(),
            cortex_weights=self.cortex.get_weights(),
            inter_region_weights=self.cortex.get_inter_region_weights(),
            ewc_protection=self._ewc_scalars,
            personality=self._personality,
            mood_baseline=self._mood_baseline,
            developmental_age=self._developmental_age,
            plasticity_rates=self.plasticity_system.all_rates(
                self._developmental_age,
                self._mood_baseline,
            ),
            consolidation_meta=self._consolidation_meta,
            inter_region_highway=self.cortex.wiring.get_highway_map(),
            neuromodulator_baseline=self._neuromodulator_baseline,
            thalamus_weights=self.thalamus.get_weights(),
            cerebellum_weights=self.cerebellum.get_weights(),
            astrocyte_usage=self.astrocyte.get_usage(),
            cortex_predictions=self.cortex.get_predictions(),
            habit_store=self.basal_ganglia.get_state(),
        )
        self.weight_store.save(state, genome=self._genome)

    def _try_restore(self) -> None:
        try:
            state = self.weight_store.load()
        except Exception as exc:
            print("Warning: failed to load saved brain state; starting fresh.")
            print(f"  Reason: {exc}")
            self.weight_store.delete()
            return
        if state is None:
            return

        self.brainstem.load_weights(state.brainstem_weights)
        self.cortex.set_weights(state.cortex_weights)
        self.cortex.set_inter_region_weights(state.inter_region_weights)
        self._ewc_scalars = state.ewc_protection
        self._personality = state.personality
        self._mood_baseline = state.mood_baseline
        self._developmental_age = state.developmental_age
        self._consolidation_meta = state.consolidation_meta or {}

        for key, strength in state.inter_region_highway.items():
            if key in self.cortex.wiring.highway_strengths:
                self.cortex.wiring.highway_strengths[key] = strength

        # Restore new component weights (backward compat: missing = keep random init)
        if state.neuromodulator_baseline is not None:
            self._neuromodulator_baseline = state.neuromodulator_baseline
        if state.thalamus_weights:
            self.thalamus.set_weights(state.thalamus_weights)
        if state.cerebellum_weights:
            self.cerebellum.set_weights(state.cerebellum_weights)
        if state.astrocyte_usage:
            self.astrocyte.set_usage(state.astrocyte_usage)
        if state.cortex_predictions:
            self.cortex.set_predictions(state.cortex_predictions)
        if state.habit_store:
            self.basal_ganglia.set_state(state.habit_store)

    def _tokenize(self, text: str) -> torch.Tensor:
        if self._tokenizer is not None:
            ids = self._tokenizer.encode(
                text,
                truncation=True,
                max_length=self._genome.topology.hippocampus_capacity,
            )
            vocab = self._genome.topology.vocab_size
            if vocab > 0:
                ids = [min(int(t), vocab - 1) for t in ids]
            return torch.tensor(ids, dtype=torch.long)

        vocab = self._genome.topology.vocab_size
        tokens = [min(b, vocab - 1) for b in text.encode("utf-8")]
        return torch.tensor(tokens, dtype=torch.long)

    def _detokenize(self, token_ids: list[int]) -> str:
        if self._tokenizer is not None:
            try:
                return self._tokenizer.decode([int(t) for t in token_ids], clean_up_tokenization_spaces=True)
            except Exception:
                pass

        bytes_list = bytes(min(t, 255) for t in token_ids)
        return bytes_list.decode("utf-8", errors="replace")

    @property
    def personality(self) -> PersonalityVector:
        return self._personality

    @property
    def developmental_age(self) -> int:
        return self._developmental_age

    @property
    def developmental_stage(self) -> str:
        return self.plasticity_system.stage(self._developmental_age).value

    def status(self) -> dict:
        mood = self.limbic.mood if self.limbic else self._mood_baseline
        neuromod = self.limbic.neuromodulators if self.limbic else self._neuromodulator_baseline
        return {
            "developmental_age": self._developmental_age,
            "developmental_stage": self.developmental_stage,
            "personality": {
                name: round(getattr(self._personality, name), 4)
                for name in PersonalityVector.TRAIT_NAMES
            },
            "mood": {
                "valence": round(mood.valence, 4),
                "arousal": round(mood.arousal, 4),
                "openness": round(mood.openness, 4),
            },
            "neuromodulators": {
                "dopamine": round(neuromod.dopamine, 4),
                "serotonin": round(neuromod.serotonin, 4),
                "acetylcholine": round(neuromod.acetylcholine, 4),
                "norepinephrine": round(neuromod.norepinephrine, 4),
            },
            "plasticity": self.plasticity_system.describe(self._developmental_age, mood),
            "hippocampus_utilization": (
                round(self.hippocampus.utilization, 4)
                if self.hippocampus else 0.0
            ),
            "session_active": self._session_active,
            "brainstem_frozen": self.brainstem.is_frozen,
            "consolidation_meta": self._consolidation_meta,
            "region_weights": (
                {
                    r.value: round(w, 4)
                    for r, w in self.cortex.voter.compute_region_weights(self._personality).items()
                }
            ),
            "inter_region_highways": {
                f"{r1.value}-{r2.value}": round(s, 6)
                for (r1, r2), s in self.cortex.wiring.highway_strengths.items()
            },
        }
