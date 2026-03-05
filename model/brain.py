"""Brain - Top-level orchestrator that wires all tiers together."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch

from .brainstem import Brainstem
from .consolidation import ConsolidationEngine, ConsolidationReport
from .cortex import Cortex
from .generation import GenerationInterface
from .genome import Genome
from .hippocampus import Hippocampus
from .limbic import LimbicSystem
from .novelty import NoveltyDetector
from .plasticity import PlasticitySystem
from .tensor import seed_all
from .types_ import (
    BrainState,
    CortexRegion,
    HebbianTrace,
    MoodState,
    PersonalityVector,
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
        self._developmental_age = 0
        self._ewc_scalars: dict[CortexRegion, torch.Tensor] = {}
        self._consolidation_meta: dict = {
            "total_sessions": 0,
            "total_consolidations": 0,
            "avg_reinforcement": 0.0,
        }

        self.brainstem = Brainstem(self._genome)
        self.cortex = Cortex(self._genome)
        self.novelty_detector = NoveltyDetector(self._genome)
        self.consolidation_engine = ConsolidationEngine(self._genome)
        self.plasticity_system = PlasticitySystem(self._genome)
        self.generation = GenerationInterface(self._genome)

        self.limbic: LimbicSystem | None = None
        self.hippocampus: Hippocampus | None = None

        self._tokenizer = tokenizer

        self._previous_model_text: str | None = None
        self._previous_traces: list[HebbianTrace] = []
        self._session_active = False

        if restore:
            self._try_restore()

    def birth(self, pretraining_corpus: list[torch.Tensor] | None = None) -> None:
        if pretraining_corpus:
            self.brainstem.pretrain(pretraining_corpus)
        else:
            self.brainstem.freeze()

        self._personality = PersonalityVector()
        self._mood_baseline = MoodState()
        self._developmental_age = 0
        self._ewc_scalars = {}
        self._save_state()

    def session_start(self) -> None:
        self.limbic = LimbicSystem(self._genome, self._mood_baseline)
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

        if self.hippocampus:
            self._developmental_age = self.hippocampus.interaction_count

        report = self._consolidate()
        self._consolidation_meta["total_sessions"] += 1
        self._consolidation_meta["total_consolidations"] += 1
        self._save_state()
        return report

    def process_turn(self, user_text: str) -> TurnResult:
        assert self._session_active, "Call session_start() before processing turns"
        assert self.limbic is not None
        assert self.hippocampus is not None

        token_ids = self._tokenize(user_text)
        structured = self.brainstem.process(token_ids)

        novelty_score = self.novelty_detector.score(structured, self.cortex.forward_readonly)

        ctx_input = self.limbic.process(structured, user_text, self._previous_model_text, novelty_score)
        reinforcement_strength = sum(s.strength for s in ctx_input.reinforcement_signals)

        combined_logits, traces, activations = self.cortex.forward(ctx_input, self._personality)

        generated_tokens = self.generation.generate_sequence(
            combined_logits,
            self.limbic.mood,
            novelty_score,
            max_tokens=64,
            brainstem=self.brainstem,
        )

        self.hippocampus.record(
            token_ids=token_ids,
            traces=traces,
            novelty_score=novelty_score,
            mood=self.limbic.mood,
            reinforcement=reinforcement_strength,
        )

        plasticity_rates = self.plasticity_system.all_rates(
            self._developmental_age,
            self.limbic.mood,
            {r: float(s.mean().item()) for r, s in self._ewc_scalars.items()},
        )

        partial_applied = self.generation.partial_weight_update(
            self.cortex,
            self._previous_traces,
            reinforcement_strength,
            plasticity_rates,
            self._ewc_scalars,
        )

        if self.hippocampus.needs_consolidation:
            self._consolidate()

        self._previous_model_text = self._detokenize(generated_tokens)
        self._previous_traces = traces
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
        )

    def prepare_turn(self, user_text: str) -> tuple[torch.Tensor, list, float, float]:
        assert self._session_active, "Call session_start() before processing turns"
        assert self.limbic is not None
        assert self.hippocampus is not None

        token_ids = self._tokenize(user_text)
        structured = self.brainstem.process(token_ids)

        novelty_score = self.novelty_detector.score(structured, self.cortex.forward_readonly)
        ctx_input = self.limbic.process(structured, user_text, self._previous_model_text, novelty_score)
        reinforcement_strength = sum(s.strength for s in ctx_input.reinforcement_signals)

        combined_logits, traces, _ = self.cortex.forward(ctx_input, self._personality)

        self.hippocampus.record(
            token_ids=token_ids,
            traces=traces,
            novelty_score=novelty_score,
            mood=self.limbic.mood,
            reinforcement=reinforcement_strength,
        )

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

        plasticity_rates = self.plasticity_system.all_rates(
            self._developmental_age,
            self.limbic.mood,
            {r: float(s.mean().item()) for r, s in self._ewc_scalars.items()},
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

        if self.hippocampus.needs_consolidation:
            self._consolidate()

        self._previous_model_text = self._detokenize(generated_tokens)
        self._previous_traces = traces
        self._developmental_age += 1

    def count_parameters(self) -> int:
        total = 0
        for arr in self.brainstem.get_weights().values():
            total += int(arr.numel())
        for region_module in self.cortex.regions.values():
            total += int(region_module.W_in.numel())
            total += int(region_module.W_hidden.numel())
            total += int(region_module.W_out.numel())
        return total

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
