"""Model - Top-level orchestrator that wires all components together.

Processing flow:
  Input arrives
    → Router module (decides where to send the signal)
    → Predictive coding (each transformer layer generates prediction, computes error)
    → Error signals propagate (not raw input)
    → Neuromodulation updates learning gains
    → Transformer layers process with sparse activations
    → Habit module checks for cached patterns
    → Corrector predicts outcome, adjusts logits
    → Decoder produces next-token logits using recurrent context
    → Output

  Between sessions:
    → DMN phase (internal thought generation)
    → Three-stage sleep consolidation
    → Astrocyte layer applies slow metabolic penalties
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import math
from pathlib import Path

import torch
import torch.nn.functional as F

from .regularizer import Regularizer
from .habit import HabitSystem
from .encoder import Encoder
from .corrector import Corrector
from .replay import ReplayEngine, ConsolidationReport
from .transformer import Transformer
from .decoder import Decoder
from .genome import Genome, mutate_genome
from .memory import Memory
Hippocampus = Memory
from .reinforcement import ReinforcementSystem
LimbicSystem = ReinforcementSystem
from .surprise import SurpriseDetector
from .adaptation import AdaptationSystem
from .tensor import DTYPE, get_device, seed_all
from .router import Router
from .types_ import (
    ModelState,
    TransformerRegion,
    HebbianTrace,
    MoodState,
    NeuromodulatorState,
    PersonalityVector,
    ReinforcementSignal,
    ReinforcementType,
    EpisodicRecord,
    FailureRecord,
    FailureMode,
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
    region_weights: dict[TransformerRegion, float]
    neuromodulators: NeuromodulatorState | None = None
    prediction_error: float = 0.0
    thalamic_routing: dict[TransformerRegion, float] | None = None
    cascade_active: bool = False


# ---------------------------------------------------------------------------
# Mortality Monitor
# ---------------------------------------------------------------------------

class MortalityMonitor:
    """Tracks rolling per-cycle loss and detects death-worthy failure states.

    An instance is considered dead when its rolling training loss stays above
    `cycle_loss_threshold` for `consecutive_cycles` consolidation cycles in a
    row.  A single bad session does not trigger death; sustained, unimproving
    failure does.

    Failure mode inference
    ----------------------
    - CATASTROPHIC_FORGETTING: early cycles had low loss, recent ones are high.
    - NEVER_LEARNED: loss consistently high with low variance.
    - REPETITIVE_OUTPUT: loss is moderate but cascades fired repeatedly
      (tracked via `record_cascade()`).
    - UNKNOWN: fallback.
    """

    def __init__(self, genome: Genome) -> None:
        self._params = genome.mortality
        self._cycle_losses: deque[float] = deque(maxlen=self._params.loss_window)
        self._consecutive_failures: int = 0
        self._high_loss_inputs: list[tuple[float, list[int]]] = []
        self._cascade_count: int = 0   # cascades fired in current window

    def record_cycle(
        self,
        loss: float,
        high_loss_inputs: list[tuple[float, list[int]]] | None = None,
    ) -> None:
        """Record the average training loss for the just-completed cycle."""
        self._cycle_losses.append(loss)
        if high_loss_inputs:
            self._high_loss_inputs.extend(high_loss_inputs)
            self._high_loss_inputs.sort(key=lambda x: x[0], reverse=True)
            self._high_loss_inputs = self._high_loss_inputs[:5]
        if loss > self._params.cycle_loss_threshold:
            self._consecutive_failures += 1
        else:
            self._consecutive_failures = 0

    def record_cascade(self) -> None:
        """Note that a catastrophic cascade fired this session."""
        self._cascade_count += 1

    def is_dying(self) -> bool:
        """True when sustained failure warrants death-and-replacement."""
        return self._consecutive_failures >= self._params.consecutive_cycles

    def _infer_failure_mode(self) -> FailureMode:
        losses = list(self._cycle_losses)
        if not losses:
            return FailureMode.UNKNOWN
        # Catastrophic forgetting: began well, ended badly
        if len(losses) >= 4:
            early = sum(losses[:2]) / 2
            late = sum(losses[-2:]) / 2
            if late > early * 2.0 and early < self._params.cycle_loss_threshold:
                return FailureMode.CATASTROPHIC_FORGETTING
        # Never learned: consistently high loss, low variance
        if len(losses) >= 3:
            avg = sum(losses) / len(losses)
            variance = sum((l - avg) ** 2 for l in losses) / len(losses)
            if avg > self._params.cycle_loss_threshold and variance < 0.5:
                return FailureMode.NEVER_LEARNED
        # Repetitive output: many cascade firings but no improvement
        if self._cascade_count >= 3:
            return FailureMode.REPETITIVE_OUTPUT
        return FailureMode.UNKNOWN

    def build_failure_record(
        self,
        predecessor_age: int,
        neuromodulators: NeuromodulatorState,
    ) -> FailureRecord:
        """Construct the immutable postmortem record before the instance dies."""
        return FailureRecord(
            failure_mode=self._infer_failure_mode(),
            predecessor_age=predecessor_age,
            cycle_losses=list(self._cycle_losses),
            neuromodulators_at_death=neuromodulators,
            failure_token_ids=[ids for _, ids in self._high_loss_inputs],
        )

    def reset(self, genome: Genome | None = None) -> None:
        if genome is not None:
            self._params = genome.mortality
        self._consecutive_failures = 0
        self._cycle_losses.clear()
        self._high_loss_inputs.clear()
        self._cascade_count = 0


# ---------------------------------------------------------------------------
# Thalamic routing stub
# ---------------------------------------------------------------------------

class Thalamus:
    """Routes encoder embeddings to cortex regions via learned gating weights.

    Initialises W_route and W_feedback matrices so that count_parameters()
    works correctly.  The stub route() returns an empty dict, which causes
    Transformer.forward to use its default routing weight of 1.0 for every
    region — equivalent to no routing bias.
    """

    def __init__(self, genome: Genome) -> None:
        topo = genome.topology
        device = get_device()
        n_regions = len(topo.regions)
        self.W_route = torch.zeros(topo.embed_dim, n_regions, device=device, dtype=DTYPE)
        self.W_feedback = torch.zeros(
            topo.transformer_hidden, topo.embed_dim, device=device, dtype=DTYPE
        )

    def route(
        self,
        embeddings: torch.Tensor,
        neuromod: NeuromodulatorState,
        novelty_score: float,
    ) -> dict[TransformerRegion, float]:
        return {}  # empty → each region uses default weight 1.0

    def receive_feedback(
        self, region_hiddens: dict[TransformerRegion, torch.Tensor]
    ) -> None:
        pass

    def get_weights(self) -> dict[str, torch.Tensor]:
        return {"W_route": self.W_route, "W_feedback": self.W_feedback}

    def set_weights(self, weights: dict[str, torch.Tensor]) -> None:
        if "W_route" in weights:
            self.W_route = weights["W_route"]
        if "W_feedback" in weights:
            self.W_feedback = weights["W_feedback"]


# ---------------------------------------------------------------------------
# Cerebellar forward-model corrector stub
# ---------------------------------------------------------------------------

class Cerebellum:
    """Forward-model corrector: pre-adjusts logits based on predicted outcome.

    Stub: initialises W_in / W_out for parameter counting and passes logits
    through unchanged until a learning rule is trained in.
    """

    def __init__(self, genome: Genome) -> None:
        topo = genome.topology
        device = get_device()
        hidden = max(topo.transformer_hidden // 2, 1)
        self.W_in = torch.zeros(topo.embed_dim, hidden, device=device, dtype=DTYPE)
        self.W_out = torch.zeros(hidden, topo.vocab_size, device=device, dtype=DTYPE)

    def pre_correct_logits(
        self, logits: torch.Tensor, embeddings: torch.Tensor
    ) -> torch.Tensor:
        return logits  # pass-through until cerebellar learning is implemented

    def train_step(self, embeddings: torch.Tensor, reinforcement: float) -> None:
        pass

    def get_weights(self) -> dict[str, torch.Tensor]:
        return {"W_in": self.W_in, "W_out": self.W_out}

    def set_weights(self, weights: dict[str, torch.Tensor]) -> None:
        if "W_in" in weights:
            self.W_in = weights["W_in"]
        if "W_out" in weights:
            self.W_out = weights["W_out"]


# ---------------------------------------------------------------------------
# Astrocyte metabolic monitor stub
# ---------------------------------------------------------------------------

class Astrocyte:
    """Tracks per-region firing activity and applies slow metabolic penalties.

    Uses the same AstrocyteParams as the Regularizer module (genome.regularizer).
    """

    def __init__(self, genome: Genome) -> None:
        self._params = genome.regularizer  # AstrocyteParams
        self._usage: dict[TransformerRegion, float] = {}

    def decay(self) -> None:
        for r in self._usage:
            self._usage[r] *= self._params.metabolic_decay

    def apply_penalties(self, cortex: object) -> None:
        pass  # stub: penalisation logic to be implemented

    def record_activity(
        self, region: TransformerRegion, fired_indices: torch.Tensor
    ) -> None:
        if region not in self._usage:
            self._usage[region] = 0.0
        if fired_indices.numel() > 0:
            self._usage[region] += float(fired_indices.numel())

    def get_usage(self) -> dict[TransformerRegion, torch.Tensor]:
        return {r: torch.tensor([v], dtype=DTYPE) for r, v in self._usage.items()}

    def set_usage(self, usage: dict[TransformerRegion, torch.Tensor]) -> None:
        self._usage = {r: float(t.item()) for r, t in usage.items()}


class Model:
    """The complete neural architecture - all modules assembled into a single
    AI model."""

    def __init__(
        self,
        genome: Genome | None = None,
        storage_dir: str | Path = "./model_data",
        seed: int = 42,
        tokenizer: object | None = None,
        restore: bool = True,
    ) -> None:
        self._genome = genome or Genome()
        seed_all(seed)
        self._storage_dir = Path(storage_dir)
        self._seed = seed

        self.weight_store = WeightStore(self._storage_dir)

        self._personality = PersonalityVector()
        self._mood_baseline = MoodState()
        self._neuromodulator_baseline = NeuromodulatorState()
        self._developmental_age = 0
        self._ewc_scalars: dict[TransformerRegion, torch.Tensor] = {}
        self._encoder_ewc_scalars: dict[str, torch.Tensor] = {}
        self._consolidation_meta: dict = {
            "total_sessions": 0,
            "total_consolidations": 0,
            "avg_reinforcement": 0.0,
        }

        # Core components
        self.encoder = Encoder(self._genome)
        self.transformer = Transformer(self._genome)
        self.surprise_detector = SurpriseDetector(self._genome)
        self.replay_engine = ReplayEngine(self._genome)
        self.adaptation_system = AdaptationSystem(self._genome)
        self.decoder = Decoder(self._genome)

        # Additional modules
        self.router = Router(self._genome)
        self.habit_system = HabitSystem(self._genome)
        self.corrector = Corrector(self._genome)
        self.regularizer = Regularizer(self._genome)

        # Backwards compatibility and biological aliases
        self.cortex = self.transformer
        self.thalamus = self.router
        self.cerebellum = self.corrector
        self.astrocyte = self.regularizer
        self.basal_ganglia = self.habit_system          # habit system = basal ganglia
        self.plasticity_system = self.adaptation_system # adaptation = plasticity gating
        self.generation = self.decoder                  # decoder = token generation
        self.novelty_detector = self.surprise_detector  # surprise = novelty detection
        self.consolidation_engine = self.replay_engine  # replay = consolidation

        self.reinforcement: ReinforcementSystem | None = None
        self.memory: Memory | None = None

        self._tokenizer = tokenizer

        self._previous_model_text: str | None = None
        self._previous_traces: list[HebbianTrace] = []
        self._last_structured = None  # cached for soft-prompt anchoring
        self._session_active = False
        self._session_token_buffer: list[torch.Tensor] = []

        # Mortality tracking (death-and-replacement)
        self._mortality_monitor = MortalityMonitor(self._genome)
        self._pending_failure_records: list[FailureRecord] = []
        self._train_loss_buffer: list[float] = []

        # gradient training support ------------------------------------------------
        # ensure all trainable tensors request gradients
        for p in self.parameters():
            p.requires_grad_(True)

        grad_cfg = self._genome.gradient
        # Cache the param list once so we don't rebuild it on every step
        self._grad_params: list[torch.Tensor] = list(self.parameters())
        if grad_cfg.lr > 0.0:
            self._optimizer = torch.optim.AdamW(
                self._grad_params,
                lr=grad_cfg.lr,
                weight_decay=grad_cfg.weight_decay,
            )
        else:
            self._optimizer = None

        if restore:
            self._try_restore()

    def parameters(self) -> list[torch.Tensor]:
        """Gather all tensors that should receive gradient updates.

        This mirrors the various components' own `parameters()` helpers so an
        optimizer can treat the entire architecture like a single model.
        """
        params: list[torch.Tensor] = []
        params.extend(self.encoder.parameters())
        params.extend(self.transformer.parameters())
        params.extend(self.router.parameters())
        params.extend(self.corrector.parameters())
        # regularizer, habit system, memory, etc. are non-gradient
        return params

    # ------------------------------------------------------------------
    # Gradient-based language modelling utilities
    # ------------------------------------------------------------------

    def _compute_lm_region_logits(self, token_ids: torch.Tensor) -> dict[TransformerRegion, torch.Tensor]:
        """Return next-token logits for each active region independently."""
        token_ids = token_ids.to(device=get_device(), dtype=torch.long)
        seq_len = int(token_ids.shape[0])
        if seq_len < 2:
            return {}

        with torch.amp.autocast("cuda", enabled=False):
            structured = self.encoder.process(token_ids)
            x = F.layer_norm(structured.embeddings[:-1], [structured.embeddings.shape[-1]])

            region_logits: dict[TransformerRegion, torch.Tensor] = {}
            for region in self._genome.topology.active_regions:
                m = self.cortex.regions.get(region)
                if m is None:
                    continue
                h = F.relu(x @ m.W_in)
                h = F.layer_norm(h, [h.shape[-1]], m._ln_scale, m._ln_bias)
                h = F.relu(h @ m.W_hidden)
                region_logits[region] = h @ m.W_out
            return region_logits

    def _compute_lm_logits(self, token_ids: torch.Tensor) -> torch.Tensor:
        """Return voter-combined next-token logits for every position."""
        token_ids = token_ids.to(device=get_device(), dtype=torch.long)
        seq_len = int(token_ids.shape[0])
        if seq_len < 2:
            return torch.empty((0, self._genome.topology.vocab_size), device=get_device())

        region_logits = self._compute_lm_region_logits(token_ids)
        if region_logits:
            region_weights = self.cortex.voter.compute_region_weights(self._personality)
            present = [region for region in region_logits if region in region_weights]
            total_weight = sum(region_weights[region] for region in present) or 1.0
            combined: torch.Tensor | None = None
            for region in present:
                weight = region_weights[region] / total_weight
                logits = region_logits[region]
                combined = logits * weight if combined is None else combined + logits * weight
            if combined is not None:
                return combined

        with torch.amp.autocast("cuda", enabled=False):
            structured = self.encoder.process(token_ids)
            x = F.layer_norm(structured.embeddings[:-1], [structured.embeddings.shape[-1]])
            return x @ self.encoder.token_embeddings.T

    def _online_train(self, token_ids: torch.Tensor) -> float | None:
        """Perform a single gradient step on the provided input tokens.

        This is a thin wrapper around :meth:`_compute_lm_logits`.  Training
        on individual sequences is noisy; for more stability use
        :meth:`_online_train_batch` and a batch size >1.

        Note: next-token prediction is just one possible feedback signal.  For
        better pattern recognition you might replace this loss with a
        downstream task or contrastive objective; see project notes for ideas.
        """
        if self._optimizer is None or token_ids.numel() < 2:
            return None
        loss = self._compute_lm_loss(token_ids)
        if loss is None:
            return None
        grad_cfg = self._genome.gradient
        self._optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self._grad_params, grad_cfg.grad_clip_norm)
        self._optimizer.step()
        self._train_loss_buffer.append(float(loss.item()))
        return float(loss.item())

    def _online_train_batch(self, token_list: list[torch.Tensor]) -> float | None:
        """Perform one gradient step over a batch of sequences.

        The losses for each sequence are averaged to compute the gradient.  This
        simple batching avoids variable-length padding by iterating, but still
        shares a single optimizer step.
        """
        if self._optimizer is None or not token_list:
            return None
        grad_cfg = self._genome.gradient
        total_loss: torch.Tensor | None = None
        count = 0
        for token_ids in token_list:
            if token_ids.numel() < 2:
                continue
            seq_loss = self._compute_lm_loss(token_ids)
            if seq_loss is None:
                continue
            total_loss = seq_loss if total_loss is None else total_loss + seq_loss
            count += 1
        if count == 0 or total_loss is None:
            return None
        loss = total_loss / count
        self._optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self._grad_params, grad_cfg.grad_clip_norm)
        self._optimizer.step()
        self._train_loss_buffer.append(float(loss.item()))
        return float(loss.item())

    def _compute_lm_loss(self, token_ids: torch.Tensor) -> torch.Tensor | None:
        """Return raw cross-entropy loss tensor without stepping the optimizer.

        Intended for gradient accumulation loops in external trainers (e.g.
        pretrain.py) where the caller manages ``zero_grad``, ``backward``, and
        ``optimizer.step`` directly.  Uses the same label-smoothing setting as
        :meth:`_online_train` so evaluation metrics are consistent.
        """
        if token_ids.numel() < 2:
            return None
        grad_cfg = self._genome.gradient

        token_ids = token_ids.to(device=get_device(), dtype=torch.long)
        seq_len = int(token_ids.shape[0])
        if seq_len >= 2:
            with torch.amp.autocast("cuda", enabled=False):
                structured = self.encoder.process(token_ids)
                x = F.layer_norm(structured.embeddings[:-1], [structured.embeddings.shape[-1]])
                targets = token_ids[1: x.shape[0] + 1].to(device=x.device)
                region_weights = self.cortex.voter.compute_region_weights(self._personality)
                total_weight = 0.0
                weighted_loss: torch.Tensor | None = None

                for region in self._genome.topology.active_regions:
                    m = self.cortex.regions.get(region)
                    if m is None:
                        continue
                    weight = float(region_weights.get(region, 0.0))
                    if weight <= 0.0:
                        continue

                    h = F.relu(x @ m.W_in)
                    h = F.layer_norm(h, [h.shape[-1]], m._ln_scale, m._ln_bias)
                    h = F.relu(h @ m.W_hidden)
                    logits = h @ m.W_out
                    region_loss = F.cross_entropy(
                        logits,
                        targets,
                        label_smoothing=grad_cfg.label_smoothing,
                    )
                    weighted_loss = region_loss * weight if weighted_loss is None else weighted_loss + region_loss * weight
                    total_weight += weight

                if weighted_loss is not None:
                    return weighted_loss / max(total_weight, 1e-8)

        logits = self._compute_lm_logits(token_ids)
        if logits.numel() == 0:
            return None

        # Append EOS token to targets if available and not already present
        token_ids_use = token_ids
        eos_id = getattr(self._tokenizer, "eos_token_id", None)
        if eos_id is not None and int(token_ids_use[-1].item()) != eos_id:
            token_ids_use = torch.cat([token_ids_use, torch.tensor([eos_id], dtype=torch.long, device=token_ids_use.device)])
        targets = token_ids_use[1: logits.shape[0] + 1].to(device=logits.device)
        return F.cross_entropy(logits, targets, label_smoothing=grad_cfg.label_smoothing)

    def _sleep_replay_train_step(self, token_ids: torch.Tensor, replay_scale: float = 1.0) -> float | None:
        if self._optimizer is None or token_ids.numel() < 2:
            return None
        loss = self._compute_lm_loss(token_ids)
        if loss is None:
            return None
        scaled_loss = loss * max(float(replay_scale), 0.1)
        self._optimizer.zero_grad()
        scaled_loss.backward()
        torch.nn.utils.clip_grad_norm_(self._grad_params, self._genome.gradient.grad_clip_norm)
        self._optimizer.step()
        self._train_loss_buffer.append(float(loss.item()))
        return float(loss.item())

    def _update_encoder_ewc(self) -> None:
        params = self._genome.ewc

        token_importance = self.encoder.token_embeddings.detach().abs().mean(dim=1)
        token_importance = token_importance / torch.clamp(token_importance.max(), min=1e-8)
        token_protection = torch.pow(token_importance, params.protection_exponent)
        token_protection = torch.clamp(
            token_protection * params.protection_growth_rate,
            0.0,
            params.max_protection,
        )

        co_importance = self.encoder.cooccurrence_weights.detach().abs().mean(dim=0)
        co_importance = co_importance / torch.clamp(co_importance.max(), min=1e-8)
        co_protection = torch.pow(co_importance, params.protection_exponent)
        co_protection = torch.clamp(
            co_protection * params.protection_growth_rate,
            0.0,
            params.max_protection,
        )

        if "token_embeddings" in self._encoder_ewc_scalars and self._encoder_ewc_scalars["token_embeddings"].shape == token_protection.shape:
            self._encoder_ewc_scalars["token_embeddings"] = torch.maximum(
                self._encoder_ewc_scalars["token_embeddings"], token_protection,
            )
        else:
            self._encoder_ewc_scalars["token_embeddings"] = token_protection

        if "cooccurrence_weights" in self._encoder_ewc_scalars and self._encoder_ewc_scalars["cooccurrence_weights"].shape == co_protection.shape:
            self._encoder_ewc_scalars["cooccurrence_weights"] = torch.maximum(
                self._encoder_ewc_scalars["cooccurrence_weights"], co_protection,
            )
        else:
            self._encoder_ewc_scalars["cooccurrence_weights"] = co_protection

    def birth(self, pretraining_corpus: list[torch.Tensor] | None = None) -> None:
        if pretraining_corpus:
            self.encoder.pretrain(pretraining_corpus)

        # Keep encoder plastic after birth so user interactions can continue
        # to shape low-level token structure online.
        self.encoder.unfreeze()

        self._personality = PersonalityVector()
        self._mood_baseline = MoodState()
        self._developmental_age = 0
        self._ewc_scalars = {}
        self._encoder_ewc_scalars = {}
        self._save_state()

    def _record_session_loss(self, avg_loss: float) -> None:
        """Record the average training loss for the just-completed session.

        Minimal implementation: append to the internal buffer for later analysis.
        """
        try:
            self._train_loss_buffer.append(float(avg_loss))
        except Exception:
            # Silently ignore failures here; recording loss is best-effort
            pass

    def session_start(self) -> None:
        self.limbic = LimbicSystem(
            self._genome, self._mood_baseline, self._neuromodulator_baseline,
        )
        self.reinforcement = self.limbic
        self.hippocampus = Hippocampus(self._genome)
        self.memory = self.hippocampus
        self.hippocampus.set_interaction_counter(self._developmental_age)

        self._previous_model_text = None
        self._previous_traces = []
        self._session_token_buffer = []
        self._session_active = True

    def session_end(self) -> ConsolidationReport | None:
        if not self._session_active:
            return None

        self._session_active = False

        if self.limbic:
            self._mood_baseline = self.limbic.session_end_baseline()
            self._neuromodulator_baseline = self.limbic.session_end_neuromodulator_baseline()
            # Propagate cascade events to mortality monitor
            if self.limbic.cascade_active:
                self._mortality_monitor.record_cascade()

        if self.hippocampus:
            self._developmental_age = self.hippocampus.interaction_count

        if self._session_token_buffer and self._optimizer is not None:
            try:
                self._online_train_batch(self._session_token_buffer)
            except Exception:
                pass
        self._session_token_buffer = []

        # DMN phase: internal thought generation before consolidation
        try:
            self._run_dmn_phase()
        except Exception:
            pass  # DMN is best-effort; don't block shutdown

        report = self._consolidate()

        # Astrocyte: decay usage counters and apply metabolic penalties
        self.astrocyte.decay()
        if self._developmental_age % self._genome.regularizer.update_interval == 0:
            self.astrocyte.apply_penalties(self.cortex)

        # Basal ganglia: decay unused habits
        self.basal_ganglia.decay_habits()

        self._consolidation_meta["total_sessions"] += 1
        self._consolidation_meta["total_consolidations"] += 1

        # Death-and-replacement check: sustained failure triggers instance cycle.
        if self._mortality_monitor.is_dying():
            self._execute_death_and_replacement()

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
        structured = self.encoder.process(token_ids)

        # Online encoder update: light Hebbian pass on this turn's tokens
        self.encoder.online_update(token_ids, ewc_scalars=self._encoder_ewc_scalars)
        self._session_token_buffer.append(token_ids.detach().cpu())

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
            norepinephrine=neuromod.norepinephrine, surprise=novelty_score,
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
                self._personality,
                max_tokens=_max_tokens,
                eos_token=getattr(self._tokenizer, "eos_token_id", None) if self._tokenizer is not None else None,
                encoder=self.encoder,
                transformer=self.cortex,
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
            cascade_active=self.limbic.cascade_active,
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
        structured = self.encoder.process(token_ids)

        # Online encoder update for streaming/chat-server path.
        self.encoder.online_update(token_ids, ewc_scalars=self._encoder_ewc_scalars)
        self._session_token_buffer.append(token_ids.detach().cpu())

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
            norepinephrine=neuromod.norepinephrine, surprise=novelty_score,
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

        self.hippocampus.record(
            token_ids=token_ids,
            traces=traces,
            novelty_score=novelty_score,
            mood=self.limbic.mood,
            reinforcement=reinforcement_strength,
        )

        # Store structured embeddings for soft-prompt anchoring in the server
        self._last_structured = structured

        # real-time gradient training on user input (language model loss)
        if self._optimizer is not None:
            try:
                self._online_train(token_ids)
            except Exception:
                pass  # training failures shouldn't break conversation

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
        for arr in self.encoder.get_weights().values():
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

    def _failure_record_to_episodics(self, fr: FailureRecord) -> list[EpisodicRecord]:
        """Convert a FailureRecord into EpisodicRecords for consolidation replay.

        Failure experiences are injected at maximum priority and with maximum
        negative reinforcement so the consolidation engine treats them as the
        single most important thing to learn from.
        """
        result = []
        for token_ids_list in fr.failure_token_ids:
            if not token_ids_list:
                continue
            tids = torch.tensor(token_ids_list, dtype=torch.long)
            rec = EpisodicRecord(
                token_ids=tids,
                hebbian_traces=[],
                novelty_score=1.0,
                mood_at_event=MoodState(valence=-1.0, arousal=1.0, openness=0.8),
                reinforcement=-1.0,
                interaction_number=fr.predecessor_age,
                repetition_count=0,
                consolidation_priority=1.0,
                prediction_error=1.0,
                neuromodulators_at_event=fr.neuromodulators_at_death,
            )
            result.append(rec)
        return result

    def _execute_death_and_replacement(self) -> None:
        """Terminate the current instance configuration and spawn a successor.

        The successor inherits all weights unchanged.  What changes is:
          1. The failure record (immutable memory) is queued for the next
             consolidation pass, so the successor wakes with knowledge of
             what killed its predecessor.
          2. The genome receives a small directional mutation based on the
             identified failure mode, preventing the same mistake from
             recurring across the lineage.
          3. Mortality counters reset so the successor gets a clean slate.
        """
        neuromod = (
            self.limbic.neuromodulators
            if hasattr(self, "limbic") and self.limbic is not None
            else self._neuromodulator_baseline
        )
        failure_record = self._mortality_monitor.build_failure_record(
            predecessor_age=self._developmental_age,
            neuromodulators=neuromod,
        )
        # Queue failure memory for the successor's first consolidation pass
        self._pending_failure_records.append(failure_record)
        # Mutate genome: small directional correction away from failure mode
        self._genome = mutate_genome(self._genome, failure_record)
        # Reset mortality tracking for the successor
        self._mortality_monitor.reset(self._genome)

    def _run_dmn_phase(self) -> None:
        """DMN: internal thought generation between sessions.

        The model generates internal "thought" sequences not directed at
        any external input.  These draw on recent hippocampus buffer
        contents and explore associative connections.

        The Hebbian traces get added to the consolidation buffer.
        """
        dmn = self._genome.dmn
        patterns = self.cortex.run_autonomous(steps=dmn.thought_steps)
        for act_dict in patterns:
            if len(act_dict) >= 2:
                self.cortex.wiring.hebbian_update(act_dict, dmn.association_strength)

    def _consolidate(self) -> ConsolidationReport:
        records = []
        if self.hippocampus:
            records = self.hippocampus.drain()

        # Inject pending immutable failure records from a predecessor death.
        # These replay at maximum priority so the successor wakes from its
        # first sleep already knowing what killed the prior instance.
        if self._pending_failure_records:
            failure_episodics = []
            for fr in self._pending_failure_records:
                failure_episodics.extend(self._failure_record_to_episodics(fr))
            records = failure_episodics + records
            self._pending_failure_records.clear()

        # Track cycle loss for mortality monitoring.
        if self._train_loss_buffer:
            _cycle_loss = sum(self._train_loss_buffer) / len(self._train_loss_buffer)
            self._train_loss_buffer.clear()
        elif records:
            _neg = [r.reinforcement for r in records if r.reinforcement < 0]
            _cycle_loss = (abs(sum(_neg) / len(_neg)) * 2.5) if _neg else 1.0
        else:
            _cycle_loss = 1.0

        _high_loss = sorted(
            [
                (abs(r.reinforcement), r.token_ids.tolist())
                for r in records
                if r.reinforcement < -0.5 and not getattr(r, 'is_immutable', False)
            ],
            reverse=True,
        )[:5]
        self._mortality_monitor.record_cycle(_cycle_loss, _high_loss or None)

        plasticity_rates = self.plasticity_system.all_rates(
            self._developmental_age,
            self._mood_baseline,
            {r: float(s.mean().item()) for r, s in self._ewc_scalars.items()},
        )

        report = self.consolidation_engine.run(
            records=records,
            transformer=self.cortex,
            ewc_scalars=self._ewc_scalars,
            personality=self._personality,
            plasticity_rates=plasticity_rates,
            developmental_age=self._developmental_age,
            sleep_trainer=self._sleep_replay_train_step,
        )
        self._update_encoder_ewc()
        return report

    def _save_state(self) -> None:
        state = ModelState(
            encoder_weights=self.encoder.get_weights(),
            encoder_ewc_protection=self._encoder_ewc_scalars,
            transformer_weights=self.transformer.get_weights(),
            inter_region_weights=self.transformer.get_inter_region_weights(),
            ewc_protection=self._ewc_scalars,
            personality=self._personality,
            mood_baseline=self._mood_baseline,
            developmental_age=self._developmental_age,
            plasticity_rates=self.adaptation_system.all_rates(
                self._developmental_age,
                self._mood_baseline,
            ),
            consolidation_meta=self._consolidation_meta,
            inter_region_highway=self.transformer.wiring.get_highway_map(),
            neuromodulator_baseline=self._neuromodulator_baseline,
            router_weights=self.router.get_weights(),
            corrector_weights=self.corrector.get_weights(),
            regularizer_usage=self.regularizer.get_usage(),
            transformer_predictions=self.transformer.get_predictions(),
            habit_store=self.habit_system.get_state(),
            thalamus_weights=self.thalamus.get_weights(),
            cerebellum_weights=self.cerebellum.get_weights(),
            astrocyte_usage=self.astrocyte.get_usage(),
            cortex_predictions=self.transformer.get_predictions(),
        )
        self.weight_store.save(state, genome=self._genome, seed=self._seed)

    def _try_restore(self) -> None:
        try:
            state = self.weight_store.load()
        except Exception as exc:
            print("Warning: failed to load saved model state; starting fresh.")
            print(f"  Reason: {exc}")
            self.weight_store.delete()
            return
        if state is None:
            return

        self.encoder.load_weights(state.encoder_weights)
        self._encoder_ewc_scalars = state.encoder_ewc_protection or {}
        self.transformer.set_weights(state.transformer_weights)
        self.transformer.set_inter_region_weights(state.inter_region_weights)
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
            "encoder_frozen": self.encoder.is_frozen,
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
