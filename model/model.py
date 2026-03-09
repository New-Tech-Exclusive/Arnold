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
from .genome import Genome
from .memory import Memory
from .reinforcement import ReinforcementSystem
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
)
from .weight_store import WeightStore


@dataclass
class TurnResult:
    """Everything produced by a single turn of processing."""

    generated_tokens: list[int]
    surprise_score: float
    mood: MoodState
    reinforcement_strength: float
    partial_update_applied: bool
    region_weights: dict[TransformerRegion, float]
    neuromodulators: NeuromodulatorState | None = None
    prediction_error: float = 0.0
    thalamic_routing: dict[TransformerRegion, float] | None = None


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

        self.reinforcement: ReinforcementSystem | None = None
        self.memory: Memory | None = None

        self._tokenizer = tokenizer

        self._previous_model_text: str | None = None
        self._previous_traces: list[HebbianTrace] = []
        self._last_structured = None  # cached for soft-prompt anchoring
        self._session_active = False

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
        optimizer can treat the entire model like a single model.
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

    def _compute_lm_logits(self, token_ids: torch.Tensor) -> torch.Tensor:
        """Return next-token logits for every position in *token_ids*.

        Uses the same parameters that inference uses — encoder causal
        self-attention followed by each active transformer region's W_in →
        W_hidden → W_out chain.  Gradients therefore flow into the weights
        that actually determine generation quality.

        Does not mutate any Hebbian state (prediction buffers, firing counters,
        etc.), so it is safe to call repeatedly during training.
        """
        token_ids = token_ids.to(device=get_device(), dtype=torch.long)
        seq_len = int(token_ids.shape[0])
        if seq_len < 2:
            return torch.empty((0, self._genome.topology.vocab_size), device=get_device())

        # Run the entire forward pass in fp32, regardless of any outer autocast.
        #
        # Two reasons:
        #  1. The encoder's hand-rolled softmax guards with 1e-12, which
        #     underflows to zero in fp16 → 0/0 = NaN.
        #  2. After Hebbian pretraining, encoder embeddings can reach magnitudes
        #     in the millions.  Feeding those into transformer matmuls under fp16 AMP
        #     (max ≈ 65504) overflows to inf → NaN cross-entropy loss.
        # fp32 handles both cases correctly; this function is not the bottleneck.
        with torch.amp.autocast("cuda", enabled=False):
            # --- Encoder: causal self-attention (no Hebbian side effects) ---
            structured = self.encoder.process(token_ids)
            # (seq_len-1, embed_dim) — prediction inputs for positions 0..T-2
            x = structured.embeddings[:-1]

            # Normalise the encoder output before the transformer.  Hebbian learning
            # amplifies values through co-occurrence → syntax → attention chains;
            # layer_norm brings them back to unit scale so the transformer matmuls
            # operate in a numerically stable regime regardless of training phase.
            x = F.layer_norm(x, [x.shape[-1]])  # still (T-1, embed_dim)

            # --- Transformer: W_in → layer_norm → W_hidden → W_out per region ---
            active_regions = self._genome.topology.active_regions
            combined: torch.Tensor | None = None
            n_regions = 0
            for region in active_regions:
                m = self.transformer.regions.get(region)
                if m is None:
                    continue
                h = F.relu(x @ m.W_in)                                     # (T, transformer_hidden)
                h = F.layer_norm(h, [h.shape[-1]], m._ln_scale, m._ln_bias)
                h = F.relu(h @ m.W_hidden)                                 # (T, transformer_hidden)
                # Accumulate incrementally: avoids holding all region logit
                # tensors in memory simultaneously (each is T × vocab_size).
                r_logits = h @ m.W_out                                     # (T, vocab_size)
                combined = r_logits if combined is None else combined + r_logits
                n_regions += 1

            if combined is not None:
                return combined / n_regions                                 # (T, vocab_size)

            # Fallback: encoder embedding dot-product with vocab matrix
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
        logits = self._compute_lm_logits(token_ids)
        if logits.numel() == 0:
            return None
        targets = token_ids[1: logits.shape[0] + 1].to(device=logits.device)
        grad_cfg = self._genome.gradient
        loss = F.cross_entropy(logits, targets, label_smoothing=grad_cfg.label_smoothing)
        self._optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self._grad_params, grad_cfg.grad_clip_norm)
        self._optimizer.step()
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
            logits = self._compute_lm_logits(token_ids)
            if logits.numel() == 0:
                continue
            targets = token_ids[1: logits.shape[0] + 1].to(device=logits.device)
            seq_loss = F.cross_entropy(logits, targets, label_smoothing=grad_cfg.label_smoothing)
            total_loss = seq_loss if total_loss is None else total_loss + seq_loss
            count += 1
        if count == 0 or total_loss is None:
            return None
        loss = total_loss / count
        self._optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self._grad_params, grad_cfg.grad_clip_norm)
        self._optimizer.step()
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
        logits = self._compute_lm_logits(token_ids)
        if logits.numel() == 0:
            return None
        targets = token_ids[1: logits.shape[0] + 1].to(device=logits.device)
        return F.cross_entropy(
            logits, targets,
            label_smoothing=self._genome.gradient.label_smoothing,
        )

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
        self._save_state()

    def session_start(self) -> None:
        self.reinforcement = ReinforcementSystem(
            self._genome, self._mood_baseline, self._neuromodulator_baseline,
        )
        self.memory = Memory(self._genome)
        self.memory.set_interaction_counter(self._developmental_age)

        self._previous_model_text = None
        self._previous_traces = []
        self._session_active = True

    def session_end(self) -> ConsolidationReport | None:
        if not self._session_active:
            return None

        self._session_active = False

        if self.reinforcement:
            self._mood_baseline = self.reinforcement.session_end_baseline()
            self._neuromodulator_baseline = self.reinforcement.session_end_neuromodulator_baseline()

        if self.memory:
            self._developmental_age = self.memory.interaction_count

        # DMN phase: internal thought generation before consolidation
        try:
            self._run_dmn_phase()
        except Exception:
            pass  # DMN is best-effort; don't block shutdown

        report = self._consolidate()

        # Astrocyte: decay usage counters and apply metabolic penalties
        self.regularizer.decay()
        if self._developmental_age % self._genome.regularizer.update_interval == 0:
            self.regularizer.apply_penalties(self.transformer)

        # Basal ganglia: decay unused habits
        self.habit_system.decay_habits()

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
        assert self.reinforcement is not None
        assert self.memory is not None

        token_ids = self._tokenize(user_text)
        structured = self.encoder.process(token_ids)

        # Online encoder update: light Hebbian pass on this turn's tokens
        self.encoder.online_update(token_ids)

        surprise_score = self.surprise_detector.score(structured, self.transformer.forward_readonly)

        # Prediction error from transformer (unified with surprise via predictive coding)
        pred_error = self.transformer.mean_prediction_error()

        forced_signals = self._forced_signals_from_reinforcement(external_reinforcement)

        ctx_input = self.reinforcement.process(
            structured, user_text, self._previous_model_text,
            surprise_score, pred_error,
            forced_signals=forced_signals,
        )
        reinforcement_strength = sum(s.strength for s in ctx_input.reinforcement_signals)
        neuromod = self.reinforcement.neuromodulators

        # Neuromodulatory gain: norepinephrine drives transformer sharpening
        self.transformer.set_neuromodulatory_gain(
            norepinephrine=neuromod.norepinephrine, surprise=surprise_score,
        )

        # Thalamic routing: decide which regions get what
        routing_weights = self.router.route(
            structured.embeddings, neuromod, surprise_score,
        )

        combined_logits, traces, activations = self.transformer.forward(
            ctx_input, self._personality, routing_weights,
        )

        # Thalamic feedback from transformer output
        region_hiddens = {r: a.hidden for r, a in activations.items()}
        self.router.receive_feedback(region_hiddens)

        # Astrocyte: record activity
        for region, act in activations.items():
            self.regularizer.record_activity(region, act.fired_indices)

        # Corrector: pre-correct logits
        combined_logits = self.corrector.pre_correct_logits(
            combined_logits, structured.embeddings,
        )

        # Basal ganglia: check for habituated shortcut
        context_emb = structured.embeddings.mean(dim=0)
        habit_tokens = self.habit_system.check_habit(context_emb)

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
                self.reinforcement.mood,
                surprise_score,
                max_tokens=_max_tokens,
                encoder=self.encoder,
                transformer=self.transformer,
                anchor_embeddings=structured.embeddings,
                norepinephrine=neuromod.norepinephrine,
            )

        self.memory.record(
            token_ids=token_ids,
            traces=traces,
            surprise_score=surprise_score,
            mood=self.reinforcement.mood,
            reinforcement=reinforcement_strength,
        )

        # Plasticity rates gated by acetylcholine
        plasticity_rates = self.adaptation_system.all_rates(
            self._developmental_age,
            self.reinforcement.mood,
            {r: float(s.mean().item()) for r, s in self._ewc_scalars.items()},
            acetylcholine=neuromod.acetylcholine,
        )

        partial_applied = self.generation.partial_weight_update(
            self.transformer,
            self._previous_traces,
            reinforcement_strength,
            plasticity_rates,
            self._ewc_scalars,
        )

        # Corrector: update forward model with actual reinforcement
        self.corrector.train_step(structured.embeddings, reinforcement_strength)

        # Basal ganglia: record sequence for potential habit formation
        self.habit_system.record_sequence(context_emb, generated_tokens, reinforcement_strength)

        if self.memory.needs_consolidation:
            self._consolidate()

        # Periodic homeostatic synaptic scaling
        homeo = self._genome.homeostatic
        if self._developmental_age > 0 and self._developmental_age % homeo.update_interval == 0:
            self.transformer.homeostatic_scale_all()

        self._previous_model_text = self._detokenize(generated_tokens)
        self._previous_traces = traces
        self._developmental_age += 1
        region_weights = self.transformer.voter.compute_region_weights(self._personality)

        return TurnResult(
            generated_tokens=generated_tokens,
            surprise_score=surprise_score,
            mood=MoodState(
                self.reinforcement.mood.valence,
                self.reinforcement.mood.arousal,
                self.reinforcement.mood.openness,
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

        Returns (logits, traces, surprise, reinforcement).
        Side effects: thalamic routing, neuromodulator update, transformer forward,
        memory record, corrector pre-correction, regularizer tracking.
        """
        assert self._session_active, "Call session_start() before processing turns"
        assert self.reinforcement is not None
        assert self.memory is not None

        token_ids = self._tokenize(user_text)
        structured = self.encoder.process(token_ids)

        # Online encoder update for streaming/chat-server path.
        self.encoder.online_update(token_ids)

        surprise_score = self.surprise_detector.score(structured, self.transformer.forward_readonly)
        pred_error = self.transformer.mean_prediction_error()

        ctx_input = self.reinforcement.process(
            structured, user_text, self._previous_model_text,
            surprise_score, pred_error,
        )
        reinforcement_strength = sum(s.strength for s in ctx_input.reinforcement_signals)
        neuromod = self.reinforcement.neuromodulators

        # Neuromodulatory gain
        self.transformer.set_neuromodulatory_gain(
            norepinephrine=neuromod.norepinephrine, surprise=surprise_score,
        )

        # Thalamic routing
        routing_weights = self.router.route(
            structured.embeddings, neuromod, surprise_score,
        )

        combined_logits, traces, activations = self.transformer.forward(
            ctx_input, self._personality, routing_weights,
        )

        # Thalamic feedback
        region_hiddens = {r: a.hidden for r, a in activations.items()}
        self.router.receive_feedback(region_hiddens)

        # Astrocyte tracking
        for region, act in activations.items():
            self.regularizer.record_activity(region, act.fired_indices)

        # Corrector pre-correction
        combined_logits = self.corrector.pre_correct_logits(
            combined_logits, structured.embeddings,
        )

        self.memory.record(
            token_ids=token_ids,
            traces=traces,
            surprise_score=surprise_score,
            mood=self.reinforcement.mood,
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

        return combined_logits, traces, surprise_score, reinforcement_strength

    def finalize_turn(
        self,
        generated_tokens: list[int],
        traces: list,
        surprise_score: float,
        reinforcement_strength: float,
    ) -> None:
        assert self.reinforcement is not None
        assert self.memory is not None

        neuromod = self.reinforcement.neuromodulators

        plasticity_rates = self.adaptation_system.all_rates(
            self._developmental_age,
            self.reinforcement.mood,
            {r: float(s.mean().item()) for r, s in self._ewc_scalars.items()},
            acetylcholine=neuromod.acetylcholine,
        )

        signal = float(max(min(surprise_score * 0.4 + abs(reinforcement_strength) * 0.6, 1.0), 0.0))

        for trace in traces:
            region = trace.region
            if region not in self.transformer.regions:
                continue
            lr = plasticity_rates.get(region, 0.01) * signal * 0.08
            ewc = self._ewc_scalars.get(region)
            if ewc is not None and int(ewc.numel()) > 0:
                lr *= float(max(min(1.0 - ewc.mean().item(), 1.0), 0.0))
            self.transformer.regions[region].hebbian_update(trace, lr)

        # Corrector: update forward model with actual reinforcement
        if self._last_structured is not None:
            self.corrector.train_step(self._last_structured.embeddings, reinforcement_strength)

        # Basal ganglia: record sequence for habit formation
        if self._last_structured is not None:
            context_emb = self._last_structured.embeddings.mean(dim=0)
            self.habit_system.record_sequence(context_emb, generated_tokens, reinforcement_strength)

        if self.memory.needs_consolidation:
            self._consolidate()

        self._previous_model_text = self._detokenize(generated_tokens)
        self._previous_traces = traces
        self._developmental_age += 1

        # Periodic homeostatic synaptic scaling
        homeo = self._genome.homeostatic
        if self._developmental_age > 0 and self._developmental_age % homeo.update_interval == 0:
            self.transformer.homeostatic_scale_all()

    def count_parameters(self) -> int:
        total = 0
        for arr in self.encoder.get_weights().values():
            total += int(arr.numel())
        for region_module in self.transformer.regions.values():
            total += int(region_module.W_in.numel())
            total += int(region_module.W_hidden.numel())
            total += int(region_module.W_out.numel())
            total += int(region_module.W_recurrent.numel())
        # Router
        total += int(self.router.W_route.numel())
        total += int(self.router.W_feedback.numel())
        # Corrector
        total += int(self.corrector.W_in.numel())
        total += int(self.corrector.W_out.numel())
        return total

    # ------------------------------------------------------------------
    # Default Mode Network
    # ------------------------------------------------------------------

    def _run_dmn_phase(self) -> None:
        """DMN: internal thought generation between sessions.

        The model generates internal "thought" sequences not directed at
        any external input.  These draw on recent memory buffer
        contents and explore associative connections.

        The Hebbian traces get added to the consolidation buffer.
        """
        dmn = self._genome.dmn
        if self.memory is None or not self.memory.size:
            return

        recent = self.memory.peek()[:5]  # peek at top-5 recent experiences
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
                self._genome.topology.transformer_hidden,
                dtype=DTYPE, device=device,
            )
            n = min(trace.post_indices.numel(), trace.activation_strengths.numel())
            valid = trace.post_indices[:n]
            valid = valid[valid < hidden.shape[0]]
            hidden[valid] = trace.activation_strengths[:valid.numel()]

            # Project hidden back to embed space for transformer input
            # Use first active region's W_in transpose as projection
            region = self._genome.topology.active_regions[0]
            if region in self.transformer.regions:
                W_in = self.transformer.regions[region].W_in
                # pseudo-inverse projection
                thought_emb = hidden[:W_in.shape[1]] @ W_in.T  # (embed_dim,)
                thought_emb = thought_emb.unsqueeze(0)  # (1, embed_dim)

                # Run transformer read-only to generate DMN traces
                activations = self.transformer.forward_readonly(thought_emb)
                for r, act in activations.items():
                    dmn_trace = self.transformer.regions[r].get_hebbian_trace()
                    if dmn_trace is not None:
                        # Apply DMN traces at reduced strength
                        self.transformer.regions[r].hebbian_update(
                            dmn_trace, dmn.association_strength,
                        )

    def _consolidate(self) -> ConsolidationReport:
        records = []
        if self.memory:
            records = self.memory.drain()

        plasticity_rates = self.adaptation_system.all_rates(
            self._developmental_age,
            self._mood_baseline,
            {r: float(s.mean().item()) for r, s in self._ewc_scalars.items()},
        )

        return self.replay_engine.run(
            records=records,
            transformer=self.transformer,
            ewc_scalars=self._ewc_scalars,
            personality=self._personality,
            plasticity_rates=plasticity_rates,
            developmental_age=self._developmental_age,
        )

    def _save_state(self) -> None:
        state = ModelState(
            encoder_weights=self.encoder.get_weights(),
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
        self.transformer.set_weights(state.transformer_weights)
        self.transformer.set_inter_region_weights(state.inter_region_weights)
        self._ewc_scalars = state.ewc_protection
        self._personality = state.personality
        self._mood_baseline = state.mood_baseline
        self._developmental_age = state.developmental_age
        self._consolidation_meta = state.consolidation_meta or {}

        for key, strength in state.inter_region_highway.items():
            if key in self.transformer.wiring.highway_strengths:
                self.transformer.wiring.highway_strengths[key] = strength

        # Restore new component weights (backward compat: missing = keep random init)
        if state.neuromodulator_baseline is not None:
            self._neuromodulator_baseline = state.neuromodulator_baseline
        if state.router_weights:
            self.router.set_weights(state.router_weights)
        if state.corrector_weights:
            self.corrector.set_weights(state.corrector_weights)
        if state.regularizer_usage:
            self.regularizer.set_usage(state.regularizer_usage)
        if state.transformer_predictions:
            self.transformer.set_predictions(state.transformer_predictions)
        if state.habit_store:
            self.habit_system.set_state(state.habit_store)

    def _tokenize(self, text: str) -> torch.Tensor:
        if self._tokenizer is not None:
            ids = self._tokenizer.encode(
                text,
                truncation=True,
                max_length=self._genome.topology.memory_capacity,
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
        return self.adaptation_system.stage(self._developmental_age).value

    def status(self) -> dict:
        mood = self.reinforcement.mood if self.reinforcement else self._mood_baseline
        neuromod = self.reinforcement.neuromodulators if self.reinforcement else self._neuromodulator_baseline
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
            "plasticity": self.adaptation_system.describe(self._developmental_age, mood),
            "memory_utilization": (
                round(self.memory.utilization, 4)
                if self.memory else 0.0
            ),
            "session_active": self._session_active,
            "encoder_frozen": self.encoder.is_frozen,
            "consolidation_meta": self._consolidation_meta,
            "region_weights": (
                {
                    r.value: round(w, 4)
                    for r, w in self.transformer.voter.compute_region_weights(self._personality).items()
                }
            ),
            "inter_region_highways": {
                f"{r1.value}-{r2.value}": round(s, 6)
                for (r1, r2), s in self.transformer.wiring.highway_strengths.items()
            },
        }
