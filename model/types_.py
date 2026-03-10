"""Shared data types for the neural architecture."""

from __future__ import annotations

import enum
from dataclasses import dataclass, field
from typing import Optional

import torch


# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------

class DevelopmentalStage(enum.Enum):
    INFANCY = "infancy"
    CHILDHOOD = "childhood"
    ADOLESCENCE = "adolescence"
    ADULTHOOD = "adulthood"


class TransformerRegion(enum.Enum):
    TEMPORAL = "temporal"
    FRONTAL = "frontal"
    PARIETAL = "parietal"
    OCCIPITAL = "occipital"


class ReinforcementType(enum.Enum):
    # Explicit
    DIRECT_CORRECTION = "direct_correction"
    DIRECT_AFFIRMATION = "direct_affirmation"
    REPEATED_CORRECTION = "repeated_correction"
    # Implicit
    RESPONSE_LENGTH = "response_length"
    FOLLOW_UP_QUESTION = "follow_up_question"
    CONVERSATION_ABANDONMENT = "conversation_abandonment"
    TONE_MIRRORING = "tone_mirroring"
    TONE_REJECTION = "tone_rejection"


class FailureMode(enum.Enum):
    """How an instance died - used to direct genome mutation."""
    REPETITIVE_OUTPUT = "repetitive_output"          # stuck in generation loops
    CATASTROPHIC_FORGETTING = "catastrophic_forgetting"  # performed well then collapsed
    NEVER_LEARNED = "never_learned"                  # loss never converged
    UNKNOWN = "unknown"


# ---------------------------------------------------------------------------
# Core data containers
# ---------------------------------------------------------------------------

@dataclass
class MoodState:
    """Rolling emotional state vector (legacy view — derived from neuromodulators)."""
    valence: float = 0.0    # -1.0 (negative) → +1.0 (positive)
    arousal: float = 0.0    #  0.0 (calm)     → 1.0  (activated)
    openness: float = 0.5   #  0.0 (guarded)  → 1.0  (receptive)

    def as_array(self) -> torch.Tensor:
        return torch.tensor([self.valence, self.arousal, self.openness], dtype=torch.float64)

    def clamp(self) -> None:
        self.valence = float(max(min(self.valence, 1.0), -1.0))
        self.arousal = float(max(min(self.arousal, 1.0), 0.0))
        self.openness = float(max(min(self.openness, 1.0), 0.0))


@dataclass
class NeuromodulatorState:
    """Four distinct neurochemical systems replacing the single mood proxy.

    Dopamine      — prediction error signal (better/worse than expected)
    Serotonin     — patience / time horizon
    Acetylcholine — learning signal (directly gates Hebbian rates)
    Norepinephrine — uncertainty / exploration drive
    """
    dopamine: float = 0.0         # -1..1  prediction error
    serotonin: float = 0.5        #  0..1  patience
    acetylcholine: float = 0.3    #  0..1  learning gate
    norepinephrine: float = 0.3   #  0..1  exploration

    def clamp(self) -> None:
        self.dopamine = float(max(min(self.dopamine, 1.0), -1.0))
        self.serotonin = float(max(min(self.serotonin, 1.0), 0.0))
        self.acetylcholine = float(max(min(self.acetylcholine, 1.0), 0.0))
        self.norepinephrine = float(max(min(self.norepinephrine, 1.0), 0.0))

    def to_mood(self) -> "MoodState":
        """Derive a legacy MoodState view from neuromodulators."""
        return MoodState(
            valence=self.dopamine,
            arousal=self.norepinephrine,
            openness=self.acetylcholine,
        )

    # ------------------------------------------------------------------
    # specialized utilities for asymmetry features
    # ------------------------------------------------------------------

    def apply_catastrophe(self) -> None:
        """Force neuromodulators to catastrophe values as described in docs."""
        self.norepinephrine = 1.0
        self.dopamine = -1.0
        self.acetylcholine = 1.0
        self.serotonin = 0.0
        self.clamp()

    def decay_toward_baseline(self, baseline: "NeuromodulatorState", rate: float) -> None:
        """Move each signal a fraction `rate` toward its baseline value."""
        self.dopamine += (baseline.dopamine - self.dopamine) * (1 - rate)
        self.serotonin += (baseline.serotonin - self.serotonin) * (1 - rate)
        self.acetylcholine += (baseline.acetylcholine - self.acetylcholine) * (1 - rate)
        self.norepinephrine += (baseline.norepinephrine - self.norepinephrine) * (1 - rate)
        self.clamp()



@dataclass
class PersonalityVector:
    """Six-trait personality profile.  Each trait ∈ [0, 1]."""
    curiosity: float = 0.5
    warmth: float = 0.5
    assertiveness: float = 0.5
    creativity: float = 0.5
    caution: float = 0.5
    humor: float = 0.5

    def as_array(self) -> torch.Tensor:
        return torch.tensor([
            self.curiosity, self.warmth, self.assertiveness,
            self.creativity, self.caution, self.humor,
        ], dtype=torch.float64)

    def from_array(self, arr: torch.Tensor) -> None:
        arr = arr.clamp(0.0, 1.0)
        (self.curiosity, self.warmth, self.assertiveness,
         self.creativity, self.caution, self.humor) = [float(v) for v in arr.tolist()]

    TRAIT_NAMES = ("curiosity", "warmth", "assertiveness",
                   "creativity", "caution", "humor")


@dataclass
class ReinforcementSignal:
    """A detected reinforcement signal from user input."""
    signal_type: ReinforcementType
    strength: float  # positive = affirmation, negative = correction. magnitude = intensity


@dataclass
class StructuredRepresentation:
    """Output of the brainstem — structured token representation."""
    token_ids: torch.Tensor          # (seq_len,) int
    embeddings: torch.Tensor         # (seq_len, embed_dim)
    positional_encoding: torch.Tensor  # (seq_len, embed_dim)
    attention_weights: torch.Tensor  # (seq_len, seq_len)


@dataclass
class ContextualizedInput:
    """Limbic-gate output — brainstem representation + emotional context."""
    structured: StructuredRepresentation
    mood: MoodState
    reinforcement_signals: list[ReinforcementSignal]
    salience: float  # 0–1, how emotionally significant


@dataclass
class HebbianTrace:
    """Record of which transformer nodes fired for an event."""
    region: TransformerRegion
    pre_indices: torch.Tensor   # indices of pre-synaptic units that fired
    post_indices: torch.Tensor  # indices of post-synaptic units that fired
    activation_strengths: torch.Tensor  # how strongly each pair fired


@dataclass
class EpisodicRecord:
    """One experience-buffer entry in the memory."""
    token_ids: torch.Tensor
    hebbian_traces: list[HebbianTrace]
    surprise_score: float
    mood_at_event: MoodState
    reinforcement: float  # net scalar
    interaction_number: int
    repetition_count: int = 0
    consolidation_priority: float = 0.0
    prediction_error: float = 0.0  # from predictive coding
    neuromodulators_at_event: Optional[NeuromodulatorState] = None

    def compute_priority(self) -> None:
        self.consolidation_priority = (
            self.surprise_score * 0.3
            + abs(self.reinforcement) * 0.5
            + self.repetition_count * 0.2
        )


@dataclass
class FailureRecord:
    """Immutable postmortem from a death-replacement cycle.

    Unlike EpisodicRecords, these are never pruned.  They persist across
    instance boundaries and are replayed at maximum priority at the start
    of the successor's first consolidation pass.
    """
    failure_mode: FailureMode
    predecessor_age: int
    cycle_losses: list[float]                    # rolling losses that triggered death
    neuromodulators_at_death: NeuromodulatorState
    failure_token_ids: list[list[int]] = field(default_factory=list)  # inputs causing peak loss
    is_immutable: bool = True                    # never prune


@dataclass
class RegionActivation:
    """Activation output from one transformer region on a forward pass."""
    region: TransformerRegion
    logits: torch.Tensor          # (vocab_size,)
    hidden: torch.Tensor          # (hidden_dim,)
    fired_indices: torch.Tensor   # which units crossed threshold
    prediction_error: float = 0.0  # mean absolute prediction error from this pass


@dataclass
class ModelState:
    """Everything that gets persisted between sessions (Tier 8 payload)."""
    encoder_weights: dict[str, torch.Tensor] = field(default_factory=dict)
    transformer_weights: dict[TransformerRegion, torch.Tensor] = field(default_factory=dict)
    inter_region_weights: dict[tuple[TransformerRegion, TransformerRegion], torch.Tensor] = field(default_factory=dict)
    ewc_protection: dict[TransformerRegion, torch.Tensor] = field(default_factory=dict)
    personality: PersonalityVector = field(default_factory=PersonalityVector)
    mood_baseline: MoodState = field(default_factory=MoodState)
    developmental_age: int = 0
    plasticity_rates: dict[TransformerRegion, float] = field(default_factory=dict)
    consolidation_meta: dict = field(default_factory=dict)
    inter_region_highway: dict[tuple[TransformerRegion, TransformerRegion], float] = field(default_factory=dict)
    # New component weights
    neuromodulator_baseline: Optional[NeuromodulatorState] = None
    router_weights: dict[str, torch.Tensor] = field(default_factory=dict)
    corrector_weights: dict[str, torch.Tensor] = field(default_factory=dict)
    regularizer_usage: dict[TransformerRegion, torch.Tensor] = field(default_factory=dict)
    transformer_predictions: dict[TransformerRegion, torch.Tensor] = field(default_factory=dict)
    habit_store: Optional[dict] = None