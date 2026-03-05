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


class CortexRegion(enum.Enum):
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


# ---------------------------------------------------------------------------
# Core data containers
# ---------------------------------------------------------------------------

@dataclass
class MoodState:
    """Rolling emotional state vector."""
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
    """Record of which cortex nodes fired for an event."""
    region: CortexRegion
    pre_indices: torch.Tensor   # indices of pre-synaptic units that fired
    post_indices: torch.Tensor  # indices of post-synaptic units that fired
    activation_strengths: torch.Tensor  # how strongly each pair fired


@dataclass
class EpisodicRecord:
    """One experience-buffer entry in the hippocampus."""
    token_ids: torch.Tensor
    hebbian_traces: list[HebbianTrace]
    novelty_score: float
    mood_at_event: MoodState
    reinforcement: float  # net scalar
    interaction_number: int
    repetition_count: int = 0
    consolidation_priority: float = 0.0

    def compute_priority(self) -> None:
        self.consolidation_priority = (
            self.novelty_score * 0.3
            + abs(self.reinforcement) * 0.5
            + self.repetition_count * 0.2
        )


@dataclass
class RegionActivation:
    """Activation output from one cortex region on a forward pass."""
    region: CortexRegion
    logits: torch.Tensor          # (vocab_size,)
    hidden: torch.Tensor          # (hidden_dim,)
    fired_indices: torch.Tensor   # which units crossed threshold


@dataclass
class BrainState:
    """Everything that gets persisted between sessions (Tier 8 payload)."""
    brainstem_weights: dict[str, torch.Tensor] = field(default_factory=dict)
    cortex_weights: dict[CortexRegion, torch.Tensor] = field(default_factory=dict)
    inter_region_weights: dict[tuple[CortexRegion, CortexRegion], torch.Tensor] = field(default_factory=dict)
    ewc_protection: dict[CortexRegion, torch.Tensor] = field(default_factory=dict)
    personality: PersonalityVector = field(default_factory=PersonalityVector)
    mood_baseline: MoodState = field(default_factory=MoodState)
    developmental_age: int = 0
    plasticity_rates: dict[CortexRegion, float] = field(default_factory=dict)
    consolidation_meta: dict = field(default_factory=dict)
    inter_region_highway: dict[tuple[CortexRegion, CortexRegion], float] = field(default_factory=dict)
