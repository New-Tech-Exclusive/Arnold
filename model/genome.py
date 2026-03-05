"""Tier 0 — The Genome.

The immutable blueprint.  Every parameter here is set once at creation and
never modified by any interaction.  This is the DNA of the model.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .types_ import CortexRegion, DevelopmentalStage


# ---------------------------------------------------------------------------
# Layer topology
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class LayerTopology:
    """Defines dimensionality of every region in the brain."""
    vocab_size: int = 30000
    embed_dim: int = 64
    brainstem_hidden: int = 128
    cortex_hidden: int = 128
    hippocampus_capacity: int = 1024
    regions: tuple[CortexRegion, ...] = (
        CortexRegion.TEMPORAL,
        CortexRegion.FRONTAL,
        CortexRegion.PARIETAL,
        CortexRegion.OCCIPITAL,
    )
    active_regions: tuple[CortexRegion, ...] = (
        CortexRegion.TEMPORAL,
        CortexRegion.FRONTAL,
        CortexRegion.PARIETAL,
    )


# ---------------------------------------------------------------------------
# Hebbian learning rule
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class HebbianParams:
    """Mathematical rules governing how connections strengthen."""
    learning_rate: float = 0.01
    decay_rate: float = 0.001        # passive weight decay per consolidation
    max_weight: float = 5.0
    min_weight: float = -5.0
    coactivation_threshold: float = 0.3  # minimum activation to count as "fired"
    strengthening_fn_exponent: float = 1.0  # Δw = η · pre^exp · post^exp


# ---------------------------------------------------------------------------
# Plasticity decay
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class PlasticityParams:
    """Controls how learning rate narrows with age."""
    p_max: float = 1.0       # plasticity at birth
    p_floor: float = 0.05    # minimum plasticity in adulthood
    lambda_decay: float = 0.0005  # global decay rate

    # Per-region λ overrides (higher = closes faster)
    region_lambdas: dict[CortexRegion, float] = field(default_factory=lambda: {
        CortexRegion.TEMPORAL: 0.0008,   # closes fastest
        CortexRegion.FRONTAL: 0.0005,    # medium
        CortexRegion.PARIETAL: 0.0003,   # closes slowest
        CortexRegion.OCCIPITAL: 0.0005,  # same as frontal (dormant anyway)
    })

    # Mood modulation bounds
    mood_plasticity_boost: float = 0.15   # max raise from high openness
    mood_plasticity_dampen: float = 0.10  # max lower from low openness


# ---------------------------------------------------------------------------
# Consolidation algorithm
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ConsolidationParams:
    """Controls offline learning between sessions."""
    replay_passes_max: int = 5          # max replays for highest-priority records
    replay_passes_min: int = 1
    pruning_threshold: float = 0.01     # weights below this get pruned
    pruning_age_threshold: int = 500    # interactions since last fire to prune
    buffer_trigger_pct: float = 0.80    # consolidation triggers at 80% buffer
    time_trigger_hours: float = 6.0     # fallback consolidation interval


# ---------------------------------------------------------------------------
# Reinforcement thresholds
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ReinforcementParams:
    """What counts as strong enough to matter."""
    explicit_strength: float = 1.0      # base strength of explicit signals
    implicit_strength: float = 0.4      # base strength of implicit signals
    immediate_update_threshold: float = 0.7  # in-session partial update fires above this
    immediate_update_magnitude: float = 0.005  # bounded in-session weight change


# ---------------------------------------------------------------------------
# Novelty scoring
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class NoveltyParams:
    """How surprise is calculated."""
    temperature: float = 1.0       # softmax temperature for prediction comparison
    arousal_boost: float = 0.3     # novelty → arousal gain
    temperature_boost: float = 0.1 # novelty → generation temperature gain


# ---------------------------------------------------------------------------
# Mood state rules
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class MoodParams:
    """How emotional context shifts."""
    valence_step: float = 0.05       # per-signal nudge magnitude
    arousal_decay: float = 0.1       # arousal decays this much per turn toward baseline
    openness_rise_rate: float = 0.02 # per-turn rise under sustained positive valence
    openness_fall_rate: float = 0.03 # per-turn fall under sustained negative valence
    session_carry_decay: float = 0.3 # fraction of session-end mood that carries forward
    arousal_baseline: float = 0.2


# ---------------------------------------------------------------------------
# Personality vector schema
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class PersonalityParams:
    """Personality trait schema and drift bounds."""
    trait_names: tuple[str, ...] = (
        "curiosity", "warmth", "assertiveness",
        "creativity", "caution", "humor",
    )
    trait_min: float = 0.0
    trait_max: float = 1.0
    default_value: float = 0.5
    max_drift_per_cycle: float = 0.02  # bounded personality shift per consolidation


# ---------------------------------------------------------------------------
# EWC protection
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class EWCParams:
    """Elastic Weight Consolidation parameters."""
    fisher_sample_size: int = 200     # samples used to estimate Fisher information
    protection_exponent: float = 2.0  # how steeply importance → protection
    max_protection: float = 0.95      # never fully lock a weight
    protection_growth_rate: float = 0.1  # how fast protection accrues per cycle


# ---------------------------------------------------------------------------
# Developmental stage boundaries
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class DevelopmentalBoundaries:
    """When each stage ends, measured in total interactions."""
    infancy_end: int = 500
    childhood_end: int = 5_000
    adolescence_end: int = 50_000
    # Everything after adolescence_end is adulthood

    def stage_for_age(self, age: int) -> DevelopmentalStage:
        if age < self.infancy_end:
            return DevelopmentalStage.INFANCY
        elif age < self.childhood_end:
            return DevelopmentalStage.CHILDHOOD
        elif age < self.adolescence_end:
            return DevelopmentalStage.ADOLESCENCE
        return DevelopmentalStage.ADULTHOOD


# ---------------------------------------------------------------------------
# Cortex voting
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class VotingParams:
    """Maps personality traits → region weight biases."""
    # Each entry: (region, trait_name, multiplier)
    # High curiosity   → parietal weight up
    # High caution     → frontal weight up
    # High warmth      → temporal weight up
    region_trait_map: tuple[tuple[CortexRegion, str, float], ...] = (
        (CortexRegion.PARIETAL, "curiosity", 0.3),
        (CortexRegion.PARIETAL, "creativity", 0.2),
        (CortexRegion.FRONTAL, "caution", 0.3),
        (CortexRegion.FRONTAL, "assertiveness", 0.15),
        (CortexRegion.TEMPORAL, "warmth", 0.3),
        (CortexRegion.TEMPORAL, "humor", 0.15),
    )
    base_weight: float = 1.0  # each region starts at this before trait modulation


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class GenerationParams:
    base_temperature: float = 0.8
    min_temperature: float = 0.3
    max_temperature: float = 1.5
    brainstem_pretrain_steps: int = 500


# ---------------------------------------------------------------------------
# Top-level genome
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Genome:
    """The complete immutable blueprint of the model."""
    topology: LayerTopology = field(default_factory=LayerTopology)
    hebbian: HebbianParams = field(default_factory=HebbianParams)
    plasticity: PlasticityParams = field(default_factory=PlasticityParams)
    consolidation: ConsolidationParams = field(default_factory=ConsolidationParams)
    reinforcement: ReinforcementParams = field(default_factory=ReinforcementParams)
    novelty: NoveltyParams = field(default_factory=NoveltyParams)
    mood: MoodParams = field(default_factory=MoodParams)
    personality: PersonalityParams = field(default_factory=PersonalityParams)
    ewc: EWCParams = field(default_factory=EWCParams)
    development: DevelopmentalBoundaries = field(default_factory=DevelopmentalBoundaries)
    voting: VotingParams = field(default_factory=VotingParams)
    generation: GenerationParams = field(default_factory=GenerationParams)
