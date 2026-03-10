"""Tier 0 — The Genome.

The immutable blueprint.  Every parameter here is set once at creation and
never modified by any interaction.  This is the DNA of the model.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field

from .types_ import TransformerRegion, DevelopmentalStage, FailureMode


# ---------------------------------------------------------------------------
# Layer topology
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class LayerTopology:
    """Defines dimensionality of every module in the AI model."""
    vocab_size: int = 30000
    embed_dim: int = 64
    encoder_hidden: int = 128
    transformer_hidden: int = 128
    memory_capacity: int = 1024
    regions: tuple[TransformerRegion, ...] = (
        TransformerRegion.TEMPORAL,
        TransformerRegion.FRONTAL,
        TransformerRegion.PARIETAL,
        TransformerRegion.OCCIPITAL,
    )
    active_regions: tuple[TransformerRegion, ...] = (
        TransformerRegion.TEMPORAL,
        TransformerRegion.FRONTAL,
        TransformerRegion.PARIETAL,
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
    region_lambdas: dict[TransformerRegion, float] = field(default_factory=lambda: {
        TransformerRegion.TEMPORAL: 0.0008,   # closes fastest
        TransformerRegion.FRONTAL: 0.0005,    # medium
        TransformerRegion.PARIETAL: 0.0003,   # closes slowest
        TransformerRegion.OCCIPITAL: 0.0005,  # same as frontal (dormant anyway)
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
    arousal_boost: float = 0.3     # surprise → arousal gain
    temperature_boost: float = 0.1 # surprise → generation temperature gain


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
# Transformer voting
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class VotingParams:
    """Maps personality traits → region weight biases."""
    # Each entry: (region, trait_name, multiplier)
    # High curiosity   → parietal weight up
    # High caution     → frontal weight up
    # High warmth      → temporal weight up
    region_trait_map: tuple[tuple[TransformerRegion, str, float], ...] = (
        (TransformerRegion.PARIETAL, "curiosity", 0.3),
        (TransformerRegion.PARIETAL, "creativity", 0.2),
        (TransformerRegion.FRONTAL, "caution", 0.3),
        (TransformerRegion.FRONTAL, "assertiveness", 0.15),
        (TransformerRegion.TEMPORAL, "warmth", 0.3),
        (TransformerRegion.TEMPORAL, "humor", 0.15),
    )
    base_weight: float = 1.0  # each region starts at this before trait modulation


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class HomeostaticParams:
    """Homeostatic synaptic scaling (Turrigiano 1998)."""
    target_rate: float = 0.10          # target fraction of max activation
    scaling_factor: float = 0.01       # multiplicative adjustment per cycle
    update_interval: int = 10          # every N turns, run scaling


@dataclass(frozen=True)
class BCMParams:
    """Bienenstock-Cooper-Munro sliding threshold."""
    theta_lr: float = 0.01             # how fast theta tracks average post
    theta_init: float = 0.3            # initial threshold (same as old fixed)


@dataclass(frozen=True)
class STDPParams:
    """Spike-Timing Dependent Plasticity."""
    a_plus: float = 0.005              # LTP amplitude (pre before post)
    a_minus: float = 0.005             # LTD amplitude  (post before pre)
    tau_plus: float = 20.0             # LTP time constant
    tau_minus: float = 20.0            # LTD time constant


@dataclass(frozen=True)
class GenerationParams:
    base_temperature: float = 0.8
    min_temperature: float = 0.3
    max_temperature: float = 1.5
    encoder_pretrain_steps: int = 500

    # Recurrent hidden state
    context_buffer_size: int = 64         # rolling attention window
    context_attention_weight: float = 0.3  # how much context pulls hidden state
    recurrent_mix: float = 0.3             # hidden state carry-forward strength

    # Soft prompt anchoring
    anchor_weight: float = 0.15            # topic anchoring bias per step

    # Entropy-adaptive sampling
    entropy_high_threshold: float = 3.5    # above → narrow sampling
    entropy_low_threshold: float = 1.0     # below → normal sampling

    # In-loop repetition penalty
    repetition_penalty: float = 1.15

    # Top-k / top-p inside model generation
    top_k: int = 50
    top_p: float = 0.95


@dataclass(frozen=True)
class GradientParams:
    """Gradient-based optimisation hyperparameters.

    This complements the Hebbian rules and lets the model learn via
    backpropagation during interaction or batch training.
    """
    lr: float = 1e-5
    # Decoupled weight decay (AdamW-style); 0.01 provides mild regularisation
    weight_decay: float = 0.01
    # Max-norm gradient clipping — prevents exploding gradients on long seqs
    grad_clip_norm: float = 1.0
    # Label smoothing for cross-entropy — prevents overconfident predictions
    label_smoothing: float = 0.1


# ---------------------------------------------------------------------------
# Neuromodulator systems (replaces single mood proxy)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class NeuromodulatorParams:
    """Four neurochemical systems with distinct update rules."""
    # Dopamine — prediction error signal
    dopamine_lr: float = 0.1
    dopamine_decay: float = 0.3
    dopamine_baseline: float = 0.0

    # Serotonin — patience / time horizon
    serotonin_lr: float = 0.02
    serotonin_decay: float = 0.05
    serotonin_baseline: float = 0.5

    # Acetylcholine — learning gate
    ach_surprise_weight: float = 0.6
    ach_reinforcement_weight: float = 0.4
    ach_decay: float = 0.1
    ach_baseline: float = 0.3

    # Norepinephrine — uncertainty / exploration
    ne_uncertainty_weight: float = 0.7
    ne_decay: float = 0.15
    ne_baseline: float = 0.3


# ---------------------------------------------------------------------------
# Thalamic routing
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class RouterParams:
    """Thalamic gating — active routing of inputs to transformer regions."""
    routing_sharpness: float = 3.0
    feedback_strength: float = 0.3
    surprise_parietal_bias: float = 0.4
    linguistic_temporal_bias: float = 0.4
    logical_frontal_bias: float = 0.4


# ---------------------------------------------------------------------------
# Predictive coding
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class PredictiveCodingParams:
    """Each transformer region predicts next input; only error propagates."""
    prediction_lr: float = 0.05
    error_gain: float = 0.3  # reduced from 1.5 — prevents logit explosion with untrained weights


# ---------------------------------------------------------------------------
# Basal ganglia — habit formation
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class HabitSystemParams:
    """Go/no-go action selection and habit store."""
    habit_threshold: float = 0.7
    habit_min_occurrences: int = 5
    max_habits: int = 500
    habit_match_threshold: float = 0.85
    habit_decay: float = 0.999


# ---------------------------------------------------------------------------
# Corrector — forward model
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class CorrectorParams:
    """Fast forward model for error correction and pre-correction."""
    hidden_dim: int = 64
    learning_rate: float = 0.01
    correction_weight: float = 0.2


# ---------------------------------------------------------------------------
# Astrocyte — metabolic penalties
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class AstrocyteParams:
    """Slow-timescale metabolic cost on overused pathways."""
    metabolic_decay: float = 0.999
    penalty_scale: float = 0.01
    target_usage: float = 0.1
    update_interval: int = 20


# ---------------------------------------------------------------------------
# Default Mode Network
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class DMNParams:
    """Internal processing between sessions."""
    thought_steps: int = 32
    num_thoughts: int = 10
    association_strength: float = 0.05


# ---------------------------------------------------------------------------
# Three-stage sleep
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class SleepParams:
    """Three-stage sleep consolidation (Tononi & Cirelli)."""
    # Stage B: Global downscaling (Synaptic Homeostasis Hypothesis)
    global_downscale_factor: float = 0.995
    # Stage C: REM — transformer autonomous processing
    rem_steps: int = 50
    rem_integration_lr: float = 0.005


@dataclass(frozen=True)
class CatastropheParams:
    """Parameters governing the catastrophic cascade response.

    When both reinforcement is severely negative AND prediction error is high
    (confident wrong answer), the four neuromodulator systems diverge in a
    qualitatively different way from ordinary negative reinforcement:
      - NE spikes    → max exploration / sampling width
      - Dopamine crashes → expectation fully reset downward
      - ACh spikes   → learning gate fully open
      - Serotonin crashes → shift to short-horizon behavioural changes
    The cascade then decays over subsequent turns.
    """
    # Trigger condition
    reinforcement_threshold: float = 0.65   # |net_signal| must exceed this
    prediction_error_threshold: float = 0.55  # prediction_error must also exceed this
    # Neuromodulator targets during cascade
    ne_target: float = 1.0           # NE spikes to max
    dopamine_target: float = -1.0    # dopamine crashes to min
    ach_target: float = 1.0          # ACh spikes to max
    serotonin_target: float = 0.0    # serotonin crashes to min
    # Persistence / decay
    decay_rate: float = 0.18         # intensity lost per subsequent turn
    min_intensity: float = 0.05      # deactivate below this threshold


@dataclass(frozen=True)
class DeathParams:
    """Parameters governing the death-and-replacement lifecycle.

    An instance dies when it fails persistently across multiple consecutive
    consolidation cycles.  The successor inherits all weights but receives
    the failure record as an immutable memory, and the genome receives a
    small directional mutation away from the failure mode.
    """
    cycle_loss_threshold: float = 3.5    # loss above this = a failing cycle
    consecutive_cycles: int = 5          # consecutive failing cycles to trigger death
    loss_window: int = 10                # rolling-window size for cycle losses
    mutation_magnitude: float = 0.05     # per-parameter adjustment magnitude
    mutation_max_cumulative: float = 0.40  # max drift from original parameter value


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
    surprise: NoveltyParams = field(default_factory=NoveltyParams)
    mood: MoodParams = field(default_factory=MoodParams)
    personality: PersonalityParams = field(default_factory=PersonalityParams)
    ewc: EWCParams = field(default_factory=EWCParams)
    development: DevelopmentalBoundaries = field(default_factory=DevelopmentalBoundaries)
    voting: VotingParams = field(default_factory=VotingParams)
    generation: GenerationParams = field(default_factory=GenerationParams)
    # parameters for backprop-based optimisation (added for real-time training)
    gradient: GradientParams = field(default_factory=GradientParams)
    homeostatic: HomeostaticParams = field(default_factory=HomeostaticParams)
    bcm: BCMParams = field(default_factory=BCMParams)
    stdp: STDPParams = field(default_factory=STDPParams)
    neuromodulator: NeuromodulatorParams = field(default_factory=NeuromodulatorParams)
    router: RouterParams = field(default_factory=RouterParams)
    predictive_coding: PredictiveCodingParams = field(default_factory=PredictiveCodingParams)
    habit_system: HabitSystemParams = field(default_factory=HabitSystemParams)
    corrector: CorrectorParams = field(default_factory=CorrectorParams)
    regularizer: AstrocyteParams = field(default_factory=AstrocyteParams)
    dmn: DMNParams = field(default_factory=DMNParams)
    sleep: SleepParams = field(default_factory=SleepParams)
    catastrophe: CatastropheParams = field(default_factory=CatastropheParams)
    mortality: DeathParams = field(default_factory=DeathParams)


# ---------------------------------------------------------------------------
# Genome mutation (used by the death-and-replacement cycle)
# ---------------------------------------------------------------------------

def mutate_genome(genome: "Genome", failure_record: "object") -> "Genome":
    """Return a new Genome with small directional adjustments based on how the
    predecessor died.  Structural parameters (dims, regions, stages) are never
    touched.  Only behavioural parameters that can be corrected by small nudges
    are eligible for mutation.

    The changes are intentionally small.  The goal is to prevent the same
    failure mode from recurring — not to redesign the architecture.
    """
    # Import here to avoid circular import at module load time.
    from .types_ import FailureMode as _FM  # noqa: F401 (already imported above)

    mag = genome.mortality.mutation_magnitude
    mode: FailureMode = getattr(failure_record, "failure_mode", FailureMode.UNKNOWN)

    if mode == FailureMode.REPETITIVE_OUTPUT:
        # Model got stuck in loops — raise the repetition penalty.
        new_gen = dataclasses.replace(
            genome.generation,
            repetition_penalty=min(genome.generation.repetition_penalty + mag, 2.0),
        )
        return dataclasses.replace(genome, generation=new_gen)

    if mode == FailureMode.CATASTROPHIC_FORGETTING:
        # Model learned early then collapsed — slow plasticity and strengthen EWC.
        new_plasticity = dataclasses.replace(
            genome.plasticity,
            p_floor=max(genome.plasticity.p_floor - mag * 0.5, 0.01),
        )
        new_ewc = dataclasses.replace(
            genome.ewc,
            protection_growth_rate=min(
                genome.ewc.protection_growth_rate + mag * 0.5, 0.5,
            ),
        )
        return dataclasses.replace(genome, plasticity=new_plasticity, ewc=new_ewc)

    if mode == FailureMode.NEVER_LEARNED:
        # Model never converged — raise Hebbian LR and minimum replay passes.
        new_hebbian = dataclasses.replace(
            genome.hebbian,
            learning_rate=min(genome.hebbian.learning_rate * (1.0 + mag), 0.1),
        )
        new_consolidation = dataclasses.replace(
            genome.consolidation,
            replay_passes_min=min(
                genome.consolidation.replay_passes_min + 1,
                genome.consolidation.replay_passes_max,
            ),
        )
        return dataclasses.replace(
            genome, hebbian=new_hebbian, consolidation=new_consolidation,
        )

    # UNKNOWN — no mutation; keep the genome as-is.
    return genome
