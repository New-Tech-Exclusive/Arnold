"""Tier 2 — The Limbic System (Reinforcement and Mood Layer).

The emotional core.  Sits between encoder and transformer.  Everything that comes
up from the encoder passes through here before the transformer sees it.

Components:
  2A — Reinforcement Detector
  2B — Mood State Vector (legacy, derived from neuromodulators)
  2C — Limbic Gate
  2D — Neuromodulator Manager (dopamine, serotonin, acetylcholine, norepinephrine)
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

import numpy as np

from .genome import Genome
from .types_ import (
    ContextualizedInput,
    MoodState,
    NeuromodulatorState,
    ReinforcementSignal,
    ReinforcementType,
    StructuredRepresentation,
)


# =====================================================================
# 2A — Reinforcement Detector
# =====================================================================

class ReinforcementDetector:
    """Reads every incoming user message for reinforcement signals.

    Runs *before* the transformer processes anything.
    """

    # Simple keyword-based heuristics for detecting signal type.
    _CORRECTION_PATTERNS = [
        re.compile(r"\b(no|wrong|incorrect|actually|not right|mistake)\b", re.I),
    ]
    _AFFIRMATION_PATTERNS = [
        re.compile(r"\b(yes|exactly|correct|right|perfect|great|thanks|good)\b", re.I),
    ]
    _FOLLOW_UP_PATTERNS = [
        re.compile(r"\?\s*$"),  # ends with question mark
        re.compile(r"\b(what about|how about|can you|could you|tell me more|and)\b", re.I),
    ]

    def __init__(self, genome: Genome) -> None:
        self._params = genome.reinforcement
        self._correction_history: list[str] = []

    def detect(
        self,
        user_text: str,
        previous_model_text: str | None,
        previous_signals: list[ReinforcementSignal] | None = None,
    ) -> list[ReinforcementSignal]:
        """Analyse user text and return detected reinforcement signals."""
        signals: list[ReinforcementSignal] = []

        # --- Explicit signals ---
        if self._matches(user_text, self._CORRECTION_PATTERNS):
            strength = -self._params.explicit_strength
            sig_type = ReinforcementType.DIRECT_CORRECTION

            # Check for repeated correction on same topic
            topic_key = user_text[:60].lower()
            if topic_key in self._correction_history:
                sig_type = ReinforcementType.REPEATED_CORRECTION
                strength *= 1.5
            self._correction_history.append(topic_key)
            # Trim history
            if len(self._correction_history) > 100:
                self._correction_history = self._correction_history[-100:]

            signals.append(ReinforcementSignal(sig_type, strength))

        if self._matches(user_text, self._AFFIRMATION_PATTERNS):
            signals.append(ReinforcementSignal(
                ReinforcementType.DIRECT_AFFIRMATION,
                self._params.explicit_strength,
            ))

        # --- Implicit signals ---

        # Response length heuristic
        if previous_model_text is not None:
            model_len = len(previous_model_text)
            user_len = len(user_text)
            if user_len > model_len * 0.5 and user_len > 100:
                signals.append(ReinforcementSignal(
                    ReinforcementType.RESPONSE_LENGTH,
                    self._params.implicit_strength * 0.6,
                ))

        # Follow-up question
        if self._matches(user_text, self._FOLLOW_UP_PATTERNS):
            signals.append(ReinforcementSignal(
                ReinforcementType.FOLLOW_UP_QUESTION,
                self._params.implicit_strength,
            ))

        # Tone mirroring / rejection (simple word overlap heuristic)
        if previous_model_text is not None:
            overlap = self._word_overlap(previous_model_text, user_text)
            if overlap > 0.3:
                signals.append(ReinforcementSignal(
                    ReinforcementType.TONE_MIRRORING,
                    self._params.implicit_strength * overlap,
                ))
            elif overlap < 0.05 and len(user_text.split()) > 10:
                signals.append(ReinforcementSignal(
                    ReinforcementType.TONE_REJECTION,
                    -self._params.implicit_strength * 0.5,
                ))

        # Conversation abandonment is detected at session level (see Model)
        return signals

    def signal_abandonment(self) -> ReinforcementSignal:
        return ReinforcementSignal(
            ReinforcementType.CONVERSATION_ABANDONMENT,
            -self._params.implicit_strength * 0.8,
        )

    # ------------------------------------------------------------------

    @staticmethod
    def _matches(text: str, patterns: list[re.Pattern]) -> bool:
        return any(p.search(text) for p in patterns)

    @staticmethod
    def _word_overlap(a: str, b: str) -> float:
        words_a = set(a.lower().split())
        words_b = set(b.lower().split())
        if not words_a or not words_b:
            return 0.0
        intersection = words_a & words_b
        return len(intersection) / min(len(words_a), len(words_b))


# =====================================================================
# 2B — Mood State Manager
# =====================================================================

class MoodManager:
    """Maintains the rolling mood state across the session."""

    def __init__(self, genome: Genome, baseline: MoodState | None = None) -> None:
        self._params = genome.mood
        self.state = MoodState(
            valence=baseline.valence if baseline else 0.0,
            arousal=baseline.arousal if baseline else self._params.arousal_baseline,
            openness=baseline.openness if baseline else 0.5,
        )
        self._turn_valences: list[float] = []

    def update(
        self,
        signals: list[ReinforcementSignal],
        surprise_score: float = 0.0,
    ) -> MoodState:
        """Update mood from this turn's reinforcement signals and surprise."""
        # Valence: nudge by net signal
        net_signal = sum(s.strength for s in signals)
        self.state.valence += self._params.valence_step * np.tanh(net_signal)
        self._turn_valences.append(self.state.valence)

        # Arousal: raised by surprise, decays toward baseline
        self.state.arousal += self._params.arousal_decay * (
            surprise_score - (self.state.arousal - self._params.arousal_baseline)
        )

        # Openness: rises with sustained positive valence, falls with sustained negative
        recent = self._turn_valences[-5:]
        avg_valence = float(np.mean(recent))
        if avg_valence > 0.1:
            self.state.openness += self._params.openness_rise_rate
        elif avg_valence < -0.1:
            self.state.openness -= self._params.openness_fall_rate

        self.state.clamp()
        return self.state

    def session_end_baseline(self) -> MoodState:
        """Produce a decayed mood baseline for the next session."""
        decay = self._params.session_carry_decay
        return MoodState(
            valence=self.state.valence * decay,
            arousal=self._params.arousal_baseline,
            openness=self.state.openness * decay + 0.5 * (1 - decay),
        )


# =====================================================================
# 2C — Limbic Gate
# =====================================================================

class LimbicGate:
    """Interface between encoder output and transformer input.

          Wraps the structured token representation with emotional context
          so the transformer never sees raw encoder output.
    """

    def __init__(self, genome: Genome) -> None:
        self._genome = genome

    def gate(
        self,
        structured: StructuredRepresentation,
        mood: MoodState,
        signals: list[ReinforcementSignal],
    ) -> ContextualizedInput:
        """Wrap encoder output with emotional context."""
        # Salience: emotional significance based on signal magnitude and mood arousal
        signal_mag = sum(abs(s.strength) for s in signals) if signals else 0.0
        salience = float(np.clip(
            signal_mag * 0.5 + mood.arousal * 0.3 + abs(mood.valence) * 0.2,
            0.0, 1.0,
        ))

        return ContextualizedInput(
            structured=structured,
            mood=MoodState(mood.valence, mood.arousal, mood.openness),
            reinforcement_signals=list(signals),
            salience=salience,
        )


# =====================================================================
# 2D — Neuromodulator Manager
# =====================================================================

class NeuromodulatorManager:
    """Four distinct neurochemical systems replacing the single mood proxy.

    Each neuromodulator has its own update rule and specific effects:

    Dopamine — prediction error signal.  Fires when outcome is BETTER than
    expected, suppressed when WORSE.  Does NOT fire for expected rewards.

    Serotonin — patience / time horizon.  High → conservative, willing to wait.
    Low → impulsive, short time horizon.

    Acetylcholine — learning gate.  High → high plasticity, pay attention.
    Directly gates Hebbian update rates.

    Norepinephrine — uncertainty / exploration.  High → explore, widen sampling.
    Directly controls generation temperature.
    """

    def __init__(
        self,
        genome: Genome,
        baseline: NeuromodulatorState | None = None,
    ) -> None:
        self._params = genome.neuromodulator
        self.state = NeuromodulatorState(
            dopamine=baseline.dopamine if baseline else self._params.dopamine_baseline,
            serotonin=baseline.serotonin if baseline else self._params.serotonin_baseline,
            acetylcholine=baseline.acetylcholine if baseline else self._params.ach_baseline,
            norepinephrine=baseline.norepinephrine if baseline else self._params.ne_baseline,
        )
        self._expected_reinforcement: float = 0.0  # running average
        self._expected_alpha: float = 0.1  # EMA smoothing

    def update(
        self,
        signals: list[ReinforcementSignal],
        surprise_score: float = 0.0,
        prediction_error_magnitude: float = 0.0,
    ) -> NeuromodulatorState:
        """Update all four neuromodulator levels."""
        p = self._params
        # Sanitize inputs — NaN from surprise/transformer must not corrupt neuromodulators
        if surprise_score != surprise_score:  # NaN check
            surprise_score = 0.5
        if prediction_error_magnitude != prediction_error_magnitude:
            prediction_error_magnitude = 0.5
        surprise_score = float(max(min(surprise_score, 1.0), 0.0))
        prediction_error_magnitude = float(max(min(prediction_error_magnitude, 2.0), 0.0))
        net_signal = sum(s.strength for s in signals)

        # --- Dopamine: prediction error (RPE) ---
        # Dopamine fires for BETTER than expected, suppressed for worse
        actual = net_signal
        rpe = actual - self._expected_reinforcement  # reward prediction error
        self._expected_reinforcement += self._expected_alpha * (actual - self._expected_reinforcement)
        self.state.dopamine += p.dopamine_lr * rpe
        self.state.dopamine += p.dopamine_decay * (p.dopamine_baseline - self.state.dopamine)

        # --- Serotonin: patience (slow-adjusting) ---
        # Positive reinforcement history → more patient
        self.state.serotonin += p.serotonin_lr * net_signal * 0.5
        self.state.serotonin += p.serotonin_decay * (p.serotonin_baseline - self.state.serotonin)

        # --- Acetylcholine: learning gate ---
        # Driven by surprise and reinforcement magnitude
        ach_signal = (
            p.ach_surprise_weight * surprise_score
            + p.ach_reinforcement_weight * abs(net_signal)
        )
        self.state.acetylcholine += p.ach_decay * (ach_signal - self.state.acetylcholine)

        # --- Norepinephrine: uncertainty / exploration ---
        # High prediction error → high uncertainty → explore
        ne_signal = (
            p.ne_uncertainty_weight * prediction_error_magnitude
            + (1.0 - p.ne_uncertainty_weight) * surprise_score
        )
        # Cap so random-weight noise (uniformly high pred error) cannot
        # saturate exploration at 1.0 from the very first turn.
        ne_signal = min(ne_signal, 0.65)
        self.state.norepinephrine += p.ne_decay * (ne_signal - self.state.norepinephrine)

        self.state.clamp()
        return self.state

    def session_end_baseline(self) -> NeuromodulatorState:
        """Decayed neuromodulator baseline for next session."""
        p = self._params
        return NeuromodulatorState(
            dopamine=p.dopamine_baseline,
            serotonin=0.7 * self.state.serotonin + 0.3 * p.serotonin_baseline,
            acetylcholine=p.ach_baseline,
            norepinephrine=p.ne_baseline,
        )


# =====================================================================
# Assembled Limbic System
# =====================================================================

class ReinforcementSystem:
    """Full Tier 2: detector + mood + neuromodulators + gate."""

    def __init__(
        self,
        genome: Genome,
        mood_baseline: MoodState | None = None,
        neuromodulator_baseline: NeuromodulatorState | None = None,
    ) -> None:
        self.detector = ReinforcementDetector(genome)
        self.mood_manager = MoodManager(genome, mood_baseline)
        self.neuromodulator_manager = NeuromodulatorManager(genome, neuromodulator_baseline)
        self.gate = LimbicGate(genome)
        self._genome = genome

    def process(
        self,
        structured: StructuredRepresentation,
        user_text: str,
        previous_model_text: str | None,
        surprise_score: float = 0.0,
        prediction_error_magnitude: float = 0.0,
        forced_signals: list[ReinforcementSignal] | None = None,
    ) -> ContextualizedInput:
        """Full reinforcement pipeline: detect → neuromodulators → mood → gate."""
        signals = list(forced_signals) if forced_signals is not None else self.detector.detect(
            user_text, previous_model_text,
        )

        # Update neuromodulators first (they are the primary system)
        neuromod = self.neuromodulator_manager.update(
            signals, surprise_score, prediction_error_magnitude,
        )

        # Mood is now derived from neuromodulators (legacy compatibility)
        mood = self.mood_manager.update(signals, surprise_score)

        return self.gate.gate(structured, mood, signals)

    @property
    def mood(self) -> MoodState:
        return self.mood_manager.state

    @property
    def neuromodulators(self) -> NeuromodulatorState:
        return self.neuromodulator_manager.state

    def session_end_baseline(self) -> MoodState:
        return self.mood_manager.session_end_baseline()

    def session_end_neuromodulator_baseline(self) -> NeuromodulatorState:
        return self.neuromodulator_manager.session_end_baseline()
