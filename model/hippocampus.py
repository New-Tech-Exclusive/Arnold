"""Tier 4 — The Hippocampus (Experience Buffer).

Short-term memory.  The staging area between experience and long-term storage.
Nothing goes directly from conversation to permanent weights — it always passes
through here first.

Records are tagged with novelty, mood, and reinforcement, then prioritised
for consolidation.
"""

from __future__ import annotations

import torch

from .genome import Genome
from .types_ import EpisodicRecord, HebbianTrace, MoodState


class Hippocampus:
    """Experience buffer that stages episodic records for consolidation."""

    def __init__(self, genome: Genome) -> None:
        self._genome = genome
        self._capacity = genome.topology.hippocampus_capacity
        self._buffer: list[EpisodicRecord] = []
        self._interaction_counter = 0

    # ------------------------------------------------------------------
    # Recording
    # ------------------------------------------------------------------

    def record(
        self,
        token_ids: torch.Tensor,
        traces: list[HebbianTrace],
        novelty_score: float,
        mood: MoodState,
        reinforcement: float,
    ) -> EpisodicRecord:
        """Add a new episodic record to the buffer."""
        self._interaction_counter += 1

        # Check for repetition — same token prefix seen before
        rep_count = self._count_repetitions(token_ids)

        record = EpisodicRecord(
            token_ids=token_ids,
            hebbian_traces=traces,
            novelty_score=novelty_score,
            mood_at_event=MoodState(mood.valence, mood.arousal, mood.openness),
            reinforcement=reinforcement,
            interaction_number=self._interaction_counter,
            repetition_count=rep_count,
        )
        record.compute_priority()
        self._buffer.append(record)

        return record

    # ------------------------------------------------------------------
    # Buffer management
    # ------------------------------------------------------------------

    @property
    def utilization(self) -> float:
        """Fraction of buffer capacity currently used."""
        return len(self._buffer) / self._capacity if self._capacity > 0 else 1.0

    @property
    def needs_consolidation(self) -> bool:
        """True when buffer hits the trigger threshold."""
        return self.utilization >= self._genome.consolidation.buffer_trigger_pct

    @property
    def size(self) -> int:
        return len(self._buffer)

    def drain(self) -> list[EpisodicRecord]:
        """Remove and return all records sorted by consolidation priority (desc)."""
        records = sorted(self._buffer, key=lambda r: r.consolidation_priority, reverse=True)
        self._buffer = []
        return records

    def peek(self) -> list[EpisodicRecord]:
        """Return records without draining (for inspection)."""
        return sorted(self._buffer, key=lambda r: r.consolidation_priority, reverse=True)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _count_repetitions(self, token_ids: torch.Tensor) -> int:
        """Count how many existing buffer records share a token-prefix overlap."""
        if len(token_ids) == 0:
            return 0
        prefix_len = min(10, len(token_ids))
        prefix = tuple(token_ids[:prefix_len].tolist())
        count = 0
        for rec in self._buffer:
            if len(rec.token_ids) >= prefix_len:
                existing_prefix = tuple(rec.token_ids[:prefix_len].tolist())
                if prefix == existing_prefix:
                    count += 1
        return count

    @property
    def interaction_count(self) -> int:
        return self._interaction_counter

    def set_interaction_counter(self, value: int) -> None:
        self._interaction_counter = value
