"""Basal Ganglia — Action Selection and Habit Formation.

Implements a go/no-go mechanism.  Frequently reinforced response patterns
get stored as chunked habits that can shortcut full transformer processing.

This is how the model develops a consistent voice over time — habituated
patterns produce recognisable stylistic tendencies that emerge automatically.
"""

from __future__ import annotations

import math

import torch

from .genome import Genome
from .tensor import DTYPE, get_device


class HabitSystem:
    """Habit store with go/no-go action selection."""

    def __init__(self, genome: Genome) -> None:
        self._genome = genome
        self._device = get_device()
        self._params = genome.habit_system

        # Habit store: list of (context_embedding, token_sequence, strength, occurrence_count)
        self._habits: list[dict] = []

    def check_habit(
        self,
        context_embedding: torch.Tensor,
    ) -> list[int] | None:
        """Check if the current context matches a habituated pattern.

        Returns the habituated token sequence if similarity exceeds threshold,
        otherwise None (full transformer processing required).
        """
        if not self._habits:
            return None

        context = context_embedding.to(self._device)
        context_norm = torch.linalg.norm(context)
        if context_norm < 1e-8:
            return None

        best_sim = -1.0
        best_idx = -1

        for i, habit in enumerate(self._habits):
            stored = habit["context"].to(self._device)
            sim = float(torch.dot(context, stored) / (context_norm * torch.linalg.norm(stored) + 1e-8))
            if sim > best_sim:
                best_sim = sim
                best_idx = i

        if best_sim >= self._params.habit_match_threshold and best_idx >= 0:
            return self._habits[best_idx]["tokens"]

        return None

    def record_sequence(
        self,
        context_embedding: torch.Tensor,
        tokens: list[int],
        reinforcement: float,
    ) -> None:
        """After generation, potentially store or update a habit.

        A sequence must be positively reinforced above threshold at least
        `habit_min_occurrences` times before becoming a habit.
        """
        if reinforcement < self._params.habit_threshold:
            return

        context = context_embedding.detach().to(self._device)
        context_norm = torch.linalg.norm(context)
        if context_norm < 1e-8:
            return

        # Check if similar context already in store
        for habit in self._habits:
            stored = habit["context"].to(self._device)
            sim = float(torch.dot(context, stored) / (context_norm * torch.linalg.norm(stored) + 1e-8))
            if sim >= self._params.habit_match_threshold:
                habit["count"] += 1
                # Running average of reinforcement
                habit["strength"] = 0.9 * habit["strength"] + 0.1 * reinforcement
                # Update tokens to latest version if count passes threshold
                if habit["count"] >= self._params.habit_min_occurrences:
                    habit["tokens"] = tokens
                return

        # New potential habit
        self._habits.append({
            "context": context.clone(),
            "tokens": tokens,
            "strength": reinforcement,
            "count": 1,
        })

        # Evict weakest if over capacity
        if len(self._habits) > self._params.max_habits:
            min_idx = min(range(len(self._habits)), key=lambda i: self._habits[i]["strength"])
            self._habits.pop(min_idx)

    def decay_habits(self) -> None:
        """Slow decay of unused habits each session."""
        for habit in self._habits:
            habit["strength"] *= self._params.habit_decay
        # Remove habits below minimum strength
        self._habits = [h for h in self._habits if h["strength"] > 0.1]

    def get_state(self) -> dict:
        """Serialise habit store for persistence."""
        return {
            "habits": [
                {
                    "context": h["context"].detach().cpu(),
                    "tokens": h["tokens"],
                    "strength": h["strength"],
                    "count": h["count"],
                }
                for h in self._habits
            ],
        }

    def set_state(self, state: dict) -> None:
        """Restore habit store from persistence."""
        self._habits = []
        for h in state.get("habits", []):
            ctx = h["context"]
            if isinstance(ctx, torch.Tensor):
                ctx = ctx.to(device=self._device, dtype=DTYPE)
            self._habits.append({
                "context": ctx,
                "tokens": h["tokens"],
                "strength": h["strength"],
                "count": h["count"],
            })
