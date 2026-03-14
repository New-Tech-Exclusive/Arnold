"""Tier 8 — The Long Term Weight Store (Persistent Model State).

Saves and loads the complete model state between sessions.
Uses numpy's npz format for efficient serialisation of weight matrices.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import tempfile

import numpy as np
import torch

from .genome import Genome, LayerTopology
from .tensor import DTYPE, get_device
from .types_ import ModelState, TransformerRegion, MoodState, NeuromodulatorState, PersonalityVector


class WeightStore:
    """Persistent storage for the model state."""

    def __init__(self, storage_dir: str | Path) -> None:
        self._dir = Path(storage_dir)
        self._dir.mkdir(parents=True, exist_ok=True)
        self._weights_path = self._dir / "model_weights.pt"
        self._legacy_weights_path = self._dir / "model_weights.npz"  # backward compat
        self._meta_path = self._dir / "model_meta.json"

    # ------------------------------------------------------------------
    # Save
    # ------------------------------------------------------------------

    def load_topology(self) -> LayerTopology | None:
        """Return the LayerTopology stored in the most recent save, or None."""
        if not self._meta_path.exists():
            return None
        try:
            meta = json.loads(self._meta_path.read_text())
            topo_data = meta.get("topology")
            if topo_data is None:
                return None
            return LayerTopology(
                vocab_size=topo_data.get("vocab_size", 30000),
                embed_dim=topo_data.get("embed_dim", 64),
                encoder_hidden=topo_data.get("encoder_hidden", 128),
                transformer_hidden=topo_data.get("transformer_hidden", topo_data.get("cortex_hidden", 128)),
                cortex_hidden=topo_data.get("cortex_hidden", 128),
                memory_capacity=topo_data.get("memory_capacity", 1024),
                hippocampus_capacity=topo_data.get("hippocampus_capacity", 1024),
            )
        except Exception:
            return None

    def save(self, state: ModelState, genome: Genome | None = None, seed: int | None = None) -> None:
        """Persist the complete model state to disk.

        Args:
            state: the ModelState dataclass containing tensor weights and other
                bookkeeping information.
            genome: optional Genome instance so topology can be saved.
            seed: optional random seed for reproducibility.  Stored in metadata.
        """
        tensors: dict[str, torch.Tensor] = {}

        # Encoder weights
        for name, arr in state.encoder_weights.items():
            tensors[f"encoder_{name}"] = arr.detach().cpu()

        for name, arr in state.encoder_ewc_protection.items():
            tensors[f"encoderewc_{name}"] = arr.detach().cpu()

        # Transformer weights
        for region, arr in state.transformer_weights.items():
            tensors[f"cortex_{region.value}"] = arr.detach().cpu()

        # Inter-region weights
        for (r1, r2), arr in state.inter_region_weights.items():
            tensors[f"interregion_{r1.value}_{r2.value}"] = arr.detach().cpu()

        # EWC protection scalars
        for region, arr in state.ewc_protection.items():
            tensors[f"ewc_{region.value}"] = arr.detach().cpu()

        # Plasticity rates as tensor
        if state.plasticity_rates:
            keys = sorted(state.plasticity_rates.keys(), key=lambda r: r.value)
            tensors["plasticity_values"] = torch.tensor(
                [state.plasticity_rates[k] for k in keys], dtype=torch.float64,
            )

        # Thalamus weights
        if state.thalamus_weights:
            for name, arr in state.thalamus_weights.items():
                tensors[f"thalamus_{name}"] = arr.detach().cpu()

        # Cerebellum weights
        if state.cerebellum_weights:
            for name, arr in state.cerebellum_weights.items():
                tensors[f"cerebellum_{name}"] = arr.detach().cpu()

        # Astrocyte usage counters
        if state.astrocyte_usage:
            for region, arr in state.astrocyte_usage.items():
                tensors[f"astrocyte_{region.value}"] = arr.detach().cpu()

        # Transformer predictions (predictive coding)
        if state.cortex_predictions:
            for region, arr in state.cortex_predictions.items():
                tensors[f"cortexpred_{region.value}"] = arr.detach().cpu()

        # Write atomically; torch.save is much faster than np.savez_compressed
        # for large tensors (no zlib overhead).
        with tempfile.NamedTemporaryFile(
            dir=str(self._dir), prefix="model_weights_", suffix=".pt", delete=False,
        ) as tmp:
            tmp_weights_path = Path(tmp.name)
        try:
            torch.save(tensors, str(tmp_weights_path))
            os.replace(str(tmp_weights_path), str(self._weights_path))
        finally:
            if tmp_weights_path.exists():
                os.remove(tmp_weights_path)

        # Metadata (JSON-serialisable)
        meta: dict = {
            "personality": {
                name: getattr(state.personality, name)
                for name in PersonalityVector.TRAIT_NAMES
            },
            # include seed if available for reproducibility
            **({"seed": seed} if seed is not None else {}),
            "mood_baseline": {
                "valence": state.mood_baseline.valence,
                "arousal": state.mood_baseline.arousal,
                "openness": state.mood_baseline.openness,
            },
            "developmental_age": state.developmental_age,
            "plasticity_rate_keys": [r.value for r in sorted(
                state.plasticity_rates.keys(), key=lambda r: r.value,
            )] if state.plasticity_rates else [],
            "consolidation_meta": state.consolidation_meta,
            "inter_region_highway": {
                f"{r1.value}_{r2.value}": v
                for (r1, r2), v in state.inter_region_highway.items()
            },
        }

        # Neuromodulator baseline
        if state.neuromodulator_baseline is not None:
            nm = state.neuromodulator_baseline
            meta["neuromodulator_baseline"] = {
                "dopamine": nm.dopamine,
                "serotonin": nm.serotonin,
                "acetylcholine": nm.acetylcholine,
                "norepinephrine": nm.norepinephrine,
            }

        # Habit store (basal ganglia) — convert tensor contexts to JSON-serialisable lists
        if state.habit_store is not None:
            serialisable_habits = []
            for h in state.habit_store.get("habits", []):
                entry = dict(h)
                if isinstance(entry.get("context"), torch.Tensor):
                    entry["context"] = entry["context"].detach().cpu().tolist()
                serialisable_habits.append(entry)
            meta["habit_store"] = {"habits": serialisable_habits}

        # Persist topology so the server can reload with matching dimensions.
        if genome is not None:
            t = genome.topology
            meta["topology"] = {
                "vocab_size": t.vocab_size,
                "embed_dim": t.embed_dim,
                "encoder_hidden": t.encoder_hidden,
                "transformer_hidden": t.transformer_hidden,
                "cortex_hidden": t.cortex_hidden,
                "memory_capacity": t.memory_capacity,
                "hippocampus_capacity": t.hippocampus_capacity,
            }

        with tempfile.NamedTemporaryFile(
            dir=str(self._dir), prefix="model_meta_", suffix=".json", mode="w", delete=False,
        ) as tmp_meta:
            tmp_meta_path = Path(tmp_meta.name)
            json.dump(meta, tmp_meta, indent=2)
        try:
            os.replace(str(tmp_meta_path), str(self._meta_path))
        finally:
            if tmp_meta_path.exists():
                os.remove(tmp_meta_path)

    # ------------------------------------------------------------------
    # Load
    # ------------------------------------------------------------------

    def load(self) -> ModelState | None:
        """Load model state from disk. Returns None if no save exists."""
        if not self._meta_path.exists():
            return None
        if not self._weights_path.exists() and not self._legacy_weights_path.exists():
            return None

        try:
            with open(self._meta_path, "r") as f:
                meta = json.load(f)

            state = ModelState()

            # Load tensors — prefer new .pt format, fall back to legacy .npz
            if self._weights_path.exists():
                raw = torch.load(
                    str(self._weights_path), map_location="cpu", weights_only=True,
                )
                data_items = {k: v.to(dtype=DTYPE, device=get_device()) for k, v in raw.items()}
            else:
                with np.load(str(self._legacy_weights_path), allow_pickle=False) as npz:
                    data_items = {
                        k: torch.as_tensor(npz[k], dtype=DTYPE, device=get_device())
                        for k in npz.files
                    }

            rate_keys = meta.get("plasticity_rate_keys", [])
            for key, tensor in data_items.items():
                if key.startswith("encoder_"):
                    state.encoder_weights[key[len("encoder_"):]] = tensor
                elif key.startswith("encoderewc_"):
                    state.encoder_ewc_protection[key[len("encoderewc_"):]] = tensor
                elif key.startswith("cortex_"):
                    state.transformer_weights[TransformerRegion(key[len("cortex_"):])] = tensor
                elif key.startswith("interregion_"):
                    parts = key[len("interregion_"):].split("_", 1)
                    if len(parts) == 2:
                        state.inter_region_weights[
                            (TransformerRegion(parts[0]), TransformerRegion(parts[1]))
                        ] = tensor
                elif key.startswith("ewc_"):
                    state.ewc_protection[TransformerRegion(key[len("ewc_"):])] = tensor
                elif key.startswith("thalamus_"):
                    state.thalamus_weights[key[len("thalamus_"):]] = tensor
                elif key.startswith("cerebellum_"):
                    state.cerebellum_weights[key[len("cerebellum_"):]] = tensor
                elif key.startswith("astrocyte_"):
                    try:
                        state.astrocyte_usage[TransformerRegion(key[len("astrocyte_"):])] = tensor
                    except ValueError:
                        pass
                elif key.startswith("cortexpred_"):
                    try:
                        state.cortex_predictions[TransformerRegion(key[len("cortexpred_"):])] = tensor
                    except ValueError:
                        pass
                elif key == "plasticity_values":
                    for i, key_name in enumerate(rate_keys):
                        if i < tensor.numel():
                            state.plasticity_rates[TransformerRegion(key_name)] = float(tensor[i].item())

            # Personality
            p = meta.get("personality", {})
            state.personality = PersonalityVector(
                curiosity=p.get("curiosity", 0.5),
                warmth=p.get("warmth", 0.5),
                assertiveness=p.get("assertiveness", 0.5),
                creativity=p.get("creativity", 0.5),
                caution=p.get("caution", 0.5),
                humor=p.get("humor", 0.5),
            )

            # Mood baseline
            m = meta.get("mood_baseline", {})
            state.mood_baseline = MoodState(
                valence=m.get("valence", 0.0),
                arousal=m.get("arousal", 0.2),
                openness=m.get("openness", 0.5),
            )

            state.developmental_age = meta.get("developmental_age", 0)
            state.consolidation_meta = meta.get("consolidation_meta", {})

            highway = meta.get("inter_region_highway", {})
            for key_str, val in highway.items():
                parts = key_str.split("_", 1)
                if len(parts) == 2:
                    state.inter_region_highway[
                        (TransformerRegion(parts[0]), TransformerRegion(parts[1]))
                    ] = float(val)

            # Neuromodulator baseline (backward compat: missing → None)
            nm_data = meta.get("neuromodulator_baseline")
            if nm_data is not None:
                state.neuromodulator_baseline = NeuromodulatorState(
                    dopamine=nm_data.get("dopamine", 0.0),
                    serotonin=nm_data.get("serotonin", 0.5),
                    acetylcholine=nm_data.get("acetylcholine", 0.3),
                    norepinephrine=nm_data.get("norepinephrine", 0.3),
                )

            # Habit store (basal ganglia) — restore tensor contexts from lists
            habit_data = meta.get("habit_store")
            if habit_data is not None:
                for h in habit_data.get("habits", []):
                    if isinstance(h.get("context"), list):
                        h["context"] = torch.tensor(
                            h["context"], dtype=DTYPE, device=get_device(),
                        )
            state.habit_store = habit_data

        except Exception as exc:
            raise RuntimeError(
                f"Failed to load model state from '{self._weights_path}': {exc}"
            ) from exc

        return state

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    def exists(self) -> bool:
        weights_exist = self._weights_path.exists() or self._legacy_weights_path.exists()
        return weights_exist and self._meta_path.exists()

    def delete(self) -> None:
        for path in (self._weights_path, self._legacy_weights_path, self._meta_path):
            if path.exists():
                os.remove(path)
