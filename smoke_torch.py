#!/usr/bin/env python3
"""Quick smoke test for the PyTorch-backed brain pipeline."""

from __future__ import annotations

from model.brain import Brain
from model.genome import Genome, LayerTopology


def main() -> None:
    genome = Genome(
        topology=LayerTopology(
            vocab_size=512,
            embed_dim=32,
            brainstem_hidden=64,
            cortex_hidden=64,
            hippocampus_capacity=128,
        )
    )
    brain = Brain(genome=genome, storage_dir="./model/pretrained_smoke_small", seed=1)
    brain.birth(pretraining_corpus=[])
    brain.session_start()
    result = brain.process_turn("hello world")
    print(f"tokens={len(result.generated_tokens)} novelty={result.novelty_score:.4f}")


if __name__ == "__main__":
    main()
