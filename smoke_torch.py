#!/usr/bin/env python3
"""Quick smoke test for the PyTorch-backed model pipeline."""

from __future__ import annotations

import sys

# Prevent triton segfault (see pretrain.py for full explanation).
sys.modules["triton"] = None

from model.model import Model
from model.genome import Genome, LayerTopology


def main() -> None:
    genome = Genome(
        topology=LayerTopology(
            vocab_size=512,
            embed_dim=32,
            encoder_hidden=64,
            transformer_hidden=64,
            memory_capacity=128,
        )
    )
    model = Model(genome=genome, storage_dir="./model/pretrained_smoke_small", seed=1)
    model.birth(pretraining_corpus=[])
    model.session_start()
    result = model.process_turn("hello world")
    print(f"tokens={len(result.generated_tokens)} surprise={result.novelty_score:.4f}")


if __name__ == "__main__":
    main()
