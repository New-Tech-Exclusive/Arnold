#!/usr/bin/env python3
"""Evaluation harness for the Arnold model.

Provides simple language-modeling metrics (cross-entropy, perplexity) and
collects a few internal signals (prediction error, surprise) for diagnostics.
You can point it at a saved model and a HuggingFace dataset and it will
iterate over a subset of examples.  Metrics are also written to TensorBoard
for visualization.

Usage:
    python evaluate.py --storage_dir ./model/pretrained --dataset wikitext --split test

The script is intentionally minimalist; it's intended to be a scaffolding you
can extend with BLEU, accuracy on downstream tasks, or integration with
Weights & Biases.
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path
from typing import Iterator

# Prevent triton segfault (see pretrain.py for full explanation).
sys.modules["triton"] = None

import torch
try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:  # tensorboard not installed (tests, minimal environments)
    SummaryWriter = None

try:
    from datasets import load_dataset
except ImportError:
    print("ERROR: 'datasets' package is not installed.\n  Install it with:  pip install datasets")
    sys.exit(1)

# reuse tokenization helpers from pretrain
from pretrain import tokenize, _detect_text_field, _extract_text_sample
from model.model import Model
from model.genome import Genome, LayerTopology, HebbianParams, GenerationParams
from model.tensor import seed_all, get_device


def iter_texts(dataset_name: str, split: str, seed: int = 42) -> Iterator[str]:
    """Yield raw text samples from a HF dataset; autodetects first text field."""
    ds = load_dataset(dataset_name, split=split, streaming=False)
    ds = ds.shuffle(seed=seed)
    first = next(iter(ds))
    field = _detect_text_field(first, None)
    for sample in ds:
        text = _extract_text_sample(sample, field)
        if text:
            yield text


def evaluate_lm(
    model: Model,
    texts: Iterator[str],
    max_seq_len: int,
    tb_writer: SummaryWriter | None = None,
    max_examples: int | None = None,
) -> dict[str, float]:
    """Run the model in evaluation mode, compute cross-entropy/perplexity."""
    losses = []
    pred_errors = []
    surprise_scores = []
    model.eval_mode = True  # just a flag; nothing else uses it yet

    for idx, text in enumerate(texts):
        if max_examples is not None and idx >= max_examples:
            break
        token_ids = tokenize(text, max_seq_len)
        if token_ids is None or token_ids.numel() < 2:
            continue
        with torch.no_grad():
            logits = model._compute_lm_logits(token_ids)
        if logits.numel() == 0:
            continue
        target = token_ids[1 : logits.shape[0] + 1]
        loss = torch.nn.functional.cross_entropy(logits, target, reduction="mean")
        losses.append(float(loss.item()))

        # internal signals from a dummy forward pass
        structured = model.encoder.process(token_ids)
        pred_errors.append(model.transformer.mean_prediction_error())
        surprise_scores.append(model.surprise_detector.score(structured, model.transformer.forward_readonly))

        if tb_writer is not None:
            tb_writer.add_scalar("eval/loss", float(loss.item()), idx)
            tb_writer.add_scalar("eval/pred_error", pred_errors[-1], idx)
            tb_writer.add_scalar("eval/surprise", surprise_scores[-1], idx)

    avg_loss = sum(losses) / max(len(losses), 1)
    ppl = math.exp(avg_loss) if avg_loss < 50 else float("inf")
    return {"loss": avg_loss, "perplexity": ppl, "prediction_error": sum(pred_errors) / max(len(pred_errors), 1),
            "surprise": sum(surprise_scores) / max(len(surprise_scores), 1)}


def main():
    parser = argparse.ArgumentParser(description="Evaluate a saved Arnold model")
    parser.add_argument("--storage_dir", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--max_examples", type=int, default=500)
    parser.add_argument("--max_seq_len", type=int, default=2048)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    seed_all(args.seed)

    # load model (assumes topology stored in metadata)
    genome = Genome()
    model = Model(genome=genome, storage_dir=args.storage_dir, restore=True)
    model.session_start()

    tb = SummaryWriter(log_dir=Path(args.storage_dir) / "eval_logs")

    texts = iter_texts(args.dataset, args.split, seed=args.seed)
    metrics = evaluate_lm(model, texts, args.max_seq_len, tb_writer=tb, max_examples=args.max_examples)

    for k, v in metrics.items():
        print(f"{k}: {v}")

    tb.close()


if __name__ == "__main__":
    main()
