#!/usr/bin/env python3
"""Pretraining - trains the Brainstem (infancy) on a HuggingFace dataset."""

from __future__ import annotations

import argparse
import gc
import json
import os
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterator

import numpy as np
import torch

try:
    from datasets import load_dataset
except ImportError:
    print("ERROR: 'datasets' package is not installed.")
    print("  Install it with:  pip install datasets")
    sys.exit(1)

_REPO = Path(__file__).parent
sys.path.insert(0, str(_REPO))

from model.brain import Brain
from model.genome import GenerationParams, Genome, HebbianParams, LayerTopology
from model.tensor import get_device, seed_all

TOKENIZER = None
TOKENIZER_VOCAB_LIMIT: int | None = None
VOCAB_SIZE = 50257


def _successful_exit() -> None:
    """Terminate cleanly after successful pretraining.

    The Hugging Face streaming stack can crash during Python 3.12 interpreter
    finalization after all useful work is already done. Clear heavyweight
    globals, flush stdio, and exit the process before that teardown path runs.
    """
    global TOKENIZER, TOKENIZER_VOCAB_LIMIT
    TOKENIZER = None
    TOKENIZER_VOCAB_LIMIT = None
    gc.collect()
    if torch.cuda.is_available():
        try:
            torch.cuda.synchronize()
        except Exception:
            pass
        torch.cuda.empty_cache()
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)


@dataclass
class TrainingConfig:
    dataset: str = "allenai/WildChat-1M"
    dataset_config: str | None = None
    split: str = "train"
    text_field: str | None = None
    streaming: bool = True

    steps: int = 500
    lr: float = 0.01
    max_seq_len: int = 2048

    # Increased default sizes (~100M params target)
    embed_dim: int = 448
    brainstem_hidden: int = 896
    cortex_hidden: int = 896
    hippocampus_capacity: int = 4096
    vocab_size: int | None = None
    tokenizer_name: str = "gpt2"

    storage_dir: str = "./model/pretrained"
    resume: bool = False
    seed: int = 42

    @classmethod
    def from_json(cls, path: str | Path) -> "TrainingConfig":
        data = json.loads(Path(path).read_text())
        valid = {k: v for k, v in data.items() if k in cls.__dataclass_fields__}
        return cls(**valid)

    @classmethod
    def from_args(cls, args: argparse.Namespace) -> "TrainingConfig":
        cfg = cls()
        if args.config:
            cfg = cls.from_json(args.config)
        # Apply named preset first so individual arg flags can still override it
        preset = getattr(args, "preset", None)
        if preset and preset in _DATASET_PRESETS:
            for key, val in _DATASET_PRESETS[preset].items():
                setattr(cfg, key, val)
        for key in cls.__dataclass_fields__.keys():
            val = getattr(args, key, None)
            if val is not None:
                setattr(cfg, key, val)
        return cfg


def _resolve_vocab_size(tokenizer: object | None, requested_vocab_size: int | None) -> int:
    if requested_vocab_size is not None:
        return int(requested_vocab_size)
    if tokenizer is not None:
        try:
            return int(len(tokenizer))
        except Exception:
            pass
        tokenizer_vocab = getattr(tokenizer, "vocab_size", None)
        if tokenizer_vocab is not None:
            return int(tokenizer_vocab)
    return VOCAB_SIZE


def tokenize(text: str, max_seq_len: int) -> torch.Tensor | None:
    global TOKENIZER, TOKENIZER_VOCAB_LIMIT
    if TOKENIZER is not None:
        toks = TOKENIZER.encode(text, truncation=True, max_length=max_seq_len)
        if TOKENIZER_VOCAB_LIMIT is not None and TOKENIZER_VOCAB_LIMIT > 0:
            toks = [min(int(t), TOKENIZER_VOCAB_LIMIT - 1) for t in toks]
        if len(toks) < 1:
            return None
        return torch.tensor(toks, dtype=torch.long, device=get_device())

    raw = text.encode("utf-8", errors="replace")
    if len(raw) < 4:
        return None
    ids = np.frombuffer(raw[:max_seq_len], dtype=np.uint8).astype(np.int64)
    return torch.tensor(ids.tolist(), dtype=torch.long, device=get_device())


_TEXT_FIELDS = (
    "text",
    "content",
    "conversation",
    "messages",
    "story",
    "document",
    "article",
    "sentence",
)


def _extract_text_value(value: Any) -> str | None:
    if isinstance(value, str):
        text = value.strip()
        return text if text else None

    if isinstance(value, dict):
        for key in ("content", "text", "message", "value", "prompt", "response"):
            extracted = _extract_text_value(value.get(key))
            if extracted:
                return extracted
        return None

    if isinstance(value, (list, tuple)):
        parts: list[str] = []
        for item in value:
            extracted = _extract_text_value(item)
            if extracted:
                parts.append(extracted)
        if parts:
            return "\n".join(parts)
        return None

    return None


def _extract_text_sample(sample: dict, field: str) -> str | None:
    return _extract_text_value(sample.get(field))


def _detect_text_field(sample: dict, override: str | None) -> str:
    if override:
        if override not in sample:
            raise KeyError(
                f"--text_field '{override}' not found in dataset. Available fields: {list(sample.keys())}"
            )
        if _extract_text_sample(sample, override) is None:
            raise KeyError(
                f"--text_field '{override}' exists but did not contain extractable text in the first sample. "
                f"Available fields: {list(sample.keys())}"
            )
        return override

    for f in _TEXT_FIELDS:
        if f in sample and _extract_text_sample(sample, f) is not None:
            return f

    for f in sample.keys():
        if _extract_text_sample(sample, f) is not None:
            return f

    raise KeyError(
        f"Could not auto-detect the text field. Available fields: {list(sample.keys())}. "
        f"Pass --text_field <name> explicitly."
    )


def iter_sequences(cfg: TrainingConfig) -> Iterator[torch.Tensor]:
    kwargs: dict[str, Any] = {"split": cfg.split}
    if cfg.dataset_config:
        kwargs["name"] = cfg.dataset_config
    if cfg.streaming:
        kwargs["streaming"] = True

    print(
        f"  Loading dataset '{cfg.dataset}'"
        + (f" ({cfg.dataset_config})" if cfg.dataset_config else "")
        + f" split='{cfg.split}' streaming={cfg.streaming} ..."
    )

    ds = load_dataset(cfg.dataset, **kwargs)
    if not cfg.streaming:
        ds = ds.shuffle(seed=cfg.seed)

    first = next(iter(ds))
    field = _detect_text_field(first, cfg.text_field)
    print(f"  Text field: '{field}'")

    while True:
        for sample in ds:
            text = _extract_text_sample(sample, field)
            if text is None:
                continue
            ids = tokenize(text, cfg.max_seq_len)
            if ids is not None:
                yield ids
        if cfg.streaming:
            ds = load_dataset(cfg.dataset, **kwargs)


def pretrain(cfg: TrainingConfig) -> None:
    storage_dir = Path(cfg.storage_dir)
    storage_dir.mkdir(parents=True, exist_ok=True)

    seed_all(cfg.seed)

    print("\n" + "=" * 64)
    print("  BRAINSTEM PRETRAINING")
    print("=" * 64)

    tokenizer = None
    if cfg.tokenizer_name and cfg.tokenizer_name.strip() != "":
        try:
            from transformers import AutoTokenizer
        except Exception:
            print("ERROR: transformers not installed. Install with: pip install transformers")
            raise
        tokenizer = AutoTokenizer.from_pretrained(cfg.tokenizer_name, use_fast=True)
        global TOKENIZER, TOKENIZER_VOCAB_LIMIT
        TOKENIZER = tokenizer

    resolved_vocab_size = _resolve_vocab_size(tokenizer, cfg.vocab_size)
    if tokenizer is not None:
        TOKENIZER_VOCAB_LIMIT = resolved_vocab_size

    topo = LayerTopology(
        vocab_size=resolved_vocab_size,
        embed_dim=cfg.embed_dim,
        brainstem_hidden=cfg.brainstem_hidden,
        cortex_hidden=cfg.cortex_hidden,
        hippocampus_capacity=cfg.hippocampus_capacity,
    )
    gen_params = GenerationParams(brainstem_pretrain_steps=cfg.steps)
    hebb = HebbianParams(learning_rate=cfg.lr)
    genome = Genome(topology=topo, hebbian=hebb, generation=gen_params)

    # Only restore a previous state when explicitly resuming.
    brain = Brain(
        genome=genome,
        storage_dir=storage_dir,
        seed=cfg.seed,
        tokenizer=tokenizer,
        restore=cfg.resume,
    )

    if cfg.resume and brain.weight_store.exists():
        print(f"  Resuming from existing state in {storage_dir}")
        print(f"  Brainstem frozen: {brain.brainstem.is_frozen}")
        if brain.brainstem.is_frozen:
            print("  Brainstem is already frozen - skipping pretraining.")
            return
    else:
        print("  New brain - pretraining from scratch")

    print("\n  Training config:")
    for k, v in asdict(cfg).items():
        print(f"    {k}: {v}")

    print(f"\n  Collecting up to {cfg.steps} training sequences from dataset...")
    sequences: list[torch.Tensor] = []
    seq_iter = iter_sequences(cfg)

    try:
        from tqdm import tqdm
    except Exception:
        tqdm = None

    if tqdm is not None:
        pbar = tqdm(total=cfg.steps, desc="Collecting sequences")
    else:
        pbar = None

    t_collect = time.perf_counter()
    while len(sequences) < cfg.steps:
        sequences.append(next(seq_iter))
        if pbar is not None:
            pbar.update(1)
        if len(sequences) % 500 == 0 and pbar is None:
            elapsed = time.perf_counter() - t_collect
            print(f"    Collected {len(sequences):>6d} / {cfg.steps}  ({elapsed:.1f}s)")
    if pbar is not None:
        pbar.close()

    avg_len = int(sum(int(s.numel()) for s in sequences) / max(len(sequences), 1))
    print(f"  Collected {len(sequences)} sequences  (avg length: {avg_len} tokens)")

    print(f"\n  Pretraining brainstem for {cfg.steps} Hebbian steps...")
    t0 = time.perf_counter()

    brain._genome = Genome(
        topology=topo,
        hebbian=hebb,
        generation=GenerationParams(brainstem_pretrain_steps=len(sequences)),
    )

    brain.birth(pretraining_corpus=sequences)

    dt = time.perf_counter() - t0
    print(f"  Pretraining complete in {dt:.1f}s  ({len(sequences) / max(dt, 1e-6):.0f} seq/s)")
    print(f"  Brainstem frozen: {brain.brainstem.is_frozen}")

    print(f"\n  Saving brain state to {storage_dir} ...")
    brain._save_state()
    print("  Saved.")

    print("\n  Sanity check - forward pass on 'hello world'...")
    sample_ids = tokenize("hello world", cfg.max_seq_len)
    if sample_ids is not None:
        structured = brain.brainstem.process(sample_ids)
        novelty = brain.novelty_detector.score(structured, brain.cortex.forward_readonly)
        print(f"    input tokens:  {int(sample_ids.numel())}")
        print(f"    embedding dim: {tuple(structured.embeddings.shape)}")
        print(f"    novelty score: {novelty:.4f}")
        print("    forward pass succeeded")

    print(f"\n  Brain state saved to: {storage_dir}")
    print("  Start the chat server with:")
    print(f"    python chat_server.py --storage_dir {storage_dir}\n")


# Named dataset presets — pass --preset <name> to quickly switch corpora.
# Individual flags (--dataset, --split, etc.) override the preset.
_DATASET_PRESETS: dict[str, dict[str, str]] = {
    "wildchat": {
        "dataset": "allenai/WildChat-1M",
        "split": "train",
    },
    "tinystories": {  # clean simple narrative; ideal for early developmental stage
        "dataset": "roneneldan/TinyStories",
        "split": "train",
    },
    "wikitext": {  # factual Wikipedia prose
        "dataset": "wikitext",
        "dataset_config": "wikitext-103-v1",
        "split": "train",
    },
    "openhermes": {  # instruction-tuning Q&A pairs
        "dataset": "teknium/OpenHermes-2.5",
        "split": "train",
    },
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Pretrain the Arnold brain brainstem on a HuggingFace dataset",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    p.add_argument("--config", default=None, help="Path to JSON training config")

    g = p.add_argument_group("Dataset")
    g.add_argument("--dataset", default=None, help="HuggingFace dataset name")
    g.add_argument("--dataset_config", default=None, help="Dataset configuration/subset")
    g.add_argument("--split", default=None, help="Dataset split to use")
    g.add_argument("--text_field", default=None, help="Name of text column")
    g.add_argument("--streaming", action="store_true", default=None, help="Use HF streaming mode")
    g.add_argument(
        "--preset", default=None,
        choices=["wildchat", "tinystories", "wikitext", "openhermes"],
        help=(
            "Named dataset preset (shortcut for --dataset + --split).\n"
            "  wildchat    — allenai/WildChat-1M (default, diverse dialogue)\n"
            "  tinystories — roneneldan/TinyStories (clean, simple narrative; best for infancy)\n"
            "  wikitext    — wikitext-103 (factual Wikipedia prose)\n"
            "  openhermes  — teknium/OpenHermes-2.5 (instruction Q&A pairs)"
        ),
    )

    g = p.add_argument_group("Training")
    g.add_argument("--steps", type=int, default=None, help="Hebbian update steps")
    g.add_argument("--lr", type=float, default=None, help="Hebbian learning rate")
    g.add_argument("--max_seq_len", type=int, default=None, help="Max tokens per sequence")

    g = p.add_argument_group("Model dimensions")
    g.add_argument("--embed_dim", type=int, default=None)
    g.add_argument("--brainstem_hidden", type=int, default=None)
    g.add_argument("--cortex_hidden", type=int, default=None)
    g.add_argument("--hippocampus_capacity", type=int, default=None)
    g.add_argument("--vocab_size", type=int, default=None)
    g.add_argument("--tokenizer_name", type=str, default=None)

    g = p.add_argument_group("Storage")
    g.add_argument("--storage_dir", default=None)
    g.add_argument("--resume", action="store_true", default=None)

    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--print_default_config", action="store_true", help="Print default config JSON and exit")

    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.print_default_config:
        print(json.dumps(asdict(TrainingConfig()), indent=2))
        raise SystemExit(0)
    cfg = TrainingConfig.from_args(args)
    pretrain(cfg)
    _successful_exit()
