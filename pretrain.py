#!/usr/bin/env python3
"""Pretraining - trains the Encoder (infancy) on a HuggingFace dataset."""

from __future__ import annotations

import argparse
import gc
import json
import math
import os
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterator

# triton crashes on import on some GPU/driver configurations (SIGSEGV in
# triton/knobs.py via torch._dynamo -> has_triton_package).  This project
# never calls torch.compile(), so blocking the import is safe and prevents
# transformers' lazy-loaded masking_utils from pulling in a broken triton.
# Must run BEFORE `import torch` so torch cannot lazily register a broken
# triton module that setdefault would then refuse to overwrite.
sys.modules["triton"] = None
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

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

from model.model import Model
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
    dataset: str = "HuggingFaceFW/fineweb-edu"
    dataset_config: str | None = None
    split: str = "train"
    text_field: str | None = None
    streaming: bool = True

    steps: int = 500
    lr: float = 0.01
    # additional gradient fine-tuning on same corpus
    gradient_steps: int = 0
    gradient_lr: float = 1e-5
    gradient_batch_size: int = 1
    freeze_encoder: bool = False
    # Mixed-precision training — halves VRAM and speeds up ~2× on modern GPUs
    use_amp: bool = True
    # Accumulate over N micro-steps before each optimizer step (larger effective batch)
    accumulation_steps: int = 4
    # Linear warmup steps for cosine LR schedule
    warmup_steps: int = 100
    max_seq_len: int = 2048
    gradient_max_seq_len: int = 384

    # Increased default sizes (~100M params target)
    embed_dim: int = 448
    encoder_hidden: int = 896
    transformer_hidden: int = 896
    memory_capacity: int = 4096
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
            return torch.tensor(toks, dtype=torch.long)

    raw = text.encode("utf-8", errors="replace")
    if len(raw) < 4:
        return None
    ids = np.frombuffer(raw[:max_seq_len], dtype=np.uint8).astype(np.int64)
    return torch.tensor(ids.tolist(), dtype=torch.long)


def _prepare_gradient_sequence(
    seq: torch.Tensor,
    max_len: int,
    step: int,
) -> torch.Tensor:
    """Crop long sequences for gradient tuning to keep VRAM bounded."""
    seq = seq.detach().cpu()
    max_len = max(8, int(max_len))
    if int(seq.numel()) <= max_len:
        return seq

    span = int(seq.numel()) - max_len
    start = (step * 9973) % max(span + 1, 1)
    return seq[start:start + max_len]


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
    print("  ENCODER PRETRAINING")
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
        encoder_hidden=cfg.encoder_hidden,
        transformer_hidden=cfg.transformer_hidden,
        memory_capacity=cfg.memory_capacity,
    )
    gen_params = GenerationParams(encoder_pretrain_steps=cfg.steps)
    hebb = HebbianParams(learning_rate=cfg.lr)
    from model.genome import GradientParams as _GP
    genome = Genome(topology=topo, hebbian=hebb, generation=gen_params)
    # override gradient hyperparams from config; keep all other defaults intact
    if cfg.gradient_lr is not None:
        genome = genome.__class__(
            **{**genome.__dict__, "gradient": _GP(lr=cfg.gradient_lr)}
        )

    # Only restore a previous state when explicitly resuming.
    model = Model(
        genome=genome,
        storage_dir=storage_dir,
        seed=cfg.seed,
        tokenizer=tokenizer,
        restore=cfg.resume,
    )

    if cfg.resume and model.weight_store.exists():
        print(f"  Resuming from existing state in {storage_dir}")
        print(f"  Encoder frozen: {model.encoder.is_frozen}")
    else:
        print("  New model - pretraining from scratch")

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

    print(f"\n  Pretraining encoder for {cfg.steps} Hebbian steps...")
    t0 = time.perf_counter()

    model._genome = Genome(
        topology=topo,
        hebbian=hebb,
        generation=GenerationParams(encoder_pretrain_steps=len(sequences)),
    )

    model.birth(pretraining_corpus=sequences)

    # optional gradient-based tuning: run a few supervised passes on the corpus
    if cfg.gradient_steps > 0 and model._optimizer is not None:
        if cfg.freeze_encoder:
            model.encoder.freeze()

        use_amp = cfg.use_amp and torch.cuda.is_available()
        n_accumulate = max(1, cfg.accumulation_steps)
        n_update_steps = math.ceil(cfg.gradient_steps / n_accumulate)
        n_warmup = min(cfg.warmup_steps, n_update_steps)
        n_seqs = len(sequences)

        # Cosine LR schedule with linear warmup (uses existing transformers dep)
        scheduler = None
        try:
            from transformers import get_cosine_schedule_with_warmup
            scheduler = get_cosine_schedule_with_warmup(
                model._optimizer,
                num_warmup_steps=n_warmup,
                num_training_steps=n_update_steps,
            )
        except Exception:
            pass  # scheduler is optional

        scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

        print(
            f"  Running {cfg.gradient_steps} gradient fine-tuning steps"
            f" (batch={cfg.gradient_batch_size}, accumulate={n_accumulate},"
            f" warmup={n_warmup}, AMP={use_amp}, grad_seq_len={cfg.gradient_max_seq_len})"
        )
        avg_loss: float | None = None
        current_grad_max_len = min(cfg.gradient_max_seq_len, cfg.max_seq_len)

        grad_pbar = None
        if tqdm is not None:
            grad_pbar = tqdm(total=cfg.gradient_steps, desc="Grad train", leave=False)

        model._optimizer.zero_grad()

        for i in range(cfg.gradient_steps):
            # --- pick sequence(s) for this micro-step ---
            if cfg.gradient_batch_size <= 1:
                seqs = [sequences[i % n_seqs]]
            else:
                seqs = [
                    sequences[(i * cfg.gradient_batch_size + j) % n_seqs]
                    for j in range(cfg.gradient_batch_size)
                ]

            prepared_seqs = [
                _prepare_gradient_sequence(seq, current_grad_max_len, i + j)
                for j, seq in enumerate(seqs)
            ]

            # --- forward + scaled backward ---
            try:
                with torch.amp.autocast("cuda", enabled=use_amp):
                    step_loss: torch.Tensor | None = None
                    count = 0
                    for seq in prepared_seqs:
                        sl = model._compute_lm_loss(seq)
                        if sl is not None:
                            step_loss = sl if step_loss is None else step_loss + sl
                            count += 1
            except torch.OutOfMemoryError:
                if torch.cuda.is_available():
                    model._optimizer.zero_grad(set_to_none=True)
                    torch.cuda.empty_cache()
                new_len = max(128, current_grad_max_len // 2)
                if new_len == current_grad_max_len:
                    raise
                current_grad_max_len = new_len
                if grad_pbar is not None:
                    grad_pbar.set_postfix_str(f"oom->seq_len={current_grad_max_len}")
                continue
            if step_loss is None or count == 0:
                if grad_pbar is not None:
                    grad_pbar.update(1)
                continue
            # normalise by accumulation window so effective gradient ≈ full-batch
            micro_loss = step_loss / (count * n_accumulate)
            try:
                scaler.scale(micro_loss).backward()
            except torch.OutOfMemoryError:
                if torch.cuda.is_available():
                    model._optimizer.zero_grad(set_to_none=True)
                    torch.cuda.empty_cache()
                new_len = max(128, current_grad_max_len // 2)
                if new_len == current_grad_max_len:
                    raise
                current_grad_max_len = new_len
                if grad_pbar is not None:
                    grad_pbar.set_postfix_str(f"oom->seq_len={current_grad_max_len}")
                continue

            # unscaled loss for display (undo both count and accumulate division)
            loss_val = float(micro_loss.detach().item()) * n_accumulate

            # --- optimizer step every accumulation_steps micro-steps ---
            is_update = (i + 1) % n_accumulate == 0 or i == cfg.gradient_steps - 1
            if is_update:
                scaler.unscale_(model._optimizer)
                torch.nn.utils.clip_grad_norm_(
                    model._grad_params, model._genome.gradient.grad_clip_norm
                )
                scale_before = scaler.get_scale()
                scaler.step(model._optimizer)
                scaler.update()
                # Only advance the LR schedule when the optimizer actually ran.
                # GradScaler skips the step (and halves the scale) when it
                # detects inf/NaN gradients; calling scheduler.step() then
                # triggers a PyTorch warning and wastes a warmup tick.
                if scheduler is not None and scaler.get_scale() >= scale_before:
                    scheduler.step()
                model._optimizer.zero_grad()

            if avg_loss is None:
                avg_loss = loss_val
            else:
                avg_loss = 0.99 * avg_loss + 0.01 * loss_val

            if grad_pbar is not None:
                grad_pbar.set_postfix(loss=f"{loss_val:.4f}", avg=f"{avg_loss:.4f}")
                grad_pbar.update(1)
            elif i % 100 == 0:
                print(f"    grad step {i}: loss={loss_val:.4f}, avg={avg_loss:.4f}")

        if grad_pbar is not None:
            grad_pbar.close()
        if cfg.freeze_encoder:
            model.encoder.unfreeze()

    dt = time.perf_counter() - t0
    print(f"  Pretraining complete in {dt:.1f}s  ({len(sequences) / max(dt, 1e-6):.0f} seq/s)")
    print(f"  Encoder frozen: {model.encoder.is_frozen}")
    # report average loss to model for potential death detection
    if avg_loss is not None:
        model._record_session_loss(avg_loss)

    print(f"\n  Saving model state to {storage_dir} ...")
    model._save_state()
    print("  Saved.")

    print("\n  Sanity check - forward pass on 'hello world'...")
    sample_ids = tokenize("hello world", cfg.max_seq_len)
    if sample_ids is not None:
        structured = model.encoder.process(sample_ids)
        surprise = model.surprise_detector.score(structured, model.transformer.forward_readonly)
        print(f"    input tokens:  {int(sample_ids.numel())}")
        print(f"    embedding dim: {tuple(structured.embeddings.shape)}")
        print(f"    surprise score: {surprise:.4f}")
        print("    forward pass succeeded")

    print(f"\n  Model state saved to: {storage_dir}")
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
        description="Pretrain the Arnold model encoder on a HuggingFace dataset",
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
    g.add_argument("--gradient_steps", type=int, default=None, help="Extra gradient-based fine-tuning steps")
    g.add_argument("--gradient_lr", type=float, default=None, help="Gradient learning rate")
    g.add_argument("--gradient_batch_size", type=int, default=None, help="Gradient batch size")
    g.add_argument("--gradient_max_seq_len", type=int, default=None, help="Max tokens per sequence during gradient fine-tuning")
    g.add_argument("--freeze_encoder", action="store_true", default=None, help="Freeze encoder during gradient tuning")
    g.add_argument("--no_amp", dest="use_amp", action="store_false", default=None, help="Disable mixed-precision (AMP) training")
    g.add_argument("--accumulation_steps", type=int, default=None, help="Gradient accumulation steps before optimizer update (default: 4)")
    g.add_argument("--warmup_steps", type=int, default=None, help="LR warmup steps for cosine schedule (default: 100)")

    g = p.add_argument_group("Model dimensions")
    g.add_argument("--embed_dim", type=int, default=None)
    g.add_argument("--encoder_hidden", type=int, default=None)
    g.add_argument("--transformer_hidden", type=int, default=None)
    g.add_argument("--memory_capacity", type=int, default=None)
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
    _successful_exit()