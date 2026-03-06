#!/usr/bin/env python3
"""
Arnold Chat Server — FastAPI + SSE streaming interface to the Brain architecture.

User messages are treated as training data (Hebbian updates fire after each
response).  The model's own generated output is NEVER fed back as training data.

Usage
-----
  python chat_server.py
    python chat_server.py --storage_dir ./model/pretrained --port 7860
    python chat_server.py --storage_dir ./model/pretrained --embed_dim 128 --cortex_hidden 256

Then open http://localhost:7860 in your browser.

Required packages
-----------------
  pip install fastapi uvicorn[standard]
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time
from pathlib import Path
from typing import AsyncGenerator

import numpy as np

try:
    import torch
except ImportError:
    print("ERROR: 'torch' not installed.")
    print("  Install with:  pip install torch")
    sys.exit(1)

try:
    from fastapi import FastAPI, Request
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.responses import HTMLResponse, StreamingResponse
    from fastapi.staticfiles import StaticFiles
    from pydantic import BaseModel, Field
    import uvicorn
except ImportError:
    print("ERROR: 'fastapi' or 'uvicorn' not installed.")
    print("  Install with:  pip install fastapi 'uvicorn[standard]'")
    sys.exit(1)

_REPO = Path(__file__).parent
sys.path.insert(0, str(_REPO))

from model.genome import Genome, LayerTopology, HebbianParams, GenerationParams
from model.brain import Brain
from model.types_ import CortexRegion, PersonalityVector

STATIC_DIR = _REPO / "static"


# =============================================================================
# Global state
# =============================================================================

brain: Brain | None = None
gen_lock = asyncio.Lock()  # one generation at a time


# =============================================================================
# FastAPI app
# =============================================================================

app = FastAPI(title="Arnold Chat", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


# =============================================================================
# Schemas
# =============================================================================

class ChatRequest(BaseModel):
    message: str
    temperature: float = Field(default=0.8, ge=0.0, le=2.0)
    top_k: int = Field(default=50, ge=1, le=1000)
    top_p: float = Field(default=0.95, ge=0.0, le=1.0)
    max_tokens: int = Field(default=256, ge=1, le=2048)
    repetition_penalty: float = Field(default=1.1, ge=1.0, le=3.0)


class ModelInfoResponse(BaseModel):
    name: str
    parameters: int
    d_model: int
    num_layers: int
    device: str
    stage: str
    age: int
    session_active: bool
    personality: dict
    mood: dict
    neuromodulators: dict | None = None


# =============================================================================
# Routes
# =============================================================================

@app.get("/", response_class=HTMLResponse)
async def index():
    html_path = STATIC_DIR / "chat.html"
    if not html_path.exists():
        return HTMLResponse(
            "<h1>chat.html not found.</h1>"
            "<p>Run the server from the repo root with the static/ directory present.</p>",
            status_code=404,
        )
    return HTMLResponse(html_path.read_text())


@app.get("/api/info")
async def info():
    assert brain is not None
    status = brain.status()
    try:
        device = str(brain.brainstem.token_embeddings.device)
    except Exception:
        device = "cpu"
    return ModelInfoResponse(
        name="Arnold",
        parameters=brain.count_parameters(),
        d_model=brain._genome.topology.embed_dim,
        num_layers=len(brain._genome.topology.active_regions),
        device=device,
        stage=status["developmental_stage"],
        age=status["developmental_age"],
        session_active=status["session_active"],
        personality=status["personality"],
        mood=status["mood"],
        neuromodulators=status.get("neuromodulators"),
    )


@app.get("/api/status")
async def status():
    assert brain is not None
    return brain.status()


@app.post("/api/new_session")
async def new_session():
    """End the current session (triggers consolidation) and start a fresh one."""
    assert brain is not None
    async with gen_lock:
        report = None
        if brain._session_active:
            loop = asyncio.get_event_loop()
            report = await loop.run_in_executor(None, brain.session_end)
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, brain.session_start)

    consolidation_info = {}
    if report:
        consolidation_info = {
            "records_replayed": report.records_replayed,
            "connections_pruned": report.connections_pruned,
            "personality_deltas": report.personality_deltas or {},
        }

    return {
        "ok": True,
        "consolidation": consolidation_info,
        "age": brain.developmental_age,
        "stage": brain.developmental_stage,
    }


@app.post("/api/chat")
async def chat(req: ChatRequest):
    """Stream generated tokens as Server-Sent Events.

    Training flow:
      1. User message is processed — Hebbian traces recorded to hippocampus.
      2. Tokens are generated and streamed to the client.
      3. After generation: post-generation weight update fires based on the
         USER's input traces.  The generated output is NEVER used as training data.
    """
    return StreamingResponse(
        generate_sse(req),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


# =============================================================================
# SSE generation
# =============================================================================

async def generate_sse(req: ChatRequest) -> AsyncGenerator[str, None]:
    token_queue: asyncio.Queue[dict | None] = asyncio.Queue()

    def _generate() -> None:
        """Synchronous generation — runs in a thread executor."""
        assert brain is not None

        def emit(event: dict) -> None:
            token_queue.put_nowait(event)

        try:
            import math
            from collections import deque

            # ----------------------------------------------------------
            # Step 1: Process user input.
            # ----------------------------------------------------------
            logits, traces, novelty, reinforcement = brain.prepare_turn(req.message)

            # Get neuromodulator state for temperature modulation
            ne_value = None
            if brain.limbic is not None:
                ne_value = brain.limbic.neuromodulators.norepinephrine

            # ----------------------------------------------------------
            # Step 2: Token-by-token generation with full quality pipeline.
            # Uses recurrent hidden state, context buffer attention,
            # repetition penalty, entropy-adaptive + top-k + top-p sampling,
            # and soft prompt anchoring.
            # ----------------------------------------------------------
            gen_params = brain._genome.generation
            generated_tokens: list[int] = []
            if isinstance(logits, torch.Tensor):
                current_logits = logits.detach().cpu()
            else:
                current_logits = torch.as_tensor(logits)
            vocab_size = current_logits.shape[0]

            bs = brain.brainstem
            hidden_dim = bs.cooccurrence_weights.shape[1]
            hidden_state = torch.zeros(hidden_dim, dtype=current_logits.dtype)
            ctx_buf: deque[torch.Tensor] = deque(maxlen=gen_params.context_buffer_size)

            # Soft prompt anchor from input embeddings
            anchor: torch.Tensor | None = None
            if brain._last_structured is not None:
                mean_emb = brain._last_structured.embeddings.mean(dim=0).cpu()
                anchor = bs.token_embeddings.cpu() @ mean_emb

            # Recurrent weight from cortex
            W_rec: torch.Tensor | None = None
            for region in brain._genome.topology.active_regions:
                if region in brain.cortex.regions:
                    W_rec = brain.cortex.regions[region].W_recurrent.cpu()
                    break

            t0 = time.perf_counter()

            for _ in range(req.max_tokens):
                # Apply anchor
                step_logits = current_logits.clone()
                if anchor is not None:
                    step_logits = step_logits + gen_params.anchor_weight * anchor

                # --- Repetition penalty ---
                if gen_params.repetition_penalty != 1.0 and generated_tokens:
                    for t_id in set(generated_tokens):
                        if 0 <= t_id < vocab_size:
                            if step_logits[t_id] > 0:
                                step_logits[t_id] /= gen_params.repetition_penalty
                            else:
                                step_logits[t_id] *= gen_params.repetition_penalty

                # Also apply server-level rep penalty on top
                lgt_np = step_logits.numpy()
                # Modulate temperature by norepinephrine: high NE → more exploration
                effective_temp = req.temperature
                if ne_value is not None:
                    effective_temp += (ne_value - 0.3) * 0.5
                    effective_temp = max(0.1, min(effective_temp, 2.0))
                tok_id = _sample(
                    lgt_np,
                    generated_tokens,
                    effective_temp,
                    req.top_k,
                    req.top_p,
                    req.repetition_penalty,
                    np.random.default_rng(),
                )

                generated_tokens.append(tok_id)

                # Decode and stream immediately
                text = _decode_token(tok_id)
                emit({"token": text})

                # --- Recurrent hidden state update ---
                emb = bs.token_embeddings[tok_id].cpu()
                co_hidden = torch.relu(emb @ bs.cooccurrence_weights.cpu())

                rec_input = co_hidden
                if W_rec is not None:
                    rec_input = gen_params.recurrent_mix * (hidden_state @ W_rec) + (1.0 - gen_params.recurrent_mix) * co_hidden
                hidden_state = torch.tanh(rec_input)

                # Context buffer attention
                ctx_buf.append(hidden_state.detach().clone())
                if len(ctx_buf) > 1:
                    stack = torch.stack(list(ctx_buf))
                    scores = torch.softmax(hidden_state @ stack.T / math.sqrt(hidden_dim), dim=0)
                    context_vec = scores @ stack
                    hidden_state = hidden_state + gen_params.context_attention_weight * context_vec

                # Produce next logits from recurrent hidden state
                recon = hidden_state @ bs.output_projection.cpu()
                next_signal = bs.token_embeddings.cpu() @ recon
                current_logits = 0.50 * current_logits + 0.50 * next_signal

            dt = time.perf_counter() - t0
            tok_s = len(generated_tokens) / max(dt, 1e-6)

            # ----------------------------------------------------------
            # Step 3: Post-generation weight update.
            # Fires after response is complete.
            # Signal source = user's input (novelty + reinforcement).
            # Model output is discarded for training purposes.
            # ----------------------------------------------------------
            brain.finalize_turn(generated_tokens, traces, novelty, reinforcement)

            emit({
                "done": True,
                "tokens": len(generated_tokens),
                "time_s": round(dt, 2),
                "tok_per_s": round(tok_s, 1),
                "novelty": round(novelty, 4),
                "reinforcement": round(reinforcement, 4),
                "age": brain.developmental_age,
                "stage": brain.developmental_stage,
                "neuromodulators": (
                    {
                        "dopamine": round(brain.limbic.neuromodulators.dopamine, 4),
                        "serotonin": round(brain.limbic.neuromodulators.serotonin, 4),
                        "acetylcholine": round(brain.limbic.neuromodulators.acetylcholine, 4),
                        "norepinephrine": round(brain.limbic.neuromodulators.norepinephrine, 4),
                    }
                    if brain.limbic else None
                ),
            })

        except Exception as exc:
            emit({"error": str(exc)})
        finally:
            token_queue.put_nowait(None)

    async with gen_lock:
        loop = asyncio.get_event_loop()
        gen_task = loop.run_in_executor(None, _generate)

        while True:
            try:
                event = await asyncio.wait_for(token_queue.get(), timeout=30.0)
            except asyncio.TimeoutError:
                yield f"data: {json.dumps({'error': 'Generation timed out'})}\n\n"
                break

            if event is None:
                break

            yield f"data: {json.dumps(event)}\n\n"

        await gen_task


# =============================================================================
# Sampling helpers
# =============================================================================

def _sample(
    logits: np.ndarray,
    seen: list[int],
    temperature: float,
    top_k: int,
    top_p: float,
    rep_penalty: float,
    rng: np.random.Generator,
) -> int:
    """Apply repetition penalty, temperature, top-k, top-p, then sample."""
    lgt = logits.copy()

    # Repetition penalty
    if rep_penalty != 1.0 and seen:
        unique_seen = list(set(seen))
        for t in unique_seen:
            if 0 <= t < len(lgt):
                lgt[t] /= rep_penalty

    # Temperature
    temp = max(temperature, 1e-8)
    lgt = lgt / temp

    # Top-k
    if top_k > 0 and top_k < len(lgt):
        threshold = np.sort(lgt)[::-1][top_k - 1]
        lgt[lgt < threshold] = -np.inf

    # Softmax
    lgt -= np.max(lgt)
    probs = np.exp(lgt)
    probs_sum = probs.sum()
    if probs_sum == 0:
        probs = np.ones(len(lgt)) / len(lgt)
    else:
        probs /= probs_sum

    # Top-p (nucleus)
    if top_p < 1.0:
        sorted_idx = np.argsort(probs)[::-1]
        cumprobs = np.cumsum(probs[sorted_idx])
        cutoff = np.searchsorted(cumprobs, top_p) + 1
        keep = sorted_idx[:cutoff]
        mask = np.zeros(len(probs), dtype=bool)
        mask[keep] = True
        probs[~mask] = 0.0
        total = probs.sum()
        if total > 0:
            probs /= total
        else:
            probs = np.ones(len(lgt)) / len(lgt)

    return int(rng.choice(len(probs), p=probs))


def _decode_token(token_id: int) -> str:
    """Decode token ID using the active brain tokenizer when available."""
    if brain is not None:
        return brain._detokenize([int(token_id)])
    b = min(token_id, 255)
    return bytes([b]).decode("utf-8", errors="replace")


# =============================================================================
# Brain startup
# =============================================================================

def build_genome(args: argparse.Namespace, vocab_size: int) -> Genome:
    from model.genome import (
        LayerTopology, HebbianParams, GenerationParams,
        PlasticityParams, ConsolidationParams, ReinforcementParams,
        NoveltyParams, MoodParams, PersonalityParams, EWCParams,
        DevelopmentalBoundaries, VotingParams,
    )
    topo = LayerTopology(
        vocab_size=vocab_size,
        embed_dim=args.embed_dim,
        brainstem_hidden=args.brainstem_hidden,
        cortex_hidden=args.cortex_hidden,
        hippocampus_capacity=args.hippocampus_capacity,
    )
    return Genome(
        topology=topo,
        hebbian=HebbianParams(learning_rate=args.lr),
        generation=GenerationParams(brainstem_pretrain_steps=500),
    )


def load_or_create_brain(args: argparse.Namespace) -> Brain:
    storage_dir = Path(args.storage_dir)
    storage_dir.mkdir(parents=True, exist_ok=True)

    # Optional tokenizer
    tokenizer = None
    if getattr(args, 'tokenizer_name', None):
        if args.tokenizer_name.strip() != "":
            try:
                from transformers import AutoTokenizer
            except Exception:
                print("ERROR: 'transformers' not installed. Install with: pip install transformers")
                raise
            tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name, use_fast=True)

    # Read saved topology first so CLI defaults never shadow a pretrained model.
    from model.weight_store import WeightStore
    saved_topo = WeightStore(storage_dir).load_topology()

    if args.vocab_size is not None:
        resolved_vocab_size = int(args.vocab_size)
    elif saved_topo is not None:
        resolved_vocab_size = saved_topo.vocab_size
    elif tokenizer is not None:
        try:
            resolved_vocab_size = int(len(tokenizer))
        except Exception:
            resolved_vocab_size = int(getattr(tokenizer, "vocab_size", 30000))
    else:
        resolved_vocab_size = 30000

    # Override CLI dimension args with the saved topology when present,
    # so the Brain is always built with the topology it was trained on.
    if saved_topo is not None:
        args.embed_dim = saved_topo.embed_dim
        args.brainstem_hidden = saved_topo.brainstem_hidden
        args.cortex_hidden = saved_topo.cortex_hidden
        args.hippocampus_capacity = saved_topo.hippocampus_capacity

    genome = build_genome(args, resolved_vocab_size)

    try:
        b = Brain(genome=genome, storage_dir=storage_dir, seed=args.seed, tokenizer=tokenizer)
    except Exception as exc:
        print("Saved brain state appears incompatible with the requested genome:")
        print(" ", str(exc))
        print("Removing existing saved state and initialising a fresh brain with the new topology.")
        # Attempt to remove saved files and re-create
        from model.weight_store import WeightStore
        ws = WeightStore(storage_dir)
        ws.delete()
        b = Brain(genome=genome, storage_dir=storage_dir, seed=args.seed, tokenizer=tokenizer)

    if not b.weight_store.exists():
        print("  No saved state found — initialising a fresh brain (brainstem will be random + trainable).")
        b.brainstem.unfreeze()
    else:
        print(f"  Loaded brain state from {storage_dir}")
        print(f"  Brainstem frozen: {b.brainstem.is_frozen}")
        print(f"  Age: {b.developmental_age} interactions")
        print(f"  Stage: {b.developmental_stage}")

    b.session_start()
    print(f"  Session started.")
    return b


# =============================================================================
# CLI
# =============================================================================

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Arnold web chat server",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Storage
    p.add_argument("--storage_dir", default="./model/pretrained",
                   help="Directory with the saved brain state")
    p.add_argument("--tokenizer_name", type=str, default="gpt2",
                   help="HuggingFace tokenizer name to use (e.g. 'gpt2'). Use empty string to disable.)")

    # Model dimensions (only matter if no saved state exists)
    p.add_argument("--vocab_size", type=int, default=None,
                   help="Override vocabulary size. Defaults to tokenizer vocab size when tokenizer is enabled.")
    p.add_argument("--embed_dim", type=int, default=128)
    p.add_argument("--brainstem_hidden", type=int, default=256)
    p.add_argument("--cortex_hidden", type=int, default=256)
    p.add_argument("--hippocampus_capacity", type=int, default=4096)
    p.add_argument("--lr", type=float, default=0.01)

    # Server
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument("--port", type=int, default=7860)
    p.add_argument("--seed", type=int, default=42)

    return p.parse_args()


def main() -> None:
    global brain

    args = parse_args()

    print("\n" + "=" * 64)
    print("  ARNOLD CHAT SERVER")
    print("=" * 64)

    brain = load_or_create_brain(args)

    n = brain.count_parameters()
    print(f"\n  Parameters: {n:,}")
    print(f"  Storage:    {args.storage_dir}")
    print(f"\n  Open http://localhost:{args.port} in your browser.")
    print(f"  User messages train the model. Generated output does not.\n")

    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")

    # Graceful shutdown: end session and save
    if brain._session_active:
        print("\nShutting down — saving brain state...")
        brain.session_end()
        print("Saved.")


if __name__ == "__main__":
    main()
