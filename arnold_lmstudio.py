#!/usr/bin/env python3
"""
arnold_lmstudio.py — Autonomous self-supervised training via LM Studio conversation.

Arnold (Hebbian model) talks to an LM Studio model over the OpenAI-compatible API.
Each exchange trains Arnold: LM Studio's responses become Arnold's training data.

Flow per turn:
  1. Arnold's text  →  LM Studio teacher (OpenAI-compatible /v1/chat/completions)
  2. LM Studio returns structured feedback with `correction` + `reinforcement`
  3. Arnold trains on the corrected text using the teacher's numeric reward
  4. Arnold generates a new reply; if it is poor, the next teacher input falls
      back to the corrected text instead of recursively feeding garbage
  5. Repeat

Every --save_every turns the session is checkpointed:
  session_end() → three-stage sleep consolidation → save → session_start()

Usage
-----
  python arnold_lmstudio.py
  python arnold_lmstudio.py --lm_url http://localhost:1234 --model "llama-3b"
  python arnold_lmstudio.py --turns 500 --seed_topic "Tell me about perception"
  python arnold_lmstudio.py --system_prompt "You are Socrates. Ask probing questions."
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import re
import signal
import sys
import time
from pathlib import Path

# Prevent triton segfault (see pretrain.py for full explanation).
sys.modules["triton"] = None

try:
    import requests
except ImportError:
    print("ERROR: 'requests' not installed.  Install with:  pip install requests")
    sys.exit(1)

_REPO = Path(__file__).parent
sys.path.insert(0, str(_REPO))

from model.model import Model
from model.genome import Genome, LayerTopology
from model.weight_store import WeightStore


DEFAULT_TEACHER_PROMPT = """You are an expert conversation teacher whose sole job is to produce a concise, machine-parseable feedback block for a downstream Hebbian model. You MUST follow these rules exactly.
Response MUST start with a single line exactly: ===FEEDBACK===
Respond ONLY with the fields below, one per line, using the exact field names and formatting. Do NOT add any extra commentary, explanation, JSON, code fences, or markup.
Numeric fields must use two decimal places. Lists must be comma-separated items on a single line.
If you detect unsafe content, set reinforcement: -1.00 and list the safety issue in issues.
Required fields and formatting (each on its own line):
coherence: 0.00-1.00
surprise: 0.00-1.00
reinforcement: -1.00..+1.00
issues: short phrase, short phrase, ...
correction: concise corrected rewrite (<=200 characters)
tokens_to_reinforce: tokenA, tokenB, ...
tokens_to_deprioritize: tokenX, tokenY, ...
why: one short sentence (<=120 chars)
Operational rules:
The input to evaluate will arrive after a line that reads exactly ===INPUT===. Only evaluate that text.
Be concise and objective. Rate coherence high for grammatical, clear meaning. Rate surprise high for unexpected but valid phrasing. Use reinforcement positive for useful, grammatical, factual content; negative for hallucination, nonsense, or unsafe content.
If you cannot parse the input, output coherence: 0.00, surprise: 0.00, reinforcement: -1.00, issues: unparsable input, correction: Please rewrite as one short grammatical sentence., empty token lists, and a why explaining parsing failure."""


# ---------------------------------------------------------------------------
# Curriculum: phase-appropriate seed sentences used when Arnold's output is
# too noisy to be a useful next-turn driver.
#
# Phase 0 — simple declarative sentences (easiest pattern for blank weights)
# Phase 1 — Q&A pairs (richer structure; used once coherence stabilises)
# Phase 2 — open conversation (default behaviour; curriculum exits here)
# ---------------------------------------------------------------------------
_CURRICULUM_SEEDS = [
    # Phase 0
    [
        "The sun rises in the east and sets in the west every day.",
        "Dogs are loyal animals that have lived alongside humans for thousands of years.",
        "Clean water is essential for all living things on Earth.",
        "Reading books expands vocabulary and improves critical thinking skills.",
        "Exercise strengthens the heart and helps maintain a healthy body weight.",
        "The moon orbits the Earth approximately once every twenty-seven days.",
        "Trees produce oxygen through photosynthesis using sunlight and carbon dioxide.",
        "A balanced diet includes vegetables, fruits, proteins, and whole grains.",
        "Learning a new language opens doors to different cultures and perspectives.",
        "Sleep is necessary for the model to consolidate memories and repair cells.",
    ],
    # Phase 1
    [
        "Question: What is gravity? Answer: Gravity is the force that pulls objects with mass toward one another.",
        "Question: How do plants make food? Answer: Plants use sunlight, water, and carbon dioxide to make sugar through photosynthesis.",
        "Question: What is the largest planet in our solar system? Answer: Jupiter is the largest planet, larger than all others combined.",
        "Question: Why is the sky blue? Answer: The sky appears blue because the atmosphere scatters short blue wavelengths of sunlight more than other colours.",
        "Question: What causes the seasons? Answer: Earth's tilted axis causes different parts to receive more sunlight at different times of year.",
    ],
]
# Coherence EMA thresholds at which to advance to the next phase
_CURRICULUM_ADVANCE_THRESHOLDS = [0.45, 0.55]  # phase 0→1, phase 1→2
_CURRICULUM_TURNS_REQUIRED = 10  # sustained turns above threshold before advancing


@dataclass
class TeacherFeedback:
    coherence: float
    surprise: float
    reinforcement: float
    correction: str
    issues: list[str]


def _parse_float(value: str) -> float | None:
    match = re.search(r"[-+]?\d+(?:\.\d+)?", value)
    if match is None:
        return None
    try:
        return float(match.group(0))
    except ValueError:
        return None


def _strip_code_fences(text: str) -> str:
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = re.sub(r"^```[a-zA-Z0-9_-]*\n?", "", stripped)
        stripped = re.sub(r"\n?```$", "", stripped)
    return stripped.strip()


def parse_teacher_feedback(text: str) -> TeacherFeedback | None:
    cleaned = _strip_code_fences(text)

    if cleaned.startswith("{"):
        try:
            payload = json.loads(cleaned)
        except json.JSONDecodeError:
            payload = None
        if isinstance(payload, dict):
            correction = str(payload.get("correction", "")).strip()
            reinforcement = _parse_float(str(payload.get("reinforcement", "")))
            coherence = _parse_float(str(payload.get("coherence", "")))
            surprise = _parse_float(str(payload.get("surprise", "")))
            if correction and reinforcement is not None:
                return TeacherFeedback(
                    coherence=max(min(coherence if coherence is not None else 0.0, 1.0), 0.0),
                    surprise=max(min(surprise if surprise is not None else 0.0, 1.0), 0.0),
                    reinforcement=max(min(reinforcement, 1.0), -1.0),
                    correction=correction,
                    issues=[str(item).strip() for item in payload.get("issues", []) if str(item).strip()],
                )

    lines = [line.strip() for line in cleaned.splitlines() if line.strip()]
    if lines and lines[0] == "===FEEDBACK===":
        lines = lines[1:]

    fields: dict[str, str] = {}
    for line in lines:
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        key = key.strip().lower()
        if key:
            fields[key] = value.strip()

    correction = fields.get("correction", "").strip()
    reinforcement = _parse_float(fields.get("reinforcement", ""))
    coherence = _parse_float(fields.get("coherence", ""))
    surprise = _parse_float(fields.get("surprise", ""))
    if not correction or reinforcement is None:
        return None

    return TeacherFeedback(
        coherence=max(min(coherence if coherence is not None else 0.0, 1.0), 0.0),
        surprise=max(min(surprise if surprise is not None else 0.0, 1.0), 0.0),
        reinforcement=max(min(reinforcement, 1.0), -1.0),
        correction=correction,
        issues=[item.strip() for item in fields.get("issues", "").split(",") if item.strip()],
    )


def _is_usable_driver_text(text: str) -> bool:
    cleaned = text.strip()
    if not cleaned:
        return False
    if "\ufffd" in cleaned or "�" in cleaned:
        return False

    words = cleaned.split()
    if len(words) < 3:
        return True

    counts: dict[str, int] = {}
    for word in words:
        counts[word] = counts.get(word, 0) + 1

    max_repeat_ratio = max(counts.values()) / max(len(words), 1)
    unique_ratio = len(counts) / max(len(words), 1)
    weird_chars = sum(
        1 for ch in cleaned
        if not (ch.isalnum() or ch.isspace() or ch in ".,?!'\"-:;()")
    )
    weird_ratio = weird_chars / max(len(cleaned), 1)
    return max_repeat_ratio < 0.28 and unique_ratio > 0.45 and weird_ratio < 0.08


# =============================================================================
# LM Studio API
# =============================================================================

def lm_chat(
    url: str,
    model: str,
    messages: list[dict],
    temperature: float = 0.8,
    max_tokens: int = 512,
    timeout: int = 60,
) -> str:
    """Send a chat request to an OpenAI-compatible /v1/chat/completions endpoint."""
    endpoint = f"{url.rstrip('/')}/v1/chat/completions"
    payload = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "stream": False,
    }
    resp = requests.post(endpoint, json=payload, timeout=timeout)
    resp.raise_for_status()
    return resp.json()["choices"][0]["message"]["content"].strip()


def detect_model(url: str, timeout: int = 10) -> list[str]:
    """Return list of loaded model IDs from LM Studio."""
    resp = requests.get(f"{url.rstrip('/')}/v1/models", timeout=timeout)
    resp.raise_for_status()
    return [m["id"] for m in resp.json().get("data", [])]


# =============================================================================
# Arnold model loader
# =============================================================================

def load_model(args: argparse.Namespace) -> Model:
    storage_dir = Path(args.storage_dir)
    storage_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = None
    if args.tokenizer_name.strip():
        try:
            from transformers import AutoTokenizer
        except ImportError:
            print("ERROR: 'transformers' not installed.  pip install transformers")
            raise
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name, use_fast=True)

    ws = WeightStore(storage_dir)
    saved_topo = ws.load_topology()

    if saved_topo is not None:
        genome = Genome(topology=saved_topo)
        print(f"  Topology from saved state: vocab={saved_topo.vocab_size} "
              f"embed={saved_topo.embed_dim} bs_h={saved_topo.encoder_hidden} "
              f"cx_h={saved_topo.transformer_hidden}")
    else:
        vocab_size = args.vocab_size or (len(tokenizer) if tokenizer else 50257)
        topo = LayerTopology(
            vocab_size=vocab_size,
            embed_dim=args.embed_dim,
            encoder_hidden=args.encoder_hidden,
            transformer_hidden=args.transformer_hidden,
        )
        genome = Genome(topology=topo)
        print(f"  No saved topology — using defaults (vocab={vocab_size})")

    model = Model(
        genome=genome,
        storage_dir=storage_dir,
        seed=args.seed,
        tokenizer=tokenizer,
    )

    if not ws.exists():
        model.encoder.unfreeze()
        print("  Fresh model — encoder trainable with random weights")
    else:
        print(f"  Loaded model state from {storage_dir}")
        print(f"  Age: {model.developmental_age} interactions  |  "
              f"Stage: {model.developmental_stage}")
        print(f"  Encoder frozen: {model.encoder.is_frozen}")

    return model


# =============================================================================
# Main loop
# =============================================================================

def run(args: argparse.Namespace) -> None:
    print("\n" + "=" * 64)
    print("  ARNOLD ↔ LM STUDIO — AUTONOMOUS TRAINING LOOP")
    print("=" * 64)

    # --- Verify LM Studio is reachable ---
    print(f"\n  Connecting to LM Studio at {args.lm_url} ...")
    try:
        available = detect_model(args.lm_url, timeout=10)
        print(f"  Available models: {available or '(none listed)'}")
        if not args.model:
            if not available:
                print("  ERROR: no models detected.  Load a model in LM Studio first.")
                sys.exit(1)
            args.model = available[0]
        print(f"  Using model: {args.model}")
    except Exception as exc:
        print(f"  ERROR: could not reach LM Studio: {exc}")
        print("  Make sure LM Studio is running with the local server enabled.")
        sys.exit(1)

    # --- Load Arnold ---
    print()
    model = load_model(args)
    model.session_start()
    print(f"  Parameters: {model.count_parameters():,}")

    # --- System prompt for LM Studio ---
    system_prompt = args.system_prompt or DEFAULT_TEACHER_PROMPT

    # --- Seed message ---
    seed = args.seed_topic or (
        "Hello! I'm Arnold, a Hebbian associative AI. "
        "I learn through conversation — each exchange actually trains my weights. "
        "What's an idea you find genuinely fascinating?"
    )

    print(f"\n  Turns: {args.turns}  |  Save every: {args.save_every} turns")
    print(f"  Seed:  {seed[:100]}{'...' if len(seed) > 100 else ''}")
    print("=" * 64)

    arnold_text = seed
    turn = 0
    total_tokens_learned = 0

    # Curriculum state (only meaningful when args.curriculum is True)
    _curr_phase: int = 0
    _phase_coherence_ema: float = 0.0
    _phase_turns_ok: int = 0
    if args.curriculum:
        print(f"  Curriculum mode ON — Phase 0 (simple sentences)")

    # Graceful Ctrl+C handler
    shutdown_requested = False
    def _handle_sigint(sig, frame):
        nonlocal shutdown_requested
        if shutdown_requested:
            print("\n  Force quit.")
            sys.exit(1)
        print("\n\n  Ctrl+C received — finishing current turn then shutting down...")
        shutdown_requested = True
    signal.signal(signal.SIGINT, _handle_sigint)

    try:
        while turn < args.turns and not shutdown_requested:
            turn += 1
            print(f"\n── Turn {turn}/{args.turns} {'─' * (50 - len(str(turn)))}")

            # ------------------------------------------------------------------
            # Step 1: Send Arnold's message to LM Studio
            # ------------------------------------------------------------------
            print(f"  Arnold → LM │ {arnold_text[:140]}{'…' if len(arnold_text) > 140 else ''}")
            request_messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": "===INPUT===\n" + arnold_text},
            ]

            try:
                lm_response = lm_chat(
                    url=args.lm_url,
                    model=args.model,
                    messages=request_messages,
                    temperature=args.lm_temperature,
                    max_tokens=args.lm_max_tokens,
                    timeout=args.timeout,
                )
            except requests.exceptions.Timeout:
                print(f"  WARN: LM Studio timed out after {args.timeout}s — skipping turn")
                continue
            except Exception as exc:
                print(f"  ERROR: LM Studio call failed: {exc}")
                break

            print(f"  LM → Arnold │ {lm_response[:140]}{'…' if len(lm_response) > 140 else ''}")

            feedback = parse_teacher_feedback(lm_response)
            if feedback is None:
                print("  Teacher │ unparseable feedback, falling back to raw response")
                training_text = lm_response.strip()
                reinforcement_override = None
                correction_text = ""
            else:
                training_text = feedback.correction.strip()
                reinforcement_override = feedback.reinforcement
                correction_text = feedback.correction.strip()
                print(
                    f"  Teacher │ coh={feedback.coherence:.2f} nov={feedback.surprise:.2f} "
                    f"reinf={feedback.reinforcement:+.2f} correction={training_text[:96]}"
                    f"{'…' if len(training_text) > 96 else ''}"
                )

            # ------------------------------------------------------------------
            # Step 2: Arnold trains on the teacher's corrected target text.
            # ------------------------------------------------------------------
            result = model.process_turn(
                training_text,
                external_reinforcement=reinforcement_override,
            )

            generated_text = model._detokenize(result.generated_tokens).strip()
            if _is_usable_driver_text(generated_text):
                arnold_text = generated_text
            elif args.curriculum and _curr_phase < 2:
                # Use a curriculum seed rather than feeding back the LM correction,
                # which often contains meta-commentary ("The input is repetitive…").
                seed_list = _CURRICULUM_SEEDS[_curr_phase]
                arnold_text = seed_list[(turn - 1) % len(seed_list)]
                print(f"  Curriculum │ Phase {_curr_phase} seed override")
            else:
                arnold_text = correction_text or training_text or "Describe one idea in a single grammatical sentence."
                print("  Driver │ using corrected text for next turn")

            # Curriculum phase advancement: track rolling coherence and advance
            # when it stays above the threshold for enough consecutive turns.
            if args.curriculum and feedback is not None:
                _phase_coherence_ema = 0.85 * _phase_coherence_ema + 0.15 * feedback.coherence
                threshold = _CURRICULUM_ADVANCE_THRESHOLDS[_curr_phase] if _curr_phase < len(_CURRICULUM_ADVANCE_THRESHOLDS) else 1.0
                if _phase_coherence_ema >= threshold:
                    _phase_turns_ok += 1
                    if _phase_turns_ok >= _CURRICULUM_TURNS_REQUIRED and _curr_phase < 2:
                        _curr_phase += 1
                        _phase_turns_ok = 0
                        phase_name = ["simple sentences", "Q&A pairs", "open chat"][_curr_phase]
                        print(f"\n  [Curriculum] Coherence EMA={_phase_coherence_ema:.2f} \u2192 Phase {_curr_phase} ({phase_name})\n")
                else:
                    _phase_turns_ok = 0

            total_tokens_learned += len(training_text.split())

            # Print neuromodulator / learning stats
            nm = result.neuromodulators
            if nm:
                print(
                    f"  Model │ surprise={result.surprise_score:.3f} "
                    f"reinf={result.reinforcement_strength:+.3f} "
                    f"DA={nm.dopamine:+.2f} 5HT={nm.serotonin:.2f} "
                    f"ACh={nm.acetylcholine:.2f} NE={nm.norepinephrine:.2f} "
                    f"age={model.developmental_age}"
                )

            # ------------------------------------------------------------------
            # Step 3: Periodic session checkpoint (sleep consolidation + save)
            # ------------------------------------------------------------------
            if turn % args.save_every == 0:
                print(f"\n  ── Checkpoint at turn {turn} ──")
                report = model.session_end()
                if report:
                    extras = []
                    if report.records_replayed:
                        extras.append(f"replayed={report.records_replayed}")
                    if report.connections_pruned:
                        extras.append(f"pruned={report.connections_pruned}")
                    if report.global_downscale_applied:
                        extras.append("downscaled")
                    if getattr(report, "rem_integrations", 0):
                        extras.append(f"rem={report.rem_integrations}")
                    print(f"  Consolidation: {', '.join(extras) if extras else 'complete'}")
                print(f"  Saved  (age={model.developmental_age}  "
                      f"tokens_seen≈{total_tokens_learned:,})")
                model.session_start()

            if args.delay > 0 and not shutdown_requested:
                time.sleep(args.delay)

    finally:
        print("\n" + "=" * 64)
        print("  Ending session — running sleep consolidation and saving...")
        try:
            report = model.session_end()
            if report and report.records_replayed:
                print(f"  Consolidation: {report.records_replayed} records replayed")
        except Exception as exc:
            print(f"  Warning: consolidation error: {exc}")
            model._save_state()

        print(f"  Final age:        {model.developmental_age} interactions")
        print(f"  Tokens seen ≈     {total_tokens_learned:,}")
        print(f"  Turns completed:  {turn}/{args.turns}")
        status = model.status()
        nm = status.get("neuromodulators", {})
        if nm:
            print(f"  Neuromodulators:  DA={nm['dopamine']:+.3f}  "
                  f"5HT={nm['serotonin']:.3f}  "
                  f"ACh={nm['acetylcholine']:.3f}  "
                  f"NE={nm['norepinephrine']:.3f}")
        print("=" * 64)


# =============================================================================
# CLI
# =============================================================================

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Arnold ↔ LM Studio autonomous conversation training",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    g = p.add_argument_group("LM Studio")
    g.add_argument("--lm_url", default="http://localhost:1234",
                   help="LM Studio server base URL")
    g.add_argument("--model", default="",
                   help="Model ID to use (auto-detects first loaded model if empty)")
    g.add_argument("--lm_temperature", type=float, default=0.0,
                   help="Sampling temperature for LM Studio responses (0.0 for deterministic feedback)")
    g.add_argument("--lm_max_tokens", type=int, default=240,
                   help="Max tokens per LM Studio response")
    g.add_argument("--system_prompt", default=None,
                   help="System prompt to guide LM Studio's conversational style")
    g.add_argument("--timeout", type=int, default=60,
                   help="HTTP request timeout in seconds")

    g = p.add_argument_group("Conversation")
    g.add_argument("--turns", type=int, default=200,
                   help="Total number of conversation turns to run")
    g.add_argument("--seed_topic", default=None,
                   help="Opening message Arnold sends to start the conversation")
    g.add_argument("--delay", type=float, default=0.5,
                   help="Seconds to pause between turns (avoid hammering the API)")
    g.add_argument("--max_history", type=int, default=20,
                   help="Max exchanges to keep in LM Studio's context window")
    g.add_argument("--curriculum", action="store_true", default=False,
                   help=(
                       "Enable curriculum training. Phases Arnold through simple "
                       "sentences → Q&A → open chat, advancing automatically once "
                       "the teacher's coherence score stays above threshold."
                   ))

    g = p.add_argument_group("Arnold model")
    g.add_argument("--storage_dir", default="./model/pretrained",
                   help="Directory containing Arnold's saved model state")
    g.add_argument("--tokenizer_name", default="gpt2",
                   help="HuggingFace tokenizer name (must match pretraining)")
    g.add_argument("--vocab_size", type=int, default=None,
                   help="Override vocab size (ignored when saved state exists)")
    g.add_argument("--embed_dim", type=int, default=448)
    g.add_argument("--encoder_hidden", type=int, default=896)
    g.add_argument("--transformer_hidden", type=int, default=896)
    g.add_argument("--seed", type=int, default=42)
    g.add_argument("--save_every", type=int, default=50,
                   help="Run sleep consolidation and save every N turns")

    return p.parse_args()


if __name__ == "__main__":
    run(parse_args())
