"""Main entry point — demonstrates the full neural architecture lifecycle.

  1. Birth: create genome, pretrain encoder
  2. Session: process multiple turns, watch mood/personality shift
  3. Consolidation: end session, run offline learning
  4. Persistence: save state, reload, verify continuity
"""

from __future__ import annotations

import shutil
import sys
import tempfile

# Prevent triton segfault (see pretrain.py for full explanation).
sys.modules["triton"] = None

import torch

from model.genome import Genome
from model.model import Model


def generate_pretraining_corpus(
    genome: Genome, n_sequences: int = 500, rng: object | None = None,
) -> list[torch.Tensor]:
    """Generate synthetic token sequences for encoder pretraining."""
    g = torch.Generator()
    g.manual_seed(0)
    vocab = genome.topology.vocab_size
    sequences = []
    for _ in range(n_sequences):
        length = int(torch.randint(10, 128, (1,), generator=g).item())
        seq = torch.randint(0, vocab, (length,), dtype=torch.long, generator=g)
        sequences.append(seq)
    return sequences


def print_status(model: Model, label: str = "") -> None:
    """Pretty-print model status."""
    status = model.status()
    print(f"\n{'=' * 60}")
    if label:
        print(f"  {label}")
        print(f"{'=' * 60}")
    print(f"  Stage:          {status['developmental_stage']}")
    print(f"  Age:            {status['developmental_age']} interactions")
    print(f"  Session active: {status['session_active']}")
    print(f"  Encoder:      {'frozen' if status['encoder_frozen'] else 'LIVE'}")

    print(f"\n  Personality:")
    for trait, val in status["personality"].items():
        bar = "█" * int(val * 20) + "░" * (20 - int(val * 20))
        print(f"    {trait:15s} {bar} {val:.4f}")

    print(f"\n  Mood:")
    for dim, val in status["mood"].items():
        print(f"    {dim:10s} {val:+.4f}")

    print(f"\n  Region weights:")
    for region, weight in status["region_weights"].items():
        print(f"    {region:10s} {weight:.4f}")

    print(f"\n  Plasticity:")
    plast = status["plasticity"]
    for region, rate in plast["per_region_plasticity"].items():
        print(f"    {region:10s} {rate:.6f}")

    print(f"\n  Memory: {status['memory_utilization']:.1%} full")
    print(f"  Consolidations: {status['consolidation_meta'].get('total_consolidations', 0)}")

    highways = status.get("inter_region_highways", {})
    if highways:
        print(f"\n  Inter-region highways:")
        for path, strength in highways.items():
            print(f"    {path:20s} {strength:.6f}")
    print()


def main() -> None:
    # Use a temp directory for this demo
    storage_dir = tempfile.mkdtemp(prefix="model_")
    print(f"Storage: {storage_dir}")

    # --- Phase 1: Birth ---
    print("\n" + "=" * 60)
    print("  PHASE 1: BIRTH")
    print("=" * 60)

    genome = Genome()
    model = Model(genome=genome, storage_dir=storage_dir, seed=42)

    # Generate synthetic pretraining data and pretrain encoder
    corpus = generate_pretraining_corpus(genome, n_sequences=500)
    model.birth(pretraining_corpus=corpus)
    print(f"  Encoder pretrained on {len(corpus)} sequences")
    print(f"  Encoder frozen: {model.encoder.is_frozen}")

    print_status(model, "After birth")

    # --- Phase 2: First session ---
    print("=" * 60)
    print("  PHASE 2: FIRST SESSION")
    print("=" * 60)

    model.session_start()

    conversation = [
        "Hello! I'm excited to start learning with you.",
        "What do you think about the relationship between mathematics and music?",
        "That's exactly right! The harmonic series is fascinating.",
        "Can you tell me more about how Fourier transforms connect these domains?",
        "Yes, perfect explanation. You're really good at drawing these connections.",
        "No, actually that's not quite right. Let me explain it differently.",
        "What about the connection between fractals and natural patterns?",
        "Great, thanks for the conversation!",
    ]

    for i, user_msg in enumerate(conversation):
        print(f"\n  Turn {i + 1}: \"{user_msg[:60]}{'...' if len(user_msg) > 60 else ''}\"")
        result = model.process_turn(user_msg)
        print(f"    Novelty:       {result.novelty_score:.3f}")
        print(f"    Mood:          v={result.mood.valence:+.3f}  "
              f"a={result.mood.arousal:.3f}  o={result.mood.openness:.3f}")
        print(f"    Reinforcement: {result.reinforcement_strength:+.3f}")
        print(f"    Partial update: {result.partial_update_applied}")
        print(f"    Tokens generated: {len(result.generated_tokens)}")

    # End session → consolidation
    print(f"\n  Ending session...")
    report = model.session_end()
    if report:
        print(f"  Consolidation report:")
        print(f"    Records replayed:  {report.records_replayed}")
        print(f"    Total replays:     {report.total_replay_passes}")
        print(f"    Connections pruned: {report.connections_pruned}")
        print(f"    Weights protected: {report.weights_protected}")
        if report.personality_deltas:
            print(f"    Personality shifts:")
            for trait, delta in report.personality_deltas.items():
                print(f"      {trait:15s} {delta:+.6f}")

    print_status(model, "After first session")

    # --- Phase 3: Persistence test ---
    print("=" * 60)
    print("  PHASE 3: PERSISTENCE TEST")
    print("=" * 60)

    old_personality = model.personality.as_array().clone()
    old_age = model.developmental_age

    # Create a new Model instance from the same storage
    model2 = Model(genome=genome, storage_dir=storage_dir, seed=42)
    new_personality = model2.personality.as_array()

    print(f"  Age preserved: {old_age} → {model2.developmental_age}")
    print(f"  Personality preserved: {torch.allclose(old_personality, new_personality)}")
    print(f"  Encoder still frozen: {model2.encoder.is_frozen}")

    print_status(model2, "Restored model")

    # --- Phase 4: Second session (showing continuity) ---
    print("=" * 60)
    print("  PHASE 4: SECOND SESSION (CONTINUITY)")
    print("=" * 60)

    model2.session_start()

    conversation2 = [
        "I'm back! Let's continue our discussion about math and music.",
        "Actually no, I want to talk about something completely different. How about cooking?",
        "Tell me about the chemistry behind bread rising.",
        "Wrong, that's not how yeast works at all.",
        "Wrong again. Please be more careful.",
    ]

    for i, user_msg in enumerate(conversation2):
        print(f"\n  Turn {i + 1}: \"{user_msg[:60]}{'...' if len(user_msg) > 60 else ''}\"")
        result = model2.process_turn(user_msg)
        print(f"    Novelty:       {result.novelty_score:.3f}")
        print(f"    Mood:          v={result.mood.valence:+.3f}  "
              f"a={result.mood.arousal:.3f}  o={result.mood.openness:.3f}")
        print(f"    Reinforcement: {result.reinforcement_strength:+.3f}")

    report2 = model2.session_end()
    print_status(model2, "After second session (mood should be lower, caution up)")

    # Show personality drift across sessions
    print("=" * 60)
    print("  PERSONALITY DRIFT ACROSS SESSIONS")
    print("=" * 60)
    for name in ("curiosity", "warmth", "assertiveness", "creativity", "caution", "humor"):
        v1 = getattr(model.personality, name)
        v2 = getattr(model2.personality, name)
        arrow = "↑" if v2 > v1 else "↓" if v2 < v1 else "="
        print(f"  {name:15s}  {v1:.4f} → {v2:.4f}  {arrow}")

    # Cleanup
    shutil.rmtree(storage_dir, ignore_errors=True)
    print(f"\nCleaned up {storage_dir}")
    print("\nDone.")


if __name__ == "__main__":
    main()
